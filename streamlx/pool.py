"""Per-layer ExpertPool: preallocated slot tensors, id->slot LRU, fetch-on-miss.

Exactness basis (F9): gather_qmm over relocated expert rows with remapped
indices is bitwise-identical to the full stacked call. The forward here is a
line-for-line mirror of mlx_lm SwitchGLU's decode path (no-sort branch),
parameterized by which triples (full tensors or pool slots) it runs over —
both the reference and the streaming path share `switch_forward`, so the ONLY
difference under test is relocation + index remap.

Eviction: LRU head, applied per-miss (identical semantics to sim/lru_sim.py's
lru_layer, so measured misses cross-validate against the simulator). v1
evicts only between steps by construction (ensure() runs before the step's
compute graph is built). Requires n_slots >= k.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from .reader import ExpertReader
from .stindex import PARTS, PROJS, SafetensorsIndex


def switch_forward(x: mx.array, indices: mx.array, triples: dict,
                   qparams: dict) -> mx.array:
    """Mirror of SwitchGLU.__call__ decode path (indices.size < 64, no bias).

    x: [B, T, D]; indices: [B, T, k] (expert ids for the full tensors, or
    slot ids for pool tensors). triples: {proj: {weight, scales, biases}}.
    qparams: {proj: (in_features, out_features, bits, group_size)}.
    """
    x = mx.expand_dims(x, (-2, -3))

    def qmm(inp, proj):
        t = triples[proj]
        _, _, bits, gs = qparams[proj]
        return mx.gather_qmm(inp, t["weight"], t["scales"], t["biases"],
                             rhs_indices=indices, transpose=True,
                             group_size=gs, bits=bits, mode="affine")

    x_up = qmm(x, "up_proj")
    x_gate = qmm(x, "gate_proj")
    x = qmm(swiglu(x_gate, x_up), "down_proj")
    return x.squeeze(-2)


class StreamingSwitchMLP:
    def __init__(self, model_dir: str, layer: int, n_slots: int,
                 index: SafetensorsIndex | None = None,
                 reader: ExpertReader | None = None):
        self.index = index or SafetensorsIndex(model_dir)
        self.reader = reader or ExpertReader(self.index)
        self.layer = layer
        self.triples_loc = self.index.switch_triples(layer)
        self.qparams = {p: self.index.quant_params(self.triples_loc[p])
                        for p in PROJS}
        k_any = self.triples_loc["gate_proj"]["weight"]
        self.n_experts = k_any.shape[0]
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        self.n_slots = n_slots

        import numpy as np  # local: only for dtype mapping at alloc

        def alloc(loc):
            dt = {"U32": mx.uint32, "BF16": mx.bfloat16,
                  "F16": mx.float16, "F32": mx.float32}[loc.dtype]
            return mx.zeros((n_slots,) + loc.shape[1:], dtype=dt)

        self.pool = {p: {part: alloc(self.triples_loc[p][part])
                         for part in PARTS} for p in PROJS}
        self.id2slot: OrderedDict[int, int] = OrderedDict()
        self.free = list(range(n_slots))
        # expert id -> slot, SENTINEL when absent; enables in-graph-free
        # vectorized remap in the wrapper (numpy take, no per-id dict walk)
        self.SENTINEL = np.uint32(0xFFFFFFFF)
        self.slot_table = np.full(self.n_experts, self.SENTINEL,
                                  dtype=np.uint32)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.fetch_s = 0.0
        self.sweep_experts = 0   # expert-rows streamed by prefill sweeps
        self.sweep_fetch_s = 0.0

    def ensure(self, ids: list[int]) -> list[int]:
        """Return slot ids for `ids` (touch order = given order, like the sim).

        Misses are slot-assigned in order (LRU-head eviction, identical
        semantics to sim/lru_sim.py) but FETCHED as one coalesced batch —
        eviction decisions never depend on fetch timing, so miss counts stay
        sim-exact while IO gets span-merged + threaded (M4 coalescing).
        """
        slots = []
        missing: list[int] = []
        for e in ids:
            if e in self.id2slot:
                self.id2slot.move_to_end(e)
                self.hits += 1
            else:
                self.misses += 1
                if self.free:
                    slot = self.free.pop()
                else:
                    old_id, slot = self.id2slot.popitem(last=False)  # LRU head
                    self.slot_table[old_id] = self.SENTINEL
                    self.evictions += 1
                self.id2slot[e] = slot
                self.slot_table[e] = slot
                missing.append(e)
            slots.append(self.id2slot[e])
        if missing:
            t0 = time.perf_counter()
            rows = self.reader.read_experts(self.triples_loc, missing)
            for e in missing:
                slot = self.id2slot[e]
                for proj in PROJS:
                    for part in PARTS:
                        self.pool[proj][part][slot] = rows[e][proj][part]
            self.fetch_s += time.perf_counter() - t0
        return slots

    def resident(self, e: int) -> bool:
        return e in self.id2slot

    def touch(self, ids: list[int]) -> None:
        """Mark resident experts most-recently-used (hit accounting only).

        The overlap path touches hits before fetching misses, so a same-step
        eviction can never reclaim a slot the GPU is still reading. Eviction
        decisions match plain `ensure` whenever n_slots >= 2k (every real
        budget); below that, touch-first merely avoids a double fetch.
        """
        for e in ids:
            self.id2slot.move_to_end(e)
            self.hits += 1

    def __call__(self, x: mx.array, expert_ids: list[int]) -> mx.array:
        if len(expert_ids) > self.n_slots:
            raise ValueError("n_slots must be >= k")
        slots = self.ensure(expert_ids)
        inds = mx.array([[slots]], dtype=mx.uint32)
        return switch_forward(x, inds, self.pool, self.qparams)

    def reset_stats(self) -> None:
        self.hits = self.misses = self.evictions = 0
        self.fetch_s = 0.0
        self.sweep_experts = 0
        self.sweep_fetch_s = 0.0

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "miss_rate": self.misses / total if total else 0.0,
                "fetch_s": round(self.fetch_s, 4),
                "sweep_experts": self.sweep_experts,
                "sweep_fetch_s": round(self.sweep_fetch_s, 4),
                "bytes_read": self.reader.bytes_read,
                "reads": self.reader.reads}
