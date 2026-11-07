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

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import wait as _wait_all

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from .reader import ExpertReader
from .stindex import PARTS, PROJS, SafetensorsIndex

# STREAMLX_KEEPALIVE=0 turns off the decode-time GPU keepalive below and the
# wrapper's early output dispatch (integrate.py).
KEEPALIVE = os.environ.get("STREAMLX_KEEPALIVE", "1") != "0"
_KEEPALIVE_PERIOD_S = 50e-6
_KEEPALIVE_LOCAL = threading.local()


class _GpuKeepalive:
    """Keeps the GPU clocked while the decode thread waits on expert reads.

    Apple GPUs lower their clock after a couple of milliseconds idle, and the
    next layer's work runs slower until it recovers. One gather at the start
    of the wait does not hold the clock (sparse bursts measured slower than a
    plain idle gap); near-continuous tiny submissions do. At most two trivial
    kernels are in flight, touching nothing the model reads. One per thread:
    an mx array is bound to the stream of the thread that made it.
    """

    def __init__(self):
        self._x = mx.zeros((8,), dtype=mx.float32)
        self._inflight: list = []

    def wait(self, futures) -> None:
        while True:
            _, pending = _wait_all(futures, timeout=_KEEPALIVE_PERIOD_S)
            if not pending:
                return
            y = self._x + 1
            mx.async_eval(y)
            self._inflight.append(y)
            if len(self._inflight) > 2:
                mx.eval(self._inflight.pop(0))


def keepalive_wait(futures) -> None:
    """Block until `futures` finish, keeping the GPU busy meanwhile."""
    ka = getattr(_KEEPALIVE_LOCAL, "ka", None)
    if ka is None:
        ka = _KEEPALIVE_LOCAL.ka = _GpuKeepalive()
    ka.wait(futures)


def switch_forward(x: mx.array, indices: mx.array, triples: dict,
                   qparams: dict, act=None, escales=None,
                   sorted_indices: bool = False, weights=None) -> mx.array:
    """Mirror of SwitchGLU.__call__ decode path (indices.size < 64, no bias).

    x: [B, T, D]; indices: [B, T, k] (expert ids for the full tensors, or
    slot ids for pool tensors). triples: {proj: {weight, scales[, biases]}}
    (mxfp modes ship no biases). qparams: {proj: (in_features,
    out_features, bits, group_size, mode)}. act: the model's activation
    module when it is not plain SwiGLU, called with SwitchGLU's (up, gate)
    order. escales: optional (gate_vec[E], out_vec[E], eidx) per-expert
    output scales applied exactly as the stock module does; eidx indexes
    the vectors by EXPERT id with the same nesting as `indices` (which may
    hold pool slot ids, a different numbering). weights: routing weights
    shaped like `indices`, for an activation that applies them itself
    before the down projection (deepseek_v41), called act(up, gate, w).
    """
    x = mx.expand_dims(x, -2 if sorted_indices else (-2, -3))

    def qmm(inp, proj):
        t = triples[proj]
        _, _, bits, gs, mode = qparams[proj]
        return mx.gather_qmm(inp, t["weight"], t["scales"], t.get("biases"),
                             rhs_indices=indices, transpose=True,
                             group_size=gs, bits=bits, mode=mode,
                             sorted_indices=sorted_indices)

    def esc(vec, eidx, like):
        sc = vec[eidx].astype(like.dtype)
        return sc.reshape(sc.shape + (1,) * (like.ndim - sc.ndim))

    x_up = qmm(x, "up_proj")
    x_gate = qmm(x, "gate_proj")
    if escales is not None:
        x_gate = x_gate * esc(escales[0], escales[2], x_gate)
    if act is None:
        h = swiglu(x_gate, x_up)
    elif weights is None:
        h = act(x_up, x_gate)
    else:
        h = act(x_up, x_gate, weights[..., None, None])
    x = qmm(h, "down_proj")
    if escales is not None:
        x = x * esc(escales[1], escales[2], x)
    return x.squeeze(-2)


class StreamingSwitchMLP:
    def __init__(self, model_dir: str, layer: int, n_slots: int,
                 index: SafetensorsIndex | None = None,
                 reader: ExpertReader | None = None,
                 tag: str | None = None):
        self.index = index or SafetensorsIndex(model_dir)
        self.reader = reader or ExpertReader(self.index)
        self.layer = layer
        self.tag = tag              # module path of a non-trunk block
        self.triples_loc = self.index.switch_triples(layer, tag=tag)
        self.qparams = {p: self.index.quant_params(self.triples_loc[p])
                        for p in PROJS}
        # parts present in this layer's tensors (affine: W/S/B; mxfp: W/S)
        self.parts = tuple(pt for pt in PARTS
                           if pt in self.triples_loc["gate_proj"])
        k_any = self.triples_loc["gate_proj"]["weight"]
        self.n_experts = k_any.shape[0]
        self.expert_bytes = sum(loc.row_stride
                                for parts in self.triples_loc.values()
                                for loc in parts.values())
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        # never more slots than experts: gather_qmm's kernel choice depends
        # on the slab's size, and stock's holds exactly the layer's experts
        n_slots = min(n_slots, self.n_experts)
        self.n_slots = n_slots
        self.act = None       # non-default activation, set at wrap time
        self.act_weights = False  # act takes the routing weights (v4.1)
        self.escales = None   # optional (gate_vec, out_vec), set at wrap time
        self._pf_suspend = False  # a pipelined step is running: no lookahead
        # the pipeline's hint for the NEXT switch call (one stream's layer):
        # .rows = the experts the stream read itself, .idx = its routing when
        # the caller already synced it (None: the switch syncs as usual)
        self.call_hint = None

        import numpy as np  # local: only for dtype mapping at alloc

        def alloc(loc):
            dt = {"U32": mx.uint32, "BF16": mx.bfloat16,
                  "F16": mx.float16, "F32": mx.float32,
                  "U8": mx.uint8, "U16": mx.uint16}[loc.dtype]
            return mx.zeros((n_slots,) + loc.shape[1:], dtype=dt)

        self.pool = {p: {part: alloc(self.triples_loc[p][part])
                         for part in self.parts} for p in PROJS}
        self.id2slot: OrderedDict[int, int] = OrderedDict()
        self.free = list(range(n_slots))
        # Eviction policy: "lru" (default) or "s3fifo" (F19: -9..-12% misses
        # on Laguna — small FIFO filters one-hit wonders, ghost queue rescues
        # repeaters). Policy only decides WHICH expert loses its slot; the
        # arithmetic never changes.
        self.policy = os.environ.get("STREAMLX_EVICT", "lru")
        self.n_small = max(1, int(n_slots * 0.1))
        self.n_main = max(1, n_slots - self.n_small)
        self.q_small: OrderedDict[int, int] = OrderedDict()  # expert -> freq
        self.q_main: OrderedDict[int, int] = OrderedDict()
        self.q_ghost: OrderedDict[int, bool] = OrderedDict()
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
        self.adopted = 0         # experts placed by prefill sweeps (no IO)
        self.next_layer_reads = 0   # read for this layer's next sweep early
        self.next_layer_used = 0    # of those, routed by that sweep
        self._pf = None          # in-flight: (pred_set, launch, fut, rank_of)
        self.pf_calls = 0        # predictions targeting this layer
        self.pf_launched = 0     # experts submitted to async read
        self.pf_used = 0         # prefetched experts the step actually routed to
        self.pf_agree_hit = 0    # |predicted ∩ actual| accumulator
        self.pf_agree_tot = 0    # |actual| accumulator
        self.pf_rank = {}        # confidence rank -> [launched, used]

    def _policy_hit(self, e: int) -> None:
        if self.policy == "s3fifo":
            q = self.q_small if e in self.q_small else self.q_main
            q[e] = min(3, q[e] + 1)
        else:
            self.id2slot.move_to_end(e)

    def _s3_evict_main(self) -> int:
        """Pop main's first zero-freq entry (decrement-and-recycle heads),
        per sim/policy_ab.py run_s3fifo."""
        while self.q_main:
            e, f = self.q_main.popitem(last=False)
            if f > 0:
                self.q_main[e] = f - 1
                continue
            return e
        raise RuntimeError("s3fifo: empty main on eviction")

    def _s3_insert(self, e: int) -> int | None:
        """Register e in its queue; return the evicted expert, or None when
        queue capacity allowed the insert without one (a free physical slot
        is guaranteed to exist in that case)."""
        if e in self.q_ghost:
            del self.q_ghost[e]
            victim = (self._s3_evict_main()
                      if len(self.q_main) >= self.n_main else None)
            self.q_main[e] = 0
            return victim
        victim = None
        if len(self.q_small) >= self.n_small:
            e2, f = self.q_small.popitem(last=False)
            if f > 0:                      # proved itself: promote to main
                if len(self.q_main) >= self.n_main:
                    victim = self._s3_evict_main()
                self.q_main[e2] = 0
            else:                          # one-hit wonder: ghost it
                if len(self.q_ghost) >= self.n_main:
                    self.q_ghost.popitem(last=False)
                self.q_ghost[e2] = True
                victim = e2
        self.q_small[e] = 0
        return victim

    def _take_slot_for(self, e: int) -> int:
        """Free a physical slot per the eviction policy and bind it to e."""
        if self.policy == "s3fifo":
            victim = self._s3_insert(e)
            if victim is not None:
                slot = self.id2slot.pop(victim)
                self.slot_table[victim] = self.SENTINEL
                self.evictions += 1
                self.free.append(slot)
            slot = self.free.pop()
        elif self.free:
            slot = self.free.pop()
        else:
            old_id, slot = self.id2slot.popitem(last=False)  # LRU head
            self.slot_table[old_id] = self.SENTINEL
            self.evictions += 1
        self.id2slot[e] = slot
        self.slot_table[e] = slot
        return slot

    def ensure(self, ids: list[int], wait=None) -> list[int]:
        """Return slot ids for `ids` (touch order = given order, like the sim).

        Misses are slot-assigned in order (eviction policy above, identical
        semantics to the simulator) but FETCHED as one coalesced batch —
        eviction decisions never depend on fetch timing, so miss counts stay
        sim-exact while IO gets span-merged + threaded (M4 coalescing).
        `wait`, when given, blocks on the batch's read futures in place of a
        plain wait (the decode path's GPU keepalive).
        """
        slots = []
        missing: list[int] = []
        for e in ids:
            if e in self.id2slot:
                self._policy_hit(e)
                self.hits += 1
            else:
                self.misses += 1
                self._take_slot_for(e)
                missing.append(e)
            slots.append(self.id2slot[e])
        if missing:
            t0 = time.perf_counter()
            rows = self.reader.read_experts(self.triples_loc, missing,
                                            wait=wait)
            for e in missing:
                slot = self.id2slot[e]
                for proj in PROJS:
                    for part in self.parts:
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
            self._policy_hit(e)
            self.hits += 1

    # Confidence cut: rank-vs-usefulness measured on Laguna decode (12 GiB,
    # battery deltas): rank 0 = 95% used, 6 = 42%, 9 = 22%; top-7 beat both
    # no-cut (46% efficiency, contention) and top-5 (starves absorption).
    _PF_TOPM = int(os.environ.get("STREAMLX_PF_TOPM", "7"))

    def prefetch_start(self, pred: list[int], ex) -> None:
        """Attention-window prefetch (F18): async raw read of the predicted
        experts for this layer's NEXT call that aren't resident. `pred` is
        ordered by router score, most confident first; only missing
        predictions with confidence rank < STREAMLX_PF_TOPM are read (the
        rank tail is where mispredictions concentrate — see pf_rank). One
        prediction in flight per layer; an unconsumed one is dropped here.
        Reads are submitted on `ex`, a dedicated executor — read_experts_raw
        fans spans onto the reader's own pool, so submitting from that same
        pool could starve it."""
        rank_of = {e: r for r, e in enumerate(pred)}
        missing = [e for e in pred if e not in self.id2slot]
        launch = [e for e in missing if rank_of[e] < self._PF_TOPM]
        for e in launch:
            self.pf_rank.setdefault(rank_of[e], [0, 0])[0] += 1
        if not launch:
            self._pf = (set(pred), [], None, rank_of)
            return
        self.pf_launched += len(launch)
        self._pf = (set(pred), launch,
                    ex.submit(self.reader.read_experts_raw,
                              self.triples_loc, launch), rank_of)

    def prefetch_absorb(self, wanted: list[int], wait=None) -> None:
        """Adopt prefetched experts this step actually routed to. Called
        before ensure(), so absorbed experts are hits with no blocking read.
        Blocks on the in-flight read only when the step needs its bytes
        (they're exactly the bytes ensure() would otherwise re-read).
        Also tallies prediction agreement for the adaptive per-layer gate
        and per-rank usefulness for the confidence cut."""
        pf, self._pf = self._pf, None
        if pf is None:
            return
        pred_set, launch, fut, rank_of = pf
        self.pf_calls += 1
        self.pf_agree_hit += len(set(wanted) & pred_set)
        self.pf_agree_tot += len(wanted)
        if fut is None:
            return
        take = [e for e in launch
                if e in set(wanted) and e not in self.id2slot]
        if not take:
            return
        try:
            if wait is not None:
                wait([fut])
            raws = fut.result()
            rows = self.reader.experts_from_raw(self.triples_loc, launch,
                                                raws)
        except Exception:
            return
        for e in take:
            self.adopt(e, rows[e])
            self.pf_rank.setdefault(rank_of[e], [0, 0])[1] += 1
        self.adopted -= len(take)       # keep sweep-adoption count pure
        self.pf_used += len(take)

    def prefetch_agreement(self) -> float:
        return (self.pf_agree_hit / self.pf_agree_tot
                if self.pf_agree_tot else 1.0)

    def adopt(self, e: int, arrays: dict) -> None:
        """Place expert e's already-read rows into a slot without IO.

        Used by the prefill sweep to warm the pool from data that just
        streamed through a mini-pool. LRU semantics mirror ensure() (evicts
        the LRU head when full); no hit/miss accounting — adoptions are free.
        """
        if e in self.id2slot:
            self._policy_hit(e)
            slot = self.id2slot[e]
        else:
            slot = self._take_slot_for(e)
        self.adopted += 1
        for proj in PROJS:
            for part in self.parts:
                self.pool[proj][part][slot] = arrays[proj][part]

    def __call__(self, x: mx.array, expert_ids: list[int]) -> mx.array:
        if len(expert_ids) > self.n_slots:
            raise ValueError("n_slots must be >= k")
        slots = self.ensure(expert_ids)
        inds = mx.array([[slots]], dtype=mx.uint32)
        es = (None if self.escales is None else
              (*self.escales, mx.array([[expert_ids]], dtype=mx.uint32)))
        return switch_forward(x, inds, self.pool, self.qparams, self.act,
                              escales=es)

    def reset_stats(self) -> None:
        self.hits = self.misses = self.evictions = 0
        self.fetch_s = 0.0
        self.sweep_experts = 0
        self.sweep_fetch_s = 0.0
        self.adopted = 0
        self.next_layer_reads = self.next_layer_used = 0
        self.pf_calls = self.pf_launched = self.pf_used = 0
        self.pf_agree_hit = self.pf_agree_tot = 0
        self.pf_rank = {}

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "miss_rate": self.misses / total if total else 0.0,
                "fetch_s": round(self.fetch_s, 4),
                "sweep_experts": self.sweep_experts,
                "sweep_fetch_s": round(self.sweep_fetch_s, 4),
                "adopted": self.adopted,
                "next_layer_reads": self.next_layer_reads,
                "next_layer_used": self.next_layer_used,
                "pf_calls": self.pf_calls,
                "pf_launched": self.pf_launched,
                "pf_used": self.pf_used,
                "pf_agree_hit": self.pf_agree_hit,
                "pf_agree_tot": self.pf_agree_tot,
                "bytes_read": self.reader.bytes_read,
                "reads": self.reader.reads}
