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
        # In-flight predictions: [(pred_set, launch, fut, rank_of, source)].
        # A list because two sources feed it (norm-ratio prefetch from the
        # exact pass, SEF scout from its own thread); the lock covers the
        # scout-thread/exact-thread handoff.
        self._pf: list = []
        self._pf_lock = threading.Lock()
        # per-source (launched, used) so nr-prefetch vs scout read
        # efficiency stays separable in telemetry
        self.pf_src = {"nr": [0, 0], "scout": [0, 0]}
        # scout emission-rank -> [launched, used] (nr keeps pf_rank; the
        # two calibrations must not mix — different ranking semantics)
        self.pf_rank_scout: dict = {}
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

    def ensure(self, ids: list[int]) -> list[int]:
        """Return slot ids for `ids` (touch order = given order, like the sim).

        Misses are slot-assigned in order (eviction policy above, identical
        semantics to the simulator) but FETCHED as one coalesced batch —
        eviction decisions never depend on fetch timing, so miss counts stay
        sim-exact while IO gets span-merged + threaded (M4 coalescing).
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
            self._policy_hit(e)
            self.hits += 1

    # Confidence cut: rank-vs-usefulness measured on Laguna decode (12 GiB,
    # battery deltas): rank 0 = 95% used, 6 = 42%, 9 = 22%; top-7 beat both
    # no-cut (46% efficiency, contention) and top-5 (starves absorption).
    _PF_TOPM = int(os.environ.get("STREAMLX_PF_TOPM", "7"))

    def prefetch_start(self, pred: list[int], ex, source: str = "nr") -> None:
        """Staged prefetch: async raw read of predicted experts for this
        layer's NEXT call that aren't resident. `pred` is ordered most
        confident first; only ranks < STREAMLX_PF_TOPM are read (the rank
        tail is where mispredictions concentrate). Sources MERGE: each
        call appends an in-flight entry, deduped against resident ids and
        other entries' launches; prefetch_absorb drains them all.
        Agreement/rank telemetry tallies only "nr" entries so the adaptive
        gate and confidence cut keep their calibrated meaning; scout tiers
        keep their own counters. Reads go on `ex`, a dedicated executor —
        read_experts_raw fans spans onto the reader's own pool, so
        submitting from that pool could starve it."""
        rank_of = {e: r for r, e in enumerate(pred)}
        with self._pf_lock:
            inflight = {e for ent in self._pf for e in ent[1]}
            missing = [e for e in pred
                       if e not in self.id2slot and e not in inflight]
            launch = [e for e in missing if rank_of[e] < self._PF_TOPM]
            if source == "nr":
                for e in launch:
                    self.pf_rank.setdefault(rank_of[e], [0, 0])[0] += 1
            elif not launch:
                return          # scout entry with nothing to read: no-op
            else:
                for e in launch:
                    self.pf_rank_scout.setdefault(rank_of[e], [0, 0])[0] += 1
            fut = None
            if launch:
                self.pf_launched += len(launch)
                self.pf_src.setdefault(source, [0, 0])[0] += len(launch)
                fut = ex.submit(self.reader.read_experts_raw,
                                self.triples_loc, launch)
            if len(self._pf) >= 8:      # stale-entry bound (skipped steps)
                self._pf.pop(0)
            self._pf.append((set(pred), launch, fut, rank_of, source))

    def prefetch_absorb(self, wanted: list[int]) -> None:
        """Adopt prefetched experts this step actually routed to. Called
        before ensure(), so absorbed experts are hits with no blocking read.
        Blocks on an in-flight read only when the step needs its bytes
        (they're exactly the bytes ensure() would otherwise re-read).
        Also tallies prediction agreement for the adaptive per-layer gate
        and per-rank usefulness for the confidence cut (nr entries only)."""
        with self._pf_lock:
            entries, self._pf = self._pf, []
        if not entries:
            return
        wset = set(wanted)
        for pred_set, launch, fut, rank_of, source in entries:
            if source == "nr":
                self.pf_calls += 1
                self.pf_agree_hit += len(wset & pred_set)
                self.pf_agree_tot += len(wanted)
            if fut is None:
                continue
            take = [e for e in launch if e in wset and e not in self.id2slot]
            if not take:
                continue
            try:
                raws = fut.result()
                rows = self.reader.experts_from_raw(self.triples_loc,
                                                    launch, raws)
            except Exception:
                continue
            for e in take:
                self.adopt(e, rows[e])
                if source == "nr":
                    self.pf_rank.setdefault(rank_of[e], [0, 0])[1] += 1
                else:
                    self.pf_rank_scout.setdefault(rank_of[e], [0, 0])[1] += 1
            self.adopted -= len(take)   # keep sweep-adoption count pure
            self.pf_used += len(take)
            self.pf_src.setdefault(source, [0, 0])[1] += len(take)

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
            for part in PARTS:
                self.pool[proj][part][slot] = arrays[proj][part]

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
        self.adopted = 0
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
                "pf_calls": self.pf_calls,
                "pf_launched": self.pf_launched,
                "pf_used": self.pf_used,
                "pf_agree_hit": self.pf_agree_hit,
                "pf_agree_tot": self.pf_agree_tot,
                "bytes_read": self.reader.bytes_read,
                "reads": self.reader.reads}
