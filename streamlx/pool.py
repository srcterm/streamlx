"""Per-layer ExpertPool: preallocated slot tensors, id->slot LRU, fetch-on-miss.

Exactness basis: gather_qmm over relocated expert rows with remapped indices
is bitwise-identical to the full stacked call. The forward here is a
line-for-line mirror of mlx_lm SwitchGLU's decode path (no-sort branch),
parameterized by which triples (full tensors or pool slots) it runs over.

Eviction: the least recently used expert leaves, applied per miss, only
between steps by construction (ensure() runs before the step's compute graph
is built). Requires n_slots >= k.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED
from concurrent.futures import wait as _wait_all

import mlx.core as mx
import numpy as np
try:
    from mlx_lm.models.activations import swiglu
except ImportError:    # mlx-lm's layout from 11-19 (#2082)
    from mlx_lm.layers.common import swiglu

from .reader import ExpertReader
from .stindex import PARTS, SafetensorsIndex

# STREAMLX_KEEPALIVE=0 turns off the decode-time GPU keepalive and state job
# below and the wrapper's early output dispatch (integrate.py).
KEEPALIVE = os.environ.get("STREAMLX_KEEPALIVE", "1") != "0"
GPU_HOLD = KEEPALIVE
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

    def wait(self, futures, first: bool = False) -> None:
        when = FIRST_COMPLETED if first else ALL_COMPLETED
        while True:
            done, pending = _wait_all(futures, timeout=_KEEPALIVE_PERIOD_S,
                                      return_when=when)
            if not pending or (first and done):
                return
            y = self._x + 1
            mx.async_eval(y)
            self._inflight.append(y)
            if len(self._inflight) > 2:
                mx.eval(self._inflight.pop(0))


class _GpuHold:
    """Holds the GPU's performance state up through a decode that syncs on
    every layer's routing.

    Such a decode gives the GPU sub-millisecond bursts with the host in
    between, and once another thread is busy next to it (one CPU-bound
    thread in any process is enough) the system can run the same work at a
    lower performance state (on an M4: 2x, at times 3.7x the time) for
    seconds on end; on a quiet machine it does not. One sustained GPU job
    of some tens of milliseconds raises the state for about two seconds;
    short jobs and the keepalive's tiny kernels do not. So the switch reports
    how long each routing sync of a one-token call took (a round trip of
    one layer, no read in it); when their recent mean runs well above the
    level seen right after a job, one job is run, a second apart at least.
    With nothing to go by, a job says where the level is: at a thread's
    first decode, then ten seconds after the last job, and twice as long
    each time such a job changed nothing. One per thread, as the keepalive.
    Pipelined steps do not report: their state drops when they read much,
    and then the reads bound the step, not the GPU (the job cost them its
    own time); the decoder's plain steps do.
    """
    WINDOW = 16        # routing syncs to a mean
    SLOW = 1.25        # a mean this many times the level: run a job
    PERIOD = 1.0       # seconds between jobs, at least
    PROBE = 10.0       # seconds without a job before one is run anyway
    MIN_JOB = 0.03     # seconds a job should last: its matmuls double below

    def __init__(self):
        self._a = None
        self._mats = 6
        self._n, self._sum = 0, 0.0
        self._level = None             # the mean when the state is up
        self._before = None            # the mean the last job followed,
        self._probing = False          # until the next mean; it was a probe
        self._probe = self.PROBE       # seconds to the next probe
        self._last = 0.0               # (long ago: the first job comes soon)

    def note(self, dt: float) -> None:
        self._sum += dt
        self._n += 1
        if self._n < self.WINDOW:
            return
        mean, self._n, self._sum = self._sum / self._n, 0, 0.0
        if self._before is not None:   # the first mean after a job
            if self._probing:          # did the probe change anything?
                self._probe = (self.PROBE if self._before > 1.1 * mean
                               else min(2 * self._probe, 8 * self.PROBE))
            self._level, self._before = mean, None
            return
        if self._level is None or mean < self._level:
            self._level = mean         # as good as it gets now
            return
        self._level *= 1.001           # a longer context costs a little more
        idle = time.perf_counter() - self._last
        slow = mean > self.SLOW * self._level and idle >= self.PERIOD
        if not slow and idle < self._probe:
            return
        self._before, self._probing = mean, not slow
        if slow:
            self._probe = self.PROBE
        self.run()

    def run(self) -> None:
        if self._a is None:
            self._a = mx.full((2048, 2048), 1 / 2048)
        t0 = time.perf_counter()
        y = self._a
        for _ in range(self._mats):
            y = y @ self._a
        mx.eval(y)
        self._last = time.perf_counter()
        if self._last - t0 < self.MIN_JOB and self._mats < 96:
            self._mats *= 2
        with _HOLD_LOCK:
            _HOLD_TOTAL[0] += 1
            _HOLD_TOTAL[1] += self._last - t0


# Jobs run and their seconds over every decode thread. A hold itself (and
# its matrix on the GPU) is its thread's alone and goes when the thread does.
_HOLD_LOCK = threading.Lock()
_HOLD_TOTAL = [0, 0.0]


def gpu_hold(dt: float) -> None:
    """A one-token decode's routing sync took dt seconds (see _GpuHold)."""
    if not GPU_HOLD:
        return
    h = getattr(_KEEPALIVE_LOCAL, "hold", None)
    if h is None:
        h = _KEEPALIVE_LOCAL.hold = _GpuHold()
    h.note(dt)


def gpu_hold_stats() -> tuple:
    """(jobs run, their seconds) over every decode thread."""
    return _HOLD_TOTAL[0], _HOLD_TOTAL[1]


def keepalive_wait(futures, first: bool = False) -> None:
    """Block until `futures` finish (first: until one of them does), keeping
    the GPU busy meanwhile."""
    ka = getattr(_KEEPALIVE_LOCAL, "ka", None)
    if ka is None:
        ka = _KEEPALIVE_LOCAL.ka = _GpuKeepalive()
    ka.wait(futures, first)


def _storage(arr: mx.array) -> memoryview:
    """The array's own memory as writable bytes (MLX exports an evaluated
    array's buffer, and the GPU reads what is written there). Hold it only
    while writing: a live view keeps MLX from updating the array in place."""
    m = memoryview(arr)
    if m.readonly or not m.c_contiguous or m.nbytes != arr.nbytes:
        raise RuntimeError("array does not export its storage")
    return m.cast("B")


def switch_forward(x: mx.array, indices: mx.array, triples: dict,
                   qparams: dict, act=None, escales=None,
                   sorted_indices: bool = False, weights=None) -> mx.array:
    """Mirror of SwitchGLU.__call__ decode path (indices.size < 64, no bias).

    x: [B, T, D]; indices: [B, T, k] (expert ids for the full tensors, or
    slot ids for pool tensors). triples: {proj: {weight, scales[, biases]
    [, global_scale]}} (the float modes ship no biases; nvfp4 may carry a
    scale per expert). qparams: {proj: (in_features,
    out_features, bits, group_size, mode)}. act: the model's activation
    module when it is not plain SwiGLU, called with SwitchGLU's (up, gate)
    order. escales: optional (gate_vec[E], out_vec[E], eidx) per-expert
    output scales applied exactly as the stock module does; eidx indexes
    the vectors by EXPERT id with the same nesting as `indices` (which may
    hold pool slot ids, a different numbering). weights: routing weights
    shaped like `indices`, for an activation that applies them itself
    before the down projection (deepseek_v41), called act(up, gate, w).

    With gate and up in one projection (inkling2: gate rows even, up rows
    odd) the call mirrors that model's Experts: act(gate, up), its own.
    """
    x = mx.expand_dims(x, -2 if sorted_indices else (-2, -3))

    def qmm(inp, proj):
        t = triples[proj]
        _, _, bits, gs, mode = qparams[proj]
        return mx.gather_qmm(inp, t["weight"], t["scales"], t.get("biases"),
                             rhs_indices=indices, transpose=True,
                             group_size=gs, bits=bits, mode=mode,
                             global_scale=t.get("global_scale"),
                             sorted_indices=sorted_indices)

    if "gate_up_proj" in triples:
        gu = qmm(x, "gate_up_proj")
        return qmm(act(gu[..., 0::2], gu[..., 1::2]), "down_proj").squeeze(-2)

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
        # the model's cache entries of this layer, in call order: one, or
        # one a visit for a looped stack (set at load)
        self.visits = [layer]
        self.tag = tag              # module path of a non-trunk block
        self.triples_loc = self.index.switch_triples(layer, tag=tag)
        # the layer's projections: gate / up / down, or gate+up / down
        self.projs = projs = tuple(self.triples_loc)
        self.qparams = {p: self.index.quant_params(self.triples_loc[p])
                        for p in projs}
        # parts present in this layer's tensors (affine: W/S/B; mxfp: W/S;
        # nvfp4 with a scale per expert: W/S/G)
        self.parts = tuple(pt for pt in PARTS
                           if pt in self.triples_loc[projs[0]])
        k_any = self.triples_loc[projs[0]]["weight"]
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
        self.min_slots = 1    # resize() floor: the host sets the routed top-k
        self.act = None       # non-default activation, set at wrap time
        self.act_weights = False  # act takes the routing weights (v4.1)
        self.escales = None   # optional (gate_vec, out_vec), set at wrap time
        # the pipeline's hint for the NEXT switch call (one stream's layer):
        # .rows = the experts the stream read itself, .idx = its routing when
        # the caller already synced it (None: the switch syncs as usual)
        self.call_hint = None

        def alloc(loc):
            dt = {"U32": mx.uint32, "BF16": mx.bfloat16,
                  "F16": mx.float16, "F32": mx.float32,
                  "U8": mx.uint8, "U16": mx.uint16}[loc.dtype]
            return mx.zeros((n_slots,) + loc.shape[1:], dtype=dt)

        self.pool = {p: {part: alloc(self.triples_loc[p][part])
                         for part in self.parts} for p in projs}
        # expert -> slot, the least recently used first (the next to go)
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
        self.sweep_resident = 0  # routed by a sweep, taken from the pool
        self.sweep_fetch_s = 0.0
        self.adopted = 0         # experts placed by prefill sweeps (no IO)
        self.next_layer_reads = 0   # read for this layer's next sweep early
        self.next_layer_used = 0    # of those, routed by that sweep

    def _take_slot_for(self, e: int) -> int:
        """A slot for e: a free one, else the least recently used expert's."""
        if self.free:
            slot = self.free.pop()
        else:
            old_id, slot = self.id2slot.popitem(last=False)
            self.slot_table[old_id] = self.SENTINEL
            self.evictions += 1
        self.id2slot[e] = slot
        self.slot_table[e] = slot
        return slot

    def ensure(self, ids: list[int], wait=None) -> list[int]:
        """Return slot ids for `ids` (touch order = given order).

        Misses are slot-assigned in order (LRU) but FETCHED as one threaded
        batch, each range straight into its slot's memory (no array is built
        and no slot copy follows): eviction decisions never depend on fetch
        timing. A batch whose read fails leaves none of its experts resident
        (forget) and raises. The slots it takes must be none a kernel in
        flight reads: the
        decode path touches its hits before calling, and a layer's earlier
        calls are evaluated by the time its routing is known.
        `wait`, when given, blocks on the batch's read futures in place of a
        plain wait (the decode path's GPU keepalive).
        """
        slots = []
        missing: list[int] = []
        for e in ids:
            if e in self.id2slot:
                self.id2slot.move_to_end(e)
                self.hits += 1
            else:
                self.misses += 1
                self._take_slot_for(e)
                missing.append(e)
            slots.append(self.id2slot[e])
        if missing:
            t0 = time.perf_counter()
            try:
                self.reader.read_into(self.slot_requests(missing), wait=wait)
            except BaseException:
                self.forget(missing)       # no read is still writing them
                raise
            self.fetch_s += time.perf_counter() - t0
        return slots

    def forget(self, ids) -> None:
        """Experts given slots whose read did not finish: the slots hold
        part of them, or what was there before, so the experts are not
        resident and the slots are free (the next use reads them again).
        Call it only once no read is writing those slots."""
        for e in ids:
            slot = self.id2slot.pop(e, None)
            if slot is not None:
                self.slot_table[e] = self.SENTINEL
                self.free.append(slot)

    def slot_requests(self, ids: list[int]) -> list:
        """reader.read_into's requests for experts that have their slots:
        each of their ranges, into its slot's memory. A part with one value
        an expert (nvfp4's global_scale) is copied from RAM here."""
        reqs = []
        for proj in self.projs:
            for part in self.parts:
                loc, mem = (self.triples_loc[proj][part],
                            _storage(self.pool[proj][part]))
                n = loc.row_stride
                if len(loc.shape) == 1:
                    vec = memoryview(self.reader._vector(loc)).cast("B")
                    for e in ids:
                        s = self.id2slot[e] * n
                        mem[s:s + n] = vec[e * n:(e + 1) * n]
                    continue
                for e in ids:
                    path, off, n = loc.expert_range(e)
                    s = self.id2slot[e] * n
                    reqs.append((path, off, n, mem[s:s + n]))
        return reqs

    def resident(self, e: int) -> bool:
        return e in self.id2slot

    def resize(self, n_slots: int) -> int:
        """Change the slot count, within [min_slots, the layer's experts].
        Fewer slots: the least recently used experts lose theirs and the
        rest move into the first slots; more: free slots are added. Kept
        experts' bytes are copied unchanged and only which experts are
        resident changes, so outputs don't. Call it with nothing in flight
        on the pool (between requests). Returns the new count."""
        n = max(self.min_slots, min(int(n_slots), self.n_experts))
        if n == self.n_slots:
            return n
        if n < self.n_slots:
            for e in list(self.id2slot)[:max(0, len(self.id2slot) - n)]:
                self.id2slot.pop(e)
                self.slot_table[e] = self.SENTINEL
                self.evictions += 1
            used = {slot: e for e, slot in self.id2slot.items()}
            free = [t for t in range(n) if t not in used]
            src = list(range(n))
            for slot, e in used.items():
                if slot >= n:              # into a free slot below n
                    t = free.pop()
                    src[t] = slot
                    self.id2slot[e] = t    # keeps its place in the LRU order
                    self.slot_table[e] = t
            idx = mx.array(src, dtype=mx.uint32)
            for proj in self.projs:
                for part in self.parts:
                    a = mx.take(self.pool[proj][part], idx, axis=0)
                    mx.eval(a)
                    self.pool[proj][part] = a
            self.free = free
        else:
            for proj in self.projs:
                for part in self.parts:
                    old = self.pool[proj][part]
                    a = mx.concatenate(
                        [old, mx.zeros((n - self.n_slots,) + old.shape[1:],
                                       dtype=old.dtype)])
                    mx.eval(a)
                    self.pool[proj][part] = a
            self.free.extend(range(self.n_slots, n))
        self.n_slots = n
        return n

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

    def adopt(self, e: int, arrays: dict) -> None:
        """Place expert e's already-read arrays ({(e, proj, part): array}, as
        the reader builds them) into a slot without IO: their bytes are
        copied into the slot's memory, as ensure() reads them there.

        Used by the prefill sweep to warm the pool from data that just
        streamed through a mini-pool, and by the pipeline for what a step
        read. LRU semantics mirror ensure() (evicts the LRU head when full);
        no hit/miss accounting — adoptions are free. As for ensure(), no
        kernel in flight may read the slot taken.
        """
        if e in self.id2slot:
            self.id2slot.move_to_end(e)
            slot = self.id2slot[e]
        else:
            slot = self._take_slot_for(e)
        self.adopted += 1
        for proj in self.projs:
            for part in self.parts:
                src = memoryview(arrays[(e, proj, part)]).cast("B")
                s = slot * len(src)
                _storage(self.pool[proj][part])[s:s + len(src)] = src

    def reset_stats(self) -> None:
        self.hits = self.misses = self.evictions = 0
        self.fetch_s = 0.0
        self.sweep_experts = 0
        self.sweep_resident = 0
        self.sweep_fetch_s = 0.0
        self.adopted = 0
        self.next_layer_reads = self.next_layer_used = 0

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
                "bytes_read": self.reader.bytes_read,
                "reads": self.reader.reads}
