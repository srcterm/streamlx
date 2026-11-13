"""Model integration: replace each layer's `switch_mlp` with a streaming
wrapper that computes the MoE FFN from the ExpertPool (pread-backed) instead
of the resident stacked tensors.

The wrapper is signature-compatible with SwitchGLU.__call__(x, indices):
the surrounding SparseMoeBlock (gate, top-k, score weighting, shared expert)
is untouched, so the only change in the graph is where expert weights live.

Sync point (by design): routed indices must be materialized per layer per
step to drive fetch decisions — this is the fundamental gate-then-fetch
dependency of expert streaming.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .pool import KEEPALIVE, StreamingSwitchMLP, keepalive_wait, switch_forward
from .reader import ExpertReader
from .stindex import PROJS, SafetensorsIndex


_PF_EX = None   # the prefill sweeps' read queue: two batches in flight


def _pf_ex():
    global _PF_EX
    if _PF_EX is None:
        _PF_EX = ThreadPoolExecutor(max_workers=2,
                                    thread_name_prefix="streamlx-pf")
    return _PF_EX


# Prefill reads run ahead of the compute through one queue: a sweep's own
# batches, then the next streamed layer's first reads, at most _NEXT_BYTES of
# experts queued and not yet used. The next layer sees the same tokens, so it
# routes most of its experts too, and the drive would otherwise idle through
# this layer's tail and the next layer's attention. The next layer's are only
# read by a sweep that routes at least _NEXT_MIN_COVERAGE of its layer's
# experts: below it (a short chunk after a prompt-cache hit), too many of
# those reads would go unused.
_NEXT_MIN_COVERAGE = 0.7
_NEXT_BYTES = 512 * 2**20

# The cache of the prefill chunk in flight (prefill_cache), and the last layer
# whose entry its streamed layers have evaluated.
_PREFILL = {"cache": None, "done": -1}


@contextmanager
def prefill_cache(cache):
    """Run a prefill chunk's forward in this: each streamed layer's MoE call
    then evaluates the cache entries of the layers up to it, so what they
    store is computed as the chunk passes. Left lazy until the host evaluates
    the cache after the forward, a stored state keeps the arrays it is made
    from: a linear-attention layer's conv window is three rows of its whole
    conv input (Qwen3.8: ~1.3 GiB over 36 layers for 1.6K tokens, ~3 GiB for
    4K), a rotating cache's window its chunk's keys. Evaluation order only:
    the values are the same."""
    prev = _PREFILL["cache"], _PREFILL["done"]
    _PREFILL["cache"], _PREFILL["done"] = cache, -1
    try:
        yield
    finally:
        _PREFILL["cache"], _PREFILL["done"] = prev


def _stock_sorted(n_routes: int, n_experts: int) -> bool:
    """Whether the stock module's call runs gather_qmm's sorted matmul
    kernel: SwitchGLU sorts its routes from 64 on, and gather_qmm takes the
    sorted kernel from 16 rows and 4 rows per expert on (MLX's GatherQMM
    rule); otherwise every row is its own matrix-vector product. streamlx
    picks the same kernel so its outputs match stock's bitwise."""
    return n_routes >= 64 and n_routes // n_experts >= 4


def _pick(w, pos, flat_shape: bool = False):
    """Routing weights of the (token, pick) pairs at flat positions `pos`,
    nested [1, 1, n] like a decode call's slot ids (flat_shape: [n])."""
    if w is None:
        return None
    sel = mx.take(w.reshape(-1), mx.array(np.asarray(pos, dtype=np.uint32)))
    return sel if flat_shape else sel[None, None]


class StreamingSwitchGLU(nn.Module):
    def __init__(self, pool: StreamingSwitchMLP):
        super().__init__()
        self.pool = pool
        # A chunk that routes more experts than the pool holds is swept
        # (_expert_sweep), this many experts a batch.
        self.sweep_batch = int(os.environ.get("STREAMLX_SWEEP_BATCH", "16"))
        self.sweep_warm = os.environ.get("STREAMLX_SWEEP_WARM", "1") != "0"
        # What a sweep adopts: "fill" (its most-routed experts into the free
        # slots only, the experts the pool holds left where decoding put
        # them) or "count" (its n_slots most-routed, evicting and retouching:
        # decode after a prefill then misses a third more).
        self.warm_policy = os.environ.get("STREAMLX_WARM_POLICY", "fill")
        # Next-layer reads: the next streamed layer's wrapper (linked at
        # load), what a sweep may still queue for it (expert ids), the reads
        # the previous layer started for this layer's next sweep ({expert id:
        # future of its batch read}), and this layer's recent sweep routing
        # (route counts, halved every sweep), which ranks what the previous
        # layer reads.
        self._next = None
        self._todo = None
        self._early = None
        self._hist = None
        # Hosts probe these to pick fused resident-tensor fast paths; None
        # sends them down the standard switch_mlp(x, indices) call.
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def _own_cache_states(self) -> None:
        """prefill_cache's per-layer step: evaluate the entries of the layers
        up to this one that this forward has not (a layer at or below the
        last one seen starts a new forward)."""
        cache, done = _PREFILL["cache"], _PREFILL["done"]
        upto = min(self.pool.layer, len(cache) - 1)
        if upto <= done:
            done = -1
        states = []
        for c in cache[done + 1:upto + 1]:
            try:
                st = c.state
            except Exception:    # an entry with nothing stored yet
                continue
            todo = [st]
            while todo:
                x = todo.pop()
                if isinstance(x, mx.array):
                    states.append(x)
                elif isinstance(x, (list, tuple)):
                    todo.extend(x)
        if states:
            mx.async_eval(states)
        _PREFILL["done"] = upto

    def _remap(self, idx: np.ndarray) -> np.ndarray:
        slots = self.pool.slot_table[idx]
        if (slots == self.pool.SENTINEL).any():
            raise RuntimeError(f"unmapped expert after ensure "
                               f"(layer {self.pool.layer})")
        return slots

    def _esc(self, eidx) -> tuple | None:
        """switch_forward's escales triple, or None (the common case). eidx:
        EXPERT-id index nested like the call site's slot indices — the scale
        vectors are expert-addressed while pool slots permute."""
        es = self.pool.escales
        if es is None:
            return None
        if not isinstance(eidx, mx.array):
            eidx = mx.array(eidx, dtype=mx.uint32)
        return (es[0], es[1], eidx)

    def _overlapped(self, x: mx.array, idx: np.ndarray,
                    wait=None, w=None) -> mx.array | None:
        """Compute resident experts on the GPU while missing ones are read.

        Baseline order per layer is strictly serial — gate sync, then a
        blocking fetch (GPU idle), then the FFN (SSD idle). Here the hit half
        is dispatched with async_eval first, so the pread overlaps real GPU
        work. Exactness is preserved because switch_forward returns per-expert
        outputs *before* the router's weighted sum: each row depends only on x
        and its own expert, so computing two subsets and permuting back is
        bitwise identical (verified against the stock model by
        examples/validate.py).

        Returns None when there is nothing to overlap (no misses, or no hits).
        `wait` is handed to the pool's fetch (the GPU keepalive).
        """
        flat = idx.reshape(-1).tolist()
        seen, hit_ids, miss_ids = set(), [], []
        for e in flat:
            if e in seen:
                continue
            seen.add(e)
            (hit_ids if self.pool.resident(e) else miss_ids).append(e)
        if not miss_ids or not hit_ids:
            return None

        self.pool.touch(hit_ids)          # protect hit slots before evicting
        hits = set(hit_ids)
        hit_pos = [i for i, e in enumerate(flat) if e in hits]
        miss_pos = [i for i, e in enumerate(flat) if e not in hits]

        slots = [self.pool.id2slot[flat[i]] for i in hit_pos]
        y_hit = switch_forward(x, mx.array([[slots]], dtype=mx.uint32),
                               self.pool.pool, self.pool.qparams,
                               self.pool.act,
                               escales=self._esc(
                                   [[[flat[i] for i in hit_pos]]]),
                               weights=_pick(w, hit_pos))
        mx.async_eval(y_hit)              # GPU starts; CPU proceeds to fetch

        self.pool.ensure(miss_ids, wait=wait)

        slots = [self.pool.id2slot[flat[i]] for i in miss_pos]
        y_miss = switch_forward(x, mx.array([[slots]], dtype=mx.uint32),
                                self.pool.pool, self.pool.qparams,
                                self.pool.act,
                                escales=self._esc(
                                    [[[flat[i] for i in miss_pos]]]),
                                weights=_pick(w, miss_pos))

        y = mx.concatenate([y_hit, y_miss], axis=-2)
        inv = np.argsort(np.array(hit_pos + miss_pos, dtype=np.uint32))
        return mx.take(y, mx.array(inv.astype(np.uint32)), axis=-2)

    def _stream_forward(self, x: mx.array, idx: np.ndarray,
                        rows: dict, w=None) -> mx.array:
        """One pipelined stream's layer: experts resident in the pool (which
        the pipeline leaves unchanged during its step) from their slots, the
        rest one by one from `rows`, the experts the stream read itself;
        subsets permuted back as in _overlapped, so bitwise identical to one
        call over the pool."""
        pool = self.pool
        flat = idx.reshape(-1).tolist()
        lost = [e for e in dict.fromkeys(flat)
                if e not in rows and not pool.resident(e)]
        if lost:
            pool.ensure(lost)        # a host changed the pool mid-step
        res = [i for i, e in enumerate(flat) if e not in rows]
        own = [i for i, e in enumerate(flat) if e in rows]
        outs = []
        if res:
            slots = [pool.id2slot[flat[i]] for i in res]
            outs.append(switch_forward(
                x, mx.array([[slots]], dtype=mx.uint32), pool.pool,
                pool.qparams, pool.act,
                escales=self._esc([[[flat[i] for i in res]]]),
                weights=_pick(w, res)))
        if own:
            # each of the stream's own experts straight from its rows (a
            # one-expert slab is a view): no copy into a shared slab
            zero = mx.zeros((1, 1, 1), dtype=mx.uint32)
            for i in own:
                e = flat[i]
                one = {p: {q: rows[e][p][q][None] for q in pool.parts}
                       for p in PROJS}
                outs.append(switch_forward(x, zero, one, pool.qparams,
                                           pool.act,
                                           escales=self._esc([[[e]]]),
                                           weights=_pick(w, [i])))
        y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-2)
        order = res + own
        if order != sorted(order):
            inv = np.argsort(np.array(order, dtype=np.uint32))
            y = mx.take(y, mx.array(inv.astype(np.uint32)), axis=-2)
        if KEEPALIVE:
            mx.async_eval(y)         # as the decode path: GPU starts now
        return y

    def _pooled(self, x: mx.array, idx: np.ndarray,
                slots: np.ndarray, w=None) -> mx.array:
        """A non-decode call over pool slots on the kernel stock's call
        uses (_stock_sorted): sorted by slot where stock sorts and takes the
        sorted kernel (a pool holds at most the layer's experts, so
        gather_qmm takes that kernel here too), else the per-row path."""
        pool = self.pool
        if not _stock_sorted(slots.size, pool.n_experts):
            return switch_forward(x, mx.array(slots), pool.pool, pool.qparams,
                                  pool.act,
                                  escales=self._esc(idx.astype(np.uint32)),
                                  weights=w)
        b, t, k = slots.shape
        flat = slots.reshape(-1)
        order = np.argsort(flat, kind="stable")
        xs = mx.take(x.reshape(b * t, -1),
                     mx.array((order // k).astype(np.uint32)), axis=0)
        eidx = idx.reshape(-1)[order].astype(np.uint32)
        y = switch_forward(xs, mx.array(flat[order]), pool.pool,
                           pool.qparams, pool.act,
                           escales=self._esc(mx.array(eidx)),
                           sorted_indices=True,
                           weights=_pick(w, order, flat_shape=True))
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        y = mx.take(y, mx.array(inv.astype(np.uint32)), axis=0)
        return y.reshape(b, t, k, -1)

    def _plan_next_layer(self, n_routed: int, n_read: int) -> None:
        """What a sweep may read ahead for the next layer (self._todo, in id
        order; see _read_next_layer): as many experts as it reads itself, the
        next layer's most-routed in recent sweeps (its lowest ids before it
        has swept) that its pool does not hold. Nothing below
        _NEXT_MIN_COVERAGE, while the next layer still holds earlier reads,
        and when its pool would hold the whole call (no sweep)."""
        nxt, self._todo = self._next, None
        if (nxt is None or nxt._early is not None
                or n_routed < _NEXT_MIN_COVERAGE * self.pool.n_experts
                or nxt.pool.n_slots >= n_routed):
            return
        p, h = nxt.pool, nxt._hist
        rank = (np.argsort(-h, kind="stable") if h is not None
                else np.arange(p.n_experts))
        self._todo = sorted([int(e) for e in rank
                             if not p.resident(int(e))][:n_read])
        nxt._early = {}

    def _read_next_layer(self, ahead: int) -> None:
        """Queue the next layer's reads that fit under _NEXT_BYTES with this
        sweep's `ahead` experts still to use. One batch per task (data-only
        arrays; their dtype views, which are ops, are made on the compute
        thread)."""
        if not self._todo:
            return
        nxt = self._next
        p, bs, early = nxt.pool, nxt.sweep_batch, nxt._early
        room = (max(bs, _NEXT_BYTES // p.expert_bytes) - len(early)
                - ahead * self.pool.expert_bytes // p.expert_bytes)
        while self._todo and min(bs, len(self._todo)) <= room:
            ids, self._todo = self._todo[:bs], self._todo[bs:]
            fut = _pf_ex().submit(p.reader.read_experts_arrays, p.triples_loc,
                                  ids)
            early.update(dict.fromkeys(ids, fut))
            p.next_layer_reads += len(ids)
            room -= len(ids)

    def _take_early(self, routed: set) -> dict:
        """The previous layer's reads for this call: those under way or done,
        of experts it routes. The ones not started are cancelled (the call
        now knows what it routes and reads exactly that)."""
        early, self._early = self._early, None
        if not early:
            return {}
        keep = {e: f for e, f in early.items()
                if not f.cancel() and e in routed}
        self.pool.next_layer_used += len(keep)
        return keep

    def _expert_sweep(self, x: mx.array, idx: np.ndarray,
                      w=None) -> mx.array:
        """Expert-major prefill: read each routed expert once, in id order.

        A chunk whose expert union exceeds the pool, run in token slices that
        fit it, re-reads evicted experts (measured 2.7x on Laguna) at
        scattered-read bandwidth. Here (position, pick) pairs are grouped by
        expert instead:
        experts stream through throwaway mini-pools in ascending id order
        (adjacent rows coalesce into long sequential spans), each read exactly
        once per chunk. Experts the pool holds are not read: their rows are
        copies of their slots, taken before any adoption below can move one.

        Bitwise-exact vs the stock module: where stock's call runs the
        sorted matmul kernel (_stock_sorted), each batch here is a run of
        experts on that kernel (padded when too sparse for it), else every
        row is a matrix-vector product as in stock; either way every output
        row depends only on its own x row and its own expert, so splitting
        the experts into batches is a permutation of independent
        computations (same basis as _overlapped).

        Batches dispatch with async_eval so the GPU computes batch i while
        batch i+1 is read; mini-pool buffers are immutable so no per-slice
        eval barrier is needed. One eval at the end materializes the layer
        output and frees the mini-pools before the next layer runs.

        Reads run ahead of the compute on the shared read queue, at most
        _NEXT_BYTES of experts queued and not yet used: the sweep's batches
        in id order, then the experts the next layer is likely to route
        (_plan_next_layer), so the drive keeps reading through this layer's
        tail and the next one's attention; that sweep takes the ones it
        routes. sweep_fetch_s counts only time blocked on IO.

        With sweep_warm (default on), the chunk's most-routed experts are
        adopted into the pool's free slots as their batch streams by (no
        extra IO), most-frequent most-recently-used; the experts the pool
        holds stay as decoding left them, since decode after the prefill
        routes more like the decode before it than like the prompt
        (warm_policy "count" fills the whole pool from the chunk).
        """
        pool = self.pool
        b, t, k = idx.shape
        P = b * t
        x_rows = x.reshape(P, -1)
        srt = _stock_sorted(idx.size, pool.n_experts)
        flat = idx.reshape(-1)                     # pair -> expert id
        row_of_pair = np.repeat(np.arange(P, dtype=np.uint32), k)

        order = np.argsort(flat, kind="stable")    # pairs grouped by expert
        sorted_e = flat[order]
        uniq, starts = np.unique(sorted_e, return_index=True)
        bounds = np.append(starts, sorted_e.size)

        counts = bounds[1:] - bounds[:-1]
        if self._hist is None:
            self._hist = np.zeros(pool.n_experts, np.float32)
        self._hist *= 0.5
        self._hist[uniq] += counts
        warm_ids: set = set()
        if self.sweep_warm:
            top = np.argsort(counts, kind="stable")[::-1]
            if self.warm_policy == "fill":
                room = pool.n_slots - len(pool.id2slot)
                top = [i for i in top.tolist()
                       if not pool.resident(int(uniq[i]))][:max(0, room)]
            else:
                top = top[:pool.n_slots]
            warm_ids = {int(uniq[i]) for i in top}
        # the routed experts the pool holds: copies of their slots (a gather
        # into new buffers, no view of a slab), made before an adoption can
        # evict one and rewrite its slot
        resident = {int(e): pool.id2slot[int(e)] for e in uniq
                    if pool.resident(int(e))}
        taken, res_pos = None, {}
        if resident:
            res_pos = {e: j for j, e in enumerate(resident)}
            sl = mx.array(list(resident.values()), dtype=mx.uint32)
            taken = {p: {q: mx.take(pool.pool[p][q], sl, axis=0)
                         for q in pool.parts} for p in PROJS}
            mx.eval(*(taken[p][q] for p in PROJS for q in pool.parts))
            pool.sweep_resident += len(resident)

        batches = [[int(e) for e in uniq[i0:i0 + self.sweep_batch]]
                   for i0 in range(0, len(uniq), self.sweep_batch)]
        early = self._take_early(set(uniq.tolist()))
        self._plan_next_layer(len(uniq), len(uniq) - len(resident))
        # what the previous layer did not start nor the pool has
        need = [[e for e in ids if e not in early and e not in resident]
                for ids in batches]
        cap = max(self.sweep_batch, _NEXT_BYTES // pool.expert_bytes)
        own, sub, ahead = {}, 0, len(early)   # ahead: queued, not yet used

        def queue(upto):   # this layer's reads in order, then the next one's
            nonlocal sub, ahead
            while sub < len(batches) and (
                    sub <= upto or ahead + len(need[sub]) <= cap):
                if need[sub]:
                    own[sub] = _pf_ex().submit(
                        pool.reader.read_experts_arrays, pool.triples_loc,
                        need[sub])
                    ahead += len(need[sub])
                sub += 1
            if sub == len(batches):
                self._read_next_layer(ahead)

        queue(0)
        outs = []
        for bi, ids in enumerate(batches):
            t0 = time.perf_counter()
            raw = {}
            # what `ahead` stops counting below is let go: a read leaves
            # `early` with the last expert it holds for this call and hands
            # over each batch's arrays as that batch takes them
            reads = [early.pop(e) for e in ids if e in early]
            if bi in own:
                reads.append(own.pop(bi))
            for f in dict.fromkeys(reads):
                arrays = f.result()
                for key in [key for key in arrays if key[0] in ids]:
                    raw[key] = arrays.pop(key)
            pool.sweep_fetch_s += time.perf_counter() - t0   # blocked wait
            # built on the reading threads; the dtype views (ops) here
            got = [e for e in ids if e not in resident]
            ahead -= len(got)
            queue(bi + 1)
            rows = pool.reader.experts_from_arrays(pool.triples_loc, got, raw)
            for e in ids:
                if e in resident:
                    rows[e] = {p: {q: taken[p][q][res_pos[e]]
                                   for q in pool.parts} for p in PROJS}
            pool.sweep_experts += len(got)
            mini = {p: {q: mx.stack([rows[e][p][q] for e in ids])
                        for q in pool.parts} for p in PROJS}
            i0 = bi * self.sweep_batch
            i1 = min(i0 + self.sweep_batch, len(uniq))
            sel = order[bounds[i0]:bounds[i1]]
            slot = np.searchsorted(uniq[i0:i1],
                                   sorted_e[bounds[i0]:bounds[i1]])
            xb = mx.take(x_rows, mx.array(row_of_pair[sel]), axis=0)
            inds = slot.astype(np.uint32)
            eb = sorted_e[bounds[i0]:bounds[i1]].astype(np.uint32)
            wb = _pick(w, sel, flat_shape=True)
            if srt:
                # a batch too sparse for the sorted kernel gets zero rows on
                # its last expert (outputs dropped), so it runs stock's kernel
                pad = max(16, 4 * len(ids)) - len(inds)
                if pad > 0:
                    xb = mx.concatenate(
                        [xb, mx.zeros((pad, xb.shape[-1]), xb.dtype)])
                    inds = np.append(inds, np.full(pad, len(ids) - 1,
                                                   np.uint32))
                    eb = np.append(eb, np.full(pad, ids[-1], np.uint32))
                    if wb is not None:
                        wb = mx.concatenate([wb, mx.zeros((pad,), wb.dtype)])
                y = switch_forward(xb, mx.array(inds), mini, pool.qparams,
                                   pool.act, escales=self._esc(mx.array(eb)),
                                   sorted_indices=True, weights=wb)  # [Nb, D]
                if pad > 0:
                    y = y[:-pad]
            else:
                y = switch_forward(xb[None], mx.array(inds)[None, :, None],
                                   mini, pool.qparams, pool.act,
                                   escales=self._esc(
                                       mx.array(eb)[None, :, None]),
                                   weights=None if wb is None
                                   else wb[None, :, None])[0, :, 0]
            mx.async_eval(y)       # GPU busy while the next batch is read
            outs.append(y)
            for e in ids:
                if e in warm_ids:
                    if e in pool.id2slot:   # adopt() would rewrite its bytes
                        pool.id2slot.move_to_end(e)
                    else:
                        pool.adopt(e, rows[e])
        if warm_ids:
            for i in np.argsort(counts, kind="stable"):  # ascending count
                e = int(uniq[i])
                if e in warm_ids and e in pool.id2slot:
                    pool.id2slot.move_to_end(e)          # most-frequent = MRU
        del taken
        y = mx.concatenate(outs, axis=0)           # rows in sorted-pair order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        y = mx.take(y, mx.array(inv.astype(np.uint32)), axis=0)
        y = y.reshape(b, t, k, -1)
        # One eval materializes the layer output AND adopted pool slots,
        # freeing this layer's mini-pools before the next layer runs.
        mx.eval(y, *(pool.pool[p][q] for p in PROJS for q in pool.parts))
        return y

    def __call__(self, x: mx.array, indices: mx.array,
                 scores: mx.array | None = None, **_kw) -> mx.array:
        if x.ndim == 2:
            # hosts that flatten [B, T, D] -> [P, D] before the switch call
            return self.__call__(x[None], indices[None],
                                 None if scores is None else scores[None],
                                 **_kw).squeeze(0)
        # `scores` / extra kwargs are fusion hints some hosts pass; returning
        # per-expert outputs (ndim == scores.ndim + 1) makes the caller
        # apply the weighted sum itself. An activation that takes the
        # routing weights (deepseek_v41) gets them, pair by pair (w).
        w = scores if self.pool.act_weights else None
        decode = indices.shape[0] * indices.shape[1] == 1  # shape is metadata
        hint, self.pool.call_hint = self.pool.call_hint, None
        if hint is not None and decode:
            idx = (np.array(indices) if hint.idx is None
                   else hint.idx.reshape(indices.shape))
            return self._stream_forward(x, idx, hint.rows, w)
        idx = np.array(indices)  # materializes the gate's selection (sync)
        if not decode and _PREFILL["cache"] is not None:
            self._own_cache_states()
        uniq = list(dict.fromkeys(idx.reshape(-1).tolist()))
        if len(uniq) > self.pool.n_slots:
            return self._expert_sweep(x, idx, w)
        if self._early is not None:
            self._take_early(set())   # the previous layer read for a sweep
        if not decode:
            self.pool.ensure(uniq)
            return self._pooled(x, idx, self._remap(idx), w)
        # Decode reads wait with the GPU kept busy (see pool._GpuKeepalive).
        ka = keepalive_wait if KEEPALIVE else None
        y = self._overlapped(x, idx, wait=ka, w=w)
        if y is None:
            self.pool.ensure(uniq, wait=ka)
            y = switch_forward(x, mx.array(self._remap(idx)), self.pool.pool,
                               self.pool.qparams, self.pool.act,
                               escales=self._esc(idx.astype(np.uint32)),
                               weights=w)
        if ka is not None:
            # Dispatch the layer's experts now, so the GPU starts on them
            # while the host builds the rest of the layer.
            mx.async_eval(y)
        return y


def _moe_block(layer):
    """The submodule holding switch_mlp: `mlp` in most mlx-lm MoE models,
    `ffn` in deepseek_v4."""
    for attr in ("mlp", "ffn"):
        m = getattr(layer, attr, None)
        if m is not None and hasattr(m, "switch_mlp"):
            return m
    return None


def _uncompile_ffn(layer) -> None:
    """Hosts that mx.compile a layer's FFN half for single-token decode must
    run it eagerly once the switch is pool-backed: the wrapper materializes
    the routing indices (a sync) to know which experts to read, and an eval
    inside a compiled function is an error."""
    if getattr(layer, "compile_ffn", None):
        layer.compile_ffn = False
    if getattr(layer, "_ffn_c", None) is not None:
        layer._ffn_c = None


def _n_args(fn) -> int:
    import inspect

    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 0


def _wrap(layer, mlp, pool) -> None:
    """Swap a block's switch_mlp for the streaming wrapper, carrying over
    what the stock module computed besides the gathers."""
    # A non-default activation (deepseek_v4's LimitedSwiGLU) must survive
    # the swap; plain SwiGLU stays on the fused default.
    act = getattr(mlp.switch_mlp, "activation", None)
    if type(act).__name__ != "SwiGLU":
        pool.act = act
        # deepseek_v41 weights each expert's activation before its down
        # projection: act(up, gate, weights)
        pool.act_weights = act is not None and _n_args(act) >= 3
    # Per-expert output scales are tiny [E] vectors riding the module being
    # dropped: materialize them now (expert tensors stay lazy). Trivial
    # all-ones stays None so models without them pay nothing.
    gsv = getattr(mlp.switch_mlp, "gate_scale", None)
    osv = getattr(mlp.switch_mlp, "out_scale", None)
    if gsv is not None and osv is not None:
        mx.eval(gsv, osv)
        if not (mx.all(gsv == 1) & mx.all(osv == 1)).item():
            pool.escales = (gsv, osv)
    _uncompile_ffn(layer)
    mlp.switch_mlp = StreamingSwitchGLU(pool)


def install_streaming(model, model_dir: str, n_slots: int):
    """Swap every MoE layer's switch_mlp for a streaming wrapper.

    Returns (pools, reader) — reader is shared (one fd set, global IO stats).
    """
    index = SafetensorsIndex(model_dir)
    reader = ExpertReader(index)
    pools: dict[int, StreamingSwitchMLP] = {}
    for i, layer in enumerate(model.layers):
        mlp = _moe_block(layer)
        if mlp is None:
            continue
        pool = StreamingSwitchMLP(model_dir, i, n_slots, index=index,
                                  reader=reader)
        _wrap(layer, mlp, pool)
        pools[i] = pool
    if not pools:
        raise RuntimeError("no MoE layers found to wrap")
    _link_next_layers(model, pools)
    return pools, reader


def _link_next_layers(model, pools: dict) -> None:
    """Point each streamed layer at the next streamed one for prefill
    next-layer reads (dense layers between them see the same tokens too)."""
    ws = [_moe_block(model.layers[i]).switch_mlp for i in sorted(pools)]
    for a, b in zip(ws, ws[1:]):
        a._next = b


def _vlm_export_shim(model_dir: str) -> dict | None:
    """mlx-vlm exports of text-only models prefix every tensor AND every
    per-path quantization override with "language_model.", which stock
    mlx-lm text impls don't expect (strict load fails on the extra keys,
    and the override table would silently never match a module). Detected
    via the safetensors index; returns a model_config overlay whose
    quantization table has both spellings, and installs a sanitize wrapper
    that strips the prefix from the weights unless the mlx-lm class wraps a
    language_model itself (the qwen3_5 family), which takes them as they
    are. None/no-op for native exports."""
    import json
    from pathlib import Path

    idx = Path(model_dir) / "model.safetensors.index.json"
    if not idx.exists():
        return None
    wm = json.loads(idx.read_text())["weight_map"]
    if not wm or not all(k.startswith("language_model.") for k in wm):
        return None

    import mlx_lm.utils as _u

    def _remap(k: str) -> str:
        # mlx-vlm wraps the router matrix in a proj Linear; mlx-lm's
        # MoEGate holds it directly as `gate.weight`.
        return k.removeprefix("language_model.").replace(
            ".mlp.gate.proj.", ".mlp.gate.")

    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    model_cls, _ = _u._get_classes(config=cfg)
    if not getattr(model_cls, "_streamlx_vlm_sanitize", False):
        prev = getattr(model_cls, "sanitize", None)

        def sanitize(self, weights):
            if prev is not None:
                weights = prev(self, weights)
            if "language_model" not in self and any(
                    k.startswith("language_model.") for k in weights):
                weights = {_remap(k): v for k, v in weights.items()}
            return weights

        model_cls.sanitize = sanitize
        model_cls._streamlx_vlm_sanitize = True

    quant = cfg.get("quantization")
    if not quant:
        return None
    return {"quantization": {**quant,
                             **{_remap(k): v for k, v in quant.items()}}}


def _routed_topk(model_dir: str, default: int = 8) -> int:
    """The checkpoint's num_experts_per_tok (top level or text_config)."""
    import json
    from pathlib import Path

    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return default
    for c in (cfg, cfg.get("text_config") or {}):
        v = c.get("num_experts_per_tok")
        if isinstance(v, int) and v > 0:
            return v
    return default


def _checkpoint_loader(model_dir: str):
    """The loader for a checkpoint mlx-lm can't run: DeepSeek-V4.1 with Engram
    layers (mlx-lm refuses them) and Qwen3.8 (not in mlx-lm): mlx-vlm has both."""
    import json
    from pathlib import Path

    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if (cfg.get("model_type") == "deepseek_v41"
            and (cfg.get("text_config") or cfg).get("engram_layer_ids")):
        try:
            from .deepseek_v41 import load_v41
        except ImportError as e:
            raise ImportError("DeepSeek-V4.1 with Engram layers needs mlx-vlm: "
                              "pip install mlx-vlm") from e
        return load_v41
    if cfg.get("model_type") == "qwen4_exp":
        try:
            from .qwen4_exp import load_qwen4_exp
        except ImportError as e:
            raise ImportError("Qwen3.8 needs mlx-vlm: pip install mlx-vlm") from e
        return load_qwen4_exp
    return None


def load_streaming_model(model_dir: str, budget_bytes: int,
                         k: int | None = None,
                         trust_remote_code: bool = False,
                         tokenizer_config: dict | None = None,
                         model_config: dict | None = None,
                         loader=None):
    """Load with lazy=True, wrap MoE layers, materialize ONLY the trunk.

    The replaced SwitchGLU modules are dropped before any eval, so expert
    tensors are never materialized: resident memory = trunk + pools. Budget
    is split per layer by that layer's actual expert byte size (equal
    bytes a layer). Every streamed layer keeps at
    least k slots (default: the model's experts per token, which one decode
    step needs at once; 8 if the config does not say).

    loader: optional host load hook for checkpoints mlx-lm's registry can't
    build (VLM-packaged model types, vendored implementations). Called as
    loader(model_dir, tokenizer_config=..., model_config=...) and must
    return (model, tokenizer) with LAZY weights, `model.layers` exposing
    the decoder stack, and MoE blocks discoverable by _moe_block. The
    mlx-lm path (and its vlm-export shim) is skipped.
    """
    if k is None:
        k = _routed_topk(model_dir)
    tok_cfg = dict(tokenizer_config or {})
    if trust_remote_code:
        tok_cfg["trust_remote_code"] = True
    if loader is None:
        loader = _checkpoint_loader(model_dir)
    if loader is not None:
        model, tokenizer = loader(model_dir, tokenizer_config=tok_cfg or None,
                                  model_config=model_config)
    else:
        from mlx_lm import load as _load

        cfg_overlay = _vlm_export_shim(model_dir) or {}
        cfg_overlay.update(model_config or {})
        model, tokenizer = _load(model_dir, lazy=True,
                                 tokenizer_config=tok_cfg,
                                 model_config=cfg_overlay or None)
    index = SafetensorsIndex(model_dir)
    reader = ExpertReader(index)
    moe_layers = [i for i, l in enumerate(model.layers)
                  if _moe_block(l) is not None]
    bpe = {i: index.layer_expert_bytes(i) for i in moe_layers}
    pools: dict[int, StreamingSwitchMLP] = {}
    for i in moe_layers:
        slots = max(k, int(budget_bytes / len(moe_layers) / bpe[i]))
        pool = StreamingSwitchMLP(model_dir, i, slots, index=index,
                                  reader=reader)
        _wrap(model.layers[i], _moe_block(model.layers[i]), pool)
        pool.min_slots = min(k, pool.n_experts)
        pools[i] = pool
    _link_next_layers(model, pools)
    # the trunk: the experts' modules were replaced before any eval and
    # never materialize
    mx.eval(model.parameters())
    # Pool slot buffers are plain-object attrs, invisible to parameters();
    # eval them on the loader thread so multi-threaded hosts never touch
    # lazy arrays stream-bound to a thread they don't run on.
    mx.eval([t for p in pools.values()
             for proj in p.pool.values() for t in proj.values()])
    return model, tokenizer, pools, reader


def pool_bytes(pools: dict) -> int:
    """The bytes the pools' slots hold."""
    return sum(p.n_slots * p.expert_bytes for p in pools.values())


def resize_pools(pools: dict, budget_bytes: float) -> int:
    """Split `budget_bytes` over `pools` the way load_streaming_model splits
    its budget (equal bytes a pool, at least its min_slots, at most its
    experts) and resize each pool to its share (StreamingSwitchMLP.resize:
    exact), shrinking ones first so the copies' transient stays one pool
    array. Returns the bytes the pools then hold."""
    n = len(pools)
    want = {i: max(p.min_slots, min(p.n_experts,
                                    int(budget_bytes / n / p.expert_bytes)))
            for i, p in pools.items()}
    for i in sorted(pools, key=lambda i: want[i] - pools[i].n_slots):
        pools[i].resize(want[i])
    mx.clear_cache()
    return pool_bytes(pools)


def stream_moe_blocks(module, index: SafetensorsIndex, reader: ExpertReader,
                      key0: int, prefix: str = "", slots: int = 1,
                      k: int | None = None) -> dict:
    """Pools for the MoE blocks of a module outside the trunk (a draft head):
    block i of module.layers, whose switch tensors are at
    `{prefix}layers.{i}` in the index's checkpoint, streams through a pool
    keyed key0 + i, as a trunk layer does. Call it before the module's
    parameters are evaluated: the replaced switch weights then never load.
    Returns {key: pool}; size them with resize_pools."""
    pools = {}
    for i, layer in enumerate(getattr(module, "layers", None) or []):
        mlp = _moe_block(layer)
        if mlp is None:
            continue
        tag = f"{prefix}layers.{i}"
        try:
            index.layer_expert_bytes(key0 + i, tag=tag)
        except KeyError:
            continue                # no streamable switch tensors there
        pool = StreamingSwitchMLP(index.model_dir, key0 + i,
                                  max(slots, k or 1), index=index,
                                  reader=reader, tag=tag)
        pool.min_slots = min(k or 1, pool.n_experts)
        _wrap(layer, mlp, pool)
        pools[key0 + i] = pool
    return pools


def aggregate_stats(pools: dict) -> dict:
    tot = {"hits": 0, "misses": 0, "evictions": 0, "fetch_s": 0.0,
           "sweep_experts": 0, "sweep_fetch_s": 0.0, "adopted": 0,
           "next_layer_reads": 0, "next_layer_used": 0}
    for p in pools.values():
        s = p.stats
        for k in ("hits", "misses", "evictions", "sweep_experts", "adopted",
                  "next_layer_reads", "next_layer_used"):
            tot[k] += s.get(k, 0)
        tot["fetch_s"] += s["fetch_s"]
        tot["sweep_fetch_s"] += s.get("sweep_fetch_s", 0.0)
    acc = tot["hits"] + tot["misses"]
    tot["miss_rate"] = tot["misses"] / acc if acc else 0.0
    tot["fetch_s"] = round(tot["fetch_s"], 3)
    tot["sweep_fetch_s"] = round(tot["sweep_fetch_s"], 3)
    return tot
