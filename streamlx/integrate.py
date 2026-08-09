"""Model integration: replace each layer's `switch_mlp` with a streaming
wrapper that computes the MoE FFN from the ExpertPool (pread-backed) instead
of the resident stacked tensors.

The wrapper is signature-compatible with SwitchGLU.__call__(x, indices):
the surrounding SparseMoeBlock (gate, top-k, score weighting, shared expert)
is untouched, so the only change in the graph is where expert weights live.

Sync point (by design, see notes/phase3-design.md): routed indices must be
materialized per layer per step to drive fetch decisions — this is the
fundamental gate-then-fetch dependency of expert streaming.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .pool import StreamingSwitchMLP, switch_forward
from .reader import ExpertReader
from .stindex import PARTS, PROJS, SafetensorsIndex


class StreamingSwitchGLU(nn.Module):
    def __init__(self, pool: StreamingSwitchMLP):
        super().__init__()
        self.pool = pool
        self.record_sink = None  # optional: list collecting (layer, idx np)
        # Prefill strategy when a chunk's expert union exceeds the pool:
        # "sweep" (expert-major, read-once) or "slice" (legacy token slices).
        self.prefill_mode = os.environ.get("STREAMLX_PREFILL", "sweep")
        self.sweep_batch = int(os.environ.get("STREAMLX_SWEEP_BATCH", "16"))
        self.sweep_warm = os.environ.get("STREAMLX_SWEEP_WARM", "1") != "0"
        self._sweep_ex = None   # lazy 1-thread executor: batch read-ahead

    def _remap(self, idx: np.ndarray) -> np.ndarray:
        slots = self.pool.slot_table[idx]
        if (slots == self.pool.SENTINEL).any():
            raise RuntimeError(f"unmapped expert after ensure "
                               f"(layer {self.pool.layer})")
        return slots

    def _overlapped(self, x: mx.array, idx: np.ndarray) -> mx.array | None:
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
                               self.pool.pool, self.pool.qparams)
        mx.async_eval(y_hit)              # GPU starts; CPU proceeds to fetch

        self.pool.ensure(miss_ids)

        slots = [self.pool.id2slot[flat[i]] for i in miss_pos]
        y_miss = switch_forward(x, mx.array([[slots]], dtype=mx.uint32),
                                self.pool.pool, self.pool.qparams)

        y = mx.concatenate([y_hit, y_miss], axis=-2)
        inv = np.argsort(np.array(hit_pos + miss_pos, dtype=np.uint32))
        return mx.take(y, mx.array(inv.astype(np.uint32)), axis=-2)

    def _expert_sweep(self, x: mx.array, idx: np.ndarray) -> mx.array:
        """Expert-major prefill: read each routed expert once, in id order.

        When a chunk's expert union exceeds the pool, the token-slicing path
        re-reads evicted experts (measured 2.7x on Laguna) at scattered-read
        bandwidth. Here (position, pick) pairs are grouped by expert instead:
        experts stream through throwaway mini-pools in ascending id order
        (adjacent rows coalesce into long sequential spans), each read exactly
        once per chunk, and the LRU pool — decode's working set — is never
        touched.

        Bitwise-exact vs the pool path: both run switch_forward on the same
        weight bytes, and every output row depends only on its own x row and
        its own expert, so regrouping pairs by expert is a permutation of
        independent computations (same basis as _overlapped, F9).

        Batches dispatch with async_eval so the GPU computes batch i while
        batch i+1 is read; mini-pool buffers are immutable so no per-slice
        eval barrier is needed. One eval at the end materializes the layer
        output and frees the mini-pools before the next layer runs.

        Reads run one batch ahead on a dedicated thread (sweep_fetch_s counts
        only time actually blocked on IO). With sweep_warm (default on), the
        chunk's most-routed experts are adopted into the LRU pool as their
        batch streams by — no extra IO — so decode after prefill starts warm;
        a final retouch leaves the most-frequent experts most-recently-used.
        """
        pool = self.pool
        b, t, k = idx.shape
        P = b * t
        x_rows = x.reshape(P, -1)
        flat = idx.reshape(-1)                     # pair -> expert id
        row_of_pair = np.repeat(np.arange(P, dtype=np.uint32), k)

        order = np.argsort(flat, kind="stable")    # pairs grouped by expert
        sorted_e = flat[order]
        uniq, starts = np.unique(sorted_e, return_index=True)
        bounds = np.append(starts, sorted_e.size)

        counts = bounds[1:] - bounds[:-1]
        warm_ids: set = set()
        if self.sweep_warm:
            top = np.argsort(counts, kind="stable")[::-1][:pool.n_slots]
            warm_ids = {int(uniq[i]) for i in top}

        if self._sweep_ex is None:
            self._sweep_ex = ThreadPoolExecutor(max_workers=1)
        batches = [[int(e) for e in uniq[i0:i0 + self.sweep_batch]]
                   for i0 in range(0, len(uniq), self.sweep_batch)]
        fut = self._sweep_ex.submit(pool.reader.read_experts_raw,
                                    pool.triples_loc, batches[0])
        outs = []
        for bi, ids in enumerate(batches):
            t0 = time.perf_counter()
            raw = fut.result()
            pool.sweep_fetch_s += time.perf_counter() - t0   # blocked wait
            if bi + 1 < len(batches):
                fut = self._sweep_ex.submit(pool.reader.read_experts_raw,
                                            pool.triples_loc,
                                            batches[bi + 1])
            # mx arrays must be created on this (compute) thread
            rows = pool.reader.experts_from_raw(pool.triples_loc, ids, raw)
            pool.sweep_experts += len(ids)
            mini = {p: {q: mx.stack([rows[e][p][q] for e in ids])
                        for q in PARTS} for p in PROJS}
            i0 = bi * self.sweep_batch
            i1 = min(i0 + self.sweep_batch, len(uniq))
            sel = order[bounds[i0]:bounds[i1]]
            slot = np.searchsorted(uniq[i0:i1],
                                   sorted_e[bounds[i0]:bounds[i1]])
            xb = mx.take(x_rows, mx.array(row_of_pair[sel]), axis=0)[None]
            inds = mx.array(slot.astype(np.uint32))[None, :, None]
            y = switch_forward(xb, inds, mini, pool.qparams)  # [1, Nb, 1, D]
            mx.async_eval(y)       # GPU busy while the next batch is read
            outs.append(y)
            for e in ids:
                if e in warm_ids:
                    pool.adopt(e, rows[e])
        if warm_ids:
            for i in np.argsort(counts, kind="stable"):  # ascending count
                e = int(uniq[i])
                if e in warm_ids and e in pool.id2slot:
                    pool.id2slot.move_to_end(e)          # most-frequent = MRU
        y = mx.concatenate(outs, axis=1)[0, :, 0]  # rows in sorted-pair order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        y = mx.take(y, mx.array(inv.astype(np.uint32)), axis=0)
        y = y.reshape(b, t, k, -1)
        # One eval materializes the layer output AND adopted pool slots,
        # freeing this layer's mini-pools before the next layer runs.
        mx.eval(y, *(pool.pool[p][q] for p in PROJS for q in PARTS))
        return y

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        idx = np.array(indices)  # materializes the gate's selection (sync)
        if self.record_sink is not None:
            self.record_sink.append((self.pool.layer, idx.copy()))
        uniq = list(dict.fromkeys(idx.reshape(-1).tolist()))
        if len(uniq) <= self.pool.n_slots:
            # fast path: whole call fits the pool (always true for decode)
            if idx.shape[0] * idx.shape[1] == 1:
                y = self._overlapped(x, idx)
                if y is not None:
                    return y
            self.pool.ensure(uniq)
            return switch_forward(x, mx.array(self._remap(idx)),
                                  self.pool.pool, self.pool.qparams)

        if self.prefill_mode == "sweep":
            return self._expert_sweep(x, idx)

        # Legacy chunked prefill: process position slices whose expert union
        # fits. Each slice's output MUST be materialized before the next
        # slice's ensure() mutates pool slots (the gather graph reads pool
        # buffers in place); mx.eval per slice enforces that.
        b, t, k = idx.shape
        flat = idx.reshape(-1, k)                     # [P, k], P = B*T
        x_flat = x.reshape(1, -1, x.shape[-1])        # [1, P, D]
        outs = []
        s, P = 0, flat.shape[0]
        while s < P:
            union: set = set()
            e = s
            while e < P:
                nxt = union | set(flat[e].tolist())
                if len(nxt) > self.pool.n_slots:
                    break
                union = nxt
                e += 1
            if e == s:
                raise RuntimeError(f"k > n_slots (layer {self.pool.layer})")
            ids = list(dict.fromkeys(flat[s:e].reshape(-1).tolist()))
            self.pool.ensure(ids)
            sl = self._remap(flat[s:e])[None]         # [1, T', k]
            y = switch_forward(x_flat[:, s:e], mx.array(sl),
                               self.pool.pool, self.pool.qparams)
            mx.eval(y)
            outs.append(y)
            s = e
        y = mx.concatenate(outs, axis=1)              # [1, P, k, D_out]
        return y.reshape(b, t, k, y.shape[-1])


def install_streaming(model, model_dir: str, n_slots: int):
    """Swap every MoE layer's switch_mlp for a streaming wrapper.

    Returns (pools, reader) — reader is shared (one fd set, global IO stats).
    """
    index = SafetensorsIndex(model_dir)
    reader = ExpertReader(index)
    pools: dict[int, StreamingSwitchMLP] = {}
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        pool = StreamingSwitchMLP(model_dir, i, n_slots, index=index,
                                  reader=reader)
        layer.mlp.switch_mlp = StreamingSwitchGLU(pool)
        pools[i] = pool
    if not pools:
        raise RuntimeError("no MoE layers found to wrap")
    return pools, reader


def plan_residency(layers: list, bpe: dict, n_exp: dict,
                   budget_bytes: int, k: int) -> tuple[set, float]:
    """Greedy water-fill: a layer goes FULLY RESIDENT when the fair share of
    the remaining budget covers all its experts (cheapest layer first, freed
    surplus cascades), provided every still-streamed layer keeps at least k
    slots. Fully-resident layers keep their stock SwitchGLU: no gate sync,
    no remap, no pool — the streaming wrapper never touches them.

    Self-selecting by regime: fetch-dominated models (budget << expert
    bytes, e.g. Laguna at 16-18 GiB) qualify no layers and are unchanged;
    overhead-dominated ones (budget ~ expert bytes, e.g. Qwen3.6 at 20 GiB)
    cascade to fully resident. Uniform-width models flip near budget ==
    total expert bytes; mixed-width models (dense-first, varying widths) go
    partial. Returns (resident layer set, budget left for pools)."""
    resident: set = set()
    streamed = list(layers)
    left = float(budget_bytes)
    for i in sorted(layers, key=lambda j: n_exp[j] * bpe[j]):
        full = n_exp[i] * bpe[i]
        min_rest = sum(k * bpe[j] for j in streamed if j != i)
        if left / len(streamed) >= full and left - full >= min_rest:
            resident.add(i)
            streamed.remove(i)
            left -= full
    return resident, left


def load_streaming_model(model_dir: str, budget_bytes: int, k: int = 8,
                         trust_remote_code: bool = False,
                         tokenizer_config: dict | None = None,
                         resident: str | None = None):
    """Load with lazy=True, wrap MoE layers, materialize ONLY the trunk.

    The replaced SwitchGLU modules are dropped before any eval, so expert
    tensors are never materialized — resident memory = trunk + pools. Budget
    is split per layer by that layer's actual expert byte size (same
    equal-bytes policy as sim/lru_sim.py).
    """
    from mlx_lm import load as _load

    tok_cfg = dict(tokenizer_config or {})
    if trust_remote_code:
        tok_cfg["trust_remote_code"] = True
    model, tokenizer = _load(model_dir, lazy=True, tokenizer_config=tok_cfg)
    index = SafetensorsIndex(model_dir)
    reader = ExpertReader(index)
    moe_layers = [i for i, l in enumerate(model.layers)
                  if hasattr(getattr(l, "mlp", None), "switch_mlp")]
    bpe = {i: index.layer_expert_bytes(i) for i in moe_layers}
    n_exp = {i: index.switch_triples(i)["gate_proj"]["weight"].shape[0]
             for i in moe_layers}
    # Residency is OPT-IN ("auto"): it trades RAM headroom (KV cache for
    # long agent/coding contexts) for speed, so the streaming default stays
    # predictable. serve.py exposes it as --resident.
    mode = resident if resident is not None else os.environ.get(
        "STREAMLX_RESIDENT", "off")
    res_layers: set = set()
    budget_left = float(budget_bytes)
    if mode not in ("off", "0"):
        res_layers, budget_left = plan_residency(moe_layers, bpe, n_exp,
                                                 budget_bytes, k)
    streamed = [i for i in moe_layers if i not in res_layers]
    pools: dict[int, StreamingSwitchMLP] = {}
    for i in streamed:
        slots = max(k, int(budget_left / len(streamed) / bpe[i]))
        pool = StreamingSwitchMLP(model_dir, i, slots, index=index,
                                  reader=reader)
        model.layers[i].mlp.switch_mlp = StreamingSwitchGLU(pool)
        pools[i] = pool
    if res_layers:
        rb = sum(n_exp[i] * bpe[i] for i in res_layers) / 2**30
        print(f"[streamlx] {len(res_layers)} fully-resident MoE layers "
              f"({rb:.1f} GiB, stock path); {len(streamed)} streamed",
              file=sys.stderr)
    # trunk + any fully-resident layers; streamed experts were replaced
    # before eval and never materialize
    mx.eval(model.parameters())
    # Pool slot buffers are plain-object attrs, invisible to parameters();
    # eval them on the loader thread so multi-threaded hosts never touch
    # lazy arrays stream-bound to a thread they don't run on.
    mx.eval([t for p in pools.values()
             for proj in p.pool.values() for t in proj.values()])
    return model, tokenizer, pools, reader


def preload_popular(pools: dict, trace_npz: str) -> int:
    """Warm-start each pool with its layer's most-popular experts (measured
    skew); sorted id order => semi-sequential reads.

    Trace axis -> model layer comes from the sidecar's layer_ids (models with
    dense layers, e.g. kimi_linear layer 0, are NOT axis-identity)."""
    import json
    from pathlib import Path

    d = np.load(trace_npz)
    e = d["experts"]
    p = Path(trace_npz)
    side = p.parent / (p.stem + ".meta.json")
    if side.exists():
        layer_ids = json.loads(side.read_text())["layer_ids"]
    else:
        layer_ids = list(range(e.shape[1]))
    axis_of = {int(lid): i for i, lid in enumerate(layer_ids)}
    total = 0
    for lid, pool in pools.items():
        ax = axis_of.get(lid)
        if ax is None:
            continue
        hist = np.bincount(e[:, ax, :].ravel(), minlength=pool.n_experts)
        top = np.argsort(hist)[::-1][: pool.n_slots]
        pool.ensure(sorted(int(x) for x in top))
        # Slice-updates from ensure() are lazy; eval per pool so buffers are
        # real before another thread (multi-threaded hosts) reads them.
        mx.eval([t for proj in pool.pool.values() for t in proj.values()])
        total += len(top)
    return total


def aggregate_stats(pools: dict) -> dict:
    tot = {"hits": 0, "misses": 0, "evictions": 0, "fetch_s": 0.0,
           "sweep_experts": 0, "sweep_fetch_s": 0.0, "adopted": 0}
    for p in pools.values():
        s = p.stats
        for k in ("hits", "misses", "evictions", "sweep_experts", "adopted"):
            tot[k] += s.get(k, 0)
        tot["fetch_s"] += s["fetch_s"]
        tot["sweep_fetch_s"] += s.get("sweep_fetch_s", 0.0)
    acc = tot["hits"] + tot["misses"]
    tot["miss_rate"] = tot["misses"] / acc if acc else 0.0
    tot["fetch_s"] = round(tot["fetch_s"], 3)
    tot["sweep_fetch_s"] = round(tot["sweep_fetch_s"], 3)
    return tot
