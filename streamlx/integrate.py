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

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .pool import StreamingSwitchMLP, switch_forward
from .reader import ExpertReader
from .stindex import SafetensorsIndex


class StreamingSwitchGLU(nn.Module):
    def __init__(self, pool: StreamingSwitchMLP):
        super().__init__()
        self.pool = pool
        self.record_sink = None  # optional: list collecting (layer, idx np)

    def _remap(self, idx: np.ndarray) -> np.ndarray:
        slots = self.pool.slot_table[idx]
        if (slots == self.pool.SENTINEL).any():
            raise RuntimeError(f"unmapped expert after ensure "
                               f"(layer {self.pool.layer})")
        return slots

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        idx = np.array(indices)  # materializes the gate's selection (sync)
        if self.record_sink is not None:
            self.record_sink.append((self.pool.layer, idx.copy()))
        uniq = list(dict.fromkeys(idx.reshape(-1).tolist()))
        if len(uniq) <= self.pool.n_slots:
            # fast path: whole call fits the pool (always true for decode)
            self.pool.ensure(uniq)
            return switch_forward(x, mx.array(self._remap(idx)),
                                  self.pool.pool, self.pool.qparams)

        # Chunked prefill: process position slices whose expert union fits.
        # Each slice's output MUST be materialized before the next slice's
        # ensure() mutates pool slots (the gather graph reads pool buffers
        # in place); mx.eval per slice enforces that.
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


def load_streaming_model(model_dir: str, budget_bytes: int, k: int = 8,
                         trust_remote_code: bool = False):
    """Load with lazy=True, wrap MoE layers, materialize ONLY the trunk.

    The replaced SwitchGLU modules are dropped before any eval, so expert
    tensors are never materialized — resident memory = trunk + pools. Budget
    is split per layer by that layer's actual expert byte size (same
    equal-bytes policy as sim/lru_sim.py).
    """
    from mlx_lm import load as _load

    tok_cfg = {"trust_remote_code": True} if trust_remote_code else {}
    model, tokenizer = _load(model_dir, lazy=True, tokenizer_config=tok_cfg)
    index = SafetensorsIndex(model_dir)
    reader = ExpertReader(index)
    moe_layers = [i for i, l in enumerate(model.layers)
                  if hasattr(getattr(l, "mlp", None), "switch_mlp")]
    pools: dict[int, StreamingSwitchMLP] = {}
    for i in moe_layers:
        bpe = index.layer_expert_bytes(i)
        slots = max(k, int(budget_bytes / len(moe_layers) / bpe))
        pool = StreamingSwitchMLP(model_dir, i, slots, index=index,
                                  reader=reader)
        model.layers[i].mlp.switch_mlp = StreamingSwitchGLU(pool)
        pools[i] = pool
    mx.eval(model.parameters())  # trunk only; experts were replaced unevaluated
    return model, tokenizer, pools, reader


def preload_popular(pools: dict, trace_npz: str) -> int:
    """Warm-start each pool with its layer's most-popular experts (measured
    skew, notes/phase2a.md); sorted id order => semi-sequential reads."""
    d = np.load(trace_npz)
    e = d["experts"]
    total = 0
    for i, pool in pools.items():
        hist = np.bincount(e[:, i, :].ravel(), minlength=pool.n_experts)
        top = np.argsort(hist)[::-1][: pool.n_slots]
        pool.ensure(sorted(int(x) for x in top))
        total += len(top)
    return total


def aggregate_stats(pools: dict) -> dict:
    tot = {"hits": 0, "misses": 0, "evictions": 0, "fetch_s": 0.0}
    for p in pools.values():
        s = p.stats
        for k in ("hits", "misses", "evictions"):
            tot[k] += s[k]
        tot["fetch_s"] += s["fetch_s"]
    acc = tot["hits"] + tot["misses"]
    tot["miss_rate"] = tot["misses"] / acc if acc else 0.0
    tot["fetch_s"] = round(tot["fetch_s"], 3)
    return tot
