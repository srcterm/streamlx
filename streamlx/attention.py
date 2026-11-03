"""Bounded prefill attention.

MLX's fused attention kernel covers some head sizes; for the others (256, as
in the full-attention layers of Qwen3.5/3.6 and Qwen3-Next)
mx.fast.scaled_dot_product_attention builds the whole score matrix: heads x
queries x keys, 4 GiB for one 4096-token prefill chunk at 32K of context,
more than the KV cache itself. install() runs such calls as blocks of query
rows over all keys, each with its rows of the mask: per row the same
computation, so the output is bitwise the unblocked call's, and the
transient stays near limit_bytes. Each block is evaluated before the next is
built (queued blocks would all hold their scores at once), so this applies
outside mx.compile only; inside it the call runs unblocked.
"""

from __future__ import annotations

import mlx.core as mx

_SDPA = mx.fast.scaled_dot_product_attention
_LIMIT = 256 * 2**20
_BUILDS: dict = {}   # (head dim, dtype) -> whether the call builds scores


def _builds_scores(d: int, dtype) -> bool:
    """Probe once per head size: does an unblocked call allocate the score
    matrix (queries x keys) rather than streaming it through a kernel?"""
    key = (d, dtype)
    if key not in _BUILDS:
        q = mx.zeros((1, 1, 64, d), dtype)
        kv = mx.zeros((1, 1, 4096, d), dtype)
        mx.eval(q, kv)
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(_SDPA(q, kv, kv, scale=1.0, mask="causal"))
        _BUILDS[key] = (mx.get_peak_memory() - base
                        >= 64 * 4096 * q.itemsize // 2)
    return _BUILDS[key]


def _blocked(q, k, v, *, scale, mask=None, **kw):
    _, h, n, d = q.shape
    s = k.shape[-2]
    if (n < 128 or h * n * s * q.itemsize <= _LIMIT
            or not (mask is None or isinstance(mask, mx.array)
                    or mask == "causal")
            or not _builds_scores(d, q.dtype)):
        return _SDPA(q, k, v, scale=scale, mask=mask, **kw)
    rows = max(64, _LIMIT // (h * s * q.itemsize) // 64 * 64)
    out = []
    try:
        for i in range(0, n, rows):
            j = min(n, i + rows)
            if isinstance(mask, str):   # causal: row r sees keys <= r + s - n
                m = (mx.arange(s)[None]
                     <= mx.arange(i + s - n, j + s - n)[:, None])
            elif mask is not None and mask.ndim >= 2 and mask.shape[-2] == n:
                m = mask[..., i:j, :]
            else:
                m = mask
            out.append(_SDPA(q[:, :, i:j], k, v, scale=scale, mask=m, **kw))
            mx.eval(out[-1])
    except (RuntimeError, ValueError):  # eval inside mx.compile: unblocked
        return _SDPA(q, k, v, scale=scale, mask=mask, **kw)
    return mx.concatenate(out, axis=2)


def install(limit_bytes: int = _LIMIT) -> None:
    """Route mx.fast.scaled_dot_product_attention through _blocked, which
    every mlx-lm model calls at call time."""
    global _LIMIT
    _LIMIT = limit_bytes
    mx.fast.scaled_dot_product_attention = _blocked
