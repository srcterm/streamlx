"""Bounded prefill attention.

For some calls (head size 256, as in the full-attention layers of Qwen3.5/3.6
and Qwen3-Next, prefill chunks after the first) MLX's
mx.fast.scaled_dot_product_attention builds the whole score matrix: heads x
queries x keys, 4 GiB for one 4096-token chunk at 32K of context, more than
the KV cache itself. install() routes the calls that would (probed once per
head sizes, dtype, mask kind and whether queries == keys) to MLX's fused
kernel where it has one (force_fused, MLX >= 0.32.2): no scores at all, float32
softmax over unrounded scores (~3x closer to a float32 reference), 5-29%
faster on 4096-token chunks on an M4 and 2-5% slower on 512-token ones. Else,
or with fused=False, they run as blocks of query rows over all keys, each with
its rows of the mask: per row the same computation, so bitwise the unfused
output, the transient near limit_bytes. Each block is evaluated before the next
is built (queued blocks would all hold their scores at once), so blocking
applies outside mx.compile only; inside it the call runs unblocked.
"""

from __future__ import annotations

import mlx.core as mx

_SDPA = mx.fast.scaled_dot_product_attention
_HAS_FUSED = "force_fused" in (_SDPA.__doc__ or "")
_FUSED = _HAS_FUSED
_LIMIT = 256 * 2**20
_BUILDS: dict = {}   # call class -> whether the stock call builds scores


def _kind(mask) -> str | None:
    if mask is None:
        return "none"
    if isinstance(mask, str):
        return mask if mask == "causal" else None
    return "array" if isinstance(mask, mx.array) else None


def _builds_scores(dq: int, dv: int, dtype, square: bool, kind: str) -> bool:
    """Probe once per call class: does the stock call allocate the score
    matrix (queries x keys) rather than stream it through a kernel?"""
    key = (dq, dv, dtype, square, kind)
    if key not in _BUILDS:
        n, s = (2048, 2048) if square else (512, 4096)
        q = mx.zeros((1, 1, n, dq), dtype)
        k = mx.zeros((1, 1, s, dq), dtype)
        v = mx.zeros((1, 1, s, dv), dtype)
        m = {"none": None, "causal": "causal",
             "array": mx.ones((n, s), mx.bool_)}[kind]
        mx.eval(q, k, v, *([m] if kind == "array" else []))
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(_SDPA(q, k, v, scale=1.0, mask=m))
        _BUILDS[key] = mx.get_peak_memory() - base >= n * s * q.itemsize // 2
    return _BUILDS[key]


def _attend(q, k, v, *, scale, mask=None, **kw):
    _, h, n, d = q.shape
    s = k.shape[-2]
    kind = _kind(mask)
    if (n < 128 or h * n * s * q.itemsize <= _LIMIT or kind is None
            or "force_fused" in kw
            or not _builds_scores(d, v.shape[-1], q.dtype, n == s, kind)):
        return _SDPA(q, k, v, scale=scale, mask=mask, **kw)
    if _FUSED:
        try:
            return _SDPA(q, k, v, scale=scale, mask=mask, force_fused=True,
                         **kw)
        except (RuntimeError, ValueError):   # no fused kernel for these sizes
            pass
    rows = max(64, _LIMIT // (h * s * q.itemsize) // 64 * 64)
    out = []
    try:
        for i in range(0, n, rows):
            j = min(n, i + rows)
            if kind == "causal":   # row r sees keys <= r + s - n
                m = (mx.arange(s)[None]
                     <= mx.arange(i + s - n, j + s - n)[:, None])
            elif kind == "array" and mask.ndim >= 2 and mask.shape[-2] == n:
                m = mask[..., i:j, :]
            else:
                m = mask
            out.append(_SDPA(q[:, :, i:j], k, v, scale=scale, mask=m, **kw))
            mx.eval(out[-1])
    except (RuntimeError, ValueError):  # eval inside mx.compile: unblocked
        return _SDPA(q, k, v, scale=scale, mask=mask, **kw)
    return mx.concatenate(out, axis=2)


def install(limit_bytes: int = _LIMIT, fused: bool = True) -> None:
    """Route mx.fast.scaled_dot_product_attention through _attend, which
    every mlx-lm model calls at call time. fused=False keeps the blocked
    path (bitwise the unfused stock output) even where MLX could fuse."""
    global _LIMIT, _FUSED
    _LIMIT = limit_bytes
    _FUSED = fused and _HAS_FUSED
    mx.fast.scaled_dot_product_attention = _attend
