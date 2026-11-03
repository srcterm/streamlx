"""Generic pipeline adapter for mlx-lm-shaped MoE decoders.

Covers the common mlx-lm layer shape

    r = attn(norm1(x), mask, cache);  h = x + r
    r = mlp(norm2(h));                out = h + r

with a MoE block that calls ``switch_mlp(x, inds)`` somewhere inside
(qwen3_moe, mixtral-likes, laguna, glm4_moe, ...). Nothing about the
block's ROUTING is re-typed here (a host's fused router kernel can differ
from the reference chain by ULPs, enough to walk a greedy sequence off
the host's own output): the
routing indices are captured by running the block itself with the
switch temporarily replaced by a recording stub (the gate matmul and any
shared expert run twice per layer, a few percent of the layer; the
expert gather runs once), and ``finish`` calls the block for real, so
whatever routing op is installed on the block class is the one used.

Drafting follows the ``mtp_forward`` / ``make_mtp_cache`` protocol a host
puts on models with an attached MTP head (``_omlx_mtp_head_prenorm``
decides whether the head wants the pre-norm hidden). Models without a
head get no drafter: the core then runs K=1 (or teacher-forced tests).

Adapters for other layer shapes (hyper-connected residuals, multi-branch
blocks) are host-provided.
"""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

_ATTN = ("self_attn", "attention", "attn")
_NORM1 = ("input_layernorm", "attention_norm", "ln_1", "pre_attention_layernorm")
_NORM2 = ("post_attention_layernorm", "ffn_norm", "ln_2", "pre_mlp_layernorm")
_MLP = ("mlp", "ffn", "feed_forward", "block_sparse_moe")


def _pick(obj, names, what):
    for n in names:
        if getattr(obj, n, None) is not None:
            return n
    raise AttributeError(f"generic pipeline adapter: no {what} among {names} "
                         f"on {type(obj).__name__}")


class _SwitchStub:
    """Stands in for switch_mlp during routing capture: records the
    indices, returns zeros of the gather's shape, touches no pool."""

    def __init__(self, dim: int):
        self.dim = dim
        self.inds = None

    def __call__(self, x, inds, *a, **kw):
        self.inds = inds
        return mx.zeros(tuple(x.shape[:-1]) + (int(inds.shape[-1]), self.dim),
                        dtype=x.dtype)


class StandardMoEAdapter:
    def __init__(self, model, drafter=None):
        self.model = model
        inner = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
        self.inner = inner
        self.layers = list(inner.layers)
        self.embed_tokens = inner.embed_tokens
        self.final_norm = getattr(inner, "norm", None)
        self.lm_head = getattr(model, "lm_head", None)
        first = next(l for l in self.layers if _find_switch(l) is not None)
        self._attn = _pick(first, _ATTN, "attention")
        self._norm1 = _pick(first, _NORM1, "pre-attention norm")
        self._norm2 = _pick(first, _NORM2, "pre-mlp norm")
        self._mlp = _pick(first, _MLP, "mlp/moe block")
        self._switch_attr = {}
        for l in self.layers:
            blk = getattr(l, self._mlp, None)
            self._switch_attr[id(blk)] = _find_switch(l)
        self.drafter = drafter if drafter is not None else _MtpDrafter.maybe(model, self)

    # -- token entry / exit ----------------------------------------------------
    def prep(self, tok: int, offset: int, pos_base=None, i: int = 0):
        ids = mx.array([[tok]])
        return ids, self.embed_tokens(ids), None

    def offset(self, cache) -> int:
        for c in cache:
            for obj in (c, getattr(c, "kv_cache", None)):
                if obj is not None and hasattr(obj, "_idx"):
                    return int(obj._idx)
            off = getattr(c, "offset", None)
            if off is not None:
                return int(off.reshape(-1)[0].item()) if isinstance(off, mx.array) else int(off)
        return 0

    def masks(self, h, cache):
        from mlx_lm.models.base import create_attention_mask

        return create_attention_mask(h, cache[0] if cache else None)

    def logits(self, h):
        out = self.final_norm(h) if self.final_norm is not None else h
        if self.lm_head is not None:
            return self.lm_head(out)
        return self.embed_tokens.as_linear(out)

    def hidden_for_draft(self, h):
        prenorm = getattr(self.model, "_omlx_mtp_head_prenorm", False)
        return h if (prenorm or self.final_norm is None) else self.final_norm(h)

    # -- the layer split -------------------------------------------------------
    def begin(self, l, h, ids, ctx, cache, pos):
        layer = self.layers[l]
        attn = getattr(layer, self._attn)
        r = attn(getattr(layer, self._norm1)(h), ctx, cache[l])
        h2 = h + r
        blk = getattr(layer, self._mlp, None)
        xn = getattr(layer, self._norm2)(h2)
        inds = None
        sw_name = self._switch_attr.get(id(blk))
        if sw_name is not None:
            real = getattr(blk, sw_name)
            stub = _SwitchStub(int(xn.shape[-1]))
            setattr(blk, sw_name, stub)
            try:
                blk(xn)
            finally:
                setattr(blk, sw_name, real)
            inds = stub.inds
        if inds is None:                      # dense layer: nothing to route
            inds = mx.zeros((0,), dtype=mx.uint32)
        return SimpleNamespace(h=h2, xn=xn, inds=inds, blk=blk)

    def finish(self, l, st):
        return st.h + st.blk(st.xn)

    # -- drafting ----------------------------------------------------------------
    @property
    def can_draft(self) -> bool:
        return self.drafter is not None

    def reset_drafter(self) -> None:
        if self.drafter is not None:
            self.drafter.reset()

    def prime_pairs(self, hidden, next_ids) -> None:
        if self.drafter is not None:
            self.drafter.prime_pairs(hidden, next_ids)

    def feed_pairs(self, pairs) -> None:
        """Committed pairs while the pipeline's gate has drafting OFF:
        keep the head's cache exact without proposing."""
        if self.drafter is None or not pairs:
            return
        self.drafter.prime_pairs(mx.concatenate([h for h, _ in pairs], axis=1),
                                 mx.array([[t for _, t in pairs]]))

    def draft(self, pairs, depth: int, spec=None):
        if self.drafter is None:
            return [], [], None
        return self.drafter.draft(pairs, depth, spec)


def _find_switch(layer):
    for name in _MLP:
        blk = getattr(layer, name, None)
        if blk is None:
            continue
        for sw in ("switch_mlp", "experts"):
            if getattr(blk, sw, None) is not None and callable(getattr(blk, sw)):
                return sw
    return None


class _MtpDrafter:
    """The host's MTP-head protocol: model.mtp_forward(hidden, ids, cache,
    return_hidden=True, logits_keep=1) + make_mtp_cache(); chained on the
    head's own hidden for depth > 1; the chain's speculative cache
    entries undone by snapshot/restore."""

    @classmethod
    def maybe(cls, model, adapter):
        if callable(getattr(model, "mtp_forward", None)) and callable(
                getattr(model, "make_mtp_cache", None)):
            try:
                if not model.make_mtp_cache():
                    return None
            except Exception:  # noqa: BLE001
                return None
            return cls(model, adapter)
        return None

    def __init__(self, model, adapter):
        self.model = model
        self.adapter = adapter
        self.cache = None
        self.fed = 0

    def reset(self):
        self.cache = self.model.make_mtp_cache()
        self.fed = 0

    def prime_pairs(self, hidden, next_ids):
        if hidden.shape[1] == 0:
            return
        if self.cache is None:
            self.reset()
        lg = self.model.mtp_forward(self.adapter.hidden_for_draft(hidden), next_ids,
                                    self.cache, logits_keep=1)
        mx.eval(lg)
        self.fed += int(hidden.shape[1])

    def draft(self, pairs, depth, spec=None):
        from .pipeline import restore_cache, sampling_logprobs, snapshot_cache

        if self.cache is None or not pairs or depth <= 0:
            return [], [], None
        rows = self.adapter.hidden_for_draft(mx.concatenate([h for h, _ in pairs], axis=1))
        committed = mx.array([[t for _, t in pairs]])
        logits, head_hidden = self.model.mtp_forward(
            rows, committed, self.cache, return_hidden=True, logits_keep=1)
        self.fed += len(pairs)
        toks, probs, qs = [], [], []
        snap = [snapshot_cache(c) for c in self.cache]
        h = head_hidden[:, -1:]
        for j in range(depth):
            lg = logits[:, -1, :]
            if spec is not None:
                lp = sampling_logprobs(lg, spec)
                tok = mx.random.categorical(lp)
                probs.append(mx.exp(lp[tok]))
                qs.append(lp)
            else:
                tok = mx.argmax(lg, axis=-1)
                probs.append(mx.max(mx.softmax(lg, axis=-1, precise=True), axis=-1))
            toks.append(tok)
            if j + 1 == depth:
                break
            logits, head_hidden = self.model.mtp_forward(
                h, tok.reshape(1, 1), self.cache, return_hidden=True)
            h = head_hidden[:, -1:]
        mx.eval(*toks, *probs, *qs)
        for c, s in zip(self.cache, snap):
            restore_cache(c, s)
        return ([int(t.item()) for t in toks], [float(p.item()) for p in probs],
                qs if spec is not None else None)
