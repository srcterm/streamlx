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

Drafting: an mlx-vlm block drafter (DSpark / DFlash heads, ``pip install
mlx-vlm``; ``load_drafter``), or the ``mtp_forward`` / ``make_mtp_cache``
protocol a host puts on models with an attached MTP head
(``mtp_head_prenorm`` decides whether the head wants the pre-norm
hidden). Models without a head get no drafter: the core then runs K=1 (or
teacher-forced tests).

Adapters for other layer shapes (hyper-connected residuals, multi-branch
blocks) are host-provided.
"""
from __future__ import annotations

from contextlib import contextmanager
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
    # begin() routes with the block's own code (the switch stub), so the
    # pipeline may hand that routing to the block's switch call in finish()
    exact_routing = True

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
        # a block drafter (DSpark, DFlash) reads these layers' outputs,
        # concatenated in its order; other drafters read the last layer's
        drafter = self.drafter
        self.tap_layers = tuple(getattr(drafter, "tap_layers", None) or ())
        self.taps = frozenset(self.tap_layers)

    def views(self, cache):
        """The core's per-layer cache list for the model's prompt cache."""
        return cache

    # -- token entry / exit ----------------------------------------------------
    def prep(self, tok: int, offset: int, pos_base=None, i: int = 0):
        ids = mx.array([[tok]])
        pos = SimpleNamespace(taps={}) if self.taps else None
        return ids, self.embed_tokens(ids), pos

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
        prenorm = getattr(self.model, "mtp_head_prenorm", False)
        return h if (prenorm or self.final_norm is None) else self.final_norm(h)

    # -- the layer split -------------------------------------------------------
    def begin(self, l, h, ids, ctx, cache, pos):
        layer = self.layers[l]
        attn = getattr(layer, self._attn)
        r = attn(getattr(layer, self._norm1)(h), ctx, cache[l])
        h2 = h + r
        blk = getattr(layer, self._mlp, None)
        xn = getattr(layer, self._norm2)(h2)
        return SimpleNamespace(h=h2, xn=xn, inds=self._route(blk, xn), blk=blk,
                               pos=pos)

    def _route(self, blk, xn):
        """The block's routed expert ids for xn, captured by running it with
        the switch replaced by a recording stub (empty for a dense block)."""
        sw_name = self._switch_attr.get(id(blk))
        if sw_name is None:
            return mx.zeros((0,), dtype=mx.uint32)
        real = getattr(blk, sw_name)
        stub = _SwitchStub(int(xn.shape[-1]))
        setattr(blk, sw_name, stub)
        try:
            blk(xn)
        finally:
            setattr(blk, sw_name, real)
        return stub.inds

    def finish(self, l, st):
        h = st.h + st.blk(st.xn)
        if l in self.taps:
            st.pos.taps[l] = h
        return h

    def draft_hidden(self, h, pos):
        """A stream's hidden for its drafter: the tap layers' outputs
        concatenated, else the last layer's output."""
        if not self.taps:
            return h
        return mx.concatenate([pos.taps[i] for i in self.tap_layers], axis=-1)

    def plain(self, tok: int, cache, pos_base=None):
        """The model's own one-token forward, expert prefetch included: the
        pipeline's plain_fn. Returns (logits, the hidden draft_hidden
        gives). Hosts that pass position ids supply their own."""
        want = self.tap_layers or (len(self.layers) - 1,)
        with capture_layer_outputs(self.model, want) as got:
            logits = self.model(mx.array([[tok]]), cache=cache)
        if not self.taps:
            return logits, got[want[0]][-1]
        return logits, mx.concatenate([got[i][-1] for i in self.tap_layers],
                                      axis=-1)

    def forward_taps(self, ids, cache):
        """The model's forward plus the draft head's context for every
        position: the tap layers' outputs, concatenated (a prefill's)."""
        with capture_layer_outputs(self.model, self.tap_layers) as got:
            logits = self.model(ids, cache=cache)
        if not self.taps:
            return logits, None
        return logits, mx.concatenate(
            [mx.concatenate(got[i], axis=1) for i in self.tap_layers], axis=-1)

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


class MistralLarge4Adapter(StandardMoEAdapter):
    """mlx-lm's mistral_large4: attention takes (mask, rope, log_pos, cache,
    shared_kv, prev_topk) and returns (out, topk), and the model threads
    per-token RoPE, latents shared across layers and the DSA indexer's top-k
    through its layers, so each stream carries its own (in `pos`)."""

    def __init__(self, model, drafter=None):
        super().__init__(model, drafter)
        from mlx_lm.models.mistral_large4 import _log_positions

        self._log_positions = _log_positions
        self._dims, self._freqs = self.inner.rope.dims, self.inner.rope.freqs()
        self._cap = getattr(model.text_args, "final_logit_softcapping", None)

    def prep(self, tok: int, offset: int, pos_base=None, i: int = 0):
        ids = mx.array([[tok]])
        dims, freqs = self._dims, self._freqs

        def rope(x):
            return mx.fast.rope(x, dims, traditional=True, base=None,
                                scale=1.0, offset=offset, freqs=freqs)

        pos = SimpleNamespace(rope=rope, log_pos=self._log_positions(offset, 1),
                              shared={}, topk=None, taps={})
        return ids, self.embed_tokens(ids), pos

    def offset(self, cache) -> int:
        return int(cache[0][0].offset)

    def masks(self, h, cache):
        return None                                   # one query per stream

    def logits(self, h):
        out = self.lm_head(self.final_norm(h))
        return mx.tanh(out / self._cap) * self._cap if self._cap else out

    def begin(self, l, h, ids, ctx, cache, pos):
        layer = self.layers[l]
        r, pos.topk = layer.self_attn(layer.input_layernorm(h), ctx, pos.rope,
                                      pos.log_pos, cache[l], pos.shared,
                                      pos.topk)
        h2 = h + r
        xn = layer.post_attention_layernorm(h2)
        return SimpleNamespace(h=h2, xn=xn, inds=self._route(layer.mlp, xn),
                               blk=layer.mlp, pos=pos)


def adapter_for(model, drafter=None):
    """The pipeline adapter for `model`'s decoder shape."""
    mt = getattr(getattr(model, "args", None), "model_type", None)
    if mt == "mistral_large4":
        return MistralLarge4Adapter(model, drafter)
    if mt == "deepseek_v41":
        from .deepseek_v41 import V41Adapter

        return V41Adapter(model, drafter)
    return StandardMoEAdapter(model, drafter)


@contextmanager
def capture_layer_outputs(model, layer_ids):
    """Record model.layers[i]'s output for every i in layer_ids during the
    forwards inside the block: yields {i: [output per call]} (a prefill's
    draft-head context)."""
    layers = model.layers
    want = {id(layers[i]): i for i in layer_ids}
    out = {i: [] for i in layer_ids}
    patched = {type(layers[i]) for i in layer_ids}
    orig = {cls: cls.__call__ for cls in patched}

    def wrap(cls):
        def call(self, *a, **kw):
            res = orig[cls](self, *a, **kw)
            i = want.get(id(self))
            if i is not None:
                out[i].append(res[0] if isinstance(res, tuple) else res)
            return res
        return call

    for cls in patched:
        cls.__call__ = wrap(cls)
    try:
        yield out
    finally:
        for cls, f in orig.items():
            cls.__call__ = f


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


class VlmDrafter:
    """The pipeline's drafter protocol over an mlx-vlm block drafter: it
    drafts a block after the last committed token from the target's
    tap-layer outputs, which the pairs carry (the adapter's draft_hidden).
    Its caches hold committed context only, so rejected drafts need no
    rollback. Context that arrives without a draft (the prompt, gated-off
    steps) waits and goes in with the next block; with every draft layer
    windowed only the last window of it is kept (the head's RoPE is
    relative, so the shift is exact)."""

    def __init__(self, draft, model):
        cfg = draft.config
        self.head = draft
        self.tap_layers = tuple(cfg.target_layer_ids)
        self.depth = int(getattr(cfg, "runtime_block_size", None)
                         or cfg.block_size) - 1
        windows = list(getattr(cfg, "windows", None) or [None])
        self._keep = None if None in windows else max(windows)
        draft.bind(_vlm_target(model, cfg))
        self.cache = None
        self.pending = []

    def reset(self) -> None:
        self.cache = self.head.make_cache()
        self.pending = []

    def prime_pairs(self, hidden, next_ids=None) -> None:
        if hidden.shape[1]:
            self.pending.append(hidden)
            if self._keep is not None:
                tail = mx.concatenate(self.pending, axis=1)[:, -self._keep:]
                self.pending = [tail]

    def draft(self, pairs, depth: int, spec=None):
        if self.cache is None or not pairs or depth <= 0:
            return [], [], None
        hidden = mx.concatenate(self.pending + [h for h, _ in pairs], axis=1)
        self.pending = []
        probs = []

        def greedy(logits):
            probs.append(mx.max(mx.softmax(logits.astype(mx.float32), axis=-1),
                                axis=-1))
            return mx.argmax(logits, axis=-1)

        toks = self.head.draft_block(pairs[-1][1], hidden, self.cache,
                                      min(depth, self.depth) + 1, greedy)
        mx.eval(toks, *probs)
        return (toks.reshape(-1).tolist(), [float(p.item()) for p in probs],
                None)


def _vlm_target(model, cfg):
    """What an mlx-vlm drafter's bind() reads off its target (its own
    model class), for an mlx-lm model: language_model.{args, model,
    lm_head}, args carrying mlx-vlm's text model_type."""
    args = getattr(model, "text_args", None) or model.args
    vlm_args = SimpleNamespace(**{**vars(args),
                                  "model_type": cfg.text_config.model_type})
    return SimpleNamespace(language_model=SimpleNamespace(
        args=vlm_args, model=model.model, lm_head=model.lm_head))


def load_drafter(path: str, model, reader=None):
    """A pipeline drafter for `model` from an mlx-vlm drafter directory, e.g.
    `python -m mlx_vlm.split_mtp --model <checkpoint with its draft head>
    --output <dir> --q-bits 4` (Mistral Large 4's DSpark head). DeepSeek-V4.1
    reads its head from the checkpoint itself: `path` is the model directory,
    `reader` the model's expert reader."""
    if getattr(getattr(model, "args", None), "model_type", None) == "deepseek_v41":
        from .deepseek_v41_dspark import V41DSpark

        return V41DSpark(model, path, reader.index, reader)
    try:
        from mlx_vlm.speculative.drafters import load_drafter as _load
    except ImportError as e:
        raise ImportError("draft heads need mlx-vlm: pip install mlx-vlm") from e
    draft, kind = _load(path)
    if kind != "dflash":
        raise ValueError(f"{path}: an mlx-vlm {kind!r} drafter; the pipeline "
                         f"takes block drafters (DSpark, DFlash)")
    return VlmDrafter(draft, model)
