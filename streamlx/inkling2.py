"""Inkling 2 through mlx-lm's inkling2: the loader for an oQ export.

mlx-lm builds the model. A checkpoint saved in mlx-lm's own layout
(mlx-community's conversions) needs nothing from here, nor does a published
source checkpoint (Inkling 2 Mini's): the shard index reads its experts
(`w13_weight` / `w2_weight`, NVFP4 or MXFP8) with their bytes as they are,
and the global scale mlx-lm derives from them at load comes from the stock
module. An oQ export stores the same weights under oMLX's runtime layout,
which `load_inkling2` maps onto mlx-lm's parameter tree without
requantizing:
  - renames (attn -> self_attn, mlp.router_weight -> mlp.gate.weight, ...);
  - the attention projections oQ stacks row-wise into one quantized matrix,
    split back by rows (a row keeps its own scales and biases, so a slice
    is exact):
      sliding layers  attn.in_proj = q | k | v | r | gate | sink
      full layers     attn.q_proj  = q | r | gate | sink
                      attn.k_proj  = k | k_sconv.weight_proj   (v likewise)
  - conv weights and attention-residual queries reshaped;
  - the looped layers' message gates (the Mini's), stacked over the four
    inputs a gate feeds (query, key, value, mlp): loop_down [4, 2, dims,
    rank] = each gate's down_proj, transposed and split into the hidden
    state's and the message's halves; loop_gain_forget [4, rank, 2 * dims]
    = its message_gate_proj | forget_gate_proj, transposed; and the layer's
    four message queries, loop_message.query [4, dims] (one row of mlx-lm's
    message_input).
The routed experts (w13 = gate and up in one projection, w2 = down) keep
their bytes (NVFP4 ones with their global scale). mlx-lm reads every module's bits, group size and mode off its
packed shapes, so the export's quantization table needs no translation.
Text only: the vision tower is not built and its tensors are dropped.

Also the pipeline's adapter for the model (`Inkling2Adapter`): its layers
pass the residual blocks along and store four convolution states, the last
one after the expert call. The looped Mini's (`Inkling2LoopAdapter`) checks
a step's drafts in one forward of mlx-lm's own loop instead.
"""
from __future__ import annotations

import inspect
import re
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx

from .pipeline import snapshot_cache
from .pipeline_adapters import StandardMoEAdapter, capture_layer_outputs, token_ids

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_ROOT = "model.language_model."
_TOP = {"model.embed.": _ROOT + "embed_tokens.",
        "model.embed_norm.": _ROOT + "embed_norm.",
        "model.norm.": _ROOT + "norm.",
        "lm_head.": "lm_head."}
# within a layer: tensors that only change their name
_RENAMES = {"attn.o_proj.": "self_attn.o_proj.",
            "attn.q_scale": "self_attn.q_l2_clip.scale",
            "attn.r_scale": "self_attn.rel_l2_clip.scale",
            "attn.rel_proj": "self_attn.rel_logits_proj.proj",
            "attn.k_static": "self_attn.k_sconv.static_weight",
            "attn.v_static": "self_attn.v_sconv.static_weight",
            "attn_norm.": "input_layernorm.",
            "q_input_norm.": "q_input_layernorm.",
            "k_input_norm.": "k_input_layernorm.",
            "v_input_norm.": "v_input_layernorm.",
            "mlp_norm.": "post_attention_layernorm.",
            "mlp.router_weight": "mlp.gate.weight",
            "mlp.router_bias": "mlp.gate.e_score_correction_bias",
            "mlp.experts.w13.": "mlp.experts.gate_up_proj.",
            "mlp.experts.w2.": "mlp.experts.down_proj."}
# [dims, kernel] -> mlx's conv1d weight [dims, kernel, 1]
_CONVS = {"attn.k_conv": "self_attn.k_sconv.conv1d.weight",
          "attn.v_conv": "self_attn.v_sconv.conv1d.weight",
          "attn_conv": "post_attention_convolution.conv1d.weight",
          "mlp_conv": "post_mlp_convolution.conv1d.weight"}
# [n, dims] -> [dims] for one query, as it is for a full layer's three
_QUERIES = {"attn_input.query": "attention_input.query",
            "mlp_input.query": "mlp_input.query"}
# a looped layer's four message gates, in the order oQ stacks them
_GATES = ("q", "k", "v", "mlp")


def _splits(text: dict, sliding: bool) -> dict:
    """{oQ projection: [(mlx-lm module under self_attn, rows)]} of a layer."""
    pre = "swa_" if sliding else ""
    heads = text[pre + "num_attention_heads"]
    kv = text[pre + "num_key_value_heads"]
    q, kvd = heads * text[pre + "head_dim"], kv * text[pre + "head_dim"]
    r = heads * text["d_rel"]
    if sliding:
        return {"in_proj": [("q_proj", q), ("k_proj", kvd), ("v_proj", kvd),
                            ("r_proj", r), ("gate_proj", q),
                            ("sink_proj", heads)]}
    conv = kv * text["sconv_kernel_size"]
    return {"q_proj": [("q_proj", q), ("r_proj", r), ("gate_proj", q),
                       ("sink_proj", heads)],
            "k_proj": [("k_proj", kvd), ("k_sconv.weight_proj", conv)],
            "v_proj": [("v_proj", kvd), ("v_sconv.weight_proj", conv)]}


def from_oq_export(weights: dict, text: dict) -> dict:
    """An oQ export's tensors under mlx-lm's names (text model only). `text`
    is the checkpoint's text_config."""
    local = set(text["local_layer_ids"])
    out, messages = {}, {}
    for key, v in weights.items():
        if key.startswith("vision_tower."):
            continue
        m = _LAYER.match(key)
        if m is None:
            if key == "model.final_input.query":
                out[_ROOT + "final_input.query"] = v.reshape(-1)
                continue
            new = next((n + key[len(o):] for o, n in _TOP.items()
                        if key.startswith(o)), None)
            if new is None:
                raise ValueError(f"unmapped Inkling 2 tensor {key}")
            out[new] = v
            continue
        n, rest = int(m.group(1)), m.group(2)
        pre = f"{_ROOT}layers.{n}."
        proj, _, part = rest.removeprefix("attn.").rpartition(".")
        rows_of = _splits(text, n in local).get(proj)
        if rest.startswith("attn.") and rows_of is not None:
            a = 0
            for name, rows in rows_of:
                out[f"{pre}self_attn.{name}.{part}"] = v[a:a + rows]
                a += rows
            if a != v.shape[0]:
                raise ValueError(f"{key}: {v.shape[0]} rows, expected {a}")
        elif rest in _CONVS:
            out[pre + _CONVS[rest]] = v[..., None]
        elif rest in _QUERIES:
            out[pre + _QUERIES[rest]] = v.reshape(-1) if v.shape[0] == 1 else v
        elif rest == "loop_down":
            for i, g in enumerate(_GATES):
                out[f"{pre}{g}_message_gate.down_proj.weight"] = (
                    v[i].reshape(-1, v.shape[-1]).T)
        elif rest == "loop_gain_forget":
            dims = v.shape[-1] // 2
            for i, g in enumerate(_GATES):
                out[f"{pre}{g}_message_gate.message_gate_proj.weight"] = (
                    v[i, :, :dims].T)
                out[f"{pre}{g}_message_gate.forget_gate_proj.weight"] = (
                    v[i, :, dims:].T)
        elif rest == "loop_message.query":
            messages[n] = v
        else:
            new = next((n_ + rest[len(o):] for o, n_ in _RENAMES.items()
                        if rest.startswith(o)), None)
            if new is None:
                raise ValueError(f"unmapped Inkling 2 tensor {key}")
            out[pre + new] = v
    if messages:
        # the looped layers in order: message_input.query [layers, 4, dims]
        out[_ROOT + "message_input.query"] = mx.stack(
            [messages[n] for n in sorted(messages)])
    return out


def is_oq_export(cfg: dict) -> bool:
    """Whether an inkling2 checkpoint (the Mini's: inkling2_loop) is in oQ's
    layout (its quantization table names oQ's modules)."""
    return (cfg.get("model_type") in ("inkling2", "inkling2_loop")
            and "model.embed" in (cfg.get("quantization") or {}))


def load_inkling2(model_dir, tokenizer_config=None, model_config=None):
    """load_streaming_model's loader for an oQ export of Inkling 2."""
    import json
    from pathlib import Path

    import mlx_lm.models.inkling2 as ink
    from mlx_lm import load

    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    text = cfg["text_config"]
    sanitize = ink.Model.sanitize

    def oq_sanitize(self, weights):
        return sanitize(self, from_oq_export(weights, text))

    ink.Model.sanitize = oq_sanitize
    try:
        return load(str(model_dir), lazy=True,
                    tokenizer_config=tokenizer_config or {},
                    model_config={"vision_config": None,
                                  **(model_config or {})})
    finally:
        ink.Model.sanitize = sanitize


class _View:
    """Layer `layer`'s entry of the prompt cache (its KV cache and its four
    convolution states), as the pipeline's cache handle."""

    def __init__(self, entry, layer: int):
        self.entry, self.layer = entry, layer


class Inkling2Adapter(StandardMoEAdapter):
    """The pipeline's adapter for mlx-lm's inkling2. begin() is the layer up
    to its router (the block's own, exact_routing), finish() the experts, the
    post-MLP convolution and the residual; each stream carries the residual
    blocks so far. The model's logits come from the last layer's finish()
    (the mix over the blocks and the final norm) and logits().

    Cache: a stream's begin() stores the layer's first three convolution
    states (the next stream's begin() reads them), its finish() the fourth,
    which may be after the next stream's begin() and so after the snapshot a
    rollback restores: restore_cache then puts back what the stream before the
    rejected one stored. A sliding layer's window is written in place, so its
    snapshot holds the arrays under handles of its own.
    """
    verify = None       # steps run as streams
    # finish() keeps the fourth states of this many positions a layer, for
    # restore_cache: a step of more streams could not roll back to its first
    max_streams = 8

    def __init__(self, model, drafter=None):
        try:    # mlx-lm's layout from 11-19 (#2082): the model is a package
            from mlx_lm.models.inkling2 import layers as stock

            self._rms = stock.unweighted_rms_norm
        except ImportError:
            import mlx_lm.models.inkling2 as stock

            self._rms = stock._unweighted_rms_norm
        self._conv_state = stock._conv_state
        self.model = model
        self.lm = model.model.language_model
        self.layers = list(model.layers)
        text = model.args.text
        self._cap = text.final_logit_softcapping
        self._vocab = text.unpadded_vocab_size
        self._keep = text.sconv_kernel_size - 1
        self.drafter = drafter
        # a draft head reads the residual ENTERING these layers and, for an
        # id past the stack, the model's final normalized output
        self.tap_layers = tuple(getattr(drafter, "tap_layers", None) or ())
        self.taps = frozenset(self.tap_layers)
        # mlx-lm's looped Inkling 2 (#2086) hands _self_attention the loop's
        # messages too; a stack without a loop passes none
        sa = inspect.signature(type(self.layers[0])._self_attention)
        self._no_messages = (None,) if "messages" in sa.parameters else ()
        self._dated = {}     # layer -> the snapshot its next begin() dates
        self._late = {}      # layer -> {position: its fourth conv state}

    # -- cache handles -----------------------------------------------------
    def views(self, cache):
        return [_View(c, i) for i, c in enumerate(cache)]

    def offset(self, views) -> int:
        return int(views[0].entry[0].offset)

    def snapshot_cache(self, v):
        kv, conv = v.entry[0], v.entry[1]
        ring = hasattr(kv, "_idx")
        snap = {"kv": ({k: mx.array(x) if isinstance(x, mx.array) else x
                        for k, x in vars(kv).items()} if ring
                       else snapshot_cache(kv)),
                "ring": ring, "conv": list(conv.cache),
                "lengths": conv.lengths, "left": conv.left_padding,
                "pos": None}
        self._dated[v.layer] = snap
        return snap

    def restore_cache(self, v, snap) -> None:
        kv, conv = v.entry[0], v.entry[1]
        if snap["ring"]:
            for k, x in snap["kv"].items():
                setattr(kv, k, mx.array(x) if isinstance(x, mx.array) else x)
        else:
            for k, x in snap["kv"][0].items():
                setattr(kv, k, x)
        conv.cache = list(snap["conv"])
        conv.lengths, conv.left_padding = snap["lengths"], snap["left"]
        # the stream before the rejected one may have finished this layer
        # after the snapshot: its fourth state is the one to keep
        late = self._late.get(v.layer, {}).get((snap["pos"] or 0) - 1)
        if late is not None:
            conv.cache[3] = late

    # -- token entry / exit ------------------------------------------------
    def prep(self, tok: int, offset: int):
        ids = mx.array([[tok]])
        pos = SimpleNamespace(offset=offset, blocks=None, taps={})
        return ids, self.lm.embed(ids), pos

    def masks(self, h, cache):
        return cache[0].entry[1].make_mask(h.shape[1])

    def logits(self, h):
        out = self.model.lm_head(h).astype(mx.float32)
        out = mx.tanh(out / self._cap) * self._cap
        return out[..., :self._vocab]

    # -- the layer split ---------------------------------------------------
    def begin(self, l, h, ids, ctx, cache, pos):
        layer, entry = self.layers[l], cache[l].entry
        snap = self._dated.pop(l, None)
        if snap is not None:
            snap["pos"] = pos.offset
        if l in self.taps:
            pos.taps[l] = h
        conv, dtype = entry[1], h.dtype
        win = list(conv.cache)
        normed = self._rms(h.astype(mx.float32))
        inputs = None
        if layer.attention_input is not None:
            inputs = layer.attention_input(pos.blocks, normed)
        x = layer._self_attention(h, inputs, entry, win, ctx,
                                  *self._no_messages)
        x, win[2] = layer.post_attention_convolution(x, win[2], ctx)
        res = x = x.astype(mx.float32)
        blocks = pos.blocks
        if layer.is_block_boundary:
            blk = normed[None]
            blocks = blk if blocks is None else mx.concatenate([blocks, blk])
        else:
            res = x = h.astype(mx.float32) + x
        xn = layer.mlp_input(blocks, self._rms(x))[..., 0, :]
        xn = layer.post_attention_layernorm(xn).astype(dtype)
        for i in range(3):
            conv[i] = self._conv_state(win[i], conv.lengths, self._keep)
        inds, weights = layer.mlp.gate(xn)
        return SimpleNamespace(h=res, xn=xn, inds=inds, weights=weights,
                               blk=layer.mlp, blocks=blocks, conv=conv,
                               ctx=ctx, dtype=dtype, pos=pos)

    def finish(self, l, st):
        layer, pos, x = self.layers[l], st.pos, st.xn
        y = st.blk.experts(x, st.inds).astype(mx.float32)
        y = (y * st.weights[..., None]).sum(axis=-2).astype(x.dtype)
        # the fourth state as the stream before this one left it (its
        # finish() ran since this stream's begin())
        y, win = layer.post_mlp_convolution(y, st.conv.cache[3], st.ctx)
        state = self._conv_state(win, st.conv.lengths, self._keep)
        st.conv[3] = state
        st.conv.advance(x.shape[1])
        late = self._late.setdefault(l, {})
        late[pos.offset] = state
        late.pop(pos.offset - self.max_streams, None)
        h = (st.h + y.astype(mx.float32)).astype(st.dtype)
        pos.blocks = st.blocks
        if l == len(self.layers) - 1:
            h = self.lm.final_input(pos.blocks, self._rms(h.astype(mx.float32)))
            h = self.lm.norm(h[..., 0, :]).astype(st.dtype)
        return h

    def draft_hidden(self, h, pos):
        """A stream's row of the head's context: its taps, the final output
        last."""
        if not self.taps:
            return h
        n = len(self.layers)
        return mx.concatenate([pos.taps[i] if i < n else h
                               for i in self.tap_layers], axis=-1)

    # -- the model's own forward -------------------------------------------
    def plain(self, tok, cache):
        return self.forward_taps(token_ids(tok), [v.entry for v in cache])

    def forward_taps(self, ids, cache):
        """The model's logits plus the head's context for every position: the
        residual entering each tap layer = the output of the layer before
        it, and the final normalized output."""
        n = len(self.layers)
        prev = [i - 1 for i in self.tap_layers if i < n]
        with capture_layer_outputs(self.model, prev) as got:
            hidden = self.lm(ids, cache=cache)
        logits = self.logits(hidden)
        if not self.taps:
            return logits, None
        return logits, mx.concatenate(
            [mx.concatenate(got[i - 1], axis=1) if i < n else hidden
             for i in self.tap_layers], axis=-1)


class Inkling2LoopAdapter(Inkling2Adapter):
    """The adapter for mlx-lm's looped Inkling 2 (Mini: layers 6 to 17 run
    three times a token, each pass with caches of its own). Its drafts are
    checked in one forward of the step's tokens (PipelinedDecoder._mode),
    mlx-lm's own loop: verify() runs it and records every convolution window
    the layers build, rollback() takes a rejected draft's rows off the KV
    caches and each convolution state from its window, at the last token
    kept. A draft head's taps count layer visits: the residual entering
    visit t, past the last visit the model's final normalized output."""
    batched = True

    def __init__(self, model, drafter=None):
        super().__init__(model, drafter)
        self.n_visits = len(self.lm.layer_visits)
        self._classes = {type(layer) for layer in self.layers}
        # the module whose _conv_state the layers call
        self._module = sys.modules[type(self.layers[0]).__module__]
        self._want = {t - 1 for t in self.tap_layers if t < self.n_visits}

    @contextmanager
    def _recording(self, windows=None):
        """During the forward inside: {visit: its output} for the visits the
        taps read and, appended to `windows`, every convolution window in
        the order the layers build them (four a visit)."""
        orig = {c: c.__call__ for c in self._classes}
        outs, count, want = {}, [0], self._want

        def wrap(f):
            def call(layer, *a, **kw):
                res = f(layer, *a, **kw)
                if count[0] in want:
                    outs[count[0]] = res[0]
                count[0] += 1
                return res
            return call

        conv_state = self._module._conv_state

        def recorded(window, lengths, n):
            windows.append(window)
            return conv_state(window, lengths, n)

        for c, f in orig.items():
            c.__call__ = wrap(f)
        if windows is not None:
            self._module._conv_state = recorded
        try:
            yield outs
        finally:
            for c, f in orig.items():
                c.__call__ = f
            self._module._conv_state = conv_state

    def forward_taps(self, ids, cache, windows=None):
        """The model's logits plus the head's rows for every position."""
        if not self.taps and windows is None:
            return self.logits(self.lm(ids, cache=cache)), None
        with self._recording(windows) as got:
            hidden = self.lm(ids, cache=cache)
        logits = self.logits(hidden)
        if not self.taps:
            return logits, None
        return logits, mx.concatenate(
            [got[t - 1] if t < self.n_visits else hidden
             for t in self.tap_layers], axis=-1)

    def verify(self, toks, views):
        """The step's tokens through the model in one forward: (logits
        [1, K, V], the head's rows [1, K, ...], what rollback() needs)."""
        windows = []
        logits, rows = self.forward_taps(mx.array([toks]),
                                         [v.entry for v in views], windows)
        return logits, rows, SimpleNamespace(windows=windows, n=len(toks))

    def rollback(self, views, rec, keep: int) -> None:
        """Keep only the step's first `keep` tokens in the caches. A window
        (RotatingKVCache) holds its rows in order after a forward of several
        tokens, the new ones last, so they come off its end; every cache
        holds the rows that the next token's window reaches back to."""
        r = rec.n - keep
        if r <= 0:
            return
        if len(rec.windows) != 4 * len(views):
            raise RuntimeError(
                f"Inkling 2 rollback: {len(rec.windows)} convolution windows "
                f"for {len(views)} layer visits (mlx-lm's layer changed?)")
        n = self._keep
        for v in views:
            kv, conv = v.entry[0], v.entry[1]
            if conv.lengths is not None:
                raise RuntimeError("Inkling 2 rollback: a batched cache")
            kv.offset -= r
            if hasattr(kv, "max_size"):
                kv._idx -= r
                kv.keys = kv.keys[..., :kv._idx, :]
                kv.values = kv.values[..., :kv._idx, :]
            conv.cache = [w[:, keep:keep + n]
                          for w in rec.windows[4 * v.layer:4 * v.layer + 4]]
        rec.n = keep
