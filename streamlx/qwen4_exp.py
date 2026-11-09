"""Qwen3.8-Flash-Next (`qwen4_exp`), through mlx-vlm's model (mlx-lm has none):
the checkpoint loader, its draft head and the pipeline adapter. Needs mlx-vlm.

Loader. mlx-vlm builds the model; `load_qwen4_exp` only
  - keeps the PLE table (30 GiB) out of the parameter tree: mlx-vlm's external
    PLE storage reads the rows a token needs from the checkpoint's byte ranges.
    An oQ export stores those rows unscaled, with one `weight_scale` for the
    table, which that storage doesn't know about: applied here,
  - turns off mlx-vlm's single-token decode path, which inlines the expert
    gather and so can't stream: decode runs the blocks prefill runs,
  - wraps the language model as a text model: `layers`, `make_cache`, logits.

Draft head. The checkpoint's own `mtp.*` tensors in mlx-vlm's drafter, resident,
at the bits the checkpoint records for them: `load_head`, `--draft-head` being
the model directory.

Adapter. A block is  [PLE] -> hc mix -> attention -> inject -> hc mix
                     -> MoE (routing boundary) -> inject
with the residual as hc_count parallel copies. begin() runs up to the router,
finish() runs the MoE and the inject, both with the block's own modules.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5_moe.language import \
    Qwen3_5MoeSparseMoeBlock as _MoeBlock
from mlx_vlm.models.qwen4_exp.language import (
    _create_qwen3_5_ssm_mask, _create_qwen4_exp_attention_mask)
from mlx_vlm.models.qwen4_exp.ple_storage import (PLE_MARKER,
                                                  build_quantized_ple_manifest)

from .pipeline_adapters import StandardMoEAdapter, _find_switch


def _route(blk, x):
    """_MoeBlock.__call__'s routing, op for op: (expert ids, their scores)."""
    gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
    k = blk.top_k
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if blk.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


class _ScaledRows:
    """A PLE table's rows times its checkpoint's weight_scale."""

    def __init__(self, table, scale):
        self.table, self.scale = table, scale

    def __call__(self, row_ids):
        return self.table(row_ids).astype(mx.bfloat16) * self.scale


def _scale_ple_rows(lm, root: Path) -> None:
    wmap = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"]
    for i, layer in enumerate(lm.model.layers):
        key = (f"language_model.model.layers.{i}.ple.ple_embedding."
               "ngram_embedding.weight_scale")
        if "ple" in layer and key in wmap:
            emb = layer.ple.ple_embedding
            scale = mx.load(str(root / wmap[key]))[key]
            mx.eval(scale)
            emb.ngram_embedding = _ScaledRows(emb.ngram_embedding, scale)


class Qwen4ExpText(nn.Module):
    """mlx-vlm's qwen4_exp language model as an mlx-lm-shaped text model."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm
        self.model_type = "qwen4_exp"
        self.args = SimpleNamespace(model_type="qwen4_exp", **{
            k: getattr(lm.args, k)
            for k in ("hidden_size", "num_hidden_layers", "vocab_size",
                      "num_experts_per_tok", "num_experts")})

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(self, inputs, cache=None, **kw):
        return self.language_model(inputs, cache=cache, **kw).logits


def load_qwen4_exp(model_dir, tokenizer_config=None, model_config=None):
    """load_streaming_model's loader for these checkpoints."""
    import mlx_vlm.utils as vu
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    root = Path(model_dir)
    cfg = json.loads((root / "config.json").read_text())
    orig = vu.load_config
    with tempfile.TemporaryDirectory(prefix="streamlx-ple-") as tmp:
        manifest = Path(tmp) / "ple-store.json"
        build_quantized_ple_manifest(root, manifest)

        def load_config(path, **kw):
            out = orig(path, **kw)
            out["text_config"]["ple_storage"] = {"manifest": str(manifest)}
            out["quantization"] = {k: v for k, v in out["quantization"].items()
                                   if PLE_MARKER not in k}
            out.update(model_config or {})
            return out

        vu.load_config = load_config
        try:
            # strict=False only for the vision encoder this text model drops
            vlm = vu.load_model(root, lazy=True, strict=False)
        finally:
            vu.load_config = orig
    lm = vlm.language_model
    lm._supports_batch_invariant_decode = lambda: False
    _scale_ple_rows(lm, root)
    tok = load_tokenizer(root, tokenizer_config_extra=tokenizer_config or {},
                         eos_token_ids=cfg.get("eos_token_id"))
    return Qwen4ExpText(lm), tok


def load_head(model_dir, model):
    """mlx-vlm's drafter on the checkpoint's `mtp.*` tensors."""
    from mlx_vlm.models.qwen4_exp.fp8 import convert_qwen4_exp_fp8_weights
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import \
        Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import \
        Qwen4ExpMTPDraftModel

    root = Path(model_dir)
    cfg = json.loads((root / "config.json").read_text())
    wmap = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"]
    keys = [k for k in wmap if k.startswith("mtp.")]
    weights = {}
    for shard in {wmap[k] for k in keys}:
        tensors = mx.load(str(root / shard))
        weights.update((k, tensors[k]) for k in keys if wmap[k] == shard)
    head = Qwen4ExpMTPDraftModel(
        Qwen4ExpMTPConfig.from_dict({"text_config": cfg["text_config"]}))
    weights = head.sanitize(convert_qwen4_exp_fp8_weights(weights))
    record = cfg["quantization"]
    default = {k: record[k] for k in ("bits", "group_size", "mode")}

    def predicate(path, _module):
        if f"{path}.scales" not in weights:
            return False
        return dict(record.get(f"mtp.{path}", default))

    nn.quantize(head, class_predicate=predicate)
    head.load_weights(list(weights.items()))
    mx.eval(head.parameters())
    head.eval()
    return head


class Qwen4ExpAdapter(StandardMoEAdapter):
    exact_routing = True

    def __init__(self, model, drafter=None):
        self.model = model
        self.lm = model.language_model
        self.inner = self.lm.model
        self.layers = list(self.inner.layers)
        self.embed_tokens = self.inner.embed_tokens
        self.final_norm = None
        self.hc_count = self.lm.args.hc_count
        self._switch_attr = {id(l.mlp): _find_switch(l) for l in self.layers}
        self.drafter = drafter
        self.tap_layers = ()
        self.taps = frozenset()
        # begin() computes the block's routing itself and finish() runs the
        # rest of the block once, while the block's __call__ is the one
        # _route() mirrors; otherwise the generic way, the block run twice
        self._own_route = all(type(l.mlp).__call__ is _MoeBlock.__call__
                              for l in self.layers)

    def offset(self, cache) -> int:
        return int(cache[self.inner.fa_idx].offset)

    def prep(self, tok: int, offset: int, pos_base=None, i: int = 0):
        ids = mx.array([[tok]])
        h = mx.tile(self.embed_tokens(ids), (1, 1, self.hc_count))
        # the position ids the model gives a decode step (its shape is what
        # the QSA cache stores)
        rd = self.lm._rope_deltas
        pos = mx.arange(1).reshape(1, -1) + (mx.array(offset) if rd is None
                                              else offset + rd)
        pid = self.lm._position_ids
        if pid is not None and pid.ndim == 3:
            pos = mx.broadcast_to(pos[None], (3, 1, 1))
        return ids, h, pos

    def masks(self, h, cache):
        return (_create_qwen4_exp_attention_mask(h, cache[self.inner.fa_idx]),
                _create_qwen3_5_ssm_mask(h, cache[self.inner.ssm_idx]))

    def logits(self, h):
        out = self.inner.hyper_connection_mixer(h)
        if self.lm.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(out)
        return self.lm.lm_head(out)

    def begin(self, l, h, ids, ctx, cache, pos):
        layer, c = self.layers[l], cache[l]
        mask = ctx[1] if layer.is_linear else ctx[0]
        if "ple" in layer:
            h = h + layer.ple(h, ids, c, mask)
        mixed, hyper, weights = layer.attn_hyper_connection(h)
        if layer.is_linear:
            branch = layer.linear_attn(mixed, mask=mask, cache=c)
        else:
            branch = layer.self_attn(mixed, mask=mask, cache=c,
                                     position_ids=pos)
        h = hyper + (branch[..., None, :]
                     * weights[..., None]).reshape(*hyper.shape)
        mixed, hyper, weights = layer.mlp_hyper_connection(h)
        st = SimpleNamespace(mixed=mixed, hyper=hyper, weights=weights,
                             blk=layer.mlp, pos=pos, scores=None)
        if self._own_route:
            st.inds, st.scores = _route(layer.mlp, mixed)
        else:
            st.inds = self._route(layer.mlp, mixed)
        return st

    def finish(self, l, st):
        if st.scores is None:
            branch = st.blk(st.mixed)
        else:   # the rest of the block's __call__, routing reused
            blk = st.blk
            y = blk.switch_mlp(st.mixed, st.inds)
            y = (y * st.scores[..., None]).sum(axis=-2)
            branch = y + (blk._shared_expert_scale(st.mixed)
                          * blk.shared_expert(st.mixed))
        return st.hyper + (branch[..., None, :]
                           * st.weights[..., None]).reshape(*st.hyper.shape)

    def plain(self, tok: int, cache, pos_base=None):
        out = self.lm(mx.array([[tok]]), cache=cache, return_hidden=True)
        return out.logits, out.hidden_states[0]

    def forward_taps(self, ids, cache):
        out = self.lm(ids, cache=cache, return_hidden=True)
        return out.logits, out.hidden_states[0]
