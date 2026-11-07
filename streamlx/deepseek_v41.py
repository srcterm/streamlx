"""DeepSeek-V4.1 with Engram layers, through mlx-vlm's deepseek_v41 (mlx-lm's
refuses them): the checkpoint loader and the pipeline adapter. Needs mlx-vlm.

Loader. mlx-vlm builds the model; `load_v41` only
  - turns an oQ export's quantization record (its per-module bits, group size
    and mode, keyed by checkpoint names) into the table mlx-vlm reads, keyed by
    mlx-vlm's module paths,
  - hands mlx-vlm's sanitize the source layout it takes (see from_oq_export),
  - keeps the Engram tables (114 GiB) out of the parameter tree: mlx-vlm reads
    them row by row from memory maps,
  - reads the token embedding row by row from the checkpoint (SSDEmbedding),
  - wraps the language model as a text model: `layers`, `make_cache`, logits.

Adapter. A block is  [engram] -> hc mix -> collapse -> attn_norm -> attention
                     -> hc expand -> hc mix -> collapse -> ffn_norm
                     -> MoE (routing boundary) -> hc expand
with the residual as hc_mult parallel copies and the next sublayer's collapse
weights (pre_mix) carried from the previous one. begin() runs up to the router,
finish() runs the MoE and the expand, both with the block's own modules.

The model keeps ONE cache object for all layers, with a position (`offset`,
advanced once per forward), a cross-layer handoff (published by source layers
and read by later ones in the same forward) and the Engram n-gram history. With
tokens in flight one layer apart:
  - each token's attention runs with its own offset, and with its own handoff
    swapped in and out around the call;
  - a token's Engram hashes are computed at its layer 0, in token order, so the
    history is sequential;
  - the core's per-layer cache handles are views on the one cache: a snapshot
    copies the layer's sliding window ring (written in place) and keeps the
    rest by reference (rebuilt functionally, or only appended past a published
    prefix); the layer-0 snapshot also carries the offset and Engram history,
    which is the rollback point (taken after the previous token set them);
  - a token's Engram rows are read off the compute thread: its hash ids,
    evaluated with layer 0's routing sync, go to reader threads at finish(0),
    which copy the rows from the memory-mapped tables without read-ahead.
"""
from __future__ import annotations

import json
import mmap
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.deepseek_v41.engram import (OffloadedEngramEmbedding,
                                                _engram_apply_gate,
                                                _engram_gate_score)
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
from mlx_vlm.models.deepseek_v41.language import (DeepseekV41Block, hc_expand,
                                                  make_identity_pre_mix)

from .pipeline_adapters import capture_layer_outputs
from .stindex import default_group_size

_W = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_ROUTED = re.compile(r"\.experts\.(w[123])\.")
_EMBED_KEY = "language_model.embed.weight"
HANDOFF = ("index_k", "candidates", "compress_kv", "topk_idxs")


# -- loader --------------------------------------------------------------------
def oq_quantization(cfg: dict) -> dict | None:
    """config["quantization"] for mlx-vlm from an oQ export's record. Routed
    experts load as ffn.switch_mlp.{gate,down,up}_proj, shared experts as
    shared_experts.{gate,down,up}_proj; the MTP stages are not part of
    mlx-vlm's model."""
    spec = (cfg.get("omlx_deepseek_v41") or {}).get("quantized_modules")
    if not spec:
        return None
    # the defaults never apply: every quantized module is listed
    out: dict = {"group_size": 64, "bits": 4, "mode": "affine"}
    for name, q in spec.items():
        if ".mtp." in name:
            continue
        parts = name.split(".")
        if len(parts) > 2 and parts[-2] == "experts" and parts[-1] in _W:
            parts[-2:] = ["switch_mlp", _W[parts[-1]]]
        elif len(parts) > 2 and parts[-2] == "shared_experts" and parts[-1] in _W:
            parts[-1] = _W[parts[-1]]
        mode = q.get("mode", "affine")
        gs = q.get("group_size") or default_group_size(mode)
        out[".".join(parts)] = {"group_size": int(gs), "bits": int(q["bits"]),
                                "mode": mode}
    return out


def from_oq_export(weights: dict) -> dict:
    """An oQ export keeps DeepSeek's names under `language_model.` and stacks
    the routed experts as experts.w1/2/3, a layout mlx-vlm's sanitize skips:
    give it the source layout it takes (prefix off, stacked experts named
    experts.{gate,down,up}_proj). Vision tensors are dropped (text decode)."""
    out = {}
    for k, v in weights.items():
        if k.startswith(("vision.", "aligner.", "image_")):
            continue
        k = k.removeprefix("language_model.")
        out[_ROUTED.sub(lambda m: f".experts.{_W[m.group(1)]}.", k)] = v
    return out


class SSDEmbedding(nn.Module):
    """A BF16 embedding table looked up from a memory map of its checkpoint
    shard: a token needs one 10 KB row, and the table's 1.23 GiB go to the
    expert cache. Same bits as nn.Embedding over the loaded table."""

    def __init__(self, shard: Path, n_header: int, t: dict):
        super().__init__()
        start, end = t["data_offsets"]
        rows = np.memmap(shard, mode="r", dtype=np.uint16,
                         offset=8 + n_header + start, shape=tuple(t["shape"]))
        if rows.nbytes != end - start:
            raise ValueError(f"{shard}: shape {t['shape']} does not match "
                             f"its {end - start} bytes")
        object.__setattr__(self, "_rows", rows)

    def __call__(self, ids):
        return mx.array(self._rows[np.asarray(ids)]).view(mx.bfloat16)


def _embedding_from_disk(lm, model_dir: str) -> None:
    """Swap an untied BF16 token embedding for SSDEmbedding; the lazy table is
    dropped unread. Other layouts stay as they are."""
    if (lm.embed_tokens.weight.dtype != mx.bfloat16
            or getattr(lm.config, "tie_word_embeddings", False)):
        return
    root = Path(model_dir)
    wmap = json.loads((root / "model.safetensors.index.json").read_text())
    name = wmap["weight_map"].get(_EMBED_KEY)
    if name is None:
        return
    with (root / name).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        t = json.loads(f.read(n))[_EMBED_KEY]
    if t["dtype"] == "BF16":
        lm.embed_tokens = SSDEmbedding(root / name, n, t)


class V41Text(nn.Module):
    """mlx-vlm's deepseek_v41 language model as an mlx-lm-shaped text model."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm
        self.model_type = "deepseek_v41"
        self.args = SimpleNamespace(model_type="deepseek_v41", **{
            k: getattr(lm.config, k)
            for k in ("hidden_size", "num_hidden_layers", "vocab_size",
                      "num_experts_per_tok", "n_routed_experts")})

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(self, inputs, cache=None, **kw):
        return self.language_model(inputs, cache=cache, **kw).logits


def load_v41(model_dir, tokenizer_config=None, model_config=None):
    """load_streaming_model's loader for these checkpoints."""
    import mlx_vlm.models.deepseek_v41.deepseek_v41 as dsm
    import mlx_vlm.utils as vu
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    orig = vu.load_config

    def load_config(path, **kw):
        cfg = orig(path, **kw)
        if not cfg.get("quantization"):
            q = oq_quantization(cfg)
            if q is not None:
                cfg["quantization"] = q
        cfg.update(model_config or {})
        return cfg

    sanitize = dsm.Model.sanitize
    loaded: set = set()

    def oq_sanitize(self, weights):
        if any(".ffn.experts.w" in k for k in weights):
            weights = from_oq_export(weights)
        out = sanitize(self, weights)
        loaded.update(out)
        return out

    vu.load_config, dsm.Model.sanitize = load_config, oq_sanitize
    try:
        # strict=False only for the vision encoder this text model drops;
        # every language-model parameter is checked below
        vlm = vu.load_model(Path(model_dir), lazy=True, strict=False)
    finally:
        vu.load_config, dsm.Model.sanitize = orig, sanitize
    lm = vlm.language_model
    # mlx-vlm also puts the whole lazy tables back into the parameter tree,
    # where evaluating the trunk would read all of them: drop them. Lookups
    # are scattered rows: no read-ahead on the maps
    for layer in lm.layers:
        emb = getattr(getattr(layer, "engram", None), "embed", None)
        if isinstance(emb, OffloadedEngramEmbedding):
            for name in ("weight", "scales", "biases"):
                if name in emb:
                    del emb[name]
            for arr in emb._arrays.values():
                arr._mmap.madvise(mmap.MADV_RANDOM)
    _embedding_from_disk(lm, model_dir)
    want = {f"language_model.{k}" for k, _ in tree_flatten(lm.parameters())}
    missing = sorted(want - loaded)
    if missing:
        raise ValueError(f"{len(missing)} language-model parameters not in "
                         f"the checkpoint: {missing[:8]}")
    tok = load_tokenizer(Path(model_dir),
                         tokenizer_config_extra=tokenizer_config or {})
    return V41Text(lm), tok


# -- Engram rows off the compute thread ----------------------------------------
def _engram_rows(emb, idx):
    """(reader thread) OffloadedEngramEmbedding._read_rows' copies, as
    data-only arrays: no MLX ops here."""
    return {name: mx.array(rows[idx]) for name, rows in emb._arrays.items()}


def _engram_apply(eg, x, rows, token_mask):
    """Engram.__call__ with its lookup's rows already read: the same ops on the
    same values."""
    emb = eg.embed
    sel = {name: a.view(emb._dtypes[name], stream=mx.cpu)
           for name, a in rows.items()}
    if emb._quantization is None:
        looked = sel["weight"]
    else:
        looked = mx.dequantize(sel["weight"], sel["scales"], sel.get("biases"),
                               **emb._quantization, dtype=mx.bfloat16)
    n_cols = looked.shape[-2]
    embeddings = looked.astype(x.dtype).reshape(*looked.shape[:-2],
                                                n_cols * eg.head_dim)
    kv = eg.wkv(fake_quant_fp8_ue8m0(embeddings))
    key, value = mx.split(kv, [eg.hc_mult * eg.dim], axis=-1)
    key = key.reshape(*key.shape[:-1], eg.hc_mult, eg.dim)
    score = _engram_gate_score(x, key, eg.q_weight, eg.k_weight,
                               mx.array(eg.dim ** -0.5, dtype=mx.float32),
                               eg.eps, eg.clamp_value)
    return _engram_apply_gate(x, value, mx.sigmoid(score), token_mask)


# -- adapter -------------------------------------------------------------------
class V41View:
    """Layer l of the shared cache, as the core's per-layer cache handle."""

    def __init__(self, entry, layer: int):
        self.entry, self.layer = entry, layer


class V41Adapter:
    exact_routing = True          # begin() routes with the MoE's own gate

    def __init__(self, model, drafter=None):
        self.model = model
        self.lm = model.language_model
        self.layers = list(self.lm.layers)
        self.cfg = self.lm.config
        self.drafter = drafter
        self.tap_layers = tuple(getattr(drafter, "tap_layers", None) or ())
        self.taps = frozenset(self.tap_layers)
        self.lm._ensure_engram_hash()
        self.image_token_id = self.cfg.image_token_id
        self.engram_layers = [
            (i, l.engram) for i, l in enumerate(self.layers)
            if isinstance(getattr(getattr(l, "engram", None), "embed", None),
                          OffloadedEngramEmbedding)]
        self._eg_pool = None
        if self.engram_layers:
            self._eg_pool = ThreadPoolExecutor(2, thread_name_prefix="engram")

    # -- cache handles -----------------------------------------------------
    def views(self, cache):
        """The core's per-layer cache list for the model's one cache."""
        return [V41View(cache[0], i) for i in range(len(self.layers))]

    def offset(self, views) -> int:
        return int(views[0].entry.offset)

    def snapshot_cache(self, v):
        e, i = v.entry, v.layer
        win = e.window[i]
        snap = {
            "window": None if win is None else mx.array(win),
            "compress": e.compress[i], "keys": e.keys[i],
            "cbuf": e._compress_buffers[i], "kbuf": e._key_buffers[i],
            "kv_state": e.kv_state[i], "score_state": e.score_state[i],
        }
        if i == 0:
            snap["offset"], snap["engram"] = e.offset, e.engram
        return snap

    def restore_cache(self, v, snap) -> None:
        e, i = v.entry, v.layer
        win = snap["window"]
        # re-copied: the ring is written in place
        e.window[i] = None if win is None else mx.array(win)
        e.compress[i], e.keys[i] = snap["compress"], snap["keys"]
        e._compress_buffers[i], e._key_buffers[i] = snap["cbuf"], snap["kbuf"]
        e.kv_state[i], e.score_state[i] = snap["kv_state"], snap["score_state"]
        if "offset" in snap:
            e.offset, e.engram = snap["offset"], snap["engram"]

    # -- token entry / exit ------------------------------------------------
    def prep(self, tok: int, offset: int, pos_base=None, i: int = 0):
        ids = mx.array([[tok]])
        h = self.lm.embed_tokens(ids)
        h = mx.broadcast_to(h[..., None, :],
                            (1, 1, self.cfg.hc_mult, self.cfg.hidden_size))
        pos = SimpleNamespace(offset=offset, handoff=None, hashes=None,
                              pre_mix=make_identity_pre_mix(1, 1, self.cfg.hc_mult),
                              taps={}, engram_rows=None,
                              image_mask=ids == self.image_token_id)
        return ids, h, pos

    def masks(self, h, cache):
        return None

    def logits(self, h):
        return self.lm.head(self.lm.norm(h))

    def draft_hidden(self, h, pos):
        if not self.taps:
            return h
        return mx.concatenate([pos.taps[i] for i in self.tap_layers], axis=-1)

    # -- the layer split ---------------------------------------------------
    def _attention(self, layer, x, e, pos):
        if pos.handoff is not None:
            for name, val in zip(HANDOFF, pos.handoff):
                setattr(e, name, val)
        e.offset = pos.offset
        out = layer.attn(x, e)
        pos.handoff = tuple(getattr(e, name) for name in HANDOFF)
        # what the token leaves behind (the layer-0 rollback point)
        e.offset = pos.offset + 1
        return out

    def begin(self, l, h, ids, ctx, cache, pos):
        layer, e = self.layers[l], cache[l].entry
        mask = ~pos.image_mask
        if l == 0 and self.lm.engram_hash is not None:
            e.offset = pos.offset
            pos.hashes = self.lm.engram_hash(ids, pos.offset, e, token_mask=mask)
            if self._eg_pool is not None:
                mx.async_eval(pos.hashes)     # done by layer 0's routing sync
        if layer.engram is not None and pos.hashes is not None:
            fut = pos.engram_rows.get(l) if pos.engram_rows else None
            if fut is not None:
                h = _engram_apply(layer.engram, h, fut.result(), mask)
            else:
                h = layer.engram(
                    h, pos.hashes[:, :, layer.engram.layer_hash_index, :], mask)
        if l in self.taps:
            # DSpark's target hidden: the hc mean entering the layer
            pos.taps[l] = mx.mean(h, axis=2)
        residual = h
        attn_pre, attn_post, attn_comb = layer.hc_mixes(
            h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        x = DeepseekV41Block.hc_pre(h, pos.pre_mix)
        x = self._attention(layer, layer.attn_norm(x), e, pos)
        x = hc_expand(x, residual, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = layer.hc_mixes(
            x, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
        xf = layer.ffn_norm(DeepseekV41Block.hc_pre(x, attn_pre))
        # DeepseekV41MoE.__call__'s first line, run once: finish() reuses it
        inds, scores = layer.ffn.gate(xf, pos.image_mask)
        return SimpleNamespace(h=x, xn=xf, inds=inds, scores=scores,
                               blk=layer.ffn, post=ffn_post, comb=ffn_comb,
                               pre=ffn_pre, pos=pos)

    def finish(self, l, st):
        # the rest of DeepseekV41MoE.__call__ with begin()'s routing
        blk, x = st.blk, st.xn
        y = blk.switch_mlp(fake_quant_fp8_ue8m0(x), st.inds, st.scores)
        y = y.astype(mx.float32).sum(-2)
        y = (y + blk.shared_experts(x).astype(mx.float32)).astype(x.dtype)
        h = hc_expand(y, st.h, st.post, st.comb)
        st.pos.pre_mix = st.pre
        if l == 0 and self._eg_pool is not None and st.pos.hashes is not None:
            hn = np.asarray(st.pos.hashes)              # evaluated: no wait
            st.pos.engram_rows = {
                i: self._eg_pool.submit(_engram_rows, eg.embed,
                                        hn[:, :, eg.layer_hash_index, :])
                for i, eg in self.engram_layers}
        if l == len(self.layers) - 1:
            h = DeepseekV41Block.hc_pre(h, st.pre)      # the final collapse
        return h

    # -- the model's own forward -------------------------------------------
    def plain(self, tok: int, cache, pos_base=None):
        return self.forward_taps(mx.array([[tok]]), [cache[0].entry])

    def forward_taps(self, ids, cache):
        """The model's forward plus the draft head's context for every
        position: the hc mean of the residual entering each tap layer = of the
        previous layer's output (no tap layer carries an Engram)."""
        prev = [i - 1 for i in self.tap_layers]
        with capture_layer_outputs(self.model, prev) as got:
            logits = self.model(ids, cache=cache)
        if not prev:
            return logits, None
        return logits, mx.concatenate(
            [mx.mean(mx.concatenate(got[i], axis=1), axis=2) for i in prev],
            axis=-1)

    # -- drafting ----------------------------------------------------------
    def reset_drafter(self):
        if self.drafter is not None:
            self.drafter.reset()

    def prime_pairs(self, hidden, next_ids):
        if self.drafter is not None:
            self.drafter.prime_pairs(hidden, next_ids)

    def feed_pairs(self, pairs):
        if self.drafter is not None and pairs:
            self.drafter.prime_pairs(
                None if pairs[0][0] is None
                else mx.concatenate([h for h, _ in pairs], 1),
                mx.array([[t for _, t in pairs]]))

    def draft(self, pairs, depth, spec=None):
        if self.drafter is None:
            return [], [], None
        return self.drafter.draft(pairs, depth, spec)
