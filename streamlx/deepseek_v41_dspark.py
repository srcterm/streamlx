"""DeepSeek-V4.1's DSpark draft head for the pipeline, on mlx-vlm's V4.1 modules.

Standalone until mlx-vlm ships a V4.1 drafter (mlx-vlm PR #2678): then delete
this file and the dispatch in pipeline_adapters.load_drafter. Nothing else
imports it.

The checkpoint ships the head as language_model.mtp.{0,1,2}: 3 stages, each a
V4.1 block (hyper-connections, MLA attention, a 128-expert top-3 MoE) whose
attention reads a CONTEXT of target hidden states instead of a causal KV cache,
plus stage 0's main_proj / main_norm (the context projection) and the last
stage's norm, Markov head and confidence head.
  context: main_x = main_norm(main_proj(taps)), taps = the mean over the hc
           copies of the residual entering target layers 37/38/39 (one row per
           committed position); each stage appends its keys (wkv -> kv_norm ->
           RoPE at the position -> FP8) to a 128-position window
  propose: ids = [anchor, noise x 4] at positions [p, p+5) -> embed, hc copies
           (pre = copy 0) -> 3 stages (attention over [context keys, block
           keys], no mask, sinks) -> collapse -> norm -> the target's head;
           token i = argmax(logits_i + Markov(token i-1)), i = 0..4;
           confidence_i = sigmoid(confidence_head([h_i, markov_i]))
Routed experts stream through streamlx pools (index tag mtp.{s}); the rest is
resident (~0.8 GiB). Reads an oQ export's layout and quantization record.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq8
from mlx_vlm.models.deepseek_v41.language import (DeepseekV41Block,
                                                  DeepseekV41MoE, hc_expand)

from .integrate import _wrap
from .pool import StreamingSwitchMLP
from .stindex import DS_PROJ as _W, default_group_size

_HEADS = ("main_proj", "main_norm", "norm", "markov_head", "confidence_head")


def _group(q: dict) -> int:
    return int(q.get("group_size") or default_group_size(q.get("mode", "affine")))


def _shared(name: str) -> str:
    return re.sub(r"^ffn\.shared_experts\.(w[123])",
                  lambda g: f"ffn.shared_experts.{_W[g.group(1)]}", name)


class _Head(nn.Module):
    def __init__(self, dims_in, dims_out):
        super().__init__()
        self.proj = nn.Linear(dims_in, dims_out, bias=False)


class MarkovHead(nn.Module):
    def __init__(self, vocab, rank):
        super().__init__()
        self.embed = nn.Embedding(vocab, rank)
        self.head = nn.Linear(rank, vocab, bias=False)

    def __call__(self, ids):
        e = self.embed(ids)
        return self.head(e), e


class DSparkStage(nn.Module):
    def __init__(self, cfg, stage: int, n_stages: int, n_taps: int):
        super().__init__()
        # mlx-vlm keeps compress_ratios for the trunk only; the stages are
        # uncompressed (local) layers
        c = copy.copy(cfg)
        c.compress_ratios = (list(cfg.compress_ratios)[:cfg.num_hidden_layers]
                             + [0] * n_stages)
        blk = DeepseekV41Block(c, cfg.num_hidden_layers + stage)
        self.blk = blk
        blk.ffn = DeepseekV41MoE(
            cfg, n_routed_experts=cfg.dspark_n_routed_experts,
            num_experts_per_tok=cfg.dspark_num_experts_per_tok)
        if stage == 0:
            self.main_proj = nn.Linear(cfg.hidden_size * n_taps, cfg.hidden_size,
                                       bias=False)
            self.main_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if stage == n_stages - 1:
            self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.markov_head = MarkovHead(cfg.vocab_size, cfg.dspark_markov_rank)
            self.confidence_head = _Head(cfg.hidden_size + cfg.dspark_markov_rank, 1)

    def keys(self, x, start: int):
        """Context / block keys at [start, start + L): the attention's kv path."""
        a = self.blk.attn
        b, n = x.shape[0], x.shape[1]
        kv = a.kv_norm(a.wkv(fq8(x)))
        kv = a.rope(kv[:, None], start).reshape(b, n, a.head_dim)
        return fq8(kv)

    def attend(self, x, ctx, start: int):
        a = self.blk.attn
        b, n = x.shape[0], x.shape[1]
        qr = a.q_norm(a.wq_a(fq8(x)))
        q = a.wq_b(fq8(qr)).reshape(b, n, a.n_heads, a.head_dim).transpose(0, 2, 1, 3)
        q = a.rope(q, start)
        kv = self.keys(x, start)
        if ctx is not None:
            kv = mx.concatenate([ctx, kv], axis=1)
        kv = kv[:, None].astype(mx.float32)
        out = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), kv, kv, scale=a.scale,
            sinks=a.attn_sink.astype(mx.float32)).astype(q.dtype)
        out = a.rope(out, start, inverse=True)
        out = out.reshape(b, a.o_groups, -1, n, a.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = a.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
        return a.wo_b(fq8(out))

    def __call__(self, h, pre, ctx, start: int):
        k = self.blk
        ap, ao, ac = k.hc_mixes(h, k.hc_attn_fn, k.hc_attn_scale, k.hc_attn_base)
        x = self.attend(k.attn_norm(DeepseekV41Block.hc_pre(h, pre)), ctx, start)
        h = hc_expand(x, h, ao, ac)
        fp, fo, fc = k.hc_mixes(h, k.hc_ffn_fn, k.hc_ffn_scale, k.hc_ffn_base)
        x = k.ffn(k.ffn_norm(DeepseekV41Block.hc_pre(h, ap)), None)
        return hc_expand(x, h, fo, fc), fp


class V41DSpark:
    """The pipeline's drafter protocol over the checkpoint's DSpark head."""

    def __init__(self, model, model_dir: str, index, reader, expert_slots: int = 16):
        self.lm = model.language_model
        cfg = self.lm.config
        root = Path(model_dir)
        raw = json.loads((root / "config.json").read_text())
        spec = (raw.get("omlx_deepseek_v41") or {}).get("quantized_modules")
        if not spec:
            raise ValueError(f"{model_dir}: no oQ quantization record, the "
                             f"V4.1 DSpark head reads that layout")
        for k in ("dspark_block_size", "dspark_noise_token_id",
                  "dspark_target_layer_ids", "dspark_markov_rank",
                  "dspark_n_routed_experts", "dspark_num_experts_per_tok",
                  "num_nextn_predict_layers"):
            setattr(cfg, k, raw["text_config"][k])
        self.cfg = cfg
        self.tap_layers = tuple(cfg.dspark_target_layer_ids)
        self.block = int(cfg.dspark_block_size)
        self.window = int(cfg.sliding_window)
        n = int(cfg.num_nextn_predict_layers)
        self.stages = [DSparkStage(cfg, s, n, len(self.tap_layers)) for s in range(n)]
        self._load(root, spec)
        # routed experts: pools keyed past the trunk, like MTP stages in
        # load_streaming_model
        self.pools = {}
        for s, st in enumerate(self.stages):
            key = cfg.num_hidden_layers + s
            pool = StreamingSwitchMLP(model_dir, key, expert_slots, index=index,
                                      reader=reader, tag=f"mtp.{s}")
            _wrap(st.blk, st.blk.ffn, pool)
            self.pools[key] = pool
        mx.eval([st.parameters() for st in self.stages])
        self.reset()

    def _load(self, root: Path, spec: dict) -> None:
        wmap = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        per_stage = [{} for _ in self.stages]
        shards = {}

        def name_of(body: str) -> str:
            body = _shared(body)
            return body if body.startswith(_HEADS) else "blk." + body

        for k, f in wmap.items():
            m = re.match(r"language_model\.mtp\.(\d+)\.(.+)$", k)
            if m is None or ".ffn.experts." in k:     # routed experts stream
                continue
            if f not in shards:
                shards[f] = mx.load(str(root / f))
            per_stage[int(m.group(1))][name_of(m.group(2))] = shards[f][k]
        for s, st in enumerate(self.stages):
            q = {}
            for mod, qs in spec.items():
                m = re.match(rf"language_model\.mtp\.{s}\.(.+)$", mod)
                if m is not None and ".ffn.experts." not in mod:
                    q[name_of(m.group(1))] = dict(
                        group_size=_group(qs), bits=int(qs["bits"]),
                        mode=qs.get("mode", "affine"))

            def pred(path, mod, q=q):
                return q.get(path, False) if hasattr(mod, "to_quantized") else False

            nn.quantize(st, class_predicate=pred)
            w = per_stage[s]
            # what mlx-vlm's sanitize does to every trunk layer: MultiLinear
            # wo_a, FP32 router weights
            k = "blk.attn.wo_a.weight"
            if k in w and w[k].ndim == 2:
                w[k] = w[k].reshape(self.cfg.o_groups, self.cfg.o_lora_rank, -1)
            k = "blk.ffn.gate.weight"
            if k in w:
                w[k] = w[k].astype(mx.float32)
            shapes = dict(tree_flatten(st.parameters()))
            bad = [(n, tuple(v.shape), tuple(shapes[n].shape)) for n, v in w.items()
                   if n in shapes and tuple(v.shape) != tuple(shapes[n].shape)]
            if bad:
                raise ValueError(f"DSpark stage {s}: shape mismatches {bad[:4]}")
            missing = [k for k, _ in tree_flatten(st.parameters())
                       if k not in w and ".switch_mlp." not in k]
            if missing:
                raise ValueError(f"DSpark stage {s}: missing {missing[:6]}")
            st.load_weights(list(w.items()), strict=False)

    # -- drafter protocol --------------------------------------------------
    def reset(self):
        self.ctx = [None] * len(self.stages)
        self.pos = 0                     # context positions appended so far
        self.pending = []

    def _append(self, taps):
        if taps is None or taps.shape[1] == 0:
            return
        x = self.stages[0].main_norm(self.stages[0].main_proj(fq8(taps)))
        start = self.pos
        for s, st in enumerate(self.stages):
            k = st.keys(x, start)
            c = k if self.ctx[s] is None else mx.concatenate([self.ctx[s], k], axis=1)
            self.ctx[s] = c[:, -self.window:]
        self.pos += taps.shape[1]

    def prime_pairs(self, hidden, next_ids=None):
        if hidden is None or not hidden.shape[1]:
            return
        self.pending.append(hidden)
        n = sum(h.shape[1] for h in self.pending)
        if n >= 2 * self.window:
            # only the last window reaches the context; the rows before it
            # just advance the position
            tail = mx.concatenate(self.pending, axis=1)[:, -self.window:]
            mx.eval(tail)
            self.pending = [tail]
            self.pos += n - self.window

    def draft(self, pairs, depth):
        if not pairs or depth <= 0:
            return [], []
        hs = self.pending + [h for h, _ in pairs if h is not None]
        self.pending = []
        if hs:
            self._append(mx.concatenate(hs, axis=1))
        cfg = self.cfg
        anchor = int(pairs[-1][1])
        ids = mx.array([[anchor] + [cfg.dspark_noise_token_id] * (self.block - 1)])
        h = mx.repeat(self.lm.embed_tokens(ids)[..., None, :], cfg.hc_mult, -2)
        pre = mx.broadcast_to((mx.arange(cfg.hc_mult) == 0).astype(mx.float32),
                              h.shape[:-1])
        for s, st in enumerate(self.stages):
            h, pre = st(h, pre, self.ctx[s], self.pos)
        final = self.stages[-1]
        hf = DeepseekV41Block.hc_pre(h, pre)
        logits = self.lm.head(final.norm(hf))
        toks, embs = [], []
        prev = mx.array([anchor])
        n = min(depth, self.block)
        for i in range(n):
            bias, e = final.markov_head(prev)
            tok = mx.argmax(logits[:, i] + bias, axis=-1)
            toks.append(tok)
            embs.append(e)
            prev = tok
        conf = mx.sigmoid(final.confidence_head.proj(mx.concatenate(
            [hf[:, :n], mx.stack(embs, axis=1).astype(hf.dtype)],
            axis=-1).astype(mx.float32)))
        mx.eval(toks, conf)
        return [int(t.item()) for t in toks], conf.reshape(-1).tolist()
