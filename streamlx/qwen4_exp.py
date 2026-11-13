"""Qwen3.8-Flash-Next (`qwen4_exp`), through mlx-vlm's model (mlx-lm has none):
the checkpoint loader, its draft head and the pipeline adapter. Needs mlx-vlm.

Loader. mlx-vlm builds the model; `load_qwen4_exp` only
  - keeps the PLE table (30 GiB) out of the parameter tree: mlx-vlm's external
    PLE storage reads the rows a token needs from the checkpoint's byte ranges
    (here with parallel preads of exactly their bytes, not its memmap's page
    faults one at a time). An oQ export stores those rows unscaled, with one
    `weight_scale` for the table, which that storage doesn't know about:
    applied here,
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
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen3_5_moe.language import \
    Qwen3_5MoeSparseMoeBlock as _MoeBlock
from mlx_vlm.models.qwen4_exp.language import (
    _create_qwen3_5_ssm_mask, _create_qwen4_exp_attention_mask)
from mlx_vlm.models.qwen4_exp.ple_storage import (PLE_MARKER,
                                                  build_quantized_ple_manifest)

from .pipeline_adapters import StandardMoEAdapter, _find_switch
from .reader import open_nocache


def _route(blk, x):
    """_MoeBlock.__call__'s routing, op for op: (expert ids, their scores)."""
    gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
    k = blk.top_k
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if blk.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


class _PreadRows:
    """A PLE store's row reads (its _read_rows, no row cache) as one pread per
    row and tensor, of exactly its bytes, on a thread pool. A row is 160
    values at Q4: 80 B of weight, 10 B of scales and 10 B of biases in three
    tensors, so the store's memmap faults three 16 KB pages a row, one at a
    time (16 rows a token); the same bytes, read in parallel. ahead() starts
    the reads of rows a later call asks for, off the compute thread."""

    _ex = None

    def __init__(self, store, workers: int = 32):
        self._ahead = None      # (sorted row ids, their rows, the reads)
        if _PreadRows._ex is None:
            _PreadRows._ex = ThreadPoolExecutor(max_workers=workers,
                                                thread_name_prefix="ple-rows")
        self.workers = workers
        fds, self.starts, self.parts = {}, [], []
        for start, _end, arrays in store._shards:
            parts = []
            for name in store._tensor_names:
                a = arrays[name]
                fd = fds.get(a.filename)
                if fd is None:
                    fd = fds[a.filename] = open_nocache(a.filename)
                parts.append((fd, a.offset, a.shape[1] * a.dtype.itemsize,
                              a.dtype))
            self.starts.append(start)
            self.parts.append(parts)
        self.starts = np.array(self.starts)
        self.cols = [arrays[name].shape[1] for name in store._tensor_names]

    def _start(self, ids):
        """(rows, reads): the arrays the rows of `ids` are being read into."""
        shard = np.searchsorted(self.starts, ids, side="right") - 1
        local = (ids - self.starts[shard]).tolist()
        shard = shard.tolist()
        outs = [np.empty((ids.size, c), p[3])
                for c, p in zip(self.cols, self.parts[0])]

        def read(lo, hi):
            for i in range(lo, hi):
                for out, (fd, off, n, dt) in zip(outs, self.parts[shard[i]]):
                    out[i] = np.frombuffer(os.pread(fd, n, off + local[i] * n),
                                           dt)

        step = max(1, -(-ids.size // (4 * self.workers)))
        return outs, [self._ex.submit(read, lo, min(ids.size, lo + step))
                      for lo in range(0, ids.size, step)]

    def ahead(self, row_ids) -> None:
        ids = np.unique(np.asarray(row_ids, dtype=np.int64).reshape(-1))
        self._ahead = (ids, *self._start(ids))

    def __call__(self, row_ids):
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        at = None
        if self._ahead is not None and ids.size and self._ahead[0].size:
            have, outs, reads = self._ahead
            at = np.minimum(np.searchsorted(have, ids), have.size - 1)
            if not (have[at] == ids).all():
                at = None
        if at is None:
            outs, reads = self._start(ids)
        for f in reads:
            f.result()
        return tuple(outs) if at is None else tuple(o[at] for o in outs)


class _Context:
    """A PLE layer's cache entry as its n-gram embedding reads it (the context
    tokens), written to by nothing."""

    def __init__(self, entry):
        self.entry = entry

    def __getitem__(self, i):
        return None if self.entry is None else self.entry[i]

    def update_window(self, *_a) -> None:
        pass


class _RowIds:
    """Stands in for an n-gram embedding's table: keeps the row ids asked for."""

    ids = None

    def __call__(self, row_ids):
        self.ids = row_ids
        return mx.zeros((*row_ids.shape, 1))


def _ple_layers(layers) -> list:
    return [(i, l) for i, l in enumerate(layers) if "ple" in l]


def _ple_ahead(layers, ids, cache) -> None:
    """Start the row reads the PLE layers will make for token ids `ids`
    [1, T], the next tokens of `cache`'s sequence: the row ids by each
    embedding's own hashing of its context and these tokens, so the reads
    run while the layers below compute."""
    for i, layer in layers:
        emb = layer.ple.ple_embedding
        table = emb.ngram_embedding
        rows = getattr(getattr(table, "table", table), "_read_rows", None)
        if not isinstance(rows, _PreadRows):
            continue
        emb.ngram_embedding = rec = _RowIds()
        try:
            emb(ids, _Context(None if cache is None else cache[i]))
        finally:
            emb.ngram_embedding = table
        rows.ahead(np.asarray(rec.ids))     # a sync on a few integer ops


class _ScaledRows:
    """A PLE table's rows times its checkpoint's weight_scale."""

    def __init__(self, table, scale):
        self.table, self.scale = table, scale

    def __call__(self, row_ids):
        return self.table(row_ids).astype(mx.bfloat16) * self.scale


def _ple_rows(lm, root: Path) -> None:
    """The PLE stores' row reads (_PreadRows) and the oQ table scale."""
    wmap = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"]
    for i, layer in _ple_layers(lm.model.layers):
        emb = layer.ple.ple_embedding
        store = emb.ngram_embedding
        if (getattr(store, "cache_rows", None) == 0
                and getattr(store, "_interleaved", 0) is None):
            store._read_rows = _PreadRows(store)
        key = (f"language_model.model.layers.{i}.ple.ple_embedding."
               "ngram_embedding.weight_scale")
        if key in wmap:
            scale = mx.load(str(root / wmap[key]))[key]
            mx.eval(scale)
            emb.ngram_embedding = _ScaledRows(store, scale)


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

    def forward(self, inputs, cache=None, **kw):
        """The language model's forward, its PLE rows read from the start."""
        if getattr(inputs, "ndim", 0) == 2:
            _ple_ahead(_ple_layers(self.layers), inputs, cache)
        return self.language_model(inputs, cache=cache, **kw)

    def __call__(self, inputs, cache=None, **kw):
        return self.forward(inputs, cache, **kw).logits


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
    _ple_rows(lm, root)
    tok = load_tokenizer(root, tokenizer_config_extra=tokenizer_config or {},
                         eos_token_ids=cfg.get("eos_token_id"))
    return Qwen4ExpText(lm), tok


def load_head(model_dir, model, reader=None):
    """mlx-vlm's drafter on the checkpoint's `mtp.*` tensors. With `reader`
    (the model's expert reader) its MoE layer streams through a pool like a
    trunk layer's (head.streamlx_pools, sized with integrate.resize_pools)
    instead of loading its 512 experts (1.3 GiB), at a decode cost
    (pipeline_adapters.load_drafter's stream_head)."""
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
    if reader is not None:
        from .integrate import stream_moe_blocks

        head.streamlx_pools = stream_moe_blocks(
            head, reader.index, reader, key0=len(model.layers), prefix="mtp.",
            k=cfg["text_config"].get("num_experts_per_tok"))
    del weights
    mx.eval(head.parameters())
    head.eval()
    return head


class Qwen4ExpAdapter(StandardMoEAdapter):
    def __init__(self, model, drafter=None):
        self.model = model
        self.lm = model.language_model
        self.inner = self.lm.model
        self.layers = list(self.inner.layers)
        self.embed_tokens = self.inner.embed_tokens
        self.hc_count = self.lm.args.hc_count
        self._switch_attr = {id(l.mlp): _find_switch(l) for l in self.layers}
        self.drafter = drafter
        self.taps = frozenset()
        self._toks = []         # the step's tokens, from prep() to masks()
        self._ple = _ple_layers(self.layers)
        # begin() computes the block's routing itself and finish() runs the
        # rest of the block once, while the block's __call__ is the one
        # _route() mirrors; otherwise the generic way, the block run twice
        self._own_route = all(type(l.mlp).__call__ is _MoeBlock.__call__
                              for l in self.layers)

    def offset(self, cache) -> int:
        return int(cache[self.inner.fa_idx].offset)

    def prep(self, tok: int, offset: int):
        self._toks.append(tok)
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
        # once a step, every stream prepared: their PLE rows start reading
        toks, self._toks = self._toks, []
        _ple_ahead(self._ple, mx.array([toks]), cache)
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
                             blk=layer.mlp, scores=None)
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

    def plain(self, tok: int, cache):
        return self.forward_taps(mx.array([[tok]]), cache)

    def forward_taps(self, ids, cache):
        out = self.model.forward(ids, cache, return_hidden=True)
        return out.logits, out.hidden_states[0]
