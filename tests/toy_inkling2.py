"""A tiny random Inkling 2 for the pipelined decoder's test: mlx-lm's own
inkling2 model class (sliding windows written in place, convolution states,
block residuals; experts in nvfp4 with a scale per expert and in affine
4-bit), in mlx-lm's own checkpoint layout. Importing it needs an mlx-lm
that has the model (see the README's note on Inkling 2).
"""
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import mlx_lm.models.inkling2 as stock
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

import toy  # noqa: F401  (the device switch)
from streamlx.integrate import load_streaming_model

TEXT = dict(hidden_size=64, num_hidden_layers=6, vocab_size=256,
            unpadded_vocab_size=250, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4,
            swa_num_key_value_heads=2, swa_head_dim=16,
            sliding_window_size=8, local_layer_ids=[0, 1, 3, 4], d_rel=8,
            rel_extent=16, log_scaling_n_floor=4, log_scaling_alpha=0.1,
            rms_norm_eps=1e-6, qk_l2_clip_threshold=22.627416997969522,
            rel_l2_clip_threshold=8.0, sconv_kernel_size=4,
            n_layers_per_attn_res_block=3, n_routed_experts=16,
            num_experts_per_tok=4, intermediate_size=64, mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0, route_scale=8.0,
            final_logit_softcapping=30.0)
VOCAB, K = TEXT["unpadded_vocab_size"], TEXT["num_experts_per_tok"]
NVFP4 = dict(group_size=16, bits=4, mode="nvfp4")
AFF4 = dict(group_size=64, bits=4, mode="affine")
DENSE = dict(group_size=32, bits=8, mode="affine")
NV_LAYERS = (0, 1, 2, 3)          # the others' experts are affine
ROOT = "model.language_model."


def build():
    m = stock.Model(stock.ModelArgs(model_type="inkling2",
                                    text_config=dict(TEXT)))
    keep = m.cast_predicate
    out = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith(".scale") or "norm" in k:
            w = 1 + 0.1 * mx.random.normal(v.shape)
        elif k.endswith(".mlp.gate.weight"):
            w = 0.3 * mx.random.normal(v.shape)
        elif k.endswith("e_score_correction_bias"):
            w = 0.05 * mx.random.normal(v.shape)
        elif v.ndim >= 2:
            w = mx.random.normal(v.shape) * (
                v.shape[-1] ** -0.5 if v.shape[-1] > 1 else 0.3)
        else:
            w = 0.5 * mx.random.normal(v.shape)
        out.append((k, w.astype(mx.bfloat16) if keep(k) else w))
    m.load_weights(out, strict=True)

    def pred(path, mod):
        if path.endswith(("gate_up_proj", "down_proj")):
            n = int(path.split(".layers.")[1].split(".")[0])
            return (dict(NVFP4, global_scale=True) if n in NV_LAYERS
                    else dict(AFF4))
        return (dict(DENSE) if isinstance(mod, (nn.Linear, nn.Embedding))
                else False)

    nn.quantize(m, class_predicate=pred)
    mx.eval(m.parameters())
    return m


class Checkpoint:
    def __init__(self, seed: int = 0):
        mx.random.seed(seed)
        self.dir = tempfile.mkdtemp(prefix="streamlx-test-inkling2-")
        quant = dict(DENSE)
        for n in range(TEXT["num_hidden_layers"]):
            if n not in NV_LAYERS:
                for p in ("gate_up_proj", "down_proj"):
                    quant[f"{ROOT}layers.{n}.mlp.experts.{p}"] = dict(AFF4)
        mx.save_safetensors(f"{self.dir}/model-00001-of-00001.safetensors",
                            dict(tree_flatten(build().parameters())))
        with open(f"{self.dir}/config.json", "w") as f:
            json.dump({"model_type": "inkling2", "text_config": TEXT,
                       "quantization": quant,
                       "vision_config": {"hidden_size": 32}}, f)

    def stock(self):
        """The checkpoint loaded resident by mlx-lm: the reference."""
        return load_model(Path(self.dir),
                          model_config={"vision_config": None})[0]

    def streaming(self, slots: int):
        """(model, pools) through load_streaming_model; the toy has no
        tokenizer, so mlx_lm.load is stood in for while it loads."""
        real = mlx_lm.load

        def load(path, lazy=False, tokenizer_config=None, model_config=None,
                 **_kw):
            model, _ = load_model(Path(path), lazy=lazy,
                                  model_config=model_config)
            return model, None

        mlx_lm.load = load
        try:
            m, _tok, pools, _reader = load_streaming_model(
                self.dir, 2**20, k=K, model_config={"vision_config": None})
        finally:
            mlx_lm.load = real
        for p in pools.values():
            p.resize(slots)
        return m, pools

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
