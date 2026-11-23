"""A tiny random qwen3_moe checkpoint for the tests: mlx-lm's own model
class with real quantized arithmetic, written to a temporary directory and
streamed through real pools and a real reader. Nothing is downloaded. The
same weights loaded resident by mlx-lm are the reference of every test:
streamlx must give its logits bit for bit.

STREAMLX_TEST_DEVICE=cpu runs the tests on the CPU (a runner without a GPU).
"""
import importlib
import json
import os
import shutil
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

if os.environ.get("STREAMLX_TEST_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)

from streamlx.integrate import install_streaming, resize_pools  # noqa: E402

ARGS = dict(model_type="qwen3_moe", hidden_size=64, num_hidden_layers=4,
            intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, num_experts=32,
            num_experts_per_tok=4, decoder_sparse_step=1, mlp_only_layers=[],
            moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=256,
            rope_theta=10000.0, tie_word_embeddings=False,
            max_position_embeddings=512, norm_topk_prob=True)
QUANT = {"group_size": 32, "bits": 4}
VOCAB = ARGS["vocab_size"]
SHARDED = "model-00001-of-00001.safetensors"

# Tiny configs of other mlx-lm MoE types, the same sizes (test_pipeline's
# LayerShapes: the layer shapes the standard adapter's split reproduces)
_BASE = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
             num_attention_heads=4, vocab_size=VOCAB, rms_norm_eps=1e-6,
             rope_theta=10000.0, max_position_embeddings=512,
             tie_word_embeddings=False)
_MLA = dict(_BASE, kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=16,
            qk_nope_head_dim=32, v_head_dim=32, num_key_value_heads=4,
            moe_intermediate_size=32, n_shared_experts=1, n_routed_experts=8,
            num_experts_per_tok=2, first_k_dense_replace=1, n_group=1,
            topk_group=1)
SHAPES = {
    "qwen3_moe": ARGS,
    "qwen2_moe": dict(_BASE, num_key_value_heads=2, num_experts=8,
                      num_experts_per_tok=2, moe_intermediate_size=32,
                      shared_expert_intermediate_size=64),
    "olmoe": dict(_BASE, num_key_value_heads=2, num_experts=8,
                  num_experts_per_tok=2),
    "glm4_moe": dict(_BASE, moe_intermediate_size=32, norm_topk_prob=True,
                     n_group=1, head_dim=16, topk_group=1, n_shared_experts=1,
                     n_routed_experts=8, routed_scaling_factor=1.0,
                     num_experts_per_tok=2, first_k_dense_replace=1,
                     num_key_value_heads=2, rope_scaling=None,
                     use_qk_norm=True, attention_bias=False,
                     partial_rotary_factor=0.5),
    "deepseek_v2": dict(_MLA, n_group=None, topk_group=None,
                        rope_scaling={"factor": 1.0, "mscale": 1.0,
                                      "mscale_all_dim": 0.0,
                                      "original_max_position_embeddings": 512,
                                      "beta_fast": 32, "beta_slow": 1}),
    "deepseek_v3": _MLA,
    "deepseek_v32": dict(_MLA, index_head_dim=32, index_n_heads=2,
                         index_topk=4),
    "glm4_moe_lite": _MLA,
    "laguna": dict(_BASE, num_key_value_heads=2, head_dim=16, sliding_window=4,
                   num_experts=8, num_experts_per_tok=2,
                   moe_intermediate_size=32,
                   shared_expert_intermediate_size=32),
    "ernie4_5_moe": dict(_BASE, num_key_value_heads=2, use_bias=False,
                         moe_num_experts=8, moe_intermediate_size=32, moe_k=2,
                         moe_num_shared_experts=1),
    # (attention that also returns its keys and values, for the next layer)
    "hunyuan": dict(_BASE, num_key_value_heads=2, attention_bias=False,
                    moe_topk=2, num_experts=16, num_shared_expert=1,
                    use_mixed_mlp_moe=True, use_qk_norm=True, use_cla=True,
                    cla_share_factor=2, moe_intermediate_size=32,
                    rope_scaling={"alpha": 1.0, "factor": 1.0,
                                  "type": "dynamic"}),
}
SHAPES = {t: dict(a, model_type=t) for t, a in SHAPES.items()}


def build(weights: str | None = None, args: dict = ARGS, dtype=None,
          quantize: bool = True):
    """The model with random weights, or with a checkpoint file's: `args`
    its config (ARGS, or one of SHAPES), `dtype` its parameters' before they
    are quantized."""
    mod = importlib.import_module(f"mlx_lm.models.{args['model_type']}")
    m = mod.Model(mod.ModelArgs.from_dict(args))
    if dtype is not None:
        m.set_dtype(dtype)
    if quantize:
        nn.quantize(m, **QUANT)
    if weights is not None:
        m.load_weights(weights)
    mx.eval(m.parameters())
    return m


class Checkpoint:
    """The toy on disk (or another tiny config: `args`, `dtype` as for
    build). `ref` is the stock resident model."""

    def __init__(self, filename: str = SHARDED, seed: int = 0,
                 args: dict = ARGS, dtype=None):
        self.args, self.dtype = args, dtype
        mx.random.seed(seed)
        self.dir = tempfile.mkdtemp(prefix="streamlx-test-")
        self.file = os.path.join(self.dir, filename)
        self.ref = build(args=args, dtype=dtype)
        mx.save_safetensors(self.file,
                            dict(tree_flatten(self.ref.parameters())),
                            metadata={"format": "mlx"})
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump({**args, "quantization": QUANT}, f)

    def rewrite(self, seed: int):
        """Replace the weights in place with another random model's (same
        path, same config, same size); returns that model, resident."""
        mx.random.seed(seed)
        m = build(args=self.args, dtype=self.dtype)
        mx.save_safetensors(self.file, dict(tree_flatten(m.parameters())),
                            metadata={"format": "mlx"})
        return m

    def streaming(self, slots: int):
        """(model, pools): the checkpoint streamed, `slots` experts a layer
        in memory."""
        m = build(self.file, self.args, self.dtype)
        pools, _ = install_streaming(m, self.dir, slots)
        return m, pools

    def held(self):
        """(model, pools): the checkpoint streamed with every expert held."""
        m, pools = self.streaming(8)
        resize_pools(pools, sum(p.n_experts * p.expert_bytes
                                for p in pools.values()))
        return m, pools

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def prompt(n: int = 12, seed: int = 0) -> list[int]:
    return np.random.default_rng(seed).integers(0, VOCAB, n).tolist()


def pick(logits) -> int:
    """mlx-lm's greedy token (generate_step): the argmax of the
    log-probabilities, not of the logits."""
    return int(mx.argmax(logits - mx.logsumexp(logits, keepdims=True)).item())


def greedy(model, ids: list[int], n: int, cache=None):
    """(seq, logits): the model's greedy continuation of `ids`, seq[0] the
    first token after them, and the logits that chose seq[i + 1]."""
    cache = make_prompt_cache(model) if cache is None else cache
    out = model(mx.array([ids]), cache=cache)
    seq, logits = [pick(out[0, -1])], []
    for _ in range(n):
        out = model(mx.array([[seq[-1]]]), cache=cache)
        mx.eval(out)
        logits.append(out[0, -1])
        seq.append(pick(out[0, -1]))
    return seq, logits


def wrong_slots(ref, pools) -> list:
    """(layer, expert) of every resident expert whose slot does not hold
    the checkpoint's bytes."""
    bad = []
    for l, pool in pools.items():
        sw = ref.layers[l].mlp.switch_mlp
        for e, slot in pool.id2slot.items():
            if not all(bool(mx.array_equal(pool.pool[p][q][slot],
                                           getattr(getattr(sw, p), q)[e]))
                       for p in pool.projs for q in pool.parts):
                bad.append((l, e))
    return bad
