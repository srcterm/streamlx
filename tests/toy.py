"""A tiny random qwen3_moe checkpoint for the tests: mlx-lm's own model
class with real quantized arithmetic, written to a temporary directory and
streamed through real pools and a real reader. Nothing is downloaded. The
same weights loaded resident by mlx-lm are the reference of every test:
streamlx must give its logits bit for bit.

STREAMLX_TEST_DEVICE=cpu runs the tests on the CPU (a runner without a GPU).
"""
import json
import os
import shutil
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe
from mlx_lm.models.cache import make_prompt_cache

if os.environ.get("STREAMLX_TEST_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)

from streamlx.integrate import install_streaming  # noqa: E402

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


def build(weights: str | None = None):
    """The model with random weights, or with a checkpoint file's."""
    m = qwen3_moe.Model(qwen3_moe.ModelArgs(**ARGS))
    nn.quantize(m, **QUANT)
    if weights is not None:
        m.load_weights(weights)
    mx.eval(m.parameters())
    return m


class Checkpoint:
    """The toy on disk. `ref` is the stock resident model."""

    def __init__(self, filename: str = SHARDED, seed: int = 0):
        mx.random.seed(seed)
        self.dir = tempfile.mkdtemp(prefix="streamlx-test-")
        self.file = os.path.join(self.dir, filename)
        self.ref = build()
        mx.save_safetensors(self.file,
                            dict(tree_flatten(self.ref.parameters())),
                            metadata={"format": "mlx"})
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump({**ARGS, "quantization": QUANT}, f)

    def rewrite(self, seed: int):
        """Replace the weights in place with another random model's (same
        path, same config, same size); returns that model, resident."""
        mx.random.seed(seed)
        m = build()
        mx.save_safetensors(self.file, dict(tree_flatten(m.parameters())),
                            metadata={"format": "mlx"})
        return m

    def streaming(self, slots: int):
        """(model, pools): the checkpoint streamed, `slots` experts a layer
        in memory."""
        m = build(self.file)
        pools, _ = install_streaming(m, self.dir, slots)
        return m, pools

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def prompt(n: int = 12, seed: int = 0) -> list[int]:
    return np.random.default_rng(seed).integers(0, VOCAB, n).tolist()


def greedy(model, ids: list[int], n: int, cache=None):
    """(seq, logits): the model's greedy continuation of `ids`, seq[0] the
    first token after them, and the logits that chose seq[i + 1]."""
    cache = make_prompt_cache(model) if cache is None else cache
    out = model(mx.array([ids]), cache=cache)
    seq, logits = [int(mx.argmax(out[0, -1]).item())], []
    for _ in range(n):
        out = model(mx.array([[seq[-1]]]), cache=cache)
        mx.eval(out)
        logits.append(out[0, -1])
        seq.append(int(mx.argmax(out[0, -1]).item()))
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
