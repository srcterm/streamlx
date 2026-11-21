"""Streaming a checkpoint gives the stock model's logits, whichever of the
file names mlx-lm writes it is saved under."""
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

import toy
from streamlx.integrate import install_streaming
from streamlx.stindex import SafetensorsIndex


class CheckpointFiles(unittest.TestCase):
    def test_sharded_and_single_file(self):
        # mlx-lm saves a small model as one model.safetensors
        for name in (toy.SHARDED, "model.safetensors"):
            with self.subTest(file=name):
                ck = toy.Checkpoint(name)
                self.addCleanup(ck.close)
                stock, _ = load_model(Path(ck.dir))       # mlx-lm's loader
                self.assertTrue(SafetensorsIndex(ck.dir).tensors)
                m, _ = load_model(Path(ck.dir), lazy=True)
                pools, _ = install_streaming(m, ck.dir, 8)
                self.assertEqual(len(pools), toy.ARGS["num_hidden_layers"])
                ids = toy.prompt()
                ca, cb = make_prompt_cache(stock), make_prompt_cache(m)
                a = stock(mx.array([ids]), cache=ca)
                b = m(mx.array([ids]), cache=cb)
                for _ in range(33):        # the prefill, then decode steps
                    mx.eval(a, b)
                    self.assertTrue(bool(mx.array_equal(a[0, -1], b[0, -1])))
                    t = mx.argmax(a[0, -1]).reshape(1, 1)
                    a, b = stock(t, cache=ca), m(t, cache=cb)

    def test_no_checkpoint(self):
        ck = toy.Checkpoint("weights.safetensors")        # not a name it reads
        self.addCleanup(ck.close)
        with self.assertRaises(FileNotFoundError):
            SafetensorsIndex(ck.dir)


if __name__ == "__main__":
    unittest.main()
