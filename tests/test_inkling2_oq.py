"""An oQ export's looped layers (Inkling 2 Mini) under mlx-lm's names: each
message gate's down, message and forget projections and the layers' message
queries come back as mlx-lm's own checkpoint loader builds them, bit for bit.
No checkpoint: random tensors packed as oQ packs them."""
import unittest

import mlx.core as mx
import numpy as np

import toy  # noqa: F401  (the device switch)
from streamlx.inkling2 import from_oq_export

DIMS, RANK = 8, 4
# from_oq_export reads the attention shapes for every layer tensor
TEXT = {"local_layer_ids": [], "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "d_rel": 2,
        "sconv_kernel_size": 4}


class LoopMapping(unittest.TestCase):
    def test_message_gates_and_queries(self):
        rng = np.random.default_rng(0)
        weights, want, queries = {}, {}, []
        for n in (2, 3):                # the looped layers, in order
            # mlx-lm's shapes: down_proj [rank, 2 * dims], message_gate_proj
            # and forget_gate_proj [dims, rank], one row per gate (q, k, v, mlp)
            down = rng.standard_normal((4, RANK, 2 * DIMS), np.float32)
            gain = rng.standard_normal((4, DIMS, RANK), np.float32)
            forget = rng.standard_normal((4, DIMS, RANK), np.float32)
            query = rng.standard_normal((4, DIMS), np.float32)
            pre = f"model.layers.{n}."
            weights[pre + "loop_down"] = mx.array(np.stack(
                [d.T.reshape(2, DIMS, RANK) for d in down]))
            weights[pre + "loop_gain_forget"] = mx.array(np.stack(
                [np.concatenate([g.T, f.T], axis=1)
                 for g, f in zip(gain, forget)]))
            weights[pre + "loop_message.query"] = mx.array(query)
            for i, gate in enumerate(("q", "k", "v", "mlp")):
                to = f"model.language_model.layers.{n}.{gate}_message_gate."
                want[to + "down_proj.weight"] = down[i]
                want[to + "message_gate_proj.weight"] = gain[i]
                want[to + "forget_gate_proj.weight"] = forget[i]
            queries.append(query)
        want["model.language_model.message_input.query"] = np.stack(queries)
        got = from_oq_export(weights, TEXT)
        self.assertEqual(sorted(got), sorted(want))
        for name, v in want.items():
            self.assertTrue(np.array_equal(np.array(got[name]), v), name)


class LoopedModelInstall(unittest.TestCase):
    """A looped Inkling 2 checkpoint (the Mini) on an mlx-lm without the
    looped model is refused with the pull request that has it, not passed on
    to fail in mlx-lm."""

    def test_names_the_looped_model(self):
        import json
        import tempfile
        from unittest import mock

        from streamlx import integrate

        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/config.json", "w") as f:
                json.dump({"model_type": "inkling2_loop"}, f)
            try:
                import mlx_lm.models.inkling2  # noqa: F401
            except ImportError:
                self.skipTest("needs an mlx-lm with the inkling2 model")
            with mock.patch.object(integrate, "_has_loop", lambda: False):
                with self.assertRaisesRegex(ImportError, "pull/2086"):
                    integrate._checkpoint_loader(d)
            with mock.patch.object(integrate, "_has_loop", lambda: True):
                self.assertIsNone(integrate._checkpoint_loader(d))


if __name__ == "__main__":
    unittest.main()
