"""The fused Inkling 2 layer visit (streamlx.inkling2_step) against mlx-lm's
own, on the toy checkpoints: a prompt chunk, decode steps and a several-row
step give every logit and every cache entry bit for bit, in at most three
quarters of the kernels a decode step; and fuse() leaves the model as it was
when its check finds a difference. Needs the GPU (the toys' nvfp4 experts)
and an mlx-lm with the layer the fused visit mirrors."""
import collections
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten

import toy  # noqa: F401  (the device switch)
from streamlx import inkling2_step as step
from streamlx.pipeline_adapters import adapter_for

try:
    import toy_inkling2 as ink
except ImportError:         # an mlx-lm without the inkling2 model
    ink = None

_VIEWS = {"Broadcast", "Reshape", "Squeeze", "ExpandDims", "Slice",
          "AsStrided", "StopGradient", "Depends", "Split", "Unflatten",
          "Flatten", "Transpose"}


def _mirrored() -> bool:
    if ink is None:
        return False
    try:
        stock = step._stock()
    except ImportError:
        return False
    return all(hasattr(stock, name) for name in step._NEEDS)


def _kernels(arr: mx.array) -> int:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.dot")
        mx.export_to_dot(path, arr)
        with open(path) as f:
            labels = re.findall(r'label ="([^"]+)"', f.read())
    count = collections.Counter(labels)
    return sum(n for k, n in count.items() if k not in _VIEWS)


def _same(a, b) -> bool:
    if isinstance(a, mx.array):
        return (a.shape == b.shape and a.dtype == b.dtype and bool(
            mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _states(cache):
    return [[c[0].state, list(c[1].cache)] for c in cache]


def _float32_v(ck, layer):
    """The checkpoint with one layer's value projection scales and biases in
    float32 (its output then float32, the stream bfloat16)."""
    path = f"{ck.dir}/model-00001-of-00001.safetensors"
    w = mx.load(path)
    for part in ("scales", "biases"):
        k = f"{ink.ROOT}layers.{layer}.self_attn.v_proj.{part}"
        w[k] = w[k].astype(mx.float32)
    mx.save_safetensors(path, w)


def _alike(test, ref, m, prompt, steps=6, emb=None):
    """The streamed model gives stock's logits and caches bit for bit over a
    prompt (or its input embeddings) and greedy steps after it."""
    ca, cb = ref.make_cache(), m.make_cache()
    ids = mx.array([prompt])
    for i in range(steps + 1):
        la = ref(ids, cache=ca, input_embeddings=emb if i == 0 else None)
        lb = m(ids, cache=cb, input_embeddings=emb if i == 0 else None)
        mx.eval(la, lb, _states(ca), _states(cb))
        test.assertTrue(_same(la, lb), f"logits, step {i}")
        test.assertTrue(_same(_states(ca), _states(cb)), f"caches, step {i}")
        ids = mx.argmax(la[0, -1]).reshape(1, 1)


@unittest.skipUnless(mx.default_device() == mx.gpu and _mirrored(),
                     "needs the GPU and an mlx-lm with the mirrored layer")
class FusedVisit(unittest.TestCase):
    def setUp(self):
        step.ENABLED = True

    def tearDown(self):
        step.ENABLED = True

    def _check(self, loop: bool, **text):
        with mock.patch.dict(ink.TEXT, text):
            ck = ink.Checkpoint(loop=loop)
        try:
            ref = ck.stock()
            m, _pools = ck.streaming(16)
            layers = m.model.language_model.layers
            self.assertTrue(all(isinstance(x, step._Fused) for x in layers))
            ca, cb = ref.make_cache(), m.make_cache()

            def both(ids):
                la, lb = ref(ids, cache=ca), m(ids, cache=cb)
                mx.eval(la, lb, _states(ca), _states(cb))
                self.assertTrue(_same(la, lb), "logits")
                self.assertTrue(_same(_states(ca), _states(cb)), "caches")
                return mx.argmax(la[0, -1]).reshape(1, 1)

            t = both(mx.array([list(range(3, 15))]))   # past the window of 8
            for _ in range(12):
                t = both(t)
            both(mx.array([[5, 9, 200]]))              # a step's drafts
            for _ in range(4):
                t = both(t)
            # the kernels of one decode step, the fused visit on and off
            counts = []
            for on in (True, False):
                step.ENABLED = on
                c = m.make_cache()
                mx.eval(m(mx.array([[3, 4, 5]]), cache=c), _states(c))
                counts.append(_kernels(m(t, cache=c)))
            self.assertLessEqual(counts[0], 0.75 * counts[1], counts)
        finally:
            ck.close()

    def test_matches_mlx_lm(self):
        self._check(loop=False)

    def test_matches_mlx_lm_clipping(self):
        """Thresholds every query, key and relative state exceeds: the
        clips scale (at the toy's own they pass most rows through)."""
        self._check(loop=ink.HAS_LOOP, qk_l2_clip_threshold=0.25,
                    rel_l2_clip_threshold=0.125)

    @unittest.skipUnless(ink is not None and ink.HAS_LOOP,
                         "needs mlx-lm's inkling2_loop")
    def test_matches_mlx_lm_looped(self):
        self._check(loop=True)

    def test_other_dtypes_keep_mlx_lm(self):
        """A layer whose value projection keeps its scales in float32 (its
        output in float32, which mlx-lm's convolution after it keeps) is not
        the kind fuse() checked for its class: it keeps mlx-lm's visit, the
        other layers take the fused one. A prompt given as input embeddings
        in float16 runs mlx-lm's code, the steps after it the fused visit
        over the keys and values it cached in float32. Bit for bit."""
        ck = ink.Checkpoint(loop=ink.HAS_LOOP)
        try:
            _float32_v(ck, 4)
            ref = ck.stock()
            m, _pools = ck.streaming(16)
            layers = m.model.language_model.layers
            self.assertEqual([isinstance(x, step._Fused) for x in layers],
                             [i != 4 for i in range(len(layers))])
            _alike(self, ref, m, [25, 26, 27])
            emb = ref.model.language_model.embed(mx.array([[25, 26, 27]]))
            _alike(self, ref, m, [25, 26, 27], emb=emb.astype(mx.float16))
        finally:
            ck.close()

    def test_head_rows_with_some_layers_fused(self):
        """A prompt chunk of more rows than the fused visit takes, through a
        model where one layer keeps mlx-lm's visit (its value projection in
        float32): the draft head's rows record each layer visit once (a
        fused layer falls back to its class's parent, which the recorder
        also wraps), equal to stock's, looped and not."""
        for loop, taps in ((False, (2, 5, 6)), (True, (2, 5, 10))):
            if loop and not ink.HAS_LOOP:
                continue
            with self.subTest(loop=loop):
                ck = ink.Checkpoint(loop=loop)
                try:
                    _float32_v(ck, 4)
                    ref = ck.stock()
                    m, _pools = ck.streaming(16)
                    self.assertEqual(
                        [isinstance(x, step._Fused)
                         for x in m.model.language_model.layers],
                        [i != 4 for i in range(ink.TEXT["num_hidden_layers"])])
                    head = SimpleNamespace(tap_layers=taps)
                    prompt = mx.array([list(range(3, 3 + step._MAX_ROWS + 1))])
                    got = [adapter_for(x, head).forward_taps(prompt,
                                                             x.make_cache())
                           for x in (ref, m)]
                    mx.eval(got)
                    self.assertTrue(_same(got[0], got[1]))
                finally:
                    ck.close()

    def test_new_weights(self):
        """Weights loaded after fuse(): what the fused visit made of the old
        ones (a norm weight's cast, the relative logits table's, the static
        convolution weights, the loop's message queries) is made again."""
        ck = ink.Checkpoint(loop=ink.HAS_LOOP)
        try:
            ref = ck.stock()
            m, _pools = ck.streaming(16)
            self.assertTrue(all(isinstance(x, step._Fused)
                                for x in m.model.language_model.layers))
            _alike(self, ref, m, [5, 6, 7], steps=1)
            mx.random.seed(3)
            made = ("sconv.static_weight", "rel_logits_proj.proj",
                    "post_attention_layernorm.weight", "message_input.query")
            new = [(name, (p + 0.5 * mx.random.normal(p.shape)).astype(
                p.dtype)) for name, p in tree_flatten(ref.parameters())
                if name.endswith(made)]
            self.assertGreaterEqual(len(new), 4 + ink.HAS_LOOP)
            for model in (ref, m):
                model.load_weights(new, strict=False)
            _alike(self, ref, m, [5, 6, 7])
            # an update in place keeps the parameter's identity: fuse() again
            values = 0.5 * mx.random.normal(
                ref.model.language_model.layers[2].self_attn.k_sconv
                .static_weight.shape)
            for model in (ref, m):
                conv = model.model.language_model.layers[2].self_attn.k_sconv
                conv.static_weight[:] = values
            self.assertEqual(step.fuse(m), len(m.model.language_model.layers))
            _alike(self, ref, m, [5, 6, 7])
        finally:
            ck.close()

    def test_compiled_groups_exact(self):
        """Each compiled group against its operations one by one, at full
        precision: the check fuse() runs first."""
        self.assertTrue(step._groups_exact(mx.bfloat16, draws=8))
        self.assertTrue(step._groups_exact(mx.float32, draws=2))

    def test_a_difference_keeps_mlx_lm(self):
        """A fused visit that computes something else (here clips a quarter
        larger) is caught by fuse()'s check: nothing is swapped."""
        ck = ink.Checkpoint(loop=ink.HAS_LOOP)
        clip = step._clip
        try:
            step.ENABLED = False                # load without fusing
            m, _pools = ck.streaming(16)
            step.ENABLED = True
            layers = m.model.language_model.layers
            before = [type(x) for x in layers]
            self.assertFalse(any(isinstance(x, step._Fused) for x in layers))

            def off(fn, x, scale, thr, tau=None):
                y = clip(fn, x, scale, thr, tau)
                return (y.astype(mx.float32) * 1.25).astype(y.dtype)

            step._clip = off
            self.assertEqual(step.fuse(m), 0)
            self.assertEqual([type(x) for x in layers], before)
            step._clip = clip
            self.assertEqual(step.fuse(m), len(layers))
            self.assertTrue(all(isinstance(x, step._Fused) for x in layers))
        finally:
            step._clip = clip
            ck.close()


if __name__ == "__main__":
    unittest.main()
