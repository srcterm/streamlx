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
from unittest import mock

import mlx.core as mx

import toy  # noqa: F401  (the device switch)
import toy_inkling2 as ink
from streamlx import inkling2_step as step

_VIEWS = {"Broadcast", "Reshape", "Squeeze", "ExpandDims", "Slice",
          "AsStrided", "StopGradient", "Depends", "Split", "Unflatten",
          "Flatten", "Transpose"}


def _mirrored() -> bool:
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

    @unittest.skipUnless(ink.HAS_LOOP, "needs mlx-lm's inkling2_loop")
    def test_matches_mlx_lm_looped(self):
        self._check(loop=True)

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
