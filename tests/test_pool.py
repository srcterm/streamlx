"""A failed expert read must leave no expert resident: an expert is in a
pool only once its bytes are."""
import errno
import os
import unittest
from unittest import mock

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import toy
from streamlx.pipeline import PipelinedDecoder
from streamlx.pipeline_adapters import adapter_for

N, PROMPT = 24, toy.prompt()


class _Fail:
    """A read call that fails its n-th call (once): EIO, or a short read."""

    def __init__(self, real, n, short=False):
        self.real, self.n, self.short = real, n, short
        self.calls, self.fired = 0, False

    def __call__(self, fd, arg, off):
        self.calls += 1
        if self.calls == self.n and not self.fired:
            self.fired = True
            if not self.short:
                raise OSError(errno.EIO, "injected read failure")
            return self.real(fd, [memoryview(arg[0])[:1]], off)
        return self.real(fd, arg, off)


class ReadFailure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()
        cls.seq, cls.want = toy.greedy(cls.ck.ref, PROMPT, N)

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    def test_ensure(self):
        for short in (False, True):
            with self.subTest(short=short):
                _, pools = self.ck.streaming(8)
                pool = pools[0]
                pool.ensure(list(range(8)))          # full of real experts
                ids = [20, 21, 22]
                hook = _Fail(os.preadv, 4, short)
                with mock.patch("os.preadv", hook):
                    with self.assertRaises(OSError):
                        pool.ensure(ids)
                self.assertTrue(hook.fired)
                self.assertEqual([e for e in ids if pool.resident(e)], [])
                self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])
                reads = pool.reader.reads
                pool.ensure(ids)                     # the retry reads again
                self.assertGreater(pool.reader.reads, reads)
                self.assertTrue(all(pool.resident(e) for e in ids))
                self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])

    def decoder(self, m, pools, K):
        ad = adapter_for(m, None)
        cache = make_prompt_cache(m)
        mx.eval(m(mx.array([PROMPT]), cache=cache))
        return ad, PipelinedDecoder(ad, pools, ad.views(cache), k_max=K,
                                    adaptive=False, self_gate=False)

    def test_pipelined_step(self):
        # reads into slots go through os.preadv, reads as arrays (no slot
        # to give: 4 slots, 4 streams) through os.pread
        for slots, K, call, nth in ((8, 3, "preadv", 5), (8, 3, "preadv", 40),
                                    (4, 4, "preadv", 23), (4, 4, "pread", 2),
                                    (4, 4, "pread", 9)):
            with self.subTest(slots=slots, K=K, call=call, nth=nth):
                m, pools = self.ck.streaming(slots)
                ad, dec = self.decoder(m, pools, K)
                hook = _Fail(getattr(os, call), nth)
                with mock.patch(f"os.{call}", hook):
                    with self.assertRaises(OSError):
                        for i in range(0, N, K):
                            dec.step(self.seq[i:i + K], len(PROMPT) + i)
                self.assertTrue(hook.fired)
                self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])
                # the same pools, a new cache: the decode is the stock model's
                ad, dec = self.decoder(m, pools, K)
                same = 0
                for i in range(0, N, K):
                    _, hid, _ = dec.step(self.seq[i:i + K], len(PROMPT) + i)
                    for j, h in enumerate(hid):
                        same += bool(mx.array_equal(ad.logits(h)[0, -1],
                                                    self.want[i + j]))
                self.assertEqual(same, N)
                self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])


if __name__ == "__main__":
    unittest.main()
