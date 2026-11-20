"""The pipelined decoder on the toy checkpoint: every stream's logits must
be the stock model's, bit for bit, whatever the number of streams, the pool
size, the cache kind and the drafts that turn out wrong."""
import importlib.util
import time
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

import toy
from streamlx.integrate import prefill_cache
from streamlx.pipeline import PipelinedDecoder
from streamlx.pipeline_adapters import adapter_for

N, PROMPT = 48, toy.prompt()


def _other(rng, t: int, vocab: int) -> int:
    """A token that is not t: a wrong draft."""
    return (t + 1 + int(rng.integers(0, vocab - 1))) % vocab


class Scheduler(unittest.TestCase):
    """Steps of K accepted tokens: the logits, and what a step leaves in
    the pools (the LRU order the tokens would have left one after another)."""

    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()
        cls.seq, cls.want = toy.greedy(cls.ck.ref, PROMPT, N)

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    def run_steps(self, slots, K):
        m, pools = self.ck.streaming(slots)
        ad = adapter_for(m, None)
        cache = make_prompt_cache(m)
        mx.eval(m(mx.array([PROMPT]), cache=cache))
        dec = PipelinedDecoder(ad, pools, ad.views(cache), k_max=K,
                               adaptive=False, self_gate=False)
        ref = {l: OrderedDict.fromkeys(p.id2slot) for l, p in pools.items()}
        settle, bad_pools = PipelinedDecoder._settle, []

        def checked(self_, s):
            done = list(s.done)
            used = {l: [list(u) for u in self_._used.get(l, ())] for l in done}
            settle(self_, s)
            for l in done:
                pool, lru = self_.pools[l], ref[l]
                for uniq in used[l]:
                    for e in uniq:
                        if e in lru:
                            lru.move_to_end(e)
                        else:
                            if len(lru) >= pool.n_slots:
                                lru.popitem(last=False)
                            lru[e] = True
                if list(pool.id2slot) != list(lru):
                    bad_pools.append(l)

        same = 0
        with mock.patch.object(PipelinedDecoder, "_settle", checked):
            for i in range(0, N, K):
                outs, hid, _ = dec.step(self.seq[i:i + K], len(PROMPT) + i)
                for j, (o, h) in enumerate(zip(outs, hid)):
                    same += (bool(mx.array_equal(ad.logits(h)[0, -1],
                                                 self.want[i + j]))
                             and o == self.seq[i + j + 1])
        return same, bad_pools, toy.wrong_slots(self.ck.ref, pools)

    def test_logits_and_pools(self):
        # 32 slots hold every expert; 4 are the routed top-k, so most of a
        # step's experts find no slot and are read as arrays
        for slots in (32, 16, 8, 4):
            for K in (1, 2, 3, 4):
                with self.subTest(slots=slots, K=K):
                    same, bad_pools, bad_slots = self.run_steps(slots, K)
                    self.assertEqual(same, N)
                    self.assertEqual(bad_pools, [])
                    self.assertEqual(bad_slots, [])


class Rollback(unittest.TestCase):
    """Wrong drafts rolled back, on a cache that appends (KVCache) and on
    windows written in place over rows they still read (RotatingKVCache)."""

    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    @staticmethod
    def caches(model, window):
        if window is None:
            return [KVCache() for _ in model.layers]
        return [RotatingKVCache(max_size=window, keep=0) for _ in model.layers]

    def test_rejected_drafts(self):
        n = 64
        for window in (None, 4, 8, 32):       # the prompt is 12 tokens
            seq, want = toy.greedy(self.ck.ref, PROMPT, n,
                                   self.caches(self.ck.ref, window))
            for wrong in (0.0, 0.4):
                with self.subTest(window=window, wrong=wrong):
                    rng = np.random.default_rng(1)
                    m, pools = self.ck.streaming(8)
                    ad = adapter_for(m, None)
                    cache = self.caches(m, window)
                    mx.eval(m(mx.array([PROMPT]), cache=cache))
                    dec = PipelinedDecoder(ad, pools, ad.views(cache), k_max=4,
                                           adaptive=False, self_gate=False)
                    i = same = rolled = 0
                    while i < n:
                        toks, a = [seq[i]], None
                        for j in range(1, min(int(rng.integers(1, 5)), n - i)):
                            t = seq[i + j]
                            if rng.random() < wrong:
                                t = _other(rng, t, toy.VOCAB)
                                a = j if a is None else a
                            toks.append(t)
                        a = len(toks) if a is None else a
                        outs, hid, snaps = dec.step(toks, len(PROMPT) + i)
                        for j in range(a):
                            same += (bool(mx.array_equal(
                                ad.logits(hid[j])[0, -1], want[i + j]))
                                and outs[j] == seq[i + j + 1])
                        dec.rollback(snaps, a)
                        rolled += len(toks) - a
                        i += a
                    self.assertEqual(same, n)
                    self.assertEqual(rolled > 0, wrong > 0)


class _Head:
    """A draft head that proposes the stock model's own continuation, with
    a wrong token in every fourth draft, and takes SLEEP seconds to draft
    and to be fed a token while the gate has drafting off."""
    tap_layers = (3,)
    SLEEP = 0.02

    def __init__(self, seq):
        self.seq, self.pos, self.calls = seq, 0, 0
        self.t_feed, self.t_draft = [], []     # what the sleeps really took

    def reset(self):
        self.pos = 0

    def _sleep(self, log):
        t0 = time.perf_counter()
        time.sleep(self.SLEEP)
        log.append(time.perf_counter() - t0)

    def prime_pairs(self, hidden, next_ids=None):
        n = int(hidden.shape[1])
        self.pos += n
        if n == 1:                             # the gate's feed
            self._sleep(self.t_feed)

    def draft(self, pairs, depth):
        self._sleep(self.t_draft)
        self.pos += len(pairs)
        self.calls += 1
        toks = list(self.seq[self.pos:self.pos + depth])
        if self.calls % 4 == 0 and toks:
            toks[-1] = (toks[-1] + 1) % toy.VOCAB
        return toks, [0.9] * len(toks)


class DecodeLoop(unittest.TestCase):
    """decode() with a draft head and the self-gate: the tokens, and what
    the gate charges each arm (its own wall time up to where its next step
    can begin: drafting = draft + step + rollback, plain = step + feed)."""

    def test_tokens_and_gate(self):
        n = 160
        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        seq, _ = toy.greedy(ck.ref, PROMPT, n)
        m, pools = ck.streaming(8)
        head = _Head(seq)
        ad = adapter_for(m, head)
        cache = make_prompt_cache(m)
        logits, rows = ad.forward_taps(mx.array([PROMPT]), cache)
        mx.eval(logits, rows)
        dec = PipelinedDecoder(ad, pools, ad.views(cache), k_max=3,
                               adaptive=False, plain_fn=ad.plain,
                               self_gate=True)
        dec.gate_probe, dec.gate_min, dec.gate_hold = 8, 4, 16
        wall = {"plain": [], "step": [], "roll": []}

        def timed(name, fn, sleep=0.0):
            def run(*a, **kw):
                t0 = time.perf_counter()
                if sleep:
                    time.sleep(sleep)
                out = fn(*a, **kw)
                if name == "plain":
                    mx.eval(out[0])
                wall[name].append(time.perf_counter() - t0)
                return out
            return run

        dec.plain_fn = timed("plain", dec.plain_fn)
        dec.step = timed("step", dec.step)
        dec.rollback = timed("roll", dec.rollback, head.SLEEP)
        out = dec.generate(seq[0], rows[:, -1:], len(PROMPT), n)
        self.assertEqual(out, seq[:len(out)])
        self.assertGreater(dec.draft_tried, dec.draft_ok)    # some rejected
        self.assertEqual(toy.wrong_slots(ck.ref, pools), [])
        on, off = dec._g_spt["on"], dec._g_spt["off"]
        self.assertIsNotNone(on)
        self.assertIsNotNone(off)
        med = lambda v: float(np.median(v))  # noqa: E731
        tokens = 1 + (dec.k_max - 1) * dec.draft_ok / dec.draft_tried
        # a probe's steps can run slower than the run's median step, not
        # faster by a sleep: the sleeps must be in what the gate charged
        self.assertGreaterEqual(off - med(wall["plain"]),
                                0.8 * med(head.t_feed))
        self.assertGreaterEqual(
            on * tokens - med(wall["step"]),
            0.8 * (med(head.t_draft) + med(wall["roll"])))


@unittest.skipUnless(importlib.util.find_spec("mlx_lm.models.inkling2")
                     and mx.default_device() == mx.gpu,
                     "needs an mlx-lm with the inkling2 model, on the GPU "
                     "(MLX runs its experts' scale on the GPU only)")
class Inkling2(unittest.TestCase):
    """The Inkling 2 adapter: sliding windows written in place and
    convolution states rolled back, and a draft head's context rows (the
    residual entering layers 2 and 5, and the final output) from pipelined
    streams, plain steps and the model's own forward."""
    TAPS = (2, 5, 6)

    def test_rejected_drafts(self):
        import toy_inkling2 as ink

        n, prompt = 72, toy.prompt(14)
        prompt = [t % ink.VOCAB for t in prompt]
        ck = ink.Checkpoint()
        self.addCleanup(ck.close)
        ref = ck.stock()
        head = SimpleNamespace(tap_layers=self.TAPS)
        rad = adapter_for(ref, head)
        cache = ref.make_cache()
        logits, rows = rad.forward_taps(mx.array([prompt]), cache)
        mx.eval(logits, rows)
        seq, want, taps = [int(mx.argmax(logits[0, -1]).item())], [], []
        for _ in range(n):
            logits, row = rad.forward_taps(mx.array([[seq[-1]]]), cache)
            mx.eval(logits, row)
            want.append(logits[0, -1])
            taps.append(row)
            seq.append(int(mx.argmax(logits[0, -1]).item()))
        hidden = ink.TEXT["hidden_size"]
        for slots in (16, 8, 4):
            for wrong in (0.0, 0.4):
                with self.subTest(slots=slots, wrong=wrong):
                    rng = np.random.default_rng(0)
                    m, pools = ck.streaming(slots)
                    ad = adapter_for(m, head)
                    cache = m.make_cache()
                    with prefill_cache(cache):
                        mx.eval(m(mx.array([prompt]), cache=cache))
                    views = ad.views(cache)
                    dec = PipelinedDecoder(ad, pools, views, k_max=4,
                                           adaptive=False, self_gate=False)
                    i = same = rows_same = steps = rolled = 0
                    while i < n:
                        if steps % 7 == 6:     # the adapter's own plain step
                            lg, row = ad.plain(seq[i], views)
                            mx.eval(lg, row)
                            same += bool(mx.array_equal(lg[0, -1], want[i]))
                            rows_same += bool(mx.array_equal(row, taps[i]))
                            i, steps = i + 1, steps + 1
                            continue
                        toks, a = [seq[i]], None
                        for j in range(1, min(int(rng.integers(1, 5)), n - i)):
                            t = seq[i + j]
                            if rng.random() < wrong:
                                t = _other(rng, t, ink.VOCAB)
                                a = j if a is None else a
                            toks.append(t)
                        a = len(toks) if a is None else a
                        outs, hid, snaps = dec.step(toks, len(prompt) + i)
                        for j in range(a):
                            lg = ad.logits(hid[j][..., -hidden:])[0, -1]
                            same += (bool(mx.array_equal(lg, want[i + j]))
                                     and outs[j] == seq[i + j + 1])
                            rows_same += bool(mx.array_equal(hid[j],
                                                             taps[i + j]))
                        dec.rollback(snaps, a)
                        rolled += len(toks) - a
                        i, steps = i + a, steps + 1
                    self.assertEqual(same, n)
                    self.assertEqual(rows_same, n)
                    self.assertEqual(rolled > 0, wrong > 0)


if __name__ == "__main__":
    unittest.main()
