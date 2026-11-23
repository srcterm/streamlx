"""The pipelined decoder on the toy checkpoint: every stream's logits must
be the stock model's, bit for bit, whatever the number of streams, the pool
size, the cache kind and the drafts that turn out wrong."""
import importlib.util
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

import toy
from streamlx import pipeline
from streamlx.integrate import prefill_cache, resize_pools
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


class _Clock:
    """The pipeline's clock in the gate test: time passes only where the
    test spends it, so what the gate charged a step is exact on any machine
    (on a hosted runner, steps and sleeps vary by more than the terms)."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def spend(self, seconds):
        self.now += seconds


class _Head:
    """A draft head that proposes the stock model's own continuation, with
    a wrong token in every fourth draft, and takes DRAFT seconds of the
    clock to draft and FEED to be fed the tokens of a step while the gate
    has drafting off."""
    tap_layers = (3,)
    DRAFT, FEED = 0.5, 0.125

    def __init__(self, seq, clock):
        self.seq, self.clock, self.pos, self.calls = seq, clock, 0, 0

    def reset(self):
        self.pos = 0

    def prime_pairs(self, hidden, next_ids=None):    # the gate's feed
        self.pos += int(hidden.shape[1])
        self.clock.spend(self.FEED)

    def draft(self, pairs, depth):
        self.clock.spend(self.DRAFT)
        self.pos += len(pairs)
        self.calls += 1
        toks = list(self.seq[self.pos:self.pos + depth])
        if self.calls % 4 == 0 and toks:
            toks[-1] = (toks[-1] + 1) % toy.VOCAB
        return toks, [0.9] * len(toks)


class DecodeLoop(unittest.TestCase):
    """decode() with a draft head and the self-gate: the tokens, and what
    the gate charges each arm (its own time up to where its next step can
    begin: drafting = draft + step + rollback, plain = step + feed)."""
    STEP, ROLLBACK, PLAIN = 1.0, 0.25, 2.0     # seconds of the clock

    def test_tokens_and_gate(self):
        n = 160
        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        seq, _ = toy.greedy(ck.ref, PROMPT, n)
        m, pools = ck.streaming(8)
        clock = _Clock()
        head = _Head(seq, clock)
        ad = adapter_for(m, head)
        cache = make_prompt_cache(m)
        logits, rows = ad.forward_taps(mx.array([PROMPT]), cache)
        mx.eval(logits, rows)
        dec = PipelinedDecoder(ad, pools, ad.views(cache), k_max=3,
                               adaptive=False, plain_fn=ad.plain,
                               self_gate=True)
        dec.gate_probe, dec.gate_min, dec.gate_hold = 8, 4, 16
        ran, charged = [], {"step": [], "plain": [], "undrafted": []}

        def spending(name, fn, seconds):
            def run(*a, **kw):
                clock.spend(seconds)
                if name != "rollback":
                    ran.append(name)
                return fn(*a, **kw)
            return run

        account = dec._gate_account

        def recording(dt, tokens):
            # a plain step with drafting on: the head had nothing to propose
            kind = ("undrafted" if ran[-1] == "plain" and dec.drafting
                    else ran[-1])
            charged[kind].append(dt + dec._g_draft)
            account(dt, tokens)

        dec.plain_fn = spending("plain", dec.plain_fn, self.PLAIN)
        dec.step = spending("step", dec.step, self.STEP)
        dec.rollback = spending("rollback", dec.rollback, self.ROLLBACK)
        dec._gate_account = recording
        with mock.patch.object(pipeline, "time", clock):
            out = dec.generate(seq[0], rows[:, -1:], len(PROMPT), n)
        self.assertEqual(out, seq[:len(out)])
        self.assertGreater(dec.draft_tried, dec.draft_ok)    # some rejected
        self.assertEqual(toy.wrong_slots(ck.ref, pools), [])
        drafting = head.DRAFT + self.STEP + self.ROLLBACK
        plain = self.PLAIN + head.FEED
        self.assertGreater(len(charged["step"]), 8)
        self.assertGreater(len(charged["plain"]), 8)
        self.assertEqual(set(charged["step"]), {drafting})
        self.assertEqual(set(charged["plain"]), {plain})
        on, off = dec._g_spt["on"], dec._g_spt["off"]
        self.assertEqual(off, plain)                 # a token a plain step
        self.assertGreaterEqual(on, drafting / dec.k_max)
        self.assertLessEqual(on, drafting)
        self.assertTrue(dec._g_decisions)


class _CountingHead(_Head):
    """_Head that checks, at each draft, that it has been given every
    committed token's pair (fed or with the draft) and no more."""

    def __init__(self, seq, clock, committed):
        super().__init__(seq, clock)
        self.committed, self.fed, self.drafts, self.late = committed, 0, [], []

    def prime_pairs(self, hidden, next_ids=None):
        self.fed += int(hidden.shape[1])
        super().prime_pairs(hidden, next_ids)

    def draft(self, pairs, depth):
        self.drafts.append(len(pairs))
        if self.pos + len(pairs) != self.committed():
            self.late.append((self.pos + len(pairs), self.committed()))
        return super().draft(pairs, depth)


class RunAhead(unittest.TestCase):
    """Every expert held: plain steps run one ahead, the next one queued on
    this one's argmax before it is read. The ids are plain decoding's; a
    reader that stops early leaves the cache just after the last id out; and
    while the gate has drafting off the head is fed nothing per token, its
    next draft getting every committed pair."""

    def setUp(self):
        self.ck = toy.Checkpoint()
        self.addCleanup(self.ck.close)
        self.seq, _ = toy.greedy(self.ck.ref, PROMPT, N)
        m, self.pools = self.ck.streaming(8)
        resize_pools(self.pools, sum(p.n_experts * p.expert_bytes
                                     for p in self.pools.values()))
        self.m = m

    def _decoder(self, head, **kw):
        ad = adapter_for(self.m, head)
        cache = make_prompt_cache(self.m)
        logits, rows = ad.forward_taps(mx.array([PROMPT]), cache)
        mx.eval(logits, rows)
        dec = PipelinedDecoder(ad, self.pools, ad.views(cache),
                               plain_fn=ad.plain, **kw)
        return dec, cache, rows[:, -1:]

    def test_plain_ids_and_cache_end(self):
        dec, cache, row = self._decoder(_Head(self.seq, _Clock()), k_max=1,
                                        self_gate=False)
        fed = []
        plain = dec.plain_fn
        dec.plain_fn = lambda tok, c: (fed.append(tok), plain(tok, c))[1]
        out = dec.generate(self.seq[0], row, len(PROMPT), N)
        self.assertEqual(out, self.seq[:N])
        # every forward after the first went in on the step before's argmax
        self.assertIsInstance(fed[0], int)
        self.assertTrue(all(isinstance(t, mx.array) for t in fed[1:]))
        self.assertEqual(cache[0].offset, len(PROMPT) + N - 1)
        # a reader that stops after k ids: the forward of the k-th ran too
        dec, cache, row = self._decoder(_Head(self.seq, _Clock()), k_max=1,
                                        self_gate=False)
        it = dec.decode(self.seq[0], row, len(PROMPT), N)
        k = 5
        got = [next(it) for _ in range(k)]
        it.close()
        self.assertEqual(got, self.seq[1:k + 1])
        self.assertEqual(cache[0].offset, len(PROMPT) + k + 1)

    def test_not_where_the_forward_reads_the_token(self):
        """An adapter whose plain forward reads the token id on the host
        (plain_lazy False: DeepSeek-V4.1's and Qwen3.8's embedding rows)
        gets every step's token as an id, in turn."""
        dec, cache, row = self._decoder(_Head(self.seq, _Clock()), k_max=1,
                                        self_gate=False)
        dec.ad.plain_lazy = False
        fed = []
        plain = dec.plain_fn
        dec.plain_fn = lambda tok, c: (fed.append(tok), plain(tok, c))[1]
        out = dec.generate(self.seq[0], row, len(PROMPT), N)
        self.assertEqual(out, self.seq[:N])
        self.assertTrue(all(isinstance(t, int) for t in fed))
        self.assertEqual(cache[0].offset, len(PROMPT) + N - 1)

    def test_head_kept_while_gated_off(self):
        out = []
        head = _CountingHead(self.seq, _Clock(), lambda: len(out))
        dec, cache, row = self._decoder(head, k_max=3, adaptive=False,
                                        self_gate=True)
        dec.gate_probe, dec.gate_min, dec.gate_hold = 8, 4, 16
        out.append(self.seq[0])
        got = dec.generate(self.seq[0], row, len(PROMPT), N,
                           on_token=out.append)
        self.assertEqual(got, self.seq[:N])
        self.assertEqual(head.late, [])      # never short of a pair
        self.assertEqual(head.fed, 0)        # nothing fed per token
        self.assertGreater(head.drafts[0], 8)    # the gated-off run's pairs
        self.assertGreater(dec.draft_ok, 0)


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


def _has_loop() -> bool:
    if (importlib.util.find_spec("mlx_lm.models.inkling2") is None
            or mx.default_device() != mx.gpu):
        return False
    import toy_inkling2 as ink

    return ink.HAS_LOOP


@unittest.skipUnless(_has_loop(), "needs an mlx-lm with the looped inkling2 "
                     "(inkling2_loop), on the GPU")
class InklingLoop(unittest.TestCase):
    """Inkling 2 Mini's looped stack through BatchedDecoder, each step's
    drafts checked in one forward: what a step's rejected drafts were leaves
    no trace in the caches (what follows is bit for bit the same), with the
    sliding windows filling and full, after one rollback or two; and a
    decode with the checkpoint's head ends with the caches where the server
    expects them."""
    TAPS = (2, 5, 10)

    @classmethod
    def setUpClass(cls):
        import toy_inkling2 as ink

        cls.ink = ink
        cls.ck = ink.Checkpoint(loop=True)

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    def test_rejected_drafts_leave_no_trace(self):
        ink, rng = self.ink, np.random.default_rng(0)
        for slots in (16, 4):
            m, _pools = self.ck.streaming(slots)
            ad = adapter_for(m, SimpleNamespace(tap_layers=self.TAPS))
            self.assertTrue(ad.batched)
            for n in (4, 14):                # windows of 8: filling, full
                prompt = [t % ink.VOCAB for t in toy.prompt(n)]
                for K, keep in ((2, 1), (4, 1), (4, 3), (5, 2)):
                    with self.subTest(slots=slots, prompt=n, K=K, keep=keep):
                        got = []
                        for variant in range(3):
                            cache = m.make_cache()
                            views = ad.views(cache)
                            with prefill_cache(cache):
                                mx.eval(ad.forward_taps(mx.array([prompt]),
                                                        cache))
                            junk = rng.integers(0, ink.VOCAB, K - keep)
                            toks = [7, 11, 13, 17][:keep] + junk.tolist()
                            _, _, rec = ad.verify(toks, views)
                            if variant == 2:         # the decode loop's two
                                ad.rollback(views, rec, keep + 1)
                            ad.rollback(views, rec, keep)
                            # what follows: a plain step (in place in the
                            # windows), a step with a rejected draft, another
                            after = list(ad.plain(37, views))
                            lg, rows, rec = ad.verify([23, 29, 31], views)
                            ad.rollback(views, rec, 2)
                            after += [lg, rows, *ad.plain(41, views)]
                            mx.eval(after)
                            got.append((ad.offset(views), *after))
                        for g in got:
                            self.assertEqual(g[0], n + keep + 4)
                            for x, y in zip(g[1:], got[0][1:]):
                                self.assertTrue(mx.array_equal(x, y))

    def test_next_layer_links_follow_the_loop(self):
        """Prefill reads ahead for the layer that runs next: every layer of
        the looped toy (layers 2 and 3 run three times) points at the one
        after it but the loop's last, which is followed by layer 2 on two
        passes and by layer 4 on the third."""
        m, _pools = self.ck.streaming(8)
        ws = [layer.mlp.experts for layer in m.layers]
        # (by identity: a module's == compares what it holds)
        nxt = [next((i for i, v in enumerate(ws) if v is w._next), None)
               for w in ws]
        self.assertEqual(nxt, [1, 2, 3, None, 5, None])

    def test_decode_with_the_head(self):
        from streamlx.inkling2 import Inkling2LoopAdapter
        from streamlx.pipeline import BatchedDecoder
        from streamlx.pipeline_adapters import load_drafter

        ink = self.ink
        m, pools = self.ck.streaming(8)
        ad = adapter_for(m, load_drafter(self.ck.dir, m))
        self.assertIsInstance(ad, Inkling2LoopAdapter)
        prompt = [t % ink.VOCAB for t in toy.prompt(14)]
        for n, early in ((40, False), (9, True)):
            cache = m.make_cache()
            views = ad.views(cache)
            ad.reset_drafter()
            with prefill_cache(cache):
                logits, rows = ad.forward_taps(mx.array([prompt]), cache)
            mx.eval(logits, rows)
            ad.prime_pairs(rows[:, :-1], None)
            dec = BatchedDecoder(ad, pools, views, k_max=4, plain_fn=ad.plain,
                                 adaptive=False, self_gate=False)
            first = int(mx.argmax(logits[0, -1]).item())
            if early:                         # a server's request that ends
                out, gen = [first], dec.decode(first, rows[:, -1:],
                                               len(prompt), 10**6)
                for t in gen:
                    out.append(t)
                    if len(out) == n:
                        break
                gen.close()
            else:
                out = dec.generate(first, rows[:, -1:], len(prompt), n)
            with self.subTest(n=n, early=early):
                self.assertEqual(len(out), n)
                self.assertTrue(all(0 <= t < ink.TEXT["vocab_size"]
                                    for t in out))
                self.assertGreater(dec.draft_tried, 0)
                # the cache holds the prompt and every token out but the
                # last, which it holds too when a later stream ran it
                self.assertIn(ad.offset(views) - len(prompt), (n - 1, n))


if __name__ == "__main__":
    unittest.main()
