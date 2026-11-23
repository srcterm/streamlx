"""A served request's end: the prompt cache must end where the stock loop
leaves it (after the last token out, run once), because the next turn
continues from it. A pipelined step can have run that token already."""
import types
import unittest

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

import toy
from streamlx import serve
from streamlx.pipeline import PipelinedDecoder
from streamlx.pipeline_adapters import adapter_for

PROMPT = toy.prompt(8)


class _Head:
    """A draft head that proposes the stock continuation: every draft is
    accepted, so requests end inside a step as well as at its end."""
    tap_layers = (3,)

    def __init__(self, seq):
        self.seq, self.pos = seq, 0

    def reset(self):
        self.pos = 0

    def prime_pairs(self, hidden, next_ids=None):
        self.pos += int(hidden.shape[1])

    def draft(self, pairs, depth):
        self.pos += len(pairs)
        toks = list(self.seq[self.pos - len(PROMPT) + 1:][:depth])
        return toks, [0.9] * len(toks)


class _Detokenizer:
    last_segment = ""

    def reset(self):
        pass

    def add_token(self, _token):
        pass

    def finalize(self):
        pass


class _Tokenizer:
    eos_token_ids = ()
    detokenizer = _Detokenizer()


class RequestEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()
        cls.stream = mx.new_stream(mx.default_device())

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    @staticmethod
    def caches(model, window):
        if window is None:
            return [KVCache() for _ in model.layers]
        return [RotatingKVCache(max_size=window, keep=0) for _ in model.layers]

    def stock(self, window, n):
        """(seq, cache) of the stock loop: the prompt but its last token,
        that token, then each of the n tokens out, once."""
        cache = self.caches(self.ck.ref, window)
        mx.eval(self.ck.ref(mx.array([PROMPT[:-1]]), cache=cache))
        seq, _ = toy.greedy(self.ck.ref, PROMPT[-1:], n, cache)
        return seq, cache

    def test_cache_ends_after_the_last_token(self):
        ref, nxt = self.ck.ref, 7
        for window in (None, 4, 8):           # the prompt is 8 tokens
            seq, _ = self.stock(window, 16)
            for n in (2, 3, 4, 5, 7, 12):
                with self.subTest(window=window, tokens=n):
                    _, want = self.stock(window, n)
                    m, pools = self.ck.streaming(8)
                    ad = adapter_for(m, _Head(seq))
                    cache = self.caches(m, window)
                    dec = PipelinedDecoder(ad, pools, None, k_max=3,
                                           adaptive=False, plain_fn=ad.plain,
                                           self_gate=False)
                    out = [r.token for r in serve._pipelined(dec, dict(
                        model=m, tokenizer=_Tokenizer(), prompt_cache=cache,
                        prompt=list(PROMPT), max_tokens=n,
                        stream=self.stream))]
                    self.assertEqual(out, seq[:n])
                    self.assertGreater(dec.draft_ok, 0)
                    self.assertEqual(cache[0].offset, want[0].offset)
                    self.assertEqual(ad.offset(cache), len(PROMPT) + n)
                    a = ref(mx.array([[nxt]]), cache=want)
                    b = m(mx.array([[nxt]]), cache=cache)
                    self.assertTrue(bool(mx.array_equal(a, b)))


class PoolsGrowBack(unittest.TestCase):
    """--memory-gib's pool sizing: a prompt chunk that takes less from the
    pools than the regrow threshold must not leave a model that fits short
    of its experts for good (it decodes from memory only while every pool
    holds its layer's)."""

    def test_whole_again(self):
        from unittest import mock

        from streamlx.integrate import pool_bytes, resize_pools
        from streamlx.pool import all_held

        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        _, pools = ck.streaming(8)
        every = sum(p.n_experts * p.expert_bytes for p in pools.values())
        resize_pools(pools, every)
        self.assertTrue(all_held(pools))
        state = {"pools": pools}
        cfg = {"budget_bytes": 2 * every}
        with mock.patch.object(serve, "_STATE", state), \
                mock.patch.object(serve, "_CFG", cfg), \
                mock.patch.object(serve, "_HYST", every // 8), \
                mock.patch.object(serve, "_GROW", every):
            serve._fit_pools(0.75 * every, "a prompt chunk")
            self.assertFalse(all_held(pools))
            self.assertLess(pool_bytes(pools), every)
            # short of the regrow threshold and of every expert: as it is
            have = pool_bytes(pools)
            serve._fit_pools(0.9 * every, "decoding")
            self.assertEqual(pool_bytes(pools), have)
            # room for every expert again, still less than _GROW more
            serve._fit_pools(1.05 * every, "decoding")
            self.assertTrue(all_held(pools))


class TokenLogprob(unittest.TestCase):
    """The plain path's log-probabilities for mlx-lm's server: the chosen
    token's comes from the token's own step (the request's sampler computes
    it), with the value an index into the array gives; other uses reach the
    array."""

    def _steps(self, dtype, n=4):
        args = types.SimpleNamespace(temperature=0.0, top_p=1.0, top_k=0,
                                     min_p=0.0, xtc_probability=0.0,
                                     xtc_threshold=0.0)
        tok = types.SimpleNamespace(eos_token_ids=[0], encode=lambda s: [1])
        real = serve._orig_sampler
        serve._orig_sampler = lambda a, t: (lambda lp: mx.argmax(lp, axis=-1))
        try:
            sampler = serve._recording_sampler(args, tok)
        finally:
            serve._orig_sampler = real
        mx.random.seed(3)
        out = []
        for _ in range(n):               # generate_step: sample, then yield
            x = mx.random.normal((1, 50)).astype(dtype)
            logprobs = x - mx.logsumexp(x, keepdims=True)
            y = sampler(logprobs)
            mx.eval(y)
            out.append(types.SimpleNamespace(token=y.item(),
                                             logprobs=logprobs.squeeze(0)))
        return out

    def test_value_and_order(self):
        for dtype in (mx.float32, mx.bfloat16):
            steps = self._steps(dtype)
            arrays = [s.logprobs for s in steps]
            for r, a in zip(serve._host_logprobs(iter(steps)), arrays):
                self.assertIsInstance(r.logprobs, serve._HostLogprobs)
                self.assertEqual(r.logprobs[r.token].item(),
                                 a[r.token].item())
                other = (r.token + 1) % 50
                self.assertEqual(r.logprobs[other].item(), a[other].item())
                self.assertEqual((-r.logprobs).tolist(), (-a).tolist())
                self.assertEqual(r.logprobs.shape, a.shape)

    def test_without_a_recorded_step(self):
        steps = self._steps(mx.float32, 2)
        serve._STATE["picked"].clear()       # e.g. another generator's
        for r in serve._host_logprobs(iter(steps)):
            self.assertIsInstance(r.logprobs, mx.array)


if __name__ == "__main__":
    unittest.main()
