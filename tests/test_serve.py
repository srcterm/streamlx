"""A served request's end: the prompt cache must end where the stock loop
leaves it (after the last token out, run once), because the next turn
continues from it. A pipelined step can have run that token already."""
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


if __name__ == "__main__":
    unittest.main()
