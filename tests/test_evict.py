"""Eviction by rank (STREAMLX_EVICT): in decode a miss evicts the resident
expert its layer's router has ranked lowest lately, not the least recently
used one. Only which experts stay resident changes, so the logits are stock's
bit for bit; the router is found by its shape in every model family the tests
build; the current token's experts never leave; a matrix whose top-k is not
the routed set is dropped after the probe (the pool stays LRU); and "auto"
ranks only where a layer's reads take long enough."""
import importlib.util
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import toy
from streamlx import integrate
from streamlx import pool as pool_mod
from streamlx.integrate import StreamingSwitchGLU

N, PROMPT = 24, toy.prompt()


def _wrappers(m):
    return [v for k, v in m.named_modules()
            if isinstance(v, StreamingSwitchGLU) and "._next" not in k]


def _routing(args):
    """The tiny checkpoint of `args` with routers that route: mlx-lm starts a
    MoEGate's matrix at zero, so in a toy every logit ties and the routed set
    is the sort's tie-break, which no ranking can follow. Such matrices get
    random values, in the checkpoint and in its stock model."""
    ck = toy.Checkpoint(args=args)
    params = dict(tree_flatten(ck.ref.parameters()))
    zero = [k for k, v in params.items() if k.endswith("gate.weight")
            and v.ndim == 2 and not mx.any(v).item()]
    for k in zero:
        params[k] = mx.random.normal(params[k].shape).astype(params[k].dtype)
    if zero:
        mx.save_safetensors(ck.file, params, metadata={"format": "mlx"})
        ck.ref.load_weights(list(params.items()))
        mx.eval(ck.ref.parameters())
    return ck


@contextmanager
def _evict(mode, floor=None):
    """The decode path's eviction mode (and auto's floor) while inside."""
    with ExitStack() as st:
        st.enter_context(mock.patch.object(integrate, "RANK_EVICT", mode))
        if floor is not None:
            st.enter_context(mock.patch.object(integrate, "RANK_MIN_FETCH_S",
                                               floor))
        yield


class Exact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()
        cls.seq, cls.want = toy.greedy(cls.ck.ref, PROMPT, N)

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    def test_same_logits(self):
        """Ranked pools down to top-k slots: stock's logits bit for bit, every
        router found and ranking, every slot holding its expert's bytes, and
        a miss count other than LRU's somewhere (the rule acted)."""
        differs = 0
        for slots in (12, 8, 6, toy.ARGS["num_experts_per_tok"]):
            with self.subTest(slots=slots):
                with _evict("lru"):
                    m, lru = self.ck.streaming(slots)
                    toy.greedy(m, PROMPT, N)
                with _evict("rank"):
                    m, pools = self.ck.streaming(slots)
                    seq, logits = toy.greedy(m, PROMPT, N)
                self.assertEqual(seq, self.seq)
                for got, want in zip(logits, self.want):
                    self.assertTrue(mx.array_equal(got, want))
                self.assertTrue(all(w._logits is not None
                                    for w in _wrappers(m)))
                self.assertTrue(all(p.ranked and p.worth is not None
                                    for p in pools.values()))
                self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])
                differs += (sum(p.misses for p in pools.values())
                            != sum(p.misses for p in lru.values()))
        self.assertGreater(differs, 0)

    def test_auto(self):
        """auto ranks a layer only once its reads take RANK_MIN_FETCH_S a
        decode token: a floor no read reaches leaves every pool LRU."""
        for floor, ranks in ((1e9, False), (0.0, True)):
            with self.subTest(floor=floor):
                with _evict("auto", floor):
                    m, pools = self.ck.streaming(6)
                    seq, _ = toy.greedy(m, PROMPT, N)
                self.assertEqual(seq, self.seq)
                self.assertTrue(all((p.worth is not None) == ranks
                                    for p in pools.values()))

    def test_lru(self):
        with _evict("lru"):
            m, pools = self.ck.streaming(6)
            toy.greedy(m, PROMPT, N)
        self.assertTrue(all(p.worth is None for p in pools.values()))


class Pool(unittest.TestCase):
    """pool.rank and ensure(routed=...) on one layer's pool of 6 slots."""

    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    def ranked(self):
        """Experts 0-5 resident (0 the least recently used); for the probe's
        calls the router placed 0, 1, 2, 3 at its top, in that order, and 5
        then 4 at its bottom."""
        _, pools = self.ck.streaming(6)
        p = pools[0]
        p.ensure(list(range(6)))
        logits = np.zeros(p.n_experts, np.float32)
        logits[[0, 1, 2, 3, 4, 5]] = [4.0, 3.0, 2.0, 1.0, -2.0, -1.0]
        for _ in range(pool_mod._RANK_PROBE):
            p.rank(logits, [0, 1, 2, 3])
        self.assertTrue(p.ranked)
        return p

    def test_lowest_worth_leaves(self):
        p = self.ranked()
        p.ensure([6, 7], routed=[6, 7, 2, 3])
        self.assertEqual(sorted(p.id2slot), [0, 1, 2, 3, 6, 7])
        self.assertEqual(toy.wrong_slots(self.ck.ref, {0: p}), [])

    def test_routed_stay(self):
        """The current token's experts stay, however low their worth (a
        kernel in flight may read their slots)."""
        p = self.ranked()
        p.ensure([6, 7], routed=[6, 7, 4, 5])
        self.assertEqual(sorted(p.id2slot), [0, 1, 4, 5, 6, 7])

    def test_without_routed(self):
        """A call with no routed set (prefill, the pipeline's own slots)
        evicts the least recently used, ranks or not."""
        p = self.ranked()
        p.ensure([6, 7])
        self.assertEqual(sorted(p.id2slot), [2, 3, 4, 5, 6, 7])

    def test_wrong_matrix_dropped(self):
        """Logits whose top-k is never the routed set: after the probe the
        pool has no ranks and evicts by recency."""
        _, pools = self.ck.streaming(6)
        p = pools[0]
        p.ensure(list(range(6)))
        logits = np.arange(p.n_experts, dtype=np.float32)
        for _ in range(pool_mod._RANK_PROBE):
            p.rank(logits, [0, 1, 2, 3])
        self.assertFalse(p.ranked)
        self.assertIsNone(p.worth)
        p.ensure([6], routed=[6, 3, 4, 5])
        self.assertEqual(sorted(p.id2slot), [1, 2, 3, 4, 5, 6])


class Hosts(unittest.TestCase):
    """The router found by shape in each tiny model family (a Linear gate,
    a gate module holding its matrix, beside shared experts and their gates):
    ranked pools give stock's logits bit for bit."""

    def test_router_found(self):
        n = pool_mod._RANK_PROBE + 4
        for t, args in toy.SHAPES.items():
            with self.subTest(model_type=t):
                ck = _routing(args)
                self.addCleanup(ck.close)
                seq, want = toy.greedy(ck.ref, PROMPT, n)
                with _evict("rank"):
                    m, pools = ck.streaming(4)
                    got_seq, got = toy.greedy(m, PROMPT, n)
                self.assertEqual(got_seq, seq)
                for a, b in zip(got, want):
                    self.assertTrue(mx.array_equal(a, b))
                self.assertTrue(all(w._logits is not None
                                    for w in _wrappers(m)))
                self.assertTrue(all(p.ranked and p.worth is not None
                                    for p in pools.values()))


@unittest.skipUnless(importlib.util.find_spec("mlx_lm.models.inkling2")
                     and mx.default_device() == mx.gpu,
                     "needs an mlx-lm with the inkling2 model, on the GPU "
                     "(MLX runs its experts' scale on the GPU only)")
class Inkling2(unittest.TestCase):
    """Inkling 2's router holds a float matrix (TopKRouter): found, ranked,
    stock's logits bit for bit (through the fused visit where it applies)."""

    def test_ranked(self):
        import toy_inkling2 as ink

        prompt = [t % ink.VOCAB for t in toy.prompt(14)]
        ck = ink.Checkpoint()
        self.addCleanup(ck.close)
        ref = ck.stock()
        cache = ref.make_cache()
        logits = ref(mx.array([prompt]), cache=cache)
        seq, want = [int(mx.argmax(logits[0, -1]).item())], []
        for _ in range(N):
            logits = ref(mx.array([[seq[-1]]]), cache=cache)
            mx.eval(logits)
            want.append(logits[0, -1])
            seq.append(int(mx.argmax(logits[0, -1]).item()))
        for slots in (8, ink.K):
            with self.subTest(slots=slots):
                with _evict("rank"):
                    m, pools = ck.streaming(slots)
                    cache = m.make_cache()
                    m(mx.array([prompt]), cache=cache)
                    for i in range(N):
                        logits = m(mx.array([[seq[i]]]), cache=cache)
                        self.assertTrue(mx.array_equal(logits[0, -1],
                                                       want[i]))
                self.assertTrue(all(w._logits is not None
                                    for w in _wrappers(m)))
                self.assertTrue(all(p.ranked and p.worth is not None
                                    for p in pools.values()))


if __name__ == "__main__":
    unittest.main()
