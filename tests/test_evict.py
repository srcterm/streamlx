"""Eviction by rank (STREAMLX_EVICT): in decode, plain or pipelined, a miss
evicts the resident expert its layer's router has ranked lowest lately, not
the least recently used one. Only which experts stay resident changes, so the
logits are stock's bit for bit; the router is found by its shape in every
model family the tests build; the current token's experts (a pipelined
step's: every stream's, and the last step's) never leave; a matrix whose
top-k is not the routed set is dropped after the probe (the pool stays LRU);
and "auto" ranks only where a layer's reads take long enough."""
import importlib.util
import time
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

import toy
from streamlx import pool as pool_mod
from streamlx.pipeline import PipelinedDecoder
from streamlx.pipeline_adapters import adapter_for

N, PROMPT = 24, toy.prompt()
# kimi_k3 with latent experts: its experts take 32 features of the 64 its
# router takes (MLA attention in every layer)
KIMI_K3 = dict(toy._BASE, model_type="kimi_k3", num_key_value_heads=4,
               q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=32,
               qk_rope_head_dim=16, v_head_dim=32, num_experts=8,
               num_experts_per_token=2, moe_intermediate_size=32,
               routed_expert_hidden_size=32, first_k_dense_replace=1,
               num_shared_experts=1,
               linear_attn_config={"kda_layers": [], "num_heads": 4})


def _routers(pools) -> bool:
    return all(p.router is not None for p in pools.values())


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
    """The eviction mode (and auto's floor) while inside."""
    with ExitStack() as st:
        st.enter_context(mock.patch.object(pool_mod, "RANK_EVICT", mode))
        if floor is not None:
            st.enter_context(mock.patch.object(pool_mod, "RANK_MIN_FETCH_S",
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
                self.assertTrue(_routers(pools))
                self.assertTrue(all(p.ranked and p.worth is not None
                                    for p in pools.values()))
                self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])
                differs += (sum(p.misses for p in pools.values())
                            != sum(p.misses for p in lru.values()))
        self.assertGreater(differs, 0)

    def test_auto(self):
        """auto ranks a layer only once its reads take RANK_MIN_FETCH_S a
        decode token: a floor no read reaches leaves every pool LRU. Every
        token routed without ranking (the prompt's too) is counted, to age
        the ranks when they resume."""
        for floor, ranks in ((1e9, False), (0.0, True)):
            with self.subTest(floor=floor):
                with _evict("auto", floor):
                    m, pools = self.ck.streaming(6)
                    seq, _ = toy.greedy(m, PROMPT, N)
                self.assertEqual(seq, self.seq)
                self.assertTrue(all((p.worth is not None) == ranks
                                    for p in pools.values()))
                self.assertTrue(all(p.since_rank == (0 if ranks else
                                                     len(PROMPT) + N)
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

    def test_kept_hold_the_rest(self):
        """Experts to keep (a pipelined step's) holding all slots but one:
        only that one is freed."""
        p = self.ranked()
        self.assertEqual(p._evict_lowest(3, [0, 1, 2, 3, 4]), 1)
        self.assertEqual(sorted(p.id2slot), [0, 1, 2, 3, 4])
        self.assertEqual(p._evict_lowest(2, [0, 1, 2, 3, 4]), 0)

    def test_scores_age(self):
        """Tokens routed without ranks in between (auto below its floor,
        a prompt) age the scores like ranked ones: ranked again after 128,
        the router's order now decides, not the one before them (0 and 1
        then at the top, now at the bottom, leave; 2 and 3 stay)."""
        p = self.ranked()
        p.since_rank = 128
        logits = np.zeros(p.n_experts, np.float32)
        logits[[4, 5, 6, 7, 0, 1]] = [4.0, 3.0, 2.0, 1.0, -2.0, -1.0]
        p.rank(logits, [4, 5, 6, 7])
        self.assertEqual(p.since_rank, 0)
        p.ensure([6, 7], routed=[4, 5, 6, 7])
        self.assertEqual(sorted(p.id2slot), [2, 3, 4, 5, 6, 7])

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
                self.assertTrue(_routers(pools))
                self.assertTrue(all(p.ranked and p.worth is not None
                                    for p in pools.values()))

    @unittest.skipUnless(importlib.util.find_spec("mlx_lm.models.kimi_k3"),
                         "needs an mlx-lm with the kimi_k3 model")
    def test_latent_experts(self):
        """kimi_k3's latent experts: the router takes the block's input, the
        experts a projection of it (32 of 64 features). The pool ranks by
        the router's own input, not the experts'."""
        ck = toy.Checkpoint(args=KIMI_K3)
        self.addCleanup(ck.close)
        seq, want = toy.greedy(ck.ref, PROMPT, N)
        with _evict("rank"):
            try:
                m, pools = ck.streaming(4)
            except KeyError as e:
                if "missing switch tensors" not in str(e):
                    raise
                # (the package layout's kimi_k3 keeps its layers under
                # model.language_model, which the index does not map)
                self.skipTest(f"streamlx does not stream this mlx-lm's "
                              f"kimi_k3: {e}")
            got_seq, got = toy.greedy(m, PROMPT, N)
        self.assertEqual(got_seq, seq)
        for a, b in zip(got, want):
            self.assertTrue(mx.array_equal(a, b))
        self.assertTrue(_routers(pools))
        self.assertTrue(all(p.ranked and p.worth is not None
                            for p in pools.values()))


class Pipelined(unittest.TestCase):
    """Pipelined steps rank too: each stream's routing ranks its layer, a
    miss takes the slot of the lowest-worth expert that no stream of the
    step nor of the last uses there, and the settle places what was read as
    arrays in the slots of the lowest worth outside the step's experts.
    Stock's logits bit for bit, drafts right or wrong, every slot holding its
    expert's bytes, and misses other than LRU's somewhere."""

    @classmethod
    def setUpClass(cls):
        cls.ck = toy.Checkpoint()
        cls.seq, cls.want = toy.greedy(cls.ck.ref, PROMPT, N)

    @classmethod
    def tearDownClass(cls):
        cls.ck.close()

    def decode(self, slots, mode, wrong=0.0, floor=None, rename=False,
               slow=0.0, start=None):
        """N tokens in steps of up to 3, each draft wrong with probability
        `wrong`: the pools, and whether every token kept had stock's logits
        and token. Checked as it runs: a stream's routing evicts no expert
        that a stream of the step or of the last uses at that layer, and a
        settle leaves the step's experts kept (the last used, as many as the
        pool has slots) resident; where the pool ranks, what either evicts
        is of the lowest worth among the experts it could have.
        rename: the adapter's state names the block's input otherwise (as
        Qwen3.8's does); slow: seconds each read takes longer; start: every
        pool's fetch_avg before the first step."""
        rng = np.random.default_rng(0)
        route, settle = PipelinedDecoder._route, PipelinedDecoder._settle

        def routed(self_, s):
            l, pool = s.l - 1, self_.pools.get(s.l - 1)
            before = set() if pool is None else set(pool.id2slot)
            route(self_, s)
            if pool is not None:
                busy = self_._last.get(l, set()).union(*self_._used[l])
                self.assertEqual((busy & before) - set(pool.id2slot), set())
                if l in self_._ranked:
                    self.lowest_left(pool, before, busy)

        def settled(self_, s):
            keep = {l: list(dict.fromkeys(
                e for u in reversed(self_._used.get(l, ()))
                for e in reversed(u)))[:self_.pools[l].n_slots]
                for l in s.done}
            before = {l: set(self_.pools[l].id2slot) for l in keep}
            settle(self_, s)
            for l, ids in keep.items():
                pool = self_.pools[l]
                self.assertTrue(all(e in pool.id2slot for e in ids))
                if l in self_._ranked:
                    self.lowest_left(pool, before[l], set(ids))

        with _evict(mode, floor), \
                mock.patch.object(PipelinedDecoder, "_route", routed), \
                mock.patch.object(PipelinedDecoder, "_settle", settled):
            m, pools = self.ck.streaming(slots)
            ad = adapter_for(m, None)
            if rename:
                begin, finish = ad.begin, ad.finish

                def renamed(*args):
                    st = begin(*args)
                    st.mixed = vars(st).pop("xn")
                    return st

                def unnamed(l, st):
                    st.xn = st.mixed
                    return finish(l, st)

                ad.begin, ad.finish = renamed, unnamed
            reader = next(iter(pools.values())).reader
            for name in ("read_into", "read_experts_arrays") if slow else ():
                def late(*args, fn=getattr(reader, name), **kw):
                    time.sleep(slow)
                    return fn(*args, **kw)

                setattr(reader, name, late)
            for p in pools.values():
                p.fetch_avg = p.fetch_avg if start is None else start
            cache = make_prompt_cache(m)
            mx.eval(m(mx.array([PROMPT]), cache=cache))
            dec = PipelinedDecoder(ad, pools, ad.views(cache), k_max=3,
                                   adaptive=False, self_gate=False)
            i, same = 0, True
            while i < N:
                toks, a = [self.seq[i]], None
                for j in range(1, min(3, N - i)):
                    if rng.random() < wrong:
                        toks.append((self.seq[i + j] + 1) % toy.VOCAB)
                        a = j if a is None else a
                    else:
                        toks.append(self.seq[i + j])
                a = len(toks) if a is None else a
                outs, hid, snaps = dec.step(toks, len(PROMPT) + i)
                for j in range(a):
                    same &= (outs[j] == self.seq[i + j + 1] and bool(
                        mx.array_equal(ad.logits(hid[j])[0, -1],
                                       self.want[i + j])))
                dec.rollback(snaps, a)
                i += a
        self.assertEqual(dec.plain_steps, 0)
        return pools, same

    def lowest_left(self, pool, before, keep):
        """The experts that left a ranked pool are of the lowest worth among
        those it held outside `keep`."""
        after = set(pool.id2slot)
        left, stayed = before - after, (before & after) - keep
        if left and stayed:
            self.assertLessEqual(max(pool.worth[list(left)]),
                                 min(pool.worth[list(stayed)]))

    def test_same_logits(self):
        differs = 0
        for slots in (16, 8, toy.ARGS["num_experts_per_tok"]):
            for wrong in (0.0, 0.4):
                with self.subTest(slots=slots, wrong=wrong):
                    lru, same = self.decode(slots, "lru", wrong)
                    self.assertTrue(same)
                    self.assertTrue(all(p.worth is None for p in lru.values()))
                    pools, same = self.decode(slots, "rank", wrong)
                    self.assertTrue(same)
                    self.assertTrue(all(p.ranked and p.worth is not None
                                        for p in pools.values()))
                    self.assertEqual(toy.wrong_slots(self.ck.ref, pools), [])
                    differs += (sum(p.misses for p in pools.values())
                                != sum(p.misses for p in lru.values()))
        self.assertGreater(differs, 0)

    def test_auto(self):
        """auto in pipelined steps: the pool's own verdict (pool.ranking);
        every stream's token counted while it does not rank."""
        for floor, ranks in ((1e9, False), (0.0, True)):
            with self.subTest(floor=floor):
                pools, same = self.decode(8, "auto", floor=floor)
                self.assertTrue(same)
                self.assertTrue(all((p.worth is not None) == ranks
                                    for p in pools.values()))
                self.assertTrue(all(p.since_rank == (0 if ranks else
                                                     len(PROMPT) + N)
                                    for p in pools.values()))

    def test_auto_from_pipelined_reads(self):
        """With no plain step to time them, the pipelined steps' own reads
        decide auto: reads 10 ms slower make every pool rank; fast reads
        bring an estimate of 80 ms under a 10 ms floor."""
        pools, same = self.decode(8, "auto", floor=1e-4, slow=0.01)
        self.assertTrue(same)
        self.assertTrue(all(p.fetch_avg >= 1e-4 and p.worth is not None
                            for p in pools.values()))
        pools, same = self.decode(8, "auto", floor=0.01, start=0.08)
        self.assertTrue(same)
        self.assertTrue(all(p.fetch_avg < 0.01 for p in pools.values()))

    def test_read_cost(self):
        """A step's reads in fetch_avg: the time some read was in flight,
        shared by bytes, per token. Two reads of 10 ms at once, layer 0's
        three times the bytes of layer 1's: 7.5 and 2.5 ms over 2 tokens."""
        m, pools = self.ck.streaming(8)
        ad = adapter_for(m, None)
        dec = PipelinedDecoder(ad, pools, [], k_max=3, adaptive=False,
                               self_gate=False)
        dec._io.extend([(1.0, 1.01, 0, 3000), (1.0, 1.01, 1, 1000)])
        dec._note_reads(2)
        a = 1 - (1 - pool_mod._FETCH_WEIGHT) ** 2
        for l, ms in ((0, 7.5), (1, 2.5), (2, 0.0), (3, 0.0)):
            self.assertAlmostEqual(pools[l].fetch_avg, a * ms / 2e3)

    def test_router_input_by_any_name(self):
        """The router's input reaches the pool from the router itself: an
        adapter whose state names the block's input otherwise (Qwen3.8's
        `mixed`) ranks as well."""
        pools, same = self.decode(8, "rank", rename=True)
        self.assertTrue(same)
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
                self.assertTrue(_routers(pools))
                self.assertTrue(all(p.ranked and p.worth is not None
                                    for p in pools.values()))


if __name__ == "__main__":
    unittest.main()
