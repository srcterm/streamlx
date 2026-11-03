"""Pipelined expert fetching: speculative decode pipelined over streaming pools.

K tokens in flight, one layer apart, on the REAL cache. Token 0 of a step
is a committed token; tokens 1..K-1 are drafts (from the host's own MTP
head, or any host drafter). Each layer is split at the routing boundary:

    begin  = attention half -> gate -> routing indices (the sync)
             -> the pool's missing experts are read asynchronously
    finish = wait -> adopt into the pool -> expert gather -> rest of layer

and the per-layer order S0.begin, [S_i.finish(l-1), S_i.begin]_{i>=1},
S0.finish keeps every stream's read in flight across the other K-1
streams' GPU phases. Because the streams touch each layer's cache in
sequence order, the computation IS the sequential decode; the only new
machinery is undoing the cache writes of drafts that turn out wrong.

The model-specific pieces are a host ADAPTER (the same pattern as
load_streaming_model's `loader=`): how to embed a token, split a layer,
finish it, get logits, and draft. Everything here is model-agnostic:
scheduling, pool IO, acceptance, snapshot/rollback, stats.

Adapter protocol (duck-typed):
    layers                      the decoder stack (len == number of layers)
    prep(tok, offset, pos_base=None, i=0) -> (ids, h, pos)
                                pos = pos_base + i when the host supplied
                                position ids, else from offset
    offset(cache) -> int        the layer-0-of-attention cache offset
    masks(h, cache) -> ctx      opaque, handed to begin()
    begin(l, h, ids, ctx, cache, pos) -> state   with `state.inds`, an
                                mx array of routed expert ids for layer l
                                (unevaluated is fine; the core syncs it)
    finish(l, state) -> h
    logits(h) -> mx array [B, 1, V]
    draft(pairs, depth) -> (list[int] drafts, list[float] confidences)
                                pairs = accepted (hidden, next_token)
                                pairs since the previous call, oldest
                                first — the drafter's own cache is its
                                business

Rollback: a shallow snapshot of every layer cache is taken right before
each stream's begin() at that layer; rejecting streams >= a restores the
snapshot taken before S_a. This is exact for caches that update
functionally (state = new array) or through an offset over an in-place
buffer (mlx-lm/mlx-vlm KVCache: the rejected rows beyond the restored
offset are never read). A cache that mutates a state array in place with
no offset would not be restorable this way — the host must then provide
snapshot/restore itself (adapter.snapshot_cache / restore_cache).
"""
from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .pool import KEEPALIVE, keepalive_wait

# -- generic cache snapshot ---------------------------------------------------

def _cache_like(x) -> bool:
    return hasattr(x, "__dict__") and any(
        hasattr(x, a) for a in ("update_and_fetch", "offset", "state", "cache"))


def snapshot_cache(c):
    """Shallow snapshot: attribute dict copied one container level deep,
    cache-like children snapshotted recursively."""
    if c is None:
        return None
    attrs = {}
    kids = {}
    for k, v in vars(c).items():
        if isinstance(v, list):
            attrs[k] = list(v)
        elif isinstance(v, tuple):
            attrs[k] = tuple(v)
        elif isinstance(v, dict):
            attrs[k] = dict(v)
        else:
            attrs[k] = v
        if isinstance(v, (list, tuple)) and v and all(_cache_like(x) for x in v):
            kids[k] = [snapshot_cache(x) for x in v]
        elif (not isinstance(v, (mx.array, list, tuple, dict, str, int,
                                 float, bool, type(None)))
              and _cache_like(v)):
            kids[k] = snapshot_cache(v)
    return attrs, kids


def restore_cache(c, snap) -> None:
    if c is None or snap is None:
        return
    attrs, kids = snap
    for k, v in attrs.items():
        if isinstance(v, list):
            v = list(v)
        elif isinstance(v, dict):
            v = dict(v)
        setattr(c, k, v)
    for k, sub in kids.items():
        cur = getattr(c, k)
        if isinstance(sub, list):
            for child, s in zip(cur, sub):
                restore_cache(child, s)
        else:
            restore_cache(cur, sub)


# -- stochastic acceptance ------------------------------------------------------

@dataclass
class SamplingSpec:
    """The host's sampling pipeline, mlx-lm order: filters (top_p, min_p,
    top_k) on log-probs, then temperature inside categorical sampling.
    temp <= 0 is greedy."""
    temp: float = 0.0
    top_p: float = 0.0
    min_p: float = 0.0
    top_k: int = 0
    min_tokens_to_keep: int = 1

    @property
    def greedy(self) -> bool:
        return self.temp <= 0


def sampling_logprobs(logits, spec: SamplingSpec):
    """Log-probs of the distribution the host's sampler draws from, for
    the last position of `logits` [..., V] -> [V] float32."""
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    lp = logits.reshape(-1, logits.shape[-1])[-1:].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    if 0.0 < spec.top_p < 1.0:
        lp = apply_top_p(lp, spec.top_p)
    if spec.min_p != 0.0:
        lp = apply_min_p(lp, spec.min_p, spec.min_tokens_to_keep)
    if spec.top_k > 0:
        lp = apply_top_k(lp, spec.top_k)
    lp = lp * (1.0 / spec.temp)
    return (lp - mx.logsumexp(lp, axis=-1, keepdims=True)).reshape(-1)


def forced_logits(tok: int, like) -> mx.array:
    """A logits array the host's sampler can only resolve to `tok`."""
    v = mx.full(like.shape, -1e9, dtype=like.dtype)
    v[..., tok] = 0
    return v


# -- streams ------------------------------------------------------------------

@dataclass
class Stream:
    tok: int
    ids: mx.array
    h: mx.array
    pos: object
    ctx: object = None
    st: object = None
    missing: list = field(default_factory=list)
    fut: object = None


class PipelinedDecoder:
    def __init__(self, adapter, pools: dict, cache: list, k_max: int = 3,
                 tau: float = 0.0, k_min: int = 1,
                 io_workers: int | None = None, adaptive: bool = True,
                 plain_fn=None, self_gate: bool = True):
        self.ad = adapter
        self.pools = pools
        self.cache = cache
        # plain_fn(tok, cache, pos_base) -> (logits [1,1,V], hidden): the
        # host's own one-token forward, routing prefetch included. A step
        # with no drafts to launch runs through it instead of a lone K=1
        # pipeline step (which would have no prefetch and no overlap). The
        # hidden must be what the adapter's draft() consumes.
        self.plain_fn = plain_fn
        # Self-gate: the pipeline pays only when fetch is a fraction of the
        # step and the drafter is byte-cheap and confident; elsewhere
        # drafting costs bytes and head compute for nothing. The gate
        # measures seconds per committed token with drafting ON (pipelined
        # steps + gated plain steps, draft time included) and OFF (plain_fn
        # only), keeps the faster arm for a hold period, and re-probes the
        # other arm afterwards. Needs plain_fn; a decoder without one
        # always drafts.
        self.self_gate = bool(self_gate) and plain_fn is not None
        self.gate_probe = 24          # steps (on) / tokens (off) per probe
        self.gate_hold = 192          # tokens between re-probes (doubles on
        self.gate_hold_max = 3072     # a repeated verdict, up to this)
        self._g_hold_cur = 192
        self.gate_margin = 0.02       # ON must beat OFF by this to hold
        self.drafting = True          # current arm
        self._g_phase = "on"          # on | off (probing or holding)
        self._g_n = 0                 # samples in the current phase
        self._g_t = 0.0               # seconds in the current phase
        self._g_tok = 0               # tokens in the current phase
        self._g_hold_left = 0         # tokens left in the hold
        self._g_spt = {"on": None, "off": None}   # sec/token per arm
        self._g_decisions: list = []  # (spt_on, spt_off, chosen)
        self.plain_steps = 0
        # a host whose caches mutate live state in place (rotating windows,
        # pooling rings) supplies its own snapshot/restore
        self._snap = getattr(adapter, "snapshot_cache", None) or snapshot_cache
        self._restore = getattr(adapter, "restore_cache", None) or restore_cache
        self.reader = next(iter(pools.values())).reader
        self.k_max = max(1, k_max)
        # k_min: streams the confidence gate may never drop below. A K=1
        # step is a plain step with no overlap at all, worse per token than
        # a K=2 step even at 50% draft acceptance, so the gate should only
        # decide the streams beyond k_min.
        self.k_min = max(1, min(k_min, self.k_max))
        self.tau = tau
        self.ex = ThreadPoolExecutor(max_workers=io_workers or self.k_max,
                                     thread_name_prefix="streamlx-pipe-io")
        self.n_layers = len(adapter.layers)
        # stats
        self.t_sync = self.t_io = self.t_adopt = self.t_draft = 0.0
        self.t_snap = self.t_rollback = 0.0
        self.n_lookups = self.n_miss = 0
        self.steps = 0
        self.accepted_hist: dict[int, int] = {}   # a -> count
        self.k_hist: dict[int, int] = {}          # K used -> count
        self.draft_tried = self.draft_ok = 0
        self.depth_tried: dict[int, int] = {}     # draft depth -> tried
        self.depth_ok: dict[int, int] = {}        # draft depth -> accepted
        self.gated = 0
        # -- adaptive K controller ------------------------------------------
        # Picks K per step to maximize expected tokens per second from two
        # online estimates: the wall time of a step at each K (EMA) and the
        # acceptance rate of a draft given its confidence (EMA per
        # confidence bucket, prior = the confidence itself). Content that
        # drafts badly (reasoning text) sinks to K=2 or 1; code climbs to
        # k_max. Subsumes a fixed tau: the confidence enters the expected
        # token count instead of a threshold. tau > 0 disables it.
        self.adaptive = adaptive
        self._t_step: dict[int, float] = {}       # K -> EMA seconds per step
        self._calib = [None] * 10                 # conf bucket -> EMA accept
        self._calib_n = [0] * 10
        self._ema = 0.15
        # The time model excludes each sequence's first two steps (the
        # post-prefill step is cold: pool churn, graph warmup) and explores:
        # a K with fewer than 3 clean samples, or unsampled for
        # _explore_every steps, gets one step.
        self._explore_every = 40
        self._t_n: dict[int, int] = {}            # K -> clean samples
        self._t_last: dict[int, int] = {}         # K -> clock of last sample
        self._t_clock = 0
        self._warm = 0                            # steps since sequence start
        self.adaptive_hist: dict[int, int] = {}
        self._draft_q = None
        # Experts a lower stream is already reading at a layer are not read
        # again by a later stream: every stream's begin() runs before any
        # stream's finish() adopts its rows, so a shared miss would
        # otherwise be read once per stream. finish() runs in stream order
        # per layer, so the first reader's rows are in the pool before the
        # later stream's compute; the wrapper's ensure() is the safety net.
        self._inflight: dict = {}          # layer -> {expert: True}
        self.n_dup = self.n_read = 0       # duplicates avoided / reads issued

    # -- layer halves over the pools ---------------------------------------
    def _begin(self, l: int, s: Stream) -> None:
        st = self.ad.begin(l, s.h, s.ids, s.ctx, self.cache, s.pos)
        s.st, s.missing, s.fut = st, [], None
        t0 = time.perf_counter()
        mx.eval(st.inds)                                   # routing sync
        self.t_sync += time.perf_counter() - t0
        pool = self.pools.get(l)
        if pool is None:
            return
        uniq = list(dict.fromkeys(np.array(st.inds).reshape(-1).tolist()))
        self.n_lookups += len(uniq)
        missing = [e for e in uniq if e not in pool.id2slot]
        self.n_miss += len(missing)
        infl = self._inflight.setdefault(l, {})
        if infl:
            dup = [e for e in missing if e in infl]
            self.n_dup += len(dup)
            missing = [e for e in missing if e not in infl]
        # touch this step's hits before any adopt() evicts the LRU head
        for e in uniq:
            if e in pool.id2slot:
                pool._policy_hit(e)
        if missing:
            self.n_read += len(missing)
            for e in missing:
                infl[e] = True
            s.missing = missing
            s.fut = self.ex.submit(self.reader.read_experts_raw,
                                   pool.triples_loc, missing)

    def _finish(self, l: int, s: Stream) -> None:
        if s.fut is not None:
            pool = self.pools[l]
            t0 = time.perf_counter()
            if KEEPALIVE:
                keepalive_wait([s.fut])                     # GPU kept busy
            raw = s.fut.result()                            # IO residual
            t1 = time.perf_counter()
            rows = self.reader.experts_from_raw(pool.triples_loc, s.missing,
                                                raw)
            for e in s.missing:
                pool.adopt(e, rows[e])
            infl = self._inflight.get(l)
            if infl:
                for e in s.missing:
                    infl.pop(e, None)
            self.t_io += t1 - t0
            self.t_adopt += time.perf_counter() - t1
            s.fut = None
        s.h = self.ad.finish(l, s.st)
        s.st = None

    # -- one step: K tokens in flight ----------------------------------------
    def step(self, toks: list[int], offset: int, pos_base=None):
        """Returns (outputs, hiddens, snaps): outputs[i] = argmax after
        stream i, hiddens[i] = its final (pre-head) hidden, snaps[l][i] =
        layer-l cache snapshot taken before stream i wrote it. `pos_base`
        is the host's position_ids for stream 0 (any shape); stream i
        gets pos_base + i, exactly what the host would pass i calls
        later."""
        # the core reads every stream's experts exactly; the wrapper's
        # lookahead would only re-read them
        for p in self.pools.values():
            p._pf_suspend = True
        self._inflight.clear()
        try:
            return self._step(toks, offset, pos_base)
        finally:
            for p in self.pools.values():
                p._pf_suspend = False

    def _step(self, toks: list[int], offset: int, pos_base=None):
        K = len(toks)
        streams = []
        for i, t in enumerate(toks):
            ids, h, pos = self.ad.prep(t, offset + i, pos_base, i)
            streams.append(Stream(t, ids, h, pos))
        ctx = self.ad.masks(streams[0].h, self.cache)
        for s in streams:
            s.ctx = ctx
        n = self.n_layers
        snaps = [[None] * K for _ in range(n)]
        for l in range(n):
            for i in range(K):
                if i > 0 and l > 0:
                    self._finish(l - 1, streams[i])
                if K > 1:
                    t0 = time.perf_counter()
                    snaps[l][i] = self._snap(self.cache[l])
                    self.t_snap += time.perf_counter() - t0
                self._begin(l, streams[i])
            self._finish(l, streams[0])
        for i in range(1, K):
            self._finish(n - 1, streams[i])
        lg = [self.ad.logits(s.h) for s in streams]
        mx.eval(*lg)
        outs = [int(mx.argmax(x[0, -1]).item()) for x in lg]
        self._last_logits = lg
        return outs, [s.h for s in streams], snaps

    def rollback(self, snaps, a: int) -> None:
        """Undo the cache writes of streams >= a (keep streams 0..a-1)."""
        if not snaps or a >= len(snaps[0]):
            return
        t0 = time.perf_counter()
        for l in range(self.n_layers):
            self._restore(self.cache[l], snaps[l][a])
        self.t_rollback += time.perf_counter() - t0

    # -- the plain fallback and the self-gate --------------------------------
    def plain_step(self, tok: int, pos_base=None):
        """One token through the host's own forward (with its prefetch).
        Returns (argmax, logits, hidden)."""
        t0 = time.perf_counter()
        logits, hidden = self.plain_fn(tok, self.cache, pos_base)
        mx.eval(logits)
        self._last_logits = [logits]
        self.plain_steps += 1
        self.k_hist[1] = self.k_hist.get(1, 0) + 1
        self._gate_account(time.perf_counter() - t0, 1)
        return int(mx.argmax(logits[0, -1]).item()), logits, hidden

    def _gate_account(self, dt: float, tokens: int) -> None:
        """Charge `dt` seconds for `tokens` committed tokens to the current
        gate phase (draft time is charged separately by _draft_for)."""
        if not self.self_gate:
            return
        self._warm_gate = getattr(self, "_warm_gate", 0) + 1
        if self._warm_gate <= 2:
            return                        # a sequence's cold steps
        self._g_t += dt
        self._g_tok += tokens
        self._g_n += 1
        if self._g_hold_left > 0:
            self._g_hold_left -= tokens
            if self._g_hold_left <= 0:
                # hold over: probe the OTHER arm
                self._g_begin(self._other())
            return
        if self._g_n >= self.gate_probe:
            self._g_spt[self._g_phase] = self._g_t / max(self._g_tok, 1)
            on, off = self._g_spt["on"], self._g_spt["off"]
            if on is None or off is None:
                self._g_begin(self._other())        # measure the other arm
                return
            chosen = "on" if on < off * (1.0 - self.gate_margin) else "off"
            # a repeated verdict doubles the hold (capped): a model whose
            # head never pays stops paying for probes of the slow arm
            if self._g_decisions and self._g_decisions[-1][2] == chosen:
                self._g_hold_cur = min(self._g_hold_cur * 2, self.gate_hold_max)
            else:
                self._g_hold_cur = self.gate_hold
            self._g_decisions.append((round(on, 4), round(off, 4), chosen))
            self._g_begin(chosen, hold=self._g_hold_cur)

    def _other(self) -> str:
        return "off" if self._g_phase == "on" else "on"

    def _g_begin(self, phase: str, hold: int = 0) -> None:
        self._g_phase = phase
        self.drafting = phase == "on"
        self._g_n = 0
        self._g_t = 0.0
        self._g_tok = 0
        self._g_hold_left = hold

    # -- the decode loop -----------------------------------------------------
    def generate(self, first_tok: int, first_hidden, offset: int,
                 n_tokens: int, on_token=None) -> list[int]:
        """Greedy pipelined decode. `first_tok` is the committed token whose
        forward has not run yet; `first_hidden` the hidden that produced
        it (the drafter's first pending pair). Returns the generated ids
        starting with first_tok, exactly like a plain greedy loop."""
        out = [first_tok]
        cur = first_tok
        pairs = [(first_hidden, first_tok)]
        depth = self.k_max - 1
        while len(out) < n_tokens:
            drafts, conf = self._draft_for(pairs, depth)
            pairs = []
            if not drafts and self.plain_fn is not None:
                nxt, _lg, h = self.plain_step(cur)
                self.steps += 1
                pairs.append((h, nxt))
                out.append(nxt)
                if on_token is not None:
                    on_token(nxt)
                offset += 1
                cur = nxt
                continue
            toks = [cur] + list(drafts)
            K = len(toks)
            self.k_hist[K] = self.k_hist.get(K, 0) + 1
            ts = time.perf_counter()
            outs, hiddens, snaps = self.step(toks, offset)
            dt = time.perf_counter() - ts
            self._learn_time(K, dt)
            self.steps += 1
            a = self._accepted(drafts, outs)
            self._learn_accept(conf, a)
            self.draft_tried += K - 1
            self.draft_ok += a - 1
            for d in range(1, K):
                self.depth_tried[d] = self.depth_tried.get(d, 0) + 1
                if d < a:
                    self.depth_ok[d] = self.depth_ok.get(d, 0) + 1
            self.accepted_hist[a] = self.accepted_hist.get(a, 0) + 1
            self._gate_account(dt, a)
            self.rollback(snaps, a)
            for i in range(a):
                pairs.append((hiddens[i], outs[i]))
                out.append(outs[i])
                if on_token is not None:
                    on_token(outs[i])
            offset += a
            cur = outs[a - 1]
            if pairs and depth == 0:
                pairs = []
        return out[:n_tokens]

    @staticmethod
    def _accepted(drafts, outs) -> int:
        """Streams 0..a-1 are valid: draft i was exactly the previous
        stream's (greedy) output."""
        a = 1
        while a - 1 < len(drafts) and drafts[a - 1] == outs[a - 1]:
            a += 1
        return a

    # -- adaptive K -----------------------------------------------------------
    @staticmethod
    def _bucket(c: float) -> int:
        return min(9, max(0, int(c * 10)))

    def _cal(self, c: float) -> float:
        v = self._calib[self._bucket(c)]
        return c if v is None else v

    def _learn_accept(self, confs, a: int) -> None:
        """Draft j (0-based) was accepted iff j < a - 1; draft j+1 was only
        tried if draft j was accepted. Per-bucket running mean for the
        first observations (fast start), EMA after."""
        for j, c in enumerate(confs):
            if j > a - 1:
                break
            ok = 1.0 if j < a - 1 else 0.0
            b = self._bucket(c)
            n = self._calib_n[b] = self._calib_n[b] + 1
            v = self._calib[b]
            if v is None:
                self._calib[b] = ok
            elif n <= 20:
                self._calib[b] = v + (ok - v) / n
            else:
                self._calib[b] = v + self._ema * (ok - v)

    def _learn_time(self, K: int, dt: float) -> None:
        self._t_clock += 1
        self._warm += 1
        if self._warm <= 2:
            return                                # cold steps are not the model
        n = self._t_n[K] = self._t_n.get(K, 0) + 1
        self._t_last[K] = self._t_clock
        v = self._t_step.get(K)
        if v is None:
            self._t_step[K] = dt
        elif n <= 5:
            self._t_step[K] = v + (dt - v) / n    # count-mean: fast start
        else:
            self._t_step[K] = v + self._ema * (dt - v)

    def _choose_k(self, confs) -> int:
        """K in [k_floor, 1 + len(confs)] maximizing E[tokens]/t_step[K].
        k_floor is 2 whenever a draft exists: a K=1 step inside a
        pipelined sequence is worse than a low-confidence K=2 step (the
        pool churns faster under K>1, so the lone plain step pays more
        fetch than the steady-state plain rate would credit it with)."""
        kmax = 1 + len(confs)
        if kmax <= 1:
            return 1
        k_floor = max(self.k_min, 2 if kmax >= 2 else 1)
        best_k, best_rate = k_floor, None
        exp_tok = 1.0
        prod = 1.0
        ref = self._t_step.get(k_floor)
        for K in range(1, kmax + 1):
            if K > 1:
                prod *= self._cal(confs[K - 2])
                exp_tok += prod
            if K < k_floor:
                continue
            t = self._t_step.get(K)
            if t is None:
                # unmeasured: optimistic so it gets tried once
                base = ref if ref is not None else 0.25
                t = base * (1.0 + 0.4 * (K - k_floor))
            rate = exp_tok / t
            if best_rate is None or rate > best_rate * 1.01:
                best_k, best_rate = K, rate
        best_k = max(k_floor, min(best_k, kmax))
        # exploration: a candidate the time model has not measured cleanly
        # (< 3 samples) or not seen for _explore_every steps gets one step
        for K in range(k_floor, kmax + 1):
            if K == best_k:
                continue
            if (self._t_n.get(K, 0) < 3
                    or self._t_clock - self._t_last.get(K, 0) > self._explore_every):
                return K
        return best_k

    def _accept_stochastic(self, drafts, q_lps, logits_list, spec):
        """Speculative-sampling acceptance against the host's distribution.
        Stream i's target p_i (its logits through the host's sampling
        pipeline) judges draft i+1: accept with prob min(1, p_i(d)/q_i(d));
        on rejection the emitted token is drawn from the residual
        max(0, p_i - q_i). Returns (a, forced): streams 0..a-1 are kept;
        forced[i] is the token the host must emit after stream i (the
        accepted draft, or the residual sample), None for the last kept
        stream when every draft was accepted (the host samples the bonus
        token from the true logits itself). Lossless: the emitted
        sequence is distributed exactly as the host's own sampling."""
        K = len(logits_list)
        forced = [None] * K
        a = 1
        for i in range(K - 1):
            d = drafts[i]
            p = sampling_logprobs(logits_list[i], spec)
            q = q_lps[i]
            pd = float(mx.exp(p[d]).item())
            qd = float(mx.exp(q[d]).item())
            u = float(mx.random.uniform().item())
            if qd <= 0.0 or u < min(1.0, pd / qd):
                forced[i] = d
                a += 1
                continue
            resid = mx.maximum(mx.exp(p) - mx.exp(q), 0.0)
            z = float(mx.sum(resid).item())
            if z <= 0.0:
                # p == q on this support: any draw from p is valid
                tok = int(mx.random.categorical(p).item())
            else:
                tok = int(mx.random.categorical(mx.log(resid / z + 1e-30)).item())
            forced[i] = tok
            break
        return a, forced

    def _draft_for(self, tok_pairs, depth):
        """Drafts for the next step. tau > 0: fixed confidence gate (with
        k_min). Otherwise the adaptive controller picks K from calibrated
        confidence and measured step times. Returns (drafts, confs) of the
        streams to run."""
        if depth <= 0 or not tok_pairs:
            return [], []
        if not self.drafting:
            # gated OFF: no head call, no stage bytes — but the head's own
            # cache must stay exact for when the gate re-arms, so feed it
            # the committed pairs without proposing (adapters that cannot
            # feed silently just re-prime on the next prefill)
            feed = getattr(self.ad, "feed_pairs", None)
            if feed is not None:
                t0 = time.perf_counter()
                feed(tok_pairs)
                self.t_draft += time.perf_counter() - t0
            else:
                self._stale_drafter = True
            return [], []
        if getattr(self, "_stale_drafter", False):
            return [], []
        t0 = time.perf_counter()
        spec = getattr(self, "sampling", None)
        if spec is not None and not spec.greedy:
            res = self.ad.draft(tok_pairs, depth, spec)
        else:
            res = self.ad.draft(tok_pairs, depth)
        dt = time.perf_counter() - t0
        self.t_draft += dt
        if self.self_gate and getattr(self, "_warm_gate", 0) > 2:
            self._g_t += dt                     # the drafter's cost is the
        drafts, conf = list(res[0]), list(res[1])
        self._draft_q = list(res[2]) if len(res) > 2 and res[2] is not None else None
        if self.tau > 0:
            keep = 0
            for c in conf:
                if c < self.tau:
                    break
                keep += 1
            keep = max(keep, self.k_min - 1)
            if keep < len(drafts):
                self.gated += 1
            if self._draft_q is not None:
                self._draft_q = self._draft_q[:keep]
            return drafts[:keep], conf[:keep]
        if self.adaptive:
            K = self._choose_k(conf)
            self.adaptive_hist[K] = self.adaptive_hist.get(K, 0) + 1
            if K - 1 < len(drafts):
                self.gated += 1
            if self._draft_q is not None:
                self._draft_q = self._draft_q[:K - 1]
            return drafts[:K - 1], conf[:K - 1]
        return drafts, conf

    def stats(self) -> dict:
        return {
            "steps": self.steps,
            "k_hist": dict(sorted(self.k_hist.items())),
            "accepted_hist": dict(sorted(self.accepted_hist.items())),
            "draft_accept": (self.draft_ok / self.draft_tried
                             if self.draft_tried else None),
            "gated_steps": self.gated,
            "depth_accept": {d: round(self.depth_ok.get(d, 0) / n, 3)
                             for d, n in sorted(self.depth_tried.items())},
            "adaptive": self.adaptive and self.tau <= 0,
            "t_step": {k: round(v, 3) for k, v in sorted(self._t_step.items())},
            "t_n": dict(sorted(self._t_n.items())),
            "calib": [None if v is None else round(v, 2) for v in self._calib],
            "miss_rate": self.n_miss / max(self.n_lookups, 1),
            "n_miss": self.n_miss,
            "n_dup": self.n_dup,
            "n_read": self.n_read,
            "t_sync": round(self.t_sync, 1),
            "t_io": round(self.t_io, 1),
            "t_adopt": round(self.t_adopt, 1),
            "t_draft": round(self.t_draft, 1),
            "t_snap": round(self.t_snap, 2),
            "t_rollback": round(self.t_rollback, 2),
            "plain_steps": self.plain_steps,
            "gate": None if not self.self_gate else {
                "arm": self._g_phase, "drafting": self.drafting,
                "hold_left": self._g_hold_left, "hold": self._g_hold_cur,
                "spt": {k: (None if v is None else round(v, 4))
                        for k, v in self._g_spt.items()},
                "decisions": list(self._g_decisions[-6:]),
            },
        }


class LookaheadDecoder(PipelinedDecoder):
    """Serve a host's ONE-token decode calls from pipelined steps.

    A host that decodes one token per model call: call(tok) must return
    the logits of tok's
    forward with the cache advanced by one. Here that call runs a whole
    K-stream step — tok plus K-1 drafts — returns stream 0's logits, and
    keeps the accepted drafts' streams PENDING. The next call is served
    from the pending stream when the host asks for exactly the token that
    stream ran on (always, under greedy: the host samples the argmax we
    already verified against); its cache writes are already in place,
    so the host sees a normal step that happens to be nearly free. Any
    other token means the pending streams were speculation the host did
    not take: their writes are rolled back to the last consumed stream
    and a fresh step starts from the requested token. Sampling therefore
    degrades to plain decode, never to a wrong result.

    Invariants checked on every call: the cache objects are the ones the
    pending streams wrote (a host that swapped caches gets a fresh
    start) and the layer offset is where the step left it (a host that
    trimmed or rewound the cache invalidates the pending streams without
    a restore — the host owns the cache now).

    The drafter is fed (hidden, next_token) pairs from CONSUMED calls
    only, with the token the host actually asked for, so the head's own
    cache is exact whatever the host samples.
    """

    def __init__(self, adapter, pools, cache, k_max=3, tau=0.0, k_min=1,
                 io_workers=None, adaptive=True, plain_fn=None, self_gate=True):
        super().__init__(adapter, pools, cache, k_max, tau, k_min, io_workers,
                         adaptive, plain_fn=plain_fn, self_gate=self_gate)
        self.pending: deque = deque()     # (input_tok, logits, hidden)
        self.snaps = None
        self.consumed = 0
        self.end_offset = None
        self.last_hidden = None
        self.primed_offset = None
        self.pairs: list = []
        self.hits = self.misses = self.invalidated = 0
        self.invalidated_by: dict[str, int] = {}
        self.sampling: SamplingSpec | None = None
        self.stochastic_steps = 0
        self.residual_draws = 0
        # STREAMLX_PIPE_TRACE=1: per-call trace (hit / step records) for
        # divergence hunting; read via .trace, cleared by reset_stats()
        self.trace = [] if os.environ.get("STREAMLX_PIPE_TRACE") == "1" else None

    def set_sampling(self, spec: SamplingSpec | None) -> None:
        """The host's sampling parameters (None or temp 0 = greedy)."""
        self.sampling = spec if (spec is not None and not spec.greedy) else None

    def prime(self, last_hidden, offset: int | None = None) -> None:
        """Hidden of the token that produced the first decode token (the
        prompt's last position); lets the very first call draft. `offset`:
        the cache offset the prompt left behind, so a later call arriving
        with a different cache container at that same offset is
        recognised as the same sequence."""
        self.last_hidden = last_hidden
        self.primed_offset = offset
        self._warm = 0

    def _cache_matches(self, cache) -> bool:
        if cache is None or cache is self.cache:
            return True
        if self.cache is None or len(cache) != len(self.cache):
            return False
        return all(a is b for a, b in zip(cache, self.cache))

    def discard_pending(self, reason: str = "") -> None:
        """Undo the cache writes of every pending (unconsumed) stream and
        forget them. Called whenever the host does anything other than
        ask for the next pending token: a passthrough forward (prefill,
        a multi-token call), a cache swap, or a rewind. If the host has
        already moved the cache's offset off the step's end, a restore is
        applied only when the offset sits exactly at the consumed point
        (then it is idempotent: the host trimmed what we would undo);
        anywhere else the host owns the cache and we only forget."""
        if not self.pending and self.snaps is None:
            return
        if self.pending and self.snaps is not None and self.cache is not None:
            try:
                offset = self.ad.offset(self.cache)
            except Exception:  # noqa: BLE001
                offset = None
            step_start = (self.end_offset - self.consumed
                          - len(self.pending)) if self.end_offset is not None \
                else None
            if offset == self.end_offset or (
                    step_start is not None and offset == step_start + self.consumed):
                self.rollback(self.snaps, self.consumed)
            self.invalidated += 1
            if reason:
                self.invalidated_by[reason] = self.invalidated_by.get(reason, 0) + 1
        self.pending.clear()
        self.snaps = None

    def call(self, tok: int, cache=None, pos_base=None):
        """Logits [1, 1, V] of `tok`'s forward; cache advanced by one.
        `pos_base`: the host's position_ids for this call, if any."""
        if not self._cache_matches(cache):
            # a host-owned cache we have not seen (first call, or a swap):
            # the old cache must not keep our speculation. If nothing is
            # pending and the new container sits exactly where priming
            # left the sequence, it is the SAME sequence in a new container
            # (a host moving a request's prefill cache into a batch cache):
            # keep the primed hidden so the first step can draft.
            same_seq = (not self.pending and self.primed_offset is not None
                        and self.ad.offset(cache) == self.primed_offset)
            self.discard_pending("cache-swap")
            self.cache = cache
            if not same_seq:
                self.last_hidden = None
                self.pairs = []
            self.primed_offset = None
        offset = self.ad.offset(self.cache)
        if self.pending:
            if self.end_offset != offset:
                # the host moved the cache under us
                self.discard_pending("offset")
                offset = self.ad.offset(self.cache)
            else:
                ptok, plog, ph = self.pending[0]
                if ptok == tok:
                    self.pending.popleft()
                    self.consumed += 1
                    self.hits += 1
                    self.pairs.append((self.last_hidden, tok))
                    self.last_hidden = ph
                    if self.trace is not None:
                        self.trace.append(("hit", tok))
                    return plog
                # speculation the host did not take
                self.misses += 1
                self.rollback(self.snaps, self.consumed)
                self.pending.clear()
                self.snaps = None
                offset = self.ad.offset(self.cache)
        if self.last_hidden is not None:
            self.pairs.append((self.last_hidden, tok))
        drafts, conf = self._draft_for(self.pairs, self.k_max - 1)
        self.pairs = []
        if not drafts and self.plain_fn is not None:
            # nothing to pipeline: the host's own step, prefetch and all
            _nxt, logits, hidden = self.plain_step(tok, pos_base)
            self.steps += 1
            self.snaps = None
            self.consumed = 0
            self.end_offset = offset + 1
            self.last_hidden = hidden
            if self.trace is not None:
                self.trace.append(("plain", tok))
            return logits
        toks = [tok] + drafts
        K = len(toks)
        self.k_hist[K] = self.k_hist.get(K, 0) + 1
        ts = time.perf_counter()
        outs, hiddens, snaps = self.step(toks, offset, pos_base)
        dt = time.perf_counter() - ts
        self._learn_time(K, dt)
        lg = self._last_logits
        self.steps += 1
        spec = self.sampling
        if spec is not None and K > 1 and self._draft_q is not None:
            a, forced = self._accept_stochastic(drafts, self._draft_q, lg, spec)
            self.stochastic_steps += 1
            if a < K:
                self.residual_draws += 1
        else:
            a = self._accepted(drafts, outs)
            forced = [None] * K
        self._learn_accept(conf, a)
        self.draft_tried += K - 1
        self.draft_ok += a - 1
        for d in range(1, K):
            self.depth_tried[d] = self.depth_tried.get(d, 0) + 1
            if d < a:
                self.depth_ok[d] = self.depth_ok.get(d, 0) + 1
        self.accepted_hist[a] = self.accepted_hist.get(a, 0) + 1
        self._gate_account(dt, a)
        self.rollback(snaps, a)
        self.snaps = snaps
        self.consumed = 1
        self.end_offset = offset + a
        rets = [lg[i] if forced[i] is None else forced_logits(forced[i], lg[i])
                for i in range(a)]
        self.pending = deque((toks[i], rets[i], hiddens[i]) for i in range(1, a))
        self.last_hidden = hiddens[0]
        if self.trace is not None:
            self.trace.append(("step", tok, K, a, list(drafts), list(outs)))
        return rets[0]

    def stats(self) -> dict:
        d = super().stats()
        d.update(hits=self.hits, misses=self.misses,
                 invalidated=self.invalidated,
                 invalidated_by=dict(self.invalidated_by),
                 sampling=None if self.sampling is None else vars(self.sampling),
                 stochastic_steps=self.stochastic_steps,
                 residual_draws=self.residual_draws)
        return d

    def reset_stats(self) -> None:
        self.t_sync = self.t_io = self.t_adopt = self.t_draft = 0.0
        self.t_snap = self.t_rollback = 0.0
        self.n_lookups = self.n_miss = 0
        self.n_dup = self.n_read = 0
        self.steps = 0
        self.accepted_hist = {}
        self.k_hist = {}
        self.draft_tried = self.draft_ok = 0
        self.depth_tried = {}
        self.depth_ok = {}
        self.gated = 0
        self.hits = self.misses = self.invalidated = 0
        self.invalidated_by = {}
        self.stochastic_steps = self.residual_draws = 0
        self.adaptive_hist = {}
        self._warm = 0                    # a new sequence: cold steps again
        self._warm_gate = 0
        self.plain_steps = 0
        self._stale_drafter = False
        # the time model and the calibration are model properties, not
        # request properties: keep them across sequences
        if self.trace is not None:
            self.trace = []
