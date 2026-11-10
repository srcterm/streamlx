"""Pipelined expert fetching: speculative decode pipelined over streaming pools.

K tokens in flight, one layer apart, on the REAL cache. Token 0 of a step
is a committed token; tokens 1..K-1 are drafts (from the host's own MTP
head, or any host drafter). Each layer is split at the routing boundary:

    begin  = attention half -> gate -> routing indices (the sync)
             -> the pool's missing experts are read asynchronously
    finish = wait -> expert gather (resident slots + the rows the stream
             read) -> rest of layer; the pool is updated once every stream
             has finished the layer

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
    prep(tok, offset) -> (ids, h, pos)   pos: the stream's own state
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
    draft_hidden(h, pos)        optional: what a pair carries for a stream
                                (default its final hidden; a DSpark head
                                wants several layers' outputs)

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

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace

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


# -- streams ------------------------------------------------------------------

@dataclass
class Stream:
    tok: int
    ids: mx.array
    h: mx.array
    pos: object
    ctx: object = None
    st: object = None
    missing: list = field(default_factory=list)   # experts it reads
    need: list = field(default_factory=list)      # + those an earlier one reads
    fut: object = None


class PipelinedDecoder:
    def __init__(self, adapter, pools: dict, cache: list, k_max: int = 3,
                 adaptive: bool = True, plain_fn=None, self_gate: bool = True):
        self.ad = adapter
        self.pools = pools
        self.cache = cache
        # plain_fn(tok, cache) -> (logits [1,1,V], hidden): the
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
        # always drafts. It starts on the plain arm and ends a probe early
        # once its arm is clearly the slower: a losing drafter costs seconds
        # per token on a slow model, a winning one gives up little. Steps
        # that pay for learning rather than reflect an arm's rate are not
        # charged: the first after a switch and the K controller's exploring.
        self.self_gate = bool(self_gate) and plain_fn is not None
        self.gate_probe = 24          # steps (on) / tokens (off) per probe
        self.gate_min = 8             # samples before a probe may end early,
        self.gate_early = 0.15        # when this much slower than the other
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
        self._g_settle = False        # skip the first step after a switch
        self._g_explore = False       # the step in flight is K exploration
        self._g_draft = 0.0           # draft seconds of the step in flight
        if self.self_gate:
            self._g_begin("off")
        self.plain_steps = 0
        # a host whose caches mutate live state in place (rotating windows,
        # pooling rings) supplies its own snapshot/restore
        self._snap = getattr(adapter, "snapshot_cache", None) or snapshot_cache
        self._restore = getattr(adapter, "restore_cache", None) or restore_cache
        self.reader = next(iter(pools.values())).reader
        self.k_max = max(1, k_max)
        self.ex = ThreadPoolExecutor(max_workers=self.k_max,
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
        # -- adaptive K controller ------------------------------------------
        # Picks K per step to maximize expected tokens per second from two
        # online estimates: the wall time of a step at each K (EMA) and the
        # acceptance rate of a draft given its confidence (EMA per
        # confidence bucket, prior = the confidence itself). Content that
        # drafts badly (reasoning text) sinks to K=2; code climbs to k_max.
        # adaptive=False runs every draft the head gives (tests).
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
        self._warm = self._warm_gate = 0          # steps since sequence start
        self._stale_drafter = False               # gated off with no feed
        # Experts a lower stream is already reading at a layer are not read
        # again by a later stream: every stream's begin() runs before any
        # stream's finish(), so a shared miss would otherwise be read once
        # per stream. finish() runs in stream order per layer, so the first
        # reader's rows are there before the later stream's compute.
        self._inflight: dict = {}          # layer -> {expert: True}
        self.n_dup = self.n_read = 0       # duplicates avoided / reads issued
        # The pools do not change during a step: a stream computes its
        # resident experts from their slots and its misses from the rows read
        # in this step (handed to the streaming switch as pool.call_hint), so
        # streams in flight never evict each other's experts, whatever the
        # budget. Once every stream has computed a layer, its pool takes what
        # the tokens would have left there one after another (_settle), but
        # only after a later routing sync: a slot write into a slab that a
        # gather still in flight reads cannot reuse the slab's buffer (MLX
        # holds a command buffer's inputs until it completes), so it would
        # copy the whole slab, every layer, every step.
        self._rows: dict = {}              # layer -> {expert: rows} this step
        self._used: dict = {}              # layer -> [stream's experts, ...]
        self._fin: dict = {}               # layer -> streams finished
        self._unsettled: list = []         # layers computed, pool not updated
        self._k_cur = 0
        # an adapter whose begin() routes with the block's own code lets the
        # switch reuse that routing instead of syncing on it a second time
        self._exact_routing = bool(getattr(adapter, "exact_routing", False))

    # -- layer halves over the pools ---------------------------------------
    def _begin(self, l: int, s: Stream) -> None:
        st = self.ad.begin(l, s.h, s.ids, s.ctx, self.cache, s.pos)
        s.st, s.missing, s.need, s.fut = st, [], [], None
        t0 = time.perf_counter()
        mx.eval(st.inds)                                   # routing sync
        self.t_sync += time.perf_counter() - t0
        pool = self.pools.get(l)
        if pool is None:
            return
        uniq = list(dict.fromkeys(np.array(st.inds).reshape(-1).tolist()))
        self._used.setdefault(l, []).append(uniq)
        self.n_lookups += len(uniq)
        missing = [e for e in uniq if e not in pool.id2slot]
        self.n_miss += len(missing)
        s.need = missing
        infl = self._inflight.setdefault(l, {})
        if infl:
            dup = [e for e in missing if e in infl]
            self.n_dup += len(dup)
            missing = [e for e in missing if e not in infl]
        if missing:
            self.n_read += len(missing)
            for e in missing:
                infl[e] = True
            s.missing = missing
            s.fut = self.ex.submit(self.reader.read_experts_arrays,
                                   pool.triples_loc, missing)

    def _finish(self, l: int, s: Stream) -> None:
        pool = self.pools.get(l)
        if s.fut is not None:
            t0 = time.perf_counter()
            if KEEPALIVE:
                keepalive_wait([s.fut])                     # GPU kept busy
            raw = s.fut.result()                            # IO residual
            t1 = time.perf_counter()
            rows = self.reader.experts_from_arrays(pool.triples_loc,
                                                   s.missing, raw)
            self._rows.setdefault(l, {}).update(rows)
            infl = self._inflight.get(l)
            if infl:
                for e in s.missing:
                    infl.pop(e, None)
            self.t_io += t1 - t0
            self.t_adopt += time.perf_counter() - t1
            s.fut = None
        if pool is None:
            s.h = self.ad.finish(l, s.st)
            s.st = None
            return
        got = self._rows.get(l, {})
        pool.call_hint = SimpleNamespace(
            rows={e: got[e] for e in s.need},
            idx=np.array(s.st.inds) if self._exact_routing else None)
        try:
            s.h = self.ad.finish(l, s.st)
        finally:
            pool.call_hint = None
        s.st = None
        self._fin[l] = self._fin.get(l, 0) + 1
        if self._fin[l] == self._k_cur:
            self._unsettled.append(l)

    def _settle_before(self, l: int) -> None:
        """Settle the computed layers below l (their gathers have completed:
        a routing sync came after their last stream's)."""
        while self._unsettled and self._unsettled[0] < l:
            self._settle(self._unsettled.pop(0))

    def _settle(self, l: int) -> None:
        """Every stream has computed layer l: leave in its pool what the
        step's tokens would have left there one after another (the last
        n_slots experts they used, the last stream's most recent) and drop
        the step's rows."""
        pool = self.pools[l]
        seq = [e for uniq in self._used.pop(l, ()) for e in uniq]
        rows = self._rows.pop(l, {})
        keep: list = []
        for e in reversed(seq):
            if e not in keep:
                keep.append(e)
                if len(keep) == pool.n_slots:
                    break
        keep.reverse()                                  # oldest first
        t0 = time.perf_counter()
        # residents first (made recent, so no adopt below evicts them),
        # then the new rows into the least recent slots
        for e in keep:
            if e in pool.id2slot:
                pool._policy_hit(e)
        new = [e for e in keep if e not in pool.id2slot and e in rows]
        for e in new:
            pool.adopt(e, rows[e])
        if new:
            # run the slot writes now: left lazy until the layer's next use,
            # every layer would hold its old slabs and the new rows (GiBs)
            mx.async_eval([a for part in pool.pool.values()
                           for a in part.values()])
        self.t_adopt += time.perf_counter() - t0

    # -- one step: K tokens in flight ----------------------------------------
    def step(self, toks: list[int], offset: int):
        """Returns (outputs, hiddens, snaps): outputs[i] = argmax after
        stream i, hiddens[i] = its final (pre-head) hidden, snaps[l][i] =
        layer-l cache snapshot taken before stream i wrote it."""
        # the core reads every stream's experts exactly; the wrapper's
        # lookahead would only re-read them
        for p in self.pools.values():
            p._pf_suspend = True
        self._inflight.clear()
        self._rows, self._used, self._fin = {}, {}, {}
        self._unsettled = []
        self._k_cur = len(toks)
        try:
            return self._step(toks, offset)
        finally:
            for p in self.pools.values():
                p._pf_suspend = False

    def _step(self, toks: list[int], offset: int):
        K = len(toks)
        streams = []
        for i, t in enumerate(toks):
            ids, h, pos = self.ad.prep(t, offset + i)
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
                if i == 0:
                    # every stream's layer l-2 gather finished before the
                    # routing syncs since its last stream's
                    self._settle_before(l - 1)
            self._finish(l, streams[0])
        for i in range(1, K):
            self._finish(n - 1, streams[i])
        lg = [self.ad.logits(s.h) for s in streams]
        mx.eval(*lg)
        self._settle_before(n)
        outs = [int(mx.argmax(x[0, -1]).item()) for x in lg]
        dh = getattr(self.ad, "draft_hidden", None)
        return outs, [dh(s.h, s.pos) if dh else s.h for s in streams], snaps

    def rollback(self, snaps, a: int) -> None:
        """Undo the cache writes of streams >= a (keep streams 0..a-1)."""
        if not snaps or a >= len(snaps[0]):
            return
        t0 = time.perf_counter()
        for l in range(self.n_layers):
            self._restore(self.cache[l], snaps[l][a])
        self.t_rollback += time.perf_counter() - t0

    # -- the plain fallback and the self-gate --------------------------------
    def plain_step(self, tok: int):
        """One token through the host's own forward (with its prefetch).
        Returns (argmax, logits, hidden)."""
        t0 = time.perf_counter()
        logits, hidden = self.plain_fn(tok, self.cache)
        mx.eval(logits)
        self.plain_steps += 1
        self.k_hist[1] = self.k_hist.get(1, 0) + 1
        self._gate_account(time.perf_counter() - t0, 1)
        return int(mx.argmax(logits[0, -1]).item()), logits, hidden

    def _gate_account(self, dt: float, tokens: int) -> None:
        """Charge `dt` seconds, plus the step's draft time, for `tokens`
        committed tokens to the current gate phase."""
        if not self.self_gate:
            return
        dt += self._g_draft
        self._g_draft = 0.0
        explore, self._g_explore = self._g_explore, False
        self._warm_gate += 1
        if self._warm_gate <= 2:
            return                        # a sequence's cold steps
        if self._g_hold_left > 0:
            self._g_hold_left -= tokens
            if self._g_hold_left <= 0:
                # hold over: probe the OTHER arm
                self._g_begin(self._other())
            return
        if self._g_settle:
            # an arm's first step pays the switch (the drafter catching up
            # on the context it skipped, the pool turning over), not its rate
            self._g_settle = False
            return
        if explore:
            # the K controller trying a K to learn its cost (a fresh time
            # model tries every K, deep ones included): not the arm's rate
            return
        self._g_t += dt
        self._g_tok += tokens
        self._g_n += 1
        spt = self._g_t / max(self._g_tok, 1)
        other = self._g_spt[self._other()]
        early = (other is not None and self._g_n >= self.gate_min
                 and spt > other * (1.0 + self.gate_early))
        if self._g_n >= self.gate_probe or early:
            self._g_spt[self._g_phase] = spt
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
        self._g_settle = True

    # -- the decode loop -----------------------------------------------------
    def generate(self, first_tok: int, first_hidden, offset: int,
                 n_tokens: int, on_token=None, stop=()) -> list[int]:
        """Greedy pipelined decode. `first_tok` is the committed token whose
        forward has not run yet; `first_hidden` the hidden that produced
        it (the drafter's first pending pair). Returns the generated ids
        starting with first_tok, exactly like a plain greedy loop, ending
        early after the first committed id in `stop`."""
        out = [first_tok]
        for tok in self.decode(first_tok, first_hidden, offset, n_tokens, stop):
            out.append(tok)
            if on_token is not None:
                on_token(tok)
        return out

    def decode(self, first_tok: int, first_hidden, offset: int, n_tokens: int,
               stop=()):
        """generate() one token at a time: yields the ids after first_tok. A
        server stops reading wherever a request ends, so however this ends,
        closed early included, the cache ends just after the last id out when
        a later stream of its step ran it, else just before it."""
        self._warm = self._warm_gate = 0           # a sequence's cold steps
        if first_tok in stop:
            return
        n = 1
        cur = first_tok
        pairs = [(first_hidden, first_tok)]
        depth = self.k_max - 1
        while n < n_tokens:
            drafts, conf = self._draft_for(pairs, depth)
            pairs = []
            if not drafts and self.plain_fn is not None:
                nxt, _lg, h = self.plain_step(cur)
                self.steps += 1
                pairs.append((h, nxt))
                n += 1
                yield nxt
                if nxt in stop:
                    return
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
            self.accepted_hist[a] = self.accepted_hist.get(a, 0) + 1
            self._gate_account(dt, a)
            self.rollback(snaps, a)
            for i in range(a):
                pairs.append((hiddens[i], outs[i]))
                n += 1
                ending = True                # also if closed at this yield
                try:
                    yield outs[i]
                    ending = outs[i] in stop or n == n_tokens
                finally:
                    if ending:               # a step can overshoot the end;
                        # stream i+1 ran outs[i] if it was kept
                        self.rollback(snaps, min(i + 2, a))
                if ending:
                    return
            offset += a
            cur = outs[a - 1]
            if pairs and depth == 0:
                pairs = []

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
        """K in [2, 1 + len(confs)] maximizing E[tokens]/t_step[K]. At
        least 2 whenever a draft exists: a K=1 step inside a pipelined
        sequence is worse than a low-confidence K=2 step (the pool churns
        faster under K>1, so the lone plain step pays more fetch than the
        steady-state plain rate would credit it with)."""
        kmax = 1 + len(confs)
        if kmax <= 1:
            return 1
        k_floor = 2
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
                self._g_explore = True
                return K
        return best_k

    def _draft_for(self, tok_pairs, depth):
        """Drafts for the next step: the head's, cut to the K the adaptive
        controller picks from calibrated confidence and measured step times.
        Returns (drafts, confs) of the streams to run."""
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
        if self._stale_drafter:
            return [], []
        t0 = time.perf_counter()
        drafts, conf = (list(v) for v in self.ad.draft(tok_pairs, depth))
        dt = time.perf_counter() - t0
        self.t_draft += dt
        self._g_draft = dt                  # charged with this step's time
        if self.adaptive:
            K = self._choose_k(conf)
            return drafts[:K - 1], conf[:K - 1]
        return drafts, conf

    def stats(self) -> dict:
        return {
            "steps": self.steps,
            "k_hist": dict(sorted(self.k_hist.items())),
            "accepted_hist": dict(sorted(self.accepted_hist.items())),
            "draft_accept": (self.draft_ok / self.draft_tried
                             if self.draft_tried else None),
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

