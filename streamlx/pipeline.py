"""Pipelined expert fetching: speculative decode pipelined over streaming pools.

K tokens in flight on the REAL cache. Token 0 of a step is a committed
token; tokens 1..K-1 are drafts (from the host's own MTP head, or any host
drafter). Each layer is split at the routing boundary:

    begin  = attention half -> gate -> routing indices (the sync)
             -> the pool's missing experts are read asynchronously,
             straight into slots the step is not using
    finish = expert gather over the pool's slots (experts that found no
             slot: from the arrays read for them) -> rest of layer

A stream runs finish(l-1), begin(l) whenever its read has arrived and the
stream before it has begun layer l (its last finish: once that stream is
through the stack); the least advanced such stream goes first, and only when
none can run does the step wait for a read. So a
stream's read is in flight across the other streams' GPU phases, and a late
read holds back its own stream, not the ones ahead of it. A begin is handed
to the GPU and its routing read back only once the next ready stream's
begin is handed over too: the GPU has a routing to run while the host
builds the next graph. Because the streams still touch each layer's cache
in sequence order, the computation IS the sequential decode; the only new
machinery is undoing the cache writes of drafts that turn out wrong.

Drafts need no head where the reply repeats the conversation: the tokens
that followed there before are the drafts (Lookup), checked like any other.

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
    verify(toks, cache) -> (logits [1, K, V], rows [1, K, ...], record)
    rollback(cache, record, keep)    optional: a step's tokens in one
                                forward of the model's, and its caches cut
                                back to the tokens kept (see _mode)

Rollback: a shallow snapshot of every layer cache is taken right before
each stream's begin() at that layer; rejecting streams >= a restores the
snapshot taken before S_a. The pipelined decode is the sequential decode
only if that restore gives back the cache exactly, so it is a condition on
the cache. It holds for caches that update functionally (state = new
array), through an offset over an in-place buffer (mlx-lm/mlx-vlm KVCache:
the rejected rows beyond the restored offset are never read), and for
mlx-lm's rotating windows, which write over rows they still read once the
window is full: their arrays are copied, not shared (_ring). Any other
cache that mutates a state array in place is not restorable this way: the
host must then provide snapshot/restore itself (adapter.snapshot_cache /
restore_cache), as deepseek_v41 and inkling2 do.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from .pipeline_adapters import greedy
from .pool import KEEPALIVE, all_held, keepalive_wait

# While drafting is off and plain steps run ahead (decode), the committed
# pairs wait for the head's next draft instead of a feed per token; past this
# many they are fed (one wait for the step in flight) to bound what is held.
_PAIRS_HELD = 256

def _lookup_n(v: str | None) -> int:
    """STREAMLX_LOOKUP: how many tokens must repeat to find a copy (Lookup),
    default 3; 0, off, false or no: none (the head's drafts only, plain
    decoding with no head). Any other word is the default."""
    v = (v or "").strip().lower()
    if v in ("off", "false", "no"):
        return 0
    return int(v) if v.isdigit() else 3


LOOKUP = _lookup_n(os.environ.get("STREAMLX_LOOKUP"))
# A copy is drafted from once it has predicted this many committed tokens in
# a row. Ordinary text repeats phrases of the prompt and of itself and then
# goes on otherwise: drafted from at once, three in four such copies were
# wrong after a token or two and cost 3 to 5% (the Mini, F172); a real copy
# holds.
_LK_MIN = 8
# A copied draft's "confidence" is not a probability but its own calibration
# bucket beside the head's ten, from this acceptance rate until measured.
_LK_CONF, _LK_PRIOR = 2.0, 0.95


class Lookup:
    """Drafts by copy. Where the last `n` tokens occurred before, in the
    prompt or in the reply so far, a model rewriting a file, quoting or
    repeating itself goes on as it did there: the tokens that followed are
    the draft, free to make and as long as the step can take. They come
    before the head's drafts, for any model with a decoder.

    `ptr` is where the text being copied continues (an index into `seq`),
    `run` how many committed tokens in a row it predicted."""

    def __init__(self, n: int = 3):
        self.n = max(1, n)
        self.reset(())

    def reset(self, context) -> None:
        """A new sequence: the prompt's tokens (all of them, cached or not)."""
        n = self.n
        self.seq = [int(t) for t in context]
        # the n tokens before a position -> that position, the latest one
        self.at = {tuple(self.seq[p - n:p]): p
                   for p in range(n, len(self.seq))}
        self.ptr = None
        self.run = 0
        self.proposed = self.kept = 0

    def push(self, tok: int) -> None:
        """A committed token."""
        seq, n, p = self.seq, self.n, len(self.seq)
        hit = self.ptr is not None and seq[self.ptr] == tok
        self.run = self.run + 1 if hit else 0
        if p >= n:
            self.at[tuple(seq[p - n:p])] = p
        seq.append(tok)
        self.ptr = self.ptr + 1 if hit else self.at.get(tuple(seq[-n:]))

    def propose(self, depth: int) -> list[int]:
        """Up to `depth` tokens the copy goes on with."""
        return [] if self.ptr is None else self.seq[self.ptr:self.ptr + depth]


# -- generic cache snapshot ---------------------------------------------------

def _cache_like(x) -> bool:
    return hasattr(x, "__dict__") and any(
        hasattr(x, a) for a in ("update_and_fetch", "offset", "state", "cache"))


def _ring(c) -> bool:
    """A bounded window written in place over rows it still reads (mlx-lm's
    and mlx-vlm's RotatingKVCache, once full: a write index and a maximum
    size): a rejected stream's write lands on live history, so a snapshot
    must hold the arrays themselves. A second handle on an array keeps its
    values when the cache writes through the first; it costs no copy of its
    own, and holds the window as it was until the snapshot goes (a window
    is bounded, so that is small; for a cache that grows with the context
    it would be the whole context, once a stream)."""
    return hasattr(c, "_idx") and hasattr(c, "max_size")


def snapshot_cache(c):
    """Shallow snapshot: attribute dict copied one container level deep,
    cache-like children snapshotted recursively; a ring's arrays copied."""
    if c is None:
        return None
    attrs = {}
    kids = {}
    ring = _ring(c)
    for k, v in vars(c).items():
        if isinstance(v, list):
            attrs[k] = list(v)
        elif isinstance(v, tuple):
            attrs[k] = tuple(v)
        elif isinstance(v, dict):
            attrs[k] = dict(v)
        elif ring and isinstance(v, mx.array):
            attrs[k] = mx.array(v)
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
    ring = _ring(c)
    for k, v in attrs.items():
        if isinstance(v, list):
            v = list(v)
        elif isinstance(v, dict):
            v = dict(v)
        elif ring and isinstance(v, mx.array):
            v = mx.array(v)        # the snapshot keeps its own
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
    i: int                      # its place in the step: token i
    tok: int
    ids: mx.array
    h: mx.array
    pos: object
    ctx: object = None
    l: int = 0                  # the next layer to begin
    st: object = None           # the layer begun, until finished
    lg: object = None           # its router's logits there, if its pool ranks
    idx: object = None          # its routing there (numpy)
    need: tuple = ()            # the routed experts its pool lacks
    reads: tuple = ()           # the reads that hold them
    out: object = None          # its logits, once through the stack
    done: list = field(default_factory=list)   # layers its finish was the
    #                             last to compute, their pools not settled


class _Read:
    """One read of a layer's experts, for every stream of the step that
    routes to them: straight into their pool slots (the future gives None),
    or as arrays when the pool had no slot to give (rows, once viewed)."""
    __slots__ = ("fut", "ids", "rows", "slots")

    def __init__(self, fut, ids, slots=False):
        self.fut, self.ids, self.rows, self.slots = fut, ids, None, slots


class PipelinedDecoder:
    def __init__(self, adapter, pools: dict, cache: list, k_max: int = 3,
                 adaptive: bool = True, plain_fn=None, self_gate: bool = True,
                 lookup: int | None = None, k_copy: int | None = None):
        self.ad = adapter
        # Drafts by copy (Lookup): the match length that starts one, default
        # LOOKUP, 0 for none. A copy that has held is drafted from whatever
        # the head has and whatever the gate's arm: its steps are neither
        # arm's rate, so they are not charged to the gate.
        n = LOOKUP if lookup is None else lookup
        self.lookup = Lookup(n) if n > 0 else None
        self._k_copy = k_copy         # None: by the kind of step (_mode)
        self.pools = pools
        self.cache = cache
        # plain_fn(tok, cache) -> (logits [1,1,V], hidden): the host's own
        # one-token forward, tok an id or the step before's argmax still
        # being computed (a step queued ahead: _plain_queue). A step with no
        # drafts to launch runs through it instead of a lone K=1 pipeline
        # step (which would compute nothing while its experts are read). The
        # hidden must be what the adapter's draft() consumes.
        self.plain_fn = plain_fn
        # Self-gate: the pipeline pays only when fetch is a fraction of the
        # step and the drafter is byte-cheap and confident; elsewhere
        # drafting costs bytes and head compute for nothing. The gate
        # measures seconds per committed token with drafting ON (pipelined
        # steps + gated plain steps, with the draft and the rollback) and
        # OFF (plain_fn, with the feed that keeps the head's cache exact):
        # each arm as it runs, up to where its next step can begin. It
        # keeps the faster arm for a hold period, and re-probes the
        # other arm afterwards. Needs plain_fn; a decoder without one
        # always drafts. It starts on the plain arm and ends a probe early
        # once its arm is clearly the slower: a losing drafter costs seconds
        # per token on a slow model, a winning one gives up little. Steps
        # that pay for learning rather than reflect an arm's rate are not
        # charged: the first after a switch and the K controller's exploring.
        # With no head (drafts by copy alone) there is nothing to gate.
        self.self_gate = (bool(self_gate) and plain_fn is not None
                          and k_max > 1)
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
        # a host whose caches mutate live state in place in a way the
        # generic snapshot does not know (pooling rings, a window inside
        # another object) supplies its own snapshot/restore
        self._snap = getattr(adapter, "snapshot_cache", None) or snapshot_cache
        self._restore = getattr(adapter, "restore_cache", None) or restore_cache
        self.reader = next(iter(pools.values())).reader
        self.k_max = max(1, k_max)
        self._mode()
        self.ex = ThreadPoolExecutor(max_workers=max(self.k_max, self.k_copy),
                                     thread_name_prefix="streamlx-pipe-io")
        self.n_layers = len(adapter.layers)
        # stats
        self.t_sync = self.t_io = self.t_adopt = self.t_draft = 0.0
        self.t_snap = self.t_rollback = 0.0
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
        # conf bucket -> EMA accept: the head's ten, then copied drafts'
        # (their prior until measured)
        self._calib = [None] * 10 + [_LK_PRIOR]
        self._calib_n = [0] * 11
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
        # every expert held (decode): a drafted step that loses costs at most
        # about a plain step and a half, not seconds of reads, so a drafting
        # probe runs to its end: cut after gate_min steps, one that began on
        # less predictable text (a docstring before the code) misjudged the
        # arm, the more so as plain steps that run ahead are cheaper
        self._held = False
        self._built = 0.0             # host seconds building the last queued
        # A stream's misses are read straight into pool slots and computed
        # from the pool like its resident experts. A slot is given up during
        # a step only by an expert that neither this step's streams nor the
        # last step's used at that layer: a gather still in flight never has
        # a slot rewritten under it, streams in flight never evict each
        # other's experts, and what the next token is likeliest to route to
        # stays, whatever the budget. Of those experts the least recently
        # used gives it up, or, where the pool ranks (pool.ranking: each
        # stream's routing ranks its layer as a plain token's does), the one
        # of the lowest worth. With no such slot the expert is read
        # as arrays instead (handed to the streaming switch as
        # pool.call_hint). Once every stream has computed a layer and an
        # eval has run it, its pool takes the order, and the experts read as
        # arrays, that the tokens would have left there one after another
        # (_settle). An expert is read once a step however many streams
        # route to it.
        self._reads: dict = {}             # layer -> {expert: _Read} this step
        self._used: dict = {}              # layer -> [stream's experts, ...]
        self._last: dict = {}              # layer -> the last step's experts
        self._fin: dict = {}               # layer -> streams finished
        self._ranked: set = set()          # layers whose pools ranked
        self._io = deque()                 # (start, end, layer, bytes) a read
        self._k_cur = 0
        # an adapter whose begin() routes with the block's own code lets the
        # switch reuse that routing instead of syncing on it a second time
        self._exact_routing = bool(getattr(adapter, "exact_routing", False))

    # -- layer halves over the pools ---------------------------------------
    def _begin(self, s: Stream) -> None:
        """The stream's next layer up to its routing, handed to the GPU: the
        host goes on meanwhile, _route reads the routing back. Where the
        layer's pool ranks, the router's logits go with it, of the input
        the router was given in begin() (pool.router_logits: whatever the
        adapter calls it)."""
        l = s.l
        pool = self.pools.get(l)
        if pool is not None:
            pool.router_x = None          # (left by an earlier call)
        s.st = self.ad.begin(l, s.h, s.ids, s.ctx, self.cache, s.pos)
        s.lg = None if pool is None else pool.router_logits()
        mx.async_eval(s.st.inds, *([] if s.lg is None else [s.lg]))
        s.l, s.need, s.reads = l + 1, (), ()

    def _route(self, s: Stream) -> None:
        """The routing of the layer the stream began (the sync); the routed
        experts no pool slot nor read of this step holds start reading."""
        l, st = s.l - 1, s.st
        t0 = time.perf_counter()
        mx.eval(st.inds, *([] if s.lg is None else [s.lg]))   # routing sync
        self.t_sync += time.perf_counter() - t0
        pool = self.pools.get(l)
        if pool is None:
            return
        s.idx = np.array(st.inds)
        uniq = list(dict.fromkeys(s.idx.reshape(-1).tolist()))
        pool.since_rank += 1
        if s.lg is not None:
            pool.rank(np.array(s.lg).reshape(-1), uniq)
            s.lg = None
            self._ranked.add(l)
        by_worth = l in self._ranked and pool.worth is not None
        self._used.setdefault(l, []).append(uniq)
        busy = self._last.get(l, set()).union(*self._used[l])
        held = self._reads.setdefault(l, {})
        new = [e for e in uniq if e not in pool.id2slot and e not in held]
        # the pool's counts and order, as for the tokens one after another:
        # an expert an earlier stream reads is there for the later one
        pool.hits += len(uniq) - len(new)
        pool.misses += len(new)
        for e in uniq:
            if e in pool.id2slot:
                pool.id2slot.move_to_end(e)
        # a slot for each new expert while one is free or an expert none of
        # this step or the last used here can give up its own: the least
        # recent expert if it is one, where the pool ranks the lowest-worth
        direct = []
        if by_worth:
            pool._evict_lowest(len(new) - len(pool.free), busy)
            direct = new[:len(pool.free)]
            for e in direct:
                pool._take_slot_for(e)
        else:
            for e in new:
                if pool.free or next(iter(pool.id2slot)) not in busy:
                    pool._take_slot_for(e)
                    direct.append(e)
        if direct:
            try:
                reqs = pool.slot_requests(direct)
            except BaseException:
                pool.forget(direct)        # slots given, nothing read
                raise
            rd = _Read(self.ex.submit(
                self._read, l, len(direct) * pool.expert_bytes,
                self.reader.read_into, reqs), direct, slots=True)
            held.update(dict.fromkeys(direct, rd))
        if len(direct) < len(new):
            rest = [e for e in new if e not in pool.id2slot]
            rd = _Read(self.ex.submit(
                self._read, l, len(rest) * pool.expert_bytes,
                self.reader.read_experts_arrays, pool.triples_loc, rest),
                rest)
            held.update(dict.fromkeys(rest, rd))
        s.need = need = [e for e in uniq if e in held]
        s.reads = tuple(dict.fromkeys(held[e] for e in need))

    def _read(self, l: int, nbytes: int, fn, *args):
        """fn(*args) on a read thread, its interval logged for the layers'
        read cost (_note_reads)."""
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self._io.append((t0, time.perf_counter(), l, nbytes))

    def _note_reads(self, K: int) -> None:
        """Each pool's reads of the step, as K tokens', to its fetch_avg
        (what auto decides by): the time some read was in flight, shared by
        bytes. Reads overlap here; a plain step's is its blocking read."""
        io, self._io = sorted(self._io), deque()
        busy, end, by_layer = 0.0, -math.inf, {}
        for t0, t1, l, nbytes in io:
            busy += max(0.0, t1 - max(t0, end))
            end = max(end, t1)
            by_layer[l] = by_layer.get(l, 0) + nbytes
        total = sum(by_layer.values())
        for l, pool in self.pools.items():
            share = by_layer.get(l, 0) / total if total else 0.0
            pool.note_reads(busy * share, K)

    def _finish(self, s: Stream) -> None:
        """The rest of the layer the stream began (its reads have arrived)."""
        l = s.l - 1
        pool = self.pools.get(l)
        if pool is None:
            s.h, s.st = self.ad.finish(l, s.st), None
            return
        t0 = time.perf_counter()
        rows = {}
        for rd in s.reads:
            if rd.rows is None:
                arrays = rd.fut.result()       # None: its experts are in slots
                rd.rows = {} if arrays is None else (
                    self.reader.experts_from_arrays(pool.triples_loc, rd.ids,
                                                    arrays))
            rows.update(rd.rows)
        self.t_adopt += time.perf_counter() - t0
        pool.call_hint = SimpleNamespace(
            rows={e: rows[e] for e in s.need if e in rows},
            idx=s.idx if self._exact_routing else None)
        try:
            s.h = self.ad.finish(l, s.st)
        finally:
            pool.call_hint = None
        s.st = None
        self._fin[l] = self._fin.get(l, 0) + 1
        if self._fin[l] == self._k_cur:
            s.done.append(l)

    def _settle(self, s: Stream) -> None:
        """Every stream has computed the layers the stream's finish
        completed and an eval has run them (the stream's next routing, or
        the step's logits): leave in each pool what the step's tokens would
        have left there one after another (the experts used last, as many
        as it has slots, in their order of use; a slot for each of them
        that was read as arrays: the least recently used expert's, or the
        lowest worth's where the pool ranked) and drop the step's reads."""
        if not s.done:
            return
        t0 = time.perf_counter()
        for l in s.done:
            pool = self.pools[l]
            held, seq = self._reads.pop(l, {}), self._used.pop(l, ())
            # the step's experts, the last used first: the ones that stay
            keep = list(dict.fromkeys(
                e for uniq in reversed(seq) for e in reversed(uniq)))
            keep = keep[:pool.n_slots]
            # those with slots go to the recent end before any adoption, so
            # an adoption takes a free slot or the slot of an expert that
            # does not stay, never one a later token of the step used
            for e in keep:
                if e in pool.id2slot:
                    pool.id2slot.move_to_end(e)
            arrays = [e for e in keep if e not in pool.id2slot]
            if arrays and l in self._ranked and pool.worth is not None:
                pool._evict_lowest(len(arrays) - len(pool.free), keep)
            for e in arrays:
                pool.adopt(e, held[e].fut.result())
            for e in reversed(keep):               # their order of use
                pool.id2slot.move_to_end(e)
            self._last[l] = {e for uniq in seq for e in uniq}
        s.done = []
        self.t_adopt += time.perf_counter() - t0

    def _ready(self, streams: list, skip):
        """(the stream to run next, the reads still awaited): the least
        advanced stream, `skip` aside (its routing is not read back yet),
        whose reads have arrived and whose predecessor is a layer ahead: it
        has begun the layer this one begins next, or, for the last layer's
        finish, gone through the stack. The cache takes the tokens in order
        in begin and in finish (Inkling 2 stores a convolution state in
        finish that the next token's finish reads)."""
        n = self.n_layers
        best, late = None, []
        for s in streams:
            if s.l > n or s is skip:
                continue
            pending = [r.fut for r in s.reads if not r.fut.done()]
            if pending:
                late += pending
            elif ((s.i == 0 or streams[s.i - 1].l > s.l)
                  and (best is None or s.l < best.l)):
                best = s
        return best, late

    # -- one step: K tokens in flight ----------------------------------------
    def _mode(self) -> None:
        """How steps run, set where a decode begins (the pools may have been
        resized since the last). As streams, K tokens in flight, each
        token's forward the model's own for one token: plain decoding bit
        for bit, and it pays where there are expert reads to hide. Or the
        step's tokens in ONE forward of the model's (the adapter's verify()
        and rollback()), where an adapter checks drafts that way (`batched`:
        Inkling 2 Mini's looped stack) or every expert is held: each weight
        is then read once for all the tokens, well under K forwards of one.
        That is not plain decoding bit for bit: a forward of K tokens
        computes each through other kernels than a forward of one (matrix
        products, attention), so logits differ in their last bits, as in
        stock's own chunked prefill, and where plain's best two tokens are
        about tied the text can take the other (the Mini: after some tens
        of tokens on a long prompt; identical on short ones).

        And the tokens of a step of copied drafts. Streams: the head's
        width (each stream in flight holds experts of its own), 4 with no
        head, none where the adapter does not split the layers. One
        forward: 16, its rows share every weight but their experts (the
        Mini's own head at 8 a step lost 7 to 16% on ordinary text against
        4, a held copy at 16 gained a third: F172). No step as streams is
        wider than the adapter can roll back (`max_streams`: Inkling 2).

        Where the kind of step changes, the self-gate's and the K
        controller's times, measured on the other kind, are dropped."""
        ad = self.ad
        was = getattr(self, "one_forward", None)
        self.one_forward = getattr(ad, "verify", None) is not None and (
            getattr(ad, "batched", False) or all_held(self.pools))
        self.k_copy = self._k_copy or (
            16 if self.one_forward else self.k_max if self.k_max > 1 else 4)
        self._streams = (None if self.one_forward
                         else getattr(ad, "max_streams", None))
        if not self.one_forward and getattr(ad, "begin", None) is None:
            self.k_copy = 1         # layers the adapter does not split
        elif self._streams:
            self.k_copy = min(self.k_copy, self._streams)
        if was is not None and was != self.one_forward:
            self._t_step, self._t_n, self._t_last = {}, {}, {}
            self._g_spt = {"on": None, "off": None}
            self._g_hold_cur = self.gate_hold
            if self.self_gate:
                self._g_begin("off")

    def step(self, toks: list[int], offset: int):
        """Returns (outputs, hiddens, snaps): outputs[i] = the greedy token
        after stream i, hiddens[i] = its final (pre-head) hidden, snaps[l][i] =
        layer-l cache snapshot taken before stream i wrote it (one forward:
        the adapter's record for its rollback). A step that raises (a read
        failed) leaves the pools usable: see _abandon. The cache then holds
        part of the step and is the caller's to drop."""
        if self.one_forward:
            logits, rows, rec = self.ad.verify(toks, self.cache)
            outs = greedy(logits[0])
            mx.eval(outs, rows)
            return (outs.tolist(),
                    [None if rows is None else rows[:, i:i + 1]
                     for i in range(len(toks))], rec)
        try:
            return self._step(toks, offset)
        except BaseException:
            self._abandon()
            raise

    def _abandon(self) -> None:
        """A step that raised: its reads are waited out (none may still be
        writing a slot when the pools are next used), and the experts of a
        read into slots that failed are not resident (pool.forget)."""
        reads, self._reads = self._reads, {}
        self._used, self._fin, self._ranked = {}, {}, set()
        self._io = deque()
        for l, held in reads.items():
            for rd in dict.fromkeys(held.values()):
                try:
                    rd.fut.result()
                except BaseException:
                    if rd.slots:
                        self.pools[l].forget(rd.ids)

    def _step(self, toks: list[int], offset: int):
        K, n = len(toks), self.n_layers
        self._reads, self._used, self._fin = {}, {}, {}
        self._ranked, self._io = set(), deque()
        self._k_cur = K
        streams = [Stream(i, t, *self.ad.prep(t, offset + i))
                   for i, t in enumerate(toks)]
        ctx = self.ad.masks(streams[0].h, self.cache)
        for s in streams:
            s.ctx = ctx
        snaps = [[None] * K for _ in range(n)]
        pend = None        # the stream whose routing the GPU is running
        while True:
            s, late = self._ready(streams, pend)
            if s is None:
                if pend is not None:         # no graph to build meanwhile
                    self._route(pend)
                    self._settle(pend)
                    pend = None
                elif late:                   # nothing can run: a read
                    t0 = time.perf_counter()
                    if KEEPALIVE:
                        keepalive_wait(late, first=True)    # GPU kept busy
                    else:
                        wait(late, return_when=FIRST_COMPLETED)
                    self.t_io += time.perf_counter() - t0
                else:
                    break                    # every stream through the stack
                continue
            if s.st is not None:
                self._finish(s)
            if s.l == n:
                s.l = n + 1                  # through the stack: its logits
                s.out = self.ad.logits(s.h)  # run while the others finish
                mx.async_eval(s.out)
                continue
            if K > 1:
                t0 = time.perf_counter()
                snaps[s.l][s.i] = self._snap(self.cache[s.l])
                self.t_snap += time.perf_counter() - t0
            self._begin(s)
            if pend is not None:             # the routing handed over before
                self._route(pend)            # this one: read back now, with
                self._settle(pend)           # this one queued behind it
            pend = s
        mx.eval(*(s.out for s in streams))
        for s in streams:                    # the layers the logits ran last
            self._settle(s)
        self._note_reads(K)
        outs =[int(greedy(s.out[0, -1]).item()) for s in streams]
        dh = getattr(self.ad, "draft_hidden", None)
        return outs, [dh(s.h, s.pos) if dh else s.h for s in streams], snaps

    def rollback(self, snaps, a: int) -> None:
        """Undo the cache writes of streams >= a (keep streams 0..a-1)."""
        t0 = time.perf_counter()
        if self.one_forward:
            self.ad.rollback(self.cache, snaps, a)
        elif snaps and a < len(snaps[0]):
            for l in range(self.n_layers):
                self._restore(self.cache[l], snaps[l][a])
        self.t_rollback += time.perf_counter() - t0

    # -- the plain fallback and the self-gate --------------------------------
    def plain_step(self, tok: int):
        """One token through the host's own forward. Returns (the greedy
        token, logits, hidden)."""
        t0 = time.perf_counter()
        logits, hidden = self.plain_fn(tok, self.cache)
        mx.eval(logits)
        self._plain_done(time.perf_counter() - t0)
        return int(greedy(logits[0, -1]).item()), logits, hidden

    def _plain_queue(self, tok):
        """A plain step on `tok` (an id, or the step before's token still
        being computed), queued on the GPU and not waited on: (its greedy
        token, hidden)."""
        t0 = time.perf_counter()
        logits, hidden = self.plain_fn(tok, self.cache)
        nxt = greedy(logits[0, -1])
        self._built = time.perf_counter() - t0      # the host's build
        mx.async_eval(nxt, *([] if hidden is None else [hidden]))
        return nxt, hidden

    def _plain_next(self, depth: int) -> bool:
        """The next step is plain whatever the head has: nothing drafts."""
        return depth <= 0 or not self.drafting or self._stale_drafter

    def _plain_done(self, dt: float, charge: bool = True) -> None:
        self.plain_steps += 1
        self.k_hist[1] = self.k_hist.get(1, 0) + 1
        if charge:
            self._gate_account(dt, 1)
        else:
            self._g_draft = 0.0

    def _gate_account(self, dt: float, tokens: int) -> None:
        """Charge `dt` seconds, plus the step's draft (or feed) time, for
        `tokens` committed tokens to the current gate phase."""
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
                 and spt > other * (1.0 + self.gate_early)
                 and not (self._held and self._g_phase == "on"))
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
                 n_tokens: int, on_token=None, stop=(),
                 context=None) -> list[int]:
        """Greedy pipelined decode. `first_tok` is the committed token whose
        forward has not run yet; `first_hidden` the hidden that produced
        it (the drafter's first pending pair). Returns the generated ids
        starting with first_tok, exactly like a plain greedy loop, ending
        early after the first committed id in `stop`. `context`: the
        prompt's tokens, which drafts by copy read (Lookup)."""
        out = [first_tok]
        for tok in self.decode(first_tok, first_hidden, offset, n_tokens, stop,
                               context):
            out.append(tok)
            if on_token is not None:
                on_token(tok)
        return out

    def decode(self, first_tok: int, first_hidden, offset: int, n_tokens: int,
               stop=(), context=None):
        """generate() one token at a time: yields the ids after first_tok. A
        server stops reading wherever a request ends, so however this ends,
        closed early included, the cache ends just after the last id out when
        a later stream of its step, or a plain step queued after it, ran it,
        else just before it. The sequence's state is set at the call, not at
        the first id read: a request that ends on its first token keeps no
        counts of the one before."""
        self._warm = self._warm_gate = 0           # a sequence's cold steps
        self._mode()
        if self.lookup is not None:
            self.lookup.reset(context if context is not None else ())
            self.lookup.push(first_tok)
        return self._decode(first_tok, first_hidden, offset, n_tokens, stop)

    def _decode(self, first_tok: int, first_hidden, offset: int,
                n_tokens: int, stop):
        lk = self.lookup
        if first_tok in stop:
            return
        n = 1
        cur = first_tok
        pairs = [(first_hidden, first_tok)]
        depth = min(self.k_max, self._streams or self.k_max) - 1
        # With every expert held a forward never waits on the host, so plain
        # steps run one ahead: the next one is built and queued on this one's
        # argmax while the GPU computes it (as mlx-lm's generate_step; same
        # ids). The committed pairs then wait for the head's next draft,
        # which takes any number, instead of a feed per token, whose eval
        # would wait behind the step in flight. Not where the forward reads
        # the token on the host (plain_lazy): building the next step would
        # wait for this one, and the gate, which is charged the build on top
        # of the step, would count the step twice.
        ahead = (self.plain_fn is not None and all_held(self.pools)
                 and getattr(self.ad, "plain_lazy", True))
        self._held = ahead
        pend = None                    # (argmax, hidden) queued on `cur`
        while n < n_tokens:
            drafts, conf = [], []
            queued = pend is not None
            copy = not queued and self._copy_ready()
            if copy:                    # (the head gets its tokens later)
                drafts, conf = self._copied()
            if depth <= 0:
                pairs = []                          # no head to keep exact
            elif copy or queued or (ahead and self._plain_next(depth)):
                if len(pairs) >= _PAIRS_HELD:
                    self._feed(pairs)
                    pairs = []
            else:
                drafts, conf = self._draft_for(pairs, depth)
                pairs = []
            if not drafts and self.plain_fn is not None:
                if ahead:
                    t0 = time.perf_counter()
                    tok, h = pend if queued else self._plain_queue(cur)
                    self._built = 0.0
                    # (not on a copy one token from being drafted from)
                    pend = (self._plain_queue(tok) if self._plain_next(depth)
                            and n + 1 < n_tokens
                            and not self._copy_ready(1) else None)
                    nxt = int(tok.item())
                    # The gate is charged what the step costs without running
                    # ahead (plus the next one's build, hidden here): with
                    # the arms some percent apart, its probes, on different
                    # stretches of text, cannot resolve the overlap's tenth,
                    # and charged as it runs the plain arm took code text
                    # from drafting at a loss (F170). A step queued before
                    # the gate re-armed drafting is neither arm's rate.
                    self._plain_done(time.perf_counter() - t0 + self._built,
                                     charge=not (queued and self.drafting))
                else:
                    nxt, _lg, h = self.plain_step(cur)
                self.steps += 1
                pairs.append((h, nxt))
                if lk is not None:
                    lk.push(nxt)
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
            if lk is not None:
                if copy:
                    lk.proposed += K - 1
                    lk.kept += a - 1
                for i in range(a):
                    lk.push(outs[i])
            tr = time.perf_counter()
            self.rollback(snaps, a)
            if copy:
                # neither arm's rate; nor is the next step's, whose draft
                # or feed takes the copy's tokens to the head
                self._g_draft, self._g_settle = 0.0, True
            else:
                # the step's cost up to where the next one can begin
                self._gate_account(dt + time.perf_counter() - tr, a)
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
            snaps = hiddens = None       # (not held into the steps after)
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
        if c >= _LK_CONF:                 # a copied draft
            return 10
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
            self._feed(tok_pairs)
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

    def _copy_ready(self, soon: int = 0) -> bool:
        """A copy has predicted _LK_MIN committed tokens (`soon`: will have
        once it predicts that many more) and has tokens to go on with."""
        lk = self.lookup
        return (lk is not None and self.k_copy > 1 and lk.ptr is not None
                and lk.run + soon >= _LK_MIN)

    def _copied(self):
        """(drafts, confs) of a step by copy: as many drafts as the copy
        has predicted tokens so far (one that ends as it starts then costs
        half a full step: a served reply lost 15 drafts to one), and as are
        each more likely kept than not at the copies' measured acceptance,
        at least one (which keeps measuring it)."""
        n, p = min(self.k_copy - 1, self.lookup.run), self._cal(_LK_CONF)
        if self.adaptive and p < 1.0:
            n = max(1, min(n, int(math.log(0.5) / math.log(max(p, 0.01)))))
        drafts = self.lookup.propose(n)
        return drafts, [_LK_CONF] * len(drafts)

    def _feed(self, pairs) -> None:
        """Gated OFF: no head call, no stage bytes, but the head's own cache
        must stay exact for when the gate re-arms, so feed it the committed
        pairs without proposing (adapters that cannot feed silently just
        re-prime on the next prefill)."""
        feed = getattr(self.ad, "feed_pairs", None)
        if feed is None:
            self._stale_drafter = True
            return
        t0 = time.perf_counter()
        feed(pairs)
        dt = time.perf_counter() - t0
        self.t_draft += dt
        self._g_draft = dt                  # this arm's cost as it runs

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
