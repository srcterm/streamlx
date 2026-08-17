"""SEF scout: same-token speculative expert fetching (NOTEBOOK F28-F41).

Each decode step, before the exact forward, a second approximate forward
of the SAME token (the "scout") is dispatched on a second GPU stream:
attention runs over read-only views of the committed KV, the MoE FFN uses
only experts already resident in the pools. Its GPU work hides inside the
exact pass's SSD-stall idle. Every gate it evaluates predicts that
layer's routing; predicted-missing experts are handed to the pools'
staged prefetch (prefetch_start), whose arrivals the exact pass adopts
via prefetch_absorb. The exact pass remains the sole source of outputs
and pool mutations — the scout writes nothing, so bit-exactness holds by
construction (A/B byte-identical gate, per F18/F27).

Threading model (load-bearing): MLX evaluation is NOT thread-safe —
concurrent eval from two threads segfaults on real-size graphs (upstream
ml-explore/mlx#2133). So every mx operation happens on the MAIN thread:
launch() builds the whole scout forward as one lazy graph (expert
selection in-graph, no per-layer CPU syncs) and dispatches it with one
mx.async_eval on the scout stream; the exact pass's wrappers then call
poll() to np-export the scout's routing a few layers ahead of themselves.
The scout graph references only MATERIALIZED arrays — cache/pool refs are
snapshotted at launch, when the previous step's logits eval has settled
every scatter — so the async-evaluated scout graph and the sync-evaluated
exact graph share no lazy nodes. Snapshot staleness is <= 1 token and
only costs prediction accuracy, never exactness.

Enable with STREAMLX_SCOUT=1 (or load_streaming_model(scout="on")). The
driver self-gates out of regimes where it cannot win — see ScoutDriver.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from .stindex import PARTS, PROJS

# The driver currently building a scout graph (main thread only; set for
# the duration of launch()'s model call). Patched gate classes stash
# their output here so the scout wrapper reuses the caller's gate nodes
# instead of re-invoking the gate.
_ACTIVE: "ScoutDriver | None" = None


class ReadOnlyCacheView:
    """One layer's KV cache, frozen at construction, never written.

    update_and_fetch returns (committed prefix + new kv) WITHOUT storing.
    Buffer refs are captured at construction — at scout launch everything
    is materialized, and the exact pass's own cache append only REBINDS
    the base attrs, so the view can never see (or share) an in-flight
    lazy node. The slice to the captured offset excludes the current
    token however far the exact pass has advanced. When `capture` is set
    (t+2 tier) the probe's own k/v is kept for the chained second probe.
    Unknown attributes proxy to the base cache (scalar reads only)."""

    # 0 = attend over the full committed prefix; N = only the last N
    # prefix tokens (keys are stored post-rope, so this is plain
    # sliding-window attention). Footprint lever for long contexts.
    WINDOW = int(os.environ.get("STREAMLX_SCOUT_WINDOW", "0"))

    def __init__(self, base, capture=False):
        self._base = base
        self._keys = base.keys
        self._values = base.values
        self._offset = base.offset
        self._capture = capture
        self.captured = None         # (k_new, v_new) lazy nodes

    @property
    def offset(self):
        return self._offset

    def update_and_fetch(self, keys, values):
        if self._capture:
            self.captured = (keys, values)
        bk, bv = self._keys, self._values
        if bk is None or self._offset == 0:
            return keys, values
        end = min(self._offset, bk.shape[2])
        start = max(0, end - self.WINDOW) if self.WINDOW else 0
        return (mx.concatenate([bk[..., start:end, :], keys], axis=2),
                mx.concatenate([bv[..., start:end, :], values], axis=2))

    def __getattr__(self, name):
        if name in ("_base", "_keys", "_values", "_offset", "_capture"):
            raise AttributeError(name)
        return getattr(self._base, name)


class _T2CacheView:
    """Probe-2 cache view (t+2 tier, F37/F38): committed exact prefix +
    the single approximate k/v probe 1 just produced + probe 2's own new
    k/v, never stored. offset is probe 2's position (prefix + 1). F31's
    compounding collapse does not apply to ONE approximate entry."""

    def __init__(self, view: ReadOnlyCacheView):
        self._keys = view._keys
        self._values = view._values
        self._offset = view._offset + 1
        self._pk, self._pv = view.captured

    @property
    def offset(self):
        return self._offset

    def update_and_fetch(self, keys, values):
        bk, bv = self._keys, self._values
        if bk is None or self._offset == 1:
            return (mx.concatenate([self._pk, keys], axis=2),
                    mx.concatenate([self._pv, values], axis=2))
        end = min(self._offset - 1, bk.shape[2])
        return (mx.concatenate([bk[..., :end, :], self._pk, keys], axis=2),
                mx.concatenate([bv[..., :end, :], self._pv, values], axis=2))


_SIMPLE_CACHES = ("KVCache", "RotatingKVCache")


def make_view(c, capture=False):
    """Read-only stand-in for one cache entry.

    KVCache gets the tuned concat view. Everything else — CacheList,
    PoolingCache, MLA rotating rings with direct attr writes, unknown
    future types — gets a generic COPY-view: mx arrays are immutable, so
    a shallow copy of the cache object is a semantically exact
    scout-local cache (every method runs unchanged; every mutation lands
    on the copy; validated bitwise against deepseek_v4, F41).
    RotatingKVCache also uses the copy-view for exact ring semantics,
    except when the t+2 capture path needs the tuned view's kv hook."""
    tn = type(c).__name__
    if tn == "KVCache" or (capture and tn in _SIMPLE_CACHES):
        return ReadOnlyCacheView(c, capture=capture)
    return _copy_cache_view(c)


def _copy_cache_view(c):
    import copy

    v = copy.copy(c)
    # containers (e.g. deepseek_v4 CacheList) still share children after
    # a shallow copy — copy cache-like children one level down
    for attr, val in list(vars(v).items()):
        if (isinstance(val, (list, tuple)) and val
                and all(hasattr(x, "update_and_fetch") or hasattr(x, "offset")
                        for x in val)):
            setattr(v, attr, type(val)(_copy_cache_view(x) for x in val))
    return v


def scout_graph_forward(wrapper, x, indices, scores, ctx):
    """The wrapper's scout-mode FFN, built as a pure graph (no CPU
    syncs): top-m resident experts by router score, everything else
    zeroed; m >= k (the default) means all resident routed experts.
    Stashes (indices, scores, resident-mask) lazily for later harvest.
    Touches no exact-pass state: no ensure, no LRU touch, no stats."""
    from .pool import switch_forward

    pool = wrapper.pool
    slot_mx, bufs = ctx.snap[pool.layer]

    w = None
    lg, ctx._last_gate = ctx._last_gate, None
    if scores is not None:
        w = scores.astype(mx.float32)
    elif isinstance(lg, tuple) and len(lg) == 2:
        # the caller's own gate output (stashed): zero extra graph ops
        indices, w = lg[0], lg[1].astype(mx.float32)
    else:
        gate = ctx.gates.get(pool.layer)
        if gate is not None:
            out = gate(x)               # fallback: same math, extra ops
            if isinstance(out, tuple):
                indices, w = out[0], out[1].astype(mx.float32)
    if w is None:                       # no scores available: given order
        w = -mx.arange(indices.shape[-1], dtype=mx.float32) * mx.ones(
            indices.shape[:-1] + (1,), dtype=mx.float32)

    kth = min(ctx.m, indices.shape[-1]) - 1
    slots = mx.take(slot_mx, indices)                     # [B,T,k] uint32
    res = slots != np.uint32(0xFFFFFFFF)                  # pool.SENTINEL
    if kth < 0:                                           # m == 0
        y = mx.zeros(x.shape[:2] + indices.shape[-1:] + (x.shape[-1],),
                     dtype=x.dtype)
    else:
        if ctx.m >= indices.shape[-1]:
            keep = res          # all resident: no selection ops needed
        else:
            wm = mx.where(res, w, mx.array(-1e30, dtype=mx.float32))
            thr = -mx.partition(-wm, kth=kth, axis=-1)[..., kth:kth + 1]
            keep = mx.logical_and(wm >= thr, res)         # ties keep extra
        y = switch_forward(x, mx.where(keep, slots, mx.array(0, mx.uint32)),
                           bufs, pool.qparams)
        y = y * keep[..., None].astype(y.dtype)
    ctx.pending[pool.layer] = (indices, w, res)           # lazy, tiny
    return y


class ScoutDriver:
    """Owns the scout stream, per-token graph dispatch, and harvest.

    The scout self-selects its regime; outside it, the norm-ratio
    prefetch resumes automatically. Gates (calibration in NOTEBOOK
    F33/F35/F40):
      - would-miss pressure ((misses + pf_used) / lookups) below
        min_pressure: near-coverage budgets, nothing to predict;
      - global routed-set agreement below min_global_agree: residency
        too thin for the scout's own resident-only approximation;
      - build cost above max_build_frac of the step time (or the
        absolute backstop): memory pressure / throttling has made the
        scout itself expensive;
      - per-layer agreement below min_agree stops that layer's
        emissions only.
    """

    def __init__(self, model, pools, gates, orig_call):
        self.model = model
        self.pools = pools
        self.gates = gates
        self.orig_call = orig_call
        # Scout FFN breadth: m >= k means ALL resident routed experts —
        # the default, since scout compute hides in stalls while fidelity
        # buys emission precision (F33). Smaller m only saves GPU flops.
        self.m = int(os.environ.get("STREAMLX_SCOUT_M", "999"))
        # poll() harvests predictions for layers (L, L+lead].
        self.lead = int(os.environ.get("STREAMLX_SCOUT_LEAD", "8"))
        # Max predicted-missing experts emitted per layer (score-ranked);
        # 0 = uncapped; -1 = MUTE: scout computes but never emits (the
        # clock-artifact control mode, F33).
        self.emit_cap = int(os.environ.get("STREAMLX_SCOUT_EMIT", "3"))
        self.min_agree = float(
            os.environ.get("STREAMLX_SCOUT_MIN_AGREE", "0.5"))
        self.min_global_agree = float(
            os.environ.get("STREAMLX_SCOUT_MIN_GLOBAL_AGREE", "0.75"))
        self.min_pressure = float(
            os.environ.get("STREAMLX_SCOUT_MIN_PRESSURE", "0.05"))
        # Build-cost gate is RELATIVE to step time (an absolute limit
        # would conflate "machine is slow" with "scout isn't worth it");
        # the absolute cap is a runaway backstop.
        self.max_build_frac = float(
            os.environ.get("STREAMLX_SCOUT_MAX_BUILD_FRAC", "0.55"))
        self.max_build = float(
            os.environ.get("STREAMLX_SCOUT_MAX_BUILD_MS", "500")) / 1e3
        # t+2 tier (F37/F38): probe 2 chains lazily on probe 1's argmax
        # and predicts the NEXT token's routing with a full token of
        # lead. Doubles the graph-encode cost; opt-in.
        self.t2 = os.environ.get("STREAMLX_SCOUT_T2", "0") == "1"
        self.stream = mx.new_stream(mx.gpu)
        # prefetch-read submitter: read_experts_raw fans spans onto the
        # reader's own pool, so submissions need their own executor
        self.pf_ex = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="streamlx-sc-pf")
        self.building = False
        self.enabled = True
        self.seq = 0
        self.snap = {}                  # layer -> (slot_table mx, buf refs)
        self.pending = {}               # layer -> (inds, scores, res) lazy
        self.harvested: set = set()
        self._last_gate = None          # stash from patched gate classes
        self.last_pred: dict[int, tuple[int, set]] = {}
        self.agree = {L: [0, 0] for L in pools}
        self._p2: dict = {}             # probe-2 pending (t+2)
        self._p2_seq = -1
        self.t2_pred: dict[int, tuple[int, set]] = {}
        self.agree2 = {L: [0, 0] for L in pools}
        self._miss0 = 0
        self._pressure_ema = None
        self._last_tot = (0, 0)
        self._build_ema = None
        self._interval_ema = None
        self._last_launch = None
        # telemetry
        self.launched = 0
        self.emitted = 0
        self.t2_emitted = 0
        self.harvest_evals = 0
        self.errors = 0
        self.t_build = 0.0      # launch(): snapshot + graph + dispatch, s
        self.t_snap = 0.0
        self.t_graph = 0.0
        self.t_dispatch = 0.0
        self.t_harvest = 0.0    # blocked in harvest evals, s

    # -- per-token entry, main thread, before the exact forward ------------
    def launch(self, inputs, cache):
        if not self.enabled or not cache:
            return
        if self.launched and self.launched % 32 == 0:
            self._pressure_gate()
            if not self.enabled:
                return
        tid = int(inputs.reshape(-1)[0].item())  # host already materialized
        self.pending = {}
        self.harvested = set()
        self.seq += 1
        self._harvest_t2()      # previous step's probe-2 -> this step's reads
        self.launched += 1
        self._miss0 = sum(p.misses for p in self.pools.values())
        tb = time.perf_counter()
        self._launch_eager(tid, cache)
        self._build_gate(time.perf_counter() - tb)

    def _launch_eager(self, tid, cache):
        t0 = time.perf_counter()
        if self.t2 and any(type(c).__name__ not in _SIMPLE_CACHES
                           for c in cache):
            self.t2 = False     # capture path is simple-cache only (v1)
            print("[streamlx] scout t+2 off: unsupported cache types",
                  file=sys.stderr)
        views = [make_view(c, capture=self.t2) for c in cache]
        self.snap = {L: (mx.array(p.slot_table),
                         {pr: {pt: p.pool[pr][pt] for pt in PARTS}
                          for pr in PROJS})
                     for L, p in self.pools.items()}
        self.building = True
        self.t_snap += time.perf_counter() - t0
        t0 = time.perf_counter()
        global _ACTIVE
        try:
            _ACTIVE = self
            with mx.stream(self.stream):
                out1 = self.orig_call(self.model, mx.array([[tid]]),
                                      cache=views)
                if self.t2:
                    # probe 2 chains LAZILY on probe 1's argmax: the
                    # predicted token never touches the main thread, so
                    # both probes dispatch here and interleave together
                    pred_tok = mx.argmax(
                        out1[:, -1:, :], axis=-1).astype(mx.uint32)
                    views2 = [_T2CacheView(v) for v in views]
                    p1 = self.pending
                    self.pending = {}
                    self.orig_call(self.model, pred_tok, cache=views2)
                    self._p2 = self.pending
                    self._p2_seq = self.seq
                    self.pending = p1
            self.t_graph += time.perf_counter() - t0
            t1 = time.perf_counter()
            flat = [a for t in self.pending.values() for a in t]
            flat += [a for t in self._p2.values() for a in t]
            if flat:
                mx.async_eval(*flat)    # one dispatch; GPU interleaves
            self.t_dispatch += time.perf_counter() - t1
            # Drop snapshot refs now: python refs held across the step
            # would block buffer donation on every exact-pass pool write
            # (each write would then copy the whole tensor).
            self.snap = {}
        except Exception:
            self.errors += 1
            if self.errors == 1:
                import traceback
                print("[streamlx] scout error (first occurrence):",
                      file=sys.stderr)
                traceback.print_exc()
            if self.errors >= 3:
                self.enabled = False
                print("[streamlx] scout disabled after repeated errors",
                      file=sys.stderr)
            self.pending = {}
            self._p2 = {}
        finally:
            _ACTIVE = None
            self._last_gate = None
            self.building = False

    def _build_gate(self, bt):
        self.t_build += bt
        e = self._build_ema
        self._build_ema = bt if e is None else 0.8 * e + 0.2 * bt
        now = time.perf_counter()
        if self._last_launch is not None:
            iv = now - self._last_launch
            e = self._interval_ema
            self._interval_ema = iv if e is None else 0.8 * e + 0.2 * iv
        self._last_launch = now
        if self.launched < 16 or self._interval_ema is None:
            return
        frac = self._build_ema / self._interval_ema
        if frac > self.max_build_frac or self._build_ema > self.max_build:
            self.enabled = False
            self.pending = {}
            self._p2 = {}
            print(f"[streamlx] scout auto-off: build cost "
                  f"{self._build_ema * 1e3:.0f} ms/token = {frac:.0%} of "
                  f"step time (limit {self.max_build_frac:.0%}, abs "
                  f"{self.max_build * 1e3:.0f} ms; memory pressure or "
                  f"throttling; norm-ratio prefetch resumes)",
                  file=sys.stderr)

    def _harvest_t2(self):
        """Emit the previous step's probe-2 predictions as prefetch reads
        for THIS step. Called at the top of launch, when probe 2's arrays
        are guaranteed materialized — every layer (shallow included) gets
        its prediction at step start."""
        p2, self._p2 = self._p2, {}
        if self.emit_cap < 0:           # muted (control mode)
            return
        if not p2 or self._p2_seq != self.seq - 1:
            return                      # stale (skipped step / new request)
        try:
            t0 = time.perf_counter()
            mx.eval(*[a for tvals in p2.values() for a in tvals])
            self.t_harvest += time.perf_counter() - t0
        except Exception:
            self.errors += 1
            return
        for L, (inds, w, res) in p2.items():
            idx = np.array(inds).reshape(-1)
            ws = np.array(w).reshape(-1)
            rs = np.array(res).reshape(-1)
            self.t2_pred[L] = (self.seq, set(idx.tolist()))
            a = self.agree2.get(L)
            if a is not None and a[1] >= 512 and a[0] < self.min_agree * a[1]:
                continue
            order = np.argsort(-ws)
            missing = [int(idx[o]) for o in order if not rs[o]]
            if self.emit_cap > 0:
                missing = missing[:self.emit_cap]
            if missing:
                self.t2_emitted += len(missing)
                self.pools[L].prefetch_start(missing, self.pf_ex,
                                             source="t2")

    # -- called from the exact pass's wrapper at each MoE layer ------------
    def poll(self, layer):
        """Harvest the scout's routing for layers in (layer, layer+lead]
        and start prefetch reads for predicted-missing experts. Gated on
        the step having produced a demand miss (no stall idle means a
        not-yet-done scout would serialize, and predictions are worthless
        there anyway). The window is materialized with ONE mx.eval,
        stride-gated so it does not degenerate to a sync per layer."""
        if self.emit_cap < 0 or not self.pending:   # -1 = muted (control)
            return
        if sum(p.misses for p in self.pools.values()) <= self._miss0:
            return
        window = [j for j in sorted(self.pending)
                  if layer < j <= layer + self.lead
                  and j not in self.harvested]
        if not window or window[0] > layer + 2:
            return
        self.harvested.update(window)
        try:
            self.harvest_evals += 1
            t0 = time.perf_counter()
            mx.eval(*[a for j in window for a in self.pending[j]])
            self.t_harvest += time.perf_counter() - t0
        except Exception:
            self.errors += 1
            return
        for j in window:
            inds, w, res = self.pending[j]
            idx = np.array(inds).reshape(-1)          # materialized: cheap
            ws = np.array(w).reshape(-1)
            rs = np.array(res).reshape(-1)
            self.last_pred[j] = (self.seq, set(idx.tolist()))
            a = self.agree.get(j)
            if a is not None and a[1] >= 512 and a[0] < self.min_agree * a[1]:
                continue                # unpredictable layer: don't emit
            order = np.argsort(-ws)
            missing = [int(idx[o]) for o in order if not rs[o]]
            if self.emit_cap:
                missing = missing[:self.emit_cap]
            if missing:
                self.emitted += len(missing)
                self.pools[j].prefetch_start(missing, self.pf_ex,
                                             source="scout")

    # -- called from the exact pass (decode path, cheap) -------------------
    def observe(self, layer, actual):
        aset = None
        lp = self.last_pred.get(layer)
        if lp is not None and lp[0] == self.seq:
            aset = set(actual)
            a = self.agree[layer]
            a[0] += len(lp[1] & aset)
            a[1] += len(aset)
        t2 = self.t2_pred.get(layer)
        if t2 is not None and t2[0] == self.seq:
            aset = set(actual) if aset is None else aset
            a = self.agree2[layer]
            a[0] += len(t2[1] & aset)
            a[1] += len(aset)

    def _pressure_gate(self):
        hits = misses = used = 0
        for p in self.pools.values():
            hits += p.hits
            misses += p.misses
            used += p.pf_used
        look, wm = hits + misses, misses + used
        dl, dw = look - self._last_tot[0], wm - self._last_tot[1]
        self._last_tot = (look, wm)
        if dl <= 0:
            return
        r = dw / dl
        e = self._pressure_ema
        self._pressure_ema = r if e is None else 0.8 * e + 0.2 * r
        if self.launched >= 128 and self._pressure_ema < self.min_pressure:
            self.enabled = False
            self.pending = {}
            print(f"[streamlx] scout auto-off: would-miss pressure "
                  f"{self._pressure_ema:.1%} < {self.min_pressure:.0%} "
                  f"(norm-ratio prefetch resumes)", file=sys.stderr)
            return
        ah = sum(a[0] for a in self.agree.values())
        at = sum(a[1] for a in self.agree.values())
        if self.launched >= 96 and at and ah / at < self.min_global_agree:
            self.enabled = False
            self.pending = {}
            self._p2 = {}
            print(f"[streamlx] scout auto-off: agreement {ah / at:.1%} < "
                  f"{self.min_global_agree:.0%} — residency too thin for "
                  f"the scout's own approximation (norm-ratio prefetch "
                  f"resumes)", file=sys.stderr)

    @property
    def stats(self):
        ah = sum(a[0] for a in self.agree.values())
        at = sum(a[1] for a in self.agree.values())
        a2h = sum(a[0] for a in self.agree2.values())
        a2t = sum(a[1] for a in self.agree2.values())
        return {"scout_launched": self.launched,
                "scout_t2": self.t2,
                "t2_emitted": self.t2_emitted,
                "t2_agree": round(a2h / a2t, 3) if a2t else None,
                "scout_emitted": self.emitted,
                "scout_harvest_evals": self.harvest_evals,
                "scout_t_build_s": round(self.t_build, 2),
                "scout_t_snap_s": round(self.t_snap, 2),
                "scout_t_graph_s": round(self.t_graph, 2),
                "scout_t_dispatch_s": round(self.t_dispatch, 2),
                "scout_t_harvest_s": round(self.t_harvest, 2),
                "scout_errors": self.errors,
                "scout_agree": round(ah / at, 3) if at else None,
                "scout_pressure": (round(self._pressure_ema, 3)
                                   if self._pressure_ema is not None
                                   else None),
                "scout_build_ema_ms": (round(self._build_ema * 1e3, 1)
                                       if self._build_ema is not None
                                       else None),
                "scout_enabled": self.enabled}


def install_scout(model, pools):
    """Wire the scout onto a streaming-wrapped model: collect per-layer
    gate refs, wrap the model class's __call__ so every single-token call
    first builds+dispatches a scout graph for the same token, and hand
    each wrapper the driver. Returns the driver."""
    from .integrate import _moe_block

    gates = {}
    for i, layer in enumerate(model.layers):
        if i in pools:
            gates[i] = getattr(_moe_block(layer), "gate", None)

    # Patch each gate CLASS so a scout build reuses the caller's gate
    # output instead of re-invoking the gate; the exact pass pays one
    # None-check per gate call.
    for gcls in {type(g) for g in gates.values() if g is not None}:
        if getattr(gcls, "_streamlx_gate_stash", False):
            continue
        gorig = gcls.__call__

        def gwrapped(self, x, _orig=gorig):
            out = _orig(self, x)
            if _ACTIVE is not None:
                _ACTIVE._last_gate = out
            return out

        gcls.__call__ = gwrapped
        gcls._streamlx_gate_stash = True

    cls = type(model)
    if getattr(cls, "_streamlx_scout_orig", None) is None:
        orig = cls.__call__

        def wrapped(self, inputs, *a, **kw):
            drv = getattr(self, "_streamlx_scout", None)
            if (drv is not None and not drv.building
                    and getattr(inputs, "ndim", 0) == 2
                    and inputs.shape[0] == 1):
                if inputs.shape[1] == 1:
                    cache = kw.get("cache", a[0] if a else None)
                    try:
                        drv.launch(inputs, cache)
                    except Exception:
                        drv.errors += 1
                else:
                    drv._p2 = {}    # prefill = new context: t+2 is stale
            return orig(self, inputs, *a, **kw)

        cls.__call__ = wrapped
        cls._streamlx_scout_orig = orig

    driver = ScoutDriver(model, pools, gates, cls._streamlx_scout_orig)
    model._streamlx_scout = driver
    for i in pools:
        _moe_block(model.layers[i]).switch_mlp._scout = driver
    return driver
