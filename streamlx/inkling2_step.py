"""Inkling 2's layer visit in fewer kernels, bit for bit mlx-lm's.

A decode step of mlx-lm's inkling2 queues about 150 GPU kernels a layer
visit (the Mini: 48 visits, 7,400 kernels a token), most of them a cast, an
add or a mask over one vector. `fuse(model)` gives the layers a visit that
computes the same values with about half the kernels. What that is worth
depends on what bounds the step. On a base M4 the Mini's weight reads are 47
of a token's 55 ms and the kernels removed cost half a microsecond each:
+3% decode. The host's work a token (building the graph, encoding its
kernels) falls by 30%, which is the bound where memory is faster.

  - a value cast twice is cast once (the residual in float32, a convolution's
    input), and constants are cast when fused, not every visit (the static
    convolution weights, the relative-position table);
  - runs of elementwise ops are one compiled kernel each (mx.compile, as
    mlx-lm's own clips and activation): a convolution's tail with the
    residual around it, the attention output's gate, a clip's two halves,
    a cast through the stream's dtype and back;
  - what a token's visits share is computed once a token: the distances from
    the query to a window's keys, with what they hide (one table for the
    sliding layers, one for the global), and a global layer's query scale;
  - a sliding layer's three message gates share the norm of its input;
  - a norm's weight is cast to the norm's dtype once, not by every call
    (mx.fast.rms_norm casts it when it differs from the input's);
  - a convolution state is a view of the window that made it (no gather).

Nothing here changes a float: every rewrite keeps the operations and their
order element by element, in the dtypes fuse() checks it in. It checks both
ways it could stop being so: every compiled group against its operations run
one by one, at full precision (a device whose compiler contracts a multiply
and an add rounds differently: the CPU's does, in the message gate's; Metal's
does not), and a visit of each kind of layer against mlx-lm's own, every
output and cache entry compared (an mlx-lm whose layer no longer is the one
mirrored here, or a layer whose parameters' dtypes give its projections
another dtype than the stream's: mlx-lm casts a convolution's output to its
input's dtype, the fused visit to the stream's). A kind of layer that differs
keeps mlx-lm's visit (every layer does when a group differs), and so does a
visit whose stream comes in another dtype than the check's (input embeddings
of another dtype). What the visit makes from parameters (casts, the static
convolution weights, the message queries) is made again when a parameter is
replaced; loading weights of other dtypes needs fuse() again.

Visits of up to _MAX_ROWS tokens without padding take this path (a decode
step, a step's drafts checked together); a prompt chunk and a padded batch
run mlx-lm's own code, on the same caches.
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten

F32 = mx.float32
_MAX_ROWS = 16
# STREAMLX_FUSED=0: mlx-lm's own visit (for an A/B; the values are the same)
ENABLED = os.environ.get("STREAMLX_FUSED", "1") != "0"


def _stock():
    try:    # mlx-lm's layout from #2082: the model is a package
        from mlx_lm.models.inkling2 import layers as stock
    except ImportError:
        import mlx_lm.models.inkling2 as stock
    return stock


# what the fused visit calls of mlx-lm's layer module
_NEEDS = ("DecoderLayer", "SlidingDecoderLayer", "FullDecoderLayer",
          "unweighted_rms_norm", "_conv_state", "_positions",
          "_key_positions", "_depth_attention")

_FNS: dict = {}


def _groups(dtype) -> dict:
    """The elementwise groups of a visit for a stream of `dtype`, as plain
    functions: {name: (function, its arguments' kinds)}. A kind is what
    _groups_exact() draws to try the function: s an array of the stream's
    dtype, f a float32 one, p a positive float32 one, c a float32 scalar."""
    def through(x):                    # float32 as the stream holds it
        return x.astype(dtype).astype(F32)

    def both(x):
        y = x.astype(dtype)
        return y, y.astype(F32)

    def round32(x):
        return through(x)

    def tail(out, resid, c):           # a convolution's output and residual
        return ((out.astype(F32) + resid) / c).astype(dtype)

    def tail32(out, resid, c):
        return through((out.astype(F32) + resid) / c)

    def tail_add(out, resid, c, base):
        return base + through((out.astype(F32) + resid) / c)

    def final(out, resid, c, residual):
        return (residual + through((out.astype(F32) + resid) / c)
                ).astype(dtype)

    def out_gate(o, gate):
        return (o.astype(F32) * mx.sigmoid(gate.astype(F32))
                ).astype(gate.dtype)

    def gate_out(f, g, h, m):          # a message gate's output
        return (1 - mx.tanh(f)) * h + mx.tanh(g) * m

    def clip_in(x):
        s = x.astype(F32)
        return s, s * s

    def clip_in_scaled(x, scale):
        s = x.astype(F32) * scale
        return s, s * s

    def clipped(s, sq_sum, thr):
        norm = mx.sqrt(sq_sum)
        return (s * mx.minimum(thr / (norm + 1e-6), 1.0)).astype(dtype)

    def clip_out(s, sq_sum, thr):
        return clipped(s, sq_sum, thr)

    def clip_out_scaled(s, sq_sum, thr, tau):
        return (clipped(s, sq_sum, thr).astype(F32) * tau).astype(dtype)

    def moe_weighted(ys, w):
        return ys.astype(F32) * w

    def dyn_weighted(windows, static, dyn):
        return windows * (static + dyn.astype(F32))

    return {"both": (both, "f"), "round32": (round32, "f"),
            "tail": (tail, "sfc"), "tail32": (tail32, "sfc"),
            "tail_add": (tail_add, "sfcf"), "final": (final, "sfcf"),
            "out_gate": (out_gate, "ss"), "gate_out": (gate_out, "ffff"),
            "clip_in": (clip_in, "s"),
            "clip_in_scaled": (clip_in_scaled, "sf"),
            "clip_out": (clip_out, "fpc"),
            "clip_out_scaled": (clip_out_scaled, "fpcc"),
            "moe_weighted": (moe_weighted, "sf"),
            "dyn_weighted": (dyn_weighted, "ffs")}


def _fns(dtype):
    """The groups compiled, a kernel each (mx.compile, as mlx-lm's own clips
    and activation)."""
    fn = _FNS.get(dtype)
    if fn is None:
        fn = _FNS[dtype] = SimpleNamespace(**{
            name: mx.compile(f, shapeless=True)
            for name, (f, _kinds) in _groups(dtype).items()})
    return fn


def _groups_exact(dtype, draws: int = 3, size: int = 4096) -> bool:
    """Whether every compiled group gives the bits of its operations run one
    by one, on this device: a compiler is free to contract a multiply and an
    add into one operation, which rounds once where two kernels round twice
    (the CPU's does, in the message gate's; Metal's does not)."""
    fn, key = _fns(dtype), mx.random.key(1)
    for name, (plain, kinds) in _groups(dtype).items():
        compiled = getattr(fn, name)
        for _ in range(draws):
            key, *ks = mx.random.split(key, len(kinds) + 1)
            args = []
            for kind, k in zip(kinds, ks):
                if kind == "c":
                    a = 1.0 + mx.random.uniform(shape=(), key=k)
                else:
                    a = 3.0 * mx.random.normal((1, 1, size), key=k)
                    a = mx.abs(a) if kind == "p" else a
                args.append(a.astype(dtype if kind == "s" else F32))
            mx.eval(args)
            if not _same(_listed(plain(*args)), _listed(compiled(*args))):
                return False
    return True


def _listed(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _scalar(v: float) -> mx.array:
    return mx.array(v, dtype=F32)


class _Memo:
    """What a token's visits share, keyed by what it depends on; a few
    entries (a token has two kinds of window)."""

    def __init__(self, size: int = 4):
        self.size, self.d = size, {}

    def get(self, key, make):
        v = self.d.get(key)
        if v is None:
            if len(self.d) >= self.size:
                self.d.pop(next(iter(self.d)))
            v = self.d[key] = make()
        return v


_MASKS = _Memo()
_TAUS = _Memo()


class _Made:
    """Values made from a parameter once, made again when the parameter is
    replaced (model.load_weights, Module.update): get() reads it, make(it,
    key) makes the value for a key (a dtype, an index, or None)."""

    def __init__(self, get, make):
        self.get, self.make, self.src, self.made = get, make, None, {}

    def __call__(self, key=None):
        src = self.get()
        if src is not self.src:
            self.src, self.made = src, {}
        v = self.made.get(key)
        if v is None:
            v = self.made[key] = self.make(src, key)
            mx.eval(v)
        return v


class _Norm:
    """An nn.RMSNorm's call with its weight already in the result's dtype
    (mx.fast.rms_norm casts the weight on every call when it is not)."""

    def __init__(self, norm):
        self.norm = norm
        self.cast = _Made(lambda: norm.weight, lambda w, dtype: w.astype(
            mx.result_type(dtype, w.dtype)))

    def __call__(self, x):
        w = self.norm.weight
        if w.dtype != x.dtype:
            w = self.cast(x.dtype)
        return mx.fast.rms_norm(x, w, self.norm.eps)


def _clip(fn, x, scale, thr, tau=None):
    """mlx-lm's _l2_clip (and, with tau, the scale by it that follows)."""
    s, sq = fn.clip_in(x) if scale is None else fn.clip_in_scaled(x, scale)
    sq = sq.sum(axis=-1, keepdims=True)
    if tau is None:
        return fn.clip_out(s, sq, thr)
    return fn.clip_out_scaled(s, sq, thr, tau)


def _window(conv, x32, state):
    """A short convolution's window (its state then the input, float32)."""
    if state is None:
        state = mx.zeros((x32.shape[0], conv.kernel_size - 1, x32.shape[-1]),
                         F32)
    return mx.concatenate([state, x32], axis=1)


def _sconv(conv, x32, state):
    """ShortConvolution up to its conv1d: (the conv's output, the window)."""
    window = _window(conv, x32, state)
    return conv.conv1d(window.astype(conv.conv1d.weight.dtype)), window


class _Fused:
    """The fused visit, mixed into mlx-lm's two layer classes (fuse())."""

    def _fused_ok(self, hidden_states, cache, conv_mask) -> bool:
        """A visit of the kind fuse() checked: up to _MAX_ROWS tokens, no
        padding, and a stream in the dtype its probe drew it in (the
        embedding's when it was fused: the fused visit casts to the stream's
        dtype where mlx-lm casts to a projection's, the same dtype only
        there); any other is mlx-lm's. (Block states, messages, and keys and
        values cached in another dtype go through the same calls in both.)"""
        return (ENABLED and cache is not None and conv_mask is None
                and hidden_states.shape[1] <= _MAX_ROWS
                and hidden_states.dtype == self._k.dtype
                and cache[1].lengths is None
                and self.mlp.sharding_group is None)

    def __call__(self, hidden_states, block_states, *, cache=None,
                 conv_mask=None, messages=None):
        if not self._fused_ok(hidden_states, cache, conv_mask):
            kw = {} if messages is None else {"messages": messages}
            return super().__call__(hidden_states, block_states, cache=cache,
                                    conv_mask=conv_mask, **kw)
        k = self._k
        dtype = hidden_states.dtype
        fn = _fns(dtype)
        conv = cache[1]
        states = list(conv.cache)
        h32 = hidden_states.astype(F32)
        normed_residual = k.rms(h32)

        attention_inputs = None
        if self.attention_input is not None:
            attention_inputs = self.attention_input(block_states,
                                                    normed_residual)
        a = self._fused_attention(hidden_states, attention_inputs, cache,
                                  states, messages, fn)
        a32 = a.astype(F32)
        out, states[2] = _sconv(self.post_attention_convolution, a32,
                                states[2])
        if self.is_block_boundary:
            residual = fn.tail32(out, a32, k.c_short)
            block = normed_residual[None]
            block_states = (block if block_states is None
                            else mx.concatenate([block_states, block]))
        else:
            residual = fn.tail_add(out, a32, k.c_short, h32)

        x = self.mlp_input(block_states, k.rms(residual))[..., 0, :]
        if messages is not None:
            x = self._fused_gate("mlp", x, k.rms(x), messages[..., 3, :], fn)
        # the experts take the stream's dtype, the router float32 of it
        xn, xn32 = fn.both(k.post_norm(x))
        inds, weights = self.mlp.gate(xn32)
        y = fn.moe_weighted(self.mlp.experts(xn, inds), weights[..., None])
        y32 = fn.round32(y.sum(axis=-2))
        out, states[3] = _sconv(self.post_mlp_convolution, y32, states[3])
        h = fn.final(out, y32, k.c_short, residual)

        n, rows = k.keep, hidden_states.shape[1]
        # one token: the window's tail as it is (the window is the old state
        # and this token's row); several, or a host recording the windows
        # (the loop adapter's rollback): mlx-lm's own gather
        gather = k.stock._conv_state
        view = rows == 1 and gather is k.conv_state
        for i, window in enumerate(states):
            conv[i] = window[:, 1:] if view else gather(window, None, n)
        conv.advance(rows)
        return h, block_states

    # -- message gates -----------------------------------------------------
    def _fused_gate(self, name, hidden, hidden_normed, message, fn):
        """MessageGate from its input's norm; its output one kernel. (Its
        two projections stay two matmuls: one over their stacked weights
        rounds some elements differently on a real checkpoint.)"""
        gate = getattr(self, name + "_message_gate")
        s = mx.concatenate([hidden_normed, self._k.rms(message)], axis=-1)
        s = gate.down_proj(s.astype(gate.down_proj.weight.dtype)).astype(F32)
        return fn.gate_out(gate.forget_gate_proj(s),
                           gate.message_gate_proj(s), hidden, message)

    def _fused_mix(self, states, messages, fn):
        q, kk, v = states
        k = self._k
        nq = k.rms(q)
        nk = nq if kk is q else k.rms(kk)
        nv = nq if v is q else (nk if v is kk else k.rms(v))
        return (self._fused_gate("q", q, nq, messages[..., 0, :], fn),
                self._fused_gate("k", kk, nk, messages[..., 1, :], fn),
                self._fused_gate("v", v, nv, messages[..., 2, :], fn))

    # -- attention ---------------------------------------------------------
    def _fused_attend(self, h, key_states, value32, cache, fn, tau=None):
        """Attention._attend from the convolved keys (stream dtype) and
        values (float32 of the stream's)."""
        attn, k = self.self_attn, self._k
        dtype = h.dtype
        B, L, _ = h.shape
        gate = attn.gate_proj(h)
        sinks = attn.sink_proj(h)
        q = attn.q_proj(h).reshape(B, L, attn.num_heads, -1)
        r = attn.r_proj(h).reshape(B, L, attn.num_heads, -1)
        key = key_states.reshape(B, L, attn.num_key_value_heads, -1)
        value = value32.reshape(B, L, attn.num_key_value_heads, -1)
        q = _clip(fn, q, attn.q_l2_clip.scale, k.qk_thr, tau)
        key = _clip(fn, key, None, k.qk_thr)
        value = k.rms(value, attn.eps).astype(dtype)
        r = _clip(fn, r, attn.rel_l2_clip.scale, k.rel_thr, tau)
        position_bias = r @ k.rel_proj(r.dtype)

        q = q.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        kv = cache[0]
        key, value = kv.update_and_fetch(key, value)
        if L == 1 and B == 1:
            out = self._fused_one(q, key, value, position_bias, sinks, kv)
        else:
            out = attn._attention(q, key, value, position_bias, sinks, kv)
        return attn.o_proj(fn.out_gate(out.reshape(B, L, -1), gate))

    def _fused_one(self, query, key, value, position_bias, sinks, cache):
        """Attention._attention's one-row branch, the distances and what they
        hide shared by the token's visits."""
        attn, stock = self.self_attn, self._k.stock
        S, end = key.shape[2], cache.offset
        R = position_bias.shape[-1]
        window = attn.sliding_window
        bdt = position_bias.dtype

        def make():
            qpos = stock._positions(end - 1, 1)
            kpos = stock._key_positions(cache, S, 1, end).reshape(S)
            distance = qpos.reshape(1) - kpos
            visible = (distance >= 0) & (kpos >= 0)
            if window:
                visible = visible & (distance < window)
            return (mx.clip(distance, 0, R - 1), visible & (distance < R),
                    mx.where(visible, mx.zeros((), bdt), -mx.inf))

        key_ = (S, end, getattr(cache, "_idx", None), type(cache), window, R,
                bdt)
        at, near, far = _MASKS.get(key_, make)
        bias = mx.take(position_bias[0, 0], at, axis=-1)
        mask = mx.where(near, bias, far).astype(query.dtype)
        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=attn.scaling,
            mask=mask[None, :, None, :],
            sinks=sinks[0, 0].astype(query.dtype))
        return out.transpose(0, 2, 1, 3)


class _FusedSliding(_Fused):
    def _fused_attention(self, hidden_states, attention_inputs, cache,
                         states, messages, fn):
        attn, k = self.self_attn, self._k
        dtype = hidden_states.dtype
        x = hidden_states
        if attention_inputs is not None:
            x = attention_inputs[..., 0, :]
        if messages is None:
            q = kh = vh = k.in_norm(x).astype(dtype)
        else:
            q, kh, vh = (k.in_norm(s).astype(dtype)
                         for s in self._fused_mix((x, x, x), messages, fn))
        kx32 = attn.k_proj(kh).astype(F32)
        out, states[0] = _sconv(attn.k_sconv, kx32, states[0])
        key_states = fn.tail(out, kx32, k.c_short)
        vx32 = attn.v_proj(vh).astype(F32)
        out, states[1] = _sconv(attn.v_sconv, vx32, states[1])
        value32 = fn.tail32(out, vx32, k.c_short)
        return self._fused_attend(q, key_states, value32, cache, fn)


class _FusedFull(_Fused):
    def _dconv(self, conv, static, x, dynamic_input, state, fn):
        """DynamicShortConvolution up to its residual: (the weighted sum,
        the input in float32, the window)."""
        B, L, _ = x.shape
        K, H = conv.kernel_size, conv.num_heads
        dyn = conv.weight_proj(dynamic_input).reshape(B, L, H, 1, K)
        x32 = x.astype(F32)
        window = _window(conv, x32, state)
        windows = mx.stack([window[:, K - 1 - lag:K - 1 - lag + L]
                            for lag in range(K)], axis=-1)
        windows = windows.reshape(B, L, H, -1, K)
        out = fn.dyn_weighted(windows, static, dyn).sum(axis=-1)
        return out.reshape(B, L, -1), x32, window

    def _fused_attention(self, hidden_states, attention_inputs, cache,
                         states, messages, fn):
        attn, k = self.self_attn, self._k
        dtype = hidden_states.dtype
        if attention_inputs is None:
            q = kh = vh = hidden_states
        else:
            q, kh, vh = (attention_inputs[..., i, :] for i in range(3))
        if messages is not None:
            q, kh, vh = self._fused_mix((q, kh, vh), messages, fn)
        q = k.q_norm(q).astype(dtype)
        kh = k.k_norm(kh).astype(dtype)
        vh = k.v_norm(vh).astype(dtype)

        out, x32, states[0] = self._dconv(attn.k_sconv, k.k_static(),
                                          attn.k_proj(kh), kh, states[0], fn)
        key_states = fn.tail(out, x32, k.c_dyn)
        out, x32, states[1] = self._dconv(attn.v_sconv, k.v_static(),
                                          attn.v_proj(vh), vh, states[1], fn)
        value32 = fn.tail32(out, x32, k.c_dyn)

        kv, rows = cache[0], hidden_states.shape[1]
        offset = kv.offset

        def make():
            pos = k.stock._positions(offset, rows).astype(F32)
            tau = 1.0 + attn.log_scaling_alpha * mx.log(
                mx.maximum((pos + 1) / attn.log_scaling_n_floor, 1.0))
            return tau[..., None, None]

        tau = _TAUS.get((offset, rows, attn.log_scaling_alpha,
                         attn.log_scaling_n_floor), make)
        return self._fused_attend(q, key_states, value32, cache, fn, tau=tau)


class _FusedMessages:
    """MessageInput with each looped layer's four queries taken out of the
    table once (indexing it is a gather a visit)."""

    def __call__(self, bank, loop_layer_idx):
        if not ENABLED:
            return super().__call__(bank, loop_layer_idx)
        stock, queries = self._k
        return stock._depth_attention(*bank, queries(loop_layer_idx))


_CLASSES: dict = {}


def _fused_class(cls, stock):
    c = _CLASSES.get(cls)
    if c is None:
        mixin = (_FusedSliding if issubclass(cls, stock.SlidingDecoderLayer)
                 else _FusedFull)
        # mlx-lm's module stays the class's: a host looks the layer's
        # helpers up there (the loop adapter's _conv_state)
        c = _CLASSES[cls] = type("Fused" + cls.__name__, (mixin, cls),
                                 {"__module__": cls.__module__})
    return c


def _fuse_messages(lm, stock):
    mi = getattr(lm, "message_input", None)
    if mi is None or isinstance(mi, _FusedMessages):
        return None
    queries = _Made(lambda: mi.query, lambda q, i: mx.contiguous(q[i]))
    for i in range(mi.query.shape[0]):
        queries(i)
    cls = type(mi)
    mi.__class__ = type("Fused" + cls.__name__, (_FusedMessages, cls),
                        {"__module__": cls.__module__})
    object.__setattr__(mi, "_k", (stock, queries))
    return mi, cls


def _static(conv):
    """A DynamicShortConvolution's static weights as its call shapes them."""
    return _Made(lambda: conv.static_weight, lambda w, _key: mx.contiguous(
        w.astype(F32).T.reshape(conv.num_heads, -1, conv.kernel_size)))


def _consts(layer, stock, text, dtype):
    """What a fused visit reads besides the layer's parameters (made again
    from a parameter that is replaced), and the stream's dtype its probe
    checks it in."""
    attn = layer.self_attn
    K = text.sconv_kernel_size
    k = SimpleNamespace(
        stock=stock, rms=stock.unweighted_rms_norm, keep=K - 1, dtype=dtype,
        conv_state=stock._conv_state,
        c_short=_scalar(math.sqrt(K + 1)), c_dyn=_scalar(math.sqrt(2 * K + 1)),
        qk_thr=_scalar(attn.q_l2_clip.threshold),
        rel_thr=_scalar(attn.rel_l2_clip.threshold),
        rel_proj=_Made(lambda: attn.rel_logits_proj.proj,
                       lambda p, dt: p.astype(dt)))
    k.post_norm = _Norm(layer.post_attention_layernorm)
    if isinstance(layer, stock.FullDecoderLayer):
        for name in ("q", "k", "v"):
            setattr(k, name + "_norm",
                    _Norm(getattr(layer, name + "_input_layernorm")))
        k.k_static, k.v_static = _static(attn.k_sconv), _static(attn.v_sconv)
        for made in (k.k_static, k.v_static):
            made()
    else:
        k.in_norm = _Norm(layer.input_layernorm)
    mx.eval(k.c_short, k.c_dyn, k.qk_thr, k.rel_thr)
    return k


def _bits(a: mx.array) -> mx.array:
    return a.view(mx.uint8) if a.size else a


def _same(a, b) -> bool:
    if isinstance(a, mx.array) or isinstance(b, mx.array):
        return (isinstance(a, mx.array) and isinstance(b, mx.array)
                and a.shape == b.shape and a.dtype == b.dtype
                and bool(mx.array_equal(_bits(a), _bits(b)).item()))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    return a == b


class _stand_in:
    """While inside, the block's experts are a cheap function of their
    inputs: the probe compares the visit around them (the call is the same
    in both), and a streamed model's pools are not read or reordered."""

    def __init__(self, mlp):
        self.mlp = mlp

    @staticmethod
    def experts(x, indices, *_a, **_kw):
        w = (indices % 5 + 1).astype(x.dtype)[..., None]
        return x[..., None, :] * w

    def __enter__(self):
        object.__setattr__(self.mlp, "experts", self.experts)

    def __exit__(self, *exc):
        object.__delattr__(self.mlp, "experts")


def _kind(layer) -> tuple:
    """What a layer's visit computes besides its inputs: its class, its place
    in its block, whether it takes the loop's messages, and the dtypes of its
    parameters (a projection's output takes its own: mlx-lm casts a
    convolution's output to its input's dtype, the fused visit to the
    stream's), its experts aside (the same call in both visits, stood in for
    by the probe). The probe tries a layer of each kind."""
    return (type(layer), layer.attention_input is None,
            layer.is_block_boundary, hasattr(layer, "q_message_gate"),
            tuple((name, a.dtype) for name, a in tree_flatten(
                layer.parameters()) if not name.startswith("mlp.experts.")))


def _probe(model, layers, stock, dtype, rows: int = 1):
    """A visit of each kind of layer (its first in the stack, with and
    without the loop's messages) through mlx-lm's code and through the fused
    one, from the same inputs (the stream's in `dtype`) and caches. Returns
    the kinds where some output or cache entry is not equal bit for bit, and
    whether the message input's are."""
    global ENABLED
    text = model.args.text
    lm = model.model.language_model
    dims = text.hidden_size
    visits = list(getattr(lm, "layer_visits", None) or lm.layers)
    cache = model.make_cache()
    key = mx.random.key(0)
    per = max(text.n_layers_per_attn_res_block, 1)
    seen, bad = set(), set()
    for layer in layers:
        kind = _kind(layer)
        if kind in seen:
            continue
        seen.add(kind)
        # (a module is a dict: its == would compare parameters)
        entry = cache[next(i for i, v in enumerate(visits) if v is layer)]
        idx = next(i for i, v in enumerate(lm.layers) if v is layer)
        n_blocks = -(-idx // per) if idx else 0
        for messages in ((False, True) if kind[3] else (False,)):
            results = []
            for fused in (False, True):
                ENABLED = fused
                c = [type(entry[0])(**({"max_size": entry[0].max_size}
                                       if hasattr(entry[0], "max_size")
                                       else {})),
                     type(entry[1])(len(entry[1].cache))]
                ks = mx.random.split(key, 4)
                outs = []
                for step in range(3):      # the third visit reads a state
                    n = 1 if step < 2 else rows
                    sk = mx.random.split(ks[step % 4], 4)
                    h = mx.random.normal((1, n, dims), key=sk[0]).astype(dtype)
                    blocks = (mx.random.normal((n_blocks, 1, n, dims),
                                               key=sk[1])
                              if n_blocks else None)
                    msg = (mx.random.normal((1, n, 4, dims), key=sk[2])
                           if messages else None)
                    kw = {} if msg is None else {"messages": msg}
                    with _stand_in(layer.mlp):
                        outs.append(layer(h, blocks, cache=c, **kw))
                state = [outs, list(c[1].cache), c[0].state]
                mx.eval(state)
                results.append(state)
            if not _same(results[0], results[1]):
                bad.add(kind)
    messages_same = True
    mi = getattr(lm, "message_input", None)
    if mi is not None:
        ks = mx.random.split(key, 2)
        bank = (mx.random.normal((2, 1, rows, dims), key=ks[0]),
                mx.random.normal((1, rows, dims), key=ks[1]))
        results = []
        for fused in (False, True):
            ENABLED = fused
            results.append([mi(bank, i) for i in range(mi.query.shape[0])])
        mx.eval(results)
        messages_same = _same(results[0], results[1])
    return bad, messages_same


def _unfuse(lm) -> None:
    """The layers and the message input back in mlx-lm's classes."""
    for m in [*lm.layers, getattr(lm, "message_input", None)]:
        if isinstance(m, (_Fused, _FusedMessages)):
            m.__class__ = type(m).__bases__[1]


def fuse(model) -> int:
    """Give mlx-lm's inkling2 layers the fused visit. Returns how many
    layers took it: 0 when switched off (STREAMLX_FUSED=0), for a model that
    is not mlx-lm's inkling2, or when the check against mlx-lm's own visit
    finds a difference in every kind of layer (the model is then as it was);
    the layers of a kind that differs keep mlx-lm's visit. A model fused
    before is checked anew (after loading weights of other dtypes)."""
    global ENABLED
    if not ENABLED:
        return 0
    before, why, bad, same, lm = [], None, set(), True, None
    try:
        stock = _stock()
        if not all(hasattr(stock, name) for name in _NEEDS):
            return 0                   # an mlx-lm before the layer mirrored
        lm = model.model.language_model
        layers = list(lm.layers)
        text = model.args.text
        if not all(isinstance(layer, stock.DecoderLayer) for layer in layers):
            return 0
        _unfuse(lm)
        dtype = lm.embed(mx.array([[0]])).dtype
        for layer in layers:
            k = _consts(layer, stock, text, dtype)
            before.append((layer, type(layer)))
            layer.__class__ = _fused_class(type(layer), stock)
            object.__setattr__(layer, "_k", k)
        swapped = _fuse_messages(lm, stock)
        if not _groups_exact(dtype):
            why = "a compiled group rounds differently on this device"
        else:
            for rows in (1, 3):
                b, s = _probe(model, layers, stock, dtype, rows)
                bad, same = bad | b, same and s
            if len(bad) == len({_kind(layer) for layer in layers}):
                why = "its values differ"
    except Exception as e:  # noqa: BLE001 — a layer this does not know
        why = f"{type(e).__name__}: {e}"
    finally:
        ENABLED = True
    if why is not None:
        for layer, cls in before:
            layer.__class__ = cls
        if lm is not None:
            _unfuse(lm)
        print("[streamlx] inkling2: mlx-lm's own layer visit is used (the "
              f"fused one does not reproduce it here: {why})",
              file=sys.stderr)
        return 0
    kept = [i for i, (layer, cls) in enumerate(before) if _kind(layer) in bad]
    for i in kept:
        layer, cls = before[i]
        layer.__class__ = cls
    if swapped is not None and not same:
        mi, cls = swapped
        mi.__class__ = cls
    parts = ([f"layers {kept}"] if kept else []) + (
        [] if same else ["the message input"])
    if parts:
        print(f"[streamlx] inkling2: mlx-lm's own visit is kept for "
              f"{' and '.join(parts)} (the fused one does not reproduce it "
              "there)", file=sys.stderr)
    return len(layers) - len(kept)
