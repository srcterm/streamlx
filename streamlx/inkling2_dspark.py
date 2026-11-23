"""Inkling 2's draft head (`mtp.safetensors` next to the checkpoint) for the
pipeline: a block drafter of the DSpark family that neither mlx-lm nor
mlx-vlm has. Standalone: nothing else imports it but
pipeline_adapters.load_drafter.

  context   per committed position, the target's residual entering layers
            5 / 11 / ... / 41 and its final normalized output (the adapter's
            taps). Each of the 5 draft layers mixes them with its own
            softmax weights (fuse_weight), norms the mix (dflare_norm) and
            projects it to its context keys and values (k_proj_ctx /
            v_proj_ctx; keys normed per head and rotated to the position).
  propose   a block of 8 rows at the positions from the last committed token
            on: its normalized embedding from the target, then the mask
            embedding 7 times; 5 pre-norm layers (attention over the
            context and the block's own rows, not causal; a dense SwiGLU
            MLP), a norm. Row i proposes the token i places after the
            anchor: the target's head on the row, corrected by a small
            recurrent head that reads the previous proposal (markov_w1), the
            row and the mix of the last committed position's taps.
Rejected proposals never enter the context, so nothing is rolled back.

All dense, bf16 in the checkpoint (3.2 GiB; Inkling 2 Mini's, 3 layers:
0.6); `bits` quantizes its projections at load (drafts are verified:
precision only costs acceptance).

The recurrent head's correction goes through the target's output matrix once
per proposal, one after another (each reads the previous proposal): for a
bf16 matrix (a source checkpoint's: Mini's is 0.9 GB) that would cost as
much as a token of the trunk. So the row's logits are computed once for all
proposals (with a 4-bit copy of a bf16 matrix), each proposal is chosen
among that row's best CANDIDATES, and only their correction (and their
logits, from the target's own matrix) is computed; the rest of the
vocabulary only enters a proposal's probability.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx import nn

BITS = 4     # the head's projections in RAM: 1.0 GiB (8: 1.7, None = bf16: 3.2)
_PART = 512  # context positions projected at a time
CANDIDATES = 64  # tokens a proposal is chosen among


def _norm(x, weight, eps):
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32),
                            eps)


class _Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, d = cfg["hidden_size"], cfg["head_dim"]
        self.heads, self.kv_heads = (cfg["num_attention_heads"],
                                     cfg["num_key_value_heads"])
        self.dim, self.eps, self.theta = d, cfg["rms_norm_eps"], cfg["rope_theta"]
        self.q_proj = nn.Linear(h, self.heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, self.kv_heads * d, bias=False)
        self.k_proj_ctx = nn.Linear(h, self.kv_heads * d, bias=False)
        self.v_proj_ctx = nn.Linear(h, self.kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.heads * d, h, bias=False)
        self.q_norm = nn.RMSNorm(d, eps=self.eps)
        self.k_norm = nn.RMSNorm(d, eps=self.eps)
        self.k_norm_ctx = nn.RMSNorm(d, eps=self.eps)

    def _heads(self, x, n):
        return x.reshape(x.shape[0], x.shape[1], n, self.dim).transpose(0, 2, 1, 3)

    def _rot(self, x, norm, start: int):
        """[1, heads, rows, dim]: normed per head, rotated to its position."""
        x = _norm(x, norm.weight, self.eps).astype(mx.bfloat16)
        return mx.fast.rope(x, self.dim, traditional=False, base=self.theta,
                            scale=1.0, offset=start)

    def context(self, mix, start: int):
        """The keys and values of context rows at positions start.."""
        k = self._heads(self.k_proj_ctx(mix), self.kv_heads)
        v = self._heads(self.v_proj_ctx(mix), self.kv_heads)
        return self._rot(k, self.k_norm_ctx, start), v

    def __call__(self, x, ctx, start: int, mask):
        q = self._rot(self._heads(self.q_proj(x), self.heads), self.q_norm, start)
        k = self._rot(self._heads(self.k_proj(x), self.kv_heads), self.k_norm, start)
        v = self._heads(self.v_proj(x), self.kv_heads)
        if ctx is not None:
            k = mx.concatenate([ctx[0], k], axis=2)
            v = mx.concatenate([ctx[1], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.dim ** -0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(x.shape[0],
                                                             x.shape[1], -1))


class _MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, m = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, m, bias=False)
        self.up_proj = nn.Linear(h, m, bias=False)
        self.down_proj = nn.Linear(m, h, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.input_layernorm = nn.RMSNorm(h, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=eps)
        self.dflare_norm = nn.RMSNorm(h, eps=eps)
        self.self_attn = _Attention(cfg)
        self.mlp = _MLP(cfg)


class _Markov(nn.Module):
    def __init__(self, cfg, vocab: int):
        super().__init__()
        h, r = cfg["hidden_size"], cfg["markov_rank"]
        self.markov_w1 = nn.Embedding(vocab, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * h, 3 * r, bias=True)
        self.hidden_out = nn.Linear(r, h, bias=True)
        self.dflare_norm = nn.RMSNorm(h, eps=cfg["rms_norm_eps"])


class _Head(nn.Module):
    def __init__(self, cfg, vocab: int):
        super().__init__()
        n, h = cfg["num_hidden_layers"], cfg["hidden_size"]
        self.fuse_weight = mx.zeros((n + 1, len(cfg["target_layer_ids"])))
        self.mask_embed = mx.zeros((h,))
        self.layers = [_Layer(cfg) for _ in range(n)]
        self.norm = nn.RMSNorm(h, eps=cfg["rms_norm_eps"])
        self.markov_head = _Markov(cfg, vocab)


class Inkling2Draft:
    """The pipeline's drafter protocol (reset / prime_pairs / draft) over
    the head. `tap_layers`: what the adapter's rows hold, in order."""

    def __init__(self, model, model_dir: str, bits: int | None = BITS):
        root = Path(model_dir)
        cfg = json.loads((root / "config.json").read_text())["mtp_config"]
        if cfg.get("markov_head_type") != "rnn_hidden":
            raise ValueError(f"{root}: draft head with a "
                             f"{cfg.get('markov_head_type')!r} Markov head")
        self.cfg = cfg
        self.tap_layers = tuple(cfg["target_layer_ids"])
        self.block = int(cfg["block_size"])
        self.depth = self.block - 1
        self.eps, self.h = cfg["rms_norm_eps"], cfg["hidden_size"]
        self.local = frozenset(cfg["local_layer_ids"])
        # a windowed layer's rows see the context this far back
        self.window = int(cfg["sliding_window_size"]) + self.block - 1
        text = model.args.text
        self._embed = model.model.language_model.embed
        self._lm_head = lm = model.lm_head
        self._cap, self._vocab = (text.final_logit_softcapping,
                                  text.unpadded_vocab_size)
        # the candidates' rows of the target's output matrix, and the matrix
        # that picks them: the target's own if quantized, else a 4-bit copy
        self._rows_of = self._quantized_rows if isinstance(
            lm, nn.QuantizedLinear) else (lambda ids: lm.weight[ids])
        self._pick = lm if isinstance(lm, nn.QuantizedLinear) else (
            nn.QuantizedLinear.from_linear(lm, group_size=64, bits=4))
        mx.eval(self._pick.parameters())
        weights = mx.load(str(root / "mtp.safetensors"))
        head = _Head(cfg, weights["markov_head.markov_w1.weight"].shape[0])
        head.load_weights(list(weights.items()), strict=True)
        if bits:
            nn.quantize(head, group_size=64, bits=bits,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear))
        mx.eval(head.parameters())
        self.head = head
        self.alpha = mx.softmax(head.fuse_weight.astype(mx.float32), axis=-1)
        mx.eval(self.alpha)
        self.reset()

    # -- context -----------------------------------------------------------
    def reset(self) -> None:
        self.n = 0                     # committed positions in the context
        self.ctx = [None] * len(self.head.layers)
        self.last = None               # the last position's taps

    def _mix(self, rows, stage: int, norm):
        """[1, n, taps x h] -> stage `stage`'s normed mix of the taps."""
        taps = rows.reshape(rows.shape[0], rows.shape[1], -1, self.h)
        a = self.alpha[stage]
        mix = (taps.astype(mx.float32) * a[:, None]).sum(axis=2)
        return _norm(mix, norm.weight, self.eps).astype(mx.bfloat16)

    def prime_pairs(self, hidden, next_ids=None) -> None:
        """Context rows of the next committed positions (a prefill chunk's
        go in by parts: a part's mixes are float32 copies of its taps)."""
        if hidden is None or not hidden.shape[1]:
            return
        for a in range(0, hidden.shape[1], _PART):
            part = hidden[:, a:a + _PART]
            new = []
            for s, layer in enumerate(self.head.layers):
                k, v = layer.self_attn.context(
                    self._mix(part, s, layer.dflare_norm), self.n)
                if self.ctx[s] is not None:
                    k = mx.concatenate([self.ctx[s][0], k], axis=2)
                    v = mx.concatenate([self.ctx[s][1], v], axis=2)
                if s in self.local:
                    k, v = k[:, :, -self.window:], v[:, :, -self.window:]
                new.append((k, v))
            self.ctx = new
            self.n += part.shape[1]
            mx.eval(self.ctx)
        self.last = hidden[:, -1:]
        mx.eval(self.last)

    # -- proposing ---------------------------------------------------------
    def _block(self, anchor: int):
        """The block's rows through the draft layers: [1, block, h]."""
        head, B, p = self.head, self.block, self.n
        x = mx.concatenate(
            [self._embed(mx.array([[anchor]])).astype(mx.bfloat16),
             mx.broadcast_to(head.mask_embed.astype(mx.bfloat16),
                             (1, B - 1, self.h))], axis=1)
        res = x.astype(mx.float32)
        x = _norm(res, head.layers[0].input_layernorm.weight, self.eps)
        for s, layer in enumerate(head.layers):
            ctx, mask = self.ctx[s], None
            if s in self.local and ctx is not None \
                    and ctx[0].shape[2] + B > self.window:
                # row i (position p + i) sees keys less than a window back
                n = ctx[0].shape[2]
                kpos = mx.concatenate([mx.arange(p - n, p), mx.arange(p, p + B)])
                mask = (mx.arange(p, p + B)[:, None] - kpos[None]) < self.window
            a = layer.self_attn(x.astype(mx.bfloat16), ctx, p, mask)
            res = res.astype(mx.bfloat16).astype(mx.float32) + a.astype(mx.float32)
            x = _norm(res, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp(x.astype(mx.bfloat16))
            res = res.astype(mx.bfloat16).astype(mx.float32) + m.astype(mx.float32)
            nxt = (head.layers[s + 1].input_layernorm if s + 1 < len(head.layers)
                   else head.norm)
            x = _norm(res, nxt.weight, self.eps)
        return x.astype(mx.bfloat16)

    def _quantized_rows(self, ids):
        """Rows `ids` of the target's quantized output matrix, dequantized."""
        lm = self._lm_head
        return mx.dequantize(lm.weight[ids], lm.scales[ids],
                             None if lm.get("biases") is None else lm.biases[ids],
                             group_size=lm.group_size, bits=lm.bits,
                             mode=getattr(lm, "mode", "affine"))

    def _softcap(self, x):
        return mx.tanh(x / self._cap) * self._cap

    def draft(self, pairs, depth: int):
        """Proposals for the tokens after the last committed one, and the
        probability each got."""
        if pairs:
            rows = [h for h, _ in pairs if h is not None]
            if rows:
                self.prime_pairs(mx.concatenate(rows, axis=1))
        if not pairs or depth <= 0 or self.last is None:
            return [], []
        anchor = int(pairs[-1][1])
        mk = self.head.markov_head
        rows = self._block(anchor)
        f = self._mix(self.last, len(self.head.layers), mk.dflare_norm)[:, 0]
        state = mx.zeros((1, self.cfg["markov_rank"]), dtype=mx.bfloat16)
        prev = mx.array([anchor])
        toks, probs = [], []
        depth = min(depth, self.depth)
        r = rows[0, 1:depth + 1]
        # each row's candidates by its own logits, their rows of the output
        # matrix, and the rest of the vocabulary's mass (for the probability)
        base = self._softcap(self._pick(r).astype(mx.float32))[:, :self._vocab]
        cand = mx.argpartition(-base, CANDIDATES - 1, axis=-1)[:, :CANDIDATES]
        rest = mx.logsumexp(mx.put_along_axis(
            base, cand, mx.array(-mx.inf, dtype=base.dtype), axis=-1), axis=-1)
        w = self._rows_of(cand)                       # [depth, C, h]
        for i in range(1, depth + 1):
            row = rows[:, i]
            z = mk.rnn_proj(mx.concatenate(
                [state, mk.markov_w1(prev).astype(mx.bfloat16), row, f], axis=-1))
            gate, c, out = mx.split(z.astype(mx.float32), 3, axis=-1)
            gate = mx.sigmoid(gate)
            state = (gate * state.astype(mx.float32)
                     + (1 - gate) * mx.tanh(c)).astype(mx.bfloat16)
            fix = mk.hidden_out(mx.tanh(out).astype(mx.bfloat16))
            # the row's logit plus the correction's, for the candidates only
            wi = w[i - 1]
            lg = ((wi @ row[0].astype(wi.dtype)).astype(mx.float32)
                  + (wi @ fix[0].astype(wi.dtype)).astype(mx.float32))
            lg = self._softcap(lg)
            best = mx.argmax(lg)
            prev = cand[i - 1, best].reshape(1)
            top = lg[best]
            m = mx.maximum(top, rest[i - 1])
            toks.append(prev)
            probs.append(mx.exp(top - m) / (mx.exp(lg - m).sum()
                                            + mx.exp(rest[i - 1] - m)))
        mx.eval(toks, probs)
        return ([int(t.item()) for t in toks], [float(p.item()) for p in probs])
