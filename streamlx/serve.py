"""OpenAI-compatible HTTP server for SSD-streamed MoE models.

Wraps stock `mlx_lm.server` (all its flags pass through) but loads the model
with streamlx: trunk resident, experts streamed from SSD on demand.

  streamlx serve <local-model-dir> [--port 8080] \
      [--memory-gib N] [--budget-gib N] [--mlx-cache-gib N] \
      [--kv-cache-dir DIR [--kv-cache-gib N]] \
      [--draft-head DIR|none [--draft-k K]] \
      [--trust-remote-code] [--tokenizer-config key=json ...]

With no options: MLX's memory is kept under 95% of the GPU working set the
system recommends (--memory-gib), the expert pools take what that leaves
(--budget-gib caps them), prompt caches go to ~/.cache/streamlx/kv, and a
checkpoint that carries a draft head (Qwen3.8, DeepSeek-V4.1, Inkling 2)
decodes pipelined with it; one without decodes plainly.

The model must be a LOCAL directory (the byte-range reader needs the shards on
disk). Requests are handled sequentially: concurrent batching multiplies the
per-step expert working set and collapses cache hit rates.

MLX's default buffer-cache limit is ~95% of RAM, so freed buffers are
hoarded forever; with most of that wired for Metal, a long agent session
runs the machine out of unpageable memory. It gets a hard cap here:
--mlx-cache-gib (default 2).

--memory-gib (default: 95% of the GPU working set the system recommends; 0:
no limit) keeps MLX's memory (active + buffer cache) under a limit by
resizing the expert pools, never above --budget-gib: before each request,
for the cache it starts from plus what its prompt and max_tokens will add
(at the bytes a token has added so far), with the rest of the peak taken from
earlier requests. A long conversation so gives slots back as its cache
grows, and a short one takes them again. A prompt chunk's prefill needs
working memory of its own, more the longer the chunk (DeepSeek-V4.1: 4 MiB a
token), while the pools sit idle: its experts pass through the sweep's
buffer. So the pools are sized for each chunk as it comes, from what earlier
chunks held a token, and grow back for decoding, which reads again what the
smaller pools dropped. Resizing copies the kept slots (~1 s for 16 GiB);
outputs don't change.

Long prefills of models whose attention head size MLX does not fuse by
default (256 in Qwen3.5/3.6) would build heads x queries x keys of scores per
layer, GiBs at coding-agent context lengths: streamlx.attention sends those
calls to MLX's fused kernel where it has one (MLX >= 0.32.2), else runs them
in blocks of query rows, bitwise the unfused output (always on).

Prefill stops where a chat prompt's history ends, before the template's
answer prompt, and the prompt cache stores that point: the next turn repeats
everything before it, so it restores it whole and prefills only what is new
(thinking templates drop a past answer's "<think>", and hybrid caches cannot
trim back to a divergence). Chunks are evened out, at most 1.25x
--prefill-step-size, instead of ending in a small one: each chunk's expert
sweep reads ~every expert.

Chat formats are streamlx.chat_formats': the prompt format and tool calls of
a checkpoint that ships no template (DeepSeek-V4.1), and the reply framing of
a model that writes its own (Inkling 2), which the server sorts into
reasoning, content and tool calls.

Prompt caches live on SSD (streamlx.kvcache; --kv-cache-dir, default
~/.cache/streamlx/kv): stored once as segments shared by every request with
the same prefix, so RAM holds only the active request's cache.
--kv-cache-gib bounds the disk use (default half of what is free there, at
most 32; oldest segments go first). A drive other than the model's avoids
sharing its bandwidth with expert reads.

A draft head decodes greedy requests through streamlx.pipeline: K tokens in
flight, so each token's expert reads overlap the others' compute, with plain
decoding's output. A checkpoint that carries one (Qwen3.8, DeepSeek-V4.1,
Inkling 2) uses it by default; --draft-head takes an mlx-vlm drafter
directory for other models, or none for plain decoding. The self-gate's
verdict carries across requests.
Requests that sample, process logits or ask for logprobs decode plainly. On a
prompt-cache hit the head's context starts at the tokens this request
prefills: drafts are verified, so that can only cost acceptance, not change
the text.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time

import mlx.core as mx
import mlx_lm.server as srv
from mlx_lm.generate import (GenerationResponse, generation_stream,
                             maybe_quantize_kv_cache, wired_limit)

from streamlx import attention, chat_formats
from streamlx.integrate import (aggregate_stats, budget_within,
                                load_streaming_model, memory_limit,
                                pool_bytes, prefill_cache, resize_pools)
from streamlx.kvcache import PagedPromptCache

_CFG = {"budget_bytes": None, "trust": False, "tokenizer_config": {},
        "draft_head": "auto", "draft_k": None, "memory_bytes": 0,
        "mlx_cache_bytes": 2 * 2**30}
_STATE = {"pools": None, "reader": None,
          "mem": {"F": None, "slope": None, "pre": None}}


_orig_sampler = srv._make_sampler


def _recording_sampler(args, tokenizer):
    # What the request asked for: stream_generate only gets the closures.
    _STATE["args"] = args
    return _orig_sampler(args, tokenizer)


srv._make_sampler = _recording_sampler


def _chunks(n: int, step: int) -> list[int]:
    """Prefill chunk sizes for n tokens: as many as step-size chunks take,
    one fewer when the last would be under a quarter step, evened out. Each
    chunk's expert sweep reads ~every expert, so a small last chunk costs as
    many bytes as a full one."""
    k = -(-n // step)
    if k > 1 and n - (k - 1) * step < step // 4:
        k -= 1
    q, r = divmod(n, k)
    return [q + 1] * r + [q] * (k - r)


_MARGIN = 2**28   # --memory-gib keeps this much of its limit free,
_HYST = 2**28     # shrinks the pools when they pass it by this much
_GROW = 2**30     # and grows them back when this much is free
_PRE_MIN = 256    # a prompt chunk from this many tokens gets room of its own:
_PRE_PRIOR = 4 * 2**20   # bytes a chunk token until a chunk has been measured
#                          (DeepSeek-V4.1's 4 MiB, the most of the models run)


def _buffer_reserve() -> int:
    """The buffer cache a peak can hold."""
    return _CFG["mlx_cache_bytes"]


def _cache_bytes(cache) -> int:
    try:
        return sum(c.nbytes for c in cache or ())
    except Exception:  # noqa: BLE001 — a cache type without nbytes
        return 0


def _fit_pools(target: float, why: str) -> None:
    """Resize every expert pool (trunk and draft head) to `target` bytes in
    all, at most --budget-gib: down by _HYST or more, up by _GROW or more
    (a resize copies every kept slot, ~1 s for 16 GiB)."""
    pools = _STATE["pools"] or {}
    have = pool_bytes(pools)
    target = min(target, _CFG["budget_bytes"])
    if not pools or not (have - target >= _HYST or target - have >= _GROW):
        return
    t0 = time.perf_counter()
    got = resize_pools(pools, max(target, 0))
    if got != have:   # (pools that hold every expert have nothing to grow)
        print(f"[streamlx] memory: pools {have / 2**30:.1f} -> "
              f"{got / 2**30:.1f} GiB ({why}; "
              f"{time.perf_counter() - t0:.1f}s)", file=sys.stderr)


def _govern_request(kwargs: dict) -> None:
    """--memory-gib, before a request: pools sized so its peak should stay
    under the limit. A peak is the pools, the cache (the one the request
    starts from plus the bytes a token has added so far times the tokens to
    come), what decoding holds besides (trunk, draft head, working set: the
    most earlier requests measured) and, while a prompt chunk is prefilled,
    that chunk's working memory, which grows with its length. The pools sit
    idle while a chunk's experts pass through the sweep's buffer, so they are
    sized for the first chunk here, for each later one by _govern_chunk, and
    for decoding once the prompt is in."""
    lim, m = _CFG["memory_bytes"], _STATE["mem"]
    if not lim:
        return
    cache, prompt = kwargs.get("prompt_cache"), kwargs.get("prompt")
    n_prompt = len(prompt) if isinstance(prompt, list) else 0
    cb = _cache_bytes(cache)
    m["base"] = (mx.get_active_memory() - cb
                 - pool_bytes(_STATE["pools"] or {}))
    # decoding's share; on the first request what is resident now plus 2 GiB
    m["rest"] = m["F"] if m["F"] is not None else m["base"] + 2 * 2**30
    m["cb0"], m["peak_rest"] = cb, 0
    m["to_come"] = max(0, kwargs.get("max_tokens", 256))
    chunk = (_chunks(n_prompt, kwargs.get("prefill_step_size", 2048))[0]
             if n_prompt else 0)
    _fit_pools(_pools_for(cb, n_prompt + m["to_come"], chunk),
               f"a prompt chunk of {chunk} tokens" if chunk >= _PRE_MIN
               else "decoding")
    _reset_peak()


def _pools_for(cb: int, tokens: int, chunk: int = 0) -> float:
    """The pool bytes that keep a peak under --memory-gib: with a cache of
    cb bytes, `tokens` more to come and, for chunk >= _PRE_MIN, a prompt chunk
    of that many tokens in prefill."""
    m = _STATE["mem"]
    pre = 0.0
    if chunk >= _PRE_MIN:
        pre = (_PRE_PRIOR if m["pre"] is None else m["pre"]) * chunk
    return (_CFG["memory_bytes"] - _MARGIN - _buffer_reserve() - m["rest"]
            - cb - (m["slope"] or 0) * tokens - pre)


def _reset_peak() -> None:
    """Start a new peak measurement, keeping the request's highest so far for
    its summary line."""
    m = _STATE["mem"]
    m["peak_all"] = max(m.get("peak_all", 0), mx.get_peak_memory())
    mx.reset_peak_memory()


def _govern_mark(cache) -> None:
    """Record decoding's peak above its pools and cache so far."""
    m = _STATE["mem"]
    rest = (mx.get_peak_memory() - pool_bytes(_STATE["pools"] or {})
            - _cache_bytes(cache))
    m["peak_rest"] = max(m.get("peak_rest", 0), rest)


def _govern_chunk(cache, done: int, n: int, nxt: int, left: int) -> None:
    """--memory-gib after a prefill chunk of n tokens. Its working memory a
    token is what its peak held above the pools, the cache and decoding's
    share; the most over chunks is kept. Then the pools are sized for the
    next chunk (nxt tokens), or after the last one for decoding: they grow
    back, and decoding reads again what the smaller pools dropped."""
    lim, m = _CFG["memory_bytes"], _STATE["mem"]
    if not lim:
        return
    cb = _cache_bytes(cache)
    _measure_slope(cb, done)
    if n >= _PRE_MIN:
        held = (mx.get_peak_memory() - cb
                - pool_bytes(_STATE["pools"] or {})
                - (m["base"] if m["F"] is None else m["F"]))
        m["pre"] = max(m["pre"] or 0.0, held / n)
    _fit_pools(_pools_for(cb, left + m["to_come"], nxt),
               f"a prompt chunk of {nxt} tokens" if nxt >= _PRE_MIN
               else "decoding")
    _reset_peak()
    if not nxt:
        m["peak_rest"] = 0


def _measure_slope(cb: int, n_tokens: int) -> None:
    """The cache bytes a token adds, from a request that continued a cache
    (one that starts empty also allocates fixed state, e.g. recurrent
    layers') over at least 256 tokens."""
    m = _STATE["mem"]
    if m.get("cb0", 0) > 0 and n_tokens >= 256 and cb > m["cb0"]:
        m["slope"] = max(m["slope"] or 0, (cb - m["cb0"]) / n_tokens)


def _govern_done(cache, n_tokens: int) -> None:
    """After a request: the bytes a token adds to the cache and decoding's
    peak above pools and cache (kept as the maximum over requests)."""
    m = _STATE["mem"]
    if not _CFG["memory_bytes"]:
        return
    _measure_slope(_cache_bytes(cache), n_tokens)
    _govern_mark(cache)
    m["F"] = max(m["F"] or 0, m["peak_rest"])


def _prefill_history(kwargs: dict, adapter=None) -> int:
    """Prefill the request's uncached prompt up to where its history ends,
    store the cache there and leave the rest to stream_generate (updates
    kwargs); returns the tokens prefilled here. With a pipeline adapter the
    draft head gets each chunk's context.

    The history ends before the tokens the chat template appends to ask for
    an answer. The next turn's prompt repeats this one up to there but not
    always beyond (a thinking template renders a past answer without the
    "<think>" it was prompted with), and hybrid caches (recurrent state,
    rotating KV) cannot trim back, so a cache stored any later is of no use
    to it. Stored there (a segment of the SSD prompt cache), the next turn
    restores it and prefills only its new tokens. Prompts without
    that suffix stop one token short of their end.
    """
    prompt, cache, model = (kwargs.get(k)
                            for k in ("prompt", "prompt_cache", "model"))
    if cache is None or model is None or not isinstance(prompt, list):
        return 0
    sfx = _STATE.get("suffix") or []
    cut = len(prompt) - (len(sfx) if sfx and prompt[-len(sfx):] == sfx
                         else 1)
    if cut < 1:
        return 0
    paged = _STATE.get("paged")
    base = paged.last[0] if paged is not None else 0
    cb = kwargs.get("prompt_progress_callback") or (lambda *_: None)
    stream = kwargs.get("stream", generation_stream)
    tokens, done = mx.array(prompt[:cut]), 0
    cb(0, len(prompt))
    with wired_limit(model, [stream]), mx.stream(stream):
        sizes = _chunks(cut, kwargs.get("prefill_step_size", 2048))
        for i, n in enumerate(sizes):
            chunk = tokens[None, done:done + n]
            with prefill_cache(cache):
                if adapter is None:
                    model(chunk, cache=cache)
                    rows = None
                else:
                    _, rows = adapter.forward_taps(chunk, cache)
            maybe_quantize_kv_cache(
                cache, kwargs.get("quantized_kv_start", 0),
                kwargs.get("kv_group_size", 64), kwargs.get("kv_bits"))
            mx.eval([c.state for c in cache], [] if rows is None else rows)
            if rows is not None:
                nxt = prompt[done + 1:done + n + 1]
                adapter.prime_pairs(rows, mx.array([nxt], dtype=mx.int32))
            done += n
            if paged is not None:
                paged.save_progress(base + done, cache)
            cb(done, len(prompt))
            mx.clear_cache()
            _govern_chunk(cache, done, n, sizes[i + 1] if i + 1 < len(sizes)
                          else 0, len(prompt) - done)
    kwargs["prompt"] = prompt[cut:]
    kwargs["prompt_progress_callback"] = lambda p, t: cb(cut + p, cut + t)
    return cut


def _pipelined_decoder(kwargs: dict):
    """The decoder if it can serve this request: greedy, nothing processing
    the logits and no logprobs (its streams keep only the argmax)."""
    args = _STATE.pop("args", None)
    if (_STATE.get("decoder") is None or args is None
            or args.sampling.temperature != 0 or args.logprobs
            or args.top_logprobs > 0 or kwargs.get("logits_processors")
            or kwargs.get("input_embeddings") is not None
            or kwargs.get("kv_bits") is not None
            or not isinstance(kwargs.get("prompt"), list)):
        return None
    return _STATE["decoder"]


class _NoLogprobs:
    """A pipelined response's logprobs: no request that reads them gets here,
    the server only looks up the token's."""

    def __getitem__(self, _token):
        return mx.array(0.0)


def _pipelined(dec, kwargs: dict):
    """stream_generate's responses from the pipelined decoder, for the prompt
    tokens _prefill_history left."""
    ad, model = dec.ad, kwargs["model"]
    tokenizer, cache = kwargs["tokenizer"], kwargs["prompt_cache"]
    prompt = kwargs["prompt"]
    stream = kwargs.get("stream", generation_stream)
    max_tokens = kwargs.get("max_tokens", 256)
    cb = kwargs.get("prompt_progress_callback") or (lambda *_: None)
    eos = set(tokenizer.eos_token_ids)
    detok = tokenizer.detokenizer
    detok.reset()
    with wired_limit(model, [stream]), mx.stream(stream):
        tic = time.perf_counter()
        cb(0, len(prompt))
        # the prompt as stock's generate_step runs it: all but the last token
        # in prefill-size chunks, then the last alone (a model can compute a
        # token differently in a chunk than alone, so this keeps plain's text)
        step, done = kwargs.get("prefill_step_size", 2048), 0
        while len(prompt) - done > 1:
            k = min(step, len(prompt) - done - 1)
            with prefill_cache(cache):
                _, rows = ad.forward_taps(
                    mx.array([prompt[done:done + k]]), cache)
            mx.eval(rows, [c.state for c in cache])
            ad.prime_pairs(rows, mx.array(
                [prompt[done + 1:done + k + 1]], dtype=mx.int32))
            done += k
            cb(done, len(prompt))
            mx.clear_cache()
        with prefill_cache(cache):
            logits, rows = ad.forward_taps(mx.array([prompt[done:]]),
                                           cache)
        mx.eval(logits, rows, [c.state for c in cache])
        cb(len(prompt), len(prompt))
        token = first = int(mx.argmax(logits[0, -1]).item())
        dec.cache = ad.views(cache)
        prompt_tps = len(prompt) / (time.perf_counter() - tic)
        tic = time.perf_counter()
        n = 0

        def response(finish_reason=None):
            return GenerationResponse(
                text=detok.last_segment, token=token, logprobs=_NoLogprobs(),
                from_draft=False, prompt_tokens=len(prompt),
                prompt_tps=prompt_tps, generation_tokens=n,
                generation_tps=n / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=finish_reason)

        start = ad.offset(dec.cache)
        later = dec.decode(first, rows[:, -1:], start,
                           max_tokens if max_tokens > 0 else 2**62, stop=eos)

        def end():
            # Stock's cache ends after the last token: the next turn restores
            # it there and prefills from the same token, so its text is
            # plain's. If no stream ran that token, run it here.
            later.close()
            if ad.offset(dec.cache) < start + n:
                ad.plain(token, dec.cache)
                mx.eval([c.state for c in cache])

        try:
            for token in itertools.chain((first,), later):
                n += 1
                if token in eos:
                    break
                detok.add_token(token)
                if n == max_tokens:
                    break
                yield response()
        except GeneratorExit:
            end()
            raise
        end()
        detok.finalize()
        yield response("stop" if token in eos else "length")


def _instrumented(fn):
    """Run the request's history prefill (_prefill_history), then the
    wrapped stream_generate; log a summary line (prefill/decode tok/s, miss
    rate, bytes read, KV restored, peak memory) once it finishes."""
    def gen(*args, **kwargs):
        pools, reader = _STATE["pools"], _STATE["reader"]
        base = aggregate_stats(pools) if pools is not None else None
        r0 = reader.bytes_read if reader else 0
        mx.reset_peak_memory()
        _STATE["mem"]["peak_all"] = 0
        _govern_request(kwargs)
        paged = _STATE.get("paged")
        dec = _pipelined_decoder(kwargs)
        if dec is not None:
            dec.ad.reset_drafter()
            drafted = dec.draft_ok, dec.draft_tried
        t0 = time.perf_counter()
        pre = _prefill_history(kwargs, dec.ad if dec is not None else None)
        t_pre = time.perf_counter() - t0
        last = None
        try:
            for last in (_pipelined(dec, kwargs) if dec is not None
                         else fn(*args, **kwargs)):
                yield last
        finally:
            # The server breaks out of its loop on stop conditions, which
            # closes this generator instead of exhausting it — log either way.
            if last is not None:
                _govern_done(kwargs.get("prompt_cache"),
                             pre + last.prompt_tokens + last.generation_tokens)
                n = pre + last.prompt_tokens
                t = t_pre + last.prompt_tokens / last.prompt_tps
                msg = (f"[streamlx] prefill {n} tok @ {n / t:.1f} tok/s | "
                       f"decode {last.generation_tokens} tok")
                if last.generation_tokens >= 2:  # rate meaningless below
                    msg += f" @ {last.generation_tps:.1f} tok/s"
                if base is not None:
                    s = aggregate_stats(pools)
                    acc = ((s["hits"] - base["hits"])
                           + (s["misses"] - base["misses"]))
                    if acc:
                        miss = (s["misses"] - base["misses"]) / acc
                        msg += f" | miss {miss * 100:.1f}%"
                if reader:
                    msg += f" | {(reader.bytes_read - r0) / 1e9:.1f} GB read"
                if paged is not None:
                    msg += (f" | kv: {paged.last[0]} of {paged.last[1]} tok "
                            f"from SSD, {paged.nbytes / 1e9:.1f} GB stored")
                if dec is not None:
                    msg += (f" | drafts {dec.draft_ok - drafted[0]}/"
                            f"{dec.draft_tried - drafted[1]}, gate "
                            f"{'on' if dec.drafting else 'off'}")
                peak = max(_STATE["mem"].get("peak_all", 0),
                           mx.get_peak_memory())
                msg += f" | peak {peak / 2**30:.1f} GiB"
                if _CFG["memory_bytes"]:
                    pb = pool_bytes(_STATE["pools"])
                    msg += f", pools {pb / 2**30:.1f} GiB"
                print(msg, file=sys.stderr)
    return gen


srv.stream_generate = _instrumented(srv.stream_generate)


class StreamingModelProvider(srv.ModelProvider):
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        # Single-model server: clients send arbitrary model names in the
        # request body (opencode, most OpenAI SDKs). Treat any requested name
        # as an alias of the model loaded at startup; the stock provider
        # would try to resolve it as a new path/repo and hit the HF hub.
        if self.model is not None:
            return self.model, self.tokenizer
        return super().load(model_path, adapter_path, draft_model_path)

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if adapter_path is not None or draft_model_path is not None:
            raise ValueError("adapters/draft models not supported with "
                             "expert streaming")
        from streamlx.pipeline_adapters import default_draft

        t0 = time.time()
        auto = _CFG["budget_bytes"] is None
        # (an automatic budget is set below, once what else is resident is
        # known: the pools start small)
        model, tokenizer, pools, reader = load_streaming_model(
            str(model_path), 2**30 if auto else _CFG["budget_bytes"],
            trust_remote_code=_CFG["trust"],
            tokenizer_config=_CFG["tokenizer_config"])
        print(f"[streamlx] trunk loaded in {time.time()-t0:.1f}s; "
              f"{len(pools)} pools", file=sys.stderr)
        _STATE["pools"], _STATE["reader"] = pools, reader
        _STATE["decoder"] = None
        head, k = default_draft(model_path, _CFG["draft_head"],
                                _CFG["draft_k"])
        drafter = None
        if head:
            from streamlx.pipeline import PipelinedDecoder
            from streamlx.pipeline_adapters import adapter_for, load_drafter

            t0 = time.time()
            try:
                drafter = load_drafter(head, model, reader)
            except Exception as e:
                if _CFG["draft_head"] != "auto":   # one that was asked for
                    raise
                print(f"[streamlx] the checkpoint's draft head did not load "
                      f"({type(e).__name__}: {e}): plain decoding",
                      file=sys.stderr)
        if drafter is not None:
            ad = adapter_for(model, drafter)
            _STATE["decoder"] = PipelinedDecoder(ad, pools, [], k_max=k,
                                                 plain_fn=ad.plain)
            print(f"[streamlx] draft head in {time.time()-t0:.1f}s, up to "
                  f"{k} tokens in flight", file=sys.stderr)
        if auto:
            # what the memory limit leaves beside the trunk and the head is
            # the most the pools may take; they start at what the first
            # request's estimate would give them (_govern_request)
            lim = _CFG["memory_bytes"]
            _CFG["budget_bytes"] = (budget_within(lim - _MARGIN,
                                                  _STATE["pools"])
                                    if lim else 8 * 2**30)
            start = (_CFG["budget_bytes"] - _buffer_reserve() - 2 * 2**30
                     if lim else _CFG["budget_bytes"])
        else:
            start = _CFG["budget_bytes"]
        got = resize_pools(_STATE["pools"], max(start, 2**30))
        slots = sum(p.n_slots for p in _STATE["pools"].values())
        print(f"[streamlx] expert pools: {slots} slots, {got / 2**30:.1f} GiB"
              f" (at most {_CFG['budget_bytes'] / 2**30:.1f})",
              file=sys.stderr)

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template
        if chat_formats.reply_framing(tokenizer):
            print("[streamlx] reply framing from the checkpoint's "
                  "response_template", file=sys.stderr)
        _STATE["suffix"] = chat_formats.generation_suffix(
            tokenizer, self.cli_args.chat_template_args)
        if _STATE["suffix"]:
            print(f"[streamlx] history ends before "
                  f"{tokenizer.decode(_STATE['suffix'])!r}", file=sys.stderr)

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.is_batchable = False  # batch-1 by design


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="most RAM the expert pools take (default: what "
                         "--memory-gib leaves; 8 without a limit)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-config", action="append", default=[],
                    metavar="KEY=JSON",
                    help="extra tokenizer kwarg, repeatable "
                         "(e.g. fix_mistral_regex=true)")
    ap.add_argument("--mlx-cache-gib", type=float, default=2.0,
                    help="cap for MLX's freed-buffer cache (default 2; "
                         "MLX's own default hoards ~95%% of RAM)")
    ap.add_argument("--kv-cache-dir", default="~/.cache/streamlx/kv",
                    help="prompt caches on SSD in this directory (default "
                         "~/.cache/streamlx/kv)")
    ap.add_argument("--kv-cache-gib", type=float, default=None,
                    help="disk space for --kv-cache-dir (default: half of "
                         "what is free there, at most 32)")
    ap.add_argument("--draft-head", default="auto",
                    help="draft head: pipelined decoding of greedy requests. "
                         "Default: the checkpoint's own if it carries one, "
                         "else plain decoding; a directory; none: plain")
    ap.add_argument("--draft-k", type=int, default=None,
                    help="tokens in flight with a draft head (default: 3 "
                         "for Qwen3.8 and Inkling 2, 6 for DeepSeek-V4.1, "
                         "else 4)")
    ap.add_argument("--memory-gib", type=float, default=None,
                    help="keep MLX's memory (active + buffer cache) under "
                         "this by resizing the expert pools (at most "
                         "--budget-gib): for each prompt chunk, then for "
                         "decoding. Default: 95%% of the GPU working set the "
                         "system recommends; 0: no limit")
    ours, rest = ap.parse_known_args()
    if {"-h", "--help"} & set(rest):   # ours, then mlx_lm.server's own
        print("streamlx serve <model dir> [options]\n\nstreamlx options:")
        print(ap.format_help().split("options:", 1)[-1].rstrip())
        print("\nmlx_lm.server options:\n", flush=True)
        sys.argv = [sys.argv[0]] + rest
        srv.main()
    if ours.budget_gib is not None:
        _CFG["budget_bytes"] = int(ours.budget_gib * 2**30)
    _CFG["trust"] = ours.trust_remote_code
    _CFG["draft_head"], _CFG["draft_k"] = ours.draft_head, ours.draft_k
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = 0
    _CFG["mlx_cache_bytes"] = int(ours.mlx_cache_gib * 2**30)
    mx.set_cache_limit(_CFG["mlx_cache_bytes"])
    _CFG["memory_bytes"] = (memory_limit() if ours.memory_gib is None
                            else int(ours.memory_gib * 2**30))
    attention.install()
    kv_dir = os.path.expanduser(ours.kv_cache_dir)
    os.makedirs(kv_dir, exist_ok=True)
    kv_gib = ours.kv_cache_gib
    if kv_gib is None:   # half of what the volume has free, at most 32
        kv_gib = max(1.0, min(32.0, shutil.disk_usage(kv_dir).free / 2**31))

    def paged(*_args, **_kwargs):
        _STATE["paged"] = PagedPromptCache(kv_dir, int(kv_gib * 2**30))
        return _STATE["paged"]

    srv.LRUPromptCache = paged
    where = f"prompt caches on SSD at {kv_dir} (<= {kv_gib:.0f} GiB)"
    lim = _CFG["memory_bytes"]
    print(f"[streamlx] ram guard: {ram / 2**30:.0f} GiB RAM, "
          + (f"MLX memory <= {lim / 2**30:.1f} GiB, " if lim else "")
          + f"{where}, mlx buffer-cache cap {ours.mlx_cache_gib:.1f} GiB",
          file=sys.stderr)
    for pair in ours.tokenizer_config:
        key, sep, val = pair.partition("=")
        if not sep:
            ap.error(f"--tokenizer-config expects KEY=JSON, got {pair!r}")
        try:
            _CFG["tokenizer_config"][key] = json.loads(val)
        except json.JSONDecodeError:
            _CFG["tokenizer_config"][key] = val  # bare strings are fine

    if not any(a.startswith("--prefill-step-size") for a in rest):
        # Each chunk's expert sweep reads ~every expert once, so prefill
        # bytes scale ~1/chunk. 4096 halves 2048's reads (Qwen3.6, 2.5K
        # prompt: 10.3 -> 7.4 s) for +1.4 GiB peak on an 8K prompt; 8192
        # costs 3 GiB more there and gains little more.
        rest += ["--prefill-step-size", "4096"]
    srv.ModelProvider = StreamingModelProvider
    sys.argv = [sys.argv[0]] + rest
    srv.main()


if __name__ == "__main__":
    main()
