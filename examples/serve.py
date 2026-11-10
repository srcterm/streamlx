"""OpenAI-compatible HTTP server for SSD-streamed MoE models.

Wraps stock `mlx_lm.server` (all its flags pass through) but loads the model
with streamlx: trunk resident, experts streamed from SSD on demand.

  python examples/serve.py --budget-gib 16 \
      --model <local-model-dir> --port 8080 [--trust-remote-code] \
      [--tokenizer-config key=json ...] \
      [--prompt-cache-gib N] [--mlx-cache-gib N] \
      [--kv-cache-dir DIR [--kv-cache-gib N]] \
      [--draft-head DIR [--draft-k 4] [--stream-draft-head]] \
      [--memory-gib N] [--prefill-memory-gib N]

--model must be a LOCAL directory (the byte-range reader needs the shards on
disk). Requests are handled sequentially: concurrent batching multiplies the
per-step expert working set and collapses cache hit rates.

RAM guard: the stock server bounds its prompt-cache LRU by sequence count
only (bytes are enforced on the batch path we don't use) and MLX's default
buffer-cache limit is ~95% of RAM, so freed buffers are hoarded forever.
With most of that wired for Metal, a long agent session runs the machine out
of unpageable memory. Both get hard byte caps here: --prompt-cache-gib
(default: RAM - budget - 12 GiB, clamped to [1, 8]) and --mlx-cache-gib
(default 2).

--memory-gib keeps MLX's memory (active + buffer cache) under a limit by
resizing the expert pools, never above --budget-gib: before each request,
for the cache it starts from plus what its prompt and max_tokens will add
(at the bytes a token has added so far), with the rest of the peak taken from
earlier requests; between prefill chunks it shrinks them if the next chunk
would pass the limit. A long conversation so gives slots back as its cache
grows, and a short one takes them again. Resizing copies the kept slots (~1 s
for 16 GiB); outputs don't change. --prefill-memory-gib caps what a prefill
holds beyond the arrays it uses: MLX's freed buffers (a third, at most
0.25 GiB) and the expert reads queued ahead of the compute (the rest, at most
0.5 GiB).

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

--kv-cache-dir moves prompt caches to SSD instead (streamlx.kvcache): stored
once as segments shared by every request with the same prefix, so RAM holds
only the active request's cache. --kv-cache-gib bounds the disk use (default
32; oldest segments go first). A drive other than the model's avoids sharing
its bandwidth with expert reads.

--draft-head decodes greedy requests through streamlx.pipeline (PEF): K tokens
in flight, so each token's expert reads overlap the others'
compute, with plain decoding's output. It takes what generate.py's does: the
model directory for Qwen3.8 and DeepSeek-V4.1, an mlx-vlm drafter directory
otherwise. With --stream-draft-head, Qwen3.8's head streams its MoE layer
through a pool taken out of --budget-gib (V4.1's stages always stream, in
pools of their own). The self-gate's verdict carries across requests.
Requests that sample, process logits or ask for logprobs decode plainly. On a
prompt-cache hit the head's context starts at the tokens this request
prefills: drafts are verified, so that can only cost acceptance, not change
the text.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import os
import sys
import time

import mlx.core as mx
import mlx_lm.server as srv
from mlx_lm.generate import (GenerationResponse, generation_stream,
                             maybe_quantize_kv_cache, wired_limit)
from mlx_lm.models.cache import can_trim_prompt_cache

from streamlx import attention
from streamlx import integrate as _integrate
from streamlx.integrate import (aggregate_stats, load_streaming_model,
                                pool_bytes, prefill_cache, resize_pools)
from streamlx.kvcache import PagedPromptCache

_CFG = {"budget_bytes": 8 * 2**30, "trust": False, "tokenizer_config": {},
        "prompt_cache_bytes": 4 * 2**30, "draft_head": None, "draft_k": 4,
        "memory_bytes": 0, "mlx_cache_bytes": 2 * 2**30,
        "prefill_cache_bytes": 0, "stream_head": False}
_STATE = {"pools": None, "reader": None, "all_pools": None,
          "mem": {"F": None, "slope": None}}


_orig_fetch = srv.LRUPromptCache.fetch_nearest_cache


def _recording_fetch(self, model, tokens):
    # Stash the request's full token key for the history snapshot.
    _STATE["fetch"] = (self, model, list(tokens))
    return _orig_fetch(self, model, tokens)


srv.LRUPromptCache.fetch_nearest_cache = _recording_fetch


_orig_lru_init = srv.LRUPromptCache.__init__


def _bounded_lru_init(self, *args, **kwargs):
    # The stock server only enforces --prompt-cache-bytes on its batch path;
    # ours is sequential, where inserts are bounded by sequence count alone.
    # Setting max_bytes makes the stock insert_cache evict on every path.
    _orig_lru_init(self, *args, **kwargs)
    self.max_bytes = min(self.max_bytes, _CFG["prompt_cache_bytes"])


srv.LRUPromptCache.__init__ = _bounded_lru_init


_orig_sampler = srv._make_sampler


def _recording_sampler(args, tokenizer):
    # What the request asked for: stream_generate only gets the closures.
    _STATE["args"] = args
    return _orig_sampler(args, tokenizer)


srv._make_sampler = _recording_sampler


def _gen_suffix(tokenizer, template_args: dict) -> list[int]:
    """The tokens the chat template appends for add_generation_prompt, e.g.
    Qwen3.6's "<|im_start|>assistant\\n<think>\\n" ([] if none)."""
    msgs = [{"role": "user", "content": "x"}]
    try:
        a, b = (list(tokenizer.apply_chat_template(
            msgs, add_generation_prompt=g, tokenize=True, **template_args))
            for g in (False, True))
    except Exception:  # noqa: BLE001 — no template, or one needing more
        return []
    return b[len(a):] if b[:len(a)] == a else []


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


def _snapshot(fetch, n: int, cache) -> None:
    """RAM prompt cache: store a copy of the live cache at tokens[:n].

    Skipped when the cache is still trimmable (the stock trim path already
    reuses those; a copy would only double memory) and when the copy alone
    would exceed half the LRU byte cap (storing it would evict everything
    else and then be evicted itself). Disable with STREAMLX_PROMPT_SNAPSHOT=0.
    """
    lru, model, tokens = fetch
    if (os.environ.get("STREAMLX_PROMPT_SNAPSHOT", "1") == "0" or n < 256
            or can_trim_prompt_cache(cache)):
        return
    snap = sum(c.nbytes for c in cache)
    if snap > lru.max_bytes // 2:
        print(f"[streamlx] prompt snapshot skipped: {snap / 2**30:.2f} GiB "
              f"copy vs {lru.max_bytes / 2**30:.1f} GiB cache cap",
              file=sys.stderr)
        return
    t0 = time.perf_counter()
    lru.insert_cache(model, tokens[:n], copy.deepcopy(cache),
                     cache_type="user")
    print(f"[streamlx] prompt snapshot: {n} tok "
          f"({time.perf_counter() - t0:.2f}s)", file=sys.stderr)


_MARGIN = 2**28   # --memory-gib keeps this much of its limit free,
_HYST = 2**28     # shrinks the pools when they pass it by this much
_GROW = 2**30     # and grows them back when this much is free


@contextlib.contextmanager
def _prefill_buffers():
    """--prefill-memory-gib: MLX's freed-buffer cap while prefilling."""
    lim = _CFG["prefill_cache_bytes"]
    if lim:
        mx.set_cache_limit(lim)
    try:
        yield
    finally:
        if lim:
            mx.set_cache_limit(_CFG["mlx_cache_bytes"])


def _buffer_reserve() -> int:
    """The buffer cache a peak can hold: prefill's cap, where peaks are."""
    return _CFG["prefill_cache_bytes"] or _CFG["mlx_cache_bytes"]


def _cache_bytes(cache) -> int:
    try:
        return sum(c.nbytes for c in cache or ())
    except Exception:  # noqa: BLE001 — a cache type without nbytes
        return 0


def _fit_pools(target: float, why: str) -> None:
    """Resize every expert pool (trunk and draft head) to `target` bytes in
    all, at most --budget-gib: down by _HYST or more, up by _GROW or more
    (a resize copies every kept slot, ~1 s for 16 GiB)."""
    pools = _STATE["all_pools"] or {}
    have = pool_bytes(pools)
    target = min(target, _CFG["budget_bytes"])
    if not pools or not (have - target >= _HYST or target - have >= _GROW):
        return
    t0 = time.perf_counter()
    got = resize_pools(pools, max(target, 0))
    print(f"[streamlx] memory: pools {have / 2**30:.1f} -> "
          f"{got / 2**30:.1f} GiB ({why}; {time.perf_counter() - t0:.1f}s)",
          file=sys.stderr)


def _govern_request(kwargs: dict) -> None:
    """--memory-gib, before a request: pools sized so its peak should stay
    under the limit. The rest of a peak (trunk, draft head, working set) is
    what earlier requests measured above their pools and caches; the cache is
    the one the request starts from plus the bytes a token has added so far
    times the tokens it will add (its prompt and max_tokens)."""
    lim, m = _CFG["memory_bytes"], _STATE["mem"]
    if not lim:
        return
    cache, prompt = kwargs.get("prompt_cache"), kwargs.get("prompt")
    n_new = (len(prompt) if isinstance(prompt, list) else 0) + max(
        0, kwargs.get("max_tokens", 256))
    cb = _cache_bytes(cache)
    pools = pool_bytes(_STATE["all_pools"] or {})
    rest = m["F"]
    if rest is None:   # first request: what is resident now, plus 2 GiB
        rest = mx.get_active_memory() - pools - cb + 2 * 2**30
    grow = (m["slope"] or 0) * n_new
    _fit_pools(lim - _MARGIN - _buffer_reserve() - rest - cb - grow,
               f"cache {cb / 2**30:.2f} + {grow / 2**30:.2f} GiB to come")
    m["cb0"], m["peak_rest"] = cb, 0
    mx.reset_peak_memory()


def _govern_mark(cache) -> None:
    """Record the request's peak above its pools and cache so far."""
    m = _STATE["mem"]
    rest = (mx.get_peak_memory() - pool_bytes(_STATE["all_pools"] or {})
            - _cache_bytes(cache))
    m["peak_rest"] = max(m.get("peak_rest", 0), rest)


def _govern_chunk(cache, done: int, left: int) -> None:
    """--memory-gib between prefill chunks: shrink the pools if the next
    chunk's peak, at the peak so far plus its tokens' cache, would pass the
    limit (the first request's estimate can be short)."""
    lim, m = _CFG["memory_bytes"], _STATE["mem"]
    if not lim or done <= 0:
        return
    cb = _cache_bytes(cache)
    _measure_slope(cb, done)
    _govern_mark(cache)
    over = (mx.get_peak_memory() + _buffer_reserve()
            + (m["slope"] or 0) * left - (lim - _MARGIN))
    if over >= _HYST:
        _fit_pools(pool_bytes(_STATE["all_pools"] or {}) - over,
                   "prefill guard")
        mx.reset_peak_memory()


def _measure_slope(cb: int, n_tokens: int) -> None:
    """The cache bytes a token adds, from a request that continued a cache
    (one that starts empty also allocates fixed state, e.g. recurrent
    layers') over at least 256 tokens."""
    m = _STATE["mem"]
    if m.get("cb0", 0) > 0 and n_tokens >= 256 and cb > m["cb0"]:
        m["slope"] = max(m["slope"] or 0, (cb - m["cb0"]) / n_tokens)


def _govern_done(cache, n_tokens: int) -> None:
    """After a request: the bytes a token adds to the cache and the peak
    above pools and cache (kept as the maximum over requests)."""
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
    to it. Stored there (a segment with --kv-cache-dir, else a RAM copy), the
    next turn restores it and prefills only its new tokens. Prompts without
    that suffix stop one token short of their end.
    """
    fetch = _STATE.pop("fetch", None)
    prompt, cache, model = (kwargs.get(k)
                            for k in ("prompt", "prompt_cache", "model"))
    if (cache is None or model is None or not isinstance(prompt, list)
            or kwargs.get("draft_model") is not None):
        return 0
    sfx = _STATE.get("suffix") or []
    cut = len(prompt) - (len(sfx) if sfx and prompt[-len(sfx):] == sfx
                         else 1)
    if cut < 1:
        return 0
    paged = _STATE.get("paged")
    base = (paged.last[0] if paged is not None
            else len(fetch[2]) - len(prompt) if fetch else 0)
    cb = kwargs.get("prompt_progress_callback") or (lambda *_: None)
    stream = kwargs.get("stream", generation_stream)
    tokens, done = mx.array(prompt[:cut]), 0
    cb(0, len(prompt))
    with wired_limit(model, [stream]), mx.stream(stream), _prefill_buffers():
        for n in _chunks(cut, kwargs.get("prefill_step_size", 2048)):
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
            _govern_chunk(cache, done, len(prompt) - done)
    if paged is None and fetch is not None:
        _snapshot(fetch, base + cut, cache)
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
            or kwargs.get("draft_model") is not None
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
        with _prefill_buffers():
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
                msg += f" | peak {mx.get_peak_memory() / 2**30:.1f} GiB"
                if _CFG["memory_bytes"]:
                    pb = pool_bytes(_STATE["all_pools"])
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
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None

        t0 = time.time()
        model, tokenizer, pools, reader = load_streaming_model(
            str(model_path), _CFG["budget_bytes"],
            trust_remote_code=_CFG["trust"],
            tokenizer_config=_CFG["tokenizer_config"])
        slots = sum(p.n_slots for p in pools.values())
        print(f"[streamlx] trunk loaded in {time.time()-t0:.1f}s; "
              f"{len(pools)} pools, {slots} slots "
              f"({_CFG['budget_bytes']/2**30:.1f} GiB)", file=sys.stderr)
        self.pools = pools
        self.reader = reader
        _STATE["pools"], _STATE["reader"] = pools, reader
        _STATE["all_pools"] = dict(pools)
        _STATE["decoder"] = None
        if _CFG["draft_head"]:
            from streamlx.pipeline import PipelinedDecoder
            from streamlx.pipeline_adapters import adapter_for, load_drafter

            t0 = time.time()
            drafter = load_drafter(_CFG["draft_head"], model, reader,
                                   stream_head=_CFG["stream_head"])
            ad = adapter_for(model, drafter)
            _STATE["decoder"] = PipelinedDecoder(ad, pools, [],
                                                 k_max=_CFG["draft_k"],
                                                 plain_fn=ad.plain)
            head_pools = getattr(drafter, "budget_pools", None) or {}
            if head_pools:
                # the head's experts share the budget with the trunk's
                _STATE["all_pools"] = {**pools, **head_pools}
                resize_pools(_STATE["all_pools"], _CFG["budget_bytes"])
            print(f"[streamlx] draft head in {time.time()-t0:.1f}s, up to "
                  f"{_CFG['draft_k']} tokens in flight"
                  + (f", {len(head_pools)} streamed MoE layer(s) in the "
                     f"budget" if head_pools else ""), file=sys.stderr)

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template
        _STATE["suffix"] = _gen_suffix(tokenizer,
                                       self.cli_args.chat_template_args)
        if _STATE["suffix"]:
            print(f"[streamlx] history ends before "
                  f"{tokenizer.decode(_STATE['suffix'])!r}", file=sys.stderr)

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = None
        self.is_batchable = False  # batch-1 by design


def _auto_prompt_cache_bytes(ram: int, budget: int) -> int:
    """Whatever RAM the budget leaves after ~12 GiB for trunk, OS, the
    active generation's cache and the per-turn copy transients — clamped
    to [1, 8] GiB. Fallback 4 GiB if RAM can't be detected."""
    if not ram:
        return 4 * 2**30
    return min(max(ram - budget - 12 * 2**30, 1 * 2**30), 8 * 2**30)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--budget-gib", type=float, default=8.0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-config", action="append", default=[],
                    metavar="KEY=JSON",
                    help="extra tokenizer kwarg, repeatable "
                         "(e.g. fix_mistral_regex=true)")
    ap.add_argument("--prompt-cache-gib", type=float, default=None,
                    help="byte cap for the server's prompt-cache LRU "
                         "(default: RAM - budget - 12 GiB, clamped to "
                         "[1, 8]); replaces --prompt-cache-bytes here")
    ap.add_argument("--mlx-cache-gib", type=float, default=2.0,
                    help="cap for MLX's freed-buffer cache (default 2; "
                         "MLX's own default hoards ~95% of RAM)")
    ap.add_argument("--kv-cache-dir", default=None,
                    help="keep prompt caches on SSD in this directory "
                         "(paged; replaces the RAM prompt cache)")
    ap.add_argument("--kv-cache-gib", type=float, default=32.0,
                    help="disk space for --kv-cache-dir (default 32)")
    ap.add_argument("--draft-head", default=None,
                    help="draft head (generate.py's --draft-head): pipelined "
                         "decoding of greedy requests")
    ap.add_argument("--draft-k", type=int, default=4,
                    help="tokens in flight with --draft-head (default 4)")
    ap.add_argument("--stream-draft-head", action="store_true",
                    help="stream the draft head's MoE experts through a pool "
                         "in --budget-gib instead of loading them (Qwen3.8: "
                         "1.3 GiB less RAM, ~15%% slower pipelined decode)")
    ap.add_argument("--prefill-memory-gib", type=float, default=None,
                    help="cap what prefill holds beyond the arrays it uses: "
                         "MLX's freed-buffer cache while prefilling (a third, "
                         "at most 0.25 GiB) and the expert reads queued "
                         "ahead of the compute (the rest, at most 0.5 GiB). "
                         "Default: --mlx-cache-gib and 0.5 GiB")
    ap.add_argument("--memory-gib", type=float, default=None,
                    help="keep MLX's memory (active + buffer cache) under "
                         "this: before each request the expert pools are "
                         "resized (at most --budget-gib) for the cache it "
                         "will grow to, from earlier requests' peaks")
    ours, rest = ap.parse_known_args()
    _CFG["budget_bytes"] = int(ours.budget_gib * 2**30)
    _CFG["trust"] = ours.trust_remote_code
    _CFG["draft_head"], _CFG["draft_k"] = ours.draft_head, ours.draft_k
    _CFG["stream_head"] = ours.stream_draft_head
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = 0
    if ours.prompt_cache_gib is not None:
        _CFG["prompt_cache_bytes"] = int(ours.prompt_cache_gib * 2**30)
    else:
        _CFG["prompt_cache_bytes"] = _auto_prompt_cache_bytes(
            ram, _CFG["budget_bytes"])
    _CFG["mlx_cache_bytes"] = int(ours.mlx_cache_gib * 2**30)
    mx.set_cache_limit(_CFG["mlx_cache_bytes"])
    if ours.memory_gib:
        _CFG["memory_bytes"] = int(ours.memory_gib * 2**30)
    if ours.prefill_memory_gib:
        cap = int(ours.prefill_memory_gib * 2**30)
        _CFG["prefill_cache_bytes"] = min(_CFG["mlx_cache_bytes"],
                                          max(2**26, min(2**28, cap // 3)))
        reads = cap - _CFG["prefill_cache_bytes"]
        _integrate._NEXT_BYTES = max(2**26, min(_integrate._NEXT_BYTES, reads))
    attention.install()
    if ours.kv_cache_dir:
        def paged(*_args, **_kwargs):
            _STATE["paged"] = PagedPromptCache(
                ours.kv_cache_dir, int(ours.kv_cache_gib * 2**30))
            return _STATE["paged"]

        srv.LRUPromptCache = paged
        where = (f"prompt caches on SSD at {ours.kv_cache_dir} "
                 f"(<= {ours.kv_cache_gib:.0f} GiB)")
    else:
        where = (f"prompt-cache cap "
                 f"{_CFG['prompt_cache_bytes'] / 2**30:.1f} GiB")
    print(f"[streamlx] ram guard: {ram / 2**30:.0f} GiB RAM, {where}, "
          f"mlx buffer-cache cap {ours.mlx_cache_gib:.1f} GiB",
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
