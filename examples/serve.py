"""OpenAI-compatible HTTP server for SSD-streamed MoE models.

Wraps stock `mlx_lm.server` (all its flags pass through) but loads the model
with streamlx: trunk resident, experts streamed from SSD on demand.

  python examples/serve.py --budget-gib 16 \
      --model <local-model-dir> --port 8080 [--trust-remote-code] \
      [--warmstart-trace trace.npz] [--tokenizer-config key=json ...] \
      [--resident]

--model must be a LOCAL directory (the byte-range reader needs the shards on
disk). Requests are handled sequentially: concurrent batching multiplies the
per-step expert working set and collapses cache hit rates.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import mlx_lm.server as srv

from streamlx.integrate import (aggregate_stats, load_streaming_model,
                                preload_popular)

_CFG = {"budget_bytes": 8 * 2**30, "warmstart_trace": None, "trust": False,
        "tokenizer_config": {}, "resident": None}
_STATE = {"pools": None, "reader": None}


_orig_fetch = srv.LRUPromptCache.fetch_nearest_cache


def _recording_fetch(self, model, tokens):
    # Stash the request's full token key for the prompt-boundary snapshot.
    _STATE["fetch"] = (self, model, list(tokens))
    return _orig_fetch(self, model, tokens)


srv.LRUPromptCache.fetch_nearest_cache = _recording_fetch


def _snapshot_progress(kwargs):
    """Cache an extra copy of the KV state at the prompt boundary.

    Agent clients rewrite history every turn (reasoning stripped, tool
    output re-rendered), so the post-generation cache the server stores is
    never a strict prefix of the next prompt — and hybrid-cache models
    (rotating KV, conv state) cannot trim back to the divergence point, so
    every turn reprocesses the whole prompt. An extra entry keyed by the
    prompt itself makes the next turn a pure extension: no trim needed.

    The copy is taken at the prefill progress callback where
    total - processed == 1: the one point where the cache holds exactly
    prompt[:-1], evaluated. (After that, generate_step builds the next
    step's graph before the first yield, so the cache is already one
    generated token ahead — one token no hybrid cache can give back.)
    Disable with STREAMLX_PROMPT_SNAPSHOT=0.
    """
    fetch = _STATE.pop("fetch", None)
    cache = kwargs.get("prompt_cache")
    if (os.environ.get("STREAMLX_PROMPT_SNAPSHOT", "1") == "0"
            or fetch is None or cache is None):
        return
    lru, model, tokens = fetch
    if len(tokens) < 256:
        return
    orig_cb = kwargs.get("prompt_progress_callback")

    def cb(processed, total):
        if total - processed == 1:
            t0 = time.perf_counter()
            lru.insert_cache(model, tokens[:-1], copy.deepcopy(cache),
                             cache_type="user")
            print(f"[streamlx] prompt snapshot: {len(tokens) - 1} tok "
                  f"({time.perf_counter() - t0:.2f}s)", file=sys.stderr)
        if orig_cb is not None:
            orig_cb(processed, total)

    kwargs["prompt_progress_callback"] = cb


def _instrumented(fn):
    """Log a per-request summary line (prefill/decode tok/s, miss rate,
    bytes read) once the wrapped stream_generate finishes."""
    def gen(*args, **kwargs):
        pools, reader = _STATE["pools"], _STATE["reader"]
        base = aggregate_stats(pools) if pools is not None else None
        r0 = reader.bytes_read if reader else 0
        _snapshot_progress(kwargs)
        last = None
        try:
            for last in fn(*args, **kwargs):
                yield last
        finally:
            # The server breaks out of its loop on stop conditions, which
            # closes this generator instead of exhausting it — log either way.
            if last is not None:
                msg = (f"[streamlx] prefill {last.prompt_tokens} tok @ "
                       f"{last.prompt_tps:.1f} tok/s | decode "
                       f"{last.generation_tokens} tok @ "
                       f"{last.generation_tps:.1f} tok/s")
                if base is not None:
                    s = aggregate_stats(pools)
                    acc = ((s["hits"] - base["hits"])
                           + (s["misses"] - base["misses"]))
                    if acc:
                        miss = (s["misses"] - base["misses"]) / acc
                        msg += f" | miss {miss * 100:.1f}%"
                if reader:
                    msg += f" | {(reader.bytes_read - r0) / 1e9:.1f} GB read"
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
            tokenizer_config=_CFG["tokenizer_config"],
            resident=_CFG["resident"])
        slots = sum(p.n_slots for p in pools.values())
        print(f"[streamlx] trunk loaded in {time.time()-t0:.1f}s; "
              f"{len(pools)} pools, {slots} slots "
              f"({_CFG['budget_bytes']/2**30:.1f} GiB)", file=sys.stderr)
        if _CFG["warmstart_trace"]:
            t0 = time.time()
            preload_popular(pools, _CFG["warmstart_trace"])
            print(f"[streamlx] warm-start: {reader.bytes_read/1e9:.1f} GB "
                  f"in {time.time()-t0:.1f}s", file=sys.stderr)
        self.pools = pools
        self.reader = reader
        _STATE["pools"], _STATE["reader"] = pools, reader

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = None
        self.is_batchable = False  # batch-1 by design


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--budget-gib", type=float, default=8.0)
    ap.add_argument("--warmstart-trace", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-config", action="append", default=[],
                    metavar="KEY=JSON",
                    help="extra tokenizer kwarg, repeatable "
                         "(e.g. fix_mistral_regex=true)")
    ap.add_argument("--resident", action="store_true",
                    help="let fully-covered MoE layers stay on the stock "
                         "resident path (faster; trades RAM headroom that "
                         "long contexts need for KV cache)")
    ours, rest = ap.parse_known_args()
    _CFG["budget_bytes"] = int(ours.budget_gib * 2**30)
    _CFG["warmstart_trace"] = ours.warmstart_trace
    _CFG["trust"] = ours.trust_remote_code
    if ours.resident:
        _CFG["resident"] = "auto"
    for pair in ours.tokenizer_config:
        key, sep, val = pair.partition("=")
        if not sep:
            ap.error(f"--tokenizer-config expects KEY=JSON, got {pair!r}")
        try:
            _CFG["tokenizer_config"][key] = json.loads(val)
        except json.JSONDecodeError:
            _CFG["tokenizer_config"][key] = val  # bare strings are fine

    srv.ModelProvider = StreamingModelProvider
    sys.argv = [sys.argv[0]] + rest
    srv.main()


if __name__ == "__main__":
    main()
