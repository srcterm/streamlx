"""Streaming generation demo: run a MoE model larger than RAM.

Streams tokens to stdout live, then prints the punchline stats
(model size vs machine RAM vs resident memory, tok/s).

  python examples/generate.py --model <local-model-dir> --budget-gib 16 \
      --prompt "..." --max-tokens 200 [--trust-remote-code] \
      [--warmstart-trace trace.npz] [--draft-head DIR [--draft-k 4]]

--draft-head decodes through streamlx.pipeline with an mlx-vlm block
drafter (pip install mlx-vlm), e.g. Mistral Large 4's bundled DSpark head
split out once with `python -m mlx_vlm.split_mtp --model <checkpoint>
--output <dir> --q-bits 4`: K tokens in flight, so each
token's expert reads overlap the others' compute; greedy output identical
to plain decoding. A self-gate times drafting against plain decoding and
keeps whichever is faster. DeepSeek-V4.1 and Qwen3.8 read their heads from
the checkpoint: --draft-head is the model directory, with --draft-k 6 for
V4.1 and 3 for Qwen3.8.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generation_stream, wired_limit
from mlx_lm.models.cache import make_prompt_cache

from streamlx.integrate import (aggregate_stats, load_streaming_model,
                                prefill_cache, preload_popular, resize_pools)

PREFILL_STEP = 4096


def model_size_gb(model_dir: str) -> float:
    return sum(f.stat().st_size
               for f in Path(model_dir).glob("*.safetensors")) / 1e9


def ram_gb() -> float:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-gib", type=float, default=16.0)
    ap.add_argument("--prompt", default="Explain, step by step, why "
                    "mixture-of-experts models can stream from SSD.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--warmstart-trace", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--draft-head", default=None,
                    help="mlx-vlm drafter directory: pipelined decoding")
    ap.add_argument("--stream-draft-head", action="store_true",
                    help="stream the draft head's MoE experts through a pool "
                         "in --budget-gib instead of loading them")
    ap.add_argument("--draft-k", type=int, default=4,
                    help="tokens in flight with --draft-head (default 4)")
    args = ap.parse_args()

    size = model_size_gb(args.model)
    ram = ram_gb()
    print(f"model: {Path(args.model).name}  ({size:.0f} GB weights)")
    print(f"machine RAM: {ram:.0f} GB  |  expert budget: "
          f"{args.budget_gib:.0f} GiB\n")

    t0 = time.time()
    model, tokenizer, pools, reader = load_streaming_model(
        args.model, int(args.budget_gib * 2**30),
        trust_remote_code=args.trust_remote_code)
    print(f"[load] trunk resident in {time.time()-t0:.1f}s "
          f"(experts stay on SSD)", file=sys.stderr)
    if args.warmstart_trace:
        t0 = time.time()
        preload_popular(pools, args.warmstart_trace)
        print(f"[load] warm-start in {time.time()-t0:.1f}s "
              f"({reader.bytes_read/1e9:.1f} GB)", file=sys.stderr)

    drafter = ad = None
    if args.draft_head:
        from streamlx.pipeline_adapters import adapter_for, load_drafter

        t0 = time.time()
        drafter = load_drafter(args.draft_head, model, reader,
                               stream_head=args.stream_draft_head)
        ad = adapter_for(model, drafter)
        head_pools = getattr(drafter, "budget_pools", None) or {}
        if head_pools:   # the head's experts share the budget with the trunk's
            resize_pools({**pools, **head_pools}, int(args.budget_gib * 2**30))
        print(f"[load] draft head in {time.time()-t0:.1f}s", file=sys.stderr)

    # what stream_generate holds while it runs: MLX's buffers stay wired
    # instead of pageable, so a large trunk is not compressed or swapped
    with wired_limit(model, [generation_stream]):
        ids = tokenizer.encode(args.prompt)
        cache = make_prompt_cache(model)
        context = []
        for i in range(0, len(ids), PREFILL_STEP):
            chunk = mx.array([ids[i:i + PREFILL_STEP]])
            with prefill_cache(cache):
                if ad is None:
                    logits = model(chunk, cache=cache)
                else:
                    logits, h = ad.forward_taps(chunk, cache)
                    context.append(h)
            mx.eval(logits, context)
        first = int(mx.argmax(logits[0, -1]).item())

        eos = set(tokenizer.eos_token_ids or ())
        detok = tokenizer.detokenizer
        detok.reset()
        for p in pools.values():
            p.reset_stats()
        reader.reset_stats()

        def emit(t: int) -> None:
            if t not in eos:
                detok.add_token(t)
                print(detok.last_segment, end="", flush=True)

        t0 = time.time()
        dec = None
        if drafter is not None:
            from streamlx.pipeline import PipelinedDecoder

            hidden = mx.concatenate(context, axis=1)
            del context
            drafter.reset()
            drafter.prime_pairs(hidden[:, :-1],
                                mx.array([ids[1:]], dtype=mx.int32))
            # plain_fn arms the self-gate: drafting only while it is faster
            dec = PipelinedDecoder(ad, pools, ad.views(cache),
                                   k_max=args.draft_k, plain_fn=ad.plain)
            emit(first)
            out = dec.generate(first, hidden[:, -1:], len(ids),
                               args.max_tokens, on_token=emit, stop=eos)
        else:
            out, tok = [first], first
            emit(first)
            while len(out) < args.max_tokens and tok not in eos:
                logits = model(mx.array([[tok]]), cache=cache)
                tok = int(mx.argmax(logits[0, -1]).item())
                out.append(tok)
                emit(tok)
        n = len([t for t in out if t not in eos])
        detok.finalize()
        print(detok.last_segment)
        dt = time.time() - t0

    st = aggregate_stats(pools)
    peak = mx.get_peak_memory() / 1e9
    print(f"\n--- {n} tokens in {dt:.1f}s = {n/dt:.2f} tok/s | "
          f"model {size:.0f} GB on {ram:.0f} GB RAM | "
          f"peak resident {peak:.1f} GB | "
          f"hit rate {1-st['miss_rate']:.1%} | "
          f"streamed {reader.bytes_read/1e9:.1f} GB from SSD ---")
    if dec is not None:
        gate = dec.stats()["gate"]
        print(f"--- pipeline: {dec.draft_ok}/{dec.draft_tried} drafts "
              f"accepted, tokens per step "
              f"{dict(sorted(dec.accepted_hist.items()))}, "
              f"{dec.plain_steps} plain steps, gate "
              f"{gate['decisions'] or gate['arm']} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
