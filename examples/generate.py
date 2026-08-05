"""Streaming generation demo: run a MoE model larger than RAM.

Streams tokens to stdout live, then prints the punchline stats
(model size vs machine RAM vs resident memory, tok/s).

  python examples/generate.py --model <local-model-dir> --budget-gib 16 \
      --prompt "..." --max-tokens 200 [--trust-remote-code] \
      [--warmstart-trace trace.npz]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache

from streamlx.integrate import (aggregate_stats, load_streaming_model,
                                preload_popular)


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

    cache = make_prompt_cache(model)
    for pid in tokenizer.encode(args.prompt):
        logits = model(mx.array([[pid]]), cache=cache)
    tok = mx.argmax(logits[:, -1:, :], axis=-1)

    detok = tokenizer.detokenizer
    detok.reset()
    for p in pools.values():
        p.reset_stats()
    reader.reset_stats()

    t0 = time.time()
    n = 0
    for _ in range(args.max_tokens):
        mx.eval(tok)
        tid = int(tok.item())
        if tid in (tokenizer.eos_token_ids or set()):
            break
        detok.add_token(tid)
        print(detok.last_segment, end="", flush=True)
        logits = model(tok, cache=cache)
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        n += 1
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
