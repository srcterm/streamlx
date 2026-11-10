"""Pool-budget sweep: measured tok/s, miss rate, and memory vs expert budget.

  python examples/sweep.py --model <local-model-dir> --budget-gib 12 \
      [--warmup 96] [--measure 192] [--trust-remote-code] [--out sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache

from streamlx.integrate import aggregate_stats, load_streaming_model

PROMPT = ("The tradeoff between memory bandwidth and storage bandwidth in "
          "local inference of mixture-of-experts models is")

CSV_FIELDS = ["ts", "budget_gib", "slots_total", "trunk_load_s",
              "warmup_tokens", "measured_tokens", "tok_s",
              "miss_rate", "miss_mb_per_token", "fetch_ms_per_expert",
              "fetch_bw_gbs", "peak_rss_gb", "mlx_peak_gb", "notes"]


def greedy_step(model, cache, tok_arr):
    logits = model(tok_arr, cache=cache)
    return mx.argmax(logits[:, -1:, :], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--warmup", type=int, default=96)
    ap.add_argument("--measure", type=int, default=192)
    ap.add_argument("--out", default="sweep.csv")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    budget = int(args.budget_gib * 2**30)
    t0 = time.time()
    model, tokenizer, pools, reader = load_streaming_model(
        args.model, budget, trust_remote_code=args.trust_remote_code)
    trunk_s = time.time() - t0
    slots_total = sum(p.n_slots for p in pools.values())
    print(f"[sweep] {args.budget_gib} GiB: trunk in {trunk_s:.1f}s, "
          f"{len(pools)} pools, {slots_total} slots", file=sys.stderr)

    cache = make_prompt_cache(model)
    for pid in tokenizer.encode(PROMPT):
        logits = model(mx.array([[pid]]), cache=cache)
    tok = mx.argmax(logits[:, -1:, :], axis=-1)
    for _ in range(args.warmup):
        tok = greedy_step(model, cache, tok)
    mx.eval(tok)

    for p in pools.values():
        p.reset_stats()
    reader.reset_stats()
    t0 = time.time()
    for _ in range(args.measure):
        tok = greedy_step(model, cache, tok)
    mx.eval(tok)
    dt = time.time() - t0
    st = aggregate_stats(pools)

    tok_s = args.measure / dt
    miss_mb = reader.bytes_read / args.measure / 1e6
    fetch_ms = st["fetch_s"] * 1000 / max(st["misses"], 1)
    fetch_bw = (reader.bytes_read / st["fetch_s"] / 1e9
                if st["fetch_s"] else 0.0)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "budget_gib": args.budget_gib, "slots_total": slots_total,
           "trunk_load_s": round(trunk_s, 1), "warmup_tokens": args.warmup,
           "measured_tokens": args.measure, "tok_s": round(tok_s, 2),
           "miss_rate": round(st["miss_rate"], 4),
           "miss_mb_per_token": round(miss_mb, 1),
           "fetch_ms_per_expert": round(fetch_ms, 2),
           "fetch_bw_gbs": round(fetch_bw, 2),
           "peak_rss_gb": round(
               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
           "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2),
           "notes": ""}
    out = Path(args.out)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[sweep] budget={args.budget_gib} GiB: tok/s={tok_s:.2f} "
          f"miss_rate={st['miss_rate']:.3f} miss={miss_mb:.0f} MB/tok "
          f"fetch_bw={fetch_bw:.2f} GB/s mlx_peak={row['mlx_peak_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
