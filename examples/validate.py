"""Correctness gate: with a model that FITS in RAM, greedy decode through
deliberately tiny streaming pools must match the full-RAM reference
token-for-token. For >RAM models (no reference possible), run twice at two
budgets instead — outputs must be identical (pool-size invariance).

  python examples/validate.py --model <local-dir> --slots 16 --tokens 256
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from streamlx.integrate import aggregate_stats, install_streaming

PROMPT = ("The tradeoff between memory bandwidth and storage bandwidth in "
          "local inference of mixture-of-experts models is")


def greedy(model, cache_factory, prompt_ids, n_new, sync_each=False):
    cache = cache_factory()
    for pid in prompt_ids:
        logits = model(mx.array([[pid]]), cache=cache)
    tok = mx.argmax(logits[:, -1:, :], axis=-1)
    toks = [tok]
    for _ in range(n_new - 1):
        logits = model(tok, cache=cache)
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        toks.append(tok)
        if sync_each:
            mx.eval(tok)
    mx.eval(*toks)
    return [int(t.item()) for t in toks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"[validate] reference model loaded in {time.time()-t0:.1f}s",
          file=sys.stderr)
    cache_factory = lambda: make_prompt_cache(model)  # noqa: E731
    prompt_ids = tokenizer.encode(PROMPT)

    ref = greedy(model, cache_factory, prompt_ids, args.tokens,
                 sync_each=True)
    pools, reader = install_streaming(model, args.model, args.slots)
    print(f"[validate] streaming installed: {len(pools)} layers x "
          f"{args.slots} slots", file=sys.stderr)
    stream = greedy(model, cache_factory, prompt_ids, args.tokens)

    st = aggregate_stats(pools)
    match = stream == ref
    print(f"CORRECTNESS GATE: {'PASS' if match else 'FAIL'} — "
          f"{sum(a == b for a, b in zip(ref, stream))}/{args.tokens} "
          f"tokens identical (miss_rate={st['miss_rate']:.3f}, "
          f"{reader.bytes_read/1e9:.1f} GB streamed)")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
