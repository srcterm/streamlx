#!/usr/bin/env python3
"""Terminal chat on a streamed model — no server, no client, no system prompt.

  python examples/chat.py --model /path/to/Laguna-S-2.1-oQ2e-fast \
      --budget-gib 18 --trust-remote-code --tokenizer-config fix_mistral_regex=true

Multi-turn: the prompt cache carries the whole conversation, so each turn
templates and feeds only the new message — history is never re-prefilled.
"""

import argparse
import json
import readline  # noqa: F401 -- input() line editing / history
import sys
import time

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from streamlx.integrate import aggregate_stats, load_streaming_model

HINT = ("[/reset new conversation · /stats pool counters · "
        "^C stops a reply · q, /quit or ^C^C quits]")
QUIT = {"q", "quit", "exit", "/q", "/quit", "/exit"}
DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def stream_print(seg: str, in_think: bool) -> bool:
    """Print a stream segment, rendering <think> spans dim instead of raw
    tags. Returns the updated in-think state."""
    while seg:
        head, tag, seg = seg.partition("</think>" if in_think else "<think>")
        print((DIM if in_think else "") + head + (RESET if in_think else ""),
              end="")
        if tag:
            in_think = not in_think
    return in_think


def main() -> None:
    ap = argparse.ArgumentParser(
        description="terminal chat on a streamed model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-gib", type=float, default=8.0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-config", action="append", default=[],
                    metavar="KEY=JSON",
                    help="extra tokenizer kwarg, repeatable "
                         "(e.g. fix_mistral_regex=true)")
    ap.add_argument("--resident", action="store_true",
                    help="let fully-covered MoE layers stay stock-resident")
    ap.add_argument("--system", default=None,
                    help="optional system prompt, sent on the first turn")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--mlx-cache-gib", type=float, default=2.0,
                    help="cap for MLX's freed-buffer cache (its own default "
                         "hoards ~95%% of RAM)")
    args = ap.parse_args()

    tok_cfg = {}
    for pair in args.tokenizer_config:
        key, sep, val = pair.partition("=")
        if not sep:
            ap.error(f"--tokenizer-config expects KEY=JSON, got {pair!r}")
        try:
            tok_cfg[key] = json.loads(val)
        except json.JSONDecodeError:
            tok_cfg[key] = val  # bare strings are fine

    mx.set_cache_limit(int(args.mlx_cache_gib * 2**30))
    t0 = time.monotonic()
    model, tokenizer, pools, reader = load_streaming_model(
        args.model,
        budget_bytes=int(args.budget_gib * 2**30),
        trust_remote_code=args.trust_remote_code,
        tokenizer_config=tok_cfg,
        resident="auto" if args.resident else None,
    )
    print(f"[ready in {time.monotonic() - t0:.0f}s · {len(pools)} streamed "
          f"layers @ {args.budget_gib:g} GiB]\n{HINT}", file=sys.stderr)

    sampler = make_sampler(args.temp, args.top_p)
    cache = make_prompt_cache(model)
    system = args.system
    armed = False  # one ^C at the prompt warns, a second in a row quits
    while True:
        try:
            query = input("\n>> ").strip()
        except KeyboardInterrupt:
            if armed:
                print()
                return
            armed = True
            print("\n(^C again, q, or /quit to exit)")
            continue
        except EOFError:
            print()
            return
        armed = False
        if not query:
            continue
        if query.lower() in QUIT:
            return
        if query == "/reset":
            cache = make_prompt_cache(model)
            system = args.system
            continue
        if query == "/stats":
            s = aggregate_stats(pools)
            tot = s["hits"] + s["misses"]
            if tot:
                print(f"[hit rate {s['hits'] / tot:.1%}]", file=sys.stderr)
            print("  ".join(f"{k}={v:.4g}" if isinstance(v, float)
                            else f"{k}={v}" for k, v in s.items()),
                  file=sys.stderr)
            continue
        if query.startswith("/"):  # unknown command, not a chat message
            print(HINT, file=sys.stderr)
            continue
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
            system = None  # the prompt cache carries it from here on
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(messages,
                                               add_generation_prompt=True)
        # Some templates (Laguna) end the generation prompt inside <think>:
        # reasoning then streams dim until the model closes the block.
        in_think = tokenizer.decode(prompt[-6:]).rstrip().endswith("<think>")
        r = None
        try:
            for r in stream_generate(model, tokenizer, prompt,
                                     max_tokens=args.max_tokens,
                                     sampler=sampler, prompt_cache=cache):
                in_think = stream_print(r.text, in_think)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(RESET, end="")
        if r is not None:
            print(f"\n[prompt {r.prompt_tokens} tok @ {r.prompt_tps:.0f} tok/s"
                  f" · decode {r.generation_tokens} tok @ "
                  f"{r.generation_tps:.1f} tok/s · peak {r.peak_memory:.1f} GB]",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
