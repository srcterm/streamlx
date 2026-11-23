"""Streaming generation demo: run a MoE model larger than RAM.

Streams tokens to stdout live, then prints the punchline stats
(model size vs machine RAM vs resident memory, tok/s).

  streamlx generate <local-model-dir> --prompt "..." [--max-tokens 200] \
      [--budget-gib N] [--raw] [--chat-template-args JSON] \
      [--draft-head DIR|none [--draft-k K]] [--trust-remote-code]

With no options the expert pools take what the memory limit leaves (95% of
the GPU working set the system recommends, less the model's resident part
and 4 GiB) and give a prompt chunk's working memory room during prefill
(integrate.prefill_plan), the prompt goes through the checkpoint's chat
template (--raw: as it is), and a checkpoint that carries a draft head
(Qwen3.8, DeepSeek-V4.1, Inkling 2) decodes pipelined with it.

A draft head decodes through streamlx.pipeline: K tokens in flight, so each
token's expert reads overlap the others' compute; greedy output identical
to plain decoding. A self-gate times drafting against plain decoding and
keeps whichever is faster. DeepSeek-V4.1, Qwen3.8 and Inkling 2 carry their
heads in the checkpoint (--draft-head <model dir> where it is not the
default); otherwise --draft-head is an mlx-vlm block drafter's directory,
e.g. Mistral Large 4's DSpark head split out once with `python -m
mlx_vlm.split_mtp --model <checkpoint> --output <dir> --q-bits 4`.
Any model, with a head or none, also drafts by copy where the reply repeats
the prompt or itself (pipeline.Lookup; --draft-head none: plain decoding).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generation_stream, wired_limit
from mlx_lm.models.cache import make_prompt_cache

from streamlx.integrate import (
    aggregate_stats,
    budget_within,
    load_streaming_model,
    memory_limit,
    pool_floor,
    prefill_cache,
    prefill_chunks,
    prefill_plan,
    resize_pools,
)
from streamlx.pipeline_adapters import greedy, token_ids
from streamlx.pool import all_held

PREFILL_STEP = 4096


def model_size_gb(model_dir: str) -> float:
    return sum(f.stat().st_size
               for f in Path(model_dir).glob("*.safetensors")) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="RAM for the expert pools (default: what the "
                         "memory limit leaves)")
    ap.add_argument("--prompt", default="Explain, step by step, why "
                    "mixture-of-experts models can stream from SSD.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="the prompt is the model's input as it is (default: "
                         "inside the checkpoint's chat template)")
    ap.add_argument("--chat-template-args", type=json.loads, default={},
                    help="JSON for the chat template, e.g. "
                         "'{\"enable_thinking\": false}'")
    ap.add_argument("--draft-head", default="auto",
                    help="draft head: pipelined decoding. Default: the "
                         "checkpoint's own if it carries one, else plain "
                         "decoding; a directory; none: plain")
    ap.add_argument("--draft-k", type=int, default=None,
                    help="tokens in flight with a draft head (default: 3 "
                         "for Qwen3.8 and Inkling 2, 6 for DeepSeek-V4.1, "
                         "else 4)")
    args = ap.parse_args()
    from streamlx.pipeline_adapters import default_draft, headless_adapter

    size = model_size_gb(args.model)
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    print(f"model: {Path(args.model).name}  ({size:.0f} GB weights)")
    mx.set_cache_limit(2 * 2**30)   # MLX's own would keep ~all freed buffers

    t0 = time.time()
    auto = args.budget_gib is None   # sized below, once the rest is resident
    model, tokenizer, pools, reader = load_streaming_model(
        args.model, 2**30 if auto else int(args.budget_gib * 2**30),
        trust_remote_code=args.trust_remote_code)
    print(f"[load] trunk resident in {time.time()-t0:.1f}s "
          f"(experts stay on SSD)", file=sys.stderr)

    drafter = ad = None
    head, draft_k = default_draft(args.model, args.draft_head, args.draft_k)
    if head:
        from streamlx.pipeline_adapters import adapter_for, load_drafter

        t0 = time.time()
        try:
            drafter = load_drafter(head, model, reader)
        except Exception as e:
            if args.draft_head != "auto":   # one that was asked for
                raise
            print(f"[load] the checkpoint's draft head did not load "
                  f"({type(e).__name__}: {e}): plain decoding",
                  file=sys.stderr)
    if drafter is not None:
        ad = adapter_for(model, drafter)
        print(f"[load] draft head in {time.time()-t0:.1f}s", file=sys.stderr)
    elif args.draft_head == "auto" and not head:
        ad = headless_adapter(model, pools)    # drafts by copy alone
    # the memory limit, less what is resident, MLX's buffer cache and 2 GiB;
    # the pools take it, or every expert when that is less
    room = (budget_within(memory_limit(), pools, reserve=4 * 2**30)
            if auto else int(args.budget_gib * 2**30))
    # (sized, not read yet: a prompt whose chunks need the pools smaller
    # would drop what was read and read it again after the prefill)
    budget = resize_pools(pools, room, fill=False)
    print(f"machine RAM: {ram:.0f} GB  |  expert budget: "
          f"{budget / 2**30:.1f} GiB\n")

    # what stream_generate holds while it runs: MLX's buffers stay wired
    # instead of pageable, so a large trunk is not compressed or swapped
    with wired_limit(model, [generation_stream]):
        if args.raw or not getattr(tokenizer, "has_chat_template", False):
            ids = tokenizer.encode(args.prompt)
        else:
            ids = list(tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True, **args.chat_template_args))
        cache = make_prompt_cache(model)
        context = []
        if auto:   # the pools shrink for the prompt chunks, then grow back
            sizes, pre = prefill_plan(len(ids), room, pool_floor(pools),
                                      PREFILL_STEP)
            if pre < budget:
                got = resize_pools(pools, pre)
                print(f"[prefill] {len(sizes)} chunk(s) of {sizes[0]} tokens:"
                      f" expert pools {got / 2**30:.1f} GiB", file=sys.stderr)
        else:
            sizes, pre = prefill_chunks(len(ids), PREFILL_STEP), budget
        if pre >= budget:      # the pools stay: what fits whole is read now
            resize_pools(pools, room)
        done = 0
        for n in sizes:
            chunk = mx.array([ids[done:done + n]])
            done += n
            with prefill_cache(cache):
                if ad is None:
                    logits = model(chunk, cache=cache)
                else:
                    logits, h = ad.forward_taps(chunk, cache)
                    context.append(h)
            mx.eval(logits, context)
        if pre < budget:
            resize_pools(pools, budget)
        first = int(greedy(logits[0, -1]).item())

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
        if ad is not None:
            from streamlx.pipeline import PipelinedDecoder

            hidden = None
            if drafter is not None:
                hidden = mx.concatenate(context, axis=1)
                drafter.reset()
                drafter.prime_pairs(hidden[:, :-1],
                                    mx.array([ids[1:]], dtype=mx.int32))
                hidden = hidden[:, -1:]
            del context
            # plain_fn arms the self-gate: drafting only while it is faster
            dec = PipelinedDecoder(ad, pools, ad.views(cache),
                                   k_max=draft_k, plain_fn=ad.plain)
            emit(first)
            out = dec.generate(first, hidden, len(ids), args.max_tokens,
                               on_token=emit, stop=eos, context=ids)
        else:
            def queue(t):
                """The forward of `t` (an id, or the step before's token
                still being computed), queued: its greedy token."""
                y = greedy(model(token_ids(t), cache=cache)[0, -1])
                mx.async_eval(y)
                return y

            # every expert held: a forward never waits on the host, so the
            # next one is queued on this one's argmax while the GPU computes
            # it (as mlx-lm's generate_step; same ids)
            ahead = all_held(pools)
            out, tok, y = [first], first, None
            emit(first)
            while len(out) < args.max_tokens and tok not in eos:
                y = queue(tok) if y is None else y
                nxt = (queue(y) if ahead and len(out) + 1 < args.max_tokens
                       else None)
                tok = int(y.item())
                out.append(tok)
                emit(tok)
                y = nxt
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
    if dec is not None and (drafter is not None or dec.draft_tried):
        gate = dec.stats()["gate"]
        lk = dec.lookup
        by_copy = (f" ({lk.kept}/{lk.proposed} by copy)"
                   if lk is not None and lk.proposed else "")
        print(f"--- pipeline: {dec.draft_ok}/{dec.draft_tried} drafts "
              f"accepted{by_copy}, tokens per step "
              f"{dict(sorted(dec.accepted_hist.items()))}, "
              f"{dec.plain_steps} plain steps"
              + (f", gate {gate['decisions'] or gate['arm']}" if gate else "")
              + " ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
