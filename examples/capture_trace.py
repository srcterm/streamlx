"""Capture a routing trace for --warmstart-trace.

Runs your own text through the model and records which experts each layer's
router picks. `preload_popular` then loads the most frequently used experts at
startup, so pools are useful within seconds instead of after a miss-heavy
first prompt.

Feed it text that resembles your real workload (your codebase, your notes,
your prompts): expert popularity is workload dependent.

  python examples/capture_trace.py --model /path/to/model --out trace.npz \
      --input notes/*.md src/*.py [--budget-gib 8] [--trust-remote-code]

Writes trace.npz plus trace.meta.json beside it. Both are needed: the sidecar
maps trace layer axis to model layer index, which is not the identity for
models whose first layers are dense.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import make_prompt_cache

from streamlx.integrate import load_streaming_model

PAD = 0xFFFF


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="trace.npz")
    ap.add_argument("--input", nargs="+", required=True,
                    help="text files to run through the model")
    ap.add_argument("--budget-gib", type=float, default=8.0,
                    help="pool budget during capture (does not affect the "
                         "trace, only how fast capture runs)")
    ap.add_argument("--chunk", type=int, default=512,
                    help="tokens per forward pass")
    ap.add_argument("--doc-tokens", type=int, default=2048,
                    help="max tokens taken from each input file")
    ap.add_argument("--max-tokens", type=int, default=20000,
                    help="stop after this many tokens total")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model, tokenizer, pools, reader = load_streaming_model(
        args.model, int(args.budget_gib * 2**30),
        trust_remote_code=args.trust_remote_code)
    layer_ids = sorted(pools.keys())
    L = len(layer_ids)
    wrappers = [model.layers[i].mlp.switch_mlp for i in layer_ids]
    print(f"[capture] model loaded in {time.time()-t0:.1f}s; {L} MoE layers "
          f"({layer_ids[0]}..{layer_ids[-1]})", file=sys.stderr)

    blocks, seq_ids, poss = [], [], []
    total, k, seq = 0, None, 0

    for path in args.input:
        if total >= args.max_tokens:
            break
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError as e:
            print(f"[capture] skipping {path}: {e}", file=sys.stderr)
            continue
        ids = tokenizer.encode(text)[: args.doc_tokens]
        if len(ids) < 2:
            continue
        ids = ids[: args.max_tokens - total]

        cache = make_prompt_cache(model)
        sink: list = []
        for w in wrappers:
            w.record_sink = sink
        try:
            for s in range(0, len(ids), args.chunk):
                model(mx.array(ids[s:s + args.chunk])[None], cache=cache)
        finally:
            for w in wrappers:
                w.record_sink = None

        per_layer: dict[int, list] = {i: [] for i in layer_ids}
        for lyr, idx in sink:
            per_layer[lyr].append(idx.reshape(-1, idx.shape[-1]))
        if k is None:
            k = per_layer[layer_ids[0]][0].shape[-1]

        T = len(ids)
        block = np.full((T, L, k), PAD, dtype=np.uint16)
        for li, lyr in enumerate(layer_ids):
            arr = np.concatenate(per_layer[lyr], axis=0)
            if arr.shape[0] != T:
                raise RuntimeError(
                    f"layer {lyr}: recorded {arr.shape[0]} rows for {T} tokens")
            block[:, li, :] = arr.astype(np.uint16)
        blocks.append(block)
        seq_ids.append(np.full(T, seq, dtype=np.uint32))
        poss.append(np.arange(T, dtype=np.uint32))
        seq += 1
        total += T
        print(f"[capture] {Path(path).name}: {T} tokens "
              f"({total}/{args.max_tokens})", file=sys.stderr)

    if not blocks:
        print("[capture] no usable input", file=sys.stderr)
        return 1

    experts = np.concatenate(blocks, axis=0)
    out = Path(args.out)
    np.savez_compressed(
        out, experts=experts, seq_id=np.concatenate(seq_ids),
        pos=np.concatenate(poss),
        prompt_class=np.zeros(experts.shape[0], dtype=np.uint8))

    index = pools[layer_ids[0]].index
    meta = {
        "schema_version": 1,
        "model": str(Path(args.model).name),
        "n_experts": int(pools[layer_ids[0]].n_experts),
        "k": int(k),
        "n_moe_layers": L,
        "layer_ids": [int(i) for i in layer_ids],
        "bytes_per_expert": {str(i): int(index.layer_expert_bytes(i))
                             for i in layer_ids},
        "tokens": int(experts.shape[0]),
        "sequences": seq,
    }
    (out.parent / (out.stem + ".meta.json")).write_text(
        json.dumps(meta, indent=2))
    print(f"[capture] wrote {out} ({experts.shape[0]} tokens x {L} layers "
          f"x {k}) and {out.stem}.meta.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
