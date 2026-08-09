# streamlx

**Run MoE models larger than RAM on Apple Silicon by streaming expert weights from SSD.**

streamlx keeps a quantized MLX checkpoint's trunk resident and leaves the routed experts, typically 85-95% of the model's bytes, on disk. Each MoE layer gets a fixed-budget LRU pool of expert slots; misses are read on demand from the safetensors shards while the GPU computes with the experts already loaded. Output is bitwise identical to the stock model.

## Install

Apple Silicon, Python 3.10+, `mlx-lm` 0.31+. Not on PyPI yet.

```sh
git clone https://github.com/srcterm/streamlx
pip install -e streamlx
```

## Serve

OpenAI-compatible; point any client or coding agent at it:

```sh
python examples/serve.py --model /path/to/model --budget-gib 12 --port 8080
```

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Models must be local directories in MLX format with the standard stacked `switch_mlp` expert layout. Validated on `laguna`, `kimi_linear`, and `qwen3_5_moe`; other mlx-lm MoE models should work unmodified.

| Flag | Default | |
|---|---|---|
| `--budget-gib` | 8.0 | RAM reserved for experts, the one knob that matters. Set it as high as fits: about 16 on a 32 GB Mac, 36 on 64 GB. Too high is worse than too low, past roughly 60% of RAM the machine swaps and throughput collapses. |
| `--trust-remote-code` | off | Needed for models with custom tokenizer code. |
| `--tokenizer-config` | none | Extra tokenizer kwargs as `key=json`, repeatable. Laguna's Mistral-derived tokenizer needs `fix_mistral_regex=true`. |
| `--warmstart-trace` | none | Preload popular experts from a routing trace. See below. |

All `mlx_lm.server` flags pass through (`--port`, `--host`, `--max-tokens`, `--temp`, ...). Requests run one at a time by design: batching unions each step's expert picks and collapses the hit rate. `--draft-model` and `--adapter-path` are unsupported.

## Performance

M4 MacBook Air, 32 GB, macOS 26.3, ~2.4 GB/s SSD, batch-1 greedy decode. Resident memory is about `trunk + budget + 1-2 GB`.

**Laguna-S-2.1** (Poolside, 117B, top-10-of-256 routing, OptiQ ~3 bpw, 44.2 GB). Stock loading dies of swap on this machine; streaming:

| Budget | Resident | tok/s | Miss rate |
|---|---|---|---|
| 8 GiB | 12 GB | 2.6 | 39% |
| 12 GiB | 17 GB | 3.1 | 28% |
| 16 GiB | 21 GB | 3.8 | 21% |
| 18 GiB | 23 GB | 4.1 | 18% |

Warm-starting the 16 GiB pool measures as a wash here (3.9 tok/s, 19.8% miss) and swaps ~1.5 GB during the 17 GB preload: at 470 expert-loads per token this budget is nowhere near covering the hot set. See the warm-start caveat below.

Prefill runs at ~32 tok/s via the expert-major sweep, roughly budget-independent: each chunk reads its routed experts once at sequential bandwidth.

**Qwen3.6-35B-A3B** (4-bit, 22.1 GB; fits in RAM at ~32 tok/s, shown for the curve):

| Budget | Resident | tok/s | Miss rate |
|---|---|---|---|
| 8 GiB | 11 GB | 7.2 | 8.3% |
| 12 GiB | 15 GB | 7.6 | 5.2% |
| 16 GiB | 19 GB | 15.9 | 3.6% |
| 16 GiB, warm-started | 19 GB | 19.0 | 1.5% |

Throughput tracks `t_compute + miss_bytes / ssd_bandwidth`.

## Warm-start

Capture a routing trace from text once, then preload each layer's most popular experts at startup:

```sh
python examples/capture_trace.py --model /path/to/model --out trace.npz --input src/*.py
python examples/serve.py --model /path/to/model --budget-gib 16 --warmstart-trace trace.npz
```

Keep `trace.meta.json` next to the `.npz`, both are needed. Only worth it when the pool nearly covers the model's experts (the last two Qwen rows above); on miss-dominated budgets it measures as a wash, since LRU reaches the same hot set within a few tokens anyway.

## Python API

```python
from streamlx.integrate import load_streaming_model, aggregate_stats
from mlx_lm import generate

model, tokenizer, pools, reader = load_streaming_model(
    "/path/to/model",
    budget_bytes=12 << 30,
    trust_remote_code=True,                        # custom tokenizer code
    tokenizer_config={"fix_mistral_regex": True},  # e.g. Laguna
)
print(generate(model, tokenizer, prompt="...", max_tokens=100))
print(aggregate_stats(pools))   # hits, misses, evictions, fetch time
```

## How it works

- The trunk loads via `mlx_lm.load(lazy=True)`; each `SwitchGLU` is swapped for a pool-backed wrapper before evaluation, so expert weights never materialize.
- Missing experts are `pread` straight from the safetensors byte ranges, span-merged into multi-MB reads for SSD throughput.
- Experts already resident start on the GPU (`async_eval`) while misses are fetched, so the SSD read overlaps compute instead of stalling it.
- Every step is exact: each expert's output depends only on the input and its own weights, so relocating experts into slots and regrouping rows cannot change a bit.
- Prefill is expert-major: when a chunk's routed set exceeds the pool, each needed expert streams through a throwaway buffer exactly once, in id order (long sequential reads), without touching the LRU pool. On Laguna this is 4.9x faster than slicing prefill to fit the pool (756-token prompt: 24 s vs 116 s). `STREAMLX_PREFILL=slice` restores the old path.

## Status

v0.1, APIs may change. oMLX server integration lives on a branch (`expert_stream` per-model setting); an mlx-lm RFC is in preparation.
