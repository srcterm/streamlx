# streamlx

**Run MoE models larger than RAM on Apple Silicon by streaming expert weights from SSD.**

streamlx keeps a quantized MLX checkpoint's trunk resident and leaves the routed experts on disk: each MoE layer gets a fixed-budget LRU pool of expert slots, and misses are read on demand from the safetensors shards while the GPU computes with the experts already loaded. Output is bitwise identical to the stock model and fetch/compute overlapping makes processing fast.

## How it works

- The trunk loads via `mlx_lm.load(lazy=True)`; each `SwitchGLU` is swapped for a pool-backed wrapper before evaluation, so expert weights never load.
- Missing experts are `pread` straight from the safetensors byte ranges, which are span-merged into multi-MB reads for SSD throughput.
- Experts already resident start on the GPU (`async_eval`) while misses are fetched, so the SSD read overlaps compute, i.e. fetch/compute overlapping, speeding up processing.
- During decode each layer also predicts the *next* layer's routing from the current residual (~70% accurate) and starts reading the predicted missing experts while attention computes, so most misses are absorbed before they can stall. On by default.
- Every step is exact: each expert's output depends only on the input and its own weights, so relocating experts into slots and regrouping rows cannot change a bit.
- Prefill is expert-major: when a chunk's routed set exceeds the pool, each needed expert streams through a throwaway buffer exactly once, in id order, which is ˜5x faster than slicing prefill to fit the pool and the chunk's hottest experts are adopted into the pool, so decode starts warm.

## Install

Apple Silicon, Python 3.10+, `mlx-lm` from git main (`pip install git+https://github.com/ml-explore/mlx-lm` — the PyPI release predates `laguna`/`qwen3_5_moe` support). Not on PyPI yet.

```sh
git clone https://github.com/srcterm/streamlx
pip install -e streamlx
```

## Serve

OpenAI-compatible; point any client or coding agent at it:

```sh
python examples/serve.py --model /path/to/model --budget-gib 12 --port 8080
```

To serve Laguna S-2.1 on 32GB ram for instance (also see config flags further below):

```sh
python examples/serve.py --model /path/to/Laguna-S-2.1-oQ2e-fast --budget-gib 18 --trust-remote-code --tokenizer-config fix_mistral_regex=true --port 8080
```

Models must be local directories in MLX format with the standard stacked `switch_mlp` expert layout. Validated on `laguna`, `kimi_linear`, and `qwen3_5_moe` (Qwen3.6); other mlx-lm MoE models should work unmodified. mlx-vlm text-only exports (tensors prefixed `language_model.`) are detected and remapped automatically.

| Flag | Default | |
|---|---|---|
| `--budget-gib` | 8.0 | RAM reserved for experts. Set it as high as fits: about 16-20 on a 32 GB Mac depending on required KV cache |
| `--trust-remote-code` | off | Needed for models with custom tokenizer code. |
| `--tokenizer-config` | none | Extra tokenizer kwargs as `key=json`, repeatable. Laguna's Mistral-derived tokenizer needs `fix_mistral_regex=true`. |
| `--resident` | off | Let fully-covered MoE layers stay stock-resident (no sync, no pool). Faster when the model nearly fits in RAM; costs KV-cache headroom. |
| `--warmstart-trace` | none | Preload popular experts from a routing trace. See below. |
| `--prompt-cache-gib` | auto | Byte cap for the server's prompt-cache LRU (default `RAM − budget − 12 GiB`, clamped to 1–8). |
| `--mlx-cache-gib` | 2.0 | Cap for MLX's freed-buffer cache (its own default hoards ~95% of RAM). |
| `--prefill-step-size` | 2048 | Prefill chunk. Expert-sweep bytes scale ~1/chunk (the mlx-lm default of 512 reads 3.7× more per prompt at identical speed). |

All `mlx_lm.server` flags pass through (`--port`, `--host`, `--temp`, ...). Requests run one at a time by design (batching collapses the expert hit rate); `--draft-model` and `--adapter-path` are unsupported.

Tuning env vars: `STREAMLX_PREFETCH=0` disables the routing-lookahead prefetch, `STREAMLX_PF_TOPM` (default 7) caps how many predicted experts are read per layer, `STREAMLX_EVICT=s3fifo` switches the pool eviction policy (default LRU).

## Performance

**M4 MacBook Air, 32 GB**, macOS 26.3, ~2.4 GB/s SSD, batch-1 greedy decode. Resident memory is about `trunk + budget + 1-2 GB`. On tested models:

**>> Laguna-S-2.1** (Poolside, 117B, top-10-of-256 routing; [oQ2e-fast](https://huggingface.co/mlx-community/Laguna-S-2.1-oQ2e-fast) community imatrix quant, 2-bit experts, 34.7 GB). Stock loading dies with swap on this machine; with streaming though:

| Budget | Resident | tok/s | Miss rate |
|---|---|---|---|
| 8 GiB | 12 GB | 3.4 | 25.9% |
| 12 GiB | 16 GB | 4.5 | 14.7% |
| 16 GiB | 20 GB | 7.2 | 7.6% |
| 18 GiB | 22 GB | 9.2 | 6.2% |

**>> Qwen3.6-35B-A3B** (4-bit, 22.1 GB; fits in RAM at ~32 tok/s, shown for the curve):

| Budget | Resident | tok/s | Miss rate |
|---|---|---|---|
| 8 GiB | 11 GB | 7.2 | 8.3% |
| 12 GiB | 15 GB | 7.6 | 5.2% |
| 16 GiB | 19 GB | 15.9 | 3.6% |
| 16 GiB, warm-started | 19 GB | 19.0 | 1.5% |

With `--resident` at 20 GiB (all 40 MoE layers fit) decode reaches 33.3 tok/s — stock speed, with experts loaded once at startup.

Throughput tracks `t_compute + miss_bytes / ssd_bandwidth`.

## Warm-start

Capture a routing trace from text once, then preload each layer's most popular experts at startup:

```sh
python examples/capture_trace.py --model /path/to/model --out trace.npz --input src/*.py
python examples/serve.py --model /path/to/model --budget-gib 16 --warmstart-trace trace.npz
```

Keep `trace.meta.json` next to the `.npz`. Only worth it when the pool nearly covers the model's experts (the last two Qwen rows); at miss-dominated budgets LRU reaches the same hot set within a few tokens anyway.

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

## Status

v0.1, APIs may change. oMLX integration: [srcterm/omlx `expert-streaming`](https://github.com/srcterm/omlx/tree/expert-streaming) (per-model `expert_stream` setting). Upstream: RFC in preparation for [mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438) (MoE expert streaming / SSD offload).
