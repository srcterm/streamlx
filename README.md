# streamlx

**A lightweight SSD-streaming engine for mixture-of-experts models on Apple Silicon.** It serves MoE checkpoints several times larger than RAM to a coding agent, with the stock model's output.

A 343 GB DeepSeek-V4.1 runs on a 32 GB MacBook Air. The dense part of the model stays in RAM; the routed experts stay on the SSD and are read when a token needs them, into a fixed memory budget.

## Quick start

```sh
pip install git+https://github.com/srcterm/streamlx

streamlx serve /path/to/model
```

That is an OpenAI-compatible server on `http://localhost:8080/v1`: point a coding agent or any client at it. The model is a local MLX checkpoint directory. Nothing else needs setting:

- MLX's memory is kept under 95% of what the system recommends for the GPU, and the expert pools take what that leaves beside the model's dense part.
- A checkpoint that carries a draft head (Qwen3.8, DeepSeek-V4.1, Inkling 2) decodes pipelined with it; any other decodes plainly.
- Prompt caches are kept on the SSD, under `~/.cache/streamlx/kv`.
- Models think before they answer. `--chat-template-args '{"enable_thinking": false}'` turns it off (Inkling 2: `'{"reasoning_effort": "none"}'`).

`streamlx generate /path/to/model --prompt "..."` runs one prompt and prints speed and memory figures.

## What makes it fast

- **Reads land where they are used.** A missed expert is read from its byte range in the checkpoint's shards straight into its pool slot, by parallel reader threads. Nothing is deserialized, converted or copied on the way.
- **The drive and the GPU work at the same time.** Experts already in the pool start on the GPU while the missing ones are read. In prefill, the next layer's experts are read while this layer computes. In decode, a draft head puts several tokens in flight one layer apart, so one token's reads hide behind the others' compute.
- **A draft head is used for IO, not for batching.** Checkpoints with a multi-token head normally verify a batch of drafted positions at once, which reads the same experts with nothing to overlap them. streamlx staggers the drafted tokens through the layers instead (pipelined expert fetching). A draft is kept only if it equals the model's own next token, a wrong one is rolled back, and a gate keeps plain decoding whenever that is faster.
- **Prefill goes by expert, not by token.** A prompt chunk routes to nearly every expert, so each layer's experts pass once through a small buffer instead of thrashing the pool, and the most used ones stay in it for decoding.
- **The prompt cache lives on the SSD.** A coding agent resends a long, mostly unchanged prompt every turn. Prompt caches are stored on disk as segments shared between turns and conversations, a request restores the longest prefix it has in common with anything stored, and RAM holds only the active request's cache.
- **It manages the machine, not only the model.** A memory limit, on by default, shrinks the pools while a long prompt chunk is prefilled and grows them back for decoding, and trades slots for context as a conversation grows: past the GPU's working set results go wrong, not just slow. Long prefills run attention without building full score matrices. Small GPU jobs keep the GPU from dropping to a lower performance state while a decode waits on reads or shares the machine with other work.

## What makes it lightweight

- **About 6,700 lines of Python** on `mlx`, `mlx-lm` and `mlx-vlm`. No custom kernels, no fork of MLX.
- **Stock checkpoints, read in place.** A header-only index finds every expert's bytes in the safetensors shards; nothing is repacked or converted.
- **The model code stays upstream's.** Only the expert layer is replaced, behind the same call signature, so routing, attention and everything else run as the stock model does. An MoE model with the standard stacked expert layout needs no code here; pipelined decoding needs one small adapter per architecture.
- **The server is `mlx_lm.server`** with the model loader swapped: its flags, endpoints and clients work unchanged.
- **Exact.** Experts run through the stock kernels from relocated rows, and pipelined greedy decoding emits the tokens plain decoding emits. `streamlx validate` checks this token for token on a model that fits in RAM. One exception: on models with attention head size 256 (Qwen3.5 / 3.6), long prefills use MLX's fused attention kernel, which differs from the stock path in the last bits.

## Models

| Model | Loaded through | Draft head |
|---|---|---|
| mlx-lm MoE models with stacked `switch_mlp` experts (tested: Qwen3.6) | mlx-lm | none: plain decoding |
| Qwen3.8-Flash-Next | mlx-vlm | in the checkpoint, 3 tokens in flight |
| DeepSeek-V4.1-Flash | mlx-vlm | in the checkpoint, 6 tokens in flight |
| Inkling 2 | mlx-lm with its `inkling2` model (mlx-lm PR #2061) | in the checkpoint, 3 tokens in flight |

mlx-vlm text-only exports (tensors prefixed `language_model.`) are detected and remapped. DeepSeek-V4.1 checkpoints ship no chat template; streamlx brings its prompt format and tool-call parser.

## Serving numbers

M4 MacBook Air, 32 GB. `streamlx serve <model>` with its defaults and thinking off, driven by a coding agent (pi): its first request is a cold prompt of about 1,600 tokens, and the task takes two or three requests. Checkpoints on a USB4 SSD, except Qwen3.6 on the internal one.

| Model | Checkpoint | Pools | Prefill | Decode | Peak memory |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B, 3-bit | 14 GB | 13.1 GiB (all of it) | 118 tok/s | 30–33 tok/s | 14.6 GiB |
| Qwen3.8-Flash-Next, oQ4e | 99 GB | 16.4–17.8 GiB | 51 tok/s | 5.5–6.2 tok/s | 23.7 GiB |
| DeepSeek-V4.1-Flash, oQ2.5e | 343 GB | 9.8 GiB | 15 tok/s | 1.7–1.8 tok/s | 23.4 GiB |
| Inkling-2-Small, oQ2e | 181 GB | 13.3–14.2 GiB | 28 tok/s | 1.4–1.7 tok/s | 23.3 GiB |

A larger-than-RAM model is bound by the drive: tokens per second is about the drive's read rate over the bytes a token misses, so more memory or a faster drive is more speed. A later turn restores its prompt from the SSD cache and prefills only what is new.

## Options

`streamlx serve` takes every `mlx_lm.server` flag (`--port`, `--host`, `--temp`, `--chat-template-args`, ...) plus:

| Flag | Default | |
|---|---|---|
| `--memory-gib` | 95% of the recommended GPU working set | keep MLX's memory under this by resizing the pools: for each prompt chunk, then for decoding; `0` turns it off |
| `--budget-gib` | what the memory limit leaves | the most RAM the expert pools take |
| `--draft-head`, `--draft-k` | the checkpoint's own head | pipelined decoding of greedy requests, tokens in flight; `none` for plain decoding |
| `--kv-cache-dir`, `--kv-cache-gib` | `~/.cache/streamlx/kv`, half the free space up to 32 | prompt caches on SSD and their disk limit |
| `--mlx-cache-gib` | 2 | cap for MLX's freed-buffer cache |
| `--prefill-step-size` | 4096 | prefill chunk; larger chunks read fewer bytes per prompt |
| `--tokenizer-config`, `--trust-remote-code` | | passed to the tokenizer |

Requests run one at a time. `STREAMLX_KEEPALIVE=0` turns the GPU pacing off.

`streamlx validate /path/to/model` checks streaming against the model held in RAM, token for token (for a model that fits).

## Python API

```python
from mlx_lm import generate
from streamlx.integrate import aggregate_stats, load_streaming_model

model, tokenizer, pools, reader = load_streaming_model("/path/to/model", 12 << 30)
print(generate(model, tokenizer, prompt="...", max_tokens=100))
print(aggregate_stats(pools))      # hits, misses, fetch time
```

## Layout

| | |
|---|---|
| `streamlx/stindex.py`, `reader.py` | byte ranges of every expert in the shards; parallel reads of them |
| `streamlx/pool.py`, `integrate.py` | the expert pools; the streaming expert layer, loaders, the prefill sweep |
| `streamlx/pipeline.py`, `pipeline_adapters.py` | pipelined decoding; the per-model adapters and draft heads |
| `streamlx/kvcache.py`, `attention.py` | the SSD prompt cache; bounded prefill attention |
| `streamlx/chat_formats.py` | prompt formats and reply framing mlx-lm does not get from the checkpoint |
| `streamlx/qwen4_exp.py`, `deepseek_v41*.py`, `inkling2*.py` | Qwen3.8, DeepSeek-V4.1 and Inkling 2 |
| `streamlx/serve.py`, `generate.py`, `validate.py`, `cli.py` | the `streamlx` command |

## Limits

- Apple Silicon, one request at a time, local checkpoint directories.
- Pipelined decoding covers greedy requests; requests that sample or ask for logprobs decode plainly.

v0.1; interfaces may change.
