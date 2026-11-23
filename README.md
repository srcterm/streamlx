# streamlx

**A lightweight SSD-streaming engine for mixture-of-experts models on Apple Silicon.** It serves MoE checkpoints several times larger than RAM to a coding agent, with the stock model's output.

A 343 GB DeepSeek-V4.1 runs on a 32 GB MacBook Air. The dense part of the model stays in RAM; the routed experts stay on the SSD and are read when a token needs them.

## Quick start

```sh
pip install git+https://github.com/srcterm/streamlx

streamlx serve /path/to/model
```

That is an OpenAI-compatible server on `http://localhost:8080/v1` for a coding agent or any client. The model is a local MLX checkpoint directory; nothing else needs setting. Models think before they answer: `--chat-template-args '{"enable_thinking": false}'` turns it off.

`streamlx generate /path/to/model --prompt "..."` runs one prompt and prints speed and memory figures.

## Features

![How streamlx serves a request](docs/pipeline.svg)

- **Pipelined expert fetching.** A checkpoint's MTP draft head puts several tokens in flight one layer apart, so one token's expert reads hide behind the others' compute. Greedy output is unchanged, and a gate keeps plain decoding if that is faster ([paper](docs/pipelined-expert-fetching.pdf)).
- **Lookup drafts.** Where a reply repeats text from the conversation, the tokens that followed it there are drafted and checked like a draft head's. Every model gets them, with or without a head ([below](#lookup-drafts)).
- **Reads overlap compute.** A missed expert is read straight into its pool slot by parallel threads while the experts already in RAM run. Prefill reads each layer's experts once per chunk, the next layer's while this one computes.
- **Prompt cache on the SSD.** Prompts are stored as segments shared across turns and conversations; a request restores its longest stored prefix and prefills only the rest.
- **Memory limit, on by default.** The pools shrink while a prompt chunk is prefilled and grow back for decoding, keeping MLX inside the GPU's working set.
- **Stock models and checkpoints.** Only the expert layer is swapped; checkpoints are read in place and the server is `mlx_lm.server`. MoE models with stacked `switch_mlp` experts run as they are; pipelined decoding needs a small adapter per architecture.
- **Exact.** Output matches the stock model token for token; `streamlx validate` checks it on a model that fits in RAM. (Drafts checked in one forward are the exception: see Limits.)
- **Small.** About 8,000 lines of Python on `mlx`, `mlx-lm` and `mlx-vlm`, with no custom kernels.

## Serving numbers

M4 MacBook Air, 32 GB. `streamlx serve <model>` with its defaults and thinking off, driven by a coding agent (pi): a cold first prompt of about 1,600 tokens, two or three requests per task. Checkpoints on a USB4 SSD (~3 GB/s peak), Qwen3.6 on the internal disk.

| Model | Checkpoint | Pools | Prefill | Decode | Peak memory |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B, 3-bit | 14 GB | 13.1 GiB (all of it) | 406 tok/s | 54–56 tok/s | 14.6 GiB |
| Inkling-2-Mini, oQ4e | 16 GB | 12.8 GiB (all of it) | 194 tok/s | 18–26 tok/s | 16.4 GiB |
| Qwen3.8-Flash-Next, oQ4e | 99 GB | 16.4–17.8 GiB | 51 tok/s | 5.5–7.1 tok/s | 23.7 GiB |
| DeepSeek-V4.1-Flash, oQ2.5e | 343 GB | 9.8 GiB | 15 tok/s | 1.7–1.8 tok/s | 23.4 GiB |
| Inkling-2-Small, oQ2e | 181 GB | 13.3–14.2 GiB | 28 tok/s | 1.5–1.7 tok/s | 23.3 GiB |

**NOTE:** A larger-than-RAM model is bound by the drive: more memory or a faster drive means more speed. A model that fits is read into the pools at load and runs at mlx-lm's speed.

## Lookup drafts

A coding agent's replies often repeat text that is already in the conversation: a file written out again with an edit, a function quoted back, a tool's output. streamlx indexes every three-token sequence of the prompt and of the reply so far. When the reply's last three tokens occurred earlier, and what followed them there has matched the next eight tokens the model wrote, the tokens after that are drafted from the earlier text. They are checked in one step like a draft head's, and any that the model would not have written are rolled back.

On ordinary text a copy rarely holds for eight tokens, so little is drafted and decoding keeps its usual speed. A model in memory checks up to 16 copied tokens in one forward; a streamed model checks as many as its draft head puts in flight. `STREAMLX_LOOKUP=0` turns lookup drafts off.

Decode speed on a reply that writes a file from the prompt out again with one rename:

| Model | Plain | The draft head | Lookup drafts |
|---|---|---|---|
| Qwen3.6-35B-A3B, 3-bit (in memory, no head) | 55 | | 99 |
| Inkling-2-Mini, oQ4e (in memory) | 18.5 | 26 | 45 |
| DeepSeek-V4.1-Flash, oQ2.5e | 1.1 | 1.5 | 1.8 |
| Inkling-2-Small, oQ2e | 1.05 | 0.9 | 1.35 |

## Options

`streamlx serve` takes every `mlx_lm.server` flag (`--port`, `--host`, `--temp`, `--chat-template-args`, ...) plus:

| Flag | Default | |
|---|---|---|
| `--memory-gib` | 95% of the recommended GPU working set | MLX's memory limit, kept by resizing the pools; `0` turns it off |
| `--budget-gib` | what the memory limit leaves | the most RAM the expert pools take |
| `--draft-head`, `--draft-k` | the checkpoint's own head | pipelined decoding and its tokens in flight; `none` decodes plainly, without lookup drafts too |
| `--kv-cache-dir`, `--kv-cache-gib` | `~/.cache/streamlx/kv`, half of the free space and what the caches hold, up to 32 | prompt caches on the SSD and their disk limit, all models together |
| `--mlx-cache-gib` | 2 | cap for MLX's freed-buffer cache |
| `--prefill-step-size` | 4096 | prefill chunk; larger chunks read fewer bytes per prompt |
| `--tokenizer-config`, `--trust-remote-code` | | passed to the tokenizer |

Requests run one at a time. `STREAMLX_KEEPALIVE=0` turns off the small GPU jobs that keep the GPU at speed while a decode waits on reads; `STREAMLX_LOOKUP=0` turns off lookup drafts.

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
| `streamlx/pipeline.py`, `pipeline_adapters.py` | pipelined decoding and lookup drafts; the per-model adapters and draft heads |
| `streamlx/kvcache.py`, `attention.py` | the SSD prompt cache; bounded prefill attention |
| `streamlx/chat_formats.py` | prompt formats and reply framing mlx-lm does not get from the checkpoint |
| `streamlx/qwen4_exp.py`, `deepseek_v41*.py`, `inkling2*.py` | Qwen3.8, DeepSeek-V4.1 and Inkling 2 |
| `streamlx/serve.py`, `generate.py`, `validate.py`, `cli.py` | the `streamlx` command |
| `tests/` | toy-checkpoint tests, bit for bit against mlx-lm's resident model: `python -m unittest discover -s tests` |

## Limits

- Apple Silicon with macOS 14 or later (MLX's requirement), one request at a time, local checkpoint directories.
- Pipelined decoding covers greedy requests; requests that sample or ask for logprobs decode plainly.
- On models with attention head size 256 (Qwen3.5 / 3.6), long prefills use MLX's fused attention kernel, which differs from the stock path in the last bits.
- Drafts checked in one forward (Inkling 2 Mini's draft head; lookup drafts on any model in memory) go through other kernels than a forward of one token, so where two tokens are about tied the text can take the other one. `--draft-head none` decodes plainly.

## Notes

**Inkling 2** needs mlx-lm with its `inkling2` model, which is still an open pull request ([mlx-lm #2061](https://github.com/ml-explore/mlx-lm/pull/2061)). Install mlx-lm from it, then serve as usual:

```sh
pip install git+https://github.com/ml-explore/mlx-lm@refs/pull/2061/head
streamlx serve /path/to/model
```

Its chat template takes `reasoning_effort` instead of `enable_thinking`: `--chat-template-args '{"reasoning_effort": "none"}'`. At `none` it still writes a short reasoning block.

**Inkling 2 Mini** is a looped model (twelve of its layers run three times a token) and needs another open pull request, [mlx-lm #2086](https://github.com/ml-explore/mlx-lm/pull/2086), which carries the model above as well: `pip install git+https://github.com/ml-explore/mlx-lm@refs/pull/2086/head`. Its 16 GB 4-bit export fits a 32 GB Mac whole, with room for a long conversation. Served with the defaults, with a document of Python source in the prompt:

| Context | Prefill (new tokens) | Decode | Writing a function from the prompt out again | Next turn's first token | Peak memory |
|---|---|---|---|---|---|
| 4k | 163 tok/s | 18 tok/s | 24 tok/s | 0.5 s | 17.7 GiB |
| 16k | 188 tok/s | 17 tok/s | 19 tok/s | 0.7 s | 18.9 GiB |
| 32k | 161 tok/s | 14 tok/s | 31 tok/s | 0.9 s | 20.2 GiB |
| 64k | 119 tok/s | 11 tok/s | 21 tok/s | 1.2 s | 21.8 GiB |
| 128k | 80 tok/s | 7 tok/s | 10 tok/s | 2.7 s | 23.7 GiB |

Decode is a reply that repeats nothing from the prompt; one that quotes it gets lookup drafts. The next turn restores the conversation from the SSD prompt cache, which holds 15 GB at 128k tokens.

v0.2; interfaces may change.
