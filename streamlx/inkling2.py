"""Inkling 2 through mlx-lm's inkling2: the loader for an oQ export.

mlx-lm builds the model and loads its own layout as it is (the source
checkpoint, mlx-community's conversions): those need nothing from here. An oQ
export stores the same weights under oMLX's runtime layout, which
`load_inkling2` maps onto mlx-lm's parameter tree without requantizing:
  - renames (attn -> self_attn, mlp.router_weight -> mlp.gate.weight, ...);
  - the attention projections oQ stacks row-wise into one quantized matrix,
    split back by rows (a row keeps its own scales and biases, so a slice
    is exact):
      sliding layers  attn.in_proj = q | k | v | r | gate | sink
      full layers     attn.q_proj  = q | r | gate | sink
                      attn.k_proj  = k | k_sconv.weight_proj   (v likewise)
  - conv weights and attention-residual queries reshaped.
The routed experts (w13 = gate and up in one projection, w2 = down) keep
their bytes. mlx-lm reads every module's bits, group size and mode off its
packed shapes, so the export's quantization table needs no translation.
Text only: the vision tower is not built and its tensors are dropped.
"""
from __future__ import annotations

import re

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_ROOT = "model.language_model."
_TOP = {"model.embed.": _ROOT + "embed_tokens.",
        "model.embed_norm.": _ROOT + "embed_norm.",
        "model.norm.": _ROOT + "norm.",
        "lm_head.": "lm_head."}
# within a layer: tensors that only change their name
_RENAMES = {"attn.o_proj.": "self_attn.o_proj.",
            "attn.q_scale": "self_attn.q_l2_clip.scale",
            "attn.r_scale": "self_attn.rel_l2_clip.scale",
            "attn.rel_proj": "self_attn.rel_logits_proj.proj",
            "attn.k_static": "self_attn.k_sconv.static_weight",
            "attn.v_static": "self_attn.v_sconv.static_weight",
            "attn_norm.": "input_layernorm.",
            "q_input_norm.": "q_input_layernorm.",
            "k_input_norm.": "k_input_layernorm.",
            "v_input_norm.": "v_input_layernorm.",
            "mlp_norm.": "post_attention_layernorm.",
            "mlp.router_weight": "mlp.gate.weight",
            "mlp.router_bias": "mlp.gate.e_score_correction_bias",
            "mlp.experts.w13.": "mlp.experts.gate_up_proj.",
            "mlp.experts.w2.": "mlp.experts.down_proj."}
# [dims, kernel] -> mlx's conv1d weight [dims, kernel, 1]
_CONVS = {"attn.k_conv": "self_attn.k_sconv.conv1d.weight",
          "attn.v_conv": "self_attn.v_sconv.conv1d.weight",
          "attn_conv": "post_attention_convolution.conv1d.weight",
          "mlp_conv": "post_mlp_convolution.conv1d.weight"}
# [n, dims] -> [dims] for one query, as it is for a full layer's three
_QUERIES = {"attn_input.query": "attention_input.query",
            "mlp_input.query": "mlp_input.query"}


def _splits(text: dict, sliding: bool) -> dict:
    """{oQ projection: [(mlx-lm module under self_attn, rows)]} of a layer."""
    pre = "swa_" if sliding else ""
    heads = text[pre + "num_attention_heads"]
    kv = text[pre + "num_key_value_heads"]
    q, kvd = heads * text[pre + "head_dim"], kv * text[pre + "head_dim"]
    r = heads * text["d_rel"]
    if sliding:
        return {"in_proj": [("q_proj", q), ("k_proj", kvd), ("v_proj", kvd),
                            ("r_proj", r), ("gate_proj", q),
                            ("sink_proj", heads)]}
    conv = kv * text["sconv_kernel_size"]
    return {"q_proj": [("q_proj", q), ("r_proj", r), ("gate_proj", q),
                       ("sink_proj", heads)],
            "k_proj": [("k_proj", kvd), ("k_sconv.weight_proj", conv)],
            "v_proj": [("v_proj", kvd), ("v_sconv.weight_proj", conv)]}


def from_oq_export(weights: dict, text: dict) -> dict:
    """An oQ export's tensors under mlx-lm's names (text model only). `text`
    is the checkpoint's text_config."""
    local = set(text["local_layer_ids"])
    out = {}
    for key, v in weights.items():
        if key.startswith("vision_tower."):
            continue
        m = _LAYER.match(key)
        if m is None:
            if key == "model.final_input.query":
                out[_ROOT + "final_input.query"] = v.reshape(-1)
                continue
            new = next((n + key[len(o):] for o, n in _TOP.items()
                        if key.startswith(o)), None)
            if new is None:
                raise ValueError(f"unmapped Inkling 2 tensor {key}")
            out[new] = v
            continue
        n, rest = int(m.group(1)), m.group(2)
        pre = f"{_ROOT}layers.{n}."
        proj, _, part = rest.removeprefix("attn.").rpartition(".")
        rows_of = _splits(text, n in local).get(proj)
        if rest.startswith("attn.") and rows_of is not None:
            a = 0
            for name, rows in rows_of:
                out[f"{pre}self_attn.{name}.{part}"] = v[a:a + rows]
                a += rows
            if a != v.shape[0]:
                raise ValueError(f"{key}: {v.shape[0]} rows, expected {a}")
        elif rest in _CONVS:
            out[pre + _CONVS[rest]] = v[..., None]
        elif rest in _QUERIES:
            out[pre + _QUERIES[rest]] = v.reshape(-1) if v.shape[0] == 1 else v
        else:
            new = next((n_ + rest[len(o):] for o, n_ in _RENAMES.items()
                        if rest.startswith(o)), None)
            if new is None:
                raise ValueError(f"unmapped Inkling 2 tensor {key}")
            out[pre + new] = v
    return out


def is_oq_export(cfg: dict) -> bool:
    """Whether an inkling2 checkpoint is in oQ's layout (its quantization
    table names oQ's modules)."""
    return (cfg.get("model_type") == "inkling2"
            and "model.embed" in (cfg.get("quantization") or {}))


def load_inkling2(model_dir, tokenizer_config=None, model_config=None):
    """load_streaming_model's loader for an oQ export of Inkling 2."""
    import json
    from pathlib import Path

    import mlx_lm.models.inkling2 as ink
    from mlx_lm import load

    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    text = cfg["text_config"]
    sanitize = ink.Model.sanitize

    def oq_sanitize(self, weights):
        return sanitize(self, from_oq_export(weights, text))

    ink.Model.sanitize = oq_sanitize
    try:
        return load(str(model_dir), lazy=True,
                    tokenizer_config=tokenizer_config or {},
                    model_config={"vision_config": None,
                                  **(model_config or {})})
    finally:
        ink.Model.sanitize = sanitize
