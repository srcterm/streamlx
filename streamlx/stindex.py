"""Safetensors byte-range index: tensor name -> (shard, absolute offset,
dtype, shape) and per-expert row ranges for stacked [n_experts, ...] tensors.

Header-only (no tensor data is read). Row-major layout means expert e of a
stacked tensor is one contiguous byte range: base + e * stride.
"""

from __future__ import annotations

import glob
import json
import re
import struct
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

DTYPE_SIZE = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U16": 2, "U8": 1,
              "F8_E4M3": 1, "F8_E5M2": 1}
PROJS = ("gate_proj", "up_proj", "down_proj")
# inkling2: gate and up in one projection, gate rows even and up rows odd
FUSED = ("gate_up_proj", "down_proj")
DS_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj",
           "w13": "gate_up_proj"}
# global_scale: nvfp4's tensor scale, one float per expert
PARTS = ("weight", "scales", "biases", "global_scale")
# a float format's (bits, group size, mode) by bits x group size
_FP_MODES = {64: (4, 16, "nvfp4"), 128: (4, 32, "mxfp4"), 256: (8, 32, "mxfp8")}


def default_group_size(mode: str) -> int:
    """mlx.nn's group size for a quantization mode that doesn't state one."""
    return {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}.get(mode, 64)


@dataclass(frozen=True)
class TensorLoc:
    path: str
    abs_start: int   # absolute file offset of tensor data
    dtype: str
    shape: tuple
    name: str = ""   # tensor key in the checkpoint
    # the values are computed at load from this tensor (the reader takes
    # them from the stock module, never from these bytes)
    derived: bool = False

    @property
    def row_stride(self) -> int:
        return prod(self.shape[1:]) * DTYPE_SIZE[self.dtype]

    def expert_range(self, e: int) -> tuple[str, int, int]:
        """(path, absolute offset, nbytes) of row e of a stacked tensor."""
        if not 0 <= e < self.shape[0]:
            raise IndexError(f"expert {e} out of range {self.shape[0]}")
        return self.path, self.abs_start + e * self.row_stride, self.row_stride


class SafetensorsIndex:
    def __init__(self, model_dir: str):
        self.model_dir = str(model_dir)
        self.tensors: dict[str, TensorLoc] = {}
        # the files stock mlx-lm loads: model-0000x-of-0000y.safetensors, or
        # a small checkpoint's single model.safetensors
        for shard in sorted(glob.glob(f"{self.model_dir}/model*.safetensors")):
            with open(shard, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
            data_base = 8 + hlen
            for name, info in hdr.items():
                if name == "__metadata__":
                    continue
                a = info["data_offsets"][0]
                self.tensors[name] = TensorLoc(
                    path=shard, abs_start=data_base + a,
                    dtype=info["dtype"], shape=tuple(info["shape"]),
                    name=name)
        if not self.tensors:
            raise FileNotFoundError(
                f"no model*.safetensors under {model_dir}")
        # Quantization table: bits/group_size come from here (mixed-precision
        # checkpoints override per path), shapes only cross-check. Keys are
        # normalized so mlx-vlm's "language_model." prefix matches either way.
        # An oQ DeepSeek-V4.1 export keeps its table under its own key.
        quant = {}
        cfg = Path(self.model_dir) / "config.json"
        if cfg.exists():
            raw = json.loads(cfg.read_text())
            quant = (raw.get("quantization")
                     or (raw.get("omlx_deepseek_v41") or {}).get("quantized_modules")
                     or {})
        self._quant_default = {k: v for k, v in quant.items()
                               if not isinstance(v, dict)}
        self._quant_by_path = {self._norm(k): v for k, v in quant.items()
                               if isinstance(v, dict)}

    @staticmethod
    def _norm(key: str) -> str:
        return key.removeprefix("language_model.")

    def switch_triples(self, layer: int,
                       tag: str | None = None) -> dict[str, dict[str, TensorLoc]]:
        """{proj: {weight/scales[/biases][/global_scale]: TensorLoc}} for
        one MoE layer, over PROJS or, for inkling2's layout, FUSED.

        `tag` overrides the module path for non-trunk MoE blocks (a draft
        head's, at `mtp.{s}` or `mtp.layers.{i}`). biases are optional (the
        float modes ship weight and scales only); weight and scales are
        required for every projection.
        """
        out: dict[str, dict[str, TensorLoc]] = {}
        base = tag if tag is not None else f"layers.{layer}"
        # Anchored at the start of the normalized name: a substring match
        # also hits an MTP head that repeats the trunk's numbering
        # (`mtp.layers.0.mlp.switch_mlp.*`). `_norm` strips `language_model.`;
        # an optional `model.` may follow it. deepseek_v4 hangs the MoE
        # block off `ffn`, not `mlp`.
        anchored = re.compile(r"^(?:model\.)?" + re.escape(base)
                              + r"\.(?:mlp|ffn)\.switch_mlp\.")
        # DeepSeek's own names for the stacked experts (V4.1 oQ exports):
        # `ffn.experts.w1/w3/w2` = gate / up / down; an oQ inkling2 export
        # has `mlp.experts.w13/w2` = gate+up / down
        ds_names = re.compile(r"^(?:model\.)?" + re.escape(base)
                              + r"\.(?:mlp|ffn)\.experts\.(w13|w[123])\."
                              + r"(weight|scales|biases|global_scale)$")
        # mlx-lm's inkling2: `mlp.experts.gate_up_proj / down_proj`
        fused = re.compile(r"^(?:model\.)?(?:language_model\.)?"
                           + re.escape(base) + r"\.mlp\.experts\."
                           + r"(gate_up_proj|down_proj)\."
                           + r"(weight|scales|biases|global_scale)$")
        # Inkling 2's source checkpoints (modelopt): `mlp.experts.w13_weight`
        # / `w2_weight` and their `_scale`, bytes mlx-lm runs as they are
        # (packed rows seen as uint32, fp8 scales as bytes); the nvfp4 tensor
        # scale is computed at load from `_amax`
        src = re.compile(r"^(?:model\.)?(?:llm\.|language_model\.)?"
                         + re.escape(base)
                         + r"\.mlp\.experts\.(w13|w2)_weight(_scale|_amax)?$")
        for name, loc in self.tensors.items():
            norm = self._norm(name)
            if (m := src.match(norm)) is not None:
                proj, suffix = DS_PROJ[m.group(1)], m.group(2)
                if suffix is None:
                    n = loc.shape[-1] * DTYPE_SIZE[loc.dtype]
                    part, loc = "weight", replace(
                        loc, dtype="U32", shape=loc.shape[:-1] + (n // 4,))
                elif suffix == "_scale":
                    part, loc = "scales", replace(loc, dtype="U8")
                else:
                    part, loc = "global_scale", replace(loc, derived=True)
                out.setdefault(proj, {})[part] = loc
                continue
            ds = ds_names.match(norm) or fused.match(norm)
            if ds is not None:
                proj = DS_PROJ.get(ds.group(1), ds.group(1))
                out.setdefault(proj, {})[ds.group(2)] = loc
                continue
            if anchored.match(norm) is None:
                continue
            for proj in PROJS:
                for part in PARTS:
                    if name.endswith(f"{proj}.{part}"):
                        out.setdefault(proj, {})[part] = loc
        projs = FUSED if "gate_up_proj" in out else PROJS
        missing = [(p, q) for p in projs for q in ("weight", "scales")
                   if q not in out.get(p, {})]
        if missing:
            raise KeyError(f"layer {base}: missing switch tensors {missing}")
        return {p: out[p] for p in projs}

    def layer_expert_bytes(self, layer: int, tag: str | None = None) -> int:
        """Bytes for ONE expert of one layer (all projections+parts)."""
        return sum(loc.row_stride
                   for parts in self.switch_triples(layer, tag).values()
                   for loc in parts.values())

    def quant_params(
            self,
            triple: dict[str, TensorLoc]) -> tuple[int, int, int, int, str]:
        """(in_features, out_features, bits, group_size, mode) for one
        projection: bits/group_size/mode from the config's quantization
        table (per-path override, else top-level, else shape inference at
        the mode's group size, affine by default), verified against the
        packed weight/scales shapes. Byte scales under an affine entry are a
        float format the table leaves out (mlx-lm's inkling2 keeps the
        source's nvfp4 / mxfp8 experts): read off the shapes."""
        w, s = triple["weight"], triple["scales"]
        n_groups, packed = s.shape[-1], w.shape[-1]
        ov = self._quant_by_path.get(self._norm(w.name)[: -len(".weight")], {})
        bits = ov.get("bits", self._quant_default.get("bits"))
        gs = ov.get("group_size", self._quant_default.get("group_size"))
        mode = ov.get("mode", self._quant_default.get("mode", "affine"))
        if (s.dtype == "U8" and mode == "affine"
                and packed * 32 // n_groups in _FP_MODES):
            bits, gs, mode = _FP_MODES[packed * 32 // n_groups]
        if gs is None:
            gs = default_group_size(mode)
        in_features = n_groups * gs
        if bits is None:
            bits = packed * 32 // in_features
        if packed * 32 != in_features * bits or bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"quant mismatch for {w.name}: W{w.shape} "
                             f"S{s.shape} with bits={bits} group_size={gs}")
        return in_features, w.shape[1], bits, gs, mode
