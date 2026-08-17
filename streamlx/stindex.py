"""Safetensors byte-range index: tensor name -> (shard, absolute offset,
dtype, shape) and per-expert row ranges for stacked [n_experts, ...] tensors.

Header-only (no tensor data is read). Row-major layout means expert e of a
stacked tensor is one contiguous byte range: base + e * stride.
"""

from __future__ import annotations

import glob
import json
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path

DTYPE_SIZE = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U16": 2, "U8": 1}
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")


@dataclass(frozen=True)
class TensorLoc:
    path: str
    abs_start: int   # absolute file offset of tensor data
    nbytes: int
    dtype: str
    shape: tuple
    name: str = ""   # tensor key in the checkpoint

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
        for shard in sorted(glob.glob(f"{self.model_dir}/model-*.safetensors")):
            with open(shard, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
            data_base = 8 + hlen
            for name, info in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = info["data_offsets"]
                self.tensors[name] = TensorLoc(
                    path=shard, abs_start=data_base + a, nbytes=b - a,
                    dtype=info["dtype"], shape=tuple(info["shape"]),
                    name=name)
        if not self.tensors:
            raise FileNotFoundError(f"no shards under {model_dir}")
        # Quantization table: bits/group_size come from here (mixed-precision
        # checkpoints override per path), shapes only cross-check. Keys are
        # normalized so mlx-vlm's "language_model." prefix matches either way.
        quant = {}
        cfg = Path(self.model_dir) / "config.json"
        if cfg.exists():
            quant = json.loads(cfg.read_text()).get("quantization") or {}
        self._quant_default = {k: v for k, v in quant.items()
                               if not isinstance(v, dict)}
        self._quant_by_path = {self._norm(k): v for k, v in quant.items()
                               if isinstance(v, dict)}

    @staticmethod
    def _norm(key: str) -> str:
        return key.removeprefix("language_model.")

    def switch_triples(self, layer: int) -> dict[str, dict[str, TensorLoc]]:
        """{proj: {weight/scales/biases: TensorLoc}} for one MoE layer."""
        out: dict[str, dict[str, TensorLoc]] = {}
        tag = f"layers.{layer}.mlp.switch_mlp."
        if not any(tag in name for name in self.tensors):
            # deepseek_v4 hangs the MoE block off `ffn`, not `mlp`
            tag = f"layers.{layer}.ffn.switch_mlp."
        for name, loc in self.tensors.items():
            if tag not in name:
                continue
            for proj in PROJS:
                for part in PARTS:
                    if name.endswith(f"{proj}.{part}"):
                        out.setdefault(proj, {})[part] = loc
        missing = [(p, q) for p in PROJS for q in PARTS
                   if q not in out.get(p, {})]
        if missing:
            raise KeyError(f"layer {layer}: missing switch tensors {missing}")
        return out

    def layer_expert_bytes(self, layer: int) -> int:
        """Bytes for ONE expert of one layer (all 3 projections, W+S+B)."""
        return sum(loc.row_stride
                   for parts in self.switch_triples(layer).values()
                   for loc in parts.values())

    def quant_params(self,
                     triple: dict[str, TensorLoc]) -> tuple[int, int, int, int]:
        """(in_features, out_features, bits, group_size) for one projection:
        bits/group_size from the config's quantization table (per-path
        override, else top-level, else shape inference at gs=64), verified
        against the packed weight/scales shapes."""
        w, s = triple["weight"], triple["scales"]
        n_groups, packed = s.shape[-1], w.shape[-1]
        ov = self._quant_by_path.get(self._norm(w.name)[: -len(".weight")], {})
        bits = ov.get("bits", self._quant_default.get("bits"))
        gs = ov.get("group_size", self._quant_default.get("group_size"))
        if gs is None:
            gs = 64
        in_features = n_groups * gs
        if bits is None:
            bits = packed * 32 // in_features
        if packed * 32 != in_features * bits or bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"quant mismatch for {w.name}: W{w.shape} "
                             f"S{s.shape} with bits={bits} group_size={gs}")
        return in_features, w.shape[1], bits, gs
