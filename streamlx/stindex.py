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
                    dtype=info["dtype"], shape=tuple(info["shape"]))
        if not self.tensors:
            raise FileNotFoundError(f"no shards under {model_dir}")

    def switch_triples(self, layer: int) -> dict[str, dict[str, TensorLoc]]:
        """{proj: {weight/scales/biases: TensorLoc}} for one MoE layer."""
        out: dict[str, dict[str, TensorLoc]] = {}
        tag = f"layers.{layer}.mlp.switch_mlp."
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

    @staticmethod
    def quant_params(triple: dict[str, TensorLoc],
                     group_size: int = 64) -> tuple[int, int, int, int]:
        """(in_features, out_features, bits, group_size) from shapes alone."""
        w, s = triple["weight"], triple["scales"]
        n_groups = s.shape[-1]
        in_features = n_groups * group_size
        packed = w.shape[-1]
        bits = packed * 32 // in_features
        if packed * 32 % in_features or bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"cannot infer bits: W{w.shape} S{s.shape}")
        return in_features, w.shape[1], bits, group_size
