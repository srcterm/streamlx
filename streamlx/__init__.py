"""streamlx — SSD expert streaming for MoE models on mlx-lm and mlx-vlm.

Batch-1 inference with a per-layer expert pool (LRU) and a pread-based miss
path; no custom kernels.
"""

from .pool import StreamingSwitchMLP, switch_forward
from .reader import ExpertReader
from .stindex import SafetensorsIndex

__all__ = ["SafetensorsIndex", "ExpertReader", "StreamingSwitchMLP",
           "switch_forward"]
