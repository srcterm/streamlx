"""streamlx — SSD expert streaming for MoE models on mlx-lm.

Phase 3 prototype (see notes/phase3-design.md). v1 scope: batch-1 decode,
per-layer ExpertPool with LRU, pread-based miss path, no custom kernels.
"""

from .pool import StreamingSwitchMLP, switch_forward
from .reader import ExpertReader
from .stindex import SafetensorsIndex

__all__ = ["SafetensorsIndex", "ExpertReader", "StreamingSwitchMLP",
           "switch_forward"]
