"""pread-based expert fetcher with range coalescing.

Miss path per notes/phase3-design.md: MLX lazy loading is tensor-granular
(F10), so expert rows are read directly by byte range. Coalescing (M4):
requests are planned per shard, sorted by offset, and merged into spans when
adjacent or within --merge-gap; spans are read concurrently by a small thread
pool (latency hiding for the many ~33 KB scales/biases rows — the device is
throughput-flat across QD but small reads are latency-bound, F7).

F_NOCACHE is set as declared intent, but per F5 (macOS 26.3) reads populate
the page cache anyway — treated as a free reclaimable L2.
"""

from __future__ import annotations

import fcntl
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from .stindex import PARTS, PROJS, SafetensorsIndex, TensorLoc

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
F_RDAHEAD = getattr(fcntl, "F_RDAHEAD", 45)

_NP_VIEW = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16,
            "F32": np.float32}
_MX_VIEW = {"BF16": mx.bfloat16}


class ExpertReader:
    """Reads expert rows of stacked switch tensors; batch-coalescing planner."""

    def __init__(self, index: SafetensorsIndex, workers: int = 6,
                 merge_gap: int = 64 * 1024, max_span: int = 16 * 1024 * 1024):
        self.index = index
        self.merge_gap = merge_gap
        self.max_span = max_span
        self._fds: dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._ex = ThreadPoolExecutor(max_workers=workers)
        self.bytes_read = 0      # actual bytes off the device (incl. gap waste)
        self.useful_bytes = 0    # bytes delivered to the pool
        self.reads = 0           # preads issued (post-merge)

    def _fd(self, path: str) -> int:
        with self._fd_lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                try:
                    fcntl.fcntl(fd, F_NOCACHE, 1)
                    fcntl.fcntl(fd, F_RDAHEAD, 0)
                except OSError:
                    pass
                self._fds[path] = fd
            return fd

    # -- planning ------------------------------------------------------------

    def _plan(self, requests):
        """requests: [(key, path, off, n)] -> spans [(path, off, n, members)]
        with members = [(key, rel_off, n)]. Merges adjacent/near ranges."""
        spans = []
        for key, path, off, n in sorted(requests, key=lambda r: (r[1], r[2])):
            if spans:
                s_path, s_off, s_n, members = spans[-1]
                gap = off - (s_off + s_n)
                if (path == s_path and 0 <= gap <= self.merge_gap
                        and (off + n) - s_off <= self.max_span):
                    spans[-1] = (s_path, s_off, (off + n) - s_off,
                                 members + [(key, off - s_off, n)])
                    continue
            spans.append((path, off, n, [(key, 0, n)]))
        return spans

    def read_ranges(self, requests) -> dict:
        """[(key, path, off, n)] -> {key: memoryview of exactly n bytes}."""
        spans = self._plan(requests)

        def read_span(span):
            path, off, n, members = span
            buf = os.pread(self._fd(path), n, off)
            if len(buf) != n:
                raise IOError(f"short read {len(buf)}/{n} at {off} of {path}")
            return buf, members

        out: dict = {}
        for fut in [self._ex.submit(read_span, s) for s in spans]:
            buf, members = fut.result()
            view = memoryview(buf)
            self.bytes_read += len(buf)
            self.reads += 1
            for key, rel, n in members:
                out[key] = view[rel:rel + n]
                self.useful_bytes += n
        return out

    # -- expert-level API ----------------------------------------------------

    def _to_mx(self, raw: memoryview, loc: TensorLoc) -> mx.array:
        arr = np.frombuffer(raw, dtype=_NP_VIEW[loc.dtype]).reshape(
            loc.shape[1:])
        out = mx.array(arr)
        if loc.dtype in _MX_VIEW:
            out = out.view(_MX_VIEW[loc.dtype])
        return out

    def read_experts_raw(self, layer_triples: dict, ids: list[int]) -> dict:
        """{(e, proj, part): memoryview} — raw bytes only, one coalesced
        batch. Safe from any thread: no mx arrays are created (mx arrays are
        stream-bound to the creating thread)."""
        requests = []
        for e in ids:
            for proj in PROJS:
                for part in PARTS:
                    loc = layer_triples[proj][part]
                    path, off, n = loc.expert_range(e)
                    requests.append(((e, proj, part), path, off, n))
        return self.read_ranges(requests)

    def experts_from_raw(self, layer_triples: dict, ids: list[int],
                         raw: dict) -> dict:
        """Materialize {e: {proj: {part: mx.array}}} from raw bytes.
        Must run on the thread that will build the compute graph."""
        return {e: {proj: {part: self._to_mx(raw[(e, proj, part)],
                                             layer_triples[proj][part])
                           for part in PARTS}
                    for proj in PROJS}
                for e in ids}

    def read_experts(self, layer_triples: dict, ids: list[int]) -> dict:
        """{expert_id: {proj: {part: mx.array}}} — one coalesced batch."""
        return self.experts_from_raw(
            layer_triples, ids, self.read_experts_raw(layer_triples, ids))

    def read_expert(self, layer_triples: dict, e: int) -> dict:
        return self.read_experts(layer_triples, [e])[e]

    def reset_stats(self) -> None:
        self.bytes_read = 0
        self.useful_bytes = 0
        self.reads = 0

    def close(self) -> None:
        self._ex.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
