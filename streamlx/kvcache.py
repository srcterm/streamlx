"""Paged prompt cache on SSD.

Coding agents resend a long, mostly unchanged prompt every turn. mlx-lm's
server keeps finished caches in a RAM LRU and deep-copies one into every
request, so at long contexts the stored caches and their copies are what
fill RAM. Here prompt caches live on disk as segments of the token sequence,
each keyed by a hash of every token before its end. Segments are written as
prefill proceeds (save_progress; an aborted request keeps what it computed)
and after generation; a request restores the longest prefix it shares with
anything stored, from any earlier turn or conversation, and RAM holds only
the active request's cache.

Layers whose cache grows along the sequence (KVCache, QSAKVCache,
QuantizedKVCache, ConcatenateKVCache) store each segment's slice: a prefix
restores from the chain of segments before it, cut at any token when every
layer is such a cache. Other layers (recurrent ArraysCache, rotating
windows) store their state at the segment's end, a window only the rows a
next token can reach, so models with them restore at segment ends only. A
CacheList (Inkling 2's layers: attention rows beside a convolution state)
stores each member its own way: stored whole, each segment would write all
the rows before it again (a 128k-token conversation, ~100 GB).

Each model's segments sit in a directory of its own under the root, named
for its path, config and weight files (weights replaced under the same path
start an empty cache), and one budget covers them all: past it the least recently used of the other models'
caches go first, a model's whole cache at a time (a server still running on
that model then misses and stores it again), then the served model's least
recently used segments. A segment is not written when it would leave the volume with
less than 1 GiB free.

A segment's arrays are stored compressed, losslessly: in blocks, each one's
bytes split into planes, the most significant first (the high byte of a
bf16 value, its sign and exponent, repeats far more than the low one), then
compressed with Apple's libcompression; Accelerate's vImage splits and joins
the planes (both are system libraries). A restore reads every segment's
part of an array straight into the array's final buffer, decoding on a pool
of threads, so a row is copied once on its way in.

Drop-in for mlx_lm's LRUPromptCache in the sequential server (one request at
a time): the same fetch_nearest_cache / insert_cache interface, plus
save_progress(n, cache) when the active request's prefill reaches token n.
"""

from __future__ import annotations

import bisect
import copy
import ctypes
import fcntl
import hashlib
import heapq
import importlib
import inspect
import json
import logging
import math
import os
import queue
import re
import shutil
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as _all_done
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .pool import _storage
from .reader import F_RDAHEAD

FORMAT = 3      # 3: arrays in compressed blocks (2: a CacheList's members
#                 each their own way, windows cut)
_STEP = 256      # tokens per hash step
_SLICED = ("KVCache", "QSAKVCache", "QuantizedKVCache", "ConcatenateKVCache")
# state leaves of a sliced cache not along axis -2: their axis, or None for what
# the cache rebuilds (Qwen3.8's indexer: positions, then its block summaries)
_LEAF_AXIS = {"QSAKVCache": {"3": -1, "4": None, "5": None}}
_TAGS = {mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32",
         mx.int32: "I32", mx.uint32: "U32", mx.int64: "I64",
         mx.uint8: "U8", mx.bool_: "BOOL"}

_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
       "I32": np.int32, "U32": np.uint32, "I64": np.int64, "U8": np.uint8,
       "BOOL": np.bool_}
_MX = {tag: dt for dt, tag in _TAGS.items()}

_MODEL_DIR = re.compile(r"[0-9a-f]{16}")   # root/<model fingerprint>
_SAMPLE = 2**16  # bytes of a weight file's middle, and of its end, fingerprinted
_FREE_FLOOR = 2**30   # what a segment write leaves free on the volume at least

# The codecs: on the four models' stored segments (bf16 attention rows,
# planes split) lzbitmap saves 18-21% (DeepSeek-V4.1's 35%) and decodes at
# ~10 GB/s a thread, faster than a drive reads; LZFSE saves 28-29% (55%)
# at ~1.3 GB/s a thread; lz4 14-16%; zlib about LZFSE's at a third of its
# speed. Recurrent states in float32 save 10-14%.
_ALGS = {"lzbitmap": 0x702, "lzfse": 0x801, "lz4": 0x100, "zlib": 0x205}
_BLOCK = 4 * 2**20   # bytes of an array a block holds
_ALLOC = 256 * 2**20  # bytes of restored arrays allocated at a time

log = logging.getLogger(__name__)


def _codec(v: str | None) -> str:
    """STREAMLX_KV_CODEC: how segments are compressed, one of _ALGS (default
    lzbitmap); none, off, false, no or 0: not at all. Any other word is the
    default."""
    v = (v or "").strip().lower()
    if v in ("none", "off", "false", "no", "0"):
        return "none"
    return v if v in _ALGS else "lzbitmap"


def _libcompression():
    try:
        lib = ctypes.CDLL("/usr/lib/libcompression.dylib")
    except OSError:     # not macOS: segments are stored as they are
        return None
    for fn in (lib.compression_encode_buffer, lib.compression_decode_buffer):
        fn.restype = ctypes.c_size_t
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                       ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    return lib


_LIB = _libcompression()
CODEC = _codec(os.environ.get("STREAMLX_KV_CODEC")) if _LIB else "none"


class _VBuf(ctypes.Structure):     # vImage_Buffer
    _fields_ = [("data", ctypes.c_void_p), ("height", ctypes.c_size_t),
                ("width", ctypes.c_size_t), ("rowBytes", ctypes.c_size_t)]


def _vimage():
    """Accelerate's vImage (a system framework): its four-plane interleave
    and deinterleave run at ~35-45 GB/s a thread where numpy's shifts
    managed 5; None where it does not load (numpy then does it)."""
    try:
        lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/"
                          "Accelerate")
    except OSError:
        return None
    for fn in (lib.vImageConvert_Planar8toARGB8888,
               lib.vImageConvert_ARGB8888toPlanar8):
        fn.restype = ctypes.c_ssize_t
        fn.argtypes = [ctypes.POINTER(_VBuf)] * 5 + [ctypes.c_uint32]
    return lib


_VIMAGE = _vimage()
_NO_TILE = 16   # kvImageDoNotTile: one thread a call (the callers are many)

_POOLS: dict = {}


def _pool(name: str, workers: int) -> ThreadPoolExecutor:
    """The writer's encoders ("enc") and the restore's readers ("dec")."""
    if name not in _POOLS:
        _POOLS[name] = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"streamlx-kv-{name}")
    return _POOLS[name]


def _planar(w: int, n: int) -> bool:
    """Whether a block of n bytes of w-byte values is split into planes:
    those of values wider than a byte that fill whole 4-byte groups."""
    return w > 1 and n % 4 == 0


def _bufs(a: np.ndarray, p: int) -> list:
    """vImage buffers of a's four planes of p bytes, the first (the most
    significant byte of each 4-byte group) last: vImage's A, R, G, B."""
    return [ctypes.byref(_VBuf(a[k * p:].ctypes.data, 1, p, p))
            for k in (3, 2, 1, 0)]


def _split(raw: np.ndarray) -> np.ndarray:
    """A block's 4-byte groups (a bf16 pair, a float32) as four planes, the
    groups' most significant byte first: the high byte of a value, its sign
    and exponent, repeats far more than the low one."""
    p, out = raw.size // 4, np.empty(raw.size, np.uint8)
    if _VIMAGE is None:
        out.reshape(4, p)[:] = raw.reshape(p, 4)[:, ::-1].T
    elif _VIMAGE.vImageConvert_ARGB8888toPlanar8(
            ctypes.byref(_VBuf(raw.ctypes.data, 1, p, 4 * p)),
            *_bufs(out, p), _NO_TILE):
        raise RuntimeError("vImage could not split a block")
    return out


def _join(planes: np.ndarray, out: np.ndarray) -> None:
    """_split's inverse, into out."""
    p = planes.size // 4
    if _VIMAGE is None:
        out.reshape(p, 4)[:] = planes.reshape(4, p)[::-1].T
    elif _VIMAGE.vImageConvert_Planar8toARGB8888(
            *_bufs(planes, p), ctypes.byref(_VBuf(out.ctypes.data, 1, p,
                                                  4 * p)), _NO_TILE):
        raise RuntimeError("vImage could not join a block")


def _encode(raw: np.ndarray, w: int, alg: int) -> np.ndarray | None:
    """A block's bytes compressed, or None when they would not shrink."""
    src = _split(raw) if _planar(w, raw.size) else raw
    dst = np.empty(src.size - 1, np.uint8)
    n = _LIB.compression_encode_buffer(dst.ctypes.data, dst.size,
                                       src.ctypes.data, src.size, None,
                                       alg) if dst.size else 0
    return dst[:n] if n else None


def _scratch(bufs: dict | None, name: str, n: int) -> np.ndarray:
    """n bytes of a thread's buffer `name` from `bufs` (a restore's, one a
    thread), grown as needed: a block then reads and decodes into memory
    already faulted in and warm in the caches, not a fresh allocation."""
    if bufs is None:
        return np.empty(n, np.uint8)
    b = bufs.get(name)
    if b is None or b.size < n:
        b = bufs[name] = np.empty(n, np.uint8)
    return b[:n]


def _decode(stored: np.ndarray, out: np.ndarray, w: int, alg: int,
            bufs: dict | None = None) -> None:
    """A block as stored -> its bytes, into out (a block that would not
    shrink is stored as it was: as many bytes as out)."""
    if stored.size == out.size:
        out[:] = stored
        return
    planar = _planar(w, out.size)
    planes = _scratch(bufs, "planes", out.size) if planar else out
    n = _LIB.compression_decode_buffer(planes.ctypes.data, planes.size,
                                       stored.ctypes.data, stored.size, None,
                                       alg)
    if n != out.size:
        raise ValueError(f"a damaged block: {n} of {out.size} bytes")
    if planar:
        _join(planes, out)


def _lines(shape, w: int, axis) -> tuple[int, int] | None:
    """(bytes of a row, of a line) of an array stored along the sequence
    (axis -2): a line is a plane of its last two dims (a head's rows), which
    a restore puts in one stretch of the restored buffer. None for others."""
    if axis != -2 or len(shape) < 2:
        return None
    rb = shape[-1] * w
    return rb, shape[-2] * rb


def _blocks(raw: np.ndarray, shape, w: int, axis) -> list[np.ndarray]:
    """An array's bytes in blocks: along the sequence, runs of whole rows
    within one line (so that each block lands in one stretch of a restored
    buffer: decoded or read straight into place); else runs of _BLOCK
    bytes."""
    ln = _lines(shape, w, axis)
    if ln is None or not raw.size:
        return [raw[i:i + _BLOCK] for i in range(0, raw.size, _BLOCK)]
    rb, lb = ln
    q = max(1, _BLOCK // rb) * rb
    return [raw[s + i:s + min(i + q, lb)]
            for s in range(0, raw.size, lb) for i in range(0, lb, q)]


def _pack(host: dict, codec: str, meta: dict) -> tuple[list, list]:
    """A segment's arrays as they go in its file: the header's entries
    [(name, dtype, shape, bytes)] and the pieces to write, in order. host:
    {name: (dtype, array, axis along the sequence or None)}. With a codec an
    array (the tokens excepted) goes in blocks (_blocks), each compressed
    or, when it would not shrink, as it is; meta["arrays"] then holds its
    dtype, shape and [bytes stored, bytes] of each block."""
    alg = _ALGS.get(codec)
    enc = _pool("enc", 4) if alg is not None else None
    todo = []
    for name, (tag, arr, axis) in host.items():
        raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
        if alg is None or name == "tokens" or not raw.size:
            todo.append((name, tag, arr.shape, [raw], None))
            continue
        blocks = _blocks(raw, arr.shape, arr.itemsize, axis)
        todo.append((name, tag, arr.shape, blocks,
                     [enc.submit(_encode, b, arr.itemsize, alg)
                      for b in blocks]))
    entries, pieces, packed = [], [], {}
    for name, tag, shape, blocks, futs in todo:
        got = [f.result() for f in futs] if futs else []
        if any(g is not None for g in got):
            stored = [b if g is None else g for b, g in zip(blocks, got)]
            packed[name] = {"dtype": tag, "shape": list(shape), "blocks": [
                [s.size, b.size] for s, b in zip(stored, blocks)]}
            tag, shape, blocks = "U8", (sum(s.size for s in stored),), stored
        entries.append((name, tag, list(shape), sum(b.size for b in blocks)))
        pieces += blocks
    if packed:
        meta.update(codec=codec, arrays=packed)
    return entries, pieces


def _stored(h: dict, meta: dict, name: str, axis) -> tuple:
    """(dtype, shape, [[bytes stored, bytes], ...]) of a segment's array; one
    as it was comes in blocks of a line each along the sequence."""
    packed = meta.get("arrays", {}).get(name)
    if packed is not None:
        return packed["dtype"], packed["shape"], packed["blocks"]
    tag, shape = h[name]["dtype"], h[name]["shape"]
    a, b = h[name]["data_offsets"]
    ln = _lines(shape, np.dtype(_NP[tag]).itemsize, axis)
    if ln is None or a == b:
        return tag, shape, [[b - a, b - a]]
    return tag, shape, [[ln[1], ln[1]]] * ((b - a) // ln[1])


def _unpack(buf: np.ndarray, h: dict, meta: dict, name: str) -> np.ndarray:
    """An array of a segment as it was, from its bytes in the file."""
    tag, shape, blocks = _stored(h, meta, name, None)
    w = np.dtype(_NP[tag]).itemsize
    raw = np.empty(math.prod(shape) * w, np.uint8)
    alg, at, out = _ALGS.get(meta.get("codec")), 0, 0
    for stored, size in blocks:
        _decode(buf[at:at + stored], raw[out:out + size], w, alg)
        at, out = at + stored, out + size
    if (at, out) != (buf.size, raw.size):
        raise ValueError(f"array {name}: its blocks do not add up")
    return raw.view(_NP[tag]).reshape(shape)


def _read(fd: int, buf: np.ndarray, off: int) -> None:
    if os.preadv(fd, [buf], off) != buf.nbytes:
        raise OSError(f"short read at {off}")


def _fill(fd: int, file: tuple, name: str, axis, at: int, rows: int,
          dst: np.ndarray, scratch: dict) -> None:
    """One segment's part of a restored array, its blocks read (and
    decoded) straight into dst, a view of the array's buffer: the segment's
    first `rows` rows along `axis` at row `at` of each line, or (axis None)
    the whole array. A block stored as it was is read straight into place,
    a compressed one into a buffer and decoded from there; a block past
    those rows is not read. scratch: the restore's buffers, {thread id:
    {name: buffer}}. (One read a block beat one an array: F185, cold 0.67
    vs 0.80 s, warm 0.14 vs 0.15 s.)"""
    bufs = scratch.setdefault(threading.get_ident(), {})
    h, meta, base = file
    a, b = h[name]["data_offsets"]
    tag, shape, blocks = _stored(h, meta, name, axis)
    w = np.dtype(_NP[tag]).itemsize
    ln = _lines(shape, w, axis)
    if axis is not None and ln is None:     # along the last axis (small)
        buf = np.empty(b - a, np.uint8)
        _read(fd, buf, base + a)
        dst[..., at:at + rows] = _unpack(buf, h, meta, name)[..., :rows]
        return
    if ln is None:
        flat = dst.reshape(-1).view(np.uint8)
    else:
        rb, lb = ln
        lines = dst.reshape(-1, *dst.shape[-2:])
    alg = _ALGS.get(meta.get("codec"))
    off, r = base + a, 0
    for stored, size in blocks:
        if ln is None:
            out, need = flat[r:r + size], size
        else:       # the block's rows of its line, those restored
            line, q = divmod(r, lb)
            q //= rb
            if (r % lb) % rb or size % rb or (q + size // rb) * rb > lb:
                raise ValueError(f"array {name}: a block across lines")
            n = min(q + size // rb, rows) - q
            out = (lines[line, at + q:at + q + n].reshape(-1).view(np.uint8)
                   if n > 0 else None)
            need = max(n, 0) * rb
        if out is not None:
            if stored == size:          # as it was: read straight in
                _read(fd, out, off)
            else:
                buf = _scratch(bufs, "stored", stored)
                _read(fd, buf, off)
                if need == size:
                    _decode(buf, out, w, alg, bufs)
                else:                   # the first rows of a block
                    tmp = _scratch(bufs, "rows", size)
                    _decode(buf, tmp, w, alg, bufs)
                    out[:] = tmp[:need]
        off, r = off + stored, r + size
    if off != base + b or r != math.prod(shape) * w:
        raise ValueError(f"array {name}: its blocks do not add up")


def _files(d: Path) -> list[Path]:
    """A model directory's segment files."""
    try:
        return list(d.glob("*.safetensors"))
    except OSError:   # gone meanwhile
        return []


def _model_dirs(root: Path) -> list[Path]:
    try:
        return [d for d in root.iterdir()
                if d.is_dir() and _MODEL_DIR.fullmatch(d.name)]
    except OSError:
        return []


def _size_and_used(files: list[Path]) -> tuple[int, float]:
    n, used = 0, 0.0
    for p in files:
        try:
            st = p.stat()
        except OSError:   # gone meanwhile
            continue
        n, used = n + st.st_size, max(used, st.st_mtime)
    return n, used


def _weights_id(mdir: Path) -> bytes:
    """What tells a checkpoint's weights from others put under the same path
    with the same config: each weight file's name, size, modification time
    and change time, and bytes from its middle and its end (a copy can keep
    the modification time; no write to the file keeps the change time)."""
    h = hashlib.sha256()
    for p in sorted(mdir.glob("*.safetensors")):
        if not p.is_file():
            continue
        st = p.stat()
        h.update(f"{p.name}/{st.st_size}/{st.st_mtime_ns}/"
                 f"{st.st_ctime_ns}/".encode())
        with open(p, "rb") as f:
            for at in (st.st_size // 2, max(0, st.st_size - _SAMPLE)):
                f.seek(at)
                h.update(f.read(_SAMPLE))
    return h.digest()


def stored_bytes(root) -> int:
    """The bytes every model's prompt caches hold under root."""
    return sum(_size_and_used(_files(d))[0] for d in _model_dirs(Path(root)))


def _sliced(c) -> bool:
    """Whether a cache grows along the sequence (axis -2) and nothing else,
    so any token range of it can be cut out and concatenated back."""
    return type(c).__name__ in _SLICED


def _window(c):
    """A rotating window (mlx-lm's RotatingKVCache) after a forward of
    several tokens holds them and the max_size - 1 rows before them, in
    order; the next forward reaches back max_size - 1 rows, a token
    max_size (`keep` rows and the last max_size - keep, then it rotates).
    A copy of c with only those rows, cut as its next update cuts them (the
    same next state: the same rows, the same write index), else c."""
    if (type(c).__name__ != "RotatingKVCache"
            or getattr(c, "keys", None) is None):
        return c
    n, size = c.keys.shape[2], c.max_size
    if n <= size or c._idx != n:    # (in order: the last update appended)
        return c
    w = copy.copy(c)
    w.keys, w.values = c._trim(n - size, c.keys), c._trim(n - size, c.values)
    w._idx = size
    return w


def settled_bytes(c) -> int:
    """The bytes a layer cache holds once its next token has run (a window
    after a prompt chunk holds the chunk too until then: _window)."""
    if type(c).__name__ == "CacheList":
        return sum(settled_bytes(m) for m in c.caches)
    return _window(c).nbytes


def _layout(specs) -> list:
    """The cache classes of a segment's layers, a CacheList's members too."""
    return [[s["cls"], _layout(s["members"]) if "members" in s else None]
            for s in specs]


def _offset(c) -> int | None:
    """Tokens a layer cache has seen (None for recurrent state)."""
    if type(c).__name__ == "CacheList":
        return max((_offset(x) or 0 for x in c.caches), default=None)
    return getattr(c, "offset", None)


def _cls(name: str):
    mod, _, qual = name.rpartition(".")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _host(a: mx.array) -> tuple[str, np.ndarray]:
    """(safetensors dtype, host copy); made on the calling thread so the
    writer thread never touches an mx array."""
    tag = _TAGS[a.dtype]
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return tag, np.array(a)


def _store(c, prefix: str, start: int, end: int, host: dict) -> dict:
    """A cache's part of the segment tokens[start:end]: its arrays into
    `host` under `prefix` (dtype, host copy, the axis it is cut along or
    None), and its spec. A CacheList's members are stored each their own
    way, under prefix + their index."""
    cls = f"{type(c).__module__}.{type(c).__qualname__}"
    if type(c).__name__ == "CacheList":
        members = [_store(m, f"{prefix}{j}.", start, end, host)
                   for j, m in enumerate(c.caches)]
        return {"cls": cls, "sliced": all(m["sliced"] for m in members),
                "members": members}
    sl = _sliced(c)
    if not sl:
        c = _window(c)
    axes = _LEAF_AXIS.get(type(c).__name__, {}) if sl else {}
    scalars = {}
    for path, v in tree_flatten(c.state):
        axis = axes.get(path, -2)
        if axis is None:
            continue
        if isinstance(v, mx.array):
            host[f"{prefix}{path}"] = (*_host(
                v if not sl else v[..., start:end, :] if axis == -2
                else v[..., start:end]), axis if sl else None)
        else:
            scalars[path] = v
    meta = getattr(c, "meta_state", "")
    return {"cls": cls, "sliced": sl, "scalars": scalars,
            "meta": list(meta) if isinstance(meta, tuple) else meta}


@dataclass
class _Seg:
    key: str
    prev: str          # key of the prefix this segment extends (root: "")
    start: int
    end: int
    nbytes: int
    used: float
    path: Path


class PagedPromptCache:
    def __init__(self, root: str, max_bytes: int,
                 pending_bytes: int = 512 * 2**20):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._pending_cap = pending_bytes
        self._dir: Path | None = None   # per model, set on first use
        self._seed = b""
        self._segs: dict[str, _Seg] = {}
        self._next: dict[str, set] = {}  # prev key -> keys extending it
        self._ends: list[int] = []       # sorted distinct segment ends
        self._bytes = 0
        self._others: list = []   # other models' caches: (last used, bytes, dir)
        self._other_bytes = 0
        self._sliced_only: bool | None = None
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._pending = 0
        self._q: queue.Queue = queue.Queue()
        self._cur = None    # active request: (tokens, last cut, last key)
        self.last = (0, 0)  # last fetch: (tokens restored, prompt tokens)
        threading.Thread(target=self._writer, daemon=True,
                         name="streamlx-kv").start()

    # -- keys ----------------------------------------------------------------

    def _open(self, model_key) -> None:
        """Bind to the served model: one directory per model fingerprint
        (path, config, weight index, weight files), its segments indexed
        from their headers."""
        if self._dir is not None:
            return
        mdir = Path(model_key[0] if isinstance(model_key, tuple)
                    else model_key)
        h = hashlib.sha256(f"streamlx-kv/{FORMAT}/{mdir.resolve()}".encode())
        for f in ("config.json", "model.safetensors.index.json"):
            if (mdir / f).exists():
                h.update((mdir / f).read_bytes())
        h.update(_weights_id(mdir))
        self._seed = h.digest()
        self._dir = self.root / h.hexdigest()[:16]
        self._dir.mkdir(parents=True, exist_ok=True)
        for p in self._dir.glob("*.tmp"):
            p.unlink(missing_ok=True)
        for p in self._dir.glob("*.safetensors"):
            try:
                meta = self._header(p)[1]
                self._add(_Seg(meta["key"], meta["prev"], meta["start"],
                               meta["end"], p.stat().st_size,
                               p.stat().st_mtime, p))
            except Exception:  # noqa: BLE001 — a torn file is dropped
                p.unlink(missing_ok=True)
                continue
            if self._sliced_only is None:
                self._sliced_only = all(x["sliced"] for x in meta["layers"])
        for s in [s for s in self._segs.values() if self._walk(s) is None]:
            self._drop(s)    # its prefix is gone (deleted, torn write)
        for d in _model_dirs(self.root):
            if d != self._dir:
                n, used = _size_and_used(_files(d))
                self._others.append((used, n, d))
        self._others.sort(key=lambda o: o[0])   # least recently used first
        self._other_bytes = sum(o[1] for o in self._others)
        with self._lock:
            self._evict()    # e.g. a smaller budget than the last server's
        log.info("kv cache %s: %d segments, %.1f GB (%.1f GB of other models)",
                 self._dir, len(self._segs), self._bytes / 1e9,
                 self._other_bytes / 1e9)

    def _chain(self, tokens) -> tuple[list[bytes], np.ndarray]:
        a = np.asarray(tokens, dtype=np.int32)
        c = [self._seed]
        for j in range(len(a) // _STEP):
            c.append(hashlib.sha256(
                c[-1] + a[j * _STEP:(j + 1) * _STEP].tobytes()).digest())
        return c, a

    @staticmethod
    def _key(chain, a, n: int) -> str:
        """Hash of tokens[:n], whatever the cut points before n."""
        j = n // _STEP
        return hashlib.sha256(chain[j] + a[j * _STEP:n].tobytes()
                              + n.to_bytes(8, "little")).hexdigest()

    # -- index -------------------------------------------------------------

    def _add(self, s: _Seg) -> None:
        old = self._segs.get(s.key)
        if old is not None:
            self._bytes -= old.nbytes
        self._segs[s.key] = s
        self._next.setdefault(s.prev, set()).add(s.key)
        i = bisect.bisect_left(self._ends, s.end)
        if i == len(self._ends) or self._ends[i] != s.end:
            self._ends.insert(i, s.end)
        self._bytes += s.nbytes

    def _drop(self, s: _Seg) -> None:
        del self._segs[s.key]
        self._next.get(s.prev, set()).discard(s.key)
        self._bytes -= s.nbytes
        s.path.unlink(missing_ok=True)

    def _forget_gone(self) -> bool:
        """Forget the segments whose files are gone (another server's budget
        removed this model's cache), so that they are stored again instead of
        being taken for stored. True when there were any."""
        with self._lock:
            gone = [s for s in self._segs.values() if not s.path.exists()]
            for s in gone:
                self._drop(s)
            alive = {s.end for s in self._segs.values()}
            self._ends = [e for e in self._ends if e in alive]
        if gone:
            log.info("kv cache: %d segments gone from disk, forgotten",
                     len(gone))
        return bool(gone)

    def _evict(self) -> None:
        """Other models' caches first, least recently used first and each
        whole; then this model's, least recently used first, leaves only: a
        segment goes once nothing extends it, so no stored chain ever loses
        its prefix."""
        while self._others and self._bytes + self._other_bytes > self.max_bytes:
            _, n, d = self._others.pop(0)
            for p in _files(d):
                p.unlink(missing_ok=True)
            self._other_bytes -= n
            log.info("kv cache: %s dropped (%.1f GB, another model's)", d,
                     n / 1e9)
        limit = self.max_bytes - self._other_bytes
        if self._bytes <= limit:
            return
        heap = [(s.used, s.key) for s in self._segs.values()
                if not self._next.get(s.key)]
        heapq.heapify(heap)
        while heap and self._bytes > limit:
            _, k = heapq.heappop(heap)
            s = self._segs.get(k)
            if s is None or self._next.get(k):
                continue
            self._drop(s)
            p = self._segs.get(s.prev)
            if p is not None and not self._next.get(p.key):
                heapq.heappush(heap, (p.used, p.key))
        alive = {s.end for s in self._segs.values()}
        self._ends = [e for e in self._ends if e in alive]

    def _walk(self, s: _Seg) -> list[_Seg] | None:
        """The segments from the start of the sequence to s, or None when
        one of them has been evicted."""
        out = [s]
        while out[-1].prev:
            p = self._segs.get(out[-1].prev)
            if p is None:
                return None
            out.append(p)
        return out[::-1] if out[-1].start == 0 else None

    # -- files --------------------------------------------------------------

    @staticmethod
    def _file(path: Path) -> tuple[dict, dict, int]:
        """A segment file's header, its streamlx metadata and where its
        arrays' bytes start."""
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            h = json.loads(f.read(n))
        return h, json.loads(h["__metadata__"]["streamlx"]), 8 + n

    @classmethod
    def _header(cls, path: Path) -> tuple[dict, dict]:
        return cls._file(path)[:2]

    @staticmethod
    def _arrays(h: dict, meta: dict) -> dict:
        """{name: (dtype, shape)} of a segment's arrays as they were stored
        (compressed ones by their metadata)."""
        packed = meta.get("arrays", {})
        return {k: ((packed[k]["dtype"], packed[k]["shape"]) if k in packed
                    else (v["dtype"], v["shape"]))
                for k, v in h.items() if k != "__metadata__"}

    def _tokens(self, s: _Seg) -> np.ndarray:
        h, _ = self._header(s.path)
        t = h["tokens"]
        a, b = t["data_offsets"]
        with open(s.path, "rb") as f:
            f.seek(8 + struct.unpack("<Q", f.read(8))[0] + a)
            return np.frombuffer(f.read(b - a), dtype=np.int32)

    def _writer(self) -> None:
        while True:
            seg, host, meta, nbytes = self._q.get()
            tmp = seg.path.with_suffix(".tmp")
            try:
                free = shutil.disk_usage(seg.path.parent).free
                if free - nbytes < _FREE_FLOOR:
                    log.warning("kv cache: segment not stored, %.1f GB free",
                                free / 1e9)
                    continue
                entries, pieces = _pack(host, CODEC, meta)
                header, off = {}, 0
                for name, tag, shape, size in entries:
                    header[name] = {"dtype": tag, "shape": shape,
                                    "data_offsets": [off, off + size]}
                    off += size
                header["__metadata__"] = {"streamlx": json.dumps(meta)}
                hb = json.dumps(header).encode()
                hb += b" " * (-len(hb) % 8)
                with open(tmp, "wb") as f:
                    f.write(struct.pack("<Q", len(hb)))
                    f.write(hb)
                    for p in pieces:
                        f.write(p.data)
                os.replace(tmp, seg.path)
                seg.nbytes = seg.path.stat().st_size
                with self._lock:
                    self._add(seg)
                    self._evict()
            except Exception:  # a failed write is a cache miss
                log.exception("kv cache: segment write failed")
                tmp.unlink(missing_ok=True)
            finally:
                with self._cv:
                    self._pending -= nbytes
                    self._cv.notify_all()

    def _save(self, chain, a, start: int, end: int, prev: str,
              cache) -> str:
        """Queue the segment tokens[start:end] of the live cache; returns its
        key. Host copies are made here (main thread), the file is written by
        the writer thread."""
        key = self._key(chain, a, end)
        with self._lock:
            if key in self._segs:
                return key
        mx.eval([v for c in cache for _, v in tree_flatten(c.state)
                 if isinstance(v, mx.array)])
        host = {"tokens": ("I32", a[start:end].copy(), None)}
        layers = [_store(c, f"{i}.", start, end, host)
                  for i, c in enumerate(cache)]
        if self._sliced_only is None:
            self._sliced_only = all(x["sliced"] for x in layers)
        meta = {"format": FORMAT, "key": key, "prev": prev, "start": start,
                "end": end, "layers": layers}
        nbytes = sum(arr.nbytes for _, arr, _ in host.values())
        with self._cv:   # bound the host copies waiting for the disk
            while self._pending and self._pending + nbytes > self._pending_cap:
                self._cv.wait()
            self._pending += nbytes
        seg = _Seg(key, prev, start, end, nbytes, time.time(),
                   self._dir / f"{key}.safetensors")
        self._q.put((seg, host, meta, nbytes))
        return key

    def _restore(self, chain: list[_Seg], partial: tuple[_Seg, int] | None):
        """Rebuild the per-layer caches from the chain (+ the first tokens
        of one more segment): sliced layers take their rows from every
        segment, the others their state from the last whole segment. Each
        array is allocated once, whole, and every segment's part of it read
        (and decoded) straight into place on the restore's threads."""
        segs = chain + ([partial[0]] if partial is not None else [])
        files = [self._file(s.path) for s in segs]
        metas = [m for _, m, _ in files]
        kinds = {json.dumps(_layout(m["layers"])) for m in metas}
        if len(kinds) != 1:   # e.g. KV quantized from some point on
            raise ValueError("segments of different cache layouts")
        n = chain[-1].end if chain else 0
        if partial is not None:
            n = partial[0].start + partial[1]
        whole = len(chain)
        stored = [self._arrays(h, m) for h, m, _ in files]
        arrays: dict = {}    # name -> (the restored array, its dtype)
        jobs = []            # (segment, name, axis, at row, rows)

        def plan(prefix, spec):
            """The arrays of the cache stored under `prefix` (`spec` its
            spec in the newest segment) and the reads that fill them."""
            if "members" in spec:     # a CacheList: its members' arrays
                for j, m in enumerate(spec["members"]):
                    plan(f"{prefix}{j}.", m)
                return
            cls = _cls(spec["cls"])
            axes = _LEAF_AXIS.get(spec["cls"].rpartition(".")[2], {})
            # a KVCache's rows go in a buffer with the room its updates
            # allocate (`step` rows at a time), as a live one has: else the
            # next token would copy every row into a new buffer
            step = getattr(cls, "step", 0) if cls.__name__ == "KVCache" else 0
            for key in [k for k in stored[-1] if k.startswith(prefix)]:
                if not spec["sliced"]:
                    tag, shape = stored[whole - 1][key]
                    arrays[key] = mx.zeros(shape, _MX[tag]), tag
                    jobs.append((whole - 1, key, None, 0, 0))
                    continue
                axis = axes.get(key[len(prefix):], -2)
                tag, shape = stored[-1][key]
                parts = [(i, stored[i][key][1][axis]) for i in range(whole)]
                if partial is not None:
                    parts.append((whole, partial[1]))
                pad = -n % step if step and axis == -2 else 0
                shape = list(shape)
                shape[axis] = sum(r for _, r in parts) + pad
                arrays[key] = mx.zeros(shape, _MX[tag]), tag
                at = 0
                for i, r in parts:
                    jobs.append((i, key, axis, at, r))
                    at += r

        for i, spec in enumerate(metas[-1]["layers"]):
            plan(f"{i}.", spec)
        # the arrays' buffers in groups of ~_ALLOC bytes: the GPU zeroes
        # the next group while the threads fill this one
        groups, size = [[]], 0
        for k, (a, _) in arrays.items():
            if size >= _ALLOC:
                groups.append([])
                size = 0
            groups[-1].append(k)
            size += a.nbytes
        by_key: dict = {}
        for job in jobs:
            by_key.setdefault(job[1], []).append(job)
        views, fds, futs = {}, [], []
        try:
            for s in segs:
                fds.append(os.open(s.path, os.O_RDONLY))
                # no read-ahead: it fetched the arrays a restore skips (the
                # states of all but the last segment), and from a drive a
                # restore took 18% longer (F185: 0.67 vs 0.57 s)
                fcntl.fcntl(fds[-1], F_RDAHEAD, 0)
            # six threads: on an M4 (4 + 6 cores) 4-8 restore alike, 6 best
            dec = _pool("dec", min(6, os.cpu_count() or 4))
            scratch: dict = {}   # the threads' buffers, gone with the restore
            mx.async_eval([arrays[k][0] for k in groups[0]])
            for g, keys in enumerate(groups):
                if g + 1 < len(groups):
                    mx.async_eval([arrays[k][0] for k in groups[g + 1]])
                mx.eval([arrays[k][0] for k in keys])
                for k in keys:
                    a, tag = arrays[k]
                    if not a.size:
                        continue
                    views[k] = np.frombuffer(_storage(a), _NP[tag]).reshape(
                        a.shape)
                    futs += [dec.submit(_fill, fds[i], files[i], k, axis, at,
                                        r, views[k], scratch)
                             for i, _, axis, at, r in by_key.get(k, ())]
            _all_done(futs)    # (a failed read: the others write on first)
            for f in futs:
                f.result()
        finally:
            _all_done(futs)    # no thread still reads a file or writes a view
            for fd in fds:
                os.close(fd)
            # a live view keeps MLX from updating an array in place
            del views

        def rebuild(prefix, spec, old):
            """The cache stored under `prefix`: `spec` its spec in the newest
            segment, `old` in the last whole one (the same layout)."""
            cls = _cls(spec["cls"])
            if "members" in spec:     # a CacheList: its members, rebuilt
                return cls(*(rebuild(f"{prefix}{j}.", m, o) for j, (m, o)
                             in enumerate(zip(spec["members"],
                                              old["members"]))))
            leaves = [(k[len(prefix):], a) for k, (a, _) in arrays.items()
                      if k.startswith(prefix)]
            src = spec if spec["sliced"] else old
            leaves += list(src["scalars"].items())
            state = tree_unflatten(leaves)
            # mlx-vlm's caches (and mlx-lm's older ones) rebuild from both
            if len(inspect.signature(cls.from_state).parameters) > 1:
                meta = src.get("meta", "")
                c = cls.from_state(state, tuple(meta) if isinstance(meta, list)
                                   else meta)
            else:
                c = cls.from_state(state)
            if spec["sliced"]:
                c.offset = n
            return c

        return [rebuild(f"{i}.", s, o) for i, (s, o) in
                enumerate(zip(metas[-1]["layers"],
                              metas[whole - 1]["layers"]))], n

    # -- the server's interface ----------------------------------------------

    def fetch_nearest_cache(self, model_key, tokens):
        self._open(model_key)
        self.flush()    # the previous turn's segments, if still in flight
        # a segment whose file is gone must not stay indexed past a lookup
        # it does not serve: its prompt would count as stored
        self._forget_gone()
        tokens = list(tokens)
        self.last = (0, len(tokens))
        chain, a = self._chain(tokens)
        best: list[_Seg] = []
        with self._lock:
            lim = bisect.bisect_right(self._ends, len(tokens) - 1)
            for e in reversed(self._ends[:lim]):
                s = self._segs.get(self._key(chain, a, e))
                if s is not None:
                    walked = self._walk(s)
                    if walked:
                        best = walked
                        break
            succ = [self._segs[k] for k in
                    self._next.get(best[-1].key if best else "", ())]
        partial = None
        if self._sliced_only:   # any token: cut into a stored successor
            at = best[-1].end if best else 0
            want = a[at:len(tokens) - 1]
            for s in succ:
                try:
                    t = self._tokens(s)
                except OSError:
                    continue
                except Exception:  # a damaged file: dropped, stored again
                    log.warning("kv cache: unreadable segment %s dropped",
                                s.path.name)
                    with self._lock:
                        if self._segs.get(s.key) is s:
                            self._drop(s)
                    continue
                m = min(len(t), len(want))
                eq = t[:m] == want[:m]
                same = m if eq.all() else int(np.argmin(eq))
                if same and (partial is None or same > partial[1]):
                    partial = (s, same)
        if not best and partial is None:
            self._cur = (tokens, 0, "")
            return None, tokens
        used = best + ([partial[0]] if partial else [])
        if (not all(s.path.exists() for s in used)
                and self._forget_gone()):   # look again without them
            return self.fetch_nearest_cache(model_key, tokens)
        try:
            cache, n = self._restore(best, partial)
        except Exception:  # unreadable segments = a miss, stored again
            log.exception("kv cache: restore failed")
            with self._lock:
                for s in used:
                    if self._segs.get(s.key) is s:
                        self._drop(s)
            self._cur = (tokens, 0, "")
            return None, tokens
        now = time.time()
        for s in used:
            s.used = now
            try:
                os.utime(s.path, (now, now))
            except OSError:
                pass
        self.last = (n, len(tokens))
        # new segments chain from the last whole segment (a partial
        # successor's tokens are re-stored with the next segment)
        self._cur = (tokens, best[-1].end if best else 0,
                     best[-1].key if best else "")
        return cache, tokens[n:]

    def request_tokens(self):
        """The active request's whole prompt (what was restored from here
        included), None before a request."""
        return None if self._cur is None else self._cur[0]

    def save_progress(self, n: int, cache) -> None:
        """Prefill reached tokens[:n] of the active request: store it."""
        if self._cur is None or self._dir is None:
            return
        tokens, cut, prev = self._cur
        if n <= cut or n > len(tokens):
            return
        chain, a = self._chain(tokens[:n])
        try:
            key = self._save(chain, a, cut, n, prev, cache)
        except Exception:  # never fail the request over it
            log.exception("kv cache: segment not stored")
            self._cur = None
            return
        self._cur = (tokens, n, key)

    def insert_cache(self, model_key, tokens, prompt_cache, *,
                     cache_type: str = "assistant") -> None:
        self._open(model_key)
        tokens = list(tokens)
        cur = self._cur
        if cur is not None and tokens[:len(cur[0])] == cur[0] and \
                cur[1] < len(tokens):
            cut, prev = cur[1], cur[2]
        else:
            cut, prev = 0, ""
        seen = [o for o in map(_offset, prompt_cache) if o is not None]
        n = min(len(tokens), max(seen)) if seen else len(tokens)
        if n > cut:
            chain, a = self._chain(tokens[:n])
            try:
                self._save(chain, a, cut, n, prev, prompt_cache)
            except Exception:  # never fail the request over it
                log.exception("kv cache: segment not stored")
        self._cur = None

    def trim_to(self, *, n_sequences=None, n_bytes=None) -> None:
        pass    # bounded on disk by max_bytes

    def flush(self) -> None:
        with self._cv:
            while self._pending:
                self._cv.wait()

    def __len__(self) -> int:
        return len(self._segs)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def stats_by_type(self) -> dict:
        return {"ssd": {"n_sequences": len(self._segs),
                        "n_bytes": self._bytes}}
