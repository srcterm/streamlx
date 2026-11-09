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
QuantizedKVCache, ConcatenateKVCache and CacheLists of them) store each
segment's slice: a prefix restores from the chain of segments before it,
cut at any token when every layer is such a cache. Other layers (recurrent
ArraysCache, rotating windows) store their whole state at the segment's end,
so models with them restore at segment ends only.

Drop-in for mlx_lm's LRUPromptCache in the sequential server (one request at
a time): the same fetch_nearest_cache / insert_cache interface, plus
save_progress(n, cache) when the active request's prefill reaches token n.
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import importlib
import inspect
import json
import logging
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

FORMAT = 1
_STEP = 256      # tokens per hash step
_SLICED = ("KVCache", "QSAKVCache", "QuantizedKVCache", "ConcatenateKVCache")
# state leaves of a sliced cache not along axis -2: their axis, or None for what
# the cache rebuilds (Qwen3.8's indexer: positions, then its block summaries)
_LEAF_AXIS = {"QSAKVCache": {"3": -1, "4": None, "5": None}}
_TAGS = {mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32",
         mx.int32: "I32", mx.uint32: "U32", mx.int64: "I64",
         mx.uint8: "U8", mx.bool_: "BOOL"}

log = logging.getLogger(__name__)


def _sliced(c) -> bool:
    """Whether a layer cache grows along the sequence (axis -2) and nothing
    else, so any token range of it can be cut out and concatenated back."""
    if type(c).__name__ == "CacheList":
        return all(_sliced(x) for x in c.caches)
    return type(c).__name__ in _SLICED


def _offset(c) -> int | None:
    """Tokens a layer cache has seen (None for recurrent state)."""
    if type(c).__name__ == "CacheList":
        return max((_offset(x) or 0 for x in c.caches), default=None)
    return getattr(c, "offset", None)


def _set_offset(c, n: int) -> None:
    if type(c).__name__ == "CacheList":
        for x in c.caches:
            _set_offset(x, n)
    elif hasattr(c, "offset"):
        c.offset = n


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
        (path, config, weight index), its segments indexed from their
        headers."""
        if self._dir is not None:
            return
        mdir = Path(model_key[0] if isinstance(model_key, tuple)
                    else model_key)
        h = hashlib.sha256(f"streamlx-kv/{FORMAT}/{mdir.resolve()}".encode())
        for f in ("config.json", "model.safetensors.index.json"):
            if (mdir / f).exists():
                h.update((mdir / f).read_bytes())
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
        log.info("kv cache %s: %d segments, %.1f GB", self._dir,
                 len(self._segs), self._bytes / 1e9)

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

    def _evict(self) -> None:
        """Least recently used first, leaves only: a segment goes once
        nothing extends it, so no stored chain ever loses its prefix."""
        if self._bytes <= self.max_bytes:
            return
        heap = [(s.used, s.key) for s in self._segs.values()
                if not self._next.get(s.key)]
        heapq.heapify(heap)
        while heap and self._bytes > self.max_bytes:
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
    def _header(path: Path) -> tuple[dict, dict]:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            h = json.loads(f.read(n))
        return h, json.loads(h["__metadata__"]["streamlx"])

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
            try:
                header, off = {}, 0
                for name, (tag, arr) in host.items():
                    header[name] = {"dtype": tag, "shape": list(arr.shape),
                                    "data_offsets": [off, off + arr.nbytes]}
                    off += arr.nbytes
                header["__metadata__"] = {"streamlx": json.dumps(meta)}
                hb = json.dumps(header).encode()
                hb += b" " * (-len(hb) % 8)
                tmp = seg.path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    f.write(struct.pack("<Q", len(hb)))
                    f.write(hb)
                    for _, arr in host.values():
                        f.write(np.ascontiguousarray(arr).data)
                os.replace(tmp, seg.path)
                seg.nbytes = seg.path.stat().st_size
                with self._lock:
                    self._add(seg)
                    self._evict()
            except Exception:  # a failed write is a cache miss
                log.exception("kv cache: segment write failed")
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
        host = {"tokens": ("I32", a[start:end].copy())}
        layers = []
        for i, c in enumerate(cache):
            sl = _sliced(c)
            axes = _LEAF_AXIS.get(type(c).__name__, {}) if sl else {}
            scalars = {}
            for path, v in tree_flatten(c.state):
                axis = axes.get(path, -2)
                if axis is None:
                    continue
                if isinstance(v, mx.array):
                    host[f"{i}.{path}"] = _host(
                        v if not sl else v[..., start:end, :] if axis == -2
                        else v[..., start:end])
                else:
                    scalars[path] = v
            meta = getattr(c, "meta_state", "")
            layers.append({"cls": f"{type(c).__module__}."
                                  f"{type(c).__qualname__}",
                           "sliced": sl, "scalars": scalars,
                           "meta": list(meta) if isinstance(meta, tuple)
                           else meta})
        if self._sliced_only is None:
            self._sliced_only = all(x["sliced"] for x in layers)
        meta = {"format": FORMAT, "key": key, "prev": prev, "start": start,
                "end": end, "layers": layers}
        nbytes = sum(arr.nbytes for _, arr in host.values())
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
        of one more segment): sliced layers concatenate their slices, the
        others take their state from the last whole segment."""
        loaded = [mx.load(str(s.path), return_metadata=True) for s in chain]
        if partial is not None:
            loaded.append(mx.load(str(partial[0].path), return_metadata=True))
        metas = [json.loads(m["streamlx"]) for _, m in loaded]
        kinds = {tuple(x["cls"] for x in m["layers"]) for m in metas}
        if len(kinds) != 1:   # e.g. KV quantized from some point on
            raise ValueError("segments of different cache layouts")
        n = chain[-1].end if chain else 0
        if partial is not None:
            n = partial[0].start + partial[1]
        whole = len(chain)
        out = []
        for i, spec in enumerate(metas[-1]["layers"]):
            prefix = f"{i}."
            names = [k[len(prefix):] for k in loaded[-1][0]
                     if k.startswith(prefix)]
            axes = _LEAF_AXIS.get(spec["cls"].rpartition(".")[2], {})
            leaves = []
            for name in names:
                if spec["sliced"]:
                    axis = axes.get(name, -2)
                    parts = [arrs[prefix + name] for arrs, _ in
                             loaded[:whole]]
                    if partial is not None:
                        v = loaded[-1][0][prefix + name]
                        parts.append(v[..., :partial[1], :] if axis == -2
                                     else v[..., :partial[1]])
                    v = (parts[0] if len(parts) == 1
                         else mx.concatenate(parts, axis=axis))
                else:
                    v = loaded[whole - 1][0][prefix + name]
                leaves.append((name, v))
            src = metas[-1] if spec["sliced"] else metas[whole - 1]
            leaves += list(src["layers"][i]["scalars"].items())
            state = tree_unflatten(leaves)
            cls = _cls(spec["cls"])
            # mlx-vlm's caches (and mlx-lm's older ones) rebuild from both
            if len(inspect.signature(cls.from_state).parameters) > 1:
                meta = src["layers"][i].get("meta", "")
                c = cls.from_state(state, tuple(meta) if isinstance(meta, list)
                                   else meta)
            else:
                c = cls.from_state(state)
            if spec["sliced"]:
                _set_offset(c, n)
            out.append(c)
        return out, n

    # -- the server's interface ----------------------------------------------

    def fetch_nearest_cache(self, model_key, tokens):
        self._open(model_key)
        self.flush()    # the previous turn's segments, if still in flight
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
                m = min(len(t), len(want))
                eq = t[:m] == want[:m]
                same = m if eq.all() else int(np.argmin(eq))
                if same and (partial is None or same > partial[1]):
                    partial = (s, same)
        if not best and partial is None:
            self._cur = (tokens, 0, "")
            return None, tokens
        try:
            cache, n = self._restore(best, partial)
        except Exception:  # unreadable segments = a miss
            log.exception("kv cache: restore failed")
            self._cur = (tokens, 0, "")
            return None, tokens
        now = time.time()
        for s in best + ([partial[0]] if partial else []):
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
