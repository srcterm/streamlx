"""The SSD prompt cache restores a history only for the weights that
computed it, and a damaged or deleted segment costs one prefill: never a
request, and never a second prefill of the same prompt."""
import importlib.util
import os
import shutil
import struct
import tempfile
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

import toy
from streamlx import kvcache
from streamlx.kvcache import PagedPromptCache

TOKENS, NEXT = toy.prompt(12), 7


def _server(root):
    return PagedPromptCache(root, 2**30)


def _store(pc, ck, model, tokens=TOKENS):
    """One request's prefill, stored: what a server does on a miss."""
    got, _ = pc.fetch_nearest_cache(ck.dir, tokens)
    assert got is None
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([tokens]), cache=cache))
    pc.insert_cache(ck.dir, tokens, cache)
    pc.flush()


class _Case(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="streamlx-test-kv-")
        self.addCleanup(shutil.rmtree, self.root, True)

    def restored(self, pc, ck, model, tokens=TOKENS):
        """Tokens the cache restores for tokens + [NEXT]; a restored cache
        must give the logits of a fresh prefill."""
        cache, rest = pc.fetch_nearest_cache(ck.dir, tokens + [NEXT])
        if cache is None:
            self.assertEqual(rest, tokens + [NEXT])
            return 0
        fresh = make_prompt_cache(model)
        mx.eval(model(mx.array([tokens]), cache=fresh))
        self.assertTrue(bool(mx.array_equal(
            model(mx.array([rest]), cache=cache),
            model(mx.array([[NEXT]]), cache=fresh))))
        return pc.last[0]


class ReplacedWeights(_Case):
    def test_same_path_new_weights(self):
        # mlx-lm saves a small model as one model.safetensors (no index
        # file); a copy can keep the files' times; and an edit can sit
        # where the sampled bytes do not look (here: no samples at all)
        cases = [(name, keep_time, 2**16) for name in
                 (toy.SHARDED, "model.safetensors")
                 for keep_time in (False, True)]
        cases.append((toy.SHARDED, True, 0))
        for name, keep_time, sample in cases:
            with self.subTest(file=name, keep_time=keep_time, sample=sample), \
                    mock.patch.object(kvcache, "_SAMPLE", sample):
                ck = toy.Checkpoint(name)
                self.addCleanup(ck.close)
                root = tempfile.mkdtemp(dir=self.root)
                _store(_server(root), ck, ck.ref)
                # a restarted server finds its model's history
                self.assertEqual(
                    self.restored(_server(root), ck, ck.ref), len(TOKENS))
                st = os.stat(ck.file)
                other = ck.rewrite(seed=1)
                self.assertEqual(os.stat(ck.file).st_size, st.st_size)
                if keep_time:
                    os.utime(ck.file, ns=(st.st_atime_ns, st.st_mtime_ns))
                pc = _server(root)
                self.assertEqual(self.restored(pc, ck, other), 0)
                # and the new weights get a cache of their own
                cache = make_prompt_cache(other)
                mx.eval(other(mx.array([TOKENS]), cache=cache))
                pc.insert_cache(ck.dir, TOKENS, cache)
                pc.flush()
                self.assertEqual(
                    self.restored(_server(root), ck, other), len(TOKENS))


class DeletedSegment(_Case):
    def test_missed_lookup(self):
        """A segment whose file is gone, met by a request it cannot serve
        (the same prompt: it ends where the segment does): forgotten at
        that miss, so the prompt is stored again."""
        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        pc = _server(self.root)
        _store(pc, ck, ck.ref)
        (seg,) = pc._segs.values()
        seg.path.unlink()
        _store(pc, ck, ck.ref)                 # a miss, stored again
        self.assertTrue(seg.path.exists())
        self.assertEqual(self.restored(pc, ck, ck.ref), len(TOKENS))


class DamagedSegment(_Case):
    def test_partial_prefix_lookup(self):
        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        # a request that shares its first two tokens with the stored one
        req = TOKENS[:2] + [(t + 1) % toy.VOCAB for t in TOKENS[2:]]
        for left in (4, 40, 300, "in the tokens"):
            with self.subTest(bytes_left=left):
                root = tempfile.mkdtemp(dir=self.root)
                pc = _server(root)
                _store(pc, ck, ck.ref)
                (seg,) = pc._segs.values()
                if left == "in the tokens":    # header whole, data cut
                    with open(seg.path, "rb") as f:
                        left = 8 + struct.unpack("<Q", f.read(8))[0] + 18
                with open(seg.path, "r+b") as f:
                    f.truncate(left)
                with self.assertLogs("streamlx.kvcache", "WARNING"):
                    self.assertEqual(self.restored(pc, ck, ck.ref, req), 0)
                self.assertEqual(len(pc), 0)
                self.assertFalse(seg.path.exists())
                # the request's prefill is stored, and restores
                cache = make_prompt_cache(ck.ref)
                mx.eval(ck.ref(mx.array([req]), cache=cache))
                pc.insert_cache(ck.dir, req + [NEXT], cache)
                pc.flush()
                self.assertEqual(self.restored(pc, ck, ck.ref, req), len(req))


class Layouts(_Case):
    """What a segment holds, and that it restores bit for bit: a window
    (RotatingKVCache) only the rows a next token can reach, a CacheList's
    members each their own way (Inkling 2: the attention rows the segment
    added or a window, beside a convolution state stored whole). Restored at
    the last of a prefill's segment ends, the caches give the logits of the
    ones never stored, for a chunk and then token by token."""
    CHUNK = 5

    def check(self, model_dir, model, make, vocab):
        tokens = [t % vocab for t in toy.prompt(40)]
        more = [t % vocab for t in toy.prompt(8, seed=1)]
        pc = _server(self.root)
        got, _ = pc.fetch_nearest_cache(model_dir, tokens)
        self.assertIsNone(got)
        cache = make()
        for n in range(self.CHUNK, len(tokens) + 1, self.CHUNK):
            mx.eval(model(mx.array([tokens[n - self.CHUNK:n]]), cache=cache))
            pc.save_progress(n, cache)    # a segment a prefill chunk
        pc.flush()
        for seg in pc._segs.values():
            header, meta = pc._header(seg.path)
            shapes = {k: s for k, (_, s) in pc._arrays(header, meta).items()}
            n = seg.end - seg.start
            self.check_rows(shapes, meta["layers"], "", n)
            # no segment holds rows of the tokens before it past a window
            self.assertLessEqual(max(s[-2] for s in shapes.values()
                                     if len(s) == 4), max(n, self.window))
        # what follows a restore: a chunk, or (a template that adds nothing
        # after the history) a token
        for steps in ([more[:5]] + [[t] for t in more[5:]],
                      [[t] for t in more[:3]] + [more[3:]]):
            pc = _server(self.root)       # a restarted server
            cache, rest = pc.fetch_nearest_cache(model_dir, tokens + more)
            self.assertEqual((pc.last[0], rest), (len(tokens), more))
            # its attention rows in buffers with a live one's room
            for c in cache:
                for kv in getattr(c, "caches", (c,)):
                    if type(kv).__name__ == "KVCache":
                        self.assertEqual(kv.offset, len(tokens))
                        self.assertEqual(kv.keys.shape[2] % kv.step, 0)
            ref = make()                  # the caches never stored
            for n in range(self.CHUNK, len(tokens) + 1, self.CHUNK):
                model(mx.array([tokens[n - self.CHUNK:n]]), cache=ref)
            for step in steps:
                a = model(mx.array([step]), cache=cache)
                b = model(mx.array([step]), cache=ref)
                self.assertTrue(bool(mx.array_equal(a, b)), (steps, step))

    def check_rows(self, shapes, specs, prefix, n):
        """Rows along the sequence of each stored array: a segment's own
        rows for a cache that grows along it, at most the window for a
        rotating one."""
        for i, spec in enumerate(specs):
            at = f"{prefix}{i}."
            if "members" in spec:
                self.check_rows(shapes, spec["members"], at, n)
                continue
            rows = [s[-2] for k, s in shapes.items()
                    if k.startswith(at) and len(s) == 4]
            if spec["sliced"]:
                self.assertEqual(set(rows), {n}, at)
            elif spec["cls"].endswith("RotatingKVCache"):
                self.assertTrue(rows, at)
                self.assertLessEqual(max(rows), self.window, at)

    def test_windows(self):
        from mlx_lm.models.cache import RotatingKVCache

        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        # (the chunks are 5 tokens; `keep` rows stay at the window's start)
        for w, keep in ((4, 0), (8, 0), (32, 0), (8, 2)):
            with self.subTest(window=w, keep=keep):
                self.root = tempfile.mkdtemp(dir=self.root)
                self.window = w
                self.check(ck.dir, ck.ref,
                           lambda: [RotatingKVCache(max_size=w, keep=keep)
                                    for _ in ck.ref.layers], toy.VOCAB)

    @unittest.skipUnless(importlib.util.find_spec("mlx_lm.models.inkling2")
                         and mx.default_device() == mx.gpu,
                         "needs an mlx-lm with the inkling2 model, on the GPU")
    def test_cache_lists(self):
        import toy_inkling2 as ink

        ck = ink.Checkpoint()
        self.addCleanup(ck.close)
        m = ck.stock()
        self.window = ink.TEXT["sliding_window_size"]
        self.check(ck.dir, m, m.make_cache, ink.VOCAB)


class Codecs(_Case):
    """Arrays stored in compressed blocks come back as they were: a block
    that would not shrink is stored as it is, a damaged array fails its
    restore; through the server, segments written by each codec (in blocks
    small enough that an array spans several) restore the live cache's rows
    bit for bit, for a prefix that ends where a segment does, inside one, or
    inside the first."""

    def test_blocks(self):
        rng = np.random.default_rng(0)
        rows = rng.integers(0, 2**16, (4, 300), dtype=np.uint16)
        rows[1:3] = 0x3F80        # two rows that shrink, two that do not
        heads = rng.integers(0, 2**16, (1, 2, 5, 60), dtype=np.uint16)
        heads[..., :30] = 0x3F80
        host = {"tokens": ("I32", np.arange(7, dtype=np.int32), None),
                "0.0": ("BF16", rows, None),
                "0.1": ("F32", rng.standard_normal((3, 70), np.float32), None),
                "0.2": ("BOOL", np.ones(5, np.bool_), None),
                "0.3": ("BF16", np.full((1, 3), 0x3F80, np.uint16), None),
                "1.0": ("BF16", heads, -2)}     # along the sequence
        for codec in kvcache._ALGS:
            with self.subTest(codec=codec), \
                    mock.patch.object(kvcache, "_BLOCK", 600):
                meta = {}
                entries, pieces = kvcache._pack(host, codec, meta)
                # numpy, where vImage does not load, stores the same bytes
                with mock.patch.object(kvcache, "_VIMAGE", None):
                    self.assertEqual(
                        b"".join(p.tobytes() for p in
                                 kvcache._pack(host, codec, {})[1]),
                        b"".join(p.tobytes() for p in pieces))
                data = np.concatenate(pieces)
                header, off = {}, 0
                for name, tag, shape, size in entries:
                    header[name] = {"dtype": tag, "shape": shape,
                                    "data_offsets": [off, off + size]}
                    off += size
                for name, (_, arr, _) in host.items():
                    a, b = header[name]["data_offsets"]
                    got = kvcache._unpack(data[a:b], header, meta, name)
                    self.assertEqual(got.shape, arr.shape, name)
                    self.assertEqual(got.tobytes(), arr.tobytes(), name)
                self.assertNotIn("tokens", meta["arrays"])
                stored = [s for s, _ in meta["arrays"]["0.0"]["blocks"]]
                self.assertEqual(len(stored), 4)
                self.assertEqual(stored.count(600), 2)  # stored as they were
                # along the sequence: whole rows (120 bytes) of one line
                # (a head: 600 bytes) a block, here all five
                lines = meta["arrays"]["1.0"]["blocks"]
                self.assertEqual([r for _, r in lines], [600, 600])
                a, b = header["0.0"]["data_offsets"]
                with self.assertRaises(ValueError):
                    kvcache._unpack(data[a:b - 1], header, meta, "0.0")
        with mock.patch.object(kvcache, "_BLOCK", 250):   # 2 rows a block
            self.assertEqual(
                [b.size for b in kvcache._blocks(
                    heads.reshape(-1).view(np.uint8), heads.shape, 2, -2)],
                [240, 240, 120] * 2)

    def test_rows(self):
        ck = toy.Checkpoint()
        self.addCleanup(ck.close)
        tokens = toy.prompt(40)
        for codec in ("none", *kvcache._ALGS):
            with self.subTest(codec=codec), \
                    mock.patch.object(kvcache, "CODEC", codec), \
                    mock.patch.object(kvcache, "_BLOCK", 256):
                root = tempfile.mkdtemp(dir=self.root)
                pc = _server(root)
                pc.fetch_nearest_cache(ck.dir, tokens)
                live = make_prompt_cache(ck.ref)
                for n in range(5, len(tokens) + 1, 5):
                    mx.eval(ck.ref(mx.array([tokens[n - 5:n]]), cache=live))
                    pc.save_progress(n, live)
                pc.flush()
                if codec == "none":
                    self.assertFalse(any("arrays" in pc._header(s.path)[1]
                                         for s in pc._segs.values()))
                for n in (len(tokens), 39, 23, 4):
                    pc = _server(root)          # a restarted server
                    cache, rest = pc.fetch_nearest_cache(
                        ck.dir, tokens[:n] + [NEXT])
                    self.assertEqual((pc.last[0], rest), (n, [NEXT]))
                    for c, ref in zip(cache, live):
                        self.assertEqual(c.offset, n)
                        self.assertEqual(c.keys.shape[2] % c.step, 0)
                        for a, b in ((c.keys, ref.keys),
                                     (c.values, ref.values)):
                            self.assertTrue(bool(mx.array_equal(
                                a[..., :n, :], b[..., :n, :])), (codec, n))


if __name__ == "__main__":
    unittest.main()
