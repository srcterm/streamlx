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
            n = seg.end - seg.start
            self.check_rows(header, meta["layers"], "", n)
            # no segment holds rows of the tokens before it past a window
            self.assertLessEqual(max(h["shape"][-2] for h in header.values()
                                     if len(h.get("shape", ())) == 4),
                                 max(n, self.window))
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

    def check_rows(self, header, specs, prefix, n):
        """Rows along the sequence of each stored array: a segment's own
        rows for a cache that grows along it, at most the window for a
        rotating one."""
        for i, spec in enumerate(specs):
            at = f"{prefix}{i}."
            if "members" in spec:
                self.check_rows(header, spec["members"], at, n)
                continue
            rows = [h["shape"][-2] for k, h in header.items()
                    if k.startswith(at) and len(h["shape"]) == 4]
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


if __name__ == "__main__":
    unittest.main()
