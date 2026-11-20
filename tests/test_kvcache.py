"""The SSD prompt cache restores a history only for the weights that
computed it, and a damaged or deleted segment costs one prefill: never a
request, and never a second prefill of the same prompt."""
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


if __name__ == "__main__":
    unittest.main()
