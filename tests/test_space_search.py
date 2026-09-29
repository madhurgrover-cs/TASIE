"""Search logic of the Space with a fake embedder (no torch):  python -m unittest discover -s tests"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fakes import DIM, FakeEncoder, FakeSTModel, hashed_bow
from query_encoder import DEFAULT_QUERY_PREFIX, STQueryEncoder
from search import (
    INDEX_FORMAT_VERSION,
    DenseIndexSource,
    EncodedQuery,
    SearchEngine,
    SearchSource,
    top_k_indices,
)

DOCS = {
    "d1": "def longest_increasing_subsequence(arr): dp = [1] * len(arr)",
    "d2": "def count_islands(grid): visited = set()  # bfs over grid cells",
    "d3": "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
    "d4": "def climb_stairs(n): ways = [1, 1]  # stairs dp modulo",
    "d5": "def is_palindrome(s): return s == s[::-1]",
}


def make_source(name: str = "apps", docs: dict[str, str] = DOCS) -> DenseIndexSource:
    ids = list(docs)
    emb = np.stack([hashed_bow(docs[i]) for i in ids])
    return DenseIndexSource(name, emb, ids, [docs[i] for i in ids],
                            urls=[f"https://example.com/{i}" for i in ids],
                            metas=[{"partition": "test"} for _ in ids])


def write_index(path: Path, docs: dict[str, str] = DOCS, **manifest_overrides) -> dict:
    ids = list(docs)
    emb = np.stack([hashed_bow(docs[i]) for i in ids]).astype(np.float32)
    np.save(path / "embeddings.npy", emb)
    with open(path / "docs.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "code": docs[i], "url": f"https://example.com/{i}",
                                "partition": "train", "language": "python"}) + "\n")
    manifest = {"format_version": INDEX_FORMAT_VERSION, "name": "fake", "model_id": "fake/model",
                "n_docs": len(ids), "dim": DIM,
                "embeddings_sha256": hashlib.sha256((path / "embeddings.npy").read_bytes()).hexdigest()}
    manifest.update(manifest_overrides)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class TopKTest(unittest.TestCase):
    def test_matches_full_sort_with_index_tiebreak(self):
        rng = np.random.default_rng(0)
        scores = rng.integers(0, 5, size=200).astype(np.float32)  # many ties
        expected = sorted(range(200), key=lambda i: (-scores[i], i))
        for k in (1, 7, 50, 200, 500):
            self.assertEqual(top_k_indices(scores, k).tolist(), expected[:k])

    def test_empty(self):
        self.assertEqual(top_k_indices(np.array([], dtype=np.float32), 3).tolist(), [])
        self.assertEqual(top_k_indices(np.ones(3, dtype=np.float32), 0).tolist(), [])


class DenseIndexSourceTest(unittest.TestCase):
    def test_ranks_by_cosine(self):
        src = make_source()
        q = EncodedQuery("x", hashed_bow("count islands in a grid"))
        hits = src.search(q, 3)
        self.assertEqual(hits[0].doc_id, "d2")
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))
        brute = np.stack([hashed_bow(t) for t in DOCS.values()]) @ q.vector
        self.assertAlmostEqual(hits[0].score, float(brute.max()), places=5)
        self.assertEqual(hits[0].url, "https://example.com/d2")
        self.assertEqual(hits[0].meta, {"partition": "test"})
        self.assertEqual(hits[0].source, "apps")

    def test_k_larger_than_corpus(self):
        self.assertEqual(len(make_source().search(EncodedQuery("x", hashed_bow("gcd")), 100)), len(DOCS))

    def test_normalises_embeddings(self):
        src = DenseIndexSource("s", np.array([[3.0, 4.0], [0.0, 2.0]]), ["a", "b"], ["", ""])
        np.testing.assert_allclose(np.linalg.norm(src.embeddings, axis=1), 1.0, rtol=1e-6)

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            make_source().search(EncodedQuery("x", np.ones(DIM + 1, dtype=np.float32)), 3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            DenseIndexSource("s", np.ones((2, 4)), ["a"], ["x", "y"])
        with self.assertRaises(ValueError):
            DenseIndexSource("s", np.ones((2, 4)), ["a", "a"], ["x", "y"])
        with self.assertRaises(ValueError):
            DenseIndexSource("s", np.ones(4), ["a"], ["x"])


class FromDirTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            write_index(Path(d))
            src = DenseIndexSource.from_dir(d, name="APPS corpus")
        self.assertEqual(src.name, "APPS corpus")
        self.assertEqual(len(src), len(DOCS))
        self.assertEqual(src.manifest["model_id"], "fake/model")
        hit = src.search(EncodedQuery("x", hashed_bow("gcd of a and b")), 1)[0]
        self.assertEqual(hit.doc_id, "d3")
        self.assertEqual(hit.meta, {"partition": "train", "language": "python"})

    def test_rejects_bad_index(self):
        cases = [{"n_docs": 99}, {"dim": 3}, {"format_version": 999}, {"embeddings_sha256": "0" * 64}]
        for overrides in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as d:
                write_index(Path(d), **overrides)
                with self.assertRaises(ValueError):
                    DenseIndexSource.from_dir(d)


class OtherSource(SearchSource):
    """A second kind of source, e.g. a versioned git-repo index: fixed hits with version meta."""

    def __init__(self, name: str, score: float):
        from search import Hit
        self.name = name
        self._hit = Hit(source=name, doc_id=f"{name}:src/x.py", score=score, code="x = 1",
                        meta={"commit": "abc123", "path": "src/x.py"})

    def search(self, query, k):
        return [self._hit][:k]

    def __len__(self):
        return 1


class SearchEngineTest(unittest.TestCase):
    def test_end_to_end(self):
        enc = FakeEncoder()
        engine = SearchEngine(enc, [make_source()])
        res = engine.search("longest increasing subsequence", k=2)
        self.assertEqual(res.hits[0].doc_id, "d1")
        self.assertEqual(len(res.hits), 2)
        self.assertEqual(res.n_searched, len(DOCS))
        self.assertEqual(enc.seen, ["longest increasing subsequence"])
        for t in (res.encode_ms, res.search_ms, res.total_ms):
            self.assertGreaterEqual(t, 0.0)
        self.assertGreaterEqual(res.total_ms, res.search_ms)

    def test_merges_sources_by_score(self):
        engine = SearchEngine(FakeEncoder(), [make_source(), OtherSource("repo@v1", 0.99), OtherSource("repo@v2", -1.0)])
        res = engine.search("gcd", k=3)
        self.assertEqual(res.hits[0].source, "repo@v1")
        self.assertEqual(res.hits[0].meta["commit"], "abc123")
        self.assertNotIn("repo@v2", [h.source for h in res.hits])
        self.assertEqual(res.n_searched, len(DOCS) + 2)
        self.assertEqual([h.score for h in res.hits], sorted((h.score for h in res.hits), reverse=True))

    def test_source_selection(self):
        engine = SearchEngine(FakeEncoder(), [make_source(), OtherSource("repo@v1", 0.99)])
        res = engine.search("gcd", k=5, sources=["apps"])
        self.assertEqual({h.source for h in res.hits}, {"apps"})
        self.assertEqual(res.n_searched, len(DOCS))
        with self.assertRaises(ValueError):
            engine.search("gcd", sources=["nope"])

    def test_bad_input(self):
        engine = SearchEngine(FakeEncoder(), [make_source()])
        for q in ("", "   \n"):
            with self.assertRaises(ValueError):
                engine.search(q)
        with self.assertRaises(ValueError):
            engine.search("gcd", k=0)
        with self.assertRaises(ValueError):
            SearchEngine(FakeEncoder(), [make_source("a"), make_source("a")])
        with self.assertRaises(ValueError):
            SearchEngine(FakeEncoder(), [])


class STQueryEncoderTest(unittest.TestCase):
    def test_prefix_and_desc_io_cleaning(self):
        model = FakeSTModel()
        enc = STQueryEncoder(model)
        statement = ("Find the gcd of an array.\n\n-----Input-----\nn and the array.\n\n"
                     "-----Output-----\nOne integer.\n\n-----Examples-----\nInput\n3\n2 4 6\nOutput\n2\n")
        vec = enc.encode_query(statement)
        sent = model.calls[0][0][0]
        self.assertTrue(sent.startswith(DEFAULT_QUERY_PREFIX))
        self.assertIn("Input:\nn and the array.", sent)
        self.assertIn("Output:\nOne integer.", sent)
        self.assertNotIn("Examples", sent)
        self.assertNotIn("2 4 6", sent)
        self.assertTrue(model.calls[0][1].get("normalize_embeddings"))
        self.assertEqual(vec.shape, (DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_short_query_unchanged(self):
        model = FakeSTModel()
        STQueryEncoder(model, query_prefix="Q: ").encode_query("  reverse a linked list ")
        self.assertEqual(model.calls[0][0], ["Q: reverse a linked list"])

    def test_engine_normalises_encoder_output(self):
        class Raw:
            def encode_query(self, text):
                return np.full(DIM, 5.0, dtype=np.float32)
        res = SearchEngine(Raw(), [make_source()]).search("anything", k=1)
        self.assertLessEqual(res.hits[0].score, 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
