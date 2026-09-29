"""
retrieval/versioned: chunking, content-hash cache, full vs incremental builds, registry,
search and CLI, on a throwaway git repo (conftest.git_repo) with a torch-free embedder.

    pytest tests/test_versioned.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import CORE_V1
from retrieval.versioned.builder import build, nearest_indexed_base
from retrieval.versioned.chunker import chunk_file, chunk_python, content_hash, normalize_code
from retrieval.versioned.cli import main as cli
from retrieval.versioned.embedder import HashingEmbedder
from retrieval.versioned.gitrepo import GitRepo
from retrieval.versioned.searcher import VersionedSearcher
from retrieval.versioned.store import EmbeddingCache, Store


def by_name(chunks):
    return {c.name: c for c in chunks}


# ---------------------------------------------------------------- chunker
class TestChunker:
    def test_python_chunks(self):
        chunks = by_name(chunk_python("pkg/core.py", CORE_V1))
        assert set(chunks) == {"<module>", "parse_header", "Session", "Session.__init__",
                               "Session.cookie_count", "Session.send", "fetch_json"}
        kinds = {n: c.kind for n, c in chunks.items()}
        assert kinds["parse_header"] == kinds["fetch_json"] == "function"
        assert kinds["Session.send"] == "method" and kinds["Session"] == "class"
        assert all(c.path == "pkg/core.py" and c.language == "python" for c in chunks.values())

    def test_line_ranges_match_source(self):
        lines = CORE_V1.split("\n")
        for c in chunk_python("pkg/core.py", CORE_V1):
            assert c.code == normalize_code("\n".join(lines[c.start_line - 1:c.end_line])), c.name

    def test_decorators_nested_and_headers(self):
        chunks = by_name(chunk_python("pkg/core.py", CORE_V1))
        assert chunks["Session.cookie_count"].code.startswith("@property")
        assert "_prepare" not in chunks and "def _prepare" in chunks["Session.send"].code
        header = chunks["Session"].code
        assert "timeout = 10" in header and "def __init__" not in header
        modules = [c for c in chunk_python("pkg/core.py", CORE_V1) if c.kind == "module"]
        assert len(modules) == 2 and {c.name for c in modules} == {"<module>"}
        assert modules[0].code == "MAX_RETRIES = 3"  # docstring + imports left out
        assert (modules[0].start_line, modules[0].end_line) == (4, 4)
        assert modules[1].code.startswith('if __name__ == "__main__":')

    def test_import_only_module_has_no_chunks(self):
        assert chunk_file("pkg/__init__.py", b"from pkg.core import parse_header\n") == []

    def test_fallbacks(self):
        (py2,) = chunk_file("legacy.py", b'print "python 2 only"\n')
        assert (py2.kind, py2.name, py2.start_line, py2.end_line) == ("file", "legacy.py", 1, 1)
        (md,) = chunk_file("docs/README.md", b"# demo\n\ntext\n")
        assert (md.kind, md.language, md.end_line) == ("file", "markdown", 3)
        assert chunk_file("logo.png", b"\x89PNG\x00\x00") == []
        assert chunk_file("empty.txt", b"   \n") == []
        assert chunk_file("big.txt", b"x" * 600_000) == []
        assert chunk_file("icon.svg", b"<svg/>") == []

    def test_hash_normalisation(self):
        method = "    def f(self):\n        return 1   \n"
        func = "def f(self):\r\n    return 1\r\n\r\n"
        assert normalize_code(method) == normalize_code(func)
        assert content_hash(normalize_code(method)) == content_hash(normalize_code(func))
        assert content_hash("def f(): return 1") != content_hash("def f(): return 2")

    def test_ids_unique_for_redefinitions(self):
        src = "class A:\n    @property\n    def x(self):\n        return 1\n\n    @x.setter\n    def x(self, v):\n        pass\n"
        ids = [c.id for c in chunk_python("a.py", src)]
        assert len(ids) == len(set(ids)) == 3


# ---------------------------------------------------------------- cache
class TestEmbeddingCache:
    def test_embed_only_missing_and_persist(self, tmp_path):
        emb = HashingEmbedder(32)
        cache = EmbeddingCache(tmp_path, emb.model_id)
        texts = {"h1": "def a(): pass", "h2": "def b(): pass"}
        assert cache.embed_missing(texts, emb.embed_documents) == (0, 2)
        assert cache.embed_missing({**texts, "h3": "def c(): pass"}, emb.embed_documents) == (2, 1)
        assert emb.documents_embedded == 3
        cache.save()
        again = EmbeddingCache(tmp_path, emb.model_id)
        assert len(again) == 3
        np.testing.assert_allclose(again.matrix(["h3", "h1"]), cache.matrix(["h3", "h1"]))
        np.testing.assert_allclose(np.linalg.norm(again.matrix(["h1", "h2", "h3"]), axis=1), 1.0, rtol=1e-6)

    def test_model_id_is_part_of_the_key(self, tmp_path):
        EmbeddingCache(tmp_path, "model-a@1").add(["h"], np.ones((1, 4)))
        assert "h" not in EmbeddingCache(tmp_path, "model-a@2")  # unsaved, and a different key anyway
        a = EmbeddingCache(tmp_path, "model-a@1")
        a.add(["h"], np.ones((1, 4)))
        a.save()
        assert "h" in EmbeddingCache(tmp_path, "model-a@1")
        assert "h" not in EmbeddingCache(tmp_path, "model-a@2")

    def test_duplicate_hashes_in_one_add(self, tmp_path):
        c = EmbeddingCache(tmp_path, "m")
        c.add(["h", "h", "g"], np.eye(3, 4))
        assert len(c) == 2 and c.matrix(["h", "g"]).shape == (2, 4)


# ---------------------------------------------------------------- builds
def rows(store: Store, commit: str) -> list[dict]:
    return store.read_chunks(commit)


def comparable(store: Store, commit: str) -> list[tuple]:
    return [(r["id"], r["hash"], r["kind"], r["start_line"], r["end_line"]) for r in rows(store, commit)]


@pytest.fixture()
def repo(git_repo):
    with GitRepo(git_repo["path"]) as r:
        yield r


class TestBuilds:
    def test_full_build(self, repo, tmp_path):
        emb = HashingEmbedder()
        store = Store(tmp_path / "s")
        rep = build(repo, store, emb, "v1", incremental=False)
        assert rep.mode == "full" and rep.base is None and rep.refs == ["v1"]
        expected = [c for e in repo.ls_tree(rep.commit) for c in chunk_file(e.path, repo.read_blob(e.blob))]
        assert rep.n_chunks == len(expected) == len(rows(store, rep.commit))
        assert rep.files_rechunked == 5  # every file in the tree, incl. binary / import-only
        assert rep.n_files == 3  # core.py, legacy.py, README.md produce chunks
        assert rep.chunks_embedded == rep.n_chunks and rep.chunks_reused == 0
        assert emb.documents_embedded == rep.embeddings_new == len({c.hash for c in expected})

    def test_incremental_equals_full_and_embeds_only_changes(self, repo, git_repo, tmp_path):
        shas = git_repo["shas"]
        emb = HashingEmbedder()
        store = Store(tmp_path / "inc")
        build(repo, store, emb, "v1", incremental=False)
        before = emb.documents_embedded
        rep = build(repo, store, emb, "v2")  # base picked automatically
        assert rep.mode == "incremental" and rep.base == shas["v1"]
        assert rep.files_rechunked == 2  # pkg/core.py (M) + pkg/retry.py (A)
        assert rep.files_deleted == 1  # README.md
        assert rep.files_carried == 1  # legacy.py (pkg/__init__.py, logo.png have no chunks)
        v1_hashes = {r["hash"] for r in rows(store, shas["v1"])}
        v2_hashes = {r["hash"] for r in rows(store, shas["v2"])}
        new = v2_hashes - v1_hashes
        assert len(new) == 2  # parse_header (edited) + backoff_delay (added); the rest of core.py is reused
        assert rep.embeddings_new == emb.documents_embedded - before == len(new)
        assert rep.chunks_reused == rep.n_chunks - rep.chunks_embedded == rep.n_chunks - 2

        full = Store(tmp_path / "full")
        build(repo, full, HashingEmbedder(), "v2", incremental=False)
        assert comparable(store, shas["v2"]) == comparable(full, shas["v2"])
        np.testing.assert_allclose(store.load_view("v2").matrix, full.load_view("v2").matrix)

    def test_moved_class_is_not_reembedded(self, repo, git_repo, tmp_path):
        store = Store(tmp_path / "s")
        emb = HashingEmbedder()
        for tag in ("v1", "v2"):
            build(repo, store, emb, tag)
        before = emb.documents_embedded
        rep = build(repo, store, emb, "v3")
        assert rep.base == git_repo["shas"]["v2"]  # nearest indexed ancestor
        assert rep.files_rechunked == 2  # core.py (M), session.py (A)
        assert rep.embeddings_new == 0 and emb.documents_embedded == before
        moved = [r for r in rows(store, rep.commit) if r["path"] == "pkg/session.py"]
        assert {r["name"] for r in moved} == {"Session", "Session.__init__", "Session.cookie_count", "Session.send"}

    def test_incremental_across_several_commits(self, repo, git_repo, tmp_path):
        s = Store(tmp_path / "skip")
        build(repo, s, HashingEmbedder(), "v1", incremental=False)
        rep = build(repo, s, HashingEmbedder(), "v3", base="v1")
        full = Store(tmp_path / "full")
        build(repo, full, HashingEmbedder(), "v3", incremental=False)
        assert rep.base == git_repo["shas"]["v1"]
        assert comparable(s, rep.commit) == comparable(full, rep.commit)

    def test_rebuild_is_a_noop(self, repo, tmp_path):
        store = Store(tmp_path / "s")
        emb = HashingEmbedder()
        build(repo, store, emb, "v1")
        n = emb.documents_embedded
        again = build(repo, store, emb, "v1")
        assert again.mode == "exists" and emb.documents_embedded == n
        forced = build(repo, store, emb, "v1", incremental=False, force=True)
        assert forced.mode == "full" and forced.embeddings_new == 0 and forced.chunks_reused == forced.n_chunks

    def test_model_mismatch_rejected(self, repo, tmp_path):
        store = Store(tmp_path / "s")
        build(repo, store, HashingEmbedder(256), "v1")
        with pytest.raises(ValueError):
            build(repo, Store(tmp_path / "s"), HashingEmbedder(64), "v2")


# ---------------------------------------------------------------- registry
class TestRegistry:
    def test_entries_and_resolution(self, repo, git_repo, tmp_path):
        shas = git_repo["shas"]
        root = tmp_path / "s"
        store = Store(root)
        emb = HashingEmbedder()
        for tag in ("v1", "v2", "v3"):
            build(repo, store, emb, tag)
        reg = Store(root).registry  # reloaded from disk
        listed = reg.list()
        assert [v["refs"] for v in listed] == [["v1"], ["v2"], ["v3"]]
        assert [v["mode"] for v in listed] == ["full", "incremental", "incremental"]
        assert [v["base"] for v in listed] == [None, shas["v1"], shas["v2"]]
        assert listed[1]["git_parent"] == shas["v1"]
        for v in listed:
            assert v["build_s"] >= 0 and v["chunks_reused"] + v["chunks_embedded"] == v["n_chunks"]
            assert v["model_id"] == emb.model_id
            meta = json.loads((root / "versions" / v["commit"] / "meta.json").read_text())
            assert meta["commit"] == v["commit"]
        assert reg.resolve("v2") == reg.resolve(shas["v2"][:7]) == shas["v2"]
        assert reg.label(shas["v2"]) == f"v2 · {shas['v2'][:7]}"
        with pytest.raises(KeyError):
            reg.resolve("v9")
        assert nearest_indexed_base(repo, Store(root), shas["v3"]) == shas["v2"]

    def test_storage_is_shared_between_versions(self, repo, tmp_path):
        store = Store(tmp_path / "s")
        emb = HashingEmbedder()
        for tag in ("v1", "v2", "v3"):
            build(repo, store, emb, tag)
        all_rows = sum(len(store.read_chunks(v["commit"])) for v in store.registry.list())
        distinct = {r["hash"] for v in store.registry.list() for r in store.read_chunks(v["commit"])}
        assert len(store.embeddings) == len(store.content) == len(distinct) < all_rows


# ---------------------------------------------------------------- search + CLI
class TestSearch:
    @pytest.fixture()
    def store(self, repo, tmp_path):
        s = Store(tmp_path / "s")
        emb = HashingEmbedder()
        for tag in ("v1", "v2", "v3"):
            build(repo, s, emb, tag)
        return s

    def test_search_per_commit(self, store):
        searcher = VersionedSearcher(store, HashingEmbedder())
        res = searcher.search("exponential backoff delay for a retry attempt", "v2", k=3)
        top = res.hits[0]
        assert (top.path, top.name, top.kind) == ("pkg/retry.py", "backoff_delay", "function")
        assert (top.start_line, top.end_line) == (1, 3) and top.location == "pkg/retry.py:1-3"
        assert res.label.startswith("v2") and res.total_ms >= res.search_ms >= 0 and res.encode_ms >= 0
        v1 = searcher.search("exponential backoff delay for a retry attempt", "v1", k=10)
        assert all(h.name != "backoff_delay" for h in v1.hits)  # not written yet at v1

    def test_same_code_different_location_per_commit(self, store):
        searcher = VersionedSearcher(store, HashingEmbedder())
        paths = {tag: searcher.search("session cookies count", tag, k=1).hits[0].path for tag in ("v2", "v3")}
        assert paths == {"v2": "pkg/core.py", "v3": "pkg/session.py"}

    def test_errors(self, store):
        searcher = VersionedSearcher(store, HashingEmbedder())
        with pytest.raises(KeyError):
            searcher.search("x", "deadbeef")
        with pytest.raises(ValueError):
            searcher.search("  ", "v1")
        with pytest.raises(ValueError):
            VersionedSearcher(store, HashingEmbedder(64))


class TestCLI:
    def test_index_update_list_query(self, git_repo, tmp_path, capsys):
        base = ["--store", str(tmp_path / "s"), "--embedder", "hashing"]
        path = str(git_repo["path"])
        assert cli(base + ["index", "--repo", path, "--commit", "v1"]) == 0
        assert "[full]" in capsys.readouterr().out
        assert cli(base + ["update", "--repo", path, "--commit", "v3"]) == 0
        out = capsys.readouterr().out
        # v1 -> v3 directly: edited parse_header + new backoff_delay; moved Session is reused
        assert "[incremental from " in out and "(2 new vectors)" in out
        assert cli(base + ["list-versions"]) == 0
        out = capsys.readouterr().out
        assert "v1" in out and "v3" in out and "incremental" in out
        assert cli(base + ["query", "--commit", "v3", "-k", "2", "exponential backoff retry"]) == 0
        out = capsys.readouterr().out
        assert "pkg/retry.py:1-3" in out and "backoff_delay" in out and " ms" in out
        assert cli(base + ["query", "--commit", "v3", "--json", "cookies"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["hits"] and {"path", "name", "start_line", "end_line", "score"} <= set(payload["hits"][0])


# ---------------------------------------------------------------- Space source
def test_space_versioned_source(repo, tmp_path):
    from search import SearchEngine
    from versioned_source import load_repo_sources

    root = tmp_path / "s"
    store = Store(root)
    emb = HashingEmbedder()
    for tag in ("v1", "v3"):
        build(repo, store, emb, tag)

    class Encoder:
        def encode_query(self, text):
            return emb.embed_query(text)

    space_store, sources = load_repo_sources(root)
    assert [s.label.split(" ")[0] for s in sources] == ["v1", "v3"]  # oldest first
    engine = SearchEngine(Encoder(), sources)
    v3 = sources[1]
    res = engine.search("session cookies count", k=2, sources=[v3.name])
    hit = res.hits[0]
    assert hit.source == v3.name and res.n_searched == len(v3)
    assert hit.meta["path"] == "pkg/session.py" and hit.meta["name"].startswith("Session")
    assert hit.meta["start_line"] <= hit.meta["end_line"] and hit.meta["commit"] == v3.commit
    assert hit.url is None  # local repo name is not an owner/name GitHub slug
