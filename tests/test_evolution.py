"""
Evolutionary retrieval (retrieval/versioned/evolution.py): search all versions at once,
grouped into lineages. Fake embedders only; conftest.git_repo has 4 versions:
  v1  pkg/core.py (parse_header, Session.*, fetch_json)
  v2  parse_header edited, pkg/retry.py added
  v3  Session moved, unchanged, to pkg/session.py
  v4  pkg/ -> src/pkg/, fetch_json edited
"""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.versioned.builder import build
from retrieval.versioned.embedder import HashingEmbedder
from retrieval.versioned.evolution import (
    ABSENT, NEW, SAME, AllVersionsIndex, canonical_path, identity_key, search_all_versions,
)
from retrieval.versioned.gitrepo import GitRepo
from retrieval.versioned.store import Store


@pytest.fixture(scope="module")
def built(git_repo, tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("evo"))
    emb = HashingEmbedder()
    with GitRepo(git_repo["path"]) as repo:
        for tag in ("v1", "v2", "v3", "v4"):
            build(repo, store, emb, tag)
    return store, emb


def lineage_named(result, name):
    matches = [lin for lin in result.lineages if any(k[0] == "name" and k[3] == name for k in lin.keys)]
    assert matches, [lin.keys for lin in result.lineages]
    return matches[0]


def states(lin):
    return [s.state for s in lin.timeline]


class TestHelpers:
    def test_canonical_path(self):
        assert canonical_path("src/requests/utils.py") == canonical_path("requests/utils.py") == "requests/utils.py"
        assert canonical_path("src") == "src" and canonical_path("tests/src/x.py") == "tests/src/x.py"
        assert canonical_path("test_requests.py") == canonical_path("tests/test_requests.py") == "tests/test_requests.py"
        assert canonical_path("setup.py") == "setup.py" and canonical_path("pkg/test_x.py") == "pkg/test_x.py"

    def test_identity_key(self):
        a = {"path": "src/pkg/core.py", "kind": "function", "name": "f", "hash": "h1"}
        b = {"path": "pkg/core.py", "kind": "function", "name": "f", "hash": "h2"}
        assert identity_key(a) == identity_key(b)
        m = {"path": "pkg/core.py", "kind": "module", "name": "<module>", "hash": "h3"}
        assert identity_key(m) == ("hash", "h3", "pkg/core.py")


class TestLineages:
    def test_edited_function_timeline_and_src_move(self, built):
        store, emb = built
        res = search_all_versions(store, emb, "parse an HTTP header line into key and value", k=5)
        lin = lineage_named(res, "parse_header")
        assert res.lineages[0] is lin
        assert states(lin) == [NEW, NEW, SAME, SAME]  # edited in v2; v4 moved to src/ with the same code
        assert lin.changed and lin.n_distinct_code == 2
        assert lin.versions_present == ["v1", "v2", "v3", "v4"]
        assert lin.timeline_text() == "v1 ● v2 ● v3 ○ v4 ○"
        # the original v1 code scores 0.786 vs 0.736 for the edit: beyond the 0.01 tie, so v1 is shown
        v1, v2, v3, v4 = (s.score for s in lin.timeline)
        assert v1 - v2 > 0.01 and v2 == v3 == v4
        assert lin.representative.version == 0 and lin.representative_score == lin.best_score
        # with a wide tie window "latest" shows v4 (same code as v2/v3), under its new src/ path
        wide = lineage_named(search_all_versions(store, emb, "parse an HTTP header line into key and value",
                                                 k=5, tie=0.1), "parse_header")
        assert wide.representative.version == 3 and wide.representative.chunk["path"] == "src/pkg/core.py"

    def test_moved_class_is_one_lineage(self, built):
        store, emb = built
        res = search_all_versions(store, emb, "session send request prepare", k=10)
        send = [lin for lin in res.lineages if any(k[0] == "name" and k[3] == "Session.send" for k in lin.keys)]
        assert len(send) == 1  # pkg/core.py (v1, v2) + pkg/session.py (v3) + src/pkg/session.py (v4)
        lin = send[0]
        assert {k[1] for k in lin.keys} == {"pkg/core.py", "pkg/session.py"}  # merged by near-dup, not by name
        assert states(lin) == [NEW, SAME, SAME, SAME] and not lin.changed

    def test_added_and_changed_later(self, built):
        store, emb = built
        res = search_all_versions(store, emb, "exponential backoff delay retry attempt", k=3)
        assert states(lineage_named(res, "backoff_delay")) == [ABSENT, NEW, SAME, SAME]
        res = search_all_versions(store, emb, "fetch json url", k=5)
        assert states(lineage_named(res, "fetch_json")) == [NEW, SAME, SAME, NEW]

    def test_one_entry_per_lineage_and_ordering(self, built):
        store, emb = built
        res = search_all_versions(store, emb, "session cookies count", k=10)
        ids = [lin.id for lin in res.lineages]
        assert len(ids) == len(set(ids))
        keys = [k for lin in res.lineages for k in lin.keys]
        assert len(keys) == len(set(keys))
        scores = [lin.best_score for lin in res.lineages]
        assert scores == sorted(scores, reverse=True)

    def test_no_reembedding(self, built):
        store, emb = built
        before = emb.documents_embedded
        idx = AllVersionsIndex(store)
        idx.search(emb.embed_query("cookies"), k=5)
        assert emb.documents_embedded == before
        assert len(idx) == len(store.embeddings) < idx.n_rows  # scored per distinct chunk

    def test_duplicate_metric(self, built):
        store, emb = built
        res = search_all_versions(store, emb, "session cookies count", k=10)
        d = res.duplicates(10)
        assert d["flat_duplicates"] > 0  # the same Session methods, once per version
        assert d["grouped_duplicates"] == 0
        assert d["grouped_unique"] >= d["flat_unique"]
        assert d["flat_duplicates"] + d["flat_unique"] == 10
        small = search_all_versions(store, emb, "session cookies count", k=3)  # metric ignores k
        assert len(small.lineages) == 3 and small.duplicates(10) == d

    def test_bad_input(self, built):
        store, emb = built
        with pytest.raises(ValueError):
            search_all_versions(store, emb, "  ")
        with pytest.raises(ValueError):
            AllVersionsIndex(store).search(emb.embed_query("x"), prefer="oldest")
        with pytest.raises(ValueError):
            AllVersionsIndex(store).search(np.ones(3))


# ---------------------------------------------------------------- synthetic store: exact scores
class FakeStore:
    """Duck-typed store: versions -> chunks, vectors chosen per hash."""

    def __init__(self, versions: dict[str, list[dict]], vectors: dict[str, list[float]]):
        self._versions = versions
        self.registry = self
        self.embeddings = self
        self.content = self
        self._vec = {h: np.asarray(v, dtype=np.float32) / np.linalg.norm(v) for h, v in vectors.items()}

    def list(self):
        return [{"commit": c, "short": c, "refs": [c], "committed_at": f"202{i}"} for i, c in enumerate(self._versions)]

    def read_chunks(self, commit):
        return self._versions[commit]

    def matrix(self, hashes):
        return np.stack([self._vec[h] for h in hashes])

    def get(self, h):
        return f"code {h}"


def chunk(name, h, path="m.py", kind="function", language="python", line=1):
    return {"id": f"{path}#{name}@{line}", "path": path, "name": name, "kind": kind,
            "start_line": line, "end_line": line + 1, "hash": h, "language": language}


def at_cos(c):
    """Two 2-D unit vectors with cosine c between them, both scoring ~0.99 against (1, 0)."""
    t = np.arccos(c) / 2
    return [np.cos(0.1 - t), np.sin(0.1 - t)], [np.cos(0.1 + t), np.sin(0.1 + t)]


def unit(angle):
    """2-D unit vector: its dot product with (1, 0) is cos(angle)."""
    return [np.cos(angle), np.sin(angle)]


class TestRanking:
    def test_prefer_latest_within_tie(self):
        # lineage f: v1 scores 0.80, v2 scores 0.795 (edited). Query = (1, 0).
        store = FakeStore({"v1": [chunk("f", "a")], "v2": [chunk("f", "b")]},
                          {"a": unit(np.arccos(0.80)), "b": unit(np.arccos(0.795))})
        idx = AllVersionsIndex(store)
        q = np.array([1.0, 0.0])
        latest = idx.search(q, k=1).lineages[0]
        assert latest.representative.version == 1 and latest.best_score == pytest.approx(0.80)
        assert latest.representative_score == pytest.approx(0.795)
        by_score = idx.search(q, k=1, prefer="score").lineages[0]
        assert by_score.representative.version == 0
        assert idx.search(q, k=1, tie=0.001).lineages[0].representative.version == 0  # outside the tie window

    def test_coexisting_near_duplicates_stay_separate(self):
        # f and g have identical vectors but both exist in v1: two different functions
        store = FakeStore({"v1": [chunk("f", "a"), chunk("g", "a2")], "v2": [chunk("g", "a2")]},
                          {"a": [1, 0.1], "a2": [1, 0.1]})
        res = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5)
        assert len(res.lineages) == 2

    def test_renamed_near_duplicate_merges(self):
        # f (v1) renamed to f2 (v2), almost the same vector: one lineage
        store = FakeStore({"v1": [chunk("f", "a")], "v2": [chunk("f2", "b")]},
                          {"a": [1, 0.10], "b": [1, 0.11]})
        res = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5)
        assert len(res.lineages) == 1
        assert [s.state for s in res.lineages[0].timeline] == [NEW, NEW]
        # below the threshold they stay apart
        store2 = FakeStore({"v1": [chunk("f", "a")], "v2": [chunk("f2", "b")]}, {"a": [1, 0], "b": [0.5, 1]})
        assert len(AllVersionsIndex(store2).search(np.array([1.0, 0.0]), k=5).lineages) == 2

    def test_identical_boilerplate_elsewhere_does_not_merge(self):
        # Response.__enter__ vs ConnectionPool.__enter__ in requests: same code, disjoint versions,
        # but different name AND different file -> not one piece of code
        store = FakeStore({"v1": [chunk("Pool.__enter__", "a", "vendor/pool.py")],
                           "v2": [chunk("Response.__enter__", "a2", "models.py")]},
                          {"a": [1, 0.1], "a2": [1, 0.1]})
        assert len(AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5).lineages) == 2

    def test_moved_file_same_name_merges(self):
        store = FakeStore({"v1": [chunk("T.test_x", "a", "test_requests.py")],
                           "v2": [chunk("T.test_x", "b", "tests/test_requests.py")]},
                          {"a": [1, 0.10], "b": [1, 0.12]})
        (lin,) = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5).lineages
        assert [s.state for s in lin.timeline] == [NEW, NEW]

    def test_module_blocks_merge_only_within_a_file(self):
        store = FakeStore({"v1": [chunk("<module>", "a", "a.py", "module")],
                           "v2": [chunk("<module>", "b", "a.py", "module"), chunk("<module>", "c", "b.py", "module")]},
                          {"a": [1, 0.10], "b": [1, 0.11], "c": [1, 0.10]})
        res = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5)
        paths = sorted(sorted({o.chunk["path"] for s in lin.timeline if s.occurrence for o in [s.occurrence]})
                       for lin in res.lineages)
        assert paths == [["a.py"], ["b.py"]]

    def test_renamed_test_class_and_moved_test_file(self):
        # v2.0.0 test_requests.py::RequestsTestCase.test_x -> tests/test_requests.py::TestRequests.test_x
        a, b = at_cos(0.92)
        store = FakeStore({"v1": [chunk("RequestsTestCase.test_x", "a", "test_requests.py", "method")],
                           "v2": [chunk("TestRequests.test_x", "b", "tests/test_requests.py", "method")]},
                          {"a": a, "b": b})
        (lin,) = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5).lineages
        assert [s.state for s in lin.timeline] == [NEW, NEW]
        # below rename_threshold (0.9): stays split
        a, b = at_cos(0.85)
        store.__init__(store._versions, {"a": a, "b": b})
        assert len(AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5).lineages) == 2

    def test_class_rename_needs_same_method_name_and_file(self):
        a, b = at_cos(0.93)  # above 0.9, below 0.95
        other_method = FakeStore({"v1": [chunk("Old.test_x", "a", "t.py", "method")],
                                  "v2": [chunk("New.test_y", "b", "t.py", "method")]}, {"a": a, "b": b})
        assert len(AllVersionsIndex(other_method).search(np.array([1.0, 0.0]), k=5).lineages) == 2
        other_file = FakeStore({"v1": [chunk("Old.test_x", "a", "t.py", "method")],
                                "v2": [chunk("New.test_x", "b", "u.py", "method")]}, {"a": a, "b": b})
        assert len(AllVersionsIndex(other_file).search(np.array([1.0, 0.0]), k=5).lineages) == 2

    def test_single_module_block_is_keyed_by_file(self):
        # requests/auth.py: one <module> block per version, cosine only 0.86 between versions
        a, b = at_cos(0.86)
        store = FakeStore({"v1": [chunk("<module>", "a", "requests/auth.py", "module")],
                           "v2": [chunk("<module>", "b", "src/requests/auth.py", "module", line=5)]},
                          {"a": a, "b": b})
        (lin,) = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=5).lineages
        assert [s.state for s in lin.timeline] == [NEW, NEW]
        # two blocks in the file: content keys, merged within the file only above 0.9
        a2, b2 = at_cos(0.86)
        multi = FakeStore({"v1": [chunk("<module>", "a", "x.py", "module"), chunk("<module>", "z", "x.py", "module", line=9)],
                           "v2": [chunk("<module>", "b", "x.py", "module")]},
                          {"a": a2, "b": b2, "z": [0.0, 1.0]})
        assert len(AllVersionsIndex(multi).search(np.array([1.0, 0.0]), k=5).lineages) == 3

    def test_module_and_non_python_files_rank_below_functions(self):
        store = FakeStore({"v1": [chunk("<module>", "m", "a.py", "module"),
                                  chunk("README.md", "r", "README.md", "file", "markdown"),
                                  chunk("legacy.py", "l", "legacy.py", "file", "python"),
                                  chunk("f", "f", "b.py")]},
                          {"m": unit(np.arccos(0.80)), "r": unit(np.arccos(0.79)),
                           "l": unit(np.arccos(0.77)), "f": unit(np.arccos(0.76))})
        idx = AllVersionsIndex(store)
        q = np.array([1.0, 0.0])
        ranked = idx.search(q, k=4).lineages
        # module 0.80-0.05=0.75, README 0.79-0.05=0.74; py2 file-level chunk is Python: no penalty
        assert [lin.representative.chunk["name"] for lin in ranked] == ["legacy.py", "f", "<module>", "README.md"]
        assert [lin.penalty for lin in ranked] == [0.0, 0.0, 0.05, 0.05]
        assert ranked[2].best_score == pytest.approx(0.80) and ranked[2].rank_score == pytest.approx(0.75)
        raw = idx.search(q, k=4, kind_penalty=0.0).lineages
        assert [lin.representative.chunk["name"] for lin in raw] == ["<module>", "README.md", "legacy.py", "f"]

    def test_equal_best_scores_newest_first(self):
        store = FakeStore({"v1": [chunk("old", "a", "a.py")], "v2": [chunk("new", "b", "b.py")]},
                          {"a": [1, 0], "b": [1, 0]})
        # identical vectors, disjoint versions -> would merge; lower the threshold's reach with dup_threshold=1.01
        res = AllVersionsIndex(store).search(np.array([1.0, 0.0]), k=2, dup_threshold=1.01)
        assert [lin.representative.chunk["name"] for lin in res.lineages] == ["new", "old"]


# ---------------------------------------------------------------- Space: All versions source + UI
def test_space_all_versions(built):
    gradio = pytest.importorskip("gradio")
    import app
    from search import SearchEngine
    from test_space_search import make_source
    from versioned_source import load_repo_sources

    store, emb = built
    _, sources = load_repo_sources(store.root, all_versions=True)
    everything = sources[-1]
    assert everything.name.endswith("@all") and len(everything) == len(store.embeddings)

    class Enc:
        def encode_query(self, t):
            return emb.embed_query(t)

    engine = SearchEngine(Enc(), [make_source(app.APPS)] + sources)
    info = app.repo_info("org/repo", sources)
    assert info.versions[0][0] == "All versions"
    assert info.default.startswith("v4")  # the newest commit stays the default
    demo = app.create_demo(engine, info)
    dropdown = [b for b in demo.blocks.values() if isinstance(b, gradio.Dropdown)][0]
    assert dropdown.value == info.default
    assert [c[0] for c in dropdown.choices][0] == "All versions"

    res = engine.search("parse an HTTP header line into key and value", k=3, sources=[everything.name])
    names = [h.meta["name"] for h in res.hits]
    assert names[0] == "parse_header" and len(names) == len(set(names))
    _, updates = app.format_result(res, " of org/repo, all versions")
    header = updates[1]["value"]
    assert "`v1 ● v2 ● v3 ○ v4 ○`" in header and "shown: v1" in header and "code changed" in header
