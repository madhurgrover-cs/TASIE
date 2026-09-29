"""search(query, commit, k): top-k chunks of one indexed commit, with timings."""
from __future__ import annotations

import time
from dataclasses import dataclass

from retrieval.versioned.embedder import Embedder
from retrieval.versioned.store import Store, VersionView


@dataclass(frozen=True)
class SearchHit:
    path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    score: float
    code: str
    language: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class SearchResponse:
    commit: str
    label: str
    hits: list[SearchHit]
    n_chunks: int
    encode_ms: float
    search_ms: float
    total_ms: float


class VersionedSearcher:
    def __init__(self, store: Store, embedder: Embedder, max_loaded: int = 8):
        if store.model_id and store.model_id != embedder.model_id:
            raise ValueError(f"store was embedded with {store.model_id!r}, query embedder is {embedder.model_id!r}")
        self.store, self.embedder, self.max_loaded = store, embedder, max_loaded
        self._views: dict[str, VersionView] = {}

    def view(self, ref: str) -> VersionView:
        commit = self.store.registry.resolve(ref)
        if commit not in self._views:
            if len(self._views) >= self.max_loaded:
                self._views.pop(next(iter(self._views)))
            self._views[commit] = self.store.load_view(commit)
        return self._views[commit]

    def search(self, query: str, commit: str, k: int = 10) -> SearchResponse:
        if not query.strip():
            raise ValueError("empty query")
        view = self.view(commit)
        t0 = time.perf_counter()
        vec = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        ranked = view.search(vec, k)
        t2 = time.perf_counter()
        hits = [SearchHit(c["path"], c["name"], c["kind"], c["start_line"], c["end_line"], score, c["code"],
                          c.get("language", "text")) for c, score in ranked]
        return SearchResponse(view.commit, self.store.registry.label(view.commit), hits, len(view),
                              (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000)


def search(store: Store, embedder: Embedder, query: str, commit: str, k: int = 10) -> SearchResponse:
    return VersionedSearcher(store, embedder).search(query, commit, k)
