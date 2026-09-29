"""
Versioned git-repo index as a SearchSource (one source per indexed commit).

The store is built by retrieval/versioned (see DEPLOY_SPACE.md) and read with
versioned_store.py, a verbatim copy of retrieval/versioned/store.py. Chunk code and
vectors are shared between commits, so loading every version costs little more than one.
"""
from __future__ import annotations

from pathlib import Path

from search import EncodedQuery, Hit, SearchSource
from versioned_store import Store, VersionView


class VersionedRepoSource(SearchSource):
    def __init__(self, view: VersionView, repo: str, label: str):
        self.view = view
        self.repo = repo
        self.commit = view.commit
        self.label = label  # e.g. "v2.32.3 · 61e2240"
        self.name = f"{repo}@{view.commit[:7]}"

    def __len__(self) -> int:
        return len(self.view)

    def _url(self, c: dict) -> str | None:
        if "/" not in self.repo or self.repo.count("/") != 1:
            return None  # not an owner/name GitHub slug
        return f"https://github.com/{self.repo}/blob/{self.commit}/{c['path']}#L{c['start_line']}-L{c['end_line']}"

    def search(self, query: EncodedQuery, k: int) -> list[Hit]:
        return [
            Hit(source=self.name, doc_id=c["id"], score=score, code=c["code"], language=c.get("language", "text"),
                url=self._url(c),
                meta={"path": c["path"], "name": c["name"], "kind": c["kind"], "start_line": c["start_line"],
                      "end_line": c["end_line"], "commit": self.commit, "version": self.label})
            for c, score in self.view.search(query.vector, k)
        ]


def load_repo_sources(root: str | Path) -> tuple[Store, list[VersionedRepoSource]]:
    """All indexed commits of a store, oldest first."""
    store = Store(root)
    sources = [VersionedRepoSource(store.load_view(v["commit"]), store.registry.repo or "repo",
                                   store.registry.label(v["commit"]))
               for v in store.registry.list()]
    return store, sources
