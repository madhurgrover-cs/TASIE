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


class AllVersionsRepoSource(SearchSource):
    """Every indexed commit at once: one hit per lineage (versioned_evolution.py)."""

    def __init__(self, store: Store, repo: str, prefer: str = "latest"):
        from versioned_evolution import AllVersionsIndex

        self.index = AllVersionsIndex(store)
        self.repo = repo
        self.prefer = prefer
        self.name = f"{repo}@all"
        self.label = "All versions"

    def __len__(self) -> int:
        return len(self.index)  # distinct chunks over all versions

    def search(self, query: EncodedQuery, k: int) -> list[Hit]:
        hits = []
        for lin in self.index.search(query.vector, k, prefer=self.prefer).lineages:
            o = lin.representative
            c = o.chunk
            version = self.index.versions[o.version]
            url = None
            if self.repo.count("/") == 1:
                url = f"https://github.com/{self.repo}/blob/{version.commit}/{c['path']}#L{c['start_line']}-L{c['end_line']}"
            hits.append(Hit(
                source=self.name, doc_id=f"lineage:{c['id']}", score=lin.best_score, code=self.index.code(o),
                language=c.get("language", "text"), url=url,
                meta={"path": c["path"], "name": c["name"], "kind": c["kind"], "start_line": c["start_line"],
                      "end_line": c["end_line"], "commit": version.commit, "version": version.label,
                      "timeline": lin.timeline_text(), "versions_present": lin.versions_present,
                      "changed": lin.changed, "shown_score": lin.representative_score}))
        return hits


def load_repo_sources(root: str | Path, all_versions: bool = False) -> tuple[Store, list[SearchSource]]:
    """All indexed commits of a store, oldest first; plus an AllVersionsRepoSource last if asked."""
    store = Store(root)
    repo = store.registry.repo or "repo"
    sources: list[SearchSource] = [VersionedRepoSource(store.load_view(v["commit"]), repo,
                                                       store.registry.label(v["commit"]))
                                   for v in store.registry.list()]
    if all_versions:
        sources.append(AllVersionsRepoSource(store, repo))
    return store, sources
