"""
Build a commit's index, fully or incrementally from an indexed commit.

full         chunk every file at the commit.
incremental  start from an indexed base commit (any commit, not only the parent):
             `git diff --name-status base commit` gives the changed files; only
             added / modified / type-changed files are re-chunked, deleted files are
             dropped, every other file's chunks are carried over as-is.
Both then embed through the content-hash cache, so only chunks whose normalised code
was never embedded by this model reach the embedder. An incremental build produces the
same index as a full build of the same commit (tests/test_versioned.py checks this).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from retrieval.versioned.chunker import chunk_file
from retrieval.versioned.embedder import Embedder
from retrieval.versioned.gitrepo import GitRepo
from retrieval.versioned.store import Store


@dataclass
class BuildReport:
    commit: str
    short: str
    mode: str  # full | incremental | exists
    base: str | None
    refs: list[str] = field(default_factory=list)
    n_files: int = 0  # files in the tree that produced chunks
    n_chunks: int = 0
    files_rechunked: int = 0
    files_carried: int = 0  # incremental: files whose chunks came from the base index
    files_deleted: int = 0
    chunks_carried: int = 0
    chunks_embedded: int = 0  # chunks whose code was embedded in this build
    chunks_reused: int = 0  # chunks whose embedding was already in the cache
    embeddings_new: int = 0  # distinct new vectors
    chunk_s: float = 0.0
    embed_s: float = 0.0
    save_s: float = 0.0
    build_s: float = 0.0

    def summary(self) -> str:
        base = f" from {self.base[:7]}" if self.base else ""
        return (f"{self.short} {','.join(self.refs)} [{self.mode}{base}] {self.n_chunks} chunks in {self.n_files} files | "
                f"re-chunked {self.files_rechunked} files, carried {self.files_carried} | "
                f"embedded {self.chunks_embedded} chunks ({self.embeddings_new} new vectors), reused {self.chunks_reused} | "
                f"{self.build_s:.2f}s (chunk {self.chunk_s:.2f}s, embed {self.embed_s:.2f}s, save {self.save_s:.2f}s)")


def _sort_key(c: dict) -> tuple:
    return (c["path"], c["start_line"], c["name"])


def nearest_indexed_base(repo: GitRepo, store: Store, commit: str) -> str | None:
    """Indexed ancestor with the fewest commits in between, else the most recent indexed commit."""
    indexed = [c for c in store.registry.versions if c != commit]
    if not indexed:
        return None
    ancestors = [c for c in indexed if repo.is_ancestor(c, commit)]
    if ancestors:
        return min(ancestors, key=lambda c: repo.commits_between(c, commit))
    # not an ancestor (e.g. another branch): git diff still works between any two commits
    return max(indexed, key=lambda c: store.registry.versions[c].get("committed_at") or "")


def build(repo: GitRepo, store: Store, embedder: Embedder, ref: str, base: str | None = None,
          incremental: bool = True, force: bool = False, refs: list[str] | None = None) -> BuildReport:
    """Index `ref`. base=None + incremental picks the nearest indexed commit automatically."""
    t0 = time.perf_counter()
    commit = repo.resolve(ref)
    if not commit:
        raise ValueError(f"unknown commit {ref!r}")
    if refs is None:  # record tag / branch names, not shas or HEAD
        refs = [] if commit.startswith(ref) or ref.upper().startswith("HEAD") else [ref]
    refs = list(dict.fromkeys(refs))
    if store.model_id is None:
        store.model_id = embedder.model_id
    elif store.model_id != embedder.model_id:
        raise ValueError(f"store uses {store.model_id!r}, embedder is {embedder.model_id!r}")
    if store.registry.repo is None:
        store.registry.repo = _repo_name(repo)

    if commit in store.registry.versions and not force:
        meta = store.registry.versions[commit]
        merged = list(dict.fromkeys(meta.get("refs", []) + refs))
        if merged != meta.get("refs", []):
            meta["refs"] = merged
            store.write_version(commit, store.read_chunks(commit), meta)
            store.save()
        return BuildReport(commit, commit[:7], "exists", meta.get("base"), merged, n_chunks=meta.get("n_chunks", 0))

    if incremental and base is None:
        base = nearest_indexed_base(repo, store, commit)
    base = repo.resolve(base) if (incremental and base) else None
    if base and base not in store.registry.versions:
        raise ValueError(f"base {base[:7]} is not indexed")

    rep = BuildReport(commit, commit[:7], "incremental" if base else "full", base, refs)
    tree = {e.path: e for e in repo.ls_tree(commit)}
    if base:
        changes = repo.diff(base, commit)
        touched = {c.path for c in changes}
        rep.files_deleted = sum(c.status == "D" for c in changes)
        to_chunk = sorted(p for p in touched if p in tree)
        carried = [c for c in store.read_chunks(base) if c["path"] not in touched]
        rep.files_carried = len({c["path"] for c in carried})
        rep.chunks_carried = len(carried)
    else:
        to_chunk = sorted(tree)
        carried = []

    new_chunks = []
    for path in to_chunk:
        new_chunks += [c.to_dict() for c in chunk_file(path, repo.read_blob(tree[path].blob))]
    rep.files_rechunked = len(to_chunk)
    for c in new_chunks:
        store.content.add(c["hash"], c["code"])
    chunks = sorted(carried + new_chunks, key=_sort_key)
    rep.chunk_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    missing_before = {c["hash"] for c in chunks if c["hash"] not in store.embeddings}
    texts = {c["hash"]: store.content.get(c["hash"]) for c in chunks}
    _, rep.embeddings_new = store.embeddings.embed_missing(texts, embedder.embed_documents)
    rep.chunks_embedded = sum(c["hash"] in missing_before for c in chunks)
    rep.chunks_reused = len(chunks) - rep.chunks_embedded
    rep.embed_s = time.perf_counter() - t1

    rep.n_chunks = len(chunks)
    rep.n_files = len({c["path"] for c in chunks})
    info = repo.commit_info(commit)
    rep.build_s = time.perf_counter() - t0  # chunk + embed; writing to disk is save_s
    meta = {**asdict(rep), **{k: info[k] for k in ("git_parent", "committed_at", "subject")},
            "model_id": embedder.model_id, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    t2 = time.perf_counter()
    store.write_version(commit, chunks, meta)
    store.save()
    rep.save_s = time.perf_counter() - t2
    return rep


def _repo_name(repo: GitRepo) -> str:
    try:
        url = repo.git("config", "--get", "remote.origin.url").strip()
    except Exception:
        url = ""
    name = url.rstrip("/").removesuffix(".git")
    if "github.com" in name:
        return name.split("github.com")[-1].lstrip(":/")
    return name or repo.path.resolve().name
