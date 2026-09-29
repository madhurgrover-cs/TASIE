"""
Evolutionary retrieval: search every indexed version of a repo at once and show each
piece of code once, as a "lineage" through the versions.

numpy only, and no imports from this repo (the store is passed in, duck-typed), so
space/versioned_evolution.py can be a byte-for-byte copy (tests/test_space_sync.py).

Scoring reuses the stored vectors: every *distinct* chunk (content hash) is scored once,
so identical code in several versions costs one dot product and is never re-embedded.

Lineages
  1. identity: same canonical path + qualified name + kind. The canonical path drops a
     leading "src/" (requests/utils.py and src/requests/utils.py are the same file).
     Top-level "<module>" blocks have no stable name, so they start as one lineage per
     (hash, path).
  2. near-duplicates: two lineages merge when some pair of their chunks has cosine
     similarity > dup_threshold (0.95) and they are plausibly the same code over time:
       - same qualified name in another file (moved: test_requests.py -> tests/...,
         urllib3/util.py -> util/url.py, charade -> chardet), or
       - same file under another name (renamed in place);
       module blocks only merge within the same file.
     Guards: lineages that coexist in the same version are never merged (two snippets
     present at the same time are different code), and identical boilerplate in unrelated
     places (`def __enter__(self): return self` in two classes) doesn't merge.

Ranking: one entry per lineage, ordered by its best score. The version shown for a
lineage is, by default, the latest version whose score is within `tie` (0.01) of the
best ("latest"); prefer="score" shows the best-scoring version instead. Lineages with
equal best scores are ordered newest-first.

Timeline per lineage, oldest to newest: "●" the code is new or changed in that version
(content hash differs from its previous appearance), "○" unchanged, "–" absent.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NEW, SAME, ABSENT = "●", "○", "–"


def canonical_path(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[1:]) if len(parts) > 1 and parts[0] == "src" else path


def identity_key(chunk: dict) -> tuple:
    if chunk["kind"] == "module":  # no stable name: identified by content (and near-dup merging)
        return ("hash", chunk["hash"], canonical_path(chunk["path"]))
    return ("name", canonical_path(chunk["path"]), chunk["kind"], chunk["name"])


@dataclass(frozen=True)
class Version:
    index: int  # 0 = oldest
    commit: str
    label: str  # first ref (tag) or short sha
    date: str


@dataclass(frozen=True)
class Occurrence:
    version: int  # Version.index
    chunk: dict  # chunks.jsonl row (path, name, kind, start_line, end_line, hash, id, ...)
    row: int  # index into AllVersionsIndex.hashes


@dataclass
class VersionState:
    version: Version
    state: str  # NEW / SAME / ABSENT
    score: float | None = None
    occurrence: Occurrence | None = None


@dataclass
class Lineage:
    id: int
    best_score: float
    representative: Occurrence  # the version shown
    representative_score: float
    timeline: list[VersionState]
    n_distinct_code: int  # distinct content hashes across versions
    keys: list[tuple] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.n_distinct_code > 1

    @property
    def versions_present(self) -> list[str]:
        return [s.version.label for s in self.timeline if s.state != ABSENT]

    def timeline_text(self) -> str:
        return " ".join(f"{s.version.label} {s.state}" for s in self.timeline)


@dataclass
class AllVersionsResult:
    lineages: list[Lineage]
    flat: list[tuple[Occurrence, float]]  # ungrouped top-k: every (version, chunk) row
    flat_lineage_ids: list[int]  # lineage of each flat hit
    n_rows: int  # chunk rows over all versions
    n_distinct: int  # distinct chunks scored
    ranked: list[Lineage] = field(default_factory=list)  # all candidate lineages, ranked (lineages = top k)

    def duplicates(self, k: int = 10) -> dict[str, int]:
        """How many of the top-k are repeats of a function already shown above them."""
        flat_ids = self.flat_lineage_ids[:k]
        grouped_ids = [lin.id for lin in (self.ranked or self.lineages)[:k]]
        return {"k": k,
                "flat_duplicates": len(flat_ids) - len(set(flat_ids)), "flat_unique": len(set(flat_ids)),
                "grouped_duplicates": len(grouped_ids) - len(set(grouped_ids)),
                "grouped_unique": len(set(grouped_ids))}


class AllVersionsIndex:
    """Every indexed version of a store, scored over distinct chunks."""

    def __init__(self, store: Any):
        self.store = store
        self.versions: list[Version] = []
        occ_by_hash: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        self.n_rows = 0
        for i, v in enumerate(store.registry.list()):
            refs = v.get("refs") or []
            self.versions.append(Version(i, v["commit"], refs[0] if refs else v["short"],
                                         (v.get("committed_at") or "")[:10]))
            for c in store.read_chunks(v["commit"]):
                occ_by_hash[c["hash"]].append((i, c))
                self.n_rows += 1
        self.hashes = list(occ_by_hash)
        self.matrix = np.asarray(store.embeddings.matrix(self.hashes), dtype=np.float32)
        self.occurrences: list[list[Occurrence]] = [
            [Occurrence(vi, c, row) for vi, c in occ_by_hash[h]] for row, h in enumerate(self.hashes)]
        self.by_key: dict[tuple, list[Occurrence]] = defaultdict(list)
        for occs in self.occurrences:
            for o in occs:
                self.by_key[identity_key(o.chunk)].append(o)

    def __len__(self) -> int:
        return len(self.hashes)

    def code(self, occ: Occurrence) -> str:
        return self.store.content.get(occ.chunk["hash"])

    def search(self, vector: np.ndarray, k: int = 10, prefer: str = "latest", tie: float = 0.01,
               dup_threshold: float = 0.95, pool: int | None = None) -> AllVersionsResult:
        if prefer not in ("latest", "score"):
            raise ValueError("prefer must be 'latest' or 'score'")
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.matrix.shape[1]}")
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.matrix @ q
        pool = pool or max(10 * k, 100)
        top_rows = np.argsort(-scores, kind="stable")[:pool]

        # ---- items = identity keys touched by the candidate pool, with ALL their occurrences
        items: dict[tuple, list[Occurrence]] = {}
        for row in top_rows:
            for o in self.occurrences[row]:
                key = identity_key(o.chunk)
                if key not in items:
                    items[key] = self.by_key[key]
        keys = list(items)
        parent = list(range(len(keys)))
        vsets = [{o.version for o in items[k_]} for k_ in keys]

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def compatible(a: tuple, b: tuple) -> bool:
            """Could items a and b be one piece of code over time (moved or renamed)?"""
            # keys: ("name", canonical path, kind, qualified name) | ("hash", hash, canonical path)
            if a[0] == "hash" or b[0] == "hash":  # module blocks: only with module blocks of the same file
                return a[0] == b[0] == "hash" and a[2] == b[2]
            return a[3] == b[3] or a[1] == b[1]  # same qualified name (moved), or same file (renamed)

        # ---- near-duplicate merging: max chunk-pair cosine between items, highest first
        item_rows = [sorted({o.row for o in items[k_]}) for k_ in keys]
        all_rows = sorted({r for rows in item_rows for r in rows})
        pos = {r: j for j, r in enumerate(all_rows)}
        sub = self.matrix[all_rows]
        cos = sub @ sub.T
        pairs = []
        for a in range(len(keys)):
            ia = [pos[r] for r in item_rows[a]]
            for b in range(a + 1, len(keys)):
                if not compatible(keys[a], keys[b]):
                    continue
                ib = [pos[r] for r in item_rows[b]]
                c = float(cos[np.ix_(ia, ib)].max())
                if c > dup_threshold:
                    pairs.append((c, a, b))
        for _, a, b in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
            ra, rb = find(a), find(b)
            if ra != rb and not (vsets[ra] & vsets[rb]):  # coexisting code is not one lineage
                parent[rb] = ra
                vsets[ra] |= vsets[rb]

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(keys)):
            groups[find(i)].append(i)

        # ---- one Lineage per group
        lineages: list[Lineage] = []
        lineage_of: dict[tuple[int, str], int] = {}  # (version, chunk id) -> lineage
        for gid, (root, members) in enumerate(sorted(groups.items())):
            occs = [o for i in members for o in items[keys[i]]]
            best_per_version: dict[int, tuple[float, Occurrence]] = {}
            for o in occs:
                s = float(scores[o.row])
                lineage_of[(o.version, o.chunk["id"])] = gid
                cur = best_per_version.get(o.version)
                if cur is None or s > cur[0]:
                    best_per_version[o.version] = (s, o)
            best = max(s for s, _ in best_per_version.values())
            if prefer == "latest":
                rep_v = max(v for v, (s, _) in best_per_version.items() if s >= best - tie)
            else:
                rep_v = max(v for v, (s, _) in best_per_version.items() if s == best)
            timeline, prev_hash = [], None
            for v in self.versions:
                if v.index in best_per_version:
                    s, o = best_per_version[v.index]
                    state = NEW if o.chunk["hash"] != prev_hash else SAME
                    prev_hash = o.chunk["hash"]
                    timeline.append(VersionState(v, state, s, o))
                else:
                    timeline.append(VersionState(v, ABSENT))
            lineages.append(Lineage(
                id=gid, best_score=best, representative=best_per_version[rep_v][1],
                representative_score=best_per_version[rep_v][0], timeline=timeline,
                n_distinct_code=len({o.chunk["hash"] for _, o in best_per_version.values()}),
                keys=[keys[i] for i in members]))
        lineages.sort(key=lambda lin: (-lin.best_score, -lin.representative.version, lin.id))

        # ---- ungrouped baseline: every (version, chunk) row, as if each version were searched and merged
        flat: list[tuple[Occurrence, float]] = []
        for row in top_rows:
            for o in sorted(self.occurrences[row], key=lambda o: -o.version):
                flat.append((o, float(scores[row])))
        flat.sort(key=lambda t: (-t[1], -t[0].version))
        flat = flat[:max(k, 10)]
        flat_ids = [lineage_of[(o.version, o.chunk["id"])] for o, _ in flat]
        return AllVersionsResult(lineages[:k], flat, flat_ids, self.n_rows, len(self.hashes), lineages)


def search_all_versions(store: Any, embedder: Any, query: str, k: int = 10, **kwargs) -> AllVersionsResult:
    """Embed the query once, search every indexed version of `store` together."""
    if not query.strip():
        raise ValueError("empty query")
    return AllVersionsIndex(store).search(embedder.embed_query(query), k, **kwargs)
