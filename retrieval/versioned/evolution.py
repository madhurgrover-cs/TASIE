"""
Evolutionary retrieval: search every indexed version of a repo at once and show each
piece of code once, as a "lineage" through the versions.

numpy only, and no imports from this repo (the store is passed in, duck-typed), so
space/versioned_evolution.py can be a byte-for-byte copy (tests/test_space_sync.py).

Scoring reuses the stored vectors: every *distinct* chunk (content hash) is scored once,
so identical code in several versions costs one dot product and is never re-embedded.

Lineages
  1. identity: same canonical path + qualified name + kind. The canonical path drops a
     leading "src/" (src/requests/utils.py == requests/utils.py) and puts root test files
     under tests/ (test_requests.py == tests/test_requests.py). A file's top-level
     "<module>" block is keyed by the file when the file has exactly one such block in
     that version; otherwise each block starts as its own lineage (hash, path).
  2. near-duplicates: two lineages merge when some pair of their chunks has cosine
     similarity above the threshold for that kind of pair:
       - same qualified name in another file (moved: urllib3/util.py -> util/url.py,
         charade -> chardet): > dup_threshold (0.95)
       - same file, another name (renamed in place): > dup_threshold (0.95)
       - same file, method with the same name in a renamed class
         (RequestsTestCase.test_x -> TestRequests.test_x): > rename_threshold (0.9)
       - module blocks of the same file: > rename_threshold (0.9)
     Guards: lineages that coexist in the same version are never merged (two snippets
     present at the same time are different code), and identical boilerplate in unrelated
     places (`def __enter__(self): return self` in two classes) does not merge.

Ranking: one entry per lineage, ordered by rank_score = best score, minus kind_penalty
(0.05) for "<module>" blocks and whole non-Python files (.rst, .md, ...), so functions
and classes come first. The version shown for a lineage is, by default, the latest
version whose score is within `tie` (0.01) of the best ("latest"); prefer="score" shows
the best-scoring version instead. Lineages with equal rank scores are ordered newest-first.

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
    if len(parts) > 1 and parts[0] == "src":
        return "/".join(parts[1:])
    if len(parts) == 1 and parts[0].startswith("test_") and parts[0].endswith(".py"):
        return f"tests/{parts[0]}"  # root test file later moved into tests/
    return path


def identity_key(chunk: dict, single_module: bool = False) -> tuple:
    """("name", canonical path, kind, qualified name) or ("hash", hash, canonical path).
    single_module: the chunk is its file's only <module> block in that version."""
    if chunk["kind"] == "module" and not single_module:  # no stable name: content + near-dup merging
        return ("hash", chunk["hash"], canonical_path(chunk["path"]))
    return ("name", canonical_path(chunk["path"]), chunk["kind"], chunk["name"])


def _is_module(key: tuple) -> bool:
    return key[0] == "hash" or key[2] == "module"


def _key_path(key: tuple) -> str:
    return key[2] if key[0] == "hash" else key[1]


def merge_threshold(a: tuple, b: tuple, dup_threshold: float, rename_threshold: float) -> float | None:
    """Cosine two items must exceed to be one piece of code over time, or None if never."""
    if _is_module(a) or _is_module(b):  # module blocks: only with module blocks of the same file
        return rename_threshold if _is_module(a) and _is_module(b) and _key_path(a) == _key_path(b) else None
    if a[3] == b[3]:
        return dup_threshold  # moved to another file
    if a[1] != b[1]:
        return None  # different name AND different file
    if a[2] == b[2] == "method" and a[3].rsplit(".", 1)[-1] == b[3].rsplit(".", 1)[-1]:
        return rename_threshold  # same method, class renamed
    return dup_threshold  # renamed in place


def penalized(chunk: dict) -> bool:
    """<module> blocks and whole non-Python files rank below functions/classes."""
    return chunk["kind"] == "module" or (chunk["kind"] == "file" and chunk.get("language") != "python")


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
    key: tuple = ()  # identity_key


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
    rank_score: float = 0.0  # best_score minus the kind penalty; the ranking key
    penalty: float = 0.0

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
        occ_by_hash: dict[str, list[tuple[int, dict, tuple]]] = defaultdict(list)
        self.n_rows = 0
        for i, v in enumerate(store.registry.list()):
            refs = v.get("refs") or []
            self.versions.append(Version(i, v["commit"], refs[0] if refs else v["short"],
                                         (v.get("committed_at") or "")[:10]))
            chunks = store.read_chunks(v["commit"])
            n_modules: dict[str, int] = defaultdict(int)
            for c in chunks:
                if c["kind"] == "module":
                    n_modules[c["path"]] += 1
            for c in chunks:
                key = identity_key(c, single_module=n_modules[c["path"]] == 1)
                occ_by_hash[c["hash"]].append((i, c, key))
                self.n_rows += 1
        self.hashes = list(occ_by_hash)
        self.matrix = np.asarray(store.embeddings.matrix(self.hashes), dtype=np.float32)
        self.occurrences: list[list[Occurrence]] = [
            [Occurrence(vi, c, row, key) for vi, c, key in occ_by_hash[h]] for row, h in enumerate(self.hashes)]
        self.by_key: dict[tuple, list[Occurrence]] = defaultdict(list)
        for occs in self.occurrences:
            for o in occs:
                self.by_key[o.key].append(o)

    def __len__(self) -> int:
        return len(self.hashes)

    def code(self, occ: Occurrence) -> str:
        return self.store.content.get(occ.chunk["hash"])

    def search(self, vector: np.ndarray, k: int = 10, prefer: str = "latest", tie: float = 0.01,
               dup_threshold: float = 0.95, rename_threshold: float = 0.9, kind_penalty: float = 0.05,
               pool: int | None = None) -> AllVersionsResult:
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
                if o.key not in items:
                    items[o.key] = self.by_key[o.key]
        keys = list(items)
        parent = list(range(len(keys)))
        vsets = [{o.version for o in items[k_]} for k_ in keys]

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

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
                threshold = merge_threshold(keys[a], keys[b], dup_threshold, rename_threshold)
                if threshold is None:
                    continue
                ib = [pos[r] for r in item_rows[b]]
                c = float(cos[np.ix_(ia, ib)].max())
                if c > threshold:
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
            rep_occ = best_per_version[rep_v][1]
            penalty = kind_penalty if penalized(rep_occ.chunk) else 0.0
            lineages.append(Lineage(
                id=gid, best_score=best, representative=rep_occ,
                representative_score=best_per_version[rep_v][0], timeline=timeline,
                n_distinct_code=len({o.chunk["hash"] for _, o in best_per_version.values()}),
                keys=[keys[i] for i in members], rank_score=best - penalty, penalty=penalty))
        lineages.sort(key=lambda lin: (-lin.rank_score, -lin.representative.version, lin.id))

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
