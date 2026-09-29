"""
Evolutionary retrieval report: search ALL indexed versions at once with the real model and
print lineage-grouped results plus the duplicate metric (for verification before any deploy).

    python -m retrieval.versioned.evolution_report                          # Hub index, CPU, 5 queries
    python -m retrieval.versioned.evolution_report --prefer score -k 5
    python -m retrieval.versioned.evolution_report --store versioned-index  # a local store

Nothing is re-embedded: the store's vectors are reused (each distinct chunk scored once),
and only the queries are encoded, with the store's exact model revision.

Duplicate metric: of the top-k, how many results repeat a function (lineage) already shown
above them. "flat" = every version searched and the rows merged by score (what you get
without grouping); "grouped" = one entry per lineage.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval.versioned.evolution import AllVersionsIndex, AllVersionsResult, canonical_path  # noqa: E402
from retrieval.versioned.store import Store  # noqa: E402

DEFAULT_INDEX = "madhurr382/repo-versions-index"
DEFAULT_QUERIES = [
    "get proxy settings from environment variables",
    "strip the Authorization header when redirected to another host",
    "encode files for a multipart/form-data POST body",
    "merge session-level settings with per-request settings",
    "check whether a URL should bypass the proxy using no_proxy",
]


def print_result(idx: AllVersionsIndex, res: AllVersionsResult, k: int) -> None:
    print(f"  grouped: {len(res.lineages)} lineages from {res.n_distinct} distinct chunks "
          f"({res.n_rows} chunk rows over {len(idx.versions)} versions)")
    for i, lin in enumerate(res.lineages[:k], 1):
        c = lin.representative.chunk
        shown = idx.versions[lin.representative.version].label
        note = "" if lin.representative_score == lin.best_score else f" (shown {lin.representative_score:.4f})"
        if lin.penalty:
            note += f" [rank {lin.rank_score:.4f}: -{lin.penalty:.2f} {c['kind']}]"
        print(f"  {i:>2}. {lin.best_score:.4f}{note}  {canonical_path(c['path'])}::{c['name']}  "
              f"[shown {shown}: {c['path']}:{c['start_line']}-{c['end_line']}]")
        print(f"      {lin.timeline_text()}   {'code changed' if lin.changed else 'code unchanged'}")
    print("  flat (no grouping), top 10:")
    for (o, s), lid in zip(res.flat[:10], res.flat_lineage_ids[:10]):
        print(f"      {s:.4f}  {idx.versions[o.version].label:<8} {o.chunk['path']}::{o.chunk['name']}  (lineage {lid})")
    d = res.duplicates(10)
    print(f"  duplicates in top 10: flat {d['flat_duplicates']}/10 ({d['flat_unique']} unique functions) "
          f"-> grouped {d['grouped_duplicates']}/10 ({d['grouped_unique']} unique)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-repo", default=DEFAULT_INDEX, help="HF dataset with the versioned store")
    ap.add_argument("--index-revision", default=None)
    ap.add_argument("--store", type=Path, default=None, help="local store instead of --index-repo")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--prefer", choices=["latest", "score"], default="latest")
    ap.add_argument("--tie", type=float, default=0.01)
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = ap.parse_args()

    root = args.store
    if root is None:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.index_repo, repo_type="dataset", revision=args.index_revision))
    store = Store(root)

    from retrieval.versioned.embedder import HashingEmbedder, STEmbedder

    model_id = store.model_id or ""
    t0 = time.perf_counter()
    if model_id.startswith("hashing-"):
        embedder = HashingEmbedder(int(model_id.split("-", 1)[1]))
    else:
        model, _, revision = model_id.rpartition("@")
        embedder = STEmbedder(model, revision=revision, device=args.device)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx = AllVersionsIndex(store)
    print(f"{store.registry.repo}: {len(idx.versions)} versions "
          f"({', '.join(v.label for v in idx.versions)}), {idx.n_rows} chunk rows, {len(idx)} distinct vectors "
          f"reused from the store (no re-embedding) | index {time.perf_counter() - t0:.2f}s, "
          f"model {model_id} loaded in {load_s:.1f}s on {args.device}")
    print(f"ranking: one entry per lineage, prefer={args.prefer}, tie={args.tie}, near-dup cosine>{args.dup_threshold}"
          f", module/non-Python file penalty 0.05 | timeline: ● new/changed  ○ unchanged  – absent")

    totals = {"flat_duplicates": 0, "grouped_duplicates": 0, "flat_unique": 0, "grouped_unique": 0}
    rows = []
    for q in args.queries:
        t0 = time.perf_counter()
        vec = embedder.embed_query(q)
        t1 = time.perf_counter()
        res = idx.search(vec, k=args.k, prefer=args.prefer, tie=args.tie, dup_threshold=args.dup_threshold)
        t2 = time.perf_counter()
        print(f"\nQ: {q}   [encode {(t1 - t0) * 1000:.0f} ms, search+group {(t2 - t1) * 1000:.0f} ms]")
        print_result(idx, res, args.k)
        d = res.duplicates(10)
        for key in totals:
            totals[key] += d[key]
        rows.append((q, d))

    n = len(rows)
    print("\nDuplicate metric (top 10 per query): results repeating a function already shown above")
    print(f"  {'query':<62} {'flat':>5} {'grouped':>8}")
    for q, d in rows:
        print(f"  {q[:62]:<62} {d['flat_duplicates']:>4}/10 {d['grouped_duplicates']:>5}/10")
    print(f"  {'mean':<62} {totals['flat_duplicates'] / n:>5.1f} {totals['grouped_duplicates'] / n:>8.1f}")
    print(f"  unique functions in top 10: flat {totals['flat_unique'] / n:.1f} -> grouped {totals['grouped_unique'] / n:.1f}")


if __name__ == "__main__":
    main()
