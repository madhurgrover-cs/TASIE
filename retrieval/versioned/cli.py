"""
python -m retrieval.versioned <command>

    index          full build of one commit
    update         incremental build from an indexed commit (nearest indexed ancestor by default)
    query          top-k chunks for a natural-language query at one commit
    list-versions  indexed commits with build stats

Examples:
    python -m retrieval.versioned index  --repo ../requests --commit v2.0.0
    python -m retrieval.versioned update --repo ../requests --commit v2.12.0
    python -m retrieval.versioned query  --commit v2.12.0 "retry a request after a redirect" -k 5
    python -m retrieval.versioned query  --commit all "retry a request after a redirect"   # every version, grouped
    python -m retrieval.versioned list-versions
    ... --embedder hashing   (no torch; for dry runs)

--embedder st (default) is madhurr382/coderankembed-apps-ft on --device (default cpu), pinned
to the store's recorded revision once the store exists.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval.versioned.builder import build  # noqa: E402
from retrieval.versioned.embedder import DEFAULT_MODEL, Embedder, HashingEmbedder, STEmbedder  # noqa: E402
from retrieval.versioned.gitrepo import GitRepo  # noqa: E402
from retrieval.versioned.searcher import VersionedSearcher  # noqa: E402
from retrieval.versioned.store import Store  # noqa: E402

DEFAULT_STORE = Path("versioned-index")


def make_embedder(args, store: Store) -> Embedder:
    """The store's model when it has one (exact revision), else --embedder / --model."""
    model_id = store.model_id
    if model_id and model_id.startswith("hashing-"):
        return HashingEmbedder(int(model_id.split("-", 1)[1]))
    if model_id:
        model, _, revision = model_id.rpartition("@")
        return STEmbedder(model, revision=None if revision == "local" else revision, device=args.device)
    if args.embedder == "hashing":
        return HashingEmbedder()
    return STEmbedder(args.model, revision=args.revision, device=args.device, batch_size=args.batch_size)


def _print_versions(store: Store, as_json: bool) -> None:
    versions = store.registry.list()
    if as_json:
        print(json.dumps(versions, indent=1))
        return
    print(f"{store.registry.repo} | model {store.model_id} | {len(versions)} versions")
    print(f"  {'commit':<8} {'refs':<14} {'date':<10} {'mode':<11} {'base':<8} {'chunks':>6} "
          f"{'embedded':>8} {'reused':>6} {'build_s':>7}")
    for v in versions:
        print(f"  {v['short']:<8} {','.join(v.get('refs', []))[:14]:<14} {(v.get('committed_at') or '')[:10]:<10} "
              f"{v['mode']:<11} {(v.get('base') or '')[:7]:<8} {v['n_chunks']:>6} {v['chunks_embedded']:>8} "
              f"{v['chunks_reused']:>6} {v['build_s']:>7.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m retrieval.versioned", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE, help=f"index store directory (default {DEFAULT_STORE})")
    ap.add_argument("--embedder", choices=["st", "hashing"], default="st")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None, help="model Hub revision (default: current main, then pinned)")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("index", "update"):
        p = sub.add_parser(name)
        p.add_argument("--repo", type=Path, required=True, help="path to a git clone")
        p.add_argument("--commit", required=True, help="sha, tag or branch")
        p.add_argument("--ref-name", action="append", default=None, help="label to record (default: --commit if a name)")
        p.add_argument("--force", action="store_true", help="rebuild even if already indexed")
        if name == "update":
            p.add_argument("--base", default=None, help="indexed commit to start from (default: nearest indexed)")

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--commit", required=True, help='sha / tag, or "all" for every version grouped by lineage')
    q.add_argument("--prefer", choices=["latest", "score"], default="latest", help="--commit all: version shown")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--json", action="store_true")
    q.add_argument("--show-code", type=int, default=0, metavar="LINES", help="print the first LINES of each hit")

    lv = sub.add_parser("list-versions")
    lv.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = Store(args.store)

    if args.cmd == "list-versions":
        _print_versions(store, args.json)
        return 0

    if args.cmd in ("index", "update"):
        embedder = make_embedder(args, store)
        with GitRepo(args.repo) as repo:
            rep = build(repo, store, embedder, args.commit, base=getattr(args, "base", None),
                        incremental=args.cmd == "update", force=args.force, refs=args.ref_name)
        print(rep.summary())
        return 0

    embedder = make_embedder(args, store)
    if args.commit == "all":
        from retrieval.versioned.evolution import AllVersionsIndex
        from retrieval.versioned.evolution_report import print_result

        idx = AllVersionsIndex(store)
        print_result(idx, idx.search(embedder.embed_query(args.text), args.k, prefer=args.prefer), args.k)
        return 0
    res = VersionedSearcher(store, embedder).search(args.text, args.commit, args.k)
    if args.json:
        print(json.dumps({"commit": res.commit, "label": res.label, "encode_ms": res.encode_ms,
                          "search_ms": res.search_ms, "total_ms": res.total_ms,
                          "hits": [h.__dict__ for h in res.hits]}, indent=1))
        return 0
    print(f"{res.label}: {len(res.hits)} of {res.n_chunks} chunks in {res.total_ms:.0f} ms "
          f"(encode {res.encode_ms:.0f} ms, search {res.search_ms:.1f} ms)")
    for i, h in enumerate(res.hits, 1):
        print(f"  {i:>2}. {h.score:.4f}  {h.location}  {h.name} ({h.kind})")
        if args.show_code:
            for line in h.code.splitlines()[:args.show_code]:
                print(f"        {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
