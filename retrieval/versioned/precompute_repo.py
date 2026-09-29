"""
Kaggle job: index a real repo at several commits, compare full vs incremental builds,
run example queries, and upload the store for the Space.

    python -m retrieval.versioned.precompute_repo                     # psf/requests, 4 tags, upload
    python -m retrieval.versioned.precompute_repo --no-upload --embedder hashing   # dry run, no torch

Steps:
  1. clone --repo-url into --workdir (full history, so any tag/commit can be indexed);
  2. chain build into --out: full build of the first ref, then each next ref incrementally
     from the previous one (only files changed per `git diff` are re-chunked, and only
     code never embedded before is embedded);
  3. cold full build of every ref in a throwaway store (no cache), for the timing comparison;
  4. step benchmark (--steps A:B ...): adjacent releases, i.e. a typical change: full build
     of A, then B incrementally vs a cold full build of B, in throwaway stores;
  5. 3 example queries per commit (top-3 with path / name / lines / score);
  6. write benchmark.json + a dataset card into --out and upload it to --upload-repo
     (dataset repo, HF_TOKEN).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval.versioned.builder import BuildReport, build  # noqa: E402
from retrieval.versioned.embedder import DEFAULT_MODEL, make_embedder  # noqa: E402
from retrieval.versioned.gitrepo import GitRepo  # noqa: E402
from retrieval.versioned.searcher import VersionedSearcher  # noqa: E402
from retrieval.versioned.store import Store  # noqa: E402

DEFAULT_REPO_URL = "https://github.com/psf/requests"
# 2013 -> 2024: spread over the project's history
DEFAULT_REFS = ["v2.0.0", "v2.12.0", "v2.25.0", "v2.32.3"]
# adjacent releases: what an everyday "rebuild after a change" costs
DEFAULT_STEPS = ["v2.32.2:v2.32.3", "v2.31.0:v2.32.0"]
DEFAULT_QUERIES = [
    "get proxy settings from environment variables",
    "strip the Authorization header when redirected to another host",
    "encode files for a multipart/form-data POST body",
]


def clone(url: str, dest: Path) -> Path:
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "--tags", "origin"], check=True)
    else:
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    return dest


def dataset_card(repo_id: str, store: Store, bench: dict) -> str:
    rows = "\n".join(
        f"| {r['ref']} | `{r['commit'][:7]}` | {r['committed_at'][:10]} | {r['n_files']} | {r['n_chunks']} | "
        f"{r['mode']} | {r['files_rechunked']} | {r['chunks_embedded']} | {r['chunks_reused']} | "
        f"{r['build_s']:.1f} | {r['full_build_s']:.1f} |"
        for r in bench["versions"])
    step_rows = "\n".join(
        f"| {s['from']} → {s['to']} | {s['files_changed']} | {s['n_chunks']} | {s['files_rechunked']} | "
        f"{s['chunks_embedded']} | {s['chunks_reused']} | {s['incremental_s']:.2f} | {s['full_s']:.2f} |"
        for s in bench.get("steps", []))
    return f"""---
license: apache-2.0
pretty_name: {store.registry.repo} versioned code index
tags:
- code-search
- embeddings
---

# {repo_id.split('/')[-1]}

Function/class-level code index of [{store.registry.repo}](https://github.com/{store.registry.repo}) at
{len(bench['versions'])} commits, for the [code-search Space](https://huggingface.co/spaces/madhurr382/code-search-demo).
Built with `retrieval/versioned` from https://github.com/madhurgrover-cs/TASIE (branch `phase2-versions`).

- Model: `{store.model_id}` (queries: desc-io cleanup + `Represent this query for searching relevant code: `)
- Chunks: Python functions / methods / class headers / top-level blocks via `ast`; one chunk per other text file
- Storage is shared: each distinct chunk's code and vector is stored once (`content.jsonl`,
  `embeddings/`), and `versions/<commit>/chunks.jsonl` lists the chunks of that commit.
- Build device: {bench['device']}. `build_s` = incremental build from the previous row (first row: full);
  `full_build_s` = cold full build of the same commit (no cache).

| ref | commit | date | files | chunks | mode | files re-chunked | chunks embedded | chunks reused | build_s | full_build_s |
|---|---|---|---|---|---|---|---|---|---|---|
{rows}

Adjacent releases (a typical change), incremental vs cold full build:

| from → to | files changed | chunks | re-chunked files | embedded | reused | incremental_s | full_s |
|---|---|---|---|---|---|---|---|
{step_rows}

The indexed code is from {store.registry.repo} (Apache-2.0).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--refs", nargs="+", default=DEFAULT_REFS, help="oldest first")
    ap.add_argument("--workdir", type=Path, default=Path("versioned-work"))
    ap.add_argument("--out", type=Path, default=Path("repo-versions-index"), help="store to build and upload")
    ap.add_argument("--embedder", choices=["st", "hashing"], default="st")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None, help="model Hub commit (default: current main)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    ap.add_argument("--skip-full", action="store_true", help="skip the cold full-build comparison")
    ap.add_argument("--steps", nargs="*", default=DEFAULT_STEPS, metavar="A:B",
                    help="adjacent ref pairs for the incremental-vs-full step benchmark")
    ap.add_argument("--upload-repo", default="madhurr382/repo-versions-index")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not args.no_upload and not token:
        raise SystemExit("HF_TOKEN is not set (needs a write token), or pass --no-upload.")
    if args.out.exists():
        shutil.rmtree(args.out)  # the uploaded store must hold exactly these refs
    args.workdir.mkdir(parents=True, exist_ok=True)
    repo_dir = clone(args.repo_url, args.workdir / "repo")

    t0 = time.perf_counter()
    embedder = make_embedder(args.embedder, args.model, args.revision, args.device, args.batch_size)
    print(f"Embedder {embedder.model_id} ({getattr(embedder, 'device', 'cpu')}) loaded in {time.perf_counter() - t0:.1f}s")

    store = Store(args.out)
    chain: list[BuildReport] = []
    full: dict[str, BuildReport] = {}
    with GitRepo(repo_dir) as repo:
        prev = None
        for ref in args.refs:
            rep = build(repo, store, embedder, ref, base=prev, incremental=prev is not None, refs=[ref])
            chain.append(rep)
            prev = rep.commit
            print("chain:", rep.summary())
        if not args.skip_full:
            for ref in args.refs:
                with tempfile.TemporaryDirectory() as tmp:
                    full[ref] = build(repo, Store(tmp), embedder, ref, incremental=False, refs=[ref])
                print("cold full:", full[ref].summary())
        steps = []
        for pair in args.steps:
            a, b = pair.split(":")
            with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
                s1 = Store(t1)
                build(repo, s1, embedder, a, incremental=False, refs=[a])
                inc = build(repo, s1, embedder, b, base=a, refs=[b])
                cold = build(repo, Store(t2), embedder, b, incremental=False, refs=[b])
            n_changed = len(repo.diff(inc.base, inc.commit))
            steps.append({"from": a, "to": b, "files_changed": n_changed, "n_chunks": inc.n_chunks,
                          "files_rechunked": inc.files_rechunked, "chunks_embedded": inc.chunks_embedded,
                          "chunks_reused": inc.chunks_reused, "incremental_s": inc.build_s,
                          "full_s": cold.build_s, "full_chunks_embedded": cold.chunks_embedded})
            print(f"step {a} -> {b}: {n_changed} files changed | incremental {inc.build_s:.2f}s "
                  f"({inc.chunks_embedded} embedded, {inc.chunks_reused} reused) vs full {cold.build_s:.2f}s "
                  f"({cold.chunks_embedded} embedded) -> {cold.build_s / max(inc.build_s, 1e-9):.1f}x")

    print(f"\n{'ref':<9} {'commit':<8} {'files':>5} {'chunks':>6} {'mode':<11} {'rechunked':>9} "
          f"{'embedded':>8} {'reused':>6} {'incr_s':>7} {'full_s':>7} {'speedup':>7}")
    versions = []
    for rep in chain:
        meta = store.registry.versions[rep.commit]
        f = full.get(rep.refs[0])
        full_s = f.build_s if f else float("nan")
        print(f"{rep.refs[0]:<9} {rep.short:<8} {rep.n_files:>5} {rep.n_chunks:>6} {rep.mode:<11} "
              f"{rep.files_rechunked:>9} {rep.chunks_embedded:>8} {rep.chunks_reused:>6} {rep.build_s:>7.2f} "
              f"{full_s:>7.2f} {full_s / rep.build_s if rep.mode == 'incremental' else 1.0:>6.1f}x")
        versions.append({**meta, "ref": rep.refs[0], "full_build_s": full_s,
                         "full_chunks_embedded": f.chunks_embedded if f else None})

    searcher = VersionedSearcher(store, embedder)
    examples = {}
    for rep in chain:
        print(f"\n== {store.registry.label(rep.commit)} ({store.registry.versions[rep.commit]['committed_at'][:10]})")
        examples[rep.refs[0]] = []
        for q in args.queries:
            res = searcher.search(q, rep.commit, k=3)
            print(f"  Q: {q}   [{res.total_ms:.0f} ms]")
            for h in res.hits:
                print(f"     {h.score:.4f}  {h.location}  {h.name}")
            examples[rep.refs[0]].append({"query": q, "total_ms": res.total_ms,
                                          "hits": [{"path": h.path, "name": h.name, "lines": [h.start_line, h.end_line],
                                                    "score": h.score} for h in res.hits]})

    bench = {"repo": store.registry.repo, "model_id": store.model_id, "embedder": args.embedder,
             "device": getattr(embedder, "device", "cpu"), "versions": versions, "steps": steps,
             "examples": examples,
             "distinct_chunks": len(store.content), "total_chunk_rows": sum(r.n_chunks for r in chain)}
    (args.out / "benchmark.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
    (args.out / "README.md").write_text(dataset_card(args.upload_repo, store, bench), encoding="utf-8")
    print(f"\nStore {args.out}: {bench['total_chunk_rows']} chunk rows over {len(chain)} versions, "
          f"{bench['distinct_chunks']} distinct chunks stored/embedded once")

    if args.no_upload:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(args.upload_repo, repo_type="dataset", exist_ok=True)
    info = api.upload_folder(repo_id=args.upload_repo, repo_type="dataset", folder_path=args.out,
                             commit_message=f"Index {store.registry.repo} at {', '.join(args.refs)}",
                             delete_patterns=["versions/**", "embeddings/**"])
    print(f"Uploaded {args.out} -> {url} (commit {info.oid})")


if __name__ == "__main__":
    main()
