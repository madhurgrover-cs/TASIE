"""
Embed the AppsRetrieval corpus once (Kaggle GPU) and publish it as a Hub dataset
for the code-search Space, which only encodes queries.

    export HF_TOKEN=hf_...   # write token (Kaggle: secret, see DEPLOY_SPACE.md)
    python retrieval/precompute_corpus.py                        # embed 8,765 docs + upload
    python retrieval/precompute_corpus.py --limit 200 --no-upload  # smoke check, writes --out-dir only

Output (--out-dir, then uploaded to --repo-id as a dataset repo):
  embeddings.npy  float32 [n_docs, dim], L2-normalised
  docs.jsonl      {"id", "code", "url", "partition", "language"} per row, same order
  manifest.json   model id + exact Hub revision, query prefix / cleaning, dims,
                  dataset revision, library versions, embeddings sha256
  README.md       dataset card

Documents get the same text as in eval (mteb's "title text" join, no doc prefix),
so Space scores match what AppsRetrieval measured. The model is loaded from a local
snapshot at a pinned revision (NomicBert's remote code ignores `revision` for Hub
weights); the Space loads the same revision from the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # run as `python retrieval/precompute_corpus.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("precompute_corpus")

# mteb AppsRetrieval metadata (mteb 2.21.8); checked against mteb when it is installed.
APPS_PATH = "CoIR-Retrieval/apps"
APPS_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
DEFAULT_MODEL = "madhurr382/coderankembed-apps-ft"
DEFAULT_REPO = "madhurr382/apps-corpus-index"
FORMAT_VERSION = 1  # space/search.py INDEX_FORMAT_VERSION


def check_mteb_metadata() -> None:
    try:
        import mteb
    except ImportError:
        return
    meta = mteb.get_task("AppsRetrieval").metadata.dataset
    if (meta["path"], meta["revision"]) != (APPS_PATH, APPS_REVISION):
        raise SystemExit(f"mteb AppsRetrieval now points at {meta}; update APPS_PATH / APPS_REVISION")


def doc_text(title: str | None, text: str) -> str:
    """mteb's corpus join (_create_dataloaders.py), as in finetune.py."""
    return f"{title} {text}".strip() if title else text


def dataset_card(repo_id: str, manifest: dict) -> str:
    return f"""---
license: mit
pretty_name: AppsRetrieval corpus index
tags:
- code-search
- embeddings
source_datasets:
- {APPS_PATH}
---

# {repo_id.split('/')[-1]}

Precomputed embeddings of the {manifest['n_docs']:,}-document `AppsRetrieval` corpus
([{APPS_PATH}](https://huggingface.co/datasets/{APPS_PATH}) @ `{APPS_REVISION[:7]}`), for the
[code-search Space](https://huggingface.co/spaces/madhurr382/code-search-demo).

- Model: [`{manifest['model_id']}`](https://huggingface.co/{manifest['model_id']}) @ `{manifest['model_revision']}`
- `embeddings.npy`: float32 `[{manifest['n_docs']}, {manifest['dim']}]`, L2-normalised (cosine = dot product)
- `docs.jsonl`: `id`, `code`, `url`, `partition`, `language`, row-aligned with the embeddings
- `manifest.json`: query prefix `{manifest['query_prefix']!r}`, query cleaning `{manifest['query_clean']}`,
  library versions, sha256 of the embeddings

Queries must be encoded with the same model revision, prefix and cleaning.
Built by `retrieval/precompute_corpus.py` in https://github.com/madhurgrover-cs/TASIE.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hub model id")
    ap.add_argument("--model-revision", default=None, help="Hub commit to embed with (default: current main)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO, help="dataset repo to upload to")
    ap.add_argument("--out-dir", type=Path, default=Path("apps-corpus-index"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="first N docs only (smoke test)")
    ap.add_argument("--no-upload", action="store_true", help="write --out-dir only")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    token = os.environ.get("HF_TOKEN")
    if not args.no_upload and not token:
        raise SystemExit("HF_TOKEN is not set (needs a write token), or pass --no-upload.")
    if args.limit and not args.no_upload:
        raise SystemExit("--limit builds a partial index; use it with --no-upload.")

    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    from retrieval.model_loading import load_st_model

    check_mteb_metadata()
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    t_total = time.perf_counter()

    # ---- model at a pinned revision
    api = HfApi(token=token)
    revision = args.model_revision or api.model_info(args.model).sha
    model_dir = Path(snapshot_download(args.model, revision=revision))
    ft_path = model_dir / "finetune_config.json"
    ft = json.loads(ft_path.read_text()) if ft_path.is_file() else {}
    query_prefix = ft.get("query_prefix", "Represent this query for searching relevant code: ")
    doc_prefix = ft.get("doc_prefix", "")
    query_clean = ft.get("query_clean", "desc-io")
    logger.info("Model %s @ %s on %s | query_prefix=%r doc_prefix=%r query_clean=%s",
                args.model, revision, device, query_prefix, doc_prefix, query_clean)
    model, buffers = load_st_model(str(model_dir), device, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # ---- corpus (the full AppsRetrieval corpus: train + test solutions)
    corpus = load_dataset(APPS_PATH, "corpus", split="corpus", revision=APPS_REVISION)
    if args.limit:
        corpus = corpus.select(range(min(args.limit, len(corpus))))
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus)
    metas = corpus["meta_information"] if "meta_information" in corpus.column_names else [{}] * len(corpus)
    partitions = corpus["partition"] if "partition" in corpus.column_names else [None] * len(corpus)
    languages = corpus["language"] if "language" in corpus.column_names else ["python"] * len(corpus)
    ids, codes = corpus["_id"], corpus["text"]
    texts = [doc_prefix + doc_text(ti, tx) for ti, tx in zip(titles, codes)]
    logger.info("Corpus: %d docs", len(texts))

    # ---- embed
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    encode_s = time.perf_counter() - t0
    logger.info("Embedded %d docs -> %s in %.1fs", len(texts), emb.shape, encode_s)

    # ---- write
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
        for i, c, m, p, lang in zip(ids, codes, metas, partitions, languages):
            f.write(json.dumps({"id": i, "code": c, "url": (m or {}).get("url") or None, "partition": p,
                                "language": (lang or "python").lower()}, ensure_ascii=False) + "\n")
    manifest = {
        "format_version": FORMAT_VERSION,
        "name": "APPS corpus",
        "language": "python",
        "model_id": args.model,
        "model_revision": revision,
        "query_prefix": query_prefix,
        "doc_prefix": doc_prefix,
        "query_clean": query_clean,
        "max_seq_length": args.max_seq_length,
        "n_docs": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "normalized": True,
        "similarity": "cosine",
        "dataset": {"path": APPS_PATH, "revision": APPS_REVISION, "config": "corpus", "limit": args.limit},
        "embeddings_sha256": hashlib.sha256((out / "embeddings.npy").read_bytes()).hexdigest(),
        "buffers_restored": buffers,
        "device": device,
        "encode_s": encode_s,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(dataset_card(args.repo_id, manifest), encoding="utf-8")
    logger.info("Wrote %s (%.1f MB embeddings)", out, (out / "embeddings.npy").stat().st_size / 1e6)

    if args.no_upload:
        print(f"Index written to {out} (not uploaded)")
        return
    url = api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    info = api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=out,
                             commit_message=f"Index {manifest['n_docs']} docs with {args.model}@{revision[:7]}",
                             allow_patterns=["embeddings.npy", "docs.jsonl", "manifest.json", "README.md"])
    print(f"Uploaded {out} -> {url} (commit {info.oid}) in {time.perf_counter() - t_total:.0f}s total")


if __name__ == "__main__":
    main()
