"""
Baseline dense-retrieval eval on MTEB "AppsRetrieval" (CoIR), CPU only.

    python -m retrieval.eval_baseline                         # full eval, CodeRankEmbed
    python -m retrieval.eval_baseline --model jinaai/jina-embeddings-v2-base-code
    python -m retrieval.eval_baseline --smoke 5 300           # 5 queries, 300 docs

Corpus embeddings are cached in retrieval/cache/ (see CodeEncoder._cache_path),
so query-side experiments don't re-encode the ~9k-doc corpus.

Written against mteb 2.21.8: AbsEncoder.encode(inputs: DataLoader[BatchedInput], *,
task_metadata, hf_split, hf_subset, prompt_type, **encode_kwargs) -> Array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("eval_baseline")

TASK_NAME = "AppsRetrieval"
DEFAULT_MODEL = "nomic-ai/CodeRankEmbed"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Bump whenever the text fed to the document encoder changes (prefixing,
# truncation strategy, normalisation...) so stale cached embeddings are ignored.
PREPROC_VERSION = "p1"

# Models that expect an instruction prefix on the query side only.
QUERY_PREFIXES = {
    "nomic-ai/CodeRankEmbed": "Represent this query for searching relevant code: ",
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


class CodeEncoder(AbsEncoder):
    """Wraps a sentence-transformers code embedding model for mteb, on CPU."""

    def __init__(
        self,
        model_name: str,
        query_max_len: int = 512,
        doc_max_len: int = 512,
        query_prefix: str | None = None,
        cache_dir: Path | None = CACHE_DIR,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        self.query_prefix = QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix
        self.cache_dir = cache_dir
        self.timings: dict[str, Any] = {
            "corpus_encode_s": 0.0,
            "query_encode_s": 0.0,
            "corpus_cache_hits": 0,
            "corpus_cache_misses": 0,
        }
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": model_name,
            "embed_dim": self.model.get_sentence_embedding_dimension(),
            "max_tokens": doc_max_len,
            "similarity_fn_name": ScoringFunction.COSINE,
            "framework": ["Sentence Transformers", "PyTorch"],
            "open_weights": True,
            "use_instructions": bool(self.query_prefix),
            "modalities": ["text"],
        })

    def _cache_path(self, texts: list[str]) -> Path:
        # Key = model + doc max length + preprocessing version + corpus content,
        # so a different corpus (smoke subset, another code version) gets its own file.
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8", errors="ignore"))
            h.update(b"\0")
        name = f"{_slug(self.model_name)}__len{self.doc_max_len}__{PREPROC_VERSION}__n{len(texts)}_{h.hexdigest()[:16]}.npy"
        return self.cache_dir / name

    def _encode_texts(self, texts: list[str], max_len: int, batch_size: int, show_progress: bool) -> np.ndarray:
        self.model.max_seq_length = max_len
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        batch_size = kwargs.get("batch_size", 32)
        show_progress = kwargs.get("show_progress_bar", True)

        if prompt_type == PromptType.query:
            t0 = time.perf_counter()
            emb = self._encode_texts([self.query_prefix + t for t in texts], self.query_max_len, batch_size, show_progress)
            dt = time.perf_counter() - t0
            self.timings["query_encode_s"] += dt
            logger.info("Encoded %d queries in %.1fs", len(texts), dt)
            return emb

        t0 = time.perf_counter()
        path = self._cache_path(texts) if self.cache_dir else None
        if path and path.exists():
            emb = np.load(path)
            self.timings["corpus_cache_hits"] += 1
            logger.info("Loaded %d cached doc embeddings from %s", len(texts), path.name)
        else:
            emb = self._encode_texts(texts, self.doc_max_len, batch_size, show_progress)
            self.timings["corpus_cache_misses"] += 1
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, emb)
                logger.info("Cached %d doc embeddings to %s", len(texts), path.name)
        dt = time.perf_counter() - t0
        self.timings["corpus_encode_s"] += dt
        logger.info("Corpus embeddings ready (%d docs) in %.1fs", len(texts), dt)
        return emb


def subsample_task(task, n_queries: int, n_docs: int, seed: int = 0) -> None:
    """Shrink a loaded retrieval task in place: first n_queries queries (that have
    qrels), all their relevant docs, padded with random distractors to n_docs."""
    rng = random.Random(seed)
    for subset, splits in task.dataset.items():
        for split, data in splits.items():
            qrels = data["relevant_docs"]
            qids = [q for q in data["queries"]["id"] if q in qrels][:n_queries]
            keep_q = set(qids)
            rel_docs = {d for q in qids for d in qrels[q]}
            others = [d for d in data["corpus"]["id"] if d not in rel_docs]
            keep_d = rel_docs | set(rng.sample(others, max(0, min(len(others), n_docs - len(rel_docs)))))

            data["queries"] = data["queries"].filter(lambda i: i in keep_q, input_columns="id")
            data["corpus"] = data["corpus"].filter(lambda i: i in keep_d, input_columns="id")
            data["relevant_docs"] = {q: qrels[q] for q in qids}
            logger.info("[%s/%s] smoke subset: %d queries, %d docs (%d relevant)",
                        subset, split, len(data["queries"]), len(data["corpus"]), len(rel_docs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-seq-length", type=int, default=512, help="default for both query and doc length")
    ap.add_argument("--query-max-len", type=int, default=None)
    ap.add_argument("--doc-max-len", type=int, default=None)
    ap.add_argument("--query-prefix", default=None, help="override the per-model query prefix ('' to disable)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-cache", action="store_true", help="don't read/write corpus embedding cache")
    ap.add_argument("--smoke", nargs=2, type=int, metavar=("N_QUERIES", "N_DOCS"))
    ap.add_argument("--out", default=None, help="default: appsretrieval_results.json (.smoke.json with --smoke)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t_total = time.perf_counter()

    encoder = CodeEncoder(
        args.model,
        query_max_len=args.query_max_len or args.max_seq_length,
        doc_max_len=args.doc_max_len or args.max_seq_length,
        query_prefix=args.query_prefix,
        cache_dir=None if args.no_cache else CACHE_DIR,
    )

    task = mteb.get_task(TASK_NAME)
    if args.smoke:
        # evaluate() leaves pre-loaded data untouched, so we can subset it first.
        task.load_data()
        subsample_task(task, *args.smoke)

    results = mteb.evaluate(
        encoder,
        task,
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,  # never serve stale scores from ~/.cache/mteb
        overwrite_strategy="always",
    )
    total_s = time.perf_counter() - t_total

    scores = results.task_results[0].scores["test"][0]
    ndcg10, mrr10 = scores["ndcg_at_10"], scores["mrr_at_10"]

    out = Path(args.out or ("appsretrieval_results.smoke.json" if args.smoke else "appsretrieval_results.json"))
    payload = {
        "task": TASK_NAME,
        "model": args.model,
        "query_max_len": encoder.query_max_len,
        "doc_max_len": encoder.doc_max_len,
        "query_prefix": encoder.query_prefix,
        "preproc_version": PREPROC_VERSION,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "mteb_version": mteb.__version__,
        "timings": {**encoder.timings, "total_s": total_s},
        "ndcg_at_10": ndcg10,
        "mrr_at_10": mrr10,
        "scores": scores,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))

    t = encoder.timings
    print(f"\n{TASK_NAME} | {args.model} | q_len={encoder.query_max_len} d_len={encoder.doc_max_len}")
    print(f"  NDCG@10 = {ndcg10:.4f}")
    print(f"  MRR@10  = {mrr10:.4f}")
    print(f"  corpus encode {t['corpus_encode_s']:.1f}s (cache hits={t['corpus_cache_hits']}), "
          f"query encode {t['query_encode_s']:.1f}s, total {total_s:.1f}s")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
