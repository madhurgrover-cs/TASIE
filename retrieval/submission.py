"""
Samsung "Agentic Code Intelligence" screening submission: AppsRetrieval (CPU).

    python retrieval/submission.py                                  # default --model (HF repo id)
    python retrieval/submission.py --model <user>/<repo>            # any fine-tuned repo or local dir
    python retrieval/submission.py --smoke 5 300                    # 5 queries / 300 docs pipeline check

Final model: nomic-ai/CodeRankEmbed fine-tuned on the APPS train split
(retrieval/finetune.py, 2 epochs, CachedMNRL, no hard negatives), dense retrieval,
cosine similarity. Full test split: NDCG@10 0.4709, MRR@10 0.4303.

PrePostPipelineEncoder does all pre/post-processing inside encode():
  pre  (queries) desc-io cleanup (retrieval/query_clean.py): keep the problem
                 description + Input/Output spec, drop samples / notes / constraints;
                 then the CodeRankEmbed query prefix.
  pre  (docs)    none (CodeRankEmbed uses no document prefix).
  post           L2-normalised float32 embeddings -> cosine similarity.

Writes appsretrieval_results.json = task_result.to_dict() (mteb's TaskResult).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if __package__ in (None, ""):  # run as `python retrieval/submission.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.eval_baseline import load_st_model, subsample_task
from retrieval.query_clean import clean_query

logger = logging.getLogger("submission")

TASK_NAME = "AppsRetrieval"
# Published with retrieval/push_model.py.
DEFAULT_MODEL = "madhurr382/coderankembed-apps-ft"
# From the nomic-ai/CodeRankEmbed model card; fine-tuning used the same prefix.
QUERY_PREFIX = "Represent this query for searching relevant code: "
DOC_PREFIX = ""
QUERY_CLEAN = "desc-io"


class PrePostPipelineEncoder(AbsEncoder):
    """Fine-tuned CodeRankEmbed with query cleanup + prefixing inside encode()."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_seq_length: int = 512,
        query_clean: str = QUERY_CLEAN,
        query_prefix: str = QUERY_PREFIX,
        doc_prefix: str = DOC_PREFIX,
    ):
        # fp32 + restore_nonpersistent_buffers() (transformers v5 leaves NomicBert's
        # rotary inv_freq / norm_factor uninitialised when loading remote code).
        self.model, _ = load_st_model(model_name, device, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length
        self.query_clean = query_clean
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": model_name,
            "embed_dim": self.model.get_sentence_embedding_dimension(),
            "max_tokens": max_seq_length,
            "similarity_fn_name": ScoringFunction.COSINE,
            "framework": ["Sentence Transformers", "PyTorch"],
            "open_weights": True,
            "use_instructions": True,
            "modalities": ["text"],
        })

    # ---- pre-processing
    def preprocess_query(self, text: str) -> str:
        return self.query_prefix + clean_query(text, self.query_clean)

    def preprocess_doc(self, text: str) -> str:
        return self.doc_prefix + text

    # ---- post-processing
    @staticmethod
    def postprocess(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        is_query = prompt_type == PromptType.query
        pre = self.preprocess_query if is_query else self.preprocess_doc
        t0 = time.perf_counter()
        emb = self.model.encode(
            [pre(t) for t in texts],
            batch_size=kwargs.get("batch_size", 32),
            convert_to_numpy=True,
            show_progress_bar=kwargs.get("show_progress_bar", True),
        )
        logger.info("Encoded %d %s in %.1fs", len(texts), "queries" if is_query else "docs",
                    time.perf_counter() - t0)
        return self.postprocess(emb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id (or local dir) of the fine-tuned model")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cpu",
                    help="screening is CPU; auto = cuda if available")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--smoke", nargs=2, type=int, metavar=("N_QUERIES", "N_DOCS"))
    ap.add_argument("--out", type=Path, default=Path("appsretrieval_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Model %s on %s", args.model, device)

    model = PrePostPipelineEncoder(args.model, device=device, max_seq_length=args.max_seq_length)
    task = mteb.get_task(TASK_NAME)
    if args.smoke:
        task.load_data()  # evaluate() keeps pre-loaded data, so the subset sticks
        subsample_task(task, *args.smoke)

    t0 = time.perf_counter()
    results = mteb.evaluate(
        model,
        task,
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,  # never serve stale scores from ~/.cache/mteb
        overwrite_strategy="always",
    )
    task_result = results.task_results[0]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)

    scores = task_result.scores["test"][0]
    print(f"\n{TASK_NAME} | {args.model} | {device}"
          + (f" | smoke {args.smoke[0]}q/{args.smoke[1]}d" if args.smoke else ""))
    print(f"  NDCG@10 {scores['ndcg_at_10']:.4f}  MRR@10 {scores['mrr_at_10']:.4f}  "
          f"({time.perf_counter() - t0:.1f}s) -> {args.out}")


if __name__ == "__main__":
    main()
