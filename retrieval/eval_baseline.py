"""
Baseline dense-retrieval eval on MTEB "AppsRetrieval" (CoIR).
Screening is CPU only; --device auto uses CUDA when available (e.g. Kaggle GPU).

    python -m retrieval.eval_baseline                         # full eval, preset coderankembed
    python -m retrieval.eval_baseline --preset jina-code --out appsretrieval_results.jina-code.json
    python -m retrieval.eval_baseline --preset sfr-code-400m --smoke 5 300

Corpus embeddings are cached in --cache-dir (default retrieval/cache/, see CodeEncoder._cache_path),
so query-side experiments don't re-encode the ~9k-doc corpus.

Written against mteb 2.21.8: AbsEncoder.encode(inputs: DataLoader[BatchedInput], *,
task_metadata, hf_split, hf_subset, prompt_type, **encode_kwargs) -> Array.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("eval_baseline")

TASK_NAME = "AppsRetrieval"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Bump whenever the text fed to the document encoder changes (prefixing,
# truncation strategy, normalisation...), or the encoder itself is fixed, so
# stale cached embeddings are ignored.
# p2: restore_nonpersistent_buffers(); p1 embeddings came from uninitialised buffers.
PREPROC_VERSION = "p2"

# Per-model settings, taken from each Hugging Face model card and the repo's
# 1_Pooling/config.json (checked 2026-09-28). Pooling is loaded from the repo by
# sentence-transformers; it is listed here only to verify it at load time.
# All three ship custom modelling code, so trust_remote_code is required.
PRESETS: dict[str, dict[str, Any]] = {
    "coderankembed": {
        "model": "nomic-ai/CodeRankEmbed",  # 137M, NomicBert, MIT
        # Card: the query "*must* include the following task instruction prefix";
        # also the "query" prompt in config_sentence_transformers.json. Docs: none.
        "query_prefix": "Represent this query for searching relevant code: ",
        "doc_prefix": "",
        "pooling": "cls",
        "trust_remote_code": True,
    },
    "jina-code": {
        "model": "jinaai/jina-embeddings-v2-base-code",  # 161M, JinaBert (ALiBi), Apache-2.0
        # Card examples encode raw queries and raw code, no prefixes.
        "query_prefix": "",
        "doc_prefix": "",
        "pooling": "mean",
        "trust_remote_code": True,
        # Its remote code (jinaai/jina-bert-v2-qk-post-norm) imports
        # find_pruneable_heads_and_indices and calls get_extended_attention_mask /
        # get_head_mask, all removed in transformers v5. Needs transformers<5,
        # which conflicts with sentence-transformers 6.x.
        "max_transformers_major": 4,
    },
    "sfr-code-400m": {
        "model": "Salesforce/SFR-Embedding-Code-400M_R",  # 434M, Alibaba "NewModel", CC-BY-NC-4.0
        # Card examples (Transformers + ST) encode raw queries and raw code, no
        # prefixes. The "Instruct: ...\nQuery: " template mteb uses is registered
        # only for the 2B model, not this one.
        "query_prefix": "",
        "doc_prefix": "",
        "pooling": "cls",
        "trust_remote_code": True,
    },
}
DEFAULT_PRESET = "coderankembed"


def restore_nonpersistent_buffers(st_model: SentenceTransformer) -> tuple[int, int]:
    """Recompute non-persistent buffers that transformers v5 leaves uninitialised.

    v5 builds the model on the meta device, refills every non-persistent buffer
    with torch.empty_like, and relies on `_init_weights` to restore them. The
    remote-code models here compute such buffers in __init__ (rotary inv_freq /
    cos / sin caches, position_ids, attention norm_factor) and their
    `_init_weights` never touches them, so they hold garbage after loading. SFR
    then fails with a CUDA index-out-of-bounds assert (Alibaba-NLP/new-impl
    discussion #14), and NomicBert silently gets wrong RoPE frequencies.

    Fix: build a throwaway copy from the config (real construction, weight init
    skipped) and copy its buffers over. Returns (restored, differed).
    """
    from transformers import PreTrainedModel

    hf = next((m for m in st_model.modules() if isinstance(m, PreTrainedModel)), None)
    if hf is None or not hasattr(hf, "named_non_persistent_buffers"):  # v4 never trashes them
        return 0, 0
    loaded = dict(hf.named_non_persistent_buffers())
    if not loaded:
        return 0, 0
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        no_init_weights = contextlib.nullcontext
    with torch.device("cpu"), no_init_weights():
        fresh = dict(type(hf)(hf.config).named_non_persistent_buffers())

    restored = differed = 0
    for name, buf in loaded.items():
        new = fresh.get(name)
        if new is None or new.shape != buf.shape:
            logger.warning("Cannot restore buffer %s (missing or shape mismatch in fresh model)", name)
            continue
        new = new.to(device=buf.device, dtype=buf.dtype)
        differed += not torch.equal(new, buf)
        parent, _, attr = name.rpartition(".")
        hf.get_submodule(parent).register_buffer(attr, new, persistent=False)
        restored += 1
    return restored, differed


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


class CodeEncoder(AbsEncoder):
    """Wraps a sentence-transformers code embedding model for mteb."""

    def __init__(
        self,
        model_name: str,
        query_max_len: int = 512,
        doc_max_len: int = 512,
        query_prefix: str = "",
        doc_prefix: str = "",
        trust_remote_code: bool = True,
        expected_pooling: str | None = None,
        cache_dir: Path | None = CACHE_DIR,
        device: str = "cpu",
    ):
        self.model_name = model_name
        # fp32: jina/SFR checkpoints are stored in fp16/bf16, and transformers v5
        # would otherwise load them in that dtype (slow and lossy on CPU).
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": torch.float32},
        )
        restored, differed = restore_nonpersistent_buffers(self.model)
        logger.info("Restored %d non-persistent buffers (%d had uninitialised values after load)",
                    restored, differed)
        self.buffers_restored = {"restored": restored, "differed": differed}
        self.pooling = next((m.pooling_mode for m in self.model if hasattr(m, "pooling_mode")), None)
        logger.info("Loaded %s: pooling=%s, dim=%s", model_name, self.pooling,
                    self.model.get_sentence_embedding_dimension())
        if expected_pooling and self.pooling != expected_pooling:
            logger.warning("Pooling mismatch for %s: repo config gives %r, preset expects %r",
                           model_name, self.pooling, expected_pooling)
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
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
        # Key = model + doc max length + preprocessing version + corpus content
        # (hashed after doc_prefix is applied), so a different corpus or doc prefix
        # (smoke subset, another code version) gets its own file.
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
        texts = [self.doc_prefix + t for t in texts]
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


def resolve_preset(preset: str | None, model: str | None) -> tuple[str, dict[str, Any]]:
    """--preset wins; else a --model matching a preset's model uses that preset;
    else an unknown --model runs as "custom" with no prefixes."""
    if preset:
        return preset, PRESETS[preset]
    if model is None:
        return DEFAULT_PRESET, PRESETS[DEFAULT_PRESET]
    for name, cfg in PRESETS.items():
        if cfg["model"] == model:
            return name, cfg
    logger.warning("No preset for %s: using no prefixes, repo pooling, trust_remote_code=True", model)
    return "custom", {"model": model, "query_prefix": "", "doc_prefix": "", "pooling": None,
                      "trust_remote_code": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help=f"model + prefixes + pooling from PRESETS (default {DEFAULT_PRESET})")
    ap.add_argument("--model", default=None, help="HF model id; picks the matching preset if there is one")
    ap.add_argument("--max-seq-length", type=int, default=512, help="default for both query and doc length")
    ap.add_argument("--query-max-len", type=int, default=None)
    ap.add_argument("--doc-max-len", type=int, default=None)
    ap.add_argument("--query-prefix", default=None, help="override the preset's query prefix ('' to disable)")
    ap.add_argument("--doc-prefix", default=None, help="override the preset's doc prefix")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = cuda if available, else cpu")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="corpus embedding cache directory")
    ap.add_argument("--no-cache", action="store_true", help="don't read/write corpus embedding cache")
    ap.add_argument("--smoke", nargs=2, type=int, metavar=("N_QUERIES", "N_DOCS"))
    ap.add_argument("--out", default=None, help="default: appsretrieval_results.json (.smoke.json with --smoke)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t_total = time.perf_counter()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s (requested: %s)", device, args.device)

    preset_name, cfg = resolve_preset(args.preset, args.model)
    max_major = cfg.get("max_transformers_major")
    if max_major is not None and int(transformers.__version__.split(".")[0]) > max_major:
        raise SystemExit(f"Preset {preset_name!r} needs transformers<{max_major + 1} "
                         f"(installed {transformers.__version__}); see the note in PRESETS.")
    model_name = args.model or cfg["model"]
    query_prefix = cfg["query_prefix"] if args.query_prefix is None else args.query_prefix
    doc_prefix = cfg["doc_prefix"] if args.doc_prefix is None else args.doc_prefix
    logger.info("Preset: %s | model=%s | query_prefix=%r | doc_prefix=%r",
                preset_name, model_name, query_prefix, doc_prefix)

    encoder = CodeEncoder(
        model_name,
        query_max_len=args.query_max_len or args.max_seq_length,
        doc_max_len=args.doc_max_len or args.max_seq_length,
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        trust_remote_code=cfg["trust_remote_code"],
        expected_pooling=cfg["pooling"],
        cache_dir=None if args.no_cache else args.cache_dir,
        device=device,
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
        "preset": preset_name,
        "model": model_name,
        "pooling": encoder.pooling,
        "buffers_restored": encoder.buffers_restored,
        "transformers_version": transformers.__version__,
        "query_max_len": encoder.query_max_len,
        "doc_max_len": encoder.doc_max_len,
        "query_prefix": encoder.query_prefix,
        "doc_prefix": encoder.doc_prefix,
        "preproc_version": PREPROC_VERSION,
        "batch_size": args.batch_size,
        "device": device,
        "cache_dir": None if args.no_cache else str(args.cache_dir),
        "smoke": args.smoke,
        "mteb_version": mteb.__version__,
        "timings": {**encoder.timings, "total_s": total_s},
        "ndcg_at_10": ndcg10,
        "mrr_at_10": mrr10,
        "scores": scores,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))

    t = encoder.timings
    print(f"\n{TASK_NAME} | {preset_name} ({model_name}) | {device} | q_len={encoder.query_max_len} d_len={encoder.doc_max_len}")
    print(f"  NDCG@10 = {ndcg10:.4f}")
    print(f"  MRR@10  = {mrr10:.4f}")
    print(f"  corpus encode {t['corpus_encode_s']:.1f}s (cache hits={t['corpus_cache_hits']}), "
          f"query encode {t['query_encode_s']:.1f}s, total {total_s:.1f}s")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
