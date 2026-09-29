"""
Retrieval eval on MTEB "AppsRetrieval" (CoIR): dense, BM25 and hybrid (RRF), with
optional query cleanup. Screening is CPU only; --device auto uses CUDA when available.

    python retrieval/eval_baseline.py                                   # dense, desc-io queries
    python retrieval/eval_baseline.py --model /kaggle/working/cre-ft    # finetune.py output (preset auto)
    python retrieval/eval_baseline.py --query-clean none desc desc-io   # ablation A
    python retrieval/eval_baseline.py --retriever dense bm25 hybrid     # ablation B
    python retrieval/eval_baseline.py --smoke 5 300 --retriever bm25

--query-clean and --retriever each take several values. Every combination runs
as one variant (the model is loaded once), and a summary table is printed at the
end. Each variant goes through mteb.evaluate(), and its official TaskResult JSON
is written to <out stem>_mteb/<variant>.json.

Corpus embeddings are cached in --cache-dir (default retrieval/cache/, see
CodeEncoder._cache_key) and in memory across variants, so query-side experiments
don't re-encode the ~9k-doc corpus.

Written against mteb 2.21.8: AbsEncoder.encode(inputs: DataLoader[BatchedInput], *,
task_metadata, hf_split, hf_subset, prompt_type, **encode_kwargs) -> Array. BM25
is a SearchProtocol model (bm25_search.py), and the hybrid is mteb.HybridSearch
(models/hybrid_wrappers.py) with RRF.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import random
import re
import sys
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

if __package__ in (None, ""):  # run as `python retrieval/eval_baseline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.bm25_search import BM25CodeSearch
from retrieval.query_clean import QUERY_CLEAN_MODES, clean_query

logger = logging.getLogger("eval_baseline")

TASK_NAME = "AppsRetrieval"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
# bm25 / hybrid are kept only for the ablation table: full-split NDCG@10 was
# ~0.05 (bm25) and ~0.16 (hybrid) vs 0.2420 dense (2026-09-28).
RETRIEVERS = ("dense", "bm25", "hybrid")
# Ablation A winner (dense NDCG@10: none 0.2368, desc 0.1824, desc-io 0.2420).
# finetune.py applies the same cleaning to training queries.
DEFAULT_QUERY_CLEAN = "desc-io"

# Bump whenever the text fed to the document encoder changes (prefixing,
# truncation strategy, normalisation...), or the encoder itself is fixed, so
# stale cached embeddings are ignored.
# p2: restore_nonpersistent_buffers(); p1 embeddings came from uninitialised buffers.
PREPROC_VERSION = "p2"

# Per-model settings, taken from the Hugging Face model card and the repo's
# 1_Pooling/config.json (checked 2026-09-28). Pooling is loaded from the repo by
# sentence-transformers; it is listed here only to verify it at load time.
# Dropped 2026-09-28: jinaai/jina-embeddings-v2-base-code (its remote code needs
# transformers<5) and Salesforce/SFR-Embedding-Code-400M_R (NDCG@10 0.0 even
# after restore_nonpersistent_buffers()).
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
}
DEFAULT_PRESET = "coderankembed"


def restore_nonpersistent_buffers(st_model: SentenceTransformer) -> tuple[int, int]:
    """Recompute non-persistent buffers that transformers v5 leaves uninitialised.

    v5 builds the model on the meta device, refills every non-persistent buffer
    with torch.empty_like, and relies on `_init_weights` to restore them. Remote-code
    models compute such buffers in __init__ (rotary inv_freq / cos / sin caches,
    position_ids, attention norm_factor), and their `_init_weights` never touches
    them, so they hold garbage after loading. NomicBert (CodeRankEmbed) then
    silently gets wrong RoPE frequencies, and Alibaba new-impl models fail with a
    CUDA index-out-of-bounds assert (Alibaba-NLP/new-impl discussion #14).

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


def _is_nomic_bert(model_name: str, trust_remote_code: bool) -> bool:
    try:
        cfg = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception as e:  # let SentenceTransformer raise the real loading error
        logger.warning("Could not read config of %s: %s", model_name, e)
        return False
    return cfg.model_type == "nomic_bert"


def load_st_model(model_name: str, device: str, trust_remote_code: bool = True) -> tuple[SentenceTransformer, dict[str, int]]:
    """Load in fp32 and repair non-persistent buffers. Shared by eval, submission and finetune.py."""
    # fp32: some checkpoints are stored in fp16/bf16, and transformers v5
    # would otherwise load them in that dtype (slow and lossy on CPU).
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
    # NomicBert's remote from_pretrained loads Hub ids through
    # state_dict_from_pretrained(safe_serialization=kwargs.get("safe_serialization", False)),
    # which only looks for pytorch_model.bin(.index.json); our Hub repo (like
    # nomic-ai/CodeRankEmbed) has only model.safetensors. Local dirs take a separate
    # branch that ignores the flag. Other architectures may reject the kwarg, so gate it.
    if _is_nomic_bert(model_name, trust_remote_code):
        model_kwargs["safe_serialization"] = True
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs,
    )
    restored, differed = restore_nonpersistent_buffers(model)
    logger.info("Restored %d non-persistent buffers (%d had uninitialised values after load)",
                restored, differed)
    return model, {"restored": restored, "differed": differed}


# Written by finetune.py next to the saved model: base preset, run_id, val scores.
FINETUNE_CONFIG = "finetune_config.json"


def read_finetune_config(model_name: str) -> dict[str, Any] | None:
    path = Path(model_name) / FINETUNE_CONFIG
    return json.loads(path.read_text()) if path.is_file() else None


def model_fingerprint(model_name: str) -> str:
    """Distinguishes different weights saved under the same local path (cache key).
    Hub ids return "" (their name already identifies them)."""
    path = Path(model_name)
    if not path.is_dir():
        return ""
    ft = read_finetune_config(model_name)
    if ft and ft.get("run_id"):
        return ft["run_id"]
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        st = f.stat()
        h.update(f"{f.relative_to(path)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


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
        self.fingerprint = model_fingerprint(model_name)
        self.model, self.buffers_restored = load_st_model(model_name, device, trust_remote_code)
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
        self._corpus_memo: dict[str, np.ndarray] = {}  # reused across variants in one run
        self.timings: dict[str, Any] = {}
        self.reset_timings()
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

    def reset_timings(self) -> None:
        self.timings = {
            "corpus_encode_s": 0.0,
            "query_encode_s": 0.0,
            "corpus_cache_hits": 0,
            "corpus_cache_misses": 0,
        }

    def _cache_key(self, texts: list[str]) -> str:
        # Key = model + doc max length + preprocessing version + corpus content
        # (hashed after doc_prefix is applied), so a different corpus or doc prefix
        # (smoke subset, another code version) gets its own file.
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8", errors="ignore"))
            h.update(b"\0")
        # Local (fine-tuned) models add a fingerprint so retraining into the same
        # --output-dir doesn't reuse stale embeddings.
        model_tag = _slug(self.model_name) + (f"_{self.fingerprint}" if self.fingerprint else "")
        return f"{model_tag}__len{self.doc_max_len}__{PREPROC_VERSION}__n{len(texts)}_{h.hexdigest()[:16]}.npy"

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
        key = self._cache_key(texts)
        path = self.cache_dir / key if self.cache_dir else None
        if key in self._corpus_memo:
            emb = self._corpus_memo[key]
            self.timings["corpus_cache_hits"] += 1
            logger.info("Reusing %d doc embeddings from memory", len(texts))
        elif path and path.exists():
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
        self._corpus_memo[key] = emb
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


def set_query_clean(task, original_queries: dict[tuple[str, str], Any], mode: str) -> None:
    """Replace each split's query texts with clean_query(text, mode), starting from
    the original texts, so variants don't compound. Applies to every retriever."""
    for (subset, split), queries in original_queries.items():
        texts = [clean_query(t, mode) for t in queries["text"]]
        task.dataset[subset][split]["queries"] = queries.remove_columns("text").add_column("text", texts)
        n_changed = sum(a != b for a, b in zip(texts, queries["text"]))
        logger.info("[%s/%s] query-clean=%s: %d/%d queries changed, mean %d -> %d chars",
                    subset, split, mode, n_changed, len(texts),
                    np.mean([len(t) for t in queries["text"]]), np.mean([len(t) for t in texts]))


def resolve_preset(preset: str | None, model: str | None) -> tuple[str, dict[str, Any]]:
    """--preset wins; else a --model matching a preset's model uses that preset;
    else a local finetune.py output uses its base model's preset; else an unknown
    --model runs as "custom" with no prefixes."""
    if preset:
        return preset, PRESETS[preset]
    if model is None:
        return DEFAULT_PRESET, PRESETS[DEFAULT_PRESET]
    for name, cfg in PRESETS.items():
        if cfg["model"] == model:
            return name, cfg
    ft = read_finetune_config(model)
    if ft and ft.get("base_preset") in PRESETS:
        logger.info("%s is fine-tuned from %s: using preset %s", model, ft["base_model"], ft["base_preset"])
        return ft["base_preset"], PRESETS[ft["base_preset"]]
    logger.warning("No preset for %s: using no prefixes, repo pooling, trust_remote_code=True", model)
    return "custom", {"model": model, "query_prefix": "", "doc_prefix": "", "pooling": None,
                      "trust_remote_code": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help=f"model + prefixes + pooling from PRESETS (default {DEFAULT_PRESET})")
    ap.add_argument("--model", default=None, help="HF model id; picks the matching preset if there is one")
    ap.add_argument("--retriever", nargs="+", choices=RETRIEVERS, default=["dense"],
                    help="dense = embedding model, bm25 = rank_bm25 over code tokens, "
                         "hybrid = RRF of the two (mteb.HybridSearch)")
    ap.add_argument("--query-clean", nargs="+", choices=QUERY_CLEAN_MODES, default=[DEFAULT_QUERY_CLEAN],
                    help="none = raw statement, desc = description only, "
                         f"desc-io = description + Input/Output spec (see query_clean.py). Default {DEFAULT_QUERY_CLEAN}")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant for --retriever hybrid")
    ap.add_argument("--bm25-weight", type=float, default=1.0,
                    help="RRF weight of BM25 relative to dense (dense weight is 1.0)")
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
    retrievers = list(dict.fromkeys(args.retriever))
    clean_modes = list(dict.fromkeys(args.query_clean))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s (requested: %s)", device, args.device)

    preset_name, cfg = resolve_preset(args.preset, args.model)
    model_name = args.model or cfg["model"]
    query_prefix = cfg["query_prefix"] if args.query_prefix is None else args.query_prefix
    doc_prefix = cfg["doc_prefix"] if args.doc_prefix is None else args.doc_prefix
    logger.info("Preset: %s | model=%s | query_prefix=%r | doc_prefix=%r",
                preset_name, model_name, query_prefix, doc_prefix)

    encoder = None
    if {"dense", "hybrid"} & set(retrievers):
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

    # evaluate() leaves pre-loaded data untouched, so the task can be subset
    # (--smoke) and its query texts swapped per variant (--query-clean) up front.
    task = mteb.get_task(TASK_NAME)
    task.load_data()
    if args.smoke:
        subsample_task(task, *args.smoke)
    original_queries = {(subset, split): data["queries"]
                        for subset, splits in task.dataset.items() for split, data in splits.items()}

    out = Path(args.out or ("appsretrieval_results.smoke.json" if args.smoke else "appsretrieval_results.json"))
    mteb_dir = out.with_name(out.stem + "_mteb")
    mteb_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, Any]] = []
    for clean_mode, retriever in itertools.product(clean_modes, retrievers):
        name = f"{retriever}/{clean_mode}"
        logger.info("=== Variant %s ===", name)
        set_query_clean(task, original_queries, clean_mode)
        if encoder is not None:
            encoder.reset_timings()
        bm25 = BM25CodeSearch() if retriever in ("bm25", "hybrid") else None
        if retriever == "dense":
            model = encoder
        elif retriever == "bm25":
            model = bm25
        else:
            model = mteb.HybridSearch([encoder, bm25], weights=[1.0, args.bm25_weight],
                                      fusion_strategy="rrf", rrf_k=args.rrf_k)

        t0 = time.perf_counter()
        results = mteb.evaluate(
            model,
            task,
            encode_kwargs={"batch_size": args.batch_size},
            cache=None,  # never serve stale scores from ~/.cache/mteb
            overwrite_strategy="always",
        )
        eval_s = time.perf_counter() - t0

        task_result = results.task_results[0]
        result_path = mteb_dir / f"{_slug(name)}.json"
        task_result.to_disk(result_path)
        scores = task_result.scores["test"][0]
        timings: dict[str, Any] = {"eval_s": eval_s}
        if retriever != "bm25":
            timings.update(encoder.timings)
        if bm25 is not None:
            timings.update(bm25.timings)
        variants.append({
            "variant": name,
            "retriever": retriever,
            "query_clean": clean_mode,
            "model": model.mteb_model_meta.name,
            "ndcg_at_10": scores["ndcg_at_10"],
            "mrr_at_10": scores["mrr_at_10"],
            "timings": timings,
            "mteb_result": str(result_path),
            "scores": scores,
        })
        logger.info("%s: NDCG@10=%.4f MRR@10=%.4f (%.1fs)",
                    name, scores["ndcg_at_10"], scores["mrr_at_10"], eval_s)
    total_s = time.perf_counter() - t_total

    payload = {
        "task": TASK_NAME,
        "preset": preset_name,
        "model": model_name,
        "model_fingerprint": model_fingerprint(model_name),
        "finetune": read_finetune_config(model_name),
        "pooling": encoder.pooling if encoder else None,
        "buffers_restored": encoder.buffers_restored if encoder else None,
        "transformers_version": transformers.__version__,
        "query_max_len": encoder.query_max_len if encoder else None,
        "doc_max_len": encoder.doc_max_len if encoder else None,
        "query_prefix": query_prefix,
        "doc_prefix": doc_prefix,
        "preproc_version": PREPROC_VERSION,
        "rrf_k": args.rrf_k,
        "bm25_weight": args.bm25_weight,
        "batch_size": args.batch_size,
        "device": device,
        "cache_dir": None if args.no_cache else str(args.cache_dir),
        "smoke": args.smoke,
        "mteb_version": mteb.__version__,
        "total_s": total_s,
        "variants": variants,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))

    lens = f" | q_len={encoder.query_max_len} d_len={encoder.doc_max_len}" if encoder else ""
    print(f"\n{TASK_NAME} | {preset_name} ({model_name}) | {device}{lens}"
          + (f" | smoke {args.smoke[0]}q/{args.smoke[1]}d" if args.smoke else ""))
    width = max(len("variant"), *(len(v["variant"]) for v in variants))
    print(f"  {'variant':<{width}}  {'NDCG@10':>8}  {'MRR@10':>8}  {'time_s':>8}")
    for v in variants:
        print(f"  {v['variant']:<{width}}  {v['ndcg_at_10']:>8.4f}  {v['mrr_at_10']:>8.4f}  "
              f"{v['timings']['eval_s']:>8.1f}")
    print(f"  total {total_s:.1f}s (incl. model load / dataset download)")
    print(f"  -> {out}, {mteb_dir}/")


if __name__ == "__main__":
    main()
