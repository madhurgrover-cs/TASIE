"""
Fine-tune CodeRankEmbed on the APPS *train* split (CoIR-Retrieval/apps), for AppsRetrieval.

    python retrieval/finetune.py --output-dir /kaggle/working/cre-ft                    # stage 1
    python retrieval/finetune.py --output-dir /kaggle/working/cre-ft-hn --hard-negatives 1   # + mined negatives
    python retrieval/finetune.py --output-dir /kaggle/working/cre-ft-smoke --smoke     # ~2 min check
    python retrieval/eval_baseline.py --model /kaggle/working/cre-ft                   # preset picked up automatically

Data (no test data is touched):
  - Pairs come from the "default" config's *train* qrels only. The test qrels and
    test queries are never loaded, and any query marked partition=test aborts the run.
  - --val-size train queries (default 500, seeded) are held out. The validation
    corpus is every doc referenced by the train qrels, so the held-out positives
    compete with the training docs. That mirrors AppsRetrieval, whose corpus also
    contains the train solutions.
  - Queries get the same preset prefix and --query-clean (default desc-io) as
    eval_baseline.py. Docs use the same title + text join as mteb.
  - The dataset path and revision come from mteb's AppsRetrieval metadata.

Training (sentence-transformers 6.1, checked in source):
  - CachedMultipleNegativesRankingLoss (GradCache): in-batch negatives over
    --batch-size, with activations computed in --mini-batch-size chunks, so a large
    effective batch fits on a T4. BatchSamplers.NO_DUPLICATES keeps duplicate
    texts out of a batch (they would be false negatives).
  - fp16 on CUDA, lr 2e-5, 10% warmup, max len 512, 2 epochs by default.
  - InformationRetrievalEvaluator runs on the validation split before training
    (epoch 0 = base model) and after every epoch. Val MRR@10 / NDCG@10 and
    per-epoch train time are logged, and the epoch with the best val MRR@10 is
    saved (--select last to keep the final epoch instead).
  - --hard-negatives N (off by default): mine N negatives per pair with the *base*
    model before training (mine_hard_negatives, relative_margin 0.05, i.e.
    NV-Retriever TopK-PercPos). Candidates exclude the held-out val docs. Pairs
    with no valid negative are dropped (the count is logged), since the loss needs
    the same columns in every row.

--output-dir gets the model (model.save) plus finetune_config.json (base preset,
run_id, args, per-epoch metrics). eval_baseline.py --model <dir> reads it to apply
the base preset's prefixes, and uses run_id in its embedding-cache key.
"""
from __future__ import annotations

import os

# The HF Trainer wraps the model in DataParallel when it sees 2 GPUs (Kaggle T4 x2),
# which CachedMNRL doesn't go through. Pin one GPU unless the caller chose.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments  # noqa: E402
from sentence_transformers.base.sampler import BatchSamplers  # noqa: E402
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator  # noqa: E402
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss  # noqa: E402
from sentence_transformers.util import mine_hard_negatives  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

if __package__ in (None, ""):  # run as `python retrieval/finetune.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mteb  # noqa: E402

from retrieval.eval_baseline import (  # noqa: E402
    DEFAULT_PRESET,
    DEFAULT_QUERY_CLEAN,
    FINETUNE_CONFIG,
    PRESETS,
    TASK_NAME,
    load_st_model,
)
from retrieval.query_clean import QUERY_CLEAN_MODES, clean_query  # noqa: E402

logger = logging.getLogger("finetune")


def load_train_split(query_prefix: str, doc_prefix: str, query_clean: str) -> dict[str, Any]:
    """Train qrels -> {qid: query text}, {cid: doc text}, {qid: [cid]} (prefix + cleaning applied)."""
    meta = mteb.get_task(TASK_NAME).metadata.dataset
    path, revision = meta["path"], meta["revision"]
    logger.info("Loading %s @ %s (train qrels only)", path, revision)

    qrels = load_dataset(path, "default", split="train", revision=revision)
    relevant: dict[str, list[str]] = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
    doc_ids = {c for cids in relevant.values() for c in cids}

    queries_ds = load_dataset(path, "queries", split="queries", revision=revision)
    queries_ds = queries_ds.filter(lambda i: i in relevant, input_columns="_id")
    if "partition" in queries_ds.column_names:
        bad = [q for q, p in zip(queries_ds["_id"], queries_ds["partition"]) if p != "train"]
        if bad:
            raise SystemExit(f"{len(bad)} train-qrel queries are not partition=train (e.g. {bad[:3]}); refusing to train.")
    queries = {q: query_prefix + clean_query(t, query_clean) for q, t in zip(queries_ds["_id"], queries_ds["text"])}

    corpus_ds = load_dataset(path, "corpus", split="corpus", revision=revision)
    corpus_ds = corpus_ds.filter(lambda i: i in doc_ids, input_columns="_id")
    titles = corpus_ds["title"] if "title" in corpus_ds.column_names else [""] * len(corpus_ds)
    # Same join as mteb's dataloader (_create_dataloaders.py): "title text" when there is a title.
    corpus = {c: doc_prefix + ((f"{ti} {tx}").strip() if ti else tx)
              for c, ti, tx in zip(corpus_ds["_id"], titles, corpus_ds["text"])}

    missing_q = set(relevant) - set(queries)
    missing_d = doc_ids - set(corpus)
    if missing_q or missing_d:
        logger.warning("Dropping qrels with missing text: %d queries, %d docs", len(missing_q), len(missing_d))
    relevant = {q: [c for c in cids if c in corpus] for q, cids in relevant.items() if q in queries}
    relevant = {q: cids for q, cids in relevant.items() if cids}
    logger.info("Train split: %d queries, %d docs, %d pairs",
                len(relevant), len(corpus), sum(len(c) for c in relevant.values()))
    return {"queries": queries, "corpus": corpus, "relevant": relevant, "path": path, "revision": revision}


def _metric(metrics: dict[str, float], suffix: str) -> float | None:
    return next((v for k, v in metrics.items() if k.endswith(suffix)), None)


class EpochTracker(TrainerCallback):
    """Per-epoch train time + val metrics; keeps the best-val-MRR weights on CPU."""

    def __init__(self, history: list[dict[str, Any]], best: dict[str, Any]):
        self.history, self.best = history, best
        self._t0 = 0.0
        self._train_s = 0.0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        # Fires before the epoch-end evaluation, so eval time is excluded.
        self._train_s = time.perf_counter() - self._t0

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        epoch = int(round(state.epoch or 0))
        mrr, ndcg = _metric(metrics or {}, "mrr@10"), _metric(metrics or {}, "ndcg@10")
        cum = sum(h["train_s"] for h in self.history) + self._train_s
        self.history.append({"epoch": epoch, "val_mrr@10": mrr, "val_ndcg@10": ndcg,
                             "train_s": self._train_s, "cum_train_s": cum})
        logger.info("Epoch %d: val MRR@10=%.4f NDCG@10=%.4f | epoch train %.1fs (cumulative %.1fs)",
                    epoch, mrr, ndcg, self._train_s, cum)
        if mrr is not None and mrr > self.best["mrr"]:
            self.best.update(mrr=mrr, epoch=epoch,
                             state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    ap.add_argument("--query-clean", choices=QUERY_CLEAN_MODES, default=DEFAULT_QUERY_CLEAN)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128, help="in-batch negatives per step (CachedMNRL batch)")
    ap.add_argument("--mini-batch-size", type=int, default=16, help="GradCache chunk size; lower if CUDA OOM")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=500, help="train queries held out for validation")
    ap.add_argument("--val-max-docs", type=int, default=None,
                    help="cap the validation corpus (val positives + random train docs); default all")
    ap.add_argument("--max-train-pairs", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--hard-negatives", type=int, default=0, metavar="N",
                    help="mine N hard negatives per pair with the base model (0 = off)")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="save the best-val-MRR epoch or the final one")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="256 train pairs, 32 val queries over 300 docs, 1 epoch, batch 32")
    args = ap.parse_args()
    if args.smoke:
        args.max_train_pairs = args.max_train_pairs or 256
        args.val_size, args.val_max_docs, args.epochs = 32, args.val_max_docs or 300, 1
        args.batch_size = min(args.batch_size, 32)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t_total = time.perf_counter()
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = device == "cuda"
    logger.info("Device: %s (CUDA_VISIBLE_DEVICES=%s), fp16=%s",
                device, os.environ.get("CUDA_VISIBLE_DEVICES"), fp16)
    cfg = PRESETS[args.preset]

    # ---- data
    data = load_train_split(cfg["query_prefix"], cfg["doc_prefix"], args.query_clean)
    queries, corpus, relevant = data["queries"], data["corpus"], data["relevant"]
    rng = random.Random(args.seed)
    qids = sorted(relevant)
    rng.shuffle(qids)
    val_qids, train_qids = qids[:args.val_size], qids[args.val_size:]
    val_docs = {c for q in val_qids for c in relevant[q]}

    pairs = [(queries[q], corpus[c]) for q in train_qids for c in relevant[q]]
    if args.max_train_pairs:
        pairs = pairs[:args.max_train_pairs]
    train_ds = Dataset.from_dict({"anchor": [a for a, _ in pairs], "positive": [p for _, p in pairs]})

    val_corpus_ids = sorted(corpus)
    if args.val_max_docs and args.val_max_docs < len(val_corpus_ids):
        others = [c for c in val_corpus_ids if c not in val_docs]
        val_corpus_ids = sorted(val_docs | set(rng.sample(others, max(0, args.val_max_docs - len(val_docs)))))
    evaluator = InformationRetrievalEvaluator(
        queries={q: queries[q] for q in val_qids},
        corpus={c: corpus[c] for c in val_corpus_ids},
        relevant_docs={q: set(relevant[q]) for q in val_qids},
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 10], precision_recall_at_k=[10], map_at_k=[10],
        batch_size=args.eval_batch_size, name="apps-val",
    )
    logger.info("Train pairs: %d | val: %d queries over %d docs", len(train_ds), len(val_qids), len(val_corpus_ids))

    # ---- model
    model, buffers = load_st_model(cfg["model"], device, cfg["trust_remote_code"])
    model.max_seq_length = args.max_seq_length

    mining: dict[str, Any] | None = None
    if args.hard_negatives > 0:
        t0 = time.perf_counter()
        n_before = len(train_ds)
        # Candidates: train docs minus held-out val positives (keeps validation clean).
        candidates = [corpus[c] for c in sorted(corpus) if c not in val_docs]
        train_ds = mine_hard_negatives(
            train_ds, model,
            anchor_column_name="anchor", positive_column_name="positive",
            corpus=candidates,
            num_negatives=args.hard_negatives,
            relative_margin=0.05,
            sampling_strategy="top",
            output_format="n-tuple",
            batch_size=args.eval_batch_size,
        )
        mining = {"num_negatives": args.hard_negatives, "relative_margin": 0.05,
                  "pairs_before": n_before, "rows_after": len(train_ds),
                  "columns": train_ds.column_names, "mine_s": time.perf_counter() - t0}
        logger.info("Hard negatives: %d -> %d rows, columns %s, %.1fs",
                    n_before, len(train_ds), train_ds.column_names, mining["mine_s"])

    # ---- epoch 0 = base model
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"mrr": float("-inf"), "epoch": None, "state": None}
    t0 = time.perf_counter()
    base_metrics = evaluator(model)
    base_mrr = _metric(base_metrics, "mrr@10")
    history.append({"epoch": 0, "val_mrr@10": base_mrr, "val_ndcg@10": _metric(base_metrics, "ndcg@10"),
                    "train_s": 0.0, "cum_train_s": 0.0})
    logger.info("Epoch 0 (base): val MRR@10=%.4f NDCG@10=%.4f (eval %.1fs)",
                base_mrr, history[0]["val_ndcg@10"], time.perf_counter() - t0)
    best.update(mrr=base_mrr, epoch=0)  # weights are only copied once an epoch beats the base

    # ---- train
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="no",  # best epoch is kept in memory by EpochTracker
        logging_steps=10,
        report_to="none",  # Kaggle images ship wandb, which would prompt for a login
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator,
        callbacks=[EpochTracker(history, best)],
    )
    t0 = time.perf_counter()
    trainer.train()
    train_wall_s = time.perf_counter() - t0

    # ---- select + save
    last_epoch = history[-1]["epoch"]
    if args.select == "best" and best["epoch"] != last_epoch:
        if best["state"] is None:
            logger.warning("No epoch beat the base model on val MRR@10 (%.4f); saving the final epoch anyway, "
                           "check the history before using it.", base_mrr)
        else:
            logger.info("Restoring epoch %d weights (best val MRR@10 %.4f)", best["epoch"], best["mrr"])
            model.load_state_dict(best["state"])
    saved_epoch = best["epoch"] if args.select == "best" and best["state"] is not None else last_epoch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_dir))

    run = {
        "run_id": uuid.uuid4().hex[:12],
        "base_model": cfg["model"],
        "base_preset": args.preset,
        "query_prefix": cfg["query_prefix"],
        "doc_prefix": cfg["doc_prefix"],
        "query_clean": args.query_clean,
        "dataset": {"path": data["path"], "revision": data["revision"], "split": "train"},
        "n_train_rows": len(train_ds),
        "n_val_queries": len(val_qids),
        "n_val_docs": len(val_corpus_ids),
        "hard_negatives": mining,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": device,
        "fp16": fp16,
        "buffers_restored": buffers,
        "history": history,
        "saved_epoch": saved_epoch,
        "train_wall_s": train_wall_s,
        "total_s": time.perf_counter() - t_total,
        "torch_version": torch.__version__,
    }
    (args.output_dir / FINETUNE_CONFIG).write_text(json.dumps(run, indent=2))

    print(f"\nFine-tune {cfg['model']} -> {args.output_dir} (run {run['run_id']})")
    print(f"  {'epoch':>5}  {'val MRR@10':>10}  {'val NDCG@10':>11}  {'train_s':>8}")
    for h in history:
        mark = "  <- saved" if h["epoch"] == saved_epoch else ""
        print(f"  {h['epoch']:>5}  {h['val_mrr@10']:>10.4f}  {h['val_ndcg@10']:>11.4f}  {h['train_s']:>8.1f}{mark}")
    print(f"  train wall {train_wall_s:.1f}s (incl. per-epoch eval), total {run['total_s']:.1f}s")


if __name__ == "__main__":
    main()
