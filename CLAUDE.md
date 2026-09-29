# CLAUDE.md

## What this repo is (and is becoming)

Originally **SAST IQ / TASIE**: a FastAPI + vanilla-JS SAST scanner (regex rules → TF-IDF/LogReg
classifier → developer feedback → retraining). It is being repurposed for Samsung's
**"Agentic Code Intelligence"** hackathon: *natural-language query → ranked code snippets*.

- **Screening metric:** NDCG@10 and MRR@10 on MTEB task `AppsRetrieval` (CoIR, dataset
  `CoIR-Retrieval/apps`, split `test`, main score `ndcg_at_10`). **CPU only.**
- **P1:** rebuild indexes quickly across code versions (incremental / diff-based re-indexing).
- **Bonus:** retrieval across *all* versions of snippets.

Nothing SAST-specific has been deleted yet. See the reuse table below before removing anything.

## Commands

```bash
# API + dashboard (http://localhost:8000/dashboard, docs at /docs)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Retrieval eval (writes appsretrieval_results.json + appsretrieval_results_mteb/<variant>.json,
# prints a summary table: variant, NDCG@10, MRR@10, time). `python -m retrieval.eval_baseline` also works.
python retrieval/eval_baseline.py                                        # dense CodeRankEmbed, desc-io queries (default)
python retrieval/eval_baseline.py --model /kaggle/working/cre-ft         # fine-tuned dir: coderankembed preset applied automatically
python retrieval/eval_baseline.py --query-clean none desc desc-io        # ablation A (query cleanup)
python retrieval/eval_baseline.py --retriever dense bm25 hybrid          # ablation B (BM25 + RRF hybrid)
python retrieval/eval_baseline.py --smoke 5 300                          # 5 queries / 300 docs → appsretrieval_results.smoke.json
python retrieval/eval_baseline.py --query-max-len 128 --doc-max-len 512  # lengths are independent; --max-seq-length sets both (default 512)
python retrieval/eval_baseline.py --device cpu --cache-dir /kaggle/working/emb_cache   # --device auto|cuda|cpu (auto = cuda if available)

# Submission (PrePostPipelineEncoder, desc-io + prefix inside encode; writes task_result.to_dict())
python retrieval/submission.py --model <hf-user>/<repo>                       # CPU by default; --smoke 5 300 for a check
HF_TOKEN=... python retrieval/push_model.py --model-dir /kaggle/working/cre-ft --repo-id <hf-user>/<repo>

# Fine-tune CodeRankEmbed on the APPS train split (GPU; writes model + finetune_config.json)
python retrieval/finetune.py --output-dir /kaggle/working/cre-ft                      # CachedMNRL, 2 epochs, best val MRR@10 kept
python retrieval/finetune.py --output-dir /kaggle/working/cre-ft-hn --hard-negatives 1  # + negatives mined with the base model
python retrieval/finetune.py --output-dir /kaggle/working/cre-ft-smoke --smoke        # tiny run to check the pipeline
```

Run Python from the **project root**, not from inside `.venv/.../site-packages/mteb`. mteb ships a
`mteb/types` package that shadows the stdlib `types` module if it is the working directory.

## Layout

```
backend/
  main.py               FastAPI app; mounts frontend/ at /dashboard
  database.py           SQLite (sast_learning.db) engine + get_db()
  models.py             ORM: Feedback, SmartMemory                          [SAST — replace]
  api/routes.py         /api/scan, /api/feedback, /api/dashboard/metrics,
                        /api/smart-memory, /api/model/versions, /api/model/retrain
  ml/model_registry.py  versioned artefacts in models/ + models/registry.json [REUSE]
  ml/retraining.py      TF-IDF + LogisticRegression on feedback             [SAST — replace]
  scanner/git_utils.py  git diff → added-line chunks; full-repo file walk   [REUSE]
  scanner/sast_core.py  SmartMemory → ML → regex 3-stage scan               [SAST — replace]
frontend/               single-page dashboard (index.html, app.js, styles.css)
retrieval/
  eval_baseline.py      mteb AbsEncoder wrapper + AppsRetrieval eval (variant grid, summary table)
  bm25_search.py        code tokenizer + rank_bm25 as an mteb SearchProtocol model (no torch/mteb at import)
  query_clean.py        APPS problem-statement cleanup for --query-clean (pure Python)
  finetune.py           CodeRankEmbed fine-tuning on APPS train (CachedMNRL, optional hard negatives)
  submission.py         screening submission: PrePostPipelineEncoder, loads the fine-tuned model from the HF Hub
  push_model.py         upload a finetune.py output dir + model card to the HF Hub (HF_TOKEN)
  cache/                corpus embedding cache (gitignored)
seed_training_data.py   seeds SAST feedback rows                            [SAST — replace]
requirements.txt        web app only (Render image); requirements-retrieval.txt = CPU torch + mteb + ST
README.md, KAGGLE.md    submission write-up and Kaggle cells; old SAST README in docs/SAST_README.md
Dockerfile, render.yaml Render deployment (runs seed + retraining on boot)
```

## Reuse vs. replace

| Component | Status | Notes for retrieval |
|---|---|---|
| `backend/scanner/git_utils.py` | **Reuse** | `get_diff_chunks()` gives changed hunks between HEAD and its first parent, which is the basis for diff-based re-indexing (P1). `get_all_files()` + `EXCLUDED_DIRS` give full-index mode. Limits: only diffs HEAD vs `parents[0]` (needs an arbitrary `base..head` variant), chunks are *added-line hunks*, not whole functions (needs function-level chunking for retrieval), and deletions/renames are skipped (index needs tombstones). |
| `backend/ml/model_registry.py` | **Reuse** | Timestamped versions, `registry.json`, `active_version`, `MODEL_DIR` env override. Currently hard-wired to a joblib `{vectorizer, classifier, threshold}` payload, so it should be generalised to store index versions (embeddings + doc-id map + commit SHA). |
| `backend/main.py`, `database.py`, `api/routes.py` (`/api/model/versions`), `frontend/*` | **Reuse** | Shell for a query demo: add a `/api/search` route and a search box on the dashboard. |
| `scanner/sast_core.py` (regex `BASELINE_PATTERNS`, 3-stage scan) | **To be replaced** | SAST-specific. |
| `ml/retraining.py` (TF-IDF + LogReg) | **To be replaced** | SAST classifier. |
| `models.py` Feedback / SmartMemory, feedback + smart-memory routes, `seed_training_data.py` | **To be replaced** | SAST feedback loop. |

## Retrieval eval: `retrieval/eval_baseline.py`

- Written against the **installed mteb 2.21.8** (checked in source, not guessed):
  - `AbsEncoder.encode(inputs: DataLoader[BatchedInput], *, task_metadata, hf_split, hf_subset, prompt_type: PromptType | None, **encode_kwargs) -> Array`. Batches are dicts; `batch["text"]` is `list[str]`.
  - `ModelMeta` is pydantic with `extra="forbid"` and many required fields, so build it with `ModelMeta.create_empty(overwrites={...})`. Avoid `ModelMeta.from_sentence_transformer_model()`: in this version it always emits a deprecation warning.
  - `mteb.evaluate(model, task, encode_kwargs=..., cache=..., overwrite_strategy=...)`. By default it reads `~/.cache/mteb` with `"only-missing"` and would silently return old scores; the script passes `cache=None, overwrite_strategy="always"`.
  - `evaluate()` does not reload or unload a task whose data is already loaded, which is how `--smoke` subsets it.
  - Metric keys: `ndcg_at_10`, `mrr_at_10` in `results.task_results[0].scores["test"][0]`.
- **Presets** (`--preset`, `PRESETS` in the script). The settings come from the HF model card and the
  repo's `1_Pooling/config.json`, not from memory. It needs `trust_remote_code=True`:

  | Preset | Model | Params | Query prefix | Doc prefix | Pooling | License |
  |---|---|---|---|---|---|---|
  | `coderankembed` (default) | `nomic-ai/CodeRankEmbed` | 137M | `"Represent this query for searching relevant code: "` (card: "*must*") | none | CLS | MIT |

  **Dropped** (2026-09-28): `Salesforce/SFR-Embedding-Code-400M_R` still scored NDCG@10 0.0 after the buffer fix.
  `jinaai/jina-embeddings-v2-base-code` has remote code that needs transformers<5 (see below).
  Pooling comes from the repo; the preset value is only checked and a mismatch is logged as a warning.
  `--model <id>` alone selects the matching preset, and unknown ids run as `custom` with no prefixes.
  `--query-prefix` / `--doc-prefix` override the preset.
- Models are loaded in **fp32** (`model_kwargs={"torch_dtype": torch.float32}`). Some checkpoints
  are stored in fp16/bf16, and transformers v5 would otherwise keep that dtype.
- `nomic-ai/CodeRankEmbed` needs `einops`. sentence-transformers 6.1 requires transformers>=5.
- **transformers v5 + remote code, uninitialised buffers.** v5 loads on the meta device and refills
  every *non-persistent* buffer with `torch.empty_like`, trusting `_init_weights` to restore them.
  The remote code for NomicBert (CodeRankEmbed: rotary `inv_freq`, attention `norm_factor`) and
  Alibaba `new-impl` (SFR: `position_ids`, rotary `inv_freq`/`cos_cached`/`sin_cached`) computes
  those buffers in `__init__` and never re-inits them, so after loading they hold garbage.
  - SFR fails with a CUDA device-side assert (index out of bounds) during the first forward pass (see
    HF discussion Alibaba-NLP/new-impl #14).
  - CodeRankEmbed doesn't crash but silently gets garbage RoPE frequencies and attention scale.
  - Fix: `restore_nonpersistent_buffers()` rebuilds the model from its config (weight init skipped
    via `transformers.initialization.no_init_weights`) and copies the buffers over. It runs for every
    model, logs `Restored N ... (M had uninitialised values)`, and records `buffers_restored` in the
    results JSON.
- **jina-code doesn't work on transformers v5** (why it was dropped). Its remote code (`jinaai/jina-bert-v2-qk-post-norm`)
  imports `find_pruneable_heads_and_indices` and calls `get_extended_attention_mask` /
  `get_head_mask` / `invert_attention_mask`, all removed in v5. It would need transformers<5, which
  conflicts with sentence-transformers 6.x.
- Embeddings are L2-normalised, and similarity is cosine.
- `--device auto|cuda|cpu` (default `auto` = CUDA if `torch.cuda.is_available()`). The resolved
  device is logged at startup and recorded in the results JSON. Screening is CPU, so use GPU for fast iteration only.

### Results so far (full AppsRetrieval test split)

**Final (Phase 1 submission):** fine-tuned CodeRankEmbed (finetune.py plain run, 2 epochs, no hard negatives) +
desc-io, dense → NDCG@10 **0.4709**, MRR@10 **0.4303**. Hard negatives (3 epochs) scored worse on val
(MRR@10 0.7266 vs 0.7503), so they are not used.

| Preset | q_len / d_len | NDCG@10 | MRR@10 | Notes |
|---|---|---|---|---|
| `coderankembed` | 512 / 512 | 0.2368 | 0.2066 | pre-fix (cache p1); q_len 1024 gave no gain |
| `coderankembed` | 512 / 512 | **0.2368** | **0.2073** | **baseline**, with the buffer fix (cache p2), Kaggle-confirmed |
| `sfr-code-400m` | 512 / 512 | 0.0 | — | still 0.0 after the fix → preset removed |

Ablations (CodeRankEmbed 512/512, full split, 2026-09-28): dense/none **0.2368**, dense/desc-io **0.2420** (best →
default `--query-clean`), dense/desc 0.1824, bm25 ~0.05, hybrid RRF ~0.16 (hurts). The BM25/hybrid code stays only
to reproduce the ablation table.

### Ablations: query cleanup (A) and BM25 hybrid (B)

- `--query-clean` and `--retriever` take several values, and every combination runs as a variant
  `<retriever>/<clean>`. The model loads once and one summary table is printed at the end. The `time_s`
  column is the wall time of that variant's `mteb.evaluate()`. Corpus embeddings are memoised in memory
  across variants, so only the first dense variant pays for corpus encoding (or the cache load).
- **A) `--query-clean`** (`retrieval/query_clean.py`) rewrites the task's query texts before
  `evaluate()`, so it applies to every retriever. `none` = raw, `desc` = text before the first section
  header, `desc-io` = description + Input/Output sections (samples, examples, notes, constraints and
  explanations dropped). It handles `-----X-----` (Codeforces/CodeChef/AtCoder), `=====X=====` (HackerRank)
  and bare `Example 1:` / `Note:` / `Constraints:` lines (LeetCode). Statements that begin with a dropped
  section fall back to the raw text. On a 900-query test sample, median length is 1607 chars raw,
  1202 for desc-io and 738 for desc.
- **B) BM25** (`retrieval/bm25_search.py`, `BM25CodeSearch`) implements mteb's `SearchProtocol`
  (`index()` / `search()` in `mteb/models/models_protocols.py`; `abstasks/retrieval.py` calls a
  SearchProtocol directly and wraps a plain encoder in `SearchEncoderWrapper`). Tokens are identifiers,
  lowercased, split on snake_case and camelCase, keeping the compound too (`numRows` → `numrows num rows`).
  Numbers, 1-char parts and a short list of English function words are dropped. Scoring uses
  `rank_bm25.BM25Okapi` statistics (k1=1.5, b=0.75, eps=0.25) computed as a sparse matmul, which matches
  `get_scores()` to 1e-13 but avoids its per-doc Python loop. Zero-score docs are not returned.
- **Hybrid** = `mteb.HybridSearch([encoder, bm25], fusion_strategy="rrf")` (mteb's own
  `models/hybrid_wrappers.py`). Each sub-model returns its top 1000, and they are fused as
  `sum w_i / (rrf_k + rank)`. Flags: `--rrf-k` (default 60) and `--bm25-weight` (default 1.0, with dense
  weight 1.0).
- Outputs: `<out>` holds the summary with per-variant scores and timings, and `<out stem>_mteb/<variant>.json`
  holds mteb's official `TaskResult` (`TaskResult.to_disk`).

### Corpus embedding cache

- `<cache-dir>/<model>__len<doc_max_len>__<PREPROC_VERSION>__n<N>_<sha256[:16]>.npy`
  (`--cache-dir`, default `retrieval/cache/`).
  The hash is over the exact document texts (after the doc prefix) in the order mteb passes them, so a smoke subset, a
  different corpus, or another code version gets its own file automatically.
- **Bump `PREPROC_VERSION`** whenever document-side preprocessing changes (prefixes, chunking,
  normalisation) or the encoder is fixed. It's `p2` since the buffer fix, so `p1` files are stale. Query-side changes (prefix, `--query-max-len`) don't need a bump, since only docs are cached.
- The device is **not** part of the key. GPU and CPU embeddings differ only by float noise, but
  report official CPU numbers from a CPU-encoded corpus (use a separate `--cache-dir` or `--no-cache`).
- `--no-cache` disables reading and writing. It's safe to delete the directory at any time.

### Timing

- Logged and written to the results JSON under `timings`: `corpus_encode_s` (cache load time on a
  hit), `query_encode_s`, `total_s` (includes model load and dataset download on first run), and
  `corpus_cache_hits` / `corpus_cache_misses`.
- The corpus dominates cost (~9k code docs vs ~3.7k short queries). Doc length has the biggest
  effect on CPU time, because attention cost grows quadratically with sequence length.

## Environment gotchas

- **All model/eval runs happen on Kaggle.** The dev laptop (4 GB RAM) is for editing only:
  Windows Smart App Control blocks torch's unsigned DLLs there
  (`WinError 4551 ... Application Control policy has blocked this file`, surfacing as
  `WinError 1114` on `c10.dll`). Local checks are limited to `py_compile` and static review.
  The FastAPI/SAST parts don't need torch.
- **Branching:** work on `retrieval-baseline`; don't merge to `main` yet. Render auto-deploys `main`.
- `requirements.txt` has only the web app deps, so the Render image doesn't pull CUDA torch. Retrieval deps live in
  `requirements-retrieval.txt`, which pins `torch==2.14.0+cpu` from the PyTorch CPU index. On Kaggle don't install
  that file (it would replace the CUDA torch); `KAGGLE.md` pip-installs the other pins directly.
- `pydantic==2.5.2` works with mteb but prints a "protected namespace model_" warning, which is harmless.

## Fine-tuning: `retrieval/finetune.py`

- **No test data:** only the `default` config's **train** qrels are loaded (dataset path and revision come
  from mteb's AppsRetrieval metadata). Test queries and qrels are never read. A train-qrel query with
  `partition != "train"` aborts the run.
- **Validation:** `--val-size` (500) train queries are held out with a seeded shuffle. The val corpus is every
  train-qrel doc, so held-out positives compete with the training docs, as they do in the AppsRetrieval corpus.
  `--val-max-docs` caps the corpus (used by `--smoke`).
- **Same text as eval:** the preset query prefix, `--query-clean` (default desc-io) and mteb's title + text doc join.
- **Training** (ST 6.1 API checked in source: `sentence_transformers.sentence_transformer.losses`, `.evaluation`,
  `base.sampler.BatchSamplers`):
  - `CachedMultipleNegativesRankingLoss` with `--batch-size` 128 in-batch negatives and `--mini-batch-size` 16 GradCache chunks.
  - `NO_DUPLICATES` sampler, fp16 on CUDA, lr 2e-5, 10% warmup, 2 epochs, max len 512.
  - `report_to="none"`, since Kaggle's wandb would prompt for a login. `save_strategy="no"`: the best epoch is kept on CPU instead.
  - `CUDA_VISIBLE_DEVICES` defaults to `0`, because 2×T4 would trigger DataParallel.
- **Per-epoch log:** `InformationRetrievalEvaluator` runs before training (epoch 0 = base) and after every epoch.
  Val MRR@10 / NDCG@10 and epoch train time are logged. `--select best` (default) saves the best-val-MRR epoch,
  and if no epoch beats the base it warns and saves the final one.
- **`--hard-negatives N`** (default 0 = off): `mine_hard_negatives` with the **base** model before training
  (relative_margin 0.05, top-N, `n-tuple`). Candidates exclude val docs. Pairs without a valid negative are dropped,
  and the row counts are logged.
- **Output:** `--output-dir` gets `model.save()` plus `finetune_config.json` (base preset, `run_id`, history,
  timings). `eval_baseline.py --model <dir>` reads `base_preset` to apply the prefixes. The corpus cache key
  includes `run_id` (or a file-stat hash for other local dirs), so retraining into the same dir never reuses
  stale embeddings.
