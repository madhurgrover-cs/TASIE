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

# Retrieval baseline eval (writes appsretrieval_results.json, prints NDCG@10 / MRR@10)
python -m retrieval.eval_baseline                                   # --preset coderankembed (default)
python -m retrieval.eval_baseline --preset jina-code --out appsretrieval_results.jina-code.json
python -m retrieval.eval_baseline --smoke 5 300                     # 5 queries / 300 docs → appsretrieval_results.smoke.json
python -m retrieval.eval_baseline --query-max-len 128 --doc-max-len 512   # lengths are independent; --max-seq-length sets both (default 512)
python -m retrieval.eval_baseline --device cpu --cache-dir /kaggle/working/emb_cache   # --device auto|cuda|cpu (auto = cuda if available)
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
  eval_baseline.py      mteb AbsEncoder wrapper + AppsRetrieval eval
  cache/                corpus embedding cache (gitignored)
seed_training_data.py   seeds SAST feedback rows                            [SAST — replace]
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
- **Presets** (`--preset`, `PRESETS` in the script). The settings come from each HF model card and the
  repo's `1_Pooling/config.json`, not from memory. All three need `trust_remote_code=True`, and all are
  <500M params so they can run on CPU:

  | Preset | Model | Params | Query prefix | Doc prefix | Pooling | License |
  |---|---|---|---|---|---|---|
  | `coderankembed` (default) | `nomic-ai/CodeRankEmbed` | 137M | `"Represent this query for searching relevant code: "` (card: "*must*") | none | CLS | MIT |
  | `jina-code` | `jinaai/jina-embeddings-v2-base-code` | 161M | none | none | mean | Apache-2.0 |
  | `sfr-code-400m` | `Salesforce/SFR-Embedding-Code-400M_R` | 434M | none | none | CLS | **CC-BY-NC-4.0** |

  Pooling comes from the repo; the preset value is only checked and a mismatch is logged as a warning.
  SFR's "Instruct: ...\nQuery: " template in mteb is registered for the **2B** model only, so the 400M card uses raw text.
  `--model <id>` alone selects the matching preset, and unknown ids run as `custom` with no prefixes.
  `--query-prefix` / `--doc-prefix` override the preset.
- Models are loaded in **fp32** (`model_kwargs={"torch_dtype": torch.float32}`). The jina and SFR
  checkpoints are stored in fp16/bf16, and transformers v5 would otherwise keep that dtype.
- `nomic-ai/CodeRankEmbed` needs `einops`. sentence-transformers 6.1 requires transformers>=5, and
  the jina (`jinaai/jina-bert-v2-qk-post-norm`) and SFR (`Alibaba-NLP/new-impl`) remote code hasn't
  been verified on transformers v5 yet.
- Embeddings are L2-normalised, and similarity is cosine.
- `--device auto|cuda|cpu` (default `auto` = CUDA if `torch.cuda.is_available()`). The resolved
  device is logged at startup and recorded in the results JSON. Screening is CPU, so use GPU for fast iteration only.

### Results so far (full AppsRetrieval test split)

| Preset | q_len / d_len | NDCG@10 | MRR@10 | Notes |
|---|---|---|---|---|
| `coderankembed` | 512 / 512 | 0.2368 | 0.2066 | baseline; q_len 1024 gave no gain, so keep 512 |

### Corpus embedding cache

- `<cache-dir>/<model>__len<doc_max_len>__<PREPROC_VERSION>__n<N>_<sha256[:16]>.npy`
  (`--cache-dir`, default `retrieval/cache/`).
  The hash is over the exact document texts (after the doc prefix) in the order mteb passes them, so a smoke subset, a
  different corpus, or another code version gets its own file automatically.
- **Bump `PREPROC_VERSION`** whenever document-side preprocessing changes (prefixes, chunking,
  normalisation). Query-side changes (prefix, `--query-max-len`) don't need a bump, since only docs are cached.
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
- `requirements.txt` pins `torch==2.14.0`. On Linux (Docker/Render) plain `pip install` pulls the
  CUDA build (several GB). For CPU-only, install with
  `--extra-index-url https://download.pytorch.org/whl/cpu`.
- `pydantic==2.5.2` works with mteb but prints a "protected namespace model_" warning, which is harmless.
