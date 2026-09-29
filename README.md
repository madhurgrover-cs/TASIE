# Agentic Code Intelligence: natural-language → code retrieval

Submission for Samsung's **Agentic Code Intelligence** hackathon: given a natural-language
query, rank the code snippets that answer it. **P0** is the screening benchmark (MTEB
`AppsRetrieval`, CPU); **P1** is retrieval across versions of a codebase, with fast
incremental re-indexing.

**Live demo:** https://huggingface.co/spaces/madhurr382/code-search-demo (APPS corpus and
psf/requests at 4 commits) ·
**Model:** [`madhurr382/coderankembed-apps-ft`](https://huggingface.co/madhurr382/coderankembed-apps-ft) ·
**Indexes:** [`madhurr382/apps-corpus-index`](https://huggingface.co/datasets/madhurr382/apps-corpus-index),
[`madhurr382/repo-versions-index`](https://huggingface.co/datasets/madhurr382/repo-versions-index)

## P0: AppsRetrieval

- **Benchmark:** MTEB `AppsRetrieval` (CoIR, dataset `CoIR-Retrieval/apps`, split `test`)
- **Metrics:** NDCG@10 (main score) and MRR@10
- **Constraint:** CPU-only inference

| | NDCG@10 | MRR@10 |
|---|---|---|
| **Final: fine-tuned CodeRankEmbed + `desc-io` query cleanup** | **0.4720** | **0.4319** |
| Base CodeRankEmbed, raw queries | 0.2368 | 0.2073 |

### Problem

APPS queries are full competitive-programming statements (median ~1.6k characters): a
story-style description, Input/Output specs, sample tests, notes and constraints. The
~9k-document corpus is Python solutions. Off-the-shelf code embedders are trained
mostly on docstring → function pairs, so they match short intents well but get lost in
long statements, where samples and constraints make up most of the text.

### Approach

1. **Encoder:** [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed)
   (137M, MIT, CLS pooling), a bi-encoder that is small enough for CPU. Queries use its
   required prefix `"Represent this query for searching relevant code: "`, documents
   have none. Embeddings are L2-normalised and scored by cosine similarity.
2. **Query cleanup (`desc-io`, [`retrieval/query_clean.py`](retrieval/query_clean.py)):**
   keep the problem description and the Input/Output sections, and drop sample tests,
   examples, notes, explanations and constraints. It handles Codeforces/CodeChef/AtCoder
   `-----Input-----` headers, HackerRank `=====X=====` headers and LeetCode-style bare
   `Example 1:` / `Note:` lines. Statements that start with a dropped section are
   kept unchanged.
3. **Fine-tuning ([`retrieval/finetune.py`](retrieval/finetune.py)):** contrastive training
   on the APPS **train** split only (test queries and qrels are never loaded).
   `CachedMultipleNegativesRankingLoss` with 128 in-batch negatives, lr 2e-5, 10% warmup,
   2 epochs, max length 512, fp16 on a Kaggle T4, and a `NO_DUPLICATES` sampler.
   Queries go through the same prefix + `desc-io` cleanup as at inference time. 500
   train queries are held out for validation, and the epoch with the best validation
   MRR@10 is kept.
4. **transformers v5 fix:** v5 leaves remote-code non-persistent buffers uninitialised
   (NomicBert's rotary `inv_freq` and attention `norm_factor`), which silently corrupts
   embeddings. `restore_nonpersistent_buffers()` in
   [`retrieval/model_loading.py`](retrieval/model_loading.py) rebuilds them after every load.

The submission encoder is `PrePostPipelineEncoder` in
[`retrieval/submission.py`](retrieval/submission.py): an mteb `AbsEncoder` that does
the query cleanup and prefixing inside `encode()` and loads the fine-tuned weights
from the Hugging Face Hub: [`madhurr382/coderankembed-apps-ft`](https://huggingface.co/madhurr382/coderankembed-apps-ft).

### Results

Full `AppsRetrieval` test split, CodeRankEmbed, query/doc max length 512.

| Step | Retrieval | Queries | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| Base model | dense | raw | 0.2368 | 0.2073 |
| + query cleanup | dense | `desc-io` | 0.2420 | — |
| **+ fine-tuning (final)** | dense | `desc-io` | **0.4720** | **0.4319** |

Ablations that did not make it in:

| Variant | Result | Why dropped |
|---|---|---|
| Query cleanup `desc` (description only) | NDCG@10 0.1824 | the I/O spec carries useful signal |
| BM25 over code identifiers (`retrieval/bm25_search.py`) | NDCG@10 ~0.05 | little lexical overlap between statements and code |
| Hybrid dense + BM25 (RRF, k=60) | NDCG@10 ~0.16 | BM25 noise drags dense down |
| Fine-tuning + mined hard negatives, 3 epochs | val MRR@10 0.7266 vs 0.7503 | worse on validation |
| `Salesforce/SFR-Embedding-Code-400M_R` | NDCG@10 0.0 | broken under transformers v5 remote code |
| `jinaai/jina-embeddings-v2-base-code` | — | remote code needs transformers<5 |

## P1: retrieval across versions

[`retrieval/versioned/`](retrieval/versioned) indexes a git repo at any commit and rebuilds
indexes incrementally:

- **Chunks:** Python functions, methods (`Session.request`), class headers and top-level blocks
  via `ast`, each with path and line range; one chunk per file for other text files and for
  code that doesn't parse.
- **Content-hash cache:** a chunk is embedded only if its (normalised-code hash, model@revision)
  is new. Code that is unchanged, moved or re-indented is never re-embedded, and code and
  vectors are stored once across versions.
- **Incremental build:** from any indexed commit, only files in `git diff` are re-chunked. It
  produces the same index as a full build (tested). A registry records each commit's base,
  build time, and chunks reused vs newly embedded.

Results on [psf/requests](https://github.com/psf/requests), model `madhurr382/coderankembed-apps-ft`,
Kaggle T4 GPU (from `registry.json` / `benchmark.json` in
[`madhurr382/repo-versions-index`](https://huggingface.co/datasets/madhurr382/repo-versions-index)):

| Rebuild after a change | Files changed | Chunks embedded (incremental vs full) | Incremental | Full | Speedup |
|---|---|---|---|---|---|
| v2.32.2 → v2.32.3 | 3 | 6 vs 773 | 0.42 s | 18.9 s | **45×** |
| v2.31.0 → v2.32.0 | 96 | 121 vs 772 | 4.6 s | 18.3 s | **3.9×** |

Across the 4 indexed versions (v2.0.0 2013, v2.12.0 2016, v2.25.0 2020, v2.32.3 2024):
3,444 chunk rows, but only 2,478 distinct chunks embedded and stored, so **28% of the embedding
work is saved** (52.8 s total build time vs 74.5 s for four cold full builds). The tags are years
apart (83–100% of files changed), so the adjacent-release rows show the everyday case.

The same query at each version, "get proxy settings from environment variables", top result:

| Version | Top result | Lines | Score |
|---|---|---|---|
| v2.0.0 | `requests/utils.py` · `get_environ_proxies` | 396–425 | 0.573 |
| v2.12.0 | `requests/utils.py` · `get_environ_proxies` | 611–620 | 0.609 |
| v2.25.0 | `requests/utils.py` · `get_environ_proxies` | 766–775 | 0.611 |
| v2.32.3 | `src/requests/utils.py` · `get_environ_proxies` | 826–835 | 0.611 |

Queries take about 20 ms on the Kaggle GPU and 22–46 ms on CPU on the live Space.

### Versioned CLI

```bash
pip install -r requirements-retrieval.txt
git clone https://github.com/psf/requests ../requests
python -m retrieval.versioned index  --repo ../requests --commit v2.31.0          # full build
python -m retrieval.versioned update --repo ../requests --commit v2.32.0          # incremental, from the nearest indexed commit
python -m retrieval.versioned query  --commit v2.32.0 -k 5 "get proxy settings from environment variables"
python -m retrieval.versioned list-versions                                       # registry: base, build time, reused vs embedded
```

Options: `--store DIR` (default `versioned-index/`), `--device cpu|cuda|auto`, and
`--embedder hashing` for a torch-free dry run. `python -m retrieval.versioned.precompute_repo`
rebuilds the published index and benchmark (see [DEPLOY_SPACE.md](DEPLOY_SPACE.md)).
The tests run without torch, using fake embedders and a throwaway git repo:
`pip install -r requirements-dev.txt && python -m pytest tests`.

## Reproduce

### Evaluate the submission on CPU

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-retrieval.txt                # CPU torch wheel + mteb + sentence-transformers
python retrieval/submission.py                           # uses madhurr382/coderankembed-apps-ft; writes appsretrieval_results.json
python retrieval/submission.py --smoke 5 300             # quick pipeline check
```

`appsretrieval_results.json` is mteb's `TaskResult.to_dict()`. The scores are under
`scores.test[0].ndcg_at_10` / `mrr_at_10`. Run from the repository root.

Corpus encoding (~9k docs at 512 tokens) dominates CPU time. Queries are short after cleanup.

### Evaluate on Kaggle GPU

See [KAGGLE.md](KAGGLE.md) for copy-paste cells. `--device cuda` only speeds up
iteration: GPU and CPU embeddings differ by float noise, and the official numbers
are from CPU.

### Train

```bash
python retrieval/finetune.py --output-dir /kaggle/working/cre-ft          # final recipe (T4, 2 epochs)
python retrieval/finetune.py --output-dir /kaggle/working/cre-ft-smoke --smoke   # pipeline check
python retrieval/eval_baseline.py --model /kaggle/working/cre-ft          # eval a local dir
HF_TOKEN=hf_... python retrieval/push_model.py --model-dir /kaggle/working/cre-ft \
    --repo-id madhurr382/coderankembed-apps-ft --ndcg 0.4720 --mrr 0.4319 --also-bin   # publish
```

`finetune.py` writes the model plus `finetune_config.json` (base preset, run id,
per-epoch validation metrics). `eval_baseline.py` also reproduces the ablations:

```bash
python retrieval/eval_baseline.py --query-clean none desc desc-io   # query cleanup ablation
python retrieval/eval_baseline.py --retriever dense bm25 hybrid     # BM25 / hybrid ablation
```

## Repository layout

```
retrieval/
  submission.py      PrePostPipelineEncoder + AppsRetrieval eval → appsretrieval_results.json
  versioned/         P1: chunker, content-hash cache, store/registry, builder, searcher, CLI
  precompute_corpus.py   APPS corpus index for the Space
  model_loading.py   load_st_model: fp32, NomicBert Hub loading, transformers v5 buffer fix
  finetune.py        CodeRankEmbed fine-tuning on APPS train
  push_model.py      upload a fine-tuned dir to the Hugging Face Hub (+ model card)
  eval_baseline.py   research harness: presets, embedding cache, ablation grid
  query_clean.py     APPS statement cleanup (pure Python)
  bm25_search.py     BM25 baseline as an mteb SearchProtocol model
space/                       Hugging Face Space (Gradio): APPS + versioned repo search, see DEPLOY_SPACE.md
tests/                       pytest, no torch needed
requirements-retrieval.txt   retrieval deps (CPU torch)
requirements.txt             web app deps only (Render image)
backend/, frontend/          FastAPI + dashboard shell from the original SAST IQ project
```

The repository started as **SAST IQ**, a self-learning SAST scanner. Its README is kept at
[docs/SAST_README.md](docs/SAST_README.md), and the web app still runs with
`pip install -r requirements.txt && python -m uvicorn backend.main:app`.
