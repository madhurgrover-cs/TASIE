---
title: Code Search Demo
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Natural-language code search on APPS and a git repo
models:
  - madhurr382/coderankembed-apps-ft
datasets:
  - madhurr382/apps-corpus-index
  - madhurr382/repo-versions-index
  - CoIR-Retrieval/apps
---

# Natural-language → code search

Describe a programming problem and get ranked Python solutions from the 8,765-snippet
[APPS](https://huggingface.co/datasets/CoIR-Retrieval/apps) corpus used by MTEB `AppsRetrieval`.

- **Model:** [`madhurr382/coderankembed-apps-ft`](https://huggingface.co/madhurr382/coderankembed-apps-ft),
  `nomic-ai/CodeRankEmbed` fine-tuned on APPS train (AppsRetrieval test NDCG@10 0.4720, MRR@10 0.4319)
- **Queries** are cleaned (description + Input/Output spec kept, samples and notes dropped) and
  prefixed with `Represent this query for searching relevant code: `, like the benchmark submission.
- **Corpus embeddings** are precomputed on GPU
  ([`madhurr382/apps-corpus-index`](https://huggingface.co/datasets/madhurr382/apps-corpus-index)).
  The Space only encodes the query (on CPU by default; "GPU (optional)" uses ZeroGPU) and does an
  exact cosine search.
- **Versioned repo search:** [psf/requests](https://github.com/psf/requests) indexed at four commits
  from 2013 to 2024 with function/class-level chunks
  ([`madhurr382/repo-versions-index`](https://huggingface.co/datasets/madhurr382/repo-versions-index)).
  Pick the repo and a commit; results show path, qualified name and line range, with a GitHub link.
  Unchanged code is chunked and embedded once across versions (incremental, git-diff based builds).

Source: https://github.com/madhurgrover-cs/TASIE (`space/`, branch `phase2-versions`).
