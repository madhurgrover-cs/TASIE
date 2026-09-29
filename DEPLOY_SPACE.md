# Deploying the code-search Space

Space: [`madhurr382/code-search-demo`](https://huggingface.co/spaces/madhurr382/code-search-demo)
(Gradio SDK 6.29.0, Python 3.11, CPU basic). Source: `space/` on branch `deploy-space`.
Don't merge this branch to `main`, because `main` auto-deploys to Render.

```
Kaggle GPU:  retrieval/precompute_corpus.py ──► dataset madhurr382/apps-corpus-index
                                                 (embeddings.npy, docs.jsonl, manifest.json)
Space (CPU): app.py ── downloads index + model@manifest revision ── encodes the query only
```

The Space never embeds the corpus. It downloads the precomputed index at startup,
then loads the model at the exact revision recorded in `manifest.json`, so query and
corpus embeddings always come from the same weights.

## (a) Kaggle: precompute and upload the index

Notebook settings: **Accelerator GPU T4**, **Internet on**, and the secret **`HF_TOKEN`**
(a write token) attached under *Add-ons → Secrets*. Run each block as its own cell.

```python
!git clone -q -b deploy-space https://github.com/madhurgrover-cs/TASIE.git /kaggle/working/TASIE
%cd /kaggle/working/TASIE
# Kaggle's CUDA torch (2.10) stays; pin the rest to the versions the model was trained with.
!pip install -q transformers==5.0.0 sentence-transformers==6.1.0 einops==0.8.2
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Optional smoke test (200 docs, no upload, ~1 min):

```python
!python retrieval/precompute_corpus.py --limit 200 --no-upload --out-dir /kaggle/working/index-smoke
```

Full run: embed all 8,765 docs on the GPU, then upload to `madhurr382/apps-corpus-index`.

```python
!python retrieval/precompute_corpus.py --out-dir /kaggle/working/apps-corpus-index
```

It prints `Uploaded ... -> https://huggingface.co/datasets/madhurr382/apps-corpus-index`.
The manifest records the model's Hub commit (`model_revision`). If you push a new model
later, rerun this cell, or the Space keeps serving the old model revision along with
its matching index.

Optional: check the Space backend on Kaggle before pushing. It runs on CPU, the same
path as the Space:

```python
!pip install -q gradio==6.29.0
%cd /kaggle/working/TASIE/space
from app import build_engine
engine = build_engine()
r = engine.search("find the length of the longest increasing subsequence", k=3)
print(f"{r.total_ms:.0f} ms", [(h.doc_id, round(h.score, 4)) for h in r.hits])
print(r.hits[0].code[:400])
%cd /kaggle/working/TASIE
```

## (b) Push `space/` to the Space

From the repo root on the laptop, on branch `deploy-space`. Nothing here needs torch.

1. Get the `hf` CLI (huggingface_hub ≥ 1.0). The project `.venv` already has it: run
   `.venv\Scriptsctivate` first. Otherwise install it:
   ```powershell
   pip install -U huggingface_hub
   ```
2. Log in with a **write** token (https://huggingface.co/settings/tokens):
   ```powershell
   hf auth login
   ```
   Alternatively, set `$env:HF_TOKEN = "hf_..."` for the session.
3. Make sure the branch is current and the tests pass:
   ```powershell
   git checkout deploy-space; git pull
   python -m unittest discover -s tests
   ```
4. Upload the folder to the Space root. This replaces the Space's placeholder `README.md`:
   ```powershell
   hf upload madhurr382/code-search-demo space . --repo-type space --exclude "__pycache__/*" --exclude "*.pyc" --commit-message "Deploy code-search demo from TASIE deploy-space"
   ```
5. Open https://huggingface.co/spaces/madhurr382/code-search-demo. The first build installs
   CPU torch and the pinned stack. At startup the app downloads the index (~35 MB) and the
   model (~550 MB), then logs `Ready: 8765 docs (dim 768) ...`. Build and run logs are on
   the Space page under *Logs*.

The index repo and the model repo are both public, so the Space needs no secrets.
If you ever make them private, add `HF_TOKEN` as a Space secret (*Settings → Variables
and secrets*); `snapshot_download` picks it up automatically.

Optional overrides (*Settings → Variables*): `INDEX_REPO`, `INDEX_REVISION` (pin an index
commit), and `MODEL_ID`, which must match the index manifest or the app refuses to start.

## Troubleshooting

- **`index ... was built with X, app expects Y`**: `MODEL_ID` doesn't match the index.
  Rebuild the index with (a), or fix the variable.
- **`embeddings.npy checksum mismatch`**: the upload was interrupted. Rerun the full cell in (a).
- **Build fails resolving torch**: `space/requirements.txt` needs its
  `--extra-index-url https://download.pytorch.org/whl/cpu` line for `torch==2.10.0+cpu`.
- **Changing `space/query_clean.py` or `space/model_loading.py`**: edit the originals in
  `retrieval/` and copy them over. `tests/test_space_sync.py` fails if the copies drift apart.

## Code layout

| File | Role |
|---|---|
| `space/app.py` | Gradio Blocks UI and `build_engine()` (downloads, loading, warm-up) |
| `space/search.py` | `SearchSource` interface, `DenseIndexSource` (exact cosine), `SearchEngine` (merges sources by score, timings) |
| `space/query_encoder.py` | `STQueryEncoder`: desc-io cleanup + query prefix, as in `retrieval/submission.py` |
| `space/query_clean.py`, `space/model_loading.py` | verbatim copies of the `retrieval/` modules |
| `retrieval/precompute_corpus.py` | Kaggle job that builds and uploads the index |
| `tests/` | fake-embedder tests: `python -m unittest discover -s tests` (the gradio tests skip without gradio) |

A versioned git-repo index, planned for the next phase, plugs in as another `SearchSource`
whose hits carry `meta={"commit", "path", ...}`. The engine already merges several sources
by score, and the UI shows a source picker as soon as there is more than one.
