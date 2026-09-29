# Deploying the code-search Space

Space: [`madhurr382/code-search-demo`](https://huggingface.co/spaces/madhurr382/code-search-demo)
(Gradio SDK 6.29.0). Source: `space/` on branch `phase2-versions` (Phase 1 version: `deploy-space`).
Never merge these branches to `main`, because `main` auto-deploys to Render.

```
Kaggle GPU:  retrieval/precompute_corpus.py ───────────► dataset madhurr382/apps-corpus-index
             retrieval/versioned/precompute_repo.py ───► dataset madhurr382/repo-versions-index
Space (CPU): app.py ── downloads both indexes + model@pinned revision ── encodes the query only
```

The Space never embeds code. It downloads the precomputed indexes at startup, then loads
the model at the exact revision recorded in the APPS manifest. The repo index must have
been embedded with that same revision, and the app checks this.

## Hardware (read first)

The app is **CPU only**. `madhurr382/code-search-demo` currently runs on **ZeroGPU**
(`zero-a10g`), and a free account can't switch it to CPU basic (the API returns HTTP 402).
ZeroGPU refuses to start an app without a `@spaces.GPU` function (`No @spaces.GPU function
detected during startup`), and it rejects the `torch==2.10.0+cpu` wheel. So this version
**will not start on the current Space**. Pick one:

1. **Create a CPU basic Space** (free) and deploy there, e.g. `madhurr382/code-search`:
   ```powershell
   python -c "from huggingface_hub import HfApi; print(HfApi().create_repo('madhurr382/code-search', repo_type='space', space_sdk='gradio', space_hardware='cpu-basic'))"
   ```
   Then use that id in step (c)4. New free Spaces default to CPU basic.
2. Switch `code-search-demo` to CPU basic under *Settings → Space hardware* (needs PRO).
3. Stay on ZeroGPU with the Phase 1 app from branch `deploy-space`, which has an optional
   GPU device for exactly this reason.

Check a Space's hardware at any time:
```powershell
python -c "from huggingface_hub import HfApi; print(HfApi().get_space_runtime('madhurr382/code-search-demo').requested_hardware)"
```

## (a) Kaggle: APPS corpus index

Already done (`madhurr382/apps-corpus-index`, commit `dc58cbf`, model revision
`c9f6787afd037f4981130ce48335723ed7e65057`). To rebuild: GPU T4, Internet on, secret `HF_TOKEN`.

```python
!git clone -q -b phase2-versions https://github.com/madhurgrover-cs/TASIE.git /kaggle/working/TASIE
%cd /kaggle/working/TASIE
# Kaggle's CUDA torch (2.10) stays; pin the rest to the versions the model was trained with.
!pip install -q transformers==5.0.0 sentence-transformers==6.1.0 einops==0.8.2
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!python retrieval/precompute_corpus.py --out-dir /kaggle/working/apps-corpus-index
```

If you push a new model, rebuild **both** indexes. The Space serves the model revision in
the APPS manifest and refuses a repo index built with a different one.

## (b) Kaggle: versioned repo index (Phase 2)

Indexes [psf/requests](https://github.com/psf/requests) at `v2.0.0` (2013), `v2.12.0` (2016),
`v2.25.0` (2020) and `v2.32.3` (2024). It prints:
- the incremental chain (full build of v2.0.0, then each tag from the previous one);
- a cold full build of every tag, to compare against;
- a step benchmark on adjacent releases (`v2.32.2→v2.32.3`, `v2.31.0→v2.32.0`), the
  everyday "rebuild after a change" case;
- 3 example queries per commit;

then uploads to `madhurr382/repo-versions-index`.

Notebook: **GPU T4**, **Internet on**, secret **`HF_TOKEN`** (write). One cell:

```python
!git clone -q -b phase2-versions https://github.com/madhurgrover-cs/TASIE.git /kaggle/working/TASIE
%cd /kaggle/working/TASIE
!pip install -q transformers==5.0.0 sentence-transformers==6.1.0 einops==0.8.2
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
# --revision must equal the APPS index's model_revision (the Space checks this)
!python -m retrieval.versioned.precompute_repo \
    --revision c9f6787afd037f4981130ce48335723ed7e65057 \
    --workdir /kaggle/working/versioned-work --out /kaggle/working/repo-versions-index
```

Expected output shape (numbers from a local dry run with the numpy hashing embedder;
with the real model on GPU, `embed` dominates and the incremental speed-up tracks
`embedded` vs `full embedded`):

```
chain: 4401620 v2.0.0  [full] 740 chunks in 109 files | ... embedded 740 chunks ...
chain: 362da46 v2.12.0 [incremental from 4401620] 1234 chunks ... embedded 876 chunks, reused 358 ...
chain: 03957eb v2.25.0 [incremental from 362da46] 697 chunks ... embedded 271 chunks, reused 426 ...
chain: 0e322af v2.32.3 [incremental from 03957eb] 773 chunks ... embedded 598 chunks, reused 175 ...
step v2.32.2 -> v2.32.3: 3 files changed | incremental ... (6 embedded, 767 reused) vs full ... (773 embedded)
step v2.31.0 -> v2.32.0: 96 files changed | incremental ... (121 embedded, 651 reused) vs full ... (772 embedded)
ref  commit  files chunks mode  rechunked embedded reused  incr_s  full_s speedup
...
== v2.32.3 · 0e322af (2024-05-29)
  Q: get proxy settings from environment variables   [.. ms]
     <score>  <path>:<start>-<end>  <qualified name>
...
Uploaded /kaggle/working/repo-versions-index -> https://huggingface.co/datasets/madhurr382/repo-versions-index
```

Optional: measure CPU build and query times, as on the Space (no upload):
```python
!python -m retrieval.versioned.precompute_repo --device cpu --no-upload --skip-full \
    --revision c9f6787afd037f4981130ce48335723ed7e65057 --out /kaggle/working/cpu-check
```

The CLI works on any clone, e.g. to index another commit later:
```python
!python -m retrieval.versioned --store /kaggle/working/repo-versions-index --device cuda \
    update --repo /kaggle/working/versioned-work/repo --commit v2.32.2
!python -m retrieval.versioned --store /kaggle/working/repo-versions-index list-versions
!python -m retrieval.versioned --store /kaggle/working/repo-versions-index --device cuda \
    query --commit v2.32.2 -k 3 "rebuild proxy configuration on redirect"
```

## (c) Re-upload `space/`

From the repo root on the laptop. Nothing here needs torch.

1. Activate the project venv, which has the `hf` CLI (`.venv\Scripts\activate`), and log in
   once with a write token: `hf auth login`. Check it with `hf auth whoami`.
2. Get the branch and run the tests. Use `pip install -r requirements-dev.txt` once, for pytest:
   ```powershell
   git checkout phase2-versions; git pull
   python -m pytest tests -q
   ```
3. Check that (b) finished and that the repo index uses the right model:
   ```powershell
   python -c "import json,urllib.request as u; r=json.load(u.urlopen('https://huggingface.co/datasets/madhurr382/repo-versions-index/resolve/main/registry.json')); print(r['model_id'], len(r['versions']))"
   ```
   Expected: `madhurr382/coderankembed-apps-ft@c9f6787afd037f4981130ce48335723ed7e65057 4`.
4. Upload to a **CPU basic** Space (see *Hardware*). Replace `SPACE` with the id you use:
   ```powershell
   $SPACE = "madhurr382/code-search"
   hf upload $SPACE space . --repo-type space --exclude "__pycache__/*" --exclude "*.pyc" --commit-message "Phase 2: versioned repo search"
   ```
5. Watch *Logs* on the Space page. At startup it downloads both indexes and the model, then logs
   `Ready: {'APPS corpus': 8765, 'psf/requests@4401620': ..., ...}`. If the repo index can't be
   loaded (e.g. wrong model revision), the app logs the reason and serves APPS only.

Both indexes and the model are public, so the Space needs no secrets. Optional variables
(*Settings → Variables*): `INDEX_REPO`, `INDEX_REVISION`, `REPO_INDEX` (set to empty to
disable the repo source), `REPO_INDEX_REVISION`, `MODEL_ID`.

## Troubleshooting

- **`No @spaces.GPU function detected during startup`**: the Space is on ZeroGPU; see *Hardware*.
- **`index ... was built with X, app expects Y`**: `MODEL_ID` doesn't match the APPS index.
- **`Repo index ... not loaded ... embedded with ...`**: rerun (b) with `--revision` set to
  the APPS manifest's `model_revision`.
- **`embeddings.npy checksum mismatch`**: the APPS upload was interrupted; rerun (a).
- **Changing `space/query_clean.py`, `space/model_loading.py` or `space/versioned_store.py`**:
  edit the originals (`retrieval/query_clean.py`, `retrieval/model_loading.py`,
  `retrieval/versioned/store.py`) and copy them over. `tests/test_space_sync.py` fails
  if the copies drift apart.

## Code layout

| File | Role |
|---|---|
| `space/app.py` | Gradio UI (source picker, commit dropdown, examples) and `build_state()` (downloads, loading, warm-up) |
| `space/search.py` | `SearchSource` interface, `DenseIndexSource` (APPS), `SearchEngine` (encode once, search selected sources) |
| `space/versioned_source.py` | `VersionedRepoSource`: one `SearchSource` per indexed commit; hits carry path / name / lines / commit |
| `space/query_encoder.py` | `STQueryEncoder`: desc-io cleanup + query prefix, as in `retrieval/submission.py` |
| `space/query_clean.py`, `model_loading.py`, `versioned_store.py` | verbatim copies of `retrieval/` modules |
| `retrieval/precompute_corpus.py` | Kaggle: APPS corpus index |
| `retrieval/versioned/` | chunker, content-hash cache, store/registry, builder, searcher, CLI, `precompute_repo.py` |
| `tests/` | `python -m pytest tests`: fake embedders and a throwaway git repo, no torch; gradio tests skip without gradio |
