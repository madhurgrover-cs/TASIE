# Kaggle runbook

Notebook settings: **Accelerator GPU T4** (x1 or x2; `finetune.py` pins one GPU),
**Internet on**. For the push step, add a write token as a secret:
*Add-ons → Secrets → `HF_TOKEN`*, attached to the notebook.

Run each block as its own cell, in order. Outputs go to `/kaggle/working/`.

## 0. Hub-only eval (no training)

Fresh notebook, **Internet on**. Use Accelerator **None** to match CPU screening, or a
GPU with `--device cuda` for a faster check. This is one cell: everything runs in
subprocesses, so it needs no restart after pip.

```python
!git clone -q -b retrieval-baseline https://github.com/madhurgrover-cs/TASIE.git /kaggle/working/TASIE
%cd /kaggle/working/TASIE
!pip install -q mteb==2.21.8 sentence-transformers==6.1.0 rank-bm25==0.2.2 einops==0.8.2
!python retrieval/submission.py --model madhurr382/coderankembed-apps-ft --device cpu --out /kaggle/working/appsretrieval_results.json
!python -c "import json; s=json.load(open('/kaggle/working/appsretrieval_results.json'))['scores']['test'][0]; print('NDCG@10', s['ndcg_at_10'], 'MRR@10', s['mrr_at_10'])"
```

Expected: NDCG@10 ≈ 0.4720, MRR@10 ≈ 0.4319. `load_st_model` passes
`model_kwargs={"safe_serialization": True}` for NomicBert, so the Hub repo loads from
`model.safetensors`.

## 1. Setup

```python
!git clone -b retrieval-baseline https://github.com/madhurgrover-cs/TASIE.git /kaggle/working/TASIE
%cd /kaggle/working/TASIE
```

```python
# Keep Kaggle's CUDA torch: don't install requirements-retrieval.txt here (it pins the CPU wheel).
!pip install -q mteb==2.21.8 sentence-transformers==6.1.0 rank-bm25==0.2.2 einops==0.8.2
import torch, transformers, sentence_transformers, mteb
print(torch.__version__, torch.cuda.is_available(), transformers.__version__,
      sentence_transformers.__version__, mteb.__version__)
```

If pip upgrades packages that were already imported, restart the session and run
`%cd /kaggle/working/TASIE` again.

## 2. Train (final recipe: 2 epochs, no hard negatives)

```python
!python retrieval/finetune.py --output-dir /kaggle/working/cre-ft
```

The run prints a per-epoch table of validation MRR@10 and NDCG@10, and the saved epoch
is marked. To check the pipeline first, run the smoke version (~2 min):

```python
!python retrieval/finetune.py --output-dir /kaggle/working/cre-ft-smoke --smoke
```

If CUDA runs out of memory, add `--mini-batch-size 8`.

## 3. Evaluate the local model (GPU, fast iteration)

```python
!python retrieval/submission.py --model /kaggle/working/cre-ft --device cuda --out /kaggle/working/appsretrieval_results.json
```

`submission.py` is the submission pipeline (`PrePostPipelineEncoder`, `desc-io`
queries). For the research harness with an embedding cache and a summary table, run:

```python
!python retrieval/eval_baseline.py --model /kaggle/working/cre-ft --cache-dir /kaggle/working/emb_cache
```

## 4. Push to the Hugging Face Hub

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```python
REPO_ID = "madhurr382/coderankembed-apps-ft"
!python retrieval/push_model.py --model-dir /kaggle/working/cre-ft --repo-id {REPO_ID} --ndcg 0.4720 --mrr 0.4319 --also-bin
```

`--also-bin` also uploads a `pytorch_model.bin` copy, so a plain
`SentenceTransformer(REPO_ID, trust_remote_code=True)` works without the
`safe_serialization` kwarg.

If `/kaggle/working/cre-ft` is gone (new session), use this cell to re-push the card
and add the `.bin` from the Hub copy:

```python
REPO_ID = "madhurr382/coderankembed-apps-ft"
from huggingface_hub import snapshot_download
snapshot_download(REPO_ID, local_dir="/kaggle/working/hub-copy")
!python retrieval/push_model.py --model-dir /kaggle/working/hub-copy --repo-id {REPO_ID} --ndcg 0.4720 --mrr 0.4319 --also-bin
```

Add `--private` to create a private repo. The evaluators need read access, so make
it public before submitting.

## 5. Evaluate from the Hub on CPU (as screening runs it)

```python
!python retrieval/submission.py --model {REPO_ID} --device cpu --out /kaggle/working/appsretrieval_results.json
```

```python
import json
r = json.load(open("/kaggle/working/appsretrieval_results.json"))
s = r["scores"]["test"][0]
print(s["ndcg_at_10"], s["mrr_at_10"])
```

`DEFAULT_MODEL` in `retrieval/submission.py` is already `madhurr382/coderankembed-apps-ft`, so a plain
`python retrieval/submission.py` evaluates the published model.
