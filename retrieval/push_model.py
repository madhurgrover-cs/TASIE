"""
Upload a fine-tuned model dir (retrieval/finetune.py --output-dir) to the Hugging Face Hub.

    export HF_TOKEN=hf_...            # write token; on Kaggle use a secret (see KAGGLE.md)
    python retrieval/push_model.py --model-dir /kaggle/working/cre-ft --repo-id <user>/coderankembed-apps-ft
    python retrieval/push_model.py ... --private --ndcg 0.4720 --mrr 0.4319 --also-bin

Uploads the model files, finetune_config.json and a generated README.md model card
(skips the trainer/ scratch dir). --also-bin adds a pytorch_model.bin copy of
model.safetensors: NomicBert's remote from_pretrained looks for pytorch_model.bin on
the Hub unless model_kwargs={"safe_serialization": True} is passed, so the copy makes
a plain SentenceTransformer(repo, trust_remote_code=True) work too. Only --also-bin
imports torch.

To re-push (new card / add the .bin) without the training dir, download the repo first:
    python -c "from huggingface_hub import snapshot_download as d; d('<repo>', local_dir='hub-copy')"
    python retrieval/push_model.py --model-dir hub-copy --repo-id <repo> --also-bin ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

FINETUNE_CONFIG = "finetune_config.json"  # same name as eval_baseline.FINETUNE_CONFIG
BIN_NAME = "pytorch_model.bin"


def write_bin_copy(model_dir: Path) -> Path:
    """pytorch_model.bin with the same state dict as model.safetensors."""
    import torch
    from safetensors.torch import load_file

    src = model_dir / "model.safetensors"
    if not src.is_file():
        raise SystemExit(f"--also-bin: {src} not found")
    dst = model_dir / BIN_NAME
    torch.save(load_file(str(src), device="cpu"), dst)
    return dst


def model_card(repo_id: str, cfg: dict, ndcg: float | None, mrr: float | None, has_bin: bool) -> str:
    bin_note = ("(this repo also ships a `pytorch_model.bin` copy, so loading without the flag works too)"
                if has_bin else "(this repo has only `model.safetensors`)")
    base = cfg.get("base_model", "nomic-ai/CodeRankEmbed")
    prefix = cfg.get("query_prefix", "Represent this query for searching relevant code: ")
    rows = "\n".join(
        f"| {h['epoch']} | {h['val_mrr@10']:.4f} | {h['val_ndcg@10']:.4f} |"
        for h in cfg.get("history", []) if h.get("val_mrr@10") is not None
    )
    hn = cfg.get("hard_negatives")
    test = (f"\n**AppsRetrieval test (MTEB):** NDCG@10 {ndcg:.4f}, MRR@10 {mrr:.4f} "
            f"(queries cleaned with `{cfg.get('query_clean', 'desc-io')}`).\n") if ndcg is not None and mrr is not None else ""
    return f"""---
license: mit
base_model: {base}
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- code-search
- sentence-transformers
- mteb
datasets:
- CoIR-Retrieval/apps
---

# {repo_id.split('/')[-1]}

`{base}` fine-tuned on the **train** split of CoIR-Retrieval/apps (natural-language
problem statement → Python solution) for the MTEB `AppsRetrieval` task. No test
queries or qrels were used.
{test}
## Usage

Queries need the base model's prefix; documents (code) have none. Load with
`trust_remote_code=True` (NomicBert) and `model_kwargs={{"safe_serialization": True}}`:
NomicBert's remote `from_pretrained` otherwise looks for `pytorch_model.bin` on the Hub
{bin_note}.

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("{repo_id}", trust_remote_code=True,
                        model_kwargs={{"safe_serialization": True}})
q = m.encode(["{prefix}" + "Given an array, return the length of its longest increasing subsequence."],
             normalize_embeddings=True)
d = m.encode(["def lis(a): ..."], normalize_embeddings=True)
print(q @ d.T)
```

For the best scores, clean APPS-style statements first: keep the description and the
Input/Output sections, drop samples, notes and constraints (`retrieval/query_clean.py`,
mode `desc-io`).

**transformers v5:** its loader leaves NomicBert's non-persistent buffers (rotary
`inv_freq`, attention `norm_factor`) uninitialised. The repo's
`retrieval/eval_baseline.py::load_st_model` rebuilds them after loading.

## Training

- Loss: CachedMultipleNegativesRankingLoss, batch {cfg.get('args', {}).get('batch_size', 128)}, lr {cfg.get('args', {}).get('lr', 2e-5)}, {cfg.get('args', {}).get('epochs', 2)} epochs, max len {cfg.get('args', {}).get('max_seq_length', 512)}, NO_DUPLICATES sampler
- Hard negatives: {"none" if not hn else hn.get("num_negatives")}
- Train rows: {cfg.get('n_train_rows')}; validation: {cfg.get('n_val_queries')} held-out train queries over {cfg.get('n_val_docs')} docs
- Saved epoch: {cfg.get('saved_epoch')} (best validation MRR@10)

| epoch | val MRR@10 | val NDCG@10 |
|---|---|---|
{rows}

Full run config: `finetune_config.json`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="finetune.py --output-dir")
    ap.add_argument("--repo-id", required=True, help="<user or org>/<name>")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--ndcg", type=float, default=None, help="AppsRetrieval test NDCG@10 for the card")
    ap.add_argument("--mrr", type=float, default=None, help="AppsRetrieval test MRR@10 for the card")
    ap.add_argument("--also-bin", action="store_true",
                    help="also upload a pytorch_model.bin copy of model.safetensors (needs torch)")
    ap.add_argument("--commit-message", default="Upload fine-tuned CodeRankEmbed (AppsRetrieval)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (needs a write token).")
    cfg_path = args.model_dir / FINETUNE_CONFIG
    if not cfg_path.is_file():
        raise SystemExit(f"{cfg_path} not found: --model-dir must be a finetune.py output dir.")
    cfg = json.loads(cfg_path.read_text())

    ignore = ["trainer/*", "trainer/**", "checkpoint-*/**", "*.tmp", ".cache/**"]
    if args.also_bin:
        print(f"Wrote {write_bin_copy(args.model_dir)}")
    else:
        ignore.append(BIN_NAME)  # never upload a stale copy from an earlier --also-bin run

    # README.md is written into the model dir so it goes up in the same commit.
    card = model_card(args.repo_id, cfg, args.ndcg, args.mrr, has_bin=args.also_bin)
    (args.model_dir / "README.md").write_text(card, encoding="utf-8")

    api = HfApi(token=token)
    url = api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        commit_message=args.commit_message,
        ignore_patterns=ignore,
    )
    print(f"Uploaded {args.model_dir} -> {url}")
    print(f"  commit {info.oid}")
    print(f"  eval:  python retrieval/submission.py --model {args.repo_id}")


if __name__ == "__main__":
    main()
