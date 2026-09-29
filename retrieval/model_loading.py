"""
Load a sentence-transformers model in fp32 and repair what transformers v5 breaks.

Shared by eval_baseline.py, submission.py, finetune.py, precompute_corpus.py and the
Hugging Face Space. space/model_loading.py is a byte-for-byte copy of this file
(tests/test_space_sync.py checks it), so keep it free of repo-internal imports.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
import transformers
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("model_loading")


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
    """Load in fp32 and repair non-persistent buffers.

    To pin a Hub commit, snapshot_download(repo, revision=...) and pass the local dir:
    NomicBert's remote from_pretrained fetches Hub weights without a revision.
    """
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
