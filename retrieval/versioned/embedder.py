"""
Pluggable embedders. The builder and searcher only need:

    model_id: str                          cache key; include the weights revision
    embed_documents(texts) -> [n, dim]     code chunks, no prefix
    embed_query(text) -> [dim]             natural-language query

HashingEmbedder is numpy-only (tests, dry runs on machines without torch).
STEmbedder is the fine-tuned CodeRankEmbed, prepared exactly like the Space and
retrieval/submission.py: desc-io query cleanup + query prefix, fp32 via load_st_model,
weights loaded from a snapshot at a pinned Hub revision.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "madhurr382/coderankembed-apps-ft"
DEFAULT_QUERY_PREFIX = "Represent this query for searching relevant code: "


class Embedder(Protocol):
    model_id: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


_TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|[a-z]+")


def _tokens(text: str) -> list[str]:
    """Identifier parts: snake_case and camelCase split, lowercased."""
    return [t.lower() for t in _TOKEN.findall(text)]


class HashingEmbedder:
    """Bag of hashed identifier parts. Deterministic, no model download."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_id = f"hashing-{dim}"
        self.documents_embedded = 0  # for tests: how much work the cache saved

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            v[int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little") % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.documents_embedded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.empty((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class STEmbedder:
    """The fine-tuned CodeRankEmbed (sentence-transformers). Imports torch on construction."""

    def __init__(self, model: str = DEFAULT_MODEL, revision: str | None = None, device: str = "cpu",
                 batch_size: int = 32, max_seq_length: int = 512):
        from huggingface_hub import HfApi, snapshot_download

        from retrieval.model_loading import load_st_model

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        is_local = Path(model).is_dir()
        if is_local:
            model_dir = Path(model)
            revision = revision or "local"
        else:
            # Pin the weights: NomicBert's remote code ignores `revision` for Hub
            # weights, so load from a snapshot at an exact commit.
            revision = revision or HfApi().model_info(model).sha
            model_dir = Path(snapshot_download(model, revision=revision))
        ft_path = model_dir / "finetune_config.json"
        ft = json.loads(ft_path.read_text()) if ft_path.is_file() else {}
        self.query_prefix = ft.get("query_prefix", DEFAULT_QUERY_PREFIX)
        self.doc_prefix = ft.get("doc_prefix", "")
        self.query_clean = ft.get("query_clean", "desc-io")
        self.model_name, self.revision, self.device, self.batch_size = model, revision, device, batch_size
        self.model_id = f"{model}@{revision}"
        self.model, _ = load_st_model(str(model_dir), device, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.doc_prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 256).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        from retrieval.query_clean import clean_query

        q = self.query_prefix + clean_query(text.strip(), self.query_clean)
        return self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)[0]


def make_embedder(kind: str, model: str = DEFAULT_MODEL, revision: str | None = None,
                  device: str = "cpu", batch_size: int = 32) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "st":
        return STEmbedder(model, revision=revision, device=device, batch_size=batch_size)
    raise ValueError(f"unknown embedder {kind!r} (hashing | st)")
