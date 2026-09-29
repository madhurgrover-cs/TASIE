"""Torch-free stand-ins for the embedding model (hashed bag of words)."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

SPACE_DIR = Path(__file__).resolve().parent.parent / "space"
if str(SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(SPACE_DIR))

DIM = 64


def hashed_bow(text: str, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEncoder:
    """QueryEncoder: records the texts it saw."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.seen: list[str] = []

    def encode_query(self, text: str) -> np.ndarray:
        self.seen.append(text)
        return hashed_bow(text, self.dim)


class FakeSTModel:
    """Mimics SentenceTransformer.encode for STQueryEncoder."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.stack([hashed_bow(t, self.dim) * 3.0 for t in texts])  # un-normalised on purpose
