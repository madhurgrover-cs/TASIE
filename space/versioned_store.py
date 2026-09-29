"""
On-disk store for versioned code indexes. numpy + json only (no torch, no git), and no
imports from this repo: space/versioned_store.py is a byte-for-byte copy
(tests/test_space_sync.py checks it).

Layout (everything relative to the store root, uploaded as-is to the Hub):
    registry.json                        repo, model, one entry per indexed commit
    content.jsonl                        {"hash", "code"}: every distinct chunk, once
    embeddings/<model slug>/hashes.json  row order of vectors.npy
    embeddings/<model slug>/vectors.npy  float32 [n, dim], L2-normalised, one row per hash
    versions/<commit>/chunks.jsonl       chunk metadata (path, name, kind, lines, hash)
    versions/<commit>/meta.json          same as the registry entry

A commit's index is its chunk list; code and vectors are looked up by content hash, so
code shared between versions is stored and embedded once.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STORE_FORMAT_VERSION = 1


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8"))


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    _atomic_write_bytes(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first; ties broken by lower index."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k < len(scores):
        kth = scores[np.argpartition(-scores, k - 1)[k - 1]]
        part = np.flatnonzero(scores >= kth)
    else:
        part = np.arange(len(scores))
    return part[np.lexsort((part, -scores[part]))][:k]


class EmbeddingCache:
    """Vectors keyed by content hash, one cache per model id (repo@revision)."""

    def __init__(self, root: Path, model_id: str):
        self.model_id = model_id
        self.dir = Path(root) / "embeddings" / slug(model_id)
        hashes_path = self.dir / "hashes.json"
        self._hashes: list[str] = json.loads(hashes_path.read_text()) if hashes_path.is_file() else []
        self._vectors = (np.load(self.dir / "vectors.npy") if self._hashes
                         else np.empty((0, 0), dtype=np.float32))
        if len(self._hashes) != self._vectors.shape[0]:
            raise ValueError(f"{self.dir}: {len(self._hashes)} hashes vs {self._vectors.shape[0]} vectors")
        self._row = {h: i for i, h in enumerate(self._hashes)}
        self._pending: list[np.ndarray] = []
        self._dirty = False

    def __len__(self) -> int:
        return len(self._row)

    def __contains__(self, h: str) -> bool:
        return h in self._row

    @property
    def dim(self) -> int | None:
        if self._vectors.size:
            return self._vectors.shape[1]
        return self._pending[0].shape[1] if self._pending else None

    def _consolidate(self) -> None:
        if self._pending:
            parts = ([self._vectors] if self._vectors.size else []) + self._pending
            self._vectors = np.concatenate(parts).astype(np.float32)
            self._pending = []

    def add(self, hashes: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(hashes):
            raise ValueError(f"{len(hashes)} hashes vs vectors of shape {vectors.shape}")
        if self.dim is not None and vectors.shape[1] != self.dim:
            raise ValueError(f"dim {vectors.shape[1]} != cache dim {self.dim}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        new = []
        for i, h in enumerate(hashes):
            if h not in self._row:  # also skips repeats within this call
                self._row[h] = len(self._hashes)
                self._hashes.append(h)
                new.append(i)
        if new:
            self._pending.append(vectors[new])
            self._dirty = True

    def embed_missing(self, texts_by_hash: dict[str, str],
                      embed: Callable[[list[str]], np.ndarray], batch_size: int = 256) -> tuple[int, int]:
        """Embed only hashes not in the cache. Returns (reused, newly_embedded)."""
        missing = [h for h in texts_by_hash if h not in self._row]
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            self.add(batch, embed([texts_by_hash[h] for h in batch]))
        return len(texts_by_hash) - len(missing), len(missing)

    def matrix(self, hashes: Sequence[str]) -> np.ndarray:
        self._consolidate()
        rows = [self._row[h] for h in hashes]  # KeyError = not embedded
        if not rows:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        return self._vectors[rows]

    def save(self) -> None:
        self._consolidate()
        if not self._dirty:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        # vectors first: a crash between the two writes leaves extra rows, which load() rejects
        # loudly instead of silently mis-aligning hashes and vectors.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".npy")
        os.close(fd)
        np.save(tmp, self._vectors)
        os.replace(tmp, self.dir / "vectors.npy")
        _write_json(self.dir / "hashes.json", self._hashes)
        self._dirty = False


class ContentStore:
    """Normalised chunk code keyed by content hash."""

    def __init__(self, root: Path):
        self.path = Path(root) / "content.jsonl"
        self._code = {r["hash"]: r["code"] for r in _read_jsonl(self.path)}
        self._dirty = False

    def __len__(self) -> int:
        return len(self._code)

    def __contains__(self, h: str) -> bool:
        return h in self._code

    def get(self, h: str) -> str:
        return self._code[h]

    def add(self, h: str, code: str) -> None:
        if h not in self._code:
            self._code[h] = code
            self._dirty = True

    def save(self) -> None:
        if self._dirty:
            _write_jsonl(self.path, [{"hash": h, "code": c} for h, c in self._code.items()])
            self._dirty = False


class Registry:
    def __init__(self, root: Path):
        self.path = Path(root) / "registry.json"
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        if data and data.get("format_version") != STORE_FORMAT_VERSION:
            raise ValueError(f"unsupported store format {data.get('format_version')!r}")
        self.repo: str | None = data.get("repo")
        self.model_id: str | None = data.get("model_id")
        self.versions: dict[str, dict] = data.get("versions", {})

    def save(self) -> None:
        _write_json(self.path, {"format_version": STORE_FORMAT_VERSION, "repo": self.repo,
                                "model_id": self.model_id, "versions": self.versions})

    def list(self) -> list[dict]:
        """Oldest commit first."""
        return sorted(self.versions.values(), key=lambda v: (v.get("committed_at") or "", v["commit"]))

    def resolve(self, ref: str) -> str:
        """Full sha from a full / short sha (>= 4 chars) or a recorded ref name (e.g. a tag)."""
        if ref in self.versions:
            return ref
        by_ref = [c for c, v in self.versions.items() if ref in v.get("refs", [])]
        by_prefix = [c for c in self.versions if len(ref) >= 4 and c.startswith(ref)]
        matches = list(dict.fromkeys(by_ref + by_prefix))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"commit {ref!r} is not indexed; indexed: "
                           f"{[v['short'] + ' ' + ','.join(v.get('refs', [])) for v in self.list()]}")
        raise KeyError(f"ambiguous commit {ref!r}: {matches}")

    def label(self, commit: str) -> str:
        v = self.versions[commit]
        refs = ", ".join(v.get("refs", []))
        return f"{refs} · {v['short']}" if refs else v["short"]


@dataclass
class VersionView:
    """One commit's index, loaded for search."""

    commit: str
    meta: dict
    chunks: list[dict]  # chunks.jsonl rows + "code"
    matrix: np.ndarray  # [n_chunks, dim], L2-normalised

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[dict, float]]:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not self.chunks:
            return []
        if v.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"query dim {v.shape[0]} != index dim {self.matrix.shape[1]}")
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        scores = self.matrix @ v
        return [(self.chunks[i], float(scores[i])) for i in top_k_indices(scores, k)]


class Store:
    def __init__(self, root: str | Path, model_id: str | None = None):
        self.root = Path(root)
        self.registry = Registry(self.root)
        self.content = ContentStore(self.root)
        if model_id and self.registry.model_id and model_id != self.registry.model_id:
            raise ValueError(f"store {self.root} was built with {self.registry.model_id!r}, not {model_id!r}; "
                             "use a separate store per model")
        self.model_id = model_id or self.registry.model_id
        self._cache: EmbeddingCache | None = None

    @property
    def embeddings(self) -> EmbeddingCache:
        if self.model_id is None:
            raise ValueError("store has no model id yet")
        if self._cache is None:
            self._cache = EmbeddingCache(self.root, self.model_id)
        return self._cache

    def version_dir(self, commit: str) -> Path:
        return self.root / "versions" / commit

    def read_chunks(self, commit: str) -> list[dict]:
        return _read_jsonl(self.version_dir(commit) / "chunks.jsonl")

    def write_version(self, commit: str, chunks: Sequence[dict], meta: dict) -> None:
        rows = [{k: v for k, v in c.items() if k != "code"} for c in chunks]
        _write_jsonl(self.version_dir(commit) / "chunks.jsonl", rows)
        _write_json(self.version_dir(commit) / "meta.json", meta)
        if self.registry.model_id is None:
            self.registry.model_id = self.model_id
        self.registry.versions[commit] = meta

    def save(self) -> None:
        """Content and vectors before the registry, so a registered version is always complete."""
        self.content.save()
        if self._cache is not None:
            self._cache.save()
        self.registry.save()

    def load_view(self, ref: str) -> VersionView:
        commit = self.registry.resolve(ref)
        rows = self.read_chunks(commit)
        chunks = [{**r, "code": self.content.get(r["hash"])} for r in rows]
        return VersionView(commit, self.registry.versions[commit], chunks,
                           self.embeddings.matrix([r["hash"] for r in rows]))
