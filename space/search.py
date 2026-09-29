"""
Search backend for the code-search Space: numpy only, no torch / gradio imports.

    QueryEncoder  text -> L2-normalised vector (query_encoder.STQueryEncoder wraps the model)
    SearchSource  anything that returns ranked Hits for an EncodedQuery
    SearchEngine  encodes once, fans out to the selected sources, merges by score

DenseIndexSource is the precomputed AppsRetrieval corpus (retrieval/precompute_corpus.py).
A versioned git-repo index (next phase) plugs in as another SearchSource: its Hits
carry the version in `meta` (e.g. {"repo": ..., "commit": ..., "path": ..., "lines": ...}),
and a source per version, or one source that searches every version, both fit the same
interface. Sources embedded with the same encoder return comparable cosine scores, so
the engine merges them by score. A source with a different scoring scale (e.g. BM25)
should normalise its scores before returning them.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

INDEX_FORMAT_VERSION = 1
EMBEDDINGS_FILE = "embeddings.npy"
DOCS_FILE = "docs.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class Hit:
    source: str
    doc_id: str
    score: float
    code: str
    language: str = "python"
    url: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedQuery:
    text: str  # what the user typed
    vector: np.ndarray  # 1-D float32, L2-normalised


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    encode_ms: float
    search_ms: float
    total_ms: float
    n_searched: int  # docs across the searched sources


class QueryEncoder(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...


class SearchSource(ABC):
    """A searchable collection of code snippets."""

    name: str

    @abstractmethod
    def search(self, query: EncodedQuery, k: int) -> list[Hit]:
        """Top-k hits, best first."""

    @abstractmethod
    def __len__(self) -> int: ...


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first; ties broken by lower index."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k < len(scores):
        # argpartition picks arbitrarily among ties at the cut-off, so keep every
        # index scoring >= the k-th best score and let the sort decide.
        kth = scores[np.argpartition(-scores, k - 1)[k - 1]]
        part = np.flatnonzero(scores >= kth)
    else:
        part = np.arange(len(scores))
    # lexsort: last key is primary -> sort by -score, then by index
    return part[np.lexsort((part, -scores[part]))][:k]


class DenseIndexSource(SearchSource):
    """In-memory exact cosine search over precomputed, L2-normalised embeddings."""

    def __init__(
        self,
        name: str,
        embeddings: np.ndarray,
        doc_ids: Sequence[str],
        codes: Sequence[str],
        urls: Sequence[str | None] | None = None,
        metas: Sequence[dict[str, Any]] | None = None,
        language: str = "python",
    ):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        n = embeddings.shape[0]
        for label, seq in (("doc_ids", doc_ids), ("codes", codes), ("urls", urls), ("metas", metas)):
            if seq is not None and len(seq) != n:
                raise ValueError(f"{label} has {len(seq)} entries for {n} embeddings")
        if len(set(doc_ids)) != n:
            raise ValueError("doc_ids must be unique")
        self.name = name
        self.embeddings = _normalise_rows(embeddings)
        self.doc_ids = list(doc_ids)
        self.codes = list(codes)
        self.urls = list(urls) if urls is not None else [None] * n
        self.metas = list(metas) if metas is not None else [{} for _ in range(n)]
        self.language = language
        self.manifest: dict[str, Any] = {}

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def search(self, query: EncodedQuery, k: int) -> list[Hit]:
        q = np.asarray(query.vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.dim} ({self.name})")
        scores = self.embeddings @ q
        return [
            Hit(source=self.name, doc_id=self.doc_ids[i], score=float(scores[i]), code=self.codes[i],
                language=self.language, url=self.urls[i], meta=self.metas[i])
            for i in top_k_indices(scores, k)
        ]

    @classmethod
    def from_dir(cls, path: str | Path, name: str | None = None, verify_checksum: bool = True) -> "DenseIndexSource":
        """Load an index written by retrieval/precompute_corpus.py."""
        path = Path(path)
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError(f"unsupported index format {manifest.get('format_version')!r}")
        emb_path = path / EMBEDDINGS_FILE
        if verify_checksum and manifest.get("embeddings_sha256"):
            digest = hashlib.sha256(emb_path.read_bytes()).hexdigest()
            if digest != manifest["embeddings_sha256"]:
                raise ValueError(f"{EMBEDDINGS_FILE} checksum mismatch (index upload incomplete?)")
        embeddings = np.load(emb_path)
        docs = [json.loads(line) for line in (path / DOCS_FILE).read_text(encoding="utf-8").splitlines() if line]
        if len(docs) != embeddings.shape[0] or manifest.get("n_docs") not in (None, len(docs)):
            raise ValueError(f"index mismatch: {len(docs)} docs, {embeddings.shape[0]} embeddings, "
                             f"manifest n_docs={manifest.get('n_docs')}")
        if manifest.get("dim") not in (None, embeddings.shape[1]):
            raise ValueError(f"manifest dim {manifest['dim']} != embeddings dim {embeddings.shape[1]}")
        src = cls(
            name=name or manifest.get("name", path.name),
            embeddings=embeddings,
            doc_ids=[d["id"] for d in docs],
            codes=[d["code"] for d in docs],
            urls=[d.get("url") for d in docs],
            metas=[{k: v for k, v in d.items() if k not in ("id", "code", "url")} for d in docs],
            language=manifest.get("language", "python"),
        )
        src.manifest = manifest
        return src


class SearchEngine:
    def __init__(self, encoder: QueryEncoder, sources: Sequence[SearchSource]):
        if not sources:
            raise ValueError("need at least one source")
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names: {names}")
        self.encoder = encoder
        self.sources = {s.name: s for s in sources}

    @property
    def source_names(self) -> list[str]:
        return list(self.sources)

    def search(self, text: str, k: int = 5, sources: Sequence[str] | None = None) -> SearchResult:
        if not text or not text.strip():
            raise ValueError("empty query")
        if k < 1:
            raise ValueError("k must be >= 1")
        selected = self.source_names if not sources else list(sources)
        unknown = [s for s in selected if s not in self.sources]
        if unknown:
            raise ValueError(f"unknown sources {unknown}; available {self.source_names}")

        t0 = time.perf_counter()
        vec = np.asarray(self.encoder.encode_query(text), dtype=np.float32).reshape(-1)
        query = EncodedQuery(text=text, vector=_normalise_rows(vec))
        t1 = time.perf_counter()
        hits = [h for name in selected for h in self.sources[name].search(query, k)]
        # stable sort keeps each source's order for equal scores
        hits = sorted(hits, key=lambda h: -h.score)[:k]
        t2 = time.perf_counter()
        return SearchResult(
            hits=hits,
            encode_ms=(t1 - t0) * 1000,
            search_ms=(t2 - t1) * 1000,
            total_ms=(t2 - t0) * 1000,
            n_searched=sum(len(self.sources[n]) for n in selected),
        )
