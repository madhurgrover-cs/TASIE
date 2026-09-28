"""
BM25 over code for mteb, as a SearchProtocol model (index() + search()).

mteb 2.21.8 (models/models_protocols.py) dispatches retrieval on the model type:
an EncoderProtocol is wrapped in SearchEncoderWrapper (encode + cosine top-k), a
SearchProtocol is called directly via
    index(corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc)
    search(queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs,
           top_ranked=None, num_proc) -> {query_id: {doc_id: score}}
This class implements the latter, so mteb.evaluate() scores it (and mteb's
HybridSearch fuses it with the dense encoder) and still writes the official
TaskResult.

Scoring is rank_bm25.BM25Okapi (idf, doc lengths, k1/b/epsilon), but evaluated as
one sparse matmul instead of BM25Okapi.get_scores(), which loops in Python over
every doc for every query token (~3.7k long queries x 9k docs would take hours).
bm25_scores() reproduces get_scores() exactly, including repeated query tokens.

Only numpy / scipy / rank_bm25 at import time (mteb is imported lazily), so the
tokenizer and scorer can be tested on the dev laptop.
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np
import scipy.sparse as sp
from rank_bm25 import BM25Okapi

logger = logging.getLogger("bm25_search")

_WORD = re.compile(r"[^\W\d_]\w*")  # starts with a letter; \w also covers Cyrillic etc.
# Pieces of one snake_case segment: "HTTPServer" -> HTTP, Server; "getX2" -> get, X, 2
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[^\W\d_A-Za-z]+")
# Function words from the natural-language side. Kept short on purpose: words
# like "find", "first", "next", "sum", "input" are meaningful in code.
STOPWORDS = frozenset("""
a an and are as at be been but by can do does each for from has have he her his how i if
in into is it its let may of on or our she so such than that the their them then there
these they this those to was we were what when where which while who will with would you your
""".split())


def tokenize_code(text: str) -> list[str]:
    """Lowercased identifier tokens with snake_case / camelCase split.

    A compound identifier emits itself plus its parts ("max_value" -> max_value,
    max, value; "numRows" -> numrows, num, rows), so exact identifier matches
    score higher than part matches. Numbers, 1-char parts and stopwords are dropped.
    """
    out: list[str] = []
    for word in _WORD.findall(text):
        parts = [p.lower() for seg in word.split("_") for p in _CAMEL.findall(seg)]
        parts = [p for p in parts if len(p) > 1 and not p.isdigit() and p not in STOPWORDS]
        whole = word.lower().strip("_")
        if len(parts) > 1 and whole not in STOPWORDS:
            out.append(whole)
        out.extend(parts)
    return out


class SparseBM25:
    """rank_bm25.BM25Okapi statistics, scored with a sparse doc-term weight matrix."""

    def __init__(self, doc_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.okapi = BM25Okapi(doc_tokens, k1=k1, b=b, epsilon=epsilon)
        ok = self.okapi
        self.vocab = {t: i for i, t in enumerate(ok.idf)}
        idf = np.fromiter(ok.idf.values(), dtype=np.float64, count=len(ok.idf))
        doc_len = np.asarray(ok.doc_len, dtype=np.float64)
        norm = k1 * (1 - b + b * doc_len / ok.avgdl)

        rows, cols, tfs = [], [], []
        for d, freqs in enumerate(ok.doc_freqs):
            rows.extend([d] * len(freqs))
            cols.extend(self.vocab[t] for t in freqs)
            tfs.extend(freqs.values())
        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)
        tf = np.asarray(tfs, dtype=np.float64)
        w = idf[cols_a] * tf * (k1 + 1) / (tf + norm[rows_a])
        # (V x N): query-count row vectors @ this = per-doc BM25 scores
        self.term_doc = sp.csr_matrix((w, (cols_a, rows_a)), shape=(len(self.vocab), ok.corpus_size))

    def query_matrix(self, query_tokens: list[list[str]]) -> sp.csr_matrix:
        # Counts, not presence: get_scores() adds a term once per occurrence.
        rows, cols = [], []
        for i, toks in enumerate(query_tokens):
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(query_tokens), len(self.vocab)))

    def bm25_scores(self, query_tokens: list[list[str]]) -> np.ndarray:
        """(n_queries x n_docs) dense scores, equal to okapi.get_scores() per query."""
        return (self.query_matrix(query_tokens) @ self.term_doc).toarray()


class BM25CodeSearch:
    """mteb SearchProtocol: rank_bm25 over tokenize_code() tokens."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, query_batch: int = 256):
        from mteb.models.model_meta import ModelMeta

        self.k1, self.b, self.query_batch = k1, b, query_batch
        self.index_model: SparseBM25 | None = None
        self.doc_ids: list[str] = []
        self.timings: dict[str, float] = {"bm25_index_s": 0.0, "bm25_search_s": 0.0}
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": "local/bm25-code",
            "model_type": ["sparse"],
            "framework": [],
            "open_weights": True,
            "use_instructions": False,
            "modalities": ["text"],
        })

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None) -> None:
        t0 = time.perf_counter()
        titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus)
        # Same title + text join as mteb's dense dataloader (_create_dataloaders.py).
        texts = [(f"{t} {x}" if t else x).strip() for t, x in zip(titles, corpus["text"])]
        self.doc_ids = list(corpus["id"])
        self.index_model = SparseBM25([tokenize_code(t) for t in texts], k1=self.k1, b=self.b)
        dt = time.perf_counter() - t0
        self.timings["bm25_index_s"] += dt
        logger.info("BM25 indexed %d docs, vocab %d, in %.1fs",
                    len(texts), len(self.index_model.vocab), dt)

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs,
               top_ranked=None, num_proc=None) -> dict[str, dict[str, float]]:
        if self.index_model is None:
            raise ValueError("Corpus must be indexed before searching.")
        t0 = time.perf_counter()
        qids = list(queries["id"])
        q_tokens = [tokenize_code(t) for t in queries["text"]]
        doc_idx = {d: i for i, d in enumerate(self.doc_ids)}
        n_docs = len(self.doc_ids)
        k = min(top_k, n_docs)

        results: dict[str, dict[str, float]] = {}
        for start in range(0, len(qids), self.query_batch):
            scores = self.index_model.bm25_scores(q_tokens[start:start + self.query_batch])
            for row, qid in zip(scores, qids[start:start + self.query_batch]):
                if top_ranked is not None:  # reranking tasks: only the given candidates
                    allowed = [doc_idx[d] for d in top_ranked.get(qid, []) if d in doc_idx]
                    masked = np.full(n_docs, -np.inf)
                    masked[allowed] = row[allowed]
                    row = masked
                top = np.argpartition(-row, k - 1)[:k] if k < n_docs else np.arange(n_docs)
                # Zero-score docs share no term with the query; leave them out so
                # rank fusion doesn't reward an arbitrary tie order.
                results[qid] = {self.doc_ids[i]: float(row[i]) for i in top if row[i] > 0}
        dt = time.perf_counter() - t0
        self.timings["bm25_search_s"] += dt
        logger.info("BM25 searched %d queries in %.1fs", len(qids), dt)
        return results

    def reset_timings(self) -> None:
        self.timings = dict.fromkeys(self.timings, 0.0)

