"""
Query side of the Space: the same pre-processing as retrieval/submission.py
(desc-io cleanup, then the model's query prefix), L2-normalised output.

No torch import: the model is passed in (anything with a sentence-transformers style
`encode`), so tests can use a fake.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from query_clean import clean_query

DEFAULT_QUERY_PREFIX = "Represent this query for searching relevant code: "
DEFAULT_QUERY_CLEAN = "desc-io"


class STQueryEncoder:
    def __init__(self, model: Any, query_prefix: str = DEFAULT_QUERY_PREFIX,
                 query_clean: str = DEFAULT_QUERY_CLEAN):
        self.model = model
        self.query_prefix = query_prefix
        self.query_clean = query_clean

    def preprocess(self, text: str) -> str:
        return self.query_prefix + clean_query(text.strip(), self.query_clean)

    def encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode([self.preprocess(text)], convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)[0]
