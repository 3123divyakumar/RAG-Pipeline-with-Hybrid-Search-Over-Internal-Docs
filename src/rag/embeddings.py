"""Embeddings — turn text into vectors so similar meanings land near each other.

Two details here are classic silent RAG bugs when missed:

1. `normalize_embeddings=True` on EVERY encode() call. Normalized (unit-length)
   vectors make cosine similarity equal to a plain dot product, so Chroma's
   distance math and our dedup math stay consistent everywhere.

2. BGE models are trained *asymmetrically*: QUERIES get an instruction prefix,
   DOCUMENTS do not. `embed_query()` prepends `settings.bge_query_prefix`;
   `embed_texts()` never does. Getting this backwards raises no error —
   retrieval quality just quietly degrades.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from rag.config import Settings


class Embedder(Protocol):
    """What the rest of the pipeline is allowed to know about an embedder."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed documents/chunks (no prefix). Returns shape (n, dim)."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query (with prefix). Returns shape (dim,)."""
        ...


class SentenceTransformersEmbedder:
    """Local sentence-transformers model (default: BAAI/bge-small-en-v1.5, 384 dims)."""

    def __init__(self, model_name: str, query_prefix: str, batch_size: int = 32) -> None:
        # Imported here, not at module top: sentence_transformers pulls in torch
        # (~seconds of import time). Only pay that when an embedder is actually built.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:  # encode([]) returns a shapeless array; keep (0, dim) so callers can vstack
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self._model.encode(
            self._query_prefix + query,
            normalize_embeddings=True,
        )


def get_embedder(settings: Settings) -> Embedder:
    """Factory — callers never name the concrete class, so swapping in an
    OpenAIEmbedder/FastEmbedEmbedder later touches only this function."""
    return SentenceTransformersEmbedder(
        model_name=settings.embedding_model,
        query_prefix=settings.bge_query_prefix,
    )
