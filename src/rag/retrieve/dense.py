"""Dense (semantic) retrieval — find chunks whose MEANING matches the query.

This module also owns `RetrievedChunk`, the shared result type for the whole
retrieve/ package. Every retriever — dense, sparse, fused, reranked — emits
the same shape, which is what lets fusion and reranking compose freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.embeddings import Embedder
    from rag.index.vector import VectorIndex


@dataclass
class RetrievedChunk:
    """One search result.

    The `score` field's MEANING depends on which retriever produced it:
    cosine similarity (dense), BM25 (sparse), RRF sum (hybrid), or a
    cross-encoder logit (reranked). They are NOT comparable across sources —
    that incomparability is exactly why fusion works on `rank`, never `score`.
    """

    chunk_id: str
    text: str
    score: float
    rank: int  # 1-based position in THIS retriever's list
    source: str  # "dense" | "sparse" | "hybrid" | "reranked"
    metadata: dict = field(default_factory=dict)

    def with_rank(self, rank: int, source: str, score: float) -> RetrievedChunk:
        """Copy with new ranking info — results are treated as immutable so a
        chunk can sit in the dense list AND the fused list without aliasing."""
        return replace(self, rank=rank, source=source, score=score)


class DenseRetriever:
    """Embed the query, ask the vector index for nearest neighbors."""

    def __init__(self, embedder: Embedder, vector_index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = vector_index

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        # embed_query (NOT embed_texts): the BGE instruction prefix is applied
        # inside the embedder — queries and documents are embedded differently
        # on purpose (asymmetric training; see embeddings.py).
        query_vec = self._embedder.embed_query(query)
        hits = self._index.query(query_vec, k)
        return [
            RetrievedChunk(
                chunk_id=h["chunk_id"],
                text=h["text"],
                score=h["score"],  # cosine similarity, bigger = better
                rank=rank,
                source="dense",
                metadata=h["metadata"],
            )
            for rank, h in enumerate(hits, start=1)
        ]
