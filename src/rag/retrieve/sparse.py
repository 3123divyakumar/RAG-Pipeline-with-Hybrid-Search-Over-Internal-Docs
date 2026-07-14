"""Sparse (keyword) retrieval — find chunks containing the query's EXACT terms.

This is what catches `response_model_exclude_none`, `HTTP_422`, config keys
and error codes — strings whose embedding neighborhoods are meaningless but
whose exact occurrence is everything. Technical documentation is full of
them; this retriever is the reason "hybrid" beats "dense-only" on this corpus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieve.dense import RetrievedChunk

if TYPE_CHECKING:
    from rag.index.sparse import SparseIndex


class SparseRetriever:
    """Thin adapter: SparseIndex dicts -> RetrievedChunk objects.

    Deliberately does nothing else. BM25 scores are unbounded and
    corpus-dependent, so there is NO attempt to normalize them into a 0-1
    range or make them comparable to cosine similarities — rank fusion
    (fusion.py) exists precisely so we never have to.
    """

    def __init__(self, sparse_index: SparseIndex) -> None:
        self._index = sparse_index

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = self._index.query(query, k)
        return [
            RetrievedChunk(
                chunk_id=h["chunk_id"],
                text=h["text"],
                score=h["score"],  # raw BM25 — meaningful only within this list
                rank=rank,
                source="sparse",
                metadata=h["metadata"],
            )
            for rank, h in enumerate(hits, start=1)
        ]
