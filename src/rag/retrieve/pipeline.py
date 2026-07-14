"""Retrieval pipeline — the facade the rest of the system calls.

One object, one method:  RetrievalPipeline.retrieve(query, mode, strategy)
runs the full two-stage retrieval:

    dense (top 10) ─┐
                    ├─ weighted RRF fuse ─ take top 20 ─ cross-encoder ─ top 5
    sparse (top 10) ┘        ("hybrid" mode)              (if enabled)

    mode="dense" skips the sparse leg and fusion — that's the dashboard's
    A/B toggle, and the comparison the eval suite quantifies.

Why a facade at all: the embedder (~130MB), the reranker (~80MB) and the
indexes are all expensive to construct. Everything heavyweight is loaded
LAZILY on first use and CACHED per strategy, so the API can create one
pipeline at startup and serve every request from it (see api/main.py's
lifespan handler).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.embeddings import get_embedder
from rag.index.sparse import SparseIndex
from rag.index.vector import VectorIndex
from rag.retrieve.dense import DenseRetriever
from rag.retrieve.fusion import reciprocal_rank_fusion
from rag.retrieve.rerank import CrossEncoderReranker
from rag.retrieve.sparse import SparseRetriever

if TYPE_CHECKING:
    from rag.config import Settings
    from rag.embeddings import Embedder
    from rag.retrieve.dense import RetrievedChunk

MODES = ("hybrid", "dense")


class RetrievalPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Lazy singletons — None until first requested, then reused forever.
        self._embedder: Embedder | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._dense: dict[str, DenseRetriever] = {}  # keyed by strategy
        self._sparse: dict[str, SparseRetriever] = {}

    # ---- lazy component accessors -------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self._settings)
        return self._embedder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self._settings)
        return self._reranker

    def dense_retriever(self, strategy: str) -> DenseRetriever:
        if strategy not in self._dense:
            self._dense[strategy] = DenseRetriever(
                self.embedder, VectorIndex(self._settings, strategy)
            )
        return self._dense[strategy]

    def sparse_retriever(self, strategy: str) -> SparseRetriever:
        if strategy not in self._sparse:
            index = SparseIndex(self._settings, strategy)
            index.load()  # fail fast here (clear error) rather than mid-query
            self._sparse[strategy] = SparseRetriever(index)
        return self._sparse[strategy]

    # ---- the one public entry point ------------------------------------

    def retrieve(
        self, query: str, *, mode: str = "hybrid", strategy: str = "recursive"
    ) -> list[RetrievedChunk]:
        """Full two-stage retrieval; returns the final top-k, best first."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        s = self._settings

        if mode == "dense":
            # Dense-only: no fusion needed, but retrieve wide enough that the
            # reranker still has candidates to choose from.
            candidates = self.dense_retriever(strategy).retrieve(query, s.rerank_top_n)
        else:
            dense = self.dense_retriever(strategy).retrieve(query, s.dense_top_k)
            sparse = self.sparse_retriever(strategy).retrieve(query, s.sparse_top_k)
            fused = reciprocal_rank_fusion(
                {"dense": dense, "sparse": sparse},
                weights={"dense": s.dense_weight, "sparse": 1.0 - s.dense_weight},
                k=s.rrf_k,
            )
            candidates = fused[: s.rerank_top_n]

        if s.rerank_enabled:
            return self.reranker.rerank(query, candidates, top_n=s.final_top_k)
        # Reranking off (e.g. a small deployed container): fused order stands.
        # This is a documented, config-driven degradation — not a bug.
        return candidates[: s.final_top_k]
