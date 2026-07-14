"""Cross-encoder reranking — a slow, accurate second opinion on the top candidates.

Bi-encoder vs cross-encoder — why this second stage exists:
- The embedder is a BI-encoder: query and document are embedded SEPARATELY
  and compared afterwards. Fast — documents are pre-embedded at ingest — but
  the model never sees query and document together, so it can't notice
  interactions like negation or "same words, different question".
- A CROSS-encoder feeds "query [SEP] document" through the model as ONE
  input, with attention flowing between them. Far more accurate relevance
  judgment — and far too slow to run over the whole corpus (it would need a
  forward pass per chunk per query).

So the pipeline retrieves wide and cheap (top ~20 via hybrid) and reranks
narrow and expensive (cross-encoder keeps the best 5). Classic two-stage
retrieval — the same shape every production search system uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import Settings
    from rag.retrieve.dense import RetrievedChunk


class CrossEncoderReranker:
    """Wraps sentence-transformers' CrossEncoder (~80MB, downloads on first use)."""

    def __init__(self, settings: Settings) -> None:
        # Deferred import: CrossEncoder pulls in torch — only pay the import
        # cost when a reranker is actually constructed (mirrors embeddings.py).
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.rerank_model)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Score every (query, candidate) pair jointly; keep the best top_n.

        The scores are raw relevance logits (roughly: >0 relevant, <0 not).
        They feed two things downstream: this ordering, and the retrieval-
        confidence signal in generate/confidence.py.
        """
        if not candidates:
            return []
        # One batched predict call — the model batches internally, so this is
        # the whole "expensive" step: ~20 forward passes per query.
        scores = self._model.predict([(query, c.text) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [
            candidates[i].with_rank(rank=rank, source="reranked", score=float(scores[i]))
            for rank, i in enumerate(order[:top_n], start=1)
        ]
