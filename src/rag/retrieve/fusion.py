"""Reciprocal Rank Fusion — merge dense and sparse result lists into one.

WHY rank-based fusion instead of mixing raw scores:
cosine similarities live in [-1, 1], BM25 scores in [0, whatever-this-corpus-
produces]. Any scheme that normalizes and averages them needs corpus-specific
calibration and breaks the day the corpus changes. Ranks are already on a
common scale — "you were 3rd on your list" means the same thing in every
list. RRF throws away the raw scores and keeps only positions, which makes it
calibration-free and embarrassingly robust (it has survived 15+ years of
attempts to beat it with fancier fusion).

The formula (Cormack & Clarke, SIGIR 2009), extended with per-list weights:

    rrf_score(c) = Σ over lists L containing c of:  weight(L) / (k + rank_L(c))

k = 60 (settings.rrf_k) dampens the difference between adjacent ranks: with
k=60, rank 1 scores 1/61 and rank 2 scores 1/62 — nearly equal, so a single
list's opinion about #1-vs-#2 can't dominate. With k=0, rank 1 would score
double rank 2 and the top of one list would steamroll everything.
"""

from __future__ import annotations

from rag.retrieve.dense import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: dict[str, list[RetrievedChunk]],  # {"dense": [...], "sparse": [...]}
    weights: dict[str, float],  # {"dense": 0.7, "sparse": 0.3}
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge ranked lists into one, best RRF score first.

    A chunk found by BOTH retrievers gets score contributions from both —
    that agreement bonus is a feature: two independent search methods agreeing
    is stronger evidence of relevance than either alone.

    Ties (possible when lists don't overlap and weights are equal) break by
    chunk_id so the output order is deterministic — nondeterministic ordering
    would make evals unreproducible.
    """
    scores: dict[str, float] = {}
    # Keep the first RetrievedChunk object we see per id — text/metadata are
    # identical across lists (same underlying chunk), only ranking info differs.
    by_id: dict[str, RetrievedChunk] = {}

    for list_name, results in result_lists.items():
        weight = weights.get(list_name, 1.0)
        for chunk in results:
            # chunk.rank is 1-based (set by the retriever that produced it).
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + chunk.rank)
            by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [
        by_id[cid].with_rank(rank=rank, source="hybrid", score=scores[cid])
        for rank, cid in enumerate(ordered_ids, start=1)
    ]
