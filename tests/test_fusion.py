"""RRF fusion tests — hand-crafted lists where the expected order is
computable on paper. Every case here exercises one property that makes
RRF trustworthy: the agreement bonus, weight sensitivity, determinism on
ties, and rank (not score) driven ordering."""

from rag.retrieve.dense import RetrievedChunk
from rag.retrieve.fusion import reciprocal_rank_fusion


def rc(chunk_id: str, rank: int, source: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=f"text of {chunk_id}", score=score, rank=rank, source=source
    )


def test_chunk_in_both_lists_beats_single_list_chunks():
    """'both' is rank 2 in each list; 'd1'/'s1' are rank 1 in one list only.
    With k=1 and equal weights:
        both: 1/(1+2) + 1/(1+2) = 0.666...
        d1:   1/(1+1)           = 0.5
    The agreement bonus must put 'both' first."""
    dense = [rc("d1", 1, "dense"), rc("both", 2, "dense")]
    sparse = [rc("s1", 1, "sparse"), rc("both", 2, "sparse")]
    fused = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 1.0, "sparse": 1.0}, k=1
    )
    assert fused[0].chunk_id == "both"


def test_weights_can_flip_an_ordering():
    """Same lists, only the weights change — the winner must change with them."""
    dense = [rc("d1", 1, "dense")]
    sparse = [rc("s1", 1, "sparse")]
    dense_heavy = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 0.9, "sparse": 0.1}, k=60
    )
    sparse_heavy = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 0.1, "sparse": 0.9}, k=60
    )
    assert dense_heavy[0].chunk_id == "d1"
    assert sparse_heavy[0].chunk_id == "s1"


def test_exact_scores_match_the_formula():
    dense = [rc("a", 1, "dense"), rc("b", 2, "dense")]
    sparse = [rc("b", 1, "sparse")]
    fused = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 0.7, "sparse": 0.3}, k=60
    )
    by_id = {c.chunk_id: c for c in fused}
    assert abs(by_id["a"].score - 0.7 / 61) < 1e-12
    assert abs(by_id["b"].score - (0.7 / 62 + 0.3 / 61)) < 1e-12


def test_ties_break_deterministically_by_chunk_id():
    """Two chunks with identical RRF scores must always come out in the same
    order (chunk_id ascending) — evals depend on reproducible rankings."""
    dense = [rc("zzz", 1, "dense")]
    sparse = [rc("aaa", 1, "sparse")]
    fused = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 0.5, "sparse": 0.5}, k=60
    )
    assert [c.chunk_id for c in fused] == ["aaa", "zzz"]


def test_raw_scores_are_ignored_only_ranks_matter():
    """A huge BM25 score on a low-ranked chunk must NOT let it outrank the
    list's #1 — fusion reads positions, never score magnitudes."""
    sparse = [rc("first", 1, "sparse", score=1.0), rc("huge", 2, "sparse", score=9999.0)]
    fused = reciprocal_rank_fusion({"sparse": sparse}, weights={"sparse": 1.0}, k=60)
    assert fused[0].chunk_id == "first"


def test_output_ranks_are_fresh_and_sequential():
    dense = [rc("a", 1, "dense"), rc("b", 2, "dense")]
    sparse = [rc("c", 1, "sparse")]
    fused = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 0.7, "sparse": 0.3}, k=60
    )
    assert [c.rank for c in fused] == [1, 2, 3]
    assert all(c.source == "hybrid" for c in fused)


def test_empty_lists_produce_empty_output():
    assert reciprocal_rank_fusion({"dense": [], "sparse": []}, weights={}, k=60) == []
