"""Dedup tests — keep-first semantics on hand-built vectors.

The vectors are unit-length by construction, so cosine similarity is a plain
dot product — exactly the invariant embeddings.py guarantees for real vectors.
"""

import numpy as np

from rag.ingest.chunkers import Chunk
from rag.ingest.dedup import dedup, find_duplicates


def make_chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc.md::fixed::{i}",
        doc_id="doc.md",
        text=f"chunk {i}",
        index=i,
        strategy="fixed",
        section=None,
        char_count=8,
    )


def unit(v) -> np.ndarray:
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_exact_duplicate_is_dropped_keep_first():
    vecs = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 0, 0])])  # 2 == copy of 0
    chunks = [make_chunk(i) for i in range(3)]
    survivors, survivor_vecs, report = dedup(chunks, vecs, threshold=0.95)
    assert [c.index for c in survivors] == [0, 1]  # first copy kept
    assert survivor_vecs.shape == (2, 3)
    assert report == [("doc.md::fixed::2", "doc.md::fixed::0")]


def test_near_duplicate_above_threshold_is_dropped():
    a = unit([1.0, 0.02, 0.0])  # ~0.999 similar to [1,0,0]
    vecs = np.stack([unit([1, 0, 0]), a])
    dups = find_duplicates(vecs, threshold=0.95)
    assert dups == {1: 0}


def test_distinct_vectors_survive():
    vecs = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])])
    chunks = [make_chunk(i) for i in range(3)]
    survivors, _, report = dedup(chunks, vecs, threshold=0.95)
    assert len(survivors) == 3
    assert report == []


def test_chain_of_duplicates_points_at_real_survivor():
    """0, 1, 2 all identical: both 1 and 2 must map to 0 (the survivor),
    never to each other — the report should always name a kept chunk."""
    v = unit([1, 1, 0])
    vecs = np.stack([v, v, v])
    dups = find_duplicates(vecs, threshold=0.95)
    assert dups == {1: 0, 2: 0}


def test_empty_input():
    survivors, vecs, report = dedup([], np.zeros((0, 3), dtype=np.float32), threshold=0.95)
    assert survivors == [] and report == []
