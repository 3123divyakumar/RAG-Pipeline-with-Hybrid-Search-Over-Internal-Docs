"""Near-duplicate detection — keep redundant chunks out of the index.

Why this exists
---------------
Documentation repeats itself: boilerplate headers, the same install snippet in
five tutorials, copied paragraphs. If five near-identical chunks live in the
index, a query matching that content fills its whole top-5 with copies of one
fact and crowds out everything else the answer needed. Removing duplicates at
ingest time (once) is far cheaper than trying to diversify results at query
time (every request).

How "near-duplicate" is measured
--------------------------------
Cosine similarity between chunk embeddings. The embeddings are already
unit-normalized (see embeddings.py), so the full pairwise similarity matrix is
just `embeddings @ embeddings.T` — one matrix multiply.

Scale note (a DECISIONS.md entry): the matrix is O(n²) memory. At this
corpus's size (~5-8k chunks) that's a few hundred MB of float32 at worst,
computed in seconds. At 1M chunks you'd swap this for an ANN index lookup
(query each vector against the index built so far) or MinHash on shingles —
same keep-first semantics, sub-quadratic cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.ingest.chunkers import Chunk


def find_duplicates(embeddings: np.ndarray, threshold: float) -> dict[int, int]:
    """Return {duplicate_index: kept_index} for every chunk that near-duplicates
    an EARLIER chunk (keep-first policy).

    Keep-first is deliberate: chunks arrive in document order, so the copy we
    keep is the one from the earliest file walk position — stable across runs,
    which keeps chunk_ids (and therefore citations and eval caches) stable too.

    Works in row blocks so peak memory stays bounded even if the corpus grows:
    block_size × n floats at a time instead of n × n.
    """
    n = embeddings.shape[0]
    duplicate_of: dict[int, int] = {}
    if n == 0:
        return duplicate_of

    block = 1024
    for start in range(0, n, block):
        end = min(start + block, n)
        # Similarities of rows [start:end] against EVERYTHING BEFORE each row.
        sims = embeddings[start:end] @ embeddings.T  # (block, n)
        for i in range(start, end):
            if i in duplicate_of:  # already marked a dup — its "kept" chunk covers it
                continue
            row = sims[i - start, :i]  # only earlier chunks matter (keep-first)
            if row.size == 0:
                continue
            j = int(np.argmax(row))
            if row[j] >= threshold and j not in duplicate_of:
                # j itself might be a dup of something even earlier — chase the
                # chain so the report always points at a real survivor.
                duplicate_of[i] = j
            elif row[j] >= threshold:
                duplicate_of[i] = duplicate_of[j]
    return duplicate_of


def dedup(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[Chunk], np.ndarray, list[tuple[str, str]]]:
    """Drop near-duplicate chunks, keeping the first occurrence.

    Returns:
      survivors      — chunks that stay, original order preserved
      survivor_vecs  — their embeddings, rows aligned with `survivors`
                       (alignment matters: the ingest pipeline feeds both
                       straight into the vector index without re-embedding)
      report         — (dropped_chunk_id, kept_chunk_id) pairs. PRINT THIS
                       during ingest: eyeballing what got dropped is how the
                       threshold gets chosen honestly. At 0.95, dropped pairs
                       should look like obvious copies; if they look like
                       merely-related paragraphs, the threshold is too low.
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) are misaligned"
        )

    duplicate_of = find_duplicates(embeddings, threshold)
    keep_mask = [i not in duplicate_of for i in range(len(chunks))]

    survivors = [c for c, keep in zip(chunks, keep_mask) if keep]
    survivor_vecs = embeddings[np.array(keep_mask, dtype=bool)] if chunks else embeddings
    report = [
        (chunks[dup].chunk_id, chunks[kept].chunk_id)
        for dup, kept in sorted(duplicate_of.items())
    ]
    return survivors, survivor_vecs, report
