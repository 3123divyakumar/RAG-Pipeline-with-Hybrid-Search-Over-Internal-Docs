"""BM25 sparse index — exact-keyword matching that dense retrieval can't do.

Why a keyword index at all
--------------------------
Embeddings are great at "meaning" and terrible at arbitrary identifiers.
`response_model_exclude_none`, `HTTP_422`, `model_config` — the embedding
neighborhoods of strings like these are mush, but their EXACT occurrence in a
chunk is a near-perfect relevance signal. Technical docs are full of them.
BM25 scores chunks by weighted exact-term overlap and is the reason "hybrid"
beats "dense-only" on this corpus.

Persistence: pickle to data/bm25_{strategy}.pkl. rank_bm25 keeps its whole
model in a few numpy arrays + dicts, so pickling the object is the idiomatic
(and fastest) way to persist it. Pickle is unsafe to load from UNTRUSTED
sources — these files are produced by our own ingest run, so that's fine.
"""

from __future__ import annotations

import pickle
import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from pathlib import Path

    from rag.config import Settings
    from rag.ingest.chunkers import Chunk

# Tokens = runs of letters, digits and underscores. Keeping "_" inside tokens
# is THE decision that makes identifier queries work: "response_model" must
# stay one token, because that's exactly what a user pastes into the search box.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """The ONLY tokenizer — used at index time AND query time.

    This is the most important function in the file. BM25 matches tokens by
    exact equality: if indexing lowercases but querying doesn't (or one splits
    on "_" and the other doesn't), scores silently collapse to near-zero with
    no error anywhere. One shared function makes that bug impossible.
    """
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    """BM25 index over one chunking strategy's chunks (mirrors VectorIndex naming)."""

    def __init__(self, settings: Settings, strategy: str) -> None:
        self.strategy = strategy
        self._path: Path = settings.data_dir / f"bm25_{strategy}.pkl"
        self._bm25: BM25Okapi | None = None
        # Chunk payloads kept alongside the BM25 arrays, positionally aligned:
        # row i of the BM25 corpus is _chunks[i]. That positional alignment is
        # the whole storage model — never reorder one without the other.
        self._chunks: list[dict] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenize every chunk and fit BM25 over the token corpus.

        Sync guarantee (an invariant, not luck): this is only ever called by
        scripts/ingest.py, with the SAME post-dedup chunk list that went into
        the VectorIndex, in the same run. That's what keeps dense and sparse
        results referring to identical chunk_ids.
        """
        self._chunks = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": {
                    "doc_id": c.doc_id,
                    "index": c.index,
                    "strategy": c.strategy,
                    "section": c.section or "",
                    "char_count": c.char_count,
                    "title": str(c.metadata.get("title", "")),
                },
            }
            for c in chunks
        ]
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)

    def load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"{self._path} not found — run `uv run python scripts/ingest.py` first"
            )
        with open(self._path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._chunks = payload["chunks"]

    def count(self) -> int:
        return len(self._chunks)

    def query(self, query: str, k: int) -> list[dict]:
        """Score every chunk against the query's tokens, return top-k.

        BM25 scores are unbounded and corpus-dependent — do NOT compare them
        to cosine similarities. Rank fusion (retrieve/fusion.py) exists
        precisely so nobody ever has to.
        """
        if self._bm25 is None:
            self.load()  # lazy-load so constructing an index object is free
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort ascending -> take the last k, reversed = top-k descending.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "chunk_id": self._chunks[i]["chunk_id"],
                "text": self._chunks[i]["text"],
                "score": float(scores[i]),
                "metadata": self._chunks[i]["metadata"],
            }
            for i in top
            if scores[i] > 0  # zero score = shares no tokens with the query; not a result
        ]
