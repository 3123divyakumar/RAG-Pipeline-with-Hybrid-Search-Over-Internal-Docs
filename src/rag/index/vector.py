"""Vector index — thin wrapper around ChromaDB.

Design decisions baked in here (each is a DECISIONS.md story):

1. ONE COLLECTION PER CHUNKING STRATEGY ("chunks_fixed", "chunks_recursive",
   "chunks_semantic"). Three parallel indexes over the same corpus, selected
   by name — that's what makes the strategy comparison a config
   change instead of a re-architecture.

2. WE EMBED OUTSIDE CHROMA. Chroma can auto-embed documents with its own
   embedding function, but then the query path (our Embedder, with the BGE
   query prefix) and the index path (Chroma's embedder) could silently drift
   apart — different model, different normalization, no error, just bad
   results. One Embedder owns ALL vectorization; Chroma only stores.

3. COSINE SPACE, SIMILARITY SCORES. The collection is created with
   hnsw:space=cosine. Chroma returns cosine *distance* (0 = identical,
   2 = opposite); we convert to similarity (1 - distance) at the boundary so
   every score in this codebase means "bigger = better". Mixed score
   conventions are how off-by-one-negation bugs happen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    import numpy as np

    from rag.config import Settings
    from rag.ingest.chunkers import Chunk


class VectorIndex:
    """Persistent Chroma collection for one chunking strategy."""

    def __init__(self, settings: Settings, strategy: str) -> None:
        self.strategy = strategy
        self.collection_name = f"chunks_{strategy}"
        # PersistentClient writes to .chroma/ on disk — survives restarts, and
        # it's the directory the OneDrive exclusion protects sync exclusion protects.
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # default is l2 — cosine must be explicit
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Store chunks with their precomputed embeddings.

        Batched because Chroma has a max-batch limit (~5k) and huge single
        calls spike memory. Chroma metadata values must be str/int/float/bool —
        anything richer gets flattened or stringified here, at the boundary.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings are misaligned")
        batch = 1000
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vecs = embeddings[start : start + batch]
            self._collection.add(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=vecs.tolist(),
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "index": c.index,
                        "strategy": c.strategy,
                        # Chroma rejects None — empty string is the "no section" marker.
                        "section": c.section or "",
                        "char_count": c.char_count,
                        "title": str(c.metadata.get("title", "")),
                    }
                    for c in part
                ],
            )

    def query(self, query_embedding: np.ndarray, k: int) -> list[dict]:
        """Return the k nearest chunks as [{chunk_id, text, score, metadata}],
        best first, score = cosine similarity (bigger = better)."""
        count = self.count()
        if count == 0:
            return []
        res = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, count),  # asking Chroma for more than it has warns loudly
            include=["documents", "metadatas", "distances"],
        )
        # Chroma's response is column-oriented and nested one level per query;
        # we always send exactly one query, hence the [0]s.
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": 1.0 - distance,  # distance -> similarity, see module docstring
                "metadata": metadata or {},
            }
            for chunk_id, text, distance, metadata in zip(
                res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0]
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection — how re-ingest starts clean
        instead of layering new chunks over stale ones."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
