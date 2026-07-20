"""Chunker tests — the properties that make retrieval trustworthy.

What we test and why:
  - coverage: no text silently vanishes at window boundaries
  - overlap: neighboring fixed chunks actually share text
  - no empties: an empty chunk would embed to a meaningless vector
  - tiny docs: one chunk, no windowing arithmetic edge cases
  - sections: recursive chunks know which heading they live under
"""

import numpy as np
import pytest

from rag.config import Settings
from rag.ingest.chunkers import chunk, chunk_fixed, chunk_recursive, chunk_semantic
from rag.ingest.loaders import Document


def make_doc(text: str) -> Document:
    return Document(
        doc_id="test/doc.md",
        text=text,
        source_path="test/doc.md",
        format="md",
        title="Test Doc",
    )


SETTINGS = Settings(_env_file=None, chunk_size=200, chunk_overlap=40)

LONG_TEXT = "\n\n".join(
    f"## Section {i}\n\n" + " ".join(f"word{i}_{j}" for j in range(60)) for i in range(5)
)


class TestFixed:
    def test_tiny_doc_is_one_chunk(self):
        chunks = chunk_fixed(make_doc("short text"), size=200, overlap=40)
        assert len(chunks) == 1
        assert chunks[0].text == "short text"
        assert chunks[0].chunk_id == "test/doc.md::fixed::0"

    def test_no_empty_chunks_and_sizes_reasonable(self):
        chunks = chunk_fixed(make_doc(LONG_TEXT), size=200, overlap=40)
        assert len(chunks) > 1
        assert all(c.text.strip() for c in chunks)
        # +80 = the whitespace-nudge allowance in chunk_fixed
        assert all(c.char_count <= 200 + 80 for c in chunks)

    def test_all_text_is_covered(self):
        """Every word of the source must appear in at least one chunk."""
        doc = make_doc(LONG_TEXT)
        chunks = chunk_fixed(doc, size=200, overlap=40)
        joined = " ".join(c.text for c in chunks)
        for word in LONG_TEXT.split():
            assert word in joined

    def test_neighbors_overlap(self):
        chunks = chunk_fixed(make_doc(LONG_TEXT), size=200, overlap=40)
        for a, b in zip(chunks, chunks[1:]):
            # The head of chunk b must appear somewhere in chunk a.
            assert b.text[:20] in a.text

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            chunk_fixed(make_doc(LONG_TEXT), size=100, overlap=100)


class TestRecursive:
    def test_tiny_doc_is_one_chunk(self):
        chunks = chunk_recursive(make_doc("## Intro\n\nshort text"), size=200, overlap=40)
        assert len(chunks) == 1
        assert chunks[0].section == "Intro"

    def test_respects_size_and_no_empties(self):
        chunks = chunk_recursive(make_doc(LONG_TEXT), size=200, overlap=40)
        assert len(chunks) > 1
        assert all(c.text.strip() for c in chunks)
        # size + overlap carry-over is the worst legitimate case
        assert all(c.char_count <= 200 + 40 + 2 for c in chunks)

    def test_sections_are_tracked(self):
        chunks = chunk_recursive(make_doc(LONG_TEXT), size=200, overlap=40)
        assert any(c.section and c.section.startswith("Section") for c in chunks)

    def test_all_words_covered(self):
        chunks = chunk_recursive(make_doc(LONG_TEXT), size=200, overlap=40)
        joined = " ".join(c.text for c in chunks)
        for word in LONG_TEXT.split():
            assert word in joined

    def test_indexes_are_sequential(self):
        chunks = chunk_recursive(make_doc(LONG_TEXT), size=200, overlap=40)
        assert [c.index for c in chunks] == list(range(len(chunks)))


class FakeEmbedder:
    """Deterministic stand-in: sentences containing 'cat' embed to one
    direction, 'dog' to an orthogonal one — so the topic boundary between the
    cat sentences and the dog sentences is the ONLY similarity dip, and the
    semantic chunker must cut exactly there."""

    def embed_texts(self, texts):
        out = []
        for t in texts:
            v = np.array([1.0, 0.0]) if "cat" in t else np.array([0.0, 1.0])
            out.append(v)
        return np.array(out)

    def embed_query(self, query):  # pragma: no cover - not used by chunkers
        return np.array([1.0, 0.0])


class TestSemantic:
    def test_cuts_at_topic_shift(self):
        text = (
            "The cat sat on the mat. The cat likes fish. A cat sleeps all day. "
            "The cat purrs loudly. Dogs bark at strangers. A dog fetches sticks. "
            "The dog wags its tail. Dogs love long walks."
        )
        chunks = chunk_semantic(make_doc(text), embedder=FakeEmbedder(), min_sentences=2)
        assert len(chunks) == 2
        assert "cat" in chunks[0].text and "dog" not in chunks[0].text.lower()
        assert "dog" in chunks[1].text.lower()

    def test_tiny_doc_is_one_chunk(self):
        chunks = chunk_semantic(make_doc("One sentence only."), embedder=FakeEmbedder())
        assert len(chunks) == 1


class TestDispatcher:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            chunk(make_doc("x"), "banana", SETTINGS)

    def test_semantic_without_embedder_raises(self):
        with pytest.raises(ValueError, match="embedder"):
            chunk(make_doc("x"), "semantic", SETTINGS)

    def test_chunk_ids_are_unique(self):
        doc = make_doc(LONG_TEXT)
        for strategy in ("fixed", "recursive"):
            chunks = chunk(doc, strategy, SETTINGS)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids))
