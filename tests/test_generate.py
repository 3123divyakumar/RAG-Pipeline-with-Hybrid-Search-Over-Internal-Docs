"""Generation-layer tests — everything that doesn't need a live LLM.

The LLM-dependent paths (verify, completeness) are exercised with a fake
client; citation parsing and confidence math are pure functions and get
tested directly. End-to-end behavior with a real model is checked by
scripts/ask.py and the eval suite, not unit tests.
"""

from rag.generate.confidence import citation_coverage, composite, retrieval_confidence
from rag.generate.generate import Citation, parse_citations
from rag.generate.prompts import REFUSAL_PHRASE, SYSTEM_PROMPT_V1, build_context, build_messages
from rag.retrieve.dense import RetrievedChunk


def rc(chunk_id: str, rank: int, score: float = 5.0, source: str = "reranked") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        score=score,
        rank=rank,
        source=source,
        metadata={"doc_id": f"docs/{chunk_id}.md", "section": "Some Section"},
    )


CHUNKS = [rc("a", 1), rc("b", 2), rc("c", 3)]


class TestParseCitations:
    def test_single_marker_maps_to_right_chunk(self):
        answer = "FastAPI validates request bodies with Pydantic [2]."
        cites = parse_citations(answer, CHUNKS)
        assert len(cites) == 1
        assert cites[0].marker == 2
        assert cites[0].chunk_id == "b"  # 1-based: [2] -> chunks[1]
        assert "[2]" not in cites[0].claim  # markers stripped from the claim

    def test_multiple_markers_on_one_sentence(self):
        answer = "You can combine Path and Query parameters [1][3]."
        cites = parse_citations(answer, CHUNKS)
        assert [c.marker for c in cites] == [1, 3]
        assert cites[0].claim == cites[1].claim  # same sentence, two citations

    def test_out_of_range_marker_is_kept_as_hallucinated(self):
        """[7] with 3 blocks must NOT vanish — it must surface as a citation
        with no chunk so verification counts it against coverage."""
        cites = parse_citations("This is definitely true [7].", CHUNKS)
        assert len(cites) == 1
        assert cites[0].chunk_id == ""

    def test_sentences_without_markers_produce_no_citations(self):
        cites = parse_citations("Just prose. No citations here at all.", CHUNKS)
        assert cites == []

    def test_multi_sentence_answer(self):
        answer = "Use response_model to filter output [1]. It uses Pydantic [2]. Done."
        cites = parse_citations(answer, CHUNKS)
        assert [c.marker for c in cites] == [1, 2]
        assert "response_model" in cites[0].claim


class TestPrompts:
    def test_context_blocks_are_numbered_in_order(self):
        ctx = build_context(CHUNKS)
        assert ctx.index("[1]") < ctx.index("[2]") < ctx.index("[3]")
        assert "docs/a.md" in ctx  # source shown for citation display

    def test_messages_shape(self):
        msgs = build_messages("how do I do X?", CHUNKS)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == SYSTEM_PROMPT_V1
        assert "how do I do X?" in msgs[1]["content"]
        assert "text of a" in msgs[1]["content"]  # chunks actually included

    def test_refusal_phrase_is_in_the_system_prompt(self):
        """generate.py string-matches this phrase — the two must stay in sync."""
        assert REFUSAL_PHRASE in SYSTEM_PROMPT_V1


class TestConfidence:
    def test_zero_citations_scores_zero_not_one(self):
        assert citation_coverage([]) == 0.0

    def test_coverage_ratio(self):
        cites = [
            Citation(marker=1, chunk_id="a", claim="x", verified=True),
            Citation(marker=2, chunk_id="b", claim="y", verified=False),
        ]
        assert citation_coverage(cites) == 0.5

    def test_retrieval_confidence_high_for_strong_rerank_scores(self):
        strong = [rc("a", 1, score=8.0), rc("b", 2, score=7.0), rc("c", 3, score=6.0)]
        assert retrieval_confidence(strong) > 0.9

    def test_retrieval_confidence_low_for_weak_scores(self):
        weak = [rc("a", 1, score=-8.0), rc("b", 2, score=-9.0)]
        assert retrieval_confidence(weak) < 0.1

    def test_retrieval_confidence_neutral_without_reranker(self):
        fused = [rc("a", 1, score=0.02, source="hybrid")]
        assert retrieval_confidence(fused) == 0.5

    def test_empty_chunks_score_zero(self):
        assert retrieval_confidence([]) == 0.0

    def test_composite_weights(self):
        assert abs(composite(1.0, 1.0, 1.0) - 1.0) < 1e-9
        assert abs(composite(1.0, 0.0, 0.0) - 0.4) < 1e-9
        assert abs(composite(0.0, 0.0, 1.0) - 0.2) < 1e-9
