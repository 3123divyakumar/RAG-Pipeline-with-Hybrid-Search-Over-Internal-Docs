"""Answer confidence — one honest number (plus its parts) attached to every answer.

Three independent signals, then a weighted mix:

  retrieval          "did we FIND good material?"   (from reranker scores)
  citation_coverage  "is the answer BACKED by it?"  (verified / total claims)
  completeness       "did we answer ALL of it?"     (one judge call)

They fail independently — great retrieval + zero verified citations means the
model freelanced; perfect citations on half the question means incompleteness
— which is why the composite reports its parts, not just the blend.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.generate.generate import Citation
    from rag.llm import LLMClient
    from rag.retrieve.dense import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceReport:
    retrieval: float  # 0-1: how relevant were the final top chunks?
    citation_coverage: float  # 0-1: verified citations / total citations
    completeness: float  # 0-1: did the answer address the whole question?
    composite: float  # weighted mix — weights documented in DECISIONS.md


def retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    """Squash the top chunks' relevance scores into 0-1.

    The final chunks carry cross-encoder logits when reranking is on
    (ms-marco models: roughly >+5 clearly relevant, <-5 clearly not — a
    sigmoid maps that range cleanly to ~0.99/~0.01). With reranking OFF the
    scores are RRF sums (~0.0-0.03), useless for absolute confidence — we
    fall back to a neutral 0.5 rather than pretend to know.

    Top-3 mean, not top-1: one lucky chunk shouldn't buy high confidence;
    an answer usually needs a couple of good sources.
    """
    if not chunks:
        return 0.0
    if chunks[0].source != "reranked":
        return 0.5  # no calibrated signal available — honest neutrality
    top = chunks[:3]
    return float(sum(1.0 / (1.0 + math.exp(-c.score)) for c in top) / len(top))


def citation_coverage(citations: list[Citation]) -> float:
    """verified / total. The edge case that matters: an answer with ZERO
    citations scores 0.0, not 1.0 — "no claims checked" must never look
    like "all claims checked out". (0/0 = 1 is the classic silent bug here.)"""
    if not citations:
        return 0.0
    verified = sum(1 for c in citations if c.verified)
    return verified / len(citations)


_COMPLETENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},  # 0.0-1.0
        "missing": {"type": "string"},  # what wasn't addressed ("" if nothing)
    },
    "required": ["score", "missing"],
    "additionalProperties": False,
}

_COMPLETENESS_SYSTEM = """\
You grade whether an ANSWER addresses every part of a QUESTION. You are NOT \
grading correctness — only coverage. Multi-part questions ("how do I do X \
and what happens if Y?") must have every part addressed.

Respond with:
- score: 1.0 = every part addressed; 0.5 = some parts addressed; 0.0 = the \
answer talks past the question.
- missing: one short phrase naming what was not addressed, or "" if nothing.
"""


def completeness(question: str, answer: str, llm: LLMClient) -> float:
    """One judge call — catches the failure where RAG answers half of a
    multi-part question fluently and SOUNDS done."""
    try:
        result = llm.judge_chat(
            [
                {"role": "system", "content": _COMPLETENESS_SYSTEM},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}"},
            ],
            schema=_COMPLETENESS_SCHEMA,
        )
        return max(0.0, min(1.0, float(result["score"])))  # clamp — judges drift
    except Exception as e:
        logger.warning("completeness judge failed: %s", e)
        return 0.5  # neutral on judge failure; don't zero an answer for infra issues


def composite(retrieval: float, coverage: float, complete: float) -> float:
    """0.4 * retrieval + 0.4 * coverage + 0.2 * completeness.

    Why these weights (the DECISIONS.md argument): retrieval and coverage are
    the two hallucination-facing signals, weighted equally; completeness is
    real but a complete WRONG answer is worse than an incomplete right one,
    so it gets half their weight. The eval suite sanity-checks the blend: low
    composite should predict wrong answers on the golden set — if it doesn't,
    retune and write down why.
    """
    return 0.4 * retrieval + 0.4 * coverage + 0.2 * complete


def build_report(
    chunks: list[RetrievedChunk],
    citations: list[Citation],
    question: str,
    answer: str,
    llm: LLMClient,
) -> ConfidenceReport:
    """Convenience assembler used by ask()."""
    r = retrieval_confidence(chunks)
    cov = citation_coverage(citations)
    comp = completeness(question, answer, llm)
    return ConfidenceReport(
        retrieval=r,
        citation_coverage=cov,
        completeness=comp,
        composite=composite(r, cov, comp),
    )
