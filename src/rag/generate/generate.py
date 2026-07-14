"""The ask() orchestrator — the end-to-end pipeline in one function.

    retrieve -> (confidence gate) -> prompt -> LLM -> parse citations
             -> verify citations -> score confidence -> AskResult

Everything the API serves and the dashboard displays is an AskResult; every
eval suite scores one. This file is deliberately just orchestration —
each real decision lives in the module that owns it (prompts/verify/confidence).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag.config import get_settings
from rag.generate.confidence import ConfidenceReport, build_report, retrieval_confidence
from rag.generate.prompts import REFUSAL_PHRASE, build_messages
from rag.generate.verify import verify_answer
from rag.llm import LLMClient
from rag.retrieve.pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from rag.config import Settings
    from rag.retrieve.dense import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    marker: int  # the [n] as written by the model
    chunk_id: str  # the chunk block [n] pointed at ("" if n was out of range)
    claim: str  # the sentence the marker was attached to
    verified: bool | None = None  # None until verify.py has run


@dataclass
class AskResult:
    question: str
    answer: str  # or the structured IDK text
    citations: list[Citation]
    chunks: list[RetrievedChunk]  # exactly what the LLM saw, in block order
    confidence: ConfidenceReport
    mode: str  # "hybrid" | "dense" — the dashboard toggle
    strategy: str  # which chunking strategy's index served this
    answered: bool  # False -> the IDK path was taken
    timings_ms: dict[str, float] = field(default_factory=dict)  # per-stage latency


# A citation marker: [3] or chains like [1][4]. Finds each bracketed integer.
_MARKER_RE = re.compile(r"\[(\d+)\]")
# Sentence boundary for claim extraction (same crudeness tradeoff as chunkers).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n")


def parse_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Turn the model's [n] markers into Citation objects.

    Per sentence: every [n] in it becomes one Citation whose `claim` is the
    sentence with markers stripped. The [n] -> chunk mapping is positional —
    block numbering in prompts.build_context() is 1-based over `chunks`, so
    chunks[n-1] is the cited chunk.

    Out-of-range markers ([7] when only 5 blocks exist) are KEPT with
    chunk_id="" — they're hallucinated citations, and verify.py must count
    them as unsupported rather than have them quietly vanish from coverage.
    """
    citations: list[Citation] = []
    for sentence in _SENT_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        markers = [int(m) for m in _MARKER_RE.findall(sentence)]
        if not markers:
            continue
        claim = _MARKER_RE.sub("", sentence).strip()
        for n in markers:
            in_range = 1 <= n <= len(chunks)
            if not in_range:
                logger.warning("citation [%d] out of range (only %d blocks)", n, len(chunks))
            citations.append(
                Citation(marker=n, chunk_id=chunks[n - 1].chunk_id if in_range else "", claim=claim)
            )
    return citations


def _idk_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """The structured "I don't know" — honest AND useful.

    Instead of a bare refusal, it reports what was searched and the near-miss
    documents, so the user can judge whether to rephrase or go read manually.
    """
    lines = [
        "I couldn't find enough relevant material in the indexed documentation "
        "to answer this confidently, so I'm not going to guess.",
        "",
        f"What I searched for: {question!r}",
    ]
    if chunks:
        lines.append("Closest matches found (none strong enough to answer from):")
        seen: set[str] = set()
        for c in chunks[:5]:
            doc = c.metadata.get("doc_id", c.chunk_id)
            section = c.metadata.get("section") or ""
            label = f"{doc}" + (f" — {section}" if section else "")
            if label not in seen:
                seen.add(label)
                lines.append(f"  - {label}")
        lines.append("Those documents may still be worth a manual look.")
    return "\n".join(lines)


def ask(
    question: str,
    *,
    mode: str = "hybrid",
    strategy: str = "recursive",
    settings: Settings | None = None,
    pipeline: RetrievalPipeline | None = None,
    llm: LLMClient | None = None,
) -> AskResult:
    """Answer a question end-to-end. See module docstring for the stages.

    `pipeline` and `llm` are injectable so the API (which keeps warm,
    model-loaded instances on app.state) and the eval runner (which reuses
    one pipeline across 50 questions) never pay startup cost per call —
    while a plain script can call ask("...") with no setup at all.
    """
    settings = settings or get_settings()
    pipeline = pipeline or RetrievalPipeline(settings)
    llm = llm or LLMClient(settings)
    timings: dict[str, float] = {}

    # -- 1. retrieve (the pipeline handles fusion + reranking internally) --
    t0 = time.perf_counter()
    chunks = pipeline.retrieve(question, mode=mode, strategy=strategy)
    timings["retrieve"] = (time.perf_counter() - t0) * 1000

    # -- 2. confidence gate: is this even answerable from what we found? --
    # Below the threshold we SKIP generation entirely: not generating is both
    # cheaper (no LLM call) and safer (junk context in, confident junk out).
    r_conf = retrieval_confidence(chunks)
    if r_conf < settings.confidence_idk_threshold:
        return AskResult(
            question=question,
            answer=_idk_answer(question, chunks),
            citations=[],
            chunks=chunks,
            confidence=ConfidenceReport(
                retrieval=r_conf, citation_coverage=0.0, completeness=0.0, composite=0.0
            ),
            mode=mode,
            strategy=strategy,
            answered=False,
            timings_ms=timings,
        )

    # -- 3. build the grounded prompt and generate --
    t0 = time.perf_counter()
    answer = llm.chat(build_messages(question, chunks))
    timings["generate"] = (time.perf_counter() - t0) * 1000

    # The model can also refuse on its own (prompt rule 3) when the gate let
    # marginal context through — that's the second half of the IDK contract.
    if REFUSAL_PHRASE.lower() in answer.lower():
        return AskResult(
            question=question,
            answer=_idk_answer(question, chunks),
            citations=[],
            chunks=chunks,
            confidence=ConfidenceReport(
                retrieval=r_conf, citation_coverage=0.0, completeness=0.0, composite=0.0
            ),
            mode=mode,
            strategy=strategy,
            answered=False,
            timings_ms=timings,
        )

    # -- 4. parse [n] markers into Citations --
    citations = parse_citations(answer, chunks)

    # -- 5. verify every citation, then score confidence --
    t0 = time.perf_counter()
    chunks_by_id = {c.chunk_id: c.text for c in chunks}
    verify_answer(citations, chunks_by_id, llm)  # sets citation.verified in place
    timings["verify"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    confidence = build_report(chunks, citations, question, answer, llm)
    timings["confidence"] = (time.perf_counter() - t0) * 1000

    return AskResult(
        question=question,
        answer=answer,
        citations=citations,
        chunks=chunks,
        confidence=confidence,
        mode=mode,
        strategy=strategy,
        answered=True,
        timings_ms=timings,
    )
