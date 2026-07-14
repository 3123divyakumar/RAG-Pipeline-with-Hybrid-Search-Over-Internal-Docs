"""Citation verification — does [1] actually support the claim it's attached to?

This is the quality layer most RAG systems skip, and the strongest
differentiator in the project: the model was TOLD to cite (prompts.py rule 1),
but nothing stops it from decorating a hallucinated sentence with a
plausible-looking [2]. So every (claim, cited chunk) pair gets independently
checked by an LLM-as-judge call with a strict binary rubric.

Known bias, designed around rather than ignored: locally, the judge is the
SAME model that wrote the answer (self-preference bias — models rate their
own outputs too kindly). Mitigations actually in use here:
  1. strict binary rubric — "partially supported" counts as NOT supported;
  2. evidence-only framing — the judge sees ONE chunk and ONE claim, never
     the full answer it might feel ownership of;
  3. the judge model is separately configurable (settings.judge_model) — the
     deployed stack points it at a different model;
  4. a manual audit measures judge-vs-human agreement on a sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.generate.generate import Citation
    from rag.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    claim: str
    chunk_id: str
    supported: bool
    rationale: str  # one judge sentence — surfaced in the dashboard next to each citation


# JSON schema for the judge's verdict — passed to json_chat so the model is
# CONSTRAINED to this shape. "supported" must be a real boolean; parsing
# free-text verdicts like "Yes, mostly" is how judge pipelines rot.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["supported", "rationale"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM = """\
You are a strict fact-checking judge. You will be given a CLAIM and an \
EVIDENCE passage. Decide whether the evidence fully supports the claim.

Rules:
- The evidence passage is the ONLY thing that counts. Your own knowledge is \
forbidden, even if the claim is true in the real world.
- "Partially supported" counts as NOT supported. Every part of the claim \
must be backed by the evidence.
- Judge the claim exactly as written, not what its author probably meant.
- rationale: one short sentence explaining your decision.
"""


def verify_citation(claim: str, chunk_text: str, llm: LLMClient) -> Verdict:
    """One judge call for one (claim, cited chunk) pair."""
    try:
        result = llm.judge_chat(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"CLAIM:\n{claim}\n\nEVIDENCE:\n{chunk_text}",
                },
            ],
            schema=_VERDICT_SCHEMA,
        )
        return Verdict(
            claim=claim,
            chunk_id="",  # filled by verify_answer, which knows the citation
            supported=bool(result["supported"]),
            rationale=str(result.get("rationale", "")),
        )
    except Exception as e:  # judge call failed (LLM down, bad JSON twice, ...)
        # Fail SAFE: an unverifiable citation counts as unsupported. Claiming
        # verification we didn't perform would defeat the entire layer.
        logger.warning("citation verification failed: %s", e)
        return Verdict(claim=claim, chunk_id="", supported=False, rationale=f"judge error: {e}")


def verify_answer(
    citations: list[Citation],
    chunks_by_id: dict[str, str],  # chunk_id -> chunk text
    llm: LLMClient,
) -> list[Verdict]:
    """Verify every citation in an answer; mutates each Citation.verified.

    Sequential on purpose: locally, Ollama processes one request at a time
    anyway, and an answer has ~3-8 citations. Parallelizing would complicate
    error handling to save nothing.
    """
    verdicts: list[Verdict] = []
    for citation in citations:
        chunk_text = chunks_by_id.get(citation.chunk_id)
        if chunk_text is None:
            # Marker pointed at a block number that doesn't exist ([7] with 5
            # blocks) — a hallucinated citation, unsupported by definition.
            verdict = Verdict(
                claim=citation.claim,
                chunk_id=citation.chunk_id,
                supported=False,
                rationale="citation points at a context block that does not exist",
            )
        else:
            verdict = verify_citation(citation.claim, chunk_text, llm)
            verdict.chunk_id = citation.chunk_id
        citation.verified = verdict.supported
        verdicts.append(verdict)
    return verdicts
