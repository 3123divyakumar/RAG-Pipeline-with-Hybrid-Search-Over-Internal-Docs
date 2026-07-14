"""Eval metrics — the four numbers that tell you whether the pipeline works.

The four, and what each one localizes when it drops:

  answer_correctness  end-to-end quality vs the golden answer   (LLM judge)
  faithfulness        is EVERY claim grounded in the context?   (LLM judge)
  retrieval_hit/MRR   did the right documents even show up?     (pure bookkeeping)
  citation_accuracy   were the [n] markers honest?              (already computed in ask())

The judge-free retrieval metrics are the debugging pivot: a bad answer with
hit@k=1 means GENERATION failed; hit@k=0 means RETRIEVAL failed. Without that
split, every regression is a fog of "the model got dumber".
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.generate.generate import AskResult
    from rag.llm import LLMClient

logger = logging.getLogger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n")


# ---------------------------------------------------------------------------
# answer_correctness — judged against the golden answer
# ---------------------------------------------------------------------------

_CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},  # 0, 0.5, or 1
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

_CORRECTNESS_SYSTEM = """\
You grade a CANDIDATE answer against a REFERENCE answer to the same question. \
Grade agreement of substance, not wording.

Scores (no other values allowed):
- 1.0: the candidate conveys the same essential facts as the reference. Extra \
CORRECT detail is fine.
- 0.5: partially correct — some key facts right, others missing or wrong. \
Also the CEILING for a correct answer that adds details not in the reference \
and not obviously true (possible hallucinations).
- 0.0: wrong, contradicts the reference, or answers a different question.

Be a skeptical grader. Confident wrong answers are 0.0, not 0.5.
"""


def answer_correctness(result: AskResult, golden: dict, llm: LLMClient) -> float:
    """LLM-as-judge against the golden answer; special-cased for unanswerables.

    For type=="unanswerable", correctness is mechanical, no judge needed:
    the RIGHT behavior is refusing (answered=False -> 1.0); an eloquent
    made-up answer is exactly the failure the type exists to catch (0.0).
    """
    if golden["type"] == "unanswerable":
        return 1.0 if not result.answered else 0.0
    if not result.answered:
        # Refused an answerable question. Wrong, but distinguishable in the
        # per-type breakdown from "answered incorrectly".
        return 0.0
    try:
        verdict = llm.judge_chat(
            [
                {"role": "system", "content": _CORRECTNESS_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{golden['question']}\n\n"
                        f"REFERENCE:\n{golden['golden_answer']}\n\n"
                        f"CANDIDATE:\n{result.answer}"
                    ),
                },
            ],
            schema=_CORRECTNESS_SCHEMA,
        )
        score = float(verdict["score"])
        return min(1.0, max(0.0, round(score * 2) / 2))  # snap to {0, 0.5, 1}
    except Exception as e:
        logger.warning("correctness judge failed for %s: %s", golden["id"], e)
        return 0.0  # fail conservative: an ungradeable answer earns nothing


# ---------------------------------------------------------------------------
# faithfulness — every claim grounded, cited or not
# ---------------------------------------------------------------------------

_FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}},
    "required": ["supported"],
    "additionalProperties": False,
}

_FAITHFULNESS_SYSTEM = """\
You will be given a CONTEXT and one CLAIM. Answer whether the claim is fully \
supported by the context. The context is the only evidence that counts — your \
own knowledge is forbidden. Partial support counts as NOT supported.
"""


def faithfulness(result: AskResult, llm: LLMClient) -> float:
    """Share of the answer's claims supported by the retrieved context.

    Citation verification (in ask()) checks CLAIMED support — [n] vs block n.
    Faithfulness catches UNCITED freelancing: sentences the model slipped in
    without a marker, judged here against the FULL context. One judge call
    per sentence; IDK answers are vacuously faithful (nothing was claimed).
    """
    if not result.answered:
        return 1.0
    claims = [s.strip() for s in _SENT_SPLIT_RE.split(result.answer) if s.strip()]
    # Strip citation markers so the judge sees prose, not bracket noise.
    claims = [re.sub(r"\[\d+\]", "", c).strip() for c in claims]
    claims = [c for c in claims if len(c) > 15]  # skip fragments like "Yes."
    if not claims:
        return 1.0
    context = "\n\n".join(c.text for c in result.chunks)
    supported = 0
    for claim in claims:
        try:
            verdict = llm.judge_chat(
                [
                    {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"},
                ],
                schema=_FAITHFULNESS_SCHEMA,
            )
            supported += bool(verdict["supported"])
        except Exception as e:
            logger.warning("faithfulness judge failed: %s", e)  # unsupported by default
    return supported / len(claims)


# ---------------------------------------------------------------------------
# retrieval hit@k and MRR — no judge, pure bookkeeping
# ---------------------------------------------------------------------------


def retrieval_hit(result: AskResult, golden: dict) -> tuple[float, float]:
    """(hit@k, MRR) over the FINAL chunk list the LLM saw.

    hit@k: 1.0 if any chunk came from a golden_doc_id. MRR: 1/rank of the
    first such chunk (0 if absent) — rewards ranking the right doc high,
    since the LLM reads top blocks with more attention than bottom ones.
    Unanswerable questions have no golden docs; retrieval metrics don't
    apply (the runner skips them in aggregation).
    """
    golden_docs = set(golden.get("golden_doc_ids") or [])
    if not golden_docs:
        return 0.0, 0.0
    for i, chunk in enumerate(result.chunks, start=1):
        if chunk.metadata.get("doc_id") in golden_docs:
            return 1.0, 1.0 / i
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# citation accuracy — reuses the verification ask() already ran
# ---------------------------------------------------------------------------


def citation_accuracy(result: AskResult) -> float | None:
    """verified / total citations. None (excluded from averages) when the
    answer had no citations — 0/0 is "no data", and coercing it to either
    0.0 or 1.0 would bias the aggregate in opposite but equally wrong ways."""
    if not result.citations:
        return None
    return sum(1 for c in result.citations if c.verified) / len(result.citations)
