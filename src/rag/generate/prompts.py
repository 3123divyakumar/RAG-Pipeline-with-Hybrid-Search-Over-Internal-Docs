"""Prompt construction — the contract between retrieval and the LLM.

Prompts are code: they're versioned (SYSTEM_PROMPT_V1), every rule has a
reason, and no prompt gets edited without re-running the eval suite (see
makes that cheap). When V2 happens, keep V1 and add a changelog comment with
the eval numbers that justified the change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.retrieve.dense import RetrievedChunk

# The grounding contract. Rule-by-rule rationale:
#   - "ONLY the context blocks": the grounding rule itself — the model's own
#     training knowledge is where hallucinations come from.
#   - "[n] after every factual claim": makes each claim mechanically checkable;
#     verify.py depends on markers mapping to numbered blocks.
#   - explicit refusal phrase: gives the model a sanctioned way OUT of
#     answering. Without it, an LLM under instruction pressure will always
#     produce SOMETHING. The exact phrase is also a machine-detectable signal
#     (generate.py checks for it) — keep the two in sync.
#   - "concise, no preamble": filler sentences ("Great question!...") aren't
#     citable claims, so they'd only dilute citation coverage.
SYSTEM_PROMPT_V1 = """\
You are a technical documentation assistant. Answer the user's question using \
ONLY the numbered context blocks provided. Follow these rules exactly:

1. Every factual claim in your answer must end with a citation marker [n] \
where n is the number of the context block that supports it. A sentence may \
carry multiple markers [1][3] if it draws on multiple blocks.
2. Use ONLY information from the context blocks. Do not use any outside \
knowledge, even if you are sure it is correct.
3. If the context blocks do not contain the information needed to answer, \
reply with exactly: I cannot answer this from the provided documentation. \
Do not guess or partially answer.
4. Be concise. No preamble, no "Based on the context", no summary of what \
you were asked. Answer directly.
"""

# The refusal phrase from rule 3, used by generate.py to detect the model's
# own "I don't know" — single source of truth for both sides of the contract.
REFUSAL_PHRASE = "I cannot answer this from the provided documentation"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as the numbered blocks the citations point back to.

    Numbering is 1-based and ORDER IS THE RERANKED ORDER — the [n] -> chunk
    mapping created here is exactly what citation parsing (generate.py),
    verification (verify.py) and the dashboard's chunk inspector all rely on.
    Change the format here and you change it for all of them.
    """
    blocks = []
    for n, chunk in enumerate(chunks, start=1):
        doc_id = chunk.metadata.get("doc_id", chunk.chunk_id)
        section = chunk.metadata.get("section") or ""
        header = f"[{n}] (source: {doc_id}" + (f" — \"{section}\")" if section else ")")
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Assemble the exact messages list LLMClient.chat() receives.

    Context goes in the USER message (after the question), not the system
    message: system prompts set persistent behavior, user messages carry
    per-request data — and some providers weight system-message instructions
    differently from data, so mixing 4KB of context into it is asking for
    the rules to get diluted.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V1},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Context blocks:\n\n{build_context(chunks)}"
            ),
        },
    ]
