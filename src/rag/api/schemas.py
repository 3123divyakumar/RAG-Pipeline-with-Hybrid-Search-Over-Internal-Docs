"""API request/response models — the public contract of the service.

These Pydantic models are deliberately a SEPARATE layer from the internal
dataclasses in generate/ and retrieve/. The duplication is the point:
internals can be refactored freely (rename a field, add one) without breaking
API consumers, because the only thing consumers see is this file. They also
drive the auto-generated OpenAPI docs page (/docs) and are what the React
dashboard's fetch calls mirror.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# /v1/ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    # min_length rejects "" at the edge with a clean 422 instead of letting an
    # empty query waste a retrieval + LLM round trip deeper in the stack.
    question: str = Field(min_length=3, max_length=2000, examples=[
        "How do I declare a request body in FastAPI?"
    ])
    mode: Literal["hybrid", "dense"] = "hybrid"  # powers the dashboard's A/B toggle
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"


class CitationOut(BaseModel):
    marker: int  # the [n] as it appears in the answer text
    chunk_id: str  # "" when the marker pointed at a nonexistent block
    claim: str
    verified: bool | None  # None = verification didn't run for this citation


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    source: str  # dense | sparse | hybrid | reranked
    doc_id: str
    section: str


class ConfidenceOut(BaseModel):
    retrieval: float
    citation_coverage: float
    completeness: float
    composite: float


class AskResponse(BaseModel):
    answer: str
    answered: bool  # False -> the IDK path fired; answer holds the structured refusal
    citations: list[CitationOut]
    chunks: list[ChunkOut]  # exactly what the LLM saw — the dashboard's chunk inspector
    confidence: ConfidenceOut
    mode: str
    strategy: str
    timings_ms: dict[str, float]


# ---------------------------------------------------------------------------
# /v1/documents and /v1/ingest
# ---------------------------------------------------------------------------


class DocumentInfo(BaseModel):
    doc_id: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    strategy: str
    total_chunks: int
    documents: list[DocumentInfo]


class IngestResponse(BaseModel):
    """Counts from one upload-and-index run — mirrors the ingest CLI summary."""

    files_received: int
    loaded: int
    chunked: int
    deduped: int  # how many near-duplicates were DROPPED
    indexed: int  # chunks that made it into both indexes
    strategy: str
