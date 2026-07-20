"""FastAPI application — the service wrapper around ask().

Run locally:
    uv run uvicorn rag.api.main:app --reload
    -> docs at http://localhost:8000/docs

The single most important pattern in this file: the LIFESPAN HANDLER loads
the heavy components (embedder ~130MB, reranker ~80MB, both indexes) ONCE at
startup and stashes them on app.state. Loading them per-request would add
seconds of latency to every call. Everything request-scoped stays cheap.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    CitationOut,
    ConfidenceOut,
    DocumentInfo,
    DocumentsResponse,
    IngestResponse,
)
from rag.config import get_settings
from rag.generate.generate import AskResult, ask
from rag.index.sparse import SparseIndex
from rag.index.vector import VectorIndex
from rag.ingest.chunkers import STRATEGIES, chunk
from rag.ingest.dedup import dedup
from rag.ingest.loaders import load_file
from rag.llm import LLMClient
from rag.retrieve.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build the shared, model-loaded singletons. Shutdown: nothing —
    all state is in-process and dies with it."""
    settings = get_settings()
    app.state.settings = settings
    app.state.pipeline = RetrievalPipeline(settings)  # lazy inside; first /ask warms it
    app.state.llm = LLMClient(settings)
    logger.info("pipeline + LLM client ready (models load lazily on first ask)")
    yield


app = FastAPI(
    title="RAG pipeline — hybrid search over technical docs",
    description="Dense + BM25 retrieval, RRF fusion, cross-encoder reranking, "
    "grounded answers with independently verified citations.",
    version="1.0.0",
    lifespan=lifespan,
)

# The Vite dev server runs on another origin — without CORS headers
# the browser refuses to show it our responses. In production the built UI is
# served from THIS app (same origin), so this list only matters in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_response(result: AskResult) -> AskResponse:
    """Internal dataclasses -> public API models. The ONLY place the two
    layers touch, so contract changes are one-file diffs."""
    return AskResponse(
        answer=result.answer,
        answered=result.answered,
        citations=[
            CitationOut(marker=c.marker, chunk_id=c.chunk_id, claim=c.claim, verified=c.verified)
            for c in result.citations
        ],
        chunks=[
            ChunkOut(
                chunk_id=c.chunk_id,
                text=c.text,
                score=c.score,
                rank=c.rank,
                source=c.source,
                doc_id=str(c.metadata.get("doc_id", "")),
                section=str(c.metadata.get("section", "")),
            )
            for c in result.chunks
        ],
        confidence=ConfidenceOut(
            retrieval=result.confidence.retrieval,
            citation_coverage=result.confidence.citation_coverage,
            completeness=result.confidence.completeness,
            composite=result.confidence.composite,
        ),
        mode=result.mode,
        strategy=result.strategy,
        timings_ms=result.timings_ms,
    )


@app.get("/health")
def health() -> dict:
    """Liveness probe — Railway (and docker compose) poll this."""
    return {"status": "ok"}


@app.post("/v1/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """The product: question in, verified-cited answer out."""
    settings = app.state.settings
    # Clear 503 up front beats a Chroma stack trace from deep inside retrieval.
    if VectorIndex(settings, request.strategy).count() == 0:
        raise HTTPException(
            status_code=503,
            detail=f"No index for strategy '{request.strategy}'. "
            "Run `uv run python scripts/ingest.py` (or POST /v1/ingest) first.",
        )
    try:
        result = ask(
            request.question,
            mode=request.mode,
            strategy=request.strategy,
            settings=settings,
            pipeline=app.state.pipeline,
            llm=app.state.llm,
        )
    except Exception as e:  # most commonly: the LLM endpoint is down
        logger.exception("ask() failed")
        raise HTTPException(status_code=502, detail=f"pipeline error: {e}") from e
    return _to_response(result)


@app.get("/v1/documents", response_model=DocumentsResponse)
def documents(strategy: str = "recursive") -> DocumentsResponse:
    """What's indexed right now: chunk counts per document."""
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"unknown strategy '{strategy}'")
    index = SparseIndex(app.state.settings, strategy)
    try:
        index.load()  # the BM25 pickle doubles as a cheap chunk catalog
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    counts: dict[str, int] = {}
    for c in index._chunks:  # noqa: SLF001 — read-only peek at our own class
        doc_id = c["metadata"]["doc_id"]
        counts[doc_id] = counts.get(doc_id, 0) + 1
    return DocumentsResponse(
        strategy=strategy,
        total_chunks=index.count(),
        documents=[
            DocumentInfo(doc_id=d, chunk_count=n) for d, n in sorted(counts.items())
        ],
    )


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest_endpoint(files: list[UploadFile], strategy: str = "recursive") -> IngestResponse:
    """Upload documents and ADD them to the index (the CLI rebuilds; this appends).

    Mirrors scripts/ingest.py stage for stage: load -> chunk -> embed ->
    dedup (within the upload) -> write both indexes. Kept synchronous: a
    demo upload is a handful of files, not gigabytes.
    """
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"unknown strategy '{strategy}'")
    settings = app.state.settings
    pipeline: RetrievalPipeline = app.state.pipeline
    docs = []
    with TemporaryDirectory() as tmp:
        for upload in files:
            # UploadFile is a spooled temp file with no real path — loaders
            # dispatch on suffix, so give each one a real file on disk.
            path = Path(tmp) / Path(upload.filename or "upload.txt").name
            path.write_bytes(upload.file.read())
            try:
                docs.append(load_file(path))
            except Exception:
                logger.exception("failed to load upload %s", upload.filename)

    chunks = []
    for doc in docs:
        chunks.extend(chunk(doc, strategy, settings, embedder=pipeline.embedder))
    if not chunks:
        raise HTTPException(status_code=422, detail="no loadable content in the upload")

    embeddings = pipeline.embedder.embed_texts([c.text for c in chunks])
    survivors, survivor_vecs, report = dedup(chunks, embeddings, settings.dedup_threshold)

    VectorIndex(settings, strategy).add(survivors, survivor_vecs)
    # BM25 can't append — rebuild it over old + new chunks (cheap: seconds).
    sparse = SparseIndex(settings, strategy)
    try:
        sparse.load()
        existing = sparse._chunks  # noqa: SLF001
    except FileNotFoundError:
        existing = []
    merged = existing + [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": {
                "doc_id": c.doc_id, "index": c.index, "strategy": c.strategy,
                "section": c.section or "", "char_count": c.char_count,
                "title": str(c.metadata.get("title", "")),
            },
        }
        for c in survivors
    ]
    from rag.index.sparse import tokenize  # local import avoids a cycle at module load
    from rank_bm25 import BM25Okapi

    sparse._chunks = merged  # noqa: SLF001
    sparse._bm25 = BM25Okapi([tokenize(c["text"]) for c in merged])  # noqa: SLF001
    sparse.save()

    return IngestResponse(
        files_received=len(files),
        loaded=len(docs),
        chunked=len(chunks),
        deduped=len(report),
        indexed=len(survivors),
        strategy=strategy,
    )


# In production the built React app is served from here too — ONE container,
# one service, no separate static host. Mounted LAST and only if the build
# exists, so it can't shadow /v1/* or /health, and dev (no dist/) still works.
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="dashboard")
