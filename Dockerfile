# RAG pipeline — single-container image: FastAPI + built React dashboard.
#
# Build:  docker build -t rag-pipeline .
# Run:    docker run -p 8000:8000 --env-file .env rag-pipeline
# (or just: docker compose up --build)
#
# Two-stage build. Stage 1 compiles the React dashboard; stage 2 is the
# Python runtime that serves API + static UI from one process. The stages
# keep node_modules (~200MB of build-only weight) out of the final image.

# ---- Stage 1: frontend build -------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build
# Copy manifests first so `npm ci` layer-caches: dependency downloads rerun
# only when package.json changes, not on every source edit.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build          # -> /build/dist

# ---- Stage 2: python runtime -------------------------------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app

# uv inside the image: same resolver as dev, so uv.lock IS the environment —
# "works on my machine" and "works in the container" become the same claim.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Layer caching again: lockfile first, code later.
COPY pyproject.toml uv.lock ./
# --no-dev: pytest/ruff have no business in production.
# The CPU-only torch that sentence-transformers pulls is the bulk of the image.
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev   # now install the project itself (fast, cached deps)

# The built dashboard lands where api/main.py looks for it (frontend/dist).
COPY --from=frontend /build/dist frontend/dist

# Pre-download the two ML models INTO the image so cold starts don't spend
# minutes (and flaky network calls) pulling from HuggingFace at runtime.
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# The indexes travel with the image (built by `scripts/ingest.py` before
# `docker build`). For a demo corpus this is simpler and more reproducible
# than a volume + seed-on-boot dance; at real scale you'd mount a volume.
COPY .chroma/ .chroma/
COPY data/bm25_*.pkl data/

EXPOSE 8000
# Railway injects $PORT; default to 8000 for local docker runs.
CMD ["sh", "-c", "uv run uvicorn rag.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
