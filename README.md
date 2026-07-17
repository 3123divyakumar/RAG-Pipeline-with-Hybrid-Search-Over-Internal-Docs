# RAG Pipeline with Hybrid Search

A production-grade Retrieval-Augmented Generation system over technical documentation:
**dense vector search + BM25 keyword search fused with Reciprocal Rank Fusion, cross-encoder
reranking, grounded answers with per-claim citation verification, and a 50+ question
evaluation suite** that measures every architectural decision in this repo.

## Results

53-question golden set, recursive chunking in both columns, fully local stack
(qwen2.5:7b as both generator and LLM-judge; run 2026-07-18 — deltas under
~0.03 are run-to-run noise):

| Metric | Hybrid + rerank | Dense-only |
|---|---|---|
| Answer correctness | **0.83** | 0.80 |
| Faithfulness | 0.84 | 0.86 |
| Retrieval hit@5 | 0.92 | 0.92 |
| MRR | **0.84** | 0.81 |
| Citation accuracy | 0.48 | 0.54 |

The 4 unanswerable trap questions scored 1.00 in every configuration — the
IDK gate refused all of them instead of hallucinating. Citation accuracy is
the strictest metric here (every claim independently verified against its
cited chunk by the judge) and the 7B judge is a noisy grader; re-scoring with
a stronger judge is the documented next step.

Chunking strategy comparison (hybrid mode, 53 questions per strategy):

| Metric | fixed | recursive | semantic |
|---|---|---|---|
| Answer correctness | 0.76 | **0.83** | 0.79 |
| Retrieval hit@5 | 0.86 | **0.92** | 0.90 |
| MRR | 0.76 | **0.84** | 0.77 |

Recursive wins or ties every headline metric — structure-aware splitting at
paragraph/heading boundaries beats both blind fixed windows and the fancier
embedding-similarity splitter on this corpus. Full per-question reports:
`uv run python scripts/run_eval.py --compare-strategies` → `data/eval_runs/`.

Everything above reproduces on a laptop with no API keys and no paid services.

## Why this isn't a LangChain quickstart

- **Hybrid retrieval, hand-built.** BM25 catches the exact identifiers
  (`response_model_exclude_none`, error codes, config keys) that embeddings blur;
  RRF fusion and the rerank stage are implemented from the papers, not imported.
- **Citations are verified, not just requested.** Every claim/citation pair goes through
  an LLM-as-judge check; unsupported citations are flagged in the response (and shown
  red in the dashboard).
- **Honest "I don't know."** Below a retrieval-confidence threshold the system returns a
  structured not-found response instead of hallucinating — and skips the LLM call entirely.
- **Decisions are measured.** Three chunking strategies indexed side by side and compared
  on the same golden set. See [DECISIONS.md](DECISIONS.md).
- **Provider-agnostic by design.** Runs 100% free/local (Ollama + sentence-transformers);
  the deployed instance swaps to a hosted 70B LLM (Groq free tier) with only env-var changes.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[docs: md/html/pdf] --> B[loaders] --> C[chunkers<br/>fixed / recursive / semantic]
        C --> D[dedup >0.95 cosine]
        D --> E[(ChromaDB<br/>dense vectors)]
        D --> F[(BM25 index)]
    end
    subgraph Query
        Q[question] --> G[dense top-10]
        Q --> H[BM25 top-10]
        E --> G
        F --> H
        G --> I[RRF fusion 0.7/0.3]
        H --> I
        I --> J[cross-encoder rerank<br/>20 → 5]
        J --> K{retrieval<br/>confidence?}
        K -- low --> L[structured IDK]
        K -- ok --> M[LLM: grounded answer + citations]
        M --> N[citation verification<br/>LLM-as-judge per claim]
        N --> O[answer + confidence report]
    end
```

## Quickstart

Local, fully free (needs [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com)
with `qwen2.5:7b` pulled, and Node for the dashboard):

```bash
git clone <repo> && cd rag-pipeline
cp .env.example .env                      # defaults already point at local Ollama
uv sync
uv run python scripts/fetch_corpus.py     # downloads the doc corpus (~244 files)
uv run python scripts/ingest.py           # chunk + embed + index (all 3 strategies)

# ask from the terminal:
uv run python scripts/ask.py "How do I declare a request body in FastAPI?"

# or run the full service + dashboard:
uv run uvicorn rag.api.main:app --reload  # API on :8000 (docs at /docs)
cd frontend && npm install && npm run dev # dashboard on :5173

# or the whole thing in one container:
docker compose up --build                 # -> http://localhost:8000 (API + built dashboard)
```

Useful CLIs:

```bash
uv run python scripts/compare_retrieval.py "response_model_exclude_none"   # dense vs sparse vs hybrid, side by side
uv run python scripts/run_eval.py --limit 5                                # smoke-test the eval loop
uv run pytest                                                              # unit tests (no LLM needed)
```

## Stack

Python 3.12 · uv · sentence-transformers (BGE-small) · ChromaDB · rank-bm25 ·
cross-encoder reranker · Ollama (local) / Groq (deployed) via one OpenAI-compatible client ·
FastAPI · React + Vite · Docker · Railway

## Repo tour

| Path | What |
|---|---|
| `src/rag/ingest/` | loaders, 3 chunking strategies, near-dup detection |
| `src/rag/index/` | Chroma + BM25 wrappers (one collection per chunking strategy) |
| `src/rag/retrieve/` | dense, sparse, RRF fusion, cross-encoder rerank, pipeline facade |
| `src/rag/generate/` | grounded prompting, ask() orchestrator, citation verification, confidence |
| `src/rag/evals/` | golden-set runner (cache-first), metrics, strategy-comparison reports |
| `src/rag/api/` | FastAPI service (serves the built React dashboard in prod) |
| `frontend/` | React dashboard: citations, chunk inspector, confidence, mode toggle |
| `data/golden/` | hand-written 53-question eval set (committed) |
| `DECISIONS.md` | why every tunable has the value it has |
