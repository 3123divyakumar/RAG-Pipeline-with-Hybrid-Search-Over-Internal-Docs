"""rag — a RAG pipeline with hybrid search over technical documentation.

Package layout:

    config.py        central settings for every tunable in the system
    embeddings.py    text -> vector
    llm.py           OpenAI-compatible chat client
    ingest/          load, chunk, dedup, index documents
    index/           vector store + BM25 index wrappers
    retrieve/        dense, sparse, RRF fusion, reranking
    generate/        grounded answers with verified citations
    evals/           golden-set evaluation harness
    api/             FastAPI service
"""

__version__ = "0.1.0"
