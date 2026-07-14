"""Central configuration for the whole pipeline.

How it works
------------
`pydantic-settings` builds a `Settings` object from three sources, in priority order:
  1. real environment variables (e.g. set by Railway in production)
  2. the `.env` file in the project root (local development)
  3. the defaults written below

Every tunable in the system (chunk sizes, top-k values, fusion weights, model
names) lives here so that retrieval and eval experiments are just config changes,
and so the deployed instance can differ from local dev without code changes.

Usage anywhere in the codebase:

    from rag.config import get_settings
    settings = get_settings()
    settings.rrf_k  # -> 60
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (any OpenAI-compatible endpoint: Ollama locally, Groq deployed) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores the key but the client requires one
    llm_model: str = "qwen2.5:7b"
    judge_model: str = "qwen2.5:7b"

    # --- Embeddings (local sentence-transformers model) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models want queries (not documents) prefixed with this instruction:
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Chunking defaults (the three strategies are compared empirically) ---
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150
    dedup_threshold: float = 0.95  # cosine similarity above which a chunk is a near-duplicate

    # --- Retrieval ---
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60  # the constant in 1 / (k + rank)
    dense_weight: float = 0.7  # sparse gets (1 - dense_weight)
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20  # candidates sent to the reranker
    final_top_k: int = 5  # chunks that reach the LLM

    # --- Generation & confidence ---
    answer_max_tokens: int = 1024
    confidence_idk_threshold: float = 0.35  # below this, return the "I don't know" response

    # --- Paths ---
    data_dir: Path = Path("data")
    chroma_dir: Path = Path(".chroma")

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def golden_path(self) -> Path:
        return self.data_dir / "golden" / "golden.jsonl"

    @property
    def eval_runs_dir(self) -> Path:
        return self.data_dir / "eval_runs"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor — the .env file is read once per process."""
    return Settings()
