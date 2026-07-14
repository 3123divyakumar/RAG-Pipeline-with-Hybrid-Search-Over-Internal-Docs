"""Side-by-side retrieval comparison — SEE why hybrid beats dense-only.

Usage:
    uv run python scripts/compare_retrieval.py "how do I return a custom status code"
    uv run python scripts/compare_retrieval.py "response_model_exclude_none" --strategy fixed

Prints three columns for the same query: dense-only, sparse-only (BM25), and
the full hybrid+rerank pipeline. Try it with an identifier-style query
(`response_model_exclude_none`) and watch dense-only miss what BM25 nails —
that contrast is the whole argument for hybrid retrieval, live.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from rag.config import get_settings
from rag.ingest.chunkers import STRATEGIES
from rag.retrieve.pipeline import RetrievalPipeline

console = Console()


def fmt(results, n: int = 5) -> list[str]:
    lines = []
    for r in results[:n]:
        section = r.metadata.get("section") or ""
        label = f" §{section}" if section else ""
        lines.append(f"{r.rank}. [{r.score:.3f}] {r.metadata.get('doc_id', r.chunk_id)}{label}")
    return lines or ["(no results)"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("-k", type=int, default=5, help="results to show per column")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RetrievalPipeline(settings)

    dense = pipeline.dense_retriever(args.strategy).retrieve(args.query, args.k)
    sparse = pipeline.sparse_retriever(args.strategy).retrieve(args.query, args.k)
    hybrid = pipeline.retrieve(args.query, mode="hybrid", strategy=args.strategy)

    table = Table(title=f"query: {args.query!r}  (strategy: {args.strategy})", show_lines=True)
    table.add_column("dense only (cosine)", overflow="fold")
    table.add_column("sparse only (BM25)", overflow="fold")
    table.add_column("hybrid + rerank (final)", overflow="fold")
    table.add_row(
        "\n".join(fmt(dense, args.k)), "\n".join(fmt(sparse, args.k)), "\n".join(fmt(hybrid, args.k))
    )
    console.print(table)
    console.print(
        "[dim]Scores are NOT comparable across columns (cosine vs BM25 vs rerank logits) — "
        "that's exactly why fusion uses ranks.[/dim]"
    )


if __name__ == "__main__":
    main()
