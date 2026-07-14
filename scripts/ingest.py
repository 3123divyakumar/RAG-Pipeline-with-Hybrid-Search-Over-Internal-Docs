"""Ingest CLI — the ONE pipeline that writes both indexes.

Usage:
    uv run python scripts/ingest.py                       # all three strategies
    uv run python scripts/ingest.py --strategy recursive  # just one
    uv run python scripts/ingest.py --show-dedup          # print what dedup dropped

For each chunking strategy this runs, in order:

    load corpus -> chunk every document -> embed all chunks -> drop
    near-duplicates -> reset + fill the Chroma collection -> build + save
    the BM25 pickle

The dense and sparse indexes are written HERE, from the SAME post-dedup chunk
list, in the SAME run. That single-writer rule is the invariant that keeps
them consistent — if you ever write one without the other, hybrid retrieval
will fuse rankings over two different universes of chunks and quietly degrade.
"""

from __future__ import annotations

import argparse
import logging
import time

from rich.console import Console
from rich.table import Table

from rag.config import get_settings
from rag.embeddings import get_embedder
from rag.index.sparse import SparseIndex
from rag.index.vector import VectorIndex
from rag.ingest.chunkers import STRATEGIES, chunk
from rag.ingest.dedup import dedup
from rag.ingest.loaders import load_corpus

console = Console()


def ingest_strategy(strategy: str, docs, embedder, settings, show_dedup: bool) -> dict:
    """Run the full pipeline for one strategy; returns counts for the summary table."""
    t0 = time.perf_counter()

    # 1. Chunk every document. (The semantic chunker embeds sentences inside
    #    this call — that's why it's noticeably slower than the other two.)
    chunks = []
    for doc in docs:
        chunks.extend(chunk(doc, strategy, settings, embedder=embedder))
    console.print(f"  chunked: {len(chunks)} chunks from {len(docs)} docs")

    # 2. Embed all chunks (documents get NO prefix — see embeddings.py).
    embeddings = embedder.embed_texts([c.text for c in chunks], show_progress=True)

    # 3. Dedup. Printing (or at least counting) the report is part of the
    #    method: inspecting dropped pairs is how the threshold stays honest.
    survivors, survivor_vecs, report = dedup(chunks, embeddings, settings.dedup_threshold)
    console.print(f"  dedup: dropped {len(report)} near-duplicates (threshold {settings.dedup_threshold})")
    if show_dedup:
        for dropped, kept in report[:50]:
            console.print(f"    [dim]dropped[/dim] {dropped}  [dim]≈ kept[/dim] {kept}")
        if len(report) > 50:
            console.print(f"    ... and {len(report) - 50} more")

    # 4. Vector index: reset first so re-ingest replaces instead of appending.
    vindex = VectorIndex(settings, strategy)
    vindex.reset()
    vindex.add(survivors, survivor_vecs)

    # 5. Sparse index over the SAME survivor list.
    sindex = SparseIndex(settings, strategy)
    sindex.build(survivors)
    sindex.save()

    return {
        "strategy": strategy,
        "chunks": len(chunks),
        "deduped": len(report),
        "indexed": len(survivors),
        "seconds": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, help="ingest only this strategy")
    parser.add_argument("--show-dedup", action="store_true", help="print dropped duplicate pairs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    strategies = [args.strategy] if args.strategy else list(STRATEGIES)

    console.print(f"[bold]Loading corpus[/bold] from {settings.corpus_dir} ...")
    docs = load_corpus(settings.corpus_dir)
    if not docs:
        console.print("[red]No documents found — run scripts/fetch_corpus.py first.[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Loading embedder[/bold] ({settings.embedding_model}) ...")
    embedder = get_embedder(settings)

    rows = []
    for strategy in strategies:
        console.print(f"\n[bold cyan]== strategy: {strategy} ==[/bold cyan]")
        rows.append(ingest_strategy(strategy, docs, embedder, settings, args.show_dedup))

    table = Table(title="Ingest summary")
    for col in ("strategy", "chunks", "deduped", "indexed", "seconds"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["strategy"], str(r["chunks"]), str(r["deduped"]), str(r["indexed"]), f"{r['seconds']:.1f}")
    console.print(table)


if __name__ == "__main__":
    main()
