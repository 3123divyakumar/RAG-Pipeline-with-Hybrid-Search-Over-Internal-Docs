"""Ask the pipeline a question from the terminal — the manual end-to-end check.

Usage:
    uv run python scripts/ask.py "How do I declare a request body in FastAPI?"
    uv run python scripts/ask.py "..." --mode dense --strategy fixed

Needs: an ingested index (scripts/ingest.py) and a running LLM endpoint
(locally: `ollama serve` with qwen2.5:7b pulled — see .env / config.py).
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.generate.generate import ask
from rag.ingest.chunkers import STRATEGIES
from rag.retrieve.pipeline import MODES

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    args = parser.parse_args()

    result = ask(args.question, mode=args.mode, strategy=args.strategy)

    console.print(Panel(result.answer, title="answer" if result.answered else "IDK path"))

    if result.citations:
        table = Table(title="citations")
        table.add_column("[n]")
        table.add_column("verified")
        table.add_column("chunk")
        table.add_column("claim", overflow="fold")
        for c in result.citations:
            table.add_row(
                f"[{c.marker}]",
                {True: "[green]yes[/green]", False: "[red]NO[/red]"}.get(c.verified, "?"),
                c.chunk_id or "[red](nonexistent block)[/red]",
                c.claim[:100],
            )
        console.print(table)

    conf = result.confidence
    console.print(
        f"confidence: [bold]{conf.composite:.2f}[/bold] "
        f"(retrieval {conf.retrieval:.2f} · coverage {conf.citation_coverage:.2f} "
        f"· completeness {conf.completeness:.2f})"
    )
    console.print(
        "timings: " + "  ".join(f"{stage} {ms:.0f}ms" for stage, ms in result.timings_ms.items())
    )


if __name__ == "__main__":
    main()
