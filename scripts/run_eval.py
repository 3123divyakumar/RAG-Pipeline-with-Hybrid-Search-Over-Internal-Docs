"""Run the eval suite. The numbers this produces are the project's proof.

Usage:
    uv run python scripts/run_eval.py                          # one suite: recursive/hybrid
    uv run python scripts/run_eval.py --limit 5                # quick smoke run
    uv run python scripts/run_eval.py --mode dense             # the hybrid-vs-dense story
    uv run python scripts/run_eval.py --compare-strategies     # 3 suites, headline table
    uv run python scripts/run_eval.py --rescore <run_id>       # re-score a cached run (no LLM generation)

Needs: ingested indexes + a running LLM endpoint. A full suite makes several
LLM calls per question (answer + judge calls), so on a local 7B model budget
roughly 1-2 minutes per question — kick it off and go do something else.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from rich.console import Console

from rag.config import get_settings
from rag.evals.report import render_strategy_comparison, render_suite_report
from rag.evals.runner import compare_strategies, rescore_run, run_suite
from rag.ingest.chunkers import STRATEGIES
from rag.llm import LLMClient
from rag.retrieve.pipeline import MODES, RetrievalPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument("--compare-strategies", action="store_true")
    parser.add_argument("--rescore", metavar="RUN_ID", help="re-score a cached run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    llm = LLMClient(settings)

    if args.rescore:
        suite = rescore_run(args.rescore, settings, llm)
        report = render_suite_report(suite)
    elif args.compare_strategies:
        pipeline = RetrievalPipeline(settings)
        results = compare_strategies(settings, pipeline, llm, mode=args.mode, limit=args.limit)
        report = render_strategy_comparison(results)
        # Also save each strategy's individual report alongside the comparison.
        for suite in results.values():
            path = settings.eval_runs_dir / f"report_{suite.run_id}.md"
            path.write_text(render_suite_report(suite), encoding="utf-8")
    else:
        pipeline = RetrievalPipeline(settings)
        suite = run_suite(args.strategy, args.mode, settings, pipeline, llm, limit=args.limit)
        report = render_suite_report(suite)

    out_name = (
        f"comparison_{datetime.now():%Y%m%d}.md" if args.compare_strategies
        else f"report_{(args.rescore or suite.run_id)}.md"
    )
    out_path = settings.eval_runs_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    console.print(report)
    console.print(f"\n[green]saved:[/green] {out_path}")


if __name__ == "__main__":
    main()
