"""Eval reports — turn suite results into the markdown tables the README quotes.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.evals.runner import SuiteResult

_METRICS = ("correctness", "faithfulness", "hit_at_k", "mrr", "citation_accuracy")


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def render_suite_report(result: SuiteResult) -> str:
    """One run: overall table, per-question-type breakdown, and the worst-5
    list — which is where every improvement idea comes from. Each worst-5 row
    links to the cached AskResult file so the failure can be read in full."""
    lines = [
        f"# Eval report — `{result.run_id}`",
        "",
        f"strategy: **{result.strategy}** · mode: **{result.mode}** · "
        f"questions: **{len(result.scores)}**",
        "",
        "## Overall",
        "",
        "| metric | score |",
        "|---|---|",
    ]
    overall = result.aggregate()
    for m in _METRICS:
        lines.append(f"| {m} | {_fmt(overall[m])} |")

    lines += ["", "## By question type", "", "| type | n | " + " | ".join(_METRICS) + " |",
              "|---|---|" + "---|" * len(_METRICS)]
    for qtype in result.question_types():
        agg = result.aggregate(qtype)
        lines.append(
            f"| {qtype} | {agg['n']} | " + " | ".join(_fmt(agg[m]) for m in _METRICS) + " |"
        )

    # Worst 5 by correctness, faithfulness as tiebreaker.
    worst = sorted(result.scores, key=lambda s: (s.correctness, s.faithfulness))[:5]
    lines += ["", "## Worst 5 questions", "",
              "| q | type | correctness | faithfulness | hit@k | raw result |",
              "|---|---|---|---|---|---|"]
    for s in worst:
        lines.append(
            f"| {s.qid} | {s.qtype} | {_fmt(s.correctness)} | {_fmt(s.faithfulness)} "
            f"| {_fmt(s.hit_at_k)} | `data/eval_runs/{result.run_id}/{s.qid}.json` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_strategy_comparison(results: dict[str, SuiteResult]) -> str:
    """The headline table: rows = metrics, columns = chunking strategies,
    plus an auto-generated winner-by-metric summary.

    Reading advice baked into the output: deltas under ~0.03 are run-to-run
    noise (LLM outputs vary slightly even at temperature 0) — don't crown a
    winner over one of those."""
    strategies = list(results)
    lines = [
        "# Chunking strategy comparison",
        "",
        f"mode: **{next(iter(results.values())).mode}** · "
        f"questions per strategy: **{len(next(iter(results.values())).scores)}**",
        "",
        "| metric | " + " | ".join(strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    aggs = {s: r.aggregate() for s, r in results.items()}
    winners: list[str] = []
    for m in _METRICS:
        row = [aggs[s][m] for s in strategies]
        lines.append(f"| {m} | " + " | ".join(_fmt(v) for v in row) + " |")
        best = max(range(len(row)), key=lambda i: row[i])
        margin = row[best] - sorted(row)[-2] if len(row) > 1 else 0.0
        note = " (within noise — treat as a tie)" if margin < 0.03 else ""
        winners.append(f"- **{m}**: {strategies[best]} ({_fmt(row[best])}){note}")

    lines += ["", "## Winner by metric", "", *winners, "",
              "_Deltas under ~0.03 are run-to-run noise; don't over-read them._", ""]
    return "\n".join(lines)
