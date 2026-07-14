"""Eval runner — run the golden set through the pipeline and score everything.

The one structural decision here: RAW RESULTS ARE CACHED BEFORE ANY SCORING.
Every AskResult is serialized to data/eval_runs/<run_id>/qNNN.json the moment
it exists. Re-scoring (a new metric, a fixed judge prompt) then costs seconds
instead of an hour of re-generation — and you re-score far more often than
you re-generate. `rescore_run()` is that free path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rag.evals.metrics import answer_correctness, citation_accuracy, faithfulness, retrieval_hit
from rag.generate.confidence import ConfidenceReport
from rag.generate.generate import AskResult, Citation, ask
from rag.retrieve.dense import RetrievedChunk

if TYPE_CHECKING:
    from rag.config import Settings
    from rag.llm import LLMClient
    from rag.retrieve.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


@dataclass
class QuestionScore:
    qid: str
    qtype: str
    correctness: float
    faithfulness: float
    hit_at_k: float
    mrr: float
    citation_acc: float | None  # None = no citations to grade (excluded from averages)
    answered: bool
    composite_confidence: float  # kept so reports can check confidence-vs-correctness


@dataclass
class SuiteResult:
    run_id: str
    strategy: str
    mode: str
    scores: list[QuestionScore] = field(default_factory=list)

    # -- aggregations ----------------------------------------------------
    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def aggregate(self, qtype: str | None = None) -> dict[str, float]:
        """Metric means, overall or for one question type.

        Per-type breakdown matters because averages bury failures:
        "faithfulness 0.9 overall" can hide "multi_hop faithfulness 0.4".
        Retrieval metrics skip questions with no golden docs (unanswerables).
        """
        rows = [s for s in self.scores if qtype is None or s.qtype == qtype]
        with_docs = [s for s in rows if s.qtype != "unanswerable"]
        cited = [s.citation_acc for s in rows if s.citation_acc is not None]
        return {
            "n": len(rows),
            "correctness": self._mean([s.correctness for s in rows]),
            "faithfulness": self._mean([s.faithfulness for s in rows]),
            "hit_at_k": self._mean([s.hit_at_k for s in with_docs]),
            "mrr": self._mean([s.mrr for s in with_docs]),
            "citation_accuracy": self._mean(cited),
        }

    def question_types(self) -> list[str]:
        return sorted({s.qtype for s in self.scores})


# ---------------------------------------------------------------------------
# (de)serialization of AskResult — dataclasses <-> JSON for the run cache
# ---------------------------------------------------------------------------


def _result_to_json(result: AskResult) -> dict:
    return dataclasses.asdict(result)  # handles the nested dataclasses recursively


def _result_from_json(payload: dict) -> AskResult:
    return AskResult(
        question=payload["question"],
        answer=payload["answer"],
        citations=[Citation(**c) for c in payload["citations"]],
        chunks=[RetrievedChunk(**c) for c in payload["chunks"]],
        confidence=ConfidenceReport(**payload["confidence"]),
        mode=payload["mode"],
        strategy=payload["strategy"],
        answered=payload["answered"],
        timings_ms=payload.get("timings_ms", {}),
    )


def load_golden(path: Path) -> list[dict]:
    """Read golden.jsonl — one JSON object per line, '#' or blank lines skipped."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            questions.append(json.loads(line))
    return questions


# ---------------------------------------------------------------------------
# the two entry points
# ---------------------------------------------------------------------------


def run_suite(
    strategy: str,
    mode: str,
    settings: Settings,
    pipeline: RetrievalPipeline,
    llm: LLMClient,
    limit: int | None = None,  # small smoke runs while iterating
) -> SuiteResult:
    """Generate + score the whole golden set for one (strategy, mode) combo."""
    golden = load_golden(settings.golden_path)
    if limit:
        golden = golden[:limit]

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{strategy}_{mode}"
    run_dir = settings.eval_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suite = SuiteResult(run_id=run_id, strategy=strategy, mode=mode)

    for i, q in enumerate(golden, 1):
        t0 = time.perf_counter()
        try:
            result = ask(
                q["question"], mode=mode, strategy=strategy,
                settings=settings, pipeline=pipeline, llm=llm,
            )
        except Exception:
            logger.exception("ask() failed on %s — skipping", q["id"])
            continue

        # CACHE FIRST (see module docstring), then score.
        cache_path = run_dir / f"{q['id']}.json"
        cache_path.write_text(
            json.dumps(_result_to_json(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        suite.scores.append(_score_one(q, result, llm))
        logger.info(
            "[%d/%d] %s (%s) scored in %.1fs",
            i, len(golden), q["id"], q["type"], time.perf_counter() - t0,
        )

    # Persist the scores next to the raw results so a run folder is self-contained.
    (run_dir / "scores.json").write_text(
        json.dumps([dataclasses.asdict(s) for s in suite.scores], indent=2), encoding="utf-8"
    )
    return suite


def rescore_run(run_id: str, settings: Settings, llm: LLMClient) -> SuiteResult:
    """Re-score a cached run WITHOUT re-generating — the payoff of cache-first.

    Use after changing a judge prompt or metric: same answers, new scores,
    minutes instead of the full generation hour.
    """
    run_dir = settings.eval_runs_dir / run_id
    _, strategy, mode = run_id.rsplit("_", 2)
    golden_by_id = {q["id"]: q for q in load_golden(settings.golden_path)}
    suite = SuiteResult(run_id=run_id, strategy=strategy, mode=mode)
    for path in sorted(run_dir.glob("q*.json")):
        q = golden_by_id.get(path.stem)
        if q is None:
            logger.warning("cached %s has no golden entry — skipping", path.stem)
            continue
        result = _result_from_json(json.loads(path.read_text(encoding="utf-8")))
        suite.scores.append(_score_one(q, result, llm))
    return suite


def _score_one(q: dict, result: AskResult, llm: LLMClient) -> QuestionScore:
    hit, mrr = retrieval_hit(result, q)
    return QuestionScore(
        qid=q["id"],
        qtype=q["type"],
        correctness=answer_correctness(result, q, llm),
        faithfulness=faithfulness(result, llm),
        hit_at_k=hit,
        mrr=mrr,
        citation_acc=citation_accuracy(result),
        answered=result.answered,
        composite_confidence=result.confidence.composite,
    )


def compare_strategies(
    settings: Settings,
    pipeline: RetrievalPipeline,
    llm: LLMClient,
    mode: str = "hybrid",
    limit: int | None = None,
) -> dict[str, SuiteResult]:
    """run_suite once per chunking strategy, same questions — the numbers the
    README leads with. Slow (3x full suite); start it and walk away."""
    from rag.ingest.chunkers import STRATEGIES

    return {s: run_suite(s, mode, settings, pipeline, llm, limit=limit) for s in STRATEGIES}
