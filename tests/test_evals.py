"""Eval-layer tests — the judge-free parts (bookkeeping metrics, aggregation,
serialization round-trip, report rendering). Judge-based metrics need a live
LLM and are exercised by scripts/run_eval.py, not unit tests."""

import json

from rag.evals.metrics import citation_accuracy, retrieval_hit
from rag.evals.runner import QuestionScore, SuiteResult, _result_from_json, _result_to_json
from rag.evals.report import render_strategy_comparison, render_suite_report
from rag.generate.confidence import ConfidenceReport
from rag.generate.generate import AskResult, Citation
from rag.retrieve.dense import RetrievedChunk


def make_result(doc_ids: list[str], answered: bool = True) -> AskResult:
    chunks = [
        RetrievedChunk(
            chunk_id=f"{d}::recursive::0",
            text=f"text from {d}",
            score=5.0,
            rank=i + 1,
            source="reranked",
            metadata={"doc_id": d},
        )
        for i, d in enumerate(doc_ids)
    ]
    return AskResult(
        question="q?",
        answer="answer [1].",
        citations=[Citation(marker=1, chunk_id=chunks[0].chunk_id if chunks else "", claim="answer", verified=True)],
        chunks=chunks,
        confidence=ConfidenceReport(retrieval=0.9, citation_coverage=1.0, completeness=1.0, composite=0.96),
        mode="hybrid",
        strategy="recursive",
        answered=answered,
        timings_ms={"retrieve": 10.0},
    )


class TestRetrievalHit:
    def test_hit_at_rank_one(self):
        result = make_result(["fastapi/tutorial/body.md", "other.md"])
        golden = {"golden_doc_ids": ["fastapi/tutorial/body.md"]}
        assert retrieval_hit(result, golden) == (1.0, 1.0)

    def test_hit_at_rank_three_gives_mrr_third(self):
        result = make_result(["a.md", "b.md", "gold.md"])
        hit, mrr = retrieval_hit(result, {"golden_doc_ids": ["gold.md"]})
        assert hit == 1.0 and abs(mrr - 1 / 3) < 1e-9

    def test_miss(self):
        result = make_result(["a.md", "b.md"])
        assert retrieval_hit(result, {"golden_doc_ids": ["gold.md"]}) == (0.0, 0.0)

    def test_no_golden_docs(self):
        result = make_result(["a.md"])
        assert retrieval_hit(result, {"golden_doc_ids": []}) == (0.0, 0.0)


class TestCitationAccuracy:
    def test_no_citations_is_none_not_zero(self):
        result = make_result(["a.md"])
        result.citations = []
        assert citation_accuracy(result) is None

    def test_ratio(self):
        result = make_result(["a.md"])
        result.citations = [
            Citation(marker=1, chunk_id="x", claim="c1", verified=True),
            Citation(marker=2, chunk_id="y", claim="c2", verified=False),
        ]
        assert citation_accuracy(result) == 0.5


class TestSerializationRoundTrip:
    def test_result_survives_json_round_trip(self):
        """The eval cache depends on this: what we save must reload into an
        identical AskResult, or rescore_run() grades different data."""
        original = make_result(["fastapi/tutorial/body.md"])
        payload = json.loads(json.dumps(_result_to_json(original)))
        restored = _result_from_json(payload)
        assert restored == original


def make_suite(strategy: str, scores: list[tuple[str, str, float]]) -> SuiteResult:
    suite = SuiteResult(run_id=f"test_{strategy}_hybrid", strategy=strategy, mode="hybrid")
    for qid, qtype, corr in scores:
        suite.scores.append(
            QuestionScore(
                qid=qid, qtype=qtype, correctness=corr, faithfulness=0.9,
                hit_at_k=1.0, mrr=0.5, citation_acc=0.8, answered=True,
                composite_confidence=0.7,
            )
        )
    return suite


class TestAggregationAndReports:
    def test_per_type_breakdown_differs_from_overall(self):
        suite = make_suite("recursive", [("q1", "lookup", 1.0), ("q2", "multi_hop", 0.0)])
        assert suite.aggregate()["correctness"] == 0.5
        assert suite.aggregate("lookup")["correctness"] == 1.0
        assert suite.aggregate("multi_hop")["correctness"] == 0.0

    def test_unanswerable_excluded_from_retrieval_metrics(self):
        suite = make_suite("recursive", [("q1", "lookup", 1.0), ("q2", "unanswerable", 1.0)])
        assert suite.aggregate()["hit_at_k"] == 1.0  # only the lookup row counts

    def test_suite_report_renders(self):
        suite = make_suite("recursive", [("q1", "lookup", 1.0), ("q2", "lookup", 0.0)])
        report = render_suite_report(suite)
        assert "Worst 5" in report and "q2" in report

    def test_comparison_report_names_a_winner(self):
        results = {
            "fixed": make_suite("fixed", [("q1", "lookup", 0.4)]),
            "recursive": make_suite("recursive", [("q1", "lookup", 0.9)]),
        }
        report = render_strategy_comparison(results)
        assert "recursive" in report and "correctness" in report
