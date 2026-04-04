"""
Unit tests for evaluation/evaluator.py.

Uses a lightweight _FakePipeline stub — no models, no FAISS, no network.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import Evaluator, EvaluationResult
from evaluation.metrics import MetricsResult, average_metrics
from retrieval.base import RetrievalResult


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------

class _FakePipeline:
    """Returns a hard-coded ordered list of doc IDs for every query."""
    def __init__(self, returned_ids: list[str]) -> None:
        self._ids = returned_ids

    def run(self, query: str) -> list[RetrievalResult]:
        return [
            RetrievalResult(doc_id=doc_id, score=1.0 / (i + 1), rank=i + 1)
            for i, doc_id in enumerate(self._ids)
        ]


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

GROUND_TRUTH = [
    {"query": "What is ML?",  "relevant_ids": ["doc-0"]},
    {"query": "What is NLP?", "relevant_ids": ["doc-1"]},
]

# Pipeline A returns the relevant doc first for both queries
PIPELINE_A = _FakePipeline(["doc-0", "doc-1", "doc-2"])
# Pipeline B misses the relevant docs entirely
PIPELINE_B = _FakePipeline(["doc-9", "doc-8", "doc-7"])


# ---------------------------------------------------------------------------
# TestEvaluator
# ---------------------------------------------------------------------------

class TestEvaluator(unittest.TestCase):

    def _make_evaluator(self, **kwargs) -> Evaluator:
        ev = Evaluator(
            pipelines={"good": PIPELINE_A, "bad": PIPELINE_B},
            k=3,
        )
        ev.set_ground_truth(GROUND_TRUTH)
        return ev

    def test_load_ground_truth_from_file(self) -> None:
        """load_ground_truth reads a JSON file and stores the queries."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.json"
            path.write_text(json.dumps(GROUND_TRUTH), encoding="utf-8")

            ev = Evaluator(pipelines={"p": PIPELINE_A}, k=3)
            ev.load_ground_truth(path)
            result = ev.run(verbose=False)
            self.assertEqual(len(result.pipeline_results["p"]), 2)

    def test_set_ground_truth(self) -> None:
        """set_ground_truth accepts an in-memory list."""
        ev = Evaluator(pipelines={"p": PIPELINE_A}, k=3)
        ev.set_ground_truth(GROUND_TRUTH)
        result = ev.run(verbose=False)
        self.assertIn("p", result.aggregate)

    def test_run_returns_all_pipeline_names(self) -> None:
        """All pipeline names appear in result.aggregate."""
        ev = self._make_evaluator()
        result = ev.run(verbose=False)
        self.assertIn("good", result.aggregate)
        self.assertIn("bad",  result.aggregate)

    def test_run_metrics_in_valid_range(self) -> None:
        """All averaged metrics are in [0.0, 1.0]."""
        ev = self._make_evaluator()
        result = ev.run(verbose=False)
        for name, agg in result.aggregate.items():
            with self.subTest(pipeline=name):
                self.assertGreaterEqual(agg.recall_at_k,    0.0)
                self.assertLessEqual(   agg.recall_at_k,    1.0)
                self.assertGreaterEqual(agg.ndcg_at_k,      0.0)
                self.assertLessEqual(   agg.ndcg_at_k,      1.0)
                self.assertGreaterEqual(agg.hit_rate_at_k,  0.0)
                self.assertLessEqual(   agg.hit_rate_at_k,  1.0)

    def test_run_no_ground_truth_raises(self) -> None:
        """run() raises RuntimeError when no ground truth is loaded."""
        ev = Evaluator(pipelines={"p": PIPELINE_A}, k=3)
        with self.assertRaises(RuntimeError):
            ev.run(verbose=False)

    def test_run_unknown_pipeline_raises(self) -> None:
        """Passing an unknown pipeline name raises ValueError."""
        ev = self._make_evaluator()
        with self.assertRaises(ValueError):
            ev.run(pipeline_names=["nonexistent"], verbose=False)

    def test_good_pipeline_beats_bad(self) -> None:
        """The pipeline that returns relevant docs first has higher NDCG."""
        ev = self._make_evaluator()
        result = ev.run(verbose=False)
        self.assertGreater(
            result.aggregate["good"].ndcg_at_k,
            result.aggregate["bad"].ndcg_at_k,
        )


# ---------------------------------------------------------------------------
# TestEvaluationResult
# ---------------------------------------------------------------------------

class TestEvaluationResult(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ev = Evaluator(
            pipelines={"alpha": PIPELINE_A, "beta": PIPELINE_B},
            k=3,
        )
        ev.set_ground_truth(GROUND_TRUTH)
        cls.result: EvaluationResult = ev.run(verbose=False)

    def test_best_pipeline_returns_name(self) -> None:
        """best_pipeline() returns the name of the highest-NDCG pipeline."""
        best = self.result.best_pipeline("ndcg_at_k")
        self.assertEqual(best, "alpha")

    def test_summary_table_contains_names(self) -> None:
        """summary_table() includes all pipeline names in its output."""
        table = self.result.summary_table()
        self.assertIn("alpha", table)
        self.assertIn("beta",  table)

    def test_save_csv(self) -> None:
        """save_csv writes a file with the expected header columns."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            self.result.save_csv(path)
            self.assertTrue(path.exists())
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            self.assertIn("pipeline", header)
            self.assertIn("query",    header)
            self.assertIn("ndcg",     header)

    def test_save_csv_row_count(self) -> None:
        """save_csv writes one row per query × pipeline."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            self.result.save_csv(path)
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            # header + 2 pipelines × 2 queries = 5 rows
            self.assertEqual(len(rows), 1 + 2 * len(GROUND_TRUTH))

    def test_save_json(self) -> None:
        """save_json writes a JSON file with pipeline keys and metric fields."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            self.result.save_json(path)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("alpha",    data)
        self.assertIn("beta",     data)
        self.assertIn("ndcg_at_k", data["alpha"])
        self.assertIn("recall_at_k", data["alpha"])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
