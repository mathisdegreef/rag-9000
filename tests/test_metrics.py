"""
Unit tests for evaluation/metrics.py.

Pure Python — no external dependencies, no mocking.
"""

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MetricsResult, compute_metrics, average_metrics


class TestComputeMetrics(unittest.TestCase):

    def test_perfect_single_relevant(self) -> None:
        """Single relevant doc retrieved at rank 1 → all metrics = 1.0."""
        result = compute_metrics(["A"], ["A"], k=5)
        self.assertAlmostEqual(result.recall_at_k,    1.0)
        self.assertAlmostEqual(result.mrr,            1.0)
        self.assertAlmostEqual(result.ndcg_at_k,      1.0)
        self.assertAlmostEqual(result.hit_rate_at_k,  1.0)

    def test_no_relevant_docs(self) -> None:
        """Empty relevant set → all metrics are 0.0."""
        result = compute_metrics(["A", "B"], [], k=5)
        self.assertEqual(result.recall_at_k,   0.0)
        self.assertEqual(result.precision_at_k, 0.0)
        self.assertEqual(result.mrr,            0.0)
        self.assertEqual(result.ndcg_at_k,      0.0)
        self.assertEqual(result.map_at_k,       0.0)
        self.assertEqual(result.hit_rate_at_k,  0.0)

    def test_partial_recall(self) -> None:
        """2 relevant docs, only 1 found in top K → recall = 0.5."""
        result = compute_metrics(["A", "C"], ["A", "B"], k=5)
        self.assertAlmostEqual(result.recall_at_k, 0.5)

    def test_precision(self) -> None:
        """3 retrieved, 1 relevant → precision = 1/3."""
        result = compute_metrics(["A", "X", "Y"], ["A"], k=3)
        self.assertAlmostEqual(result.precision_at_k, 1 / 3, places=5)

    def test_mrr_second_rank(self) -> None:
        """First relevant doc appears at rank 2 → MRR = 0.5."""
        result = compute_metrics(["X", "A"], ["A"], k=5)
        self.assertAlmostEqual(result.mrr, 0.5)

    def test_mrr_no_hit(self) -> None:
        """No relevant doc in retrieved list → MRR = 0.0."""
        result = compute_metrics(["X", "Y"], ["A"], k=5)
        self.assertAlmostEqual(result.mrr, 0.0)

    def test_ndcg_perfect(self) -> None:
        """Relevant doc at rank 1 → NDCG = 1.0 (DCG == IDCG)."""
        result = compute_metrics(["A"], ["A"], k=3)
        self.assertAlmostEqual(result.ndcg_at_k, 1.0)

    def test_ndcg_no_hits(self) -> None:
        """No relevant doc retrieved → NDCG = 0.0."""
        result = compute_metrics(["X", "Y"], ["A"], k=5)
        self.assertAlmostEqual(result.ndcg_at_k, 0.0)

    def test_ndcg_rank2(self) -> None:
        """Single relevant doc at rank 2 — verify formula manually."""
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        result = compute_metrics(["X", "A"], ["A"], k=5)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(result.ndcg_at_k, expected, places=5)

    def test_map_single_relevant_rank1(self) -> None:
        """Single relevant doc at rank 1 → MAP = 1.0."""
        result = compute_metrics(["A", "X"], ["A"], k=5)
        self.assertAlmostEqual(result.map_at_k, 1.0)

    def test_map_single_relevant_rank2(self) -> None:
        """Single relevant doc at rank 2 → MAP = 0.5."""
        result = compute_metrics(["X", "A"], ["A"], k=5)
        self.assertAlmostEqual(result.map_at_k, 0.5)

    def test_hit_rate_zero(self) -> None:
        """No relevant doc in top K → hit_rate = 0.0."""
        result = compute_metrics(["X", "Y"], ["A"], k=5)
        self.assertEqual(result.hit_rate_at_k, 0.0)

    def test_hit_rate_one(self) -> None:
        """Any relevant doc in top K → hit_rate = 1.0."""
        result = compute_metrics(["X", "A", "Y"], ["A"], k=5)
        self.assertEqual(result.hit_rate_at_k, 1.0)

    def test_k_cutoff_respected(self) -> None:
        """Only the first K retrieved docs are considered."""
        # Relevant doc is at position 4 (0-indexed), k=3 → should not be found
        result = compute_metrics(["X", "Y", "Z", "A"], ["A"], k=3)
        self.assertEqual(result.hit_rate_at_k, 0.0)
        self.assertEqual(result.recall_at_k,   0.0)


class TestAverageMetrics(unittest.TestCase):

    def test_average_two_results(self) -> None:
        r1 = MetricsResult(recall_at_k=0.8, precision_at_k=0.4, mrr=0.6,
                           ndcg_at_k=0.7, map_at_k=0.5, hit_rate_at_k=1.0, k=10)
        r2 = MetricsResult(recall_at_k=0.6, precision_at_k=0.2, mrr=0.4,
                           ndcg_at_k=0.5, map_at_k=0.3, hit_rate_at_k=0.0, k=10)
        avg = average_metrics([r1, r2])
        self.assertAlmostEqual(avg.recall_at_k,    0.7)
        self.assertAlmostEqual(avg.precision_at_k, 0.3)
        self.assertAlmostEqual(avg.mrr,             0.5)
        self.assertAlmostEqual(avg.ndcg_at_k,      0.6)
        self.assertAlmostEqual(avg.map_at_k,       0.4)
        self.assertAlmostEqual(avg.hit_rate_at_k,  0.5)

    def test_empty_list_returns_default(self) -> None:
        avg = average_metrics([])
        self.assertEqual(avg.recall_at_k,    0.0)
        self.assertEqual(avg.precision_at_k, 0.0)
        self.assertEqual(avg.mrr,            0.0)
        self.assertEqual(avg.ndcg_at_k,      0.0)
        self.assertEqual(avg.map_at_k,       0.0)
        self.assertEqual(avg.hit_rate_at_k,  0.0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
