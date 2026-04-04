"""
Unit tests for retrieval/rrf.py.

Pure Python — no external dependencies, no mocking.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.base import RetrievalResult
from retrieval.rrf import reciprocal_rank_fusion


def _make_results(doc_ids: list[str], start_rank: int = 1) -> list[RetrievalResult]:
    """Build a ranked list of RetrievalResult from a list of doc IDs."""
    return [
        RetrievalResult(doc_id=d, score=1.0 / (i + 1), rank=i + start_rank)
        for i, d in enumerate(doc_ids)
    ]


class TestReciprocalRankFusion(unittest.TestCase):

    def test_empty_input(self) -> None:
        """No result lists → empty output."""
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_single_list_passthrough(self) -> None:
        """One list in → same doc IDs out, same relative order."""
        lst = _make_results(["A", "B", "C"])
        fused = reciprocal_rank_fusion([lst])
        self.assertEqual([r.doc_id for r in fused], ["A", "B", "C"])

    def test_shared_doc_gets_boosted(self) -> None:
        """
        Doc 'A' appears rank-1 in both lists.
        Doc 'B' appears rank-1 in only one list.
        'A' must score higher than 'B'.
        """
        list1 = _make_results(["A", "B"])
        list2 = _make_results(["A", "C"])
        fused = reciprocal_rank_fusion([list1, list2])
        scores = {r.doc_id: r.score for r in fused}
        self.assertGreater(scores["A"], scores["B"])
        self.assertGreater(scores["A"], scores["C"])

    def test_score_formula(self) -> None:
        """Doc at rank 1 with k=60 → RRF score = 1/61."""
        lst = [RetrievalResult(doc_id="X", score=1.0, rank=1)]
        fused = reciprocal_rank_fusion([lst], k=60)
        self.assertAlmostEqual(fused[0].score, 1.0 / 61, places=8)

    def test_top_k_truncation(self) -> None:
        """top_k=2 returns exactly 2 results."""
        lst = _make_results(["A", "B", "C", "D"])
        fused = reciprocal_rank_fusion([lst], top_k=2)
        self.assertEqual(len(fused), 2)

    def test_rank_field_is_1_based(self) -> None:
        """Returned results have rank = 1, 2, 3, …"""
        lst = _make_results(["A", "B", "C"])
        fused = reciprocal_rank_fusion([lst])
        for expected_rank, result in enumerate(fused, start=1):
            self.assertEqual(result.rank, expected_rank)

    def test_uses_stored_rank_over_position(self) -> None:
        """
        If result.rank=3 (not position 0), the formula uses 3 — not 1.
        Compare two lists: one doc at stored rank=1, another at stored rank=3.
        The rank-1 doc must win.
        """
        high_rank = RetrievalResult(doc_id="TOP", score=0.5, rank=1)
        low_rank  = RetrievalResult(doc_id="BOT", score=0.9, rank=3)
        # Same list — position-0 is BOT but rank=3, position-1 is TOP but rank=1
        fused = reciprocal_rank_fusion([[low_rank, high_rank]], k=60)
        self.assertEqual(fused[0].doc_id, "TOP")

    def test_two_separate_docs_scores(self) -> None:
        """Docs that appear in separate lists keep their individual RRF contributions."""
        list1 = [RetrievalResult(doc_id="A", score=1.0, rank=1)]
        list2 = [RetrievalResult(doc_id="B", score=1.0, rank=1)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        scores = {r.doc_id: r.score for r in fused}
        # Both appear once at rank 1 → equal scores
        self.assertAlmostEqual(scores["A"], scores["B"], places=8)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
