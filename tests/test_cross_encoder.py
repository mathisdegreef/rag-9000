"""
Unit tests for retrieval/cross_encoder.py.

CrossEncoder is mocked — no model download, no network.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.document_store import DocumentStore
from retrieval.base import RetrievalResult
from retrieval.cross_encoder import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

DOCS = [
    {"id": "doc-0", "text": "Machine learning basics.",        "metadata": {}},
    {"id": "doc-1", "text": "Deep learning and neural nets.",  "metadata": {}},
    {"id": "doc-2", "text": "Natural language processing.",    "metadata": {}},
]

CANDIDATES = [
    RetrievalResult(doc_id="doc-0", score=0.9, rank=1),
    RetrievalResult(doc_id="doc-1", score=0.7, rank=2),
    RetrievalResult(doc_id="doc-2", score=0.5, rank=3),
]


def _make_reranker(scores: list[float]) -> CrossEncoderReranker:
    """Build a CrossEncoderReranker whose .predict() returns the given scores."""
    store = DocumentStore(DOCS)
    with patch("retrieval.cross_encoder.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores, dtype=np.float32)
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker(store=store, model_name="mock-ce")
        reranker.model = mock_model
    return reranker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker(unittest.TestCase):

    def test_rerank_returns_top_k(self) -> None:
        """3 candidates, top_k=2 → exactly 2 results."""
        reranker = _make_reranker([0.3, 0.9, 0.6])
        results = reranker.rerank("query", CANDIDATES, top_k=2)
        self.assertEqual(len(results), 2)

    def test_rerank_scores_descending(self) -> None:
        """Results are ordered by cross-encoder score, highest first."""
        reranker = _make_reranker([0.3, 0.9, 0.6])
        results = reranker.rerank("query", CANDIDATES, top_k=3)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_rerank_top_result_matches_highest_score(self) -> None:
        """The candidate with the highest mock score is ranked first."""
        # doc-1 gets the highest score (0.9)
        reranker = _make_reranker([0.3, 0.9, 0.6])
        results = reranker.rerank("query", CANDIDATES, top_k=3)
        self.assertEqual(results[0].doc_id, "doc-1")

    def test_predict_called_with_query_text_pairs(self) -> None:
        """predict() must receive (query, doc_text) pairs for every candidate."""
        reranker = _make_reranker([0.1, 0.2, 0.3])
        query = "What is ML?"
        reranker.rerank(query, CANDIDATES, top_k=3)

        expected_pairs = [
            (query, DOCS[0]["text"]),
            (query, DOCS[1]["text"]),
            (query, DOCS[2]["text"]),
        ]
        reranker.model.predict.assert_called_once()
        actual_pairs = reranker.model.predict.call_args[0][0]
        self.assertEqual(actual_pairs, expected_pairs)

    def test_empty_candidates_returns_empty(self) -> None:
        """Empty candidate list → empty result, predict() not called."""
        reranker = _make_reranker([])
        results = reranker.rerank("query", [], top_k=5)
        self.assertEqual(results, [])
        reranker.model.predict.assert_not_called()

    def test_rank_field_is_1_based(self) -> None:
        """Returned results have rank values 1, 2, 3, …"""
        reranker = _make_reranker([0.5, 0.8, 0.2])
        results = reranker.rerank("query", CANDIDATES, top_k=3)
        for expected_rank, r in enumerate(results, start=1):
            self.assertEqual(r.rank, expected_rank)

    def test_score_reflects_cross_encoder_output(self) -> None:
        """result.score equals the raw cross-encoder predict value."""
        scores = [0.11, 0.99, 0.55]
        reranker = _make_reranker(scores)
        results = reranker.rerank("query", CANDIDATES, top_k=3)
        # After sorting descending: 0.99, 0.55, 0.11
        self.assertAlmostEqual(results[0].score, 0.99, places=4)
        self.assertAlmostEqual(results[1].score, 0.55, places=4)
        self.assertAlmostEqual(results[2].score, 0.11, places=4)

    def test_top_k_larger_than_candidates(self) -> None:
        """top_k > len(candidates) → all candidates returned (no error)."""
        reranker = _make_reranker([0.4, 0.7, 0.2])
        results = reranker.rerank("query", CANDIDATES, top_k=100)
        self.assertEqual(len(results), len(CANDIDATES))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
