"""
Unit tests for retrieval/bi_encoder.py.

SentenceTransformer is mocked — no model download, no network.
A tiny synthetic FAISS index is built in a tempfile.TemporaryDirectory.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import faiss
except ImportError as e:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu") from e

from data.document_store import DocumentStore
from retrieval.bi_encoder import BiEncoderRetriever


# ---------------------------------------------------------------------------
# Synthetic index helpers (reused from test_faiss_index.py pattern)
# ---------------------------------------------------------------------------

DIM = 4
N_DOCS = 5


def _make_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_temp_index(tmp_dir: Path) -> tuple[np.ndarray, list[str]]:
    vectors = _make_unit_vectors(N_DOCS, DIM)
    doc_ids = [f"doc-{i}" for i in range(N_DOCS)]

    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    faiss.write_index(index, str(tmp_dir / "index.faiss"))

    docs = [
        {"id": doc_ids[i], "text": f"Text {i}.", "metadata": {}}
        for i in range(N_DOCS)
    ]
    (tmp_dir / "documents.json").write_text(json.dumps(docs), encoding="utf-8")

    return vectors, doc_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBiEncoderRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_dir = Path(cls._tmp.name)
        cls.vectors, cls.doc_ids = _build_temp_index(cls.tmp_dir)
        cls.store = DocumentStore.from_json(cls.tmp_dir / "documents.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _make_retriever(self, query_vec: np.ndarray | None = None, **kwargs) -> BiEncoderRetriever:
        """
        Build a BiEncoderRetriever with a mocked SentenceTransformer.
        The mock's encode() returns query_vec (default: vectors[0]).
        """
        if query_vec is None:
            query_vec = self.vectors[0:1]

        with patch("retrieval.bi_encoder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            # encode() always returns the provided vector
            mock_model.encode.return_value = query_vec
            MockST.return_value = mock_model

            retriever = BiEncoderRetriever(
                store=self.store,
                index_path=str(self.tmp_dir / "index.faiss"),
                model_name="mock-model",
                **kwargs,
            )
            # Keep mock alive on the retriever for inspection
            retriever.model = mock_model
        return retriever

    # ------------------------------------------------------------------

    def test_retrieve_top_k(self) -> None:
        """retrieve(q, k=3) returns exactly 3 results."""
        retriever = self._make_retriever()
        results = retriever.retrieve("anything", top_k=3)
        self.assertEqual(len(results), 3)

    def test_scores_descending(self) -> None:
        """Scores must be in non-increasing order."""
        retriever = self._make_retriever()
        results = retriever.retrieve("anything", top_k=N_DOCS)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_exact_match_top(self) -> None:
        """Query vector identical to doc-2 → doc-2 is the top result."""
        retriever = self._make_retriever(query_vec=self.vectors[2:3])
        results = retriever.retrieve("anything", top_k=1)
        self.assertEqual(results[0].doc_id, "doc-2")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

    def test_l2_index_negates_score(self) -> None:
        """With is_l2_index=True, all returned scores must be ≤ 0."""
        # Build an L2 index for this test
        l2_index = faiss.IndexFlatL2(DIM)
        l2_index.add(self.vectors)
        l2_path = str(self.tmp_dir / "index_l2.faiss")
        faiss.write_index(l2_index, l2_path)

        with patch("retrieval.bi_encoder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = self.vectors[0:1]
            MockST.return_value = mock_model

            retriever = BiEncoderRetriever(
                store=self.store,
                index_path=l2_path,
                model_name="mock",
                is_l2_index=True,
            )
            retriever.model = mock_model

        results = retriever.retrieve("anything", top_k=3)
        for r in results:
            self.assertLessEqual(r.score, 0.0)

    def test_batch_retrieve_shape(self) -> None:
        """batch_retrieve(['q1','q2'], k=2) → list of 2 result lists."""
        with patch("retrieval.bi_encoder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            # batch encode returns (n_queries, DIM)
            mock_model.encode.return_value = self.vectors[:2]
            MockST.return_value = mock_model

            retriever = BiEncoderRetriever(
                store=self.store,
                index_path=str(self.tmp_dir / "index.faiss"),
                model_name="mock",
            )
            retriever.model = mock_model

        results = retriever.batch_retrieve(["q1", "q2"], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(results[1], list)

    def test_batch_retrieve_consistent_with_retrieve(self) -> None:
        """Single-element batch must return the same top doc_id as retrieve()."""
        query_vec = self.vectors[3:4]

        with patch("retrieval.bi_encoder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = query_vec
            MockST.return_value = mock_model

            retriever = BiEncoderRetriever(
                store=self.store,
                index_path=str(self.tmp_dir / "index.faiss"),
                model_name="mock",
            )
            retriever.model = mock_model

        single_result = retriever.retrieve("anything", top_k=1)

        # For batch_retrieve we need encode to return (1, DIM) array
        retriever.model.encode.return_value = query_vec
        batch_result = retriever.batch_retrieve(["anything"], top_k=1)

        self.assertEqual(single_result[0].doc_id, batch_result[0][0].doc_id)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
