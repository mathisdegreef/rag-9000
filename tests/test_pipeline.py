"""
Unit tests for retrieval/pipeline.py.

- BM25-only configs use a real BM25Retriever (pure Python, no mocking).
- Bi-encoder configs use a mocked SentenceTransformer + a synthetic FAISS index
  written to a tempfile.TemporaryDirectory.
- Reranker tests mock CrossEncoder.
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

from config import PipelineConfig
from data.document_store import DocumentStore
from retrieval.base import BaseReranker, BaseRetriever, RetrievalResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.pipeline import RetrieverPipeline, build_pipeline


# ---------------------------------------------------------------------------
# Shared corpus
# ---------------------------------------------------------------------------

DOCS = [
    {"id": "doc-0", "text": "machine learning algorithms",      "metadata": {}},
    {"id": "doc-1", "text": "deep learning neural networks",    "metadata": {}},
    {"id": "doc-2", "text": "natural language processing NLP",  "metadata": {}},
    {"id": "doc-3", "text": "computer vision image recognition","metadata": {}},
    {"id": "doc-4", "text": "reinforcement learning rewards",   "metadata": {}},
]

DIM = 4
N_DOCS = len(DOCS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_faiss_index(tmp_dir: Path, vectors: np.ndarray) -> str:
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    path = str(tmp_dir / "index.faiss")
    faiss.write_index(index, path)
    return path


class _FakeRetriever(BaseRetriever):
    """Returns a fixed ordered list for every query."""

    def __init__(self, doc_ids: list[str]) -> None:
        self._ids = doc_ids

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(doc_id=d, score=1.0 / (i + 1), rank=i + 1)
            for i, d in enumerate(self._ids[:top_k])
        ]


class _FakeReranker(BaseReranker):
    """Reverses the candidate list (so we can verify it was called)."""

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        reversed_list = list(reversed(candidates))[:top_k]
        return [
            RetrievalResult(doc_id=r.doc_id, score=r.score, rank=i + 1)
            for i, r in enumerate(reversed_list)
        ]


# ---------------------------------------------------------------------------
# TestRetrieverPipeline
# ---------------------------------------------------------------------------

class TestRetrieverPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = DocumentStore(DOCS)
        cls.bm25 = BM25Retriever(cls.store)

    # ------------------------------------------------------------------

    def test_bm25_only_runs(self) -> None:
        """BM25-only config returns non-empty results for a keyword query."""
        config = PipelineConfig(name="bm25_only", retrievers=["bm25"], top_k_final=3)
        pipeline = RetrieverPipeline(config=config, retrievers={"bm25": self.bm25})
        results = pipeline.run("machine learning")
        self.assertGreater(len(results), 0)

    def test_top_k_final_respected(self) -> None:
        """Only top_k_final results are returned."""
        config = PipelineConfig(name="bm25_k2", retrievers=["bm25"], top_k_final=2)
        pipeline = RetrieverPipeline(config=config, retrievers={"bm25": self.bm25})
        results = pipeline.run("learning")
        self.assertLessEqual(len(results), 2)

    def test_hybrid_rrf_merges_two_retrievers(self) -> None:
        """With two fake retrievers the fused result contains docs from both."""
        ret_a = _FakeRetriever(["doc-0", "doc-1"])
        ret_b = _FakeRetriever(["doc-2", "doc-3"])
        config = PipelineConfig(
            name="hybrid",
            retrievers=["bi_encoder", "bm25"],
            fusion="rrf",
            top_k_retrieve=4,
            top_k_final=4,
        )
        pipeline = RetrieverPipeline(
            config=config,
            retrievers={"bi_encoder": ret_a, "bm25": ret_b},
        )
        results = pipeline.run("anything")
        result_ids = {r.doc_id for r in results}
        # Results from both retrievers must be present after fusion
        self.assertIn("doc-0", result_ids)
        self.assertIn("doc-2", result_ids)

    def test_reranker_is_called(self) -> None:
        """When a reranker is configured its output is returned."""
        ret = _FakeRetriever(["doc-0", "doc-1", "doc-2"])
        reranker = _FakeReranker()
        config = PipelineConfig(
            name="reranked",
            retrievers=["bm25"],
            reranker="cross_encoder",
            top_k_retrieve=3,
            top_k_final=3,
        )
        pipeline = RetrieverPipeline(
            config=config,
            retrievers={"bm25": ret},
            reranker=reranker,
        )
        results = pipeline.run("anything")
        # _FakeReranker reverses the list, so doc-2 is first
        self.assertEqual(results[0].doc_id, "doc-2")

    def test_missing_retriever_raises(self) -> None:
        """Config lists 'bi_encoder' but only 'bm25' is supplied → ValueError."""
        config = PipelineConfig(name="dense", retrievers=["bi_encoder"])
        with self.assertRaises(ValueError):
            RetrieverPipeline(
                config=config,
                retrievers={"bm25": self.bm25},
            )

    def test_reranker_required_but_not_supplied_raises(self) -> None:
        """Config requires a reranker but reranker=None → ValueError."""
        config = PipelineConfig(
            name="needs_reranker",
            retrievers=["bm25"],
            reranker="cross_encoder",
        )
        with self.assertRaises(ValueError):
            RetrieverPipeline(
                config=config,
                retrievers={"bm25": self.bm25},
                reranker=None,
            )

    def test_batch_run_sequential_bm25(self) -> None:
        """batch_run on a BM25 pipeline falls back to sequential run()."""
        config = PipelineConfig(name="bm25_batch", retrievers=["bm25"], top_k_final=2)
        pipeline = RetrieverPipeline(config=config, retrievers={"bm25": self.bm25})
        results = pipeline.batch_run(["machine", "vision"])
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(results[1], list)

    def test_rank_fields_start_at_1(self) -> None:
        """Results returned by run() have rank 1, 2, 3, … with no reranker."""
        config = PipelineConfig(name="bm25_ranks", retrievers=["bm25"], top_k_final=3)
        pipeline = RetrieverPipeline(config=config, retrievers={"bm25": self.bm25})
        results = pipeline.run("learning")
        for expected, r in enumerate(results, start=1):
            self.assertEqual(r.rank, expected)


# ---------------------------------------------------------------------------
# TestBuildPipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_dir = Path(cls._tmp.name)
        cls.vectors = _make_unit_vectors(N_DOCS, DIM)
        cls.store = DocumentStore(DOCS)

        # Write documents.json (not needed by build_pipeline, but by DocumentStore)
        docs_path = cls.tmp_dir / "documents.json"
        docs_path.write_text(json.dumps(DOCS), encoding="utf-8")

        # Build FAISS index
        cls.index_path = _build_faiss_index(cls.tmp_dir, cls.vectors)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ------------------------------------------------------------------

    def test_bm25_only_builds_without_faiss(self) -> None:
        """build_pipeline with bm25-only config needs no FAISS index."""
        config = PipelineConfig(name="bm25_only", retrievers=["bm25"])
        pipeline = build_pipeline(config, self.store, faiss_index_path=None)
        self.assertIsInstance(pipeline, RetrieverPipeline)

    def test_bi_encoder_without_index_raises(self) -> None:
        """build_pipeline raises ValueError when bi_encoder needs a FAISS path."""
        config = PipelineConfig(name="dense", retrievers=["bi_encoder"])
        with self.assertRaises(ValueError):
            build_pipeline(config, self.store, faiss_index_path=None)

    def test_bi_encoder_with_index_builds(self) -> None:
        """build_pipeline with bi_encoder config + mocked ST succeeds."""
        config = PipelineConfig(name="dense", retrievers=["bi_encoder"])
        with patch("retrieval.bi_encoder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = self.vectors[0:1]
            MockST.return_value = mock_model

            pipeline = build_pipeline(
                config,
                self.store,
                faiss_index_path=self.index_path,
            )
        self.assertIsInstance(pipeline, RetrieverPipeline)

    def test_reranker_config_builds(self) -> None:
        """build_pipeline attaches a CrossEncoderReranker when configured."""
        config = PipelineConfig(
            name="bm25_reranked",
            retrievers=["bm25"],
            reranker="cross_encoder",
        )
        with patch("retrieval.cross_encoder.CrossEncoder") as MockCE:
            MockCE.return_value = MagicMock()
            pipeline = build_pipeline(config, self.store)

        self.assertIsNotNone(pipeline.reranker)

    def test_batch_run_bi_encoder_uses_batch_retrieve(self) -> None:
        """batch_run on a bi-encoder-only pipeline uses vectorised batch_retrieve."""
        config = PipelineConfig(
            name="dense_batch",
            retrievers=["bi_encoder"],
            top_k_retrieve=3,
            top_k_final=2,
        )
        with patch("retrieval.bi_encoder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            # batch encode: (2, DIM) for 2 queries
            mock_model.encode.return_value = self.vectors[:2]
            MockST.return_value = mock_model

            pipeline = build_pipeline(
                config,
                self.store,
                faiss_index_path=self.index_path,
            )
            pipeline.retrievers["bi_encoder"].model = mock_model

        results = pipeline.batch_run(["q1", "q2"])
        self.assertEqual(len(results), 2)
        for res_list in results:
            self.assertLessEqual(len(res_list), config.top_k_final)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
