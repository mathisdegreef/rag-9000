"""
Unittests for set_up_faiss_index.py and the FAISS index/DocumentStore integration.

All tests are self-contained: no network calls, no model downloads.
A tiny synthetic FAISS index is built in a temporary directory.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Make sure the project root is on sys.path regardless of how pytest is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import faiss
except ImportError as e:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu") from e

from data.document_store import DocumentStore
from set_up_faiss_index import HFEmbeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4
N_DOCS = 5


def _make_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Return *n* random L2-normalised float32 vectors of size *dim*."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_temp_index(tmp_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
    Write index.faiss, doc_ids.json, and documents.json to *tmp_dir*.
    Returns (vectors, doc_ids).
    """
    vectors = _make_unit_vectors(N_DOCS, DIM)
    doc_ids = [f"doc-{i}" for i in range(N_DOCS)]

    # FAISS index
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    faiss.write_index(index, str(tmp_dir / "index.faiss"))

    # doc_ids.json
    (tmp_dir / "doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")

    # documents.json (minimal schema expected by DocumentStore)
    docs = [
        {"id": doc_ids[i], "text": f"Sample text for document {i}.", "metadata": {}}
        for i in range(N_DOCS)
    ]
    (tmp_dir / "documents.json").write_text(json.dumps(docs), encoding="utf-8")

    return vectors, doc_ids


# ---------------------------------------------------------------------------
# TestFaissIndex — tests FAISS index read/search behaviour
# ---------------------------------------------------------------------------

class TestFaissIndex(unittest.TestCase):
    """Tests that verify FAISS index loading and nearest-neighbour retrieval."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_dir = Path(cls._tmp.name)
        cls.vectors, cls.doc_ids = _build_temp_index(cls.tmp_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_index_loads(self) -> None:
        """FAISS index can be read from disk and has the correct number of vectors."""
        index = faiss.read_index(str(self.tmp_dir / "index.faiss"))
        self.assertEqual(index.ntotal, N_DOCS)

    def test_index_dimension(self) -> None:
        """FAISS index has the correct embedding dimension."""
        index = faiss.read_index(str(self.tmp_dir / "index.faiss"))
        self.assertEqual(index.d, DIM)

    def test_fetch_by_position(self) -> None:
        """
        Querying with a vector identical to doc-2 must return doc-2 as the
        top result (inner-product search on L2-normalised vectors → score = 1.0).
        """
        index = faiss.read_index(str(self.tmp_dir / "index.faiss"))
        doc_ids = json.loads((self.tmp_dir / "doc_ids.json").read_text())

        query = self.vectors[2:3]  # shape (1, DIM)
        scores, indices = index.search(query, k=1)

        top_idx = int(indices[0][0])
        top_score = float(scores[0][0])

        self.assertEqual(doc_ids[top_idx], "doc-2")
        self.assertAlmostEqual(top_score, 1.0, places=5)

    def test_top_k_returns_k_results(self) -> None:
        """Searching for top-3 results actually returns 3 results."""
        index = faiss.read_index(str(self.tmp_dir / "index.faiss"))
        query = self.vectors[0:1]
        scores, indices = index.search(query, k=3)
        self.assertEqual(indices.shape, (1, 3))
        self.assertTrue(np.all(indices[0] != -1))

    def test_doc_ids_match(self) -> None:
        """doc_ids.json length equals index.ntotal."""
        index = faiss.read_index(str(self.tmp_dir / "index.faiss"))
        doc_ids = json.loads((self.tmp_dir / "doc_ids.json").read_text())
        self.assertEqual(len(doc_ids), index.ntotal)

    def test_documentstore_integration(self) -> None:
        """
        DocumentStore loaded from documents.json aligns with doc_ids.json:
        position 2 must map to 'doc-2'.
        """
        store = DocumentStore.from_json(self.tmp_dir / "documents.json")
        self.assertEqual(store.idx_to_id(2), "doc-2")
        self.assertEqual(len(store), N_DOCS)

    def test_documentstore_text_retrieval(self) -> None:
        """DocumentStore.get_by_id returns the correct document text."""
        store = DocumentStore.from_json(self.tmp_dir / "documents.json")
        doc = store.get_by_id("doc-0")
        self.assertEqual(doc["id"], "doc-0")
        self.assertIn("Sample text for document 0", doc["text"])

    def test_full_round_trip(self) -> None:
        """
        End-to-end: query the FAISS index, map indices to IDs via doc_ids.json,
        fetch documents from DocumentStore — all must be consistent.
        """
        index = faiss.read_index(str(self.tmp_dir / "index.faiss"))
        doc_ids = json.loads((self.tmp_dir / "doc_ids.json").read_text())
        store = DocumentStore.from_json(self.tmp_dir / "documents.json")

        query = self.vectors[3:4]
        _, indices = index.search(query, k=N_DOCS)

        for idx in indices[0]:
            self.assertNotEqual(idx, -1)
            fetched_id = doc_ids[int(idx)]
            doc = store.get_by_id(fetched_id)
            self.assertEqual(doc["id"], fetched_id)


# ---------------------------------------------------------------------------
# TestHFEmbeddingsInterface — tests HFEmbeddings without loading a real model
# ---------------------------------------------------------------------------

class TestHFEmbeddingsInterface(unittest.TestCase):
    """
    Tests HFEmbeddings behaviour using mocked SentenceTransformer.
    No network access or model download required.
    """

    def test_class_is_importable(self) -> None:
        """HFEmbeddings can be imported without errors."""
        from set_up_faiss_index import HFEmbeddings  # noqa: F401  (re-import to be explicit)
        self.assertTrue(True)

    def test_dim_property(self) -> None:
        """HFEmbeddings.dim delegates to the underlying model."""
        with patch("set_up_faiss_index.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            MockST.return_value = mock_model

            embedder = HFEmbeddings(model_name="mock-model")
            self.assertEqual(embedder.dim, 1024)

    def test_encode_documents_no_prompt(self) -> None:
        """encode_documents must NOT pass a prompt_name to model.encode()."""
        with patch("set_up_faiss_index.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            fake_embeddings = np.ones((2, 4), dtype=np.float32)
            mock_model.encode.return_value = fake_embeddings
            MockST.return_value = mock_model

            embedder = HFEmbeddings(model_name="mock-model")
            embedder.encode_documents(["text A", "text B"], batch_size=2, show_progress=False)

            call_kwargs = mock_model.encode.call_args.kwargs
            self.assertNotIn("prompt_name", call_kwargs,
                             "encode_documents must not pass prompt_name")
            self.assertTrue(call_kwargs.get("normalize_embeddings"),
                            "encode_documents must normalise embeddings")

    def test_encode_queries_uses_prompt(self) -> None:
        """encode_queries must pass prompt_name='web_search_query' to model.encode()."""
        with patch("set_up_faiss_index.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            fake_embeddings = np.ones((1, 4), dtype=np.float32)
            mock_model.encode.return_value = fake_embeddings
            MockST.return_value = mock_model

            embedder = HFEmbeddings(model_name="mock-model")
            embedder.encode_queries(["what is RAG?"], batch_size=1, show_progress=False)

            call_kwargs = mock_model.encode.call_args.kwargs
            self.assertEqual(call_kwargs.get("prompt_name"), "web_search_query",
                             "encode_queries must use prompt_name='web_search_query'")
            self.assertTrue(call_kwargs.get("normalize_embeddings"),
                            "encode_queries must normalise embeddings")

    def test_encode_documents_returns_float32(self) -> None:
        """encode_documents output is always float32 regardless of model dtype."""
        with patch("set_up_faiss_index.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            # Simulate a model returning float16
            mock_model.encode.return_value = np.ones((1, 4), dtype=np.float16)
            MockST.return_value = mock_model

            embedder = HFEmbeddings(model_name="mock-model")
            result = embedder.encode_documents(["hello"], show_progress=False)
            self.assertEqual(result.dtype, np.float32)

    def test_encode_queries_returns_float32(self) -> None:
        """encode_queries output is always float32 regardless of model dtype."""
        with patch("set_up_faiss_index.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            mock_model.encode.return_value = np.ones((1, 4), dtype=np.float16)
            MockST.return_value = mock_model

            embedder = HFEmbeddings(model_name="mock-model")
            result = embedder.encode_queries(["hello"], show_progress=False)
            self.assertEqual(result.dtype, np.float32)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
