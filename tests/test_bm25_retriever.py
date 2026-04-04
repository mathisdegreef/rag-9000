"""
Unit tests for retrieval/bm25_retriever.py.

No mocking needed — rank_bm25 is a pure-Python library with no network access.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.document_store import DocumentStore
from retrieval.bm25_retriever import BM25Retriever, _default_tokenize


# ---------------------------------------------------------------------------
# Small in-memory corpus
# ---------------------------------------------------------------------------

DOCS = [
    {"id": "doc-0", "text": "machine learning algorithms",    "metadata": {}},
    {"id": "doc-1", "text": "deep learning neural networks",  "metadata": {}},
    {"id": "doc-2", "text": "natural language processing NLP", "metadata": {}},
    {"id": "doc-3", "text": "computer vision image recognition", "metadata": {}},
    {"id": "doc-4", "text": "reinforcement learning rewards",  "metadata": {}},
]


class TestDefaultTokenize(unittest.TestCase):

    def test_lowercase(self) -> None:
        tokens = _default_tokenize("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_non_alphanumeric_split(self) -> None:
        tokens = _default_tokenize("foo.bar-baz")
        self.assertIn("foo", tokens)
        self.assertIn("bar", tokens)
        self.assertIn("baz", tokens)

    def test_all_lowercase_output(self) -> None:
        for token in _default_tokenize("MixedCase INPUT"):
            self.assertEqual(token, token.lower())


class TestBM25Retriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = DocumentStore(DOCS)
        cls.retriever = BM25Retriever(cls.store)

    def test_retrieve_max_k(self) -> None:
        """retrieve(q, k=2) returns at most 2 results."""
        results = self.retriever.retrieve("machine learning", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_scores_descending(self) -> None:
        """Returned results are ordered by descending BM25 score."""
        results = self.retriever.retrieve("learning", top_k=5)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_keyword_match_ranks_first(self) -> None:
        """Doc containing the exact query terms should be the top result."""
        results = self.retriever.retrieve("computer vision", top_k=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].doc_id, "doc-3")

    def test_no_match_returns_empty(self) -> None:
        """A query with zero BM25 scores for all docs returns an empty list."""
        results = self.retriever.retrieve("xyzzy frobnicator", top_k=5)
        self.assertEqual(results, [])

    def test_result_ids_come_from_store(self) -> None:
        """All result doc_ids are valid IDs in the store."""
        valid_ids = set(self.store.ids)
        results = self.retriever.retrieve("learning", top_k=5)
        for r in results:
            self.assertIn(r.doc_id, valid_ids)

    def test_custom_tokenizer(self) -> None:
        """A custom tokenizer is used instead of the default."""
        call_log = []

        def tracking_tokenizer(text: str) -> list[str]:
            call_log.append(text)
            return text.lower().split()

        BM25Retriever(self.store, tokenizer=tracking_tokenizer)
        # The tokenizer should have been called once per document during init
        self.assertEqual(len(call_log), len(DOCS))

    def test_batch_retrieve_returns_one_list_per_query(self) -> None:
        """batch_retrieve(['q1', 'q2']) → list of 2 result lists."""
        results = self.retriever.batch_retrieve(["machine", "vision"], top_k=3)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(results[1], list)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
