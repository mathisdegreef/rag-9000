"""
Unit tests for data/load_hf_dataset.py.

All HuggingFace I/O is mocked — no network access, no model downloads.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_hf_dataset import _iter_batches, load_and_convert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_dataset(rows: list[dict]):
    """Return a MagicMock that mimics a HuggingFace Dataset."""
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=len(rows))
    mock_ds.to_list.return_value = rows
    # Support .shuffle().select() chaining
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds
    return mock_ds


SAMPLE_ROWS = [
    {"context": "Context about AI.",    "question": "What is AI?",  "answer": "Artificial Intelligence."},
    {"context": "Context about ML.",    "question": "What is ML?",  "answer": "Machine Learning."},
    {"context": "Context about NLP.",   "question": "What is NLP?", "answer": "Natural Language Processing."},
]


# ---------------------------------------------------------------------------
# TestIterBatches
# ---------------------------------------------------------------------------

class TestIterBatches(unittest.TestCase):

    def test_even_batches(self) -> None:
        batches = list(_iter_batches(list(range(6)), batch_size=2))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], [0, 1])
        self.assertEqual(batches[2], [4, 5])

    def test_last_batch_smaller(self) -> None:
        batches = list(_iter_batches(list(range(5)), batch_size=2))
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[-1]), 1)

    def test_single_batch(self) -> None:
        batches = list(_iter_batches([1, 2, 3], batch_size=100))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], [1, 2, 3])

    def test_empty_input(self) -> None:
        batches = list(_iter_batches([], batch_size=4))
        self.assertEqual(batches, [])


# ---------------------------------------------------------------------------
# TestLoadAndConvert
# ---------------------------------------------------------------------------

class TestLoadAndConvert(unittest.TestCase):

    def _run(self, rows=None, **kwargs):
        """Patch load_dataset and run load_and_convert with the given rows."""
        rows = rows or SAMPLE_ROWS
        mock_ds = _make_mock_dataset(rows)
        with patch("datasets.load_dataset", return_value=mock_ds):
            return load_and_convert(split="train", **kwargs)

    # ------------------------------------------------------------------

    def test_basic_conversion(self) -> None:
        """3 rows → 3 documents, 3 ground-truth entries."""
        docs, gt = self._run()
        self.assertEqual(len(docs), 3)
        self.assertEqual(len(gt),   3)

    def test_document_schema(self) -> None:
        """Every document has keys 'id', 'text', 'metadata'."""
        docs, _ = self._run()
        for doc in docs:
            self.assertIn("id",       doc)
            self.assertIn("text",     doc)
            self.assertIn("metadata", doc)

    def test_gt_schema(self) -> None:
        """Every GT entry has keys 'query' and 'relevant_ids'."""
        _, gt = self._run()
        for entry in gt:
            self.assertIn("query",        entry)
            self.assertIn("relevant_ids", entry)
            self.assertIsInstance(entry["relevant_ids"], list)

    def test_relevant_id_points_to_document(self) -> None:
        """Each GT entry's relevant_id matches a document id."""
        docs, gt = self._run()
        doc_ids = {d["id"] for d in docs}
        for entry in gt:
            for rid in entry["relevant_ids"]:
                self.assertIn(rid, doc_ids)

    def test_deduplication(self) -> None:
        """Two rows with identical context + deduplicate=True → 1 document, 2 GT entries."""
        rows = [
            {"context": "Same context.", "question": "Q1?", "answer": "A1"},
            {"context": "Same context.", "question": "Q2?", "answer": "A2"},
        ]
        docs, gt = self._run(rows=rows, deduplicate=True)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(gt),   2)
        # Both GT entries must point to the same document
        self.assertEqual(gt[0]["relevant_ids"], gt[1]["relevant_ids"])

    def test_no_deduplication_by_default(self) -> None:
        """Two rows with identical context but deduplicate=False → 2 documents."""
        rows = [
            {"context": "Same.", "question": "Q1?", "answer": "A1"},
            {"context": "Same.", "question": "Q2?", "answer": "A2"},
        ]
        docs, _ = self._run(rows=rows, deduplicate=False)
        self.assertEqual(len(docs), 2)

    def test_empty_question_skipped(self) -> None:
        """Rows with an empty question produce no ground-truth entry."""
        rows = [
            {"context": "Context A.", "question": "Real question?", "answer": "A"},
            {"context": "Context B.", "question": "",               "answer": "B"},
        ]
        docs, gt = self._run(rows=rows)
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(gt),   1)
        self.assertEqual(gt[0]["query"], "Real question?")

    def test_split_all_calls_concatenate(self) -> None:
        """split='all' triggers concatenate_datasets."""
        mock_ds = _make_mock_dataset(SAMPLE_ROWS)
        mock_ds_dict = {"train": mock_ds, "test": mock_ds}

        with patch("datasets.load_dataset",     return_value=mock_ds_dict), \
             patch("datasets.concatenate_datasets", return_value=mock_ds) as mock_concat:
            load_and_convert(split="all")
            mock_concat.assert_called_once()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
