"""
Unit tests for data/document_store.py.

All tests are self-contained — no files on disk, no network calls.
Temporary files are written to tempfile.TemporaryDirectory where needed.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.document_store import DocumentStore


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"id": "doc-0", "text": "Machine learning basics.", "metadata": {"source": "web"}},
    {"id": "doc-1", "text": "Deep learning advances.",  "metadata": {"source": "web"}},
    {"id": "doc-2", "text": "Natural language processing.", "metadata": {}},
]


# ---------------------------------------------------------------------------
# TestDocumentStoreInit
# ---------------------------------------------------------------------------

class TestDocumentStoreInit(unittest.TestCase):

    def test_empty_raises(self) -> None:
        """DocumentStore([]) must raise ValueError."""
        with self.assertRaises(ValueError):
            DocumentStore([])

    def test_texts_and_ids_parallel(self) -> None:
        """store.texts[i] and store.ids[i] match the i-th document."""
        store = DocumentStore(SAMPLE_DOCS)
        for i, doc in enumerate(SAMPLE_DOCS):
            self.assertEqual(store.texts[i], doc["text"])
            self.assertEqual(store.ids[i],   str(doc["id"]))


# ---------------------------------------------------------------------------
# TestDocumentStoreFromJson
# ---------------------------------------------------------------------------

class TestDocumentStoreFromJson(unittest.TestCase):

    def test_from_json_roundtrip(self) -> None:
        """Write docs to a tmp JSON file, load via from_json, check count."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.json"
            path.write_text(json.dumps(SAMPLE_DOCS), encoding="utf-8")

            store = DocumentStore.from_json(path)
            self.assertEqual(len(store), len(SAMPLE_DOCS))

    def test_from_json_missing_file_raises(self) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            DocumentStore.from_json("/nonexistent/path/docs.json")


# ---------------------------------------------------------------------------
# TestDocumentStoreLookups
# ---------------------------------------------------------------------------

class TestDocumentStoreLookups(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = DocumentStore(SAMPLE_DOCS)

    def test_get_by_id(self) -> None:
        doc = self.store.get_by_id("doc-1")
        self.assertEqual(doc["id"], "doc-1")
        self.assertEqual(doc["text"], "Deep learning advances.")

    def test_get_by_id_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.store.get_by_id("does-not-exist")

    def test_get_by_index(self) -> None:
        doc = self.store.get_by_index(2)
        self.assertEqual(doc["id"], "doc-2")

    def test_idx_to_id(self) -> None:
        self.assertEqual(self.store.idx_to_id(0), "doc-0")
        self.assertEqual(self.store.idx_to_id(2), "doc-2")

    def test_id_to_idx(self) -> None:
        self.assertEqual(self.store.id_to_idx("doc-0"), 0)
        self.assertEqual(self.store.id_to_idx("doc-2"), 2)

    def test_len(self) -> None:
        self.assertEqual(len(self.store), 3)

    def test_repr_is_string(self) -> None:
        r = repr(self.store)
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
