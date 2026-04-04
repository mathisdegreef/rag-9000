"""
Unit tests for data/csv_to_json.py.

All tests write real CSV files to tempfile.TemporaryDirectory — no mocking needed.
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.csv_to_json import csv_to_documents


def _write_csv(tmp_dir: Path, filename: str, content: str) -> Path:
    path = tmp_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


class TestCsvToDocuments(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------------

    def test_basic_conversion(self) -> None:
        """Simple 3-row CSV → 3 documents with correct id and text."""
        csv_path = _write_csv(self.tmp, "test.csv", (
            "url,content\n"
            "http://a.com,text A\n"
            "http://b.com,text B\n"
            "http://c.com,text C\n"
        ))
        docs = csv_to_documents(csv_path, text_col="content", id_col="url")
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0]["id"],   "http://a.com")
        self.assertEqual(docs[0]["text"], "text A")
        self.assertEqual(docs[2]["id"],   "http://c.com")

    def test_auto_id_when_no_id_col(self) -> None:
        """When id_col is None, IDs are '0', '1', '2', …"""
        csv_path = _write_csv(self.tmp, "test.csv", (
            "content\n"
            "alpha\n"
            "beta\n"
        ))
        docs = csv_to_documents(csv_path, text_col="content", id_col=None)
        self.assertEqual(docs[0]["id"], "0")
        self.assertEqual(docs[1]["id"], "1")

    def test_explicit_id_col(self) -> None:
        """id_col='url' → IDs taken from that column."""
        csv_path = _write_csv(self.tmp, "test.csv", (
            "url,content\n"
            "page-42,hello\n"
        ))
        docs = csv_to_documents(csv_path, text_col="content", id_col="url")
        self.assertEqual(docs[0]["id"], "page-42")

    def test_prepend_extra_cols(self) -> None:
        """extra_text_cols=['title'] prepends title + '\\n\\n' + content."""
        csv_path = _write_csv(self.tmp, "test.csv", (
            "url,title,content\n"
            "http://x.com,My Title,My Content\n"
        ))
        docs = csv_to_documents(
            csv_path, text_col="content", id_col="url",
            extra_text_cols=["title"],
        )
        self.assertIn("My Title", docs[0]["text"])
        self.assertIn("My Content", docs[0]["text"])
        self.assertIn("\n\n", docs[0]["text"])
        # title must come before content
        self.assertLess(
            docs[0]["text"].index("My Title"),
            docs[0]["text"].index("My Content"),
        )

    def test_missing_text_col_raises(self) -> None:
        """Passing a text_col that doesn't exist raises ValueError."""
        csv_path = _write_csv(self.tmp, "test.csv", "url,body\nhttp://a.com,hello\n")
        with self.assertRaises(ValueError):
            csv_to_documents(csv_path, text_col="content")

    def test_missing_id_col_raises(self) -> None:
        """Passing an id_col that doesn't exist raises ValueError."""
        csv_path = _write_csv(self.tmp, "test.csv", "content\nhello\n")
        with self.assertRaises(ValueError):
            csv_to_documents(csv_path, text_col="content", id_col="nonexistent")

    def test_missing_extra_col_raises(self) -> None:
        """Passing a non-existent column in extra_text_cols raises ValueError."""
        csv_path = _write_csv(self.tmp, "test.csv", "content\nhello\n")
        with self.assertRaises(ValueError):
            csv_to_documents(csv_path, text_col="content", extra_text_cols=["ghost"])

    def test_metadata_contains_remaining_cols(self) -> None:
        """Columns not used as text appear in the metadata dict."""
        csv_path = _write_csv(self.tmp, "test.csv", (
            "url,title,content,views\n"
            "http://a.com,Title A,body A,100\n"
        ))
        docs = csv_to_documents(
            csv_path, text_col="content", id_col="url",
            extra_text_cols=["title"],
        )
        # 'views' is not text or id → must be in metadata
        self.assertIn("views", docs[0]["metadata"])
        self.assertEqual(str(docs[0]["metadata"]["views"]), "100")

    def test_output_schema_keys(self) -> None:
        """Every document has exactly the keys 'id', 'text', 'metadata'."""
        csv_path = _write_csv(self.tmp, "test.csv", "content\nhello\n")
        docs = csv_to_documents(csv_path, text_col="content")
        for doc in docs:
            self.assertIn("id",       doc)
            self.assertIn("text",     doc)
            self.assertIn("metadata", doc)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
