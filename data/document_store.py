"""
DocumentStore: loads the JSON document corpus and exposes lookup helpers
used by every retriever.

JSON schema expected (produced by csv_to_json.py):
[
  {"id": "...", "text": "...", "metadata": {...}},
  ...
]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DocumentStore:
    """
    In-memory document corpus.

    Attributes
    ----------
    documents : list[dict]
        All documents in insertion order.
    texts : list[str]
        Parallel list of raw text strings (for BM25 / cross-encoder).
    ids : list[str]
        Parallel list of document IDs.
    _id_to_idx : dict[str, int]
        Fast ID → positional-index lookup.
    """

    def __init__(self, documents: list[dict]) -> None:
        if not documents:
            raise ValueError("DocumentStore requires at least one document.")
        self.documents = documents
        self.texts: list[str] = [d["text"] for d in documents]
        self.ids: list[str] = [str(d["id"]) for d in documents]
        self._id_to_idx: dict[str, int] = {doc_id: i for i, doc_id in enumerate(self.ids)}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "DocumentStore":
        with open(path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        return cls(documents)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_by_id(self, doc_id: str) -> dict[str, Any]:
        idx = self._id_to_idx[doc_id]
        return self.documents[idx]

    def get_by_index(self, idx: int) -> dict[str, Any]:
        return self.documents[idx]

    def idx_to_id(self, idx: int) -> str:
        return self.ids[idx]

    def id_to_idx(self, doc_id: str) -> int:
        return self._id_to_idx[doc_id]

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"DocumentStore(n={len(self)})"
