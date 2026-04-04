"""
Cross-encoder reranker.

A cross-encoder takes (query, document) pairs as input and produces a
relevance score. It is significantly more accurate than a bi-encoder but
also more expensive — so it operates on a small candidate pool (typically
the top-50 from a first-stage retriever).

This implementation uses Hugging Face sentence-transformers CrossEncoder
models, which are available off-the-shelf:
  - "cross-encoder/ms-marco-MiniLM-L-6-v2"   (fast, English)
  - "cross-encoder/ms-marco-electra-base"     (higher accuracy)

Usage
-----
    from data import DocumentStore
    from retrieval import CrossEncoderReranker

    store = DocumentStore.from_json("documents.json")
    reranker = CrossEncoderReranker(store=store)
    reranked = reranker.rerank(query, candidates, top_k=10)
"""

from __future__ import annotations

from typing import List

from data.document_store import DocumentStore
from .base import BaseReranker, RetrievalResult

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker.

    Parameters
    ----------
    store:
        DocumentStore used to fetch document texts by ID.
    model_name:
        HuggingFace model name or local path of a cross-encoder model.
    max_length:
        Maximum token length for the cross-encoder input.
        Documents will be truncated if they exceed this.
    batch_size:
        Inference batch size.
    """

    def __init__(
        self,
        store: DocumentStore,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.store = store
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def rerank(
        self, query: str, candidates: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        if not candidates:
            return []

        # Build (query, passage) pairs
        pairs = [
            (query, self.store.get_by_id(r.doc_id)["text"])
            for r in candidates
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        # Re-sort by cross-encoder score
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RetrievalResult(doc_id=result.doc_id, score=float(score), rank=rank)
            for rank, (result, score) in enumerate(scored[:top_k], start=1)
        ]
