"""
Abstract base classes for retrievers and rerankers.

Every retriever returns a list of RetrievalResult, ordered by descending score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalResult:
    """A single retrieved document with its score."""
    doc_id: str
    score: float
    # Rank within this retriever's result list (1-based)
    rank: int = 0

    def __repr__(self) -> str:
        return f"RetrievalResult(id={self.doc_id!r}, score={self.score:.4f}, rank={self.rank})"


class BaseRetriever(ABC):
    """First-stage retriever interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Return up to top_k results for query, sorted by descending score."""
        ...

    def batch_retrieve(
        self, queries: List[str], top_k: int
    ) -> List[List[RetrievalResult]]:
        """Default: loop over retrieve(). Override for vectorised batch calls."""
        return [self.retrieve(q, top_k) for q in queries]


class BaseReranker(ABC):
    """Second-stage reranker interface."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        """Re-score candidates and return the top_k, sorted descending."""
        ...
