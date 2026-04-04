"""
BM25 sparse retriever using the `rank-bm25` library.

The index is built at construction time from the DocumentStore's texts.
Tokenisation is deliberately simple (whitespace + lowercase) so it works
out-of-the-box, but you can inject a custom tokeniser for your use case.

Usage
-----
    from data import DocumentStore
    from retrieval import BM25Retriever

    store = DocumentStore.from_json("documents.json")
    retriever = BM25Retriever(store)
    results = retriever.retrieve("machine learning tutorial", top_k=10)
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from data.document_store import DocumentStore
from .base import BaseRetriever, RetrievalResult

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError("Install rank-bm25: pip install rank-bm25") from e


def _default_tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric chars."""
    return re.split(r"\W+", text.lower())


class BM25Retriever(BaseRetriever):
    """
    Sparse BM25 (Okapi BM25) retriever.

    Parameters
    ----------
    store:
        DocumentStore with the corpus.
    tokenizer:
        Callable that maps a string to a list of tokens.
        Defaults to simple lowercase word tokenisation.
    """

    def __init__(
        self,
        store: DocumentStore,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.store = store
        self._tokenize = tokenizer or _default_tokenize

        tokenized_corpus = [self._tokenize(text) for text in store.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)  # ndarray of shape (n_docs,)

        # argsort descending, take top_k
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score == 0.0:
                break  # BM25 scores are non-negative; 0 means no match
            doc_id = self.store.idx_to_id(int(idx))
            results.append(RetrievalResult(doc_id=doc_id, score=score, rank=rank))

        return results
