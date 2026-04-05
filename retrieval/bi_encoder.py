"""
Bi-encoder dense retriever backed by a FAISS index.

The FAISS index and the corresponding document IDs must be built separately
(you handle the embedding step). This retriever simply wraps:
  1. A pre-built FAISS index on disk (or in memory)
  2. A query encoder (sentence-transformers model)

Index assumptions
-----------------
- The FAISS index was built from embeddings in the same order as the
  DocumentStore's `ids` list.  Position i in the index corresponds to
  DocumentStore.ids[i].
- Inner-product (IP) or L2 index both work; scores are returned as-is from
  FAISS (higher = more similar for IP, lower = more similar for L2 — we
  negate L2 to maintain "higher = better" convention).

Usage
-----
    from data import DocumentStore
    from retrieval import BiEncoderRetriever

    store = DocumentStore.from_json("documents.json")
    retriever = BiEncoderRetriever(
        store=store,
        index_path="index.faiss",
        model_name="microsoft/harrier-oss-v1-0.6b",
    )
    results = retriever.retrieve("what is machine learning?", top_k=10)
"""

from __future__ import annotations

import numpy as np

from data.document_store import DocumentStore
from .base import BaseRetriever, RetrievalResult

try:
    import faiss
except ImportError as e:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e


class BiEncoderRetriever(BaseRetriever):
    """
    Dense retriever: encodes the query with a bi-encoder, then runs a nearest-
    neighbour search on a pre-built FAISS index.

    Parameters
    ----------
    store:
        DocumentStore containing the corpus.
    index_path:
        Path to the serialised FAISS index file.
    model_name:
        Sentence-transformers model name or local path used to encode queries.
    normalize_embeddings:
        If True, L2-normalise query embeddings before the FAISS search.
        Required for cosine-similarity with IndexFlatIP.
    is_l2_index:
        Set to True if the FAISS index uses L2 distance so that scores are
        negated (making higher = better).
    """

    def __init__(
        self,
        store: DocumentStore,
        index_path: str,
        model_name: str = "microsoft/harrier-oss-v1-0.6b",
        normalize_embeddings: bool = True,
        is_l2_index: bool = False,
    ) -> None:
        self.store = store
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings
        self.is_l2_index = is_l2_index

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            adjusted_score = float(-score) if self.is_l2_index else float(score)
            doc_id = self.store.idx_to_id(int(idx))
            results.append(RetrievalResult(doc_id=doc_id, score=adjusted_score, rank=rank))

        return results

    def batch_retrieve(
        self, queries: list[str], top_k: int
    ) -> list[list[RetrievalResult]]:
        """Vectorised batch encoding for efficiency."""
        query_vecs = self.model.encode(
            queries,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            batch_size=64,
        ).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores_batch, indices_batch = self.index.search(query_vecs, k)

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx == -1:
                    continue
                adjusted_score = float(-score) if self.is_l2_index else float(score)
                doc_id = self.store.idx_to_id(int(idx))
                results.append(RetrievalResult(doc_id=doc_id, score=adjusted_score, rank=rank))
            all_results.append(results)
        return all_results
