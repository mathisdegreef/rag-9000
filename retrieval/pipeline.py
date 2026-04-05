"""
RetrieverPipeline: orchestrates first-stage retrieval, optional fusion,
and optional reranking into a single callable.

The pipeline is fully config-driven (see config.py). A single pipeline
instance is created per PipelineConfig and can be evaluated over many
queries efficiently.

Architecture
------------
query
  │
  ├─► BiEncoderRetriever  ─┐
  │                         ├─► RRF fusion ─► CrossEncoderReranker ─► results
  └─► BM25Retriever       ─┘        (optional)        (optional)

Any leg can be removed via PipelineConfig.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config import PipelineConfig
from data.document_store import DocumentStore
from .base import BaseRetriever, BaseReranker, RetrievalResult
from .rrf import reciprocal_rank_fusion


class RetrieverPipeline:
    """
    A fully assembled retrieval pipeline driven by a PipelineConfig.

    Parameters
    ----------
    config:
        Pipeline configuration.
    retrievers:
        Dict mapping retriever name ("bi_encoder", "bm25") to instantiated
        BaseRetriever objects.  Only the names listed in config.retrievers
        are used.
    reranker:
        Instantiated BaseReranker, or None.  Required when
        config.reranker is set.
    """

    def __init__(
        self,
        config: PipelineConfig,
        retrievers: Dict[str, BaseRetriever],
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self.config = config

        # Validate that the required retrievers are supplied
        missing = set(config.retrievers) - set(retrievers)
        if missing:
            raise ValueError(f"Missing retriever(s) for config '{config.name}': {missing}")
        self.retrievers = {name: retrievers[name] for name in config.retrievers}

        if config.reranker and reranker is None:
            raise ValueError(
                f"Config '{config.name}' requires a reranker but none was supplied."
            )
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> List[RetrievalResult]:
        """Execute the full pipeline for a single query."""
        # Stage 1 – first-stage retrieval
        candidate_lists = [
            retriever.retrieve(query, self.config.top_k_retrieve)
            for retriever in self.retrievers.values()
        ]

        # Stage 2 – fusion (only when multiple retrievers)
        if len(candidate_lists) == 1:
            candidates = candidate_lists[0]
        else:
            candidates = reciprocal_rank_fusion(
                candidate_lists,
                k=self.config.rrf_k,
                top_k=self.config.top_k_retrieve,
            )

        # Stage 3 – optional reranking
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, self.config.top_k_final)
        else:
            candidates = candidates[: self.config.top_k_final]
            # Refresh rank numbers after truncation
            for rank, r in enumerate(candidates, start=1):
                r.rank = rank

        return candidates

    def batch_run(self, queries: List[str]) -> List[List[RetrievalResult]]:
        """
        Run the pipeline over a batch of queries.

        For bi-encoder-only pipelines (no reranker, single retriever),
        batch encoding is used for efficiency.  All other configurations
        fall back to sequential processing.
        """
        use_batch_encode = (
            len(self.config.retrievers) == 1
            and "bi_encoder" in self.config.retrievers
            and self.reranker is None
            and hasattr(list(self.retrievers.values())[0], "batch_retrieve")
        )

        if use_batch_encode:
            retriever = list(self.retrievers.values())[0]
            candidate_lists = retriever.batch_retrieve(queries, self.config.top_k_retrieve)
            results = []
            for candidates in candidate_lists:
                candidates = candidates[: self.config.top_k_final]
                for rank, r in enumerate(candidates, start=1):
                    r.rank = rank
                results.append(candidates)
            return results

        return [self.run(q) for q in queries]

    def __repr__(self) -> str:
        return f"RetrieverPipeline({self.config.summary()})"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_pipeline(
    config: PipelineConfig,
    store: DocumentStore,
    faiss_index_path: Optional[str] = None,
    bi_encoder_model: str = "microsoft/harrier-oss-v1-0.6b",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> RetrieverPipeline:
    """
    Convenience factory: instantiates all required retrievers and assembles
    the pipeline from a PipelineConfig.

    Parameters
    ----------
    config:
        Pipeline configuration.
    store:
        DocumentStore with the corpus.
    faiss_index_path:
        Path to the FAISS index file.  Required when "bi_encoder" is in
        config.retrievers.
    bi_encoder_model:
        Sentence-transformers model for query encoding.
    cross_encoder_model:
        Cross-encoder model for reranking.
    """
    retrievers: Dict[str, BaseRetriever] = {}

    if "bi_encoder" in config.retrievers:
        if faiss_index_path is None:
            raise ValueError("faiss_index_path must be provided for bi_encoder retriever.")
        from .bi_encoder import BiEncoderRetriever
        retrievers["bi_encoder"] = BiEncoderRetriever(
            store=store,
            index_path=faiss_index_path,
            model_name=bi_encoder_model,
        )

    if "bm25" in config.retrievers:
        from .bm25_retriever import BM25Retriever
        retrievers["bm25"] = BM25Retriever(store=store)

    reranker: Optional[BaseReranker] = None
    if config.reranker == "cross_encoder":
        from .cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(store=store, model_name=cross_encoder_model)

    return RetrieverPipeline(config=config, retrievers=retrievers, reranker=reranker)
