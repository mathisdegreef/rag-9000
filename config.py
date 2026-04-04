"""
Pipeline configuration dataclasses.

A PipelineConfig defines one complete retrieval strategy. Mix and match:
  - retrievers: which first-stage retrievers to run ("bi_encoder", "bm25", or both)
  - fusion:     how to merge multi-retriever results ("rrf" | None)
  - reranker:   optional second-stage reranker ("cross_encoder" | None)
  - top_k_*:    how many candidates to keep at each stage

Examples
--------
Single bi-encoder only:
    PipelineConfig(name="dense", retrievers=["bi_encoder"])

BM25 only:
    PipelineConfig(name="bm25", retrievers=["bm25"])

Hybrid with RRF, no reranker:
    PipelineConfig(name="hybrid_rrf", retrievers=["bi_encoder", "bm25"], fusion="rrf")

Full pipeline – hybrid + cross-encoder reranker:
    PipelineConfig(name="full", retrievers=["bi_encoder", "bm25"], fusion="rrf", reranker="cross_encoder")
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


Retriever = Literal["bi_encoder", "bm25"]
Fusion = Literal["rrf"]
Reranker = Literal["cross_encoder"]


@dataclass
class PipelineConfig:
    name: str

    # First-stage: which retrievers to use
    retrievers: List[Retriever] = field(default_factory=lambda: ["bi_encoder"])

    # How to merge results when multiple retrievers are active
    # Required (and only meaningful) when len(retrievers) > 1
    fusion: Optional[Fusion] = "rrf"

    # Optional second-stage reranker
    reranker: Optional[Reranker] = None

    # Candidate pool size coming out of each first-stage retriever
    top_k_retrieve: int = 50

    # Final result size (after reranking, or directly from first-stage)
    top_k_final: int = 10

    # ---- RRF hyperparameter ----
    rrf_k: int = 60  # typical default; higher → smoother fusion

    def __post_init__(self) -> None:
        if not self.retrievers:
            raise ValueError("At least one retriever must be specified.")
        if len(self.retrievers) > 1 and self.fusion is None:
            raise ValueError(
                "fusion must be set when using multiple retrievers. "
                "Currently only 'rrf' is supported."
            )
        unknown = set(self.retrievers) - {"bi_encoder", "bm25"}
        if unknown:
            raise ValueError(f"Unknown retriever(s): {unknown}")
        if self.reranker not in (None, "cross_encoder"):
            raise ValueError(f"Unknown reranker: {self.reranker!r}")

    def summary(self) -> str:
        parts = [f"retrievers={self.retrievers}"]
        if len(self.retrievers) > 1:
            parts.append(f"fusion={self.fusion}(k={self.rrf_k})")
        if self.reranker:
            parts.append(f"reranker={self.reranker}")
        parts.append(f"top_k={self.top_k_retrieve}→{self.top_k_final}")
        return f"PipelineConfig({self.name}: {', '.join(parts)})"


# ---------------------------------------------------------------------------
# Pre-built configurations for quick experimentation
# ---------------------------------------------------------------------------

PRESET_CONFIGS: List[PipelineConfig] = [
    PipelineConfig(
        name="bi_encoder_only",
        retrievers=["bi_encoder"],
        reranker=None,
    ),
    PipelineConfig(
        name="bm25_only",
        retrievers=["bm25"],
        reranker=None,
    ),
    PipelineConfig(
        name="bi_encoder_reranked",
        retrievers=["bi_encoder"],
        reranker="cross_encoder",
    ),
    PipelineConfig(
        name="bm25_reranked",
        retrievers=["bm25"],
        reranker="cross_encoder",
    ),
    PipelineConfig(
        name="hybrid_rrf",
        retrievers=["bi_encoder", "bm25"],
        fusion="rrf",
        reranker=None,
    ),
    PipelineConfig(
        name="hybrid_rrf_reranked",
        retrievers=["bi_encoder", "bm25"],
        fusion="rrf",
        reranker="cross_encoder",
    ),
]
