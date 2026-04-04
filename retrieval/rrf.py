"""
Reciprocal Rank Fusion (RRF).

RRF merges multiple ranked lists into a single list without requiring
score normalisation, using only each document's rank position.

Formula (Cormack et al., 2009):
    RRF_score(d) = Σ  1 / (k + rank_i(d))
                  i

where rank_i(d) is document d's 1-based rank in list i, and k is a
smoothing constant (default 60).

Why RRF is robust:
- Insensitive to score magnitude differences between retrievers.
- Documents appearing in multiple lists get a natural score boost.
- Simple, parameter-light, consistently competitive with learned fusion.
"""

from __future__ import annotations

from typing import Dict, List

from .base import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: List[List[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """
    Fuse multiple ranked lists into one using Reciprocal Rank Fusion.

    Parameters
    ----------
    result_lists:
        Each element is an ordered list of RetrievalResult from one retriever.
        Order within each list is assumed to be descending by score (rank 1
        is the best). The `.rank` field is used if set; otherwise position in
        the list is used.
    k:
        RRF smoothing constant. Higher k → less aggressive rank boosting.
    top_k:
        Truncate final list to this many results. None = return all.

    Returns
    -------
    Merged list sorted by descending RRF score, with updated `.rank` fields.
    """
    scores: Dict[str, float] = {}

    for result_list in result_lists:
        for position, result in enumerate(result_list):
            # Use the stored rank if available, otherwise fall back to position
            rank = result.rank if result.rank > 0 else (position + 1)
            scores[result.doc_id] = scores.get(result.doc_id, 0.0) + 1.0 / (k + rank)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        merged = merged[:top_k]

    return [
        RetrievalResult(doc_id=doc_id, score=rrf_score, rank=rank)
        for rank, (doc_id, rrf_score) in enumerate(merged, start=1)
    ]
