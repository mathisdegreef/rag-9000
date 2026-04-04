"""
Standard information-retrieval metrics for retrieval evaluation.

All metrics operate on a single query at a time and assume a binary
relevance judgement (relevant / not relevant).

Supported metrics
-----------------
- Recall@K         fraction of relevant docs found in top-K
- Precision@K      fraction of top-K results that are relevant
- MRR              Mean Reciprocal Rank (first-hit rank)
- NDCG@K           Normalised Discounted Cumulative Gain
- MAP@K            Mean Average Precision
- Hit Rate@K       1 if any relevant doc is in top-K, else 0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Set


@dataclass
class MetricsResult:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    map_at_k: float = 0.0
    hit_rate_at_k: float = 0.0
    k: int = 0
    num_relevant: int = 0
    num_retrieved: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str] | List[str],
    k: int,
) -> MetricsResult:
    """
    Compute retrieval metrics for a single query.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved document IDs (best first).  Only the
        first ``k`` entries are considered.
    relevant_ids:
        Set (or list) of ground-truth relevant document IDs.
    k:
        Cut-off rank.

    Returns
    -------
    MetricsResult with all metrics computed at rank k.
    """
    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]
    n_rel = len(relevant_set)

    if n_rel == 0:
        return MetricsResult(k=k, num_relevant=0, num_retrieved=len(retrieved_at_k))

    # ---------- Recall@K ----------
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    recall = hits / n_rel

    # ---------- Precision@K ----------
    precision = hits / len(retrieved_at_k) if retrieved_at_k else 0.0

    # ---------- Hit Rate@K ----------
    hit_rate = 1.0 if hits > 0 else 0.0

    # ---------- MRR ----------
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    # ---------- NDCG@K ----------
    # Binary relevance: gain = 1 if relevant, 0 otherwise
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_at_k, start=1)
        if doc_id in relevant_set
    )
    # Ideal DCG: all relevant docs at the top (up to k)
    ideal_hits = min(n_rel, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # ---------- MAP@K ----------
    num_hits = 0
    sum_precision = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            num_hits += 1
            sum_precision += num_hits / rank
    map_k = sum_precision / min(n_rel, k) if n_rel > 0 else 0.0

    return MetricsResult(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        ndcg_at_k=ndcg,
        map_at_k=map_k,
        hit_rate_at_k=hit_rate,
        k=k,
        num_relevant=n_rel,
        num_retrieved=len(retrieved_at_k),
    )


def average_metrics(results: List[MetricsResult]) -> MetricsResult:
    """Macro-average a list of per-query MetricsResult objects."""
    if not results:
        return MetricsResult()
    n = len(results)
    return MetricsResult(
        recall_at_k=sum(r.recall_at_k for r in results) / n,
        precision_at_k=sum(r.precision_at_k for r in results) / n,
        mrr=sum(r.mrr for r in results) / n,
        ndcg_at_k=sum(r.ndcg_at_k for r in results) / n,
        map_at_k=sum(r.map_at_k for r in results) / n,
        hit_rate_at_k=sum(r.hit_rate_at_k for r in results) / n,
        k=results[0].k,
        num_relevant=round(sum(r.num_relevant for r in results) / n),
        num_retrieved=round(sum(r.num_retrieved for r in results) / n),
    )
