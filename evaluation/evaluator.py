"""
Evaluator: runs one or more RetrieverPipelines over a ground-truth dataset,
computes retrieval metrics, and produces a comparison report.

Ground-truth format
-------------------
A list of dicts:
[
  {
    "query":        "What is machine learning?",
    "relevant_ids": ["doc-42", "doc-17"]   # list of ground-truth doc IDs
  },
  ...
]

This can be loaded from JSON (see Evaluator.from_json_ground_truth) or
constructed programmatically.

Usage
-----
    from evaluation import Evaluator
    from retrieval.pipeline import build_pipeline
    from config import PRESET_CONFIGS
    from data import DocumentStore

    store = DocumentStore.from_json("documents.json")

    pipelines = {
        cfg.name: build_pipeline(cfg, store, faiss_index_path="index.faiss")
        for cfg in PRESET_CONFIGS
    }

    evaluator = Evaluator(pipelines=pipelines, k=10)
    evaluator.load_ground_truth("ground_truth.json")
    report = evaluator.run()
    print(report.summary_table())
    report.save_csv("results.csv")
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from retrieval.pipeline import RetrieverPipeline
from .metrics import MetricsResult, compute_metrics, average_metrics


@dataclass
class QueryResult:
    query: str
    pipeline_name: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    metrics: MetricsResult
    latency_ms: float


@dataclass
class EvaluationResult:
    """Aggregated results for all pipelines across all queries."""
    pipeline_results: Dict[str, List[QueryResult]] = field(default_factory=dict)
    aggregate: Dict[str, MetricsResult] = field(default_factory=dict)
    k: int = 10

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_table(self, fmt: str = "simple") -> str:
        """
        Return a formatted comparison table (requires `tabulate`).

        fmt: any tabulate table format, e.g. "simple", "github", "latex"
        """
        try:
            from tabulate import tabulate
        except ImportError:
            return self._plain_table()

        headers = [
            "Pipeline",
            f"Recall@{self.k}",
            f"Precision@{self.k}",
            "MRR",
            f"NDCG@{self.k}",
            f"MAP@{self.k}",
            f"HitRate@{self.k}",
            "Latency(ms)",
        ]
        rows = []
        for name, agg in self.aggregate.items():
            avg_latency = sum(
                r.latency_ms for r in self.pipeline_results[name]
            ) / len(self.pipeline_results[name])
            rows.append([
                name,
                f"{agg.recall_at_k:.4f}",
                f"{agg.precision_at_k:.4f}",
                f"{agg.mrr:.4f}",
                f"{agg.ndcg_at_k:.4f}",
                f"{agg.map_at_k:.4f}",
                f"{agg.hit_rate_at_k:.4f}",
                f"{avg_latency:.1f}",
            ])
        # Sort by NDCG descending
        rows.sort(key=lambda r: r[4], reverse=True)
        return tabulate(rows, headers=headers, tablefmt=fmt)

    def _plain_table(self) -> str:
        lines = [f"{'Pipeline':<30} {'Recall':>8} {'MRR':>8} {'NDCG':>8} {'HitRate':>9}"]
        lines.append("-" * 70)
        for name, agg in self.aggregate.items():
            lines.append(
                f"{name:<30} {agg.recall_at_k:>8.4f} {agg.mrr:>8.4f} "
                f"{agg.ndcg_at_k:>8.4f} {agg.hit_rate_at_k:>9.4f}"
            )
        return "\n".join(lines)

    def save_csv(self, path: str | Path) -> None:
        """Write per-query results for all pipelines to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pipeline", "query",
                "recall", "precision", "mrr", "ndcg", "map", "hit_rate",
                "latency_ms", "retrieved_ids", "relevant_ids",
            ])
            for name, query_results in self.pipeline_results.items():
                for qr in query_results:
                    m = qr.metrics
                    writer.writerow([
                        name, qr.query,
                        m.recall_at_k, m.precision_at_k, m.mrr,
                        m.ndcg_at_k, m.map_at_k, m.hit_rate_at_k,
                        qr.latency_ms,
                        "|".join(qr.retrieved_ids),
                        "|".join(qr.relevant_ids),
                    ])

    def save_json(self, path: str | Path) -> None:
        """Write aggregated metrics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: agg.to_dict() for name, agg in self.aggregate.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def best_pipeline(self, metric: str = "ndcg_at_k") -> str:
        """Return the name of the pipeline with the highest value for `metric`."""
        return max(self.aggregate, key=lambda name: getattr(self.aggregate[name], metric))


class Evaluator:
    """
    Runs retrieval pipelines over a ground-truth dataset and computes metrics.

    Parameters
    ----------
    pipelines:
        Dict mapping pipeline name → RetrieverPipeline instance.
    k:
        Rank cut-off for all metrics.
    """

    def __init__(
        self,
        pipelines: Dict[str, RetrieverPipeline],
        k: int = 10,
    ) -> None:
        self.pipelines = pipelines
        self.k = k
        self._ground_truth: List[dict] = []

    # ------------------------------------------------------------------
    # Ground truth loading
    # ------------------------------------------------------------------

    def load_ground_truth(self, path: str | Path) -> "Evaluator":
        """Load ground truth from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self._ground_truth = json.load(f)
        return self

    def set_ground_truth(self, ground_truth: List[dict]) -> "Evaluator":
        """Set ground truth from a list of dicts (in-memory)."""
        self._ground_truth = ground_truth
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        pipeline_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate all (or a subset of) pipelines.

        Parameters
        ----------
        pipeline_names:
            If given, only these pipelines are evaluated. Otherwise all.
        verbose:
            Show tqdm progress bars.
        """
        if not self._ground_truth:
            raise RuntimeError("No ground truth loaded. Call load_ground_truth() first.")

        names_to_run = pipeline_names or list(self.pipelines.keys())
        result = EvaluationResult(k=self.k)

        for name in names_to_run:
            if name not in self.pipelines:
                raise ValueError(f"Pipeline '{name}' not found.")
            pipeline = self.pipelines[name]
            query_results: List[QueryResult] = []

            iterator = tqdm(
                self._ground_truth,
                desc=f"Evaluating {name}",
                disable=not verbose,
                unit="query",
            )

            for item in iterator:
                query = item["query"]
                relevant_ids = item["relevant_ids"]

                t0 = time.perf_counter()
                retrieved = pipeline.run(query)
                latency_ms = (time.perf_counter() - t0) * 1000

                retrieved_ids = [r.doc_id for r in retrieved]
                metrics = compute_metrics(retrieved_ids, relevant_ids, self.k)

                query_results.append(QueryResult(
                    query=query,
                    pipeline_name=name,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=list(relevant_ids),
                    metrics=metrics,
                    latency_ms=latency_ms,
                ))

            result.pipeline_results[name] = query_results
            result.aggregate[name] = average_metrics([qr.metrics for qr in query_results])

        return result

    def compare(
        self,
        configs: Optional[List[str]] = None,
        verbose: bool = True,
        output_csv: Optional[str] = None,
        output_json: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Shortcut: run + print table + optionally save results.
        """
        result = self.run(pipeline_names=configs, verbose=verbose)
        print("\n" + result.summary_table(fmt="github") + "\n")
        print(f"Best pipeline (NDCG@{self.k}): {result.best_pipeline()}")
        if output_csv:
            result.save_csv(output_csv)
            print(f"Per-query CSV saved to: {output_csv}")
        if output_json:
            result.save_json(output_json)
            print(f"Aggregate JSON saved to: {output_json}")
        return result
