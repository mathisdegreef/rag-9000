"""
CLI entrypoint for the retrieval evaluation pipeline.

Typical usage
-------------
# Evaluate all preset configs:
python run_evaluation.py \
    --docs      documents.json \
    --index     index.faiss \
    --gt        ground_truth.json \
    --k         10 \
    --out-csv   results/eval.csv \
    --out-json  results/eval.json

# Evaluate only specific pipelines:
python run_evaluation.py \
    --docs documents.json --index index.faiss --gt ground_truth.json \
    --pipelines bm25_only hybrid_rrf_reranked

# Use a custom bi-encoder model:
python run_evaluation.py \
    --docs documents.json --index index.faiss --gt ground_truth.json \
    --bi-encoder-model sentence-transformers/all-mpnet-base-v2

Ground truth JSON format
------------------------
[
  {"query": "...", "relevant_ids": ["doc-1", "doc-7"]},
  ...
]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from config import PRESET_CONFIGS, PipelineConfig
from data import DocumentStore
from retrieval.pipeline import build_pipeline, RetrieverPipeline
from evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval pipeline configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--docs", required=True, help="Path to documents.json")
    parser.add_argument("--index", default=None, help="Path to FAISS index file (required for bi_encoder)")
    parser.add_argument("--gt", required=True, help="Path to ground_truth.json")

    # Evaluation
    parser.add_argument("--k", type=int, default=10, help="Rank cut-off for metrics")
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=None,
        help=(
            "Names of preset pipelines to evaluate. "
            "Default: all presets. "
            "Available: " + ", ".join(c.name for c in PRESET_CONFIGS)
        ),
    )

    # Model overrides
    parser.add_argument(
        "--bi-encoder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model for bi-encoder",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking",
    )

    # Output
    parser.add_argument("--out-csv", default=None, help="Save per-query CSV to this path")
    parser.add_argument("--out-json", default=None, help="Save aggregate JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars")

    return parser.parse_args()


def build_pipelines(
    configs: List[PipelineConfig],
    store: DocumentStore,
    faiss_index_path: Optional[str],
    bi_encoder_model: str,
    cross_encoder_model: str,
) -> dict[str, RetrieverPipeline]:
    pipelines = {}
    for cfg in configs:
        needs_faiss = "bi_encoder" in cfg.retrievers
        if needs_faiss and faiss_index_path is None:
            print(
                f"[skip] '{cfg.name}' requires bi_encoder but --index was not provided.",
                file=sys.stderr,
            )
            continue
        print(f"[build] {cfg.summary()}", file=sys.stderr)
        pipelines[cfg.name] = build_pipeline(
            config=cfg,
            store=store,
            faiss_index_path=faiss_index_path,
            bi_encoder_model=bi_encoder_model,
            cross_encoder_model=cross_encoder_model,
        )
    return pipelines


def main() -> None:
    args = parse_args()

    # Load corpus
    print(f"Loading documents from {args.docs} …", file=sys.stderr)
    store = DocumentStore.from_json(args.docs)
    print(f"  {len(store)} documents loaded.", file=sys.stderr)

    # Determine which preset configs to run
    preset_map = {c.name: c for c in PRESET_CONFIGS}
    if args.pipelines:
        unknown = set(args.pipelines) - set(preset_map)
        if unknown:
            print(f"Unknown pipeline names: {unknown}. Available: {list(preset_map)}", file=sys.stderr)
            sys.exit(1)
        selected_configs = [preset_map[n] for n in args.pipelines]
    else:
        selected_configs = list(PRESET_CONFIGS)

    # Build pipelines
    print("Building pipelines …", file=sys.stderr)
    pipelines = build_pipelines(
        configs=selected_configs,
        store=store,
        faiss_index_path=args.index,
        bi_encoder_model=args.bi_encoder_model,
        cross_encoder_model=args.cross_encoder_model,
    )

    if not pipelines:
        print("No pipelines could be built. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    evaluator = Evaluator(pipelines=pipelines, k=args.k)
    evaluator.load_ground_truth(args.gt)

    evaluator.compare(
        verbose=not args.quiet,
        output_csv=args.out_csv,
        output_json=args.out_json,
    )


if __name__ == "__main__":
    main()
