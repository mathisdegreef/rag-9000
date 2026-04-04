"""
Load a HuggingFace RAG dataset and convert it to the pipeline's native formats.

Default dataset: neural-bridge/rag-dataset-1200
  Fields : context (document text), question (query), answer (expected answer)
  Splits : train (960), test (240)
  License: Apache 2.0

Output files
------------
documents.json   — corpus consumed by DocumentStore / retrievers
ground_truth.json — query–relevant_id pairs consumed by Evaluator

Usage
-----
    # Defaults: full train split → documents.json + ground_truth.json
    python data/load_hf_dataset.py

    # Custom dataset, test split, 100 samples, batch size 32
    python data/load_hf_dataset.py \\
        --dataset neural-bridge/rag-dataset-1200 \\
        --split test \\
        --sample-size 100 \\
        --batch-size 32 \\
        --docs-out data/documents.json \\
        --gt-out   data/ground_truth.json

    # Skip ground-truth generation (documents only)
    python data/load_hf_dataset.py --no-ground-truth

    # Deduplicate contexts so each unique passage becomes one document
    python data/load_hf_dataset.py --deduplicate

    # Preview without writing files
    python data/load_hf_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_batches(items: list, batch_size: int) -> Iterator[list]:
    """Yield successive fixed-size batches from *items*."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def load_and_convert(
    dataset_name: str = "neural-bridge/rag-dataset-1200",
    split: str = "train",
    sample_size: int | None = None,
    batch_size: int = 64,
    context_col: str = "context",
    question_col: str = "question",
    answer_col: str = "answer",
    deduplicate: bool = False,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Download *dataset_name* from HuggingFace and return ``(documents, ground_truth)``.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier (e.g. ``"neural-bridge/rag-dataset-1200"``).
    split:
        Dataset split to load (``"train"``, ``"test"``, ``"all"``).
        Use ``"all"`` to concatenate every available split.
    sample_size:
        If set, randomly sample this many rows before processing.
    batch_size:
        Number of rows processed per iteration (controls memory / progress
        granularity).
    context_col:
        Column name containing the document passage.
    question_col:
        Column name containing the query.
    answer_col:
        Column name containing the expected answer (stored as metadata).
    deduplicate:
        When True, identical context strings are collapsed into a single
        document; all questions that reference a context map to the same doc ID.
    seed:
        Random seed used when ``sample_size`` is set.

    Returns
    -------
    documents : list[dict]
        Each dict has keys ``id``, ``text``, ``metadata``.
    ground_truth : list[dict]
        Each dict has keys ``query``, ``relevant_ids``.
    """
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        _log("ERROR: 'datasets' package not found. Install it with: pip install datasets")
        sys.exit(1)

    # ---- Load ----------------------------------------------------------------
    _log(f"Loading '{dataset_name}' (split='{split}') from HuggingFace …")
    if split == "all":
        ds_dict = load_dataset(dataset_name)
        dataset = concatenate_datasets(list(ds_dict.values()))
    else:
        dataset = load_dataset(dataset_name, split=split)

    total = len(dataset)
    _log(f"  Downloaded {total:,} rows.")

    # ---- Sample --------------------------------------------------------------
    if sample_size is not None and sample_size < total:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        _log(f"  Sampled {sample_size:,} rows (seed={seed}).")

    rows = dataset.to_list()

    # ---- Build documents & ground-truth in batches ---------------------------
    # context → doc_id mapping (supports deduplication)
    context_to_id: dict[str, str] = {}
    documents_map: dict[str, dict] = {}   # doc_id → document dict
    ground_truth: list[dict] = []

    global_doc_idx = 0

    for batch in _iter_batches(rows, batch_size):
        for row in batch:
            context: str = str(row.get(context_col, "")).strip()
            question: str = str(row.get(question_col, "")).strip()
            answer: str   = str(row.get(answer_col, "")).strip()

            # Assign / reuse a doc ID
            if deduplicate and context in context_to_id:
                doc_id = context_to_id[context]
            else:
                doc_id = f"doc-{global_doc_idx:06d}"
                global_doc_idx += 1
                context_to_id[context] = doc_id
                documents_map[doc_id] = {
                    "id": doc_id,
                    "text": context,
                    "metadata": {"source": dataset_name, "split": split},
                }

            # Ground-truth: one entry per question pointing to its document
            if question:
                ground_truth.append({
                    "query": question,
                    "relevant_ids": [doc_id],
                    "answer": answer,
                })

    documents = list(documents_map.values())

    if deduplicate:
        _log(f"  Deduplicated: {len(rows):,} rows → {len(documents):,} unique documents.")

    return documents, ground_truth


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace RAG dataset and convert it to pipeline format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        default="neural-bridge/rag-dataset-1200",
        help="HuggingFace dataset identifier.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation", "all"],
        help="Dataset split to load. Use 'all' to concatenate every split.",
    )

    # Sampling / batching
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N rows. Omit to use the full split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Rows processed per batch (affects memory usage).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    # Column mapping
    parser.add_argument("--context-col", default="context", help="Column with document text.")
    parser.add_argument("--question-col", default="question", help="Column with query text.")
    parser.add_argument("--answer-col",   default="answer",   help="Column with expected answer.")

    # Output paths
    parser.add_argument(
        "--docs-out",
        default="documents.json",
        metavar="PATH",
        help="Output path for documents.json.",
    )
    parser.add_argument(
        "--gt-out",
        default="ground_truth.json",
        metavar="PATH",
        help="Output path for ground_truth.json.",
    )

    # Flags
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Collapse identical context passages into a single document.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip writing ground_truth.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and convert data but do not write any files.",
    )

    args = parser.parse_args()

    # ---- Run conversion ------------------------------------------------------
    documents, ground_truth = load_and_convert(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        context_col=args.context_col,
        question_col=args.question_col,
        answer_col=args.answer_col,
        deduplicate=args.deduplicate,
        seed=args.seed,
    )

    _log(f"\nResult: {len(documents):,} documents, {len(ground_truth):,} ground-truth queries.")

    if args.dry_run:
        _log("Dry-run mode — no files written.")
        # Print a preview
        _log("\n--- documents[0] ---")
        _log(json.dumps(documents[0], indent=2, ensure_ascii=False))
        if ground_truth:
            _log("\n--- ground_truth[0] ---")
            _log(json.dumps(ground_truth[0], indent=2, ensure_ascii=False))
        return

    # ---- Write documents -----------------------------------------------------
    docs_path = Path(args.docs_out)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    _log(f"Wrote {len(documents):,} documents → {docs_path}")

    # ---- Write ground truth --------------------------------------------------
    if not args.no_ground_truth:
        gt_path = Path(args.gt_out)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        # Evaluator expects list[{query, relevant_ids}]; drop the answer key
        gt_for_eval = [{"query": r["query"], "relevant_ids": r["relevant_ids"]} for r in ground_truth]
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_for_eval, f, ensure_ascii=False, indent=2)
        _log(f"Wrote {len(gt_for_eval):,} queries   → {gt_path}")


if __name__ == "__main__":
    main()
