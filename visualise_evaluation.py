"""
Visualise evaluation results produced by run_evaluation.py.

Reads
-----
--json PATH   Aggregate JSON written by run_evaluation.py --out-json
              (maps pipeline name → MetricsResult dict)
--csv  PATH   Per-query CSV written by run_evaluation.py --out-csv
              (one row per query × pipeline)

Writes PNG files to --out-dir (default: results/):
  metrics_bar.png      Grouped bar chart: aggregate metrics per pipeline
  radar.png            Radar chart: multi-metric pipeline comparison
  latency_bar.png      Mean query latency per pipeline          (needs --csv)
  metrics_boxplot.png  Per-query metric distributions           (needs --csv)

Usage
-----
    # After running:
    #   python run_evaluation.py ... --out-csv results/eval.csv --out-json results/eval.json

    python visualise_evaluation.py \\
        --json results/eval.json \\
        --csv  results/eval.csv \\
        --out-dir results/plots

    # Show interactively instead of (or in addition to) saving:
    python visualise_evaluation.py --json results/eval.json --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Backend selection must happen before pyplot is imported.
# We do a quick argv scan so the rest of the module can import pyplot normally.
# ---------------------------------------------------------------------------

import matplotlib as _mpl
if "--show" not in sys.argv:
    _mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# (display label, JSON key, CSV column)
METRICS: list[tuple[str, str, str]] = [
    ("Recall",    "recall_at_k",    "recall"),
    ("Precision", "precision_at_k", "precision"),
    ("MRR",       "mrr",            "mrr"),
    ("NDCG",      "ndcg_at_k",      "ndcg"),
    ("MAP",       "map_at_k",       "map"),
    ("HitRate",   "hit_rate_at_k",  "hit_rate"),
]


def _pipeline_colors(pipelines: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(pipelines)}


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path):
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required for CSV-based plots. pip install pandas", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def _save_or_show(fig: plt.Figure, path: Path | None, show: bool) -> None:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — grouped bar: aggregate metrics
# ---------------------------------------------------------------------------

def plot_metrics_bar(data: dict, k: int, out_path: Path | None, show: bool) -> None:
    """One group per metric, one bar per pipeline."""
    pipelines     = list(data.keys())
    labels        = [m[0] for m in METRICS]
    json_keys     = [m[1] for m in METRICS]
    n_metrics     = len(labels)
    n_pipelines   = len(pipelines)
    bar_width     = 0.8 / n_pipelines
    x             = np.arange(n_metrics)
    colors        = _pipeline_colors(pipelines)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.8), 5))

    for i, pipeline in enumerate(pipelines):
        offsets = x + (i - n_pipelines / 2 + 0.5) * bar_width
        values  = [data[pipeline].get(key, 0.0) for key in json_keys]
        ax.bar(offsets, values, width=bar_width, label=pipeline, color=colors[pipeline])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}@{k}" if lbl not in ("MRR",) else lbl for lbl in labels], fontsize=11)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Score")
    ax.set_title(f"Aggregate retrieval metrics by pipeline  (K={k})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_or_show(fig, out_path, show)


# ---------------------------------------------------------------------------
# Plot 2 — radar chart: aggregate metrics
# ---------------------------------------------------------------------------

def plot_radar(data: dict, k: int, out_path: Path | None, show: bool) -> None:
    """One polygon per pipeline across all aggregate metrics."""
    pipelines   = list(data.keys())
    labels      = [m[0] for m in METRICS]
    json_keys   = [m[1] for m in METRICS]
    n           = len(labels)
    colors      = _pipeline_colors(pipelines)

    angles      = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles     += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for pipeline in pipelines:
        values  = [data[pipeline].get(key, 0.0) for key in json_keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=pipeline, color=colors[pipeline])
        ax.fill(angles, values, alpha=0.08, color=colors[pipeline])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(f"Pipeline comparison — radar chart  (K={k})", fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    _save_or_show(fig, out_path, show)


# ---------------------------------------------------------------------------
# Plot 3 — latency bar (from CSV)
# ---------------------------------------------------------------------------

def plot_latency_bar(df, out_path: Path | None, show: bool) -> None:
    """Horizontal bar chart of mean query latency per pipeline."""
    latency = df.groupby("pipeline")["latency_ms"].mean().sort_values()
    colors  = _pipeline_colors(list(latency.index))

    fig, ax = plt.subplots(figsize=(8, max(3, len(latency) * 0.7)))
    bars = ax.barh(
        latency.index,
        latency.values,
        color=[colors[p] for p in latency.index],
    )
    ax.bar_label(bars, fmt="%.1f ms", padding=5, fontsize=9)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title("Mean query latency by pipeline")
    ax.set_xlim(0, latency.max() * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_or_show(fig, out_path, show)


# ---------------------------------------------------------------------------
# Plot 4 — box plots: per-query metric distributions (from CSV)
# ---------------------------------------------------------------------------

def plot_metrics_boxplot(df, k: int, out_path: Path | None, show: bool) -> None:
    """One subplot per metric; box plots grouped by pipeline."""
    pipelines  = df["pipeline"].unique().tolist()
    colors     = _pipeline_colors(pipelines)
    n_metrics  = len(METRICS)
    ncols      = 3
    nrows      = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8))
    axes = axes.flatten()

    for idx, (label, _, csv_col) in enumerate(METRICS):
        ax = axes[idx]
        per_pipeline = [df[df["pipeline"] == p][csv_col].dropna().values for p in pipelines]

        bp = ax.boxplot(
            per_pipeline,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, pipeline in zip(bp["boxes"], pipelines):
            patch.set_facecolor(colors[pipeline])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(pipelines) + 1))
        ax.set_xticklabels(pipelines, rotation=40, ha="right", fontsize=7)
        short_label = label if label == "MRR" else f"{label}@{k}"
        ax.set_title(short_label, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Per-query metric distributions by pipeline  (K={k})", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, out_path, show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise retrieval evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json", metavar="PATH",
                        help="Aggregate JSON written by run_evaluation.py --out-json.")
    parser.add_argument("--csv",  metavar="PATH",
                        help="Per-query CSV written by run_evaluation.py --out-csv.")
    parser.add_argument("--out-dir", default="results", metavar="DIR",
                        help="Directory where PNG plots are written.")
    parser.add_argument("--show", action="store_true",
                        help="Display each plot interactively (in addition to saving).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json and not args.csv:
        print("ERROR: provide at least one of --json or --csv.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_data = _load_json(Path(args.json)) if args.json else None
    csv_df    = _load_csv(Path(args.csv))   if args.csv  else None

    # Determine K from the JSON or fall back to "?"
    k: int | str = "?"
    if json_data:
        first = next(iter(json_data.values()), {})
        k = first.get("k", "?")

    print(f"Generating plots → {out_dir}/")

    # --- JSON-based plots ---
    if json_data is not None:
        plot_metrics_bar(json_data, k, out_dir / "metrics_bar.png",  args.show)
        plot_radar(      json_data, k, out_dir / "radar.png",         args.show)

    # --- CSV-based plots ---
    if csv_df is not None:
        plot_latency_bar(    csv_df,     out_dir / "latency_bar.png",      args.show)
        plot_metrics_boxplot(csv_df, k,  out_dir / "metrics_boxplot.png",  args.show)

    print("Done.")


if __name__ == "__main__":
    main()
