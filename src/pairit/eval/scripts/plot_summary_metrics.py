#!/usr/bin/env python3
"""Plot summary metrics saved by `scripts/summarize_results.py`.

Creates one image per dataset for (a) "other" metrics and (b) distribution metrics.

Usage:
  python scripts/plot_summary_metrics.py \
      --other results/pipeline/summary_other_print.csv \
      --dist results/pipeline/summary_distribution_print.csv \
      --out results/pipeline/figures
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_DIRECTIONS = {
    # Other metrics
    "valid": "higher",
    "novel": "higher",
    "internal_diversity": "higher",
    "improved": "higher",
    "sim_seed": "higher",
    # Distribution metrics
    "fcd": "lower",
    "snn_mean": "higher",
    "fragment_cosine": "higher",
    "scaffold_cosine": "higher",
    "coverage_ratio": "higher",
    "w1_macro_mean": "lower",
    # gen LL - ref LL; less negative is better
    "kde_ll_delta_macro_mean": "higher",
}


def _build_method_palette(methods: list[str]) -> tuple[dict[str, tuple[float, float, float]], list[str]]:
    # Keep methods in appearance order to stay consistent with summary CSVs.
    ordered_methods = list(dict.fromkeys(methods))
    pairit_methods = [m for m in ordered_methods if m.startswith("pairit")]
    baseline_methods = [m for m in ordered_methods if not m.startswith("pairit")]

    palette: dict[str, tuple[float, float, float]] = {}
    base_colors = sns.color_palette("Blues", n_colors=max(len(baseline_methods) + 2, 3))[2:]
    pairit_colors = sns.color_palette("Oranges", n_colors=max(len(pairit_methods) + 2, 3))[2:]

    for m, c in zip(baseline_methods, base_colors):
        palette[m] = c
    for m, c in zip(pairit_methods, pairit_colors):
        palette[m] = c
    return palette, ordered_methods


def _is_better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "higher" else a < b


def plot_metrics(df: pd.DataFrame, metrics: list[str], dataset: str, out_dir: Path, title_suffix: str):
    df = df[df["dataset"] == dataset].copy()
    if df.empty:
        return

    palette, method_order = _build_method_palette(df["method"].tolist())
    n_metrics = len(metrics)
    ncols = 3
    nrows = ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric not in df.columns:
            ax.text(0.5, 0.5, f"{metric}\n(not present)", ha="center", va="center")
            ax.set_axis_off()
            continue

        direction = METRIC_DIRECTIONS.get(metric, "higher")
        metric_df = df[["method", metric]].dropna(subset=[metric]).copy()
        if metric_df.empty:
            ax.text(0.5, 0.5, f"{metric}\n(no data)", ha="center", va="center")
            ax.set_axis_off()
            continue

        sns.barplot(
            x="method",
            y=metric,
            data=metric_df,
            ax=ax,
            order=method_order,
            hue="method",
            hue_order=method_order,
            palette=palette,
            dodge=False,
            legend=False,
        )
        ax.set_title(f"{metric}\n({direction} is better)")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)

        # Dashed reference line at the best PairIt value for this metric.
        pairit_vals = metric_df[metric_df["method"].str.startswith("pairit")][metric]
        baseline_vals = metric_df[~metric_df["method"].str.startswith("pairit")][metric]
        if not pairit_vals.empty:
            best_pairit = pairit_vals.max() if direction == "higher" else pairit_vals.min()
            ax.axhline(best_pairit, color="#D55E00", linestyle="--", linewidth=1.4, alpha=0.9)
            ax.text(
                0.02,
                0.98,
                f"Best PairIt: {best_pairit:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#8C3C00",
            )
            if not baseline_vals.empty:
                best_baseline = baseline_vals.max() if direction == "higher" else baseline_vals.min()
                status = "better" if _is_better(best_pairit, best_baseline, direction) else "worse"
                if best_pairit == best_baseline:
                    status = "tie"
                ax.text(
                    0.02,
                    0.89,
                    f"vs best baseline: {status}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#4A4A4A",
                )

    # turn off unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_axis_off()

    plt.suptitle(f"{dataset} — {title_suffix}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{title_suffix.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--other", type=Path, required=True, help="CSV with other metrics")
    p.add_argument("--dist", type=Path, required=True, help="CSV with distribution metrics")
    p.add_argument("--out", type=Path, default=Path("results/pipeline/figures"))
    args = p.parse_args()

    other = pd.read_csv(args.other)
    dist = pd.read_csv(args.dist)

    # Ensure method/dataset columns present
    for df in (other, dist):
        if "method" not in df.columns or "dataset" not in df.columns:
            raise SystemExit("Input CSVs must contain 'method' and 'dataset' columns")

    datasets = sorted(set(other["dataset"].dropna().unique().tolist() + dist["dataset"].dropna().unique().tolist()))
    # Define metrics to plot (exclude method/dataset)
    other_metrics = [c for c in other.columns if c not in ("method", "dataset")]
    dist_metrics = [c for c in dist.columns if c not in ("method", "dataset")]

    for ds in datasets:
        print(f"Plotting dataset: {ds}")
        plot_metrics(other, other_metrics, ds, args.out, "other_metrics")
        plot_metrics(dist, dist_metrics, ds, args.out, "distribution_metrics")


if __name__ == "__main__":
    main()
