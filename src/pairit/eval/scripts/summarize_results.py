#!/usr/bin/env python3
"""
Read processed average CSVs and print paper-ready summary tables.

Examples:
  python scripts/summarize_results.py
  python scripts/summarize_results.py --output-root results/pipeline --dataset bace
  python scripts/summarize_results.py --methods nfbo ddsbm-bridge ddsbm-sb --metrics method,dataset,valid,v_n_i,sim_seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if _REPO not in sys.path:
    sys.path.insert(0, str(_REPO))

from pairit.eval.pipeline import DEFAULT_SUMMARY_METRICS, load_average_results, summarize_average


def main() -> int:
    parser = argparse.ArgumentParser(description="Print summary from processed *_average.csv files.")
    parser.add_argument("--output-root", type=Path, default=Path("results/pipeline"))
    parser.add_argument("--methods", nargs="*", default=None, help="Method list (canonical or legacy aliases).")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_SUMMARY_METRICS),
        help="Comma-separated average-table columns to print.",
    )
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional path to save summary CSV.")
    args = parser.parse_args()

    avg = load_average_results(args.output_root, methods=args.methods)
    if len(avg) == 0:
        print(f"No *_average.csv files found under {args.output_root}")
        return 1

    if args.dataset:
        avg = avg[avg["dataset"] == args.dataset].copy()
    if len(avg) == 0:
        print("No rows left after filtering.")
        return 1

    metric_cols = [c.strip() for c in args.metrics.split(",") if c.strip()]
    summary = summarize_average(avg, metrics=metric_cols)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.save_csv, index=False)
        print(f"\nSaved: {args.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

