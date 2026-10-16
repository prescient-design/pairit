#!/usr/bin/env python3
"""
Print LaTeX table* blocks for all 4 comparison experiments, matching the
reference paper format:

  Columns (in order):
    Joint (v_n_i) | Valid | Novel | Unique | Diverse | Improved | SimSeed
    FCD↓ | SNN↑ | FragCos↑ | ScafCos↑ | Cov↑ | W1↓ | ΔLL↑

  One table per dataset (bace / bbbp / hiv) per experiment.
  4 experiments × 3 datasets = 12 tables total.

  Requires: \\usepackage{booktabs,makecell,adjustbox}

Usage:
    python scripts/print_latex_tables.py
    python scripts/print_latex_tables.py --output-root results/comparison_matrix_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from pairit.eval.extra_baselines import extra_baselines, insert_after

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DATASETS = ["bace", "bbbp", "hiv", "herg"]

# ---------------------------------------------------------------------------
# Column specification
# (csv_col, latex_header, fmt, direction)
# ---------------------------------------------------------------------------
COLUMNS: List[Tuple[str, str, str, str]] = [
    ("joint_metric",             r"\makecell{Joint \\ (v\_n\_i)}",     ".2f", "higher"),
    ("valid",                    "Valid",                                ".2f", "higher"),
    ("novel",                    "Novel",                                ".2f", "higher"),
    ("unique",                   "Unique",                               ".2f", "higher"),
    ("internal_diversity",       "Diverse",                              ".2f", "higher"),
    ("improved",                 "Improved",                             ".2f", "higher"),
    ("sim_seed",                 "SimSeed",                              ".2f", "higher"),
    ("fcd",                      r"FCD$\downarrow$",                     ".2f", "lower"),
    ("snn_mean",                 r"SNN$\uparrow$",                       ".2f", "higher"),
    ("fragment_cosine",          r"FragCos$\uparrow$",                   ".2f", "higher"),
    ("scaffold_cosine",          r"ScafCos$\uparrow$",                   ".2f", "higher"),
    ("coverage_ratio",           r"Cov$\uparrow$",                       ".2f", "higher"),
    ("w1_macro_mean",            r"W1$\downarrow$",                      ".2f", "lower"),
    ("kde_ll_delta_macro_mean",  r"$\Delta$LL$\uparrow$",                ".2f", "higher"),
]

# ---------------------------------------------------------------------------
# Method ordering and display names
# ---------------------------------------------------------------------------
# Baselines (printed first, in this order, HierG2G commented out)
BASELINE_METHODS: Dict[str, List[str]] = {
    "real-only": [
        "nn-lookup-real-only-{sel}",
        "ddsbm-bridge-real-only-{sel}",
        "ddsbm-sb-real-only-{sel}",
        "hierg2g-real-only-{sel}",
        "mmp-prop-{sel}",
        "nfbo-real-only-{sel}",
    ],
}

# PairIT methods (printed after \hline)
# All PairIT variants (base, Beam, RL) go through the SAME best-of-64 + top-5
# selection protocol as every baseline (n_gen=64 → all64 → first5/best64).
PAIRIT_METHODS: Dict[str, Dict[str, List[str]]] = {
    "real-only": {
        "first5": [
            "pairit-real-only-first5",
            "pairit-beam-real-only-first5",
            "pairit-rl-real-only-first5",
        ],
        "best64": [
            "pairit-real-only-best64",
            "pairit-beam-real-only-best64",
            "pairit-rl-real-only-best64",
        ],
    },
}

DISPLAY_NAME: Dict[str, str] = {
    # baselines
    "nn-lookup-real-only-first5":          "NN Lookup",
    "nn-lookup-real-only-best64":          "NN Lookup",
    "ddsbm-bridge-real-only-first5":       "DSSBM (bridge)",
    "ddsbm-bridge-real-only-best64":       "DSSBM (bridge)",
    "ddsbm-sb-real-only-first5":           "DSSBM (sb)",
    "ddsbm-sb-real-only-best64":           "DSSBM (sb)",
    "hierg2g-real-only-first5":            "HierG2G",
    "hierg2g-real-only-best64":            "HierG2G",
    "mmp-prop-first5":                     "MMP",
    "mmp-prop-best64":                     "MMP",
    "nfbo-real-only-first5":               "NF-BO",
    "nfbo-real-only-best64":               "NF-BO",
    # pairit
    "pairit-real-only-first5":             r"\textbf{PairIT}",
    "pairit-real-only-best64":             r"\textbf{PairIT}",
    "pairit-beam-real-only-first5":        r"\textbf{PairIT-Beam}",
    "pairit-beam-real-only-best64":        r"\textbf{PairIT-Beam}",
    "pairit-rl-real-only-first5":          r"\textbf{PairIT-RL}",
    "pairit-rl-real-only-best64":          r"\textbf{PairIT-RL}",
}

HIERG2G_KEYS = {
    "hierg2g-real-only-first5", "hierg2g-real-only-best64",
}

DATASET_CAPTION: Dict[str, str] = {
    "bace": "MoleculeNet BACE bioactivity improvement ($0 \\rightarrow 1$) under seeded generation.",
    "bbbp": "MoleculeNet BBBP bioactivity improvement ($0 \\rightarrow 1$) under seeded generation.",
    "hiv":  "MoleculeNet HIV bioactivity improvement ($0 \\rightarrow 1$) under seeded generation.",
    "herg": "TDC hERG blockade reduction ($1 \\rightarrow 0$) under seeded generation.",
}


def _baseline_templates(train_regime: str) -> List[str]:
    templates = BASELINE_METHODS[train_regime]
    for extra in extra_baselines():
        templates = insert_after(
            templates, f"{extra.after_family}-real-only-{{sel}}", f"{extra.family}-real-only-{{sel}}"
        )
    return templates


def _display_name(method: str) -> str:
    for extra in extra_baselines():
        if method in extra.latex_names:
            return extra.latex_names[method]
    return DISPLAY_NAME.get(method, method)


def _resolve_baseline_methods(train_regime: str, selection_mode: str) -> List[str]:
    sel = selection_mode  # "first5" or "best64"
    return [t.format(sel=sel) for t in _baseline_templates(train_regime)]


def _compute_unique(per_gen: pd.DataFrame) -> pd.DataFrame:
    """Unique valid SMILES / total generated, per (method, dataset), as percent."""
    rows = []
    for (method, dataset), grp in per_gen.groupby(["method", "dataset"], sort=False):
        n_total = len(grp)
        valid = grp[grp["is_valid"].fillna(False)]
        n_unique = valid["generated_smiles"].nunique()
        rows.append({
            "method": method,
            "dataset": dataset,
            "unique": 100.0 * n_unique / n_total if n_total > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def _fmt(val: object, fmt: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "--"
    return f"{float(val):{fmt}}"


def _best_method(
    methods: List[str],
    col: str,
    direction: str,
    avg: pd.DataFrame,
    dataset: str,
) -> Optional[str]:
    sub = avg[(avg["dataset"] == dataset) & (avg["method"].isin(methods))]
    sub = sub[sub[col].notna()] if col in sub.columns else pd.DataFrame()
    if len(sub) == 0:
        return None
    idx = sub[col].idxmax() if direction == "higher" else sub[col].idxmin()
    return str(sub.loc[idx, "method"])


def _print_table(
    dataset: str,
    train_regime: str,
    selection_mode: str,
    baseline_methods: List[str],
    pairit_methods: List[str],
    avg: pd.DataFrame,
    exp_label: str,
) -> None:
    all_methods = baseline_methods + pairit_methods
    n_data_cols = len(COLUMNS)
    col_spec = "l" + "c" * n_data_cols

    # Collect best method per column (across all methods, for underline)
    best: Dict[str, Optional[str]] = {}
    for col, _, _, direction in COLUMNS:
        best[col] = _best_method(all_methods, col, direction, avg, dataset)

    def _row(method: str) -> str:
        sub = avg[(avg["dataset"] == dataset) & (avg["method"] == method)]
        display = _display_name(method)
        cells = [display]
        for col, _, fmt, _ in COLUMNS:
            val = None
            if len(sub) > 0 and col in sub.columns:
                v = sub.iloc[0][col]
                if pd.notna(v):
                    val = float(v)
            cell = _fmt(val, fmt)
            if val is not None and best.get(col) == method:
                cell = r"\underline{" + cell + "}"
            cells.append(cell)
        return " & ".join(cells) + r" \\"

    # Column header
    headers = [col[1] for col in COLUMNS]
    label = f"tab:{dataset}_{train_regime.replace('+', 'plus')}_{selection_mode}"

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(f"\\caption{{{DATASET_CAPTION[dataset]} {exp_label}.}}")
    print(f"\\label{{{label}}}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    print("Model & " + " & ".join(headers) + r" \\")
    print(r"\midrule")

    for method in baseline_methods:
        row = _row(method)
        if method in HIERG2G_KEYS:
            # Commented out as in reference
            for line in row.split("\n"):
                print("% " + line)
        else:
            print(row)

    print(r"\hline")

    for method in pairit_methods:
        print(_row(method))

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table*}")


# ---------------------------------------------------------------------------
# Columns for the std table (real-only best64 only): remove W1 and ΔLL,
# add std for regular metrics, no std for distribution metrics.
# (csv_col, latex_header, fmt, direction, per_seed_col_for_std)
# ---------------------------------------------------------------------------
COLUMNS_STD: List[Tuple[str, str, str, str, Optional[str]]] = [
    ("joint_metric",        r"\makecell{Joint \\ (v\_n\_i)}",    ".2f", "higher", "v_n_i"),
    ("valid",               "Valid",                               ".2f", "higher", "valid"),
    ("novel",               "Novel",                               ".2f", "higher", "novel"),
    ("unique",              "Unique",                              ".2f", "higher", None),
    ("internal_diversity",  "Diverse",                             ".2f", "higher", "internal_diversity"),
    ("improved",            "Improved",                            ".2f", "higher", "improved"),
    ("sim_seed",            "SimSeed",                             ".2f", "higher", "sim_seed"),
    ("fcd",                 r"FCD$\downarrow$",                    ".2f", "lower",  None),
    ("snn_mean",            r"SNN$\uparrow$",                      ".2f", "higher", None),
    ("fragment_cosine",     r"FragCos$\uparrow$",                  ".2f", "higher", None),
    ("scaffold_cosine",     r"ScafCos$\uparrow$",                  ".2f", "higher", None),
    ("coverage_ratio",      r"Cov$\uparrow$",                      ".2f", "higher", None),
]


def _compute_std_lookup(per_seed: pd.DataFrame, dataset: str, train_regime: str, selection_mode: str) -> Dict[Tuple[str, str], float]:
    """Compute per-seed std (ddof=1) for each (method, metric) pair."""
    std_metrics = ["v_n_i", "valid", "novel", "improved", "sim_seed", "internal_diversity"]
    # Include train_regime-agnostic methods (e.g. MMP) alongside the target regime
    sub = per_seed[
        (per_seed["dataset"] == dataset) &
        (per_seed["train_regime"].isin([train_regime, "agnostic"])) &
        (per_seed["selection_mode"] == selection_mode)
    ].copy()
    result: Dict[Tuple[str, str], float] = {}
    for method, grp in sub.groupby("method", sort=False):
        for metric in std_metrics:
            if metric not in grp.columns:
                continue
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
            if len(vals) > 1:
                result[(str(method), metric)] = float(vals.std(ddof=1))
            elif len(vals) == 1:
                result[(str(method), metric)] = 0.0
    return result


def _fmt_std(val: object, std_val: Optional[float], fmt: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "--"
    s = f"{float(val):{fmt}}"
    if std_val is not None and not pd.isna(std_val):
        s += r"{\scriptsize$\pm$" + f"{float(std_val):{fmt}}" + r"}"
    return s


def _print_table_std(
    dataset: str,
    train_regime: str,
    selection_mode: str,
    baseline_methods: List[str],
    pairit_methods: List[str],
    avg: pd.DataFrame,
    exp_label: str,
    std_lookup: Dict[Tuple[str, str], float],
) -> None:
    all_methods = baseline_methods + pairit_methods
    n_data_cols = len(COLUMNS_STD)
    col_spec = "l" + "c" * n_data_cols

    best: Dict[str, Optional[str]] = {}
    for col, _, _, direction, _ in COLUMNS_STD:
        best[col] = _best_method(all_methods, col, direction, avg, dataset)

    def _row(method: str) -> str:
        sub = avg[(avg["dataset"] == dataset) & (avg["method"] == method)]
        display = _display_name(method)
        cells = [display]
        for col, _, fmt, _, per_seed_col in COLUMNS_STD:
            val = None
            if len(sub) > 0 and col in sub.columns:
                v = sub.iloc[0][col]
                if pd.notna(v):
                    val = float(v)
            std_val = None
            if per_seed_col is not None:
                std_val = std_lookup.get((method, per_seed_col))
            cell = _fmt_std(val, std_val, fmt)
            if val is not None and best.get(col) == method:
                cell = r"\underline{" + cell + "}"
            cells.append(cell)
        return " & ".join(cells) + r" \\"

    headers = [col[1] for col in COLUMNS_STD]
    label = f"tab:{dataset}_{train_regime.replace('+', 'plus')}_{selection_mode}_std"

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(f"\\caption{{{DATASET_CAPTION[dataset]} {exp_label}.}}")
    print(f"\\label{{{label}}}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    print("Model & " + " & ".join(headers) + r" \\")
    print(r"\midrule")

    for method in baseline_methods:
        row = _row(method)
        if method in HIERG2G_KEYS:
            for line in row.split("\n"):
                print("% " + line)
        else:
            print(row)

    print(r"\hline")

    for method in pairit_methods:
        print(_row(method))

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table*}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print LaTeX tables for all 4 comparison experiments.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/comparison_matrix_final"),
    )
    parser.add_argument(
        "--std",
        action="store_true",
        help="Print std tables for real-only best64 only (3 tables).",
    )
    args = parser.parse_args()
    root = args.output_root

    dist_avg_path = root / "distribution" / "distribution_average.csv"
    per_gen_path  = root / "regular" / "regular_per_generation.csv"
    per_seed_path = root / "regular" / "regular_per_seed.csv"

    if not dist_avg_path.exists():
        print(f"ERROR: {dist_avg_path} not found.", file=sys.stderr)
        return 1

    avg = pd.read_csv(dist_avg_path)
    if "joint_metric" not in avg.columns and "v_n_i" in avg.columns:
        avg["joint_metric"] = avg["v_n_i"]

    # Uniqueness comes from the eval-side `unique` column in the average CSV
    # (per-seed Unique@5 — see pairit.eval.evaluate:_per_seed_unique_at_5).
    # Do NOT re-merge a locally computed value: the older cross-seed pool
    # formula uses a different denominator and will not match the main tables.
    if "unique" not in avg.columns:
        print(f"Warning: distribution_average.csv missing 'unique' column — Unique column will show --", file=sys.stderr)
        avg["unique"] = float("nan")

    if args.std:
        if not per_seed_path.exists():
            print(f"ERROR: {per_seed_path} not found.", file=sys.stderr)
            return 1
        per_seed = pd.read_csv(per_seed_path)
        train_regime, selection_mode = "real-only", "best64"
        exp_label = "Training: real-only, best of 64"
        baseline_methods = _resolve_baseline_methods(train_regime, selection_mode)
        pairit_methods   = PAIRIT_METHODS[train_regime][selection_mode]

        print("% LaTeX tables with std — real-only | best64")
        print("% Requires: \\usepackage{booktabs,makecell,adjustbox}")
        print("% Underlined values = best in column. Subscript = std across seeds (ddof=1).")
        print(f"\n% {'=' * 60}")
        print(f"% EXPERIMENT: {train_regime} | {selection_mode} (with std)")
        print(f"% {'=' * 60}")

        for dataset in DATASETS:
            std_lookup = _compute_std_lookup(per_seed, dataset, train_regime, selection_mode)
            print()
            _print_table_std(
                dataset, train_regime, selection_mode,
                baseline_methods, pairit_methods,
                avg, exp_label, std_lookup,
            )
        return 0

    experiments = [
        ("real-only",      "first5",  "Training: real-only, 5 generations"),
        ("real-only",      "best64",  "Training: real-only, best of 64"),
    ]

    print("% LaTeX tables — generated by scripts/print_latex_tables.py")
    print("% Requires: \\usepackage{booktabs,makecell,adjustbox}")
    print("% Underlined values = best in column.")

    for train_regime, selection_mode, exp_label in experiments:
        baseline_methods = _resolve_baseline_methods(train_regime, selection_mode)
        pairit_methods   = PAIRIT_METHODS[train_regime][selection_mode]

        print(f"\n\n% {'=' * 60}")
        print(f"% EXPERIMENT: {train_regime} | {selection_mode}")
        print(f"% {'=' * 60}")

        for dataset in DATASETS:
            print()
            _print_table(
                dataset, train_regime, selection_mode,
                baseline_methods, pairit_methods,
                avg, exp_label,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
