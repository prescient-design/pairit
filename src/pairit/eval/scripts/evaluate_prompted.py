"""Explicit-input CLI for raw-attempt matched-pair prompting evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pairit.eval.decisions import file_facts
from pairit.eval.oracles import FingerprintJoblibOracle
from pairit.eval.prompted import TRUTH_COLUMNS, evaluate_prompted_generations


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"pair_id": str}, keep_default_na=False)
    raise ValueError(f"Expected an explicit .csv or .parquet input: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True, help="Independent expected request roster, including original raw truth")
    parser.add_argument("--training-reference", type=Path, required=True, help="Fine-tuning training molecules used for novelty")
    parser.add_argument("--training-smiles-column", required=True)
    parser.add_argument("--attempts-per-request", type=int, required=True)
    parser.add_argument("--activity-property", help="Raw request property name, e.g. target; selects measured 0-to-1 pairs")
    parser.add_argument("--activity-oracle", type=Path, help="Explicit training-only fingerprint RF .joblib")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; must not already exist")
    args = parser.parse_args(argv)
    if bool(args.activity_property) != bool(args.activity_oracle):
        parser.error("--activity-property and --activity-oracle must be supplied together")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    reference = _read(args.training_reference)
    if args.training_smiles_column not in reference:
        parser.error(f"Missing reference column {args.training_smiles_column!r}")
    oracle = None
    if args.activity_oracle is not None:
        oracle = FingerprintJoblibOracle.from_path(args.activity_oracle, direction="increasing", threshold=0.5, comparison=">")
    result = evaluate_prompted_generations(
        _read(args.generations), _read(args.requests),
        training_smiles=reference[args.training_smiles_column].tolist(),
        attempts_per_request=args.attempts_per_request,
        activity_property=args.activity_property, activity_oracle=oracle,
    )
    result.summary["inputs"] = {
        name: file_facts(getattr(args, name).resolve())
        for name in ("generations", "requests", "training_reference")
    }
    result.summary["inputs"]["training_smiles_column"] = args.training_smiles_column
    args.output_dir.mkdir(parents=True)
    # CSV metadata remains portable JSON, never Python list repr.
    for column in TRUTH_COLUMNS:
        result.per_generation[column] = result.per_generation[column].map(json.dumps)
    result.per_generation.to_csv(args.output_dir / "per_generation.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result.summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result.summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
