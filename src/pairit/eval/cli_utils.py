"""
Shared CLI helpers for evaluation scripts.

Keeps path defaults and CSV input parsing/tagging logic in one place so
multiple scripts don't duplicate the same behavior.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from pairit._assets import ASSETS
from .config import PairitEvalConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("PAIRIT_EVAL_DATA_ROOT", str(ASSETS)))

# The MoleculeNet RF oracles bundled in this repo under
# moleculenet_property_evaluators/ were fit on the FULL dataset, test split
# included, and score ROC-AUC 1.000 on held-out data. They measure memorisation,
# not skill. They must not be the default: a run with no --evaluator-dir would
# silently produce numbers that cannot be published.
#
# The default is therefore the retrained, train-split-only set (test ROC-AUC
# 0.918 / 0.910 / 0.800). Override with --evaluator-dir or
# $PAIRIT_EVAL_EVALUATOR_DIR.
LEAKED_EVALUATOR_DIR = REPO_ROOT / "moleculenet_property_evaluators"

DEFAULT_EVALUATOR_DIR = Path(
    os.environ.get(
        "PAIRIT_EVAL_EVALUATOR_DIR",
        str(ASSETS / "oracles" / "moleculenet" / "evaluators"),
    )
)
#: Novelty reference for MoleculeNet: the REAL pairs split.
#:
#: Override with $PAIRIT_EVAL_MOLECULENET_TRAIN_DIR when evaluating a model trained
#: on a different pool (novelty should always be checked against a superset of the
#: molecules the model actually saw during training).
DEFAULT_MOLECULENET_TRAIN_DIR = Path(
    os.environ.get(
        "PAIRIT_EVAL_MOLECULENET_TRAIN_DIR",
        str(ASSETS / "data" / "moleculenet"),
    )
)
DEFAULT_MOLECULENET_TEST_DIR = Path(
    os.environ.get(
        "PAIRIT_EVAL_MOLECULENET_TEST_DIR",
        str(ASSETS / "data" / "moleculenet"),
    )
)
DEFAULT_ZINC250K_TRAIN_MOLECULES = Path(
    os.environ.get(
        "PAIRIT_EVAL_ZINC250K_TRAIN_MOLECULES",
        str(ASSETS / "data" / "zinc" / "train" / "molecules.parquet"),
    )
)
DEFAULT_ZINC250K_TEST_CSV = Path(
    os.environ.get(
        "PAIRIT_EVAL_ZINC250K_TEST_CSV",
        str(ASSETS / "data" / "zinc" / "test" / "test_subset_5k.csv"),
    )
)
DEFAULT_RESULTS_DIR = Path(os.environ.get("PAIRIT_EVAL_RESULTS_DIR", str(REPO_ROOT / "results")))

# Paper-regen workspace roots. Used by the internal/visualization/ paper-regen scripts.
# Both are env-var configurable so the same scripts can point at a fresh rerun
# directory without editing code.
DEFAULT_COMPARISON_OUTPUT_ROOT = Path(
    os.environ.get(
        "PAIRIT_EVAL_COMPARISON_OUTPUT_ROOT",
        str(DEFAULT_RESULTS_DIR / "comparison_matrix_final"),
    )
)
DEFAULT_PAPER_FIGURES_DIR = Path(
    os.environ.get(
        "PAIRIT_EVAL_PAPER_FIGURES_DIR",
        str(DEFAULT_RESULTS_DIR / "paper_figures"),
    )
)


def parse_named_inputs(raw_inputs: List[str]) -> List[Tuple[Optional[str], Path]]:
    """
    Parse positional args as either:
    - plain file path
    - method_name:path
    """
    parsed: List[Tuple[Optional[str], Path]] = []
    for inp in raw_inputs:
        if ":" in inp and not Path(inp).exists():
            name, path_str = inp.split(":", 1)
            parsed.append((name.strip(), Path(path_str.strip())))
        else:
            parsed.append((None, Path(inp)))
    return parsed


def load_input_file(path: Path, low_memory: bool = True) -> pd.DataFrame:
    """Load a CSV or Parquet file based on extension."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=low_memory)


def load_and_tag_inputs(
    inputs: List[Tuple[Optional[str], Path]],
    default_method: Optional[str] = None,
    low_memory: bool = True,
) -> pd.DataFrame:
    """
    Load one or more CSV/Parquet files and ensure a 'method' column exists.
    """
    frames = []
    for method_name, file_path in inputs:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        df = load_input_file(file_path, low_memory=low_memory)
        if "method" not in df.columns:
            df["method"] = method_name or default_method or file_path.stem
        elif method_name:
            df["method"] = method_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# Backward-compatible alias
load_and_tag_csv_inputs = load_and_tag_inputs


def path_if_exists(path: Optional[Union[Path, str]]) -> Optional[Path]:
    """Return the path if it exists, else None.

    Accepts a string as well as a Path: the CLI passes Path but the Python API
    and config files pass strings, and the str case used to raise
    AttributeError rather than doing the obvious thing.
    """
    if not path:
        return None
    p = path if isinstance(path, Path) else Path(str(path))
    try:
        return p if p.exists() else None
    except OSError:
        # An unreadable ancestor is a missing path as far as the caller is
        # concerned; it must not take the whole run down.
        return None


def build_eval_config(
    evaluator_dir: Optional[Path] = DEFAULT_EVALUATOR_DIR,
    moleculenet_train_dir: Optional[Path] = DEFAULT_MOLECULENET_TRAIN_DIR,
    moleculenet_test_dir: Optional[Path] = DEFAULT_MOLECULENET_TEST_DIR,
    zinc250k_train_molecules: Optional[Path] = DEFAULT_ZINC250K_TRAIN_MOLECULES,
    zinc250k_test_csv: Optional[Path] = DEFAULT_ZINC250K_TEST_CSV,
    select_oracle: Optional[str] = None,
    score_oracle: Optional[str] = None,
    minimol_python: Optional[Path] = None,
    offline_oracles: bool = False,
) -> PairitEvalConfig:
    """
    Build PairitEvalConfig with paths auto-disabled when they don't exist.

    `select_oracle` and `score_oracle` are passed through untouched -- they may be
    a directory, a .joblib, a prediction table, or the bare name "minimol", and
    coercing them to Path would break the last of those.
    """
    _warn_if_leaked(evaluator_dir)
    _warn_if_leaked(select_oracle)
    _warn_if_leaked(score_oracle)
    return PairitEvalConfig(
        evaluator_dir=path_if_exists(evaluator_dir),
        moleculenet_train_dir=path_if_exists(moleculenet_train_dir),
        moleculenet_test_dir=path_if_exists(moleculenet_test_dir),
        zinc250k_train_molecules=path_if_exists(zinc250k_train_molecules),
        zinc250k_test_csv=path_if_exists(zinc250k_test_csv),
        select_oracle=select_oracle,
        score_oracle=score_oracle,
        minimol_python=minimol_python,
        offline_oracles=offline_oracles,
    )


def n_generations_map(df: pd.DataFrame, n_generations: int) -> dict:
    return {ds: n_generations for ds in df["dataset"].dropna().unique()}


def _warn_if_leaked(spec) -> None:
    """Refuse to let the leaked MoleculeNet oracles be used silently.

    They were fit on their own test split and score ROC-AUC 1.000 on held-out
    data. Any number computed with them is measuring memorisation. A warning is
    the minimum; the caller has to have chosen this explicitly.
    """
    if spec is None:
        return
    try:
        resolved = Path(str(spec)).resolve()
    except (OSError, ValueError):
        return
    leaked = LEAKED_EVALUATOR_DIR.resolve()
    if resolved == leaked or leaked in resolved.parents:
        warnings.warn(
            f"{spec} is the LEAKED MoleculeNet oracle set: those models were fit on the "
            f"full dataset including the test split and score ROC-AUC 1.000 on held-out "
            f"data, so any improvement or V,I,N computed with them measures memorisation. "
            f"Use the train-split-only evaluators instead "
            f"(evaluators_train_only, test ROC-AUC 0.918 / 0.910 / 0.800).",
            stacklevel=3,
        )
