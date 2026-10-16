"""
Normalize raw generation tables to a canonical seed × gen grid.

Seed ordering is determined by the reference test set for each dataset
(MoleculeNet real_pairs or zinc250k_test_icml). Seeds are matched by
canonical SMILES and assigned contiguous seed_idx values in reference order.
gen_idx is remapped to 0..N-1. Missing (seed, gen) slots are filled with
NaN generated_smiles so every method occupies the same grid shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rdkit import Chem

from .config import DATASET_CONFIG, MOLECULENET_DATASETS, PairitEvalConfig


REQUIRED_COLUMNS = [
    "method",
    "dataset",
    "seed_idx",
    "seed_smiles",
    "seed_label",
    "gen_idx",
    "generated_smiles",
]


def _clean_generated_smiles(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _canonicalize_smiles(s: Any) -> Optional[str]:
    if pd.isna(s) or not str(s).strip():
        return None
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _load_reference_seed_df(dataset: str, config: PairitEvalConfig, seed_label: int = 0) -> pd.DataFrame:
    if dataset in MOLECULENET_DATASETS:
        if not config.moleculenet_test_dir:
            raise ValueError(f"moleculenet_test_dir is required for dataset '{dataset}'")
        label_col = config.dataset_config.get(dataset, {}).get("label_col", DATASET_CONFIG[dataset]["label_col"])
        # The bundled assets use `molecules.parquet`; older research trees use
        # `molecules_<dataset>_real.parquet`. Accept either.
        split_dir = Path(config.moleculenet_test_dir) / dataset / "test"
        candidates = [
            split_dir / "molecules.parquet",
            split_dir / f"molecules_{dataset}_real.parquet",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No molecule table for '{dataset}' test split; looked for "
                + " and ".join(str(p) for p in candidates)
            )
        df = pd.read_parquet(path)
        if label_col in df.columns:
            df = df[df[label_col] == seed_label]
        out = df[["smiles"]].copy().reset_index(drop=True)
        out["seed_smiles"] = out["smiles"].astype(str)
        out["seed_label"] = seed_label
        out["seed_idx"] = range(len(out))
        return out[["seed_idx", "seed_smiles", "seed_label"]]

    if dataset == "zinc250k":
        if not config.zinc250k_test_csv:
            raise ValueError("zinc250k_test_csv is required for dataset 'zinc250k'")
        df = pd.read_csv(config.zinc250k_test_csv)
        out = df[["smiles"]].copy().reset_index(drop=True)
        out["seed_smiles"] = out["smiles"].astype(str)
        out["seed_label"] = 0
        out["seed_idx"] = range(len(out))
        return out[["seed_idx", "seed_smiles", "seed_label"]]

    # Any other dataset: take the test-set path from its config entry, the same
    # escape hatch `train_smiles` provides for novelty. Without this the TDC
    # datasets cannot be normalized at all and --no-normalize is their only
    # working path, which silently excludes them from every normalized table.
    ds_config = (config.dataset_config or {}).get(dataset, {}) or {}
    test_files = ds_config.get("test_smiles")
    if test_files:
        if isinstance(test_files, (str, Path)):
            test_files = [test_files]
        frames = []
        smiles_col = ds_config.get("smiles_col", "smiles")
        for f in test_files:
            path = Path(f)
            if not path.exists():
                continue
            part = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            label_col = ds_config.get("label_col")
            if label_col and label_col in part.columns:
                part = part[part[label_col] == seed_label]
            frames.append(part[[smiles_col]].rename(columns={smiles_col: "smiles"}))
        if frames:
            df = pd.concat(frames, ignore_index=True).drop_duplicates("smiles")
            out = df.reset_index(drop=True)
            out["seed_smiles"] = out["smiles"].astype(str)
            out["seed_label"] = seed_label
            out["seed_idx"] = range(len(out))
            return out[["seed_idx", "seed_smiles", "seed_label"]]

    raise ValueError(
        f"No canonical seed set for dataset {dataset!r}. Either add a 'test_smiles' path to "
        f"its DATASET_CONFIG entry, or pass --no-normalize to score the file as it stands."
    )


def _normalize_gen_idx(series: pd.Series) -> pd.Series:
    vals = [v for v in series.dropna().tolist()]
    unique_vals = sorted({int(v) for v in vals})
    mapping = {old: new for new, old in enumerate(unique_vals)}
    return series.map(lambda x: mapping.get(int(x)) if pd.notna(x) else x)


def normalize_generation_block(
    df: pd.DataFrame,
    dataset: str,
    config: PairitEvalConfig,
    n_generations: Optional[int] = None,
    seed_label: int = 0,
    pad_missing: bool = True,
) -> pd.DataFrame:
    """
    Normalize one dataset block onto the canonical seed x generation grid.

    - seed order comes from the real class-0 test set
    - seed_idx is rebuilt from that canonical order
    - gen_idx is remapped to a contiguous 0..N-1 block
    - when `pad_missing=True` (default): missing seeds/generations are filled
      with NaN generated_smiles rows so every (method, seed, gen_idx) slot
      exists — needed for the raw-64 pass where some methods produce fewer
      than 64 native gens and we still want a 64-wide
      denominator per seed.
    - when `pad_missing=False`: skip the padding merge; keep only the rows
      that actually exist. Correct for the top-5 pass, where
      `_select_best5_joint` intentionally returns <5 rows for seeds whose
      64 gens contained fewer than 5 unique valid canonicals ("short seeds").
      Padding those with NaN placeholders contaminates VIN denominators by
      0-8pp.
    """
    block = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in block.columns:
            raise ValueError(f"Missing required column '{col}'")
    if len(block) == 0:
        return block

    block = block[block["dataset"] == dataset].copy()
    if len(block) == 0:
        return block

    block["seed_smiles_raw"] = block["seed_smiles"]
    block["seed_smiles_canon"] = block["seed_smiles"].apply(_canonicalize_smiles)
    block["generated_smiles"] = block["generated_smiles"].apply(_clean_generated_smiles)
    block["gen_idx"] = _normalize_gen_idx(block["gen_idx"]).astype("Int64")

    ref = _load_reference_seed_df(dataset, config, seed_label=seed_label).copy()
    ref["seed_smiles_canon"] = ref["seed_smiles"].apply(_canonicalize_smiles)

    # Match only canonicalizable seeds. Unmatched rows are dropped and represented by NaN grid rows.
    ref_match = ref.dropna(subset=["seed_smiles_canon"]).drop_duplicates(subset=["seed_smiles_canon"])
    block = block.merge(
        ref_match[["seed_idx", "seed_smiles", "seed_label", "seed_smiles_canon"]],
        on="seed_smiles_canon",
        how="left",
        suffixes=("", "_ref"),
    )

    matched = block[block["seed_idx_ref"].notna()].copy()
    matched["seed_idx"] = matched["seed_idx_ref"].astype(int)
    matched["seed_smiles"] = matched["seed_smiles_ref"]
    matched["seed_label"] = matched["seed_label_ref"].astype(int)

    if n_generations is None:
        observed = matched["gen_idx"].dropna().astype(int)
        n_generations = int(observed.max()) + 1 if len(observed) else 0
    if n_generations <= 0:
        raise ValueError(f"Could not infer n_generations for dataset '{dataset}'")

    methods = matched["method"].dropna().unique().tolist()
    if not methods:
        methods = block["method"].dropna().unique().tolist()

    all_rows: List[pd.DataFrame] = []
    ref_base = ref[["seed_idx", "seed_smiles", "seed_label"]].copy()
    # Method-invariant seeds × gens cross-join, built once for the pad_missing
    # branch. Each iteration only assigns `method` on a copy.
    seed_gen_grid: Optional[pd.DataFrame] = None
    if pad_missing:
        gen_grid = pd.DataFrame({"gen_idx": list(range(n_generations))})
        seed_gen_grid = ref_base.assign(_k=1).merge(gen_grid.assign(_k=1), on="_k").drop(columns="_k")
        seed_gen_grid["dataset"] = dataset
    for method in methods:
        method_rows = matched[matched["method"] == method].copy()

        method_rows = method_rows.sort_values(["seed_idx", "gen_idx"]).drop_duplicates(
            subset=["method", "dataset", "seed_idx", "gen_idx"], keep="first"
        )
        if pad_missing:
            # Left-join actual rows into the full (seeds × gens) grid. Missing
            # (seed, gen_idx) slots become NaN-generated_smiles rows.
            grid = seed_gen_grid.copy()
            grid["method"] = method
            merged = grid.merge(
                method_rows[["method", "dataset", "seed_idx", "gen_idx", "generated_smiles"]],
                on=["method", "dataset", "seed_idx", "gen_idx"],
                how="left",
            )
        else:
            # No padding — just retain the actual rows. Short seeds keep their
            # real (< n_generations) row count. Downstream metrics (VIN etc.)
            # divide by len(df), so denominators are honest instead of inflated
            # by placeholder rows that carry is_valid=False.
            merged = method_rows.merge(
                ref_base,
                on="seed_idx",
                how="inner",
                suffixes=("_drop", ""),
            )
            # Drop any columns that got _drop suffix from the merge collision
            merged = merged[[c for c in merged.columns if not c.endswith("_drop")]]
            for col in REQUIRED_COLUMNS:
                if col not in merged.columns:
                    merged[col] = None
        all_rows.append(merged[REQUIRED_COLUMNS])

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["method", "dataset", "seed_idx", "gen_idx"]).reset_index(drop=True)
    return out


def normalize_generation_table(
    df: pd.DataFrame,
    config: PairitEvalConfig,
    n_generations_by_dataset: Optional[Dict[str, int]] = None,
    seed_label_by_dataset: Optional[Dict[str, int]] = None,
    pad_missing: bool = True,
) -> pd.DataFrame:
    """Normalize a mixed-method generation table dataset-by-dataset.

    `pad_missing`: pass False for the top-5 pass so short seeds are kept
    at their real length instead of being placeholder-padded to n_gens
    (which deflates VIN by counting empty-slot rows in the denominator).
    """
    n_generations_by_dataset = n_generations_by_dataset or {}
    seed_label_by_dataset = seed_label_by_dataset or {}
    pieces: List[pd.DataFrame] = []
    for dataset in df["dataset"].dropna().unique():
        n_gen = n_generations_by_dataset.get(dataset)
        seed_label = seed_label_by_dataset.get(dataset, 0)
        pieces.append(normalize_generation_block(df, dataset, config, n_generations=n_gen, seed_label=seed_label, pad_missing=pad_missing))
    return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0].copy()
