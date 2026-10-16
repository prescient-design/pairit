"""
Loaders that convert raw method outputs into the canonical generation schema:

method, dataset, seed_idx, seed_smiles, seed_label, gen_idx, generated_smiles
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from rdkit import Chem

from .cli_utils import DATA_ROOT, DEFAULT_MOLECULENET_TEST_DIR, DEFAULT_ZINC250K_TEST_CSV
from .config import get_default_generation_paths, get_mmp_generation_paths
from .methods import ensure_supported_method


N_DDSBM_SEEDS = 5
MOLECULENET_DATASETS = ("bace", "bbbp", "hiv")

LABEL_COLS = {"bace": "Class", "bbbp": "p_np", "hiv": "HIV_active"}
CANONICAL_COLUMNS = [
    "method",
    "dataset",
    "seed_idx",
    "seed_smiles",
    "seed_label",
    "gen_idx",
    "generated_smiles",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _canon(smi: str) -> str:
    """Canonical SMILES for matching across datasets."""
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol, canonical=True) if mol else str(smi)


def _seed_order_map(smiles: pd.Series) -> Dict[str, int]:
    ordered = list(dict.fromkeys(smiles.astype(str).tolist()))
    return {s: i for i, s in enumerate(ordered)}


def _load_test_class0(dataset: str, moleculenet_test_dir: Path = DEFAULT_MOLECULENET_TEST_DIR) -> pd.DataFrame:
    """Load class-0 test molecules from parquet, preserving row order."""
    label_col = LABEL_COLS[dataset]
    # Bundled assets use `molecules.parquet`; older trees use the long name.
    split_dir = moleculenet_test_dir / dataset / "test"
    candidates = [split_dir / "molecules.parquet", split_dir / f"molecules_{dataset}_real.parquet"]
    path = next((p for p in candidates if p.exists()), candidates[-1])
    df = pd.read_parquet(path)
    return df[df[label_col] == 0].reset_index(drop=True)


def _normalize_pairit_block(raw: pd.DataFrame, dataset: str, method: str) -> pd.DataFrame:
    if len(raw) == 0:
        return _empty_df()
    if "seed_smiles" not in raw.columns or "generated_smiles" not in raw.columns:
        raise ValueError(f"Pairit raw CSV for dataset '{dataset}' must include seed_smiles/generated_smiles columns.")

    seed_order = _seed_order_map(raw["seed_smiles"])
    gen_idx = raw.groupby("seed_smiles").cumcount().astype(int)
    seed_label = raw["seed_label"].astype(int) if "seed_label" in raw.columns else 0
    gen_smiles = raw["generated_smiles"].where(raw["generated_smiles"].notna(), "")

    return pd.DataFrame(
        {
            "method": method,
            "dataset": dataset,
            "seed_idx": raw["seed_smiles"].map(seed_order).astype(int),
            "seed_smiles": raw["seed_smiles"].astype(str),
            "seed_label": seed_label,
            "gen_idx": gen_idx,
            "generated_smiles": gen_smiles.astype(str),
        }
    )


def _normalize_canonical_block(raw: pd.DataFrame, dataset: str, method: str) -> pd.DataFrame:
    out = raw[CANONICAL_COLUMNS].copy()
    out["method"] = method
    out["dataset"] = dataset
    out["seed_idx"] = pd.to_numeric(out["seed_idx"], errors="coerce").fillna(0).astype(int)
    out["seed_label"] = pd.to_numeric(out["seed_label"], errors="coerce").fillna(0).astype(int)
    out["gen_idx"] = pd.to_numeric(out["gen_idx"], errors="coerce").fillna(0).astype(int)
    out["seed_smiles"] = out["seed_smiles"].astype(str)
    out["generated_smiles"] = out["generated_smiles"].fillna("").astype(str)
    return out[CANONICAL_COLUMNS]


def _load_hierg2g_translation(path: Path, dataset: str, method: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_df()
    pairs = pd.read_csv(path, sep=r"\s+", names=["seed_smiles", "generated_smiles"], engine="python")
    if len(pairs) == 0:
        return _empty_df()

    pairs["seed_smiles"] = pairs["seed_smiles"].astype(str)
    pairs["generated_smiles"] = pairs["generated_smiles"].astype(str)
    pairs["gen_idx"] = pairs.groupby("seed_smiles").cumcount().astype(int)
    seed_order = _seed_order_map(pairs["seed_smiles"])

    return pd.DataFrame(
        {
            "method": method,
            "dataset": dataset,
            "seed_idx": pairs["seed_smiles"].map(seed_order).astype(int),
            "seed_smiles": pairs["seed_smiles"],
            "seed_label": 0,
            "gen_idx": pairs["gen_idx"],
            "generated_smiles": pairs["generated_smiles"],
        }
    )


def load_nfbo_moleculenet() -> pd.DataFrame:
    """
    Normalize NFBO MoleculeNet raw CSVs.

    Raw columns: seed_smiles, seed_class, generation_rank, generated_smiles, ...
    """
    frames = []
    path_map = get_default_generation_paths()["nfbo"]["moleculenet"]
    for dataset, path in path_map.items():
        raw = pd.read_csv(path)
        seed_order = _seed_order_map(raw["seed_smiles"])
        frames.append(
            pd.DataFrame(
                {
                    "method": "nfbo",
                    "dataset": dataset,
                    "seed_idx": raw["seed_smiles"].map(seed_order).astype(int),
                    "seed_smiles": raw["seed_smiles"].astype(str),
                    "seed_label": raw["seed_class"].astype(int),
                    "gen_idx": raw["generation_rank"].astype(int),
                    "generated_smiles": raw["generated_smiles"].astype(str),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_nfbo_zinc250k() -> pd.DataFrame:
    """
    Normalize NFBO Zinc250k logP-improvement CSVs.

    Raw columns: source_smiles, source_logp, generation_idx, generated_smiles, ...
    """
    pattern = get_default_generation_paths()["nfbo"]["zinc250k_glob"]
    files = sorted(glob.glob(pattern))
    if not files:
        return _empty_df()

    frames = []
    global_seed_idx = 0
    for file_path in files:
        raw = pd.read_csv(file_path)
        for smi in raw["source_smiles"].unique():
            sub = raw[raw["source_smiles"] == smi]
            frames.append(
                pd.DataFrame(
                    {
                        "method": "nfbo",
                        "dataset": "zinc250k",
                        "seed_idx": global_seed_idx,
                        "seed_smiles": str(smi),
                        "seed_label": 0,
                        "gen_idx": sub["generation_idx"].astype(int).values,
                        "generated_smiles": sub["generated_smiles"].astype(str).values,
                    }
                )
            )
            global_seed_idx += 1

    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_ddsbm_moleculenet(variant: str, moleculenet_test_dir: Path = DEFAULT_MOLECULENET_TEST_DIR) -> pd.DataFrame:
    """
    Normalize DDSBM MoleculeNet result files (bridge or sb).

    Raw columns: idx, success, smiles, relaxed_smiles, ...
    """
    paths_map = get_default_generation_paths()[variant]["moleculenet"]
    frames = []

    for dataset, base_dir in paths_map.items():
        class0 = _load_test_class0(dataset, moleculenet_test_dir=moleculenet_test_dir)
        for seed_num in range(N_DDSBM_SEEDS):
            fpath = Path(base_dir) / f"result_forward_seed{seed_num}.csv"
            if not fpath.exists():
                continue
            raw = pd.read_csv(fpath)

            seed_idx_list: List[int] = []
            seed_smiles_list: List[str] = []
            gen_smiles_list: List[str] = []
            for _, row in raw.iterrows():
                idx = int(row["idx"])
                seed_row = class0.iloc[idx]
                seed_idx_list.append(idx)
                seed_smiles_list.append(str(seed_row["smiles"]))
                smi = row["smiles"] if row["success"] and pd.notna(row["smiles"]) else ""
                gen_smiles_list.append(str(smi))

            frames.append(
                pd.DataFrame(
                    {
                        "method": variant,
                        "dataset": dataset,
                        "seed_idx": seed_idx_list,
                        "seed_smiles": seed_smiles_list,
                        "seed_label": 0,
                        "gen_idx": seed_num,
                        "generated_smiles": gen_smiles_list,
                    }
                )
            )

    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_ddsbm_zinc250k(variant: str, zinc250k_test_csv: Path = DEFAULT_ZINC250K_TEST_CSV) -> pd.DataFrame:
    """
    Normalize DDSBM Zinc250k result files (bridge or sb).
    """
    base_dir = Path(get_default_generation_paths()[variant]["zinc250k"])
    if not base_dir.exists():
        return _empty_df()

    test_df = pd.read_csv(zinc250k_test_csv)
    frames = []
    for seed_num in range(N_DDSBM_SEEDS):
        fpath = base_dir / f"result_forward_seed{seed_num}.csv"
        if not fpath.exists():
            continue
        raw = pd.read_csv(fpath)

        seed_idx_list: List[int] = []
        seed_smiles_list: List[str] = []
        gen_smiles_list: List[str] = []
        for _, row in raw.iterrows():
            idx = int(row["idx"])
            seed_idx_list.append(idx)
            seed_smiles_list.append(str(test_df.iloc[idx]["smiles"]))
            smi = row["smiles"] if row["success"] and pd.notna(row["smiles"]) else ""
            gen_smiles_list.append(str(smi))

        frames.append(
            pd.DataFrame(
                {
                    "method": variant,
                    "dataset": "zinc250k",
                    "seed_idx": seed_idx_list,
                    "seed_smiles": seed_smiles_list,
                    "seed_label": 0,
                    "gen_idx": seed_num,
                    "generated_smiles": gen_smiles_list,
                }
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_mmp_moleculenet(moleculenet_test_dir: Path = DEFAULT_MOLECULENET_TEST_DIR) -> pd.DataFrame:
    """
    Normalize MMP baseline CSVs to evaluation format.

    Uses the legacy raw MMP baseline CSVs, filters to class-0 real test seeds,
    and keeps the first 5 generations per seed.
    """
    frames = []
    for dataset, path in get_mmp_generation_paths(method="mmp", kind="top5").items():
        if not path.exists():
            continue
        raw = pd.read_csv(path)

        if set(CANONICAL_COLUMNS).issubset(raw.columns):
            frames.append(_normalize_canonical_block(raw, dataset=dataset, method="mmp"))
            continue

        raw = raw[(raw["split"] == "test") & (raw["query_label"] == 0)].copy()

        class0 = _load_test_class0(dataset, moleculenet_test_dir=moleculenet_test_dir)
        real_seeds_canon = {_canon(s) for s in class0["smiles"]}

        raw["_canon"] = raw["query_smiles"].apply(_canon)
        raw = raw[raw["_canon"].isin(real_seeds_canon)]

        canon_to_idx = {_canon(s): i for i, s in enumerate(class0["smiles"])}
        raw["seed_idx"] = raw["_canon"].map(canon_to_idx)
        raw = raw.dropna(subset=["seed_idx"])
        raw["seed_idx"] = raw["seed_idx"].astype(int)

        raw = raw.sort_values(["seed_idx", "gen_idx"])
        raw = raw.groupby("seed_idx").head(5)
        raw["gen_idx"] = raw.groupby("seed_idx").cumcount()

        frames.append(
            pd.DataFrame(
                {
                    "method": "mmp",
                    "dataset": dataset,
                    "seed_idx": raw["seed_idx"].values,
                    "seed_smiles": raw["query_smiles"].astype(str).values,
                    "seed_label": raw["query_label"].astype(int).values,
                    "gen_idx": raw["gen_idx"].astype(int).values,
                    "generated_smiles": raw["generated_smiles"].astype(str).values,
                }
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_mmp_prop_moleculenet() -> pd.DataFrame:
    """Normalize property-aware MMP baseline top-5 CSVs to evaluation format."""
    frames = []
    for dataset, path in get_mmp_generation_paths(method="mmp-prop", kind="top5").items():
        path = Path(path)
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        if not set(CANONICAL_COLUMNS).issubset(raw.columns):
            raise ValueError(f"mmp-prop CSV for dataset '{dataset}' is not in canonical format: {path}")
        frames.append(_normalize_canonical_block(raw, dataset=dataset, method="mmp-prop"))
    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_pairit_moleculenet(variant: str) -> pd.DataFrame:
    """
    Load pre-exported Pairit MoleculeNet generations and rebuild stable
    seed/gen indexing.

    The registered paths are parquet; dispatch on the suffix so a CSV export
    still works.
    """
    path_map = get_default_generation_paths()[variant]["moleculenet"]
    frames = []
    for dataset, path in path_map.items():
        path = Path(path)
        if not path.exists():
            continue
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frames.append(_normalize_pairit_block(raw, dataset=dataset, method=variant))
    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


NN_METHODS = ("nn-real-only", "nn-real-only-zinc")


def _load_nn_lookup(method: str) -> pd.DataFrame:
    """Load nearest-neighbor lookup baseline CSVs (already in canonical format)."""
    path_map = get_default_generation_paths()[method]["datasets"]
    frames = []
    for dataset, path in path_map.items():
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df)
    frames = [f for f in frames if len(f) > 0]
    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def _load_hierg2g(version: str) -> pd.DataFrame:
    method = f"hierg2g-{version}"
    path_map = get_default_generation_paths()[method]["datasets"]
    frames = []
    for dataset, path in path_map.items():
        frames.append(_load_hierg2g_translation(Path(path), dataset=dataset, method=method))
    frames = [f for f in frames if len(f) > 0]
    return pd.concat(frames, ignore_index=True) if frames else _empty_df()


def load_method_moleculenet(method: str) -> pd.DataFrame:
    canonical = ensure_supported_method(method)
    if canonical == "nfbo":
        return load_nfbo_moleculenet()
    if canonical == "mmp":
        return load_mmp_moleculenet()
    if canonical == "mmp-prop":
        return load_mmp_prop_moleculenet()
    if canonical in ("ddsbm-bridge", "ddsbm-sb"):
        return load_ddsbm_moleculenet(canonical)
    if canonical in ("pairit-from-zinc", "pairit-real-only"):
        return load_pairit_moleculenet(canonical)
    if canonical in ("hierg2g-04", "hierg2g-06"):
        df = _load_hierg2g(canonical.split("-")[-1])
        return df[df["dataset"].isin(MOLECULENET_DATASETS)].reset_index(drop=True)
    if canonical in NN_METHODS:
        df = _load_nn_lookup(canonical)
        return df[df["dataset"].isin(MOLECULENET_DATASETS)].reset_index(drop=True)
    raise ValueError(f"Unsupported method for MoleculeNet raw loading: {method}")


def load_method_zinc250k(method: str) -> pd.DataFrame:
    canonical = ensure_supported_method(method)
    if canonical == "nfbo":
        return load_nfbo_zinc250k()
    if canonical in ("ddsbm-bridge", "ddsbm-sb"):
        return load_ddsbm_zinc250k(canonical)
    if canonical in ("hierg2g-04", "hierg2g-06"):
        df = _load_hierg2g(canonical.split("-")[-1])
        return df[df["dataset"] == "zinc250k"].reset_index(drop=True)
    if canonical == "nn-real-only":
        df = _load_nn_lookup(canonical)
        return df[df["dataset"] == "zinc250k"].reset_index(drop=True)
    # Other NN variants and Pairit/MMP do not have zinc250k.
    return _empty_df()


def load_method_all(method: str) -> pd.DataFrame:
    """
    Load all available datasets for one method (MoleculeNet + optional zinc250k).
    """
    parts = [load_method_moleculenet(method), load_method_zinc250k(method)]
    parts = [p for p in parts if len(p) > 0]
    return pd.concat(parts, ignore_index=True) if parts else _empty_df()


# ---------------------------------------------------------------------------
# Path-driven loading
#
# The loaders above are keyed by *method name* and resolve paths from
# _CANONICAL_GENERATION_PATHS -- hardcoded absolute paths with dated run
# directories. That is fine for the canonical set and useless for anything new.
#
# load_generations() is keyed by *path*, so a registry row (path + format) is
# directly consumable and a new method needs one line in a TSV rather than a
# code change.
# ---------------------------------------------------------------------------

#: Recognised registry `format` values.
FORMATS = ("pairit_csv", "csv", "parquet", "hierg2g_txt", "ddsbm_dir")


def _pick(columns, *candidates):
    """First candidate present in `columns`, else None."""
    lowered = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def load_generations(
    path,
    fmt: str = "pairit_csv",
    dataset: str = "",
    method: str = "",
    ddsbm_prefer_relaxed: bool = False,
    ddsbm_seeds: Optional[int] = None,
    ddsbm_reference_seeds: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a generation file into the canonical 7-column frame.

    Parameters
    ----------
    path
        File (or directory, for ``ddsbm_dir``) to read.
    fmt
        One of :data:`FORMATS`. ``pairit_csv``/``csv``/``parquet`` all expect
        seed/generated SMILES columns and differ only in how they are read;
        column names are matched case-insensitively against a few known
        spellings so a file does not have to be rewritten to be registered.
    ddsbm_prefer_relaxed
        DDSBM writes both ``smiles`` and ``relaxed_smiles``. Preferring the
        relaxed form substitutes a *different molecule*, so it is off by default
        and always recorded. Three loaders in this project have historically
        disagreed on this; the choice belongs to the caller, explicitly.

    Returns
    -------
    DataFrame with :data:`CANONICAL_COLUMNS`.
    """
    path = Path(path)
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Expected one of {FORMATS}.")
    if not path.exists():
        raise FileNotFoundError(f"Generations not found: {path}")

    if fmt == "hierg2g_txt":
        return _load_hierg2g_translation(path, dataset, method)
    if fmt == "ddsbm_dir":
        return _load_ddsbm_dir(
            path, dataset, method,
            prefer_relaxed=ddsbm_prefer_relaxed,
            n_seeds=ddsbm_seeds,
            reference_seeds=ddsbm_reference_seeds,
        )

    raw = pd.read_parquet(path) if fmt == "parquet" or path.suffix == ".parquet" else pd.read_csv(path)
    if len(raw) == 0:
        return _empty_df()

    # Column spellings seen across the projects that feed this: pairit exports,
    # lobster eval CSVs (smiles_generated / smiles_conditioning), and translation
    # dumps. Matching a few known names beats requiring every file be rewritten.
    seed_col = _pick(
        raw.columns,
        "seed_smiles", "smiles_seed", "smiles_conditioning", "seed",
        "source_smiles", "input_smiles", "src", "parent_smiles",
    )
    gen_col = _pick(
        raw.columns,
        "generated_smiles", "smiles_generated", "generation", "gen_smiles",
        "target_smiles", "tgt", "smiles",
    )
    if seed_col is None or gen_col is None:
        raise ValueError(
            f"{path}: could not find seed/generated SMILES columns among {list(raw.columns)}. "
            f"Expected something like seed_smiles/generated_smiles."
        )

    work = raw.rename(columns={seed_col: "seed_smiles", gen_col: "generated_smiles"})
    return _normalize_pairit_block(work, dataset, method)


def _load_ddsbm_dir(
    directory: Path,
    dataset: str,
    method: str,
    prefer_relaxed: bool = False,
    n_seeds: Optional[int] = None,
    reference_seeds: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read every ``result_{forward,backward}_seed*.csv`` in a DDSBM output directory.

    Each file is one **random seed** -- a single stochastic sampling run -- and holds one
    row per **starting molecule**. So 64 files over 135 starting molecules is 64
    generations each, not 64 molecules.

    Unlike :func:`load_ddsbm_moleculenet`, this reads whatever files are present rather
    than assuming a fixed count.

    ``reference_seeds`` resolves the starting molecule when the files do not name it.
    MoleculeNet DDSBM output carries only an ``idx`` column, so without this every row
    gets an empty ``seed_smiles`` and the whole directory collapses to ONE starting
    molecule -- which silently turns 8,640 generations over 135 molecules into 8,640
    generations over 1. The list must be in the same order ``idx`` counts in (the
    dataset's class-0 test molecules, file order), NOT sorted.
    """
    # DDSBM writes `result_forward_seed*.csv` or `result_backward_seed*.csv` depending on
    # which direction of the bridge was sampled, and the choice is per method, not per
    # dataset: across the TDC datasets `ddsbm_bridge` ships backward while `ddsbm_sb`
    # ships forward. Accept whichever the directory holds, but refuse to choose when both
    # are present -- picking the wrong direction silently evaluates the reverse
    # transformation and still returns a full, plausible-looking frame.
    forward = sorted(directory.glob("result_forward_seed*.csv"))
    backward = sorted(directory.glob("result_backward_seed*.csv"))
    if forward and backward:
        raise ValueError(
            f"{directory}: holds both result_forward_seed*.csv ({len(forward)}) and "
            f"result_backward_seed*.csv ({len(backward)}). The direction is ambiguous; "
            f"point the registry at a directory containing only one, because the wrong "
            f"direction evaluates the reverse transformation without any error."
        )
    files = forward or backward
    if n_seeds is not None:
        files = files[:n_seeds]
    if not files:
        raise FileNotFoundError(
            f"No result_forward_seed*.csv or result_backward_seed*.csv under {directory}"
        )

    frames = []
    out_of_range = 0
    for gen_idx, fpath in enumerate(files):
        raw = pd.read_csv(fpath)
        seed_col = _pick(raw.columns, "seed_smiles", "source_smiles", "src")
        gens = []
        for _, row in raw.iterrows():
            smi = ""
            if bool(row.get("success", True)):
                smi = row.get("smiles")
                if (pd.isna(smi) or not str(smi) or str(smi) == "nan") and prefer_relaxed:
                    smi = row.get("relaxed_smiles")
                if pd.isna(smi) or str(smi) == "nan":
                    smi = ""
            gens.append(str(smi))

        idx = raw["idx"].astype(int) if "idx" in raw else pd.Series(range(len(raw)))
        if seed_col:
            seeds = raw[seed_col].astype(str)
        elif reference_seeds is not None:
            n_ref = len(reference_seeds)
            # An out-of-range idx means this directory and the reference set describe
            # different test splits. Count it and raise below rather than dropping the
            # row, which is how a mismatched pairing becomes a quietly short table.
            out_of_range += int((idx >= n_ref).sum() + (idx < 0).sum())
            seeds = idx.map(
                lambda i: reference_seeds[i] if 0 <= i < n_ref else ""
            ).astype(str)
        else:
            seeds = ""

        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "dataset": dataset,
                    "seed_idx": idx,
                    "seed_smiles": seeds,
                    "seed_label": 0,
                    "gen_idx": gen_idx,
                    "generated_smiles": gens,
                }
            )
        )

    if out_of_range:
        raise ValueError(
            f"{directory}: {out_of_range} row(s) have an `idx` outside the "
            f"{len(reference_seeds)} reference starting molecules supplied. The DDSBM run "
            f"and the reference seed set are not the same test split."
        )
    return pd.concat(frames, ignore_index=True) if frames else _empty_df()
