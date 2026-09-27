"""
Stage 1: Compute metrics on generations.

Input: "normalized" = one row per generated molecule with fixed column names (see REQUIRED_COLUMNS).
No value scaling (e.g. z-score) is implied; the table layout/schema is standardized.
Output: per_generation, per_seed, and average (quick view) DataFrames.

Metric naming:
- Valid (V): parses as a molecule.
- Novel (N): not memorised from the finetune training set.
- Improved (I): task success — seed was class 0 and generation is predicted class 1 (prob >= 0.5).
  Seed label comes from CSV (seed_label); predicted label/prob for the generated molecule comes
  only from the dataset evaluator (e.g. bace_b.joblib for dataset bace). No property name like
  "Class" is specified for improvement — the dataset (bace, bbbp, hiv) determines which evaluator.
- VIN (v_n_i): valid AND novel AND improved. The single headline composite for this codebase.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs, Descriptors
from tqdm import tqdm

from .config import PairitEvalConfig, DATASET_CONFIG, MOLECULENET_DATASETS, TDC_DATASETS
from . import oracles, property_metrics


@dataclass
class Stage1Result:
    """Result of Stage 1: metrics at three levels."""

    per_generation: pd.DataFrame  # one row per generated molecule
    per_seed: pd.DataFrame        # one row per (method, dataset, seed)
    average: pd.DataFrame          # one row per (method, dataset) — quick view
    #: Per-dataset resources actually used, keyed by dataset. Carries the
    #: resolved oracles (with their live scored/unscorable counters) and the
    #: improvement direction, so the run record can report what really ran
    #: rather than what was requested. Not part of the metric output.
    resources: Optional[Dict[str, Dict[str, Any]]] = None

# Suppress RDKit logs
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=DeprecationWarning)

REQUIRED_COLUMNS = ["method", "dataset", "seed_idx", "seed_smiles", "seed_label", "gen_idx", "generated_smiles"]

COMPUTED_COLUMNS = [
    "is_valid", "is_novel", "v_n_i",
    "predicted_prob", "predicted_label", "improvement", "maintenance",
    "tanimoto",
]


# ---------- Helpers (diversity, property) ----------

def _get_max_alkyl_chain_length(mol: Chem.Mol) -> int:
    if mol is None:
        return 0
    patterns = [
        ("[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]", 6),
        ("[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]", 5),
    ]
    for pattern_str, length in patterns:
        pat = Chem.MolFromSmarts(pattern_str)
        if pat is not None and mol.HasSubstructMatch(pat):
            return length
    return 0


def _get_max_ring_size(mol: Chem.Mol) -> int:
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    return max((len(r) for r in ri.AtomRings()), default=0)


def _compute_tanimoto(s1: str, s2: str) -> Optional[float]:
    try:
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None:
            return None
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


def compute_tanimoto(s1: str, s2: str) -> Optional[float]:
    """Public API: Tanimoto similarity between two SMILES (Morgan ECFP4, radius=2, nBits=2048)."""
    return _compute_tanimoto(s1, s2)


def mean_similarity_to_seeds(
    seeds: Sequence[str],
    gens: Sequence[str],
    nbits: int = 2048,
    radius: int = 2,
) -> Optional[float]:
    """
    Mean Tanimoto similarity between paired (seed, gen) molecules.
    Uses Morgan ECFP4 (radius=2, nBits=nbits). Invalid pairs are skipped.
    """
    sims: List[float] = []
    for s, g in zip(seeds, gens):
        t = _compute_tanimoto(str(s) if s else "", str(g) if g else "")
        if t is not None:
            sims.append(t)
    return float(np.mean(sims)) if sims else None


def max_similarity_to_gt(
    generated_smiles: Sequence[str],
    gt_smiles: Sequence[str],
    nbits: int = 2048,
    radius: int = 2,
) -> Optional[float]:
    """
    For each generated molecule, max Tanimoto to any ground-truth molecule; return mean of those max.
    Uses Morgan ECFP4. Invalid SMILES are skipped.
    """
    gt_list = [str(s).strip() for s in gt_smiles if s and str(s).strip()]
    gt_fps: List[Any] = []
    for s in gt_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        if fp is not None:
            gt_fps.append(fp)
    if not gt_fps:
        return None
    max_sims: List[float] = []
    for s in generated_smiles:
        if not s or not str(s).strip():
            continue
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, gt_fps)
        if sims:
            max_sims.append(float(max(sims)))
    return float(np.mean(max_sims)) if max_sims else None


def internal_diversity(
    smiles_list: List[str], nbits: int = 2048, radius: int = 2, deduplicate: bool = True
) -> Tuple[Optional[float], int, int]:
    mols = []
    for s in smiles_list:
        if pd.isna(s) or s == "" or s == "None":
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    n_valid = len(mols)
    if len(mols) < 2:
        return None, n_valid, n_valid
    if deduplicate:
        seen = set()
        unique_mols = []
        for m in mols:
            c = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
            if c not in seen:
                seen.add(c)
                unique_mols.append(m)
        mols = unique_mols
    n_unique = len(mols)
    if len(mols) < 2:
        return None, n_valid, n_unique
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols]
    total = 0.0
    n_pairs = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            total += 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            n_pairs += 1
    return (total / n_pairs if n_pairs else None), n_valid, n_unique


def get_seed_mol_stats(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mw": 0, "max_ring": 0, "max_chain": 0}
    return {
        "mw": Descriptors.MolWt(mol),
        "max_ring": _get_max_ring_size(mol),
        "max_chain": _get_max_alkyl_chain_length(mol),
    }


# ---------- Loaders (require config) ----------

def _canonicalize_smiles_list(smiles_series) -> List[str]:
    """Canonicalize a pandas Series / list of SMILES, skipping invalid ones."""
    out = []
    for s in smiles_series:
        if pd.isna(s) or not str(s).strip():
            continue
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return out


def _canonicalize_smiles_set(smiles_series) -> Set[str]:
    return set(_canonicalize_smiles_list(smiles_series))


def _canonicalize_smiles(s: Any) -> Optional[str]:
    """Canonicalize one SMILES string, returning None for blank/invalid values."""
    if pd.isna(s) or not str(s).strip():
        return None
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _read_smiles_from_csv(path: Path, smiles_col: str = "smiles") -> pd.Series:
    df = pd.read_csv(path)
    return df[smiles_col].dropna()


def _read_smiles_from_parquet(path: Path, smiles_col: str = "smiles",
                               split_filter: Optional[str] = None) -> pd.Series:
    df = pd.read_parquet(path)
    if split_filter and "split" in df.columns:
        df = df[df["split"] == split_filter]
    return df[smiles_col].dropna()


# --- MoleculeNet parquet loaders ---

def _moleculenet_train_path(dataset: str, moleculenet_train_dir: Path) -> Optional[Path]:
    """Locate the train molecules file for real-only MoleculeNet splits.

    Tolerant of both file-naming conventions: the short-name ASSETS bundle
    layout (`molecules.parquet`) and the long-name canonical on-disk layout
    (`molecules_{ds}_real.parquet`). Returns the first that exists, or None.
    """
    short = moleculenet_train_dir / dataset / "train" / "molecules.parquet"
    if short.exists():
        return short
    long_ = moleculenet_train_dir / dataset / "train" / f"molecules_{dataset}_real.parquet"
    return long_ if long_.exists() else None


def _load_moleculenet_train_smiles(dataset: str, moleculenet_train_dir: Path) -> Set[str]:
    """Load train molecules from {dataset}/train/molecules_{dataset}_real.parquet.
    Returns canonical SMILES set for novelty checking."""
    path = _moleculenet_train_path(dataset, moleculenet_train_dir)
    if path is None:
        # Loud. An empty novelty reference makes every generation "novel", which inflates
        # `novel` and every composite built on it, and looks like a clean run.
        raise FileNotFoundError(
            f"No train molecules for {dataset!r} under {moleculenet_train_dir} "
            f"(expected molecules_{dataset}_real.parquet). "
            f"Novelty would silently be measured against nothing."
        )
    return _canonicalize_smiles_set(_read_smiles_from_parquet(path, "smiles"))


def _load_moleculenet_train_class1_smiles_list(
    dataset: str, moleculenet_train_dir: Path,
) -> List[str]:
    """Load class-1 train molecules as canonical SMILES list for distribution
    metrics reference (A3). Uses the per-dataset `label_col` from DATASET_CONFIG
    (BACE: `Class`, BBBP: `p_np`, HIV: `HIV_active`, etc.).

    Fail loud on missing data. This is the reference set for FCD/SNN/etc; a
    silent empty list would trip the outer guard in evaluate() but also mask
    genuine misconfigurations of moleculenet_train_dir. Better to fail here
    with the specific reason (path missing vs label_col missing)."""
    path = _moleculenet_train_path(dataset, moleculenet_train_dir)
    if path is None:
        raise FileNotFoundError(
            f"Class-1 train molecules for {dataset!r} not found under "
            f"{moleculenet_train_dir}. Expected molecules.parquet or "
            f"molecules_{dataset}_real.parquet in {dataset}/train/. "
            f"Fix moleculenet_train_dir or don't request distribution metrics."
        )
    df = pd.read_parquet(path)
    label_col = DATASET_CONFIG.get(dataset, {}).get("label_col")
    if not label_col:
        raise ValueError(
            f"DATASET_CONFIG[{dataset!r}] is missing 'label_col' — cannot filter "
            f"class-1 molecules for the distribution reference set."
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Column {label_col!r} not found in {path}. Available: "
            f"{list(df.columns)}. Cannot filter class-1 molecules."
        )
    return _canonicalize_smiles_list(df[df[label_col] == 1]["smiles"])


def _load_moleculenet_test_smiles(
    dataset: str, moleculenet_test_dir: Path,
) -> List[str]:
    """Load test molecules for sim_gt_mean. Fail loud if the configured path
    exists as a directory but the expected parquet is missing — a silent
    empty here would zero the sim_gt_mean column for the whole method with
    no warning. If the caller doesn't want sim_gt_mean, pass
    moleculenet_test_dir=None instead of pointing at an empty dir."""
    short = moleculenet_test_dir / dataset / "test" / "molecules.parquet"
    long_ = moleculenet_test_dir / dataset / "test" / f"molecules_{dataset}_real.parquet"
    path = short if short.exists() else long_
    if not path.exists():
        raise FileNotFoundError(
            f"Test molecules for {dataset!r} not found under "
            f"{moleculenet_test_dir}. Expected molecules.parquet or "
            f"molecules_{dataset}_real.parquet in {dataset}/test/. "
            f"Fix moleculenet_test_dir or set it to None to skip sim_gt_mean."
        )
    return _canonicalize_smiles_list(_read_smiles_from_parquet(path, "smiles"))


def _load_moleculenet_test_class0_smiles(dataset: str, moleculenet_test_dir: Path,
                                         dataset_config: Dict[str, Dict[str, Any]]) -> List[str]:
    """Load ordered class-0 test seed molecules for stable seed indexing.
    Fail loud on missing data — silent empty here breaks seed_idx alignment
    across methods and inflates novelty counts. Callers that don't want
    stable indexing should pass moleculenet_test_dir=None."""
    short = moleculenet_test_dir / dataset / "test" / "molecules.parquet"
    long_ = moleculenet_test_dir / dataset / "test" / f"molecules_{dataset}_real.parquet"
    path = short if short.exists() else long_
    if not path.exists():
        raise FileNotFoundError(
            f"Test molecules for {dataset!r} not found under "
            f"{moleculenet_test_dir}. Expected molecules.parquet or "
            f"molecules_{dataset}_real.parquet in {dataset}/test/. "
            f"Fix moleculenet_test_dir or drop this dataset."
        )
    df = pd.read_parquet(path)
    label_col = dataset_config.get(dataset, {}).get("label_col")
    if not label_col:
        raise ValueError(
            f"DATASET_CONFIG[{dataset!r}] is missing 'label_col' — cannot filter "
            f"class-0 seeds for stable seed indexing."
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Column {label_col!r} not found in {path}. Available: "
            f"{list(df.columns)}. Cannot filter class-0 seeds."
        )
    return _canonicalize_smiles_list(df[df[label_col] == 0]["smiles"])


# --- TDC loaders (generic; driven by DATASET_CONFIG paths) ---
#
# The MoleculeNet loaders above assume moleculenet_train_dir / moleculenet_test_dir
# from PairitEvalConfig. TDC datasets (currently only hERG in the eval pipeline)
# instead carry per-dataset paths in DATASET_CONFIG["train_smiles"] /
# DATASET_CONFIG["test_smiles"], and also carry `direction` — which flips
# "seed class" and "target (reference) class" relative to MoleculeNet. Keep
# these helpers strictly TDC-only so the MoleculeNet code path is unchanged.
#
# hERG convention: direction=decreasing → seeds are class 1 (blockers), target
# is class 0 (non-blockers). Distribution reference = class 0 train molecules.


def _tdc_seed_class(dataset: str, dataset_config: Dict[str, Dict[str, Any]]) -> int:
    """Which class are seeds drawn from for this TDC dataset?

    direction=increasing → seeds are class 0 (want to reach class 1).
    direction=decreasing → seeds are class 1 (want to reach class 0).
    """
    direction = dataset_config.get(dataset, {}).get("direction", "increasing")
    return 1 if str(direction).lower() == "decreasing" else 0


def _tdc_target_class(dataset: str, dataset_config: Dict[str, Dict[str, Any]]) -> int:
    """Which class is the improvement target for this TDC dataset (opposite of
    _tdc_seed_class)? Used to pick the distribution-metric reference class."""
    return 1 - _tdc_seed_class(dataset, dataset_config)


def _tdc_parquet_path(
    dataset: str, dataset_config: Dict[str, Dict[str, Any]], *, kind: str,
) -> Optional[Path]:
    """First existing path in DATASET_CONFIG[dataset][f'{kind}_smiles'], or None.

    `kind` is either 'test' or 'train'. Accepts a single path or a list of
    fallback paths in the config.
    """
    entries = dataset_config.get(dataset, {}).get(f"{kind}_smiles") or []
    if isinstance(entries, (str, Path)):
        entries = [entries]
    for entry in entries:
        p = Path(entry)
        if p.exists():
            return p
    return None


def _tdc_test_parquet_path(dataset: str, dataset_config: Dict[str, Dict[str, Any]]) -> Optional[Path]:
    return _tdc_parquet_path(dataset, dataset_config, kind="test")


def _tdc_train_parquet_path(dataset: str, dataset_config: Dict[str, Dict[str, Any]]) -> Optional[Path]:
    return _tdc_parquet_path(dataset, dataset_config, kind="train")


def _load_tdc_class_filtered_smiles(
    dataset: str, dataset_config: Dict[str, Dict[str, Any]],
    *, kind: str, class_id: int, purpose: str,
) -> List[str]:
    """Load TDC {test|train} parquet, filter to `class_id` on label_col, return
    canonical SMILES. Fail loud on missing parquet or missing/misconfigured
    label_col — a silent empty list breaks per-seed alignment across methods
    AND makes every generation look novel.

    `purpose` is a short phrase (e.g. "seed-class alignment") appended to
    error messages so the caller can tell which loader failed.
    """
    path = _tdc_parquet_path(dataset, dataset_config, kind=kind)
    if path is None:
        raise FileNotFoundError(
            f"TDC {kind} molecules for {dataset!r} not found. Expected "
            f"DATASET_CONFIG[{dataset!r}][{kind + '_smiles'!r}] to point at an "
            f"existing parquet ({purpose})."
        )
    df = pd.read_parquet(path)
    label_col = dataset_config.get(dataset, {}).get("label_col")
    if not label_col:
        raise ValueError(
            f"DATASET_CONFIG[{dataset!r}] is missing 'label_col' — cannot filter "
            f"class-{class_id} molecules ({purpose})."
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Column {label_col!r} not found in {path}. Available: "
            f"{list(df.columns)}. Cannot filter class-{class_id} molecules."
        )
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int)
    return _canonicalize_smiles_list(df[labels == class_id]["smiles"])


def _load_tdc_test_seed_class_smiles(
    dataset: str, dataset_config: Dict[str, Dict[str, Any]],
) -> List[str]:
    """TDC analogue of `_load_moleculenet_test_class0_smiles`: return the ORDERED
    canonical SMILES list of the seed class for stable seed indexing."""
    return _load_tdc_class_filtered_smiles(
        dataset, dataset_config,
        kind="test",
        class_id=_tdc_seed_class(dataset, dataset_config),
        purpose="stable seed indexing",
    )


def _load_tdc_test_smiles(
    dataset: str, dataset_config: Dict[str, Dict[str, Any]],
) -> List[str]:
    """TDC analogue of `_load_moleculenet_test_smiles`: return the full test-set
    SMILES for sim_gt_mean. Fail loud on missing parquet."""
    path = _tdc_test_parquet_path(dataset, dataset_config)
    if path is None:
        raise FileNotFoundError(
            f"TDC test molecules for {dataset!r} not found (checked DATASET_CONFIG"
            f"[{dataset!r}]['test_smiles']). Cannot compute sim_gt_mean."
        )
    return _canonicalize_smiles_list(_read_smiles_from_parquet(path, "smiles"))


def _load_tdc_train_target_class_smiles_list(
    dataset: str, dataset_config: Dict[str, Dict[str, Any]],
) -> List[str]:
    """TDC analogue of `_load_moleculenet_train_class1_smiles_list`: return the
    canonical SMILES list of TARGET-class train molecules (class 0 for
    direction=decreasing, class 1 for direction=increasing). Used as the
    distribution-metric reference set."""
    return _load_tdc_class_filtered_smiles(
        dataset, dataset_config,
        kind="train",
        class_id=_tdc_target_class(dataset, dataset_config),
        purpose="distribution-metric reference set",
    )


# --- Zinc250k loaders ---

def _load_zinc250k_train_smiles(zinc250k_train_molecules: Path) -> Set[str]:
    """Load zinc250k train molecules from parquet (filter split=train)."""
    if not zinc250k_train_molecules.exists():
        return set()
    return _canonicalize_smiles_set(
        _read_smiles_from_parquet(zinc250k_train_molecules, "smiles", split_filter="train")
    )


def _load_zinc250k_train_smiles_list(zinc250k_train_molecules: Path) -> List[str]:
    """Load zinc250k train molecules as list for distribution metrics reference."""
    if not zinc250k_train_molecules.exists():
        return []
    return _canonicalize_smiles_list(
        _read_smiles_from_parquet(zinc250k_train_molecules, "smiles", split_filter="train")
    )


def _load_zinc250k_test_smiles(zinc250k_test_csv: Path) -> List[str]:
    """Load zinc250k test molecules from CSV."""
    if not zinc250k_test_csv.exists():
        return []
    return _canonicalize_smiles_list(_read_smiles_from_csv(zinc250k_test_csv, "smiles"))


def _normalize_seed_indexing(df: pd.DataFrame, config: Optional[PairitEvalConfig]) -> pd.DataFrame:
    """
    Rebuild seed_idx from seed_smiles using dataset reference order when available.

    This makes per-seed aggregation robust to malformed input seed_idx values and aligns
    methods to a shared index whenever they use the same seed molecules.
    """
    if len(df) == 0:
        return df

    out = df.copy()
    if "original_seed_idx" not in out.columns:
        out["original_seed_idx"] = out["seed_idx"]
    out["_seed_smiles_canon"] = out["seed_smiles"].apply(_canonicalize_smiles)

    dataset_cfg = config.dataset_config if config else DATASET_CONFIG
    for ds in out["dataset"].dropna().unique():
        ds_mask = out["dataset"] == ds
        ds_ref: List[str] = []
        if config:
            if ds in MOLECULENET_DATASETS and config.moleculenet_test_dir:
                ds_ref = _load_moleculenet_test_class0_smiles(ds, Path(config.moleculenet_test_dir), dataset_cfg)
            elif ds == "zinc250k" and config.zinc250k_test_csv:
                ds_ref = _load_zinc250k_test_smiles(Path(config.zinc250k_test_csv))
            elif ds in TDC_DATASETS:
                # hERG (direction=decreasing) → seeds are class 1 blockers;
                # BACE/BBBP/HIV-style TDC datasets (direction=increasing) → class 0.
                # _tdc_seed_class picks the right class from DATASET_CONFIG.
                ds_ref = _load_tdc_test_seed_class_smiles(ds, dataset_cfg)

        ref_index = {smi: i for i, smi in enumerate(ds_ref)}
        next_idx = len(ref_index)
        extra_index: Dict[str, int] = {}

        # Keep a stable fallback order for any seeds not found in the dataset reference.
        ds_canon = out.loc[ds_mask, "_seed_smiles_canon"]
        for smi in ds_canon.dropna():
            if smi not in ref_index and smi not in extra_index:
                extra_index[smi] = next_idx
                next_idx += 1

        combined_index = {**ref_index, **extra_index}
        remapped = ds_canon.map(combined_index)
        fallback = out.loc[ds_mask, "original_seed_idx"]
        out.loc[ds_mask, "seed_idx"] = remapped.where(remapped.notna(), fallback).astype(int)

    out = out.drop(columns=["_seed_smiles_canon"])
    return out


# --- Unified loader dispatchers ---

def training_smiles_sources(dataset: str, config) -> list:
    """Paths `_load_training_smiles` will read for this dataset, in order.

    Exists so a run can RECORD what novelty was checked against. Without it the answer
    lives only in code, and a reader cannot tell whether a `novel` figure was measured
    against the fine-tuning split, the pretraining corpus, both, or nothing.
    """
    from .raw_loaders import MOLECULENET_DATASETS as _MN

    out = []
    if dataset in _MN and config.moleculenet_train_dir:
        found = _moleculenet_train_path(dataset, Path(config.moleculenet_train_dir))
        if found is not None:
            out.append(found)
    elif dataset == "zinc250k" and config.zinc250k_train_molecules:
        out.append(Path(config.zinc250k_train_molecules))
    out += [Path(p) for p in _declared_train_paths(dataset, config)]
    return out


def _declared_train_paths(dataset: str, config) -> list:
    """Extra novelty references a dataset declares in DATASET_CONFIG."""
    paths = (config.dataset_config or {}).get(dataset, {}).get("train_smiles")
    if not paths:
        return []
    return [paths] if isinstance(paths, (str, Path)) else list(paths)


def _load_training_smiles(dataset: str, config: PairitEvalConfig) -> Optional[Set[str]]:
    """Load train-set SMILES as canonical set for novelty.
    Returns None when no training data is configured (novelty cannot be computed)."""
    if dataset in MOLECULENET_DATASETS and config.moleculenet_train_dir:
        # Finetune-only novelty: reference set is the MoleculeNet finetune
        # split for this dataset only. ZINC-250K pretraining is deliberately
        # excluded — the paper's novelty question is "did the model produce
        # something outside its task-specific supervision," not "outside
        # anything it has ever seen." Any additional sets a dataset declares
        # via DATASET_CONFIG["train_smiles"] are still unioned in.
        combined = _load_moleculenet_train_smiles(dataset, Path(config.moleculenet_train_dir))
        for pth in _declared_train_paths(dataset, config):
            combined |= _load_smiles_file(Path(pth))
        return combined or None
    if dataset == "zinc250k" and config.zinc250k_train_molecules:
        return _load_zinc250k_train_smiles(Path(config.zinc250k_train_molecules))
    # Generic path: a dataset may declare its own training molecules in DATASET_CONFIG
    # via "train_smiles" (a parquet/csv path, or a list of them). Without this, novelty
    # is undefined for any dataset outside MoleculeNet/zinc250k and V,N,R,I collapses
    # to 0 rather than being reported as unavailable.
    paths = (config.dataset_config or {}).get(dataset, {}).get("train_smiles")
    if paths:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        combined: Set[str] = set()
        for pth in paths:
            combined |= _load_smiles_file(Path(pth))
        return combined or None
    return None


def _load_smiles_file(path: Path) -> Set[str]:
    """Read a parquet/csv of molecules and return canonical isomeric SMILES.

    Fail loud on missing file. A silent empty set makes every generation appear
    "novel" — every V,N,I metric silently inflated with no warning. If the
    caller genuinely wants to skip novelty for a dataset, it should pass
    training_smiles=None explicitly upstream, not rely on a missing file
    returning an empty set."""
    if not path.exists():
        raise FileNotFoundError(
            f"train_smiles file for novelty reference not found: {path}. "
            f"Novelty would silently be measured against nothing (every generation "
            f"marked novel), so refusing to continue. Fix the path or omit this "
            f"reference from DATASET_CONFIG['train_smiles']."
        )
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    out: Set[str] = set()
    for smi in df[col].dropna():
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            out.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return out


def _load_test_smiles(dataset: str, config: PairitEvalConfig) -> List[str]:
    """Load test-set SMILES list for sim_gt_mean."""
    if dataset in MOLECULENET_DATASETS and config.moleculenet_test_dir:
        return _load_moleculenet_test_smiles(dataset, Path(config.moleculenet_test_dir))
    if dataset == "zinc250k" and config.zinc250k_test_csv:
        return _load_zinc250k_test_smiles(Path(config.zinc250k_test_csv))
    if dataset in TDC_DATASETS:
        return _load_tdc_test_smiles(dataset, config.dataset_config or DATASET_CONFIG)
    return []


def _load_distribution_reference(dataset: str, config: PairitEvalConfig) -> List[str]:
    """Load reference SMILES list for distribution metrics (FCD/SNN/cosine).

    Fixed policy (A3): TARGET-class train molecules for classification datasets,
    full train set for zinc250k. For MoleculeNet (direction=increasing) target
    is class 1; for hERG (direction=decreasing) target is class 0. Rationale
    unchanged from the MoleculeNet original: the model generates
    seed_class → target_class improvements, and the test target-class slice is
    too small for stable FCD, so use the (larger) train target-class slice.
    Not user-configurable — see `internal/docs/notes/` if we ever need to revisit this."""
    if dataset in MOLECULENET_DATASETS and config.moleculenet_train_dir:
        return _load_moleculenet_train_class1_smiles_list(
            dataset, Path(config.moleculenet_train_dir),
        )
    if dataset == "zinc250k" and config.zinc250k_train_molecules:
        return _load_zinc250k_train_smiles_list(Path(config.zinc250k_train_molecules))
    if dataset in TDC_DATASETS:
        return _load_tdc_train_target_class_smiles_list(
            dataset, config.dataset_config or DATASET_CONFIG,
        )
    return []



# ---------- Per-molecule evaluation for one (method, dataset) block ----------

def _evaluate_one_block(
    df: pd.DataFrame,
    method: str,
    dataset: str,
    training_smiles: Optional[Set[str]],
    evaluator: Optional[Any],
    continuous_improvement_spec: Optional[Any] = None,
    min_property_delta: float = 1.0,
    extra_train_smiles: Optional[Dict[str, Set[str]]] = None,
    *,
    select_oracle: Optional[Any] = None,
    score_oracle: Optional[Any] = None,
    use_select_oracle_for_prediction: bool = False,
    direction: str = "increasing",
) -> pd.DataFrame:
    """Evaluate one (method, dataset) block of generations.

    Two-oracle protocol: `select_oracle` (train-only RF) is used to score the
    64 candidates during the ALL-64 selection pass; `score_oracle` (MiniMol) is
    used to re-score the top-5 selected candidates for the final published
    metrics. `use_select_oracle_for_prediction=True` routes this call's
    `predicted_prob` / `predicted_label` through select_oracle so downstream
    selection ranks by RF confidence, not MiniMol confidence — preventing the
    RL model from being able to reward-hack against the same scorer used at
    training time.
    """
    rows = []

    all_smiles = df["generated_smiles"].tolist()
    active_oracle = select_oracle if (use_select_oracle_for_prediction and select_oracle is not None) else score_oracle
    if active_oracle is not None:
        # An Oracle carries its own threshold, direction and comparison operator.
        predictions = active_oracle.predict_labels(all_smiles)
    elif evaluator is not None:
        predictions = property_metrics.batch_predict_binary_property(all_smiles, evaluator)
    else:
        predictions = [(None, None)] * len(all_smiles)

    for i, (_, row) in enumerate(df.iterrows()):
        seed_idx = row["seed_idx"]
        seed_smiles = row["seed_smiles"]
        seed_label = int(row["seed_label"]) if pd.notna(row["seed_label"]) else 0
        gen_smiles = row["generated_smiles"]
        variant = row.get("variant")
        molecule_type = row.get("molecule_type")

        pred_prob, pred_label = predictions[i] if i < len(predictions) else (None, None)

        def empty_row(valid: bool = False) -> Dict[str, Any]:
            base = {
                "method": method,
                "dataset": dataset,
                "seed_idx": seed_idx,
                "seed_smiles": seed_smiles,
                "seed_label": seed_label,
                "gen_idx": row["gen_idx"],
                "generated_smiles": "" if not valid else gen_smiles,
                "variant": variant,
                "molecule_type": molecule_type,
                "is_valid": valid,
                "is_novel": False,
                "v_n_i": False,
                "predicted_prob": None,
                "predicted_label": None,
                "improvement": False,
                "maintenance": False,
                "tanimoto": None,
            }
            if extra_train_smiles:
                for ref_name in extra_train_smiles:
                    base[f"is_memorized_{ref_name}"] = False
            return base

        if pd.isna(gen_smiles) or gen_smiles == "" or gen_smiles == "None":
            rows.append(empty_row(False))
            continue
        mol = Chem.MolFromSmiles(gen_smiles)
        if mol is None:
            rows.append(empty_row(False))
            continue

        tanimoto = None
        is_duplicate = False
        if seed_smiles and str(seed_smiles).strip():
            tanimoto = _compute_tanimoto(str(seed_smiles), gen_smiles)
            if tanimoto is not None and abs(tanimoto - 1.0) < 1e-6:
                is_duplicate = True

        # All novelty/memorization use canonical isomeric SMILES (Lobster-aligned)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if is_duplicate:
            is_novel = False
        elif training_smiles is None:
            is_novel = None  # Cannot determine novelty without training data
        else:
            is_novel = canonical not in training_smiles

        # Extra reference set memorization checks
        extra_memorized: Dict[str, bool] = {}
        if extra_train_smiles:
            for ref_name, ref_set in extra_train_smiles.items():
                extra_memorized[ref_name] = canonical in ref_set or is_duplicate

        # I in V+N+I: binary = improved (0→1); continuous = delta >= min_property_delta
        if continuous_improvement_spec is not None and not is_duplicate:
            # Continuous property improvement (e.g. zinc250k logP)
            try:
                seed_mol = Chem.MolFromSmiles(str(seed_smiles)) if seed_smiles else None
                if seed_mol is not None and mol is not None:
                    v_seed = continuous_improvement_spec.fn(seed_mol)
                    v_gen = continuous_improvement_spec.fn(mol)
                    if v_seed is not None and v_gen is not None:
                        delta = float(v_gen) - float(v_seed)
                        direction = (continuous_improvement_spec.direction or "higher").lower()
                        if direction == "higher":
                            improvement = delta >= min_property_delta
                        else:
                            improvement = (-delta) >= min_property_delta
                    else:
                        improvement = False
                else:
                    improvement = False
            except Exception:
                improvement = False
        else:
            improvement = property_metrics.compute_improvement(
                seed_label, pred_label, is_duplicate, direction=direction
            )
        # V_N_I: valid AND improved AND NOT memorized (Lobster 3l filtering).
        # Kept as `v_n_i` for output-schema compatibility; the historical "M"
        # (maintenance) term was dropped when 1->1 support was removed. Novel
        # and not-memorized are the same predicate here.
        # When is_novel is None (no training data), v_n_i cannot be determined.
        if is_novel is None:
            v_n_i = None
        else:
            v_n_i = improvement and is_novel

        row_dict = {
            **empty_row(True),
            "generated_smiles": gen_smiles,
            "is_novel": is_novel,
            "v_n_i": v_n_i,
            "predicted_prob": pred_prob,
            "predicted_label": pred_label,
            "improvement": improvement,
            "maintenance": False,
            "tanimoto": tanimoto,
        }
        if extra_train_smiles:
            for ref_name, is_mem in extra_memorized.items():
                row_dict[f"is_memorized_{ref_name}"] = is_mem
        rows.append(row_dict)
    return pd.DataFrame(rows)


# ---------- Summary metrics from per-molecule df ----------

def compute_summary_metrics(per_mol_df: pd.DataFrame) -> Dict[str, Any]:
    df = per_mol_df
    n_total = len(df)
    # Coverage-aware fixed denominator (opt-in via PAIRIT_EVAL_FIXED_DENOMINATOR=1):
    # replace `n_total` (actual row count — depressed when some seeds got zero
    # candidates) with n_test_seeds × top_k so per-generation rates reflect
    # method-side coverage failures honestly. top_k is sniffed from the data:
    # max group size >= 32 → all64 pass (64), else → top-5 pass (5).
    # n_test_seeds is per-dataset from DATASET_CONFIG. Both must be resolvable
    # for the override to fire, otherwise fall back to dynamic n_total.
    fixed_denom_on = os.environ.get("PAIRIT_EVAL_FIXED_DENOMINATOR") == "1"
    ds_here: Optional[str] = None
    n_test_seeds: Optional[int] = None
    if fixed_denom_on and len(df) and "dataset" in df.columns:
        ds_here = str(df["dataset"].iloc[0])
        n_test_seeds = DATASET_CONFIG.get(ds_here, {}).get("n_test_seeds")
        if n_test_seeds and "seed_idx" in df.columns:
            per_seed_sizes = df.groupby("seed_idx").size()
            max_group = int(per_seed_sizes.max()) if len(per_seed_sizes) else 0
            inferred_top_k = 64 if max_group >= 32 else 5
            n_total = int(n_test_seeds) * inferred_top_k
    if n_total == 0:
        return {}
    if "seed_smiles" in df.columns:
        processed_by_seed = df.groupby("seed_smiles")["generated_smiles"].apply(lambda s: s.notna().any())
    else:
        processed_by_seed = pd.Series(dtype=bool)
    n_seed_total = int(len(processed_by_seed))
    n_seed_processed = int(processed_by_seed.sum()) if n_seed_total else 0
    valid_all = df[df["is_valid"]]
    df_no_dup = df.copy()
    # Seed-copy detection uses a defensive `>= 1.0 - 1e-6` rather than `== 1.0`.
    # RDKit canonical Morgan fingerprints should give exactly 1.0 for identical
    # molecules (float equality holds because the bit vectors are literally
    # equal), but the epsilon guards against numerical drift if the tanimoto
    # comes from a source that pre-computes it via ratio arithmetic. Same
    # epsilon is used consistently at all seed-copy checks in this module.
    df_no_dup["is_seed_dup"] = (df_no_dup["tanimoto"].notna()) & (df_no_dup["tanimoto"] >= 1.0 - 1e-6)
    valid_no_seed_dup = valid_all[~df_no_dup.loc[valid_all.index, "is_seed_dup"]]
    # `n_valid` = "did the model output parseable SMILES?" — includes seed-copies
    # (which are still valid SMILES) so a method that copies the seed verbatim
    # (e.g. HierG2G BBBP) shows Validity ~100%, not 0%. The prior convention
    # excluded seed-copies from the Validity numerator, conflating "parseable"
    # with "usable non-seed-copy output". Downstream metrics that legitimately
    # exclude seed-copies (Novel-conditional-on-valid, per-seed Unique@5, QED
    # mean, V,I,N seed-similarity restriction) continue to use `valid_no_seed_dup`.
    n_valid = len(valid_all)
    n_total_valid = len(valid_all)
    # Novelty: None values mean training data was unavailable
    _has_novelty_data = valid_all["is_novel"].notna().any() if len(valid_all) else False
    n_novel = int(valid_all["is_novel"].sum()) if _has_novelty_data else None
    n_v_n_i = int(valid_all["v_n_i"].sum()) if "v_n_i" in valid_all.columns and valid_all["v_n_i"].notna().any() else (0 if _has_novelty_data else None)
    unique_canon = set()
    for _, r in valid_no_seed_dup.iterrows():
        s = r.get("generated_smiles")
        if pd.notna(s) and s:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                unique_canon.add(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))

    # Intra-seed canonical dedup: mark rows whose canonical SMILES already
    # appeared earlier under the same (seed, method, dataset) as `is_repeated`.
    # The top-5 pass never triggers this because selection already deduped,
    # but the raw-64 pass depends on it, as does any consumer that reads a
    # raw-64 per_generation CSV directly.
    df_no_dup["canon"] = None
    for idx, r in df_no_dup.iterrows():
        s = r.get("generated_smiles")
        if pd.notna(s) and s:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                df_no_dup.at[idx, "canon"] = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    df_no_dup["is_repeated"] = False
    group_cols = ["seed_smiles", "method", "dataset"]
    if all(c in df_no_dup.columns for c in group_cols):
        for (_seed, _method, _ds), g in df_no_dup.groupby(group_cols):
            seen = set()
            for idx in g.index:
                c = df_no_dup.at[idx, "canon"]
                if c and c in seen:
                    df_no_dup.at[idx, "is_repeated"] = True
                if c:
                    seen.add(c)
    else:
        seen = set()
        for idx in df_no_dup.index:
            c = df_no_dup.at[idx, "canon"]
            if c and c in seen:
                df_no_dup.at[idx, "is_repeated"] = True
            if c:
                seen.add(c)
    df_no_dup = df_no_dup[~(df_no_dup["is_seed_dup"] | df_no_dup["is_repeated"])]
    valid_df = df_no_dup[df_no_dup["is_valid"]]
    n_v = len(valid_df)
    if n_v == 0:
        n_vi = 0
        n_vn = 0 if _has_novelty_data else None
        tanimoto_mean = tanimoto_std = diversity_mean = diversity_std = sim_seed_v_n_i = None
    else:
        n_vi = int(valid_df["improvement"].sum())
        if _has_novelty_data:
            _novel_filled = valid_df["is_novel"].fillna(False)
            n_vn = int(_novel_filled.sum())
        else:
            n_vn = None
        t_vals = valid_df["tanimoto"].dropna()
        t_vals = t_vals[t_vals < 1.0 - 1e-6]
        tanimoto_mean = float(t_vals.mean()) if len(t_vals) else None
        tanimoto_std = float(t_vals.std()) if len(t_vals) > 1 else None
        # V_N_I-restricted sim_seed (valid AND improved AND NOT memorized)
        v_n_i_df = valid_df[valid_df["v_n_i"].fillna(False)] if "v_n_i" in valid_df.columns else valid_df.iloc[0:0]
        t_v_n_i = v_n_i_df["tanimoto"].dropna()
        t_v_n_i = t_v_n_i[t_v_n_i < 1.0 - 1e-6] if len(t_v_n_i) else t_v_n_i
        sim_seed_v_n_i = float(t_v_n_i.mean()) if len(t_v_n_i) else None
        seed_diversities = []
        if "seed_smiles" in df.columns and "generated_smiles" in df.columns:
            for seed_smiles in df["seed_smiles"].unique():
                sub = df[(df["seed_smiles"] == seed_smiles) & (df["is_valid"])]
                gens = sub["generated_smiles"].dropna().tolist()
                if len(gens) >= 2:
                    try:
                        div, _, n_u = internal_diversity(gens, deduplicate=True)
                        if div is not None and n_u >= 2:
                            seed_diversities.append(div)
                    except Exception:
                        pass
        diversity_mean = float(np.mean(seed_diversities)) if seed_diversities else None
        diversity_std = float(np.std(seed_diversities)) if len(seed_diversities) > 1 else None

    n_improved = int(df_no_dup[df_no_dup["is_valid"]]["improvement"].sum()) if n_v else 0
    # `n_maintained` is a dead 1->1 metric kept for output-schema compatibility.
    n_maintained = 0

    # Novelty-dependent metrics: NaN when training data was unavailable
    if _has_novelty_data:
        novel_pct = 100 * n_novel / n_total_valid if n_total_valid else 0
        v_n_pct = 100 * (n_vn if n_v else 0) / n_total if n_total else 0
        _novel_filled_vdf = valid_df["is_novel"].fillna(False) if n_v else pd.Series(dtype=bool)
        n_v_n_i = int((_novel_filled_vdf & valid_df["improvement"]).sum()) if n_v else 0
        v_n_i_pct = 100 * n_v_n_i / n_total if n_total and n_v_n_i is not None else 0.0
    elif n_total_valid == 0:
        # No valid generation at all. The novelty-dependent composite is then
        # determinately zero -- nothing can be valid-and-improved-and-novel when
        # nothing is valid -- regardless of whether a novelty reference exists.
        #
        # Reporting None here would make a method that produced 100% unparseable
        # output look like MISSING DATA rather than a zero, so it could be
        # silently dropped from a table or averaged around. Only `novel` itself
        # stays None: it is a rate conditional on validity, i.e. 0/0.
        novel_pct = None
        v_n_pct = 0.0
        n_v_n_i = 0
        v_n_i_pct = 0.0
    else:
        # A novelty reference was genuinely not configured, so novelty is
        # unknown rather than zero.
        novel_pct = None
        v_n_pct = None
        n_v_n_i = None
        v_n_i_pct = None

    result = {
        "n_total": n_total,
        "n_processed": n_seed_processed,
        "n_valid": n_valid,
        "n_v_n": n_vn if n_v else (0 if _has_novelty_data else None),
        "n_v_i": n_vi if n_v else 0,
        "n_v_n_i": n_v_n_i,
        "n_novel": n_novel,
        "n_improved": n_improved,
        "n_maintained": n_maintained,
        "processed": 100 * n_seed_processed / n_seed_total if n_seed_total else 0,
        "valid": 100 * n_valid / n_total,
        "v_n": v_n_pct,
        "v_i": 100 * (n_vi if n_v else 0) / n_total if n_total else 0,
        "v_n_i": v_n_i_pct,
        "novel": novel_pct,
        "improved": 100 * n_improved / n_v if n_v else 0,
        "maintained": 0.0,
        "sim_seed": tanimoto_mean,
        "sim_seed_std": tanimoto_std,
        "sim_seed_v_n_i": sim_seed_v_n_i if n_v else None,
        "internal_diversity": diversity_mean,
        "internal_diversity_std": diversity_std,
        # `unique` = per-seed Unique@5. For each seed, count distinct canonical
        # SMILES among the valid, non-seed-duplicate top-5 rows and divide by
        # 5 (fixed denominator = paper's top-5 slot count). Mean across seeds.
        # A seed with 3 unique canonicals in top-5 → 3/5 = 60%; a seed that
        # produced only seed-copies contributes 0/5 = 0% to the mean IF it
        # still has any row in the top-5 selection (dedup keeps distinct
        # canonicals; seed-copies drop out here via valid_no_seed_dup).
        "unique": _per_seed_unique_at_5(
            valid_no_seed_dup,
            all_seed_ids=df["seed_idx"].unique() if "seed_idx" in df.columns else None,
            # Under fixed-denominator mode, include ALL expected test seeds in
            # the roster (missing seeds contribute 0/5) so Unique@5 and V,I,N
            # give consistent coverage signals. Same env flag as the n_total
            # override above.
            n_expected_seeds=n_test_seeds if fixed_denom_on else None,
        ),
        "qed": _mean_qed(valid_no_seed_dup["generated_smiles"]),
    }
    return result


def _per_seed_unique_at_5(
    valid_no_seed_dup: pd.DataFrame,
    all_seed_ids: Optional[Sequence] = None,
    n_expected_seeds: Optional[int] = None,
) -> Optional[float]:
    """Per-seed Unique@5 = mean over seeds of (distinct canonical SMILES in
    valid non-seed-dup top-5 / 5). Denominator is fixed at 5 (paper's top-5
    slot count) so a seed with only 3 unique canonicals contributes 60%.

    `all_seed_ids` is the full seed roster for this method-dataset. Seeds in
    that roster with zero valid non-seed-duplicate rows count as 0/5 = 0% —
    critical for HierG2G-style methods that copy the seed verbatim (many
    seeds produce zero non-seed-copy canonicals; those SHOULD drag the mean
    down, not be silently omitted).

    `n_expected_seeds` (Option A): when set, the mean is taken over exactly
    this many seeds. Seeds present in the data contribute their real count;
    missing seeds contribute 0. Handles the "method addressed only 78 of 85
    seeds" case (MMP-Prop hERG, PairIT-Beam hERG) by making Unique@5 reflect
    coverage failure. Ignored (falls back to actual-seed-count behavior) when
    None.

    Returns None only when there are no seeds at all (empty input).
    """
    if all_seed_ids is not None:
        seed_roster = list(pd.Series(all_seed_ids).dropna().unique())
    elif valid_no_seed_dup is not None and "seed_idx" in valid_no_seed_dup.columns:
        seed_roster = list(pd.Series(valid_no_seed_dup["seed_idx"]).dropna().unique())
    else:
        seed_roster = []
    if not seed_roster and not n_expected_seeds:
        return None
    counts_per_seed = {sid: 0 for sid in seed_roster}
    if valid_no_seed_dup is not None and len(valid_no_seed_dup) and "seed_idx" in valid_no_seed_dup.columns:
        for sid, seed_group in valid_no_seed_dup.groupby("seed_idx", sort=False):
            if sid not in counts_per_seed:
                # Row references a seed not in the passed-in roster — include
                # it rather than silently drop, so no per-seed value is lost.
                counts_per_seed[sid] = 0
            canons: Set[str] = set()
            for s in seed_group["generated_smiles"]:
                if pd.notna(s) and s:
                    m = Chem.MolFromSmiles(str(s))
                    if m is not None:
                        canons.add(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))
            counts_per_seed[sid] = len(canons)
    if n_expected_seeds and n_expected_seeds > 0:
        # Option A: sum over observed seeds, divide by n_expected_seeds. Missing
        # seeds implicitly contribute 0 (coverage failure). This differs from
        # np.mean() over observed seeds only when coverage < 100%.
        total_fraction = sum(c / 5.0 for c in counts_per_seed.values())
        return 100.0 * total_fraction / int(n_expected_seeds)
    return 100.0 * float(np.mean([c / 5.0 for c in counts_per_seed.values()]))


def _mean_qed(smiles: pd.Series) -> Optional[float]:
    """Mean QED over parseable molecules, or None if none parse.

    Reported on the same population as `unique` (valid, seed-duplicates removed) so the
    two are read off the same denominator. QED occasionally raises on odd valences rather
    than returning a value, so failures are skipped rather than counted as zero -- a zero
    would drag the mean toward "undruglike" for what is really a computation failure.
    """
    from rdkit.Chem import QED

    vals = []
    for s in smiles:
        if not (pd.notna(s) and s):
            continue
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        try:
            vals.append(QED.qed(m))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else None


def _build_per_seed(per_mol_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-seed summary from per-generation DataFrame. One row per (method, dataset, seed_idx)."""
    if len(per_mol_df) == 0:
        return pd.DataFrame()
    rows = []
    for (method, dataset, seed_idx, seed_smiles), grp in per_mol_df.groupby(
        ["method", "dataset", "seed_idx", "seed_smiles"]
    ):
        valid = grp[grp["is_valid"]]
        n_total = len(grp)
        is_processed = bool(grp["generated_smiles"].notna().any())
        n_valid = len(valid)
        _has_novel = valid["is_novel"].notna().any() if len(valid) else False
        n_improved = int(valid["improvement"].sum())
        t = valid["tanimoto"].dropna()
        t = t[t < 1.0 - 1e-6]
        tanimoto_mean = float(t.mean()) if len(t) else None
        gens = valid["generated_smiles"].dropna().tolist()
        div, _, _ = internal_diversity(gens, deduplicate=True) if len(gens) >= 2 else (None, 0, 0)
        if _has_novel:
            _novel_filled = valid["is_novel"].fillna(False)
            n_novel = int(_novel_filled.sum())
            n_v_n_i = int((_novel_filled & valid["improvement"]).sum())
        else:
            n_novel = None
            n_v_n_i = None
        rows.append({
            "method": method,
            "dataset": dataset,
            "seed_idx": seed_idx,
            "seed_smiles": seed_smiles,
            "n_total": n_total,
            "n_processed": int(is_processed),
            "n_valid": n_valid,
            "n_v_n": n_novel,
            "n_v_i": n_improved,
            "n_v_n_i": n_v_n_i,
            "n_novel": n_novel,
            "n_improved_or_maintained": n_improved,
            "processed": 100.0 if is_processed else 0.0,
            "valid": 100 * n_valid / n_total if n_total else 0,
            "v_n": (100 * n_novel / n_total if n_total else 0) if n_novel is not None else None,
            "v_i": 100 * n_improved / n_total if n_total else 0,
            "v_n_i": (100 * n_v_n_i / n_total if n_total else 0) if n_v_n_i is not None else None,
            "novel": (100 * n_novel / n_valid if n_valid else 0) if n_novel is not None else None,
            "improved": 100 * n_improved / n_valid if n_valid else 0,
            "sim_seed": tanimoto_mean,
            "internal_diversity": div,
        })
    return pd.DataFrame(rows)


def v_n_i_mask(per_generation_df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask for V_N_I filtering (Lobster 3l): valid AND improved AND NOT memorized.
    Use to filter per-generation rows for reporting metrics on 'good' generations only.
    """
    if per_generation_df is None or len(per_generation_df) == 0:
        return pd.Series(dtype=bool)
    if "v_n_i" in per_generation_df.columns:
        return per_generation_df["v_n_i"].fillna(False)
    valid = per_generation_df["is_valid"].fillna(False)
    imp = per_generation_df["improvement"].fillna(False)
    novel = per_generation_df["is_novel"].fillna(False)
    return valid & imp & novel


# ---------- Public API ----------

def evaluate(
    df: pd.DataFrame,
    config: Optional[PairitEvalConfig] = None,
    methods: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    verbose: bool = True,
    compute_distribution_metrics: bool = False,
    compute_property_metrics: bool = False,
    property_names: Optional[List[str]] = None,
    property_delta_binning: Optional[Dict[str, Any]] = None,
    property_preservation_names: Optional[List[str]] = None,
    property_sim_band_range: Optional[Tuple[float, float]] = (0.4, 0.6),
    property_top_k: Optional[int] = None,
    min_property_delta: float = 1.0,
    reuse_precomputed: bool = False,
    extra_train_smiles: Optional[Dict[str, Set[str]]] = None,
    use_select_oracle_for_prediction: bool = False,
) -> Stage1Result:
    """
    Stage 1: Compute metrics on generations. Returns per_generation, per_seed, and average.

    Parameters
    ----------
    df : DataFrame
        Must have columns: method, dataset, seed_idx, seed_smiles, seed_label, gen_idx, generated_smiles.
    config : PairitEvalConfig or None
        If provided, enables novelty (moleculenet_train_dir / zinc250k_train_molecules)
        and improvement (evaluator_dir). If None, novelty and improvement are skipped.
    methods : list of str or None
        If set, filter df to these methods only.
    datasets : list of str or None
        If set, filter df to these datasets only.
    verbose : bool
        Whether to use tqdm and print progress.
    compute_distribution_metrics : bool
        If True, compute slow distribution-level metrics (FCD, SNN, FragCos, ScafCos,
        Coverage, W1, KDE/LL) per (method, dataset) and add them to the average table.
        Requires config with moleculenet_train_dir / zinc250k_train_molecules.
        Reference set is class-1 train molecules for MoleculeNet, full train
        set for zinc250k (see `_load_distribution_reference`).
        Missing deps (e.g. scipy, fcd) cause that metric to be None.
    compute_property_metrics : bool
        If True, compute continuous property improvement stats (mean_delta, frac_improved, etc.)
        per property and add to the average table. Requires property_names. These are extras
        (e.g. logp, qed); binary improvement does not use property_names —
        it uses only the dataset evaluator (seed_label from CSV, predicted from evaluator).
    property_names : list of str or None
        Property names for the optional property_metrics only (e.g. ["logp", "qed"]).
        Not used for binary improvement; the dataset (bace, bbbp, hiv) alone determines the evaluator.
    property_delta_binning : dict or None
        Optional. If provided, keys are property names and values are PropertyBinningConfig
        (from property_metrics). For each key present in the block's property_specs, adds
        property_mean_bin / property_median_bin to the average table (binned |delta| summary).
    property_preservation_names : list of str or None
        Optional. RDKit-only property names (e.g. logp, qed, sa, mw, tpsa, hbd, hba,
        rotatable_bonds, frac_sp3_c) for property preservation MAD. For each, adds
        preservation_mad/{name} = mean(|prop(gen) - prop(seed)|) to the average table,
        plus         preservation_mad_macro_mean. Classifier properties (Class, p_np, HIV_active)
        are ignored for preservation.
    property_sim_band_range : tuple (lo, hi) or None
        When not None and property metrics are computed, also compute improvement stats
        restricted to pairs where lo <= Tanimoto(seed, gen) <= hi. Default (0.4, 0.6).
        Adds property_*_sim_band/{name} to the average table. Use None to disable.
    property_top_k : int or None
        When not None and property metrics are computed, add best property value and
        mean of top-k values among valid generations (property_best/{name},
        property_mean_top_k/{name}). Requires property_specs.
    min_property_delta : float
        For continuous-property datasets (e.g. zinc250k with logP): minimum delta to count
        as "improvement" in v_n_i. Default 1.0 (logP units). Use 3.0 for stricter.

    Returns
    -------
    Stage1Result
        .per_generation : one row per generated molecule (is_valid, is_novel, improvement, tanimoto, ...)
        .per_seed       : one row per (method, dataset, seed) with counts and pct_* for that seed
        .average        : one row per (method, dataset) — quick view for comparing methods or monitoring training
    """
    df = df.copy()
    # For continuous datasets (e.g. zinc250k), seed_label is optional; default to 0
    if "seed_label" not in df.columns:
        datasets_present = set(df["dataset"].unique()) if "dataset" in df.columns else set()
        all_continuous = all(
            (config.dataset_config if config else DATASET_CONFIG).get(ds, {}).get("evaluator_type") == "continuous"
            for ds in datasets_present
        )
        if all_continuous and datasets_present:
            df["seed_label"] = 0
        else:
            raise ValueError("DataFrame must have column 'seed_label'")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"DataFrame must have column '{col}'")
    if methods is not None:
        df = df[df["method"].isin(methods)]
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]
    if len(df) == 0:
        return Stage1Result(per_generation=pd.DataFrame(), per_seed=pd.DataFrame(), average=pd.DataFrame())
    df = _normalize_seed_indexing(df, config)

    evaluator_dir = config.get_evaluator_dir() if config else None

    resources: Dict[str, Dict[str, Any]] = {}
    for ds in df["dataset"].unique():
        resources[ds] = {
            "training_smiles": None,
            "evaluator": None,
            "select_oracle": None,
            "score_oracle": None,
            "direction": "increasing",
            "test_smiles": None,
            "distribution_ref_smiles": None,
        }
        if config:
            resources[ds]["training_smiles"] = _load_training_smiles(ds, config)
            resources[ds]["test_smiles"] = _load_test_smiles(ds, config)
            if compute_distribution_metrics:
                resources[ds]["distribution_ref_smiles"] = _load_distribution_reference(ds, config)
        ds_config = (config.dataset_config if config else DATASET_CONFIG).get(ds, {})
        evaluator_type = ds_config.get("evaluator_type", "binary")
        # Family-aware oracle dispatch. DEFAULT_EVALUATOR_DIR points at
        # ASSETS/oracles/moleculenet/evaluators/ for backwards compat, but
        # hERG (and other TDC datasets) live at ASSETS/oracles/tdc/evaluators/.
        # Rebase the family segment per-dataset so a single evaluator_dir
        # config value handles both MoleculeNet + TDC in one pipeline pass.
        # Family is read from DATASET_CONFIG and defaults to "moleculenet" so
        # pre-family configs still resolve correctly.
        #
        # The rebase MUST happen above the `if evaluator_type == "binary"`
        # gate so BOTH the RF-evaluator load (below) AND the resolve_oracle_pair
        # call (further down) get the family-rebased dir — otherwise a TDC
        # dataset picks up /oracles/moleculenet/ in resolve_oracle_pair even
        # after the first load succeeds, and crashes on the missing joblib.
        family = ds_config.get("family", "moleculenet")
        if evaluator_dir and family != "moleculenet" and "/oracles/moleculenet/" in str(evaluator_dir):
            ds_evaluator_dir = Path(
                str(evaluator_dir).replace("/oracles/moleculenet/", f"/oracles/{family}/")
            )
        else:
            ds_evaluator_dir = evaluator_dir
        if evaluator_type == "binary" and evaluator_dir:
            try:
                resources[ds]["evaluator"] = property_metrics.load_evaluator(
                    ds, ds_evaluator_dir, dataset_config=config.dataset_config if config else None
                )
            except FileNotFoundError as e:
                # Fail loud: a missing evaluator silently zeros the Improved
                # column and every VIN-based composite. That's undetectable
                # in the output CSV (looks like the method just didn't
                # improve anything). Either fix the evaluator_dir or drop
                # the dataset from `datasets=`.
                raise FileNotFoundError(
                    f"Evaluator not loaded for dataset '{ds}' (family={family}, "
                    f"resolved dir={ds_evaluator_dir}): {e}. "
                    f"Binary improvement would be 0 for every generation and every "
                    f"VIN metric silently zeroed — refusing to continue. Point "
                    f"evaluator_dir at the correct .joblib directory (e.g. "
                    f"ASSETS/oracles/{{family}}/evaluators/) or drop '{ds}' from "
                    f"the datasets list."
                ) from e
        # Improvement direction. Explicit per dataset: hERG and FreeSolv are
        # `decreasing`, and defaulting them to `increasing` inverts improvement.
        resources[ds]["direction"] = oracles.normalize_direction(ds_config.get("direction"))

        # Selection and scoring oracles. When only one is configured both point at
        # it, which is exactly the historical single-oracle behaviour.
        select_spec = getattr(config, "select_oracle", None) if config else None
        score_spec = getattr(config, "score_oracle", None) if config else None
        if select_spec is not None or score_spec is not None:
            try:
                sel, sco = oracles.resolve_oracle_pair(
                    select_spec,
                    score_spec,
                    ds_evaluator_dir if evaluator_type == "binary" else None,
                    ds,
                    dataset_config=config.dataset_config if config else None,
                )
                resources[ds]["select_oracle"] = sel
                resources[ds]["score_oracle"] = sco
                # Runtime two-oracle guard: reward-hacking config is when
                # `use_select_oracle_for_prediction=True` (all-64 selection
                # pass) but select == score, so the same oracle both filters
                # AND judges. Fail loud here rather than let bad numbers land.
                if use_select_oracle_for_prediction and sel is not None and sel is sco:
                    raise RuntimeError(
                        f"Two-oracle protocol violated for dataset '{ds}': "
                        f"select_oracle and score_oracle resolved to the SAME oracle "
                        f"({type(sel).__name__}). This is the reward-hacking "
                        f"configuration we exclude. Provide distinct oracles: e.g. "
                        f"evaluator_dir=<RF joblib dir> and score_oracle='minimol'."
                    )
            except (FileNotFoundError, ValueError) as e:
                raise RuntimeError(f"Could not resolve oracles for dataset '{ds}': {e}") from e

        # Property metrics (continuous + optional binned): need specs per dataset for classifier props
        # For continuous datasets (e.g. zinc250k), use default_property if property_names not specified
        resources[ds]["property_specs"] = []
        effective_property_names = property_names
        if compute_property_metrics and not effective_property_names and ds_config.get("default_property"):
            effective_property_names = [ds_config["default_property"]]
        if compute_property_metrics and effective_property_names:
            try:
                resources[ds]["property_specs"] = property_metrics.get_property_specs(
                    effective_property_names,
                    evaluator_dir=evaluator_dir if evaluator_type == "binary" else None,
                    dataset_config=config.dataset_config if config else None,
                )
            except (ValueError, FileNotFoundError):
                pass
        # For continuous datasets: spec used for improvement in v_n_i (delta >= min_property_delta)
        resources[ds]["continuous_improvement_spec"] = (
            resources[ds]["property_specs"][0] if evaluator_type == "continuous" and resources[ds]["property_specs"] else None
        )

    # Property preservation MAD: RDKit-only specs (no evaluator)
    preservation_specs: List[Any] = []
    if property_preservation_names:
        rdkit_only = [n for n in property_preservation_names if n in property_metrics.PROPERTY_REGISTRY]
        if rdkit_only:
            try:
                preservation_specs = property_metrics.get_property_specs(
                    rdkit_only, evaluator_dir=None, dataset_config=None
                )
            except (ValueError, FileNotFoundError):
                pass

    all_per_mol = []
    summary_list = []
    groupers = list(df.groupby(["method", "dataset"]))
    it = tqdm(groupers, desc="Evaluate") if verbose else groupers
    for (method, dataset), sub in it:
        res = resources.get(
            dataset,
            {"training_smiles": None, "evaluator": None, "test_smiles": None,
             "distribution_ref_smiles": None, "property_specs": [], "continuous_improvement_spec": None,
             "select_oracle": None, "score_oracle": None, "direction": "increasing"},
        )
        if reuse_precomputed and all(c in sub.columns for c in COMPUTED_COLUMNS):
            keep_cols = list(REQUIRED_COLUMNS) + list(COMPUTED_COLUMNS)
            for opt in ("variant", "molecule_type"):
                if opt in sub.columns:
                    keep_cols.append(opt)
            per_mol = sub[keep_cols].copy().reset_index(drop=True)
        else:
            per_mol = _evaluate_one_block(
                sub,
                method,
                dataset,
                res["training_smiles"],
                res.get("evaluator"),
                continuous_improvement_spec=res.get("continuous_improvement_spec"),
                min_property_delta=min_property_delta,
                extra_train_smiles=extra_train_smiles,
                select_oracle=res.get("select_oracle"),
                score_oracle=res.get("score_oracle"),
                use_select_oracle_for_prediction=use_select_oracle_for_prediction,
                direction=res.get("direction", "increasing"),
            )
        all_per_mol.append(per_mol)
        sm = compute_summary_metrics(per_mol)
        sm["method"] = method
        sm["dataset"] = dataset

        is_seed_dup = (per_mol["tanimoto"].notna()) & (per_mol["tanimoto"] >= 1.0 - 1e-6)
        valid_df = per_mol[(per_mol["is_valid"]) & ~is_seed_dup]
        gen_smiles = valid_df["generated_smiles"].dropna().unique().tolist()

        if res.get("test_smiles") and gen_smiles:
            sm["sim_gt_mean"] = max_similarity_to_gt(gen_smiles, res["test_smiles"])
            v_n_i_valid = valid_df["v_n_i"] if "v_n_i" in valid_df.columns else pd.Series(False, index=valid_df.index)
            gen_smiles_v_n_i = valid_df.loc[v_n_i_valid, "generated_smiles"].dropna().unique().tolist()
            if gen_smiles_v_n_i:
                sm["sim_gt_mean_v_n_i"] = max_similarity_to_gt(gen_smiles_v_n_i, res["test_smiles"])

        if compute_distribution_metrics and gen_smiles:
            # Fail loud: distribution metrics MUST use the configured class-1
            # reference (A3). Silent fallback to test_smiles (which is the
            # sim_gt_mean reference — a different, larger set) was masking
            # misconfigured moleculenet_train_dir or class-1 filters that
            # yielded empty. If the class-1 ref legitimately isn't available,
            # distribution metrics should be OFF, not silently wrong.
            dist_ref = res.get("distribution_ref_smiles")
            if not dist_ref:
                raise ValueError(
                    f"Distribution metrics requested for method={method!r} "
                    f"dataset={dataset!r} but distribution_ref_smiles is empty. "
                    f"Check moleculenet_train_dir points at data with class-1 "
                    f"molecules for this dataset. Do NOT fall back to test_smiles "
                    f"— that is a different reference set for sim_gt_mean."
                )
            from . import distribution_metrics as dm
            dist = dm.compute_distribution_metrics(
                gen_smiles, dist_ref, min_generated=1
            )
            sm.update(dist)

        # Continuous and optional binned property metrics
        if res.get("property_specs"):
            seeds = per_mol["seed_smiles"].tolist()
            gens = per_mol["generated_smiles"].tolist()
            for spec in res["property_specs"]:
                try:
                    stats = property_metrics.property_improvement_stats(seeds, gens, spec)
                    sm[f"property_mean_delta/{spec.name}"] = stats["mean_delta"]
                    sm[f"property_median_delta/{spec.name}"] = stats["median_delta"]
                    sm[f"property_std_delta/{spec.name}"] = stats["std_delta"]
                    sm[f"property_frac_improved/{spec.name}"] = stats["frac_improved"]
                    sm[f"property_n/{spec.name}"] = stats["n"]
                except Exception:
                    pass
                if property_delta_binning and spec.name in property_delta_binning:
                    try:
                        bin_cfg = property_delta_binning[spec.name]
                        bin_summary = property_metrics.property_delta_bin_summary(
                            seeds, gens, spec, bin_cfg
                        )
                        sm[f"property_mean_bin/{spec.name}"] = bin_summary["mean_bin"]
                        sm[f"property_median_bin/{spec.name}"] = bin_summary["median_bin"]
                    except Exception:
                        pass

        # Property improvement under similarity constraint (default band 0.4–0.6)
        if res.get("property_specs") and property_sim_band_range is not None:
            sim_lo, sim_hi = property_sim_band_range[0], property_sim_band_range[1]
            valid = per_mol["is_valid"].fillna(False)
            t = per_mol["tanimoto"]
            in_band = valid & t.notna() & (t >= sim_lo) & (t <= sim_hi)
            if in_band.any():
                band_df = per_mol.loc[in_band]
                seeds_band = band_df["seed_smiles"].tolist()
                gens_band = band_df["generated_smiles"].tolist()
                for spec in res["property_specs"]:
                    try:
                        stats_band = property_metrics.property_improvement_stats(
                            seeds_band, gens_band, spec
                        )
                        sm[f"property_mean_delta_sim_band/{spec.name}"] = stats_band["mean_delta"]
                        sm[f"property_median_delta_sim_band/{spec.name}"] = stats_band["median_delta"]
                        sm[f"property_std_delta_sim_band/{spec.name}"] = stats_band["std_delta"]
                        sm[f"property_frac_improved_sim_band/{spec.name}"] = stats_band["frac_improved"]
                        sm[f"property_n_sim_band/{spec.name}"] = stats_band["n"]
                    except Exception:
                        pass

        # Top-k best property values among valid generations
        if res.get("property_specs") and property_top_k is not None and property_top_k > 0:
            valid_gens = per_mol.loc[per_mol["is_valid"], "generated_smiles"].dropna().tolist()
            if valid_gens:
                for spec in res["property_specs"]:
                    try:
                        topk = property_metrics.property_top_k_stats(
                            valid_gens, spec, property_top_k
                        )
                        sm[f"property_best/{spec.name}"] = topk["best"]
                        sm[f"property_mean_top_k/{spec.name}"] = topk["mean_top_k"]
                        sm[f"property_n_top_k/{spec.name}"] = topk["n_top_k"]
                    except Exception:
                        pass

        # Property preservation MAD (RDKit-only; user chooses which properties)
        if preservation_specs:
            seeds_list = per_mol["seed_smiles"].tolist()
            gens_list = per_mol["generated_smiles"].tolist()
            try:
                mad_out = property_metrics.property_preservation_mad(
                    seeds_list, gens_list, preservation_specs
                )
                sm.update(mad_out)
            except Exception:
                pass

        summary_list.append(sm)

    per_generation_df = pd.concat(all_per_mol, ignore_index=True) if all_per_mol else pd.DataFrame()
    per_seed_df = _build_per_seed(per_generation_df) if len(per_generation_df) else pd.DataFrame()
    average_df = pd.DataFrame(summary_list) if summary_list else pd.DataFrame()
    return Stage1Result(
        per_generation=per_generation_df,
        per_seed=per_seed_df,
        average=average_df,
        resources=resources,
    )
