"""
Comparison-matrix pipeline for baseline 64-generation runs and PairIt variants.

Run this module from an environment with Parquet support (for PairIT best-of-N inputs),
for example the user's `dsbm-b200` environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pairit_eval_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem

from pairit._assets import ASSETS
from .cli_utils import DEFAULT_MOLECULENET_TEST_DIR, build_eval_config
from .config import DATASET_CONFIG, PairitEvalConfig
from .extra_baselines import extra_baselines, insert_after
from .evaluate import evaluate
from .normalize import normalize_generation_table
from .select import canonical as _canonical_smiles


def _seed_label_by_dataset(datasets: Sequence[str]) -> Dict[str, int]:
    """Map each dataset → the class label of its seeds (0 or 1).

    MoleculeNet convention (direction=increasing): seeds are class 0. Two TDC
    datasets in the pipeline flip this: hERG (direction=decreasing) seeds are
    blockers = class 1. Without this map `normalize_generation_table` filters
    the reference test parquet to class 0 seeds for every dataset — for hERG
    that yields 26 seeds instead of 85, silently dropping 70% of the seeds
    every method actually generated for.
    """
    out: Dict[str, int] = {}
    for ds in datasets:
        direction = str(DATASET_CONFIG.get(ds, {}).get("direction", "increasing")).lower()
        out[ds] = 1 if direction == "decreasing" else 0
    return out


DATASETS: Tuple[str, ...] = ("bace", "bbbp", "hiv", "herg")
# MoleculeNet subset — used by loaders / plot code that only knows the original
# three datasets (test-parquet loaders, plot titles, per-dataset MMP paths, etc.).
MOLECULENET_DATASETS: Tuple[str, ...] = ("bace", "bbbp", "hiv")
# TDC subset — used to route hERG-family sources under `assets/generations/<method>/
# tdc/herg/…` rather than the MoleculeNet layout.
TDC_DATASETS: Tuple[str, ...] = ("herg",)
CANONICAL_COLUMNS: List[str] = [
    "method",
    "dataset",
    "seed_idx",
    "seed_smiles",
    "seed_label",
    "gen_idx",
    "generated_smiles",
]

REAL_ONLY_TRAIN_DIR = ASSETS / "data" / "moleculenet"

REGULAR_METRICS: List[str] = [
    "v_n_i",
    "valid",
    "novel",
    "improved",
    "sim_seed",
    "internal_diversity",
]
REGULAR_DIRECTIONS = {
    "v_n_i": "higher",
    "valid": "higher",
    "novel": "higher",
    "improved": "higher",
    "sim_seed": "higher",
    "internal_diversity": "higher",
}

DISTRIBUTION_METRICS: List[str] = [
    "fcd",
    "snn_mean",
    "fragment_cosine",
    "scaffold_cosine",
    "coverage_ratio",
    "w1_macro_mean",
    "kde_ll_delta_macro_mean",
]
DISTRIBUTION_DIRECTIONS = {
    "fcd": "lower",
    "snn_mean": "higher",
    "fragment_cosine": "higher",
    "scaffold_cosine": "higher",
    "coverage_ratio": "higher",
    "w1_macro_mean": "lower",
    "kde_ll_delta_macro_mean": "higher",
}


@dataclass(frozen=True)
class SourceSpec:
    base_method: str
    display_base: str
    family: str
    source_group: str
    dataset: str
    train_regime: str
    novelty_reference: str
    source_kind: str
    path: Path
    source_n_generations: int


def _canonicalize_smiles_set(smiles: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for smi in smiles:
        if pd.isna(smi) or not str(smi).strip():
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            out.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return out


def _canonicalize_one_smiles(smi: object) -> Optional[str]:
    if pd.isna(smi) or not str(smi).strip():
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _seed_order_map(smiles: Sequence[object]) -> Dict[str, int]:
    ordered = list(dict.fromkeys(str(s) for s in smiles))
    return {s: i for i, s in enumerate(ordered)}


def _string_or_blank(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str)


def _moleculenet_split_parquet(root: Path, dataset: str, split: str) -> Path:
    """Locate the molecules parquet for one (dataset, split), tolerant of both
    the ASSETS bundle naming (`molecules.parquet`) and the canonical on-disk
    naming (`molecules_{ds}_real.parquet`). Returns the first that exists.

    A single loader used to hardcode one convention and then break under the
    other — this shim makes the loaders portable across a fresh ASSETS
    checkout and an existing researcher's tree without any env-var dance."""
    short = root / dataset / split / "molecules.parquet"
    if short.exists():
        return short
    long_ = root / dataset / split / f"molecules_{dataset}_real.parquet"
    return long_  # returned even if missing — caller gets a clear FileNotFoundError


def _load_test_class0(dataset: str, moleculenet_test_dir: Path = DEFAULT_MOLECULENET_TEST_DIR) -> pd.DataFrame:
    label_cols = {"bace": "Class", "bbbp": "p_np", "hiv": "HIV_active"}
    path = _moleculenet_split_parquet(moleculenet_test_dir, dataset, "test")
    df = pd.read_parquet(path)
    return df[df[label_cols[dataset]] == 0].reset_index(drop=True)


def _load_train_smiles(dataset: str) -> Set[str]:
    path = _moleculenet_split_parquet(REAL_ONLY_TRAIN_DIR, dataset, "train")
    df = pd.read_parquet(path)
    return _canonicalize_smiles_set(df["smiles"])


def _all64_method_name(spec: SourceSpec) -> str:
    return f"{spec.base_method}-all64"


def _final_method_name(spec: SourceSpec, selection_mode: str) -> str:
    if spec.source_n_generations == 64:
        return f"{spec.base_method}-{selection_mode}"
    return spec.base_method


def _display_name(spec: SourceSpec, selection_mode: str) -> str:
    if spec.source_n_generations == 64:
        suffix = "first5" if selection_mode == "first5" else "best64"
        return f"{spec.display_base} | {suffix}"
    return f"{spec.display_base} | direct5"


def _dataset_family(dataset: str) -> str:
    """Return the on-disk generations root for one dataset ('moleculenet' or 'tdc').

    ASSETS generations layout keys datasets by family: `assets/generations/
    {method}/moleculenet/{bace,bbbp,hiv}/…` vs `assets/generations/{method}/
    tdc/{herg,…}/…`. Reads the family declared on `DATASET_CONFIG[dataset]`
    so a new dataset only needs a config entry, not a code edit here.
    """
    family = DATASET_CONFIG.get(dataset, {}).get("family")
    if family:
        return family
    raise ValueError(
        f"Unknown dataset family for '{dataset}' — set DATASET_CONFIG[{dataset!r}]['family']."
    )


def _build_specs() -> List[SourceSpec]:
    """Assemble the SourceSpec list for the paper's baseline comparison.

    Every path is ASSETS-relative under `assets/generations/{method}/
    {moleculenet|tdc}/{dataset}/{filename}`. A missing file silently skips its
    spec, so a partial bundle runs the baselines it has. The bundle location is
    overridable at runtime via PAIRIT_ASSETS_DIR.

    Registered families: nfbo, ddsbm-bridge, ddsbm-sb, hierg2g, mmp,
    mmp-prop (if present), pairit (direct5), pairit-beam, pairit-rl, plus any
    family registered through `extra_baselines`.

    hERG (TDC) shares the same methods as MoleculeNet with two caveats:
      * hierg2g / nfbo hERG CSVs use the collaborator's schema
        (`seed_smiles, seed_class, generation_rank, generated_smiles, …`), which
        `_load_nfbo_64` already parses; both route through `source_kind=
        "nfbo_csv"`. No format conversion — the loader is the format contract.
      * MMP-Prop hERG requires a separate build (see internal/baselines/mmp/) and is
        skipped silently until `assets/generations/mmp/tdc/herg/generations.csv`
        exists.
    """
    specs: List[SourceSpec] = []

    # HierG2G — flat-text translations file per MoleculeNet dataset; hERG uses
    # the collab CSV format (parsed by `_load_nfbo_64`) instead.
    for dataset in MOLECULENET_DATASETS:
        path = ASSETS / "generations" / "hierg2g" / "moleculenet" / dataset / "translations.txt"
        if not path.exists():
            continue
        specs.append(
            SourceSpec(
                base_method="hierg2g-real-only",
                display_base="hierg2g | real-only",
                family="hierg2g",
                source_group="baseline",
                dataset=dataset,
                train_regime="real-only",
                novelty_reference="real",
                source_kind="hierg2g_txt",
                path=path,
                source_n_generations=64,
            )
        )
    # hERG hierg2g uses the collab CSV schema shared with nfbo.
    for dataset in TDC_DATASETS:
        path = ASSETS / "generations" / "hierg2g" / "tdc" / dataset / "generations.csv"
        if not path.exists():
            continue
        specs.append(
            SourceSpec(
                base_method="hierg2g-real-only",
                display_base="hierg2g | real-only",
                family="hierg2g",
                source_group="baseline",
                dataset=dataset,
                train_regime="real-only",
                novelty_reference="real",
                source_kind="nfbo_csv",
                path=path,
                source_n_generations=64,
            )
        )

    # Families registered through the extra_baselines hook.
    for extra in extra_baselines():
        specs.extend(extra.build_specs())

    # NN-Lookup — nearest-neighbor retrieval baseline. For each class-0 test
    # seed, top-64 real-train neighbors by ECFP4 Tanimoto (retrieval pool =
    # both classes, no model). Ships as canonical-schema CSVs written by
    # baselines/nearest_neighbor/build_nn_lookup_v5.py. MoleculeNet only (no
    # hERG); silently skipped per-dataset if the CSV is missing.
    for dataset in MOLECULENET_DATASETS:
        path = ASSETS / "generations" / "nn-lookup" / "moleculenet" / dataset / "generations.csv"
        if not path.exists():
            continue
        specs.append(
            SourceSpec(
                base_method="nn-lookup-real-only",
                display_base="nn-lookup | real-only",
                family="nn-lookup",
                source_group="baseline",
                dataset=dataset,
                train_regime="real-only",
                novelty_reference="real",
                source_kind="canonical_csv",
                path=path,
                source_n_generations=64,
            )
        )

    # NFBO / DDSBM-Bridge / DDSBM-SB — read from the local ASSETS bundle. The
    # loaders below consume the ASSETS file format (single CSV for NFBO, single
    # `all_generations.parquet` for DDSBM). Missing files silently skip so a
    # partial bundle still runs the baselines it has. Loader format is identical
    # across MoleculeNet and TDC — only the family directory (moleculenet/tdc)
    # differs.
    baseline_family_files = [
        ("nfbo",         "nfbo_csv",       "generations.csv"),
        ("ddsbm-bridge", "ddsbm_parquet",  "all_generations.parquet"),
        ("ddsbm-sb",     "ddsbm_parquet",  "all_generations.parquet"),
    ]
    for family, source_kind, filename in baseline_family_files:
        for dataset in DATASETS:
            path = ASSETS / "generations" / family / _dataset_family(dataset) / dataset / filename
            if not path.exists():
                continue
            specs.append(
                SourceSpec(
                    base_method=f"{family}-real-only",
                    display_base=f"{family} | real-only",
                    family=family,
                    source_group="baseline",
                    dataset=dataset,
                    train_regime="real-only",
                    novelty_reference="real",
                    source_kind=source_kind,
                    path=path,
                    source_n_generations=64,
                )
            )

    # MMP-Prop (property-scored MMP transformations) — one CSV per dataset in
    # canonical eval schema. The ASSETS file `mmp/moleculenet/{ds}/generations
    # .csv` is byte-identical (md5) to the historical `mmp_property_baseline`
    # file, so what looks like "plain MMP" in the ASSETS layout is actually
    # MMP-Property. Renamed to `mmp-prop` throughout so identifiers match the
    # actual data. Table display label is kept as "MMP" (paper convention,
    # since we don't compare against plain MMP anywhere). hERG MMP-Prop is
    # produced by internal/baselines/mmp/generate_mmp_property_baseline_herg.py and
    # silently skipped if missing.
    for dataset in DATASETS:
        path = ASSETS / "generations" / "mmp" / _dataset_family(dataset) / dataset / "generations.csv"
        if not path.exists():
            continue
        specs.append(
            SourceSpec(
                base_method="mmp-prop",
                display_base="mmp-prop",
                family="mmp-prop",
                source_group="baseline",
                dataset=dataset,
                train_regime="agnostic",
                novelty_reference="real",
                source_kind="mmp_csv",
                path=path,
                source_n_generations=64,
            )
        )

    # PairIT + PairIT-Beam + PairIT-RL — all read from assets/generations/
    # <method>/{moleculenet|tdc}/<ds>/{all_generations.parquet | generations.csv}.
    # ALL three go through the SAME best-of-64 + top-5 selection protocol as
    # every baseline (source_n_generations=64 → all64_specs → selection). No
    # method uses a special direct-N path — one protocol for all methods
    # ensures apples-to-apples comparison per user's mandate.
    #
    # File-format quirk: MoleculeNet pairit / pairit-beam ship the raw
    # `all_generations.parquet` (many rounds; `_load_pairit_allgens_64` picks
    # the first 64 per seed). hERG pairit ships the already-normalised
    # canonical-schema `generations.csv`. The underlying model is the same;
    # the export format is what differs. The per-dataset preference below
    # picks canonical_csv when both exist, else falls back to the raw parquet.
    # New RL inference runs can also export all_generations.parquet.
    pairit_family_preferences: List[Tuple[str, List[Tuple[str, str]], int]] = [
        # (method_name, [(source_kind, filename), … preferred order], source_n_gen)
        ("pairit",      [("canonical_csv", "generations.csv"),
                         ("pairit_allgens_parquet", "all_generations.parquet")], 64),
        ("pairit-beam", [("pairit_allgens_parquet", "all_generations.parquet"),
                         ("canonical_csv", "generations.csv")],                   64),
        ("pairit-rl",   [("canonical_csv", "generations.csv"),
                         ("pairit_allgens_parquet", "all_generations.parquet")], 64),
    ]
    for method_name, format_prefs, n_gen in pairit_family_preferences:
        for dataset in DATASETS:
            path: Optional[Path] = None
            source_kind: Optional[str] = None
            for kind, filename in format_prefs:
                candidate = ASSETS / "generations" / method_name / _dataset_family(dataset) / dataset / filename
                if candidate.exists():
                    path = candidate
                    source_kind = kind
                    break
            if path is None:
                continue
            specs.append(
                SourceSpec(
                    base_method=f"{method_name}-real-only",
                    display_base=f"{method_name} | real-only",
                    family=method_name,
                    source_group="pairit",
                    dataset=dataset,
                    train_regime="real-only",
                    novelty_reference="real",
                    source_kind=source_kind,
                    path=path,
                    source_n_generations=n_gen,
                )
            )

    # BACE pretraining-objective k-sweep ablation (appendix
    # `tab:pretrain_pairs_vs_single`, Fig 19, Fig 20). One spec per (variant,
    # replicate-seed) so each replicate lands as its own row in the pipeline
    # output — mean ± std across seeds is computed downstream by the LaTeX
    # writer. `source_group="ablation"` keeps these out of Tables 6-8
    # (which filter on baseline/pairit) and out of the plot code (which
    # keys on train_regime + specific family names). All variants use
    # `_load_canonical_csv`: the per_generation CSVs are already in the
    # canonical schema; the loader strips their pre-computed columns and
    # re-scores from scratch through the two-oracle protocol.
    # NOTE: the k-sweep raw outputs are not bundled with the release. Point
    # PAIRIT_KSWEEP_ROOT at your own copy to regenerate these rows; without
    # it the specs are silently skipped and the ablation rows come out empty.
    _KSWEEP_ROOT = Path(
        os.environ.get("PAIRIT_KSWEEP_ROOT", str(ASSETS / "generations" / "pretrain_ksweep")),
    )
    if not _KSWEEP_ROOT.exists():
        # Ablation rows (tab:pretrain_pairs_vs_single, Fig 19, Fig 20) require
        # this pool; without it those artifacts render empty. Warn loudly so a
        # reproduction attempt does not silently produce a blank table.
        print(
            f"[comparison_matrix] WARNING: PAIRIT_KSWEEP_ROOT ({_KSWEEP_ROOT}) "
            f"does not exist. Pretraining-ablation rows will be missing from "
            f"tab:pretrain_pairs_vs_single, Fig 19, and Fig 20. Set "
            f"PAIRIT_KSWEEP_ROOT to your k-sweep raw-output directory to "
            f"regenerate these artifacts.",
            file=sys.stderr,
            flush=True,
        )
    def _ksweep_gen_csv(run_dir: str) -> Optional[Path]:
        """Latest dated run under seedsonly_temp0p5_n64/test/*/generated_molecules/…"""
        cand = sorted((_KSWEEP_ROOT / run_dir / "evaluation_results" /
                       "seedsonly_temp0p5_n64" / "test").glob(
                       "*/generated_molecules/"
                       "conditional_step_9999_generated_molecules_pairit_per_generation.csv"))
        return cand[-1] if cand else None

    # (variant_id, run-dir template, list-of-seeds). variant_id becomes the
    # base_method prefix; seed is suffixed as `-s{seed}`.
    ksweep_variants: List[Tuple[str, str, List[int]]] = [
        ("pretrain-singlemol-uniform",
         "singlemolabl_ft_bace_ignorepairs_seed{seed}",   [42, 123, 234, 345, 456]),
        ("pretrain-singlemol-eps1seed",
         "singlemolabl_ft_bace_justfirst_seed{seed}",     [42, 123, 234, 345, 456]),
        ("pretrain-tani-k1",
         "singlemolabl_ft_bace_pair_tani_1nbr_seed{seed}", [42, 123, 234, 345, 456]),
        ("pretrain-tani-k2",
         "singlemolabl_ft_bace_pair_tani_2nbr_seed{seed}", [123, 234, 345, 456]),  # 42 missing
        ("pretrain-tani-k3",
         "singlemolabl_ft_bace_pair_tani_3nbr_seed{seed}", [42, 123, 234, 345, 456]),
        ("pretrain-tani-k5",
         "singlemolabl_ft_bace_pair_tani_5nbr_seed{seed}", [42, 123, 234, 345, 456]),
        ("pretrain-tani-k10",
         "singlemolabl_ft_bace_pair_tani_10nbr_seed{seed}",[42, 123, 234, 345, 456]),
        ("pretrain-tani-k20",
         "singlemolabl_ft_bace_pair_tani_20nbr_seed{seed}",[42, 123, 234, 345, 456]),
        ("pretrain-tani-kall",
         "singlemolabl_ft_bace_pair_tani_allnbr_seed{seed}",[42, 123, 234, 345, 456]),
        ("pretrain-none",
         "try1_scratch_bace_CLM_68M_seed{seed}",           [42, 123]),  # only 2 seeds trained
    ]
    for variant_id, run_tmpl, seeds in ksweep_variants:
        for seed in seeds:
            path = _ksweep_gen_csv(run_tmpl.format(seed=seed))
            if path is None:
                continue
            base_method = f"{variant_id}-s{seed}-real-only"
            specs.append(
                SourceSpec(
                    base_method=base_method,
                    display_base=f"{variant_id} | s{seed}",
                    family=variant_id,
                    source_group="ablation",
                    dataset="bace",
                    train_regime="real-only",
                    novelty_reference="real",
                    source_kind="canonical_csv",
                    path=path,
                    source_n_generations=64,
                )
            )

    return specs


def _load_hierg2g_64(spec: SourceSpec) -> pd.DataFrame:
    raw = pd.read_csv(spec.path, sep=r"\s+", names=["seed_smiles", "generated_smiles"], engine="python")
    seed_order = _seed_order_map(raw["seed_smiles"])
    out = pd.DataFrame(
        {
            "method": _all64_method_name(spec),
            "dataset": spec.dataset,
            "seed_idx": raw["seed_smiles"].map(seed_order).astype(int),
            "seed_smiles": raw["seed_smiles"].astype(str),
            "seed_label": 0,
            "gen_idx": raw.groupby("seed_smiles").cumcount().astype(int),
            "generated_smiles": raw["generated_smiles"].astype(str),
        }
    )
    return out[CANONICAL_COLUMNS]


def _load_nfbo_64(spec: SourceSpec) -> pd.DataFrame:
    raw = pd.read_csv(spec.path)
    seed_order = _seed_order_map(raw["seed_smiles"])
    gen_idx = pd.to_numeric(raw["generation_rank"], errors="coerce").fillna(1).astype(int) - 1
    out = pd.DataFrame(
        {
            "method": _all64_method_name(spec),
            "dataset": spec.dataset,
            "seed_idx": raw["seed_smiles"].map(seed_order).astype(int),
            "seed_smiles": raw["seed_smiles"].astype(str),
            "seed_label": pd.to_numeric(raw["seed_class"], errors="coerce").fillna(0).astype(int),
            "gen_idx": gen_idx.astype(int),
            "generated_smiles": _string_or_blank(raw["generated_smiles"]),
        }
    )
    return out[CANONICAL_COLUMNS]


def _load_ddsbm_parquet(spec: SourceSpec) -> pd.DataFrame:
    """DDSBM ASSETS format: single `all_generations.parquet` per (method, dataset)
    with columns: seed_idx, gen_idx, smiles_seed, smiles_generated, is_valid,
    smiles_generated_relaxed, is_valid_relaxed. Blank/invalid rows keep their
    seed_smiles but generated_smiles is emptied so downstream validity metrics
    see the failure. Prefer `smiles_generated` first; fall back to the relaxed
    variant only when the strict one is invalid AND `is_valid_relaxed` is True
    (matches the historical dir-of-CSVs loader's precedence, which required the
    row-level ``success`` flag before accepting either SMILES)."""
    raw = pd.read_parquet(spec.path)
    strict = raw["smiles_generated"].astype(str)
    strict_valid = raw["is_valid"].fillna(False).astype(bool)
    if "smiles_generated_relaxed" in raw.columns:
        relaxed = raw["smiles_generated_relaxed"].astype(str)
        # Only accept the relaxed fallback when its own validity flag is set;
        # otherwise the strict-invalid row must fail loudly rather than smuggle
        # in an invalid relaxed SMILES that inflates validity downstream.
        if "is_valid_relaxed" in raw.columns:
            relaxed_valid = raw["is_valid_relaxed"].fillna(False).astype(bool)
            relaxed_gated = relaxed.where(relaxed_valid, "")
        else:
            relaxed_gated = relaxed
        gen_smiles = strict.where(strict_valid, relaxed_gated)
    else:
        gen_smiles = strict.where(strict_valid, "")
    gen_smiles = gen_smiles.where(gen_smiles.astype(str) != "nan", "")
    out = pd.DataFrame(
        {
            "method": _all64_method_name(spec),
            "dataset": spec.dataset,
            "seed_idx": pd.to_numeric(raw["seed_idx"], errors="coerce").fillna(0).astype(int),
            "seed_smiles": raw["smiles_seed"].astype(str),
            "seed_label": 0,
            "gen_idx": pd.to_numeric(raw["gen_idx"], errors="coerce").fillna(0).astype(int),
            "generated_smiles": _string_or_blank(gen_smiles),
        }
    )
    return out[CANONICAL_COLUMNS]


def _load_ddsbm_64(spec: SourceSpec, moleculenet_test_dir: Path) -> pd.DataFrame:
    class0 = _load_test_class0(spec.dataset, moleculenet_test_dir=moleculenet_test_dir)
    frames: List[pd.DataFrame] = []
    for gen_idx in range(64):
        path = spec.path / f"result_forward_seed{gen_idx}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        gen_smiles: List[str] = []
        seed_smiles: List[str] = []
        seed_idx: List[int] = []
        for _, row in raw.iterrows():
            idx = int(row["idx"])
            seed_row = class0.iloc[idx]
            seed_idx.append(idx)
            seed_smiles.append(str(seed_row["smiles"]))
            smiles = ""
            if bool(row.get("success", False)):
                smiles = str(row.get("smiles") or row.get("relaxed_smiles") or "")
                if smiles == "nan":
                    smiles = ""
            gen_smiles.append(smiles)
        frames.append(
            pd.DataFrame(
                {
                    "method": _all64_method_name(spec),
                    "dataset": spec.dataset,
                    "seed_idx": seed_idx,
                    "seed_smiles": seed_smiles,
                    "seed_label": 0,
                    "gen_idx": gen_idx,
                    "generated_smiles": gen_smiles,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS] if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)


def _load_canonical_csv(spec: SourceSpec) -> pd.DataFrame:
    raw = pd.read_csv(spec.path)
    out = raw[CANONICAL_COLUMNS].copy()
    # `-all64` suffix so downstream selection can rename to `-first5` / `-best64`
    # for methods that go through the best-of-N pipeline (n_gen=64). For direct
    # methods (n_gen != 64) keep the base name unchanged.
    out["method"] = _all64_method_name(spec) if spec.source_n_generations == 64 else spec.base_method
    out["dataset"] = spec.dataset
    out["seed_idx"] = pd.to_numeric(out["seed_idx"], errors="coerce").fillna(0).astype(int)
    out["seed_label"] = pd.to_numeric(out["seed_label"], errors="coerce").fillna(0).astype(int)
    out["gen_idx"] = pd.to_numeric(out["gen_idx"], errors="coerce").fillna(0).astype(int)
    out["seed_smiles"] = out["seed_smiles"].astype(str)
    out["generated_smiles"] = _string_or_blank(out["generated_smiles"])
    return out[CANONICAL_COLUMNS]


def _load_mmp_64(spec: SourceSpec, moleculenet_test_dir: Path) -> pd.DataFrame:
    """MMP baseline ASSETS format: single generations.csv per dataset with
    columns matching the canonical eval schema (seed_smiles, seed_idx, gen_idx,
    generated_smiles, ...). Older versions ALSO carry query_label/split, in
    which case we filter to split=='test' and seeds in the "undesirable"
    class (0 for increasing-direction datasets like BACE/BBBP/HIV, 1 for
    decreasing-direction datasets like hERG). If those columns aren't
    present the file has already been narrowed and no filter is applied.
    `moleculenet_test_dir` is unused by this loader now (kept in the
    signature for dispatcher compatibility)."""
    del moleculenet_test_dir  # legacy plumbing — not used by the ASSETS format
    raw = pd.read_csv(spec.path)
    if "split" in raw.columns and "query_label" in raw.columns:
        # Pick the seed class per dataset direction. Increasing (0->1): seeds
        # are class 0. Decreasing (1->0): seeds are class 1. Hardcoded == 0
        # dropped every hERG row.
        ds_cfg = DATASET_CONFIG.get(spec.dataset, {})
        direction = str(ds_cfg.get("direction", "increasing")).lower()
        seed_class = 1 if direction in ("decreasing", "lower", "min") else 0
        raw = raw[(raw["split"] == "test") &
                  (pd.to_numeric(raw["query_label"], errors="coerce").fillna(0) == seed_class)].copy()

    raw = raw.sort_values(["seed_idx", "gen_idx"]).groupby("seed_idx", as_index=False, sort=False).head(64).copy()
    raw["gen_idx"] = raw.groupby("seed_idx").cumcount().astype(int)

    out = pd.DataFrame(
        {
            "method": _all64_method_name(spec),
            "dataset": spec.dataset,
            "seed_idx": pd.to_numeric(raw["seed_idx"], errors="coerce").fillna(0).astype(int),
            "seed_smiles": raw["seed_smiles"].astype(str),
            "seed_label": 0,
            "gen_idx": raw["gen_idx"].astype(int),
            "generated_smiles": _string_or_blank(raw["generated_smiles"]),
        }
    )
    return out[CANONICAL_COLUMNS]


def _load_pairit_allgens_64(spec: SourceSpec) -> pd.DataFrame:
    raw = pd.read_parquet(spec.path)
    raw = raw.sort_values(["seed_idx", "round", "generation_idx"]).groupby("seed_idx", as_index=False, sort=False).head(64).copy()
    raw["gen_idx"] = raw.groupby("seed_idx").cumcount().astype(int)
    out = pd.DataFrame(
        {
            "method": _all64_method_name(spec),
            "dataset": spec.dataset,
            "seed_idx": pd.to_numeric(raw["seed_idx"], errors="coerce").fillna(0).astype(int),
            "seed_smiles": raw["smiles_seed"].astype(str),
            "seed_label": 0,
            "gen_idx": raw["gen_idx"].astype(int),
            "generated_smiles": _string_or_blank(raw["smiles_generated"]),
        }
    )
    return out[CANONICAL_COLUMNS]


def load_source_frame(spec: SourceSpec, config: PairitEvalConfig) -> pd.DataFrame:
    if spec.source_kind == "hierg2g_txt":
        return _load_hierg2g_64(spec)
    if spec.source_kind == "nfbo_csv":
        return _load_nfbo_64(spec)
    if spec.source_kind == "ddsbm_dir":
        return _load_ddsbm_64(spec, moleculenet_test_dir=Path(config.moleculenet_test_dir))
    if spec.source_kind == "ddsbm_parquet":
        return _load_ddsbm_parquet(spec)
    if spec.source_kind == "mmp_csv":
        return _load_mmp_64(spec, moleculenet_test_dir=Path(config.moleculenet_test_dir))
    if spec.source_kind == "canonical_csv":
        return _load_canonical_csv(spec)
    if spec.source_kind == "pairit_allgens_parquet":
        return _load_pairit_allgens_64(spec)
    for extra in extra_baselines():
        if spec.source_kind in extra.loaders:
            return extra.loaders[spec.source_kind](spec)
    raise ValueError(f"Unsupported source_kind: {spec.source_kind}")


def _save_method_inputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, sub in df.groupby("method", sort=True):
        sub.to_csv(out_dir / f"{method}.csv", index=False)


def _component_count(df: pd.DataFrame) -> pd.Series:
    valid = df["is_valid"].fillna(False).astype(int)
    novel = df["is_novel"].fillna(False).astype(int)
    improved = df["improvement"].fillna(False).astype(int)
    return valid + novel + improved


def _select_first5(per_generation_df: pd.DataFrame) -> pd.DataFrame:
    df = per_generation_df.sort_values(["method", "dataset", "seed_idx", "gen_idx"]).copy()
    selected = df.groupby(["method", "dataset", "seed_idx"], as_index=False, sort=False).head(5).copy()
    selected["gen_idx"] = selected.groupby(["method", "dataset", "seed_idx"]).cumcount().astype(int)
    selected["method"] = selected["method"].astype(str).str.replace("-all64", "-first5", regex=False)
    return selected[CANONICAL_COLUMNS]


def _select_best5_joint(per_generation_df: pd.DataFrame) -> pd.DataFrame:
    """Dedup-first best-5 selection: sort by joint quality, dedup canonical
    SMILES within each seed, then take the top 5. Ensures the retained
    generations are distinct molecules and exposes intra-seed collapse
    (a seed can end up with <5 gens if the model can't produce 5 distinct
    valid canonicals). This matches the paper's canonical selection.
    """
    df = per_generation_df.copy()
    improved = df["improvement"].fillna(False)
    df["joint_full"] = (
        df["is_valid"].fillna(False)
        & df["is_novel"].fillna(False)
        & improved
    )
    df["joint_components"] = _component_count(df)
    df["sort_predicted_prob"] = pd.to_numeric(df["predicted_prob"], errors="coerce").fillna(-np.inf)
    df["sort_tanimoto"] = pd.to_numeric(df["tanimoto"], errors="coerce").fillna(-np.inf)

    # Selection policy: dedup canonical SMILES within seed + drop invalids
    # entirely before picking top-5. If the model produced fewer than 5 unique
    # valid canonicals in 64 gens, the seed's top-5 is "short" (has <5 rows).
    # We DO NOT fill remaining slots with placeholder invalid rows — that
    # leaks pipeline-artifact rows into Validity/Novel/Improved denominators
    # and misattributes diversity-collapse to Validity.
    #
    # Diversity-collapse is surfaced honestly through:
    # (a) per-seed Unique@5 in compute_summary_metrics (fixed denominator of 5),
    # (b) short seeds appearing in n_total, which naturally depresses V,I,N.
    # Validity/Novel/Improved are sourced from the raw-64 pass instead so
    # they measure per-generation model behavior, not top-5-selection artifacts.
    df["_canon"] = [_canonical_smiles(s) for s in df["generated_smiles"]]
    df_valid = df[df["_canon"].notna()].copy()

    sorted_df = df_valid.sort_values(
        [
            "method",
            "dataset",
            "seed_idx",
            "joint_full",
            "joint_components",
            "sort_predicted_prob",
            "sort_tanimoto",
            "gen_idx",
        ],
        ascending=[True, True, True, False, False, False, False, True],
        kind="mergesort",
    )

    # Within-seed dedup: keep highest-quality occurrence of each canonical.
    deduped = sorted_df.drop_duplicates(
        subset=["method", "dataset", "seed_idx", "_canon"],
        keep="first",
    )

    selected = (
        deduped
        .groupby(["method", "dataset", "seed_idx"], as_index=False, sort=False)
        .head(5)
        .copy()
    )
    selected["gen_idx"] = selected.groupby(["method", "dataset", "seed_idx"]).cumcount().astype(int)
    selected["method"] = selected["method"].astype(str).str.replace("-all64", "-best64", regex=False)
    return selected[CANONICAL_COLUMNS]


def _selection_summary(per_generation_df: pd.DataFrame) -> pd.DataFrame:
    df = per_generation_df.copy()
    improved = df["improvement"].fillna(False)
    df["joint_full"] = (
        df["is_valid"].fillna(False)
        & df["is_novel"].fillna(False)
        & improved
    )
    per_seed = (
        df.groupby(["method", "dataset", "seed_idx"], as_index=False)
        .agg(
            n_joint_full=("joint_full", "sum"),
            n_valid=("is_valid", "sum"),
            n_rows=("generated_smiles", "size"),
        )
    )
    summary = (
        per_seed.groupby(["method", "dataset"], as_index=False)
        .agg(
            n_seeds=("seed_idx", "nunique"),
            mean_joint_full=("n_joint_full", "mean"),
            median_joint_full=("n_joint_full", "median"),
            seeds_with_lt5_joint=("n_joint_full", lambda s: int((s < 5).sum())),
        )
    )
    return summary


def _build_method_meta(specs: Sequence[SourceSpec]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for spec in specs:
        if spec.source_n_generations == 64:
            method_rows = [
                ("first5", _final_method_name(spec, "first5"), _display_name(spec, "first5")),
                ("best64", _final_method_name(spec, "best64"), _display_name(spec, "best64")),
            ]
        else:
            method_rows = [("direct5", spec.base_method, _display_name(spec, "direct5"))]

        for selection_mode, method, display_name in method_rows:
            if method in seen:
                continue
            seen.add(method)
            rows.append(
                {
                    "method": method,
                    "display_name": display_name,
                    "family": spec.family,
                    "source_group": spec.source_group,
                    "train_regime": spec.train_regime,
                    "selection_mode": selection_mode,
                }
            )
    meta = pd.DataFrame(rows)
    order_map = {
        "nn-lookup": -1,  # printed first (naive retrieval reference)
        "nfbo": 0,
        "ddsbm-bridge": 1,
        "ddsbm-sb": 2,
        "hierg2g": 4,
        "mmp-prop": 5,
        "mmp-prop": 6,
        "pairit": 7,
        "pairit-beam": 8,
        "pairit-rl": 9,
    }
    order_map.update({extra.family: extra.family_order for extra in extra_baselines()})
    train_map = {
        "real-only": 0,
        "agnostic": 2,
    }
    selection_map = {"direct5": 0, "first5": 1, "best64": 2}
    meta["family_order"] = meta["family"].map(order_map).fillna(99).astype(int)
    meta["train_order"] = meta["train_regime"].map(train_map).fillna(99).astype(int)
    meta["selection_order"] = meta["selection_mode"].map(selection_map).fillna(99).astype(int)
    return meta.sort_values(["family_order", "train_order", "selection_order", "method"]).reset_index(drop=True)


def _attach_meta(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    return df.merge(meta, on="method", how="left")


def materialize_inputs(
    specs: Sequence[SourceSpec],
    *,
    output_root: Path,
    config: PairitEvalConfig,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    materialized_dir = output_root / "materialized"
    all64_input_dir = materialized_dir / "all64_inputs"
    direct_input_dir = materialized_dir / "direct_inputs"
    final_input_dir = materialized_dir / "final_inputs"
    all64_eval_dir = materialized_dir / "all64_eval"
    all64_eval_dir.mkdir(parents=True, exist_ok=True)

    all64_frames: List[pd.DataFrame] = []
    direct_frames: List[pd.DataFrame] = []
    all64_specs = [s for s in specs if s.source_n_generations == 64]
    direct_specs = [s for s in specs if s.source_n_generations != 64]

    if verbose:
        print(
            f"[materialize] loading {len(all64_specs)} all64 sources and "
            f"{len(direct_specs)} direct sources"
        )

    for spec in specs:
        df = load_source_frame(spec, config)
        if spec.source_n_generations == 64:
            all64_frames.append(df)
        else:
            direct_frames.append(df)

    if all64_frames:
        raw_all64 = pd.concat(all64_frames, ignore_index=True)
        _save_method_inputs(raw_all64, all64_input_dir)
        normalized_all64 = normalize_generation_table(
            raw_all64,
            config=config,
            n_generations_by_dataset={dataset: 64 for dataset in DATASETS},
            seed_label_by_dataset=_seed_label_by_dataset(DATASETS),
        )
        normalized_all64.to_csv(all64_eval_dir / "all64_normalized.csv", index=False)
        if verbose:
            print(
                f"[materialize] evaluating all64 candidates: raw_rows={len(raw_all64)} "
                f"normalized_rows={len(normalized_all64)}"
            )
        # Two-oracle protocol: the all-64 pass predicts with `select_oracle`
        # (train-only RF) so downstream selection ranks by RF confidence, NOT
        # by the same MiniMol scores used in final reporting. `use_select_oracle
        # _for_prediction=True` is the plumbing that routes RF into
        # `_evaluate_one_block`. Without it, the RL model would be scored on
        # both selection AND reporting with MiniMol — the exact reward-hacking
        # setup we're preventing.
        all64_result = evaluate(
            normalized_all64,
            config=config,
            verbose=verbose,
            use_select_oracle_for_prediction=True,
        )
        if verbose:
            print(
                f"[materialize] all64 evaluation complete: per_generation_rows={len(all64_result.per_generation)}"
            )
        all64_result.per_generation.to_csv(all64_eval_dir / "all64_per_generation.csv", index=False)
        all64_result.per_seed.to_csv(all64_eval_dir / "all64_per_seed.csv", index=False)
        all64_result.average.to_csv(all64_eval_dir / "all64_average.csv", index=False)
        _selection_summary(all64_result.per_generation).to_csv(
            materialized_dir / "selection_summary_all64.csv", index=False
        )
        first5_df = _select_first5(all64_result.per_generation)
        best5_df = _select_best5_joint(all64_result.per_generation)
    else:
        first5_df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        best5_df = pd.DataFrame(columns=CANONICAL_COLUMNS)

    if direct_frames:
        direct_df = pd.concat(direct_frames, ignore_index=True)
        _save_method_inputs(direct_df, direct_input_dir)
    else:
        direct_df = pd.DataFrame(columns=CANONICAL_COLUMNS)

    final_frames = [df for df in (direct_df, first5_df, best5_df) if len(df) > 0]
    final_df = pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
    _save_method_inputs(final_df, final_input_dir)
    method_meta = _build_method_meta(specs)
    method_meta.to_csv(materialized_dir / "method_metadata.csv", index=False)
    if verbose:
        print(
            f"[materialize] final comparison inputs ready: rows={len(final_df)} methods={final_df['method'].nunique() if len(final_df) else 0}"
        )
    return final_df, method_meta, _selection_summary(
        pd.read_csv(all64_eval_dir / "all64_per_generation.csv")
    ) if (all64_eval_dir / "all64_per_generation.csv").exists() else pd.DataFrame()


def evaluate_final_inputs(
    final_df: pd.DataFrame,
    specs: Sequence[SourceSpec],
    *,
    output_root: Path,
    config: PairitEvalConfig,
    method_meta: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    regular_dir = output_root / "regular"
    distribution_dir = output_root / "distribution"
    regular_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir.mkdir(parents=True, exist_ok=True)

    # Top-5 pass: pad to k=5 per seed (default `pad_missing=True`). Implements
    # a budget denominator `n_budget = k × n_seeds` — a seed that produced
    # fewer than 5 unique valid canonicals gets its top-5 padded with NaN
    # placeholders (is_valid=False), so VIN/Valid/Novel for that seed
    # correctly divide by 5 rather than by the (smaller) actual row count.
    # This penalizes short-seed and coverage-failure methods honestly.
    normalized = normalize_generation_table(
        final_df,
        config=config,
        n_generations_by_dataset={dataset: 5 for dataset in DATASETS},
        seed_label_by_dataset=_seed_label_by_dataset(DATASETS),
    )
    normalized.to_csv(output_root / "final_normalized.csv", index=False)
    if verbose:
        print(
            f"[evaluate] final normalized table ready: rows={len(normalized)} methods={normalized['method'].nunique() if len(normalized) else 0}"
        )

    regular = evaluate(
        normalized,
        config=config,
        verbose=verbose,
    )
    if verbose:
        print("[evaluate] regular metrics pass complete")
    distribution = evaluate(
        normalized,
        config=config,
        verbose=verbose,
        compute_distribution_metrics=True,
    )
    if verbose:
        print("[evaluate] distribution metrics pass complete")

    # All metrics (Valid, Novel, Improved, VIN, SNN, SimSeed, Unique@5) are
    # computed on the TOP-5 selection. Computing them all off the same pool
    # is what guarantees the invariant VIN <= Improved holds (splicing
    # Valid/Novel/Improved from the raw-64 pass can otherwise produce
    # VIN > Improved anomalies). The k*n_seeds budget denominator makes
    # short-seeds / coverage failure honestly penalize both. The raw-64 pass
    # output is still produced (it feeds selection) but its aggregate metrics
    # aren't spliced into the final table.
    regular_avg = _attach_meta(regular.average, method_meta)

    outputs = {
        "regular_per_generation": _attach_meta(regular.per_generation, method_meta),
        "regular_per_seed": _attach_meta(regular.per_seed, method_meta),
        "regular_average": regular_avg,
        "distribution_per_generation": _attach_meta(distribution.per_generation, method_meta),
        "distribution_per_seed": _attach_meta(distribution.per_seed, method_meta),
        "distribution_average": _attach_meta(distribution.average, method_meta),
    }

    for key, df in outputs.items():
        target_dir = regular_dir if key.startswith("regular_") else distribution_dir
        df.to_csv(target_dir / f"{key}.csv", index=False)
    return outputs


def build_summary_tables(outputs: Dict[str, pd.DataFrame], output_root: Path) -> Dict[str, pd.DataFrame]:
    summaries_dir = output_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    regular = outputs["regular_average"].copy()
    distribution = outputs["distribution_average"].copy()
    if len(regular):
        regular["joint_metric"] = pd.to_numeric(regular["v_n_i"], errors="coerce")
    if len(distribution):
        distribution["joint_metric"] = pd.to_numeric(distribution["v_n_i"], errors="coerce")

    regular_cols = [
        "dataset",
        "source_group",
        "family",
        "train_regime",
        "selection_mode",
        "method",
        "display_name",
        "joint_metric",
        *REGULAR_METRICS,
    ]
    distribution_cols = [
        "dataset",
        "source_group",
        "family",
        "train_regime",
        "selection_mode",
        "method",
        "display_name",
        "joint_metric",
        *DISTRIBUTION_METRICS,
    ]

    regular_summary = regular[[c for c in regular_cols if c in regular.columns]].copy()
    distribution_summary = distribution[[c for c in distribution_cols if c in distribution.columns]].copy()

    sort_cols = ["dataset", "source_group", "family", "train_regime", "selection_mode", "method"]
    regular_summary = regular_summary.sort_values([c for c in sort_cols if c in regular_summary.columns]).reset_index(drop=True)
    distribution_summary = distribution_summary.sort_values(
        [c for c in sort_cols if c in distribution_summary.columns]
    ).reset_index(drop=True)

    regular_summary.to_csv(summaries_dir / "regular_summary_all.csv", index=False)
    distribution_summary.to_csv(summaries_dir / "distribution_summary_all.csv", index=False)
    for dataset in DATASETS:
        regular_summary[regular_summary["dataset"] == dataset].to_csv(
            summaries_dir / f"{dataset}_regular_summary.csv", index=False
        )
        distribution_summary[distribution_summary["dataset"] == dataset].to_csv(
            summaries_dir / f"{dataset}_distribution_summary.csv", index=False
        )

    return {"regular": regular_summary, "distribution": distribution_summary}


def _method_palette() -> Dict[str, str]:
    # Baselines keep distinct families; all PairIt variants share one warm palette.
    palette = {
        "nfbo": "#6F89AE",
        "ddsbm-bridge": "#84B06B",
        "ddsbm-sb": "#9EC6C4",
        "hierg2g": "#7A708B",
        "mmp-prop": "#C9B2CC",
        "pairit-first5": "#F6C88B",
        "pairit-best64": "#F0A09E",
        "pairit-beam-first5": "#D96D6A",
        "pairit-beam-best64": "#B44448",
        "pairit-rl-first5": "#F0855C",
        "pairit-rl-best64": "#8B2E30",
    }
    palette.update({extra.family: extra.color for extra in extra_baselines()})
    return palette


def _plot_method_order(train_regime: str) -> List[str]:
    if train_regime == "real-only":
        order = [
            "nfbo-real-only-first5",
            "ddsbm-bridge-real-only-first5",
            "ddsbm-sb-real-only-first5",
            "hierg2g-real-only-first5",
            "mmp-prop-first5",
            "pairit-real-only-first5",
            "pairit-beam-real-only-first5",
            "pairit-rl-real-only-first5",
            "nfbo-real-only-best64",
            "ddsbm-bridge-real-only-best64",
            "ddsbm-sb-real-only-best64",
            "hierg2g-real-only-best64",
            "mmp-prop-best64",
            "pairit-real-only-best64",
            "pairit-beam-real-only-best64",
            "pairit-rl-real-only-best64",
        ]
        for extra in extra_baselines():
            for sel in ("first5", "best64"):
                order = insert_after(order, f"{extra.after_family}-real-only-{sel}", f"{extra.family}-real-only-{sel}")
        return order


def _plot_method_labels() -> Dict[str, str]:
    labels = {
        "nfbo-real-only-first5": "NFBO 5",
        "ddsbm-bridge-real-only-first5": "DDSBM-Bridge 5",
        "ddsbm-sb-real-only-first5": "DDSBM-SB 5",
        "hierg2g-real-only-first5": "HierG2G 5",
        "mmp-prop-first5": "MMP 5",
        "pairit-real-only-first5": "PairIT 5",
        "pairit-beam-real-only-first5": "PairIT-Beam 5",
        "pairit-rl-real-only-first5": "PairIT-RL 5",
        "nfbo-real-only-best64": "NFBO 64",
        "ddsbm-bridge-real-only-best64": "DDSBM-Bridge 64",
        "ddsbm-sb-real-only-best64": "DDSBM-SB 64",
        "hierg2g-real-only-best64": "HierG2G 64",
        "mmp-prop-best64": "MMP 64",
        "pairit-real-only-best64": "PairIT 64",
        "pairit-beam-real-only-best64": "PairIT-Beam 64",
        "pairit-rl-real-only-best64": "PairIT-RL 64",
    }
    for extra in extra_baselines():
        labels.update(extra.plot_labels)
    return labels


def _method_color(method: str) -> str:
    palette = _method_palette()
    if method.startswith("nfbo-"):
        return palette["nfbo"]
    if method.startswith("ddsbm-bridge-"):
        return palette["ddsbm-bridge"]
    if method.startswith("ddsbm-sb-"):
        return palette["ddsbm-sb"]
    if method.startswith("hierg2g-"):
        return palette["hierg2g"]
    for extra in extra_baselines():
        if method.startswith(f"{extra.family}-"):
            return palette[extra.family]
    if method.startswith("mmp-prop-"):
        return palette["mmp-prop"]
    if method.startswith("pairit-beam-") and method.endswith("-best64"):
        return palette["pairit-beam-best64"]
    if method.startswith("pairit-beam-"):
        return palette["pairit-beam-first5"]
    if method.startswith("pairit-rl-") and method.endswith("-best64"):
        return palette["pairit-rl-best64"]
    if method.startswith("pairit-rl-"):
        return palette["pairit-rl-first5"]
    if method.startswith("pairit-") and method.endswith("-best64"):
        return palette["pairit-best64"]
    if method.startswith("pairit-"):
        return palette["pairit-first5"]
    return "#888888"


def _metric_label(metric: str, value: float) -> str:
    percentage_metrics = {"joint_metric", "v_n_i", "valid", "novel", "improved"}
    if metric in percentage_metrics:
        return f"{int(round(value))}"
    return f"{value:.2f}"


def _metric_ylim(metric: str, values: pd.Series, max_err: float = 0.0) -> Tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) == 0:
        return (0.0, 1.0)
    vmin = float(min(0.0, clean.min()))
    vmax = float(clean.max()) + max_err
    span = max(vmax - vmin, 1.0 if vmax > 1.5 else 0.1)
    return (vmin - 0.05 * span, vmax + 0.22 * span)


def _compute_std_lookup(
    per_seed_df: pd.DataFrame,
    dataset: str,
) -> Dict[Tuple[str, str], float]:
    """Return {(method, metric): std_across_seeds} for one dataset.

    Std is computed with ddof=1 (sample std) across seed_idx rows for each
    (method, dataset) group.  Only metrics present in the per-seed CSV are
    included; distribution metrics are not (they have no per-seed decomposition).
    """
    per_seed_metrics = [
        "v_n_i", "valid", "novel", "improved",
        "sim_seed", "internal_diversity",
    ]
    sub = per_seed_df[per_seed_df["dataset"] == dataset].copy()
    result: Dict[Tuple[str, str], float] = {}
    for method, grp in sub.groupby("method", sort=False):
        for metric in per_seed_metrics:
            if metric not in grp.columns:
                continue
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
            if len(vals) > 1:
                result[(str(method), metric)] = float(vals.std(ddof=1))
            elif len(vals) == 1:
                result[(str(method), metric)] = 0.0
        # joint_metric mirrors v_n_i for all methods in these plots
        if (str(method), "v_n_i") in result:
            result[(str(method), "joint_metric")] = result[(str(method), "v_n_i")]
    return result


def _plot_metric_bars(
    summary: pd.DataFrame,
    dataset: str,
    train_regime: str,
    metrics: Sequence[str],
    directions: Dict[str, str],
    out_path: Path,
    title: str,
    std_lookup: Optional[Dict[Tuple[str, str], float]] = None,
) -> None:
    sub = summary[summary["dataset"] == dataset].copy()
    sub = sub[(sub["train_regime"] == train_regime) | (sub["family"] == "mmp-prop")].copy()
    if len(sub) == 0:
        return

    method_order = [m for m in _plot_method_order(train_regime) if m in set(sub["method"])]
    if not method_order:
        return
    sub = sub[sub["method"].isin(method_order)].copy()
    label_map = _plot_method_labels()
    sub["plot_label"] = sub["method"].map(label_map).fillna(sub["display_name"])
    order = method_order
    colors = {method: _method_color(method) for method in order}

    n_metrics = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))
    fig_width = max(16, 0.78 * len(order) + 6)
    fig_height = 4.8 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = sub[["method", "plot_label", metric]].dropna().copy() if metric in sub.columns else pd.DataFrame()
        if len(metric_df) == 0:
            ax.set_axis_off()
            continue
        metric_df["method"] = pd.Categorical(metric_df["method"], categories=order, ordered=True)
        metric_df = metric_df.sort_values("method")
        x = np.arange(len(metric_df))
        y = metric_df[metric].astype(float).to_numpy()
        methods_in_order = metric_df["method"].astype(str).tolist()

        # Build per-bar std error array (zeros where no std is available)
        yerr: Optional[np.ndarray] = None
        if std_lookup is not None:
            raw = np.array(
                [std_lookup.get((m, metric), np.nan) for m in methods_in_order],
                dtype=float,
            )
            if not np.all(np.isnan(raw)):
                yerr = np.where(np.isnan(raw), 0.0, raw)

        max_err = float(np.nanmax(yerr)) if yerr is not None else 0.0
        ymin, ymax = _metric_ylim(metric, metric_df[metric], max_err=max_err)

        bars = ax.bar(
            x,
            y,
            color=[colors[method] for method in methods_in_order],
            edgecolor="#6A6A6A",
            linewidth=0.8,
            alpha=0.95,
            yerr=yerr,
            error_kw={
                "capsize": 3,
                "ecolor": "#333333",
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
        )
        ax.set_title(f"{metric} ({directions.get(metric, 'higher')} better)")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_df["plot_label"].astype(str), rotation=50, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(ymin, ymax)

        err_arr = yerr if yerr is not None else np.zeros_like(y)
        for bar, value, err in zip(bars, y, err_arr):
            span = ymax - ymin
            cap_top = float(value) + float(err) if value >= 0 else float(value)
            y_text = cap_top + 0.015 * span if value >= 0 else cap_top - 0.02 * span
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                _metric_label(metric, float(value)),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                color="#333333",
            )

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_axis_off()

    fig.suptitle(f"{dataset} | {train_regime} | {title}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _draw_confidence_ellipse(
    ax: "plt.Axes",
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    color: str,
    n_std: float = 1.0,
    alpha: float = 0.15,
) -> None:
    """Draw a 1-σ confidence ellipse derived from the 2-D sample covariance."""
    from matplotlib.patches import Ellipse

    if len(x_vals) < 3:
        return
    cov = np.cov(x_vals, y_vals)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2.0 * n_std * np.sqrt(max(float(eigenvalues[0]), 0.0))
    height = 2.0 * n_std * np.sqrt(max(float(eigenvalues[1]), 0.0))
    ellipse = Ellipse(
        xy=(float(np.mean(x_vals)), float(np.mean(y_vals))),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1.0,
        linestyle="--",
    )
    ax.add_patch(ellipse)


# Short labels used in scatter plots (applied by family prefix for conciseness)
_SCATTER_LABELS: Dict[str, str] = {
    "nfbo-real-only-best64": "NFBO",
    "nfbo-real-only-first5": "NFBO-5",
    "ddsbm-bridge-real-only-best64": "DDSBM-B",
    "ddsbm-bridge-real-only-first5": "DDSBM-B-5",
    "ddsbm-sb-real-only-best64": "DDSBM-SB",
    "ddsbm-sb-real-only-first5": "DDSBM-SB-5",
    "hierg2g-real-only-best64": "HierG2G",
    "hierg2g-real-only-first5": "HierG2G-5",
    "mmp-prop-best64": "MMP",
    "mmp-prop-first5": "MMP-5",
    "mmp-prop-best64": "MMP+Prop",
    "mmp-prop-first5": "MMP+Prop-5",
    "pairit-real-only-best64": "PairIT",
    "pairit-real-only-first5": "PairIT-5",
    "pairit-beam-real-only-best64": "PairIT-Beam",
    "pairit-beam-real-only-first5": "PairIT-Beam-5",
    "pairit-rl-real-only-best64": "PairIT-RL",
    "pairit-rl-real-only-first5": "PairIT-RL-5",
}


def _scatter_labels() -> Dict[str, str]:
    labels = dict(_SCATTER_LABELS)
    for extra in extra_baselines():
        labels.update(extra.scatter_labels)
    return labels


def _scatter_marker(method: str) -> str:
    if method.startswith("pairit-"):
        return "*"
    if method.startswith("mmp-prop-"):
        return "s"
    if method.startswith("mmp-"):
        return "D"
    return "o"


def _scatter_marker_size(method: str) -> int:
    if method.startswith("pairit-"):
        return 220
    if method.startswith("mmp-prop-"):
        return 95
    if method.startswith("mmp-"):
        return 90
    return 90


def plot_scatter_sim_joint(
    summary_tables: Dict[str, pd.DataFrame],
    output_root: Path,
    per_seed_df: Optional[pd.DataFrame] = None,
    train_regime: str = "real-only",
    selection_mode: str = "best64",
    pairit_family: str = "pairit-beam",
) -> None:
    """Scatter plot of sim_seed (x) vs v_n_i (y), one point per method.

    Each point is the mean across seeds; a 1-σ confidence ellipse shows the
    2-D spread of (sim_seed, v_n_i) over seeds. Produces one figure with
    one panel per dataset (BACE / BBBP / HIV) and saves it under figures/.
    """
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    regular = summary_tables["regular"].copy()

    # ── method selection ──────────────────────────────────────────────────
    # Baselines: correct train_regime + selection_mode; exclude other PairIT families
    baseline_families = {"nfbo", "ddsbm-bridge", "ddsbm-sb", "hierg2g"} | {e.family for e in extra_baselines()}

    baseline_mask = (
        (regular["family"].isin(baseline_families))
        & (regular["train_regime"] == train_regime)
        & (regular["selection_mode"] == selection_mode)
    )
    pairit_mask = (
        (regular["family"] == pairit_family)
        & (regular["train_regime"] == train_regime)
        & (regular["selection_mode"] == selection_mode)
    )
    sub = regular[baseline_mask | pairit_mask].copy()

    if len(sub) == 0:
        return

    slug = train_regime.replace("+", "_plus_").replace("-", "_")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    dataset_titles = {"bace": "BACE", "bbbp": "BBBP", "hiv": "HIV"}

    for ax, dataset in zip(axes, DATASETS):
        dsub = sub[sub["dataset"] == dataset].copy()
        if len(dsub) == 0:
            ax.set_axis_off()
            continue

        ax.set_xlabel("Sim. to Seed (Tanimoto)", fontsize=11)
        ax.set_ylabel("Joint Metric v_n_i (%)", fontsize=11)
        ax.set_title(dataset_titles.get(dataset, dataset), fontsize=13, fontweight="bold")
        ax.grid(alpha=0.25, zorder=0)

        for _, row in dsub.iterrows():
            method = str(row["method"])
            color = _method_color(method)
            marker = _scatter_marker(method)
            ms = _scatter_marker_size(method)
            mx = float(row["sim_seed"]) if not pd.isna(row["sim_seed"]) else np.nan
            my = float(row["v_n_i"]) if not pd.isna(row["v_n_i"]) else np.nan
            if np.isnan(mx) or np.isnan(my):
                continue

            # Confidence ellipse from per-seed distribution
            if per_seed_df is not None:
                seed_sub = per_seed_df[
                    (per_seed_df["method"] == method) & (per_seed_df["dataset"] == dataset)
                ][["sim_seed", "v_n_i"]].copy()
                seed_sub["sim_seed"] = pd.to_numeric(seed_sub["sim_seed"], errors="coerce")
                seed_sub["v_n_i"] = pd.to_numeric(seed_sub["v_n_i"], errors="coerce")
                seed_sub = seed_sub.dropna()  # drop rows where either value is NaN
                xs = seed_sub["sim_seed"].to_numpy()
                ys = seed_sub["v_n_i"].to_numpy()
                if len(xs) >= 3:
                    _draw_confidence_ellipse(ax, xs, ys, color=color, n_std=1.0, alpha=0.15)

            # Scatter point (drawn on top of ellipse)
            ax.scatter(
                mx, my,
                color=color,
                marker=marker,
                s=ms,
                zorder=5,
                edgecolors="#333333",
                linewidths=0.6,
            )

            label = _scatter_labels().get(method, method)
            ax.annotate(
                label,
                xy=(mx, my),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                color="#222222",
                zorder=6,
            )

        # Legend proxy patches by family
        from matplotlib.lines import Line2D

        legend_entries: List[Tuple[str, str, str]] = [
            ("NFBO", "nfbo", "o"),
            ("DDSBM-Bridge", "ddsbm-bridge", "o"),
            ("DDSBM-SB", "ddsbm-sb", "o"),
            ("HierG2G", "hierg2g", "o"),
            (pairit_family.replace("pairit-", "PairIT-").replace("pairit", "PairIT"), "pairit", "*"),
        ]
        palette = _method_palette()
        family_colors = {
            "nfbo": palette["nfbo"],
            "ddsbm-bridge": palette["ddsbm-bridge"],
            "ddsbm-sb": palette["ddsbm-sb"],
            "hierg2g": palette["hierg2g"],
            "pairit": palette["pairit-beam-best64"],
        }
        handles = [
            Line2D(
                [0], [0],
                marker=mk,
                color="w",
                markerfacecolor=family_colors[fam],
                markeredgecolor="#333333",
                markeredgewidth=0.6,
                markersize=8 if mk == "*" else 7,
                label=lbl,
            )
            for lbl, fam, mk in legend_entries
            if fam in family_colors
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)

    regime_label = train_regime.replace("-", " ").replace("+", " + ")
    fig.suptitle(
        f"Similarity to Seed vs Joint Metric  |  {regime_label}  |  {selection_mode}",
        fontsize=12,
    )
    out_path = figures_dir / f"scatter_sim_joint_{slug}_{selection_mode}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[scatter] saved → {out_path}")


def plot_summaries(
    summary_tables: Dict[str, pd.DataFrame],
    output_root: Path,
    per_seed_df: Optional[pd.DataFrame] = None,
) -> None:
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    regular = summary_tables["regular"]
    distribution = summary_tables["distribution"]

    # If no per_seed_df supplied, try loading it from the output directory
    if per_seed_df is None:
        candidate = output_root / "regular" / "regular_per_seed.csv"
        if candidate.exists():
            per_seed_df = pd.read_csv(candidate)

    for dataset in DATASETS:
        std_lookup = _compute_std_lookup(per_seed_df, dataset) if per_seed_df is not None else None
        for train_regime in ("real-only",):
            slug = train_regime.replace("+", "_plus_").replace("-", "_")
            _plot_metric_bars(
                regular,
                dataset,
                train_regime,
                ["joint_metric"],
                {"joint_metric": "higher"},
                figures_dir / f"{dataset}_{slug}_joint_metric.png",
                "Joint Metric",
                std_lookup=std_lookup,
            )
            _plot_metric_bars(
                regular,
                dataset,
                train_regime,
                REGULAR_METRICS,
                REGULAR_DIRECTIONS,
                figures_dir / f"{dataset}_{slug}_regular_metrics.png",
                "Regular Metrics",
                std_lookup=std_lookup,
            )
            _plot_metric_bars(
                distribution,
                dataset,
                train_regime,
                DISTRIBUTION_METRICS,
                DISTRIBUTION_DIRECTIONS,
                figures_dir / f"{dataset}_{slug}_distribution_metrics.png",
                "Distribution Metrics",
                std_lookup=None,  # distribution metrics have no per-seed decomposition
            )

    # Scatter plot: sim_seed vs v_n_i — real-only / best64 / pairit-beam only
    plot_scatter_sim_joint(
        summary_tables,
        output_root,
        per_seed_df=per_seed_df,
        train_regime="real-only",
        selection_mode="best64",
        pairit_family="pairit-beam",
    )


def print_regular_tables(summary_tables: Dict[str, pd.DataFrame]) -> None:
    regular = summary_tables["regular"]
    for dataset in DATASETS:
        sub = regular[regular["dataset"] == dataset].copy()
        if len(sub) == 0:
            continue
        cols = [
            c
            for c in [
                "display_name",
                "train_regime",
                "selection_mode",
                "joint_metric",
                "valid",
                "novel",
                "improved",
                "sim_seed",
                "internal_diversity",
            ]
            if c in sub.columns
        ]
        print(f"\n=== {dataset.upper()} regular summary ===")
        print(sub[cols].to_string(index=False))


def run_full_comparison(
    *,
    output_root: Path,
    config: Optional[PairitEvalConfig] = None,
    verbose: bool = True,
    include_source_groups: Optional[Set[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Regenerate the MoleculeNet comparison tables (Tables 6-11).

    Paper protocol requires the two-stage oracle pipeline:
      * select_oracle = train-only RF (used to rank 64 candidates → top-5)
      * score_oracle  = MiniMol (used to judge the selected top-5)

    When called with `config=None` this constructs that setup by default. The
    guard below refuses to run with a single-oracle config (both slots empty
    OR both resolved to the same oracle), so we can never silently produce
    numbers that don't match the published protocol.

    MiniMol subprocess env vars (required unless PAIRIT_MINIMOL_PYTHON is
    unset AND every candidate is already cached in PAIRIT_MINIMOL_PRED_DIR):
      * PAIRIT_MINIMOL_PYTHON  — interpreter with `minimol` installed
      * PAIRIT_MINIMOL_PRED_DIR — writable cache dir (falls back to
                                  ASSETS/oracles/minimol_preds)
      * PYTHONPATH must include the fork's src/ so the worker can
        `python -m pairit.eval.oracles_minimol_worker`.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    eval_config = config or build_eval_config(
        moleculenet_train_dir=REAL_ONLY_TRAIN_DIR,
        moleculenet_test_dir=DEFAULT_MOLECULENET_TEST_DIR,
        score_oracle="minimol",
    )
    # Two-oracle protocol guard. Both slots MUST be filled AND distinct.
    # select_oracle defaults to evaluator_dir (train-only RF) via
    # resolve_oracle_pair's fallback; score_oracle MUST be set explicitly
    # (paper protocol: 'minimol'). A missing score_oracle silently falls
    # back to RF-only mode; a select_oracle that resolves to the SAME oracle
    # as score_oracle is the reward-hacking configuration we're preventing —
    # same model both filters candidates AND judges the winners.
    if not eval_config.score_oracle:
        raise ValueError(
            "run_full_comparison requires the two-stage oracle protocol. "
            "score_oracle is not set — silent fallback to single-oracle mode "
            "is deliberately disabled to prevent producing numbers that don't "
            "match the published protocol. "
            "Build config via build_eval_config(score_oracle='minimol', ...) "
            "and set PAIRIT_MINIMOL_PYTHON to a Python venv with `minimol` "
            "installed. See internal/docs/handoff/moleculenet_oracle_session.md for the "
            "full env-var recipe."
        )
    # Best-effort structural check: refuse when select_spec explicitly equals
    # score_spec. The real select == score collision is only knowable AFTER
    # resolve_oracle_pair runs per-dataset; that check is enforced inside
    # evaluate() below.
    if eval_config.select_oracle is not None and eval_config.select_oracle == eval_config.score_oracle:
        raise ValueError(
            f"select_oracle and score_oracle are identical ({eval_config.score_oracle!r}). "
            "Two-oracle protocol requires DIFFERENT oracles: RF filters, MiniMol judges. "
            "Using the same oracle for both is the reward-hacking configuration we exclude."
        )
    specs = _build_specs()
    if include_source_groups is not None:
        before = len(specs)
        specs = [s for s in specs if s.source_group in include_source_groups]
        if verbose:
            print(
                f"[run] source_group filter kept {len(specs)}/{before} specs "
                f"(groups={sorted(include_source_groups)})"
            )
    if verbose:
        print(f"[run] built comparison manifest with {len(specs)} dataset-specific source entries")
    final_df, method_meta, selection_summary = materialize_inputs(
        specs,
        output_root=output_root,
        config=eval_config,
        verbose=verbose,
    )
    if len(selection_summary):
        selection_summary.to_csv(output_root / "materialized" / "selection_summary_all64.csv", index=False)
    outputs = evaluate_final_inputs(
        final_df,
        specs,
        output_root=output_root,
        config=eval_config,
        method_meta=method_meta,
        verbose=verbose,
    )
    summary_tables = build_summary_tables(outputs, output_root)
    plot_summaries(summary_tables, output_root)
    print_regular_tables(summary_tables)
    return outputs
