"""
Configuration for pairit_eval: paths, dataset definitions, and default
generation sources.

Used for novelty (training SMILES) and improvement (evaluator). If not provided,
validity, relevance (strict), and internal diversity still run.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pairit._assets import ASSETS

# Data root and baseline generations root. Both default to the bundled assets/
# tree; override with the env vars below to point at data hosted elsewhere.
_BASE = Path(os.environ.get("PAIRIT_EVAL_DATA_ROOT", str(ASSETS)))
_RAW_OUTPUTS_BASE = Path(
    os.environ.get(
        "PAIRIT_EVAL_RAW_OUTPUTS_BASE",
        str(ASSETS / "generations"),
    )
)


DATASET_CONFIG = {
    "bace": {
        # Novelty reference = FINETUNE TRAIN ONLY. A generation copied from
        # ZINC (pretraining corpus, not in the finetune train split) isn't
        # "memorised" for the finetuning task -- it's a pretraining artefact
        # that penalises every pretrained model equally, so ZINC is NOT
        # unioned in here. VIN is unchanged (Improved filter dominates
        # ZINC-hits) but Novel% is 0-13 pts higher under this stricter
        # reference than if ZINC were unioned.
        "train_smiles": [
            str(ASSETS / "data" / "moleculenet" / "bace" / "train" / "molecules.parquet"),
            # ZINC-250K is unioned in as a memorisation reference: pretrained
            # models can reproduce ZINC molecules verbatim, which would otherwise
            # count as novel. The same union applies to every dataset.
            str(ASSETS / "data" / "zinc" / "train" / "molecules.parquet"),
        ],"smiles_col": "smiles", "label_col": "Class", "task_name": "bace_b", "evaluator_type": "binary", "direction": "increasing", "family": "moleculenet", "n_test_seeds": 135},
    "bbbp": {
        # Novelty reference = FINETUNE TRAIN ONLY. A generation copied from
        # ZINC (pretraining corpus, not in the finetune train split) isn't
        # "memorised" for the finetuning task -- it's a pretraining artefact
        # that penalises every pretrained model equally, so ZINC is NOT
        # unioned in here. VIN is unchanged (Improved filter dominates
        # ZINC-hits) but Novel% is 0-13 pts higher under this stricter
        # reference than if ZINC were unioned.
        "train_smiles": [
            str(ASSETS / "data" / "moleculenet" / "bbbp" / "train" / "molecules.parquet"),
            str(ASSETS / "data" / "zinc" / "train" / "molecules.parquet"),
        ],"smiles_col": "smiles", "label_col": "p_np", "task_name": "bbbp_b", "evaluator_type": "binary", "direction": "increasing", "family": "moleculenet", "n_test_seeds": 88},
    "hiv": {
        # Novelty reference = FINETUNE TRAIN ONLY. A generation copied from
        # ZINC (pretraining corpus, not in the finetune train split) isn't
        # "memorised" for the finetuning task -- it's a pretraining artefact
        # that penalises every pretrained model equally, so ZINC is NOT
        # unioned in here. VIN is unchanged (Improved filter dominates
        # ZINC-hits) but Novel% is 0-13 pts higher under this stricter
        # reference than if ZINC were unioned.
        "train_smiles": [
            str(ASSETS / "data" / "moleculenet" / "hiv" / "train" / "molecules.parquet"),
            str(ASSETS / "data" / "zinc" / "train" / "molecules.parquet"),
        ],"smiles_col": "smiles", "label_col": "HIV_active", "task_name": "hiv_b", "evaluator_type": "binary", "direction": "increasing", "family": "moleculenet", "n_test_seeds": 231},
    "zinc250k": {
        "smiles_col": "smiles",
        "label_col": None,
        "task_name": None,
        "evaluator_type": "continuous",
        "default_property": "logp",
        "direction": "increasing",
        "family": "zinc",
    },
    # --- TDC datasets (Tanimoto-kNN pairs) ---
    # Caco2 / FreeSolv have measured continuous targets, so improvement is a property
    # delta. hERG's measured label is binary, so it uses the classifier path.
    "caco2_wang": {
        # Novelty reference = finetune train only (Caco2_Wang train). See
        # bace/bbbp/hiv above for the finetune-only rationale.
        "train_smiles": [
            str(ASSETS / "data" / "tdc" / "caco2_wang" / "train" / "molecules.parquet"),
        ],
        "smiles_col": "smiles",
        "label_col": None,
        "task_name": "caco2_wang",
        "evaluator_type": "continuous",
        "default_property": "caco2_papp",
        "direction": "increasing",
        "family": "tdc",
        "test_smiles": [
            str(ASSETS / "data" / "tdc" / "caco2_wang" / "test" / "molecules.parquet"),
        ],
    },
    "freesolv": {
        # Novelty reference = finetune train only (FreeSolv train). See
        # bace/bbbp/hiv above for the finetune-only rationale.
        "train_smiles": [
            str(ASSETS / "data" / "tdc" / "freesolv" / "train" / "molecules.parquet"),
        ],
        "smiles_col": "smiles",
        "label_col": None,
        "task_name": "freesolv",
        "evaluator_type": "continuous",
        "default_property": "freesolv_dg",
        "direction": "decreasing",
        "family": "tdc",
        "test_smiles": [
            str(ASSETS / "data" / "tdc" / "freesolv" / "test" / "molecules.parquet"),
        ],
    },
    "herg": {
        # Novelty reference = hERG train unioned with ZINC-250K train, so that
        # pretrained models cannot pass memorised ZINC molecules off as novel.
        "train_smiles": [
            str(ASSETS / "data" / "tdc" / "herg" / "train" / "molecules.parquet"),
            str(ASSETS / "data" / "zinc" / "train" / "molecules.parquet"),
        ],
        "smiles_col": "smiles",
        # `Y` is the binary blocker label in the TDC hERG parquet (1=blocker,
        # 0=non-blocker). label_col is required for the distribution-reference
        # filter and stable seed indexing; leaving it None makes both helpers
        # raise "cannot filter class-{0,1}" and the whole distribution pass
        # aborts for hERG.
        "label_col": "Y",
        "task_name": "herg_b",
        "evaluator_type": "binary",
        "test_smiles": [
            str(ASSETS / "data" / "tdc" / "herg" / "test" / "molecules.parquet"),
        ],
        # Improvement is blocker -> non-blocker, i.e. predicted label 1 -> 0. Without
        # this the binary improvement path scores hERG exactly backwards.
        "direction": "decreasing",
        # Family = subdir under ASSETS/oracles/ where the RF evaluator + MiniMol
        # head live. Also used by the coverage-aware fixed denominator (Option A):
        # n_test_seeds is the expected class-1 blocker test-seed count.
        "family": "tdc",
        "n_test_seeds": 85,
    },
    # --- CYP substrate (multi-property; three endpoints on ONE molecule table) ---
    #
    # Registered as three separate datasets, one per isoform, because they must be
    # REPORTED SEPARATELY. Their oracles differ sharply in quality -- held-out test
    # ROC-AUC 0.587 / 0.771 / 0.645 for 2C9 / 2D6 / 3A4 -- so an aggregate V,I,N
    # would blend a near-chance signal with a usable one and read as a single
    # trustworthy number. CYP2C9 in particular must never be reported alone; see
    # internal/docs/cyp_substrate_dataset.md in the lobster repo.
    #
    # All three share the same molecule tables and the same novelty references, so
    # only `label_col` / `task_name` / `default_property` differ between them.
    **{
        f"cyp{iso}_substrate": {
            # Novelty reference = finetune train only (CYP_Substrate train).
            # See bace/bbbp/hiv above for the finetune-only rationale.
            "train_smiles": [
                str(ASSETS / "data" / "cyp_substrate" / "train" / "molecules.parquet"),
            ],
            "smiles_col": "smiles",
            "label_col": f"cyp{iso}_substrate",
            "task_name": f"cyp{iso}_substrate",
            "evaluator_type": "binary",
            "default_property": f"cyp{iso}_substrate",
            # "Fewer metabolising CYPs is better" -- a defensible default, NOT a
            # settled fact. Metabolic redundancy is often desirable, since a drug
            # cleared by several CYPs is buffered against one pathway being
            # inhibited or genetically absent. Same status as FreeSolv's direction.
            "direction": "decreasing",
            "family": "cyp_substrate",
            "test_smiles": [
                str(ASSETS / "data" / "cyp_substrate" / "test" / "molecules.parquet"),
            ],
        }
        for iso in ("2c9", "2d6", "3a4")
    },
}
MOLECULENET_DATASETS = ["bace", "bbbp", "hiv"]
TDC_DATASETS = ["caco2_wang", "freesolv", "herg"]
CYP_DATASETS = ["cyp2c9_substrate", "cyp2d6_substrate", "cyp3a4_substrate"]
DEFAULT_DATASETS = MOLECULENET_DATASETS + ["zinc250k"] + TDC_DATASETS


_PAIRIT_FROM_ZINC = {
    "bace": _RAW_OUTPUTS_BASE / "pairit/from_zinc/bace.csv",
    "bbbp": _RAW_OUTPUTS_BASE / "pairit/from_zinc/bbbp.csv",
    "hiv": _RAW_OUTPUTS_BASE / "pairit/from_zinc/hiv.csv",
}
_PAIRIT_REAL_ONLY = {
    "bace": ASSETS / "generations" / "pairit" / "moleculenet" / "bace" / "all_generations.parquet",
    "bbbp": ASSETS / "generations" / "pairit" / "moleculenet" / "bbbp" / "all_generations.parquet",
    "hiv": ASSETS / "generations" / "pairit" / "moleculenet" / "hiv" / "all_generations.parquet",
}
_HIERG2G_04 = {
    "bace": _RAW_OUTPUTS_BASE / "hierg2g/bace/04/bace_directed_from_logp04/translations_epoch11.txt",
    "bbbp": _RAW_OUTPUTS_BASE / "hierg2g/bbbp/04/bbbp_directed_from_logp04/translations_epoch11.txt",
    "hiv": _RAW_OUTPUTS_BASE / "hierg2g/hiv/04/hiv_directed_from_logp04/translations_epoch11.txt",
    "zinc250k": _RAW_OUTPUTS_BASE / "hierg2g/zinc250k/04/zinc250k_directed_from_logp04/translations_ckpt22.txt",
}
_HIERG2G_06 = {
    "bace": _RAW_OUTPUTS_BASE / "hierg2g/bace/06/bace_directed_from_logp06/translations_epoch11.txt",
    "bbbp": _RAW_OUTPUTS_BASE / "hierg2g/bbbp/06/bbbp_directed_from_logp06/translations_epoch11.txt",
    "hiv": _RAW_OUTPUTS_BASE / "hierg2g/hiv/06/hiv_directed_from_logp06/translations_epoch11.txt",
    "zinc250k": _RAW_OUTPUTS_BASE / "hierg2g/zinc250k/06/zinc250k_directed_from_logp06/translations_ckpt22.txt",
}

# Nearest-neighbor lookup baseline — generated by baselines/nearest_neighbor/build_nn_lookup.py
_NN_GEN_BASE = ASSETS / "generations" / "nn-lookup"
_NN_REAL_ONLY = {
    "bace": _NN_GEN_BASE / "nn-real-only/bace.csv",
    "bbbp": _NN_GEN_BASE / "nn-real-only/bbbp.csv",
    "hiv": _NN_GEN_BASE / "nn-real-only/hiv.csv",
    "zinc250k": _NN_GEN_BASE / "nn-real-only/zinc250k.csv",
}
_NN_REAL_ONLY_ZINC = {
    "bace": _NN_GEN_BASE / "nn-real-only-zinc/bace.csv",
    "bbbp": _NN_GEN_BASE / "nn-real-only-zinc/bbbp.csv",
    "hiv": _NN_GEN_BASE / "nn-real-only-zinc/hiv.csv",
}

_MMP_LEGACY_ALL64 = {
    "bace": _BASE / "datasets/moleculenet/mmp_baseline/bace_b_mmp_baseline_all_generations.csv",
    "bbbp": _BASE / "datasets/moleculenet/mmp_baseline/bbbp_b_mmp_baseline_all_generations.csv",
    "hiv": _BASE / "datasets/moleculenet/mmp_baseline/hiv_b_mmp_baseline_all_generations.csv",
}
_MMP_PROP_BASE = _BASE / "pairit_eval/results/raw_outputs/mmp_property_baseline"
_MMP_PROP_ALL64 = {
    dataset: _MMP_PROP_BASE / f"{DATASET_CONFIG[dataset]['task_name']}_mmp_property_baseline_all_generations.csv"
    for dataset in MOLECULENET_DATASETS
}
_MMP_PROP_TOP5 = {
    dataset: _MMP_PROP_BASE / f"{DATASET_CONFIG[dataset]['task_name']}_mmp_property_baseline_top5.csv"
    for dataset in MOLECULENET_DATASETS
}


_CANONICAL_GENERATION_PATHS = {
    "nfbo": {
        "moleculenet": {
            "bace": _RAW_OUTPUTS_BASE / "nfbo/bace.csv",
            "bbbp": _RAW_OUTPUTS_BASE / "nfbo/bbbp.csv",
            "hiv": _RAW_OUTPUTS_BASE / "nfbo/hiv.csv",
        },
        "zinc250k_glob": str(_RAW_OUTPUTS_BASE / "nfbo/zinc250k/logp_improvement_*_seed42.csv"),
    },
    "ddsbm-bridge": {
        "moleculenet": {
            "bace": _RAW_OUTPUTS_BASE / "ddsbm-bridge/bace",
            "bbbp": _RAW_OUTPUTS_BASE / "ddsbm-bridge/bbbp",
            "hiv": _RAW_OUTPUTS_BASE / "ddsbm-bridge/hiv",
        },
        "zinc250k": _RAW_OUTPUTS_BASE / "ddsbm-bridge/zinc250k",
    },
    "ddsbm-sb": {
        "moleculenet": {
            "bace": _RAW_OUTPUTS_BASE / "ddsbm-sb/bace",
            "bbbp": _RAW_OUTPUTS_BASE / "ddsbm-sb/bbbp",
            "hiv": _RAW_OUTPUTS_BASE / "ddsbm-sb/hiv",
        },
        "zinc250k": _RAW_OUTPUTS_BASE / "ddsbm-sb/zinc250k",
    },
    "mmp": {
        "moleculenet": {
            "bace": _MMP_LEGACY_ALL64["bace"],
            "bbbp": _MMP_LEGACY_ALL64["bbbp"],
            "hiv": _MMP_LEGACY_ALL64["hiv"],
        },
    },
    "mmp-prop": {
        "moleculenet": {
            "bace": _MMP_PROP_TOP5["bace"],
            "bbbp": _MMP_PROP_TOP5["bbbp"],
            "hiv": _MMP_PROP_TOP5["hiv"],
        },
    },
    "pairit-from-zinc": {
        "moleculenet": dict(_PAIRIT_FROM_ZINC),
        "zinc250k_parquet": _RAW_OUTPUTS_BASE / "pairit/from_zinc/conditional_step_9999_generated_molecules.parquet",
    },
    "pairit-real-only": {
        "moleculenet": dict(_PAIRIT_REAL_ONLY),
    },
    "hierg2g-04": {
        "datasets": dict(_HIERG2G_04),
    },
    "hierg2g-06": {
        "datasets": dict(_HIERG2G_06),
    },
    # Nearest-neighbor lookup baselines
    "nn-real-only": {
        "datasets": dict(_NN_REAL_ONLY),
    },
    "nn-real-only-zinc": {
        "datasets": dict(_NN_REAL_ONLY_ZINC),
    },
}


# Backward-compatible aliases for older scripts.
DEFAULT_GENERATION_PATHS = {
    **_CANONICAL_GENERATION_PATHS,
    "ddsbm_bridge": _CANONICAL_GENERATION_PATHS["ddsbm-bridge"],
    "ddsbm_sb": _CANONICAL_GENERATION_PATHS["ddsbm-sb"],
    "pairit": {
        "from_zinc": dict(_PAIRIT_FROM_ZINC),
    },
}


def get_default_generation_paths() -> Dict[str, Dict[str, object]]:
    return DEFAULT_GENERATION_PATHS


def get_mmp_generation_paths(method: str = "mmp", kind: str = "all64") -> Dict[str, Path]:
    """
    Return MoleculeNet MMP generation paths for one MMP variant.

    `mmp` points to the legacy baseline exports.
    `mmp-prop` points to the property-aware baseline exports.
    """
    if kind not in {"all64", "top5"}:
        raise ValueError(f"Unsupported MMP generation path kind: {kind}")
    if method == "mmp":
        return dict(_MMP_LEGACY_ALL64)
    if method == "mmp-prop":
        return dict(_MMP_PROP_ALL64 if kind == "all64" else _MMP_PROP_TOP5)
    raise ValueError(f"Unsupported MMP method: {method}")


def get_preferred_mmp_generation_paths(kind: str = "all64") -> Dict[str, Path]:
    """Backward-compatible alias for the property-aware MMP baseline paths."""
    return get_mmp_generation_paths(method="mmp-prop", kind=kind)


@dataclass
class PairitEvalConfig:
    """
    Paths and dataset list for evaluation.

    Parameters
    ----------
    evaluator_dir : directory containing {task_name}.joblib (e.g. bace_b.joblib)
    moleculenet_train_dir : base dir with {dataset}/train/molecules_{dataset}_real.parquet
        (e.g. .../real_pairs_by_split). Used for novelty + distribution metrics reference.
    moleculenet_test_dir : base dir with {dataset}/test/molecules_{dataset}_real.parquet
        (e.g. .../real_pairs_by_split). Used for sim_gt_mean.
    zinc250k_train_molecules : path to molecules.parquet with split column (filter to train).
        Used for novelty + distribution metrics reference.
    zinc250k_test_csv : path to zinc250k_test_icml.csv. Used for sim_gt_mean.

    Distribution-metric reference set (FCD/SNN/cosine) is fixed to class-1
    train molecules for MoleculeNet and full train for zinc250k — see
    `_load_distribution_reference` in evaluate.py. Not user-configurable.
    """

    evaluator_dir: Optional[Union[Path, str]] = None

    moleculenet_train_dir: Optional[Union[Path, str]] = None
    moleculenet_test_dir: Optional[Union[Path, str]] = None
    zinc250k_train_molecules: Optional[Union[Path, str]] = None
    zinc250k_test_csv: Optional[Union[Path, str]] = None

    # Separate oracles for selecting candidates and for judging them. Either may
    # be a directory, a .joblib, a smiles/prediction table, the string "minimol",
    # or an Oracle instance. When both are None, `evaluator_dir` is used for both
    # and behaviour is exactly as it was before oracles existed. When only
    # `score_oracle` is given, selection uses it too.
    #
    # Deliberately NOT in _PATH_FIELDS: __post_init__ would coerce them to Path
    # and destroy a bare name like "minimol" or a live Oracle object.
    select_oracle: Optional[Any] = None
    score_oracle: Optional[Any] = None

    # Interpreter used to score molecules an out-of-process oracle is missing.
    # Falls back to $PAIRIT_MINIMOL_PYTHON.
    minimol_python: Optional[Union[Path, str]] = None
    # Fail instead of scoring when an out-of-process oracle has cache misses.
    offline_oracles: bool = False

    datasets: List[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    dataset_config: Dict[str, Dict[str, str]] = field(default_factory=lambda: dict(DATASET_CONFIG))

    _PATH_FIELDS = (
        "evaluator_dir",
        "moleculenet_train_dir", "moleculenet_test_dir",
        "zinc250k_train_molecules", "zinc250k_test_csv",
    )

    def __post_init__(self) -> None:
        for attr in self._PATH_FIELDS:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, Path):
                setattr(self, attr, Path(val))

    def get_evaluator_dir(self) -> Optional[Path]:
        return Path(self.evaluator_dir) if self.evaluator_dir else None
