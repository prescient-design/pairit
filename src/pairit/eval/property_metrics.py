"""
Property metrics: PropertySpec, registry, compute_properties, binary improvement, binning.

- Binary improvement: threshold classifier probability at 0.5 → improvement.
- Continuous: use predicted probability (MoleculeNet) or RDKit descriptor (logP, QED, SA, ...) directly.
- Binning: PropertyBinningConfig + value_to_bin / delta_to_bin with default cutoffs from lobster.
"""

from __future__ import annotations

import os
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, DataStructs, Descriptors, QED, rdMolDescriptors

from pairit._assets import ASSETS
from .config import DATASET_CONFIG

# Optional SA (sascorer)
try:
    import sascorer as _sascorer_mod
except ImportError:
    _sascorer_mod = None


# ---------------------------------------------------------------------------
# PropertySpec and registry (RDKit + classifier-backed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertySpec:
    """
    Spec for a single molecular property.

    - name: e.g. 'logp', 'qed', 'Class'
    - fn: (Chem.Mol) -> float | None  (continuous value; for MoleculeNet this is predicted probability)
    - direction: 'higher' | 'lower'
    - min_delta: minimal change to count as improved (default 0.0)
    """

    name: str
    fn: Callable[[Chem.Mol], Optional[float]]
    direction: str = "higher"
    min_delta: float = 0.0


def _safe_compute(fn: Callable[..., Any], mol: Chem.Mol) -> Optional[float]:
    try:
        return float(fn(mol))
    except Exception:
        return None


def _compute_logp(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(Crippen.MolLogP, mol)


def _compute_qed(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(QED.qed, mol)


def _compute_sa(mol: Chem.Mol) -> Optional[float]:
    if _sascorer_mod is None:
        return None
    try:
        return float(_sascorer_mod.calculateScore(mol))
    except Exception:
        return None


def _compute_mw(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(Descriptors.MolWt, mol)


def _compute_tpsa(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(Descriptors.TPSA, mol)


def _compute_hbd(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(Descriptors.NumHDonors, mol)


def _compute_hba(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(rdMolDescriptors.CalcNumHBA, mol)


def _compute_rotatable_bonds(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(Descriptors.NumRotatableBonds, mol)


def _compute_frac_sp3_c(mol: Chem.Mol) -> Optional[float]:
    return _safe_compute(Descriptors.FractionCSP3, mol)


# Built-in registry: name -> (compute_fn, direction)


# --- TDC regression targets (RF oracles, loaded lazily) ---------------------------
# These predict a measured assay value rather than a cheap RDKit descriptor, so they
# need the trained oracle. Loaded on first use so importing this module never depends
# on the evaluator directory being present.
_TDC_EVALUATOR_DIR = os.environ.get(
    "TDC_EVALUATOR_DIR",
    str(ASSETS / "oracles" / "tdc" / "evaluators"),
)
_TDC_ORACLE_CACHE: Dict[str, Any] = {}


def _tdc_predict(mol: Chem.Mol, joblib_name: str) -> Optional[float]:
    """Score one molecule with a TDC RF oracle (ECFP4, Morgan r=2, 2048 bits)."""
    try:
        if joblib_name not in _TDC_ORACLE_CACHE:
            import joblib as _joblib

            _TDC_ORACLE_CACHE[joblib_name] = _joblib.load(
                os.path.join(_TDC_EVALUATOR_DIR, joblib_name)
            )
        model = _TDC_ORACLE_CACHE[joblib_name]
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        X = arr.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0, 1])
        return float(model.predict(X)[0])
    except Exception:
        return None


def _compute_caco2_papp(mol: Chem.Mol) -> Optional[float]:
    """Caco-2 apparent permeability, log10 cm/s. Higher is more permeable."""
    return _tdc_predict(mol, "caco2_wang.joblib")


def _compute_freesolv_dg(mol: Chem.Mol) -> Optional[float]:
    """Hydration free energy, kcal/mol. Lower (more negative) is better hydrated."""
    return _tdc_predict(mol, "freesolv.joblib")


def _compute_herg_block(mol: Chem.Mol) -> Optional[float]:
    """hERG blockade probability. Lower is safer."""
    return _tdc_predict(mol, "herg_b.joblib")


PROPERTY_REGISTRY: Dict[str, Tuple[Callable[[Chem.Mol], Optional[float]], str]] = {
    "logp": (_compute_logp, "higher"),
    "qed": (_compute_qed, "higher"),
    "sa": (_compute_sa, "lower"),
    "mw": (_compute_mw, "higher"),
    "tpsa": (_compute_tpsa, "lower"),
    "hbd": (_compute_hbd, "lower"),
    "hba": (_compute_hba, "lower"),
    "rotatable_bonds": (_compute_rotatable_bonds, "lower"),
    "frac_sp3_c": (_compute_frac_sp3_c, "higher"),
    # TDC targets: direction matches the training configs
    "caco2_papp": (_compute_caco2_papp, "higher"),
    "freesolv_dg": (_compute_freesolv_dg, "lower"),
    "herg_block": (_compute_herg_block, "lower"),
}

# MoleculeNet: property name -> task_name for .joblib (continuous = probability; binary = threshold 0.5)
CLASSIFIER_PROPERTY_TASK: Dict[str, str] = {
    "Class": "bace_b",
    "p_np": "bbbp_b",
    "HIV_active": "hiv_b",
}


def _make_classifier_spec(name: str, evaluator: Any, nbits: int = 2048, radius: int = 2) -> PropertySpec:
    """PropertySpec that returns predicted probability (continuous); binary is threshold 0.5 elsewhere."""

    def fn(mol: Chem.Mol) -> Optional[float]:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros(nbits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            p = evaluator.predict_proba(arr.reshape(1, -1))[:, 1]
            return float(p[0])
        except Exception:
            return None

    return PropertySpec(name=name, fn=fn, direction="higher", min_delta=0.0)


def get_property_specs(
    property_names: List[str],
    evaluator_dir: Optional[Union[Path, str]] = None,
    dataset_config: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[PropertySpec]:
    """
    Get PropertySpec list for the given property names.

    - RDKit properties (logp, qed, sa, ...): no evaluator_dir needed.
    - Classifier properties (Class, p_np, HIV_active): require evaluator_dir; they use
      predicted probability as the continuous value (binary = threshold 0.5 in batch_predict_binary_property).
    """
    config = dataset_config if dataset_config is not None else DATASET_CONFIG
    evaluator_path = Path(evaluator_dir) if evaluator_dir else None
    specs: List[PropertySpec] = []

    for name in property_names:
        if name in PROPERTY_REGISTRY:
            fn, direction = PROPERTY_REGISTRY[name]
            specs.append(PropertySpec(name=name, fn=fn, direction=direction, min_delta=0.0))
        elif name in CLASSIFIER_PROPERTY_TASK:
            if evaluator_path is None:
                raise ValueError(
                    f"Classifier property '{name}' requires evaluator_dir to load "
                    f"{CLASSIFIER_PROPERTY_TASK[name]}.joblib"
                )
            task = CLASSIFIER_PROPERTY_TASK[name]
            path = evaluator_path / f"{task}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Evaluator not found: {path}")
            evaluator = joblib.load(path)
            specs.append(_make_classifier_spec(name, evaluator))
        else:
            available = ", ".join(sorted(set(PROPERTY_REGISTRY) | set(CLASSIFIER_PROPERTY_TASK)))
            raise ValueError(f"Unknown property '{name}'. Available: {available}")
    return specs


def compute_properties(smiles: str, specs: Sequence[PropertySpec]) -> Dict[str, Optional[float]]:
    """Compute property values for a single SMILES. Returns name -> value (None if failed)."""
    out: Dict[str, Optional[float]] = {s.name: None for s in specs}
    if not smiles or not str(smiles).strip():
        return out
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return out
        for s in specs:
            try:
                v = s.fn(mol)
                out[s.name] = float(v) if v is not None else None
            except Exception:
                pass
        return out
    except Exception:
        return out


def get_available_properties() -> List[str]:
    """All registered property names (RDKit + classifier)."""
    return sorted(set(PROPERTY_REGISTRY) | set(CLASSIFIER_PROPERTY_TASK))


# ---------------------------------------------------------------------------
# Binary improvement (evaluator load, batch predict, improvement)
# ---------------------------------------------------------------------------


def load_evaluator(
    dataset: str,
    evaluator_dir: Union[Path, str],
    dataset_config: Optional[Dict[str, Dict[str, str]]] = None,
) -> Any:
    """Load the joblib evaluator for this dataset (task_name from config)."""
    config = dataset_config if dataset_config is not None else DATASET_CONFIG
    task_name = config.get(dataset, {}).get("task_name", dataset)
    path = Path(evaluator_dir) / f"{task_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Evaluator not found: {path}")
    return joblib.load(path)


def batch_predict_binary_property(
    smiles_list: List[str],
    evaluator: Any,
    nbits: int = 2048,
    radius: int = 2,
    threshold: float = 0.5,
) -> List[Tuple[Optional[float], Optional[int]]]:
    """
    For each SMILES: (prob, label) with label = 1 if prob >= threshold else 0.
    Invalid SMILES get (None, None).
    """
    results: List[Tuple[Optional[float], Optional[int]]] = [(None, None)] * len(smiles_list)
    valid_idx, fps = [], []
    for i, s in enumerate(smiles_list):
        if pd.isna(s) or s == "" or s == "None":
            results[i] = (None, None)
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            results[i] = (None, None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_idx.append(i)
    if fps:
        probs = evaluator.predict_proba(np.vstack(fps))[:, 1]
        for j, idx in enumerate(valid_idx):
            p = float(probs[j])
            results[idx] = (p, 1 if p >= threshold else 0)
    return results


def compute_improvement(
    seed_label: int,
    pred_label: Optional[int],
    is_duplicate: bool,
    direction: str = "increasing",
) -> bool:
    """Did this generation improve on its seed (undesirable class -> desirable class)?

    Parameters
    ----------
    seed_label
        The seed's own class label, 0 or 1.
    pred_label
        The generation's predicted class label, or None if unscorable.
    is_duplicate
        True when the generation is identical to its seed. Improvement cannot be
        claimed for a molecule that did not change.
    direction
        ``"increasing"`` (default) when label 1 is the desirable class, as for
        BACE, BBBP and HIV. ``"decreasing"`` when label 0 is desirable -- hERG
        improvement is blocker -> non-blocker, so the target label is 0.

        This argument defaults to the historical behaviour, so existing callers
        are unaffected. Passing it wrong inverts improvement for the dataset,
        which is why every entry in DATASET_CONFIG now states its direction
        explicitly rather than relying on a default.

    Returns
    -------
    improvement
        True when the seed is in the undesirable class and the generation is in the
        desirable class. False when the generation is a duplicate or unscorable.

    Notes
    -----
    **This does not measure prompt adherence, and cannot be adapted to.** It judges
    against one direction fixed for the whole dataset (`DATASET_CONFIG[ds]["direction"]`).
    Property-prompted models are told per example which way to move each property, and
    the instruction for a given property points up on some pairs and down on others --
    the target is `sign(property(ground-truth partner) - property(seed))`, which is
    per-pair. A "leave this unchanged" instruction has no counterpart here at all, and
    a generation can be `improvement=True` while doing the opposite of what it was
    asked. pairit_eval holds no representation of the prompt.

    Prompted evaluation lives in the `lobster` repo -- `compute_direction_adherence` in
    `notebooks/multiprop_rl_analysis.py`, with null baselines in
    `scripts/lookup_baseline_direction_adherence.py` and `scripts/cyp_lookup_baseline.py`.
    Those score through `pairit_eval.resolve_oracle` so thresholds and directions stay
    consistent with the numbers computed here. See DECISIONS.md section 2.0.

    Examples
    --------
    >>> compute_improvement(0, 1, False)
    True
    >>> compute_improvement(0, 1, False, direction="decreasing")
    False
    >>> compute_improvement(1, 0, False, direction="decreasing")
    True
    """
    if is_duplicate or pred_label is None:
        return False
    good, bad = (1, 0) if str(direction).lower() in ("increasing", "higher", "max") else (0, 1)
    return seed_label == bad and pred_label == good


# ---------------------------------------------------------------------------
# Continuous property improvement stats
# ---------------------------------------------------------------------------


def is_valid_smiles(smiles: str, sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None) -> bool:
    """RDKit-parseable and optional sanity check. Public API for validity outside evaluate()."""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        return sanity_fn(m) if sanity_fn else True
    except Exception:
        return False


def _valid_numeric_deltas(
    seeds: Sequence[str],
    gens: Sequence[str],
    spec: PropertySpec,
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
) -> List[float]:
    """Numeric deltas (gen - seed) for valid (seed, gen) pairs; invalid/missing -> skipped."""
    out: List[float] = []
    for s, g in zip(seeds, gens):
        if not (is_valid_smiles(s, sanity_fn) and is_valid_smiles(g, sanity_fn)):
            continue
        ms, mg = Chem.MolFromSmiles(s), Chem.MolFromSmiles(g)
        if ms is None or mg is None:
            continue
        try:
            vs = spec.fn(ms)
            vg = spec.fn(mg)
        except Exception:
            vs, vg = None, None
        if vs is None or vg is None:
            continue
        try:
            d = float(vg) - float(vs)
        except Exception:
            continue
        if np.isfinite(d):
            out.append(d)
    return out


def property_improvement_stats(
    seeds: Sequence[str],
    gens: Sequence[str],
    spec: PropertySpec,
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
) -> Dict[str, Any]:
    """
    Continuous property improvement over valid (seed, gen) pairs.
    Returns mean_delta, median_delta, std_delta, frac_improved, n.
    """
    deltas = _valid_numeric_deltas(seeds, gens, spec, sanity_fn=sanity_fn)
    if not deltas:
        return {"mean_delta": None, "median_delta": None, "std_delta": None, "n": 0, "frac_improved": None}
    direction = (spec.direction or "higher").lower()
    improved_flags = (
        [d > spec.min_delta for d in deltas]
        if direction == "higher"
        else [(-d) > spec.min_delta for d in deltas]
    )
    return {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(np.std(deltas, ddof=0)),
        "n": len(deltas),
        "frac_improved": float(np.mean(improved_flags)),
    }


def property_top_k_stats(
    generated_smiles: Sequence[str],
    spec: PropertySpec,
    k: int,
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
) -> Dict[str, Any]:
    """
    Among valid generated molecules, return best property value and mean of top-k.
    direction='higher' -> best = max, top-k = k highest; direction='lower' -> best = min, top-k = k lowest.
    Returns best, mean_top_k, n (number of valid values), n_top_k (length of top-k list).
    """
    values: List[float] = []
    for s in generated_smiles:
        if not s or not str(s).strip():
            continue
        if not is_valid_smiles(s, sanity_fn):
            continue
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            v = spec.fn(mol)
            if v is not None and np.isfinite(v):
                values.append(float(v))
        except Exception:
            continue
    if not values:
        return {"best": None, "mean_top_k": None, "n": 0, "n_top_k": 0}
    direction = (spec.direction or "higher").lower()
    if direction == "higher":
        values_sorted = sorted(values, reverse=True)
    else:
        values_sorted = sorted(values)
    top_k = values_sorted[:k]
    best = top_k[0] if top_k else None
    mean_top_k = float(np.mean(top_k)) if top_k else None
    return {
        "best": best,
        "mean_top_k": mean_top_k,
        "n": len(values),
        "n_top_k": len(top_k),
    }


def property_preservation_mad(
    seeds: Sequence[str],
    gens: Sequence[str],
    specs: Sequence[PropertySpec],
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
) -> Dict[str, Any]:
    """
    Property preservation: MAD(prop) = mean(|prop(gen_i) - prop(seed_i)|) per property.
    Measures how well non-target properties are preserved when optimizing a target.
    Returns preservation_mad/{name} for each spec and preservation_mad_macro_mean.
    """
    out: Dict[str, Any] = {}
    mad_list: List[float] = []
    for spec in specs:
        deltas = _valid_numeric_deltas(seeds, gens, spec, sanity_fn=sanity_fn)
        if not deltas:
            out[f"preservation_mad/{spec.name}"] = None
            continue
        abs_deltas = [abs(d) for d in deltas]
        mad = float(np.mean(abs_deltas))
        out[f"preservation_mad/{spec.name}"] = mad
        mad_list.append(mad)
    out["preservation_mad_macro_mean"] = float(np.mean(mad_list)) if mad_list else None
    return out


def property_delta_bin_summary(
    seeds: Sequence[str],
    gens: Sequence[str],
    spec: PropertySpec,
    delta_binning: PropertyBinningConfig,
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
) -> Dict[str, Any]:
    """
    Binned summary of |delta| for valid (seed, gen) pairs.
    Returns mean_bin, median_bin, n (and optionally bin_counts).
    """
    deltas = _valid_numeric_deltas(seeds, gens, spec, sanity_fn=sanity_fn)
    if not deltas:
        return {"mean_bin": None, "median_bin": None, "n": 0}
    bins_list = [delta_to_bin(binning=delta_binning, delta=d, magnitude=True) for d in deltas]
    return {
        "mean_bin": float(np.mean(bins_list)),
        "median_bin": float(np.median(bins_list)),
        "n": len(bins_list),
    }


# ---------------------------------------------------------------------------
# Binning (reused from lobster: PropertyBinningConfig, value_to_bin, delta_to_bin, bin_to_range)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyBinningConfig:
    """
    Binning for property value or delta magnitude.
    mode='fixed' with range gives equal-width bins. clip=True maps outside range to bin 0 / last.
    """

    num_bins: int
    range: Optional[Tuple[float, float]] = None
    mode: str = "fixed"
    clip: bool = True


def _validate_bins(num_bins: int, bin_idx: int) -> int:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if bin_idx < 0 or bin_idx >= num_bins:
        raise ValueError(f"bin_idx out of range [0,{num_bins-1}], got {bin_idx}")
    return int(bin_idx)


def _validate_range(rng: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = float(rng[0]), float(rng[1])
    if not (hi > lo):
        raise ValueError(f"range must satisfy hi>lo, got {rng!r}")
    return lo, hi


def bin_to_range(*, binning: PropertyBinningConfig, bin_idx: int) -> Tuple[float, float]:
    """Convert bin index to numeric interval [lo, hi); last bin is [lo, hi]."""
    _validate_bins(binning.num_bins, bin_idx)
    if binning.range is None:
        raise ValueError("binning.range required for bin_to_range()")
    lo, hi = _validate_range(binning.range)
    width = (hi - lo) / float(binning.num_bins)
    b0 = lo + width * float(bin_idx)
    b1 = lo + width * float(bin_idx + 1)
    if bin_idx == binning.num_bins - 1:
        b1 = hi
    return float(b0), float(b1)


def value_to_bin(*, binning: PropertyBinningConfig, value: float) -> int:
    """Map numeric value to bin index in [0, num_bins-1]. Uses clip if binning.clip."""
    if binning.range is None:
        raise ValueError("binning.range required for value_to_bin()")
    lo, hi = _validate_range(binning.range)
    x = float(value)
    if binning.clip:
        if x <= lo:
            return 0
        if x >= hi:
            return binning.num_bins - 1
    else:
        if x < lo or x > hi:
            raise ValueError(f"value {x} out of range {binning.range}")
    width = (hi - lo) / float(binning.num_bins)
    idx = int((x - lo) / width)
    if idx < 0:
        idx = 0
    if idx >= binning.num_bins:
        idx = binning.num_bins - 1
    return idx


def delta_to_bin(*, binning: PropertyBinningConfig, delta: float, magnitude: bool = True) -> int:
    """Map delta to bin index; by default uses |delta| (magnitude)."""
    d = abs(float(delta)) if magnitude else float(delta)
    return value_to_bin(binning=binning, value=d)


# Default binning cutoffs (from lobster: zinc_all_props / visualize_property_prompting)
NUM_VALUE_BINS = 32
NUM_DELTA_BINS = 16

VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "logp": (-2.0, 6.0),
    "qed": (0.0, 1.0),
    "sa": (1.0, 10.0),
    "mw": (100.0, 700.0),
    "tpsa": (0.0, 200.0),
    "hbd": (0.0, 10.0),
    "hba": (0.0, 20.0),
    "rotatable_bonds": (0.0, 20.0),
    "frac_sp3_c": (0.0, 1.0),
    "Class": (0.0, 1.0),
    "p_np": (0.0, 1.0),
    "HIV_active": (0.0, 1.0),
}

DELTA_RANGES: Dict[str, Tuple[float, float]] = {
    "logp": (0.0, 3.0),
    "qed": (0.0, 0.5),
    "sa": (0.0, 5.0),
    "mw": (0.0, 300.0),
    "tpsa": (0.0, 100.0),
    "hbd": (0.0, 10.0),
    "hba": (0.0, 20.0),
    "rotatable_bonds": (0.0, 20.0),
    "frac_sp3_c": (0.0, 1.0),
    "Class": (0.0, 1.0),
    "p_np": (0.0, 1.0),
    "HIV_active": (0.0, 1.0),
}


def get_default_value_binning(property_name: str, num_bins: int = NUM_VALUE_BINS) -> PropertyBinningConfig:
    """Default value binning for a property (from VALUE_RANGES)."""
    rng = VALUE_RANGES.get(property_name)
    if rng is None:
        raise ValueError(f"No default value range for '{property_name}'. Known: {list(VALUE_RANGES)}")
    return PropertyBinningConfig(num_bins=num_bins, range=rng, mode="fixed", clip=True)


def get_default_delta_binning(property_name: str, num_bins: int = NUM_DELTA_BINS) -> PropertyBinningConfig:
    """Default delta magnitude binning for a property (from DELTA_RANGES)."""
    rng = DELTA_RANGES.get(property_name)
    if rng is None:
        raise ValueError(f"No default delta range for '{property_name}'. Known: {list(DELTA_RANGES)}")
    return PropertyBinningConfig(num_bins=num_bins, range=rng, mode="fixed", clip=True)
