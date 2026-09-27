# molecule_metrics.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import random

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, QED, Descriptors, Crippen, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

# ----------- Re-exports from pairit.eval (backward compatibility) -----------
from pairit.eval import PropertySpec  # noqa: F401
from pairit.eval import compute_properties  # noqa: F401
from pairit.eval import is_valid_smiles  # noqa: F401
from pairit.eval import property_improvement_stats  # noqa: F401
from pairit.eval import internal_diversity  # noqa: F401
from pairit.eval import mean_similarity_to_seeds  # noqa: F401
from pairit.eval import (
    compute_distribution_metrics as _pe_compute_distribution_metrics,
    property_preservation_mad as _pe_property_preservation_mad,
)


# ----------- Backward-compat wrappers for removed functions ---------------
# These wrap pairit_eval to match the old pairit return signatures so that
# legacy callers (e.g. old_molecule_validation_callback) don't break.


def snn_to_reference(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    return_list: bool = False,
    **_kwargs,
) -> Dict[str, Optional[float]]:
    """Backward-compat wrapper around ``pairit_eval.compute_distribution_metrics``."""
    dist = _pe_compute_distribution_metrics(
        generated_smiles, reference_smiles, min_generated=1
    )
    return {
        "mean_snn": dist.get("snn_mean"),
        "median_snn": dist.get("snn_median"),
        "n_gen_used": len(generated_smiles),
        "n_ref_used": len(reference_smiles),
        "nn_list": None,
    }


def coverage_heavy_atom_types(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    **_kwargs,
) -> Dict[str, object]:
    """Backward-compat wrapper around ``pairit_eval.compute_distribution_metrics``."""
    dist = _pe_compute_distribution_metrics(
        generated_smiles, reference_smiles, min_generated=1
    )
    return {"ratio": dist.get("coverage_ratio")}


def fragment_and_scaffold_similarity(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    **_kwargs,
) -> Dict[str, object]:
    """Backward-compat wrapper around ``pairit_eval.compute_distribution_metrics``."""
    dist = _pe_compute_distribution_metrics(
        generated_smiles, reference_smiles, min_generated=1
    )
    return {
        "frag_cosine": dist.get("fragment_cosine"),
        "scaf_cosine": dist.get("scaffold_cosine"),
    }


def w1_and_kde_marginals_between_sets(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
) -> Dict:
    """Backward-compat wrapper around ``pairit_eval.compute_distribution_metrics``."""
    dist = _pe_compute_distribution_metrics(
        generated_smiles, reference_smiles, min_generated=1
    )
    return {
        "_macro_mean_w1": dist.get("w1_macro_mean"),
        "_macro_mean_w1_norm": dist.get("w1_norm_macro_mean"),
        "_macro_mean_kde_ll_delta": dist.get("kde_ll_delta_macro_mean"),
    }


def fcd_chemnet_distance(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    min_required: int = 10,
    **_kwargs,
) -> Optional[float]:
    """Backward-compat wrapper around ``pairit_eval.compute_distribution_metrics``."""
    dist = _pe_compute_distribution_metrics(
        generated_smiles, reference_smiles, min_generated=min_required
    )
    return dist.get("fcd")


def property_preservation_mad(
    seeds: Sequence[str],
    gens: Sequence[str],
    specs,
    **kwargs,
) -> Dict:
    """Backward-compat wrapper that translates pairit_eval keys to old pairit keys."""
    result = _pe_property_preservation_mad(seeds, gens, specs, **kwargs)
    out: Dict = {}
    for spec in specs:
        per_mad = result.get(f"preservation_mad/{spec.name}")
        out[spec.name] = {"mad": per_mad, "n": 0}
    out["_macro_mean_mad"] = result.get("preservation_mad_macro_mean")
    return out




# ---------------------- basic utils ----------------------

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonical, isomeric SMILES or None if parsing fails."""
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def ecfp4_bits(smiles: str, nbits: int = 2048, radius: int = 2):
    """ECFP4 bit vector (or RDKit fingerprint object) or None."""
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None
    try:
        # modern generator
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=nbits)
        return gen.GetFingerprint(m)
    except Exception:
        # fallback (older RDKit)
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


# ---------------------- property machinery ----------------------
# PropertySpec and compute_properties are re-exported from pairit.eval above.


def is_improved(spec: PropertySpec, base: Optional[float], gen: Optional[float]) -> bool:
    if base is None or gen is None:
        return False
    delta = gen - base
    return (delta > spec.min_delta) if spec.direction == "higher" else ((-delta) > spec.min_delta)


# ---------------------- set-level metrics ----------------------

def validity_fraction(smiles_list: Sequence[str], sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None) -> float:
    vals = [is_valid_smiles(s, sanity_fn) for s in smiles_list]
    return float(np.mean(vals)) if len(vals) else 0.0


def uniqueness_fraction(smiles_list: Sequence[str], sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None) -> Tuple[float, int]:
    """Uniqueness among valid canonical SMILES; returns (fraction, num_valid)."""
    canon_valid = []
    for s in smiles_list:
        if is_valid_smiles(s, sanity_fn):
            c = canonicalize_smiles(s)
            if c:
                canon_valid.append(c)
    n = len(canon_valid)
    if n == 0:
        return 0.0, 0
    return float(len(set(canon_valid)) / n), n


def novelty_fraction(smiles_list: Sequence[str], training_canon: Set[str],
                     sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None) -> Tuple[float, int]:
    """Novelty among valid canonical SMILES wrt a canonicalized training set; returns (fraction, num_valid)."""
    canon_valid = []
    for s in smiles_list:
        if is_valid_smiles(s, sanity_fn):
            c = canonicalize_smiles(s)
            if c:
                canon_valid.append(c)
    n = len(canon_valid)
    if n == 0:
        return 0.0, 0
    num_novel = sum(1 for c in canon_valid if c not in training_canon)
    return float(num_novel / n), n


# ---------------------- neighborhood similarity summaries ----------------------

def tanimoto_similarity_smiles(a: str, b: str, nbits: int = 2048, radius: int = 2) -> Optional[float]:
    fpa = ecfp4_bits(a, nbits=nbits, radius=radius)
    fpb = ecfp4_bits(b, nbits=nbits, radius=radius)
    if fpa is None or fpb is None:
        return None
    return float(TanimotoSimilarity(fpa, fpb))


def neighborhood_similarity_summaries(
    gens: Sequence[str],
    neigh_lists: Sequence[Sequence[str]],
    similarity_fn: Optional[Callable[[str, str], Optional[float]]] = None,
) -> Dict[str, List[Optional[float]]]:
    """
    Summarize similarities {sim(g, n): n in neighborhood} with median, p75, and nearest (max).
    """
    if similarity_fn is None:
        similarity_fn = tanimoto_similarity_smiles

    med, p75, nn = [], [], []
    for g, neighs in zip(gens, neigh_lists):
        sims: List[float] = []
        if g:
            for n in (neighs or []):
                if not n:
                    continue
                try:
                    s = similarity_fn(g, n)
                    if s is not None and np.isfinite(s):
                        sims.append(float(s))
                except Exception:
                    pass
        if sims:
            arr = np.asarray(sims, float)
            med.append(float(np.median(arr)))
            p75.append(float(np.percentile(arr, 75)))
            nn.append(float(np.max(arr)))
        else:
            med.append(None)
            p75.append(None)
            nn.append(None)
    return {"median": med, "p75": p75, "nearest": nn}


# ---------------------- per-sample property deltas ----------------------

def property_deltas(seeds: Sequence[str], gens: Sequence[str], specs: Sequence[PropertySpec]) -> List[Dict[str, Optional[float]]]:
    out: List[Dict[str, Optional[float]]] = []
    for s, g in zip(seeds, gens):
        ps, pg = compute_properties(s, specs), compute_properties(g, specs)
        out.append({
            sp.name: None if (ps[sp.name] is None or pg[sp.name] is None) else (pg[sp.name] - ps[sp.name])
            for sp in specs
        })
    return out


def fraction_valid_and_improved(
    seeds: Sequence[str], gens: Sequence[str], specs: Sequence[PropertySpec],
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None
) -> Tuple[Dict[str, float], float]:
    """Fractions per property (and any) where gen is valid, passes sanity, and improves beyond spec.min_delta."""
    counts = {sp.name: 0 for sp in specs}
    any_count = 0
    total = len(gens)
    for s, g in zip(seeds, gens):
        if not (is_valid_smiles(g, sanity_fn) and is_valid_smiles(s, sanity_fn)):
            continue
        ms, mg = Chem.MolFromSmiles(s), Chem.MolFromSmiles(g)
        per_any = False
        for sp in specs:
            try:
                bs = sp.fn(ms); bg = sp.fn(mg)
            except Exception:
                bs, bg = None, None
            if is_improved(sp, bs, bg):
                counts[sp.name] += 1
                per_any = True
        if per_any:
            any_count += 1
    fracs = {k: (counts[k] / float(total) if total else 0.0) for k in counts}
    return fracs, (any_count / float(total) if total else 0.0)

# --- Top-k best property ------------------------------------------------------

def topk_best_property(
    smiles_list: Sequence[str],
    spec: PropertySpec,
    k: int = 3,
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
    deduplicate: bool = False,
    return_smiles: bool = True,
) -> Dict[str, Any]:
    """
    Compute `spec.fn` for each (valid) molecule and return the top-k according to `spec.direction`.

    Returns:
      {
        'top_values': [float, ...],             # best first
        'top_smiles': [str, ...] | None,        # aligned with top_values if return_smiles=True
        'k_effective': int,                     # min(k, num_scored)
        'num_scored': int,                      # valid molecules with a numeric score
        'num_input': int,                       # len(smiles_list)
        'best_value': float | None,
        'mean_topk': float | None
      }
    """
    # 1) filter to valid (and optionally dedup canonical SMILES)
    usable: List[str] = []
    if deduplicate:
        seen = set()
        for s in smiles_list:
            if not is_valid_smiles(s, sanity_fn):
                continue
            c = canonicalize_smiles(s)
            if c and c not in seen:
                seen.add(c)
                usable.append(c)
    else:
        for s in smiles_list:
            if is_valid_smiles(s, sanity_fn):
                usable.append(s)

    # 2) score each molecule using the provided PropertySpec
    scored: List[Tuple[float, str]] = []
    for s in usable:
        try:
            m = Chem.MolFromSmiles(s)
            v = spec.fn(m) if m is not None else None
            if v is None:
                continue
            v = float(v)
            if np.isnan(v) or np.isinf(v):
                continue
            scored.append((v, s))
        except Exception:
            continue

    if not scored:
        return {
            "top_values": [],
            "top_smiles": [] if return_smiles else None,
            "k_effective": 0,
            "num_scored": 0,
            "num_input": len(smiles_list),
            "best_value": None,
            "mean_topk": None,
        }

    # 3) sort by direction (higher is better vs lower is better)
    reverse = True if spec.direction == "higher" else False
    scored.sort(key=lambda t: t[0], reverse=reverse)

    # 4) slice top-k and package results
    top = scored[:k]
    top_vals = [v for v, _ in top]
    return {
        "top_values": top_vals,
        "top_smiles": [s for _, s in top] if return_smiles else None,
        "k_effective": len(top),
        "num_scored": len(scored),
        "num_input": len(smiles_list),
        "best_value": top_vals[0] if top_vals else None,
        "mean_topk": float(np.mean(top_vals)) if top_vals else None,
    }


def topk_best_property_from_fn(
    smiles_list: Sequence[str],
    fn: Callable[[Chem.Mol], Optional[float]],
    direction: str = "higher",
    k: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience wrapper that builds a PropertySpec on the fly."""
    spec = PropertySpec(name="custom", fn=fn, direction=direction)
    return topk_best_property(smiles_list, spec, k=k, **kwargs)


# ---------------------- simple RDKit feature summaries (W1 + KDE) ----------------------

FEATURE_NAMES = (
    "num_rings", "num_aromatic_rings", "num_heterocycles",
    "heavy_atom_count", "frac_sp3_c", "hba", "hbd", "tpsa", "logp"
)


def feature_vector(smiles: str) -> Optional[Dict[str, float]]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        ring_info = m.GetRingInfo()
        ring_atom_cycles = ring_info.AtomRings()
        num_hetero = sum(
            1 for cyc in ring_atom_cycles
            if any(m.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in cyc)
        )
        return {
            "num_rings": float(rdMolDescriptors.CalcNumRings(m)),
            "num_aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(m)),
            "num_heterocycles": float(num_hetero),
            "heavy_atom_count": float(Descriptors.HeavyAtomCount(m)),
            "frac_sp3_c": float(Descriptors.FractionCSP3(m)),
            "hba": float(rdMolDescriptors.CalcNumHBA(m)),
            "hbd": float(rdMolDescriptors.CalcNumHBD(m)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
            "logp": float(Crippen.MolLogP(m)),
        }
    except Exception:
        return None


def _kde_logprob_1d(values: np.ndarray, x: float) -> float:
    n = values.size
    if n == 0:
        return -1e9
    std = float(np.std(values)) if n > 1 else 0.0
    iqr = float(np.subtract(*np.percentile(values, [75, 25]))) if n > 1 else 0.0
    sigma = max(std, iqr / 1.349, 1e-3)
    h = 1.06 * sigma * (n ** (-1 / 5)) if n > 1 else 0.2
    z = (x - values) / h
    log_k = -0.5 * (z ** 2) - 0.5 * np.log(2.0 * np.pi) - np.log(h)
    m = np.max(log_k)
    return float(m + np.log(np.sum(np.exp(log_k - m))) - np.log(n))


def _wasserstein_1d_exact(x: np.ndarray, y: np.ndarray) -> float:
    """Exact W1 between two 1D empirical distributions with equal weights."""
    x = np.sort(np.asarray(x, float)); y = np.sort(np.asarray(y, float))
    nx, ny = x.size, y.size
    i = j = 0
    cdf_x = cdf_y = 0.0
    z_cur = min(x[0], y[0])
    while i < nx and x[i] == z_cur:
        i += 1; cdf_x = i / nx
    while j < ny and y[j] == z_cur:
        j += 1; cdf_y = j / ny
    area = 0.0
    while i < nx or j < ny:
        next_x = x[i] if i < nx else np.inf
        next_y = y[j] if j < ny else np.inf
        z_next = next_x if next_x < next_y else next_y
        area += abs(cdf_x - cdf_y) * (z_next - z_cur)
        z_cur = z_next
        while i < nx and x[i] == z_cur:
            i += 1; cdf_x = i / nx
        while j < ny and y[j] == z_cur:
            j += 1; cdf_y = j / ny
    return float(area)


def _collect_feature_arrays(smiles_list: Sequence[str]) -> Dict[str, np.ndarray]:
    buckets = {k: [] for k in FEATURE_NAMES}
    for s in smiles_list:
        fv = feature_vector(s)
        if not fv:
            continue
        for k in FEATURE_NAMES:
            buckets[k].append(fv[k])
    return {k: np.asarray(v, float) for k, v in buckets.items() if len(v) > 0}


def neighborhood_alignment_proxy(
    neigh_lists: Sequence[Sequence[str]], seeds: Sequence[str], gens: Sequence[str]
) -> Tuple[List[Dict[str, float]], List[Optional[float]], List[Optional[float]]]:
    """For each sample: (per-feature ΔE|X−x|, KDE_ll_input, KDE_ll_gen). Δ<0 => gen closer to neighborhood."""
    deltas_list, ll_in_list, ll_gen_list = [], [], []
    for neighs, s, g in zip(neigh_lists, seeds, gens):
        feats = {k: [] for k in FEATURE_NAMES}
        for n in (neighs or []):
            fv = feature_vector(n)
            if fv:
                for k in FEATURE_NAMES:
                    feats[k].append(fv[k])
        fi = feature_vector(s) if s else None
        fg = feature_vector(g) if g else None
        if any(len(v) for v in feats.values()) and fi and fg:
            deltas: Dict[str, float] = {}
            ll_in = 0.0
            ll_gen = 0.0
            for k, arr in feats.items():
                arr = np.asarray(arr, float)
                if arr.size == 0:
                    continue
                deltas[k] = float(np.mean(np.abs(arr - fi[k])) - np.mean(np.abs(arr - fg[k])))
                ll_in += _kde_logprob_1d(arr, fi[k])
                ll_gen += _kde_logprob_1d(arr, fg[k])
            deltas_list.append(deltas)
            ll_in_list.append(ll_in)
            ll_gen_list.append(ll_gen)
        else:
            deltas_list.append({})
            ll_in_list.append(None)
            ll_gen_list.append(None)
    return deltas_list, ll_in_list, ll_gen_list


# ---------------------- seed-conditioned and reference metrics ----------------------

def similarities_to_seeds(
    seeds: Sequence[str],
    gens: Sequence[str],
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
    nbits: int = 2048,
    radius: int = 2,
) -> List[Optional[float]]:
    sims: List[Optional[float]] = []
    for s, g in zip(seeds, gens):
        if not (is_valid_smiles(s, sanity_fn) and is_valid_smiles(g, sanity_fn)):
            sims.append(None)
            continue
        sims.append(tanimoto_similarity_smiles(s, g, nbits=nbits, radius=radius))
    return sims


def mean_similarity_to_seeds(seeds: Sequence[str], gens: Sequence[str], **kwargs) -> Optional[float]:
    sims = [x for x in similarities_to_seeds(seeds, gens, **kwargs) if x is not None]
    return float(np.mean(sims)) if sims else None


def max_similarity_to_gt(
    gens: Sequence[str],
    seeds: Sequence[str],
    gt_lists: Sequence[Sequence[str]],
    *,
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
    nbits: int = 2048,
    radius: int = 2,
    similarity_fn: Optional[Callable[[str, str], Optional[float]]] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Per index i, return:
      gen_max_sim[i]  = max_t sim(gen_i,  t) over t in gt_lists[i]
      seed_max_sim[i] = max_t sim(seed_i, t)
      delta_sim[i]    = gen_max_sim[i] - seed_max_sim[i]
    """
    if similarity_fn is None:
        def similarity_fn(a: str, b: str) -> Optional[float]:
            return tanimoto_similarity_smiles(a, b, nbits=nbits, radius=radius)

    gen_max, seed_max, delta = [], [], []

    for g, s, gts in zip(gens, seeds, gt_lists):
        g_ok = (g is not None) and is_valid_smiles(g, sanity_fn)
        s_ok = (s is not None) and is_valid_smiles(s, sanity_fn)
        valid_gts = [t for t in (gts or []) if (t is not None and is_valid_smiles(t, sanity_fn))]

        gsims, ssims = [], []
        if valid_gts:
            if g_ok:
                for t in valid_gts:
                    try:
                        sv = similarity_fn(g, t)
                        if sv is not None and np.isfinite(sv): gsims.append(float(sv))
                    except Exception:
                        pass
            if s_ok:
                for t in valid_gts:
                    try:
                        sv = similarity_fn(s, t)
                        if sv is not None and np.isfinite(sv): ssims.append(float(sv))
                    except Exception:
                        pass

        gm = float(np.max(gsims)) if gsims else None
        sm = float(np.max(ssims)) if ssims else None
        gen_max.append(gm)
        seed_max.append(sm)
        delta.append(None if (gm is None or sm is None) else (gm - sm))

    return gen_max, seed_max, delta


def property_improvement_under_similarity_constraint(
    seeds: Sequence[str],
    gens: Sequence[str],
    spec: PropertySpec,
    *,
    sim_range: Tuple[float, float] = (0.4, 0.6),
    sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None,
    nbits: int = 2048,
    radius: int = 2,
    success_threshold: Optional[float] = None,
    success_direction: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Mean Δprop and (optional) success rate for pairs whose seed↔gen Tanimoto lies in a band."""
    lo, hi = sim_range
    n_total = min(len(seeds), len(gens))
    n_valid = 0
    n_in_range = 0

    deltas: List[float] = []
    sims: List[float] = []
    success_flags: List[bool] = []

    dirn = (success_direction or spec.direction or "higher").lower()

    for s, g in zip(seeds, gens):
        if not (is_valid_smiles(s, sanity_fn) and is_valid_smiles(g, sanity_fn)):
            continue
        ms, mg = Chem.MolFromSmiles(s), Chem.MolFromSmiles(g)
        if ms is None or mg is None:
            continue

        try:
            vs = spec.fn(ms); vg = spec.fn(mg)
        except Exception:
            vs, vg = None, None
        if vs is None or vg is None:
            continue

        sim = tanimoto_similarity_smiles(s, g, nbits=nbits, radius=radius)
        if sim is None:
            continue
        n_valid += 1
        if sim < lo or sim > hi:
            continue
        n_in_range += 1

        d = float(vg) - float(vs)
        if not np.isfinite(d):
            continue
        deltas.append(d)
        sims.append(sim)

        if success_threshold is not None:
            success_flags.append(float(vg) <= float(success_threshold) if dirn == "lower"
                                 else float(vg) >= float(success_threshold))

    if not deltas:
        return {
            "mean_delta": None, "median_delta": None, "std_delta": None,
            "frac_improved": None,
            "success_rate": (None if success_threshold is None else 0.0),
            "mean_similarity": None,
            "n_total": n_total, "n_valid": n_valid, "n_in_range": n_in_range,
            "sim_min": lo, "sim_max": hi,
            "threshold": success_threshold, "direction": dirn,
        }

    improved_flags = [d > spec.min_delta for d in deltas] if spec.direction.lower() == "higher" \
        else [(-d) > spec.min_delta for d in deltas]

    return {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(np.std(deltas, ddof=0)),
        "frac_improved": float(np.mean(improved_flags)),
        "success_rate": (float(np.mean(success_flags)) if success_threshold is not None and success_flags
                         else (0.0 if success_threshold is not None else None)),
        "mean_similarity": float(np.mean(sims)) if sims else None,
        "n_total": n_total, "n_valid": n_valid, "n_in_range": n_in_range,
        "sim_min": lo, "sim_max": hi,
        "threshold": success_threshold, "direction": dirn,
    }


# ---------------------- smoke test (toy) ----------------------

def _fmt(x):
    return "None" if x is None else f"{x:.3f}"


try:
    from atomic_datasets.utils.rdkit import is_molecule_sane as _proj_is_molecule_sane
    is_molecule_sane = _proj_is_molecule_sane
except Exception:
    def is_molecule_sane(m: Chem.Mol) -> bool:
        try:
            Chem.SanitizeMol(m)
            return True
        except Exception:
            return False


try:
    import sascorer as _sascorer_mod
    _HAS_SA = True
except Exception:
    _sascorer_mod = None
    _HAS_SA = False


def penalized_logp_fn(m: Chem.Mol) -> Optional[float]:
    """JT-VAE style: logP - SA - cycle penalty (>6)."""
    try:
        lp = float(Crippen.MolLogP(m))
        sa = float(_sascorer_mod.calculateScore(m)) if _HAS_SA else 0.0
        rings = m.GetRingInfo().AtomRings()
        max_cycle = max((len(r) for r in rings), default=0)
        cycle_pen = max(0, max_cycle - 6)
        return lp - sa - cycle_pen
    except Exception:
        return None


if __name__ == "__main__":
    from pairit.eval import (
        compute_distribution_metrics as pe_compute_distribution_metrics,
        property_preservation_mad as pe_property_preservation_mad,
    )

    refs = ["c1ccccc1", "CCO", "CCN", "CCS", "CP"]
    gens = ["CCO", "CCCl", "CBr"]
    seeds = ["CCO", "CCN", "CCS"]

    print(f"[Validity] {validity_fraction(gens) * 100:.2f}% valid")
    print(f"[Sane] {validity_fraction(gens, sanity_fn=is_molecule_sane) * 100:.2f}% pass med-chem sanity")

    uniq_frac, uniq_nvalid = uniqueness_fraction(gens)
    print(f"[Uniqueness] {uniq_frac*100:.2f}% among {uniq_nvalid} valid")

    training_canon = {c for s in refs for c in [canonicalize_smiles(s)] if c}
    nov_frac, nov_nvalid = novelty_fraction(gens, training_canon)
    print(f"[Novelty] {nov_frac*100:.2f}% among {nov_nvalid} valid")

    sims = similarities_to_seeds(seeds, gens)
    ms = mean_similarity_to_seeds(seeds, gens)
    print(f"[Similarity->Seed] per-sample={[('None' if s is None else f'{s:.3f}') for s in sims]} mean={_fmt(ms)}")

    gt_lists = [["CCO", "CCOC", "CC=O"], ["CCN", "CCCN", "CNC"], ["CCS", "CCCS", "CSC"]]
    gen_max_sim, seed_max_sim, delta_sim = max_similarity_to_gt(gens, seeds, gt_lists)
    _fmtlist = lambda L: ['None' if v is None else f'{v:.3f}' for v in L]
    print("[Similarity->GT (NN)] gen:", _fmtlist(gen_max_sim), " | seed:", _fmtlist(seed_max_sim))

    dist = pe_compute_distribution_metrics(gens, refs, min_generated=2)
    print(f"[Distribution] snn_mean={_fmt(dist.get('snn_mean'))} fcd={_fmt(dist.get('fcd'))}")

    div, n_valid, _ = internal_diversity(gens, deduplicate=True)
    print(f"[InternalDiversity] diversity={_fmt(div)}, n_valid={n_valid}")

    logp_spec = PropertySpec("logp", lambda m: Crippen.MolLogP(m), direction="higher")
    out = topk_best_property(gens, logp_spec, k=3, deduplicate=True)
    print("[TopK logP] best=", _fmt(out["best_value"]), " mean_topk=", _fmt(out["mean_topk"]))

    pen_logp_spec = PropertySpec("p_logp", penalized_logp_fn, direction="higher", min_delta=0.0)
    imp = property_improvement_stats(seeds, gens, pen_logp_spec)
    print("[Improvement p_logP] mean_delta=", _fmt(imp["mean_delta"]), " frac_improved=", ("None" if imp["frac_improved"] is None else f"{imp['frac_improved']:.3f}"))

    qed_spec = PropertySpec("qed", lambda m: QED.qed(m), direction="higher")
    sa_spec = PropertySpec("sa", (lambda m: float(_sascorer_mod.calculateScore(m))) if _HAS_SA else (lambda m: None), direction="lower")
    mad = pe_property_preservation_mad(seeds, gens, [qed_spec, sa_spec])
    print("[Property Preservation MAD] qed=", _fmt(mad.get("preservation_mad/qed")), " sa=", _fmt(mad.get("preservation_mad/sa")))

    fracs, any_frac = fraction_valid_and_improved(seeds, gens, [pen_logp_spec, qed_spec], sanity_fn=is_molecule_sane)
    props_str = " ".join([f"{k}={v:.3f}" for k, v in fracs.items()])
    print(f"[Valid+Sane+Improved] {props_str}  any={any_frac:.3f}")

    res = property_improvement_under_similarity_constraint(seeds, gens, pen_logp_spec, sim_range=(0.4, 0.6), success_threshold=1.0)
    print("[Delta under sim] mean_delta=", _fmt(res["mean_delta"]), " n_in_range=", res["n_in_range"])
