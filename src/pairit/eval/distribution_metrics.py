"""
Distribution-level metrics (FCD, SNN, FragCos, ScafCos, Coverage, W1, KDE/LL).

All logic is in this module. Used only when compute_distribution_metrics=True in evaluate().
If a dependency is missing (e.g. scipy for FCD, or fcd/fcd_torch for ChemNet), that metric
is skipped and returned as None.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Helpers (validity, fingerprints, canonicalize)
# ---------------------------------------------------------------------------


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _is_valid_smiles(smiles: str, sanity_fn: Optional[Callable[[Chem.Mol], bool]] = None) -> bool:
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return False
        return sanity_fn(m) if sanity_fn else True
    except Exception:
        return False


def _ecfp4_bits(smiles: str, nbits: int = 2048, radius: int = 2):
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=nbits)
        return gen.GetFingerprint(m)
    except Exception:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


# ---------------------------------------------------------------------------
# W1 / KDE (feature vectors, wasserstein, kde logprob)
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "num_rings", "num_aromatic_rings", "num_heterocycles",
    "heavy_atom_count", "frac_sp3_c", "hba", "hbd", "tpsa", "logp"
)


def _feature_vector(smiles: str) -> Optional[Dict[str, float]]:
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
    x = np.sort(np.asarray(x, float))
    y = np.sort(np.asarray(y, float))
    nx, ny = x.size, y.size
    i = j = 0
    cdf_x = cdf_y = 0.0
    z_cur = min(x[0], y[0])
    while i < nx and x[i] == z_cur:
        i += 1
        cdf_x = i / nx
    while j < ny and y[j] == z_cur:
        j += 1
        cdf_y = j / ny
    area = 0.0
    while i < nx or j < ny:
        next_x = x[i] if i < nx else np.inf
        next_y = y[j] if j < ny else np.inf
        z_next = next_x if next_x < next_y else next_y
        area += abs(cdf_x - cdf_y) * (z_next - z_cur)
        z_cur = z_next
        while i < nx and x[i] == z_cur:
            i += 1
            cdf_x = i / nx
        while j < ny and y[j] == z_cur:
            j += 1
            cdf_y = j / ny
    return float(area)


def _collect_feature_arrays(smiles_list: Sequence[str]) -> Dict[str, np.ndarray]:
    buckets = {k: [] for k in FEATURE_NAMES}
    for s in smiles_list:
        fv = _feature_vector(s)
        if not fv:
            continue
        for k in FEATURE_NAMES:
            buckets[k].append(fv[k])
    return {k: np.asarray(v, float) for k, v in buckets.items() if len(v) > 0}


def _w1_and_kde_marginals_between_sets(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
) -> Dict[str, Any]:
    ref = _collect_feature_arrays(reference_smiles)
    gen = _collect_feature_arrays(generated_smiles)
    per: Dict[str, Dict[str, Optional[float]]] = {}
    w1_list, w1n_list, dll_list = [], [], []
    result: Dict[str, Any] = {
        "_macro_mean_w1": None,
        "_macro_mean_w1_norm": None,
        "_macro_mean_kde_ll_delta": None,
    }

    for k in FEATURE_NAMES:
        xr = ref.get(k)
        xg = gen.get(k)
        if xr is None or xg is None or xr.size == 0 or xg.size == 0:
            continue
        w1 = _wasserstein_1d_exact(xg, xr)
        q75, q25 = np.percentile(xr, [75, 25])
        iqr = float(q75 - q25)
        sd = float(np.std(xr))
        denom = iqr if iqr > 0 else (sd if sd > 0 else 1.0)
        w1_norm = w1 / denom
        ll_gen = float(np.mean([_kde_logprob_1d(xr, v) for v in xg])) if xr.size >= 2 else None
        ll_ref = float(np.mean([_kde_logprob_1d(xr, v) for v in xr])) if xr.size >= 2 else None
        ll_delta = None if (ll_gen is None or ll_ref is None) else (ll_gen - ll_ref)
        w1_list.append(w1)
        w1n_list.append(w1_norm)
        if ll_delta is not None:
            dll_list.append(ll_delta)
        result[f"w1_delta_conditional/{k}"] = float(w1)
        result[f"kde_ll_in/{k}"] = ll_ref
        result[f"kde_ll_gen/{k}"] = ll_gen
        result[f"kde_ll_delta/{k}"] = ll_delta

    result["_macro_mean_w1"] = float(np.mean(w1_list)) if w1_list else None
    result["_macro_mean_w1_norm"] = float(np.mean(w1n_list)) if w1n_list else None
    result["_macro_mean_kde_ll_delta"] = float(np.mean(dll_list)) if dll_list else None
    return result


# ---------------------------------------------------------------------------
# Coverage (heavy-atom types)
# ---------------------------------------------------------------------------


def _heavy_atom_types_from_smiles(smiles: str) -> Optional[Set[str]]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        return {atom.GetSymbol() for atom in m.GetAtoms() if atom.GetAtomicNum() > 1}
    except Exception:
        return None


def _heavy_atom_type_set(smiles_list: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for s in smiles_list:
        if not _is_valid_smiles(s):
            continue
        types = _heavy_atom_types_from_smiles(s)
        if types:
            out |= types
    return out


def _coverage_heavy_atom_types(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
) -> Dict[str, Any]:
    ref_types = _heavy_atom_type_set(reference_smiles)
    gen_types = _heavy_atom_type_set(generated_smiles)
    hit = gen_types & ref_types
    denom = max(1, len(ref_types))
    ratio = float(len(hit) / denom)
    return {
        "ratio": ratio,
        "n_hit": len(hit),
        "n_ref_types": len(ref_types),
        "n_gen_types": len(gen_types),
        "hit_over_expected": ratio,
    }


# ---------------------------------------------------------------------------
# Fragment / Scaffold similarity (BRICS, Murcko, cosine)
# ---------------------------------------------------------------------------

_BRICS_ATTACH_RE = re.compile(r"\[\d\*\]")


def _brics_fragments_for_smiles(smiles: str, generic_attachment: bool = True) -> Optional[Set[str]]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        frags = BRICS.BRICSDecompose(m)
        out: Set[str] = set()
        for f in frags:
            if generic_attachment:
                f = _BRICS_ATTACH_RE.sub("[*]", f)
            c = _canonicalize_smiles(f)
            if c:
                out.add(c)
        return out
    except Exception:
        return None


def _murcko_scaffold_for_smiles(smiles: str, genericize: bool = True) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is None:
            return None
        if genericize:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _frequency_counter(
    smiles_list: Sequence[str],
    extractor: Callable[[str], Optional[Set[str]]],
) -> Counter[str]:
    C: Counter[str] = Counter()
    for s in smiles_list:
        if not _is_valid_smiles(s):
            continue
        tok = extractor(s)
        if tok is None:
            continue
        C.update(tok)
    return C


def _cosine_from_counters(a: Counter[str], b: Counter[str]) -> Optional[float]:
    if not a or not b:
        return None
    keys = set(a.keys()) | set(b.keys())
    dot = na = nb = 0.0
    for k in keys:
        va = float(a.get(k, 0))
        vb = float(b.get(k, 0))
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na ** 0.5 * nb ** 0.5)


def _fragment_and_scaffold_similarity(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
) -> Dict[str, Optional[float]]:
    def frag_extractor(s: str) -> Optional[Set[str]]:
        return _brics_fragments_for_smiles(s, generic_attachment=True)

    def scaf_set(s: str) -> Set[str]:
        v = _murcko_scaffold_for_smiles(s, genericize=True)
        return {v} if v else set()

    frag_gen = _frequency_counter(generated_smiles, frag_extractor)
    frag_ref = _frequency_counter(reference_smiles, frag_extractor)
    scaf_gen = _frequency_counter(generated_smiles, scaf_set)
    scaf_ref = _frequency_counter(reference_smiles, scaf_set)
    frag_cos = _cosine_from_counters(frag_gen, frag_ref)
    scaf_cos = _cosine_from_counters(scaf_gen, scaf_ref)
    return {
        "frag_cosine": float(frag_cos) if frag_cos is not None else None,
        "scaf_cosine": float(scaf_cos) if scaf_cos is not None else None,
    }


# ---------------------------------------------------------------------------
# FCD (Fréchet ChemNet Distance) — scipy and embedder optional
# ---------------------------------------------------------------------------


def _frechet_distance_gaussians(
    mu1: np.ndarray, cov1: np.ndarray,
    mu2: np.ndarray, cov2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    try:
        from scipy.linalg import sqrtm
    except ImportError:
        return float("nan")
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=float))
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)
    d = mu1.shape[0]
    cov1 = cov1 + eps * np.eye(d, dtype=float)
    cov2 = cov2 + eps * np.eye(d, dtype=float)
    diff = mu1 - mu2
    m = float(np.dot(diff, diff))
    cov_prod_sqrt = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real
    tr = float(np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(cov_prod_sqrt))
    return m + tr


def _default_chemnet_embedder() -> Optional[Callable[[Sequence[str]], np.ndarray]]:
    try:
        from fcd_torch import FCD
        _fcd_model = FCD(device="cpu", n_jobs=0, batch_size=512)
        def _embed(smiles_list):
            return _fcd_model.get_predictions(smiles_list)
        return _embed
    except Exception:
        pass
    try:
        from fcd import get_predictions as keras_or_torch_get_predictions
        def _embed(smiles_list):
            return keras_or_torch_get_predictions(smiles_list)
        return _embed
    except Exception:
        pass
    return None


def _fcd_chemnet_distance(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    min_required: int = 10,
    ridge: float = 1e-6,
) -> Optional[float]:
    emb_fn = _default_chemnet_embedder()
    if emb_fn is None:
        return None
    gens = [s for s in generated_smiles if _is_valid_smiles(s)]
    refs = [s for s in reference_smiles if _is_valid_smiles(s)]
    try:
        E_g = np.asarray(emb_fn(gens))
        E_r = np.asarray(emb_fn(refs))
    except Exception:
        return None
    if E_g.ndim != 2 or E_r.ndim != 2 or E_g.shape[1] != E_r.shape[1]:
        return None
    if len(E_g) < min_required or len(E_r) < min_required:
        return None
    mu_g = E_g.mean(axis=0)
    mu_r = E_r.mean(axis=0)
    Xg = E_g - mu_g
    Xr = E_r - mu_r
    cov_g = (Xg.T @ Xg) / max(1, (E_g.shape[0] - 1))
    cov_r = (Xr.T @ Xr) / max(1, (E_r.shape[0] - 1))
    return _frechet_distance_gaussians(mu_g, cov_g, mu_r, cov_r, eps=ridge)


# ---------------------------------------------------------------------------
# SNN (single nearest neighbor to reference)
# ---------------------------------------------------------------------------


def _snn_to_reference(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    nbits: int = 2048,
    radius: int = 2,
) -> Dict[str, Optional[float]]:
    """Mean/median nearest-neighbor Tanimoto from each valid generated molecule
    to any molecule in the reference set.

    Also returns novelty-filtered versions (`mean_snn_novel`, `median_snn_novel`)
    that skip generated molecules whose canonical SMILES appears in the reference
    set (i.e., memorized). Removes the memorization inflation where a
    memorized gen contributes Tanimoto=1.0 (self-match) to the average.
    """
    ref_seen: Set[str] = set()
    ref_fps: List[Any] = []
    for s in reference_smiles:
        if not _is_valid_smiles(s):
            continue
        c = _canonicalize_smiles(s)
        if not c or c in ref_seen:
            continue
        ref_seen.add(c)
        fp = _ecfp4_bits(c, nbits=nbits, radius=radius)
        if fp is not None:
            ref_fps.append(fp)
    if len(ref_fps) == 0:
        return {
            "mean_snn": None, "median_snn": None,
            "mean_snn_novel": None, "median_snn_novel": None,
        }

    nn_list: List[float] = []
    nn_list_novel: List[float] = []
    for s in generated_smiles:
        if not _is_valid_smiles(s):
            continue
        fp = _ecfp4_bits(s, nbits=nbits, radius=radius)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        if not sims:
            continue
        max_sim = float(max(sims))
        nn_list.append(max_sim)
        # Novelty filter: skip if canonical SMILES matches any reference
        c = _canonicalize_smiles(s)
        if c and c not in ref_seen:
            nn_list_novel.append(max_sim)
    if not nn_list:
        return {
            "mean_snn": None, "median_snn": None,
            "mean_snn_novel": None, "median_snn_novel": None,
        }
    out = {
        "mean_snn": float(np.mean(nn_list)),
        "median_snn": float(np.median(nn_list)),
        "mean_snn_novel": float(np.mean(nn_list_novel)) if nn_list_novel else None,
        "median_snn_novel": float(np.median(nn_list_novel)) if nn_list_novel else None,
    }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Always True: distribution metrics are implemented in this module. Individual metrics may still be None if deps (e.g. scipy, fcd) are missing."""
    return True


def compute_distribution_metrics(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    min_generated: int = 10,
    return_diagnostics: bool = False,
) -> Dict[str, Any]:
    """
    Compute FCD, SNN, fragment/scaffold cosine, coverage, W1, KDE LL-delta.
    All logic is in this module. Missing deps (scipy, fcd) cause that metric to be None.

    If return_diagnostics is True, the returned dict includes a "diagnostics" key with
    n_gen_used, n_ref_used, and coverage diagnostics (n_hit, n_ref_types, hit_over_expected).
    """
    out: Dict[str, Any] = {
        "fcd": None,
        "snn_mean": None,
        "snn_median": None,
        # Novelty-filtered SNN: excludes memorized gens (canonical SMILES in
        # reference set) — removes Tanimoto=1.0 self-match inflation.
        "snn_mean_novel": None,
        "snn_median_novel": None,
        "fragment_cosine": None,
        "scaffold_cosine": None,
        "coverage_ratio": None,
        "w1_macro_mean": None,
        "w1_norm_macro_mean": None,
        "kde_ll_delta_macro_mean": None,
    }
    gen_list = [s for s in generated_smiles if s and str(s).strip()]
    gen_list = [s for s in gen_list if _is_valid_smiles(s)]
    if len(gen_list) < min_generated:
        if return_diagnostics:
            out["diagnostics"] = {"n_gen_used": len(gen_list), "n_ref_used": 0}
        return out
    ref_list = [s for s in reference_smiles if s and str(s).strip()]
    if not ref_list:
        if return_diagnostics:
            out["diagnostics"] = {"n_gen_used": len(gen_list), "n_ref_used": 0}
        return out

    diagnostics: Dict[str, Any] = {
        "n_gen_used": len(gen_list),
        "n_ref_used": len(ref_list),
    }

    try:
        fcd = _fcd_chemnet_distance(gen_list, ref_list, min_required=min_generated)
        out["fcd"] = float(fcd) if fcd is not None and np.isfinite(fcd) else None
    except Exception:
        pass

    try:
        snn = _snn_to_reference(gen_list, ref_list)
        for key in ("mean_snn", "median_snn", "mean_snn_novel", "median_snn_novel"):
            v = snn.get(key)
            # Output naming: mean_snn -> snn_mean; median_snn -> snn_median (etc).
            out_key = key.replace("mean_snn", "snn_mean").replace("median_snn", "snn_median")
            out[out_key] = float(v) if v is not None else None
    except Exception:
        pass

    try:
        fs = _fragment_and_scaffold_similarity(gen_list, ref_list)
        out["fragment_cosine"] = fs.get("frag_cosine")
        out["scaffold_cosine"] = fs.get("scaf_cosine")
    except Exception:
        pass

    try:
        cov = _coverage_heavy_atom_types(gen_list, ref_list)
        out["coverage_ratio"] = cov.get("ratio")
        if return_diagnostics:
            diagnostics["coverage_n_hit"] = cov.get("n_hit")
            diagnostics["coverage_n_ref_types"] = cov.get("n_ref_types")
            diagnostics["coverage_n_gen_types"] = cov.get("n_gen_types")
            diagnostics["hit_over_expected"] = cov.get("hit_over_expected")
    except Exception:
        pass

    try:
        w1kde = _w1_and_kde_marginals_between_sets(gen_list, ref_list)
        out["w1_macro_mean"] = w1kde.get("_macro_mean_w1")
        out["w1_norm_macro_mean"] = w1kde.get("_macro_mean_w1_norm")
        out["kde_ll_delta_macro_mean"] = w1kde.get("_macro_mean_kde_ll_delta")
        for k in FEATURE_NAMES:
            for suffix in ("w1_delta_conditional", "kde_ll_in", "kde_ll_gen", "kde_ll_delta"):
                key = f"{suffix}/{k}"
                if key in w1kde and w1kde[key] is not None:
                    out[key] = w1kde[key]
    except Exception:
        pass

    if return_diagnostics:
        out["diagnostics"] = diagnostics

    return out
