"""
Canonical method naming and dataset coverage for evaluation pipelines.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


CANONICAL_METHODS: List[str] = [
    "nfbo",
    "ddsbm-bridge",
    "ddsbm-sb",
    "mmp",
    "mmp-prop",
    "pairit-from-zinc",
    "pairit-real-only",
    # "hierg2g-04",  # kept for reference/support, but excluded from default comparisons
    "hierg2g-06",
    # Nearest-neighbor lookup baselines
    "nn-real-only",
    "nn-real-only-zinc",
]


METHOD_ALIASES: Dict[str, str] = {
    # NN lookup aliases
    "nn_real_only": "nn-real-only",
    "nn_real_only_zinc": "nn-real-only-zinc",
    # DDSBM legacy CLI names
    "ddsbm_bridge": "ddsbm-bridge",
    "ddsbm_sb": "ddsbm-sb",
    # MMP variants
    "mmp_prop": "mmp-prop",
    "mmp-prop": "mmp-prop",
    # Pairit variant aliases
    "pairit-from_zinc": "pairit-from-zinc",
    "pairit_from_zinc": "pairit-from-zinc",
    "pairit-real_only": "pairit-real-only",
    "pairit_real_only": "pairit-real-only",
    # HierG2G aliases
    "hierg2g04": "hierg2g-04",
    "hierg2g_04": "hierg2g-04",
    "hierg2g-4": "hierg2g-04",
    "hierg2g06": "hierg2g-06",
    "hierg2g_06": "hierg2g-06",
    "hierg2g-6": "hierg2g-06",
}


METHOD_DATASETS: Dict[str, List[str]] = {
    "nfbo": ["bace", "bbbp", "hiv", "zinc250k"],
    "ddsbm-bridge": ["bace", "bbbp", "hiv", "zinc250k"],
    "ddsbm-sb": ["bace", "bbbp", "hiv", "zinc250k"],
    "mmp": ["bace", "bbbp", "hiv"],
    "mmp-prop": ["bace", "bbbp", "hiv"],
    "pairit-from-zinc": ["bace", "bbbp", "hiv"],
    "pairit-real-only": ["bace", "bbbp", "hiv"],
    "hierg2g-04": ["bace", "bbbp", "hiv", "zinc250k"],
    "hierg2g-06": ["bace", "bbbp", "hiv", "zinc250k"],
    # NN lookup baseline — real-only sources
    "nn-real-only": ["bace", "bbbp", "hiv", "zinc250k"],
    "nn-real-only-zinc": ["bace", "bbbp", "hiv"],
}


def normalize_method_name(method: str) -> str:
    key = str(method).strip()
    return METHOD_ALIASES.get(key, key)


def ensure_supported_method(method: str) -> str:
    canonical = normalize_method_name(method)
    if canonical not in METHOD_DATASETS:
        supported = ", ".join(sorted(METHOD_DATASETS))
        raise ValueError(f"Unsupported method '{method}' (canonical '{canonical}'). Supported: {supported}")
    return canonical


def resolve_methods(methods: Sequence[str] | None) -> List[str]:
    if not methods:
        return list(CANONICAL_METHODS)
    out: List[str] = []
    seen = set()
    for m in methods:
        canonical = ensure_supported_method(m)
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out
