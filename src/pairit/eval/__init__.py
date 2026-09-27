"""
pairit_eval: Stage 1 = compute metrics on generations; Stage 2 = compare methods (later).

Use on a single method's CSV/DataFrame or on all methods concatenated.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .config import PairitEvalConfig, get_default_generation_paths
from .oracles import (
    CallableOracle,
    FingerprintJoblibOracle,
    Oracle,
    PrecomputedTableOracle,
    PropertySpecOracle,
    SubprocessOracle,
    normalize_direction,
    resolve_oracle,
    resolve_oracle_pair,
)
from .evaluate import (
    COMPUTED_COLUMNS,
    Stage1Result,
    compute_summary_metrics,
    compute_tanimoto,
    evaluate,
    get_seed_mol_stats,
    internal_diversity,
    max_similarity_to_gt,
    mean_similarity_to_seeds,
    v_n_i_mask,
)
from .property_metrics import (
    PropertyBinningConfig,
    PropertySpec,
    batch_predict_binary_property,
    compute_improvement,
    compute_properties,
    delta_to_bin,
    get_available_properties,
    get_default_delta_binning,
    get_default_value_binning,
    get_property_specs,
    is_valid_smiles,
    load_evaluator,
    property_delta_bin_summary,
    property_improvement_stats,
    property_preservation_mad,
    property_top_k_stats,
    value_to_bin,
)
from .distribution_metrics import compute_distribution_metrics, is_available
from .normalize import normalize_generation_block, normalize_generation_table
from .methods import CANONICAL_METHODS, METHOD_DATASETS, METHOD_ALIASES, ensure_supported_method, normalize_method_name, resolve_methods
from .pipeline import (
    DEFAULT_SUMMARY_METRICS,
    MethodRunOutput,
    load_average_results,
    run_method_pipeline,
    run_pipeline,
    summarize_average,
)

__all__ = [
    "COMPUTED_COLUMNS",
    "evaluate",
    "evaluate_from_csv",
    "PairitEvalConfig",
    "get_default_generation_paths",
    "Stage1Result",
    # Per-molecule and summary (validity, novelty, sim_seed, etc.)
    "compute_tanimoto",
    "compute_summary_metrics",
    "get_seed_mol_stats",
    "is_valid_smiles",
    "internal_diversity",
    "mean_similarity_to_seeds",
    "max_similarity_to_gt",
    "v_n_i_mask",
    # Property metrics (PropertySpec, continuous, binning)
    "PropertySpec",
    "PropertyBinningConfig",
    "get_property_specs",
    "compute_properties",
    "load_evaluator",
    "batch_predict_binary_property",
    "compute_improvement",
    "Oracle",
    "FingerprintJoblibOracle",
    "PrecomputedTableOracle",
    "SubprocessOracle",
    "CallableOracle",
    "PropertySpecOracle",
    "resolve_oracle",
    "resolve_oracle_pair",
    "normalize_direction",
    "property_improvement_stats",
    "property_preservation_mad",
    "property_top_k_stats",
    "property_delta_bin_summary",
    "get_available_properties",
    "get_default_value_binning",
    "get_default_delta_binning",
    "value_to_bin",
    "delta_to_bin",
    # Distribution metrics (FCD, SNN, W1, etc.)
    "compute_distribution_metrics",
    "is_available",
    # Normalization helpers
    "normalize_generation_block",
    "normalize_generation_table",
    # Method naming helpers
    "CANONICAL_METHODS",
    "METHOD_DATASETS",
    "METHOD_ALIASES",
    "normalize_method_name",
    "ensure_supported_method",
    "resolve_methods",
    # Pipeline helpers
    "MethodRunOutput",
    "DEFAULT_SUMMARY_METRICS",
    "run_method_pipeline",
    "run_pipeline",
    "load_average_results",
    "summarize_average",
]


def evaluate_from_csv(
    csv_path: Union[Path, str],
    config: Optional[PairitEvalConfig] = None,
    methods: Optional[list] = None,
    datasets: Optional[list] = None,
    verbose: bool = True,
    compute_distribution_metrics: bool = False,
    compute_property_metrics: bool = False,
    property_names: Optional[List[str]] = None,
    property_delta_binning: Optional[Dict[str, Any]] = None,
    property_preservation_names: Optional[List[str]] = None,
    property_sim_band_range: Optional[tuple] = (0.4, 0.6),
    property_top_k: Optional[int] = None,
    min_property_delta: float = 1.0,
    reuse_precomputed: bool = False,
) -> Stage1Result:
    """
    Load normalized CSV and run Stage 1 (same as evaluate(pd.read_csv(csv_path), ...)).
    """
    df = pd.read_csv(csv_path)
    return evaluate(
        df, config=config, methods=methods, datasets=datasets,
        verbose=verbose,
        compute_distribution_metrics=compute_distribution_metrics,
        compute_property_metrics=compute_property_metrics,
        property_names=property_names,
        property_delta_binning=property_delta_binning,
        property_preservation_names=property_preservation_names,
        property_sim_band_range=property_sim_band_range,
        property_top_k=property_top_k,
        min_property_delta=min_property_delta,
        reuse_precomputed=reuse_precomputed,
    )
