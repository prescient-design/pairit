"""
High-level pipeline helpers:
1) load raw generations
2) normalize to canonical seed/gen grid
3) evaluate and save per-generation/per-seed/average CSVs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .cli_utils import build_eval_config
from .config import DATASET_CONFIG, PairitEvalConfig
from .evaluate import Stage1Result, evaluate
from .methods import ensure_supported_method, resolve_methods
from .normalize import normalize_generation_table
from .raw_loaders import load_method_all


DEFAULT_SUMMARY_METRICS = [
    "method",
    "dataset",
    "valid",
    "novel",
    "internal_diversity",
    "unique",
    "qed",
    "improved",
    "sim_seed",
    "fcd",
    "snn_mean",
    "fragment_cosine",
    "scaffold_cosine",
    "coverage_ratio",
    "w1_macro_mean",
    "kde_ll_delta_macro_mean",
]


@dataclass
class MethodRunOutput:
    method: str
    normalized_path: Optional[Path]
    per_generation_path: Optional[Path]
    per_seed_path: Optional[Path]
    average_path: Optional[Path]
    n_input_rows: int
    n_normalized_rows: int
    n_per_generation_rows: int
    n_per_seed_rows: int
    n_average_rows: int


def _should_compute_property(
    datasets: Iterable[str],
    property_mode: str,
) -> tuple[bool, Optional[List[str]]]:
    mode = str(property_mode).strip().lower()
    if mode == "off":
        return False, None
    if mode == "on":
        return True, None
    datasets_set = set(datasets)
    zinc250k_default = "zinc250k" in datasets_set and DATASET_CONFIG.get("zinc250k", {}).get("default_property")
    if zinc250k_default:
        return True, [DATASET_CONFIG["zinc250k"]["default_property"]]
    return False, None


def _dataset_n_generations(df: pd.DataFrame, default_n_generations: int) -> Dict[str, int]:
    return {ds: int(default_n_generations) for ds in df["dataset"].dropna().unique().tolist()}


def _write_csv_if(path: Path, df: pd.DataFrame, enabled: bool) -> Optional[Path]:
    if not enabled:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_method_pipeline(
    method: str,
    *,
    output_root: Path,
    config: PairitEvalConfig,
    n_generations: int = 5,
    datasets: Optional[Sequence[str]] = None,
    compute_distribution_metrics: bool = False,
    property_mode: str = "auto",
    min_property_delta: float = 1.0,
    verbose: bool = True,
    save_normalized: bool = True,
    save_per_generation: bool = True,
    save_per_seed: bool = True,
    save_average: bool = True,
    reuse_precomputed: bool = False,
    normalize: bool = True,
) -> MethodRunOutput:
    canonical = ensure_supported_method(method)
    out_dir = Path(output_root) / canonical
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_method_all(canonical)
    if datasets:
        ds_set = {str(d) for d in datasets}
        raw = raw[raw["dataset"].isin(ds_set)].copy()

    if len(raw) == 0:
        raise ValueError(f"No raw generations found for method '{canonical}'.")

    if normalize:
        normalized = normalize_generation_table(
            raw,
            config=config,
            n_generations_by_dataset=_dataset_n_generations(raw, n_generations),
        )
    else:
        normalized = raw

    compute_property, property_names = _should_compute_property(
        datasets=normalized["dataset"].dropna().unique().tolist(),
        property_mode=property_mode,
    )

    result = evaluate(
        normalized,
        config=config,
        methods=[canonical],
        verbose=verbose,
        compute_distribution_metrics=compute_distribution_metrics,
        compute_property_metrics=compute_property,
        property_names=property_names,
        min_property_delta=min_property_delta,
        reuse_precomputed=reuse_precomputed,
    )

    normalized_path = _write_csv_if(
        out_dir / f"{canonical}_normalized.csv",
        normalized,
        save_normalized,
    )
    per_generation_path = _write_csv_if(
        out_dir / f"{canonical}_per_generation.csv",
        result.per_generation,
        save_per_generation,
    )
    per_seed_path = _write_csv_if(
        out_dir / f"{canonical}_per_seed.csv",
        result.per_seed,
        save_per_seed,
    )
    average_path = _write_csv_if(
        out_dir / f"{canonical}_average.csv",
        result.average,
        save_average,
    )

    return MethodRunOutput(
        method=canonical,
        normalized_path=normalized_path,
        per_generation_path=per_generation_path,
        per_seed_path=per_seed_path,
        average_path=average_path,
        n_input_rows=len(raw),
        n_normalized_rows=len(normalized),
        n_per_generation_rows=len(result.per_generation),
        n_per_seed_rows=len(result.per_seed),
        n_average_rows=len(result.average),
    )


def run_pipeline(
    methods: Sequence[str] | None,
    *,
    output_root: Path,
    config: Optional[PairitEvalConfig] = None,
    n_generations: int = 5,
    datasets: Optional[Sequence[str]] = None,
    compute_distribution_metrics: bool = False,
    property_mode: str = "auto",
    min_property_delta: float = 1.0,
    verbose: bool = True,
    fail_fast: bool = False,
    save_normalized: bool = True,
    save_per_generation: bool = True,
    save_per_seed: bool = True,
    save_average: bool = True,
    reuse_precomputed: bool = False,
    normalize: bool = True,
) -> List[MethodRunOutput]:
    eval_config = config or build_eval_config()
    selected_methods = resolve_methods(methods)
    outputs: List[MethodRunOutput] = []
    errors: List[str] = []

    for method in selected_methods:
        try:
            out = run_method_pipeline(
                method,
                output_root=output_root,
                config=eval_config,
                n_generations=n_generations,
                datasets=datasets,
                compute_distribution_metrics=compute_distribution_metrics,
                property_mode=property_mode,
                min_property_delta=min_property_delta,
                verbose=verbose,
                save_normalized=save_normalized,
                save_per_generation=save_per_generation,
                save_per_seed=save_per_seed,
                save_average=save_average,
                reuse_precomputed=reuse_precomputed,
                normalize=normalize,
            )
            outputs.append(out)
        except Exception as exc:
            msg = f"{method}: {exc}"
            errors.append(msg)
            if fail_fast:
                raise

    if errors:
        error_path = Path(output_root) / "_errors.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")

    return outputs


def load_average_results(
    output_root: Path,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    selected = set(resolve_methods(methods)) if methods else None
    for path in sorted(Path(output_root).glob("*/*_average.csv")):
        method = path.parent.name
        if selected is not None and method not in selected:
            continue
        df = pd.read_csv(path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_average(
    average_df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if len(average_df) == 0:
        return average_df
    cols = [c for c in (metrics or DEFAULT_SUMMARY_METRICS) if c in average_df.columns]
    if not cols:
        cols = average_df.columns.tolist()
    out = average_df[cols].copy()
    sort_cols = [c for c in ("dataset", "method") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out
