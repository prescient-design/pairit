"""
Base Metrics Computation
========================

Shared base class for conditional and unconditional molecule generation metrics.

Provides:
- ``SampleResult`` dataclass for structured process_sample returns.
- ``MoleculeMetricsComputer`` base class with shared buffer management,
  per-sample helpers, and epoch-level computation methods.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from atomic_datasets.utils.rdkit import is_molecule_sane

from pairit.eval import (
    PropertySpec,
    compute_distribution_metrics as pe_compute_distribution_metrics,
    internal_diversity as pe_internal_diversity,
    mean_similarity_to_seeds as pe_mean_similarity_to_seeds,
    property_improvement_stats as pe_property_improvement_stats,
    property_preservation_mad as pe_property_preservation_mad,
    property_top_k_stats as pe_property_top_k_stats,
)

from pairit.callbacks.molecule_metrics import (
    FEATURE_NAMES,
    property_improvement_under_similarity_constraint,
    uniqueness_fraction,
    validity_fraction,
)

from .timing import TimingTracker

# ---------------------------------------------------------------------------
# Lazy import of optional distance functions
# ---------------------------------------------------------------------------
try:
    from pair_gen.utils_mol import get_tanimoto_distance
except ImportError:
    get_tanimoto_distance = None


def _has_trainer(pl_module) -> bool:
    """Check if a Lightning module has an attached trainer."""
    return hasattr(pl_module, "_trainer") and pl_module._trainer is not None


def _safe_log(pl_module, name: str, value, **kwargs):
    """Log to *pl_module* only when a trainer is attached."""
    if _has_trainer(pl_module):
        pl_module.log(name, value, **kwargs)


# ---------------------------------------------------------------------------
# SampleResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Result of processing a single molecule pair.

    Replaces the previous raw tuples returned by ``process_sample`` with
    named, typed fields for readability at call sites.

    Parameters
    ----------
    valid_and_improved : dict[str, bool]
        Mapping from property name to whether the generation is both valid
        and improved for that property.
    tanimoto_dist : float | None
        Tanimoto distance between seed and generated molecule.
    shape_tanimoto_dist : float | None
        3-D shape Tanimoto distance between seed and generated molecule.
    seed_props : dict[str, float] | None
        Computed property values for the seed molecule.
    gen_props : dict[str, float] | None
        Computed property values for the generated molecule.
    is_memorized : bool | None
        Whether the generated molecule was found in a reference set.
    prompt_agreement : dict[str, dict[str, bool]] | None
        Per-property prompt agreement information (conditional only).
    input_tanimoto_dist : float | None
        Tanimoto distance of the original input pair (conditional only).
    input_shape_tanimoto_dist : float | None
        Shape Tanimoto distance of the original input pair (conditional only).
    distance_percentile : float | None
        Percentile rank of the seed→gen distance (conditional only).
    """

    valid_and_improved: dict[str, bool] = field(default_factory=dict)
    tanimoto_dist: float | None = None
    shape_tanimoto_dist: float | None = None
    seed_props: dict[str, float] | None = None
    gen_props: dict[str, float] | None = None
    is_memorized: bool | None = None
    prompt_agreement: dict[str, dict[str, bool]] | None = None
    # Conditional-only fields (None for unconditional):
    input_tanimoto_dist: float | None = None
    input_shape_tanimoto_dist: float | None = None
    distance_percentile: float | None = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

_NaN = float("nan")


def _safe_mean(vals: list[float] | Sequence[float]) -> float | None:
    """Return the mean of *vals*, or ``None`` if empty."""
    return float(np.mean(vals)) if vals else None


class MoleculeMetricsComputer:
    """Shared base for conditional and unconditional metrics computers.

    Manages common constructor parameters, epoch buffers, per-sample helpers,
    and reusable epoch-level computation blocks (pair metrics, distributional
    metrics, uniqueness / novelty).

    Subclasses override ``reset``, ``process_sample``, and ``compute_metrics``
    to add mode-specific behaviour.

    Parameters
    ----------
    property_specs : list[PropertySpec] | None
        Properties to track (name, fn, direction, min_delta).
    distance_name : str
        Name of the distance metric used for logging (default ``"tanimoto"``).
    reference_canon_sets_by_name : dict[str, set[str]] | None
        Named canonical SMILES sets for memorisation and novelty checks.
    default_reference_name : str | None
        Primary reference set name.
    reference_smiles_by_name : dict[str, list[str]] | None
        Named reference SMILES pools for distributional metrics
        (SNN, FCD, diversity, coverage, fragment/scaffold, W1/KDE).
    distance_fn : Callable | None
        Pairwise distance function ``(smi_a, smi_b) -> float``.
    expensive_metrics_to_compute : list[str] | None
        Which expensive metric groups to enable.  Recognised keys:
        ``"snn"``, ``"fcd"``, ``"internal_diversity"``,
        ``"fragment_scaffold"``, ``"w1_kde"``, ``"memorization"``.
    timing_tracker : TimingTracker | None
        Optional operation-timing tracker.
    check_memorization_from_any_reference : bool
        If ``True``, memorisation checks union across all reference sets.
    fcd_min_required : int
        Minimum sample count required for FCD computation.
    metric_prefix : str
        Prefix prepended to per-sample logged metric names
        (``"val"`` during training, ``""`` during standalone evaluation).
    """

    def __init__(
        self,
        property_specs: list[PropertySpec] | None = None,
        distance_name: str = "tanimoto",
        reference_canon_sets_by_name: dict[str, set[str]] | None = None,
        default_reference_name: str | None = None,
        reference_smiles_by_name: dict[str, list[str]] | None = None,
        distance_fn: Callable | None = None,
        expensive_metrics_to_compute: list[str] | None = None,
        timing_tracker: TimingTracker | None = None,
        check_memorization_from_any_reference: bool = True,
        fcd_min_required: int = 10,
        metric_prefix: str = "",
    ):
        self.property_specs: list[PropertySpec] = property_specs or []
        self.distance_name = distance_name

        # Canonical sets — used for memorisation / novelty
        self.reference_canon_sets_by_name: dict[str, set[str]] = (
            reference_canon_sets_by_name or {}
        )
        self.default_reference_name: str | None = default_reference_name or (
            next(iter(self.reference_canon_sets_by_name))
            if self.reference_canon_sets_by_name
            else None
        )
        self.training_canon_set: set[str] | None = (
            self.reference_canon_sets_by_name.get(self.default_reference_name)
            if self.default_reference_name is not None
            else None
        )

        # SMILES pools — used for distributional metrics
        self.reference_smiles_by_name: dict[str, list[str]] = (
            reference_smiles_by_name or {}
        )
        self._use_named_references: bool = len(self.reference_smiles_by_name) > 1
        # Backward-compatible single reference list
        self.reference_smiles: list[str] | None = (
            self.reference_smiles_by_name.get(self.default_reference_name)
            if self.default_reference_name is not None
            else None
        )

        self.distance_fn = distance_fn or get_tanimoto_distance
        self.expensive_metrics_to_compute: list[str] = (
            expensive_metrics_to_compute or []
        )
        self.timing_tracker = timing_tracker
        self.check_memorization_from_any_reference = (
            check_memorization_from_any_reference
        )
        self.fcd_min_required = fcd_min_required
        self.metric_prefix = metric_prefix

        # Initialise common buffers
        self._reset_base()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _reset_base(self) -> None:
        """Reset epoch buffers shared between conditional and unconditional."""
        self.epoch_seeds: list[str] = []
        self.epoch_gens: list[str] = []
        self.epoch_all_gens: list[str] = []
        self.epoch_valid_canon_gens: list[str] = []
        self.epoch_seed_to_gen_dist: list[float] = []

    # ------------------------------------------------------------------
    # Timing helper
    # ------------------------------------------------------------------

    def _time(self, name: str):
        """Return a timing context manager, or :func:`nullcontext` if none."""
        if self.timing_tracker is not None:
            return self.timing_tracker.time_operation(name)
        return nullcontext()

    # ------------------------------------------------------------------
    # Per-sample helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid(smiles: str) -> bool:
        """Quick RDKit + sanity validity check."""
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and is_molecule_sane(mol)
        except Exception:
            return False

    def _log_sample(
        self,
        pl_module,
        name: str,
        value,
        *,
        dataset_key: str | None = None,
        **kwargs,
    ) -> None:
        """Log a per-sample metric respecting *metric_prefix* and *dataset_key*.

        Applies the stored ``metric_prefix`` (e.g. ``"val"``) and, when
        *dataset_key* is given, nests the metric under that key so that
        per-dataset metrics don't collide in mixed-dataloader mode.

        Parameters
        ----------
        pl_module
            PyTorch Lightning module with a ``log`` method.
        name : str
            Prefix-free metric name, e.g. ``"logp_conditional_input"``.
        value
            Scalar value (or tensor) to log.
        dataset_key : str | None
            Optional dataset identifier for mixed-dataloader namespacing.
        **kwargs
            Forwarded to :func:`_safe_log` (e.g. ``sync_dist``, ``on_epoch``).
        """
        if self.metric_prefix:
            if dataset_key:
                full_name = f"{self.metric_prefix}/{dataset_key}/{name}"
            else:
                full_name = f"{self.metric_prefix}_{name}"
        else:
            full_name = name
        _safe_log(pl_module, full_name, value, **kwargs)

    def _accumulate_pair(
        self,
        seed: str,
        gen: str,
        canonical_gen: str | None,
        is_valid: bool,
    ) -> None:
        """Append a seed/gen pair to the shared epoch buffers."""
        if seed and gen:
            self.epoch_seeds.append(seed)
            self.epoch_gens.append(gen)
        if gen:
            self.epoch_all_gens.append(gen)
        if is_valid and canonical_gen:
            self.epoch_valid_canon_gens.append(canonical_gen)

    # ------------------------------------------------------------------
    # Memorisation helpers
    # ------------------------------------------------------------------

    def _check_memorization_any(self, canonical_gen: str) -> bool:
        """Return ``True`` if *canonical_gen* is in **any** reference set."""
        if not self.reference_canon_sets_by_name:
            return False
        return any(
            canonical_gen in ref_set
            for ref_set in self.reference_canon_sets_by_name.values()
            if ref_set
        )

    def _check_memorization_default(self, canonical_gen: str) -> bool:
        """Return ``True`` if *canonical_gen* is in the default reference set."""
        if self.training_canon_set is None:
            return False
        return canonical_gen in self.training_canon_set

    # ------------------------------------------------------------------
    # Metric-name helpers
    # ------------------------------------------------------------------

    def _ref_metric(self, metric_name: str, ref_name: str) -> str:
        """Suffix *metric_name* with ``/<ref_name>`` when using named refs."""
        if self._use_named_references and ref_name != "_default":
            return f"{metric_name}/{ref_name}"
        return metric_name

    # ------------------------------------------------------------------
    # Epoch-level: pair metrics
    # ------------------------------------------------------------------

    def _compute_pair_metrics(self, suffix: str = "") -> dict[str, float]:
        """Property improvement, similarity, preservation, top-k metrics.

        These metrics compare aligned ``(seed, gen)`` pairs accumulated
        during the epoch.

        Parameters
        ----------
        suffix : str
            Appended to every metric key (e.g. ``""`` for conditional,
            ``"_unconditional"`` for unconditional).

        Returns
        -------
        dict[str, float]
            Metric name → value mapping.  ``NaN`` is used for metrics
            that cannot be computed (e.g. empty buffers).
        """
        metrics: dict[str, float] = {}
        seeds = self.epoch_seeds
        gens = self.epoch_gens
        all_gens = self.epoch_all_gens

        if not seeds or not gens:
            return metrics

        # -- Similarity to seeds -----------------------------------------
        sim = pe_mean_similarity_to_seeds(seeds, gens)
        metrics[f"similarity_to_seed_mean{suffix}"] = (
            float(sim) if sim is not None else _NaN
        )

        # -- Per-property improvement stats ------------------------------
        for spec in self.property_specs:
            stats = pe_property_improvement_stats(seeds, gens, spec)
            for stat_key, abbrev in [
                ("mean_delta", "mean"),
                ("median_delta", "median"),
                ("std_delta", "std"),
            ]:
                val = stats.get(stat_key)
                metrics[f"property_delta_{abbrev}{suffix}/{spec.name}"] = (
                    float(val) if val is not None else _NaN
                )
            frac = stats.get("frac_improved")
            metrics[f"property_frac_improved{suffix}/{spec.name}"] = (
                float(frac) if frac is not None else _NaN
            )

        # -- Property preservation MAD -----------------------------------
        if self.property_specs:
            mad_results = pe_property_preservation_mad(
                seeds, gens, self.property_specs
            )
            macro_mad = mad_results.get("preservation_mad_macro_mean")
            metrics[f"property_preservation_mad_macro{suffix}"] = (
                float(macro_mad) if macro_mad is not None else _NaN
            )
            for spec in self.property_specs:
                per_mad = mad_results.get(f"preservation_mad/{spec.name}")
                metrics[f"property_preservation_mad{suffix}/{spec.name}"] = (
                    float(per_mad) if per_mad is not None else _NaN
                )

        # -- Improvement under similarity constraint ---------------------
        for spec in self.property_specs:
            res = property_improvement_under_similarity_constraint(
                seeds, gens, spec, sim_range=(0.4, 0.6)
            )
            for key, mname in [
                ("mean_delta", f"delta_under_sim{suffix}/{spec.name}"),
                ("frac_improved", f"frac_improved_under_sim{suffix}/{spec.name}"),
            ]:
                val = res.get(key)
                metrics[mname] = float(val) if val is not None else _NaN

        # -- Validity fraction (includes invalid gens) -------------------
        if all_gens:
            vf = validity_fraction(all_gens, sanity_fn=is_molecule_sane)
            metrics[f"validity_fraction{suffix}"] = (
                float(vf) if vf is not None else _NaN
            )
        else:
            metrics[f"validity_fraction{suffix}"] = _NaN

        # -- Top-k best property ----------------------------------------
        for spec in self.property_specs:
            topk = pe_property_top_k_stats(gens, spec, k=3)
            best_val = topk.get("best")
            mean_topk = topk.get("mean_top_k")
            metrics[f"topk_best{suffix}/{spec.name}_best"] = (
                float(best_val) if best_val is not None else _NaN
            )
            metrics[f"topk_best{suffix}/{spec.name}_mean_topk"] = (
                float(mean_topk) if mean_topk is not None else _NaN
            )

        return metrics

    # ------------------------------------------------------------------
    # Epoch-level: distributional metrics
    # ------------------------------------------------------------------

    def _compute_distributional_metrics(
        self, suffix: str = ""
    ) -> dict[str, float]:
        """SNN, FCD, internal diversity, coverage, fragment/scaffold, W1/KDE.

        These pool-level metrics compare the set of valid generated
        molecules against one or more named reference SMILES pools.

        Uses ``pairit_eval.compute_distribution_metrics`` for bundled
        computation of SNN, FCD, fragment/scaffold, coverage, and W1/KDE.
        Internal diversity is computed separately via
        ``pairit_eval.internal_diversity``.

        Parameters
        ----------
        suffix : str
            Appended to every metric key (e.g. ``"_conditional"``).

        Returns
        -------
        dict[str, float]
        """
        metrics: dict[str, float] = {}
        gen_pool = self.epoch_valid_canon_gens
        has_gen = len(gen_pool) > 0

        if not has_gen:
            return metrics

        # Build reference pools map
        ref_pools: dict[str, list[str]] = dict(self.reference_smiles_by_name)
        if not ref_pools and self.reference_smiles:
            ref_pools = {"_default": self.reference_smiles}
        if not ref_pools:
            return metrics

        # -- Internal diversity (standalone, no reference needed) --------
        if "internal_diversity" in self.expensive_metrics_to_compute:
            div_value = None
            try:
                with self._time("distributional/internal_diversity"):
                    div_value, _, _ = pe_internal_diversity(
                        gen_pool, deduplicate=True
                    )
            except Exception:
                pass
            metrics[f"diversity{suffix}"] = (
                float(div_value) if div_value is not None else _NaN
            )

        # -- Bundled distribution metrics per reference pool ------------
        # Only call the expensive pe_compute_distribution_metrics when at
        # least one distributional metric is requested.  Coverage was
        # previously ungated, which caused the full FCD/SNN/W1 bundle to
        # run even when only memorization was requested.
        _distributional_gates = {"snn", "fcd", "fragment_scaffold", "w1_kde", "coverage"}
        _need_bundled = bool(_distributional_gates & set(self.expensive_metrics_to_compute))

        for ref_name, ref_pool in ref_pools.items():
            if not ref_pool:
                continue

            dist: dict = {}
            if _need_bundled:
                try:
                    with self._time(f"distributional/bundled/{ref_name}"):
                        dist = pe_compute_distribution_metrics(
                            gen_pool,
                            ref_pool,
                            min_generated=self.fcd_min_required,
                        )
                except Exception:
                    dist = {}

            # Map pairit_eval keys → pairit metric keys, gated on
            # expensive_metrics_to_compute to preserve current behaviour.
            _key_map: list[tuple[str, str, str]] = []

            if "snn" in self.expensive_metrics_to_compute:
                _key_map.append(("snn_mean", f"snn_mean{suffix}", "snn"))
                _key_map.append(("snn_median", f"snn_median{suffix}", "snn"))

            if "fcd" in self.expensive_metrics_to_compute:
                _key_map.append(("fcd", f"fcd{suffix}", "fcd"))

            if "fragment_scaffold" in self.expensive_metrics_to_compute:
                _key_map.append(
                    ("fragment_cosine", f"fragment_cosine{suffix}", "fragment_scaffold")
                )
                _key_map.append(
                    ("scaffold_cosine", f"scaffold_cosine{suffix}", "fragment_scaffold")
                )

            # Coverage is gated like other distributional metrics
            if "coverage" in self.expensive_metrics_to_compute:
                _key_map.append(
                    ("coverage_ratio", f"coverage_ratio{suffix}", "coverage")
                )

            if "w1_kde" in self.expensive_metrics_to_compute:
                _key_map.append(
                    ("w1_macro_mean", f"feature_w1_macro{suffix}", "w1_kde")
                )
                _key_map.append(
                    ("w1_norm_macro_mean", f"feature_w1_norm_macro{suffix}", "w1_kde")
                )
                _key_map.append(
                    ("kde_ll_delta_macro_mean", f"feature_kde_delta_macro{suffix}", "w1_kde")
                )

            for pe_key, lobster_key, gate in _key_map:
                val = dist.get(pe_key)
                metrics[self._ref_metric(lobster_key, ref_name)] = (
                    float(val) if val is not None else _NaN
                )

        return metrics

    # ------------------------------------------------------------------
    # Epoch-level: uniqueness / novelty
    # ------------------------------------------------------------------

    def _compute_uniqueness_novelty(
        self, suffix: str = ""
    ) -> dict[str, float]:
        """Uniqueness and novelty among valid generated canonical SMILES.

        Parameters
        ----------
        suffix : str
            Appended to metric keys (e.g. ``"_conditional"``).

        Returns
        -------
        dict[str, float]
        """
        metrics: dict[str, float] = {}
        num_valid = len(self.epoch_valid_canon_gens)

        uniqueness_value: float | None = None
        novelty_value: float | None = None

        if num_valid > 0:
            unique_valid = len(set(self.epoch_valid_canon_gens))
            uniqueness_value = float(unique_valid) / num_valid

            if self.training_canon_set is not None:
                num_novel = sum(
                    1
                    for s in self.epoch_valid_canon_gens
                    if s not in self.training_canon_set
                )
                novelty_value = float(num_novel) / num_valid
        elif self.training_canon_set is not None:
            novelty_value = _NaN

        metrics[f"uniqueness{suffix}_valid"] = (
            float(uniqueness_value) if uniqueness_value is not None else _NaN
        )
        metrics[f"novelty{suffix}_valid"] = (
            float(novelty_value) if novelty_value is not None else _NaN
        )

        # Per-reference novelty
        if num_valid > 0 and self.reference_canon_sets_by_name:
            for ref_name in sorted(self.reference_canon_sets_by_name):
                ref_set = self.reference_canon_sets_by_name.get(ref_name) or set()
                if not ref_set:
                    rate = None
                else:
                    num_novel_ref = sum(
                        1
                        for s in self.epoch_valid_canon_gens
                        if s not in ref_set
                    )
                    rate = float(num_novel_ref) / num_valid
                metrics[f"novelty{suffix}_valid/{ref_name}"] = (
                    float(rate) if rate is not None else _NaN
                )

        return metrics
