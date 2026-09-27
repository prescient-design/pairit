"""
Unconditional Metrics Computation
==================================

Computes and logs all metrics for unconditional molecule generation.
"""

import math
import traceback
from contextlib import nullcontext
from typing import List, Optional, Tuple, Dict
import torch
from atomic_datasets.utils.rdkit import is_molecule_sane

from pairit.eval import (
    compute_distribution_metrics as pe_compute_distribution_metrics,
    internal_diversity as pe_internal_diversity,
)

# Import metrics that stay in pairit
from pairit.callbacks.molecule_metrics import (
    validity_fraction,
    uniqueness_fraction,
    novelty_fraction,
)
from .base_metrics import (
    MoleculeMetricsComputer,
    SampleResult,
    _has_trainer,
    _safe_log,
)
from .timing import TimingTracker
from .property_registry import PropertySpec, compute_all_properties_for_smiles

# Import for distance computation
try:
    from pair_gen.utils_mol import get_tanimoto_distance, get_shape_tanimoto_distance
except ImportError:
    get_tanimoto_distance = None
    get_shape_tanimoto_distance = None


class UnconditionalMetricsComputer(MoleculeMetricsComputer):
    """
    Computes all unconditional generation metrics.

    Compares unconditionally generated molecules against a reference set
    to evaluate diversity, similarity to training data, and feature alignment.
    """
    
    def __init__(self,
                 reference_smiles: Optional[List[str]] = None,
                 reference_smiles_by_name: Optional[Dict[str, List[str]]] = None,
                 default_reference_name: Optional[str] = None,
                 fcd_min_required: int = 10,
                 expensive_metrics_to_compute: Optional[List[str]] = None,
                 training_canon_set: Optional[set] = None,
                 reference_canon_sets_by_name: Optional[Dict[str, set]] = None,
                 timing_tracker: Optional[TimingTracker] = None,
                 property_specs: Optional[List[PropertySpec]] = None,
                 check_memorization_from_any_reference: bool = True,
                 metric_prefix: str = "val"):
        """
        Initialize metrics computer.

        Args:
            reference_smiles: Reference SMILES for comparison (e.g., training set)
            reference_smiles_by_name: Named reference SMILES pools for distributional metrics
            default_reference_name: Primary reference set name
            fcd_min_required: Minimum molecules required to compute FCD
            expensive_metrics_to_compute: List of expensive metrics to compute
            training_canon_set: Set of canonical training SMILES for memorization check
            reference_canon_sets_by_name: Named canonical reference sets for memorization
            timing_tracker: TimingTracker for logging fine-grained timings
            property_specs: List of property specifications for sample saving
            check_memorization_from_any_reference: If True, memorization checks union of all reference sets
            metric_prefix: Prefix for per-sample logged metrics ("val" during training, "" during eval)
        """
        # Build reference_smiles_by_name from reference_smiles if needed (backward compat)
        ref_smiles_by_name = reference_smiles_by_name or {}
        if not ref_smiles_by_name and reference_smiles:
            inferred_default = default_reference_name or "_default"
            ref_smiles_by_name = {inferred_default: reference_smiles}

        super().__init__(
            property_specs=property_specs,
            distance_name="tanimoto",
            reference_canon_sets_by_name=reference_canon_sets_by_name,
            default_reference_name=default_reference_name,
            reference_smiles_by_name=ref_smiles_by_name,
            distance_fn=get_tanimoto_distance,
            expensive_metrics_to_compute=expensive_metrics_to_compute or [],
            timing_tracker=timing_tracker,
            check_memorization_from_any_reference=check_memorization_from_any_reference,
            fcd_min_required=fcd_min_required,
            metric_prefix=metric_prefix,
        )

        # Backward-compatible training canon set (may differ from base logic
        # when caller passes training_canon_set explicitly without reference_canon_sets_by_name)
        if training_canon_set is not None:
            self.training_canon_set = training_canon_set

        # Unconditional-specific buffers
        self._reset_unconditional()
    
    def reset(self):
        """Reset all accumulation buffers (base + unconditional)."""
        self._reset_base()
        self._reset_unconditional()

    def _reset_unconditional(self):
        """Reset unconditional-specific accumulation buffers."""
        self.epoch_uncond_valid_gen_smiles: List[str] = []
        self.epoch_first_valid_count: int = 0
        self.epoch_second_valid_count: int = 0
        self.epoch_total_checked: int = 0
        self.epoch_first_mol_memorized: int = 0
        self.epoch_first_mol_checked: int = 0
        self.epoch_second_mol_memorized: int = 0
        self.epoch_second_mol_checked: int = 0
        self.epoch_first_mol_memorized_by_ref: Dict[str, int] = {k: 0 for k in self.reference_canon_sets_by_name.keys()}
        self.epoch_first_mol_checked_by_ref: Dict[str, int] = {k: 0 for k in self.reference_canon_sets_by_name.keys()}
        self.epoch_second_mol_memorized_by_ref: Dict[str, int] = {k: 0 for k in self.reference_canon_sets_by_name.keys()}
        self.epoch_second_mol_checked_by_ref: Dict[str, int] = {k: 0 for k in self.reference_canon_sets_by_name.keys()}

    def accumulate_valid_smiles(self, smiles: str):
        """
        Add a valid generated SMILES to the buffer.
        
        Args:
            smiles: Valid SMILES string
        """
        self.epoch_uncond_valid_gen_smiles.append(smiles)
    
    def process_sample(self, pl_module, first_smiles: str, second_smiles: str,
                      first_valid: bool, second_valid: bool) -> SampleResult:
        """
        Process a single unconditional generation sample: compute distance metrics, properties, and track validity.

        Args:
            pl_module: PyTorch Lightning module for logging
            first_smiles: First generated SMILES
            second_smiles: Second generated SMILES
            first_valid: Whether first molecule is valid
            second_valid: Whether second molecule is valid

        Returns:
            SampleResult with tanimoto_dist, shape_tanimoto_dist, seed_props (first), gen_props (second)
        """
        # Track validity
        self.epoch_total_checked += 1
        if first_valid:
            self.epoch_first_valid_count += 1
        if second_valid:
            self.epoch_second_valid_count += 1

        # Accumulate pair data into base buffers for pair metrics
        # Map first → seed, second → gen for pair improvement computation
        if first_valid and second_valid and first_smiles and second_smiles:
            from rdkit import Chem
            canon_second = None
            mol = Chem.MolFromSmiles(second_smiles)
            if mol is not None:
                canon_second = Chem.MolToSmiles(mol)
            self._accumulate_pair(first_smiles, second_smiles, canon_second, second_valid)
        
        tanimoto_dist = None
        shape_tanimoto_dist = None
        first_props = None
        second_props = None
        
        # Compute properties for sample saving and logging
        if self.property_specs:
            if first_valid and first_smiles:
                try:
                    first_props = compute_all_properties_for_smiles(first_smiles, self.property_specs)
                    # Log per-sample properties to wandb
                    if first_props:
                        for spec in self.property_specs:
                            if (val := first_props.get(spec.name)) is not None:
                                _safe_log(pl_module,
                                    f"val_{spec.name}_unconditional_first",
                                    torch.tensor(float(val), device=pl_module.device),
                                    sync_dist=False,  # per-sample logging; aggregate explicitly later
                                    on_step=False,
                                    on_epoch=True,
                                )
                except Exception as exc:
                    print(f"[Warning] Failed to compute properties for first molecule '{first_smiles}': {exc}")
            
            if second_valid and second_smiles:
                try:
                    second_props = compute_all_properties_for_smiles(second_smiles, self.property_specs)
                    # Log per-sample properties to wandb
                    if second_props:
                        for spec in self.property_specs:
                            if (val := second_props.get(spec.name)) is not None:
                                _safe_log(pl_module,
                                    f"val_{spec.name}_unconditional_second",
                                    torch.tensor(float(val), device=pl_module.device),
                                    sync_dist=False,  # per-sample logging; aggregate explicitly later
                                    on_step=False,
                                    on_epoch=True,
                                )
                except Exception as exc:
                    print(f"[Warning] Failed to compute properties for second molecule '{second_smiles}': {exc}")
            
            # Log property deltas
            if first_props and second_props:
                for spec in self.property_specs:
                    fi, si = first_props.get(spec.name), second_props.get(spec.name)
                    if fi is not None and si is not None:
                        delta = float(si - fi)
                        _safe_log(pl_module,
                            f"val_{spec.name}_delta_unconditional",
                            torch.tensor(delta, device=pl_module.device),
                            sync_dist=False,
                            on_step=False,
                            on_epoch=True,
                        )
        
        if first_valid and second_valid and first_smiles and second_smiles:
            try:
                with self._time("unconditional/distance_computation"):
                    if get_tanimoto_distance is not None:
                        tanimoto_dist = get_tanimoto_distance(first_smiles, second_smiles)
                        if tanimoto_dist is not None:
                            _safe_log(pl_module,
                                "val_tanimoto_distance_unconditional",
                                torch.tensor(tanimoto_dist, device=pl_module.device),
                                sync_dist=False,
                                on_step=False,
                                on_epoch=True,
                            )
            except Exception as exc:
                print(
                    "[Warning] Unconditional distance computation failed "
                    f"for first '{first_smiles}' / second '{second_smiles}': {exc}"
                )
                traceback.print_exc()
            
            # Log shape_tanimoto metric (matching regular tanimoto with sync_dist=False)
            # This is outside the try block to ensure it always executes, even if distance computation fails
            shape_tanimoto_dist_value = float('nan')
            if get_shape_tanimoto_distance is not None:
                try:
                    computed_dist = get_shape_tanimoto_distance(first_smiles, second_smiles, verbose=False)
                    if computed_dist is not None:
                        shape_tanimoto_dist_value = float(computed_dist)
                        shape_tanimoto_dist = computed_dist  # Set for return value
                except Exception as exc:
                    print(
                        "[Warning] Unconditional shape tanimoto computation failed "
                        f"for first '{first_smiles}' / second '{second_smiles}': {exc}"
                    )
                    traceback.print_exc()
            
            # Log with sync_dist=False to match regular tanimoto_distance_unconditional
            _safe_log(pl_module,
                "val_shape_tanimoto_distance_unconditional",
                torch.tensor(shape_tanimoto_dist_value, device=pl_module.device),
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        
        return SampleResult(
            tanimoto_dist=tanimoto_dist,
            shape_tanimoto_dist=shape_tanimoto_dist,
            seed_props=first_props,
            gen_props=second_props,
        )
    
    def check_memorization(self, first_smiles: Optional[str], second_smiles: Optional[str],
                          first_valid: bool, second_valid: bool):
        """
        Check if generated molecules are in training set (memorization check).
        Only called when "memorization" is in expensive_metrics_to_compute.
        
        Args:
            first_smiles: Canonical first molecule SMILES (can be None if invalid)
            second_smiles: Canonical second molecule SMILES (can be None if invalid)
            first_valid: Whether first molecule is valid
            second_valid: Whether second molecule is valid
        """
        if "memorization" not in self.expensive_metrics_to_compute:
            return
        
        with self._time("unconditional/memorization_check"):
            # Training-set memorization (backward-compatible)
            if self.training_canon_set:
                if first_valid and first_smiles:
                    self.epoch_first_mol_checked += 1
                    if first_smiles in self.training_canon_set:
                        self.epoch_first_mol_memorized += 1
                
                if second_valid and second_smiles:
                    self.epoch_second_mol_checked += 1
                    if second_smiles in self.training_canon_set:
                        self.epoch_second_mol_memorized += 1

            # Named reference memorization
            if self.reference_canon_sets_by_name:
                for ref_name, ref_set in self.reference_canon_sets_by_name.items():
                    if not ref_set:
                        continue
                    if first_valid and first_smiles:
                        self.epoch_first_mol_checked_by_ref[ref_name] = self.epoch_first_mol_checked_by_ref.get(ref_name, 0) + 1
                        if first_smiles in ref_set:
                            self.epoch_first_mol_memorized_by_ref[ref_name] = self.epoch_first_mol_memorized_by_ref.get(ref_name, 0) + 1
                    if second_valid and second_smiles:
                        self.epoch_second_mol_checked_by_ref[ref_name] = self.epoch_second_mol_checked_by_ref.get(ref_name, 0) + 1
                        if second_smiles in ref_set:
                            self.epoch_second_mol_memorized_by_ref[ref_name] = self.epoch_second_mol_memorized_by_ref.get(ref_name, 0) + 1
    
    def compute_and_log(self, registry):
        """
        Compute all unconditional metrics and register them.
        
        Args:
            registry: MetricRegistry to register metrics
        """
        gen_pool = self.epoch_uncond_valid_gen_smiles
        has_gen = len(gen_pool) > 0
        # Determine reference pools (named or legacy single)
        if self._use_named_references and self.reference_smiles_by_name:
            ref_pools_by_name: Dict[str, List[str]] = self.reference_smiles_by_name
        else:
            ref_pools_by_name = {"_default": (self.reference_smiles or [])}
        
        if not has_gen:
            print("[Info] No valid unconditional generations on this rank; logging NaNs for dataset metrics.")
        if self._use_named_references:
            if not any(len(v) > 0 for v in ref_pools_by_name.values()):
                print("[Info] No reference pools provided; logging NaNs for dataset-level metrics.")
        else:
            if not any(len(v) > 0 for v in ref_pools_by_name.values()):
                print("[Info] No reference pool provided; logging NaNs for dataset-level metrics.")
        
        def _register(metric_name: str, value: Optional[float],
                      description: str, category: str):
            display_value = math.nan if value is None else float(value)
            registry.register(metric_name, display_value, description, category)

        def _ref_metric(metric_name: str, ref_name: str) -> str:
            """
            If using named references, suffix reference-dependent metrics with /<ref_name>.
            Otherwise, keep legacy metric names unchanged.
            """
            if self._use_named_references and ref_name != "_default":
                return f"{metric_name}/{ref_name}"
            return metric_name
        
        # Validity metrics (always computed)
        # Always register as fractions for DDP sync (0/0 → NaN when no data)
        total_valid = self.epoch_first_valid_count + self.epoch_second_valid_count
        total_checked = self.epoch_total_checked * 2  # Each sample has 2 molecules

        registry.register_fraction("molecule_validity", total_valid, total_checked,
                                   "Overall validity accuracy (all molecules)", "Unconditional - Basic")
        registry.register_fraction("molecule_validity_first", self.epoch_first_valid_count, self.epoch_total_checked,
                                   "Validity accuracy for first molecules only", "Unconditional - Basic")
        registry.register_fraction("molecule_validity_second", self.epoch_second_valid_count, self.epoch_total_checked,
                                   "Validity accuracy for second molecules only", "Unconditional - Basic")
        
        # Internal diversity (standalone, no reference needed)
        if "internal_diversity" in self.expensive_metrics_to_compute:
            div_value = None
            n_valid = None
            if has_gen:
                try:
                    with self._time("unconditional/internal_diversity"):
                        div_value, n_valid, _ = pe_internal_diversity(gen_pool, deduplicate=True)
                except Exception as exc:
                    print(f"[Warning] Internal diversity computation failed: {exc}")
                    traceback.print_exc()
            _register("diversity_unconditional", div_value,
                      "Internal diversity", "Unconditional - Diversity")
            _register("diversity_n_pairs_unconditional", n_valid,
                      "Number of valid molecules for diversity", "Unconditional - Counts")

        # Bundled distribution metrics per reference pool (SNN, FCD, coverage, fragment/scaffold, W1/KDE)
        # Only call the expensive bundled computation when at least one
        # distributional metric is requested via expensive_metrics_to_compute.
        _distributional_gates = {"snn", "fcd", "fragment_scaffold", "w1_kde", "coverage"}
        _need_bundled = bool(_distributional_gates & set(self.expensive_metrics_to_compute))

        for ref_name, ref_pool in ref_pools_by_name.items():
            has_ref = len(ref_pool) > 0
            dist: Dict = {}
            if has_gen and has_ref and _need_bundled:
                try:
                    with self._time(f"unconditional/distribution/{ref_name}"):
                        dist = pe_compute_distribution_metrics(
                            gen_pool,
                            ref_pool,
                            min_generated=self.fcd_min_required,
                        )
                except Exception as exc:
                    print(f"[Warning] Distribution metrics failed for ref '{ref_name}': {exc}")
                    traceback.print_exc()

            # Map pairit_eval keys → pairit metric keys, gated on expensive_metrics_to_compute
            _key_map: List[Tuple[str, str, Optional[str]]] = []

            if "snn" in self.expensive_metrics_to_compute:
                _key_map.append(("snn_mean", "snn_mean_unconditional", "snn"))
                _key_map.append(("snn_median", "snn_median_unconditional", "snn"))

            if "fcd" in self.expensive_metrics_to_compute:
                _key_map.append(("fcd", "fcd_unconditional", "fcd"))

            if "fragment_scaffold" in self.expensive_metrics_to_compute:
                _key_map.append(("fragment_cosine", "fragment_cosine_unconditional", "fragment_scaffold"))
                _key_map.append(("scaffold_cosine", "scaffold_cosine_unconditional", "fragment_scaffold"))

            # Coverage is always computed
            _key_map.append(("coverage_ratio", "coverage_ratio_unconditional", None))

            if "w1_kde" in self.expensive_metrics_to_compute:
                _key_map.append(("w1_macro_mean", "feature_w1_macro_unconditional", "w1_kde"))
                _key_map.append(("w1_norm_macro_mean", "feature_w1_norm_macro_unconditional", "w1_kde"))
                _key_map.append(("kde_ll_delta_macro_mean", "feature_kde_delta_macro_unconditional", "w1_kde"))

            for pe_key, lobster_key, gate in _key_map:
                val = dist.get(pe_key)
                _register(
                    _ref_metric(lobster_key, ref_name),
                    float(val) if val is not None else None,
                    "",
                    "Unconditional - Distribution",
                )
        
        # NEW: Pair metrics (improvement stats, preservation, top-k) via base class
        pair_metrics = self._compute_pair_metrics("_unconditional")
        for name, value in pair_metrics.items():
            display = value if not math.isnan(value) else None
            _register(name, display, "", "Unconditional - Pair Metrics")

        # NEW: Uniqueness / novelty via base class
        uq_nov = self._compute_uniqueness_novelty("_unconditional")
        for name, value in uq_nov.items():
            display = value if not math.isnan(value) else None
            _register(name, display, "", "Unconditional - Uniqueness/Novelty")

        # Sample counts (always defined)
        _register(
            "uncond_valid_count",
            float(len(gen_pool)),
            "Number of valid unconditional molecules",
            "Unconditional - Counts",
        )

        # Memorization metrics — always register as fractions for DDP sync
        if "memorization" in self.expensive_metrics_to_compute:
            registry.register_fraction(
                "memorization_rate_unconditional_first",
                self.epoch_first_mol_memorized, self.epoch_first_mol_checked,
                f"Fraction of first molecules in training set ({self.epoch_first_mol_memorized}/{self.epoch_first_mol_checked})",
                "Unconditional - Memorization",
            )
            registry.register_fraction(
                "memorization_rate_unconditional_second",
                self.epoch_second_mol_memorized, self.epoch_second_mol_checked,
                f"Fraction of second molecules in training set ({self.epoch_second_mol_memorized}/{self.epoch_second_mol_checked})",
                "Unconditional - Memorization",
            )
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                c1 = int(self.epoch_first_mol_memorized_by_ref.get(ref_name, 0))
                n1 = int(self.epoch_first_mol_checked_by_ref.get(ref_name, 0))
                c2 = int(self.epoch_second_mol_memorized_by_ref.get(ref_name, 0))
                n2 = int(self.epoch_second_mol_checked_by_ref.get(ref_name, 0))
                registry.register_fraction(
                    f"memorization_rate_unconditional_first/{ref_name}",
                    c1, n1,
                    f"Fraction of first molecules in reference '{ref_name}' ({c1}/{n1})",
                    "Unconditional - Memorization",
                )
                registry.register_fraction(
                    f"memorization_rate_unconditional_second/{ref_name}",
                    c2, n2,
                    f"Fraction of second molecules in reference '{ref_name}' ({c2}/{n2})",
                    "Unconditional - Memorization",
                )
    
    def compute_and_return_dict(self) -> dict:
        """
        Compute metrics and return as dictionary (for evaluate()).
        
        Returns:
            Dict mapping metric names to values
        """
        gen_pool = self.epoch_uncond_valid_gen_smiles
        ref_pool = self.reference_smiles
        
        metrics = {}

        # Memorization metrics (computed from per-sample check_memorization calls,
        # independent of gen_pool which only tracks second molecules)
        if "memorization" in self.expensive_metrics_to_compute:
            if self.epoch_first_mol_checked > 0:
                metrics["memorization_rate_unconditional_first"] = (
                    self.epoch_first_mol_memorized / self.epoch_first_mol_checked
                )
            if self.epoch_second_mol_checked > 0:
                metrics["memorization_rate_unconditional_second"] = (
                    self.epoch_second_mol_memorized / self.epoch_second_mol_checked
                )
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                c1 = int(self.epoch_first_mol_memorized_by_ref.get(ref_name, 0))
                n1 = int(self.epoch_first_mol_checked_by_ref.get(ref_name, 0))
                c2 = int(self.epoch_second_mol_memorized_by_ref.get(ref_name, 0))
                n2 = int(self.epoch_second_mol_checked_by_ref.get(ref_name, 0))
                if n1 > 0:
                    metrics[f"memorization_rate_unconditional_first/{ref_name}"] = c1 / n1
                if n2 > 0:
                    metrics[f"memorization_rate_unconditional_second/{ref_name}"] = c2 / n2

        if not gen_pool:
            return metrics
        
        # Always compute simple metrics
        metrics["validity_fraction_unconditional"] = validity_fraction(gen_pool, sanity_fn=is_molecule_sane)
        uq, _ = uniqueness_fraction(gen_pool, sanity_fn=is_molecule_sane)
        metrics["uniqueness_unconditional"] = uq
        
        if not ref_pool:
            return metrics

        # Internal diversity (standalone)
        if "internal_diversity" in self.expensive_metrics_to_compute:
            try:
                div, n_valid, _ = pe_internal_diversity(gen_pool, deduplicate=True)
                metrics["diversity_unconditional"] = div
                metrics["diversity_n_pairs_unconditional"] = n_valid
            except Exception as exc:
                print(f"[Warning] (dict) Internal diversity computation failed: {exc}")
                traceback.print_exc()

        # Bundled distribution metrics via pairit_eval
        # Only call when at least one distributional metric is requested.
        _distributional_gates = {"snn", "fcd", "fragment_scaffold", "w1_kde", "coverage"}
        _need_bundled = bool(_distributional_gates & set(self.expensive_metrics_to_compute))
        dist: Dict = {}
        if _need_bundled:
            try:
                dist = pe_compute_distribution_metrics(
                    gen_pool, ref_pool, min_generated=self.fcd_min_required
                )
            except Exception as exc:
                print(f"[Warning] (dict) Distribution metrics computation failed: {exc}")
                traceback.print_exc()

        if "snn" in self.expensive_metrics_to_compute:
            if dist.get("snn_mean") is not None:
                metrics["snn_mean_unconditional"] = dist["snn_mean"]
                metrics["snn_median_unconditional"] = dist["snn_median"]

        if "fragment_scaffold" in self.expensive_metrics_to_compute:
            if dist.get("fragment_cosine") is not None:
                metrics["fragment_cosine_unconditional"] = dist["fragment_cosine"]
            if dist.get("scaffold_cosine") is not None:
                metrics["scaffold_cosine_unconditional"] = dist["scaffold_cosine"]

        if "w1_kde" in self.expensive_metrics_to_compute:
            if dist.get("w1_macro_mean") is not None:
                metrics["feature_w1_macro_unconditional"] = dist["w1_macro_mean"]
            if dist.get("w1_norm_macro_mean") is not None:
                metrics["feature_w1_norm_macro_unconditional"] = dist["w1_norm_macro_mean"]
            if dist.get("kde_ll_delta_macro_mean") is not None:
                metrics["feature_kde_delta_macro_unconditional"] = dist["kde_ll_delta_macro_mean"]

        if "fcd" in self.expensive_metrics_to_compute:
            if dist.get("fcd") is not None:
                metrics["fcd_unconditional"] = dist["fcd"]
        
        # NEW: Pair metrics (improvement stats, preservation, top-k) via base class
        pair_metrics = self._compute_pair_metrics("_unconditional")
        metrics.update(pair_metrics)

        # NEW: Uniqueness / novelty via base class
        uq_nov = self._compute_uniqueness_novelty("_unconditional")
        metrics.update(uq_nov)

        # Sample count
        metrics["uncond_valid_count"] = len(gen_pool)

        return metrics

