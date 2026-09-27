"""
Conditional Metrics Computation
================================

Computes and logs all metrics for conditional molecule generation.
"""

from contextlib import nullcontext
import math
import traceback
from typing import List, Dict, Optional, Set, Callable, Tuple, Sequence
import numpy as np
import torch
from atomic_datasets.utils.rdkit import is_molecule_sane
from rdkit import Chem
from rdkit.Chem import Descriptors

from pairit.eval import (
    PropertySpec,
    compute_properties,
    property_improvement_stats as pe_property_improvement_stats,
    property_preservation_mad as pe_property_preservation_mad,
    property_top_k_stats as pe_property_top_k_stats,
)

# Keep pairit-only functions (per-sample returns, no pairit_eval equivalent)
from pairit.callbacks.molecule_metrics import (
    is_improved as metrics_is_improved,
    neighborhood_alignment_proxy,
    property_improvement_under_similarity_constraint,
    similarities_to_seeds,
    mean_similarity_to_seeds,
    max_similarity_to_gt,
    validity_fraction,
    uniqueness_fraction,
    novelty_fraction,
    FEATURE_NAMES,
)

# Import for distance computation
try:
    from pair_gen.utils_mol import get_tanimoto_distance, get_shape_tanimoto_distance
except ImportError:
    get_tanimoto_distance = None
    get_shape_tanimoto_distance = None

# Import fingerprint-cached distance utilities
from .distance_utils import (
    compute_min_distance_to_set,
    compute_distances_with_fingerprint_caching,
)
from .base_metrics import (
    MoleculeMetricsComputer,
    SampleResult,
    _has_trainer,
    _safe_log,
    _safe_mean,
)
from .timing import TimingTracker


class ConditionalMetricsComputer(MoleculeMetricsComputer):
    """
    Computes all conditional generation metrics.

    Accumulates data during conditional generation, then computes and logs
    comprehensive metrics at the end of the epoch.
    """

    def __init__(self,
                 property_specs: List[PropertySpec],
                 distance_name: str = "tanimoto",
                 reference_canon_sets_by_name: Optional[Dict[str, Set[str]]] = None,
                 default_reference_name: Optional[str] = None,
                 distance_fn: Optional[Callable] = None,
                 neighbor_provider: Optional[Callable] = None,
                 expensive_metrics_to_compute: Optional[List[str]] = None,
                 timing_tracker: Optional[TimingTracker] = None,
                 distance_percentile_fn: Optional[Callable[[str, float], Optional[float]]] = None,
                 check_memorization_from_any_reference: bool = True,
                 reference_smiles_by_name: Optional[Dict[str, List[str]]] = None,
                 fcd_min_required: int = 10,
                 metric_prefix: str = "val"):
        """
        Initialize metrics computer.

        Args:
            property_specs: List of property specifications for evaluation
            distance_name: Name of distance metric (for logging)
            reference_canon_sets_by_name: Named canonical SMILES sets for memorization/novelty
            default_reference_name: Primary reference set name
            distance_fn: Function to compute distances between molecules
            neighbor_provider: Function to provide neighbors for a molecule
            expensive_metrics_to_compute: List of expensive metrics to compute
            timing_tracker: TimingTracker for logging fine-grained timings
            distance_percentile_fn: Optional function to compute distance percentile given (seed_smiles, distance) -> percentile
            check_memorization_from_any_reference: If True, memorization checks union of all reference sets
            reference_smiles_by_name: Named reference SMILES pools for distributional metrics
            fcd_min_required: Minimum samples for FCD computation
            metric_prefix: Prefix for per-sample logged metrics ("val" during training, "" during eval)
        """
        super().__init__(
            property_specs=property_specs,
            distance_name=distance_name,
            reference_canon_sets_by_name=reference_canon_sets_by_name,
            default_reference_name=default_reference_name,
            reference_smiles_by_name=reference_smiles_by_name,
            distance_fn=distance_fn,
            expensive_metrics_to_compute=expensive_metrics_to_compute if expensive_metrics_to_compute is not None else ["snn", "fcd", "internal_diversity", "fragment_scaffold", "w1_kde"],
            timing_tracker=timing_tracker,
            check_memorization_from_any_reference=check_memorization_from_any_reference,
            fcd_min_required=fcd_min_required,
            metric_prefix=metric_prefix,
        )
        self.neighbor_provider = neighbor_provider
        self.distance_percentile_fn = distance_percentile_fn

        # Reset buffers (conditional-specific)
        self._reset_conditional()
    
    def reset(self):
        """Reset all accumulation buffers (base + conditional)."""
        self._reset_base()
        self._reset_conditional()

    def _reset_conditional(self):
        """Reset conditional-specific accumulation buffers."""
        # Distance and neighborhood metrics
        self.epoch_dist_to_gt: List[float] = []
        self.epoch_gt_min_delta: List[float] = []
        self.epoch_neigh_median_dist: List[float] = []
        self.epoch_neigh_p75_dist: List[float] = []
        self.epoch_neigh_nn_dist: List[float] = []
        
        # Distributional alignment
        self.epoch_kde_ll_in: List[float] = []
        self.epoch_kde_ll_gen: List[float] = []
        self.epoch_w1_deltas: Dict[str, List[float]] = {k: [] for k in FEATURE_NAMES}
        
        # GT-related distances
        self.epoch_input_to_gt_min_dist: List[float] = []  # min distance(seed -> GT)
        self.epoch_gen_to_gt_min_dist: List[float] = []    # min distance(gen  -> GT)

        # Seed-to-gen distance (conditional generation distance vs seed)
        self.epoch_seed_to_gen_dist: List[float] = []
        
        # Distance percentiles (for aggregate statistics)
        self.epoch_distance_percentiles: List[float] = []

        # GT lists (conditional-specific, not in base)
        self.epoch_cond_gt_lists: List[List[str]] = []
        
        # Memorization tracking (expensive metric)
        self.epoch_memorized_count: int = 0  # Count of valid gens in training set
        self.epoch_memorized_checked: int = 0  # Count of valid gens checked
        # Additional named memorization reference sets (canonical)
        self.epoch_memorized_count_by_ref: Dict[str, int] = {k: 0 for k in self.reference_canon_sets_by_name.keys()}
        self.epoch_memorized_checked_by_ref: Dict[str, int] = {k: 0 for k in self.reference_canon_sets_by_name.keys()}
        
        # Prompt agreement tracking (per property)
        # Maps property_name -> list of (direction_match, magnitude_match) tuples
        self.epoch_prompt_direction_agreement: Dict[str, List[bool]] = {}
        self.epoch_prompt_magnitude_agreement: Dict[str, List[bool]] = {}
        # Strict direction agreement: excludes "any" direction samples (which trivially match)
        self.epoch_prompt_direction_agreement_strict: Dict[str, List[bool]] = {}
        self.epoch_prompt_direction_memorized_strict: Dict[str, List[bool]] = {}
        
        # Memorization status per sample (for filtering prompt agreement metrics)
        # Maps property_name -> list of is_memorized flags (parallel to direction/magnitude lists)
        self.epoch_prompt_direction_memorized: Dict[str, List[bool]] = {}
        self.epoch_prompt_magnitude_memorized: Dict[str, List[bool]] = {}
        
        # Per-reference memorization status for prompt agreement filtering
        # Maps ref_name -> property_name -> list of is_memorized_in_ref flags
        self.epoch_prompt_direction_memorized_by_ref: Dict[str, Dict[str, List[bool]]] = {
            ref_name: {} for ref_name in self.reference_canon_sets_by_name.keys()
        }
        self.epoch_prompt_magnitude_memorized_by_ref: Dict[str, Dict[str, List[bool]]] = {
            ref_name: {} for ref_name in self.reference_canon_sets_by_name.keys()
        }
        
        # Bin tracking for random baseline computation (per property)
        # Maps property_name -> list of actual/prompted delta bins
        self.epoch_prompt_actual_bins: Dict[str, List[int]] = {}
        self.epoch_prompt_prompted_bins: Dict[str, List[int]] = {}

        # Bin error tracking (per property)
        # Maps property_name -> list of bin errors (actual_bin - prompt_bin)
        self.epoch_prompt_magnitude_bin_errors: Dict[str, List[int]] = {}
        self.epoch_prompt_magnitude_bin_errors_memorized: Dict[str, List[bool]] = {}  # Parallel list for filtering
        self.epoch_prompt_magnitude_bin_errors_memorized_by_ref: Dict[str, Dict[str, List[bool]]] = {
            ref_name: {} for ref_name in self.reference_canon_sets_by_name.keys()
        }
        
        # Distance prompt agreement tracking
        self.epoch_distance_prompt_agreement: List[bool] = []
        self.epoch_distance_prompt_memorized: List[bool] = []  # Parallel list for filtering
        self.epoch_distance_prompt_memorized_by_ref: Dict[str, List[bool]] = {
            ref_name: [] for ref_name in self.reference_canon_sets_by_name.keys()
        }
        
        # All properties agree tracking (per sample)
        # True if ALL prompted properties agree for this sample
        self.epoch_all_properties_direction_agree: List[bool] = []
        self.epoch_all_properties_magnitude_agree: List[bool] = []
        self.epoch_all_properties_direction_memorized: List[bool] = []  # Parallel list for filtering
        self.epoch_all_properties_magnitude_memorized: List[bool] = []  # Parallel list for filtering
        self.epoch_all_properties_direction_memorized_by_ref: Dict[str, List[bool]] = {
            ref_name: [] for ref_name in self.reference_canon_sets_by_name.keys()
        }
        self.epoch_all_properties_magnitude_memorized_by_ref: Dict[str, List[bool]] = {
            ref_name: [] for ref_name in self.reference_canon_sets_by_name.keys()
        }
        
        # Tracking for valid+improved+not_memorized samples (per property)
        # Maps property_name -> list of bool flags (parallel to epoch_seed_to_gen_dist)
        self.epoch_seed_to_gen_V_N_I_flags: Dict[str, List[bool]] = {}
        # Maps property_name -> list of bool flags (parallel to epoch_input_to_gt_min_dist)
        self.epoch_input_to_gt_V_N_I_flags: Dict[str, List[bool]] = {}

    # Backward-compatible aliases for renamed buffers
    @property
    def epoch_cond_seeds(self) -> List[str]:
        return self.epoch_seeds

    @property
    def epoch_cond_gens(self) -> List[str]:
        return self.epoch_gens

    @property
    def epoch_cond_all_gens(self) -> List[str]:
        return self.epoch_all_gens

    @property
    def epoch_valid_cond_gen_canon(self) -> List[str]:
        return self.epoch_valid_canon_gens

    def process_sample(
        self,
        pl_module,
        seed_smiles: str,
        gen_smiles: str,
        is_valid: bool,
        canonical_gen: Optional[str] = None,
        gt_candidates: Optional[List[str]] = None,
        neighbors_smiles: Optional[List[str]] = None,
        input_smiles1: Optional[str] = None,
        input_smiles2: Optional[str] = None,
        dataset_key: Optional[str] = None,
        parsed_prompts: Optional[Sequence] = None,
        value_binning_by_property: Optional[Dict[str, any]] = None,
        delta_binning_by_property: Optional[Dict[str, any]] = None,
        value_binning_default: Optional[any] = None,
        delta_binning_default: Optional[any] = None,
        distance_prompt_bin: Optional[int] = None,
        distance_binning_mode: Optional[str] = None,
        distance_num_bins: Optional[int] = None,
    ) -> SampleResult:
        """
        Process a single sample: compute metrics, log per-sample values, accumulate for aggregates.
        
        This is the MAIN method that does ALL per-sample computation and logging.
        The main callback should just call this with raw SMILES data.
        
        Args:
            pl_module: PyTorch Lightning module for logging
            seed_smiles: Input/conditioning SMILES
            gen_smiles: Generated SMILES  
            is_valid: Whether generated molecule is RDKit-valid
            canonical_gen: Canonical form of generated SMILES
            gt_candidates: Optional list of ground truth molecules
            neighbors_smiles: Optional list of neighbor molecules
            input_smiles1: Optional first SMILES from validation pair (for input pair distance)
            input_smiles2: Optional second SMILES from validation pair (for input pair distance)
            
        Returns:
            Tuple of (valid_and_improved dict, tanimoto_dist, shape_tanimoto_dist, input_tanimoto_dist, input_shape_tanimoto_dist, input_props, gen_props, is_memorized, distance_percentile, prompt_agreement)
            Distance values and properties are None if not computed. is_memorized is None if not enabled or not applicable.
            distance_percentile is None if not computed or distance_percentile_fn is not available.
            prompt_agreement is a dict mapping property_name -> {'direction_match': bool, 'magnitude_match': bool} or None if not computed.
        """
        def _name(metric_name: str) -> str:
            """
            In mixed validation, namespace per-sample logs by dataset_key so metrics
            don't collide across multiple val dataloaders.
            """
            if not dataset_key:
                return metric_name
            if metric_name.startswith("val_"):
                return f"val/{dataset_key}/{metric_name[4:]}"
            return metric_name

        valid_and_improved = {}
        input_props: Optional[Dict[str, float]] = None
        gen_props: Optional[Dict[str, float]] = None
        prompt_agreement: Optional[Dict[str, Dict[str, bool]]] = None  # Initialize early to avoid UnboundLocalError
        
        # 1. Compute and log properties
        if seed_smiles and gen_smiles:
            with self._time("conditional/property_computation"):
                input_props = compute_properties(seed_smiles, self.property_specs)
                gen_props = compute_properties(gen_smiles, self.property_specs)
            
            # Log per-sample properties
            if input_props:
                for spec in self.property_specs:
                    if (val := input_props.get(spec.name)) is not None:
                        _safe_log(pl_module,
                            _name(f"val_{spec.name}_conditional_input"),
                            torch.tensor(float(val), device=pl_module.device),
                            sync_dist=False,  # per-sample logging; aggregate explicitly later
                            on_step=False,
                            on_epoch=True,
                        )
            
            if gen_props:
                for spec in self.property_specs:
                    if (val := gen_props.get(spec.name)) is not None:
                        _safe_log(pl_module,
                            _name(f"val_{spec.name}_conditional_generated"),
                            torch.tensor(float(val), device=pl_module.device),
                            sync_dist=False,
                            on_step=False,
                            on_epoch=True,
                        )
            
            # Log property deltas
            if input_props and gen_props:
                for spec in self.property_specs:
                    bi, gi = input_props.get(spec.name), gen_props.get(spec.name)
                    if bi is not None and gi is not None:
                        delta = float(gi - bi)
                        _safe_log(pl_module,
                            _name(f"val_{spec.name}_delta_conditional"),
                            torch.tensor(delta, device=pl_module.device),
                            sync_dist=False,
                            on_step=False,
                            on_epoch=True,
                        )
                        
                        # Check if improved (for return value)
                        if is_valid:
                            valid_and_improved[spec.name] = metrics_is_improved(spec, bi, gi)
            
            # Compute prompt agreement metrics (directionality and magnitude)
            if parsed_prompts and input_props and gen_props:
                try:
                    from pairit.prompting.property_prompting import delta_to_bin
                    
                    prompt_agreement = {}
                    for prompt in parsed_prompts:
                        prop_name = prompt.property_name
                        if prop_name not in input_props or prop_name not in gen_props:
                            continue
                        
                        seed_val = input_props.get(prop_name)
                        gen_val = gen_props.get(prop_name)
                        if seed_val is None or gen_val is None:
                            continue
                        
                        actual_delta = float(gen_val - seed_val)
                        direction_match = None
                        magnitude_match = None
                        
                        # Check directionality agreement
                        if prompt.direction is not None:
                            if prompt.direction == "up":
                                direction_match = actual_delta > 0.0
                            elif prompt.direction == "down":
                                direction_match = actual_delta < 0.0
                            elif prompt.direction == "any":
                                direction_match = True  # "any" always matches

                            if direction_match is not None:
                                # Initialize lists if needed
                                if prop_name not in self.epoch_prompt_direction_agreement:
                                    self.epoch_prompt_direction_agreement[prop_name] = []
                                    self.epoch_prompt_direction_memorized[prop_name] = []
                                    # Also initialize per-reference parallel lists
                                    for ref_name in self.epoch_prompt_direction_memorized_by_ref:
                                        self.epoch_prompt_direction_memorized_by_ref[ref_name][prop_name] = []

                                # Store direction match (memorization status will be added later)
                                self.epoch_prompt_direction_agreement[prop_name].append(direction_match)
                                # Placeholder for memorization - will be updated when memorization is checked
                                self.epoch_prompt_direction_memorized[prop_name].append(False)
                                # Append placeholder to each per-reference list
                                for ref_name in self.epoch_prompt_direction_memorized_by_ref:
                                    self.epoch_prompt_direction_memorized_by_ref[ref_name][prop_name].append(False)

                                # Strict tracking: only up/down (excludes "any" which trivially matches)
                                if prompt.direction in ("up", "down"):
                                    if prop_name not in self.epoch_prompt_direction_agreement_strict:
                                        self.epoch_prompt_direction_agreement_strict[prop_name] = []
                                        self.epoch_prompt_direction_memorized_strict[prop_name] = []
                                    self.epoch_prompt_direction_agreement_strict[prop_name].append(direction_match)
                                    self.epoch_prompt_direction_memorized_strict[prop_name].append(False)
                                
                                # Log per-sample directionality agreement
                                _safe_log(pl_module,
                                    _name(f"val_{prop_name}_prompt_direction_match"),
                                    torch.tensor(1.0 if direction_match else 0.0, device=pl_module.device),
                                    sync_dist=False,
                                    on_step=False,
                                    on_epoch=True,
                                )
                        
                        # Check magnitude agreement (if delta_bin is available)
                        if prompt.delta_bin is not None and delta_binning_by_property is not None:
                            # Get delta binning config
                            delta_binning = delta_binning_by_property.get(prop_name, delta_binning_default)
                            if delta_binning is not None:
                                try:
                                    actual_delta_bin = delta_to_bin(binning=delta_binning, delta=actual_delta, magnitude=True)
                                    # Respect delta_mode for bounded prompts
                                    delta_mode = getattr(prompt, "delta_mode", "exact")
                                    if delta_mode == "atmost":
                                        magnitude_match = (actual_delta_bin <= prompt.delta_bin)
                                    elif delta_mode == "atleast":
                                        magnitude_match = (actual_delta_bin >= prompt.delta_bin)
                                    else:  # exact
                                        magnitude_match = (actual_delta_bin == prompt.delta_bin)
                                    
                                    # Initialize lists if needed
                                    if prop_name not in self.epoch_prompt_magnitude_agreement:
                                        self.epoch_prompt_magnitude_agreement[prop_name] = []
                                        self.epoch_prompt_magnitude_memorized[prop_name] = []
                                        self.epoch_prompt_magnitude_bin_errors[prop_name] = []
                                        self.epoch_prompt_magnitude_bin_errors_memorized[prop_name] = []
                                        self.epoch_prompt_actual_bins[prop_name] = []
                                        self.epoch_prompt_prompted_bins[prop_name] = []
                                        # Also initialize per-reference parallel lists
                                        for ref_name in self.epoch_prompt_magnitude_memorized_by_ref:
                                            self.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop_name] = []
                                        for ref_name in self.epoch_prompt_magnitude_bin_errors_memorized_by_ref:
                                            self.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop_name] = []

                                    # Store magnitude match (memorization status will be added later)
                                    self.epoch_prompt_magnitude_agreement[prop_name].append(magnitude_match)
                                    # Placeholder for memorization - will be updated when memorization is checked
                                    self.epoch_prompt_magnitude_memorized[prop_name].append(False)
                                    # Append placeholder to each per-reference list
                                    for ref_name in self.epoch_prompt_magnitude_memorized_by_ref:
                                        self.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop_name].append(False)

                                    # Store bins for random baseline computation
                                    self.epoch_prompt_actual_bins[prop_name].append(actual_delta_bin)
                                    self.epoch_prompt_prompted_bins[prop_name].append(prompt.delta_bin)

                                    # Store bin error (actual_bin - prompt_bin)
                                    bin_error = actual_delta_bin - prompt.delta_bin
                                    self.epoch_prompt_magnitude_bin_errors[prop_name].append(bin_error)
                                    # Placeholder for memorization - will be updated when memorization is checked
                                    self.epoch_prompt_magnitude_bin_errors_memorized[prop_name].append(False)
                                    for ref_name in self.epoch_prompt_magnitude_bin_errors_memorized_by_ref:
                                        self.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop_name].append(False)
                                    
                                    # Log per-sample magnitude agreement
                                    _safe_log(pl_module,
                                        _name(f"val_{prop_name}_prompt_magnitude_match"),
                                        torch.tensor(1.0 if magnitude_match else 0.0, device=pl_module.device),
                                        sync_dist=False,
                                        on_step=False,
                                        on_epoch=True,
                                    )
                                    
                                    # Log per-sample bin error
                                    _safe_log(pl_module,
                                        _name(f"val_{prop_name}_prompt_magnitude_bin_error"),
                                        torch.tensor(float(bin_error), device=pl_module.device),
                                        sync_dist=False,
                                        on_step=False,
                                        on_epoch=True,
                                    )
                                except Exception:
                                    # If binning fails, skip magnitude check
                                    pass
                        
                        prompt_agreement[prop_name] = {
                            'direction_match': direction_match,
                            'magnitude_match': magnitude_match,
                        }
                    
                    # Check if ALL properties agree for this sample
                    if prompt_agreement:
                        # Check direction: all properties with direction prompts must agree
                        all_direction_agree = True
                        direction_prompts_count = 0
                        for prop_name, agreement in prompt_agreement.items():
                            if agreement['direction_match'] is not None:
                                direction_prompts_count += 1
                                if not agreement['direction_match']:
                                    all_direction_agree = False
                                    break
                        
                        # Only record if there were any direction prompts
                        if direction_prompts_count > 0:
                            self.epoch_all_properties_direction_agree.append(all_direction_agree)
                            self.epoch_all_properties_direction_memorized.append(False)
                            for ref_name in self.epoch_all_properties_direction_memorized_by_ref:
                                self.epoch_all_properties_direction_memorized_by_ref[ref_name].append(False)

                        # Check magnitude: all properties with magnitude prompts must agree
                        all_magnitude_agree = True
                        magnitude_prompts_count = 0
                        for prop_name, agreement in prompt_agreement.items():
                            if agreement['magnitude_match'] is not None:
                                magnitude_prompts_count += 1
                                if not agreement['magnitude_match']:
                                    all_magnitude_agree = False
                                    break

                        # Only record if there were any magnitude prompts
                        if magnitude_prompts_count > 0:
                            self.epoch_all_properties_magnitude_agree.append(all_magnitude_agree)
                            self.epoch_all_properties_magnitude_memorized.append(False)
                            for ref_name in self.epoch_all_properties_magnitude_memorized_by_ref:
                                self.epoch_all_properties_magnitude_memorized_by_ref[ref_name].append(False)
                            
                except Exception as exc:
                    print(
                        f"[Warning] Prompt agreement computation failed "
                        f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
                    )
                    import traceback
                    traceback.print_exc()
                    prompt_agreement = None

                # Debug printouts for logP specifically
                try:
                    if (lp_seed := input_props.get("logp")) is not None and (lp_gen := gen_props.get("logp")) is not None:
                        lp_delta = float(lp_gen) - float(lp_seed)
                        print(f"[ConditionalLogP] seed={lp_seed:.6f} gen={lp_gen:.6f} delta={lp_delta:.6f} | seed_smiles={seed_smiles} | gen_smiles={gen_smiles}")
                except Exception as exc:
                    print(
                        "[Warning] Failed to log conditional logP debug output "
                        f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
                    )
                    traceback.print_exc()
        
        # 2. Compute and log distance to GT (using fingerprint caching)
        if is_valid and seed_smiles and gen_smiles and gt_candidates:
            try:
                # Use cached distance computation to avoid redundant fingerprinting
                with self._time("conditional/distance_to_gt"):
                    d_gen_min = compute_min_distance_to_set(gen_smiles, gt_candidates)
                    d_inp_min = compute_min_distance_to_set(seed_smiles, gt_candidates)
                
                # Accumulate for epoch summary
                self.epoch_dist_to_gt.append(d_gen_min)
                self.epoch_gt_min_delta.append(d_gen_min - d_inp_min)
                self.epoch_input_to_gt_min_dist.append(d_inp_min)
                self.epoch_gen_to_gt_min_dist.append(d_gen_min)
                
                # Initialize V_N_I flags for input_to_gt (will be updated after memorization check)
                for spec in self.property_specs:
                    if spec.name not in self.epoch_input_to_gt_V_N_I_flags:
                        self.epoch_input_to_gt_V_N_I_flags[spec.name] = []
                    # Placeholder - will be updated after memorization is checked
                    self.epoch_input_to_gt_V_N_I_flags[spec.name].append(False)
                
                # Log per-sample
                _safe_log(pl_module,
                    _name(f"val_{self.distance_name}_distance_to_gt_conditional"),
                    torch.tensor(d_gen_min, device=pl_module.device),
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                _safe_log(pl_module,
                    _name(f"val_{self.distance_name}_distance_input_to_gt_conditional"),
                    torch.tensor(d_inp_min, device=pl_module.device),
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                _safe_log(pl_module,
                    _name(f"val_{self.distance_name}_gt_closest_delta_conditional"),
                    torch.tensor(d_gen_min - d_inp_min, device=pl_module.device),
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
            except Exception as exc:
                print(
                    "[Warning] Conditional distance-to-GT computation failed "
                    f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
                )
                traceback.print_exc()
        
        # 3. Compute neighborhood metrics (using fingerprint caching)
        if is_valid and seed_smiles and gen_smiles:
            neighs = self._get_neighbors(seed_smiles, neighbors_smiles)
            if neighs:
                # Use cached distance computation to avoid redundant fingerprinting
                with self._time("conditional/neighborhood_distance"):
                    dists = compute_distances_with_fingerprint_caching(gen_smiles, neighs)
                # Filter out invalid distances
                dists = [d for d in dists if d is not None and np.isfinite(d)]
                
                if dists:
                    arr = np.asarray(dists, dtype=float)
                    self.epoch_neigh_median_dist.append(float(np.median(arr)))
                    self.epoch_neigh_p75_dist.append(float(np.percentile(arr, 75)))
                    self.epoch_neigh_nn_dist.append(float(np.min(arr)))
                
                # 4. Distributional alignment (neighborhood proxy)
                try:
                    with self._time("conditional/neighborhood_alignment_proxy"):
                        deltas_list, ll_in_list, ll_gen_list = neighborhood_alignment_proxy(
                            [neighs], [seed_smiles], [gen_smiles]
                        )
                    deltas = deltas_list[0] if deltas_list else {}
                    for k, dv in deltas.items():
                        self.epoch_w1_deltas[k].append(-float(dv))
                    if ll_in_list and ll_in_list[0] is not None:
                        self.epoch_kde_ll_in.append(float(ll_in_list[0]))
                    if ll_gen_list and ll_gen_list[0] is not None:
                        self.epoch_kde_ll_gen.append(float(ll_gen_list[0]))
                except Exception as exc:
                    print(
                        "[Warning] Conditional neighborhood alignment proxy failed "
                        f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
                    )
                    traceback.print_exc()
        
        # 5. Accumulate for epoch-level metrics
        self.accumulate_sample(seed_smiles, gen_smiles, is_valid, canonical_gen, gt_candidates)
        
        # 5b. Check memorization (expensive metric) and capture result
        is_memorized: Optional[bool] = None
        if "memorization" in self.expensive_metrics_to_compute:
            if is_valid and canonical_gen:
                if self.check_memorization_from_any_reference:
                    # Check if molecule is in ANY of the reference sets
                    if self.reference_canon_sets_by_name:
                        is_memorized = any(
                            canonical_gen in ref_set 
                            for ref_set in self.reference_canon_sets_by_name.values()
                            if ref_set
                        )
                    else:
                        is_memorized = False
                else:
                    # Check only the default reference set (backward-compatible behavior)
                    if self.training_canon_set is not None:
                        is_memorized = canonical_gen in self.training_canon_set
                
                # Update internal tracking (always tracks default reference)
                self.check_memorization(canonical_gen, is_valid)
            # Also track memorization against any named reference sets (if provided)
            self.check_memorization_named_refs(canonical_gen, is_valid)
        
        # Update memorization status in prompt agreement lists (if prompt agreement was computed)
        if is_memorized is not None and prompt_agreement is not None:
            for prop_name in prompt_agreement.keys():
                # Update the last entry in direction memorization list
                if prop_name in self.epoch_prompt_direction_memorized and len(self.epoch_prompt_direction_memorized[prop_name]) > 0:
                    self.epoch_prompt_direction_memorized[prop_name][-1] = is_memorized
                # Update strict direction memorization list
                if prop_name in self.epoch_prompt_direction_memorized_strict and len(self.epoch_prompt_direction_memorized_strict[prop_name]) > 0:
                    self.epoch_prompt_direction_memorized_strict[prop_name][-1] = is_memorized
                # Update the last entry in magnitude memorization list
                if prop_name in self.epoch_prompt_magnitude_memorized and len(self.epoch_prompt_magnitude_memorized[prop_name]) > 0:
                    self.epoch_prompt_magnitude_memorized[prop_name][-1] = is_memorized
                # Update the last entry in bin errors memorization list
                if prop_name in self.epoch_prompt_magnitude_bin_errors_memorized and len(self.epoch_prompt_magnitude_bin_errors_memorized[prop_name]) > 0:
                    self.epoch_prompt_magnitude_bin_errors_memorized[prop_name][-1] = is_memorized
            
            # Update memorization status for "all properties agree" flags
            # Check if this sample had direction prompts
            direction_prompts_count = sum(1 for agreement in prompt_agreement.values() if agreement.get('direction_match') is not None)
            if direction_prompts_count > 0 and len(self.epoch_all_properties_direction_memorized) > 0:
                self.epoch_all_properties_direction_memorized[-1] = is_memorized
            
            # Check if this sample had magnitude prompts
            magnitude_prompts_count = sum(1 for agreement in prompt_agreement.values() if agreement.get('magnitude_match') is not None)
            if magnitude_prompts_count > 0 and len(self.epoch_all_properties_magnitude_memorized) > 0:
                self.epoch_all_properties_magnitude_memorized[-1] = is_memorized
        
        # Update memorization status for distance prompt agreement
        if is_memorized is not None and len(self.epoch_distance_prompt_agreement) > 0:
            # Ensure the memorization list has the same length as the agreement list
            while len(self.epoch_distance_prompt_memorized) < len(self.epoch_distance_prompt_agreement):
                self.epoch_distance_prompt_memorized.append(False)
            # Update the last entry
            if len(self.epoch_distance_prompt_memorized) == len(self.epoch_distance_prompt_agreement):
                self.epoch_distance_prompt_memorized[-1] = is_memorized
        
        # Update per-reference memorization status (for each named reference set)
        if is_valid and canonical_gen and prompt_agreement is not None and self.reference_canon_sets_by_name:
            # Recalculate direction/magnitude prompt counts
            direction_prompts_count = sum(1 for agreement in prompt_agreement.values() if agreement.get('direction_match') is not None)
            magnitude_prompts_count = sum(1 for agreement in prompt_agreement.values() if agreement.get('magnitude_match') is not None)
            
            for ref_name, ref_set in self.reference_canon_sets_by_name.items():
                if not ref_set:
                    continue
                is_mem_in_ref = canonical_gen in ref_set
                
                # Update per-reference memorization for each property's direction/magnitude tracking
                for prop_name in prompt_agreement.keys():
                    # Direction memorization
                    if prop_name in self.epoch_prompt_direction_memorized_by_ref.get(ref_name, {}):
                        if len(self.epoch_prompt_direction_memorized_by_ref[ref_name][prop_name]) > 0:
                            self.epoch_prompt_direction_memorized_by_ref[ref_name][prop_name][-1] = is_mem_in_ref
                    
                    # Magnitude memorization
                    if prop_name in self.epoch_prompt_magnitude_memorized_by_ref.get(ref_name, {}):
                        if len(self.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop_name]) > 0:
                            self.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop_name][-1] = is_mem_in_ref
                    
                    # Bin errors memorization
                    if prop_name in self.epoch_prompt_magnitude_bin_errors_memorized_by_ref.get(ref_name, {}):
                        if len(self.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop_name]) > 0:
                            self.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop_name][-1] = is_mem_in_ref
                
                # Update "all properties agree" per-reference tracking
                if direction_prompts_count > 0 and len(self.epoch_all_properties_direction_memorized_by_ref.get(ref_name, [])) > 0:
                    self.epoch_all_properties_direction_memorized_by_ref[ref_name][-1] = is_mem_in_ref
                
                if magnitude_prompts_count > 0 and len(self.epoch_all_properties_magnitude_memorized_by_ref.get(ref_name, [])) > 0:
                    self.epoch_all_properties_magnitude_memorized_by_ref[ref_name][-1] = is_mem_in_ref
                
                # Update distance prompt per-reference tracking
                if len(self.epoch_distance_prompt_agreement) > 0:
                    while len(self.epoch_distance_prompt_memorized_by_ref[ref_name]) < len(self.epoch_distance_prompt_agreement):
                        self.epoch_distance_prompt_memorized_by_ref[ref_name].append(False)
                    if len(self.epoch_distance_prompt_memorized_by_ref[ref_name]) == len(self.epoch_distance_prompt_agreement):
                        self.epoch_distance_prompt_memorized_by_ref[ref_name][-1] = is_mem_in_ref
        
        # Update V_N_I flags after memorization is known
        # For each property, check if this sample is valid+improved+not_memorized
        if is_valid and is_memorized is not None and valid_and_improved:
            is_not_memorized = not is_memorized
            for spec in self.property_specs:
                prop_name = spec.name
                is_improved = valid_and_improved.get(prop_name, False)
                v_n_i_flag = is_valid and is_improved and is_not_memorized
                
                # Update seed_to_gen flags (if this sample has a seed_to_gen distance)
                if prop_name in self.epoch_seed_to_gen_V_N_I_flags and len(self.epoch_seed_to_gen_V_N_I_flags[prop_name]) > 0:
                    # Update the last entry (most recently appended)
                    self.epoch_seed_to_gen_V_N_I_flags[prop_name][-1] = v_n_i_flag
                
                # Update input_to_gt flags (if this sample has an input_to_gt distance)
                if prop_name in self.epoch_input_to_gt_V_N_I_flags and len(self.epoch_input_to_gt_V_N_I_flags[prop_name]) > 0:
                    # Update the last entry (most recently appended)
                    self.epoch_input_to_gt_V_N_I_flags[prop_name][-1] = v_n_i_flag

        # 6. Seed-to-generated distance (conditional generation distance vs seed)
        tanimoto_dist = None
        shape_tanimoto_dist = None
        distance_percentile = None
        try:
            if is_valid and seed_smiles and gen_smiles:
                with self._time("conditional/seed_to_gen_distance"):
                    d_seed_gen = float(self.distance_fn(seed_smiles, gen_smiles))
                    if np.isfinite(d_seed_gen):
                        self.epoch_seed_to_gen_dist.append(d_seed_gen)
                        tanimoto_dist = d_seed_gen
                        
                        # Initialize V_N_I flags for seed_to_gen (will be updated after memorization check)
                        for spec in self.property_specs:
                            if spec.name not in self.epoch_seed_to_gen_V_N_I_flags:
                                self.epoch_seed_to_gen_V_N_I_flags[spec.name] = []
                            # Placeholder - will be updated after memorization is checked
                            self.epoch_seed_to_gen_V_N_I_flags[spec.name].append(False)
                        
                        # Log per-sample seed→gen Tanimoto distance
                        _safe_log(pl_module,
                            _name("val_tanimoto_distance_conditional"),
                            torch.tensor(d_seed_gen, device=pl_module.device),
                            sync_dist=False,
                            on_step=False,
                            on_epoch=True,
                        )
                        
                        # Compute distance percentile if function is available
                        if self.distance_percentile_fn is not None:
                            try:
                                distance_percentile = self.distance_percentile_fn(seed_smiles, d_seed_gen)
                                if distance_percentile is not None and np.isfinite(distance_percentile):
                                    # Accumulate for epoch-level aggregation
                                    self.epoch_distance_percentiles.append(distance_percentile)
                                    
                                    # Log distance percentile to wandb
                                    _safe_log(pl_module,
                                        _name(f"val_{self.distance_name}_distance_percentile_conditional"),
                                        torch.tensor(distance_percentile, device=pl_module.device),
                                        sync_dist=False,
                                        on_step=False,
                                        on_epoch=True,
                                    )
                                    
                                    # Check distance prompt agreement if prompt bin is available
                                    if distance_prompt_bin is not None and distance_binning_mode is not None and distance_num_bins is not None:
                                        try:
                                            from pairit.prompting.distance_prompting import percentile_to_bin, DistanceBinningMode
                                            
                                            # Convert actual percentile to bin
                                            actual_bin = percentile_to_bin(
                                                percentile=distance_percentile,
                                                mode=distance_binning_mode,  # type: ignore
                                                num_bins=distance_num_bins,
                                            )
                                            
                                            # Check if bins match
                                            bin_match = (actual_bin == distance_prompt_bin)
                                            
                                            # Accumulate for aggregate statistics
                                            self.epoch_distance_prompt_agreement.append(bin_match)
                                            # Placeholder for memorization - will be updated when memorization is checked
                                            self.epoch_distance_prompt_memorized.append(False)
                                            
                                            # Log per-sample agreement
                                            _safe_log(pl_module,
                                                _name(f"val_{self.distance_name}_distance_prompt_match"),
                                                torch.tensor(1.0 if bin_match else 0.0, device=pl_module.device),
                                                sync_dist=False,
                                                on_step=False,
                                                on_epoch=True,
                                            )
                                        except Exception as exc:
                                            # Don't fail validation if bin comparison fails
                                            print(
                                                f"[Warning] Distance prompt agreement computation failed "
                                                f"for seed '{seed_smiles}' / percentile {distance_percentile}: {exc}"
                                            )
                            except Exception as exc:
                                # Don't fail validation if percentile computation fails
                                print(
                                    f"[Warning] Distance percentile computation failed "
                                    f"for seed '{seed_smiles}' / distance {d_seed_gen}: {exc}"
                                )
                                distance_percentile = None
                    
                    # Compute heavy atom counts and related metrics
                    try:
                        seed_mol = Chem.MolFromSmiles(seed_smiles)
                        gen_mol = Chem.MolFromSmiles(gen_smiles)
                        if seed_mol is not None and gen_mol is not None:
                            seed_heavy_atoms = Descriptors.HeavyAtomCount(seed_mol)
                            gen_heavy_atoms = Descriptors.HeavyAtomCount(gen_mol)
                            heavy_atom_delta = gen_heavy_atoms - seed_heavy_atoms
                            
                            # Log heavy atom change (number after - number before)
                            _safe_log(pl_module,
                                _name("val_heavy_atom_delta_conditional"),
                                torch.tensor(heavy_atom_delta, device=pl_module.device),
                                sync_dist=False,
                                on_step=False,
                                on_epoch=True,
                            )
                            
                            # Log tanimoto distance normalized by seed heavy atoms
                            if seed_heavy_atoms > 0:
                                tanimoto_per_heavy_atom = d_seed_gen / seed_heavy_atoms
                                _safe_log(pl_module,
                                    _name("val_tanimoto_distance_per_seed_heavy_atom_conditional"),
                                    torch.tensor(tanimoto_per_heavy_atom, device=pl_module.device),
                                    sync_dist=False,
                                    on_step=False,
                                    on_epoch=True,
                                )
                    except Exception as exc:
                        print(
                            "[Warning] Heavy atom count computation failed "
                            f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
                        )
                        traceback.print_exc()
        except Exception as exc:
            print(
                "[Warning] Conditional seed-to-gen distance computation failed "
                f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
            )
            traceback.print_exc()
        
        # Log shape_tanimoto metric (matching regular tanimoto with sync_dist=False)
        # This is outside the try block to ensure it always executes, even if distance computation fails
        if is_valid and seed_smiles and gen_smiles:
            shape_tanimoto_dist_value = float('nan')
            if get_shape_tanimoto_distance is not None:
                try:
                    computed_dist = get_shape_tanimoto_distance(seed_smiles, gen_smiles, verbose=False)
                    if computed_dist is not None:
                        shape_tanimoto_dist_value = float(computed_dist)
                        shape_tanimoto_dist = computed_dist  # Set for return value
                except Exception as exc:
                    print(
                        "[Warning] Conditional shape tanimoto computation failed "
                        f"for seed '{seed_smiles}' / gen '{gen_smiles}': {exc}"
                    )
                    traceback.print_exc()
            
            # Log with sync_dist=False to match regular tanimoto_distance_conditional
            _safe_log(pl_module,
                _name("val_shape_tanimoto_distance_conditional"),
                torch.tensor(shape_tanimoto_dist_value, device=pl_module.device),
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        
        # 7. Input pair distance metrics (between original validation pair)
        input_tanimoto_dist = None
        input_shape_tanimoto_dist = None
        if input_smiles1 and input_smiles2:
            try:
                from .smiles_utils import SmilesUtils
                if (SmilesUtils.is_valid(input_smiles1) and 
                    SmilesUtils.is_valid(input_smiles2)):
                    with self._time("conditional/input_pair_distance"):
                        if get_tanimoto_distance is not None:
                            input_tanimoto_dist = get_tanimoto_distance(input_smiles1, input_smiles2)
                            if input_tanimoto_dist is not None:
                                _safe_log(pl_module,
                                    "val_tanimoto_distance_input",
                                    torch.tensor(input_tanimoto_dist, device=pl_module.device),
                                    sync_dist=False,  # Changed from True: per-sample metric, aggregate at epoch end
                                    on_step=False,
                                    on_epoch=True,
                                )
                        
                        if get_shape_tanimoto_distance is not None:
                            input_shape_tanimoto_dist = get_shape_tanimoto_distance(input_smiles1, input_smiles2, verbose=False)
                            if input_shape_tanimoto_dist is not None:
                                _safe_log(pl_module,
                                    "val_shape_tanimoto_distance_input",
                                    torch.tensor(input_shape_tanimoto_dist, device=pl_module.device),
                                    sync_dist=False,  # Changed from True: per-sample metric, aggregate at epoch end
                                    on_step=False,
                                    on_epoch=True,
                                )
            except Exception as exc:
                print(
                    "[Warning] Conditional input pair distance computation failed "
                    f"for input1 '{input_smiles1}' / input2 '{input_smiles2}': {exc}"
                )
                traceback.print_exc()
        
        # Return structured result (replaces raw tuple)
        return SampleResult(
            valid_and_improved=valid_and_improved,
            tanimoto_dist=tanimoto_dist,
            shape_tanimoto_dist=shape_tanimoto_dist,
            seed_props=input_props,
            gen_props=gen_props,
            is_memorized=is_memorized,
            prompt_agreement=prompt_agreement,
            input_tanimoto_dist=input_tanimoto_dist,
            input_shape_tanimoto_dist=input_shape_tanimoto_dist,
            distance_percentile=distance_percentile,
        )
    
    def _get_neighbors(self, seed_smiles: str, neighbors_batch: Optional[List[str]]) -> List[str]:
        """Get neighbors for a seed SMILES."""
        if neighbors_batch and len(neighbors_batch) > 0:
            return [n for n in neighbors_batch if n and self._is_valid(n)]
        if self.neighbor_provider and seed_smiles:
            try:
                neighs = self.neighbor_provider(seed_smiles) or []
                return [n for n in neighs if n and self._is_valid(n)]
            except Exception as exc:
                print(f"[Warning] Neighbor provider failed for seed '{seed_smiles}': {exc}")
                traceback.print_exc()
                return []
        return []
    
    def accumulate_sample(self, seed_smiles: str, gen_smiles: str,
                         is_valid: bool, canonical_gen: Optional[str] = None,
                         gt_list: Optional[List[str]] = None):
        """
        Accumulate data from a single generated sample.

        Args:
            seed_smiles: Input/conditioning SMILES
            gen_smiles: Generated SMILES
            is_valid: Whether generated molecule is valid
            canonical_gen: Canonical form of generated SMILES
            gt_list: Optional list of ground truth molecules for comparison
        """
        # Populate shared base buffers (epoch_seeds, epoch_gens, epoch_all_gens, epoch_valid_canon_gens)
        self._accumulate_pair(seed_smiles, gen_smiles, canonical_gen, is_valid)

        # Conditional-specific: GT lists
        if seed_smiles and gen_smiles and gt_list is not None:
            self.epoch_cond_gt_lists.append(gt_list)
    
    def check_memorization(self, canonical_gen: Optional[str], is_valid: bool):
        """
        Check if generated molecule is in training set (memorization check).
        Only called when "memorization" is in expensive_metrics_to_compute.
        
        If check_memorization_from_any_reference is True, this checks against ALL reference sets.
        Otherwise, it checks only the default reference set (training_canon_set).
        
        Args:
            canonical_gen: Canonical form of generated SMILES (can be None if invalid)
            is_valid: Whether generated molecule is valid
        """
        if "memorization" not in self.expensive_metrics_to_compute:
            return
        
        if not is_valid or not canonical_gen:
            return
        
        with self._time("conditional/memorization_check"):
            if self.check_memorization_from_any_reference:
                # Check if molecule is in ANY of the reference sets
                if self.reference_canon_sets_by_name:
                    self.epoch_memorized_checked += 1
                    is_mem = any(
                        canonical_gen in ref_set 
                        for ref_set in self.reference_canon_sets_by_name.values()
                        if ref_set
                    )
                    if is_mem:
                        self.epoch_memorized_count += 1
            else:
                # Check only the default reference set (backward-compatible behavior)
                if self.training_canon_set is not None:
                    self.epoch_memorized_checked += 1
                    if canonical_gen in self.training_canon_set:
                        self.epoch_memorized_count += 1

    def check_memorization_named_refs(self, canonical_gen: Optional[str], is_valid: bool):
        """
        Check if generated molecule is in any named reference sets (memorization check).
        Uses the same enable/disable gate as the training-set memorization check.

        Args:
            canonical_gen: Canonical SMILES (or None if invalid)
            is_valid: Whether generated molecule is valid
        """
        if "memorization" not in self.expensive_metrics_to_compute:
            return
        if not self.reference_canon_sets_by_name:
            return
        if not (is_valid and canonical_gen):
            return

        # Keep timing separate so it doesn't get conflated with the training-set check
        with self._time("conditional/memorization_check_named_refs"):
            for ref_name, ref_set in self.reference_canon_sets_by_name.items():
                if not ref_set:
                    continue
                self.epoch_memorized_checked_by_ref[ref_name] = self.epoch_memorized_checked_by_ref.get(ref_name, 0) + 1
                if canonical_gen in ref_set:
                    self.epoch_memorized_count_by_ref[ref_name] = self.epoch_memorized_count_by_ref.get(ref_name, 0) + 1
    
    def compute_and_log(self, pl_module, registry):
        """
        Compute all conditional metrics and register them.
        
        Args:
            pl_module: PyTorch Lightning module (for device placement)
            registry: MetricRegistry to register metrics
        """
        # Helper for safe mean
        def _safe_mean(vals: List[float]) -> Optional[float]:
            return float(np.mean(vals)) if vals else None
        
        def _register(metric_name: str, value: Optional[float],
                      description: str, category: str):
            display_value = math.nan if value is None else float(value)
            registry.register(metric_name, display_value, description, category)
        
        # Distance and neighborhood metrics (always register to keep ranks aligned)
        dist_to_gt_mean = _safe_mean(self.epoch_dist_to_gt)
        _register(
            f"{self.distance_name}_distance_to_gt_conditional_mean",
            dist_to_gt_mean,
            "Mean distance to ground truth",
            "Conditional - Distances",
        )
        
        input_to_gt_mean = _safe_mean(self.epoch_input_to_gt_min_dist)
        _register(
            f"{self.distance_name}_distance_input_to_gt_conditional_mean",
            input_to_gt_mean,
            "Mean distance (seed to GT)",
            "Conditional - Distances",
        )
        
        seed_to_gen_mean = _safe_mean(self.epoch_seed_to_gen_dist)
        _register(
            f"{self.distance_name}_seed_to_gen_conditional_mean",
            seed_to_gen_mean,
            "Mean distance (seed to gen)",
            "Conditional - Distances",
        )
        diff_mean = None
        if (
            self.epoch_input_to_gt_min_dist
            and self.epoch_seed_to_gen_dist
        ):
            n = min(
                len(self.epoch_input_to_gt_min_dist),
                len(self.epoch_seed_to_gen_dist),
            )
            if n > 0:
                diff_vals = [
                    self.epoch_input_to_gt_min_dist[i]
                    - self.epoch_seed_to_gen_dist[i]
                    for i in range(n)
                ]
                if diff_vals:
                    diff_mean = float(np.mean(diff_vals))
        _register(
            f"{self.distance_name}_seed_to_gt_minus_seed_to_gen_conditional_mean",
            diff_mean,
            "Mean (seed→GT − seed→gen) distance",
            "Conditional - Distances",
        )
        
        # Filtered distance metrics for valid+improved+not_memorized samples (per property)
        for spec in self.property_specs:
            prop_name = spec.name
            
            # Filter seed_to_gen distances
            seed_v_n_i_mean = None
            if prop_name in self.epoch_seed_to_gen_V_N_I_flags:
                flags = self.epoch_seed_to_gen_V_N_I_flags[prop_name]
                if len(flags) == len(self.epoch_seed_to_gen_dist):
                    filtered_distances = [
                        self.epoch_seed_to_gen_dist[i] 
                        for i in range(len(flags)) 
                        if flags[i]
                    ]
                    if filtered_distances:
                        seed_v_n_i_mean = float(np.mean(filtered_distances))
            
            _register(
                f"{self.distance_name}_seed_to_gen_V_N_I_mean/{prop_name}",
                seed_v_n_i_mean,
                f"Mean distance (seed to gen) for valid+improved+not_memorized samples ({prop_name})",
                "Conditional - Distances",
            )
            
            # Filter input_to_gt distances
            input_v_n_i_mean = None
            if prop_name in self.epoch_input_to_gt_V_N_I_flags:
                flags = self.epoch_input_to_gt_V_N_I_flags[prop_name]
                if len(flags) == len(self.epoch_input_to_gt_min_dist):
                    filtered_distances = [
                        self.epoch_input_to_gt_min_dist[i] 
                        for i in range(len(flags)) 
                        if flags[i]
                    ]
                    if filtered_distances:
                        input_v_n_i_mean = float(np.mean(filtered_distances))
            
            _register(
                f"{self.distance_name}_distance_input_to_gt_V_N_I_mean/{prop_name}",
                input_v_n_i_mean,
                f"Mean distance (seed to GT) for valid+improved+not_memorized samples ({prop_name})",
                "Conditional - Distances",
            )
            
            # Compute difference (seed_to_gt - seed_to_gen) for V_N_I samples
            diff_v_n_i_mean = None
            if seed_v_n_i_mean is not None and input_v_n_i_mean is not None:
                diff_v_n_i_mean = input_v_n_i_mean - seed_v_n_i_mean
            
            _register(
                f"{self.distance_name}_seed_to_gt_minus_seed_to_gen_V_N_I_mean/{prop_name}",
                diff_v_n_i_mean,
                f"Mean (seed→GT − seed→gen) distance for valid+improved+not_memorized samples ({prop_name})",
                "Conditional - Distances",
            )
        
        neigh_median = _safe_mean(self.epoch_neigh_median_dist)
        _register(
            f"{self.distance_name}_neighborhood_distance_median_conditional",
            neigh_median,
            "Median distance to neighbors",
            "Conditional - Distances",
        )
        
        neigh_p75 = _safe_mean(self.epoch_neigh_p75_dist)
        _register(
            f"{self.distance_name}_neighborhood_distance_p75_conditional",
            neigh_p75,
            "75th percentile distance to neighbors",
            "Conditional - Distances",
        )
        
        neigh_nn = _safe_mean(self.epoch_neigh_nn_dist)
        _register(
            f"{self.distance_name}_nearest_neighbor_distance_conditional",
            neigh_nn,
            "Nearest neighbor distance",
            "Conditional - Distances",
        )
        
        for feat_name in FEATURE_NAMES:
            feat_mean = _safe_mean(self.epoch_w1_deltas[feat_name])
            _register(
                f"w1_delta_conditional/{feat_name}",
                feat_mean,
                f"W1 delta for {feat_name}",
                "Conditional - Distributional Alignment",
            )
        
        if self.epoch_kde_ll_in:
            ll_in_mean = float(np.mean(self.epoch_kde_ll_in))
        else:
            ll_in_mean = None
        if self.epoch_kde_ll_gen:
            ll_gen_mean = float(np.mean(self.epoch_kde_ll_gen))
        else:
            ll_gen_mean = None
        delta_ll = (
            ll_gen_mean - ll_in_mean
            if ll_in_mean is not None and ll_gen_mean is not None
            else None
        )
        _register("kde_ll_in_conditional", ll_in_mean,
                  "KDE log-likelihood input", "Conditional - Distributional Alignment")
        _register("kde_ll_gen_conditional", ll_gen_mean,
                  "KDE log-likelihood generated", "Conditional - Distributional Alignment")
        _register("kde_ll_delta_conditional", delta_ll,
                  "KDE log-likelihood delta", "Conditional - Distributional Alignment")
        
        gt_delta_mean = _safe_mean(self.epoch_gt_min_delta)
        _register(
            f"{self.distance_name}_gt_closest_delta_conditional_mean",
            gt_delta_mean,
            "Mean delta to closest GT",
            "Conditional - Distances",
        )
        
        # Distance percentile aggregate statistics
        if self.epoch_distance_percentiles:
            percentile_mean = float(np.mean(self.epoch_distance_percentiles))
            percentile_median = float(np.median(self.epoch_distance_percentiles))
            percentile_p25 = float(np.percentile(self.epoch_distance_percentiles, 25))
            percentile_p75 = float(np.percentile(self.epoch_distance_percentiles, 75))
            
            _register(
                f"{self.distance_name}_distance_percentile_conditional_mean",
                percentile_mean,
                "Mean distance percentile (0-100)",
                "Conditional - Distances",
            )
            _register(
                f"{self.distance_name}_distance_percentile_conditional_median",
                percentile_median,
                "Median distance percentile (0-100)",
                "Conditional - Distances",
            )
            _register(
                f"{self.distance_name}_distance_percentile_conditional_p25",
                percentile_p25,
                "25th percentile of distance percentiles",
                "Conditional - Distances",
            )
            _register(
                f"{self.distance_name}_distance_percentile_conditional_p75",
                percentile_p75,
                "75th percentile of distance percentiles",
                "Conditional - Distances",
            )
        
        num_valid_gen = len(self.epoch_valid_cond_gen_canon)
        # Always register fraction metrics (even with 0/0) for DDP synchronization.
        # log_all() handles denom=0 by producing NaN.
        registry.register_fraction(
            "uniqueness_conditional_valid",
            len(set(self.epoch_valid_cond_gen_canon)) if num_valid_gen > 0 else 0,
            num_valid_gen,
            "Fraction of unique valid molecules",
            "Conditional - Basic",
        )
        if self.training_canon_set is not None:
            num_novel = (
                sum(1 for s in self.epoch_valid_cond_gen_canon if s not in self.training_canon_set)
                if num_valid_gen > 0 else 0
            )
            registry.register_fraction(
                "novelty_conditional_valid",
                num_novel, num_valid_gen,
                "Fraction of novel molecules",
                "Conditional - Basic",
            )
        else:
            _register("novelty_conditional_valid", None, "Fraction of novel molecules", "Conditional - Basic")
        # Also compute novelty against each named reference set (canonical membership).
        if self.reference_canon_sets_by_name:
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                ref_set = self.reference_canon_sets_by_name.get(ref_name) or set()
                if not ref_set:
                    _register(f"novelty_conditional_valid/{ref_name}", None,
                              f"Fraction of novel molecules vs reference '{ref_name}'", "Conditional - Basic")
                else:
                    num_novel_ref = (
                        sum(1 for s in self.epoch_valid_cond_gen_canon if s not in ref_set)
                        if num_valid_gen > 0 else 0
                    )
                    registry.register_fraction(
                        f"novelty_conditional_valid/{ref_name}",
                        num_novel_ref, num_valid_gen,
                        f"Fraction of novel molecules vs reference '{ref_name}'",
                        "Conditional - Basic",
                    )
        
        # Prompt agreement aggregate statistics (per property)
        # Iterate over deterministic property names (not data-dependent dict keys)
        # to ensure all ranks register the same set of fractions for DDP sync.
        _prop_names = sorted(spec.name for spec in self.property_specs)
        for prop_name in _prop_names:
            direction_matches = self.epoch_prompt_direction_agreement.get(prop_name, [])
            # Always register for DDP sync (0/0 when no data)
            registry.register_fraction(
                f"{prop_name}_prompt_direction_agreement_rate",
                sum(direction_matches) if direction_matches else 0,
                len(direction_matches),
                f"Fraction of samples where {prop_name} direction matches prompt",
                "Conditional - Prompt Agreement",
            )

            # Non-memorized version
            non_memorized_matches = []
            if direction_matches and prop_name in self.epoch_prompt_direction_memorized:
                memorized_flags = self.epoch_prompt_direction_memorized[prop_name]
                if len(memorized_flags) == len(direction_matches):
                    non_memorized_matches = [
                        match for match, is_mem in zip(direction_matches, memorized_flags) if not is_mem
                    ]
            registry.register_fraction(
                f"{prop_name}_prompt_direction_agreement_rate_non_memorized",
                sum(non_memorized_matches) if non_memorized_matches else 0,
                len(non_memorized_matches),
                f"Fraction of non-memorized samples where {prop_name} direction matches prompt",
                "Conditional - Prompt Agreement",
            )

            # Per-reference non-memorized versions
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                ref_non_memorized_matches = []
                if direction_matches and ref_name in self.epoch_prompt_direction_memorized_by_ref:
                    if prop_name in self.epoch_prompt_direction_memorized_by_ref[ref_name]:
                        ref_memorized_flags = self.epoch_prompt_direction_memorized_by_ref[ref_name][prop_name]
                        if len(ref_memorized_flags) == len(direction_matches):
                            ref_non_memorized_matches = [
                                match for match, is_mem in zip(direction_matches, ref_memorized_flags) if not is_mem
                            ]
                registry.register_fraction(
                    f"{prop_name}_prompt_direction_agreement_rate_non_memorized/{ref_name}",
                    sum(ref_non_memorized_matches) if ref_non_memorized_matches else 0,
                    len(ref_non_memorized_matches),
                    f"Fraction of samples not in '{ref_name}' where {prop_name} direction matches prompt",
                    "Conditional - Prompt Agreement",
                )
        
        # Strict direction agreement (excludes "any" direction, which trivially matches)
        for prop_name in _prop_names:
            strict_matches = self.epoch_prompt_direction_agreement_strict.get(prop_name, [])
            # Always register for DDP sync
            registry.register_fraction(
                f"{prop_name}_prompt_direction_agreement_rate_strict",
                sum(strict_matches) if strict_matches else 0,
                len(strict_matches),
                f"Fraction of up/down-prompted samples where {prop_name} direction matches (excludes 'any')",
                "Conditional - Prompt Agreement",
            )
            nm = []
            if strict_matches and prop_name in self.epoch_prompt_direction_memorized_strict:
                mem_flags = self.epoch_prompt_direction_memorized_strict[prop_name]
                if len(mem_flags) == len(strict_matches):
                    nm = [m for m, is_mem in zip(strict_matches, mem_flags) if not is_mem]
            registry.register_fraction(
                f"{prop_name}_prompt_direction_agreement_rate_strict_non_memorized",
                sum(nm) if nm else 0,
                len(nm),
                f"Strict direction agreement (non-memorized, excludes 'any')",
                "Conditional - Prompt Agreement",
            )

        for prop_name in _prop_names:
            magnitude_matches = self.epoch_prompt_magnitude_agreement.get(prop_name, [])
            # Always register for DDP sync
            registry.register_fraction(
                f"{prop_name}_prompt_magnitude_agreement_rate",
                sum(magnitude_matches) if magnitude_matches else 0,
                len(magnitude_matches),
                f"Fraction of samples where {prop_name} magnitude bin matches prompt",
                "Conditional - Prompt Agreement",
            )

            # Non-memorized version
            non_memorized_matches = []
            if magnitude_matches and prop_name in self.epoch_prompt_magnitude_memorized:
                memorized_flags = self.epoch_prompt_magnitude_memorized[prop_name]
                if len(memorized_flags) == len(magnitude_matches):
                    non_memorized_matches = [
                        match for match, is_mem in zip(magnitude_matches, memorized_flags) if not is_mem
                    ]
            registry.register_fraction(
                f"{prop_name}_prompt_magnitude_agreement_rate_non_memorized",
                sum(non_memorized_matches) if non_memorized_matches else 0,
                len(non_memorized_matches),
                f"Fraction of non-memorized samples where {prop_name} magnitude bin matches prompt",
                "Conditional - Prompt Agreement",
            )

            # Per-reference non-memorized versions
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                ref_non_memorized_matches = []
                if magnitude_matches and ref_name in self.epoch_prompt_magnitude_memorized_by_ref:
                    if prop_name in self.epoch_prompt_magnitude_memorized_by_ref[ref_name]:
                        ref_memorized_flags = self.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop_name]
                        if len(ref_memorized_flags) == len(magnitude_matches):
                            ref_non_memorized_matches = [
                                match for match, is_mem in zip(magnitude_matches, ref_memorized_flags) if not is_mem
                            ]
                registry.register_fraction(
                    f"{prop_name}_prompt_magnitude_agreement_rate_non_memorized/{ref_name}",
                    sum(ref_non_memorized_matches) if ref_non_memorized_matches else 0,
                    len(ref_non_memorized_matches),
                    f"Fraction of samples not in '{ref_name}' where {prop_name} magnitude bin matches prompt",
                    "Conditional - Prompt Agreement",
                )
            
            # Bin error statistics (magnitude of error) - only for non-memorized samples
            # Always register all bin error metrics for DDP sync (NaN when no data)
            nm_mean_abs_error = None
            nm_mean_error = None
            nm_median_abs_error = None
            nm_p75_abs_error = None
            nm_p95_abs_error = None
            if prop_name in self.epoch_prompt_magnitude_bin_errors:
                bin_errors = self.epoch_prompt_magnitude_bin_errors[prop_name]
                if bin_errors and prop_name in self.epoch_prompt_magnitude_bin_errors_memorized:
                    memorized_flags = self.epoch_prompt_magnitude_bin_errors_memorized[prop_name]
                    if len(memorized_flags) == len(bin_errors):
                        non_memorized_errors = [
                            e for e, is_mem in zip(bin_errors, memorized_flags) if not is_mem
                        ]
                        if non_memorized_errors:
                            non_memorized_abs_errors = [abs(e) for e in non_memorized_errors]
                            nm_mean_abs_error = float(np.mean(non_memorized_abs_errors))
                            nm_mean_error = float(np.mean(non_memorized_errors))
                            nm_median_abs_error = float(np.median(non_memorized_abs_errors))
                            nm_p75_abs_error = float(np.percentile(non_memorized_abs_errors, 75))
                            nm_p95_abs_error = float(np.percentile(non_memorized_abs_errors, 95))
            _register(f"{prop_name}_prompt_magnitude_bin_error_mae", nm_mean_abs_error,
                      f"Mean absolute bin error (non-memorized) for {prop_name}", "Conditional - Prompt Agreement")
            _register(f"{prop_name}_prompt_magnitude_bin_error_mean", nm_mean_error,
                      f"Mean bin error bias (non-memorized) for {prop_name}", "Conditional - Prompt Agreement")
            _register(f"{prop_name}_prompt_magnitude_bin_error_median_abs", nm_median_abs_error,
                      f"Median absolute bin error (non-memorized) for {prop_name}", "Conditional - Prompt Agreement")
            _register(f"{prop_name}_prompt_magnitude_bin_error_p75_abs", nm_p75_abs_error,
                      f"75th percentile absolute bin error (non-memorized) for {prop_name}", "Conditional - Prompt Agreement")
            _register(f"{prop_name}_prompt_magnitude_bin_error_p95_abs", nm_p95_abs_error,
                      f"95th percentile absolute bin error (non-memorized) for {prop_name}", "Conditional - Prompt Agreement")

            # Per-reference bin error statistics
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                ref_nm_mean_abs_error = None
                ref_nm_mean_error = None
                ref_nm_median_abs_error = None
                ref_nm_p75_abs_error = None
                ref_nm_p95_abs_error = None
                if prop_name in self.epoch_prompt_magnitude_bin_errors:
                    bin_errors = self.epoch_prompt_magnitude_bin_errors[prop_name]
                    if bin_errors and ref_name in self.epoch_prompt_magnitude_bin_errors_memorized_by_ref:
                        if prop_name in self.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name]:
                            ref_memorized_flags = self.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop_name]
                            if len(ref_memorized_flags) == len(bin_errors):
                                ref_non_memorized_errors = [
                                    e for e, is_mem in zip(bin_errors, ref_memorized_flags) if not is_mem
                                ]
                                if ref_non_memorized_errors:
                                    ref_non_memorized_abs_errors = [abs(e) for e in ref_non_memorized_errors]
                                    ref_nm_mean_abs_error = float(np.mean(ref_non_memorized_abs_errors))
                                    ref_nm_mean_error = float(np.mean(ref_non_memorized_errors))
                                    ref_nm_median_abs_error = float(np.median(ref_non_memorized_abs_errors))
                                    ref_nm_p75_abs_error = float(np.percentile(ref_non_memorized_abs_errors, 75))
                                    ref_nm_p95_abs_error = float(np.percentile(ref_non_memorized_abs_errors, 95))
                _register(f"{prop_name}_prompt_magnitude_bin_error_mae/{ref_name}", ref_nm_mean_abs_error,
                          f"MAE bin error (not in '{ref_name}') for {prop_name}", "Conditional - Prompt Agreement")
                _register(f"{prop_name}_prompt_magnitude_bin_error_mean/{ref_name}", ref_nm_mean_error,
                          f"Mean bin error (not in '{ref_name}') for {prop_name}", "Conditional - Prompt Agreement")
                _register(f"{prop_name}_prompt_magnitude_bin_error_median_abs/{ref_name}", ref_nm_median_abs_error,
                          f"Median abs bin error (not in '{ref_name}') for {prop_name}", "Conditional - Prompt Agreement")
                _register(f"{prop_name}_prompt_magnitude_bin_error_p75_abs/{ref_name}", ref_nm_p75_abs_error,
                          f"P75 abs bin error (not in '{ref_name}') for {prop_name}", "Conditional - Prompt Agreement")
                _register(f"{prop_name}_prompt_magnitude_bin_error_p95_abs/{ref_name}", ref_nm_p95_abs_error,
                          f"P95 abs bin error (not in '{ref_name}') for {prop_name}", "Conditional - Prompt Agreement")

        # Random baselines for prompt agreement
        # Always register for all properties (deterministic iteration for DDP sync)
        # Direction: 0.5 (coin flip — random model has no directional bias)
        for prop_name in _prop_names:
            _register(
                f"{prop_name}_prompt_direction_agreement_rate_random_baseline",
                0.5,
                f"Random baseline for {prop_name} direction agreement (coin flip)",
                "Conditional - Prompt Agreement Baselines",
            )
            _register(
                f"{prop_name}_prompt_direction_agreement_rate_strict_random_baseline",
                0.5,
                f"Random baseline for {prop_name} strict direction agreement (coin flip)",
                "Conditional - Prompt Agreement Baselines",
            )

        # Magnitude: compute empirical baselines from observed bin distributions
        from collections import Counter
        for prop_name in _prop_names:
            actual_bins = self.epoch_prompt_actual_bins.get(prop_name, [])
            prompted_bins = self.epoch_prompt_prompted_bins.get(prop_name, [])
            if not actual_bins:
                # No data — register NaN baselines for DDP sync
                _register(f"{prop_name}_prompt_magnitude_agreement_rate_uniform_baseline", None,
                          f"Uniform random baseline for {prop_name} magnitude", "Conditional - Prompt Agreement Baselines")
                _register(f"{prop_name}_prompt_magnitude_agreement_rate_empirical_baseline", None,
                          f"Empirical random baseline for {prop_name} magnitude", "Conditional - Prompt Agreement Baselines")
                _register(f"{prop_name}_prompt_magnitude_agreement_rate_mode_baseline", None,
                          f"Mode baseline for {prop_name} magnitude", "Conditional - Prompt Agreement Baselines")
                continue
            n = len(actual_bins)
            # Uniform baseline: 1/num_bins (use max observed bin + 1 as num_bins)
            max_bin = max(max(actual_bins), max(prompted_bins))
            num_bins = max_bin + 1
            _register(
                f"{prop_name}_prompt_magnitude_agreement_rate_uniform_baseline",
                1.0 / num_bins,
                f"Uniform random baseline for {prop_name} magnitude (1/{num_bins} bins)",
                "Conditional - Prompt Agreement Baselines",
            )
            # Empirical baseline: P(match) if actual and prompted bins are drawn
            # independently from their respective marginal distributions.
            # = sum_b (P_actual(b) * P_prompted(b))
            actual_counts = Counter(actual_bins)
            prompted_counts = Counter(prompted_bins)
            empirical_match_prob = sum(
                (actual_counts.get(b, 0) / n) * (prompted_counts.get(b, 0) / n)
                for b in set(actual_bins) | set(prompted_bins)
            )
            _register(
                f"{prop_name}_prompt_magnitude_agreement_rate_empirical_baseline",
                empirical_match_prob,
                f"Empirical random baseline for {prop_name} magnitude (marginal overlap)",
                "Conditional - Prompt Agreement Baselines",
            )
            # Most-common-bin baseline: always predict the most common actual bin
            most_common_actual_bin = actual_counts.most_common(1)[0][0]
            mode_baseline = prompted_counts.get(most_common_actual_bin, 0) / n
            _register(
                f"{prop_name}_prompt_magnitude_agreement_rate_mode_baseline",
                mode_baseline,
                f"Mode baseline for {prop_name} magnitude (always predict most common actual bin {most_common_actual_bin})",
                "Conditional - Prompt Agreement Baselines",
            )

        # Distance prompt agreement aggregate statistics
        # Always register for DDP sync
        dist_agree = self.epoch_distance_prompt_agreement
        registry.register_fraction(
            f"{self.distance_name}_distance_prompt_agreement_rate",
            sum(dist_agree) if dist_agree else 0,
            len(dist_agree),
            "Fraction of samples where distance bin matches prompt",
            "Conditional - Prompt Agreement",
        )

        # Non-memorized version
        non_memorized_matches = []
        if dist_agree and len(self.epoch_distance_prompt_memorized) == len(dist_agree):
            non_memorized_matches = [
                match for match, is_mem in zip(dist_agree, self.epoch_distance_prompt_memorized) if not is_mem
            ]
        registry.register_fraction(
            f"{self.distance_name}_distance_prompt_agreement_rate_non_memorized",
            sum(non_memorized_matches) if non_memorized_matches else 0,
            len(non_memorized_matches),
            "Fraction of non-memorized samples where distance bin matches prompt",
            "Conditional - Prompt Agreement",
        )

        # Per-reference non-memorized versions
        for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
            ref_non_memorized_matches = []
            if dist_agree and ref_name in self.epoch_distance_prompt_memorized_by_ref:
                ref_memorized_flags = self.epoch_distance_prompt_memorized_by_ref[ref_name]
                if len(ref_memorized_flags) == len(dist_agree):
                    ref_non_memorized_matches = [
                        match for match, is_mem in zip(dist_agree, ref_memorized_flags) if not is_mem
                    ]
            registry.register_fraction(
                f"{self.distance_name}_distance_prompt_agreement_rate_non_memorized/{ref_name}",
                sum(ref_non_memorized_matches) if ref_non_memorized_matches else 0,
                len(ref_non_memorized_matches),
                f"Fraction of samples not in '{ref_name}' where distance bin matches prompt",
                "Conditional - Prompt Agreement",
            )
        
        # All properties agree aggregate statistics
        # Always register for DDP sync
        all_dir = self.epoch_all_properties_direction_agree
        registry.register_fraction(
            "all_properties_direction_agree_rate",
            sum(all_dir) if all_dir else 0,
            len(all_dir),
            "Fraction of samples where ALL prompted properties' directions match",
            "Conditional - Prompt Agreement",
        )

        # Non-memorized version
        non_memorized_matches = []
        if all_dir and len(self.epoch_all_properties_direction_memorized) == len(all_dir):
            non_memorized_matches = [
                match for match, is_mem in zip(all_dir, self.epoch_all_properties_direction_memorized) if not is_mem
            ]
        registry.register_fraction(
            "all_properties_direction_agree_rate_non_memorized",
            sum(non_memorized_matches) if non_memorized_matches else 0,
            len(non_memorized_matches),
            "Fraction of non-memorized samples where ALL prompted properties' directions match",
            "Conditional - Prompt Agreement",
        )

        # Per-reference non-memorized versions
        for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
            ref_non_memorized_matches = []
            if all_dir and ref_name in self.epoch_all_properties_direction_memorized_by_ref:
                ref_memorized_flags = self.epoch_all_properties_direction_memorized_by_ref[ref_name]
                if len(ref_memorized_flags) == len(all_dir):
                    ref_non_memorized_matches = [
                        match for match, is_mem in zip(all_dir, ref_memorized_flags) if not is_mem
                    ]
            registry.register_fraction(
                f"all_properties_direction_agree_rate_non_memorized/{ref_name}",
                sum(ref_non_memorized_matches) if ref_non_memorized_matches else 0,
                len(ref_non_memorized_matches),
                f"Fraction of samples not in '{ref_name}' where ALL properties' directions match",
                "Conditional - Prompt Agreement",
            )

        all_mag = self.epoch_all_properties_magnitude_agree
        registry.register_fraction(
            "all_properties_magnitude_agree_rate",
            sum(all_mag) if all_mag else 0,
            len(all_mag),
            "Fraction of samples where ALL prompted properties' magnitudes match",
            "Conditional - Prompt Agreement",
        )

        # Non-memorized version
        non_memorized_matches = []
        if all_mag and len(self.epoch_all_properties_magnitude_memorized) == len(all_mag):
            non_memorized_matches = [
                match for match, is_mem in zip(all_mag, self.epoch_all_properties_magnitude_memorized) if not is_mem
            ]
        registry.register_fraction(
            "all_properties_magnitude_agree_rate_non_memorized",
            sum(non_memorized_matches) if non_memorized_matches else 0,
            len(non_memorized_matches),
            "Fraction of non-memorized samples where ALL prompted properties' magnitudes match",
            "Conditional - Prompt Agreement",
        )

        # Per-reference non-memorized versions
        for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
            ref_non_memorized_matches = []
            if all_mag and ref_name in self.epoch_all_properties_magnitude_memorized_by_ref:
                ref_memorized_flags = self.epoch_all_properties_magnitude_memorized_by_ref[ref_name]
                if len(ref_memorized_flags) == len(all_mag):
                    ref_non_memorized_matches = [
                        match for match, is_mem in zip(all_mag, ref_memorized_flags) if not is_mem
                    ]
            registry.register_fraction(
                f"all_properties_magnitude_agree_rate_non_memorized/{ref_name}",
                sum(ref_non_memorized_matches) if ref_non_memorized_matches else 0,
                len(ref_non_memorized_matches),
                f"Fraction of samples not in '{ref_name}' where ALL properties' magnitudes match",
                "Conditional - Prompt Agreement",
            )
        
        # Comprehensive metrics (delegates to helper for detailed metrics)
        self._compute_comprehensive_metrics(registry)
        
        # Memorization metrics
        if "memorization" in self.expensive_metrics_to_compute:
            # Always register as fraction for DDP sync (0/0 → NaN when no data)
            registry.register_fraction(
                "memorization_rate_conditional",
                self.epoch_memorized_count, self.epoch_memorized_checked,
                f"Fraction of valid conditional generations in training set ({self.epoch_memorized_count}/{self.epoch_memorized_checked})",
                "Conditional - Memorization",
            )
            # Per-reference memorization rates (named reference sets)
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                checked = int(self.epoch_memorized_checked_by_ref.get(ref_name, 0))
                count = int(self.epoch_memorized_count_by_ref.get(ref_name, 0))
                registry.register_fraction(
                    f"memorization_rate_conditional/{ref_name}",
                    count, checked,
                    f"Fraction of valid conditional generations in reference '{ref_name}' ({count}/{checked})",
                    "Conditional - Memorization",
                )
        
        # NEW: Distributional metrics for conditional generations (SNN, FCD, diversity, etc.)
        distributional = self._compute_distributional_metrics("_conditional")
        for name, value in distributional.items():
            display = value if not math.isnan(value) else None
            _register(name, display, "", "Conditional - Distributional")

        # Sample counts (always register)
        _register(
            "cond_pairs_count",
            float(len(self.epoch_cond_seeds)),
            "Number of conditional pairs",
            "Conditional - Counts",
        )
        _register(
            "cond_valid_count",
            float(len(self.epoch_valid_cond_gen_canon)),
            "Number of valid conditionals",
            "Conditional - Counts",
        )

    def _compute_comprehensive_metrics(self, registry):
        """Compute comprehensive metrics using molecule_metrics functions."""
        def _register(metric_name: str, value: Optional[float],
                      description: str, category: str):
            display_value = math.nan if value is None else float(value)
            registry.register(metric_name, display_value, description, category)
        
        # Similarity to seeds
        sim_mean = None
        try:
            with self._time("conditional/mean_similarity_to_seeds"):
                sim_mean = mean_similarity_to_seeds(
                    self.epoch_cond_seeds, self.epoch_cond_gens
                )
        except Exception as exc:
            print(f"[Warning] Similarity to seed failed: {exc}")
            traceback.print_exc()
        _register(
            "similarity_to_seed_mean",
            sim_mean,
            "Mean similarity to seed",
            "Conditional - Similarities",
        )
        
        # Max similarity to ground truth
        # CRITICAL: Always register these metrics (even if data is empty) for DDP synchronization
        # With smaller datasets, some ranks may have no GT lists, causing different metrics to be logged
        gen_mean = None
        seed_mean = None
        delta_mean = None
        if self.epoch_cond_gt_lists and len(self.epoch_cond_gt_lists) == len(self.epoch_cond_gens):
            try:
                with self._time("conditional/max_similarity_to_gt"):
                    gen_max, seed_max, delta = max_similarity_to_gt(
                        self.epoch_cond_gens,
                        self.epoch_cond_seeds,
                        self.epoch_cond_gt_lists,
                    )
                def _mean_non_none(values: List[Optional[float]]) -> Optional[float]:
                    valid = [v for v in values if v is not None]
                    return float(np.mean(valid)) if valid else None
                gen_mean = _mean_non_none(gen_max)
                seed_mean = _mean_non_none(seed_max)
                delta_mean = _mean_non_none(delta)
            except Exception as exc:
                print(f"[Warning] Max similarity to GT failed: {exc}")
                traceback.print_exc()
        else:
            if not self.epoch_cond_gt_lists:
                print("[Info] No ground-truth lists available for conditional GT similarity.")
            # Still register metrics as NaN to ensure DDP synchronization
        
        _register(
            "max_similarity_to_gt_gen_mean",
            gen_mean,
            "Mean max similarity (gen to GT)",
            "Conditional - Similarities",
        )
        _register(
            "max_similarity_to_gt_seed_mean",
            seed_mean,
            "Mean max similarity (seed to GT)",
            "Conditional - Similarities",
        )
        _register(
            "max_similarity_to_gt_delta_mean",
            delta_mean,
            "Mean similarity delta",
            "Conditional - Similarities",
        )
        
        # Property improvements and related metrics
        # Debug: Check if we have valid seeds and gens
        num_seeds = len(self.epoch_cond_seeds)
        num_gens = len(self.epoch_cond_gens)
        if num_seeds > 0 and num_gens > 0:
            # Sample a few to check validity
            sample_seeds = self.epoch_cond_seeds[:min(3, num_seeds)]
            sample_gens = self.epoch_cond_gens[:min(3, num_gens)]
            valid_seeds = 0
            for s in sample_seeds:
                if s:
                    mol = Chem.MolFromSmiles(s)
                    if mol and is_molecule_sane(mol):
                        valid_seeds += 1
            valid_gens = 0
            for g in sample_gens:
                if g:
                    mol = Chem.MolFromSmiles(g)
                    if mol and is_molecule_sane(mol):
                        valid_gens += 1
            if valid_seeds == 0:
                print(f"[Warning] Property metrics: {num_seeds} seeds in epoch_cond_seeds, but sample shows 0 valid SMILES. Sample seeds: {sample_seeds}")
            if valid_gens == 0:
                print(f"[Warning] Property metrics: {num_gens} gens in epoch_cond_gens, but sample shows 0 valid SMILES. Sample gens: {sample_gens}")
        
        for spec in self.property_specs:
            stats = None
            try:
                with self._time(f"conditional/property_improvement_stats/{spec.name}"):
                    stats = pe_property_improvement_stats(
                        self.epoch_cond_seeds,
                        self.epoch_cond_gens,
                        spec,
                        sanity_fn=is_molecule_sane,
                    )
                if stats and stats.get("n", 0) == 0:
                    print(f"[Info] Property stats for {spec.name}: n=0 (no valid deltas computed). Total seeds: {num_seeds}, total gens: {num_gens}")
            except Exception as exc:
                print(f"[Warning] Property stats failed for {spec.name}: {exc}")
                traceback.print_exc()
            
            mean_delta = stats["mean_delta"] if stats else None
            median_delta = stats["median_delta"] if stats else None
            std_delta = stats["std_delta"] if stats else None
            frac_improved = stats["frac_improved"] if stats else None
            
            _register(
                f"property_delta_mean/{spec.name}",
                mean_delta,
                f"Mean {spec.name} delta",
                "Conditional - Property Improvements",
            )
            _register(
                f"property_delta_median/{spec.name}",
                median_delta,
                f"Median {spec.name} delta",
                "Conditional - Property Improvements",
            )
            _register(
                f"property_delta_std/{spec.name}",
                std_delta,
                f"Std {spec.name} delta",
                "Conditional - Property Improvements",
            )
            _register(
                f"property_frac_improved/{spec.name}",
                frac_improved,
                f"Fraction improved ({spec.name})",
                "Conditional - Property Improvements",
            )
        
        mad = None
        try:
            with self._time("conditional/property_preservation_mad"):
                mad = pe_property_preservation_mad(
                    self.epoch_cond_seeds,
                    self.epoch_cond_gens,
                    self.property_specs,
                    sanity_fn=is_molecule_sane,
                )
        except Exception as exc:
            print(f"[Warning] Property preservation MAD failed: {exc}")
            traceback.print_exc()

        macro_mad = mad.get("preservation_mad_macro_mean") if mad else None
        _register(
            "property_preservation_mad_macro",
            macro_mad,
            "Macro mean preservation MAD",
            "Conditional - Property Preservation",
        )
        for spec in self.property_specs:
            per_prop_mad = None
            if mad:
                per_prop_mad = mad.get(f"preservation_mad/{spec.name}")
            _register(
                f"property_preservation_mad/{spec.name}",
                per_prop_mad,
                f"Preservation MAD ({spec.name})",
                "Conditional - Property Preservation",
            )
        
        for spec in self.property_specs:
            res = None
            try:
                with self._time(f"conditional/improvement_under_similarity/{spec.name}"):
                    res = property_improvement_under_similarity_constraint(
                        self.epoch_cond_seeds,
                        self.epoch_cond_gens,
                        spec,
                        sim_range=(0.4, 0.6),
                        sanity_fn=is_molecule_sane,
                    )
            except Exception as exc:
                print(f"[Warning] Constrained improvement failed for {spec.name}: {exc}")
                traceback.print_exc()
            mean_delta = res["mean_delta"] if res else None
            frac_improved = res["frac_improved"] if res else None
            _register(
                f"delta_under_sim/{spec.name}",
                mean_delta,
                f"Delta under similarity ({spec.name})",
                "Conditional - Constrained Improvement",
            )
            _register(
                f"frac_improved_under_sim/{spec.name}",
                frac_improved,
                f"Frac improved under sim ({spec.name})",
                "Conditional - Constrained Improvement",
            )
        
        # Validity fraction (over all generations, including invalid)
        # CRITICAL: Always register these metrics (even if data is empty) for DDP synchronization
        validity_value = None
        try:
            if self.epoch_cond_all_gens:
                with self._time("conditional/validity_fraction"):
                    validity_value = validity_fraction(
                        self.epoch_cond_all_gens, sanity_fn=is_molecule_sane
                    )
        except Exception as exc:
            print(f"[Warning] Validity fraction failed: {exc}")
            traceback.print_exc()
        # Always register, even if None (will be NaN)
        _register(
            "validity_fraction_conditional",
            validity_value,
            "Validity fraction",
            "Conditional - Basic",
        )
        
        # Top-k properties
        # CRITICAL: Always register these metrics (even if data is empty) for DDP synchronization
        for spec in self.property_specs:
            best_value = None
            mean_topk = None
            try:
                if self.epoch_valid_cond_gen_canon:
                    with self._time(f"conditional/topk_best/{spec.name}"):
                        tk = pe_property_top_k_stats(
                            self.epoch_valid_cond_gen_canon,
                            spec,
                            k=3,
                            sanity_fn=is_molecule_sane,
                        )
                    best_value = tk.get("best")
                    mean_topk = tk.get("mean_top_k")
            except Exception as exc:
                print(f"[Warning] Top-k best property failed for {spec.name}: {exc}")
                traceback.print_exc()
            # Always register, even if None (will be NaN)
            _register(
                f"topk_best/{spec.name}_best",
                best_value,
                f"Best {spec.name} value",
                "Conditional - Top-K",
            )
            _register(
                f"topk_best/{spec.name}_mean_topk",
                mean_topk,
                f"Mean top-3 {spec.name}",
                "Conditional - Top-K",
            )
    
    def compute_and_return_dict(self) -> dict:
        """
        Compute metrics and return as dictionary (for evaluate()).
        
        Returns:
            Dict mapping metric names to values
        """
        metrics = {}
        
        # Helper for safe mean
        def _safe_mean(vals: List[float]) -> Optional[float]:
            return float(np.mean(vals)) if vals else None
        
        # Always compute simple metrics
        if self.epoch_cond_all_gens:
            try:
                vf = validity_fraction(self.epoch_cond_all_gens, sanity_fn=is_molecule_sane)
                metrics["validity_fraction_conditional"] = vf
            except Exception as exc:
                print(f"[Warning] (dict) Validity fraction failed: {exc}")
                traceback.print_exc()

        # Uniqueness and novelty (using pre-canonicalized data)
        num_valid = len(self.epoch_valid_cond_gen_canon)
        if num_valid > 0:
            unique_count = len(set(self.epoch_valid_cond_gen_canon))
            metrics["uniqueness_conditional_valid"] = float(unique_count) / num_valid
            
            if self.training_canon_set is not None:
                novel_count = sum(1 for s in self.epoch_valid_cond_gen_canon if s not in self.training_canon_set)
                metrics["novelty_conditional_valid"] = float(novel_count) / num_valid
        
        # Moderate cost metrics (always compute)
        if self.epoch_cond_seeds and self.epoch_cond_gens:
            try:
                sim_mean = mean_similarity_to_seeds(self.epoch_cond_seeds, self.epoch_cond_gens)
                if sim_mean is not None:
                    metrics["similarity_to_seed_mean"] = sim_mean
            except Exception as exc:
                print(f"[Warning] (dict) Similarity to seed failed: {exc}")
                traceback.print_exc()

            # Max similarity to GT
            if self.epoch_cond_gt_lists and len(self.epoch_cond_gt_lists) == len(self.epoch_cond_gens):
                try:
                    gen_max, seed_max, delta = max_similarity_to_gt(
                        self.epoch_cond_gens, self.epoch_cond_seeds, self.epoch_cond_gt_lists
                    )
                    
                    def _mean_non_none(x):
                        xs = [v for v in x if v is not None]
                        return float(np.mean(xs)) if xs else None
                    
                    if gmean := _mean_non_none(gen_max):
                        metrics["max_similarity_to_gt_gen_mean"] = gmean
                    if smean := _mean_non_none(seed_max):
                        metrics["max_similarity_to_gt_seed_mean"] = smean
                    if dmean := _mean_non_none(delta):
                        metrics["max_similarity_to_gt_delta_mean"] = dmean
                except Exception as exc:
                    print(f"[Warning] (dict) Max similarity to GT failed: {exc}")
                    traceback.print_exc()
        
        # Property improvements (always compute)
        for spec in self.property_specs:
            try:
                stats = pe_property_improvement_stats(
                    self.epoch_cond_seeds, self.epoch_cond_gens, spec,
                    sanity_fn=is_molecule_sane
                )
                if stats["mean_delta"] is not None:
                    metrics[f"property_delta_mean/{spec.name}"] = stats["mean_delta"]
                    metrics[f"property_delta_median/{spec.name}"] = stats["median_delta"]
                    metrics[f"property_delta_std/{spec.name}"] = stats["std_delta"]
                    if stats["frac_improved"] is not None:
                        metrics[f"property_frac_improved/{spec.name}"] = stats["frac_improved"]
            except Exception as exc:
                print(f"[Warning] (dict) Property stats failed for {spec.name}: {exc}")
                traceback.print_exc()

        # V&I&NM rate: fraction of ALL conditional generations that are valid, improved, and not memorized
        # Uses the V_N_I flags tracked per property during memorization updates
        for spec in self.property_specs:
            prop_name = spec.name
            if prop_name in self.epoch_seed_to_gen_V_N_I_flags:
                flags = self.epoch_seed_to_gen_V_N_I_flags[prop_name]
                if flags:
                    n_vinm = sum(1 for f in flags if f)
                    n_total = len(flags)
                    metrics[f"V_N_I_rate/{prop_name}"] = float(n_vinm) / n_total
                    metrics[f"V_N_I_count/{prop_name}"] = n_vinm
                    metrics[f"V_N_I_total/{prop_name}"] = n_total

        # Distance and neighborhood metrics (always compute)
        if m := _safe_mean(self.epoch_dist_to_gt):
            metrics[f"{self.distance_name}_distance_to_gt_conditional_mean"] = m
        if m := _safe_mean(self.epoch_input_to_gt_min_dist):
            metrics[f"{self.distance_name}_distance_input_to_gt_conditional_mean"] = m
        if self.epoch_input_to_gt_min_dist and self.epoch_seed_to_gen_dist:
            n = min(len(self.epoch_input_to_gt_min_dist), len(self.epoch_seed_to_gen_dist))
            if n > 0:
                diff_vals = [self.epoch_input_to_gt_min_dist[i] - self.epoch_seed_to_gen_dist[i] for i in range(n)]
                if diff_vals:
                    metrics[f"{self.distance_name}_seed_to_gt_minus_seed_to_gen_conditional_mean"] = float(np.mean(diff_vals))
        
        # Filtered distance metrics for valid+improved+not_memorized samples (per property)
        for spec in self.property_specs:
            prop_name = spec.name
            
            # Filter seed_to_gen distances
            if prop_name in self.epoch_seed_to_gen_V_N_I_flags:
                flags = self.epoch_seed_to_gen_V_N_I_flags[prop_name]
                if len(flags) == len(self.epoch_seed_to_gen_dist):
                    filtered_distances = [
                        self.epoch_seed_to_gen_dist[i] 
                        for i in range(len(flags)) 
                        if flags[i]
                    ]
                    if filtered_distances:
                        metrics[f"{self.distance_name}_seed_to_gen_V_N_I_mean/{prop_name}"] = float(np.mean(filtered_distances))
            
            # Filter input_to_gt distances
            if prop_name in self.epoch_input_to_gt_V_N_I_flags:
                flags = self.epoch_input_to_gt_V_N_I_flags[prop_name]
                if len(flags) == len(self.epoch_input_to_gt_min_dist):
                    filtered_distances = [
                        self.epoch_input_to_gt_min_dist[i] 
                        for i in range(len(flags)) 
                        if flags[i]
                    ]
                    if filtered_distances:
                        metrics[f"{self.distance_name}_distance_input_to_gt_V_N_I_mean/{prop_name}"] = float(np.mean(filtered_distances))
            
            # Compute difference (seed_to_gt - seed_to_gen) for V_N_I samples
            seed_v_n_i_distances = []
            input_v_n_i_distances = []
            
            if prop_name in self.epoch_seed_to_gen_V_N_I_flags:
                seed_flags = self.epoch_seed_to_gen_V_N_I_flags[prop_name]
                if len(seed_flags) == len(self.epoch_seed_to_gen_dist):
                    seed_v_n_i_distances = [
                        self.epoch_seed_to_gen_dist[i] 
                        for i in range(len(seed_flags)) 
                        if seed_flags[i]
                    ]
            
            if prop_name in self.epoch_input_to_gt_V_N_I_flags:
                input_flags = self.epoch_input_to_gt_V_N_I_flags[prop_name]
                if len(input_flags) == len(self.epoch_input_to_gt_min_dist):
                    input_v_n_i_distances = [
                        self.epoch_input_to_gt_min_dist[i] 
                        for i in range(len(input_flags)) 
                        if input_flags[i]
                    ]
            
            # Compute difference if we have both sets of samples
            # Since the lists may not align perfectly (input_to_gt is a subset), 
            # we compute means separately and log the difference of means
            if seed_v_n_i_distances and input_v_n_i_distances:
                seed_mean = float(np.mean(seed_v_n_i_distances))
                input_mean = float(np.mean(input_v_n_i_distances))
                diff_mean = input_mean - seed_mean
                metrics[f"{self.distance_name}_seed_to_gt_minus_seed_to_gen_V_N_I_mean/{prop_name}"] = diff_mean
        
        if m := _safe_mean(self.epoch_neigh_median_dist):
            metrics[f"{self.distance_name}_neighborhood_distance_median_conditional"] = m
        
        if m := _safe_mean(self.epoch_neigh_p75_dist):
            metrics[f"{self.distance_name}_neighborhood_distance_p75_conditional"] = m
        
        if m := _safe_mean(self.epoch_neigh_nn_dist):
            metrics[f"{self.distance_name}_nearest_neighbor_distance_conditional"] = m
        
        # Distance percentile aggregate statistics
        if self.epoch_distance_percentiles:
            metrics[f"{self.distance_name}_distance_percentile_conditional_mean"] = float(np.mean(self.epoch_distance_percentiles))
            metrics[f"{self.distance_name}_distance_percentile_conditional_median"] = float(np.median(self.epoch_distance_percentiles))
            metrics[f"{self.distance_name}_distance_percentile_conditional_p25"] = float(np.percentile(self.epoch_distance_percentiles, 25))
            metrics[f"{self.distance_name}_distance_percentile_conditional_p75"] = float(np.percentile(self.epoch_distance_percentiles, 75))
        
        # Distance prompt agreement aggregate statistics
        if self.epoch_distance_prompt_agreement:
            agreement_rate = float(sum(self.epoch_distance_prompt_agreement)) / len(self.epoch_distance_prompt_agreement)
            metrics[f"{self.distance_name}_distance_prompt_agreement_rate"] = agreement_rate
        
        # W1 deltas (always compute)
        for feat_name in FEATURE_NAMES:
            if m := _safe_mean(self.epoch_w1_deltas[feat_name]):
                metrics[f"w1_delta_conditional/{feat_name}"] = m
        
        # KDE log-likelihoods (always compute)
        if self.epoch_kde_ll_in and self.epoch_kde_ll_gen:
            ll_in_mean = float(np.mean(self.epoch_kde_ll_in))
            ll_gen_mean = float(np.mean(self.epoch_kde_ll_gen))
            metrics["kde_ll_in_conditional"] = ll_in_mean
            metrics["kde_ll_gen_conditional"] = ll_gen_mean
            metrics["kde_ll_delta_conditional"] = ll_gen_mean - ll_in_mean
        
        # GT delta (always compute)
        if self.epoch_gt_min_delta:
            metrics[f"{self.distance_name}_gt_closest_delta_conditional_mean"] = float(np.mean(self.epoch_gt_min_delta))
        
        # Sample counts
        metrics["cond_pairs_count"] = len(self.epoch_cond_seeds)
        metrics["cond_valid_count"] = len(self.epoch_valid_cond_gen_canon)
        
        # Prompt agreement aggregate statistics (per property)
        for prop_name in sorted(self.epoch_prompt_direction_agreement.keys()):
            direction_matches = self.epoch_prompt_direction_agreement[prop_name]
            if direction_matches:
                direction_agreement_rate = float(sum(direction_matches)) / len(direction_matches)
                metrics[f"{prop_name}_prompt_direction_agreement_rate"] = direction_agreement_rate
                metrics[f"{prop_name}_prompt_direction_count"] = len(direction_matches)

                # Non-memorized version
                if prop_name in self.epoch_prompt_direction_memorized:
                    memorized_flags = self.epoch_prompt_direction_memorized[prop_name]
                    if len(memorized_flags) == len(direction_matches):
                        non_memorized_matches = [
                            match for match, is_mem in zip(direction_matches, memorized_flags) if not is_mem
                        ]
                        if non_memorized_matches:
                            non_memorized_rate = float(sum(non_memorized_matches)) / len(non_memorized_matches)
                            metrics[f"{prop_name}_prompt_direction_agreement_rate_non_memorized"] = non_memorized_rate
        
        # Strict direction agreement (excludes "any" direction samples)
        for prop_name in sorted(self.epoch_prompt_direction_agreement_strict.keys()):
            strict_matches = self.epoch_prompt_direction_agreement_strict[prop_name]
            if strict_matches:
                strict_rate = float(sum(strict_matches)) / len(strict_matches)
                metrics[f"{prop_name}_prompt_direction_agreement_rate_strict"] = strict_rate
                metrics[f"{prop_name}_prompt_direction_count_strict"] = len(strict_matches)
                if prop_name in self.epoch_prompt_direction_memorized_strict:
                    mem_flags = self.epoch_prompt_direction_memorized_strict[prop_name]
                    if len(mem_flags) == len(strict_matches):
                        nm = [m for m, is_mem in zip(strict_matches, mem_flags) if not is_mem]
                        if nm:
                            metrics[f"{prop_name}_prompt_direction_agreement_rate_strict_non_memorized"] = float(sum(nm)) / len(nm)

        for prop_name in sorted(self.epoch_prompt_magnitude_agreement.keys()):
            magnitude_matches = self.epoch_prompt_magnitude_agreement[prop_name]
            if magnitude_matches:
                magnitude_agreement_rate = float(sum(magnitude_matches)) / len(magnitude_matches)
                metrics[f"{prop_name}_prompt_magnitude_agreement_rate"] = magnitude_agreement_rate
                
                # Non-memorized version
                if prop_name in self.epoch_prompt_magnitude_memorized:
                    memorized_flags = self.epoch_prompt_magnitude_memorized[prop_name]
                    if len(memorized_flags) == len(magnitude_matches):
                        non_memorized_matches = [
                            match for match, is_mem in zip(magnitude_matches, memorized_flags) if not is_mem
                        ]
                        if non_memorized_matches:
                            non_memorized_rate = float(sum(non_memorized_matches)) / len(non_memorized_matches)
                            metrics[f"{prop_name}_prompt_magnitude_agreement_rate_non_memorized"] = non_memorized_rate
            
            # Bin error statistics (magnitude of error) - only for non-memorized samples
            if prop_name in self.epoch_prompt_magnitude_bin_errors:
                bin_errors = self.epoch_prompt_magnitude_bin_errors[prop_name]
                if bin_errors and prop_name in self.epoch_prompt_magnitude_bin_errors_memorized:
                    memorized_flags = self.epoch_prompt_magnitude_bin_errors_memorized[prop_name]
                    if len(memorized_flags) == len(bin_errors):
                        # Only compute statistics for non-memorized samples
                        non_memorized_errors = [
                            e for e, is_mem in zip(bin_errors, memorized_flags) if not is_mem
                        ]
                        if non_memorized_errors:
                            non_memorized_abs_errors = [abs(e) for e in non_memorized_errors]
                            nm_mean_abs_error = float(np.mean(non_memorized_abs_errors))
                            nm_mean_error = float(np.mean(non_memorized_errors))
                            nm_median_abs_error = float(np.median(non_memorized_abs_errors))
                            nm_p75_abs_error = float(np.percentile(non_memorized_abs_errors, 75))
                            nm_p95_abs_error = float(np.percentile(non_memorized_abs_errors, 95))
                            
                            metrics[f"{prop_name}_prompt_magnitude_bin_error_mae"] = nm_mean_abs_error
                            metrics[f"{prop_name}_prompt_magnitude_bin_error_mean"] = nm_mean_error
                            metrics[f"{prop_name}_prompt_magnitude_bin_error_median_abs"] = nm_median_abs_error
                            metrics[f"{prop_name}_prompt_magnitude_bin_error_p75_abs"] = nm_p75_abs_error
                            metrics[f"{prop_name}_prompt_magnitude_bin_error_p95_abs"] = nm_p95_abs_error
        
        # Distance prompt agreement aggregate statistics
        if self.epoch_distance_prompt_agreement:
            agreement_rate = float(sum(self.epoch_distance_prompt_agreement)) / len(self.epoch_distance_prompt_agreement)
            metrics[f"{self.distance_name}_distance_prompt_agreement_rate"] = agreement_rate
            
            # Non-memorized version
            if len(self.epoch_distance_prompt_memorized) == len(self.epoch_distance_prompt_agreement):
                non_memorized_matches = [
                    match for match, is_mem in zip(self.epoch_distance_prompt_agreement, self.epoch_distance_prompt_memorized) if not is_mem
                ]
                if non_memorized_matches:
                    non_memorized_rate = float(sum(non_memorized_matches)) / len(non_memorized_matches)
                    metrics[f"{self.distance_name}_distance_prompt_agreement_rate_non_memorized"] = non_memorized_rate
        
        # All properties agree aggregate statistics
        if self.epoch_all_properties_direction_agree:
            all_agree_rate = float(sum(self.epoch_all_properties_direction_agree)) / len(self.epoch_all_properties_direction_agree)
            metrics["all_properties_direction_agree_rate"] = all_agree_rate
            
            # Non-memorized version
            if len(self.epoch_all_properties_direction_memorized) == len(self.epoch_all_properties_direction_agree):
                non_memorized_matches = [
                    match for match, is_mem in zip(self.epoch_all_properties_direction_agree, self.epoch_all_properties_direction_memorized) if not is_mem
                ]
                if non_memorized_matches:
                    non_memorized_rate = float(sum(non_memorized_matches)) / len(non_memorized_matches)
                    metrics["all_properties_direction_agree_rate_non_memorized"] = non_memorized_rate
        
        if self.epoch_all_properties_magnitude_agree:
            all_agree_rate = float(sum(self.epoch_all_properties_magnitude_agree)) / len(self.epoch_all_properties_magnitude_agree)
            metrics["all_properties_magnitude_agree_rate"] = all_agree_rate
            
            # Non-memorized version
            if len(self.epoch_all_properties_magnitude_memorized) == len(self.epoch_all_properties_magnitude_agree):
                non_memorized_matches = [
                    match for match, is_mem in zip(self.epoch_all_properties_magnitude_agree, self.epoch_all_properties_magnitude_memorized) if not is_mem
                ]
                if non_memorized_matches:
                    non_memorized_rate = float(sum(non_memorized_matches)) / len(non_memorized_matches)
                    metrics["all_properties_magnitude_agree_rate_non_memorized"] = non_memorized_rate
        
        # Memorization metrics
        if "memorization" in self.expensive_metrics_to_compute:
            if self.epoch_memorized_checked > 0:
                metrics["memorization_rate_conditional"] = (
                    self.epoch_memorized_count / self.epoch_memorized_checked
                )
            for ref_name in sorted(self.reference_canon_sets_by_name.keys()):
                checked = int(self.epoch_memorized_checked_by_ref.get(ref_name, 0))
                count = int(self.epoch_memorized_count_by_ref.get(ref_name, 0))
                if checked > 0:
                    metrics[f"memorization_rate_conditional/{ref_name}"] = (count / checked)

        # NEW: Distributional metrics for conditional generations (SNN, FCD, diversity, etc.)
        distributional = self._compute_distributional_metrics("_conditional")
        metrics.update(distributional)

        return metrics

