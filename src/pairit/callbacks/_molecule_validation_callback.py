import random
import time
import sys
import os
import hashlib
import re
from collections.abc import Callable
from typing import Optional, List, Dict, Tuple, Collection
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from atomic_datasets.utils.rdkit import is_molecule_sane

from .molecule_validation.property_registry import (
    get_property_specs,
    DEFAULT_PROPERTIES,
    compute_all_properties_for_smiles,
    get_seed_mol_stats,
    load_dataset_statistics_from_files,
)
from .molecule_validation.timing import TimingTracker
from .molecule_metrics import tanimoto_similarity_smiles
from . import sascorer

# disable logs
RDLogger.DisableLog('rdApp.*')


# Import refactored validation modules
from .molecule_validation import (
    ValidationPrinter,
    shorten_str,
    MetricRegistry,
    SmilesUtils,
    MoleculeGenerator,
    ConditionalMetricsComputer,
    UnconditionalMetricsComputer,
)

# Import RDKit and atomic_datasets for validity checking
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from atomic_datasets.utils.rdkit import is_molecule_sane
RDKIT_AVAILABLE = True

from pairit.eval import PropertySpec
from .molecule_metrics import FEATURE_NAMES as METRIC_FEATURE_NAMES

# Import distance functions from pair_gen
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from pair_gen.utils_mol import get_tanimoto_distance, get_shape_tanimoto_distance, generate_paired_smiles_grid, generate_ground_truth_grid, compute_single_molecule_snn

"""
MOLECULE VALIDATION CALLBACK - LOGGED METRICS DOCUMENTATION
============================================================

This module contains callbacks that log various metrics during molecule generation and validation.
Below is a comprehensive guide to all metrics that get logged:

## MOLECULE VALIDITY METRICS
These metrics measure the fraction of generated molecules that pass RDKit validation:

1. **val_molecule_validity** (float, 0.0-1.0)
   - Overall validity rate for UNCONDITIONAL generation, where the model generates molecule pairs from scratch
   - Formula: (valid_first_molecules + valid_second_molecules) / (2 * total_samples)
   - Logged: During validation epochs (on_validation_epoch_end)

2. **val_molecule_validity_first** (float, 0.0-1.0)
   - Validity rate for first molecules in UNCONDITIONAL generation
   - Formula: valid_first_molecules / total_samples
   - Logged: During validation epochs (on_validation_epoch_end)
   - Expected to be < 1.0 since these are generated from scratch

3. **val_molecule_validity_second** (float, 0.0-1.0)
   - Validity rate for second molecules in UNCONDITIONAL generation
   - Formula: valid_second_molecules / total_samples
   - Logged: During validation epochs (on_validation_epoch_end)
   - Expected to be < 1.0 since these are generated from scratch

4. **val_molecule_validity_conditional** (float, 0.0-1.0)
   - Validity rate for CONDITIONAL generation, where model generates improved molecules given input molecules
   - Formula: valid_conditional_generations / batch_size
   - Logged: During validation batches (on_validation_batch_end)
   - Input molecules from dataset are always valid (1.0), so this measures validity of generated improvements

## MOLECULAR DISTANCE METRICS
These metrics measure molecular similarity using Tanimoto and Shape Tanimoto distances:

5. **val_tanimoto_distance_input** (float, 0.0-1.0)
   - Tanimoto distance between the two input molecules in training pairs
   - Measures structural similarity of input molecule pairs
   - Logged: During validation batches (on_validation_batch_end)

6. **val_shape_tanimoto_distance_input** (float, 0.0-1.0)
   - Shape-based Tanimoto distance between the two input molecules in training pairs
   - Measures 3D shape similarity of input molecule pairs
   - Logged: During validation batches (on_validation_batch_end)

7. **val_tanimoto_distance_conditional** (float, 0.0-1.0)
   - Tanimoto distance between conditioning molecule and generated molecule
   - Measures how structurally similar the generated molecule is to the input
   - Logged: During conditional generation (on_validation_batch_end)

8. **val_shape_tanimoto_distance_conditional** (float, 0.0-1.0)
   - Shape-based Tanimoto distance between conditioning molecule and generated molecule
   - Measures how similar the 3D shape of generated molecule is to the input
   - Logged: During conditional generation (on_validation_batch_end)

9. **val_tanimoto_distance_unconditional** (float, 0.0-1.0)
   - Tanimoto distance between first and second molecules in unconditional generation
   - Measures structural similarity in unconditionally generated molecule pairs
   - Logged: During unconditional generation (on_validation_epoch_end)

10. **val_shape_tanimoto_distance_unconditional** (float, 0.0-1.0)
    - Shape-based Tanimoto distance between first and second molecules in unconditional generation
    - Measures 3D shape similarity in unconditionally generated molecule pairs
    - Logged: During unconditional generation (on_validation_epoch_end)

## FIRST MOLECULE PERPLEXITY METRICS
These metrics measure model performance on predicting the first molecule in pairs:

11. **val_first_molecule_loss** (float, typically 0.0-10.0)
    - Average cross-entropy loss for predicting the first molecule in each pair
    - Lower values indicate better prediction accuracy
    - Logged: During validation batches (on_validation_batch_end)

12. **val_first_molecule_perplexity** (float, typically 1.0-1000.0)
    - Perplexity (exp(loss)) for predicting the first molecule in each pair
    - Lower values indicate better prediction accuracy
    - Logged: During validation batches (on_validation_batch_end)

13. **val_first_molecule_samples** (int)
    - Number of valid samples used to compute first molecule metrics
    - Used for debugging and understanding sample sizes
    - Logged: During validation batches (on_validation_batch_end)

## GENERATION TYPES EXPLAINED

**Conditional Generation**: Model generates a molecule given an input molecule as conditioning
- Uses input molecules from the validation dataloader
- Logs: validity_conditional, tanimoto_distance_conditional, shape_tanimoto_distance_conditional

**Unconditional Generation**: Model generates molecule pairs from scratch using CLS tokens
- Starts generation from special CLS tokens without conditioning
- Logs: validity (overall, first, second), tanimoto_distance_unconditional, shape_tanimoto_distance_unconditional

**First Molecule Analysis**: Model predicts the first molecule in training pairs
- Computes loss/perplexity on the first molecule of each input pair
- Logs: first_molecule_loss, first_molecule_perplexity, first_molecule_samples

## INTERPRETATION GUIDE

- **Validity metrics (0.0-1.0)**: Higher is better (1.0 = all molecules valid)
- **Distance metrics (0.0-1.0)**: 
  - 0.0 = identical molecules
  - 1.0 = completely different molecules
  - For conditional generation: moderate values (0.3-0.7) often indicate good diversity while maintaining similarity
- **Loss metrics**: Lower is better
- **Perplexity metrics**: Lower is better (1.0 = perfect prediction)

## WHEN METRICS ARE LOGGED

- **During Training**: All metrics logged at validation steps/epochs
- **During Evaluation**: Same metrics returned by evaluate() methods
- **Logging Frequency**: Controlled by PyTorch Lightning's validation frequency settings
"""

class MoleculeValidationCallback(Callback):
    """
    PyTorch Lightning callback for molecule validation during training.
    
    This callback performs molecule generation and validity checking during validation
    epochs. It supports both unconditional generation (starting from CLS token) and
    conditional generation (using input sequences as conditioning). The callback
    generates molecules, validates them using RDKit, and computes various molecular
    similarity metrics.
    
    The callback operates in two modes:
    1. **Conditional generation**: Extracts conditioning sequences from input batches
       and generates new molecules based on those sequences. --> called during validation batch (since needs access to data)
        Currently, operates on whole batches at once, but should yield approximately num_unconditional_generations molecules
    2. **Unconditional generation**: Generates molecules from scratch starting with
       a CLS token. --> called at end of validation epoch
        Currently, generates num_unconditional_generations molecules precisely
    
    Both modes include comprehensive timing measurements and validity checking.

    This class also contains many helper functions as static class functions.
    
    Attributes
    ----------
    num_unconditional_generations : int
        The number of unconditional molecules to generate during validation.
        Default is 100.
    num_conditional_generations : int
        The number of conditional molecules to generate during validation.
        Default is None.
    save_generated_samples : bool
        Whether to save generated molecules to disk. Default is True.
    max_samples_to_save : int
        Target number of samples to save. Default is 50.
    save_dir : str
        Directory to save generated molecules. Default is "generated_molecules".
    """
    
    # Constants
    HASH_SPACE_SIZE = 2**32  # Size of hash space for deterministic sample selection
    EVALUATE_SPECIAL_STEP_ID = 9999  # Special step ID used in evaluate() method
    DEFAULT_MAX_SAMPLES_TO_SAVE = 50  # Default maximum samples to save
    
    def __init__(self, num_unconditional_generations: int = 100, 
                 num_conditional_generations: Optional[int] = None,
                 save_generated_samples: bool = True, 
                 max_samples_to_save: int = 50, save_dir: str = "generated_molecules",
                 neighbor_provider: Optional[Callable[[str], List[str]]] = None,
                 distance_fn: Optional[Callable[[str, str], float]] = None,
                 distance_name: str = "tanimoto",
                 property_specs: Optional[List[PropertySpec]] = None,
                 property_names: Optional[List[str]] = None,
                 training_smiles_canonical: Optional[Collection[str]] = None,
                 reference_smiles_for_unconditional: Optional[Collection[str]] = None,
                 training_smiles_file: Optional[str | List[str]] = None,
                 reference_smiles_file: Optional[str | List[str]] = None,
                 # Unified reference configuration (preferred).
                 # List of dicts with at least {name: <str>, path|file|files: <str|list[str]>}.
                 # Each named reference is used for:
                 # - membership metrics (memorization/novelty) via canonical SMILES
                 # - distribution comparison metrics (SNN/FCD/etc) via raw SMILES
                 reference_smiles_files: Optional[List[dict]] = None,
                 # Back-compat: legacy named memorization references (deprecated; prefer reference_smiles_files).
                 memorization_reference_smiles_files: Optional[dict[str, str | List[str]] | List[dict]] = None,
                 smiles_column: str = "smiles",
                 split_column: Optional[str] = "split",
                 training_split_value: str = "train",
                 fcd_min_required: int = 10,
                 expensive_metrics_to_compute: Optional[List[str]] = [],
                 check_memorization_from_any_reference: bool = True,
                 visualization_property_name: str = "logp",
                 temperature: float = 0.95,
                 save_pdfs: bool = False,
                 num_generations_per_seed: int = 1,
                 utility_key: Optional[str] = None,
                 utility_direction: Optional[str] = None,
                 metric_prefix: Optional[str] = None,
                 only_dataloader_idx: Optional[int] = None,
                 unconditional_prompt_text: Optional[str] = None,
                 split_metrics_by_dataset: str = "auto",
                 unconditional_per_dataset: bool = False,
                 prompting: Optional[dict] = None,
                 dataset_stats: Optional[Dict[str, int]] = None,
                 export_pairit_csv: bool = False,
                 pairit_method_name: str = "pairit",
                 pairit_dataset_name: Optional[str] = None,
                 pairit_seed_label_column: Optional[str] = None,
                 num_conditional_metric_samples: int | None = None,
                 generation_only: bool = False,
                 pairit_generation_seed: int | None = None,
                 export_prompt_metadata: bool = False):
        """
        Initialize the MoleculeValidationCallback.
        
        Parameters
        ----------
        num_unconditional_generations : int, default=100
            The number of unconditional molecules to generate during each validation epoch.
            This controls how many molecules are generated from scratch starting with
            the CLS token.
        num_conditional_generations : Optional[int], default=None
            The number of conditional molecules to generate during each validation epoch.
            If None, defaults to num_unconditional_generations. This allows separate control
            over conditional vs unconditional generation (e.g., 10 conditional, 64 unconditional).
        save_generated_samples : bool, default=False
            Whether to save generated molecules to disk for later analysis.
        max_samples_to_save : int, default=50
            Target number of samples to save. The actual save fraction is computed as
            max_samples_to_save / num_unconditional_generations. For example, if 
            num_unconditional_generations=100 and max_samples_to_save=50, then 50% of
            samples will be saved. Uses deterministic hash-based sampling.
        save_dir : str, default="generated_molecules"
            Directory where generated molecules will be saved as parquet files.
        expensive_metrics_to_compute : Optional[List[str]], default=[]
            List of expensive metrics to compute. If None, defaults to all available metrics:
            ["snn", "fcd", "internal_diversity", "fragment_scaffold", "w1_kde"]. Available options:
            - "snn": Single Nearest Neighbor similarity to reference set
            - "fcd": Fréchet ChemNet Distance (requires ChemNet model)
            - "internal_diversity": Internal diversity within generated set
            - "fragment_scaffold": Fragment and scaffold similarity
            - "w1_kde": Wasserstein-1 and KDE marginal distributions
        check_memorization_from_any_reference : bool, default=True
            If True, the generic "is_memorized" flag checks if the molecule is in ANY of the
            reference sets (union of all reference_smiles_files). If False (default), only checks
            the default reference set. Individual per-reference memorization flags are always
            logged separately regardless of this setting. This affects the generic
            "memorization_rate_conditional" metric and the "is_memorized" column in parquet files,
            as well as filtering for "*_non_memorized" prompt agreement metrics.
        training_smiles_file : Optional[str | List[str]], default=None
            Path to parquet file(s) containing training SMILES. Can be either a single file path
            or a list of file paths. If provided, SMILES will be loaded and canonicalized for
            novelty computation. If a list is provided, SMILES from all files will be combined.
        reference_smiles_file : Optional[str | List[str]], default=None
            Path to parquet file(s) containing reference SMILES for dataset-level metrics.
            Can be either a single file path or a list of file paths. If provided, these SMILES
            will be used as the reference set for metrics like SNN. If a list is provided,
            SMILES from all files will be combined.
        smiles_column : str, default="smiles"
            Name of the column containing SMILES strings in the parquet files.
        split_column : Optional[str], default="split"
            Name of the column containing split information (train/val/test). If None, all
            SMILES from the file(s) will be loaded.
        training_split_value : str, default="train"
            Value in split_column to filter for when loading SMILES (e.g., "train", "val", "test").
        property_specs : Optional[List[PropertySpec]], default=None
            List of PropertySpec objects defining which molecular properties to compute and track.
            If None and property_names is also None, defaults to ["logp", "qed", "sa"].
            Cannot be specified together with property_names.
        property_names : Optional[List[str]], default=None
            List of property names to compute (e.g., ["logp", "qed", "sa"]). Simpler alternative
            to property_specs for common properties. Available options include: "logp", "qed", "sa",
            "mw", "tpsa", "hbd", "hba", "rotatable_bonds". Cannot be specified together with property_specs.
            Custom properties can be registered using register_custom_property() from property_registry.
        temperature : float, default=0.95
            Sampling temperature for molecule generation. Controls randomness in generation:
            - Lower values (e.g., 0.7) make generation more deterministic
            - Higher values (e.g., 1.0) make generation more random
            - Default of 0.95 balances exploration and quality
        save_pdfs : bool, default=False
            Whether to save PDF visualizations of generated molecules. PDFs are memory-intensive,
            so this is disabled by default. PNGs are always generated for wandb logging.
        num_generations_per_seed : int, default=1
            Number of molecules to generate per conditioning molecule (conditional generation only).
            If set to N > 1, the model will generate N different molecules for each seed molecule.
            All generations are saved and contribute to metrics. For example, with 64 seeds and
            num_generations_per_seed=5, you'll get 320 total conditional generations.
        utility_key : Optional[str], default=None
            The property name that corresponds to the utility used during training (e.g., "logp", "gap", "sa").
            If provided along with utility_direction, the improvement direction for this property will be
            overridden to match the training configuration, ensuring consistent improvement metrics.
        utility_direction : Optional[str], default=None
            Direction for utility improvement: "increasing" (higher is better) or "decreasing" (lower is better).
            If provided along with utility_key, overrides the default direction from the property registry
            for the matching property. This ensures improvement metrics align with the training objective.
            
        Notes
        -----
        When save_generated_samples=True, the callback saves different data structures
        for conditional vs unconditional generation:
        
        **Conditional Generation** (saved during validation batches):
        - smiles_generated: The newly generated molecule SMILES
        - smiles_conditioning: The input molecule used as conditioning
        - smiles_input_1, smiles_input_2: Original input molecule pair from dataset
        - tokens_generated: Generated token sequence
        - tokens_conditioning: Input conditioning token sequence
        - is_valid: Whether generated molecule is valid
        - tanimoto_distance, shape_tanimoto_distance: Similarity metrics
        - generation_time: Time taken to generate
        - sample_id: Unique identifier
        
        **Unconditional Generation** (saved during validation epochs):
        - smiles_generated: First generated molecule SMILES
        - smiles_second: Second generated molecule SMILES (if present)
        - tokens_generated: Generated token sequence
        - is_valid: Whether first molecule is valid
        - is_second_valid: Whether second molecule is valid
        - tanimoto_distance, shape_tanimoto_distance: Similarity between generated molecules
        - generation_time: Time taken to generate
        - sample_id: Unique identifier
        """
        super().__init__()
        self.num_unconditional_generations = num_unconditional_generations
        self.num_conditional_generations = num_conditional_generations if num_conditional_generations is not None else num_unconditional_generations
        self.num_conditional_metric_samples = num_conditional_metric_samples  # None = all

        self.save_generated_samples = save_generated_samples
        self.max_samples_to_save = max_samples_to_save

        # Compute save_fraction based on max_samples_to_save and num_unconditional_generations
        self.save_fraction = min(1.0, max_samples_to_save / max(1, num_unconditional_generations))

        self.save_dir = Path(save_dir)
        self.visualization_property_name = visualization_property_name
        self.temperature = temperature
        self.save_pdfs = save_pdfs
        self.num_generations_per_seed = num_generations_per_seed

        # Per-dataset namespacing / multi-dataloader support
        self.metric_prefix = metric_prefix
        self.only_dataloader_idx = only_dataloader_idx
        self.unconditional_prompt_text = unconditional_prompt_text
        self.split_metrics_by_dataset = split_metrics_by_dataset
        self.unconditional_per_dataset = bool(unconditional_per_dataset)

        # Optional property prompting config: used to (a) select which properties to compute/log,
        # and (b) decode prompt bins into numeric ranges for saved samples.
        self._prompting = prompting or {}
        self._prompt_prop_names: list[str] = []
        self._value_binning_default = None
        self._delta_binning_default = None
        self._value_binning_per_prop = {}
        self._delta_binning_per_prop = {}
        self._prop_value_width = 2
        self._prop_delta_width = 2

        try:
            prop_cfg = (self._prompting or {}).get("properties", None) or {}
            if bool(prop_cfg.get("enabled", False)):
                self._prompt_prop_names = [str(x) for x in (prop_cfg.get("names", None) or [])]
                if self._prompt_prop_names:
                    # Compute union of prompting properties and callback property_names
                    # This ensures we compute all properties needed for both prompting and validation
                    callback_prop_names = property_names if property_names is not None else []
                    # The generic target slot is pair metadata, not a descriptor.
                    all_prop_names = list(dict.fromkeys(
                        [p for p in self._prompt_prop_names if p != "target"] + callback_prop_names
                    ))
                    property_names = all_prop_names

                from pairit.prompting.property_prompting import PropertyBinningConfig as _PBC

                vb = prop_cfg.get("value_binning", None) or {}
                db = prop_cfg.get("delta_binning", None) or {}

                def _as_range_tuple(x):
                    if x is None:
                        return None
                    try:
                        seq = list(x)
                    except Exception:
                        return None
                    if len(seq) != 2:
                        return None
                    try:
                        return (float(seq[0]), float(seq[1]))
                    except Exception:
                        return None

                def _mk_default(d: dict, *, fallback_bins: int) -> _PBC:
                    mode = str(d.get("mode", "fixed"))
                    default = d.get("default", None) or {}
                    num_bins = int(default.get("num_bins", fallback_bins))
                    rng_t = _as_range_tuple(default.get("range", None))
                    return _PBC(num_bins=num_bins, range=rng_t, mode=mode, clip=True)

                self._value_binning_default = _mk_default(vb, fallback_bins=32)
                self._delta_binning_default = _mk_default(db, fallback_bins=16)
                for pname, spec in (vb.get("per_property", None) or {}).items():
                    s = spec or {}
                    rng_t = _as_range_tuple(s.get("range", None))
                    self._value_binning_per_prop[str(pname)] = _PBC(
                        num_bins=int(s.get("num_bins", self._value_binning_default.num_bins)),
                        range=rng_t,
                        mode=str(s.get("mode", self._value_binning_default.mode)),
                        clip=True,
                    )
                for pname, spec in (db.get("per_property", None) or {}).items():
                    s = spec or {}
                    rng_t = _as_range_tuple(s.get("range", None))
                    self._delta_binning_per_prop[str(pname)] = _PBC(
                        num_bins=int(s.get("num_bins", self._delta_binning_default.num_bins)),
                        range=rng_t,
                        mode=str(s.get("mode", self._delta_binning_default.mode)),
                        clip=True,
                    )

                self._prop_value_width = max(2, len(str(self._value_binning_default.num_bins - 1)))
                self._prop_delta_width = max(2, len(str(self._delta_binning_default.num_bins - 1)))
        except Exception:
            # If prompting config is malformed, fail fast so we don't silently mis-log.
            raise
        
        # Distance prompting config (for distance prompt agreement checking)
        self._distance_binning_mode: Optional[str] = None
        self._distance_num_bins: Optional[int] = None
        try:
            dist_cfg = (self._prompting or {}).get("distance", None) or {}
            if bool(dist_cfg.get("enabled", False)):
                binning_cfg = dist_cfg.get("binning", None) or {}
                self._distance_binning_mode = str(binning_cfg.get("mode", "curve_fit"))
                self._distance_num_bins = int(binning_cfg.get("num_bins", 20))
        except Exception:
            # If distance prompting config is malformed, just skip it
            pass

        # Internal per-dataset metric computers (used only when split is enabled)
        self._cond_metrics_by_key = {}
        self._uncond_metrics_by_key = {}
        self._dataset_key_to_id = {}

        # Track how many conditional generations we've performed this epoch (per process)
        self._conditional_generated_this_epoch = 0
        self._prompt_requests = {}
        # In mixed validation (multiple val dataloaders), apply conditional quotas per dataset_key.
        self._conditional_generated_this_epoch_by_key: Dict[str, int] = {}
        
        # Track print statements to limit output pollution
        self._print_count = 0
        self._max_prints = 2
        
        # Create deterministic random state for consistent sampling
        self._random_state = random.Random(42)  # Fixed seed for reproducibility
        
        # Comprehensive metrics initialization
        self.neighbor_provider = neighbor_provider
        self.distance_fn = distance_fn or get_tanimoto_distance
        self.distance_name = distance_name
        
        # Expensive metrics configuration
        self.expensive_metrics_to_compute = expensive_metrics_to_compute
        self.check_memorization_from_any_reference = check_memorization_from_any_reference
        
        # ---- Unified reference_smiles_files configuration (preferred) ----
        # We derive:
        # - canonical sets for membership metrics (novelty/memorization)
        # - raw SMILES pools for distribution comparison metrics (SNN/FCD/etc)
        # - dataset statistics for relevancy checks (max_ring_size, max_alkyl_chain)
        self._reference_smiles_by_name: Dict[str, List[str]] = {}
        self._reference_canon_sets_by_name: Dict[str, set] = {}
        self._reference_dataset_stats_by_name: Dict[str, Dict[str, int]] = {}
        self._default_reference_name: Optional[str] = None

        def _normalize_reference_smiles_files(
            spec,
        ) -> List[dict]:
            """
            Normalize reference_smiles_files into a list of dicts.
            Preferred input is list-of-dicts; mapping is supported for convenience.
            """
            if spec is None:
                return []
            # Hydra/OmegaConf may provide DictConfig/ListConfig; convert to plain containers first.
            try:
                from omegaconf import OmegaConf  # type: ignore
                if not isinstance(spec, (dict, list, tuple, str)):
                    spec = OmegaConf.to_container(spec, resolve=True)
            except Exception:
                pass
            # Sometimes CLI overrides arrive as a single string; try to parse as YAML.
            if isinstance(spec, str):
                try:
                    import yaml  # type: ignore
                    spec = yaml.safe_load(spec)
                except Exception:
                    # If parsing fails, treat as unset so legacy fallbacks can apply.
                    return []
            if isinstance(spec, dict):
                # mapping: {name: path_or_files}
                out = []
                for k, v in spec.items():
                    out.append({"name": str(k), "files": v})
                return out
            if isinstance(spec, (list, tuple)):
                return [x for x in spec if isinstance(x, dict)]
            return []

        def _extract_name_and_files(item) -> tuple[Optional[str], Optional[str | List[str]]]:
            # Item may be a DictConfig; attempt conversion.
            try:
                from omegaconf import OmegaConf  # type: ignore
                if not isinstance(item, dict):
                    item = OmegaConf.to_container(item, resolve=True)
            except Exception:
                pass
            if not isinstance(item, dict):
                return None, None
            name = item.get("name", None) or item.get("ref", None) or item.get("id", None)
            files = item.get("files", None) or item.get("path", None) or item.get("file", None)
            if not (isinstance(name, str) and name):
                return None, None
            if files is None:
                return None, None
            return name, files

        def _canonicalize_smiles_list(smiles_list: List[str]) -> List[str]:
            out: List[str] = []
            for s in smiles_list or []:
                if not isinstance(s, str) or not s:
                    continue
                try:
                    c = SmilesUtils.canonicalize(s)
                except Exception:
                    c = None
                if c:
                    out.append(c)
            return out

        ref_specs = _normalize_reference_smiles_files(reference_smiles_files)

        # Back-compat: if unified list not provided, fall back to the legacy knobs.
        # (Users should migrate to reference_smiles_files.)
        if not ref_specs:
            # Legacy training file becomes the default reference baseline.
            if training_smiles_file is not None:
                ref_specs.append({"name": "train", "files": training_smiles_file, "default": True})
            # Legacy reference file becomes another named reference for distribution metrics.
            if reference_smiles_file is not None:
                ref_specs.append({"name": "reference", "files": reference_smiles_file, "default": (not bool(ref_specs))})
            # Legacy named memorization refs (mapping or list-of-dicts)
            if memorization_reference_smiles_files is not None:
                # Accept mapping or list-of-dicts
                mem_specs = _normalize_reference_smiles_files(memorization_reference_smiles_files)  # type: ignore[arg-type]
                for ms in mem_specs:
                    n, f = _extract_name_and_files(ms)
                    if n and f is not None:
                        ref_specs.append({"name": n, "files": f, "default": False})

        # Load reference sets from files if specified
        if ref_specs:
            for item in ref_specs:
                name, files = _extract_name_and_files(item)
                if name is None or files is None:
                    continue

                # Determine default reference name
                if self._default_reference_name is None and bool(item.get("default", False)):
                    self._default_reference_name = name

                # Load raw SMILES once, then canonicalize for membership metrics
                raw_list = self._load_smiles_from_files(
                    files, smiles_column, "split", training_split_value, canonicalize=False
                )
                canon_list = _canonicalize_smiles_list(raw_list)
                self._reference_smiles_by_name[name] = list(raw_list)
                self._reference_canon_sets_by_name[name] = set(canon_list)
                file_desc = f"{len(files)} files" if isinstance(files, list) else str(files)
                print(
                    f"Loaded reference '{name}': {len(raw_list)} raw SMILES + {len(self._reference_canon_sets_by_name[name])} canonical SMILES from {file_desc}"
                )
                
        # Back-compat: if canonical/raw references were provided directly (small datasets),
        # use them only if we still have no reference specs.
        if not self._reference_smiles_by_name and not self._reference_canon_sets_by_name:
            if training_smiles_canonical is not None:
                self._reference_canon_sets_by_name["train"] = set(training_smiles_canonical)
                self._default_reference_name = "train"
            if reference_smiles_for_unconditional is not None:
                self._reference_smiles_by_name["reference"] = list(reference_smiles_for_unconditional)
                if self._default_reference_name is None:
                    self._default_reference_name = "reference"

        # Ensure we have a default reference name if any references exist
        if self._default_reference_name is None:
            if self._reference_smiles_by_name:
                self._default_reference_name = next(iter(self._reference_smiles_by_name.keys()))
            elif self._reference_canon_sets_by_name:
                self._default_reference_name = next(iter(self._reference_canon_sets_by_name.keys()))

        # Backward-compatible aliases:
        # - "training canon set" becomes the default reference's canonical set
        # - "reference smiles" becomes the default reference's raw pool
        self._training_canon_set = (
            set(self._reference_canon_sets_by_name.get(self._default_reference_name, set()))
            if self._default_reference_name is not None
            else None
        )
        self._reference_smiles = (
            list(self._reference_smiles_by_name.get(self._default_reference_name, []))
            if self._default_reference_name is not None
            else None
        )
        
        # ---- modular property specs (NEW) ----
        def _safe_sa(m):
            try:
                return float(sascorer.calculateScore(m))
            except Exception:
                return None
        
        # Handle property configuration (can use property_specs directly or property_names for convenience)
        if property_specs is not None and property_names is not None:
            raise ValueError("Cannot specify both property_specs and property_names")
        
        if property_names is not None:
            # Use property_registry for simple configuration
            self.property_specs: List[PropertySpec] = get_property_specs(property_names)
        elif property_specs is not None:
            # Use provided specs directly
            self.property_specs: List[PropertySpec] = property_specs
        else:
            # Default: LogP, QED, SA (backward compatibility)
            self.property_specs: List[PropertySpec] = get_property_specs(DEFAULT_PROPERTIES)
        
        # Override direction for utility_key if both utility_key and utility_direction are provided
        if utility_key is not None and utility_direction is not None:
            if utility_direction not in ["increasing", "decreasing"]:
                raise ValueError(f"utility_direction must be 'increasing' or 'decreasing', got '{utility_direction}'")
            spec_direction = "higher" if utility_direction == "increasing" else "lower"
            
            updated_specs = []
            found_utility_key = False
            for spec in self.property_specs:
                if spec.name == utility_key:
                    updated_specs.append(PropertySpec(
                        spec.name,
                        spec.fn,
                        direction=spec_direction,
                        min_delta=spec.min_delta
                    ))
                    found_utility_key = True
                    print(f"Overriding direction for property '{utility_key}' to '{spec_direction}' (from utility_direction='{utility_direction}')")
                else:
                    updated_specs.append(spec)
            
            if found_utility_key:
                self.property_specs = updated_specs
            else:
                print(f"Warning: utility_key '{utility_key}' not found in property_specs. Available properties: {[s.name for s in self.property_specs]}")
        
        # FCD settings
        self._fcd_min_required = int(fcd_min_required)
        
        # Default dataset stats (used as fallback if no reference files provided or stats computation fails)
        self.dataset_stats = dataset_stats if dataset_stats is not None else {'max_ring_size': 8, 'max_alkyl_chain': 5}

        # pairit_eval CSV export settings
        self.export_pairit_csv = export_pairit_csv
        self.export_prompt_metadata = bool(export_prompt_metadata)
        self.pairit_generation_seed = pairit_generation_seed
        if self.export_prompt_metadata:
            if not save_generated_samples or not export_pairit_csv:
                raise ValueError("Prompt metadata export requires sample saving and export_pairit_csv")
            if num_conditional_metric_samples is not None:
                raise ValueError("Prompt export requires processing every requested attempt")
        if pairit_generation_seed is not None:
            from pairit.prompting.metadata import generation_pair_seed
            generation_pair_seed(pairit_generation_seed, "")  # validate the explicit seed
            if not self.export_prompt_metadata:
                raise ValueError("Pair-local generation requires immutable prompt metadata")
        self.pairit_method_name = pairit_method_name
        self.pairit_dataset_name = pairit_dataset_name
        self.pairit_seed_label_column = pairit_seed_label_column

        # Generation-only mode: skip expensive metric computation
        self.generation_only = generation_only

        # Initialize storage
        if self.save_generated_samples:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._generated_samples = []
            if self.export_prompt_metadata:
                print(f"Saving every prompted attempt to {save_dir}; visualization limit: {max_samples_to_save}")
            else:
                print(f"Molecule saving enabled: will save up to {max_samples_to_save} samples ({self.save_fraction*100:.1f}%) to {save_dir}")
        
        # ===== Initialize modular components =====
        self.printer = ValidationPrinter(verbose=True)
        self.generator = MoleculeGenerator()
        self.timing_tracker = TimingTracker()
        
        # Distance percentile function (will be set lazily if dataset is available)
        self._distance_percentile_fn: Optional[Callable[[str, float], Optional[float]]] = None
        
        self.conditional_metrics = ConditionalMetricsComputer(
            property_specs=self.property_specs,
            distance_name=self.distance_name,
            reference_canon_sets_by_name=self._reference_canon_sets_by_name,
            default_reference_name=self._default_reference_name,
            reference_smiles_by_name=self._reference_smiles_by_name,
            distance_fn=self.distance_fn,
            neighbor_provider=self.neighbor_provider,
            expensive_metrics_to_compute=self.expensive_metrics_to_compute,
            timing_tracker=self.timing_tracker,
            distance_percentile_fn=None,  # Will be set lazily
            check_memorization_from_any_reference=self.check_memorization_from_any_reference,
            fcd_min_required=self._fcd_min_required,
            metric_prefix="val",
        )

        self.unconditional_metrics = UnconditionalMetricsComputer(
            reference_smiles=self._reference_smiles,
            reference_smiles_by_name=self._reference_smiles_by_name,
            default_reference_name=self._default_reference_name,
            fcd_min_required=self._fcd_min_required,
            expensive_metrics_to_compute=self.expensive_metrics_to_compute,
            training_canon_set=self._training_canon_set,
            reference_canon_sets_by_name=self._reference_canon_sets_by_name,
            timing_tracker=self.timing_tracker,
            property_specs=self.property_specs,
            check_memorization_from_any_reference=self.check_memorization_from_any_reference,
            metric_prefix="val",
        )

    def _split_enabled(self, trainer: Optional[pl.Trainer] = None) -> bool:
        mode = (self.split_metrics_by_dataset or "auto").lower()
        if mode not in {"auto", "on", "off"}:
            raise ValueError("split_metrics_by_dataset must be one of {'auto','on','off'}")
        if mode == "on":
            return True
        if mode == "off":
            return False
        # auto: enable only when multiple val dataloaders exist
        if trainer is None:
            return False
        n = getattr(trainer, "num_val_batches", None)
        if isinstance(n, (list, tuple)):
            return len(n) > 1
        return False

    @staticmethod
    def _dataset_key_from_batch(batch: dict) -> str:
        name = batch.get("dataset_name", None)
        if isinstance(name, str) and name:
            return name
        ds_id = batch.get("dataset_id", None)
        try:
            if hasattr(ds_id, "reshape"):
                # torch tensor: take first element
                ds_id = int(ds_id.reshape(-1)[0].item())
            elif isinstance(ds_id, (list, tuple)) and ds_id:
                ds_id = int(ds_id[0])
            elif ds_id is not None:
                ds_id = int(ds_id)
        except Exception:
            ds_id = None
        return f"ds_{ds_id}" if ds_id is not None else "unknown_dataset"

    def _metric_name(self, base: str, dataset_key: Optional[str], trainer: Optional[pl.Trainer] = None) -> str:
        """
        Build a metric name that preserves single-dataset behavior by default.
        """
        if dataset_key is not None and self._split_enabled(trainer):
            # Avoid double "val_" prefix when nesting under val/<dataset_key>/...
            metric_leaf = base[4:] if base.startswith("val_") else base
            return f"val/{dataset_key}/{metric_leaf}"
        return base

    def _binning_for(self, prop_name: str, kind: str):
        if kind == "value":
            return self._value_binning_per_prop.get(prop_name, self._value_binning_default)
        return self._delta_binning_per_prop.get(prop_name, self._delta_binning_default)

    def _prompt_ranges_from_parsed(self, parsed):
        """
        Convert parsed prompt bins into numeric ranges when ranges are known.
        """
        from pairit.prompting.property_prompting import bin_to_range as _bin_to_range

        out = []
        for p in parsed:
            start_rng = None
            delta_rng = None
            if p.start_bin is not None:
                vb = self._binning_for(p.property_name, "value")
                if vb is not None and vb.range is not None:
                    start_rng = _bin_to_range(binning=vb, bin_idx=int(p.start_bin))
            if p.delta_bin is not None:
                db = self._binning_for(p.property_name, "delta")
                if db is not None and db.range is not None:
                    delta_rng = _bin_to_range(binning=db, bin_idx=int(p.delta_bin))
            out.append((p, start_rng, delta_rng))
        return out

    def _get_distance_percentile_fn(self, trainer: Optional[pl.Trainer] = None) -> Optional[Callable[[str, float], Optional[float]]]:
        """
        Create a function to compute distance percentiles from the dataset if available.
        
        Returns None if dataset is not available or doesn't support distance percentiles.
        """
        if self._distance_percentile_fn is not None:
            return self._distance_percentile_fn
        
        # Try to get dataset from trainer's datamodule
        if trainer is None or not hasattr(trainer, 'datamodule') or trainer.datamodule is None:
            return None
        
        try:
            # Check if datamodule has a val_dataset with distance distribution support
            datamodule = trainer.datamodule
            if not hasattr(datamodule, '_val_dataset') or datamodule._val_dataset is None:
                return None
            
            dataset = datamodule._val_dataset
            # Check if dataset has the required methods
            if not (hasattr(dataset, '_get_molecule_idx') and hasattr(dataset, '_get_distance_percentile')):
                return None
            
            # Create closure that captures the dataset
            def compute_percentile(seed_smiles: str, distance: float) -> Optional[float]:
                """Compute distance percentile for given seed SMILES and distance."""
                try:
                    molecule_a_idx = dataset._get_molecule_idx(seed_smiles)
                    if molecule_a_idx is None:
                        return None
                    return dataset._get_distance_percentile(molecule_a_idx, distance)
                except Exception:
                    return None
            
            self._distance_percentile_fn = compute_percentile
            return compute_percentile
        except Exception:
            # If anything fails, just return None (percentiles won't be computed)
            return None

    def _get_conditional_metrics_for_key(self, dataset_key: str, trainer: Optional[pl.Trainer] = None):
        if dataset_key in self._cond_metrics_by_key:
            cm = self._cond_metrics_by_key[dataset_key]
            # Update distance_percentile_fn if not set
            if cm.distance_percentile_fn is None:
                percentile_fn = self._get_distance_percentile_fn(trainer)
                if percentile_fn is not None:
                    cm.distance_percentile_fn = percentile_fn
            return cm
        # Try to get distance percentile function if available
        percentile_fn = self._get_distance_percentile_fn(trainer) if trainer is not None else self._distance_percentile_fn
        cm = ConditionalMetricsComputer(
            property_specs=self.property_specs,
            distance_name=self.distance_name,
            reference_canon_sets_by_name=self._reference_canon_sets_by_name,
            default_reference_name=self._default_reference_name,
            reference_smiles_by_name=self._reference_smiles_by_name,
            distance_fn=self.distance_fn,
            neighbor_provider=self.neighbor_provider,
            expensive_metrics_to_compute=self.expensive_metrics_to_compute,
            timing_tracker=self.timing_tracker,
            distance_percentile_fn=percentile_fn,
            check_memorization_from_any_reference=self.check_memorization_from_any_reference,
            fcd_min_required=self._fcd_min_required,
            metric_prefix="val",
        )
        self._cond_metrics_by_key[dataset_key] = cm
        return cm

    def _get_unconditional_metrics_for_key(self, dataset_key: str):
        if dataset_key in self._uncond_metrics_by_key:
            return self._uncond_metrics_by_key[dataset_key]
        um = UnconditionalMetricsComputer(
            reference_smiles=self._reference_smiles,
            reference_smiles_by_name=self._reference_smiles_by_name,
            default_reference_name=self._default_reference_name,
            fcd_min_required=self._fcd_min_required,
            expensive_metrics_to_compute=self.expensive_metrics_to_compute,
            training_canon_set=self._training_canon_set,
            reference_canon_sets_by_name=self._reference_canon_sets_by_name,
            timing_tracker=self.timing_tracker,
            property_specs=self.property_specs,
            check_memorization_from_any_reference=self.check_memorization_from_any_reference,
            metric_prefix="val",
        )
        self._uncond_metrics_by_key[dataset_key] = um
        return um
    
    @staticmethod
    def _load_smiles_from_files(
        file_paths: str | List[str],
        smiles_column: str = "smiles",
        split_column: Optional[str] = "split",
        split_value: str = "train",
        canonicalize: bool = False
    ) -> List[str]:
        """
        Load SMILES strings from one or more parquet files.
        
        Parameters
        ----------
        file_paths : str | List[str]
            Path to a single parquet file or list of paths to multiple parquet files
        smiles_column : str
            Name of column containing SMILES strings
        split_column : Optional[str]
            Name of column containing split information (e.g., "train", "val", "test")
            If None, loads all SMILES from file
        split_value : str
            Value in split_column to filter for (e.g., "train")
        canonicalize : bool
            Whether to canonicalize SMILES using RDKit
            
        Returns
        -------
        List[str]
            Combined list of SMILES strings from all files
        """
        # Normalize to list for uniform processing
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        all_smiles = []
        for file_path in file_paths:
            try:
                if str(file_path).endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"[Warning] Failed to load SMILES from {file_path}: {e}")
                continue
            
            if smiles_column not in df.columns:
                print(f"[Warning] Column '{smiles_column}' not found in {file_path}. Available columns: {df.columns.tolist()}")
                continue
            
            # Filter by split if split_column is provided
            if split_column is not None:
                if split_column not in df.columns:
                    print(f"[Warning] Split column '{split_column}' not found in {file_path}. Loading all SMILES.")
                    smiles_list = df[smiles_column].tolist()
                else:
                    smiles_list = df[df[split_column] == split_value][smiles_column].tolist()
            else:
                smiles_list = df[smiles_column].tolist()
            
            # Canonicalize if requested
            if canonicalize:
                canonical_smiles = []
                for smi in smiles_list:
                    if smi is None or not isinstance(smi, str):
                        continue
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                    except Exception:
                        continue
                smiles_list = canonical_smiles
            else:
                # Filter out None values
                smiles_list = [s for s in smiles_list if s is not None and isinstance(s, str)]
            
            all_smiles.extend(smiles_list)
            print(f"  Loaded {len(smiles_list)} SMILES from {file_path}")
        
        return all_smiles
    
    def _should_print(self) -> bool:
        """
        Check if we should print debug information to avoid log pollution.
        Only prints the first _max_prints times.
        """
        if self._print_count < self._max_prints:
            self._print_count += 1
            return True
        return False
    
    # NEW: Property computation helpers
    def _safe_valid(self, s: str) -> bool:
        """Check if SMILES is valid using SmilesUtils."""
        return SmilesUtils.is_valid(s, use_sanity_check=True)

    def _neighbors_for(self, conditioned_smiles: str, neighbors_from_batch: Optional[List[str]]) -> List[str]:
        """Get neighbors for a conditioned SMILES string."""
        if neighbors_from_batch and len(neighbors_from_batch) > 0:
            return neighbors_from_batch
        if self.neighbor_provider is not None and conditioned_smiles:
            try:
                return self.neighbor_provider(conditioned_smiles) or []
            except Exception as e:
                # Neighbor provider failures are unexpected and may indicate configuration issues
                print(f"[Warning] Neighbor provider failed for SMILES '{conditioned_smiles[:50]}...': {e}")
                return []
        return []

    def _log_properties(self, pl_module, props: Dict[str, Optional[float]], context: str):
        """Log properties with DDP compatibility."""
        # Skip if no trainer attached (e.g., during standalone evaluation)
        if not self._has_trainer(pl_module):
            return
        
        for k, v in props.items():
            if v is None:
                continue
            # Move to GPU before logging for sync_dist
            prop_tensor = torch.tensor(float(v), device=pl_module.device)
            pl_module.log(f"val_{k}_{context}", prop_tensor, sync_dist=True, on_step=False, on_epoch=True)

    def _should_save_sample(self, sample_id: str) -> bool:
        """
        Deterministically decide whether to save this sample based on hash.
        
        Uses hash of sample_id to ensure deterministic selection that doesn't
        vary between runs with the same parameters.
        
        Args:
            sample_id: Unique identifier for the sample
            
        Returns:
            True if this sample should be saved
            
        Examples:
            save_fraction=0.5  -> saves ~50% of samples
            save_fraction=0.1  -> saves ~10% of samples
            save_fraction=1.0  -> saves ~100% of samples
        """
        if not self.save_generated_samples:
            return False
        
        # Use hash for deterministic selection
        hash_value = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
        
        # Calculate threshold based on save_fraction
        # save_fraction=0.5 means 50% of hash space should be below threshold
        threshold = self.save_fraction * MoleculeValidationCallback.HASH_SPACE_SIZE
        
        return (hash_value % MoleculeValidationCallback.HASH_SPACE_SIZE) < threshold
    
    def _save_sample(self, sample_data: dict, step_type: str, step_id: int):
        """
        Save a generated sample to the buffer.
        
        Args:
            sample_data: Dictionary containing sample information
            step_type: Type of generation ('conditional' or 'unconditional')
            step_id: Step/batch/epoch identifier
        """
        if not self.save_generated_samples:
            return
        
        # Add metadata
        sample_data.update({
            'step_type': step_type,
            'step_id': step_id,
            'timestamp': time.time()
        })
        
        # Add to buffer for later flushing
        self._generated_samples.append(sample_data)

    @staticmethod
    def _has_trainer(pl_module) -> bool:
        """
        Safely check if pl_module has an attached trainer.
        
        The trainer property raises RuntimeError if not attached, so we need
        to check the private _trainer attribute or use try-except.
        """
        try:
            # Check if _trainer attribute exists and is not None
            return hasattr(pl_module, '_trainer') and pl_module._trainer is not None
        except Exception:
            return False
    
    @staticmethod
    def _find_callback_instance(pl_module):
        """
        Find MoleculeValidationCallback instance from trainer callbacks.
        
        This helper method extracts the repeated pattern of finding the callback
        instance from the trainer's callback list.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module with trainer attribute
            
        Returns
        -------
        MoleculeValidationCallback or None
            The callback instance if found, None otherwise
        """
        try:
            for c in getattr(pl_module.trainer, "callbacks", []) or []:
                if isinstance(c, MoleculeValidationCallback):
                    return c
        except Exception:
            pass
        return None
    
    @staticmethod
    def _mean_slice(arr: List, start: int) -> Optional[float]:
        """
        Compute mean of array slice starting from index.
        
        Parameters
        ----------
        arr : List
            List of values (may contain None)
        start : int
            Starting index for the slice
            
        Returns
        -------
        Optional[float]
            Mean of non-None values in slice, or None if slice is empty
        """
        sl = arr[start:]
        sl = [float(x) for x in sl if x is not None]
        return (sum(sl) / len(sl)) if sl else None
    
    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> float:
        """
        Safely divide, returning 0.0 if denominator is 0.
        
        Parameters
        ----------
        numerator : int
            Numerator value
        denominator : int
            Denominator value
            
        Returns
        -------
        float
            Division result, or 0.0 if denominator is 0
        """
        return float(numerator) / denominator if denominator > 0 else 0.0

    @staticmethod
    def _sanitize_ref_name(name: str) -> str:
        """
        Sanitize a user-provided reference nickname for use in metric suffixes / parquet column names.
        """
        if not isinstance(name, str):
            return "ref"
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        return cleaned if cleaned else "ref"

    def _with_metric_prefix(self, name: str) -> str:
        if getattr(self, "metric_prefix", None):
            return f"{self.metric_prefix}/{name}"
        return name

    def _metric_name_is_dataset_namespaced(self, name: str) -> bool:
        """
        Return True if `name` already encodes which validation dataloader/dataset it came from.

        In mixed validation (multiple val dataloaders), Lightning will otherwise append
        a `/dataloader_idx_*` suffix. When we already namespace metrics like `val/<dataset_key>/...`,
        that suffix is redundant and can push metric names over length limits.
        """
        # Known dataset keys (populated as we encounter datasets during validation).
        keys = set()
        keys.update(getattr(self, "_cond_metrics_by_key", {}).keys())
        keys.update(getattr(self, "_uncond_metrics_by_key", {}).keys())
        keys.update(getattr(self, "_conditional_generated_this_epoch_by_key", {}).keys())
        if not keys:
            return False
        for k in keys:
            # Match either beginning of name or nested under some metric_prefix.
            if name.startswith(f"val/{k}/") or f"/val/{k}/" in name:
                return True
        return False
    
    def _log_step(self, pl_module, name: str, value: float) -> None:
        """Helper to log a scalar in a consistent, DDP-safe way (epoch-only)."""
        try:
            # Avoid Lightning warning when no Trainer is attached (e.g., during standalone evaluate())
            if not self._has_trainer(pl_module):
                return
            tensor_val = torch.tensor(float(value), device=pl_module.device)
            full_name = self._with_metric_prefix(name)
            pl_module.log(
                full_name,
                tensor_val,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
                # If metric name already includes dataset_key (e.g. val/zinc/...), avoid redundant suffix.
                add_dataloader_idx=(not self._metric_name_is_dataset_namespaced(full_name)),
            )
        except Exception as e:
            # Logging failures should not crash training, but we should be aware of them
            print(f"[Warning] Failed to log metric '{name}': {e}")
    
    def _export_pairit_csv(self, df: pd.DataFrame, parquet_path: Path) -> None:
        """Export a pairit_eval-compatible CSV alongside a parquet file.

        Maps internal column names to the pairit_eval schema and writes
        a ``_pairit.csv`` file next to the given *parquet_path*.
        """
        if df is None or len(df) == 0:
            return

        csv_path = Path(str(parquet_path).removesuffix(".parquet") + "_pairit.csv")

        col_map = {
            "smiles_conditioning": "seed_smiles",
            "smiles_generated": "generated_smiles",
            "seed_index": "seed_idx",
            "generation_index": "gen_idx",
        }
        out = pd.DataFrame()
        for src, dst in col_map.items():
            if src in df.columns:
                out[dst] = df[src]
            else:
                out[dst] = 0

        # seed_label: use the property column if configured, else 0
        if self.pairit_seed_label_column and f"property_cond_{self.pairit_seed_label_column}" in df.columns:
            out["seed_label"] = df[f"property_cond_{self.pairit_seed_label_column}"]
        else:
            out["seed_label"] = 0

        out["method"] = self.pairit_method_name
        out["dataset"] = self.pairit_dataset_name or ""

        if self.export_prompt_metadata:
            import json
            required = {
                "pair_id", "seed_smiles", "target_smiles", "generation_index", "smiles_generated",
                "prompt_target_direction_property_names", "prompt_target_raw_direction_signs",
                "prompt_target_raw_deltas", "prompt_target_raw_seed_values", "prompt_target_raw_target_values",
            }
            if not required.issubset(df.columns):
                raise ValueError(f"Missing prompt export columns: {sorted(required - set(df.columns))}")
            if set(df.pair_id) != set(self._prompt_requests):
                raise ValueError("Generation export does not cover all requested pairs")
            for pair_id, group in df.groupby("pair_id", sort=False):
                if sorted(group.generation_index.tolist()) != list(range(self.num_generations_per_seed)):
                    raise ValueError(f"Incomplete or duplicated generation slots for pair {pair_id}")
            out["seed_smiles"] = df["seed_smiles"]
            out["seed_idx"] = df["pair_id"]
            for column in df.columns:
                if column in ("pair_id", "target_smiles", "generation_pair_seed", "evaluation_generation_seed") or column.startswith("prompt_"):
                    out[column] = df[column].map(
                        lambda value: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                    )
            requests = pd.DataFrame(list(self._prompt_requests.values()))
            for column in requests.columns:
                requests[column] = requests[column].map(
                    lambda value: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                )
            requests.to_csv(csv_path.with_name(csv_path.stem + "_requests.csv"), index=False)

        # Reorder columns to match pairit_eval expectation
        ordered_cols = ["method", "dataset", "seed_idx", "seed_smiles", "seed_label", "gen_idx", "generated_smiles"]
        extra_cols = [c for c in out.columns if c not in ordered_cols]
        out = out[[c for c in ordered_cols if c in out.columns] + extra_cols]

        out.to_csv(csv_path, index=False)
        print(f"Exported pairit_eval CSV ({len(out)} rows) to {csv_path}")

    def _prompt_batch_metadata(self, batch, count, dataset_key=None):
        """Record requests before generation; keep original strings out of decoding."""
        if not self.export_prompt_metadata:
            return [None] * count
        import json
        from pairit.prompting.metadata import generation_pair_seed, pair_identifier

        if "prompt_metadata" not in batch or len(batch["prompt_metadata"]) < count:
            raise ValueError("Enable prompting.properties.export_metadata on the input dataset")
        namespace = str(dataset_key or self.pairit_dataset_name or "")
        if not namespace:
            raise ValueError("Prompt export requires an explicit dataset namespace")
        if self._prompt_requests and next(iter(self._prompt_requests.values()))["dataset"] != namespace:
            raise ValueError("Prompt export requires one dataset per callback")
        rows = []
        for encoded in batch["prompt_metadata"][:count]:
            row = json.loads(encoded)
            row["dataset"] = namespace
            row["pair_id"] = pair_identifier(
                namespace, row["seed_smiles"], row["target_smiles"],
            )
            if row["pair_id"] in self._prompt_requests:
                raise ValueError(f"Repeated input pair {row['pair_id']}; use one request per ordered pair")
            if self.pairit_generation_seed is not None:
                row["evaluation_generation_seed"] = self.pairit_generation_seed
                row["generation_pair_seed"] = generation_pair_seed(self.pairit_generation_seed, row["pair_id"])
            self._prompt_requests[row["pair_id"]] = row
            rows.append(row)
        return rows

    def _flush_samples(self, step_type: str, step_id: int, pl_module: pl.LightningModule, trainer: pl.Trainer = None):
        """
        Save collected samples to disk and clear the buffer. Use for both conditional and unconditional generations.
        
        Parameters
        ----------
        step_type : str
            Type of generation step ('conditional' or 'unconditional')
        step_id : int
            Identifier for the step (batch_idx or epoch number)
        """
        if not self._generated_samples:
            return
        
        # Always clear the buffer on all ranks
        samples_to_save = self._generated_samples.copy()
        self._generated_samples = []
        
        # Only rank 0 saves files and creates visualizations
        if rank_zero_only.rank == 0:
            df = pd.DataFrame(samples_to_save)
            def _sanitize_dataset_part(s: str) -> str:
                s = (s or "").strip()
                s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
                return s

            # Save parquet with dataset identity in filename when available.
            # If multiple dataset keys are present in the buffer, write one parquet per dataset key.
            if "dataset_key" in df.columns:
                # Prefer using real dataset keys; do not emit "unknown_dataset" in filenames.
                ds_raw = df["dataset_key"]
                try:
                    ds_series = ds_raw.astype(object)
                except Exception:
                    ds_series = ds_raw

                # Sanitize per-row so matching is stable even if dataset names contain spaces.
                ds_sanitized = ds_series.fillna("").astype(str).map(_sanitize_dataset_part)
                ds_keys = [k for k in ds_sanitized.unique().tolist() if k]

                if len(ds_keys) > 1:
                    # Write one file per dataset key; any rows without a dataset key go to the base filename.
                    for ds_key in ds_keys:
                        mask = ds_sanitized == ds_key
                        df_k = df.loc[mask]
                        if len(df_k) == 0:
                            continue
                        filename = self.save_dir / f"{step_type}_{ds_key}_step_{step_id}_generated_molecules.parquet"
                        df_k.to_parquet(filename, index=False)
                        print(f"Saved {len(df_k)} generated molecules to {filename}")
                    # Save leftover (no dataset key) rows, if any
                    leftover = df.loc[ds_sanitized == ""]
                    if len(leftover) > 0:
                        filename = self.save_dir / f"{step_type}_step_{step_id}_generated_molecules.parquet"
                        leftover.to_parquet(filename, index=False)
                        print(f"Saved {len(leftover)} generated molecules to {filename}")
                elif len(ds_keys) == 1:
                    dataset_part = ds_keys[0]
                    filename = self.save_dir / f"{step_type}_{dataset_part}_step_{step_id}_generated_molecules.parquet"
                    df.to_parquet(filename, index=False)
                    print(f"Saved {len(df)} generated molecules to {filename}")
                else:
                    filename = self.save_dir / f"{step_type}_step_{step_id}_generated_molecules.parquet"
                    df.to_parquet(filename, index=False)
                    print(f"Saved {len(df)} generated molecules to {filename}")
            else:
                filename = self.save_dir / f"{step_type}_step_{step_id}_generated_molecules.parquet"
                df.to_parquet(filename, index=False)
                print(f"Saved {len(df)} generated molecules to {filename}")
            
            # Export pairit_eval CSV for conditional generations
            if self.export_pairit_csv and step_type == "conditional":
                self._export_pairit_csv(df, filename)

            if self.max_samples_to_save <= 0:
                return

            # Try to create a visualization grid of the first max_samples_to_save samples
            try:
                # In mixed validation, conditional samples are per-dataset. Log one grid per dataset name.
                viz_groups: List[Tuple[Optional[str], pd.DataFrame]] = [(None, df)]
                if step_type == "conditional" and "dataset_key" in df.columns:
                    try:
                        ds_series = df["dataset_key"].fillna("").astype(str)
                        ds_names = [s for s in ds_series.unique().tolist() if str(s).strip() != ""]
                    except Exception:
                        ds_names = []
                    if ds_names:
                        viz_groups = [(ds_name, df.loc[df["dataset_key"].astype(str) == str(ds_name)]) for ds_name in ds_names]

                for dataset_name, df_viz in viz_groups:
                    # Use per-dataset dataframe for visualization
                    df = df_viz
                    max_viz_samples = min(self.max_samples_to_save, len(df))
                    if max_viz_samples <= 0:
                        continue

                    # Stable component for W&B keys (but keep the human name in caption).
                    media_ds_key = None
                    if dataset_name is not None and str(dataset_name).strip() != "":
                        media_ds_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name).strip())
                
                    # Get SMILES strings for visualization based on step type
                    if step_type == 'conditional':
                        smiles_list1 = df['smiles_conditioning'].head(max_viz_samples).tolist()
                        smiles_list2 = df['smiles_generated'].head(max_viz_samples).tolist()
                    
                        # Extract property deltas
                        prop_delta_col = f'property_delta_{self.visualization_property_name}'
                        property_deltas = df[prop_delta_col].head(max_viz_samples).tolist() if prop_delta_col in df.columns else None
                        if property_deltas is not None:
                            num_valid_deltas = len([d for d in property_deltas if d is not None])
                            print(f"[Debug] Conditional: Extracted {num_valid_deltas}/{len(property_deltas)} property deltas for {self.visualization_property_name}")
                    
                        # Extract memorization flags (check if generated SMILES in training set)
                        memorization_flags = None
                        if "memorization" in self.expensive_metrics_to_compute and self._training_canon_set is not None:
                            memorization_flags = []
                            for gen_smiles in smiles_list2:
                                if gen_smiles:
                                    canon = SmilesUtils.canonicalize(gen_smiles)
                                    memorization_flags.append(canon in self._training_canon_set if canon else False)
                                else:
                                    memorization_flags.append(False)
                            num_memorized = sum(memorization_flags)
                            print(f"[Debug] Conditional: Found {num_memorized}/{len(memorization_flags)} memorized molecules")
                    
                        # Extract GT candidates (use smiles_input_2 which is the second molecule from the validation pair)
                        gt_candidates = df['smiles_input_2'].head(max_viz_samples).tolist() if 'smiles_input_2' in df.columns else None
                        if gt_candidates is not None:
                            num_valid_gt = len([gt for gt in gt_candidates if gt is not None])
                            print(f"[Debug] Conditional: Extracted {num_valid_gt}/{len(gt_candidates)} GT candidates from validation pairs")
                    else:  # unconditional
                        smiles_list1 = df['smiles_generated'].head(max_viz_samples).tolist()
                        smiles_list2 = df['smiles_second'].head(max_viz_samples).tolist()
                    
                        # Extract property deltas
                        prop_delta_col = f'property_delta_{self.visualization_property_name}'
                        property_deltas = df[prop_delta_col].head(max_viz_samples).tolist() if prop_delta_col in df.columns else None
                        if property_deltas is not None:
                            num_valid_deltas = len([d for d in property_deltas if d is not None])
                            print(f"[Debug] Unconditional: Extracted {num_valid_deltas}/{len(property_deltas)} property deltas for {self.visualization_property_name}")
                    
                        # Extract memorization flags
                        memorization_flags_list1 = None
                        memorization_flags_list2 = None
                        if "memorization" in self.expensive_metrics_to_compute:
                            if 'is_first_memorized' in df.columns:
                                memorization_flags_list1 = df['is_first_memorized'].head(max_viz_samples).tolist()
                                num_first_mem = sum(memorization_flags_list1) if memorization_flags_list1 else 0
                                print(f"[Debug] Unconditional: Found {num_first_mem}/{len(memorization_flags_list1)} memorized first molecules")
                            if 'is_second_memorized' in df.columns:
                                memorization_flags_list2 = df['is_second_memorized'].head(max_viz_samples).tolist()
                                num_second_mem = sum(memorization_flags_list2) if memorization_flags_list2 else 0
                                print(f"[Debug] Unconditional: Found {num_second_mem}/{len(memorization_flags_list2)} memorized second molecules")
                
                    # Generate and save the grid visualizations: PDF (matplotlib) and PNG (RDKit)
                    # Use trainer.global_step if available, otherwise fall back to pl_module.global_step
                    global_step = trainer.global_step if trainer is not None else pl_module.global_step
                
                    # Generate and save PDF only if save_pdfs is enabled
                    if self.save_pdfs:
                        grid_pdf = f"{step_type}_step_{global_step}_molecule_grid.pdf"
                    
                        if step_type == 'conditional':
                            print(f"[Debug] Calling generate_paired_smiles_grid with property_deltas={property_deltas is not None}, memorization_flags_list2={memorization_flags is not None}, property_name={self.visualization_property_name}")
                            generate_paired_smiles_grid(
                                smiles_list1, smiles_list2,
                                n_cols=min(10, len(smiles_list1)),
                                save_dir=self.save_dir,
                                save_name=grid_pdf,
                                use_rdkit=False,
                                property_deltas=property_deltas,
                                memorization_flags_list1=None,  # Conditioning not checked
                                memorization_flags_list2=memorization_flags,
                                property_name=self.visualization_property_name,
                            )
                        else:  # unconditional
                            generate_paired_smiles_grid(
                                smiles_list1, smiles_list2,
                                n_cols=min(10, len(smiles_list1)),
                                save_dir=self.save_dir,
                                save_name=grid_pdf,
                                use_rdkit=False,
                                property_deltas=property_deltas,
                                memorization_flags_list1=memorization_flags_list1,
                                memorization_flags_list2=memorization_flags_list2,
                                property_name=self.visualization_property_name,
                            )
                        print(f"Saved molecule grid visualization (PDF) to {grid_pdf}")

                # Generate PNGs in memory for wandb logging (saves disk space)
                grid_png_img = None
                if step_type == 'conditional':
                    grid_png_img = generate_paired_smiles_grid(
                        smiles_list1, smiles_list2,
                        n_cols=min(10, len(smiles_list1)),
                        save_dir=None,  # Don't save to disk - generate in memory only
                        save_name=None,
                        use_rdkit=True,
                        property_deltas=property_deltas,
                        memorization_flags_list1=None,  # Conditioning not checked
                        memorization_flags_list2=memorization_flags,
                        property_name=self.visualization_property_name,
                    )
                else:  # unconditional
                    grid_png_img = generate_paired_smiles_grid(
                        smiles_list1, smiles_list2,
                        n_cols=min(10, len(smiles_list1)),
                        save_dir=None,  # Don't save to disk - generate in memory only
                        save_name=None,
                        use_rdkit=True,
                        property_deltas=property_deltas,
                        memorization_flags_list1=memorization_flags_list1,
                        memorization_flags_list2=memorization_flags_list2,
                        property_name=self.visualization_property_name,
                    )
                print(f"Generated molecule grid visualization (PNG) in memory for wandb")


                
                
                # Generate ground truth grid for conditional generation only
                # Always generate it using smiles_input_2 (the second molecule from validation pairs)
                if step_type == 'conditional':
                    gt_grid_png_img = None
                    # Use smiles_input_2 which is the second molecule from the validation pair (ground truth)
                    gt_smiles_from_pairs = df['smiles_input_2'].head(max_viz_samples).tolist() if 'smiles_input_2' in df.columns else None
                    
                    if gt_smiles_from_pairs is not None and len(gt_smiles_from_pairs) > 0:
                        # Match length to smiles_list1, keeping None values (they'll be handled as empty strings in the grid)
                        gt_smiles_filtered = list(gt_smiles_from_pairs[:len(smiles_list1)])
                        # Pad with None if needed to match length
                        while len(gt_smiles_filtered) < len(smiles_list1):
                            gt_smiles_filtered.append(None)
                        
                        num_valid_gt = len([gt for gt in gt_smiles_filtered if gt is not None])
                        print(f"[Debug] Generating ground truth grid with {len(smiles_list1)} samples, {num_valid_gt} non-None GT molecules from validation pairs")
                        print(f"[Debug] Calling generate_ground_truth_grid with property_deltas={property_deltas is not None}, memorization_flags={memorization_flags is not None}, property_name={self.visualization_property_name}")
                        
                        # Generate and save ground truth PDF only if save_pdfs is enabled
                        if self.save_pdfs:
                            gt_grid_pdf = f"{step_type}_step_{global_step}_ground_truth_grid.pdf"
                            generate_ground_truth_grid(
                                conditioning_smiles=smiles_list1,
                                generated_smiles=smiles_list2,
                                gt_smiles=gt_smiles_filtered,
                                n_cols=min(10, len(smiles_list1)),
                                property_deltas=property_deltas,
                                memorization_flags=memorization_flags,
                                property_name=self.visualization_property_name,
                                save_dir=self.save_dir,
                                save_name=gt_grid_pdf,
                                use_rdkit=False,
                            )
                            print(f"Saved ground truth grid visualization (PDF) to {gt_grid_pdf}")
                        
                        # Generate ground truth grid PNG in memory for wandb
                        gt_grid_png_img = generate_ground_truth_grid(
                            conditioning_smiles=smiles_list1,
                            generated_smiles=smiles_list2,
                            gt_smiles=gt_smiles_filtered,
                            n_cols=min(10, len(smiles_list1)),
                            property_deltas=property_deltas,
                            memorization_flags=memorization_flags,
                            property_name=self.visualization_property_name,
                            save_dir=None,  # Don't save to disk - generate in memory only
                            save_name=None,
                            use_rdkit=True,
                        )
                        print(f"Generated ground truth grid visualization (PNG) in memory for wandb")
                        
                    if gt_grid_png_img is None:
                        print(f"[Info] Skipping ground truth grid: smiles_input_2 column not found or empty")
                    else:
                        # Log ground truth grid to wandb using in-memory PIL Image
                        try:
                            logger = getattr(pl_module, "logger", None)
                            if isinstance(logger, WandbLogger) and hasattr(logger, "experiment") and logger.experiment is not None:
                                caption = f"dataset={dataset_name}" if dataset_name else None
                                wandb_image_gt = wandb.Image(gt_grid_png_img, caption=caption)  # Pass PIL Image directly
                                key = f"Media/{step_type}_ground_truth_grid/{media_ds_key}" if media_ds_key else f"Media/{step_type}_ground_truth_grid"
                                logger.experiment.log({
                                    key: wandb_image_gt,
                                    "global_step": global_step
                                })
                                print(f"Logged ground truth grid visualization to wandb")
                        except Exception as e:
                            print(f"Failed to log ground truth grid to wandb: {str(e)}")

                # Log the grid visualization to wandb using in-memory PIL Image
                # Note: PNGs are generated in memory only (not saved to disk) to save storage
                try:
                    logger = getattr(pl_module, "logger", None)
                    
                    if isinstance(logger, WandbLogger) and hasattr(logger, "experiment") and logger.experiment is not None:
                        caption = f"dataset={dataset_name}" if dataset_name else None
                        wandb_image = wandb.Image(grid_png_img, caption=caption)  # Pass PIL Image directly
                        key = f"Media/{step_type}_molecule_grid/{media_ds_key}" if media_ds_key else f"Media/{step_type}_molecule_grid"
                        logger.experiment.log({
                            key: wandb_image,
                            "global_step": global_step
                        })
                        print(f"Logged molecule grid visualization to wandb")
                    else:
                        print("Skipping wandb log: no WandbLogger attached")
                except Exception as e:
                    print(f"Failed to log molecule grid to wandb: {str(e)}")
                
            except Exception as e:
                print(f"Failed to generate molecule grid visualization: {str(e)}")
            
            # Ensure any matplotlib figures are closed to prevent memory leaks
            try:
                plt.close('all')
                print('just closed all matplotlib figures')
            except Exception as e:
                print(f"Failed to close matplotlib figures: {str(e)}")
                pass
    
    def _should_run_validation_batch(self, trainer: pl.Trainer, batch_idx: int, batch_size: int = None, dataloader_idx: int = 0) -> bool:
        """
        Determine if a validation batch should be processed, based on desired number of CONDITIONAL validation samples.
        It calculates how many batches are needed to achieve the desired number of validation samples 
        and selects batches with appropriate stride to distribute them evenly across the validation set.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance containing validation information.
        batch_idx : int
            The index of the current validation batch being considered.
        batch_size : int, optional
            The size of the current batch. If provided, used to calculate exact
            number of batches needed. If None, uses a default estimate of 16.
            
        Returns
        -------
        bool
            True if the batch should be processed, False otherwise.
            
        Notes
        -----
        The method ensures at least one batch is always processed and distributes
        the selected batches evenly across the validation set using stride calculation.
        Uses num_conditional_generations to determine sampling.
        """
        try:
            num_val_batches = trainer.num_val_batches[int(dataloader_idx)]
        except Exception:
            num_val_batches = trainer.num_val_batches[0] if isinstance(trainer.num_val_batches, (list, tuple)) else trainer.num_val_batches
        
        # If we have the batch size, use it to compute exact number of batches needed
        batches_needed = max(1, (self.num_conditional_generations + batch_size - 1) // batch_size)  # Ceiling division
        batches_to_use = min(batches_needed, num_val_batches)

        # Use deterministic seed based on (num_val_batches, dataloader_idx).
        # random.Random only accepts scalar seed types, so combine into a single int.
        seed_int = (int(num_val_batches) * 1_000_003) ^ int(dataloader_idx)
        rng = random.Random(seed_int)
        # Generate deterministic set of batch indices to use
        selected_batches = set(rng.sample(range(num_val_batches), batches_to_use))
        
        return batch_idx in selected_batches
    
    def _process_conditional_generation_sample_post_generation(
        self,
        pl_module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_sequence: torch.Tensor,
        batch_idx: int,
        sample_idx: int,
        generation_time: float = 0.0,
        neighbors_smiles: Optional[List[str]] = None,
        generation_idx: int = 0,
        metrics_computer=None,
        dataset_key: Optional[str] = None,
        dataloader_idx: Optional[int] = None,
        prompt_metadata: Optional[dict] = None,
    ) -> Tuple[bool, Dict[str, bool], Optional[bool], Optional[str]]:
        """
        Process a single generated molecule for conditional generation validation.
        
        CLEAN VERSION: Just extracts SMILES and delegates ALL computation to ConditionalMetricsComputer.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module
        input_ids : torch.Tensor
            Original input token IDs for the sample
        attention_mask : torch.Tensor
            Attention mask indicating valid tokens
        generated_sequence : torch.Tensor
            The generated token sequence
        batch_idx : int
            Index of the current batch
        sample_idx : int
            Index of the sample within the batch (seed index)
        generation_time : float
            Time taken to generate this sample
        neighbors_smiles : Optional[List[str]]
            Optional list of neighbor SMILES
        generation_idx : int
            Index of this generation for the same seed (0 for first, 1 for second, etc.)
            
        Returns
        -------
        tuple
            (is_valid, valid_and_improved dict, is_memorized, canonical_gen)
        """
        # 1. Extract SMILES strings
        unmasked_ids, cond_str = SmilesUtils.extract_conditioning_from_batch(
            input_ids, attention_mask, pl_module.tokenizer
        )
        if generated_sequence.numel() == 0 or unmasked_ids is None:
            if prompt_metadata is not None:
                self._save_sample({
                    **prompt_metadata,
                    "smiles_generated": "",
                    "smiles_conditioning": prompt_metadata["seed_smiles"],
                    "smiles_input_1": prompt_metadata["seed_smiles"],
                    "smiles_input_2": prompt_metadata["target_smiles"],
                    "generation_index": generation_idx,
                    "seed_index": sample_idx,
                    "is_valid": False,
                    "generation_error": "empty_generation" if generated_sequence.numel() == 0 else "empty_conditioning",
                    "dataset_key": dataset_key,
                }, "conditional", batch_idx)
            return False, {}, None, None

        generated_as_list = [int(g) for g in generated_sequence.reshape(-1)]
        conditioned_smiles, new_smiles = SmilesUtils.tokens_to_smiles(
            pl_module.tokenizer, generated_as_list
        )
        
        # Debug printing
        if self._should_print():
            gen_full_str = pl_module.tokenizer.decode(generated_sequence.detach().cpu().tolist())
            gen_new_ids = generated_sequence.detach().cpu()[len(unmasked_ids):]
            gen_new_str = pl_module.tokenizer.decode(gen_new_ids.tolist()) if gen_new_ids.numel() > 0 else ""
            self.printer.print_debug(
                f"[Conditional] cond='{shorten_str(cond_str)}' full='{shorten_str(gen_full_str)}' new='{shorten_str(gen_new_str)}'"
            )
            self.printer.print_debug(
                f"conditioned_smiles: {shorten_str(conditioned_smiles)} new_smiles: {shorten_str(new_smiles)}" 
            )

        # 2. Basic validity check
        is_valid = self._safe_valid(new_smiles) if new_smiles else False
        canonical_gen = SmilesUtils.canonicalize(new_smiles) if (is_valid and new_smiles) else None

        # 3. Get GT candidates
        full_input_smiles1, full_input_smiles2 = SmilesUtils.tokens_to_smiles(
            pl_module.tokenizer, input_ids.reshape(-1).tolist()
        )
        
        # Clean any remaining prompt tokens from extracted SMILES strings (for property computation only)
        # NOTE: This does NOT affect the model input - prompt tokens remain in input_ids for generation.
        # This only cleans the SMILES strings extracted from tokens, which are used for property computation.
        # Handles property prompt tokens (<pc>, <p_logp>, <p0_logp_16>, <pd_logp_06>, <pdir_up>, etc.)
        # AND distance prompt tokens (<dc>, <d_00>, <d_03>, etc.)
        prompt_token_pattern = r'<(?:pc|p(?:0|d|sd|dir|dmin|dmax)?_[^>]*|dc|d_\d+)>'
        if full_input_smiles1:
            full_input_smiles1 = re.sub(prompt_token_pattern, '', full_input_smiles1).strip()
        if full_input_smiles2:
            full_input_smiles2 = re.sub(prompt_token_pattern, '', full_input_smiles2).strip()
        if conditioned_smiles:
            conditioned_smiles = re.sub(prompt_token_pattern, '', conditioned_smiles).strip()
        if prompt_metadata is not None:
            full_input_smiles1 = prompt_metadata["seed_smiles"]
            full_input_smiles2 = prompt_metadata["target_smiles"]
            conditioned_smiles = full_input_smiles1
        
        gt_candidates = []
        if neighbors_smiles and isinstance(neighbors_smiles, (list, tuple)) and len(neighbors_smiles) > 0:
            gt_candidates = [sm for sm in neighbors_smiles if self._safe_valid(sm)]
        elif full_input_smiles2 and self._safe_valid(full_input_smiles2):
            gt_candidates = [full_input_smiles2]

        # 5. DELEGATE ALL PER-SAMPLE COMPUTATION TO METRICS COMPUTER
        # This includes distance metrics, properties, and all other metrics
        if metrics_computer is None:
            metrics_computer = self.conditional_metrics
            # Try to get distance percentile function if not already set
            if metrics_computer.distance_percentile_fn is None:
                # Try to get trainer from pl_module if available
                # Note: Lightning's trainer property raises RuntimeError if not attached
                try:
                    trainer = pl_module.trainer
                except RuntimeError:
                    trainer = None
                percentile_fn = self._get_distance_percentile_fn(trainer)
                if percentile_fn is not None:
                    metrics_computer.distance_percentile_fn = percentile_fn

        # Parse property prompts if available
        parsed_prompts = None
        if self._prompt_prop_names and unmasked_ids is not None:
            try:
                from pairit.prompting.property_prompting import parse_property_prompt_from_ids as _parse
                parsed_prompts = _parse(pl_module.tokenizer, unmasked_ids.tolist())
            except Exception:
                parsed_prompts = None
        
        # Parse distance prompt bin if available
        if prompt_metadata is not None:
            from pairit.prompting.property_prompting import ParsedPropertyPrompt
            parsed_prompts = [ParsedPropertyPrompt(**p) for p in prompt_metadata["prompt_target_prompts"]]

        distance_prompt_bin = None
        if self._distance_binning_mode is not None and unmasked_ids is not None:
            try:
                from pairit.prompting.distance_prompting import parse_distance_prompt_from_ids
                distance_prompt_bin = parse_distance_prompt_from_ids(pl_module.tokenizer, unmasked_ids.tolist())
            except Exception:
                distance_prompt_bin = None

        # Use full_input_smiles1 as seed_smiles for property computation in prompting mode
        # conditioned_smiles from generated sequence may not be valid SMILES due to prompt tokens
        seed_smiles_for_props = full_input_smiles1 if (full_input_smiles1 and self._safe_valid(full_input_smiles1)) else conditioned_smiles
        
        _sample_result = metrics_computer.process_sample(
            pl_module=pl_module,
            seed_smiles=seed_smiles_for_props,
            gen_smiles=new_smiles,
            is_valid=is_valid,
            canonical_gen=canonical_gen,
            gt_candidates=gt_candidates,
            neighbors_smiles=neighbors_smiles,
            input_smiles1=full_input_smiles1,
            input_smiles2=full_input_smiles2,
            dataset_key=dataset_key,
            parsed_prompts=parsed_prompts,
            value_binning_by_property=self._value_binning_per_prop,
            delta_binning_by_property=self._delta_binning_per_prop,
            value_binning_default=self._value_binning_default,
            delta_binning_default=self._delta_binning_default,
            distance_prompt_bin=distance_prompt_bin,
            distance_binning_mode=self._distance_binning_mode,
            distance_num_bins=self._distance_num_bins,
        )
        # Unpack SampleResult into local variables for downstream compatibility
        valid_and_improved = _sample_result.valid_and_improved
        tanimoto_dist = _sample_result.tanimoto_dist
        shape_tanimoto_dist = _sample_result.shape_tanimoto_dist
        input_tanimoto_dist = _sample_result.input_tanimoto_dist
        input_shape_tanimoto_dist = _sample_result.input_shape_tanimoto_dist
        input_props = _sample_result.seed_props
        gen_props = _sample_result.gen_props
        is_memorized = _sample_result.is_memorized
        distance_percentile = _sample_result.distance_percentile
        prompt_agreement = _sample_result.prompt_agreement

        # 6. Save sample if enabled (using data computed by metrics computer)
        sample_id = f"cond_{batch_idx}_{sample_idx}_gen{generation_idx}_{hash(str(conditioned_smiles))}"
        if prompt_metadata is not None or self._should_save_sample(sample_id):
            sample_data = {
                'smiles_generated': new_smiles,
                'smiles_conditioning': conditioned_smiles,
                'smiles_input_1': full_input_smiles1,
                'smiles_input_2': full_input_smiles2,
                'tokens_generated': generated_as_list,
                'tokens_conditioning': unmasked_ids.tolist() if unmasked_ids is not None else None,
                'is_valid': is_valid,
                'tanimoto_distance': float(tanimoto_dist) if tanimoto_dist is not None else None,
                'shape_tanimoto_distance': float(shape_tanimoto_dist) if shape_tanimoto_dist is not None else None,
                'input_tanimoto_distance': float(input_tanimoto_dist) if input_tanimoto_dist is not None else None,
                'input_shape_tanimoto_distance': float(input_shape_tanimoto_dist) if input_shape_tanimoto_dist is not None else None,
                'distance_percentile': float(distance_percentile) if distance_percentile is not None else None,
                'generation_time': generation_time,
                'sample_id': sample_id,
                'generation_index': generation_idx,  # Track which generation this is (0, 1, 2, ...)
                'seed_index': sample_idx,  # Track which seed this came from
                'gt_candidate_selected': gt_candidates[0] if gt_candidates else None,
                # Provenance for mixed validation:
                'dataset_key': dataset_key,
                'dataloader_idx': int(dataloader_idx) if dataloader_idx is not None else None,
            }
            if prompt_metadata is not None:
                sample_data.update(prompt_metadata)

            # Parse and log prompt spec (property prompting) from conditioning ids.
            if self._prompt_prop_names:
                try:
                    from pairit.prompting.property_prompting import parse_property_prompt_from_ids as _parse
                    from pairit.prompting.property_prompting import format_property_prompt_summary as _fmt
                    parsed = _parse(pl_module.tokenizer, unmasked_ids.tolist() if unmasked_ids is not None else [])
                    sample_data["prompted_properties"] = [p.property_name for p in parsed]
                    # Human-readable rendering for quick inspection
                    try:
                        sample_data["prompt_summary"] = _fmt(
                            parsed,
                            value_binning_by_property=self._value_binning_per_prop,
                            delta_binning_by_property=self._delta_binning_per_prop,
                            value_binning=self._value_binning_default,
                            delta_binning=self._delta_binning_default,
                            show_ranges=True,
                            decimals=3,
                        )
                    except Exception:
                        sample_data["prompt_summary"] = _fmt(parsed, show_ranges=False)
                    for p, start_rng, delta_rng in self._prompt_ranges_from_parsed(parsed):
                        prop = p.property_name
                        if prop not in self._prompt_prop_names:
                            continue
                        if p.start_bin is not None:
                            sample_data[f"prompt_{prop}_start_bin"] = int(p.start_bin)
                        if start_rng is not None:
                            sample_data[f"prompt_{prop}_start_range_lo"] = float(start_rng[0])
                            sample_data[f"prompt_{prop}_start_range_hi"] = float(start_rng[1])
                        if p.direction is not None:
                            sample_data[f"prompt_{prop}_dir"] = str(p.direction)
                        if p.delta_bin is not None:
                            sample_data[f"prompt_{prop}_delta_bin"] = int(p.delta_bin)
                        if delta_rng is not None:
                            sample_data[f"prompt_{prop}_delta_range_lo"] = float(delta_rng[0])
                            sample_data[f"prompt_{prop}_delta_range_hi"] = float(delta_rng[1])
                except Exception:
                    # Parsing should never break validation; just skip prompt logging on failure.
                    pass
            
            # Add computed properties from metrics computer (no recomputation needed)
            # Save all computed properties (no filtering)
            if gen_props:
                for prop_name, prop_value in gen_props.items():
                    sample_data[f'property_gen_{prop_name}'] = prop_value
            
            if input_props:
                for prop_name, prop_value in input_props.items():
                    sample_data[f'property_cond_{prop_name}'] = prop_value
                
                # Add deltas
                if gen_props:
                    for prop_name in gen_props.keys():
                        if gen_props.get(prop_name) is not None and input_props.get(prop_name) is not None:
                            sample_data[f'property_delta_{prop_name}'] = gen_props[prop_name] - input_props[prop_name]
            
            # Add prompt agreement information
            if prompt_agreement:
                for prop_name, agreement in prompt_agreement.items():
                    if agreement.get('direction_match') is not None:
                        sample_data[f'prompt_{prop_name}_direction_match'] = bool(agreement['direction_match'])
                    if agreement.get('magnitude_match') is not None:
                        sample_data[f'prompt_{prop_name}_magnitude_match'] = bool(agreement['magnitude_match'])
            
            # Add distance prompt information
            if distance_prompt_bin is not None:
                sample_data['prompt_distance_bin'] = int(distance_prompt_bin)
                # Compute distance prompt match if we have percentile and binning config
                if distance_percentile is not None and self._distance_binning_mode is not None and self._distance_num_bins is not None:
                    try:
                        from pairit.prompting.distance_prompting import percentile_to_bin
                        actual_bin = percentile_to_bin(
                            percentile=distance_percentile,
                            mode=self._distance_binning_mode,  # type: ignore
                            num_bins=self._distance_num_bins,
                        )
                        sample_data['prompt_distance_match'] = bool(actual_bin == distance_prompt_bin)
                    except Exception:
                        # If computation fails, just skip
                        pass
            
            # Add SNN reference info if enabled and available
            if "snn" in self.expensive_metrics_to_compute and self._reference_smiles and is_valid and new_smiles:
                try:
                    closest_ref_smiles, closest_ref_similarity, closest_ref_index = compute_single_molecule_snn(
                        new_smiles, self._reference_smiles
                    )
                    sample_data.update({
                        'closest_ref_smiles': closest_ref_smiles,
                        'closest_ref_similarity': closest_ref_similarity,
                        'closest_ref_index': closest_ref_index
                    })
                except Exception:
                    # If SNN computation fails, just skip adding reference info
                    pass
            
            # Add per-property improvement flags (valid_and_improved is {prop_name: bool})
            if valid_and_improved:
                for prop_name, is_improved in valid_and_improved.items():
                    sample_data[f'improved_{prop_name}'] = is_improved

            # Add memorization status (will be True/False/None depending on whether it was computed)
            sample_data['is_memorized'] = is_memorized

            # Also add per-reference memorization flags (named reference sets)
            if "memorization" in self.expensive_metrics_to_compute and getattr(self, "_reference_canon_sets_by_name", None):
                for ref_name, ref_set in self._reference_canon_sets_by_name.items():
                    col = f"is_memorized_{self._sanitize_ref_name(ref_name)}"
                    if is_valid and canonical_gen and ref_set:
                        sample_data[col] = (canonical_gen in ref_set)
                    else:
                        # None = not checked / not applicable (e.g., invalid molecule)
                        sample_data[col] = None
            
            self._save_sample(sample_data, 'conditional', batch_idx)

        return is_valid, valid_and_improved, is_memorized, canonical_gen


    @property
    def _effective_metric_samples(self) -> int:
        """Number of conditional seeds for which full metrics are computed."""
        if self.num_conditional_metric_samples is not None:
            return min(self.num_conditional_metric_samples, self.num_conditional_generations)
        return self.num_conditional_generations

    def _process_conditional_generation_lightweight(
        self,
        pl_module,
        input_ids,
        attention_mask,
        generated_sequence,
    ) -> tuple[bool, str | None]:
        """Cheap validity-only processing (no metrics_computer, no memorization).

        Returns
        -------
        tuple
            (is_valid, canonical_gen)
        """
        if generated_sequence.numel() == 0:
            return False, None

        unmasked_ids, cond_str = SmilesUtils.extract_conditioning_from_batch(
            input_ids, attention_mask, pl_module.tokenizer
        )
        if unmasked_ids is None:
            return False, None

        generated_as_list = [int(g) for g in generated_sequence.reshape(-1)]
        _conditioned_smiles, new_smiles = SmilesUtils.tokens_to_smiles(
            pl_module.tokenizer, generated_as_list
        )

        is_valid = self._safe_valid(new_smiles) if new_smiles else False
        canonical_gen = SmilesUtils.canonicalize(new_smiles) if (is_valid and new_smiles) else None
        return is_valid, canonical_gen

    def _register_conditional_fraction_metrics(self, registry, trainer, pl_module) -> None:
        """Register epoch-level conditional fraction metrics and log everything to WandB.

        This must run even when unconditional generation is skipped so that
        conditional validity/improvement/VIN metrics are always logged.
        """
        _t_metric_reg_start = time.time()
        registry.register_fraction(
            "molecule_validity_conditional",
            self._epoch_cond_valid_gens, self._epoch_cond_total_gens,
            "Overall conditional validity", "Conditional - Basic",
        )
        _metric_denom = self._epoch_cond_metric_total_gens
        for spec in self.property_specs:
            registry.register_fraction(
                f"fraction_valid_and_improved_conditional/{spec.name}",
                self._epoch_cond_improved_counts.get(spec.name, 0), _metric_denom,
                f"Fraction valid+improved ({spec.name})", "Conditional - Basic",
            )
        registry.register_fraction(
            "fraction_valid_and_improved_conditional/any",
            self._epoch_cond_improved_any, _metric_denom,
            "Fraction valid+improved (any property)", "Conditional - Basic",
        )
        if "memorization" in self.expensive_metrics_to_compute:
            for spec in self.property_specs:
                registry.register_fraction(
                    f"fraction_of_valid_and_not_memorized_and_improved_from_all_conditional/{spec.name}",
                    self._epoch_cond_vnim_counts.get(spec.name, 0), _metric_denom,
                    f"Valid+NM+improved ({spec.name})", "Conditional - Basic",
                )
            registry.register_fraction(
                "fraction_of_valid_and_not_memorized_and_improved_from_all_conditional/any",
                self._epoch_cond_vnim_any, _metric_denom,
                "Valid+NM+improved (any property)", "Conditional - Basic",
            )
            for ref_name in sorted(self._epoch_cond_vnim_counts_by_ref.keys()):
                for spec in self.property_specs:
                    registry.register_fraction(
                        f"fraction_of_valid_and_not_memorized_and_improved_from_all_conditional/{ref_name}/{spec.name}",
                        self._epoch_cond_vnim_counts_by_ref[ref_name].get(spec.name, 0),
                        _metric_denom,
                        f"Valid+NM+improved ({spec.name}) vs {ref_name}", "Conditional - Basic",
                    )
                registry.register_fraction(
                    f"fraction_of_valid_and_not_memorized_and_improved_from_all_conditional/{ref_name}/any",
                    self._epoch_cond_vnim_any_by_ref.get(ref_name, 0), _metric_denom,
                    f"Valid+NM+improved (any) vs {ref_name}", "Conditional - Basic",
                )
        self.timing_tracker.record("metric_registration", time.time() - _t_metric_reg_start)

        # Log all metrics
        # The global registry contains fraction metrics (validity, V/I/N, etc.) that are
        # accumulated across ALL validation dataloaders. These must always be logged,
        # even in split mode, because per-dataset ConditionalMetricsComputers do not
        # track these fractions — they only track distances, uniqueness, memorization, etc.
        _t_logging_start = time.time()
        if rank_zero_only.rank == 0:
            self.printer.print_header(f"Validation Metrics - Epoch {trainer.current_epoch}")
            registry.log_all(pl_module, self.printer, prefix=self._with_metric_prefix("val"), echo=True)
        else:
            registry.log_all(pl_module, None, prefix=self._with_metric_prefix("val"), echo=False)
        self.timing_tracker.record("metric_logging", time.time() - _t_logging_start)

        if rank_zero_only.rank == 0:
            self.timing_tracker.print_summary(epoch=trainer.current_epoch)
        self.timing_tracker.log_all(pl_module, prefix="timing")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Propagate epoch to datamodule for epoch-varying component dropout."""
        dm = trainer.datamodule
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(trainer.current_epoch)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Reset per-epoch counters for conditional generation.
        """
        self._prompt_requests = {}
        # Detect DDP world_size and compute per-rank generation quotas so the
        # global total stays constant regardless of the number of ranks.
        # Priority: trainer.world_size > dist.get_world_size() > env WORLD_SIZE.
        # When using torchrun + Lightning SingleDeviceStrategy, neither
        # trainer.world_size nor dist.is_initialized() reflect the true world
        # size, so we fall back to the WORLD_SIZE env var set by torchrun.
        import torch.distributed as _dist
        import os
        _trainer_ws = getattr(trainer, 'world_size', 1)
        if _trainer_ws > 1:
            self._world_size = _trainer_ws
        elif _dist.is_initialized():
            self._world_size = _dist.get_world_size()
        else:
            self._world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if self.export_prompt_metadata and self._world_size != 1:
            raise ValueError("Complete prompt export currently requires single-process evaluation")
        self._effective_conditional_gens = max(1, self.num_conditional_generations // self._world_size)
        self._effective_unconditional_gens = max(1, self.num_unconditional_generations // self._world_size)
        self._effective_conditional_metric_gens = max(1, self._effective_metric_samples // self._world_size)

        self._conditional_generated_this_epoch = 0
        self._conditional_generated_this_epoch_by_key = {}
        self._printed_conditional_batch_summary = False
        self._conditional_metric_processed_this_epoch = 0
        self._conditional_metric_processed_this_epoch_by_key: dict[str, int] = {}

        # Epoch-level accumulators for conditional fraction metrics (DDP-correct)
        self._epoch_cond_total_gens = 0
        self._epoch_cond_valid_gens = 0
        self._epoch_cond_metric_total_gens = 0  # denominator for improvement/VIN (metrics subset)
        self._epoch_cond_improved_counts: dict[str, int] = {}
        self._epoch_cond_improved_any = 0
        self._epoch_cond_vnim_counts: dict[str, int] = {}  # valid+not_memorized+improved
        self._epoch_cond_vnim_any = 0
        self._epoch_cond_vnim_counts_by_ref: dict[str, dict[str, int]] = {}
        self._epoch_cond_vnim_any_by_ref: dict[str, int] = {}
        self._epoch_cond_relevant_by_ref: dict[str, int] = {}
        self._epoch_cond_relevancy_details_by_ref: dict[str, dict[str, int]] = {}
        
        # Reset metrics computers (they manage their own buffers now)
        if self._split_enabled(trainer):
            self._cond_metrics_by_key = {}
            self._uncond_metrics_by_key = {}
            self._dataset_key_to_id = {}
        else:
            self.conditional_metrics.reset()
            self.unconditional_metrics.reset()
        
        # Reset timing tracker for new epoch and start wall-clock
        self.timing_tracker.clear()
        self.timing_tracker.start_epoch()

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Called after each validation batch to perform conditional molecule generation validation.
        
        This method is triggered by PyTorch Lightning after each validation batch.
        It performs batched conditional molecule generation, processes the results,
        and logs validity metrics. The method includes comprehensive timing
        measurements for performance analysis.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer object with sep_token_id, eos_token_id, pad_token_id, cls_token and decode()
            - model: huggingface-compatible model exposing .generate()
            - _max_length: int, maximum tokens to generate
        outputs
            Outputs from the validation step (unused in this callback).
        batch : dict
            Dictionary containing the batch data with keys:
            - "input_ids": torch.LongTensor, shape [batch_size, seq_len]
            - "attention_mask": torch.LongTensor or BoolTensor, shape [batch_size, seq_len]
        batch_idx : int
            Index of the current validation batch.
            
        Returns
        -------
        None
            Results are logged to the pl_module.
            
        Notes
        -----
        This method performs conditional generation by:
        1. Extracting conditioning sequences from input batches
        2. Generating new molecules using batched model forward pass
        3. Processing each generated molecule individually for validity checking
        4. Computing distance metrics between conditioning and generated molecules
        5. Logging results with timing information
        """
        if (self.only_dataloader_idx is not None) and (int(dataloader_idx) != int(self.only_dataloader_idx)):
            return

        dataset_key = self._dataset_key_from_batch(batch) if self._split_enabled(trainer) else None
        cond_metrics = self._get_conditional_metrics_for_key(dataset_key, trainer) if dataset_key is not None else self.conditional_metrics
        # Try to update distance_percentile_fn if not set
        if cond_metrics.distance_percentile_fn is None:
            percentile_fn = self._get_distance_percentile_fn(trainer)
            if percentile_fn is not None:
                cond_metrics.distance_percentile_fn = percentile_fn
        if dataset_key is not None:
            # Track dataset_id for optional per-dataset unconditional prompting later
            ds_id = batch.get("dataset_id", None)
            try:
                if hasattr(ds_id, "reshape"):
                    ds_id = int(ds_id.reshape(-1)[0].item())
                elif isinstance(ds_id, (list, tuple)) and ds_id:
                    ds_id = int(ds_id[0])
                elif ds_id is not None:
                    ds_id = int(ds_id)
            except Exception:
                ds_id = None
            if ds_id is not None:
                self._dataset_key_to_id.setdefault(dataset_key, ds_id)

        if self._should_print():
            self.printer.print_debug(f"on_validation_batch_end: batch_idx={batch_idx}")
        
        # Skip validation generation if neither unconditional nor conditional generations are requested
        if self.num_unconditional_generations <= 0 and self.num_conditional_generations <= 0:
            return
            
        # Only run this occasionally to avoid slowing down training too much
        # Only run on a subset of batches to get desired number of total generations
        if not self._should_run_validation_batch(trainer, batch_idx, batch["input_ids"].shape[0], dataloader_idx=dataloader_idx):
            return
            
        batch_start_time = time.time()
        
        # Determine how many samples we still need to process this epoch.
        # IMPORTANT: in mixed-val (multiple validation loaders), apply the quota PER dataset.
        if dataset_key is not None and self._split_enabled(trainer):
            already_processed = int(self._conditional_generated_this_epoch_by_key.get(dataset_key, 0))
        else:
            already_processed = int(self._conditional_generated_this_epoch)
        remaining_needed = max(0, self._effective_conditional_gens - already_processed)
        if remaining_needed <= 0:
            return

        # Subsample within the batch to avoid exceeding the target per epoch
        batch_size = batch["input_ids"].shape[0]
        process_count = min(batch_size, remaining_needed)

        # Determine how many seeds in this batch get full metric processing vs lightweight (validity-only).
        if dataset_key is not None and self._split_enabled(trainer):
            already_metric_processed = int(self._conditional_metric_processed_this_epoch_by_key.get(dataset_key, 0))
        else:
            already_metric_processed = int(self._conditional_metric_processed_this_epoch)
        remaining_metric_seeds = max(0, self._effective_conditional_metric_gens - already_metric_processed)
        seeds_for_metrics_this_batch = min(process_count, remaining_metric_seeds)

        if self._should_print():
            self.printer.print_debug(f"Processing conditional batch {batch_idx}: {process_count}/{batch_size} samples (remaining={remaining_needed})")
            if self.num_conditional_metric_samples is not None:
                self.printer.print_debug(f"  metric subset: {seeds_for_metrics_this_batch}/{process_count} seeds get full metrics")

        # Batch generate all samples in this batch (with timing)
        # If num_generations_per_seed > 1, we'll get multiple generations per seed
        prompt_rows = self._prompt_batch_metadata(batch, process_count, dataset_key)
        pair_seeds = ([row["generation_pair_seed"] for row in prompt_rows]
                      if self.pairit_generation_seed is not None else None)
        with self.timing_tracker.time_operation("conditional/generation"):
            generation_start_time = time.time()
            all_generated_sequences = MoleculeGenerator.batch_generate_conditional(
                pl_module, batch["input_ids"][:process_count], batch["attention_mask"][:process_count], 
                temperature=self.temperature, num_generations_per_seed=self.num_generations_per_seed,
                pair_seeds=pair_seeds,
            )
            generation_time = time.time() - generation_start_time
            total_generations = process_count * self.num_generations_per_seed
            per_sample_generation_time = generation_time / total_generations
            if self._should_print():
                self.printer.print_timing(f"Batched conditional generation: {generation_time:.3f}s for {process_count} seeds ({total_generations} total generations)")
        
        # Process each generated sequence individually for validity checks (with timing)
        # Note: if num_generations_per_seed > 1, all_generated_sequences has length = process_count * num_generations_per_seed
        # Results are ordered as: [seed0_gen0, seed0_gen1, ..., seed1_gen0, seed1_gen1, ...]
        # IMPORTANT: total_generations is the denominator for validity fraction (includes ALL generations)
        with self.timing_tracker.time_operation("conditional/processing"):
            processing_start_time = time.time()
            valid_generations = 0
            improved_counts: Dict[str, int] = {spec.name: 0 for spec in self.property_specs}
            improved_any_count = 0
            # Track valid + not memorized + improved counts (only if memorization is enabled)
            valid_not_memorized_improved_counts: Dict[str, int] = {spec.name: 0 for spec in self.property_specs}
            valid_not_memorized_improved_any_count = 0
            # Also track per-reference "not memorized" counts (one entry per reference_smiles_files item)
            valid_not_memorized_improved_counts_by_ref: Dict[str, Dict[str, int]] = {}
            valid_not_memorized_improved_any_count_by_ref: Dict[str, int] = {}
            if "memorization" in self.expensive_metrics_to_compute and getattr(self, "_reference_canon_sets_by_name", None):
                for ref_name in sorted(self._reference_canon_sets_by_name.keys()):
                    valid_not_memorized_improved_counts_by_ref[ref_name] = {spec.name: 0 for spec in self.property_specs}
                    valid_not_memorized_improved_any_count_by_ref[ref_name] = 0
            
            # Relevancy tracking per reference
            relevant_by_ref: Dict[str, int] = {}
            relevancy_details_by_ref: Dict[str, Dict[str, int]] = {}
            # Initialize per-reference tracking if relevancy is enabled
            
            neighbors_batch = batch.get("neighbors_smiles", None)
            for i, generated_sequence in enumerate(all_generated_sequences):
                # Map generation index back to seed index
                seed_idx = i // self.num_generations_per_seed
                generation_idx = i % self.num_generations_per_seed

                # Get the input_ids and attention_mask for the seed
                input_ids = batch["input_ids"][seed_idx]
                attention_mask = batch["attention_mask"][seed_idx]

                # Decide: full metrics path vs lightweight (validity-only) path
                use_full_metrics = seed_idx < seeds_for_metrics_this_batch

                if use_full_metrics:
                    neighbors_for_i = neighbors_batch[seed_idx] if neighbors_batch is not None and seed_idx < len(neighbors_batch) else None
                    is_valid, flags, is_memorized, canonical_gen = self._process_conditional_generation_sample_post_generation(
                        pl_module,
                        input_ids,
                        attention_mask,
                        generated_sequence,
                        batch_idx,
                        seed_idx,
                        per_sample_generation_time,
                        neighbors_for_i,
                        generation_idx,
                        metrics_computer=cond_metrics,
                        dataset_key=dataset_key,
                        dataloader_idx=dataloader_idx,
                        prompt_metadata=prompt_rows[seed_idx],
                    )
                else:
                    # Lightweight path: only validity check, no metrics_computer call
                    is_valid, canonical_gen = self._process_conditional_generation_lightweight(
                        pl_module, input_ids, attention_mask, generated_sequence,
                    )
                    flags = {}
                    is_memorized = None
                if is_valid:
                    valid_generations += 1
                
                # Relevancy check (only for valid molecules) - per reference
                any_flag = False
                for spec in self.property_specs:
                    if flags.get(spec.name, False):
                        improved_counts[spec.name] += 1
                        any_flag = True
                if any_flag:
                    improved_any_count += 1
                
                # Track valid + not memorized + improved
                if "memorization" in self.expensive_metrics_to_compute:
                    # (A) Default reference ("training_canon_set") — only if memorization was computed.
                    if is_memorized is not None and is_valid and not is_memorized:
                        any_improved_flag = False
                        for spec in self.property_specs:
                            if flags.get(spec.name, False):
                                valid_not_memorized_improved_counts[spec.name] += 1
                                any_improved_flag = True
                        if any_improved_flag:
                            valid_not_memorized_improved_any_count += 1

                    # (B) Per-reference (one metric entry per reference_smiles_files item).
                    # Uses canonical membership against each named reference set.
                    if is_valid and canonical_gen and valid_not_memorized_improved_counts_by_ref:
                        for ref_name, ref_set in self._reference_canon_sets_by_name.items():
                            ref_set = ref_set or set()
                            is_mem_ref = (canonical_gen in ref_set) if ref_set else False
                            if is_mem_ref:
                                continue
                            any_improved_flag_ref = False
                            for spec in self.property_specs:
                                if flags.get(spec.name, False):
                                    valid_not_memorized_improved_counts_by_ref[ref_name][spec.name] += 1
                                    any_improved_flag_ref = True
                            if any_improved_flag_ref:
                                valid_not_memorized_improved_any_count_by_ref[ref_name] += 1
        
        processing_time = time.time() - processing_start_time
        if self._should_print():
            self.printer.print_timing(f"Conditional post-processing: {processing_time:.3f}s for {process_count} samples")

        batch_total_time = time.time() - batch_start_time
        if self._should_print():
            self.printer.print_timing(f"Total conditional batch {batch_idx}: {batch_total_time:.3f}s for {process_count} samples")

        # Update per-epoch counters (track seeds, not total generations)
        frac_valid = float(valid_generations) / total_generations if total_generations > 0 else 0.0
        if dataset_key is not None and self._split_enabled(trainer):
            self._conditional_generated_this_epoch_by_key[dataset_key] = already_processed + process_count
            self._conditional_metric_processed_this_epoch_by_key[dataset_key] = (
                self._conditional_metric_processed_this_epoch_by_key.get(dataset_key, 0) + seeds_for_metrics_this_batch
            )
        else:
            self._conditional_generated_this_epoch += process_count
            self._conditional_metric_processed_this_epoch += seeds_for_metrics_this_batch

        # --- Accumulate raw counts into epoch-level accumulators (DDP-correct) ---
        # These will be registered as fractions at epoch end via register_fraction().
        self._epoch_cond_total_gens += total_generations
        self._epoch_cond_valid_gens += valid_generations
        self._epoch_cond_metric_total_gens += seeds_for_metrics_this_batch * self.num_generations_per_seed

        for spec in self.property_specs:
            self._epoch_cond_improved_counts[spec.name] = self._epoch_cond_improved_counts.get(spec.name, 0) + improved_counts[spec.name]
        self._epoch_cond_improved_any += improved_any_count

        # Relevancy accumulators

        # Valid+not_memorized+improved accumulators
        if "memorization" in self.expensive_metrics_to_compute:
            for spec in self.property_specs:
                self._epoch_cond_vnim_counts[spec.name] = self._epoch_cond_vnim_counts.get(spec.name, 0) + valid_not_memorized_improved_counts[spec.name]
            self._epoch_cond_vnim_any += valid_not_memorized_improved_any_count

            if valid_not_memorized_improved_counts_by_ref:
                for ref_name in sorted(valid_not_memorized_improved_counts_by_ref.keys()):
                    if ref_name not in self._epoch_cond_vnim_counts_by_ref:
                        self._epoch_cond_vnim_counts_by_ref[ref_name] = {}
                    for spec in self.property_specs:
                        self._epoch_cond_vnim_counts_by_ref[ref_name][spec.name] = (
                            self._epoch_cond_vnim_counts_by_ref[ref_name].get(spec.name, 0)
                            + valid_not_memorized_improved_counts_by_ref[ref_name][spec.name]
                        )
                    self._epoch_cond_vnim_any_by_ref[ref_name] = (
                        self._epoch_cond_vnim_any_by_ref.get(ref_name, 0)
                        + valid_not_memorized_improved_any_count_by_ref.get(ref_name, 0)
                    )

        # Print concise batch summary (no logging — metrics logged at epoch end)
        if self._should_print():
            self.printer.print_info(f"[Conditional] validity={frac_valid:.3f} ({valid_generations}/{total_generations})")
            any_frac = float(improved_any_count) / total_generations if total_generations > 0 else 0.0
            self.printer.print_info(
                "[Conditional] improved=" + ", ".join(
                    [f"{spec.name}:{(float(improved_counts[spec.name]) / total_generations if total_generations > 0 else 0.0):.3f}" for spec in self.property_specs]
                ) + f", any:{any_frac:.3f}"
            )

        # Optional compact summary of conditional metrics for the FIRST printed batch only
        if not getattr(self, "_printed_conditional_batch_summary", False):
            try:
                cm = self.conditional_metrics
                # Track starting indices captured just before processing began
                # If not present (first run after code update), fall back to using entire arrays
                starts = getattr(self, "_cm_batch_starts", None)
                if starts is None:
                    # Create starts as zeroes so we summarize current batch fully next time
                    starts = {
                        "dist_to_gt": 0,
                        "gt_min_delta": 0,
                        "neigh_median": 0,
                        "neigh_p75": 0,
                        "neigh_nn": 0,
                        "kde_ll_in": 0,
                        "kde_ll_gen": 0,
                        "w1": {k: 0 for k in METRIC_FEATURE_NAMES},
                    }
                # Compute summary using static method
                dist_mean = MoleculeValidationCallback._mean_slice(cm.epoch_dist_to_gt, starts.get("dist_to_gt", 0))
                delta_mean = MoleculeValidationCallback._mean_slice(cm.epoch_gt_min_delta, starts.get("gt_min_delta", 0))
                neigh_med = MoleculeValidationCallback._mean_slice(cm.epoch_neigh_median_dist, starts.get("neigh_median", 0))
                neigh_p75 = MoleculeValidationCallback._mean_slice(cm.epoch_neigh_p75_dist, starts.get("neigh_p75", 0))
                neigh_nn  = MoleculeValidationCallback._mean_slice(cm.epoch_neigh_nn_dist, starts.get("neigh_nn", 0))
                kde_in    = MoleculeValidationCallback._mean_slice(cm.epoch_kde_ll_in, starts.get("kde_ll_in", 0))
                kde_gen   = MoleculeValidationCallback._mean_slice(cm.epoch_kde_ll_gen, starts.get("kde_ll_gen", 0))

                # Macro W1 delta across features (mean of feature means), signed so more negative is closer
                w1_means = []
                for feat in METRIC_FEATURE_NAMES:
                    lst = cm.epoch_w1_deltas.get(feat, [])
                    start_idx = starts.get("w1", {}).get(feat, 0)
                    if start_idx is None:
                        start_idx = 0
                    vals = [float(v) for v in lst[start_idx:] if v is not None]
                    if vals:
                        w1_means.append(sum(vals) / len(vals))
                w1_macro = (sum(w1_means) / len(w1_means)) if w1_means else None

                if self._should_print():
                    self.printer.print_subheader("Conditional batch summary (preview)")
                    if dist_mean is not None:
                        self.printer.print_metric("cond_distance_to_gt_mean", dist_mean, description="mean over this batch")
                    if delta_mean is not None:
                        self.printer.print_metric("cond_gt_closest_delta_mean", delta_mean, description="gen−input vs GT")
                    if neigh_med is not None:
                        self.printer.print_metric("cond_neighborhood_median_dist", neigh_med)
                    if neigh_p75 is not None:
                        self.printer.print_metric("cond_neighborhood_p75_dist", neigh_p75)
                    if neigh_nn is not None:
                        self.printer.print_metric("cond_neighborhood_nearest_dist", neigh_nn)
                    if kde_in is not None:
                        self.printer.print_metric("cond_kde_ll_input", kde_in)
                    if kde_gen is not None:
                        self.printer.print_metric("cond_kde_ll_generated", kde_gen)
                    if w1_macro is not None:
                        self.printer.print_metric("cond_w1_delta_macro_mean", w1_macro, description="mean over features (neg=closer)")
                self._printed_conditional_batch_summary = True
            except Exception as e:
                # Best-effort; do not fail training due to preview printing
                print(f"[Warning] Failed to print conditional batch summary: {e}")
                self._printed_conditional_batch_summary = True
        
        # Flush saved samples for this batch
        if self.save_generated_samples:
            self._flush_samples('conditional', trainer.global_step, pl_module, trainer)
    
    def _process_unconditional_generation_sample_post_generation(
        self,
        pl_module,
        sampled_tokens: list,
        cls_token_id: int,
        epoch: int,
        sample_idx: int,
        metrics_computer=None,
        dataset_key: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_dataset_id: Optional[int] = None,
    ) -> tuple:
        """
        Process a single generated molecule for unconditional generation validation.
        
        This method processes a generated molecule sequence for unconditional generation
        validation. It converts the token sequence to SMILES, validates the molecules,
        computes distance metrics, and determines if the generation was successful.
        This method is called after batched generation to process individual samples.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer with cls_token_id, pad/eos tokens and decode()
            - model: model with generation capabilities
        sampled_tokens : list
            The generated token IDs as a list of integers, shape [generated_seq_len].
        cls_token_id : int
            Token ID for the CLS token used as the starting point.
        epoch : int
            Current epoch number (for saving purposes).
        sample_idx : int
            Index of the sample within the epoch (for saving purposes).
            
        Returns
        -------
        tuple
            A tuple containing:
            - first_valid: bool, True if the first molecule is valid
            - second_valid: bool, True if the second molecule is valid (if present)
            
        Notes
        -----
        This method performs several operations:
        1. Converts generated tokens to SMILES strings
        2. Validates molecules using RDKit
        3. Computes distance metrics between generated molecules
        4. Logs results and prints generation details
        5. Optionally saves sample data for later analysis
        """
        # Get conditioning string
        cond_str = getattr(pl_module.tokenizer, "cls_token", None)
        if cond_str is None:
            cond_str = pl_module.tokenizer.decode([cls_token_id])
        
        # Convert tokens to SMILES
        smiles_string, second_smiles_string = SmilesUtils.tokens_to_smiles(pl_module.tokenizer, sampled_tokens)
        
        # Print generation details
        if sampled_tokens is not None and len(sampled_tokens) > 0:
            gen_ids = sampled_tokens
            gen_full_str = pl_module.tokenizer.decode(gen_ids)
            gen_new_ids = gen_ids[1:]
            gen_new_str = pl_module.tokenizer.decode(gen_new_ids)
            if self._should_print():
                print(f"\n[MoleculeValidationCallback] Unconditional generate: \n    cond='{shorten_str(cond_str)}'\n    generated_full='{shorten_str(gen_full_str)}'\n  generated_new='{shorten_str(gen_new_str)}'")
                print(f"    first smiles : {shorten_str(smiles_string)}\n    second smiles: {shorten_str(second_smiles_string)}")
        
        # Check validity
        first_valid = SmilesUtils.is_valid(smiles_string)
        second_valid = False
        
        if second_smiles_string is not None:
            second_valid = SmilesUtils.is_valid(second_smiles_string)
        
        # Compute distance metrics and properties using metrics computer
        if metrics_computer is None:
            metrics_computer = self.unconditional_metrics

        _sample_result = metrics_computer.process_sample(
            pl_module, smiles_string, second_smiles_string, first_valid, second_valid
        )
        # Unpack SampleResult into local variables for downstream compatibility
        tanimoto_dist = _sample_result.tanimoto_dist
        shape_tanimoto_dist = _sample_result.shape_tanimoto_dist
        first_props = _sample_result.seed_props
        second_props = _sample_result.gen_props
        
        # Try to infer any dataset-identity prompt present in the generated sequence itself.
        # This is useful when unconditional generation is prompted with dataset tokens,
        # or when the model emits them spontaneously.
        generated_dataset_id = None
        generated_dataset_prompt = None
        try:
            # Decode only a small prefix for speed; dataset tokens (if present) should be at the front.
            prefix_text = pl_module.tokenizer.decode(sampled_tokens[:16]) if sampled_tokens else ""
            m = re.search(r"<ds><ds_(\d+)>", prefix_text)
            if m:
                generated_dataset_id = int(m.group(1))
                generated_dataset_prompt = m.group(0)
        except Exception:
            pass

        # Save sample if enabled (using data computed by metrics computer)
        sample_id = f"uncond_{epoch}_{sample_idx}_{hash(str(smiles_string))}"
        if self._should_save_sample(sample_id):
            sample_data = {
                'smiles_generated': smiles_string,
                'smiles_second': second_smiles_string,
                'tokens_generated': sampled_tokens,
                'is_valid': first_valid,
                'is_second_valid': second_valid,
                'tanimoto_distance': float(tanimoto_dist) if tanimoto_dist is not None else None,
                'shape_tanimoto_distance': float(shape_tanimoto_dist) if shape_tanimoto_dist is not None else None,
                'generation_time': 0.0,  # Could measure this if needed
                'sample_id': sample_id,
                # Provenance for unconditional-per-dataset runs:
                'dataset_key': dataset_key,
                # What we *prompted* with (if anything)
                'prompt_text': prompt_text,
                'prompt_dataset_id': int(prompt_dataset_id) if prompt_dataset_id is not None else None,
                # What the model *actually emitted* (if it included dataset identity tokens)
                'generated_dataset_prompt': generated_dataset_prompt,
                'generated_dataset_id': generated_dataset_id,
            }

            # If we were prompted (text), parse and log prompt spec for verification.
            if self._prompt_prop_names and prompt_text:
                try:
                    # Encode prompt_text without adding <cls>/<eos>
                    ids = pl_module.tokenizer.encode(prompt_text, add_special_tokens=False)
                    from pairit.prompting.property_prompting import parse_property_prompt_from_ids as _parse
                    from pairit.prompting.property_prompting import format_property_prompt_summary as _fmt
                    parsed = _parse(pl_module.tokenizer, ids)
                    sample_data["prompted_properties"] = [p.property_name for p in parsed]
                    try:
                        sample_data["prompt_summary"] = _fmt(
                            parsed,
                            value_binning_by_property=self._value_binning_per_prop,
                            delta_binning_by_property=self._delta_binning_per_prop,
                            value_binning=self._value_binning_default,
                            delta_binning=self._delta_binning_default,
                            show_ranges=True,
                            decimals=3,
                        )
                    except Exception:
                        sample_data["prompt_summary"] = _fmt(parsed, show_ranges=False)
                    for p, start_rng, delta_rng in self._prompt_ranges_from_parsed(parsed):
                        prop = p.property_name
                        if prop not in self._prompt_prop_names:
                            continue
                        if p.start_bin is not None:
                            sample_data[f"prompt_{prop}_start_bin"] = int(p.start_bin)
                        if start_rng is not None:
                            sample_data[f"prompt_{prop}_start_range_lo"] = float(start_rng[0])
                            sample_data[f"prompt_{prop}_start_range_hi"] = float(start_rng[1])
                        if p.direction is not None:
                            sample_data[f"prompt_{prop}_dir"] = str(p.direction)
                        if p.delta_bin is not None:
                            sample_data[f"prompt_{prop}_delta_bin"] = int(p.delta_bin)
                        if delta_rng is not None:
                            sample_data[f"prompt_{prop}_delta_range_lo"] = float(delta_rng[0])
                            sample_data[f"prompt_{prop}_delta_range_hi"] = float(delta_rng[1])
                except Exception:
                    pass
            
            # Add computed properties from metrics computer (no recomputation needed)
            # Save all computed properties (no filtering)
            if first_props:
                for prop_name, prop_value in first_props.items():
                    sample_data[f'property_first_{prop_name}'] = prop_value
            
            if second_props:
                for prop_name, prop_value in second_props.items():
                    sample_data[f'property_second_{prop_name}'] = prop_value
                
                # Add deltas between first and second molecule
                if first_props:
                    for prop_name in second_props.keys():
                        if second_props.get(prop_name) is not None and first_props.get(prop_name) is not None:
                            sample_data[f'property_delta_{prop_name}'] = second_props[prop_name] - first_props[prop_name]
            
            # Add SNN reference info if enabled and available (for second molecule)
            if "snn" in self.expensive_metrics_to_compute and self._reference_smiles and second_valid and second_smiles_string:
                try:
                    closest_ref_smiles, closest_ref_similarity, closest_ref_index = compute_single_molecule_snn(
                        second_smiles_string, self._reference_smiles
                    )
                    sample_data.update({
                        'closest_ref_smiles': closest_ref_smiles,
                        'closest_ref_similarity': closest_ref_similarity,
                        'closest_ref_index': closest_ref_index
                    })
                except Exception:
                    # If SNN computation fails, just skip adding reference info
                    pass
            
            # Add memorization status if enabled
            if "memorization" in self.expensive_metrics_to_compute:
                is_first_memorized = False
                is_second_memorized = False
                first_canon = None
                second_canon = None
                
                if first_valid and smiles_string:
                    first_canon = SmilesUtils.canonicalize(smiles_string)
                if second_valid and second_smiles_string:
                    second_canon = SmilesUtils.canonicalize(second_smiles_string)
                
                if self.check_memorization_from_any_reference:
                    # Check if molecule is in ANY of the reference sets
                    if self._reference_canon_sets_by_name:
                        if first_canon:
                            is_first_memorized = any(
                                first_canon in ref_set 
                                for ref_set in self._reference_canon_sets_by_name.values()
                                if ref_set
                            )
                        if second_canon:
                            is_second_memorized = any(
                                second_canon in ref_set 
                                for ref_set in self._reference_canon_sets_by_name.values()
                                if ref_set
                            )
                else:
                    # Check only the default reference set (backward-compatible behavior)
                    if self._training_canon_set is not None:
                        if first_canon:
                            is_first_memorized = first_canon in self._training_canon_set
                        if second_canon:
                            is_second_memorized = second_canon in self._training_canon_set
                
                sample_data['is_first_memorized'] = is_first_memorized
                sample_data['is_second_memorized'] = is_second_memorized

            # Also add per-reference memorization flags (named reference sets)
            if "memorization" in self.expensive_metrics_to_compute and getattr(self, "_reference_canon_sets_by_name", None):
                # Reuse canonicalized SMILES from above if we already computed them, otherwise canonicalize
                if 'first_canon' not in locals() or first_canon is None:
                    first_canon = SmilesUtils.canonicalize(smiles_string) if first_valid and smiles_string else None
                if 'second_canon' not in locals() or second_canon is None:
                    second_canon = SmilesUtils.canonicalize(second_smiles_string) if second_valid and second_smiles_string else None
                for ref_name, ref_set in self._reference_canon_sets_by_name.items():
                    suffix = self._sanitize_ref_name(ref_name)
                    col1 = f"is_first_memorized_{suffix}"
                    col2 = f"is_second_memorized_{suffix}"
                    sample_data[col1] = (first_canon in ref_set) if (first_canon and ref_set) else None
                    sample_data[col2] = (second_canon in ref_set) if (second_canon and ref_set) else None
            
            self._save_sample(sample_data, 'unconditional', epoch)
        
        return first_valid, second_valid
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Compute validity accuracy over num_unconditional_generations unconditional samples.
        
        This method is triggered by PyTorch Lightning at the end of each validation epoch.
        It performs batched unconditional molecule generation starting from CLS tokens,
        processes the results, and logs comprehensive validity metrics. The method
        includes detailed timing measurements for performance analysis.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing:
            - tokenizer: tokenizer with cls_token_id, pad/eos tokens and decode()
            - model: huggingface-compatible model exposing .generate()
            - _max_length: int, maximum tokens to generate
            
        Returns
        -------
        None
            Results are logged to the pl_module and printed to console.
            
        Notes
        -----
        This method performs unconditional generation by:
        1. Generating molecules from scratch starting with CLS tokens
        2. Converting generated tokens to SMILES strings
        3. Validating molecules using RDKit
        4. Computing distance metrics between generated molecules
        5. Logging comprehensive validity statistics with timing information
        
        The method logs three main metrics:
        - val_molecule_validity: Overall validity accuracy
        - val_molecule_validity_first: Validity accuracy for first molecules
        - val_molecule_validity_second: Validity accuracy for second molecules
        """
        self.printer.print_section(f"Validation Epoch {trainer.current_epoch} Started")
        
        # === NEW: Use MetricRegistry for organized metric logging ===
        registry = MetricRegistry()
        
        # Compute and log conditional metrics.
        # In mixed-dataset mode, we can split metrics by dataset_id/name automatically.
        if not self.generation_only:
            with self.timing_tracker.time_operation("conditional/metrics_aggregation"):
                if self._split_enabled(trainer):
                    for ds_key, cm in self._cond_metrics_by_key.items():
                        reg_ds = MetricRegistry()
                        cm.compute_and_log(pl_module, reg_ds)
                        ds_prefix = self._with_metric_prefix(f"val/{ds_key}")
                        num_metrics = len(reg_ds.metrics) + len(reg_ds.fractions)
                        print(f"[MixedMetrics] Logging {num_metrics} metrics for dataset '{ds_key}' with prefix '{ds_prefix}'")
                        if num_metrics > 0:
                            sample_names = list(reg_ds.metrics.keys())[:3] + list(reg_ds.fractions.keys())[:3]
                            print(f"[MixedMetrics]   Sample metric names: {sample_names}")
                        reg_ds.log_all(pl_module, self.printer, prefix=ds_prefix, echo=True)
                else:
                    self.conditional_metrics.compute_and_log(pl_module, registry)
        else:
            self.printer.print_info("generation_only=True: skipping metric computation")

        # Skip unconditional generation if not needed, but always register+log conditional metrics first
        if self.generation_only:
            self.printer.print_info("generation_only=True: skipping unconditional generation")
            return
        if self.num_unconditional_generations <= 0:
            self.printer.print_info("Skipping unconditional generation: num_unconditional_generations is 0")
            self._register_conditional_fraction_metrics(registry, trainer, pl_module)
            return

        # Set to evaluation mode
        pl_module.eval()

        # Get CLS token ID
        cls_token_id = SmilesUtils.get_cls_token_id(pl_module.tokenizer)
        if cls_token_id is None:
            self.printer.print_warning("Could not find CLS token, skipping validity check")
            return
        
        self.printer.print_info(f"Starting unconditional generation: {self._effective_unconditional_gens} samples (world_size={self._world_size})")
        
        # Unconditional generation:
        # - Default: run once per epoch (unchanged)
        # - Optional: run once per dataset key (more expensive) when split mode is enabled

        def _tokenize_prompt_text(prompt_text: Optional[str]) -> Optional[list[int]]:
            if not prompt_text:
                return None
            try:
                enc = pl_module.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                ids = enc.get("input_ids", None)
                if isinstance(ids, torch.Tensor):
                    return ids.reshape(-1).tolist()
                if isinstance(ids, list):
                    return ids
            except Exception as e:
                print(f"[Warning] Failed to tokenize unconditional prompt: {e}")
            return None

        if self._split_enabled(trainer) and self.unconditional_per_dataset:
            # Run unconditional generation per dataset key (identified from validation batches)
            try:
                from pairit.data.dataset_identity import dataset_prompt_text as _ds_prompt_text
            except Exception:
                _ds_prompt_text = None

            keys = list(self._dataset_key_to_id.keys()) or list(self._cond_metrics_by_key.keys())
            with self.timing_tracker.time_operation("unconditional/generation"):
                generation_start_time = time.time()
                for ds_key in keys:
                    ds_id = self._dataset_key_to_id.get(ds_key, None)
                    prompt_text = None
                    if _ds_prompt_text is not None and ds_id is not None:
                        prompt_text = _ds_prompt_text(int(ds_id))
                    # Fallback to user-provided unconditional_prompt_text if present
                    if prompt_text is None:
                        prompt_text = getattr(self, "unconditional_prompt_text", None)

                    prompt_token_ids = _tokenize_prompt_text(prompt_text)
                    all_sampled_tokens = MoleculeGenerator.generate_unconditional_batch(
                        pl_module,
                        cls_token_id,
                        self._effective_unconditional_gens,
                        temperature=self.temperature,
                        prompt_token_ids=prompt_token_ids,
                    )

                    um = self._get_unconditional_metrics_for_key(ds_key)

                    # Relevancy tracking for per-dataset unconditional generation (per reference)
                    relevant_first_by_ref: Dict[str, int] = {}
                    relevant_second_by_ref: Dict[str, int] = {}
                    relevancy_details_by_ref_uncond: Dict[str, Dict[str, int]] = {}
                    # Initialize per-reference tracking if relevancy is enabled

                    # Process sequences for this dataset key
                    for i, sampled_tokens in enumerate(all_sampled_tokens):
                        first_valid, second_valid = self._process_unconditional_generation_sample_post_generation(
                            pl_module,
                            sampled_tokens,
                            cls_token_id,
                            trainer.current_epoch,
                            i,
                            metrics_computer=um,
                            dataset_key=ds_key,
                            prompt_text=prompt_text,
                            prompt_dataset_id=ds_id,
                        )

                        first_smiles_string, second_smiles_string = SmilesUtils.tokens_to_smiles(pl_module.tokenizer, sampled_tokens)
                        
                        # Relevancy check for per-dataset unconditional generation (per reference)
                        
                        if second_smiles_string is not None and self._safe_valid(second_smiles_string):
                            um.accumulate_valid_smiles(second_smiles_string)

                        if "memorization" in self.expensive_metrics_to_compute:
                            first_smiles_string, _ = SmilesUtils.tokens_to_smiles(pl_module.tokenizer, sampled_tokens)
                            first_canon = SmilesUtils.canonicalize(first_smiles_string) if first_valid and first_smiles_string else None
                            second_canon = SmilesUtils.canonicalize(second_smiles_string) if second_valid and second_smiles_string else None
                            um.check_memorization(first_canon, second_canon, first_valid, second_valid)

                generation_time = time.time() - generation_start_time
                self.printer.print_timing(f"Batched unconditional generation (per dataset): {generation_time:.3f}s for {len(keys)} datasets")
        else:
            # Optional unconditional prompt (after CLS). This allows per-dataset unconditional metrics.
            prompt_token_ids = _tokenize_prompt_text(getattr(self, "unconditional_prompt_text", None))
        
            # Batch generate all molecules at once (with timing)
            with self.timing_tracker.time_operation("unconditional/generation"):
                generation_start_time = time.time()
                all_sampled_tokens = MoleculeGenerator.generate_unconditional_batch(
                    pl_module,
                    cls_token_id,
                    self._effective_unconditional_gens,
                    temperature=self.temperature,
                    prompt_token_ids=prompt_token_ids,
                )
                generation_time = time.time() - generation_start_time
                self.printer.print_timing(
                    f"Batched unconditional generation: {generation_time:.3f}s for {self._effective_unconditional_gens} samples"
                )
            
            # Process each generated sequence individually for validity checks (with timing)
            with self.timing_tracker.time_operation("unconditional/processing"):
                processing_start_time = time.time()
                # Relevancy tracking for unconditional generation (per reference)
                relevant_first_by_ref_uncond: Dict[str, int] = {}
                relevant_second_by_ref_uncond: Dict[str, int] = {}
                relevancy_details_uncond_by_ref: Dict[str, Dict[str, int]] = {}
                # Initialize per-reference tracking if relevancy is enabled
                
                for i, sampled_tokens in enumerate(all_sampled_tokens):
                    first_valid, second_valid = self._process_unconditional_generation_sample_post_generation(
                        pl_module,
                        sampled_tokens,
                        cls_token_id,
                        trainer.current_epoch,
                        i,
                        prompt_text=getattr(self, "unconditional_prompt_text", None),
                    )
                    
                    # Extract SMILES for relevancy check and dataset metrics
                    first_smiles_string, second_smiles_string = SmilesUtils.tokens_to_smiles(pl_module.tokenizer, sampled_tokens)
                    
                    # Relevancy check for unconditional generation (no seed, so only check rings/chains) - per reference
                    
                    # Collect valid molecules for dataset-level metrics
                    if second_smiles_string is not None and self._safe_valid(second_smiles_string):
                        self.unconditional_metrics.accumulate_valid_smiles(second_smiles_string)
                
                # Memorization check (expensive metric)
                if "memorization" in self.expensive_metrics_to_compute:
                    first_smiles_string, _ = SmilesUtils.tokens_to_smiles(pl_module.tokenizer, sampled_tokens)
                    first_canon = SmilesUtils.canonicalize(first_smiles_string) if first_valid and first_smiles_string else None
                    second_canon = SmilesUtils.canonicalize(second_smiles_string) if second_valid and second_smiles_string else None
                    self.unconditional_metrics.check_memorization(first_canon, second_canon, first_valid, second_valid)

                    if (i + 1) % 10 == 0:
                        total_checked = self.unconditional_metrics.epoch_total_checked
                        first_count = self.unconditional_metrics.epoch_first_valid_count
                        second_count = self.unconditional_metrics.epoch_second_valid_count
                        current_first_accuracy = first_count / total_checked if total_checked > 0 else 0.0
                        current_second_accuracy = second_count / total_checked if total_checked > 0 else 0.0
                        self.printer.print_progress(
                            f"Unconditional generation progress: {i+1}/{self._effective_unconditional_gens} | "
                            f"Validity: mol1={current_first_accuracy:.3f} ({first_count}/{total_checked}), "
                            f"mol2={current_second_accuracy:.3f} ({second_count}/{total_checked})"
                        )

                processing_time = time.time() - processing_start_time
                self.printer.print_timing(
                    f"Post-generation processing: {processing_time:.3f}s for {self._effective_unconditional_gens} samples"
                )

        # === Unconditional metrics ===
        # Default behavior: compute once (unchanged).
        # Optional: split unconditional metrics per dataset key (more expensive).
        with self.timing_tracker.time_operation("unconditional/metrics_computation"):
            if self._split_enabled(trainer) and self.unconditional_per_dataset:
                for ds_key, um in self._uncond_metrics_by_key.items():
                    reg_ds = MetricRegistry()
                    um.compute_and_log(reg_ds)
                    reg_ds.log_all(pl_module, None, prefix=self._with_metric_prefix(f"val/{ds_key}"), echo=False)
            else:
                self.unconditional_metrics.compute_and_log(registry)
        
        # Register relevancy metrics for unconditional generation as fractions (DDP-correct)
        # Always register for DDP sync (0/0 → NaN when no data)
        
        # Register conditional fraction metrics + log everything (already done above
        # when unconditional was skipped; here it includes unconditional metrics too)
        self._register_conditional_fraction_metrics(registry, trainer, pl_module)

        # Flush saved samples for this epoch
        if self.save_generated_samples:
            self._flush_samples('unconditional', trainer.global_step, pl_module, trainer)

        # Synchronize all ranks after rank 0 finishes wandb logging
        # This prevents NCCL timeout when ModelCheckpoint callback runs
        # IMPORTANT: barrier() must be called on ALL ranks, not just rank 0
        if dist.is_initialized():
            try:
                if torch.cuda.is_available() and hasattr(pl_module.device, 'index') and pl_module.device.index is not None:
                    torch.cuda.set_device(pl_module.device.index)
                dist.barrier()
            except Exception as e:
                print(f"[Warning] Barrier failed on rank {rank_zero_only.rank}: {e}. Continuing anyway.")
                import traceback
                traceback.print_exc()
        
        # Set back to training mode
        pl_module.train()

    def evaluate(
        self,
        module: pl.LightningModule,
        dataloader: DataLoader,
        max_evals: int | None = None,
    ) -> dict[str, float]:
        """
        Standalone evaluation that performs both conditional (batch-based)
        and unconditional generation, mirroring on_validation_batch_end and
        on_validation_epoch_end behavior.

        Parameters
        ----------
        module : pl.LightningModule
            Model whose generate/tokenizer APIs are used for sampling.
        dataloader : DataLoader
            DataLoader containing the validation data.
        max_evals : int | None
            Maximum number of evaluations to perform.

        Returns
        -------
        Dict[str, float]
            Dictionary containing conditional and unconditional validity metrics and counts.
            Now includes comprehensive metrics:
            - Basic validity metrics (val_molecule_validity, val_molecule_validity_first, etc.)
            - Simple metrics: validity_fraction, uniqueness_fraction, novelty_fraction
            - Moderate cost metrics: similarity_to_seed_mean, max_similarity_to_gt_*, property_*_delta
            - Expensive metrics (if enabled): snn_mean_unconditional, diversity_unconditional, 
              fragment_cosine_unconditional, feature_w1_macro_unconditional, fcd_unconditional
            - Sample counts: cond_pairs_count, cond_valid_count, uncond_valid_count
        """
        if self.export_prompt_metadata:
            import os
            import torch.distributed as dist
            world_size = max(int(os.environ.get("WORLD_SIZE", "1")),
                             dist.get_world_size() if dist.is_initialized() else 1)
            if world_size != 1:
                raise ValueError("Complete prompt export currently requires single-process evaluation")
        module.eval()
        self._prompt_requests = {}

        # Reset metrics computers for fresh evaluation
        self.conditional_metrics.reset()
        self.unconditional_metrics.reset()

        cls_token_id = SmilesUtils.get_cls_token_id(module.tokenizer)
        if cls_token_id is None:
            return {
                "val_molecule_validity": 0.0,
                "val_molecule_validity_first": 0.0,
                "val_molecule_validity_second": 0.0,
                "num_generated": 0,
                "val_molecule_validity_conditional": 0.0,
                "num_conditional_samples": 0,
            }

        # 1) Conditional generation over validation batches (up to desired samples)
        total_cond_valid = 0
        total_cond_samples = 0
        # Relevancy tracking for conditional generation in evaluate() (accumulated across batches, per reference)
        total_relevant_cond_by_ref: Dict[str, int] = {}
        total_relevancy_details_cond_by_ref: Dict[str, Dict[str, int]] = {}
        # Initialize per-reference tracking if relevancy is enabled
        # Use num_conditional_generations for conditional generation (not num_unconditional_generations)
        num_conditional_samples = int(self.num_conditional_generations)
        if isinstance(max_evals, int):
            num_conditional_samples = min(num_conditional_samples, max_evals)
        
        # Skip conditional generation if num_conditional_generations is 0
        if num_conditional_samples > 0:
            for batch_idx, batch in enumerate(dataloader):
                if total_cond_samples >= num_conditional_samples:
                    break

                count = min(len(batch["input_ids"]), num_conditional_samples - total_cond_samples)
                input_ids = batch["input_ids"][:count]
                attention_mask = batch["attention_mask"][:count]
                batch_size = input_ids.shape[0]
                prompt_rows = self._prompt_batch_metadata(batch, count)
                pair_seeds = ([row["generation_pair_seed"] for row in prompt_rows]
                              if self.pairit_generation_seed is not None else None)

                generated_list = MoleculeGenerator.batch_generate_conditional(
                    module, input_ids, attention_mask, temperature=self.temperature, 
                    num_generations_per_seed=self.num_generations_per_seed, pair_seeds=pair_seeds,
                )

                valid_generations = 0
                # Relevancy tracking for this batch (will be accumulated per reference)
                batch_relevant_by_ref: Dict[str, int] = {}
                batch_relevancy_details_by_ref: Dict[str, Dict[str, int]] = {}
                # Initialize per-reference tracking for this batch
                
                # Note: generated_list has length = batch_size * num_generations_per_seed
                # Results are ordered as: [seed0_gen0, seed0_gen1, ..., seed1_gen0, seed1_gen1, ...]
                for i, gen_seq in enumerate(generated_list):
                    # Map generation index back to seed index
                    seed_idx = i // self.num_generations_per_seed
                    generation_idx = i % self.num_generations_per_seed
                    
                    # Get the input_ids and attention_mask for the seed
                    inp_ids = input_ids[seed_idx]
                    attn_mask = attention_mask[seed_idx]
                    
                    # Note: Function returns (is_valid, flags, is_memorized, canonical_gen)
                    is_valid, flags, is_memorized, canonical_gen = self._process_conditional_generation_sample_post_generation(
                        module, inp_ids, attn_mask, gen_seq, batch_idx, seed_idx, 0.0,
                        neighbors_smiles=None, generation_idx=generation_idx,
                        prompt_metadata=prompt_rows[seed_idx],
                    )
                    if is_valid:
                        valid_generations += 1
                    
                    # Relevancy check for conditional generation in evaluate()
                    
                    # Note: Memorization and property tracking is done internally by ConditionalMetricsComputer
                    # Metrics will be computed via cond_metrics.compute_and_return_dict() below

                total_cond_valid += valid_generations
                # Accumulate relevancy metrics across batches (per reference)
                if batch_relevant_by_ref:
                    for ref_name in batch_relevant_by_ref.keys():
                        if ref_name not in total_relevant_cond_by_ref:
                            total_relevant_cond_by_ref[ref_name] = 0
                            total_relevancy_details_cond_by_ref[ref_name] = {'fragmented': 0, 'large_ring': 0, 'long_chain': 0}
                        total_relevant_cond_by_ref[ref_name] += batch_relevant_by_ref[ref_name]
                        for key in total_relevancy_details_cond_by_ref[ref_name]:
                            total_relevancy_details_cond_by_ref[ref_name][key] += batch_relevancy_details_by_ref[ref_name][key]
                # The input batch was sliced to the remaining seed budget above.
                total_cond_samples += batch_size

            # Flush conditional samples
            if self.save_generated_samples:
                self._flush_samples('conditional', MoleculeValidationCallback.EVALUATE_SPECIAL_STEP_ID, module, None)
        else:
            print(f"Skipping conditional generation: num_conditional_generations is {self.num_conditional_generations}")

        # Compute conditional fraction
        if total_cond_samples > 0:
            cond_fraction = total_cond_valid / (total_cond_samples * self.num_generations_per_seed)
        else:
            cond_fraction = 0.0
        
        # Compute conditional relevancy fractions per reference (if enabled)
        cond_relevance_fractions_by_ref: Dict[str, float] = {}

        # 2) Unconditional generation (use num_unconditional_generations)
        num_unconditional_samples = int(self.num_unconditional_generations)
        if isinstance(max_evals, int):
            num_unconditional_samples = min(num_unconditional_samples, max_evals)
        
        first_valid_molecules = 0
        second_valid_molecules = 0
        total_checked = 0
        # Relevancy tracking for unconditional generation in evaluate() (per reference)
        relevant_first_eval_by_ref: Dict[str, int] = {}
        relevant_second_eval_by_ref: Dict[str, int] = {}
        relevancy_details_uncond_eval_by_ref: Dict[str, Dict[str, int]] = {}
        # Initialize per-reference tracking if relevancy is enabled
        
        # Skip unconditional generation if num_unconditional_generations is 0
        if num_unconditional_samples > 0:
            prompt_token_ids = None
            if getattr(self, "unconditional_prompt_text", None):
                try:
                    enc = module.tokenizer(
                        self.unconditional_prompt_text,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )
                    prompt_token_ids = enc.get("input_ids", None)
                    if isinstance(prompt_token_ids, torch.Tensor):
                        prompt_token_ids = prompt_token_ids.reshape(-1).tolist()
                except Exception as e:
                    print(f"[Warning] Failed to tokenize unconditional_prompt_text: {e}")
                    prompt_token_ids = None

            all_sampled_tokens = MoleculeGenerator.generate_unconditional_batch(
                module,
                cls_token_id=cls_token_id,
                num_samples=num_unconditional_samples,
                temperature=self.temperature,
                prompt_token_ids=prompt_token_ids,
            )

            for i, sampled_tokens in enumerate(all_sampled_tokens):
                first_valid, second_valid = self._process_unconditional_generation_sample_post_generation(
                    module, sampled_tokens=sampled_tokens, cls_token_id=cls_token_id, epoch=MoleculeValidationCallback.EVALUATE_SPECIAL_STEP_ID, sample_idx=i
                )
                total_checked += 1
                if first_valid:
                    first_valid_molecules += 1
                if second_valid:
                    second_valid_molecules += 1

                # Memorization check (expensive metric)
                if "memorization" in self.expensive_metrics_to_compute:
                    first_smiles_string, second_smiles_string = SmilesUtils.tokens_to_smiles(module.tokenizer, sampled_tokens)
                    first_canon = SmilesUtils.canonicalize(first_smiles_string) if first_valid and first_smiles_string else None
                    second_canon = SmilesUtils.canonicalize(second_smiles_string) if second_valid and second_smiles_string else None
                    self.unconditional_metrics.check_memorization(first_canon, second_canon, first_valid, second_valid)

                # Accumulate valid SMILES for comprehensive metrics
                _, second_smiles = SmilesUtils.tokens_to_smiles(module.tokenizer, sampled_tokens)
                if second_valid and second_smiles:
                    canonical = SmilesUtils.canonicalize(second_smiles)
                    if canonical:
                        self.unconditional_metrics.accumulate_valid_smiles(canonical)

            # Flush unconditional samples
            if self.save_generated_samples:
                self._flush_samples('unconditional', MoleculeValidationCallback.EVALUATE_SPECIAL_STEP_ID, module, None)
        else:
            print(f"Skipping unconditional generation: num_unconditional_generations is {self.num_unconditional_generations}")

        overall = float(first_valid_molecules + second_valid_molecules) / (2*total_checked) if total_checked > 0 else 0.0
        first_only = float(first_valid_molecules) / total_checked if total_checked > 0 else 0.0
        second_only = float(second_valid_molecules) / total_checked if total_checked > 0 else 0.0

        metrics = {
            "val_molecule_validity": overall,
            "val_molecule_validity_first": first_only,
            "val_molecule_validity_second": second_only,
            "num_generated": total_checked,
            "val_molecule_validity_conditional": cond_fraction,
            "num_conditional_samples": total_cond_samples,
        }
        
        # Add relevancy metrics if enabled (per reference)
        
        # Note: Memorization metrics in evaluate() mode are computed by ConditionalMetricsComputer
        # and returned via cond_metrics.compute_and_return_dict() below

        # Generation-only mode: skip expensive metric computation, return basic counts
        if self.generation_only:
            return {
                "val_molecule_validity_conditional": cond_fraction,
                "num_conditional_samples": total_cond_samples,
                "num_generated": total_checked,
                "generation_only": True,
            }

        # Compute comprehensive metrics using metrics computers
        print(f"Computing comprehensive metrics for conditional generations\n")
        cond_metrics = self.conditional_metrics.compute_and_return_dict()
        print(f"Computing comprehensive metrics for unconditional generations\n")
        print(f"expensive_metrics_to_compute: {self.unconditional_metrics.expensive_metrics_to_compute}")
        uncond_metrics = self.unconditional_metrics.compute_and_return_dict()

        # Merge with existing basic metrics
        metrics.update(cond_metrics)
        metrics.update(uncond_metrics)

        # Add "val_" prefix to all new metrics for consistency
        metrics = {
            (f"val_{k}" if not k.startswith("val_") else k): v 
            for k, v in metrics.items()
        }

        return metrics

