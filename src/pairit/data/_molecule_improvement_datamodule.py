import os
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache

import lightning.pytorch as pl
import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split

from pairit.tokenization import SmilesTokenizerFast
from pairit.transforms import TokenizerTransform
import pandas as pd
from pair_gen.utils_mol import filter_top_pairs_per_molecule
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from pairit.data.dataset_identity import dataset_prompt_text
from pairit.prompting.property_prompting import (
    PropertyBinningConfig,
    ParsedPropertyPrompt,
    build_property_prompt_text,
    delta_to_bin,
    signed_delta_to_bin,
    value_to_bin,
)
from pairit.prompting.metadata import prompt_metadata, stable_prompt_seed
from pairit.prompting.distance_prompting import (
    DistanceDistributionParams,
    build_distance_prompt_text,
    distance_to_bin_fixed_quartiles,
    percentile_from_beta_cdf,
    percentile_to_bin,
)


def _read_tabular_file(path: str | Path, **kwargs) -> "polars.DataFrame":
    """Read a parquet or CSV file into a polars DataFrame based on extension."""
    import polars
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return polars.read_csv(str(path), **kwargs)
    return polars.read_parquet(str(path), **kwargs)


# Optional torch distributed support for DDP-aware cache behavior
try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - optional
    torch = None
    dist = None

"""
File Format Requirements

The dataset requires two types of parquet files in the root directory:

1. molecules.parquet
   Required columns:
   - smiles (str): SMILES string representation of each molecule
   - split (str): One of "train", "val", or "test"
   - {utility_key} (float): Utility score column matching utility_key parameter
   
   Optional columns:
   - Any additional molecular property columns

2. Pairs files (specified by pair_filename parameter)
   Split-specific naming: pairs_{split}.parquet
   
   Required columns:
   - molecule_a_idx (int): Index of first molecule in molecules.parquet
   - molecule_b_idx (int): Index of second molecule in molecules.parquet 
   - some_distance_metric (e.g. morgan_tanimoto_distance) (float): some distance, e.g. Tanimoto distance between Morgan fingerprints
   
   Optional columns:
   - split (str): One of "train", "val", or "test" 
     Only required if using a single pairs file rather than split-specific files
   - shape_tanimoto_distance (float): 3D shape Tanimoto distance (optional, but required if shape_tanimoto_percentile or shape_tanimoto_num_pairs is provided)

The pairs file(s) should contain filtered molecule pairs based on:
- some distance, e.g. Tanimoto distance but can be anything, since the column name is passed in as distance_type
- optionally, shape_tanimoto_distance (the 3D shape Tanimoto distance)
- Maximum pairs per molecule limits

The dataset will further filter these pairs based on:
- delta: Maximum allowed distance threshold
- epsilon: Minimum required utility improvement
- distance_type: name of distance metric to use for filtering
- shape_tanimoto_percentile/num_pairs: Additional, optional shape-based filtering

For each pair, the second molecule (molecule_b_idx) must have higher utility than 
the first (molecule_a_idx) by at least epsilon to be considered an improvement.
"""



class MoleculeImprovementDataset(Dataset):
    """
    Dataset for training a model to improve molecules based on cached distance pairs.
    
    Loads two tables from a root directory:
    1. molecules.parquet: SMILES, multiple utility scores, and splits
    2. pairs.parquet: item indices, distances, and splits
    
    Filters by distance threshold, utility improvement, and split.
    Ensures the second molecule in each pair is always the better one.
    Training format: <bos>x<sep>x'<eos> where x' is the improved molecule
    """
    
    def __init__(
        self,
        root: str,  # Data root directory containing molecules.parquet and pairs.parquet
        molecule_filename: Union[str, List[str]] = "molecules.parquet", # name(s) of the molecules file(s) within root
        pair_filename: Union[str, List[str]] = "pairs.parquet", # name(s) of the pairs file(s) within root, contains shape and regular tanimoto distances
        ignore_pairs: bool = False,  # If True, ignore pairs and load single molecules from molecules parquet
        seeds_only_conditional: bool = False,  # If True with ignore_pairs, append <sep> for conditional generation
        filter_column: str | None = None,  # Column name to filter molecules (e.g., "Class")
        filter_value: float | None = None,  # Value to filter on (e.g., 0)
        split: Optional[str] = None,  # Optional split filter ("train"|"val"|"test") used only when ignore_pairs=True
        utility_key: str = "utility_0",  # Which utility column to use
        utility_key_fallback: Optional[str] = None,  # Fallback utility column to use when utility_key is None/null/False
        delta: float = None,  # Distance threshold
        epsilon: float = 0.1,  # Utility improvement threshold
        utility_threshold: Optional[float] = None,  # Keep only pairs that CROSS this utility value
        transform_fn: Optional[Callable] = None,  # Tokenizer transform function
        distance_type: Optional[str] = None,  # "morgan_tanimoto_distance" or "shape_tanimoto_distance"
        shape_tanimoto_percentile: Optional[float] = None,  # Keep closest X% of pairs by shape Tanimoto distance per molecule
        shape_tanimoto_num_pairs: Optional[int] = None,  # Keep closest N pairs by shape Tanimoto distance per molecule
        use_cache: bool = True,  # Whether to use cached filtered pairs if available
        duplicate: bool = False,  # Whether to duplicate instead of using the second molecule, for debugging purposes
        just_first: bool = False,  # Whether to just use the first molecule, for debugging purposes
        utility_direction: str = "increasing",  # "increasing" or "decreasing" - whether second molecule should have increasing or decreasing utility
        max_cached_row_groups: int = 10,  # Maximum number of row groups to cache
        cache_dir: Optional[str] = os.environ.get("PAIRIT_CACHE_DIR", "/tmp/pairit_ds_cache"),  # Directory for cache files; overridable via PAIRIT_CACHE_DIR env var
        load_all_in_memory: bool = False,  # For multi-file: load all data into memory at once (high memory, no caching)
        max_num_occurrences: Optional[int] = None,  # Limit occurrences of identical first SMILES in final pairs (e.g., 1 enforces uniqueness)
        max_num_occurrences_criterion: Optional[str] = "first",  # Criterion for selecting pairs: "first" (take first N), "min_distance" (select N with minimum distance_type), "min_shape_tanimoto" (select N with minimum shape_tanimoto_distance)
        dataset_id: int = 0,  # Dataset identity integer used for prompting
        use_dataset_identity_prompt: bool = False,  # If True, prefix sequences with dataset identity tokens
        prompting: Optional[dict] = None,  # Optional property prompting config (see hydra plan)
        # Subsetting controls (applied while building training_pairs; memory-efficient vs torch Subset)
        subset_fraction: Optional[float] = None,  # If set (<1), randomly sample this fraction of valid pairs
        subset_max_samples: Optional[int] = None,  # If set, randomly sample at most this many valid pairs
        subset_seed: int = 0,  # Seed for deterministic random sampling
        # Back-compat alias (previous debug cap). Prefer subset_max_samples.
        max_pairs_to_load: Optional[int] = None,
        keep_original_ordering: bool = False,  # If True, preserve the ordering of pairs from the parquet file, overriding epsilon-based ordering or other behavior
        utility_negate: bool = False,  # If True, negate utility values after reading from column. Use when the column scale is inverted relative to the desired improvement direction (e.g., ZINC 'sa' column is ~0-1 higher=better, but RDKit SA is ~1-10 lower=better).
        is_training: bool = True,  # Prompt dropout/bounded-delta augmentation applies only to training datasets.
    ):
        """
        Initialize the dataset.

        Args:
            root: Root directory containing molecules.parquet and pairs.parquet
            molecule_filename: Name(s) of molecules file(s) - can be string or list of strings
            pair_filename: Name(s) of pairs file(s) - can be string or list of strings
            utility_key: Which utility column to use (e.g., "utility_0", "gap", "homo", etc.)
            utility_key_fallback: Optional fallback utility column to use when utility_key is None/null/False
            delta: Maximum distance threshold for molecule pairs
            epsilon: Minimum utility improvement threshold
            transform_fn: Tokenizer transform function (will create default if None)
            use_cache: Whether to use cached filtered pairs if available
            utility_direction: "increasing" or "decreasing" - whether second molecule should have increasing or decreasing utility
            max_cached_row_groups: Maximum number of row groups to cache for multi-file scenarios
            cache_dir: Directory for cache files (default: /tmp/lobster_cache). Set to writable location if root is read-only.
            load_all_in_memory: For multi-file mode, load all data into memory at once (high memory, for testing/comparison)
            max_num_occurrences: If provided, only include up to this many pairs for any given first SMILES string after ordering (1 enforces uniqueness)
            max_num_occurrences_criterion: Criterion for selecting pairs when max_num_occurrences is set:
                - "first": Take the first N pairs (default, current behavior)
                - "min_distance": Select N pairs with minimum distance (uses distance_type)
                - "min_shape_tanimoto": Select N pairs with minimum shape_tanimoto_distance
            keep_original_ordering: If True, preserve the ordering of pairs from the parquet file, overriding epsilon-based ordering or other behavior
            utility_negate: If True, negate utility values after reading from column. Use when
                the parquet column scale is inverted relative to the desired improvement direction
                (e.g., ZINC 'sa' column is ~0-1 higher=better, but RDKit SA is ~1-10 lower=better).
        """
        self.root = Path(root)
        self.utility_key = utility_key
        self.utility_negate = utility_negate
        self.utility_key_fallback = utility_key_fallback
        self.delta = delta
        self.epsilon = epsilon
        self.utility_threshold = None if utility_threshold in (None, 'None') else float(utility_threshold)
        self.distance_type = distance_type
        self.shape_tanimoto_percentile = shape_tanimoto_percentile
        self.shape_tanimoto_num_pairs = shape_tanimoto_num_pairs
        self.use_cache = use_cache
        self.duplicate = duplicate
        self.just_first = just_first
        self.max_cached_row_groups = max_cached_row_groups
        self.load_all_in_memory = load_all_in_memory
        self.max_num_occurrences = max_num_occurrences
        self.max_num_occurrences_criterion = max_num_occurrences_criterion if max_num_occurrences_criterion is not None else "first"
        self.dataset_id = int(dataset_id)
        self.use_dataset_identity_prompt = bool(use_dataset_identity_prompt)
        self._is_training = bool(is_training)
        
        # Set ignore_pairs and split early so they're available for distance prompting validation
        self.ignore_pairs = ignore_pairs
        self.seeds_only_conditional = seeds_only_conditional
        self.filter_column = filter_column
        self.filter_value = filter_value
        self.split = split
        
        # Handle both single files and lists of files early so they're available for distance prompting validation
        if isinstance(molecule_filename, str):
            self.molecule_filenames = [molecule_filename]
            self.use_multi_file = False
        else:
            self.molecule_filenames = molecule_filename
            self.use_multi_file = True
            
        if isinstance(pair_filename, str):
            self.pair_filenames = [pair_filename]
        else:
            self.pair_filenames = pair_filename
        
        # Initialize file_paths early and set up paths if not ignoring pairs
        # (needed for distance prompting validation)
        self.file_paths = []
        if not self.ignore_pairs:
            # Set up file paths early for distance prompting validation
            for mol_file, pair_file in zip(self.molecule_filenames, self.pair_filenames):
                mol_path = self.root / mol_file
                pair_path = self.root / pair_file
                
                # Check if files exist
                if not mol_path.exists():
                    raise FileNotFoundError(f"Molecules file not found: {mol_path}")
                if not pair_path.exists():
                    raise FileNotFoundError(f"Pairs file not found: {pair_path}")
                
                self.file_paths.append((mol_path, pair_path))
            
            # For backward compatibility, set single file paths
            if not self.use_multi_file:
                self.molecules_parquet_path = self.file_paths[0][0]
                self.pairs_parquet_path = self.file_paths[0][1]

        # ---------------- property prompting (optional) ----------------
        self._prompting = prompting or {}
        self._prop_prompt_enabled = False
        self._prop_names: list[str] = []
        self._prop_selection_mode: str = "random_one"
        self._prop_selection_k: int = 1
        self._prop_selection_seed: int = 0
        self._prop_include_type: bool = True
        self._prop_include_start_value: str = "bin"  # off|bin|literal (literal not implemented)
        self._prop_include_direction: bool = True
        self._prop_include_magnitude: str = "bin"  # off|bin|literal (literal not implemented)
        self._prop_randomize_tokens: bool = False  # If True, replace property tokens with random values (ablation)
        self._prop_export_metadata = False
        self._prop_prefer_columns: bool = True
        self._prop_compute_if_missing: bool = True
        self._prop_column_map: dict[str, str] = {}
        self._value_binning_default: PropertyBinningConfig | None = None
        self._delta_binning_default: PropertyBinningConfig | None = None
        self._value_binning_per_prop: dict[str, PropertyBinningConfig] = {}
        self._delta_binning_per_prop: dict[str, PropertyBinningConfig] = {}
        self._value_width: int = 2
        self._delta_width: int = 2
        self._prop_cache: dict[str, dict[str, float | None]] = {}
        self._prop_from_columns: dict[str, dict[str, float | None]] | None = None
        # Property dropout: drop entire properties from prompt (training-time augmentation)
        self._prop_property_dropout_enabled: bool = False
        self._prop_property_dropout_rate: float = 0.0
        # Component dropout and bounded delta (training-time augmentation)
        self._prop_component_dropout_enabled: bool = False
        self._prop_component_dropout_rate: float = 0.0
        self._prop_component_dropout_epoch_vary: bool = True
        self._prop_bounded_delta_enabled: bool = False
        self._prop_bounded_delta_fraction: float = 0.0
        self._prop_bounded_delta_mode: str = "both"
        self._prop_selection_explicit_names: list[str] = []
        # Spawned persistent workers retain their Dataset copy. A shared scalar
        # makes parent epoch updates visible without restarting the workers.
        self._epoch_state = torch.zeros((), dtype=torch.long).share_memory_()

        try:
            prop_cfg = (self._prompting or {}).get("properties", None) or {}
            self._prop_prompt_enabled = bool(prop_cfg.get("enabled", False))
            if self._prop_prompt_enabled:
                self._prop_names = [str(x) for x in (prop_cfg.get("names", None) or [])]
                if not self._prop_names:
                    raise ValueError("prompting.properties.enabled=true but prompting.properties.names is empty")

                sel = prop_cfg.get("selection", None) or {}
                self._prop_selection_mode = str(sel.get("mode", "random_one"))
                self._prop_selection_k = int(sel.get("k", 1))
                self._prop_selection_seed = int(sel.get("seed", 0))

                inc = prop_cfg.get("include", None) or {}
                self._prop_include_type = bool(inc.get("type", True))
                self._prop_include_start_value = str(inc.get("start_value", "bin"))
                self._prop_include_direction = bool(inc.get("direction", True))
                self._prop_include_magnitude = str(inc.get("magnitude", "bin"))
                
                # Randomization ablation: replace property tokens with random values
                self._prop_randomize_tokens = bool(prop_cfg.get("randomize_tokens", False))
                self._prop_export_metadata = bool(prop_cfg.get("export_metadata", False))

                # Property dropout config (drop entire properties from prompt)
                prop_dropout_cfg = inc.get("property_dropout", None) or {}
                self._prop_property_dropout_enabled = bool(prop_dropout_cfg.get("enabled", False))
                self._prop_property_dropout_rate = float(prop_dropout_cfg.get("rate", 0.0))

                # Component dropout config
                dropout_cfg = inc.get("component_dropout", None) or {}
                self._prop_component_dropout_enabled = bool(dropout_cfg.get("enabled", False))
                self._prop_component_dropout_rate = float(dropout_cfg.get("rate", 0.0))
                self._prop_component_dropout_epoch_vary = bool(dropout_cfg.get("epoch_vary", True))

                # Bounded delta config
                bounded_cfg = inc.get("bounded_delta", None) or {}
                self._prop_bounded_delta_enabled = bool(bounded_cfg.get("enabled", False))
                self._prop_bounded_delta_fraction = float(bounded_cfg.get("fraction", 0.0))
                self._prop_bounded_delta_mode = str(bounded_cfg.get("mode", "both"))

                # Eval override config
                evo = prop_cfg.get("eval_override", None) or {}
                self._eval_override_enabled = bool(evo.get("enabled", False))
                _evo_start = evo.get("start_bin", None)
                self._eval_override_start_bin = int(_evo_start) if _evo_start is not None else None
                _evo_dir = evo.get("direction", None)
                self._eval_override_direction = str(_evo_dir) if _evo_dir is not None else None
                _evo_delta = evo.get("delta_bin", None)
                self._eval_override_delta_bin = int(_evo_delta) if _evo_delta is not None else None
                _evo_signed = evo.get("signed_delta_bin", None)
                self._eval_override_signed_delta_bin = int(_evo_signed) if _evo_signed is not None else None
                self._eval_override_delta_mode = str(evo.get("delta_mode", "exact"))
                self._eval_override_inc_start = bool(evo.get("include_start_value", True))
                self._eval_override_inc_dir = bool(evo.get("include_direction", True))
                self._eval_override_inc_mag = bool(evo.get("include_magnitude", True))

                # Explicit selection names
                self._prop_selection_explicit_names = [
                    str(x) for x in (sel.get("explicit_names", None) or [])
                ]

                src = prop_cfg.get("source", None) or {}
                self._prop_prefer_columns = bool(src.get("prefer_columns", True))
                self._prop_compute_if_missing = bool(src.get("compute_if_missing", True))
                self._prop_column_map = {str(k): str(v) for k, v in (src.get("column_map", None) or {}).items()
                                         if v is not None and str(v)}
                self._prop_fallback_column_map = {str(k): str(v) for k, v in (src.get("fallback_column_map", None) or {}).items()}

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

                def _mk_binning(d: dict, *, kind: str) -> PropertyBinningConfig:
                    mode = str(d.get("mode", "fixed"))
                    default = d.get("default", None) or {}
                    num_bins = int(default.get("num_bins", 32))
                    rng_t = _as_range_tuple(default.get("range", None))
                    return PropertyBinningConfig(num_bins=num_bins, range=rng_t, mode=mode, clip=True)

                self._value_binning_default = _mk_binning(vb, kind="value")
                self._delta_binning_default = _mk_binning(db, kind="delta")

                for pname, spec in (vb.get("per_property", None) or {}).items():
                    s = spec or {}
                    rng_t = _as_range_tuple(s.get("range", None))
                    self._value_binning_per_prop[str(pname)] = PropertyBinningConfig(
                        num_bins=int(s.get("num_bins", self._value_binning_default.num_bins if self._value_binning_default else 32)),
                        range=rng_t,
                        mode=str(s.get("mode", self._value_binning_default.mode if self._value_binning_default else "fixed")),
                        clip=True,
                    )
                for pname, spec in (db.get("per_property", None) or {}).items():
                    s = spec or {}
                    rng_t = _as_range_tuple(s.get("range", None))
                    self._delta_binning_per_prop[str(pname)] = PropertyBinningConfig(
                        num_bins=int(s.get("num_bins", self._delta_binning_default.num_bins if self._delta_binning_default else 16)),
                        range=rng_t,
                        mode=str(s.get("mode", self._delta_binning_default.mode if self._delta_binning_default else "fixed")),
                        clip=True,
                    )

                # Width must match tokenizer token format (SmilesTokenizer2Fast uses width=max(2, len(str(num_bins-1))))
                self._value_width = max(2, len(str((self._value_binning_default.num_bins - 1) if self._value_binning_default else 31)))
                self._delta_width = max(2, len(str((self._delta_binning_default.num_bins - 1) if self._delta_binning_default else 15)))
        except Exception as e:
            raise

        # ---------------- distance prompting (optional) ----------------
        self._distance_prompt_enabled = False
        self._distance_distributions: dict[int, DistanceDistributionParams] = {}
        self._distance_binning_mode: str = "curve_fit"
        self._distance_num_bins: int = 20
        self._distance_column: str = "morgan_tanimoto_distance"
        self._distance_is_similarity: bool = False
        self._distance_distribution_file: str = "distance_distributions.parquet"
        self._distance_compute_if_missing: bool = True
        self._distance_min_pairs_per_molecule: int = 10
        self._distance_max_fraction_low_pairs: float = 0.1
        self._distance_bin_width: int = 2
        self._distance_eval_override_enabled: bool = False
        self._distance_eval_override_bin: Optional[int] = None
        self._molecule_idx_cache: dict[str, int] = {}
        self._pair_distance_cache: dict[tuple[int, int], float] = {}

        try:
            dist_cfg = (self._prompting or {}).get("distance", None) or {}
            self._distance_prompt_enabled = bool(dist_cfg.get("enabled", False))
            if self._distance_prompt_enabled:
                binning_cfg = dist_cfg.get("binning", None) or {}
                self._distance_binning_mode = str(binning_cfg.get("mode", "curve_fit"))
                self._distance_num_bins = int(binning_cfg.get("num_bins", 20))

                source_cfg = dist_cfg.get("source", None) or {}
                self._distance_distribution_file = str(source_cfg.get("distribution_file", "distance_distributions.parquet"))
                self._distance_column = str(source_cfg.get("distance_column", "morgan_tanimoto_distance"))
                self._distance_is_similarity = bool(source_cfg.get("is_similarity", False))
                self._distance_min_pairs_per_molecule = int(source_cfg.get("min_pairs_per_molecule", 10))
                self._distance_max_fraction_low_pairs = float(source_cfg.get("max_fraction_low_pairs", 0.1))
                self._distance_compute_if_missing = bool(source_cfg.get("compute_if_missing", True))

                # Width must match tokenizer token format
                self._distance_bin_width = max(2, len(str(self._distance_num_bins - 1)))

                # Eval override: inject a fixed target distance bin at inference time,
                # independent of any ground-truth pair. This is what enables controllable
                # similarity-conditioned generation (analogous to the property eval_override).
                dist_evo = dist_cfg.get("eval_override", None) or {}
                self._distance_eval_override_enabled = bool(dist_evo.get("enabled", False))
                _dist_evo_bin = dist_evo.get("bin", None)
                self._distance_eval_override_bin = int(_dist_evo_bin) if _dist_evo_bin is not None else None

                # Load or compute distance distributions. Only needed when deriving bins
                # from real pairs; the eval override sets the bin directly, so we skip the
                # (potentially expensive) per-molecule Beta fitting in that case.
                if not self._distance_eval_override_enabled:
                    self._load_or_compute_distance_distributions()
        except Exception as e:
            raise

        # Normalize subset args
        self.subset_fraction = None
        if subset_fraction not in (None, "None"):
            self.subset_fraction = float(subset_fraction)

        self.subset_max_samples = None
        if subset_max_samples not in (None, "None"):
            self.subset_max_samples = int(subset_max_samples)
        elif max_pairs_to_load not in (None, "None"):
            self.subset_max_samples = int(max_pairs_to_load)

        self.subset_seed = int(subset_seed)
        self._subset_applied = (self.subset_fraction is not None) or (self.subset_max_samples is not None)

        # If we're subsetting, "load_all_in_memory" defeats the purpose (huge RAM hit).
        if self._subset_applied and self.load_all_in_memory:
            print("Warning: subset_* is set, disabling load_all_in_memory for memory efficiency.")
            self.load_all_in_memory = False
        
        # Setup cache directory
        if cache_dir is None:
            # Default to /tmp/lobster_cache
            import tempfile
            self.cache_dir = Path(tempfile.gettempdir()) / "lobster_cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        # Create a unique subfolder based on dataset characteristics
        # This prevents cache collisions between different datasets
        cache_id_parts = [
            ("root", str(Path(root).resolve())),
            ("utility", utility_key or "none"),
            ("utility_fallback", utility_key_fallback or "none"),
        ]
        # Add file names to cache ID for multi-file scenarios
        if isinstance(molecule_filename, list):
            cache_id_parts.append(("mol_files", "_".join(sorted(molecule_filename))))
        else:
            cache_id_parts.append(("mol_file", molecule_filename))
        if isinstance(pair_filename, list):
            cache_id_parts.append(("pair_files", "_".join(sorted(pair_filename))))
        else:
            cache_id_parts.append(("pair_file", pair_filename))
        
        cache_id_str = "|".join(f"{k}={v}" for k, v in sorted(cache_id_parts))
        cache_id_hash = hashlib.sha1(cache_id_str.encode("utf-8")).hexdigest()[:12]
        self.dataset_cache_dir = self.cache_dir / f"dataset_{cache_id_hash}"
        self.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we have the same number of molecule and pair files
        if len(self.molecule_filenames) != len(self.pair_filenames):
            raise ValueError(f"Number of molecule files ({len(self.molecule_filenames)}) must match number of pair files ({len(self.pair_filenames)})")
        # Validate utility_direction parameter with graceful default behavior
        if utility_direction is None or utility_direction == "None":
            # Graceful default: use "increasing" if not specified
            self.utility_direction = "increasing"
            print("Warning: utility_direction not specified, defaulting to 'increasing'")
        elif utility_direction not in ["increasing", "decreasing"]:
            # Check for old parameter names for backward compatibility
            if utility_direction in ["higher", "lower"]:
                # Convert old names to new names
                old_to_new = {"higher": "increasing", "lower": "decreasing"}
                self.utility_direction = old_to_new[utility_direction]
                print(f"Warning: '{utility_direction}' is deprecated, using '{self.utility_direction}' instead")
            else:
                raise ValueError(f"utility_direction must be 'increasing' or 'decreasing', got '{utility_direction}'")
        else:
            self.utility_direction = utility_direction
        
        self.keep_original_ordering = keep_original_ordering
        
        # Set up file paths for ignore_pairs case (non-ignore_pairs case was handled earlier)
        if self.ignore_pairs:
            # Only validate molecule files
            self.molecule_paths = []
            for mol_file in self.molecule_filenames:
                mol_path = self.root / mol_file
                if not mol_path.exists():
                    raise FileNotFoundError(f"Molecules file not found: {mol_path}")
                self.molecule_paths.append(mol_path)
            # For backward compatibility, set single file path
            if not self.use_multi_file:
                self.molecules_parquet_path = self.molecule_paths[0]
        
        # Initialize transform
        if transform_fn is None:
            # maybe get rid of this? shouldn't be called
            # Create default SMILES tokenizer transform
            from pairit.tokenization import SmilesTokenizerFast
            from pairit.transforms import TokenizerTransform
            
            tokenizer = SmilesTokenizerFast()
            self.transform = TokenizerTransform(
                tokenizer,
                padding="max_length",
                truncation=True,
                max_length=512,  # Default max length
            )
        else:
            self.transform = transform_fn
        
        if self.ignore_pairs:
            # Load SMILES list from molecules file(s)
            self._smiles_list = self._load_smiles_from_molecules()
        else:
            # Load and filter training pairs
            if self.use_multi_file:
                # For multi-file scenarios, use row group caching
                if self._subset_applied:
                    print(f"[WARNING] subset_fraction/subset_max_samples is set but dataset uses multi-file mode.")
                    print(f"[WARNING]   Subsetting is NOT applied for multi-file datasets. All pairs will be used.")
                    print(f"[WARNING]   To use subsetting, use single-file mode or implement multi-file subsetting support.")
                self._setup_multi_file_data()
            else:
                # For single file, use existing logic
                self.training_pairs = self._load_and_filter_pairs()

    # -------------------------
    # Utility fallback helper
    # -------------------------
    def _apply_utility_fallback(self, molecules_df):
        """
        Apply utility_key_fallback to fill null/None/False values in utility_key column.
        
        Args:
            molecules_df: polars DataFrame with molecule data
            
        Returns:
            Modified molecules_df with fallback values applied
        """
        import polars
        
        # Only apply if utility_key_fallback is provided and utility_key exists
        if self.utility_key_fallback is None or self.utility_key_fallback == 'None':
            return molecules_df
        
        if self.utility_key is None or self.utility_key == 'None':
            return molecules_df
        
        # Check if utility_key column exists
        if self.utility_key not in molecules_df.columns:
            return molecules_df
        
        # Check if utility_key_fallback column exists
        if self.utility_key_fallback not in molecules_df.columns:
            print(f"Warning: utility_key_fallback column '{self.utility_key_fallback}' not found. Available columns: {molecules_df.columns}")
            return molecules_df
        
        # Count null values before applying fallback
        null_count_before = molecules_df[self.utility_key].null_count()
        
        # Apply fallback: fill null values with values from utility_key_fallback
        # Also handle False values if the column is boolean
        molecules_df = molecules_df.with_columns(
            polars.when(polars.col(self.utility_key).is_null())
            .then(polars.col(self.utility_key_fallback))
            .otherwise(polars.col(self.utility_key))
            .alias(self.utility_key)
        )
        
        # Also handle False values (if column type allows)
        # Check if there are any False values that should be replaced
        if molecules_df[self.utility_key].dtype == polars.Boolean:
            molecules_df = molecules_df.with_columns(
                polars.when(polars.col(self.utility_key) == False)
                .then(polars.col(self.utility_key_fallback))
                .otherwise(polars.col(self.utility_key))
                .alias(self.utility_key)
            )
        
        null_count_after = molecules_df[self.utility_key].null_count()
        if null_count_before > 0:
            filled_count = null_count_before - null_count_after
            print(f"Applied utility_key_fallback '{self.utility_key_fallback}': filled {filled_count} null values in '{self.utility_key}' column")
        
        return molecules_df
    
    # -------------------------
    # DDP helpers
    # -------------------------
    @staticmethod
    def _dist_is_initialized() -> bool:
        try:
            return (dist is not None) and dist.is_available() and dist.is_initialized()
        except Exception:
            return False

    @staticmethod
    def _rank() -> int:
        if MoleculeImprovementDataset._dist_is_initialized():
            try:
                return dist.get_rank()
            except Exception:
                return 0
        return 0

    @staticmethod
    def _is_rank0() -> bool:
        return MoleculeImprovementDataset._rank() == 0

    @staticmethod
    def _barrier() -> None:
        if MoleculeImprovementDataset._dist_is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
    
    def _setup_multi_file_data(self):
        """Setup data for multi-file scenarios with row group caching."""
        # Build file metadata for efficient access
        self.file_metadata = []
        self._total_pairs = 0
        
        for file_idx, (mol_path, pair_path) in enumerate(self.file_paths):
            # Get metadata from parquet files
            mol_parquet = pq.ParquetFile(mol_path)
            pair_parquet = pq.ParquetFile(pair_path)
            
            file_info = {
                'file_idx': file_idx,
                'mol_path': mol_path,
                'pair_path': pair_path,
                'mol_parquet': mol_parquet,
                'pair_parquet': pair_parquet,
                'num_molecules': mol_parquet.metadata.num_rows,
                'num_pairs': pair_parquet.metadata.num_rows,
                'num_row_groups': pair_parquet.num_row_groups,
                'start_idx': self._total_pairs
            }
            
            self.file_metadata.append(file_info)
            self._total_pairs += file_info['num_pairs']
        
        # If load_all_in_memory, preload all data
        if self.load_all_in_memory:
            print(f"Loading all {len(self.file_paths)} files into memory (load_all_in_memory=True)...")
            total_pairs_counted = 0
            for file_info in self.file_metadata:
                # Load entire molecule and pair tables into memory
                mol_table = file_info['mol_parquet'].read()
                file_info['pair_table_cached'] = file_info['pair_parquet'].read()
                total_pairs_counted += len(file_info['pair_table_cached'])
                
                # Apply utility_key_fallback if provided (convert to polars, apply, convert back)
                if self.utility_key_fallback is not None and self.utility_key_fallback != 'None':
                    import polars
                    mol_df = polars.from_arrow(mol_table)
                    mol_df = self._apply_utility_fallback(mol_df)
                    mol_table = mol_df.to_arrow()
                
                file_info['mol_table_cached'] = mol_table
            # Print summary of molecules and pairs per file
            print("\nMolecules and pairs per file:")
            for file_info in self.file_metadata:
                # Calculate table sizes in bytes
                mol_size = file_info['mol_table_cached'].nbytes
                pair_size = file_info['pair_table_cached'].nbytes
                
                # Convert to MB or GB as appropriate
                def format_size(bytes):
                    if bytes >= 1024**3:
                        return f"{bytes/1024**3:.2f} GB"
                    else:
                        return f"{bytes/1024**2:.2f} MB"
                
                print(f"File {file_info['file_idx']}:")
                print(f"  Molecules file: {file_info['mol_path'].name}")
                print(f"    Number of molecules: {file_info['num_molecules']:,}")
                print(f"    Table size: {format_size(mol_size)}")
                print(f"  Pairs file: {file_info['pair_path'].name}")
                print(f"    Number of pairs: {file_info['num_pairs']:,}")
                print(f"    Table size: {format_size(pair_size)}")
                print(f"    Number of row groups: {file_info['num_row_groups']}")
                print()
            print(f"Loaded {self._total_pairs} total pairs into memory (counted {total_pairs_counted} pairs)")
            
            print(f"max_num_occurrences: {self.max_num_occurrences}")
            # Apply max_num_occurrences filtering if specified
            if self.max_num_occurrences is not None and self.max_num_occurrences != 'None':
                print(f"\nApplying max_num_occurrences filtering (<= {self.max_num_occurrences} per first SMILES)...")
                self._apply_max_occurrences_filtering()
        else:
            # Setup row group caching - lazy initialization to avoid fork/pickle issues
            # The actual lru_cache wrapper is created on first access via _get_load_row_group()
            self._load_row_group_cache = None
    
    def _apply_max_occurrences_filtering(self):
        """
        Apply max_num_occurrences filtering to in-memory multi-file pairs.
        Builds an index mapping that tracks which pairs to keep.
        
        This method filters directly at the index level to avoid issues with
        duplicate pairs being incorrectly matched when using set-based lookups.
        """
        # Log criterion with distance key info for "min_distance"
        if self.max_num_occurrences_criterion == "min_distance":
            dist_key = self.distance_type if self.distance_type else 'morgan_tanimoto_distance'
            print(f"Applying max_num_occurrences filtering with criterion: {self.max_num_occurrences_criterion} (using distance key: '{dist_key}')...")
        else:
            print(f"Applying max_num_occurrences filtering with criterion: {self.max_num_occurrences_criterion}...")
        
        # First, collect all pairs with their metadata
        pairs_with_metadata = []
        
        for file_info in self.file_metadata:
            mol_table = file_info['mol_table_cached']
            pair_table = file_info['pair_table_cached']
            file_idx = file_info['file_idx']
            
            for local_idx in range(len(pair_table)):
                mol_a_idx = pair_table['molecule_a_idx'][local_idx].as_py()
                mol_b_idx = pair_table['molecule_b_idx'][local_idx].as_py()
                smiles_a = mol_table['smiles'][mol_a_idx].as_py()
                smiles_b = mol_table['smiles'][mol_b_idx].as_py()
                
                # Extract distance information
                distance_info = {}
                if self.distance_type and self.distance_type in pair_table.column_names:
                    try:
                        distance_info[self.distance_type] = pair_table[self.distance_type][local_idx].as_py()
                    except (KeyError, IndexError):
                        pass
                if 'morgan_tanimoto_distance' in pair_table.column_names:
                    try:
                        distance_info['morgan_tanimoto_distance'] = pair_table['morgan_tanimoto_distance'][local_idx].as_py()
                    except (KeyError, IndexError):
                        pass
                if 'shape_tanimoto_distance' in pair_table.column_names:
                    try:
                        distance_info['shape_tanimoto_distance'] = pair_table['shape_tanimoto_distance'][local_idx].as_py()
                    except (KeyError, IndexError):
                        pass
                
                # Handle null utility_key case for deterministic ordering
                if self.utility_key is None or self.utility_key == 'None':
                    # Use same hash logic as in __getitem__
                    if hash((smiles_a, smiles_b)) % 2 == 0:
                        first_smiles = smiles_a
                    else:
                        first_smiles = smiles_b
                else:
                    # Get utilities to determine which is first (worse molecule)
                    utility_a = mol_table[self.utility_key][mol_a_idx].as_py()
                    utility_b = mol_table[self.utility_key][mol_b_idx].as_py()
                    if self.utility_negate:
                        utility_a, utility_b = -utility_a, -utility_b

                    # Determine first molecule based on utility direction
                    if self.utility_direction == "increasing":
                        first_smiles = smiles_a if utility_b > utility_a else smiles_b
                    else:  # decreasing
                        first_smiles = smiles_a if utility_b < utility_a else smiles_b
                
                pairs_with_metadata.append({
                    'first_smiles': first_smiles,
                    'file_idx': file_idx,
                    'local_idx': local_idx,
                    'distance_info': distance_info
                })
        
        # Group pairs by first SMILES
        pairs_by_first_smiles = {}
        for pair_meta in pairs_with_metadata:
            first_smiles = pair_meta['first_smiles']
            if first_smiles not in pairs_by_first_smiles:
                pairs_by_first_smiles[first_smiles] = []
            pairs_by_first_smiles[first_smiles].append(pair_meta)
        
        # Select pairs based on criterion
        valid_indices = []
        for first_smiles, pairs in pairs_by_first_smiles.items():
            if len(pairs) <= self.max_num_occurrences:
                # Keep all pairs if we have fewer than the limit
                for pair_meta in pairs:
                    valid_indices.append((pair_meta['file_idx'], pair_meta['local_idx']))
            else:
                # Apply criterion-based selection
                if self.max_num_occurrences_criterion == "first":
                    # Take first N pairs (original behavior)
                    for pair_meta in pairs[:self.max_num_occurrences]:
                        valid_indices.append((pair_meta['file_idx'], pair_meta['local_idx']))
                elif self.max_num_occurrences_criterion == "min_distance":
                    # Select N pairs with minimum distance (using distance_type, or morgan_tanimoto_distance as default)
                    dist_key = self.distance_type if self.distance_type else 'morgan_tanimoto_distance'
                    if dist_key in pairs[0]['distance_info']:
                        # Sort by distance (ascending)
                        sorted_pairs = sorted(pairs, key=lambda p: p['distance_info'].get(dist_key, float('inf')))
                        for pair_meta in sorted_pairs[:self.max_num_occurrences]:
                            valid_indices.append((pair_meta['file_idx'], pair_meta['local_idx']))
                    else:
                        # Fallback to first if distance not available
                        print(f"Warning: distance key '{dist_key}' not available in pair data, falling back to 'first' criterion")
                        for pair_meta in pairs[:self.max_num_occurrences]:
                            valid_indices.append((pair_meta['file_idx'], pair_meta['local_idx']))
                elif self.max_num_occurrences_criterion == "min_shape_tanimoto":
                    # Select N pairs with minimum shape_tanimoto_distance
                    if 'shape_tanimoto_distance' in pairs[0]['distance_info']:
                        # Sort by shape_tanimoto_distance (ascending)
                        sorted_pairs = sorted(pairs, key=lambda p: p['distance_info'].get('shape_tanimoto_distance', float('inf')))
                        for pair_meta in sorted_pairs[:self.max_num_occurrences]:
                            valid_indices.append((pair_meta['file_idx'], pair_meta['local_idx']))
                    else:
                        # Fallback to first if shape_tanimoto_distance not available
                        print(f"Warning: shape_tanimoto_distance not available, falling back to 'first' criterion")
                        for pair_meta in pairs[:self.max_num_occurrences]:
                            valid_indices.append((pair_meta['file_idx'], pair_meta['local_idx']))
                else:
                    raise ValueError(f"Unknown max_num_occurrences_criterion: {self.max_num_occurrences_criterion}. "
                                   f"Must be one of: 'first', 'min_distance', 'min_shape_tanimoto'")
        
        # Store the filtered indices mapping
        self._filtered_indices = valid_indices
        original_count = self._total_pairs
        self._total_pairs = len(valid_indices)
        
        print(f"After max_num_occurrences filtering (<= {self.max_num_occurrences} per first SMILES, criterion={self.max_num_occurrences_criterion}): {self._total_pairs} pairs (from {original_count}, kept {100*self._total_pairs/original_count:.1f}%)")
        print(f"Number of unique first SMILES: {len(pairs_by_first_smiles)}")
    
    def _load_row_group_impl(self, file_idx: int, row_group_idx: int):
        """Load specific row group from file."""
        file_info = self.file_metadata[file_idx]
        pair_parquet = file_info['pair_parquet']
        
        # Load the specific row group
        row_group = pair_parquet.read_row_group(row_group_idx)
        return row_group
    
    def _load_row_group(self, file_idx: int, row_group_idx: int):
        """Load row group with lazy lru_cache initialization.
        
        The lru_cache is created lazily on first call to avoid fork/pickle issues
        with DataLoader workers. Each worker process will create its own fresh cache.
        """
        if self._load_row_group_cache is None:
            self._load_row_group_cache = lru_cache(maxsize=self.max_cached_row_groups)(self._load_row_group_impl)
        return self._load_row_group_cache(file_idx, row_group_idx)
    
    def _get_file_and_local_index(self, global_idx: int):
        """Convert global index to file index and local index within that file."""
        for file_info in self.file_metadata:
            if global_idx < file_info['start_idx'] + file_info['num_pairs']:
                local_idx = global_idx - file_info['start_idx']
                return file_info['file_idx'], local_idx
        
        raise IndexError(f"Index {global_idx} out of range for {self._total_pairs} total pairs")
    
    def _get_row_group_and_local_idx(self, file_idx: int, local_idx: int):
        """Get row group index and local index within that row group."""
        file_info = self.file_metadata[file_idx]
        pair_parquet = file_info['pair_parquet']
        
        # Build row group offset cache if not exists
        if 'row_group_offsets' not in file_info:
            # Build a cumulative offset for each row group
            offsets = [0]
            for rg_idx in range(pair_parquet.num_row_groups):
                rg_num_rows = pair_parquet.metadata.row_group(rg_idx).num_rows
                offsets.append(offsets[-1] + rg_num_rows)
            file_info['row_group_offsets'] = offsets
        
        offsets = file_info['row_group_offsets']
        
        # Binary search to find the right row group
        for rg_idx in range(len(offsets) - 1):
            if offsets[rg_idx] <= local_idx < offsets[rg_idx + 1]:
                local_row_idx = local_idx - offsets[rg_idx]
                return rg_idx, local_row_idx
        
        raise IndexError(f"Local index {local_idx} out of range for file {file_idx}")
        
    def _build_pairs_cache_path(self) -> Path:
        """Build a deterministic cache path for filtered pairs based on inputs."""
        def _norm_value(v):
            if v in (None, "None"):
                return None
            if isinstance(v, float):
                return round(v, 6)
            return v

        key_items = [
            ("pairs_path", str(self.pairs_parquet_path.resolve())),
            ("distance_type", self.distance_type or ""),
            ("delta", _norm_value(self.delta)),
            ("epsilon", _norm_value(self.epsilon)),
            ("shape_percentile", _norm_value(self.shape_tanimoto_percentile)),
            ("shape_num_pairs", _norm_value(self.shape_tanimoto_num_pairs)),
            ("utility_direction", self.utility_direction),
            ("utility_negate", str(self.utility_negate)),
            ("max_num_occurrences", _norm_value(self.max_num_occurrences)),
            ("utility_key_fallback", self.utility_key_fallback or ""),
            ("utility_threshold", _norm_value(self.utility_threshold)),
        ]
        key_items.sort(key=lambda kv: kv[0])
        cache_key_str = "|".join(f"{k}={v}" for k, v in key_items)
        cache_hash = hashlib.sha1(cache_key_str.encode("utf-8")).hexdigest()[:16]
        return self.dataset_cache_dir / f"filtered_pairs_{cache_hash}.parquet"

    def _try_load_pairs_cache(self, cache_path: Path) -> Optional[List[Tuple[str, str, float, dict]]]:
        """Attempt to load cached training pairs from cache_path."""
        import polars

        if cache_path.exists():
            print(f"Loading cached training pairs from {cache_path}...")
            try:
                cached_df = polars.read_parquet(cache_path)
                # Convert to 4-tuple format with empty distance_info dict for backward compatibility
                training_pairs = [
                    (row['smiles_i'], row['smiles_j'], row['utility_improvement'], {})
                    for row in cached_df.iter_rows(named=True)
                ]
                print(f"Loaded {len(training_pairs)} cached training pairs from {cache_path}")
                return training_pairs
            except Exception as e:
                print(f"Warning: failed to load cache from {cache_path}: {e}")
                print("Proceeding with fresh data processing...")
                try:
                    cache_path.unlink()
                    print(f"Removed corrupted cache file: {cache_path}")
                except Exception as cleanup_e:
                    print(f"Warning: failed to remove corrupted cache file: {cleanup_e}")
        else:
            # If in DDP and not rank0, wait for potential writer to produce cache
            if self._dist_is_initialized() and not self._is_rank0():
                self._barrier()
                if cache_path.exists():
                    try:
                        cached_df = polars.read_parquet(cache_path)
                        training_pairs = [
                            (row['smiles_i'], row['smiles_j'], row['utility_improvement'])
                            for row in cached_df.iter_rows(named=True)
                        ]
                        print(f"Loaded {len(training_pairs)} cached training pairs from {cache_path} after barrier")
                        return training_pairs
                    except Exception as e:
                        print(f"Warning: failed to load cache after barrier from {cache_path}: {e}")
        return None

    def _write_pairs_cache(self, cache_path: Path, training_pairs: List[Tuple[str, str, float, dict]]) -> None:
        """Write training_pairs to cache_path atomically with verification."""
        import polars

        # Only rank 0 writes the cache; others will wait on a barrier
        if not self._is_rank0():
            return

        if cache_path.exists():
            # Nothing to do
            self._barrier()
            return

        try:
            cache_data = []
            for smiles_i, smiles_j, utility_improvement, distance_info in training_pairs:
                cache_data.append({
                    'smiles_i': smiles_i,
                    'smiles_j': smiles_j,
                    'utility_improvement': utility_improvement
                })
            if cache_data:
                cache_df = polars.DataFrame(cache_data)
                temp_cache_path = cache_path.with_suffix('.tmp')
                cache_df.write_parquet(temp_cache_path)
                try:
                    test_df = polars.read_parquet(temp_cache_path)
                    if len(test_df) == len(cache_data):
                        temp_cache_path.rename(cache_path)
                        print(f"Cached final training pairs to {cache_path}")
                    else:
                        print(f"Warning: Cache verification failed - expected {len(cache_data)} rows, got {len(test_df)}")
                        temp_cache_path.unlink()
                except Exception as verify_e:
                    print(f"Warning: Cache verification failed: {verify_e}")
                    temp_cache_path.unlink()
        except Exception as e:
            print(f"Warning: failed to write cache to {cache_path}: {e}")
            temp_cache_path = cache_path.with_suffix('.tmp')
            if temp_cache_path.exists():
                try:
                    temp_cache_path.unlink()
                except:
                    pass
        finally:
            # Signal other ranks that writing attempt completed
            self._barrier()

    def _filter_pairs_by_max_occurrences(
        self, 
        training_pairs: List[Tuple[str, str, float, dict]]
    ) -> List[Tuple[str, str, float, dict]]:
        """
        Filter training pairs to limit the number of occurrences per first SMILES.
        
        Args:
            training_pairs: List of (smiles_first, smiles_second, utility_improvement, distance_info) tuples
                           where smiles_first is the worse molecule and smiles_second is the better one
        
        Returns:
            Filtered list of training pairs
        """
        # Determine actual distance key for "min_distance" criterion
        if self.max_num_occurrences_criterion == "min_distance":
            dist_key = self.distance_type if self.distance_type else 'morgan_tanimoto_distance'
            print(f"Now limiting pairs using criterion: {self.max_num_occurrences_criterion} (using distance key: '{dist_key}')...")
        else:
            print(f"Now limiting pairs using criterion: {self.max_num_occurrences_criterion}...")
        
        # Group pairs by first SMILES
        pairs_by_first_smiles = {}
        for pair in training_pairs:
            smiles_i, smiles_j, utility_improvement, distance_info = pair
            if smiles_i not in pairs_by_first_smiles:
                pairs_by_first_smiles[smiles_i] = []
            pairs_by_first_smiles[smiles_i].append(pair)
        
        limited_pairs = []
        
        # Select pairs based on criterion
        for first_smiles, pairs in pairs_by_first_smiles.items():
            if len(pairs) <= self.max_num_occurrences:
                # Keep all pairs if we have fewer than the limit
                limited_pairs.extend(pairs)
            else:
                # Apply criterion-based selection
                if self.max_num_occurrences_criterion == "first":
                    # Take first N pairs (original behavior)
                    limited_pairs.extend(pairs[:self.max_num_occurrences])
                elif self.max_num_occurrences_criterion == "min_distance":
                    # Select N pairs with minimum distance (using distance_type, or morgan_tanimoto_distance as default)
                    dist_key = self.distance_type if self.distance_type else 'morgan_tanimoto_distance'
                    if dist_key in pairs[0][3]:
                        # Sort by distance (ascending)
                        sorted_pairs = sorted(pairs, key=lambda p: p[3].get(dist_key, float('inf')))
                        limited_pairs.extend(sorted_pairs[:self.max_num_occurrences])
                    else:
                        # Fallback to first if distance not available
                        print(f"Warning: distance key '{dist_key}' not available in pair data, falling back to 'first' criterion")
                        limited_pairs.extend(pairs[:self.max_num_occurrences])
                elif self.max_num_occurrences_criterion == "min_shape_tanimoto":
                    # Select N pairs with minimum shape_tanimoto_distance
                    if 'shape_tanimoto_distance' in pairs[0][3]:
                        # Sort by shape_tanimoto_distance (ascending)
                        sorted_pairs = sorted(pairs, key=lambda p: p[3].get('shape_tanimoto_distance', float('inf')))
                        limited_pairs.extend(sorted_pairs[:self.max_num_occurrences])
                    else:
                        # Fallback to first if shape_tanimoto_distance not available
                        print(f"Warning: shape_tanimoto_distance not available, falling back to 'first' criterion")
                        limited_pairs.extend(pairs[:self.max_num_occurrences])
                else:
                    raise ValueError(f"Unknown max_num_occurrences_criterion: {self.max_num_occurrences_criterion}. "
                                   f"Must be one of: 'first', 'min_distance', 'min_shape_tanimoto'")
        
        print(f"After max_num_occurrences filtering (<= {self.max_num_occurrences} per first SMILES, criterion={self.max_num_occurrences_criterion}): {len(limited_pairs)} pairs (from {len(training_pairs)}, kept {100*len(limited_pairs)/len(training_pairs):.1f}%)")
        print(f"Number of unique first SMILES: {len(pairs_by_first_smiles)}")
        
        return limited_pairs

    def _load_and_filter_pairs(self) -> List[Tuple[str, str, float, dict]]:
        """
        Load data from both tables and filter by distance, utility, and split.
        Ensures the second molecule is always the better one.
        
        Returns:
            List of (smiles_worse, smiles_better, utility_improvement, distance_info) tuples
        """
        import polars
        
        # Load molecules table
        print(f'Loading molecules table from {self.molecules_parquet_path}')
        molecules_df = _read_tabular_file(self.molecules_parquet_path)

        # Add item_id column if it doesn't exist
        if "item_id" not in molecules_df.columns:
            molecules_df = molecules_df.with_columns(polars.Series(name="item_id", values=range(len(molecules_df))))
        
        # Apply utility_key_fallback if provided
        molecules_df = self._apply_utility_fallback(molecules_df)

        # I think this is unnecessary
        # if "split" in molecules_df.columns:
        #     molecules_df = molecules_df.filter(polars.col("split") == self.split)
        # else:
        #     molecules_df = molecules_df.filter(polars.col(self.split) == True)
        
        # Handle utility key - allow for null utility_key
        if self.utility_key is None or self.utility_key == 'None':
            print("Warning: utility_key is null. Using all pairs with random ordering.")
            utility_lookup = None
        else:
            # Check if utility column exists
            if self.utility_key not in molecules_df.columns:
                available_columns = [col for col in molecules_df.columns if col.startswith('utility_') or col in ['gap', 'homo', 'lumo', 'u0', 'u', 'h', 'g', 'cv', 'mu', 'alpha', 'r2', 'zpve']]
                raise ValueError(f"Utility column '{self.utility_key}' not found. Available utility columns: {available_columns}")
            
            # Create utility lookup dictionary
            if self.utility_negate:
                utility_lookup = {row['item_id']: -row[self.utility_key] for row in molecules_df.iter_rows(named=True)}
            else:
                utility_lookup = {row['item_id']: row[self.utility_key] for row in molecules_df.iter_rows(named=True)}
        
        # Create lookup dictionaries
        smiles_lookup = {row['item_id']: row['smiles'] for row in molecules_df.iter_rows(named=True)}
        
        # Optional caching for filtered pairs to avoid re-reading and re-filtering large files.
        # If we're subsetting, caching full filtered pairs defeats the purpose (would still load/store all pairs),
        # so we bypass cache in that case.
        if self.use_cache and not self._subset_applied:
            cache_path = self._build_pairs_cache_path()
            cached = self._try_load_pairs_cache(cache_path)
            if cached is not None:
                return cached
        
            # Clean up any leftover temporary files (only rank0)
            if self._is_rank0():
                temp_cache_path = cache_path.with_suffix('.tmp')
                if temp_cache_path.exists():
                    try:
                        temp_cache_path.unlink()
                        print(f"Cleaned up leftover temporary file: {temp_cache_path}")
                    except:
                        pass
        
        # only get here if there's no cached file
        # Process data from scratch
        # Load pairs table. not using scan_parquet, so it's loading everything I think?

        print(f'Loading pairs table from {self.pairs_parquet_path}')
        pairs_df = polars.read_parquet(self.pairs_parquet_path)
        
        # removing because pairs should only be created from train split already, by pair_gen script
        #pairs_df = pairs_df.filter(polars.col("split") == self.split)
        
        print(f"Loaded {len(pairs_df)} pairs using utility '{self.utility_key}'")
        
        # Filter by distance threshold if specified
        if self.delta is not None and self.delta != 'None':
            if self.distance_type is None:
                raise ValueError(f"Distance type must be specified if delta is provided as {self.delta}")
            pairs_df = pairs_df.filter(polars.col(self.distance_type) < self.delta)
            print(f"After distance filtering (< {self.delta}): {len(pairs_df)} pairs")
        
        # Filter by shape Tanimoto if specified
        if self.shape_tanimoto_percentile is not None or self.shape_tanimoto_num_pairs is not None:
            # Check if shape_tanimoto_distance column exists
            if 'shape_tanimoto_distance' not in pairs_df.columns:
                print(f"Warning: shape_tanimoto_distance column not found in pairs file. Skipping shape Tanimoto filtering.")
                print(f"Available columns: {pairs_df.columns}")
            else:
                # Convert to pandas DataFrame for the utility function
                print('number of pairs before shape tanimoto filtering', len(pairs_df), 'shape_tanimoto_percentile', self.shape_tanimoto_percentile, 'shape_tanimoto_num_pairs', self.shape_tanimoto_num_pairs)
                if self.shape_tanimoto_num_pairs is not None or self.shape_tanimoto_percentile is not None:
                    pairs_pandas = pairs_df.to_pandas()
                    
                    # Use the utility function to filter by shape Tanimoto
                    filtered_pairs_pandas = filter_top_pairs_per_molecule(
                        pairs_pandas, 
                        property_key='shape_tanimoto_distance',
                        percentile=self.shape_tanimoto_percentile,
                        num_pairs=self.shape_tanimoto_num_pairs
                    )

                    print('len(filtered_pairs_pandas)', len(filtered_pairs_pandas))
                
                    # Convert back to polars DataFrame
                    pairs_df = polars.from_pandas(filtered_pairs_pandas)
                
                if self.shape_tanimoto_percentile is not None:
                    print(f"After shape Tanimoto percentile filtering ({self.shape_tanimoto_percentile}%): {len(pairs_df)} pairs")
                else:
                    print(f"After shape Tanimoto num_pairs filtering ({self.shape_tanimoto_num_pairs} pairs per molecule): {len(pairs_df)} pairs")

        def _iter_training_pairs():
            # Yield filtered (smiles_worse, smiles_better, utility_improvement, distance_info) tuples
            # distance_info is a dict with available distance metrics
            for row in pairs_df.iter_rows(named=True):
                item_i = row["molecule_a_idx"]
                item_j = row["molecule_b_idx"]

                smiles_i = smiles_lookup.get(item_i)
                smiles_j = smiles_lookup.get(item_j)
                if smiles_i is None or smiles_j is None:
                    continue

                # Extract distance information
                distance_info = {}
                if self.distance_type and self.distance_type in row:
                    distance_info[self.distance_type] = row[self.distance_type]
                if 'morgan_tanimoto_distance' in row:
                    distance_info['morgan_tanimoto_distance'] = row['morgan_tanimoto_distance']
                if 'shape_tanimoto_distance' in row:
                    distance_info['shape_tanimoto_distance'] = row['shape_tanimoto_distance']

                if handle_null_utility_key:
                    if self.keep_original_ordering:
                        # Preserve original ordering from parquet file
                        yield (smiles_i, smiles_j, 0.0, distance_info)
                    else:
                        # Deterministic ordering based on hash for reproducibility
                        if hash((smiles_i, smiles_j)) % 2 == 0:
                            yield (smiles_i, smiles_j, 0.0, distance_info)
                        else:
                            yield (smiles_j, smiles_i, 0.0, distance_info)
                    continue

                utility_i = utility_lookup.get(item_i)
                utility_j = utility_lookup.get(item_j)
                if utility_i is None or utility_j is None:
                    continue

                utility_improvement = utility_j - utility_i

                if handle_null_utility:
                    if self.keep_original_ordering:
                        # Preserve original ordering from parquet file
                        yield (smiles_i, smiles_j, abs(utility_j - utility_i), distance_info)
                    else:
                        if hash((smiles_i, smiles_j)) % 2 == 0:
                            yield (smiles_i, smiles_j, abs(utility_j - utility_i), distance_info)
                        else:
                            yield (smiles_j, smiles_i, abs(utility_j - utility_i), distance_info)
                    continue

                # print(f'epsilon {self.epsilon} utility_improvement {utility_improvement} abs(utility_improvement) {abs(utility_improvement)}')
                # breakpoint()
                if self.epsilon is not None and self.epsilon > 0 and abs(utility_improvement) < self.epsilon:
                    continue

                # Threshold crossing: require the pair to straddle utility_threshold, so the
                # improved molecule lands past an absolute bar rather than merely being
                # relatively better. Direction-independent; reordering happens below.
                if self.utility_threshold is not None:
                    lo, hi = min(utility_i, utility_j), max(utility_i, utility_j)
                    if not (lo < self.utility_threshold <= hi):
                        continue

                if self.keep_original_ordering:
                    # Preserve original ordering from parquet file (molecule_a_idx, molecule_b_idx)
                    yield (smiles_i, smiles_j, utility_improvement, distance_info)
                else:
                    # Apply utility_direction-based reordering
                    if self.utility_direction == "increasing":
                        if utility_improvement > 0:
                            yield (smiles_i, smiles_j, utility_improvement, distance_info)
                        else:
                            yield (smiles_j, smiles_i, -utility_improvement, distance_info)
                    else:
                        if utility_improvement < 0:
                            yield (smiles_i, smiles_j, -utility_improvement, distance_info)
                        else:
                            yield (smiles_j, smiles_i, utility_improvement, distance_info)

        # Filter by utility improvement and ensure correct ordering
        training_pairs: List[Tuple[str, str, float, dict]] = []
        
        # Check if we need to handle null delta/epsilon case
        handle_null_utility = (self.delta is None or self.delta == 'None') and (self.epsilon is None or self.epsilon == 'None')
        
        # Check if we need to handle null utility_key case
        handle_null_utility_key = utility_lookup is None
        
        if handle_null_utility:
            print("Warning: Both delta and epsilon are null. Using all pairs with random ordering.")
        
        if handle_null_utility_key:
            print("Warning: utility_key is null. Using all pairs with random ordering.")
        
        if not self._subset_applied:
            training_pairs = list(_iter_training_pairs())
        else:
            # Two-pass: count, then reservoir sample. Keeps memory bounded by subset size.
            print(f"[SUBSET] Applying subsetting to dataset (after epsilon/delta filtering)...")
            if self.subset_fraction is not None:
                print(f"[SUBSET]   subset_fraction: {self.subset_fraction} (will be applied to filtered pairs)")
            if self.subset_max_samples is not None:
                print(f"[SUBSET]   subset_max_samples: {self.subset_max_samples:,} (controlled dataset size)")
            print(f"[SUBSET]   subset_seed: {self.subset_seed}")
            
            total_valid = 0
            for _ in _iter_training_pairs():
                total_valid += 1

            # Compute desired subset size: if both fraction and max_samples provided, take the smaller.
            desired = total_valid
            if self.subset_fraction is not None:
                f = float(self.subset_fraction)
                if f <= 0.0:
                    desired = 0
                elif f < 1.0:
                    desired = int(total_valid * f)  # floor (smaller/equal)
                    desired = max(1, desired) if total_valid > 0 else 0
                else:
                    desired = total_valid
            if self.subset_max_samples is not None:
                m = int(self.subset_max_samples)
                desired = min(desired, max(0, m))

            print(f"[SUBSET]   Total valid pairs found: {total_valid:,}")
            print(f"[SUBSET]   Desired subset size: {desired:,}")

            if desired <= 0 or total_valid <= 0:
                training_pairs = []
                print(f"[SUBSET]   Result: Empty dataset (desired={desired}, total_valid={total_valid})")
            elif desired >= total_valid:
                training_pairs = list(_iter_training_pairs())
                print(f"[SUBSET]   Result: Using all pairs (desired >= total_valid)")
            else:
                print(f"[SUBSET]   Sampling {desired:,} pairs from {total_valid:,} using reservoir sampling...")
                rng = random.Random(int(self.subset_seed))
                sample: List[Tuple[str, str, float, dict]] = []
                seen = 0
                for pair in _iter_training_pairs():
                    seen += 1
                    if len(sample) < desired:
                        sample.append(pair)
                    else:
                        j = rng.randrange(seen)
                        if j < desired:
                            sample[j] = pair
                training_pairs = sample
                print(f"[SUBSET]   Result: Successfully sampled {len(training_pairs):,} pairs ({100.0*len(training_pairs)/total_valid:.2f}% of total)")
        
        if handle_null_utility_key:
            print(f"Using all pairs with random ordering (no utility): {len(training_pairs)} pairs")
        elif handle_null_utility:
            print(f"Using all pairs with random ordering: {len(training_pairs)} pairs")
        else:
            print(f"After utility filtering (> {self.epsilon}): {len(training_pairs)} pairs")
        
        # Optionally limit the number of occurrences per first SMILES after ordering
        if self.max_num_occurrences is not None and self.max_num_occurrences != 'None':
            training_pairs = self._filter_pairs_by_max_occurrences(training_pairs)

        # Save final training pairs to cache for next time (only rank0 writes)
        if self.use_cache and not self._subset_applied:
            self._write_pairs_cache(cache_path, training_pairs)
            # Non-rank0 ranks will have waited in _try_load_pairs_cache; nothing else needed here
        
        return training_pairs
    
    def __len__(self) -> int:
        if self.ignore_pairs:
            return len(self._smiles_list)
        if self.use_multi_file:
            return self._total_pairs
        else:
            return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        prompt = dataset_prompt_text(self.dataset_id) if self.use_dataset_identity_prompt else ""
        if self.ignore_pairs:
            # Single-molecule mode: construct sequence from a single SMILES
            smiles = self._get_smiles_item(idx)
            if self.seeds_only_conditional:
                # Seeds-only conditional: format as seed<sep> so callback generates after sep
                sequence = f"{self._build_full_prompt(smiles1=smiles, smiles2=None)}{smiles}<sep>"
            else:
                # Original unconditional mode: no separator
                sequence = f"{self._build_full_prompt(smiles1=smiles, smiles2=None)}{smiles}"
        else:
            if self.use_multi_file:
                smiles1, smiles2, utility_improvement = self._get_multi_file_item(idx)
            else:
                smiles1, smiles2, utility_improvement, _ = self.training_pairs[idx]

            if self.just_first:
                sequence = f"{self._build_full_prompt(smiles1=smiles1, smiles2=None)}{smiles1}"
            else:
                if self.duplicate:
                    smiles2 = smiles1
                
                # Construct the training sequence: <cls>smiles1<sep>smiles2<eos>
                # smiles1 is the worse molecule, smiles2 is the better molecule
                sequence = f"{self._build_full_prompt(smiles1=smiles1, smiles2=smiles2)}{smiles1}<sep>{smiles2}"
        
        # Tokenize the sequence using the transform function
        tokenized = self.transform(sequence) #[smiles1, smiles2]) #sequence)

        # Create labels tensor and mask out padding tokens with -100
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100 # new based on docs https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaForCausalLM

        res = {
            "input_ids": tokenized["input_ids"],
            #"labels": labels,
            "attention_mask": tokenized["attention_mask"],
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }

        if self._prop_export_metadata:
            if self.ignore_pairs or self.just_first:
                raise ValueError("Prompt metadata export requires paired conditional examples")
            res["prompt_metadata"] = prompt_metadata(
                smiles1, smiles2,
                self._build_property_prompts(smiles1=smiles1, smiles2=smiles2),
                self._build_property_prompts(smiles1=smiles1, smiles2=smiles2, randomize_tokens=False),
            )

        return res #, [] # as label # removing this

    # ---------------- distance prompt helpers ----------------
    def _get_cache_dir(self) -> Path:
        """Get or create the dataset cache directory."""
        if hasattr(self, 'dataset_cache_dir') and self.dataset_cache_dir is not None:
            return self.dataset_cache_dir
        
        # Cache directory not set up yet, create it now using same logic as __init__
        # First, ensure cache_dir is set up (may not be initialized yet)
        if not hasattr(self, 'cache_dir') or self.cache_dir is None:
            import tempfile
            # Use default cache directory (will be overridden later in __init__ if cache_dir param was provided)
            cache_dir_path = Path(tempfile.gettempdir()) / "lobster_cache"
        else:
            cache_dir_path = Path(self.cache_dir)
        
        cache_id_parts = [
            ("root", str(self.root.resolve())),
            ("utility", self.utility_key or "none"),
            ("utility_fallback", self.utility_key_fallback or "none"),
        ]
        # Add file names to cache ID for multi-file scenarios
        # Use molecule_filename/pair_filename from __init__ params if available
        if hasattr(self, 'molecule_filenames'):
            if isinstance(self.molecule_filenames, list):
                cache_id_parts.append(("mol_files", "_".join(sorted(self.molecule_filenames))))
            else:
                cache_id_parts.append(("mol_file", self.molecule_filenames))
        elif hasattr(self, 'molecule_filename'):
            # Fallback to original parameter name
            if isinstance(self.molecule_filename, list):
                cache_id_parts.append(("mol_files", "_".join(sorted(self.molecule_filename))))
            else:
                cache_id_parts.append(("mol_file", self.molecule_filename))
        
        if hasattr(self, 'pair_filenames'):
            if isinstance(self.pair_filenames, list):
                cache_id_parts.append(("pair_files", "_".join(sorted(self.pair_filenames))))
            else:
                cache_id_parts.append(("pair_file", self.pair_filenames))
        elif hasattr(self, 'pair_filename'):
            # Fallback to original parameter name
            if isinstance(self.pair_filename, list):
                cache_id_parts.append(("pair_files", "_".join(sorted(self.pair_filename))))
            else:
                cache_id_parts.append(("pair_file", self.pair_filename))
        
        cache_id_str = "|".join(f"{k}={v}" for k, v in sorted(cache_id_parts))
        cache_id_hash = hashlib.sha1(cache_id_str.encode("utf-8")).hexdigest()[:12]
        dataset_cache_dir = cache_dir_path / f"dataset_{cache_id_hash}"
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        # Store for future use
        self.dataset_cache_dir = dataset_cache_dir
        # Also store cache_dir if not already stored
        if not hasattr(self, 'cache_dir'):
            self.cache_dir = cache_dir_path
        return dataset_cache_dir

    def _load_or_compute_distance_distributions(self) -> None:
        """Load distance distributions from file or compute if missing."""
        distribution_path = self.root / self._distance_distribution_file
        cache_path = self._get_cache_dir() / self._distance_distribution_file

        # Try to load from data root first, then cache
        if distribution_path.exists():
            print(f"Loading distance distributions from {distribution_path}...")
            self._distance_distributions = self._load_distance_distributions(distribution_path)
            print(f"Loaded {len(self._distance_distributions)} distance distributions")
        elif cache_path.exists():
            print(f"Loading distance distributions from cache: {cache_path}...")
            self._distance_distributions = self._load_distance_distributions(cache_path)
            print(f"Loaded {len(self._distance_distributions)} distance distributions from cache")
        else:
            if self._distance_compute_if_missing:
                print(f"Distance distribution file not found: {distribution_path}")
                print("Computing distance distributions...")
                # Validate dataset first
                self._validate_dataset_for_distance_prompting()
                # Compute distributions
                self._distance_distributions = self._compute_distance_distributions()
                # Try to save to file (may fail if directory is read-only)
                try:
                    self._save_distance_distributions(distribution_path)
                    print(f"Computed and saved {len(self._distance_distributions)} distance distributions to {distribution_path}")
                except PermissionError:
                    # Try cache directory as fallback
                    cache_path = self._get_cache_dir() / self._distance_distribution_file
                    try:
                        self._save_distance_distributions(cache_path)
                        print(f"Computed and saved {len(self._distance_distributions)} distance distributions to cache: {cache_path}")
                        print(f"Note: Data directory {self.root} is read-only. Using cache directory instead.")
                    except Exception as e:
                        raise PermissionError(
                            f"Cannot write distance distributions file. "
                            f"Tried {distribution_path} (read-only) and {cache_path} (failed: {e}). "
                            f"Please run scripts/precompute_distance_distributions.py manually to generate the file, "
                            f"or ensure write permissions to either the data directory or cache directory."
                        ) from e
            else:
                raise FileNotFoundError(
                    f"Distance distribution file not found: {distribution_path}. "
                    "Run scripts/precompute_distance_distributions.py first, or set "
                    "prompting.distance.source.compute_if_missing=true"
                )

    def _load_distance_distributions(self, path: Path) -> dict[int, DistanceDistributionParams]:
        """Load distance distributions from parquet file."""
        import polars as pl

        df = pl.read_parquet(path)
        distributions: dict[int, DistanceDistributionParams] = {}

        for row in df.iter_rows(named=True):
            molecule_a_idx = int(row["molecule_a_idx"])
            distributions[molecule_a_idx] = DistanceDistributionParams(
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                p0=float(row["p0"]),
                p25=float(row["p25"]),
                p50=float(row["p50"]),
                p75=float(row["p75"]),
                p100=float(row["p100"]),
                num_pairs=int(row["num_pairs"]),
            )

        return distributions

    def _validate_dataset_for_distance_prompting(self) -> None:
        """Validate that dataset has sufficient pairs per molecule for distance prompting."""
        if self.ignore_pairs:
            raise ValueError("Distance prompting requires pairs data (ignore_pairs=False)")

        import polars as pl

        # Load pairs to check
        if self.use_multi_file:
            # For multi-file, we'd need to check all files - simplified check
            # Just check first file as representative
            if len(self.file_paths) == 0:
                raise ValueError("No pair files found")
            pair_path = self.file_paths[0][1]
        else:
            pair_path = self.pairs_parquet_path

        pairs_df = pl.read_parquet(pair_path)

        if "molecule_a_idx" not in pairs_df.columns:
            raise ValueError("molecule_a_idx column not found in pairs file")

        if self._distance_column not in pairs_df.columns:
            raise ValueError(f"Distance column '{self._distance_column}' not found in pairs file")

        # Count pairs per molecule_a
        counts = pairs_df.group_by("molecule_a_idx").agg(pl.count().alias("count"))
        low_pairs = counts.filter(pl.col("count") < self._distance_min_pairs_per_molecule)
        num_low = len(low_pairs)
        total_molecules = len(counts)

        if total_molecules == 0:
            raise ValueError("No molecules found in pairs file")

        fraction_low = num_low / total_molecules if total_molecules > 0 else 1.0

        if fraction_low > self._distance_max_fraction_low_pairs:
            raise ValueError(
                f"Distance percentile prompting requires sufficient pairs per molecule_a. "
                f"Found {num_low} molecules ({fraction_low*100:.1f}%) with < {self._distance_min_pairs_per_molecule} pairs. "
                f"This feature is designed for large datasets like ZINC, not small datasets like BACE. "
                f"Consider using a different prompting method or increasing the number of pairs per molecule."
            )

        if num_low > 0:
            print(f"Warning: {num_low} molecules ({fraction_low*100:.1f}%) have < {self._distance_min_pairs_per_molecule} pairs. "
                  f"These will use uniform distribution.")

    def _compute_distance_distributions(self) -> dict[int, DistanceDistributionParams]:
        """Compute distance distributions from pairs data."""
        from pairit.prompting.distance_prompting import (
            compute_percentiles_from_beta,
            fit_beta_distribution,
        )
        import polars as pl
        import numpy as np

        distributions: dict[int, DistanceDistributionParams] = {}

        # Load pairs
        if self.use_multi_file:
            # For multi-file, combine all pairs
            all_pairs = []
            for mol_path, pair_path in self.file_paths:
                pairs_df = pl.read_parquet(pair_path)
                all_pairs.append(pairs_df)
            pairs_df = pl.concat(all_pairs)
        else:
            pairs_df = pl.read_parquet(self.pairs_parquet_path)

        # Group by molecule_a_idx
        grouped = pairs_df.group_by("molecule_a_idx")

        for group_name, group_df in grouped:
            molecule_a_idx = group_name[0] if isinstance(group_name, tuple) else group_name

            # Get distances for this molecule
            raw_values = group_df[self._distance_column].drop_nulls().to_list()
            # Convert similarity to distance if needed
            distances = [self._convert_to_distance(v) for v in raw_values]
            num_pairs = len(distances)

            if num_pairs < self._distance_min_pairs_per_molecule:
                # Use uniform distribution
                alpha, beta_param = 1.0, 1.0
                p0, p25, p50, p75, p100 = 0.0, 0.25, 0.5, 0.75, 1.0
            else:
                # Fit Beta distribution
                alpha, beta_param = fit_beta_distribution(distances)

                # Compute percentiles
                try:
                    p0, p25, p50, p75, p100 = compute_percentiles_from_beta(alpha, beta_param)
                except Exception:
                    # Fallback to empirical percentiles
                    dist_array = np.array(distances)
                    p0 = float(np.min(dist_array))
                    p25 = float(np.percentile(dist_array, 25))
                    p50 = float(np.percentile(dist_array, 50))
                    p75 = float(np.percentile(dist_array, 75))
                    p100 = float(np.max(dist_array))

            distributions[int(molecule_a_idx)] = DistanceDistributionParams(
                alpha=float(alpha),
                beta=float(beta_param),
                p0=float(p0),
                p25=float(p25),
                p50=float(p50),
                p75=float(p75),
                p100=float(p100),
                num_pairs=int(num_pairs),
            )

        return distributions

    def _save_distance_distributions(self, path: Path) -> None:
        """Save distance distributions to parquet file."""
        import polars as pl

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for molecule_a_idx, params in self._distance_distributions.items():
            data.append({
                "molecule_a_idx": molecule_a_idx,
                "alpha": params.alpha,
                "beta": params.beta,
                "p0": params.p0,
                "p25": params.p25,
                "p50": params.p50,
                "p75": params.p75,
                "p100": params.p100,
                "num_pairs": params.num_pairs,
            })

        df = pl.DataFrame(data)
        df.write_parquet(path)

    def _get_distance_percentile(self, molecule_a_idx: int, distance: float) -> Optional[float]:
        """Compute percentile of distance for given molecule_a."""
        if molecule_a_idx not in self._distance_distributions:
            # Use uniform distribution as fallback
            return distance * 100.0  # Simple linear mapping

        params = self._distance_distributions[molecule_a_idx]
        try:
            return percentile_from_beta_cdf(distance, params.alpha, params.beta)
        except Exception:
            # Fallback to uniform
            return distance * 100.0

    def _get_distance_bin(self, molecule_a_idx: int, distance: float) -> Optional[int]:
        """Get distance bin index for given molecule_a and distance."""
        if self._distance_binning_mode == "fixed_quartiles":
            # For fixed_quartiles, compare distance directly to stored percentile values
            if molecule_a_idx not in self._distance_distributions:
                # Use uniform distribution as fallback
                stored_percentiles = (0.0, 0.25, 0.5, 0.75, 1.0)
            else:
                params = self._distance_distributions[molecule_a_idx]
                stored_percentiles = (params.p0, params.p25, params.p50, params.p75, params.p100)
            
            from pairit.prompting.distance_prompting import distance_to_bin_fixed_quartiles
            return distance_to_bin_fixed_quartiles(distance, stored_percentiles)
        else:  # curve_fit
            # For curve_fit, compute percentile first, then bin
            percentile = self._get_distance_percentile(molecule_a_idx, distance)
            if percentile is None:
                return None
            
            return percentile_to_bin(
                percentile,
                self._distance_binning_mode,
                self._distance_num_bins,
                None,  # stored_percentiles not used in curve_fit mode
            )

    # ---------------- property prompt helpers ----------------
    def _build_full_prompt(self, *, smiles1: str, smiles2: Optional[str]) -> str:
        base = dataset_prompt_text(self.dataset_id) if self.use_dataset_identity_prompt else ""
        
        # Build property prompts
        prop_prompt = ""
        if self._prop_prompt_enabled:
            # Currently only bin-based prompting is supported; literal numeric prompting can be added later.
            if self._prop_include_start_value not in ("off", "bin"):
                raise ValueError(f"Unsupported prompting.properties.include.start_value={self._prop_include_start_value!r}")
            if self._prop_include_magnitude not in ("off", "bin", "signed_bin"):
                raise ValueError(f"Unsupported prompting.properties.include.magnitude={self._prop_include_magnitude!r}")

            prompts = self._build_property_prompts(smiles1=smiles1, smiles2=smiles2)
            if prompts:
                prop_prompt = build_property_prompt_text(
                    prompts,
                    include_start_marker=True,
                    value_width=self._value_width,
                    delta_width=self._delta_width,
                )

        # Build distance prompt
        dist_prompt = ""
        if (
            self._distance_prompt_enabled
            and self._distance_eval_override_enabled
            and self._distance_eval_override_bin is not None
        ):
            # Inference-time controllable generation: inject the desired distance bin for
            # every seed, regardless of any ground-truth pair (smiles2). This is the path
            # that lets a user *specify* a target structural distance.
            dist_prompt = build_distance_prompt_text(
                self._distance_eval_override_bin,
                include_start_marker=True,
                bin_width=self._distance_bin_width,
            )
        elif self._distance_prompt_enabled and smiles2 is not None:
            # Training / paired eval: derive the distance bin from the real pair distance.
            # Get molecule_a_idx from smiles1
            # We need to find the index of smiles1 in the molecules file
            molecule_a_idx = self._get_molecule_idx(smiles1)
            if molecule_a_idx is not None:
                # Get distance from pairs data
                distance = self._get_pair_distance(molecule_a_idx, smiles1, smiles2)
                if distance is not None:
                    bin_idx = self._get_distance_bin(molecule_a_idx, distance)
                    if bin_idx is not None:
                        dist_prompt = build_distance_prompt_text(
                            bin_idx,
                            include_start_marker=True,
                            bin_width=self._distance_bin_width,
                        )

        return base + prop_prompt + dist_prompt

    def _get_molecule_idx(self, smiles: str) -> Optional[int]:
        """Get molecule index from SMILES string (with caching)."""
        if smiles in self._molecule_idx_cache:
            return self._molecule_idx_cache[smiles]

        import polars as pl

        if self.use_multi_file:
            # Check all files
            for mol_path, _ in self.file_paths:
                try:
                    df = _read_tabular_file(mol_path, columns=["smiles", "item_id"])
                    matches = df.filter(pl.col("smiles") == smiles)
                    if len(matches) > 0:
                        idx = int(matches["item_id"][0])
                        self._molecule_idx_cache[smiles] = idx
                        return idx
                except Exception:
                    continue
        else:
            mol_path = self.molecules_parquet_path
            if mol_path is not None:
                try:
                    df = _read_tabular_file(mol_path, columns=["smiles", "item_id"])
                    matches = df.filter(pl.col("smiles") == smiles)
                    if len(matches) > 0:
                        idx = int(matches["item_id"][0])
                        self._molecule_idx_cache[smiles] = idx
                        return idx
                except Exception:
                    pass

        return None

    def _convert_to_distance(self, value: float) -> float:
        """Convert similarity to distance if is_similarity is True, otherwise return value as-is."""
        if self._distance_is_similarity:
            return 1.0 - value
        return value

    def _get_pair_distance(self, molecule_a_idx: int, smiles1: str, smiles2: str) -> Optional[float]:
        """Get distance for a pair from pairs data (with caching)."""
        # Get molecule_b_idx
        molecule_b_idx = self._get_molecule_idx(smiles2)
        if molecule_b_idx is None:
            return None

        # Check cache
        cache_key = (molecule_a_idx, molecule_b_idx)
        if cache_key in self._pair_distance_cache:
            return self._pair_distance_cache[cache_key]

        import polars as pl

        # Load pairs and find matching pair
        if self.use_multi_file:
            # Check all files
            for mol_path, pair_path in self.file_paths:
                try:
                    pairs_df = pl.read_parquet(pair_path, columns=["molecule_a_idx", "molecule_b_idx", self._distance_column])
                    match = pairs_df.filter(
                        (pl.col("molecule_a_idx") == molecule_a_idx) &
                        (pl.col("molecule_b_idx") == molecule_b_idx)
                    )
                    if len(match) > 0:
                        raw_value = float(match[self._distance_column][0])
                        distance = self._convert_to_distance(raw_value)
                        self._pair_distance_cache[cache_key] = distance
                        return distance
                except Exception:
                    continue
        else:
            try:
                pairs_df = pl.read_parquet(self.pairs_parquet_path, columns=["molecule_a_idx", "molecule_b_idx", self._distance_column])
                match = pairs_df.filter(
                    (pl.col("molecule_a_idx") == molecule_a_idx) &
                    (pl.col("molecule_b_idx") == molecule_b_idx)
                )
                if len(match) > 0:
                    raw_value = float(match[self._distance_column][0])
                    distance = self._convert_to_distance(raw_value)
                    self._pair_distance_cache[cache_key] = distance
                    return distance
            except Exception:
                pass

        return None

    @property
    def _current_epoch(self) -> int:
        return int(self._epoch_state.item())

    def set_epoch(self, epoch: int) -> None:
        """Update dropout's epoch in the parent and persistent worker copies."""
        self._epoch_state.fill_(int(epoch))

    def _build_property_prompts(
        self, *, smiles1: str, smiles2: Optional[str], randomize_tokens: bool | None = None,
    ) -> list[ParsedPropertyPrompt]:
        names = list(self._prop_names)
        use_randomized_tokens = self._prop_randomize_tokens if randomize_tokens is None else randomize_tokens
        stable = (stable_prompt_seed("prompt_source", smiles1) if smiles2 is None
                  else stable_prompt_seed("prompt_pair", smiles1, smiles2))
        row_seed = int(self._prop_selection_seed) + int(self.dataset_id) * 10_000 + stable
        r = random.Random(row_seed)
        # Drawing random bins must not change selection or the target proxy.
        token_random = random.Random(row_seed + 1_000_000_007)
        if self._prop_selection_mode == "all":
            chosen = names
        elif self._prop_selection_mode == "explicit":
            chosen = [n for n in self._prop_selection_explicit_names if n in names]
        elif self._prop_selection_mode == "random_k":
            k = max(1, min(int(self._prop_selection_k), len(names)))
            chosen = r.sample(names, k=k)
        else:  # random_one (default)
            chosen = [r.choice(names)]

        # Property dropout: independently drop each chosen property with some probability.
        # This allows the model to see 0, 1, ..., N properties during training so it can
        # handle any subset at inference time.
        if (self._is_training and self._prop_property_dropout_enabled
                and not self._eval_override_enabled
                and not self._prop_randomize_tokens):
            epoch_offset = self._current_epoch if self._prop_component_dropout_epoch_vary else 0
            chosen = [
                p for p in chosen
                if random.Random(stable_prompt_seed("property_dropout", row_seed, epoch_offset, p)).random()
                >= self._prop_property_dropout_rate
            ]

        target_is_mapped = self._prop_column_map.get("target") not in (None, "", "target")
        real_names = [p for p in names if p != "target"]
        requested = [p for p in chosen if p != "target" or target_is_mapped]
        if "target" in chosen and not target_is_mapped:
            if not real_names:
                raise ValueError("Unmapped target needs at least one descriptor to mirror")
            requested = sorted(set(requested) | set(real_names))
        props1 = self._get_properties_for_smiles(smiles1, requested)
        props2 = self._get_properties_for_smiles(smiles2, requested) if smiles2 else {}

        out: list[ParsedPropertyPrompt] = []
        for p in chosen:
            start_bin = None
            direction: Optional[str] = None
            delta_bin = None
            signed_delta_bin = None

            p_for_binning = r.choice(real_names) if p == "target" and not target_is_mapped else p
            v1 = props1.get(p_for_binning)
            v2 = props2.get(p_for_binning)
            raw_delta = float(v2) - float(v1) if v1 is not None and v2 is not None else None
            raw_direction = (None if raw_delta is None else
                             "up" if raw_delta > 0 else "down" if raw_delta < 0 else "any")

            if self._prop_include_start_value == "bin":
                vb = self._value_binning_per_prop.get(p_for_binning, self._value_binning_default)
                if vb is None:
                    raise ValueError("value binning config missing")
                if vb.mode == "inferred" and vb.range is None:
                    vb = self._infer_binning_range(vb, prop_name=p_for_binning, kind="value")

                if self._eval_override_enabled and self._eval_override_start_bin is not None:
                    start_bin = self._eval_override_start_bin
                elif use_randomized_tokens:
                    # Randomize start_bin: random value from 0 to num_bins-1
                    start_bin = token_random.randint(0, vb.num_bins - 1)
                elif v1 is not None:
                    start_bin = value_to_bin(binning=vb, value=float(v1))
                else:
                    start_bin = None

            if self._prop_include_direction:
                if self._eval_override_enabled and self._eval_override_direction is not None:
                    direction = self._eval_override_direction
                elif use_randomized_tokens:
                    # Randomize direction: random choice from ["up", "down", "any"]
                    direction = token_random.choice(["up", "down"])
                elif v1 is not None and v2 is not None:
                    d = float(v2) - float(v1)
                    if d > 0:
                        direction = "up"
                    elif d < 0:
                        direction = "down"
                    else:
                        direction = "any"
                else:
                    direction = "any"

            if self._prop_include_magnitude == "bin":
                db = self._delta_binning_per_prop.get(p_for_binning, self._delta_binning_default)
                if db is None:
                    raise ValueError("delta binning config missing")
                if db.mode == "inferred" and db.range is None:
                    db = self._infer_binning_range(db, prop_name=p_for_binning, kind="delta")

                if self._eval_override_enabled and self._eval_override_delta_bin is not None:
                    delta_bin = self._eval_override_delta_bin
                elif use_randomized_tokens:
                    # Randomize delta_bin: random value from 0 to num_bins-1
                    delta_bin = token_random.randint(0, db.num_bins - 1)
                elif v1 is not None and v2 is not None:
                    delta_bin = delta_to_bin(binning=db, delta=float(v2) - float(v1), magnitude=True)
                else:
                    delta_bin = None
            elif self._prop_include_magnitude == "signed_bin":
                db = self._delta_binning_per_prop.get(p_for_binning, self._delta_binning_default)
                if db is None:
                    raise ValueError("delta binning config missing")
                if db.mode == "inferred" and db.range is None:
                    db = self._infer_binning_range(db, prop_name=p_for_binning, kind="delta")
                if self._eval_override_enabled and self._eval_override_signed_delta_bin is not None:
                    signed_delta_bin = self._eval_override_signed_delta_bin
                    if not 0 <= signed_delta_bin < db.num_bins:
                        raise ValueError("signed_delta_bin override is outside the configured bins")
                elif use_randomized_tokens:
                    signed_delta_bin = token_random.randint(0, db.num_bins - 1)
                elif raw_delta is not None:
                    signed_delta_bin = signed_delta_to_bin(binning=db, delta=raw_delta)

            # --- Eval override: component dropping and delta_mode ---
            delta_mode: str = "exact"
            if self._eval_override_enabled:
                delta_mode = self._eval_override_delta_mode
                if not self._eval_override_inc_start:
                    start_bin = None
                if not self._eval_override_inc_dir:
                    direction = None
                if not self._eval_override_inc_mag:
                    delta_bin = None
                    signed_delta_bin = None
                # Skip component dropout and bounded delta (training-only)
            else:
                # --- Per-property component dropout ---
                if self._is_training and self._prop_component_dropout_enabled and not self._prop_randomize_tokens:
                    epoch_offset = self._current_epoch if self._prop_component_dropout_epoch_vary else 0
                    dr = random.Random(stable_prompt_seed("component_dropout", row_seed, epoch_offset, p))
                    rate = self._prop_component_dropout_rate
                    if dr.random() < rate:
                        start_bin = None
                    if dr.random() < rate:
                        direction = None
                    if dr.random() < rate:
                        delta_bin = None
                        signed_delta_bin = None

                # --- Bounded delta sampling ---
                # When enabled, a fraction of examples replace the exact delta
                # token with a bounded variant (atmost / atleast).  The bound is
                # relaxed from the true bin by a *geometric* offset so that most
                # bounds are tight (0-2 bins of slack) with exponentially
                # decreasing probability of looser bounds.  This matches how
                # bounded tokens are used at inference (users specify meaningful
                # bounds, not vacuous ones).
                if (self._is_training and self._prop_bounded_delta_enabled
                        and delta_bin is not None
                        and not self._prop_randomize_tokens):
                    if r.random() < self._prop_bounded_delta_fraction:
                        if self._prop_bounded_delta_mode == "both":
                            delta_mode = r.choice(["atmost", "atleast"])
                        else:
                            delta_mode = self._prop_bounded_delta_mode

                        db = self._delta_binning_per_prop.get(p, self._delta_binning_default)
                        max_bin = db.num_bins - 1

                        # Geometric offset: P(offset=k) = 0.5^(k+1), capped at
                        # available range.  Mean offset ≈ 1 bin; ~50% are within
                        # 1 bin, ~75% within 2, ~87.5% within 3.
                        if delta_mode == "atmost":
                            remaining = max_bin - delta_bin
                            offset = 0
                            while offset < remaining and r.random() < 0.5:
                                offset += 1
                            delta_bin = delta_bin + offset
                        elif delta_mode == "atleast":
                            remaining = delta_bin
                            offset = 0
                            while offset < remaining and r.random() < 0.5:
                                offset += 1
                            delta_bin = delta_bin - offset

            out.append(ParsedPropertyPrompt(
                property_name=p, start_bin=start_bin, direction=direction,
                delta_bin=delta_bin, delta_mode=delta_mode, signed_delta_bin=signed_delta_bin,
                raw_direction=raw_direction, raw_delta=raw_delta,
                raw_seed_value=float(v1) if v1 is not None else None,
                raw_target_value=float(v2) if v2 is not None else None,
            ))
        return out

    def _get_properties_for_smiles(self, smiles: Optional[str], prop_names: Sequence[str]) -> dict[str, float | None]:
        if not smiles:
            return {p: None for p in prop_names}
        cached = self._prop_cache.setdefault(smiles, {})
        missing = [p for p in prop_names if p not in cached]
        if not missing:
            return {p: cached[p] for p in prop_names}

        # Try columns-backed table first (if built)
        if self._prop_from_columns is None and self._prop_prefer_columns and not self.use_multi_file:
            self._prop_from_columns = self._build_property_table_from_columns()

        row = self._prop_from_columns.get(smiles, None) if self._prop_from_columns else None
        out: dict[str, float | None] = {}
        for p in missing:
            if row is not None and p in row and row[p] is not None:
                out[p] = row[p]
                continue
            if not self._prop_compute_if_missing:
                out[p] = None
                continue
            try:
                out[p] = self._compute_property_rdkit(smiles, p)
            except ValueError:
                # Property cannot be computed (e.g., "Class" must come from columns)
                out[p] = None

        cached.update(out)
        return {p: cached[p] for p in prop_names}

    def _build_property_table_from_columns(self) -> dict[str, dict[str, float | None]]:
        """
        Build a mapping smiles -> {prop_name: value} from molecules parquet columns (if present).

        This is optional and best-effort; missing columns are simply skipped.
        Falls back to fallback_column_map columns when primary column values are null.
        """
        import polars

        smiles_col = "smiles"
        prop_to_col = {p: self._prop_column_map.get(p) or p for p in self._prop_names}
        # Also include fallback columns
        prop_to_fallback_col = {p: self._prop_fallback_column_map.get(p, None) for p in self._prop_names}
        fallback_cols = [c for c in prop_to_fallback_col.values() if c is not None]
        
        # Build list of columns to request
        requested_cols = [smiles_col] + sorted(set(list(prop_to_col.values()) + fallback_cols))

        try:
            # Try reading with requested columns first
            df = _read_tabular_file(self.molecules_parquet_path, columns=requested_cols)
        except Exception:
            # If that fails (e.g., some columns don't exist), read all columns and filter
            try:
                df = _read_tabular_file(self.molecules_parquet_path)
                # Filter to only requested columns that exist
                available_cols = [c for c in requested_cols if c in df.columns]
                if smiles_col not in available_cols:
                    return {}
                df = df.select(available_cols)
            except Exception:
                return {}
        
        if smiles_col not in df.columns:
            return {}

        # Filter to only columns that actually exist in the parquet file
        present_props = {p: c for p, c in prop_to_col.items() if c in df.columns}
        present_fallback_cols = {p: c for p, c in prop_to_fallback_col.items() if c is not None and c in df.columns}
        if not present_props:
            return {}

        out: dict[str, dict[str, float | None]] = {}
        for row in df.iter_rows(named=True):
            smi = row.get(smiles_col, None)
            if not smi or smi in out:
                continue
            rec: dict[str, float | None] = {}
            for p, c in present_props.items():
                v = row.get(c, None)
                # Check fallback column if primary value is None
                if v is None and p in present_fallback_cols:
                    fallback_col = present_fallback_cols[p]
                    v = row.get(fallback_col, None)
                rec[p] = float(v) if v is not None else None
            out[str(smi)] = rec
        return out

    def _compute_property_rdkit(self, smiles: str, prop_name: str) -> float | None:
        """
        Compute a medchem property using RDKit (and local SA scorer).

        We intentionally do NOT import the callback property registry here, because it may
        load external evaluators at import time (not suitable for training data workers).
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Descriptors, QED
            from pairit.callbacks import sascorer

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            if prop_name == "logp":
                return float(Crippen.MolLogP(mol))
            if prop_name == "qed":
                return float(QED.qed(mol))
            if prop_name == "sa":
                return float(sascorer.calculateScore(mol))
            if prop_name == "mw":
                return float(Descriptors.MolWt(mol))
            if prop_name == "tpsa":
                return float(Descriptors.TPSA(mol))
            if prop_name == "hbd":
                return float(Descriptors.NumHDonors(mol))
            if prop_name == "hba":
                return float(Descriptors.NumHAcceptors(mol))
            if prop_name == "rotatable_bonds":
                return float(Descriptors.NumRotatableBonds(mol))
            if prop_name == "frac_sp3_c":
                return float(Descriptors.FractionCSP3(mol))
        except Exception:
            return None
        raise ValueError(f"Unknown property name for prompting: {prop_name!r}")

    def _infer_binning_range(self, binning: PropertyBinningConfig, *, prop_name: str, kind: str) -> PropertyBinningConfig:
        """
        Best-effort range inference for mode='inferred' when range is not provided.
        """
        # Simple heuristic: compute min/max over up to N molecules from columns (if available),
        # else compute on a subset of smiles with RDKit.
        N = 50_000
        vals: list[float] = []

        # Prefer columns if available
        if self._prop_from_columns is None and self._prop_prefer_columns and not self.use_multi_file:
            self._prop_from_columns = self._build_property_table_from_columns()
        if self._prop_from_columns:
            for i, (smi, rec) in enumerate(self._prop_from_columns.items()):
                v = rec.get(prop_name, None)
                if v is not None:
                    vals.append(float(v))
                if i >= N:
                    break
        else:
            # Fallback to RDKit on a subset of known smiles
            if self.use_multi_file:
                # Multi-file inference can be expensive and complicated; require explicit ranges.
                rng = (0.0, 1.0)
                return PropertyBinningConfig(num_bins=binning.num_bins, range=rng, mode=binning.mode, clip=binning.clip)
            smiles_pool: list[str] = []
            if self.ignore_pairs:
                smiles_pool = list(self._smiles_list)[:N]
            else:
                # sample from pairs
                for i, (s1, s2, _u, _) in enumerate(self.training_pairs[:N]):  # type: ignore[index]
                    smiles_pool.append(s1)
                    if s2:
                        smiles_pool.append(s2)
                smiles_pool = smiles_pool[:N]
            for smi in smiles_pool:
                v = self._compute_property_rdkit(smi, prop_name)
                if v is not None:
                    vals.append(float(v))

        if not vals:
            # Fallback: use a conservative dummy range
            rng = (0.0, 1.0) if kind == "value" else (0.0, 1.0)
            return PropertyBinningConfig(num_bins=binning.num_bins, range=rng, mode=binning.mode, clip=binning.clip)

        lo = float(min(vals))
        hi = float(max(vals))
        if not (hi > lo):
            hi = lo + 1e-6
        return PropertyBinningConfig(num_bins=binning.num_bins, range=(lo, hi), mode=binning.mode, clip=binning.clip)

    def _load_smiles_from_molecules(self) -> List[str]:
        """Load SMILES strings from molecules parquet file(s), optionally filtering by split.
        Used when ignore_pairs=True.
        """
        import polars
        smiles: List[str] = []
        if self.use_multi_file:
            paths = self.molecule_paths
        else:
            paths = [self.molecules_parquet_path]
        for mol_path in paths:
            df = _read_tabular_file(mol_path)
            if self.split is not None:
                if "split" in df.columns:
                    df = df.filter(polars.col("split") == self.split)
                elif self.split in df.columns:
                    # Handle boolean split columns if present
                    df = df.filter(polars.col(self.split) == True)
            if self.filter_column is not None and self.filter_value is not None:
                if self.filter_column in df.columns:
                    df = df.filter(polars.col(self.filter_column) == self.filter_value)
                else:
                    print(f"Warning: filter_column '{self.filter_column}' not found in {mol_path}")
            smiles.extend(df["smiles"].to_list())
        return smiles

    def _get_smiles_item(self, idx: int) -> str:
        """Get single SMILES string for ignore_pairs mode."""
        return self._smiles_list[idx]
    
    def _get_multi_file_item(self, idx: int):
        """Get item from multi-file dataset using row group caching or in-memory tables."""
        # If we have filtered indices (from max_num_occurrences), use them
        if hasattr(self, '_filtered_indices'):
            file_idx, local_idx = self._filtered_indices[idx]
        else:
            file_idx, local_idx = self._get_file_and_local_index(idx)
        
        file_info = self.file_metadata[file_idx]
        
        # If load_all_in_memory, use cached tables
        if self.load_all_in_memory:
            pair_table = file_info['pair_table_cached']
            mol_table = file_info['mol_table_cached']
            
            # Extract molecule indices
            mol_a_idx = pair_table['molecule_a_idx'][local_idx].as_py()
            mol_b_idx = pair_table['molecule_b_idx'][local_idx].as_py()
            
            # Get SMILES strings
            smiles_a = mol_table['smiles'][mol_a_idx].as_py()
            smiles_b = mol_table['smiles'][mol_b_idx].as_py()
        else:
            # Use row group caching
            # Get row group and local index within that group
            row_group_idx, local_row_idx = self._get_row_group_and_local_idx(file_idx, local_idx)
            
            # Load row group (cached)
            row_group = self._load_row_group(file_idx, row_group_idx)
            
            # Get the specific row from the row group
            pair_row = row_group.take([local_row_idx])
            
            # Extract molecule indices
            mol_a_idx = pair_row['molecule_a_idx'][0].as_py()
            mol_b_idx = pair_row['molecule_b_idx'][0].as_py()
            
            # Get SMILES strings
            # Cache processed mol_table if not already cached (for utility_key_fallback)
            if 'mol_table_cached' not in file_info:
                mol_table = file_info['mol_parquet'].read()
                # Apply utility_key_fallback if provided
                if self.utility_key_fallback is not None and self.utility_key_fallback != 'None':
                    import polars
                    mol_df = polars.from_arrow(mol_table)
                    mol_df = self._apply_utility_fallback(mol_df)
                    mol_table = mol_df.to_arrow()
                file_info['mol_table_cached'] = mol_table
            else:
                mol_table = file_info['mol_table_cached']
            
            smiles_a = mol_table['smiles'][mol_a_idx].as_py()
            smiles_b = mol_table['smiles'][mol_b_idx].as_py()
        
        # Handle null utility_key case (deterministic ordering based on hash)
        if self.utility_key is None or self.utility_key == 'None':
            # Use hash for deterministic but varied ordering
            if hash((smiles_a, smiles_b)) % 2 == 0:
                return (smiles_a, smiles_b, 0.0)
            else:
                return (smiles_b, smiles_a, 0.0)
        
        # Get utilities
        utility_a = mol_table[self.utility_key][mol_a_idx].as_py()
        utility_b = mol_table[self.utility_key][mol_b_idx].as_py()
        if self.utility_negate:
            utility_a, utility_b = -utility_a, -utility_b

        # Calculate utility improvement
        utility_improvement = utility_b - utility_a
        
        # Apply utility direction logic
        if self.utility_direction == "increasing":
            if utility_improvement > 0:
                return (smiles_a, smiles_b, utility_improvement)
            else:
                return (smiles_b, smiles_a, -utility_improvement)
        else:  # decreasing
            if utility_improvement < 0:
                return (smiles_a, smiles_b, -utility_improvement)
            else:
                return (smiles_b, smiles_a, utility_improvement)

    def cleanup_cache(self) -> None:
        """
        Clean up the cache directory for this dataset.
        
        Call this after training is complete if you want to free up disk space.
        Only rank 0 performs the cleanup in DDP scenarios.
        
        Example:
            dataset.cleanup_cache()
        """
        if not self._is_rank0():
            return
        
        import shutil
        if self.dataset_cache_dir.exists():
            try:
                shutil.rmtree(self.dataset_cache_dir)
                print(f"Cleaned up cache directory: {self.dataset_cache_dir}")
            except Exception as e:
                print(f"Warning: failed to clean up cache directory {self.dataset_cache_dir}: {e}")
    
    @staticmethod
    def cleanup_all_caches(cache_dir: Optional[str] = None) -> None:
        """
        Clean up all pairit dataset caches.
        
        Args:
            cache_dir: Cache directory to clean (default: /tmp/lobster_cache)
        
        Example:
            MoleculeImprovementDataset.cleanup_all_caches()
            # Or with custom cache dir:
            MoleculeImprovementDataset.cleanup_all_caches("/path/to/cache")
        """
        import shutil
        import tempfile
        
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "lobster_cache"
        else:
            cache_dir = Path(cache_dir)

        print(f"Cleaning up all caches in: {cache_dir}")
        # Ask for user confirmation before deleting
        response = input(f"Are you sure you want to delete all caches in {cache_dir}? [y/N] ")
        if response.lower() != 'y':
            print("Cache cleanup cancelled")
            return
        
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"Cleaned up all caches in: {cache_dir}")
            except Exception as e:
                print(f"Warning: failed to clean up cache directory {cache_dir}: {e}")

    @staticmethod
    def find_molecule_pairs_static(
        input_smiles: str,
        molecules_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
        smiles_column: str = "smiles",
        molecule_a_idx_column: str = "molecule_a_idx",
        molecule_b_idx_column: str = "molecule_b_idx"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Static method to find pairs and molecules related to an input SMILES string.
        
        Args:
            input_smiles: The SMILES string to search for
            molecules_df: DataFrame containing molecule data with SMILES and indices
            pairs_df: DataFrame containing molecule pairs with indices
            smiles_column: Name of the SMILES column in molecules_df
            molecule_a_idx_column: Name of the first molecule index column in pairs_df
            molecule_b_idx_column: Name of the second molecule index column in pairs_df
            
        Returns:
            Tuple of (relevant_pairs_df, relevant_molecules_df):
            - relevant_pairs_df: Subset of pairs_df containing pairs with the input SMILES
            - relevant_molecules_df: Subset of molecules_df containing molecules that appear in the pairs
        """
        # Find the molecule index for the input SMILES
        molecule_matches = molecules_df[molecules_df[smiles_column] == input_smiles]
        
        if len(molecule_matches) == 0:
            # Return empty DataFrames if SMILES not found
            return pd.DataFrame(), pd.DataFrame()
        
        # Get the molecule index from the item_id column (matching the pattern from _load_and_filter_pairs)
        if "item_id" in molecules_df.columns:
            molecule_idx = molecule_matches["item_id"].iloc[0]
        else:
            # Fallback to pandas index if item_id doesn't exist
            molecule_idx = molecule_matches.index[0]
        
        # Find pairs where this molecule appears as either molecule_a or molecule_b
        relevant_pairs = pairs_df[
            (pairs_df[molecule_a_idx_column] == molecule_idx) | 
            (pairs_df[molecule_b_idx_column] == molecule_idx)
        ]
        
        if len(relevant_pairs) == 0:
            # Return empty DataFrames if no pairs found
            return pd.DataFrame(), pd.DataFrame()
        
        # Get all unique molecule indices from the relevant pairs
        all_molecule_indices = set()
        all_molecule_indices.update(relevant_pairs[molecule_a_idx_column].tolist())
        all_molecule_indices.update(relevant_pairs[molecule_b_idx_column].tolist())
        
        # Get the relevant molecules using the correct index column
        if "item_id" in molecules_df.columns:
            relevant_molecules = molecules_df[molecules_df["item_id"].isin(all_molecule_indices)]
        else:
            # Fallback to pandas index if item_id doesn't exist
            relevant_molecules = molecules_df[molecules_df.index.isin(all_molecule_indices)]
        
        return relevant_pairs, relevant_molecules

    def find_molecule_pairs(self, input_smiles: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Non-static method to find pairs and molecules related to an input SMILES string.
        Uses the instance's loaded data.
        
        Args:
            input_smiles: The SMILES string to search for
            
        Returns:
            Tuple of (relevant_pairs_df, relevant_molecules_df):
            - relevant_pairs_df: Subset of pairs_df containing pairs with the input SMILES
            - relevant_molecules_df: Subset of molecules_df containing molecules that appear in the pairs
        """
        import polars
        
        # Load the molecules and pairs dataframes
        molecules_df = _read_tabular_file(self.molecules_parquet_path)
        pairs_df = polars.read_parquet(self.pairs_parquet_path)
        
        # Convert to pandas for easier manipulation
        molecules_pandas = molecules_df.to_pandas()
        pairs_pandas = pairs_df.to_pandas()
        
        # Add item_id column if it doesn't exist (matching the pattern from _load_and_filter_pairs)
        if "item_id" not in molecules_pandas.columns:
            molecules_pandas["item_id"] = range(len(molecules_pandas))
        
        # Use the static method
        relevant_pairs, relevant_molecules = self.find_molecule_pairs_static(
            input_smiles=input_smiles,
            molecules_df=molecules_pandas,
            pairs_df=pairs_pandas,
            smiles_column="smiles",
            molecule_a_idx_column="molecule_a_idx",
            molecule_b_idx_column="molecule_b_idx"
        )
        
        return relevant_pairs, relevant_molecules


class MoleculeImprovementLightningDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for molecule improvement training.
    
    This follows pairit's conventions and integrates with the existing training pipeline.
    """
    
    def __init__(
        self,
        root: str,  # Data root directory
        molecule_filename: Union[str, List[str]] = "molecules.parquet", # name(s) of the molecules file(s) within root
        train_pair_filename: Union[str, List[str]] = "pairs_train.parquet", # name(s) of the pairs file(s) within root
        val_pair_filename: Union[str, List[str]] = "pairs_val.parquet", # name(s) of the pairs file(s) within root
        test_pair_filename: Optional[Union[str, List[str]]] = None,  # name(s) of the pairs file(s) within root (optional)
        val_molecule_filename: Optional[Union[str, List[str]]] = None,  # name(s) of val molecules file(s) (optional, defaults to molecule_filename)
        test_molecule_filename: Optional[Union[str, List[str]]] = None,  # name(s) of test molecules file(s) (optional, defaults to molecule_filename)
        ignore_pairs: bool = False,  # If True, ignore pairs and load single molecules from molecules parquet
        seeds_only_conditional: bool = False,  # If True with ignore_pairs, append <sep> for conditional generation
        filter_column: str | None = None,  # Column name to filter molecules (e.g., "Class")
        filter_value: float | None = None,  # Value to filter on (e.g., 0)
        utility_key: str = "utility_0",  # Which utility column to use
        utility_key_fallback: Optional[str] = None,  # Fallback utility column to use when utility_key is None/null/False
        delta: float = 0.3,
        epsilon: float = 0.1,
        utility_threshold: Optional[float] = None,
        transform_fn: Optional[Callable] = None,  # Tokenizer transform function
        shape_tanimoto_percentile: Optional[float] = None,  # Keep closest X% of pairs by shape Tanimoto distance per molecule
        shape_tanimoto_num_pairs: Optional[int] = None,  # Keep closest N pairs by shape Tanimoto distance per molecule
        distance_type: Optional[str] = None,  # "morgan_tanimoto_distance" or "shape_tanimoto_distance"
        use_cache: bool = True,  # Whether to use cached filtered pairs if available,
        duplicate: bool = False,  # Whether to duplicate instead of using the second molecule, for debugging purposes
        just_first: bool = False,  # Whether to just use the first molecule, for debugging purposes
        utility_direction: str = "increasing",  # "increasing" or "decreasing" - whether second molecule should have increasing or decreasing utility
        max_cached_row_groups: int = 10,  # Maximum number of row groups to cache for multi-file scenarios
        cache_dir: Optional[str] = None,  # Directory for cache files (default: /tmp/lobster_cache)
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        max_train_samples: Optional[Union[int, float]] = None,
        load_all_in_memory: bool = False,
        max_num_occurrences: Optional[int] = None,
        max_num_occurrences_criterion: Optional[str] = "first",  # Criterion for selecting pairs: "first", "min_distance", "min_shape_tanimoto"
        debug_remix_splits: bool = False,  # Mix train/val and re-divide randomly for debugging
        debug_remix_seed: int = 42,  # Seed for reproducible remixing
        dataset_id: int = 0,  # Dataset identity integer used for prompting
        use_dataset_identity_prompt: bool = False,  # If True, prefix sequences with dataset identity tokens
        prompting: Optional[dict] = None,  # Optional property prompting config (see hydra plan)
        max_pairs_to_load_train: Optional[int] = None,
        max_pairs_to_load_val: Optional[int] = None,
        train_subset_fraction: Optional[Union[float, int]] = None,  # If int >= 1, treated as max_samples; if float 0-1, treated as fraction
        val_subset_fraction: Optional[Union[float, int]] = None,  # If int >= 1, treated as max_samples; if float 0-1, treated as fraction
        train_subset_seed: int = 0,
        val_subset_seed: int = 0,
        keep_original_ordering: bool = False,  # If True, preserve the ordering of pairs from the parquet file, overriding epsilon-based ordering or other behavior
    ) -> None:
        """
        Initialize the MoleculeImprovementLightningDataModule.
        
        Args:
            root: Root directory containing molecules.parquet and pairs.parquet
            molecule_filename: Name of the molecules file within root
            train_pair_filename: Name of the train pairs file within root
            val_pair_filename: Name of the val pairs file within root
            test_pair_filename: Name of the test pairs file within root
            utility_key: Which utility column to use (e.g., "utility_0", "gap", "homo", etc.)
            utility_key_fallback: Optional fallback utility column to use when utility_key is None/null/False
            delta: Distance threshold for molecule pairs
            epsilon: Utility improvement threshold
            transform_fn: Tokenizer transform function
            use_cache: Whether to use cached filtered pairs if available
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            max_train_samples: Maximum number of training samples to use (for debugging).
                Can be an integer > 1 (absolute count) or a float between 0 and 1 (fraction of dataset).
                If float, it will be converted to integer by rounding (fraction * dataset_length).
            load_all_in_memory: Whether to load all pairs into memory at initialization
            max_num_occurrences: If provided, only include up to this many pairs for any given first SMILES string after ordering (1 enforces uniqueness)
            max_num_occurrences_criterion: Criterion for selecting pairs when max_num_occurrences is set:
                - "first": Take the first N pairs (default, current behavior)
                - "min_distance": Select N pairs with minimum distance (uses distance_type)
                - "min_shape_tanimoto": Select N pairs with minimum shape_tanimoto_distance
            debug_remix_splits: If True, combine train and val datasets, shuffle, and re-split randomly (for debugging distribution shift)
            debug_remix_seed: Random seed for reproducible remixing when debug_remix_splits is True
            keep_original_ordering: If True, preserve the ordering of pairs from the parquet file, overriding epsilon-based ordering or other behavior
        """
        super().__init__()

        # Validate shape Tanimoto filtering parameters
        if shape_tanimoto_percentile is not None and shape_tanimoto_num_pairs is not None:
            raise ValueError("Cannot specify both shape_tanimoto_percentile and shape_tanimoto_num_pairs. Use one or the other.")

        self._root = root
        self._molecule_filename = molecule_filename
        self._val_molecule_filename = val_molecule_filename if val_molecule_filename is not None else molecule_filename
        self._test_molecule_filename = test_molecule_filename if test_molecule_filename is not None else molecule_filename
        self._train_pair_filename = train_pair_filename
        self._val_pair_filename = val_pair_filename
        self._test_pair_filename = test_pair_filename
        self._utility_key = utility_key
        self._utility_key_fallback = utility_key_fallback
        self._ignore_pairs = ignore_pairs
        self._seeds_only_conditional = seeds_only_conditional
        self._filter_column = filter_column
        self._filter_value = filter_value
        self._delta = delta
        self._epsilon = epsilon
        self._utility_threshold = utility_threshold
        self._transform_fn = transform_fn
        self._shape_tanimoto_percentile = shape_tanimoto_percentile
        self._shape_tanimoto_num_pairs = shape_tanimoto_num_pairs
        self._distance_type = distance_type
        self._use_cache = use_cache
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._max_train_samples = max_train_samples
        self._duplicate = duplicate
        self._just_first = just_first
        self._utility_direction = utility_direction
        self._max_cached_row_groups = max_cached_row_groups
        self._cache_dir = cache_dir
        self._load_all_in_memory = load_all_in_memory
        self._max_num_occurrences = max_num_occurrences
        self._max_num_occurrences_criterion = max_num_occurrences_criterion if max_num_occurrences_criterion is not None else "first"
        self._debug_remix_splits = debug_remix_splits
        self._debug_remix_seed = debug_remix_seed
        self._dataset_id = int(dataset_id)
        self._use_dataset_identity_prompt = bool(use_dataset_identity_prompt)
        self._prompting = prompting or {}
        self._max_pairs_to_load_train = max_pairs_to_load_train
        self._max_pairs_to_load_val = max_pairs_to_load_val
        
        # Parse train_subset_fraction: if int >= 1, treat as max_samples; if float 0-1, treat as fraction
        self._train_subset_fraction = None
        self._train_subset_max_samples = None
        if train_subset_fraction is not None:
            if isinstance(train_subset_fraction, int) and train_subset_fraction >= 1:
                # Integer >= 1: treat as max_samples for controlled dataset size
                self._train_subset_max_samples = train_subset_fraction
            elif isinstance(train_subset_fraction, float) and 0 < train_subset_fraction <= 1:
                # Float 0-1: treat as fraction
                self._train_subset_fraction = train_subset_fraction
            else:
                raise ValueError(f"train_subset_fraction must be either an int >= 1 (max_samples) or float 0-1 (fraction), got {train_subset_fraction} (type: {type(train_subset_fraction)})")
        
        # Parse val_subset_fraction: if int >= 1, treat as max_samples; if float 0-1, treat as fraction
        self._val_subset_fraction = None
        self._val_subset_max_samples = None
        if val_subset_fraction is not None:
            if isinstance(val_subset_fraction, int) and val_subset_fraction >= 1:
                # Integer >= 1: treat as max_samples for controlled dataset size
                self._val_subset_max_samples = val_subset_fraction
            elif isinstance(val_subset_fraction, float) and 0 < val_subset_fraction <= 1:
                # Float 0-1: treat as fraction
                self._val_subset_fraction = val_subset_fraction
            else:
                raise ValueError(f"val_subset_fraction must be either an int >= 1 (max_samples) or float 0-1 (fraction), got {val_subset_fraction} (type: {type(val_subset_fraction)})")
        
        self._train_subset_seed = int(train_subset_seed)
        self._val_subset_seed = int(val_subset_seed)
        self._keep_original_ordering = keep_original_ordering
        
        # Initialize datasets
        self._train_dataset: Union[MoleculeImprovementDataset, None] = None
        self._val_dataset: Union[MoleculeImprovementDataset, None] = None
        self._test_dataset: Union[MoleculeImprovementDataset, None] = None
        
    def prepare_data(self) -> None:
        """Prepare the dataset by creating training pairs."""
        # This is called once per node
        pass
        
    def setup(self, stage: str = "fit") -> None:
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage == "test":
            # Create datasets for each split
            # Determine which subset_max_samples to use (take minimum if both are provided)
            if self._train_subset_max_samples is not None and self._max_pairs_to_load_train is not None:
                train_subset_max_samples = min(self._train_subset_max_samples, self._max_pairs_to_load_train)
                print(f"[DATAMODULE] Both train_subset_fraction (as int={self._train_subset_max_samples}) and max_pairs_to_load_train ({self._max_pairs_to_load_train}) provided. Using minimum: {train_subset_max_samples}")
            else:
                train_subset_max_samples = self._train_subset_max_samples if self._train_subset_max_samples is not None else self._max_pairs_to_load_train
            
            if self._train_subset_fraction is not None:
                print(f"[DATAMODULE] Creating train dataset with subset_fraction={self._train_subset_fraction}, subset_seed={self._train_subset_seed}")
            elif self._train_subset_max_samples is not None:
                print(f"[DATAMODULE] Creating train dataset with subset_max_samples={self._train_subset_max_samples} (from train_subset_fraction as integer), subset_seed={self._train_subset_seed}")
            elif self._max_pairs_to_load_train is not None:
                print(f"[DATAMODULE] Creating train dataset with subset_max_samples={self._max_pairs_to_load_train}")
            
            self._train_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._train_pair_filename,
                molecule_filename=self._molecule_filename,
                ignore_pairs=self._ignore_pairs,
                split="train" if self._ignore_pairs else None,
                utility_key=self._utility_key,
                utility_key_fallback=self._utility_key_fallback,
                delta=self._delta,
                epsilon=self._epsilon,
                utility_threshold=self._utility_threshold,
                transform_fn=self._transform_fn,
                shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
                distance_type=self._distance_type,
                use_cache=self._use_cache,
                duplicate=self._duplicate,
                just_first=self._just_first,
                utility_direction=self._utility_direction,
                max_cached_row_groups=self._max_cached_row_groups,
                cache_dir=self._cache_dir,
                load_all_in_memory=self._load_all_in_memory,
                max_num_occurrences=self._max_num_occurrences,
                max_num_occurrences_criterion=self._max_num_occurrences_criterion,
                dataset_id=self._dataset_id,
                use_dataset_identity_prompt=self._use_dataset_identity_prompt,
                prompting=self._prompting,
                subset_fraction=self._train_subset_fraction,
                subset_max_samples=train_subset_max_samples,
                subset_seed=self._train_subset_seed,
                keep_original_ordering=self._keep_original_ordering,
            )

            if self._max_train_samples is not None:
                # Convert fractional value (0-1) to integer count if needed
                if isinstance(self._max_train_samples, float):
                    if not (0 < self._max_train_samples <= 1):
                        raise ValueError(f"max_train_samples as float must be between 0 and 1, got {self._max_train_samples}")
                    max_samples = round(self._max_train_samples * len(self._train_dataset))
                else:
                    if self._max_train_samples <= 0:
                        raise ValueError(f"max_train_samples as integer must be > 0, got {self._max_train_samples}")
                    max_samples = self._max_train_samples
                
                self._train_dataset, _ = random_split(self._train_dataset, [max_samples, len(self._train_dataset) - max_samples])
            
            # Determine which subset_max_samples to use (take minimum if both are provided)
            if self._val_subset_max_samples is not None and self._max_pairs_to_load_val is not None:
                val_subset_max_samples = min(self._val_subset_max_samples, self._max_pairs_to_load_val)
                print(f"[DATAMODULE] Both val_subset_fraction (as int={self._val_subset_max_samples}) and max_pairs_to_load_val ({self._max_pairs_to_load_val}) provided. Using minimum: {val_subset_max_samples}")
            else:
                val_subset_max_samples = self._val_subset_max_samples if self._val_subset_max_samples is not None else self._max_pairs_to_load_val
            
            if self._val_subset_fraction is not None:
                print(f"[DATAMODULE] Creating val dataset with subset_fraction={self._val_subset_fraction}, subset_seed={self._val_subset_seed}")
            elif self._val_subset_max_samples is not None:
                print(f"[DATAMODULE] Creating val dataset with subset_max_samples={self._val_subset_max_samples} (from val_subset_fraction as integer), subset_seed={self._val_subset_seed}")
            elif self._max_pairs_to_load_val is not None:
                print(f"[DATAMODULE] Creating val dataset with subset_max_samples={self._max_pairs_to_load_val}")
            
            # When seeds_only_conditional, files are already pre-split by directory;
            # the split column inside may not match, so skip split filtering.
            val_split = None if self._seeds_only_conditional else ("val" if self._ignore_pairs else None)
            self._val_dataset = MoleculeImprovementDataset(
                root=self._root,
                is_training=False,
                pair_filename=self._val_pair_filename,
                molecule_filename=self._val_molecule_filename,
                ignore_pairs=self._ignore_pairs,
                seeds_only_conditional=self._seeds_only_conditional,
                filter_column=self._filter_column,
                filter_value=self._filter_value,
                split=val_split,
                utility_key=self._utility_key,
                utility_key_fallback=self._utility_key_fallback,
                delta=self._delta,
                epsilon=self._epsilon,
                utility_threshold=self._utility_threshold,
                transform_fn=self._transform_fn,
                shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
                distance_type=self._distance_type,
                use_cache=self._use_cache,
                duplicate=self._duplicate,
                just_first=self._just_first,
                utility_direction=self._utility_direction,
                max_cached_row_groups=self._max_cached_row_groups,
                cache_dir=self._cache_dir,
                load_all_in_memory=self._load_all_in_memory,
                max_num_occurrences=None,
                dataset_id=self._dataset_id,
                use_dataset_identity_prompt=self._use_dataset_identity_prompt,
                prompting=self._prompting,
                subset_fraction=self._val_subset_fraction,
                subset_max_samples=val_subset_max_samples,
                subset_seed=self._val_subset_seed,
                keep_original_ordering=self._keep_original_ordering,
            )

            
            # Debug: Remix and re-split datasets to check for distribution shift
            if self._debug_remix_splits:
                self._train_dataset, self._val_dataset = self._remix_and_resplit(
                    self._train_dataset, 
                    self._val_dataset,
                    seed=self._debug_remix_seed
                )
                print(f"[DEBUG] Remixed train/val splits with seed {self._debug_remix_seed}")
                print(f"[DEBUG] New train size: {len(self._train_dataset)}, val size: {len(self._val_dataset)}")
            
            # Only create test dataset if test filename is provided
            test_split = None if self._seeds_only_conditional else ("test" if self._ignore_pairs else None)
            if self._test_pair_filename is not None:
                self._test_dataset = MoleculeImprovementDataset(
                    root=self._root,
                    is_training=False,
                    pair_filename=self._test_pair_filename,
                    molecule_filename=self._test_molecule_filename,
                    ignore_pairs=self._ignore_pairs,
                    seeds_only_conditional=self._seeds_only_conditional,
                    filter_column=self._filter_column,
                    filter_value=self._filter_value,
                    split=test_split,
                    utility_key=self._utility_key,
                    utility_key_fallback=self._utility_key_fallback,
                    delta=self._delta,
                    epsilon=self._epsilon,
                    utility_threshold=self._utility_threshold,
                    transform_fn=self._transform_fn,
                    shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                    shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
                    distance_type=self._distance_type,
                    use_cache=self._use_cache,
                    duplicate=self._duplicate,
                    just_first=self._just_first,
                    utility_direction=self._utility_direction,
                    max_cached_row_groups=self._max_cached_row_groups,
                    load_all_in_memory=self._load_all_in_memory,
                    max_num_occurrences=None,
                    dataset_id=self._dataset_id,
                    use_dataset_identity_prompt=self._use_dataset_identity_prompt,
                    prompting=self._prompting,
                    keep_original_ordering=self._keep_original_ordering,
                )
            else:
                self._test_dataset = None
            
        if stage == "predict":
            # For prediction, use the test dataset if available, otherwise use val
            predict_test_split = None if self._seeds_only_conditional else ("test" if self._ignore_pairs else None)
            if self._test_pair_filename is not None:
                self._test_dataset = MoleculeImprovementDataset(
                    root=self._root,
                    is_training=False,
                    pair_filename=self._test_pair_filename,
                    molecule_filename=self._test_molecule_filename,
                    ignore_pairs=self._ignore_pairs,
                    seeds_only_conditional=self._seeds_only_conditional,
                    filter_column=self._filter_column,
                    filter_value=self._filter_value,
                    split=predict_test_split,
                    utility_key=self._utility_key,
                    utility_key_fallback=self._utility_key_fallback,
                    delta=self._delta,
                    epsilon=self._epsilon,
                    utility_threshold=self._utility_threshold,
                    transform_fn=self._transform_fn,
                    shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                    shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
                    distance_type=self._distance_type,
                    use_cache=self._use_cache,
                    duplicate=self._duplicate,
                    just_first=self._just_first,
                    utility_direction=self._utility_direction,
                    max_cached_row_groups=self._max_cached_row_groups,
                    cache_dir=self._cache_dir,
                    load_all_in_memory=self._load_all_in_memory,
                    max_num_occurrences=None,
                    dataset_id=self._dataset_id,
                    use_dataset_identity_prompt=self._use_dataset_identity_prompt,
                    prompting=self._prompting,
                    keep_original_ordering=self._keep_original_ordering,
                )
            else:
                # Use validation dataset for prediction if no test dataset
                predict_val_split = None if self._seeds_only_conditional else ("val" if self._ignore_pairs else None)
                self._test_dataset = MoleculeImprovementDataset(
                    root=self._root,
                    is_training=False,
                    pair_filename=self._val_pair_filename,
                    molecule_filename=self._val_molecule_filename,
                    ignore_pairs=self._ignore_pairs,
                    split=predict_val_split,
                    utility_key=self._utility_key,
                    utility_key_fallback=self._utility_key_fallback,
                    delta=self._delta,
                    epsilon=self._epsilon,
                    utility_threshold=self._utility_threshold,
                    transform_fn=self._transform_fn,
                    shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                    shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
                    distance_type=self._distance_type,
                    use_cache=self._use_cache,
                    duplicate=self._duplicate,
                    just_first=self._just_first,
                    utility_direction=self._utility_direction,
                    max_cached_row_groups=self._max_cached_row_groups,
                    cache_dir=self._cache_dir,
                    load_all_in_memory=self._load_all_in_memory,
                    max_num_occurrences=None,
                    dataset_id=self._dataset_id,
                    use_dataset_identity_prompt=self._use_dataset_identity_prompt,
                    keep_original_ordering=self._keep_original_ordering,
                )
            
    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to train dataset for epoch-varying component dropout."""
        dataset = self._train_dataset
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self._train_dataset is None:
            raise ValueError("Train dataset not initialized. Call setup() first.")
        
        # Use spawn to avoid fork-safety issues with lru_cache locks and PyArrow objects
        mp_context = "spawn" if self._num_workers > 0 else None
        
        # Timeout to detect DataLoader hangs - will raise RuntimeError if exceeded
        timeout = 300 if self._num_workers > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"[TRAIN DATALOADER CONFIG]")
        print(f"  dataset_len:             {len(self._train_dataset):,}")
        print(f"  batch_size:              {self._batch_size}")
        print(f"  num_workers:             {self._num_workers}")
        print(f"  pin_memory:              {self._pin_memory}")
        print(f"  multiprocessing_context: {mp_context}")
        print(f"  persistent_workers:      {self._num_workers > 0}")
        print(f"  timeout:                 {timeout}s")
        print(f"")
        if timeout:
            print(f"  NOTE: Worker batches time out after {timeout} seconds")
        print(f"{'='*70}\n")
            
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            multiprocessing_context=mp_context,
            persistent_workers=self._num_workers > 0,  # Avoid repeated worker spawning
            timeout=timeout,
        )
        
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self._val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup() first.")
        
        # Use spawn to avoid fork-safety issues with lru_cache locks and PyArrow objects
        mp_context = "spawn" if self._num_workers > 0 else None
        
        # Timeout to detect DataLoader hangs - will raise RuntimeError if exceeded
        timeout = 300 if self._num_workers > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"[VAL DATALOADER CONFIG]")
        print(f"  dataset_len:             {len(self._val_dataset):,}")
        print(f"  batch_size:              {self._batch_size}")
        print(f"  num_workers:             {self._num_workers}")
        print(f"  pin_memory:              {self._pin_memory}")
        print(f"  multiprocessing_context: {mp_context}")
        print(f"  persistent_workers:      {self._num_workers > 0}")
        print(f"  timeout:                 {timeout}s")
        print(f"{'='*70}\n")
            
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
            multiprocessing_context=mp_context,
            persistent_workers=self._num_workers > 0,
            timeout=timeout,
        )
        
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self._test_dataset is None:
            # If no test dataset, return None (PyTorch Lightning will skip test)
            return None
        
        # Use spawn to avoid fork-safety issues with lru_cache locks and PyArrow objects
        mp_context = "spawn" if self._num_workers > 0 else None
        
        # Timeout to detect DataLoader hangs - will raise RuntimeError if exceeded
        timeout = 300 if self._num_workers > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"[TEST DATALOADER CONFIG]")
        print(f"  dataset_len:             {len(self._test_dataset):,}")
        print(f"  batch_size:              {self._batch_size}")
        print(f"  num_workers:             {self._num_workers}")
        print(f"  pin_memory:              {self._pin_memory}")
        print(f"  multiprocessing_context: {mp_context}")
        print(f"  persistent_workers:      {self._num_workers > 0}")
        print(f"  timeout:                 {timeout}s")
        print(f"{'='*70}\n")
            
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
            multiprocessing_context=mp_context,
            persistent_workers=self._num_workers > 0,
            timeout=timeout,
        )
        
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader."""
        if self._test_dataset is None:
            raise ValueError("Test dataset not initialized. Call setup() first.")
        
        # Use spawn to avoid fork-safety issues with lru_cache locks and PyArrow objects
        mp_context = "spawn" if self._num_workers > 0 else None
        
        # Timeout to detect DataLoader hangs - will raise RuntimeError if exceeded
        timeout = 300 if self._num_workers > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"[PREDICT DATALOADER CONFIG]")
        print(f"  dataset_len:             {len(self._test_dataset):,}")
        print(f"  batch_size:              {self._batch_size}")
        print(f"  num_workers:             {self._num_workers}")
        print(f"  pin_memory:              {self._pin_memory}")
        print(f"  multiprocessing_context: {mp_context}")
        print(f"  persistent_workers:      {self._num_workers > 0}")
        print(f"  timeout:                 {timeout}s")
        print(f"{'='*70}\n")
            
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
            multiprocessing_context=mp_context,
            persistent_workers=self._num_workers > 0,
            timeout=timeout,
        )
    
    def _remix_and_resplit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """
        Remix and re-split train and validation datasets for debugging distribution shift.
        
        This method combines the train and validation datasets, shuffles them randomly,
        and then re-splits them back into train and validation sets with the same sizes
        as the original splits. This is useful for debugging whether performance differences
        between train and validation are due to distribution shift or overfitting.
        
        Args:
            train_dataset: Original training dataset
            val_dataset: Original validation dataset
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (new_train_dataset, new_val_dataset) with shuffled and re-split data
        """
        from torch.utils.data import ConcatDataset, Subset
        
        print("Remixing and re-splitting train and validation datasets")
        # Concatenate both datasets
        combined_dataset = ConcatDataset([train_dataset, val_dataset])
        total_size = len(combined_dataset)
        
        # Create shuffled indices
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_size, generator=generator).tolist()
        
        # Calculate split sizes (maintain original proportions)
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create new datasets using Subset
        new_train_dataset = Subset(combined_dataset, train_indices)
        new_val_dataset = Subset(combined_dataset, val_indices)
        
        return new_train_dataset, new_val_dataset
