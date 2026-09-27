"""Shared utilities for inference-time compute strategies.

Extracted from ``pairit.cmdline.iterative_generate`` so that best-of-N,
beam search, the refilter tool, and the original iterative generator can
all reuse the same logic.

Key functions
-------------
- ``load_model(cfg)`` — load a CLM checkpoint with PyTorch 2.6+ compat
- ``setup_datamodule_and_dataloader(cfg)`` — instantiate data + get split
- ``extract_seeds_from_dataloader(...)`` — pull seed SMILES from paired data
- ``generate_from_seeds(...)`` — batched conditional generation via
  ``MoleculeGenerator.batch_generate_conditional``
- ``compute_properties(smiles_list, property_names)`` — RDKit property
  computation for a list of SMILES
- ``compute_tanimoto_distances(...)`` — Morgan fingerprint Tanimoto
- ``check_memorization_batch(...)`` — check against reference canonical sets
- ``build_scored_candidates(...)`` — full pipeline: validate → properties →
  memorization → improvement → distances → ``ScoredCandidate`` objects
- ``extract_from_eval_config(cfg)`` — pull reference files, properties,
  ranking property/direction, temperature from a ``pairit_eval`` config
- ``candidates_to_dataframe(...)`` — convert to pandas DataFrame
- ``save_results(...)`` — write parquet + optional YAML metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra.utils
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from pairit.callbacks.molecule_validation.distance_utils import (
    compute_distances_with_fingerprint_caching,
)
from pairit.callbacks.molecule_validation.generation import MoleculeGenerator
from pairit.callbacks.molecule_validation.property_registry import (
    PROPERTY_REGISTRY,
    compute_all_properties_for_smiles,
    get_property_specs,
    get_seed_mol_stats,
)
from pairit.callbacks.molecule_validation.smiles_utils import SmilesUtils
from pairit.cmdline._tokenizer_inheritance import (
    get_checkpoint_path_for_inheritance,
    prepare_tokenizer_inheritance,
)
from pairit.inference.scorers import ScoredCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference SMILES loading (for memorization checking)
# ---------------------------------------------------------------------------

def load_reference_smiles_files(
    reference_specs: list[dict],
    smiles_column: str = "smiles",
) -> tuple[dict[str, set[str]], str | None]:
    """Load reference SMILES files for memorization checking.

    Parameters
    ----------
    reference_specs : list[dict]
        Each dict has ``name``, ``path`` (or ``files``), and optional ``default``.
    smiles_column : str
        Column name for SMILES in the files.

    Returns
    -------
    reference_canon_sets : dict[str, set[str]]
        Mapping from reference name to set of canonical SMILES.
    default_reference_name : str | None
        Name of the default reference set.
    """
    reference_canon_sets: dict[str, set[str]] = {}
    default_reference_name: str | None = None

    if not reference_specs:
        return reference_canon_sets, default_reference_name

    for spec in reference_specs:
        name = spec.get("name")
        files = spec.get("path") or spec.get("files") or spec.get("file")
        is_default = spec.get("default", False)

        if not name or not files:
            continue

        if isinstance(files, str):
            files = [files]

        all_smiles: list[str] = []
        for f in files:
            all_smiles.extend(_load_smiles_from_file(f, smiles_column))

        canonical = _canonicalize_smiles_list(all_smiles)
        reference_canon_sets[name] = set(canonical)

        logger.info(
            f"  Loaded reference '{name}': {len(canonical)} canonical SMILES "
            f"from {len(files)} file(s)"
        )

        if is_default or default_reference_name is None:
            default_reference_name = name

    return reference_canon_sets, default_reference_name


def _canonicalize_smiles_list(smiles_list: list[str]) -> list[str]:
    canonical = []
    for s in smiles_list:
        if not s:
            continue
        try:
            c = SmilesUtils.canonicalize(s)
            if c:
                canonical.append(c)
        except Exception:
            pass
    return canonical


def _load_smiles_from_file(file_path: str, smiles_col: str) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"Reference file not found: {file_path}")
        return []

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in [".csv", ".tsv"]:
            sep = "\t" if path.suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            logger.warning(f"Unsupported file format: {path.suffix}")
            return []

        col = smiles_col
        if col not in df.columns:
            for c in ["smiles", "SMILES", "canonical_smiles"]:
                if c in df.columns:
                    col = c
                    break
            else:
                logger.warning(f"No SMILES column found in {file_path}")
                return []

        return df[col].dropna().tolist()
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return []


# ---------------------------------------------------------------------------
# Seed extraction
# ---------------------------------------------------------------------------

def extract_seeds_from_dataloader(
    dataloader,
    tokenizer,
    max_seeds: int | None = None,
    deduplicate: bool = True,
) -> list[str]:
    """Extract seed SMILES from dataloader batches.

    For paired data, extracts the first molecule (before SEP token) from each
    sequence.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch dataloader returning batches with ``input_ids``.
    tokenizer : PreTrainedTokenizer
        Tokenizer for decoding token IDs to SMILES.
    max_seeds : int | None
        Maximum number of seeds to extract. None means extract all.
    deduplicate : bool
        Whether to remove duplicate SMILES.

    Returns
    -------
    list[str]
        List of seed SMILES strings.
    """
    seeds: list[str] = []
    seen: set[str] = set() if deduplicate else None
    rejected: list[str] = []

    for batch in tqdm(dataloader, desc="Extracting seeds"):
        input_ids = batch["input_ids"]
        for i in range(input_ids.shape[0]):
            tokens = input_ids[i]
            smiles1, _smiles2 = SmilesUtils.tokens_to_smiles(tokenizer, tokens)
            if smiles1 and SmilesUtils.is_valid(smiles1):
                if deduplicate:
                    canonical = SmilesUtils.canonicalize(smiles1)
                    if canonical and canonical not in seen:
                        seen.add(canonical)
                        seeds.append(canonical)
                else:
                    seeds.append(smiles1)

                if max_seeds and len(seeds) >= max_seeds:
                    return seeds
            elif smiles1:
                rejected.append(smiles1)

    # `is_valid` rejects more than malformed SMILES, and used to do it silently. Callers
    # guarantee an output row only for seeds in this list, so a rejected starting molecule
    # disappears from the results entirely -- it does not appear as a failure, it simply
    # is not there, and the reported denominator shrinks to match. That silence hid a real
    # loss: the MoleculeNet best-of-64 runs covered 83 of 88 BBBP and 214 of 231 HIV
    # starting molecules with no indication anything was missing.
    #
    # Two things get rejected here that RDKit itself parses without complaint:
    #   * disconnected molecules (salts, hydrates, counterions -- any SMILES with a "."),
    #     because is_valid applies is_molecule_sane, which requires a single fragment;
    #   * molecules whose SMILES uses a token outside the tokenizer vocabulary (e.g. the
    #     metal in an organometallic complex), because seeds are decoded from token ids
    #     and the decoded string comes back containing <unk>.
    # Whether those belong in a given evaluation is a judgement call. Hiding them is not.
    if rejected:
        n_frag = sum(1 for s in rejected if "." in s)
        n_unk = sum(1 for s in rejected if getattr(tokenizer, "unk_token", None) and tokenizer.unk_token in s)
        logger.warning(
            "extract_seeds_from_dataloader: dropped %d starting molecule(s) that failed "
            "SmilesUtils.is_valid -- %d disconnected (contain '.'), %d with an "
            "out-of-vocabulary token. These will be ABSENT from the results rather than "
            "counted as failures. Examples: %s",
            len(rejected), n_frag, n_unk, "; ".join(s[:60] for s in rejected[:3]),
        )

    return seeds


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def apply_tokenizer_inheritance(cfg: DictConfig) -> None:
    """Apply tokenizer inheritance from checkpoint to config.

    Handles both ``cfg.data`` (training) and ``cfg.datamodule`` (evaluation)
    config structures.
    """
    ckpt_path_for_inheritance = get_checkpoint_path_for_inheritance(cfg)
    if not ckpt_path_for_inheritance:
        return

    needs_cleanup = False
    if "data" not in cfg and "datamodule" in cfg:
        with open_dict(cfg):
            cfg.data = cfg.datamodule
        needs_cleanup = True

    try:
        summary = prepare_tokenizer_inheritance(
            cfg, ckpt_path_for_inheritance, verbose=True
        )
        if summary["inherited"]:
            logger.info(
                f"[Tokenizer Inheritance] Inherited "
                f"{summary['num_tokens_inherited']} tokens from checkpoint"
            )
        if needs_cleanup and "transform_fn" in cfg.data:
            with open_dict(cfg):
                cfg.datamodule.transform_fn = cfg.data.transform_fn
    finally:
        if needs_cleanup and "data" in cfg:
            with open_dict(cfg):
                del cfg.data


def load_model(cfg: DictConfig, device: torch.device | None = None):
    """Load a model from a Hydra config with checkpoint.

    Parameters
    ----------
    cfg : DictConfig
        Must contain ``cfg.model`` with ``_target_`` and ``ckpt_path``.
    device : torch.device | None
        Device to move the model to. Auto-detects if None.

    Returns
    -------
    model
        Loaded PyTorch Lightning module in eval mode.
    device
        The device the model is on.
    """
    if not (hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None):
        raise ValueError("model.ckpt_path is required")

    logger.info(f"  Checkpoint: {cfg.model.ckpt_path}")
    model_cls = hydra.utils.get_class(cfg.model._target_)

    try:
        model = model_cls.load_from_checkpoint(cfg.model.ckpt_path)
        logger.info("Successfully loaded model from checkpoint")
    except Exception as e:
        if "weights_only" in str(e) or "Unsupported global" in str(e) or "WeightsUnpickler" in str(e):
            logger.warning(
                f"Failed to load with weights_only=True, trying manual load: {e}"
            )
            checkpoint = torch.load(
                cfg.model.ckpt_path, map_location="cpu", weights_only=False
            )
            model_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.model, resolve=True)
            )
            for key in ["run_name", "extra_name"]:
                if key in model_cfg:
                    del model_cfg[key]
            if "ckpt_path" in model_cfg:
                model_cfg.ckpt_path = None
            model = hydra.utils.instantiate(model_cfg)
            state = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
            logger.info("Successfully loaded model from checkpoint (manual load)")
        else:
            raise

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"  Model on {device}")
    return model, device


# ---------------------------------------------------------------------------
# Datamodule + dataloader
# ---------------------------------------------------------------------------

def setup_datamodule_and_dataloader(cfg: DictConfig):
    """Instantiate datamodule and return the requested split's dataloader.

    Parameters
    ----------
    cfg : DictConfig
        Must contain ``cfg.datamodule`` and ``cfg.split``.

    Returns
    -------
    datamodule
        The instantiated datamodule.
    dataloader
        Dataloader for the requested split.
    """
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()

    split = cfg.split
    if split == "val":
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        dataloader = datamodule.test_dataloader()
    elif split == "train":
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    logger.info(f"  Using {split} split with {len(dataloader)} batches")
    return datamodule, dataloader


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_from_seeds(
    pl_module,
    seeds: list[str],
    num_generations_per_seed: int,
    temperature: float = 0.95,
    batch_size: int = 8,
) -> tuple[list[str], list[int]]:
    """Generate molecules from seed SMILES.

    Parameters
    ----------
    pl_module : LightningModule
        Model with tokenizer attached.
    seeds : list[str]
        Seed SMILES strings.
    num_generations_per_seed : int
        Number of generations per seed.
    temperature : float
        Sampling temperature.
    batch_size : int
        Batch size for generation.

    Returns
    -------
    generated_smiles : list[str]
        Generated SMILES (length = len(seeds) * num_generations_per_seed).
    seed_indices : list[int]
        Index of the seed that produced each generation.
    """
    tokenizer = pl_module.tokenizer
    generated_smiles: list[str] = []
    seed_indices: list[int] = []
    num_batches = (len(seeds) + batch_size - 1) // batch_size

    for batch_start in tqdm(
        range(0, len(seeds), batch_size), total=num_batches, desc="Generating"
    ):
        batch_end = min(batch_start + batch_size, len(seeds))
        batch_seeds = seeds[batch_start:batch_end]

        encoded = tokenizer(
            batch_seeds,
            padding="longest",
            truncation=True,
            max_length=pl_module._max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(pl_module.model.device)
        attention_mask = encoded["attention_mask"].to(pl_module.model.device)

        generated_tokens = MoleculeGenerator.batch_generate_conditional(
            pl_module,
            input_ids,
            attention_mask,
            temperature=temperature,
            num_generations_per_seed=num_generations_per_seed,
        )

        for i, tokens in enumerate(generated_tokens):
            local_seed_idx = i // num_generations_per_seed
            global_seed_idx = batch_start + local_seed_idx

            if len(tokens) == 0:
                generated_smiles.append("")
                seed_indices.append(global_seed_idx)
                continue

            smiles1, smiles2 = SmilesUtils.tokens_to_smiles(tokenizer, tokens)
            gen_smiles = smiles2 if smiles2 else smiles1
            generated_smiles.append(gen_smiles or "")
            seed_indices.append(global_seed_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generated_smiles, seed_indices


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

def compute_properties(
    smiles_list: list[str],
    property_names: list[str],
) -> dict[str, list[float | None]]:
    """Compute molecular properties for a list of SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings.
    property_names : list[str]
        Property names to compute (must be in PROPERTY_REGISTRY).

    Returns
    -------
    dict[str, list[float | None]]
        Mapping from property name to list of values (None for failures).
    """
    property_specs = get_property_specs(property_names)
    results: dict[str, list[float | None]] = {name: [] for name in property_names}

    for smiles in smiles_list:
        props = compute_all_properties_for_smiles(smiles, property_specs)
        for name in property_names:
            results[name].append(props.get(name))

    return results


# ---------------------------------------------------------------------------
# Tanimoto distances
# ---------------------------------------------------------------------------

def compute_tanimoto_distances(
    generated_smiles: list[str],
    seeds: list[str],
    seed_indices: list[int],
) -> list[float | None]:
    """Compute Tanimoto distances from each generated molecule to its seed.

    Parameters
    ----------
    generated_smiles : list[str]
        Generated SMILES.
    seeds : list[str]
        Seed SMILES.
    seed_indices : list[int]
        Seed index for each generated molecule.

    Returns
    -------
    list[float | None]
        Tanimoto distance for each molecule (None if invalid).
    """
    distances: list[float | None] = []
    for gen_smiles, seed_idx in zip(generated_smiles, seed_indices):
        if not gen_smiles or not SmilesUtils.is_valid(gen_smiles):
            distances.append(None)
            continue
        seed_smiles = seeds[seed_idx]
        dists = compute_distances_with_fingerprint_caching(gen_smiles, [seed_smiles])
        distances.append(dists[0] if dists else None)
    return distances


# ---------------------------------------------------------------------------
# Memorization checking
# ---------------------------------------------------------------------------

def check_memorization_batch(
    generated_smiles: list[str],
    validity: list[bool],
    reference_canon_sets: dict[str, set[str]],
) -> tuple[list[bool], list[dict[str, bool]]]:
    """Check memorization for a batch of generated molecules.

    Parameters
    ----------
    generated_smiles : list[str]
        Generated SMILES.
    validity : list[bool]
        Validity flags.
    reference_canon_sets : dict[str, set[str]]
        Reference sets to check against.

    Returns
    -------
    is_memorized_any : list[bool]
        Whether each molecule is memorized in ANY reference set.
    memorized_by_ref : list[dict[str, bool]]
        Per-reference memorization status.
    """
    all_ref = set()
    for ref_set in reference_canon_sets.values():
        all_ref.update(ref_set)

    is_memorized_list: list[bool] = []
    memorized_by_ref_list: list[dict[str, bool]] = []

    for smiles, valid in zip(generated_smiles, validity):
        if not valid or not smiles:
            is_memorized_list.append(False)
            memorized_by_ref_list.append({})
            continue

        canonical = SmilesUtils.canonicalize(smiles)
        if not canonical:
            is_memorized_list.append(False)
            memorized_by_ref_list.append({})
            continue

        by_ref = {
            ref_name: canonical in ref_set
            for ref_name, ref_set in reference_canon_sets.items()
        }
        is_memorized_list.append(canonical in all_ref)
        memorized_by_ref_list.append(by_ref)

    return is_memorized_list, memorized_by_ref_list


# ---------------------------------------------------------------------------
# Improvement checking
# ---------------------------------------------------------------------------

def check_improvement(
    gen_value: float | None,
    seed_value: float | None,
    direction: str,
    threshold: float | None = None,
) -> bool:
    """Check if a generated molecule improved over the seed for a property.

    Parameters
    ----------
    gen_value : float | None
        Property value of the generated molecule.
    seed_value : float | None
        Property value of the seed molecule.
    direction : str
        ``"increasing"`` or ``"decreasing"``.
    threshold : float | None
        If set, improvement means crossing this threshold (e.g., 0.5 for
        binary classifiers). For ``"increasing"``, gen_value > threshold.
        For ``"decreasing"``, gen_value < threshold.
        If None, improvement means gen_value > seed_value (increasing)
        or gen_value < seed_value (decreasing).

    Returns
    -------
    bool
        True if the generated molecule improved.
    """
    if gen_value is None:
        return False
    if threshold is not None:
        if direction == "increasing":
            return gen_value > threshold
        elif direction == "decreasing":
            return gen_value < threshold
        return False
    if seed_value is None:
        return False
    if direction == "increasing":
        return gen_value > seed_value
    elif direction == "decreasing":
        return gen_value < seed_value
    return False


# ---------------------------------------------------------------------------
# Scored candidate construction
# ---------------------------------------------------------------------------

def build_scored_candidates(
    generated_smiles: list[str],
    seeds: list[str],
    seed_indices: list[int],
    property_names: list[str],
    ranking_property: str,
    ranking_direction: str,
    reference_canon_sets: dict[str, set[str]],
    round_num: int = 0,
    num_generations_per_seed: int = 1,
    compute_distances: bool = True,
    parent_smiles_map: dict[str, str] | None = None,
    lineage_map: dict[str, list[str]] | None = None,
    improvement_threshold: float | None = None,
) -> list[ScoredCandidate]:
    """Build scored candidates from raw generation outputs.

    This is the main pipeline: validate -> compute properties -> check
    memorization -> check improvement -> compute distances -> assemble
    ``ScoredCandidate`` objects.

    Parameters
    ----------
    generated_smiles : list[str]
        Raw generated SMILES.
    seeds : list[str]
        Seed SMILES that produced these generations.
    seed_indices : list[int]
        Seed index for each generation.
    property_names : list[str]
        Properties to compute.
    ranking_property : str
        The property used for improvement checking.
    ranking_direction : str
        ``"increasing"`` or ``"decreasing"``.
    reference_canon_sets : dict[str, set[str]]
        Reference sets for memorization checking (empty dict to skip).
    round_num : int
        Current generation round.
    num_generations_per_seed : int
        Generations per seed (for generation_idx).
    compute_distances : bool
        Whether to compute Tanimoto distances.
    parent_smiles_map : dict[str, str] | None
        Mapping from seed SMILES to parent SMILES (for lineage).
    lineage_map : dict[str, list[str]] | None
        Mapping from seed SMILES to full lineage list.
    improvement_threshold : float | None
        If set, improvement means crossing this absolute threshold
        (e.g., 0.5 for binary classifiers) rather than exceeding the
        seed value. Applied to the ranking property only.

    Returns
    -------
    list[ScoredCandidate]
        Scored candidates with all properties and flags.
    """
    n = len(generated_smiles)

    # 1. Validate
    validity = [SmilesUtils.is_valid(s) for s in generated_smiles]
    n_valid = sum(validity)
    logger.info(f"    Valid: {n_valid}/{n} ({100 * n_valid / max(1, n):.1f}%)")

    # 2. Compute properties for generated molecules
    gen_properties = compute_properties(generated_smiles, property_names)

    # 3. Compute seed properties (deduplicated)
    unique_seed_smiles = list(dict.fromkeys(seeds))
    seed_props_raw = compute_properties(unique_seed_smiles, property_names)
    seed_props_by_smiles: dict[str, dict[str, float | None]] = {}
    for i, s in enumerate(unique_seed_smiles):
        seed_props_by_smiles[s] = {
            name: seed_props_raw[name][i] for name in property_names
        }

    # 4. Check memorization
    if reference_canon_sets:
        is_memorized, memorized_by_ref = check_memorization_batch(
            generated_smiles, validity, reference_canon_sets
        )
        n_mem = sum(is_memorized)
        logger.info(
            f"    Memorized: {n_mem}/{n_valid} valid "
            f"({100 * n_mem / max(1, n_valid):.1f}%)"
        )
    else:
        is_memorized = [False] * n
        memorized_by_ref = [{}] * n

    # 5. Compute distances
    if compute_distances:
        distances = compute_tanimoto_distances(generated_smiles, seeds, seed_indices)
    else:
        distances = [None] * n

    # 6. Get property direction map for improvement checking
    direction_map: dict[str, str] = {}
    for pname in property_names:
        if pname == ranking_property:
            direction_map[pname] = ranking_direction
        elif pname in PROPERTY_REGISTRY:
            direction_map[pname] = PROPERTY_REGISTRY[pname][1]
        else:
            direction_map[pname] = "increasing"

    # 7. Assemble candidates
    candidates: list[ScoredCandidate] = []
    for i in range(n):
        seed_idx = seed_indices[i]
        seed_smi = seeds[seed_idx]
        gen_idx = i % num_generations_per_seed

        canonical = SmilesUtils.canonicalize(generated_smiles[i]) if validity[i] else None

        props = {name: gen_properties[name][i] for name in property_names}
        s_props = seed_props_by_smiles.get(seed_smi, {})

        improved: dict[str, bool] = {}
        for pname in property_names:
            # Use threshold for the ranking property, marginal for others
            thresh = improvement_threshold if pname == ranking_property else None
            improved[pname] = check_improvement(
                props.get(pname), s_props.get(pname), direction_map[pname],
                threshold=thresh,
            )

        parent = parent_smiles_map.get(seed_smi) if parent_smiles_map else None
        lineage = lineage_map.get(seed_smi) if lineage_map else None

        candidates.append(
            ScoredCandidate(
                smiles=generated_smiles[i],
                canonical=canonical,
                seed_smiles=seed_smi,
                seed_idx=seed_idx,
                is_valid=validity[i],
                is_memorized=is_memorized[i],
                memorized_by_ref=memorized_by_ref[i],
                is_improved=improved,
                properties=props,
                seed_properties=s_props,
                tanimoto_to_seed=distances[i],
                round_num=round_num,
                generation_idx=gen_idx,
                parent_smiles=parent,
                lineage=lineage,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def extract_from_eval_config(cfg: DictConfig) -> dict:
    """Extract inference-relevant settings from a ``pairit_eval`` config.

    Reads the MoleculeValidationCallback entry in ``cfg.callbacks`` to pull
    out reference files, property names, temperature, etc., and maps them
    to the flat keys that best-of-N / beam search expect.

    Parameters
    ----------
    cfg : DictConfig
        A Hydra config loaded from a ``*_pair_evaluate.yaml`` file.

    Returns
    -------
    dict
        Keys that can be merged into an inference config:
        ``reference_smiles_files``, ``property_names``, ``ranking_property``,
        ``ranking_direction``, ``temperature``.
    """
    result: dict = {}

    # Find the MoleculeValidationCallback in callbacks list
    callbacks = cfg.get("callbacks", [])
    mvc = None
    if callbacks:
        for cb in callbacks:
            target = cb.get("_target_", "")
            if "MoleculeValidationCallback" in target:
                mvc = cb
                break

    if mvc:
        # Reference SMILES files
        ref_files = mvc.get("reference_smiles_files", None)
        if ref_files is not None:
            result["reference_smiles_files"] = OmegaConf.to_container(
                ref_files, resolve=True
            )

        # Property names
        prop_names = mvc.get("property_names", None)
        if prop_names is not None:
            result["property_names"] = list(
                OmegaConf.to_container(prop_names, resolve=True)
            )

        # Ranking property = first property name or utility_key
        utility_key = mvc.get("utility_key", None)
        if utility_key:
            result["ranking_property"] = str(utility_key)
        elif result.get("property_names"):
            result["ranking_property"] = result["property_names"][0]

        # Direction from utility_direction or PROPERTY_REGISTRY default
        utility_dir = mvc.get("utility_direction", None)
        if utility_dir:
            result["ranking_direction"] = str(utility_dir)
        elif result.get("ranking_property") and result["ranking_property"] in PROPERTY_REGISTRY:
            result["ranking_direction"] = PROPERTY_REGISTRY[result["ranking_property"]][1]

        # Temperature
        temp = mvc.get("temperature", None)
        if temp is not None:
            result["temperature"] = float(temp)

    return result


def candidates_to_dataframe(candidates: list[ScoredCandidate]) -> pd.DataFrame:
    """Convert a list of ScoredCandidates to a DataFrame."""
    if not candidates:
        return pd.DataFrame()
    return pd.DataFrame([c.to_dict() for c in candidates])


def save_results(
    df: pd.DataFrame,
    output_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save results DataFrame to parquet with optional YAML metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"  Saved {len(df)} results to {path}")

    if metadata:
        meta_path = path.with_suffix(".yaml")
        with open(meta_path, "w") as f:
            OmegaConf.save(config=OmegaConf.create(metadata), f=f)
        logger.info(f"  Saved metadata to {meta_path}")
