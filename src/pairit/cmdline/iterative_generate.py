# Register safe globals for PyTorch 2.6+ checkpoint loading FIRST
# This must happen before any torch.load calls (including in Lightning)
import pairit.cmdline._checkpoint_compat  # noqa: F401  # Imported for side effects

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import hydra.utils
import lightning.pytorch as pl
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from pairit.callbacks.molecule_validation.generation import MoleculeGenerator
from pairit.callbacks.molecule_validation.smiles_utils import SmilesUtils
from pairit.callbacks.molecule_validation.property_registry import (
    PROPERTY_REGISTRY,
    get_property_specs,
    compute_all_properties_for_smiles,
)
from pairit.callbacks.molecule_validation.distance_utils import (
    compute_distances_with_fingerprint_caching,
)
from pairit.cmdline._tokenizer_inheritance import (
    prepare_tokenizer_inheritance,
    get_checkpoint_path_for_inheritance,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_reference_smiles_files(
    reference_specs: list[dict],
    smiles_column: str = "smiles",
) -> tuple[dict[str, set[str]], str | None]:
    """
    Load reference SMILES files for memorization checking.

    Parameters
    ----------
    reference_specs : list[dict]
        List of reference specifications, each with:
        - name: identifier for the reference set
        - path or files: path(s) to parquet/csv files
        - default: optional, marks one as the default reference
    smiles_column : str
        Column name for SMILES in files.

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

    def _canonicalize_smiles_list(smiles_list: list[str]) -> list[str]:
        """Canonicalize a list of SMILES, skipping invalid ones."""
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
        """Load SMILES from a single file."""
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

            # Find SMILES column
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

    for spec in reference_specs:
        # Extract name and files from spec
        name = spec.get("name")
        files = spec.get("path") or spec.get("files") or spec.get("file")
        is_default = spec.get("default", False)

        if not name or not files:
            continue

        # Normalize files to list
        if isinstance(files, str):
            files = [files]

        # Load SMILES from all files
        all_smiles = []
        for f in files:
            all_smiles.extend(_load_smiles_from_file(f, smiles_column))

        # Canonicalize
        canonical = _canonicalize_smiles_list(all_smiles)
        reference_canon_sets[name] = set(canonical)

        logger.info(f"  Loaded reference '{name}': {len(canonical)} canonical SMILES from {len(files)} file(s)")

        if is_default or default_reference_name is None:
            default_reference_name = name

    return reference_canon_sets, default_reference_name


def extract_seeds_from_dataloader(dataloader, tokenizer, max_seeds: int | None = None, deduplicate: bool = True) -> list[str]:
    """
    Extract seed SMILES from dataloader batches.

    For paired data, extracts the first molecule (before SEP token) from each sequence.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch dataloader returning batches with input_ids and attention_mask.
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
    seeds = []
    seen = set() if deduplicate else None

    for batch in tqdm(dataloader, desc="Extracting seeds"):
        input_ids = batch["input_ids"]

        for i in range(input_ids.shape[0]):
            tokens = input_ids[i]

            # Decode to get SMILES - tokens_to_smiles returns (smiles1, smiles2)
            smiles1, smiles2 = SmilesUtils.tokens_to_smiles(tokenizer, tokens)

            # Use smiles1 (the seed/first molecule)
            if smiles1 and SmilesUtils.is_valid(smiles1):
                if deduplicate:
                    canonical = SmilesUtils.canonicalize(smiles1)
                    if canonical and canonical not in seen:
                        seen.add(canonical)
                        seeds.append(canonical)
                else:
                    seeds.append(smiles1)

                # Check if we've reached max_seeds
                if max_seeds and len(seeds) >= max_seeds:
                    return seeds

    return seeds


class IterativeGenerator:
    """
    Iteratively generates molecules by applying a model to seed molecules,
    scoring generations by a configurable property, selecting top-k molecules
    as new seeds, and repeating for N rounds.
    """

    def __init__(
        self,
        pl_module,
        config: DictConfig,
        initial_seeds: list[str],
        reference_canon_sets: dict[str, set[str]] | None = None,
        default_reference_name: str | None = None,
    ):
        """
        Initialize the iterative generator.

        Parameters
        ----------
        pl_module : LightningModule
            The loaded model with tokenizer attached.
        config : DictConfig
            Hydra configuration containing generation, selection, seeds, etc.
        initial_seeds : list[str]
            List of initial seed SMILES extracted from dataloader.
        reference_canon_sets : dict[str, set[str]] | None
            Reference SMILES sets for memorization checking.
        default_reference_name : str | None
            Name of the default reference set.
        """
        self.pl_module = pl_module
        self.config = config
        self.tokenizer = pl_module.tokenizer
        self.initial_seeds = initial_seeds

        # Reference sets for memorization checking
        self.reference_canon_sets = reference_canon_sets or {}
        self.default_reference_name = default_reference_name

        # Union of all reference sets for "any" memorization check
        self.all_reference_smiles: set[str] = set()
        for ref_set in self.reference_canon_sets.values():
            self.all_reference_smiles.update(ref_set)

        # Validate configuration
        self._validate_config()

        logger.info(f"Initialized with {len(self.initial_seeds)} seeds")
        if self.reference_canon_sets:
            logger.info(f"Memorization checking enabled with {len(self.reference_canon_sets)} reference set(s)")
            logger.info(f"  Total unique reference SMILES: {len(self.all_reference_smiles)}")

        # Get property specs for computing properties
        property_names = list(config.properties.property_names or [])
        # Ensure optimization property is included
        opt_prop = config.selection.property_name
        if opt_prop not in property_names:
            property_names.append(opt_prop)
        self.property_specs = get_property_specs(property_names)
        self.property_names = property_names

        logger.info(f"Will compute properties: {self.property_names}")
        logger.info(f"Optimizing for: {opt_prop} ({config.selection.direction})")

    def _validate_config(self):
        """Validate configuration parameters."""
        cfg = self.config

        # Check property name is valid
        if cfg.selection.property_name not in PROPERTY_REGISTRY:
            available = ", ".join(sorted(PROPERTY_REGISTRY.keys()))
            raise ValueError(
                f"Unknown property '{cfg.selection.property_name}'. "
                f"Available: {available}"
            )

        # Check direction is valid
        if cfg.selection.direction not in ["increasing", "decreasing"]:
            raise ValueError(
                f"Invalid direction '{cfg.selection.direction}'. "
                "Must be 'increasing' or 'decreasing'"
            )

    def _check_memorization(self, canonical_smiles: str | None) -> tuple[bool, dict[str, bool]]:
        """
        Check if a molecule is memorized (exists in reference sets).

        Parameters
        ----------
        canonical_smiles : str | None
            Canonical SMILES to check.

        Returns
        -------
        is_memorized_any : bool
            True if molecule is in ANY reference set.
        memorized_by_ref : dict[str, bool]
            Per-reference memorization status.
        """
        if not canonical_smiles:
            return False, {}

        memorized_by_ref = {}
        for ref_name, ref_set in self.reference_canon_sets.items():
            memorized_by_ref[ref_name] = canonical_smiles in ref_set

        is_memorized_any = canonical_smiles in self.all_reference_smiles

        return is_memorized_any, memorized_by_ref

    def _tokenize_seeds(self, seeds: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize seed SMILES for generation.

        Returns
        -------
        input_ids : torch.Tensor
            Shape [batch_size, seq_len]
        attention_mask : torch.Tensor
            Shape [batch_size, seq_len]
        """
        # Tokenize seeds with padding
        encoded = self.tokenizer(
            seeds,
            padding="longest",
            truncation=True,
            max_length=self.pl_module._max_length,
            return_tensors="pt",
        )

        return encoded["input_ids"], encoded["attention_mask"]

    def _generate_from_seeds(
        self, seeds: list[str], num_generations_per_seed: int
    ) -> tuple[list[str], list[int]]:
        """
        Generate molecules from seed SMILES.

        Parameters
        ----------
        seeds : list[str]
            List of seed SMILES strings.
        num_generations_per_seed : int
            Number of generations per seed.

        Returns
        -------
        generated_smiles : list[str]
            List of generated SMILES (length = len(seeds) * num_generations_per_seed).
        seed_indices : list[int]
            Index of the seed that produced each generation.
        """
        batch_size = self.config.generation.get("batch_size", 8)
        generated_smiles = []
        seed_indices = []
        num_batches = (len(seeds) + batch_size - 1) // batch_size

        # Process seeds in batches to avoid OOM
        for batch_start in tqdm(range(0, len(seeds), batch_size), total=num_batches, desc="Generating"):
            batch_end = min(batch_start + batch_size, len(seeds))
            batch_seeds = seeds[batch_start:batch_end]

            # Tokenize batch of seeds
            input_ids, attention_mask = self._tokenize_seeds(batch_seeds)
            input_ids = input_ids.to(self.pl_module.model.device)
            attention_mask = attention_mask.to(self.pl_module.model.device)

            # Generate using batch_generate_conditional
            generated_tokens = MoleculeGenerator.batch_generate_conditional(
                self.pl_module,
                input_ids,
                attention_mask,
                temperature=self.config.generation.temperature,
                num_generations_per_seed=num_generations_per_seed,
            )

            # Convert tokens to SMILES
            for i, tokens in enumerate(generated_tokens):
                # Calculate the global seed index
                local_seed_idx = i // num_generations_per_seed
                global_seed_idx = batch_start + local_seed_idx

                if len(tokens) == 0:
                    # Empty generation
                    generated_smiles.append("")
                    seed_indices.append(global_seed_idx)
                    continue

                # Decode tokens to SMILES (get the second part after SEP)
                smiles1, smiles2 = SmilesUtils.tokens_to_smiles(self.tokenizer, tokens)

                # Use smiles2 if available (the generated part), otherwise smiles1
                gen_smiles = smiles2 if smiles2 else smiles1
                generated_smiles.append(gen_smiles or "")
                seed_indices.append(global_seed_idx)

            # Clear CUDA cache between batches to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return generated_smiles, seed_indices

    def _compute_properties(self, smiles_list: list[str]) -> dict[str, list[float | None]]:
        """Compute all configured properties for a list of SMILES."""
        from rdkit import Chem

        results = {name: [] for name in self.property_names}
        n_failed = 0
        failed_examples = []

        for smiles in smiles_list:
            props = compute_all_properties_for_smiles(smiles, self.property_specs)

            for name in self.property_names:
                results[name].append(props.get(name))

            # Track failures for debugging
            opt_prop = self.config.selection.property_name
            if props.get(opt_prop) is None:
                n_failed += 1
                if len(failed_examples) < 5:
                    # Debug: test RDKit parsing directly
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                    is_valid_check = SmilesUtils.is_valid(smiles)
                    failed_examples.append({
                        "smiles": repr(smiles[:80]) if smiles else "<empty>",
                        "len": len(smiles) if smiles else 0,
                        "rdkit_mol": mol is not None,
                        "is_valid": is_valid_check,
                    })

        if n_failed > 0:
            logger.warning(f"    Property computation failed for {n_failed}/{len(smiles_list)} molecules")
            for ex in failed_examples:
                logger.warning(f"    Failed example: {ex}")

        return results

    def _compute_distances(
        self, generated_smiles: list[str], seeds: list[str], seed_indices: list[int]
    ) -> list[float | None]:
        """Compute Tanimoto distances from each generated molecule to its seed."""
        distances = []

        for gen_smiles, seed_idx in zip(generated_smiles, seed_indices):
            if not gen_smiles or not SmilesUtils.is_valid(gen_smiles):
                distances.append(None)
                continue

            seed_smiles = seeds[seed_idx]
            dists = compute_distances_with_fingerprint_caching(gen_smiles, [seed_smiles])
            distances.append(dists[0] if dists else None)

        return distances

    def _check_memorization_batch(
        self, generated_smiles: list[str], validity: list[bool]
    ) -> tuple[list[bool], list[dict[str, bool]]]:
        """
        Check memorization for a batch of generated molecules.

        Returns
        -------
        is_memorized_list : list[bool]
            Whether each molecule is memorized in ANY reference set.
        memorized_by_ref_list : list[dict[str, bool]]
            Per-reference memorization status for each molecule.
        """
        is_memorized_list = []
        memorized_by_ref_list = []

        for smiles, valid in zip(generated_smiles, validity):
            if not valid or not smiles:
                is_memorized_list.append(False)
                memorized_by_ref_list.append({})
                continue

            canonical = SmilesUtils.canonicalize(smiles)
            is_mem, mem_by_ref = self._check_memorization(canonical)
            is_memorized_list.append(is_mem)
            memorized_by_ref_list.append(mem_by_ref)

        return is_memorized_list, memorized_by_ref_list

    def _select_top_k_per_seed(
        self,
        smiles_list: list[str],
        scores: list[float | None],
        validity: list[bool],
        seed_indices: list[int],
        is_memorized: list[bool] | None,
        k: int,
        direction: str,
        deduplicate: bool = True,
        prioritize_novel: bool = False,
    ) -> list[str]:
        """
        Select top-k molecules PER SEED based on scores.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings.
        scores : list[float | None]
            Property scores for each molecule.
        validity : list[bool]
            Whether each molecule is valid.
        seed_indices : list[int]
            Seed index for each molecule.
        is_memorized : list[bool] | None
            Whether each molecule is memorized.
        k : int
            Number of molecules to select PER SEED.
        direction : str
            'increasing' (higher scores better) or 'decreasing' (lower scores better).
        deduplicate : bool
            Whether to remove duplicates before selection.
        prioritize_novel : bool
            If True, prefer novel (non-memorized) molecules over memorized ones.

        Returns
        -------
        list[str]
            Selected SMILES strings (one best per seed that has valid generations).
        """
        # Group by seed index
        seed_to_candidates: dict[int, list[tuple[str, float, bool]]] = defaultdict(list)
        seen_global = set()

        # Debug counters
        n_invalid = 0
        n_no_score = 0
        n_canonicalize_failed = 0
        n_duplicate = 0

        for i, (smiles, score, valid, seed_idx) in enumerate(
            zip(smiles_list, scores, validity, seed_indices)
        ):
            if not valid:
                n_invalid += 1
                continue
            if score is None:
                n_no_score += 1
                continue

            canonical = SmilesUtils.canonicalize(smiles)
            if canonical is None:
                n_canonicalize_failed += 1
                continue

            if deduplicate and canonical in seen_global:
                n_duplicate += 1
                continue

            if deduplicate:
                seen_global.add(canonical)

            # Get memorization status
            is_mem = is_memorized[i] if is_memorized else False

            seed_to_candidates[seed_idx].append((canonical, score, is_mem))

        # Log debug info
        total_candidates = sum(len(c) for c in seed_to_candidates.values())
        logger.info(
            f"    Selection filter stats: invalid={n_invalid}, no_score={n_no_score}, "
            f"canonicalize_failed={n_canonicalize_failed}, duplicate={n_duplicate}, "
            f"candidates={total_candidates} across {len(seed_to_candidates)} seeds"
        )

        # Select top-k from each seed
        selected = []
        reverse = direction == "increasing"  # Higher is better

        n_seeds_with_novel = 0
        n_seeds_all_memorized = 0

        for seed_idx in sorted(seed_to_candidates.keys()):
            candidates = seed_to_candidates[seed_idx]

            if prioritize_novel and is_memorized is not None:
                # Separate novel and memorized candidates
                novel = [(s, sc, m) for s, sc, m in candidates if not m]
                memorized = [(s, sc, m) for s, sc, m in candidates if m]

                if novel:
                    n_seeds_with_novel += 1
                    # Sort novel by score and take top-k
                    novel.sort(key=lambda x: x[1], reverse=reverse)
                    for s, _, _ in novel[:k]:
                        selected.append(s)
                elif memorized:
                    n_seeds_all_memorized += 1
                    # Fall back to memorized if no novel
                    memorized.sort(key=lambda x: x[1], reverse=reverse)
                    for s, _, _ in memorized[:k]:
                        selected.append(s)
            else:
                # Sort all by score
                candidates.sort(key=lambda x: x[1], reverse=reverse)
                for s, _, _ in candidates[:k]:
                    selected.append(s)

        if prioritize_novel and is_memorized is not None:
            logger.info(
                f"    Novel prioritization: {n_seeds_with_novel} seeds had novel molecules, "
                f"{n_seeds_all_memorized} seeds had only memorized"
            )

        return selected

    def _build_round_dataframe(
        self,
        round_num: int,
        seeds: list[str],
        generated_smiles: list[str],
        seed_indices: list[int],
        validity: list[bool],
        distances: list[float | None],
        properties: dict[str, list[float | None]],
        selected_set: set[str],
        is_memorized: list[bool] | None,
        memorized_by_ref: list[dict[str, bool]] | None,
    ) -> pd.DataFrame:
        """Build DataFrame for one round of generation."""
        opt_prop = self.config.selection.property_name

        records = []
        for i, (gen_smiles, seed_idx, valid) in enumerate(
            zip(generated_smiles, seed_indices, validity)
        ):
            gen_idx = i % self.config.generation.num_generations_per_seed
            seed_smiles = seeds[seed_idx]

            # Get canonical form
            canonical = SmilesUtils.canonicalize(gen_smiles) if valid else None

            # Check if selected
            is_selected = canonical in selected_set if canonical else False

            record = {
                "round": round_num,
                "seed_idx": seed_idx,
                "generation_idx": gen_idx,
                "smiles_seed": seed_smiles,
                "smiles_generated": gen_smiles,
                "smiles_canonical": canonical,
                "is_valid": valid,
                "tanimoto_distance": distances[i] if self.config.distance.compute else None,
                "score": properties[opt_prop][i],
                "selected": is_selected,
                "is_memorized": is_memorized[i] if is_memorized else None,
            }

            # Add per-reference memorization columns
            if memorized_by_ref and memorized_by_ref[i]:
                for ref_name, is_mem in memorized_by_ref[i].items():
                    record[f"memorized_in_{ref_name}"] = is_mem

            # Add all property columns
            for prop_name in self.property_names:
                record[prop_name] = properties[prop_name][i]

            records.append(record)

        return pd.DataFrame(records)

    def run(self) -> pd.DataFrame:
        """
        Run the iterative generation loop.

        Returns
        -------
        pd.DataFrame
            All results across all rounds.
        """
        cfg = self.config
        all_results = []
        current_seeds = self.initial_seeds.copy()

        # Get memorization settings
        check_memorization = bool(self.reference_canon_sets)
        prioritize_novel = cfg.selection.get("prioritize_novel", False)

        if prioritize_novel and not check_memorization:
            logger.warning("prioritize_novel=True but no reference_smiles_files provided. Ignoring.")
            prioritize_novel = False

        for round_num in range(cfg.generation.num_rounds):
            logger.info("")
            logger.info(
                f"{'=' * 50}\n"
                f"  ROUND {round_num + 1}/{cfg.generation.num_rounds} "
                f"({len(current_seeds)} seeds)\n"
                f"{'=' * 50}"
            )

            # 1. Generate molecules from current seeds
            logger.info(f"  [Round {round_num + 1} Step 1/7] Generating molecules...")
            generated_smiles, seed_indices = self._generate_from_seeds(
                current_seeds, cfg.generation.num_generations_per_seed
            )
            logger.info(f"    Generated {len(generated_smiles)} molecules")

            # 2. Validate molecules
            logger.info(f"  [Round {round_num + 1} Step 2/7] Validating molecules...")
            validity = [SmilesUtils.is_valid(s) for s in generated_smiles]
            num_valid = sum(validity)
            logger.info(f"    Valid: {num_valid}/{len(generated_smiles)} ({100*num_valid/len(generated_smiles):.1f}%)")

            # 3. Compute properties
            logger.info(f"  [Round {round_num + 1} Step 3/7] Computing properties...")
            properties = self._compute_properties(generated_smiles)
            logger.info(f"    Computed {len(self.property_names)} properties")

            # 4. Compute distances
            logger.info(f"  [Round {round_num + 1} Step 4/7] Computing distances...")
            if cfg.distance.compute:
                distances = self._compute_distances(
                    generated_smiles, current_seeds, seed_indices
                )
                logger.info(f"    Computed Tanimoto distances")
            else:
                distances = [None] * len(generated_smiles)
                logger.info(f"    Skipped (disabled)")

            # 5. Check memorization
            logger.info(f"  [Round {round_num + 1} Step 5/7] Checking memorization...")
            if check_memorization:
                is_memorized, memorized_by_ref = self._check_memorization_batch(
                    generated_smiles, validity
                )
                n_memorized = sum(is_memorized)
                logger.info(f"    Memorized: {n_memorized}/{num_valid} valid molecules ({100*n_memorized/max(1,num_valid):.1f}%)")
            else:
                is_memorized = None
                memorized_by_ref = None
                logger.info(f"    Skipped (no reference files)")

            # 6. Select top-k valid molecules PER SEED for next round
            combined_validity = validity
            logger.info(f"  [Round {round_num + 1} Step 6/6] Selecting top-k per seed...")
            selected_smiles = self._select_top_k_per_seed(
                generated_smiles,
                properties[cfg.selection.property_name],
                combined_validity,
                seed_indices,
                is_memorized,
                cfg.selection.top_k,
                cfg.selection.direction,
                cfg.selection.deduplicate,
                prioritize_novel,
            )
            selected_set = set(selected_smiles)
            logger.info(f"    Selected {len(selected_smiles)} molecules for next round")

            # Log top scores
            if selected_smiles:
                top_scores = []
                for s in selected_smiles[:5]:
                    idx = None
                    for i, (gen, valid) in enumerate(zip(generated_smiles, combined_validity)):
                        if valid and SmilesUtils.canonicalize(gen) == s:
                            idx = i
                            break
                    if idx is not None:
                        top_scores.append(properties[cfg.selection.property_name][idx])
                if top_scores:
                    logger.info(
                        f"    Top {cfg.selection.property_name} scores: "
                        f"{[f'{s:.3f}' if s else 'N/A' for s in top_scores]}"
                    )

            # 7. Build round dataframe
            round_df = self._build_round_dataframe(
                round_num,
                current_seeds,
                generated_smiles,
                seed_indices,
                validity,
                distances,
                properties,
                selected_set,
                is_memorized,
                memorized_by_ref,
            )
            all_results.append(round_df)

            # 9. Save per-round results if configured
            if cfg.output.save_per_round:
                round_path = Path(cfg.output.dir) / f"round_{round_num}.parquet"
                round_path.parent.mkdir(parents=True, exist_ok=True)
                round_df.to_parquet(round_path)
                logger.info(f"    Saved round {round_num} results to {round_path}")

            logger.info(f"  Round {round_num + 1} complete: {len(round_df)} molecules processed")

            # 10. Update seeds for next round
            if len(selected_smiles) == 0:
                logger.warning(f"  No valid molecules in round {round_num + 1}. Stopping early.")
                break

            current_seeds = selected_smiles

        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()


def save_results(df: pd.DataFrame, output_path: str, metadata: dict[str, Any] | None = None):
    """Save results to parquet with optional metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path)
    logger.info(f"  Saved {len(df)} results to {path}")

    # Optionally save metadata as yaml
    if metadata:
        meta_path = path.with_suffix(".yaml")
        with open(meta_path, "w") as f:
            OmegaConf.save(config=OmegaConf.create(metadata), f=f)
        logger.info(f"  Saved metadata to {meta_path}")


@hydra.main(config_path="../hydra_config", config_name="iterative_generate", version_base=None)
def iterative_generate(cfg: DictConfig) -> None:
    """Run iterative molecule generation."""
    logger.info("=" * 60)
    logger.info("ITERATIVE MOLECULE GENERATION")
    logger.info("=" * 60)

    # #37: seed RNG for reproducible sampling. See best_of_n.py rationale.
    _seed = int(cfg.get("seed", 42))
    pl.seed_everything(_seed, workers=True)
    logger.info(f"Seeded inference RNG with seed={_seed}")

    # Print key configuration summary
    logger.info("")
    logger.info("Configuration Summary:")
    logger.info(f"  Checkpoint: {cfg.model.ckpt_path}")
    logger.info(f"  Temperature: {cfg.generation.temperature}")
    logger.info(f"  Batch size: {cfg.generation.get('batch_size', 8)}")
    logger.info(f"  Num rounds: {cfg.generation.num_rounds}")
    logger.info(f"  Generations per seed: {cfg.generation.num_generations_per_seed}")
    logger.info(f"  Top-k selection (per seed): {cfg.selection.top_k}")
    logger.info(f"  Optimize property: {cfg.selection.property_name} ({cfg.selection.direction})")
    logger.info(f"  Prioritize novel: {cfg.selection.get('prioritize_novel', False)}")
    logger.info(f"  Output dir: {cfg.output.dir}")
    logger.info("")
    logger.info("Full config:\n%s", OmegaConf.to_yaml(cfg))

    # Tokenizer inheritance: extract special tokens from checkpoint and inject into config
    # This MUST happen before datamodule/model instantiation so the tokenizer gets the right tokens
    # Handle both cfg.data (training) and cfg.datamodule (evaluation) config structures
    ckpt_path_for_inheritance = get_checkpoint_path_for_inheritance(cfg)
    if ckpt_path_for_inheritance:
        # For evaluation configs, datamodule is aliased as 'datamodule' (via data@datamodule)
        # The inheritance function expects cfg.data, so we need to adapt the config
        # Temporarily create cfg.data if it doesn't exist (pointing to datamodule)
        needs_cleanup = False
        if "data" not in cfg and "datamodule" in cfg:
            with open_dict(cfg):
                cfg.data = cfg.datamodule
            needs_cleanup = True

        try:
            inheritance_summary = prepare_tokenizer_inheritance(
                cfg, ckpt_path_for_inheritance, verbose=True
            )
            if inheritance_summary["inherited"]:
                logger.info(f"[Tokenizer Inheritance] Inherited {inheritance_summary['num_tokens_inherited']} tokens from checkpoint")

            # If we created cfg.data temporarily, copy the updated transform_fn back to datamodule
            if needs_cleanup and "transform_fn" in cfg.data:
                with open_dict(cfg):
                    cfg.datamodule.transform_fn = cfg.data.transform_fn
        finally:
            # Clean up temporary cfg.data if we created it
            if needs_cleanup and "data" in cfg:
                with open_dict(cfg):
                    del cfg.data

    # Instantiate model
    logger.info("")
    logger.info("[Stage 1/6] Loading model...")
    if hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        logger.info(f"  Checkpoint: {cfg.model.ckpt_path}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        try:
            # Try loading with Lightning's method first (uses safe globals if registered)
            model = model_cls.load_from_checkpoint(cfg.model.ckpt_path)
            logger.info("Successfully loaded model from checkpoint")
        except Exception as e:
            if "weights_only" in str(e) or "Unsupported global" in str(e) or "WeightsUnpickler" in str(e):
                # Fallback: manually load checkpoint with weights_only=False
                logger.warning(f"Failed to load with weights_only=True, trying manual load: {e}")
                checkpoint = torch.load(cfg.model.ckpt_path, map_location='cpu', weights_only=False)
                # Create a copy of model config without evaluation-specific keys
                model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
                # Remove evaluation-specific keys that aren't model init args
                for key in ['run_name', 'extra_name']:
                    if key in model_cfg:
                        del model_cfg[key]
                # Set ckpt_path to None since we're handling checkpoint loading manually
                if 'ckpt_path' in model_cfg:
                    model_cfg.ckpt_path = None
                # Instantiate model with filtered config
                model = hydra.utils.instantiate(model_cfg)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                logger.info("Successfully loaded model from checkpoint (manual load)")
            else:
                raise
    else:
        raise ValueError("model.ckpt_path is required")

    # Determine device
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    model.to(device)
    model.eval()
    logger.info("  Model loaded and ready")

    # Load reference SMILES for memorization checking
    logger.info("")
    logger.info("[Stage 2/6] Loading reference SMILES...")
    reference_specs = OmegaConf.to_container(cfg.get("reference_smiles_files", []) or [], resolve=True)
    if reference_specs:
        reference_canon_sets, default_reference_name = load_reference_smiles_files(
            reference_specs,
            smiles_column=cfg.get("smiles_column", "smiles"),
        )
        logger.info(f"  Loaded {len(reference_canon_sets)} reference set(s)")
    else:
        reference_canon_sets = {}
        default_reference_name = None
        logger.info("  No reference_smiles_files configured (memorization checking disabled)")

    # Instantiate datamodule
    logger.info("")
    logger.info("[Stage 3/6] Setting up datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()

    # Get dataloader for specified split
    if cfg.split == "val":
        dataloader = datamodule.val_dataloader()
    elif cfg.split == "test":
        dataloader = datamodule.test_dataloader()
    elif cfg.split == "train":
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f"Invalid split: {cfg.split}")

    logger.info(f"  Using {cfg.split} split with {len(dataloader)} batches")

    # Extract seeds from dataloader
    logger.info("")
    logger.info("[Stage 4/6] Extracting seeds from dataloader...")
    initial_seeds = extract_seeds_from_dataloader(
        dataloader,
        model.tokenizer,
        max_seeds=cfg.seeds.max_seeds,
        deduplicate=cfg.seeds.deduplicate,
    )
    logger.info(f"  Extracted {len(initial_seeds)} unique seeds from {cfg.split} split")

    if not initial_seeds:
        raise ValueError("No valid seeds extracted from dataloader")

    # Create generator and run
    logger.info("")
    logger.info("[Stage 5/6] Running iterative generation...")
    generator = IterativeGenerator(
        model,
        cfg,
        initial_seeds,
        reference_canon_sets=reference_canon_sets,
        default_reference_name=default_reference_name,
    )

    with torch.no_grad():
        results_df = generator.run()

    # Save results
    logger.info("")
    logger.info("[Stage 6/6] Saving results...")
    output_path = Path(cfg.output.dir) / cfg.output.filename
    metadata = OmegaConf.to_container(cfg, resolve=True)
    save_results(results_df, str(output_path), metadata)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ITERATIVE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total molecules generated: {len(results_df)}")
    logger.info(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    iterative_generate()
