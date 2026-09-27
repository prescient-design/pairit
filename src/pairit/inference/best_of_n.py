"""Best-of-N inference-time compute strategy.

Algorithm
---------
For each seed molecule in the test set:

1. **Generate** N candidate molecules by sampling from the CLM conditioned
   on the seed SMILES (temperature-controlled stochastic decoding).
2. **Validate** each candidate with RDKit (reject unparseable SMILES).
3. **Compute properties** for every candidate (logP, QED, SA, etc.) and
   for the seed molecule. Compute Tanimoto distance to the seed.
4. **Check memorization** against all reference training sets — a molecule
   is memorized if its canonical SMILES appears in ANY reference set.
5. **Check improvement** — for the ranking property, does the candidate
   improve over the seed? (Higher for ``increasing``, lower for
   ``decreasing``.)
6. **V,I,N hard gate** — keep only candidates that are simultaneously
   Valid, Improved, and Novel (not memorized). If no candidates pass for
   a given seed, fall back to all valid candidates (configurable).
7. **Select top-1** from the V,I,N pool by the ranking property value.

All N generations are saved with pre-computed properties (no recomputation
needed for downstream analysis). The selected molecule is flagged.

This is a single-round strategy — no iteration. For iterative refinement
use ``pairit_beam_search`` or ``pairit_iterative_gen``.

Inputs
------
- A trained CLM checkpoint (``model.ckpt_path``)
- A datamodule providing seed molecules (defaults to ZINC test split)
- Reference SMILES files for memorization checking (optional but recommended)
- Alternatively: ``eval_config=zinc_pair_evaluate`` to inherit all settings
  from an existing evaluation config

Outputs
-------
- ``all_generations.parquet`` — every generated molecule with properties
- ``summary.parquet`` — the selected best-of-N molecules
- ``config.yaml`` — full Hydra config for reproducibility
"""

from __future__ import annotations

# Register safe globals for PyTorch 2.6+ checkpoint loading FIRST
import pairit.cmdline._checkpoint_compat  # noqa: F401

import logging
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf

from pairit.cmdline.evaluate import extract_run_name_from_path
from pairit.inference.filters import VINFilter
from pairit.inference.strategies import TopKPerSeed
from pairit.inference.utils import (
    apply_tokenizer_inheritance,
    build_scored_candidates,
    candidates_to_dataframe,
    extract_from_eval_config,
    extract_seeds_from_dataloader,
    generate_from_seeds,
    load_model,
    load_reference_smiles_files,
    save_results,
    setup_datamodule_and_dataloader,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OmegaConf.register_new_resolver(
    "extract_run_name", extract_run_name_from_path, replace=True
)


class BestOfNGenerator:
    """Generate N molecules per seed and select the best after V,I,N filtering.

    Parameters
    ----------
    pl_module : LightningModule
        Loaded model with tokenizer.
    config : DictConfig
        Full Hydra config.
    initial_seeds : list[str]
        Seed SMILES.
    reference_canon_sets : dict[str, set[str]]
        Reference sets for memorization checking.
    """

    def __init__(
        self,
        pl_module,
        config: DictConfig,
        initial_seeds: list[str],
        reference_canon_sets: dict[str, set[str]] | None = None,
    ):
        self.pl_module = pl_module
        self.config = config
        self.seeds = initial_seeds
        self.reference_canon_sets = reference_canon_sets or {}

        bon = config.best_of_n
        self.n = bon.n
        self.batch_size = bon.batch_size
        self.temperature = config.generation.temperature
        self.ranking_property = bon.ranking_property
        self.ranking_direction = bon.ranking_direction
        self.vin_hard_gate = bon.vin_hard_gate
        self.vin_fallback = bon.vin_fallback
        self.improvement_threshold = bon.get("improvement_threshold", None)
        self.top_k = bon.get("top_k", 1)

        # Property names: always include ranking property + any extras
        extra = list(config.properties.get("property_names", []) or [])
        names = list(dict.fromkeys([self.ranking_property] + extra))
        self.property_names = names

        logger.info(f"BestOfN: {self.n} generations/seed, "
                     f"ranking by {self.ranking_property} ({self.ranking_direction})")
        logger.info(f"  V,I,N hard gate: {self.vin_hard_gate} (fallback={self.vin_fallback})")
        logger.info(f"  Properties: {self.property_names}")
        logger.info(f"  Seeds: {len(self.seeds)}")

    def run(self) -> tuple[list, list]:
        """Run best-of-N generation.

        Returns
        -------
        all_candidates : list[ScoredCandidate]
            Every generated molecule with properties.
        selected : list[ScoredCandidate]
            The selected best molecules.
        """
        logger.info(f"\n{'=' * 50}")
        logger.info(f"  BEST-OF-{self.n} GENERATION ({len(self.seeds)} seeds)")
        logger.info(f"{'=' * 50}")

        # Generate
        logger.info("[Step 1/3] Generating molecules...")
        generated_smiles, seed_indices = generate_from_seeds(
            self.pl_module,
            self.seeds,
            num_generations_per_seed=self.n,
            temperature=self.temperature,
            batch_size=self.batch_size,
        )
        logger.info(f"  Generated {len(generated_smiles)} molecules")

        # Score
        logger.info("[Step 2/3] Scoring candidates...")
        candidates = build_scored_candidates(
            generated_smiles=generated_smiles,
            seeds=self.seeds,
            seed_indices=seed_indices,
            property_names=self.property_names,
            ranking_property=self.ranking_property,
            ranking_direction=self.ranking_direction,
            reference_canon_sets=self.reference_canon_sets,
            round_num=0,
            num_generations_per_seed=self.n,
            compute_distances=True,
            improvement_threshold=self.improvement_threshold,
        )

        # Filter + Select
        logger.info("[Step 3/3] Filtering and selecting...")
        if self.vin_hard_gate:
            vin_filter = VINFilter(
                ranking_property=self.ranking_property,
                fallback=self.vin_fallback,
            )
            filtered = vin_filter.filter(candidates, per_seed=True)
        else:
            filtered = [c for c in candidates if c.is_valid]

        strategy = TopKPerSeed()
        selected = strategy.select(
            filtered,
            k=self.top_k,
            property_name=self.ranking_property,
            direction=self.ranking_direction,
        )

        # Guarantee every seed has output — fill missing seeds with best
        # valid candidate or the seed itself
        selected_seeds = {c.seed_idx for c in selected}
        all_seed_idxs = set(range(len(self.seeds)))
        missing_seeds = all_seed_idxs - selected_seeds
        if missing_seeds:
            logger.info(f"    {len(missing_seeds)} seeds missing from selection, adding fallbacks")
            for sid in sorted(missing_seeds):
                # Try best valid candidate for this seed
                seed_cands = [c for c in candidates if c.seed_idx == sid and c.is_valid]
                if seed_cands:
                    seed_cands.sort(
                        key=lambda c: c.properties.get(self.ranking_property, float("-inf")),
                        reverse=(self.ranking_direction == "increasing"),
                    )
                    best = seed_cands[0]
                    best.selected = True
                    selected.append(best)
                else:
                    # No valid candidates at all — return seed itself
                    from pairit.inference.scorers import ScoredCandidate as _SC
                    fallback = _SC(
                        smiles=self.seeds[sid], canonical=self.seeds[sid],
                        seed_smiles=self.seeds[sid], seed_idx=sid,
                        is_valid=True, selected=True, round_num=0,
                    )
                    selected.append(fallback)

        # Log summary
        n_valid = sum(1 for c in candidates if c.is_valid)
        n_vin = sum(
            1 for c in candidates if c.is_vin(self.ranking_property)
        )
        logger.info(f"\n  Summary:")
        logger.info(f"    Total generated: {len(candidates)}")
        logger.info(f"    Valid: {n_valid}")
        logger.info(f"    V,I,N ({self.ranking_property}): {n_vin}")
        logger.info(f"    Selected: {len(selected)}")

        if selected:
            scores = [
                c.properties.get(self.ranking_property)
                for c in selected
                if c.properties.get(self.ranking_property) is not None
            ]
            if scores:
                logger.info(
                    f"    Best {self.ranking_property}: "
                    f"mean={sum(scores)/len(scores):.3f}, "
                    f"max={max(scores):.3f}, min={min(scores):.3f}"
                )

        return candidates, selected


def _apply_eval_config_overlay(cfg: DictConfig) -> None:
    """If ``cfg.eval_config`` is set, load that eval YAML and overlay its
    reference files, property names, ranking property/direction, and
    temperature onto the inference config.
    """
    eval_config_name = cfg.get("eval_config", None)
    if not eval_config_name:
        return

    from hydra import compose, initialize_config_dir
    import os

    config_dir = os.path.join(
        os.path.dirname(__file__), "..", "hydra_config"
    )
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        eval_cfg = compose(config_name=eval_config_name)

    extracted = extract_from_eval_config(eval_cfg)
    logger.info(f"[Eval Config Overlay] Extracted from '{eval_config_name}': {list(extracted.keys())}")

    from omegaconf import open_dict

    with open_dict(cfg):
        if "reference_smiles_files" in extracted and not cfg.get("reference_smiles_files"):
            cfg.reference_smiles_files = extracted["reference_smiles_files"]
            logger.info(f"  -> reference_smiles_files: {len(extracted['reference_smiles_files'])} ref(s)")

        if "ranking_property" in extracted:
            cfg.best_of_n.ranking_property = extracted["ranking_property"]
            logger.info(f"  -> ranking_property: {extracted['ranking_property']}")

        if "ranking_direction" in extracted:
            cfg.best_of_n.ranking_direction = extracted["ranking_direction"]
            logger.info(f"  -> ranking_direction: {extracted['ranking_direction']}")

        if "temperature" in extracted:
            cfg.generation.temperature = extracted["temperature"]
            logger.info(f"  -> temperature: {extracted['temperature']}")

        if "property_names" in extracted:
            cfg.properties.property_names = extracted["property_names"]
            logger.info(f"  -> property_names: {extracted['property_names']}")


@hydra.main(
    config_path="../hydra_config", config_name="best_of_n", version_base=None
)
def best_of_n_main(cfg: DictConfig) -> None:
    """Run best-of-N molecule generation."""
    logger.info("=" * 60)
    logger.info("BEST-OF-N MOLECULE GENERATION")
    logger.info("=" * 60)

    # #37: seed torch/numpy/python-random up front so temperature sampling is
    # reproducible SMILES-by-SMILES across identical inputs. Aggregate stats
    # were stable pre-fix, but per-generation output drifted between reruns
    # -- this closes that gap. Default 42, overridable via `seed=<n>`.
    _seed = int(cfg.get("seed", 42))
    pl.seed_everything(_seed, workers=True)
    logger.info(f"Seeded inference RNG with seed={_seed}")

    # Eval config overlay: inherit settings from an existing eval YAML
    _apply_eval_config_overlay(cfg)

    logger.info(f"\nFull config:\n{OmegaConf.to_yaml(cfg)}")

    # Tokenizer inheritance
    apply_tokenizer_inheritance(cfg)

    # Load model
    logger.info("\n[Stage 1/5] Loading model...")
    device_str = cfg.get("device", None)
    device = torch.device(device_str) if device_str else None
    model, device = load_model(cfg, device)

    # Load references
    logger.info("\n[Stage 2/5] Loading reference SMILES...")
    ref_specs = OmegaConf.to_container(
        cfg.get("reference_smiles_files", []) or [], resolve=True
    )
    if ref_specs:
        ref_sets, _ = load_reference_smiles_files(
            ref_specs, smiles_column=cfg.get("smiles_column", "smiles")
        )
    else:
        ref_sets = {}
        logger.info("  No reference files (memorization checking disabled)")

    # Setup data
    logger.info("\n[Stage 3/5] Setting up datamodule...")
    _dm, dataloader = setup_datamodule_and_dataloader(cfg)

    # Extract seeds
    logger.info("\n[Stage 4/5] Extracting seeds...")
    seeds = extract_seeds_from_dataloader(
        dataloader,
        model.tokenizer,
        max_seeds=cfg.seeds.max_seeds,
        deduplicate=cfg.seeds.deduplicate,
    )
    logger.info(f"  Extracted {len(seeds)} seeds")
    if not seeds:
        raise ValueError("No valid seeds extracted from dataloader")

    # Run
    logger.info("\n[Stage 5/5] Running best-of-N...")
    generator = BestOfNGenerator(model, cfg, seeds, ref_sets)

    with torch.no_grad():
        all_candidates, selected = generator.run()

    # Save outputs
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Save all generations
    all_df = candidates_to_dataframe(all_candidates)
    save_results(all_df, str(output_dir / "all_generations.parquet"))

    # Save selected (summary)
    selected_df = candidates_to_dataframe(selected)
    save_results(selected_df, str(output_dir / "summary.parquet"))

    logger.info(f"\n{'=' * 60}")
    logger.info("BEST-OF-N COMPLETE")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  All generations: {len(all_df)} molecules")
    logger.info(f"  Selected: {len(selected_df)} molecules")
    logger.info(f"{'=' * 60}")


def main():
    best_of_n_main()


if __name__ == "__main__":
    main()
