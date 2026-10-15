"""Beam search inference-time compute strategy (per-seed).

Algorithm
---------
For EACH seed molecule independently, beam search maintains a small beam
of B candidate molecules and iteratively refines them over R rounds:

For each seed molecule:
  0. **Initialize** the beam with the seed molecule itself.

  For each round r = 1, ..., R:
    1. **Expand** — for each of the B beam members, generate K candidate
       molecules by sampling from the CLM. This produces up to B*K new
       candidates per round per seed.
    2. **Score** — validate each candidate, compute all properties, check
       memorization, compute Tanimoto distance to the ORIGINAL seed.
    3. **V,I,N hard gate** — keep only candidates that are Valid, Improved
       (crosses the activity threshold, e.g. gen > 0.5), and Novel (not
       memorized). Fall back to valid-only if none pass.
    4. **Prune** — select the top-B candidates by ranking property from
       the V,I,N pool (deduplicating by canonical SMILES). Previously
       seen molecules are excluded unless ``allow_revisit=true``.
    5. **Track lineage** — each molecule records its parent chain back to
       the original seed.

  After R rounds, the best molecule from this seed's beam is the output.

This runs independently for every seed, so all seeds get evaluated.
The output contains one selected molecule per seed (the best from the
final beam), plus all generations and per-round beam states.

Inputs
------
- A trained CLM checkpoint
- A datamodule providing seed molecules
- Reference SMILES files for memorization checking

Outputs
-------
- ``all_generations.parquet`` — every generated molecule across all rounds/seeds
- ``summary.parquet`` — best molecule per seed from final beam
- ``per_round/round_{i}.parquet`` — all generations from round i
- ``trajectories.parquet`` — lineage traces per seed
- ``config.yaml`` — full Hydra config
"""

from __future__ import annotations

# Register safe globals for PyTorch 2.6+ checkpoint loading FIRST
import pairit.cmdline._checkpoint_compat  # noqa: F401

import logging
from pathlib import Path

import hydra
import lightning.pytorch as pl
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from pairit.cmdline.evaluate import extract_run_name_from_path
from pairit.inference.filters import VINFilter
from pairit.inference.scorers import ScoredCandidate
from pairit.inference.utils import (
    apply_tokenizer_inheritance,
    build_scored_candidates,
    candidates_to_dataframe,
    compute_properties,
    compute_tanimoto_distances,
    extract_from_eval_config,
    extract_seeds_from_dataloader,
    generate_from_seeds,
    load_model,
    load_reference_smiles_files,
    save_results,
    setup_datamodule_and_dataloader,
)
from pairit.callbacks.molecule_validation.smiles_utils import SmilesUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OmegaConf.register_new_resolver(
    "extract_run_name", extract_run_name_from_path, replace=True
)


class BeamSearchGenerator:
    """Per-seed beam search over molecule generation with V,I,N filtering.

    Parameters
    ----------
    pl_module : LightningModule
        Loaded model with tokenizer.
    config : DictConfig
        Full Hydra config.
    initial_seeds : list[str]
        All seed SMILES (every seed gets its own beam).
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
        self.reference_canon_sets = reference_canon_sets or {}

        bs = config.beam_search
        self.beam_width = bs.beam_width
        self.expansions_per_candidate = bs.expansions_per_candidate
        self.num_rounds = bs.num_rounds
        self.temperature = config.generation.temperature
        self.batch_size = config.generation.get("batch_size", 8)
        self.ranking_property = bs.ranking_property
        self.ranking_direction = bs.ranking_direction
        self.vin_hard_gate = bs.vin_hard_gate
        self.vin_fallback = bs.vin_fallback
        self.allow_revisit = bs.allow_revisit
        self.track_lineage = bs.track_lineage
        self.improvement_threshold = bs.get("improvement_threshold", None)

        self.initial_seeds = initial_seeds

        # Property names
        extra = list(config.properties.get("property_names", []) or [])
        self.property_names = list(
            dict.fromkeys([self.ranking_property] + extra)
        )

        logger.info(
            f"BeamSearch (per-seed): width={self.beam_width}, "
            f"expansions={self.expansions_per_candidate}, "
            f"rounds={self.num_rounds}"
        )
        logger.info(f"  Ranking: {self.ranking_property} ({self.ranking_direction})")
        if self.improvement_threshold is not None:
            logger.info(f"  Improvement threshold: {self.improvement_threshold}")
        logger.info(f"  V,I,N: {self.vin_hard_gate} (fallback={self.vin_fallback})")
        logger.info(f"  Seeds: {len(self.initial_seeds)} (each gets own beam)")

    def _run_single_seed(
        self,
        seed_smiles: str,
        seed_idx: int,
    ) -> tuple[list[ScoredCandidate], ScoredCandidate | None, list[str]]:
        """Run beam search for a single seed.

        Returns
        -------
        all_candidates : list[ScoredCandidate]
            Every generated molecule across all rounds for this seed.
        best : ScoredCandidate | None
            The best molecule from the final beam.
        lineage : list[str]
            SMILES lineage of the best molecule.
        """
        prop = self.ranking_property
        reverse = self.ranking_direction == "increasing"

        # Current beam: list of (canonical_smiles, lineage)
        beam: list[tuple[str, list[str]]] = [(seed_smiles, [seed_smiles])]
        seen: set[str] = {seed_smiles} if not self.allow_revisit else set()
        all_candidates: list[ScoredCandidate] = []

        vin_filter = VINFilter(
            ranking_property=prop, fallback=self.vin_fallback,
        ) if self.vin_hard_gate else None

        for round_num in range(self.num_rounds):
            # Expand: generate from each beam member
            beam_smiles = [smi for smi, _lin in beam]
            if not beam_smiles:
                break

            generated_smiles, gen_seed_indices = generate_from_seeds(
                self.pl_module,
                beam_smiles,
                num_generations_per_seed=self.expansions_per_candidate,
                temperature=self.temperature,
                batch_size=self.batch_size,
            )

            # Build parent/lineage maps for this round
            parent_map = {smi: seed_smiles for smi, _lin in beam}
            lineage_map = {smi: lin.copy() for smi, lin in beam}

            # Score candidates (all measure against original seed for tanimoto)
            # We pass beam_smiles as seeds but compute tanimoto to original seed separately
            candidates = build_scored_candidates(
                generated_smiles=generated_smiles,
                seeds=beam_smiles,
                seed_indices=gen_seed_indices,
                property_names=self.property_names,
                ranking_property=self.ranking_property,
                ranking_direction=self.ranking_direction,
                reference_canon_sets=self.reference_canon_sets,
                round_num=round_num,
                num_generations_per_seed=self.expansions_per_candidate,
                compute_distances=False,  # we'll compute to original seed
                parent_smiles_map=parent_map,
                lineage_map=lineage_map,
                improvement_threshold=self.improvement_threshold,
            )

            # Compute tanimoto to ORIGINAL seed (not the beam parent)
            for c in candidates:
                if c.is_valid and c.canonical:
                    dists = compute_tanimoto_distances(
                        [c.canonical], [seed_smiles], [0]
                    )
                    c.tanimoto_to_seed = dists[0]
                # Override seed_smiles/seed_idx to be the original seed
                c.seed_smiles = seed_smiles
                c.seed_idx = seed_idx

            # Attach lineage
            if self.track_lineage:
                for c in candidates:
                    parent_lin = lineage_map.get(
                        beam_smiles[gen_seed_indices[candidates.index(c)] if candidates.index(c) < len(gen_seed_indices) else 0],
                        [seed_smiles]
                    )
                    c.lineage = parent_lin + [c.canonical or c.smiles]

            # Filter out revisited
            if not self.allow_revisit:
                candidates = [
                    c for c in candidates
                    if not c.canonical or c.canonical not in seen
                ]

            all_candidates.extend(candidates)

            # V,I,N filter
            if vin_filter:
                filtered = vin_filter.filter(candidates, per_seed=False)
            else:
                filtered = [c for c in candidates if c.is_valid]

            # Sort by property and take top beam_width
            scoreable = [
                c for c in filtered
                if c.properties.get(prop) is not None
            ]
            # Deduplicate
            deduped: list[ScoredCandidate] = []
            seen_canonical: set[str] = set()
            for c in scoreable:
                if c.canonical and c.canonical not in seen_canonical:
                    seen_canonical.add(c.canonical)
                    deduped.append(c)

            deduped.sort(
                key=lambda c: c.properties.get(prop, float("-inf")),
                reverse=reverse,
            )
            new_beam_candidates = deduped[: self.beam_width]

            # Update seen set
            if not self.allow_revisit:
                for c in candidates:
                    if c.canonical:
                        seen.add(c.canonical)

            # Update beam
            beam = []
            for c in new_beam_candidates:
                c.selected = True
                lin = c.lineage if c.lineage else [seed_smiles, c.canonical or c.smiles]
                beam.append((c.canonical or c.smiles, lin))

            if not beam:
                # Fall back: keep the original seed if beam emptied
                beam = [(seed_smiles, [seed_smiles])]

        # Select best from final beam
        best = None
        if all_candidates:
            selected = [c for c in all_candidates if c.selected]
            if selected:
                selected.sort(
                    key=lambda c: c.properties.get(prop, float("-inf")),
                    reverse=reverse,
                )
                best = selected[0]

        # Guarantee output: if no candidate survived, return the seed itself
        if best is None:
            best = ScoredCandidate(
                smiles=seed_smiles, canonical=seed_smiles,
                seed_smiles=seed_smiles, seed_idx=seed_idx,
                is_valid=True, selected=True, round_num=-1,
            )

        best_lineage = best.lineage if best.lineage else [seed_smiles]
        return all_candidates, best, best_lineage

    def run(self) -> tuple[list[ScoredCandidate], list[ScoredCandidate], pd.DataFrame]:
        """Run per-seed beam search for all seeds.

        Returns
        -------
        all_candidates : list[ScoredCandidate]
            Every generated molecule across all rounds and seeds.
        selected : list[ScoredCandidate]
            Best molecule per seed.
        trajectories_df : pd.DataFrame
            Lineage traces.
        """
        all_candidates: list[ScoredCandidate] = []
        selected: list[ScoredCandidate] = []
        trajectory_records: list[dict] = []

        for seed_idx, seed_smi in enumerate(self.initial_seeds):
            if seed_idx % 10 == 0:
                logger.info(
                    f"  Seed {seed_idx + 1}/{len(self.initial_seeds)}: {seed_smi[:40]}..."
                )

            seed_candidates, best, lineage = self._run_single_seed(seed_smi, seed_idx)
            all_candidates.extend(seed_candidates)

            if best is not None:
                selected.append(best)

            # Build trajectory record
            if self.track_lineage and lineage:
                record: dict = {
                    "trajectory_id": seed_idx,
                    "initial_seed": seed_smi,
                    "final_smiles": lineage[-1] if lineage else seed_smi,
                    "num_rounds": len(lineage) - 1,
                }
                # Per-step SMILES
                for i, smi in enumerate(lineage):
                    record[f"round_{i}_smiles"] = smi

                # Compute properties for each step
                lineage_props = compute_properties(lineage, self.property_names)
                for i in range(len(lineage)):
                    for prop_name in self.property_names:
                        record[f"round_{i}_{prop_name}"] = lineage_props[prop_name][i]

                # Total deltas
                if len(lineage) >= 2:
                    for prop_name in self.property_names:
                        v0 = lineage_props[prop_name][0]
                        vf = lineage_props[prop_name][-1]
                        if v0 is not None and vf is not None:
                            record[f"total_{prop_name}_delta"] = vf - v0

                    # Tanimoto drift
                    drift = compute_tanimoto_distances(
                        [lineage[-1]], [lineage[0]], [0]
                    )
                    record["total_tanimoto_drift"] = drift[0]

                    for i, smi in enumerate(lineage[1:], 1):
                        d = compute_tanimoto_distances([smi], [lineage[0]], [0])
                        record[f"round_{i}_tanimoto_to_seed"] = d[0]

                trajectory_records.append(record)

        trajectories_df = pd.DataFrame(trajectory_records) if trajectory_records else pd.DataFrame()

        logger.info(f"\n  Beam search complete: {len(selected)}/{len(self.initial_seeds)} seeds produced output")
        n_vin = sum(1 for c in selected if c.is_vin(self.ranking_property))
        logger.info(f"  V,I,N among selected: {n_vin}/{len(selected)}")

        return all_candidates, selected, trajectories_df


def _apply_eval_config_overlay(cfg: DictConfig) -> None:
    """If ``cfg.eval_config`` is set, overlay settings from that eval YAML."""
    eval_config_name = cfg.get("eval_config", None)
    if not eval_config_name:
        return

    import os

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # This runs inside a @hydra.main app, so Hydra is normally already
    # initialised and its search path already contains hydra_config/ --
    # compose directly rather than initialising a second time, which raises.
    if GlobalHydra.instance().is_initialized():
        eval_cfg = compose(config_name=eval_config_name)
    else:
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "hydra_config")
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            eval_cfg = compose(config_name=eval_config_name)

    extracted = extract_from_eval_config(eval_cfg)
    logger.info(f"[Eval Config Overlay] Extracted from '{eval_config_name}': {list(extracted.keys())}")

    from omegaconf import open_dict

    with open_dict(cfg):
        if "reference_smiles_files" in extracted and not cfg.get("reference_smiles_files"):
            cfg.reference_smiles_files = extracted["reference_smiles_files"]

        if "ranking_property" in extracted:
            cfg.beam_search.ranking_property = extracted["ranking_property"]

        if "ranking_direction" in extracted:
            cfg.beam_search.ranking_direction = extracted["ranking_direction"]

        if "temperature" in extracted:
            cfg.generation.temperature = extracted["temperature"]

        if "property_names" in extracted:
            cfg.properties.property_names = extracted["property_names"]


@hydra.main(
    config_path="../hydra_config", config_name="beam_search", version_base=None
)
def beam_search_main(cfg: DictConfig) -> None:
    """Run per-seed beam search molecule generation."""
    logger.info("=" * 60)
    logger.info("BEAM SEARCH MOLECULE GENERATION (PER-SEED)")
    logger.info("=" * 60)

    # #37: seed RNG for reproducible sampling. See best_of_n.py rationale.
    _seed = int(cfg.get("seed", 42))
    pl.seed_everything(_seed, workers=True)
    logger.info(f"Seeded inference RNG with seed={_seed}")

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

    # Run beam search
    logger.info("\n[Stage 5/5] Running per-seed beam search...")
    generator = BeamSearchGenerator(model, cfg, seeds, ref_sets)

    with torch.no_grad():
        all_candidates, selected, trajectories_df = generator.run()

    # Save outputs
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)

    all_df = candidates_to_dataframe(all_candidates)
    save_results(all_df, str(output_dir / "all_generations.parquet"))

    selected_df = candidates_to_dataframe(selected)
    save_results(selected_df, str(output_dir / "summary.parquet"))

    # Per-round parquets
    if "round" in all_df.columns:
        per_round_dir = output_dir / "per_round"
        per_round_dir.mkdir(parents=True, exist_ok=True)
        for r in sorted(all_df["round"].unique()):
            rdf = all_df[all_df["round"] == r]
            save_results(rdf, str(per_round_dir / f"round_{r}.parquet"))

    # Trajectories
    if len(trajectories_df) > 0:
        save_results(trajectories_df, str(output_dir / "trajectories.parquet"))
        logger.info(f"  Built {len(trajectories_df)} trajectories")

    logger.info(f"\n{'=' * 60}")
    logger.info("BEAM SEARCH COMPLETE")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  All generations: {len(all_df)} molecules")
    logger.info(f"  Selected (1 per seed): {len(selected_df)} molecules")
    logger.info(f"{'=' * 60}")


def main():
    beam_search_main()


if __name__ == "__main__":
    main()
