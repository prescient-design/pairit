"""Reward functions for RL post-training of molecule improvement models.

Each reward function is a callable class following TRL's signature::

    reward_fn(completions, prompts, **kwargs) -> list[float]

TRL's ``reward_weights`` parameter handles weighted combination of multiple
reward functions, so no composite wrapper is needed.

Extra dataset columns (e.g. ``seed_smiles``) are forwarded to reward functions
as keyword arguments by TRL's GRPOTrainer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

logger = logging.getLogger(__name__)

# Suppress RDKit warnings for invalid SMILES
from rdkit import rdBase
rdBase.DisableLog("rdApp.error")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def extract_smiles(completion_str: str) -> str:
    """Extract a clean SMILES string from a TRL-decoded completion.

    The SMILES tokenizer decode inserts spaces between characters
    (e.g. ``"C C ( = O ) N"``).  This function strips special tokens
    and rejoins to produce a valid SMILES string.
    """
    # Remove special tokens
    for tok in ("<eos>", "<pad>", "<cls>", "<sep>", "<mask>", "<unk>"):
        completion_str = completion_str.replace(tok, "")
    # Remove spaces (SMILES tokenizer decode artifact)
    return "".join(completion_str.split())


def canonicalize(smiles: str) -> str | None:
    """Canonicalize a SMILES string, returning None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_tanimoto(smiles1: str, smiles2: str) -> float | None:
    """Compute Tanimoto similarity between two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = gen.GetFingerprint(mol1)
        fp2 = gen.GetFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return None


def load_reference_smiles(reference_file_specs: list[str]) -> dict[str, set[str]]:
    """Load canonical SMILES sets from ``name:path`` specifications.

    Parameters
    ----------
    reference_file_specs : list[str]
        Each entry is ``"name:/path/to/molecules.parquet"``.

    Returns
    -------
    dict mapping reference name to set of canonical SMILES.
    """
    import pandas as pd

    reference_sets: dict[str, set[str]] = {}
    for spec in reference_file_specs:
        if ":" not in spec:
            logger.warning("Skipping malformed reference spec (expected name:path): %s", spec)
            continue
        name, path = spec.split(":", 1)
        try:
            df = pd.read_parquet(path)
            smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
            raw_smiles = df[smiles_col].dropna().tolist()
            canonical = {canonicalize(s) for s in raw_smiles}
            canonical.discard(None)
            reference_sets[name] = canonical
            logger.info("Loaded %d canonical SMILES for reference '%s'", len(canonical), name)
        except Exception as e:
            logger.warning("Failed to load reference '%s' from %s: %s", name, path, e)
    return reference_sets


# ---------------------------------------------------------------------------
# Reward function classes
# ---------------------------------------------------------------------------

class ValidityReward:
    """Binary reward: 1.0 if the generated SMILES is valid, 0.0 otherwise."""

    __name__ = "validity"

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion in completions:
            smiles = extract_smiles(completion)
            mol = Chem.MolFromSmiles(smiles)
            rewards.append(1.0 if mol is not None else 0.0)

        if log_metric is not None:
            log_metric("rewards/validity_rate", np.mean(rewards))
        return rewards


class InvalidPenaltyReward:
    """Penalty for invalid SMILES: -1.0 if invalid, 0.0 otherwise."""

    __name__ = "invalid_penalty"

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion in completions:
            smiles = extract_smiles(completion)
            mol = Chem.MolFromSmiles(smiles)
            rewards.append(-1.0 if mol is None else 0.0)
        return rewards


class PropertyImprovementReward:
    """Continuous reward based on oracle property prediction.

    Supports multiple modes for computing the reward from oracle probabilities:

    - ``delta`` (default): ``prob(gen) - prob(seed)``. Rewards relative improvement.
    - ``absolute``: ``prob(gen)``. Rewards generating high-probability molecules directly.
    - ``threshold_bonus``: ``delta + bonus * (prob(gen) > threshold)``. Rewards crossing
      the decision boundary with a discrete bonus on top of continuous improvement.
    - ``soft_threshold``: ``sigmoid(scale * (prob(gen) - threshold))``. Continuous reward
      concentrated around the decision boundary. Smoothest option.

    ``direction`` controls which way is "better". Every mode above is written for
    ``increasing`` (higher oracle probability = better), which is correct for MoleculeNet
    (active / permeable / active) but **exactly wrong for hERG**, whose oracle predicts
    P(blocker) -- there, higher probability means more cardiotoxic. Passing
    ``direction="decreasing"`` flips the sign so the reward rewards *lowering* the
    probability. Default stays ``increasing`` so existing configs are unaffected.
    """

    __name__ = "property_improvement"

    def __init__(
        self,
        evaluator: Any,
        task_name: str,
        mode: str = "delta",
        threshold: float = 0.5,
        bonus: float = 1.0,
        soft_scale: float = 10.0,
        direction: str = "increasing",
    ):
        self.evaluator = evaluator
        self.task_name = task_name
        self.mode = mode
        self.threshold = threshold
        self.bonus = bonus
        self.soft_scale = soft_scale
        if direction not in ("increasing", "decreasing"):
            raise ValueError(f"direction must be increasing/decreasing, got {direction!r}")
        self.direction = direction
        # +1 keeps the original behaviour; -1 makes "better" mean a lower probability.
        self._sign = 1.0 if direction == "increasing" else -1.0

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        seed_smiles: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        if seed_smiles is None:
            logger.warning("PropertyImprovementReward: no seed_smiles provided")
            return [0.0] * len(completions)

        gen_smiles_list = [extract_smiles(c) for c in completions]

        # Batch oracle predictions for efficiency
        gen_probs = self.evaluator.predict(gen_smiles_list)
        seed_probs = self.evaluator.predict(list(seed_smiles))

        rewards = []
        valid_probs = []
        n_active = 0
        for i, gen_smi in enumerate(gen_smiles_list):
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is None:
                rewards.append(0.0)
                continue

            gp = float(gen_probs[i])
            sp = float(seed_probs[i])
            valid_probs.append(gp)
            # "on the good side of the boundary", direction-aware
            crossed = (gp >= self.threshold) if self._sign > 0 else (gp <= self.threshold)
            if crossed:
                n_active += 1

            k = self._sign
            if self.mode == "delta":
                rewards.append(k * (gp - sp))
            elif self.mode == "absolute":
                rewards.append(k * gp)
            elif self.mode == "threshold_bonus":
                rewards.append(k * (gp - sp) + self.bonus * float(crossed))
            elif self.mode == "soft_threshold":
                # sigmoid centred on the threshold, rising in the improving direction
                rewards.append(float(1.0 / (1.0 + np.exp(-self.soft_scale * k * (gp - self.threshold)))))
            else:
                rewards.append(k * (gp - sp))  # fallback to delta

        if log_metric is not None and valid_probs:
            log_metric("rewards/oracle_prob_mean", np.mean(valid_probs))
            log_metric("rewards/active_rate", n_active / max(len(valid_probs), 1))
        return rewards


class StructuralSimilarityReward:
    """Tanimoto similarity between generated and seed molecule.

    Returns similarity in [0, 1] for valid, non-identity molecules.
    Returns 0.0 for invalid molecules.
    """

    __name__ = "structural_similarity"

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        seed_smiles: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        if seed_smiles is None:
            return [0.0] * len(completions)

        rewards = []
        tanimotos = []
        for completion, seed in zip(completions, seed_smiles):
            gen_smi = extract_smiles(completion)
            sim = compute_tanimoto(seed, gen_smi)
            if sim is not None:
                rewards.append(sim)
                tanimotos.append(sim)
            else:
                rewards.append(0.0)

        if log_metric is not None and tanimotos:
            log_metric("rewards/tanimoto_mean", np.mean(tanimotos))
            log_metric("rewards/tanimoto_std", np.std(tanimotos))
        return rewards


class IdentityPenaltyReward:
    """Penalty for generating an exact copy of the seed molecule.

    Compares canonical SMILES: -1.0 if identical, 0.0 otherwise.
    """

    __name__ = "identity_penalty"

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        seed_smiles: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        if seed_smiles is None:
            return [0.0] * len(completions)

        rewards = []
        for completion, seed in zip(completions, seed_smiles):
            gen_smi = extract_smiles(completion)
            canon_gen = canonicalize(gen_smi)
            canon_seed = canonicalize(seed)
            is_copy = (canon_gen is not None) and (canon_gen == canon_seed)
            rewards.append(-1.0 if is_copy else 0.0)

        if log_metric is not None:
            identity_rate = sum(1 for r in rewards if r < 0) / max(len(rewards), 1)
            log_metric("rewards/identity_rate", identity_rate)
        return rewards


class MemorizationPenaltyReward:
    """Penalty for generating molecules that appear in any training set.

    Compares canonical SMILES against one or more reference sets.
    """

    __name__ = "memorization_penalty"

    def __init__(self, reference_sets: dict[str, set[str]]):
        self.reference_sets = reference_sets
        total = sum(len(s) for s in reference_sets.values())
        logger.info(
            "MemorizationPenaltyReward: %d reference sets, %d total molecules",
            len(reference_sets), total,
        )

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion in completions:
            gen_smi = extract_smiles(completion)
            canon = canonicalize(gen_smi)
            is_memorized = False
            if canon is not None:
                is_memorized = any(canon in s for s in self.reference_sets.values())
            rewards.append(-1.0 if is_memorized else 0.0)

        if log_metric is not None:
            mem_rate = sum(1 for r in rewards if r < 0) / max(len(rewards), 1)
            log_metric("rewards/memorization_rate", mem_rate)
        return rewards


class ValidityGatedPropertyReward:
    """Combined reward that gates property improvement on validity.

    Invalid molecules get a fixed negative reward. Valid molecules get
    their oracle property improvement score (continuous). This creates
    a natural separation: invalid < 0 < valid-but-worse < valid-and-improved,
    giving GRPO meaningful within-group variance even when most generations
    are valid.

    Optionally penalizes identity (copying seed) and memorization.
    """

    __name__ = "validity_gated_property"

    def __init__(
        self,
        evaluator: Any,
        task_name: str,
        invalid_reward: float = -1.0,
        identity_penalty: float = -0.5,
        memorization_penalty: float = -0.5,
        similarity_weight: float = 0.0,
        property_mode: str = "delta",
        active_threshold: float = 0.5,
        active_bonus: float = 1.0,
        similarity_threshold: float = 0.0,
        similarity_threshold_penalty: float = -0.5,
        reference_sets: dict[str, set[str]] | None = None,
        direction: str = "increasing",
    ):
        self.evaluator = evaluator
        self.task_name = task_name
        # Which way is "better". See PropertyImprovementReward.direction -- the oracle for
        # hERG predicts P(blocker), so "increasing" would reward making molecules MORE
        # cardiotoxic. Default preserves the original behaviour for MoleculeNet.
        if direction not in ("increasing", "decreasing"):
            raise ValueError(f"direction must be increasing/decreasing, got {direction!r}")
        self.direction = direction
        self._sign = 1.0 if direction == "increasing" else -1.0
        self.invalid_reward = invalid_reward
        self.identity_penalty = identity_penalty
        self.memorization_penalty = memorization_penalty
        self.similarity_weight = similarity_weight
        self.property_mode = property_mode
        self.active_threshold = active_threshold
        self.active_bonus = active_bonus
        # If similarity_threshold > 0, generations with Tanimoto(g, s) below this
        # value get the similarity_threshold_penalty (treated like memorization/identity).
        # Dataset-independent way to anchor generations near the seed.
        self.similarity_threshold = similarity_threshold
        self.similarity_threshold_penalty = similarity_threshold_penalty
        self.reference_sets = reference_sets or {}

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        seed_smiles: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        if seed_smiles is None:
            return [0.0] * len(completions)

        gen_smiles_list = [extract_smiles(c) for c in completions]
        seed_list = list(seed_smiles)

        # Batch oracle predictions for all molecules (valid or not — oracle handles invalid)
        gen_probs = self.evaluator.predict(gen_smiles_list)
        seed_probs = self.evaluator.predict(seed_list)

        rewards = []
        n_valid = 0
        n_invalid = 0
        n_identity = 0
        n_memorized = 0
        n_improved = 0
        improvements = []

        for i, (gen_smi, seed_smi) in enumerate(zip(gen_smiles_list, seed_list)):
            mol = Chem.MolFromSmiles(gen_smi)

            # Gate: invalid molecules get fixed negative reward
            if mol is None:
                rewards.append(self.invalid_reward)
                n_invalid += 1
                continue

            n_valid += 1
            canon_gen = canonicalize(gen_smi)
            canon_seed = canonicalize(seed_smi)

            # Penalize identity (copying seed)
            if canon_gen == canon_seed:
                rewards.append(self.identity_penalty)
                n_identity += 1
                continue

            # Penalize memorization
            if canon_gen and any(canon_gen in s for s in self.reference_sets.values()):
                rewards.append(self.memorization_penalty)
                n_memorized += 1
                continue

            # Hard similarity gate: penalize off-target generations.
            # Computed once if needed; reused below for similarity_weight term.
            tanimoto_for_gate = None
            if self.similarity_threshold > 0:
                tanimoto_for_gate = compute_tanimoto(seed_smi, gen_smi) or 0.0
                if tanimoto_for_gate < self.similarity_threshold:
                    rewards.append(self.similarity_threshold_penalty)
                    continue

            # Valid, novel, not a copy — reward based on property + optional similarity
            gp = float(gen_probs[i])
            sp = float(seed_probs[i])

            # k = +1 when higher probability is better, -1 when lower is (e.g. hERG).
            k = self._sign
            crossed = (gp >= self.active_threshold) if k > 0 else (gp <= self.active_threshold)

            if self.property_mode == "active":
                # Signed distance past the boundary in the improving direction.
                prop_reward = k * (gp - self.active_threshold)
            elif self.property_mode == "threshold_bonus":
                prop_reward = k * (gp - sp) + self.active_bonus * float(crossed)
            else:  # "delta" (default)
                prop_reward = k * (gp - sp)

            reward = prop_reward
            if self.similarity_weight > 0:
                # Reuse the value computed for the gate if we already computed it
                tanimoto = tanimoto_for_gate
                if tanimoto is None:
                    tanimoto = compute_tanimoto(seed_smi, gen_smi) or 0.0
                reward += self.similarity_weight * tanimoto
            rewards.append(reward)
            improvements.append(prop_reward)
            if gp >= self.active_threshold:
                n_improved += 1

        total = max(len(completions), 1)
        if log_metric is not None:
            log_metric("rewards/gated_validity_rate", n_valid / total)
            log_metric("rewards/gated_identity_rate", n_identity / total)
            log_metric("rewards/gated_memorization_rate", n_memorized / total)
            log_metric("rewards/gated_improvement_rate", n_improved / total)
            if improvements:
                log_metric("rewards/gated_improvement_mean", np.mean(improvements))
            log_metric("rewards/gated_novel_valid_rate",
                        (n_valid - n_identity - n_memorized) / total)
            if self.similarity_weight > 0 and improvements:
                # Recompute tanimoto for logging (only for valid/novel/non-copy molecules)
                tanimotos = []
                for i, (gen_smi, seed_smi) in enumerate(zip(gen_smiles_list, seed_list)):
                    t = compute_tanimoto(seed_smi, gen_smi)
                    if t is not None and Chem.MolFromSmiles(gen_smi) is not None:
                        tanimotos.append(t)
                if tanimotos:
                    log_metric("rewards/gated_tanimoto_mean", np.mean(tanimotos))

        return rewards


class DiversityMetricReward:
    """Logging-only reward that tracks generation diversity.

    Always returns 0.0 (does not affect training). Used with weight=0
    to monitor diversity collapse via WandB.
    """

    __name__ = "diversity_metric"

    def __init__(self, num_generations: int = 8):
        self.num_generations = num_generations

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        # Compute unique fraction within each generation group
        gen_smiles = [extract_smiles(c) for c in completions]
        canonical = [canonicalize(s) for s in gen_smiles]

        # Group by prompt (num_generations per prompt)
        unique_fractions = []
        for i in range(0, len(canonical), self.num_generations):
            group = canonical[i : i + self.num_generations]
            valid = [s for s in group if s is not None]
            if valid:
                unique_fractions.append(len(set(valid)) / len(valid))

        if log_metric is not None and unique_fractions:
            log_metric("rewards/unique_fraction", np.mean(unique_fractions))

        # Overall unique count
        all_valid = [s for s in canonical if s is not None]
        if log_metric is not None and all_valid:
            log_metric("rewards/unique_count", len(set(all_valid)))
            log_metric("rewards/unique_rate", len(set(all_valid)) / len(all_valid))

        return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Binning config loading (for prompted RL)
# ---------------------------------------------------------------------------

def load_binning_configs(
    yaml_path: str | None = None,
) -> tuple[dict, dict]:
    """Load value and delta binning configs from the prompting YAML.

    Returns
    -------
    value_binning : dict mapping property name to PropertyBinningConfig
    delta_binning : dict mapping property name to PropertyBinningConfig
    """
    import importlib.resources
    import yaml
    from pairit.prompting.property_prompting import PropertyBinningConfig

    if yaml_path is None:
        yaml_path = str(
            importlib.resources.files("pairit")
            / "hydra_config" / "prompting" / "zinc_all_props.yaml"
        )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    props_cfg = cfg.get("properties", cfg)

    value_binning = {}
    vb = props_cfg.get("value_binning", {})
    default_num_vbins = vb.get("default", {}).get("num_bins", 32)
    for prop_name, spec in vb.get("per_property", {}).items():
        rng = spec.get("range")
        if rng:
            value_binning[prop_name] = PropertyBinningConfig(
                num_bins=spec.get("num_bins", default_num_vbins),
                range=tuple(rng),
                clip=True,
            )

    delta_binning = {}
    db = props_cfg.get("delta_binning", {})
    default_num_dbins = db.get("default", {}).get("num_bins", 16)
    for prop_name, spec in db.get("per_property", {}).items():
        rng = spec.get("range")
        if rng:
            delta_binning[prop_name] = PropertyBinningConfig(
                num_bins=spec.get("num_bins", default_num_dbins),
                range=tuple(rng),
                clip=True,
            )

    logger.info("Loaded binning configs: %d value, %d delta properties",
                len(value_binning), len(delta_binning))
    return value_binning, delta_binning


class PromptAdherenceReward:
    """Reward for following property prompt instructions.

    Measures how well the generated molecule matches the prompted direction
    and magnitude for each prompted property. Returns a continuous reward
    in [0, 1] averaged across prompted properties.

    For each prompted property:
    - Direction: +1 if the property changed in the prompted direction, 0 otherwise.
      Skipped if direction was not in the prompt (dropout).
    - Magnitude: 1 - |actual_bin - target_bin| / num_bins. Exact match = 1, max error = 0.
      Skipped if delta_bin was not in the prompt (dropout).

    These are averaged across all prompted components that are present.
    """

    __name__ = "prompt_adherence"

    def __init__(
        self,
        tokenizer,
        value_binning: dict,
        delta_binning: dict,
        direction_weight: float = 1.0,
        magnitude_weight: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.value_binning = value_binning
        self.delta_binning = delta_binning
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        seed_smiles: list[str] | None = None,
        prompt_tokens: list | None = None,
        log_metric: Any = None,
        **kwargs,
    ) -> list[float]:
        if seed_smiles is None or prompt_tokens is None:
            return [0.0] * len(completions)

        from pairit.prompting.property_prompting import (
            parse_property_prompt_from_ids,
            delta_to_bin,
        )
        from pairit.callbacks.molecule_validation.property_registry import (
            compute_all_properties_for_smiles,
        )

        gen_smiles_list = [extract_smiles(c) for c in completions]

        rewards = []
        direction_matches = []
        magnitude_scores = []

        for i, (gen_smi, seed_smi) in enumerate(zip(gen_smiles_list, seed_smiles)):
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is None:
                rewards.append(0.0)
                continue

            # Parse the prompt tokens for this generation
            ptoks = prompt_tokens[i] if i < len(prompt_tokens) else []
            if not ptoks:
                rewards.append(0.0)
                continue

            parsed = parse_property_prompt_from_ids(self.tokenizer, ptoks)
            if not parsed:
                rewards.append(0.0)
                continue

            # Compute properties for seed and generated
            seed_props = compute_all_properties_for_smiles(seed_smi)
            gen_props = compute_all_properties_for_smiles(gen_smi)

            if not seed_props or not gen_props:
                rewards.append(0.0)
                continue

            # Compute adherence for each prompted property
            component_scores = []

            for prompt in parsed:
                prop_name = prompt.property_name
                seed_val = seed_props.get(prop_name)
                gen_val = gen_props.get(prop_name)

                if seed_val is None or gen_val is None:
                    continue

                actual_delta = float(gen_val - seed_val)

                # Direction agreement
                if prompt.direction is not None:
                    if prompt.direction == "up":
                        match = actual_delta > 0.0
                    elif prompt.direction == "down":
                        match = actual_delta < 0.0
                    else:  # "any"
                        match = True
                    component_scores.append(self.direction_weight * float(match))
                    direction_matches.append(match)

                # Magnitude agreement (continuous)
                if prompt.delta_bin is not None and prop_name in self.delta_binning:
                    db = self.delta_binning[prop_name]
                    actual_bin = delta_to_bin(binning=db, delta=actual_delta, magnitude=True)
                    bin_error = abs(actual_bin - prompt.delta_bin)
                    mag_score = 1.0 - bin_error / max(db.num_bins, 1)
                    component_scores.append(self.magnitude_weight * mag_score)
                    magnitude_scores.append(mag_score)

            if component_scores:
                rewards.append(float(np.mean(component_scores)))
            else:
                rewards.append(0.0)

        if log_metric is not None:
            if direction_matches:
                log_metric("rewards/prompt_direction_agreement",
                           np.mean(direction_matches))
            if magnitude_scores:
                log_metric("rewards/prompt_magnitude_score",
                           np.mean(magnitude_scores))

        return rewards
