"""Dataset utilities for RL post-training.

Handles loading seed molecules and constructing datasets in the format
expected by TRL's GRPOTrainer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_seeds_from_parquet(
    parquet_path: str,
    smiles_col: str = "smiles",
    split_col: str | None = "split",
    split_vals: list[str] | None = None,
    exclude_splits: list[str] | None = None,
    max_seeds: int | None = None,
) -> list[str]:
    """Load unique seed SMILES from a parquet file.

    Parameters
    ----------
    parquet_path : str
        Path to molecules parquet file.
    smiles_col : str
        Column containing SMILES strings.
    split_col : str | None
        Column for split filtering. None to skip filtering.
    split_vals : list[str] | None
        Which splits to include (e.g. ["train"]).
        None to include all (subject to exclude_splits).
    exclude_splits : list[str] | None
        Splits to exclude (e.g. ["val", "test"]). Applied after split_vals.
    max_seeds : int | None
        Maximum number of seeds to return. None for all.

    Returns
    -------
    list[str]
        Unique SMILES strings.
    """
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    # Log full split distribution for transparency
    if split_col and split_col in df.columns:
        split_counts = df[split_col].value_counts().to_dict()
        logger.info("Split distribution in parquet: %s", split_counts)

        if split_vals:
            df = df[df[split_col].isin(split_vals)]
            logger.info("Filtered to splits %s: %d rows", split_vals, len(df))
        if exclude_splits:
            before = len(df)
            df = df[~df[split_col].isin(exclude_splits)]
            excluded_count = before - len(df)
            if excluded_count > 0:
                logger.info("Excluded %d rows from splits %s: %d rows remaining",
                            excluded_count, exclude_splits, len(df))

        # Warn if test/val molecules are included
        remaining_splits = df[split_col].unique().tolist() if split_col in df.columns else []
        for danger_split in ["test", "val", "validation"]:
            if danger_split in remaining_splits:
                logger.warning(
                    "WARNING: '%s' split molecules are included in RL training seeds! "
                    "This will leak test data. Use --exclude-splits to remove them.",
                    danger_split,
                )
    else:
        logger.warning("No '%s' column found in parquet — cannot verify train/test split", split_col)

    smiles_list = df[smiles_col].dropna().unique().tolist()
    logger.info("Found %d unique SMILES", len(smiles_list))

    if max_seeds is not None:
        smiles_list = smiles_list[:max_seeds]
        logger.info("Truncated to %d seeds", len(smiles_list))

    return smiles_list


def build_grpo_dataset(
    seed_smiles_list: list[str],
    tokenizer: "PreTrainedTokenizerBase",
) -> Dataset:
    """Build a HuggingFace Dataset for GRPO training.

    Each row has a ``prompt`` (token IDs up to and including ``<sep>``)
    and ``seed_smiles`` (raw string for reward functions).

    The prompt format matches the SFT training format::

        <cls> seed_smiles <sep>

    Parameters
    ----------
    seed_smiles_list : list[str]
        Raw SMILES strings to use as generation seeds.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer (should have ``cls_token_id``, ``sep_token_id``).

    Returns
    -------
    Dataset
        With columns ``prompt`` (list[int]) and ``seed_smiles`` (str).
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    prompts = []
    valid_seeds = []
    for smiles in seed_smiles_list:
        # Tokenize SMILES without special tokens (we add them manually)
        token_ids = tokenizer.encode(smiles, add_special_tokens=False)
        # Build prompt: <cls> smiles_tokens <sep>
        prompt_ids = [cls_id] + token_ids + [sep_id]
        prompts.append(prompt_ids)
        valid_seeds.append(smiles)

    dataset = Dataset.from_dict({"prompt": prompts, "seed_smiles": valid_seeds})
    logger.info("Built GRPO dataset: %d seeds", len(dataset))
    return dataset


def build_prompted_grpo_dataset(
    seed_smiles_list: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    property_names: list[str],
    value_binning: dict,
    delta_binning: dict,
    prompt_mode: str = "random",
    fixed_direction: str | None = None,
    fixed_delta_bin: int | None = None,
    dropout_rate: float = 0.0,
    seed: int = 0,
) -> Dataset:
    """Build a GRPO dataset with property prompts in the conditioning sequence.

    Parameters
    ----------
    seed_smiles_list : list[str]
        Raw SMILES strings to use as generation seeds.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer with property prompt tokens in vocabulary.
    property_names : list[str]
        Properties to include in prompts (e.g. ["logp"] or ["logp", "sa"]).
    value_binning : dict
        Property name -> PropertyBinningConfig for value bins.
    delta_binning : dict
        Property name -> PropertyBinningConfig for delta bins.
    prompt_mode : str
        "random" (sample random direction/delta per seed) or
        "fixed" (same prompt for all seeds, using fixed_direction/fixed_delta_bin).
    fixed_direction : str | None
        Direction for fixed mode ("up" or "down").
    fixed_delta_bin : int | None
        Delta bin for fixed mode.
    dropout_rate : float
        Probability of dropping each prompt component (direction, delta_bin)
        independently. 0.0 = no dropout.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dataset
        With columns: ``prompt`` (list[int]), ``seed_smiles`` (str),
        ``prompt_tokens`` (list[int]).
    """
    import random as _random
    from pairit.prompting.property_prompting import (
        ParsedPropertyPrompt,
        build_property_prompt_text,
        value_to_bin,
    )
    from pairit.callbacks.molecule_validation.property_registry import (
        compute_all_properties_for_smiles,
    )

    rng = _random.Random(seed)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    all_prompts = []
    all_seeds = []
    all_prompt_tokens = []

    for smiles in seed_smiles_list:
        # Compute seed properties
        seed_props = compute_all_properties_for_smiles(smiles)
        if not seed_props:
            continue

        # Build property prompt for each requested property
        parsed_prompts = []
        for prop_name in property_names:
            seed_val = seed_props.get(prop_name)
            if seed_val is None or prop_name not in value_binning:
                continue

            vb = value_binning[prop_name]
            db = delta_binning.get(prop_name)

            # Start bin from seed's actual value
            start_bin = value_to_bin(binning=vb, value=seed_val)

            # Direction
            if prompt_mode == "fixed" and fixed_direction:
                direction = fixed_direction
            else:
                direction = rng.choice(["up", "down"])

            # Delta bin
            if prompt_mode == "fixed" and fixed_delta_bin is not None:
                delta_bin = fixed_delta_bin
            elif db is not None:
                delta_bin = rng.randint(0, db.num_bins - 1)
            else:
                delta_bin = None

            # Apply component dropout
            if dropout_rate > 0:
                if rng.random() < dropout_rate:
                    direction = None
                if rng.random() < dropout_rate:
                    delta_bin = None

            parsed_prompts.append(ParsedPropertyPrompt(
                property_name=prop_name,
                start_bin=start_bin,
                direction=direction,
                delta_bin=delta_bin,
                delta_mode="exact",
            ))

        if not parsed_prompts:
            continue

        # Build prompt text and tokenize
        prompt_text = build_property_prompt_text(parsed_prompts)
        prompt_tok_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Full conditioning: <cls> prompt_tokens smiles_tokens <sep>
        smiles_tok_ids = tokenizer.encode(smiles, add_special_tokens=False)
        full_prompt = [cls_id] + prompt_tok_ids + smiles_tok_ids + [sep_id]

        all_prompts.append(full_prompt)
        all_seeds.append(smiles)
        all_prompt_tokens.append(prompt_tok_ids)

    dataset = Dataset.from_dict({
        "prompt": all_prompts,
        "seed_smiles": all_seeds,
        "prompt_tokens": all_prompt_tokens,
    })
    logger.info("Built prompted GRPO dataset: %d seeds, properties=%s, mode=%s",
                len(dataset), property_names, prompt_mode)
    return dataset


