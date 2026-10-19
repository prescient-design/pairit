"""Utilities for converting between Lightning and HuggingFace checkpoints.

Enables RL post-training with TRL (which expects HuggingFace models) while
preserving compatibility with pairit_eval (which expects Lightning checkpoints).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizerBase

# Historical paper checkpoints store tokenizer classes under their former name.
import pairit.cmdline._checkpoint_compat  # noqa: F401

logger = logging.getLogger(__name__)


def load_hf_for_export(checkpoint_dir: str, base_checkpoint_path: str) -> LlamaForCausalLM:
    """Load full weights, or merge saved adapters into their original SFT base.

    Use the explicit Lightning base for adapters: their recorded HF base path
    may point to temporary storage on a training worker that no longer exists.
    """
    if (Path(checkpoint_dir) / "adapter_config.json").exists():
        from peft import PeftModel

        base_model, _ = lightning_to_hf(base_checkpoint_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        return model.merge_and_unload()
    return LlamaForCausalLM.from_pretrained(checkpoint_dir)


def lightning_to_hf(
    ckpt_path: str,
) -> tuple[LlamaForCausalLM, "PreTrainedTokenizerBase"]:
    """Load a Lightning checkpoint and extract the HuggingFace model + tokenizer.

    Parameters
    ----------
    ckpt_path : str
        Path to a PairITCLM Lightning ``.ckpt`` file.

    Returns
    -------
    model : LlamaForCausalLM
        The HuggingFace causal LM with trained weights.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer stored in the checkpoint (with any special tokens
        that were added during training).
    """
    logger.info("Loading Lightning checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- Extract tokenizer from hyper_parameters.transform_fn ---
    hp = checkpoint.get("hyper_parameters", {})
    transform_fn = hp.get("transform_fn")
    if transform_fn is None:
        raise ValueError(
            f"Checkpoint at {ckpt_path} has no hyper_parameters.transform_fn. "
            "Cannot extract tokenizer."
        )
    tokenizer = getattr(transform_fn, "tokenizer", None) or getattr(
        transform_fn, "_auto_tokenizer", None
    )
    if tokenizer is None:
        raise ValueError("Could not find tokenizer in checkpoint transform_fn.")

    tokenizer.padding_side = "left"  # required by TRL for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Extracted tokenizer (vocab_size=%d)", len(tokenizer))

    # --- Extract model config from checkpoint ---
    # The Lightning state_dict keys look like "model.lm_head.weight", etc.
    raw_state_dict = checkpoint["state_dict"]
    hf_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("model."):
            hf_state_dict[key[len("model.") :]] = value
        else:
            hf_state_dict[key] = value

    # Reconstruct config from checkpoint's hyper_parameters
    config = _build_config_from_checkpoint(hp, tokenizer)
    model = LlamaForCausalLM(config)

    # Load weights (strict=False in case of minor mismatches from version changes)
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading state_dict: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading state_dict: %s", unexpected)

    # Save to a temp directory so TRL can load a reference model copy when beta > 0.
    # TRL's GRPOTrainer reads model.config._name_or_path to create the ref model;
    # since we built the model manually, that field is empty and TRL crashes.
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="lobster_hf_ref_")
    model.save_pretrained(tmpdir)
    tokenizer.save_pretrained(tmpdir)
    model = LlamaForCausalLM.from_pretrained(tmpdir)
    logger.info("Saved and reloaded HF model via %s (for TRL ref model compat)", tmpdir)

    # Restore left padding for TRL
    tokenizer.padding_side = "left"

    logger.info(
        "Loaded LlamaForCausalLM (%d params)",
        sum(p.numel() for p in model.parameters()),
    )
    return model, tokenizer


def hf_to_lightning(
    hf_model: LlamaForCausalLM,
    tokenizer: "PreTrainedTokenizerBase",
    original_ckpt_path: str,
    output_path: str,
) -> None:
    """Save a HuggingFace model back into a Lightning checkpoint.

    Re-uses the ``hyper_parameters`` from the original SFT checkpoint so that
    ``pairit_eval`` can load the result with ``PairITCLM.load_from_checkpoint``.

    Parameters
    ----------
    hf_model : LlamaForCausalLM
        The (possibly RL-trained) HuggingFace model.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer (may have been modified, e.g. padding_side).
    original_ckpt_path : str
        Path to the original SFT Lightning checkpoint (used as template
        for hyper_parameters).
    output_path : str
        Where to save the new ``.ckpt`` file.
    """
    hf_state_dict = hf_model.state_dict()
    adapter_keys = [
        key for key in hf_state_dict
        if ".lora_" in key or ".base_layer." in key or key.startswith("base_model.")
    ]
    if adapter_keys:
        raise ValueError(
            "Merge adapters before exporting a plain Lightning checkpoint; "
            f"unmerged parameter names include {adapter_keys[:3]}"
        )

    logger.info("Loading original checkpoint template: %s", original_ckpt_path)
    original = torch.load(original_ckpt_path, map_location="cpu", weights_only=False)

    # Build new state_dict with "model." prefix
    new_state_dict = {f"model.{k}": v for k, v in hf_state_dict.items()}

    # Build checkpoint preserving all Lightning metadata from the original
    new_checkpoint = copy.deepcopy(original)
    new_checkpoint["state_dict"] = new_state_dict
    new_checkpoint["epoch"] = 0
    new_checkpoint["global_step"] = 0

    # Update tokenizer in hyper_parameters so eval gets the right vocab
    hp = new_checkpoint["hyper_parameters"]
    transform_fn = hp.get("transform_fn")
    if transform_fn is not None:
        # Restore right-padding for eval (TRL uses left, eval uses right)
        tokenizer_copy = copy.deepcopy(tokenizer)
        tokenizer_copy.padding_side = "right"
        if hasattr(transform_fn, "tokenizer"):
            transform_fn.tokenizer = tokenizer_copy
        elif hasattr(transform_fn, "_auto_tokenizer"):
            transform_fn._auto_tokenizer = tokenizer_copy

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    logger.info("Saved Lightning checkpoint to %s", output_path)


def _build_config_from_checkpoint(
    hp: dict, tokenizer: "PreTrainedTokenizerBase"
) -> LlamaConfig:
    """Reconstruct a LlamaConfig from checkpoint hyper_parameters."""
    from pairit.model._clm_configuration import PCLM_CONFIG_ARGS

    model_name = hp.get("model_name", "CLM_68M")
    config_args = PCLM_CONFIG_ARGS[model_name]

    max_length = hp.get("max_length", 512)
    num_key_value_heads = hp.get("num_key_value_heads") or config_args["num_attention_heads"]
    attention_bias = hp.get("attention_bias", False)

    config = LlamaConfig(
        **config_args,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=max_length,
        num_key_value_heads=num_key_value_heads,
        attention_bias=attention_bias,
    )
    return config
