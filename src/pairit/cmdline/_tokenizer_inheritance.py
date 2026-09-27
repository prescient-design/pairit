"""
Tokenizer inheritance utilities for checkpoint compatibility.

When loading from a checkpoint, the checkpoint may contain special tokens (e.g., property prompt
tokens) that the current config doesn't define. This module provides utilities to:

1. Extract special tokens from a checkpoint's tokenizer
2. Validate that bin widths are compatible between checkpoint and current config
3. Prepare the tokenizer config to inherit tokens from the checkpoint

This ensures that:
- All checkpoint token IDs are preserved (model embeddings remain valid)
- New tokens can be added without conflict
- Incompatible configurations (different bin widths) are caught early with clear errors
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

# Register safe globals for PyTorch 2.6+ checkpoint loading
import pairit.cmdline._checkpoint_compat  # noqa: F401


def extract_checkpoint_tokenizer_info(ckpt_path: str) -> dict:
    """
    Extract tokenizer information from a checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file.

    Returns
    -------
    dict
        Dictionary containing:
        - success: bool - whether extraction succeeded
        - error: str | None - error message if failed
        - vocab_size: int | None - tokenizer vocabulary size
        - additional_special_tokens: list[str] - list of additional special tokens
        - value_bin_widths: set[int] - detected widths for value bin tokens
        - delta_bin_widths: set[int] - detected widths for delta bin tokens
        - distance_bin_widths: set[int] - detected widths for distance bin tokens
        - property_names: set[str] - detected property names from tokens
    """
    result = {
        "success": False,
        "error": None,
        "vocab_size": None,
        "additional_special_tokens": [],
        "value_bin_widths": set(),
        "delta_bin_widths": set(),
        "distance_bin_widths": set(),
        "property_names": set(),
    }

    if not ckpt_path:
        result["error"] = "No checkpoint path provided"
        return result

    ckpt_path_obj = Path(ckpt_path)
    if not ckpt_path_obj.exists():
        result["error"] = f"Checkpoint not found: {ckpt_path}"
        return result

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        result["error"] = f"Failed to load checkpoint: {e}"
        return result

    # Navigate to tokenizer
    hyper_params = checkpoint.get('hyper_parameters', {})
    if not hyper_params:
        result["error"] = "No hyper_parameters found in checkpoint"
        return result

    transform_fn = hyper_params.get('transform_fn')
    if transform_fn is None:
        result["error"] = "No transform_fn found in hyper_parameters"
        return result

    tokenizer = getattr(transform_fn, 'tokenizer', None)
    if tokenizer is None:
        result["error"] = "No tokenizer found in transform_fn"
        return result

    # Extract info
    result["vocab_size"] = len(tokenizer.get_vocab())
    result["additional_special_tokens"] = list(tokenizer.additional_special_tokens or [])

    # Analyze tokens to extract bin widths and property names
    for tok in result["additional_special_tokens"]:
        # Value bins: <p0_{name}_{bin}>
        match = re.match(r'^<p0_([^_]+(?:_[^_]+)*)_(\d+)>$', tok)
        if match:
            prop_name, bin_idx = match.groups()
            result["property_names"].add(prop_name)
            result["value_bin_widths"].add(len(bin_idx))
            continue

        # Delta bins: <pd_{name}_{bin}>
        match = re.match(r'^<pd_([^_]+(?:_[^_]+)*)_(\d+)>$', tok)
        if match:
            prop_name, bin_idx = match.groups()
            result["property_names"].add(prop_name)
            result["delta_bin_widths"].add(len(bin_idx))
            continue

        # Distance bins: <d_{bin}>
        match = re.match(r'^<d_(\d+)>$', tok)
        if match:
            bin_idx = match.group(1)
            result["distance_bin_widths"].add(len(bin_idx))
            continue

        # Property type tokens: <p_{name}>
        match = re.match(r'^<p_([^>]+)>$', tok)
        if match:
            result["property_names"].add(match.group(1))

    result["success"] = True
    return result


def compute_bin_widths_from_config(cfg: DictConfig) -> dict:
    """
    Compute expected bin widths from the current config.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra config.

    Returns
    -------
    dict
        Dictionary with keys:
        - value_width: int | None - expected value bin width
        - delta_width: int | None - expected delta bin width
        - distance_width: int | None - expected distance bin width
    """
    result = {
        "value_width": None,
        "delta_width": None,
        "distance_width": None,
    }

    # Try to get property prompt config
    try:
        tokenizer_cfg = cfg.data.transform_fn.tokenizer
        if tokenizer_cfg.get("add_property_prompt_tokens", False):
            prop_tokens = tokenizer_cfg.get("property_prompt_tokens", {})
            if prop_tokens:
                num_value_bins = int(prop_tokens.get("num_value_bins", 0) or 0)
                num_delta_bins = int(prop_tokens.get("num_delta_bins", 0) or 0)
                if num_value_bins > 0:
                    result["value_width"] = max(2, len(str(num_value_bins - 1)))
                if num_delta_bins > 0:
                    result["delta_width"] = max(2, len(str(num_delta_bins - 1)))

        # Distance tokens
        if tokenizer_cfg.get("add_distance_prompt_tokens", False):
            dist_tokens = tokenizer_cfg.get("distance_prompt_tokens", {})
            if dist_tokens:
                num_bins = int(dist_tokens.get("num_bins", 0) or 0)
                if num_bins > 0:
                    result["distance_width"] = max(2, len(str(num_bins - 1)))
    except Exception:
        pass  # Config structure may differ

    return result


def validate_bin_width_compatibility(
    ckpt_info: dict,
    config_widths: dict,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate that bin widths from checkpoint are compatible with current config.

    Parameters
    ----------
    ckpt_info : dict
        Output from extract_checkpoint_tokenizer_info().
    config_widths : dict
        Output from compute_bin_widths_from_config().
    strict : bool
        If True, raise ValueError on mismatch. If False, return issues list.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of issue descriptions)
    """
    issues = []

    # Value bin widths
    ckpt_value_widths = ckpt_info.get("value_bin_widths", set())
    config_value_width = config_widths.get("value_width")
    if ckpt_value_widths and config_value_width is not None:
        if config_value_width not in ckpt_value_widths:
            issues.append(
                f"Value bin width mismatch: checkpoint uses width(s) {ckpt_value_widths}, "
                f"but current config would use width {config_value_width}. "
                f"This would create duplicate tokens with different formats (e.g., <p0_logp_00> vs <p0_logp_000>). "
                f"Ensure num_value_bins produces the same width: "
                f"1-99 bins → width 2, 100-999 bins → width 3, etc."
            )

    # Delta bin widths
    ckpt_delta_widths = ckpt_info.get("delta_bin_widths", set())
    config_delta_width = config_widths.get("delta_width")
    if ckpt_delta_widths and config_delta_width is not None:
        if config_delta_width not in ckpt_delta_widths:
            issues.append(
                f"Delta bin width mismatch: checkpoint uses width(s) {ckpt_delta_widths}, "
                f"but current config would use width {config_delta_width}. "
                f"This would create duplicate tokens with different formats. "
                f"Ensure num_delta_bins produces the same width."
            )

    # Distance bin widths
    ckpt_distance_widths = ckpt_info.get("distance_bin_widths", set())
    config_distance_width = config_widths.get("distance_width")
    if ckpt_distance_widths and config_distance_width is not None:
        if config_distance_width not in ckpt_distance_widths:
            issues.append(
                f"Distance bin width mismatch: checkpoint uses width(s) {ckpt_distance_widths}, "
                f"but current config would use width {config_distance_width}. "
                f"This would create duplicate tokens with different formats."
            )

    is_valid = len(issues) == 0

    if not is_valid and strict:
        raise ValueError(
            "Checkpoint tokenizer incompatible with current config:\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )

    return is_valid, issues


def prepare_tokenizer_inheritance(cfg: DictConfig, ckpt_path: str | None, verbose: bool = True) -> dict:
    """
    Analyze checkpoint tokenizer and prepare config for token inheritance.

    This function:
    1. Extracts special tokens from the checkpoint's tokenizer
    2. Validates bin width compatibility
    3. Modifies cfg.data.transform_fn.tokenizer to inherit those tokens
    4. Prints detailed diagnostics (if verbose=True)

    Parameters
    ----------
    cfg : DictConfig
        The Hydra config (will be modified in place).
    ckpt_path : str | None
        Path to checkpoint, or None to skip inheritance.
    verbose : bool
        If True, print detailed diagnostics. Set to False for non-rank-0 workers.

    Returns
    -------
    dict
        Summary of what was done:
        - inherited: bool - whether tokens were inherited
        - num_tokens_inherited: int - number of tokens to inherit
        - checkpoint_vocab_size: int | None
        - checkpoint_properties: list[str]
        - issues: list[str] - any warnings
    """
    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    summary = {
        "inherited": False,
        "num_tokens_inherited": 0,
        "checkpoint_vocab_size": None,
        "checkpoint_properties": [],
        "issues": [],
    }

    if not ckpt_path:
        _print("[Tokenizer Inheritance] No checkpoint path provided, skipping inheritance.")
        return summary

    _print("\n" + "=" * 70)
    _print("TOKENIZER INHERITANCE CHECK")
    _print("=" * 70)
    _print(f"Checkpoint: {ckpt_path}")

    # Extract checkpoint tokenizer info
    ckpt_info = extract_checkpoint_tokenizer_info(ckpt_path)

    if not ckpt_info["success"]:
        _print(f"\n⚠️  Could not extract tokenizer from checkpoint: {ckpt_info['error']}")
        _print("    Proceeding without token inheritance (this may cause issues if checkpoint has special tokens)")
        summary["issues"].append(f"Could not extract tokenizer: {ckpt_info['error']}")
        _print("=" * 70 + "\n")
        return summary

    _print(f"\nCheckpoint tokenizer:")
    _print(f"  Vocab size: {ckpt_info['vocab_size']}")
    _print(f"  Additional special tokens: {len(ckpt_info['additional_special_tokens'])}")

    summary["checkpoint_vocab_size"] = ckpt_info["vocab_size"]

    if not ckpt_info["additional_special_tokens"]:
        _print("\n✓ Checkpoint has no additional special tokens to inherit.")
        _print("=" * 70 + "\n")
        return summary

    # Categorize checkpoint tokens
    ckpt_tokens = ckpt_info["additional_special_tokens"]
    dataset_tokens = [t for t in ckpt_tokens if t.startswith('<ds')]
    property_tokens = [t for t in ckpt_tokens if t.startswith('<p') and not t.startswith('<pad')]
    distance_tokens = [t for t in ckpt_tokens if t.startswith('<d') and not t.startswith('<ds')]
    other_tokens = [t for t in ckpt_tokens if t not in dataset_tokens + property_tokens + distance_tokens]

    _print(f"\n  Token breakdown:")
    if dataset_tokens:
        _print(f"    Dataset identity tokens: {len(dataset_tokens)}")
    if property_tokens:
        _print(f"    Property prompt tokens: {len(property_tokens)}")
        _print(f"    Properties detected: {sorted(ckpt_info['property_names'])}")
        summary["checkpoint_properties"] = sorted(ckpt_info["property_names"])
        if ckpt_info["value_bin_widths"]:
            _print(f"    Value bin width(s): {ckpt_info['value_bin_widths']}")
        if ckpt_info["delta_bin_widths"]:
            _print(f"    Delta bin width(s): {ckpt_info['delta_bin_widths']}")
    if distance_tokens:
        _print(f"    Distance prompt tokens: {len(distance_tokens)}")
        if ckpt_info["distance_bin_widths"]:
            _print(f"    Distance bin width(s): {ckpt_info['distance_bin_widths']}")
    if other_tokens:
        _print(f"    Other tokens: {len(other_tokens)}")

    # Compute current config widths
    config_widths = compute_bin_widths_from_config(cfg)

    _print(f"\n  Current config bin widths:")
    _print(f"    Value: {config_widths['value_width']}")
    _print(f"    Delta: {config_widths['delta_width']}")
    _print(f"    Distance: {config_widths['distance_width']}")

    # Validate compatibility (this will raise if strict=True and there's a mismatch)
    is_valid, issues = validate_bin_width_compatibility(ckpt_info, config_widths, strict=True)
    if issues:
        for issue in issues:
            _print(f"\n⚠️  {issue}")
        summary["issues"].extend(issues)

    # Analyze overlap with current config
    # We need to figure out what tokens the current config would generate
    current_would_generate = set()
    try:
        tokenizer_cfg = cfg.data.transform_fn.tokenizer

        # Dataset tokens
        if tokenizer_cfg.get("add_dataset_tokens", False):
            max_id = tokenizer_cfg.get("max_dataset_id", 0) or 0
            current_would_generate.add("<ds>")
            for i in range(max_id + 1):
                current_would_generate.add(f"<ds_{i}>")

        # Property tokens
        if tokenizer_cfg.get("add_property_prompt_tokens", False):
            prop_cfg = tokenizer_cfg.get("property_prompt_tokens", {})
            if prop_cfg:
                prop_names = sorted(set(str(p) for p in (prop_cfg.get("property_names") or [])))
                num_value = int(prop_cfg.get("num_value_bins", 0) or 0)
                num_delta = int(prop_cfg.get("num_delta_bins", 0) or 0)
                v_width = max(2, len(str(num_value - 1))) if num_value > 0 else 2
                d_width = max(2, len(str(num_delta - 1))) if num_delta > 0 else 2

                current_would_generate.add("<pc>")
                current_would_generate.update(["<pdir_up>", "<pdir_down>", "<pdir_any>"])
                for name in prop_names:
                    current_would_generate.add(f"<p_{name}>")
                    for b in range(num_value):
                        current_would_generate.add(f"<p0_{name}_{b:0{v_width}d}>")
                    for b in range(num_delta):
                        current_would_generate.add(f"<pd_{name}_{b:0{d_width}d}>")

        # Distance tokens
        if tokenizer_cfg.get("add_distance_prompt_tokens", False):
            dist_cfg = tokenizer_cfg.get("distance_prompt_tokens", {})
            if dist_cfg:
                num_bins = int(dist_cfg.get("num_bins", 0) or 0)
                if num_bins > 0:
                    width = max(2, len(str(num_bins - 1)))
                    current_would_generate.add("<dc>")
                    for b in range(num_bins):
                        current_would_generate.add(f"<d_{b:0{width}d}>")
    except Exception as e:
        _print(f"\n⚠️  Could not analyze current config tokens: {e}")
        summary["issues"].append(f"Could not analyze current config: {e}")

    ckpt_token_set = set(ckpt_tokens)
    overlap = ckpt_token_set & current_would_generate
    only_in_ckpt = ckpt_token_set - current_would_generate
    only_in_config = current_would_generate - ckpt_token_set

    _print(f"\n  Token overlap analysis:")
    _print(f"    Tokens in both checkpoint and current config: {len(overlap)}")
    _print(f"    Tokens only in checkpoint (will be inherited): {len(only_in_ckpt)}")
    _print(f"    Tokens only in current config (will be added): {len(only_in_config)}")

    if only_in_ckpt:
        # Categorize what's being inherited
        inherited_dataset = [t for t in only_in_ckpt if t.startswith('<ds')]
        inherited_property = [t for t in only_in_ckpt if t.startswith('<p') and not t.startswith('<pad')]
        inherited_distance = [t for t in only_in_ckpt if t.startswith('<d') and not t.startswith('<ds')]
        inherited_other = [t for t in only_in_ckpt if t not in inherited_dataset + inherited_property + inherited_distance]

        _print(f"\n    Inherited token breakdown:")
        if inherited_dataset:
            _print(f"      Dataset tokens: {len(inherited_dataset)}")
        if inherited_property:
            _print(f"      Property tokens: {len(inherited_property)}")
            # Extract unique properties being inherited
            inherited_props = set()
            for t in inherited_property:
                match = re.match(r'^<p_([^>]+)>$', t)
                if match:
                    inherited_props.add(match.group(1))
                match = re.match(r'^<p[0d]_([^_]+(?:_[^_]+)*)_\d+>$', t)
                if match:
                    inherited_props.add(match.group(1))
            if inherited_props:
                _print(f"        Properties: {sorted(inherited_props)}")
        if inherited_distance:
            _print(f"      Distance tokens: {len(inherited_distance)}")
        if inherited_other:
            _print(f"      Other tokens: {len(inherited_other)}")

    if only_in_config:
        _print(f"\n    Sample of new tokens to add: {list(only_in_config)[:5]}{'...' if len(only_in_config) > 5 else ''}")

    # Inject inherited tokens into config
    with open_dict(cfg):
        cfg.data.transform_fn.tokenizer.inherit_special_tokens = list(ckpt_tokens)

    summary["inherited"] = True
    summary["num_tokens_inherited"] = len(ckpt_tokens)

    _print(f"\n✓ Configured tokenizer to inherit {len(ckpt_tokens)} special tokens from checkpoint.")
    _print(f"  Final vocab will have: {len(ckpt_token_set | current_would_generate)} special tokens")
    _print("=" * 70 + "\n")

    return summary


def get_checkpoint_path_for_inheritance(cfg: DictConfig) -> str | None:
    """
    Get the checkpoint path from config, handling various config structures.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra config.

    Returns
    -------
    str | None
        The checkpoint path, or None if not found.
    """
    # Try model.ckpt_path first
    try:
        ckpt_path = cfg.model.get("ckpt_path")
        if ckpt_path and ckpt_path != "null":
            # Handle quoted paths from bash
            if isinstance(ckpt_path, str):
                s = ckpt_path.strip()
                if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                    s = s[1:-1].strip()
                return s if s and s != "null" else None
            return ckpt_path
    except Exception:
        pass

    return None
