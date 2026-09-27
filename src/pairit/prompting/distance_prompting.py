from __future__ import annotations

"""
Distance prompting utilities.

This module provides functions for:
- Fitting Beta distributions to Tanimoto distance distributions
- Computing percentiles from distributions
- Converting percentiles to bins
- Building distance prompt text
- Parsing distance prompts from token IDs

Important: prompts must not include whitespace; the SMILES tokenizer treats whitespace
as unknown tokens and it will corrupt the conditioning prefix.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

try:
    from scipy.stats import beta
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    beta = None


DistanceBinningMode = Literal["curve_fit", "fixed_quartiles"]


@dataclass(frozen=True)
class DistanceDistributionParams:
    """Parameters for a Beta distribution fit to distance data."""

    alpha: float
    beta: float
    p0: float  # 0th percentile
    p25: float  # 25th percentile
    p50: float  # 50th percentile
    p75: float  # 75th percentile
    p100: float  # 100th percentile
    num_pairs: int  # Number of pairs used for fitting


def fit_beta_distribution(distances: Sequence[float]) -> tuple[float, float]:
    """
    Fit a Beta distribution to a sequence of distances (0-1 range).

    Args:
        distances: Sequence of distance values in [0, 1]

    Returns:
        Tuple of (alpha, beta) parameters

    Raises:
        ValueError: If scipy is not available or fitting fails
    """
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy is required for Beta distribution fitting")

    import numpy as np

    distances_array = np.array(distances, dtype=float)
    
    # Handle edge cases
    if len(distances_array) == 0:
        # Uniform distribution
        return (1.0, 1.0)
    
    if len(distances_array) == 1:
        # Single value - use uniform
        return (1.0, 1.0)
    
    # Check if all values are the same
    if np.allclose(distances_array, distances_array[0]):
        # Uniform distribution
        return (1.0, 1.0)
    
    # Clip to [0, 1] range
    distances_array = np.clip(distances_array, 0.0, 1.0)
    
    try:
        # Fit Beta distribution
        # Using method of moments as initial guess, then MLE
        alpha, beta_param, loc, scale = beta.fit(distances_array, floc=0.0, fscale=1.0)
        return (float(alpha), float(beta_param))
    except Exception:
        # Fallback to uniform
        return (1.0, 1.0)


def compute_percentiles_from_beta(alpha: float, beta_param: float) -> tuple[float, float, float, float, float]:
    """
    Compute 0%, 25%, 50%, 75%, 100% percentiles from Beta distribution.

    Args:
        alpha: Alpha parameter of Beta distribution
        beta_param: Beta parameter of Beta distribution

    Returns:
        Tuple of (p0, p25, p50, p75, p100)
    """
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy is required for percentile computation")

    dist = beta(alpha, beta_param)
    p0 = float(dist.ppf(0.0))
    p25 = float(dist.ppf(0.25))
    p50 = float(dist.ppf(0.50))
    p75 = float(dist.ppf(0.75))
    p100 = float(dist.ppf(1.0))
    
    return (p0, p25, p50, p75, p100)


def percentile_from_beta_cdf(distance: float, alpha: float, beta_param: float) -> float:
    """
    Compute percentile (0-100) of a distance value using Beta distribution CDF.

    Args:
        distance: Distance value in [0, 1]
        alpha: Alpha parameter of Beta distribution
        beta_param: Beta parameter of Beta distribution

    Returns:
        Percentile value in [0, 100]
    """
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy is required for percentile computation")

    dist = beta(alpha, beta_param)
    percentile = float(dist.cdf(distance)) * 100.0
    return max(0.0, min(100.0, percentile))  # Clip to [0, 100]


def distance_to_bin_fixed_quartiles(
    distance: float,
    stored_percentiles: tuple[float, float, float, float, float],
) -> int:
    """
    Convert a distance value to a bin index using fixed quartiles.

    Args:
        distance: Distance value in [0, 1]
        stored_percentiles: Tuple of (p0, p25, p50, p75, p100) - distance values at percentiles

    Returns:
        Bin index in [0, 3]
    """
    p0, p25, p50, p75, p100 = stored_percentiles
    distance = max(0.0, min(1.0, float(distance)))  # Clip to [0, 1]
    
    if distance <= p25:
        return 0
    elif distance <= p50:
        return 1
    elif distance <= p75:
        return 2
    else:
        return 3


def percentile_to_bin(
    percentile: float,
    mode: DistanceBinningMode,
    num_bins: int,
    stored_percentiles: Optional[tuple[float, float, float, float, float]] = None,
) -> int:
    """
    Convert a percentile value (0-100) to a bin index.

    Args:
        percentile: Percentile value in [0, 100]
        mode: Binning mode - "curve_fit" or "fixed_quartiles"
        num_bins: Number of bins (used in curve_fit mode)
        stored_percentiles: Tuple of (p0, p25, p50, p75, p100) for fixed_quartiles mode (not used here)

    Returns:
        Bin index in [0, num_bins-1] for curve_fit, or [0, 3] for fixed_quartiles
    """
    percentile = max(0.0, min(100.0, float(percentile)))  # Clip to [0, 100]

    if mode == "fixed_quartiles":
        # For fixed_quartiles mode, map percentiles directly to 4 bins
        if percentile <= 25.0:
            return 0
        elif percentile <= 50.0:
            return 1
        elif percentile <= 75.0:
            return 2
        else:
            return 3
    else:  # curve_fit
        # Divide 0-100 range into num_bins equal bins
        bin_width = 100.0 / float(num_bins)
        bin_idx = int(percentile / bin_width)
        # Ensure bin_idx is in valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        return bin_idx


def build_distance_prompt_text(bin_idx: int, *, include_start_marker: bool = True, bin_width: int = 2) -> str:
    """
    Build the compact distance prompt prefix (no whitespace).

    Args:
        bin_idx: Bin index
        include_start_marker: Whether to include <dc> start marker
        bin_width: Width for zero-padding bin index

    Returns:
        Prompt string (e.g., "<dc><d_05>")
    """
    parts: list[str] = []
    if include_start_marker:
        parts.append("<dc>")
    parts.append(f"<d_{bin_idx:0{bin_width}d}>")
    return "".join(parts)


def parse_distance_prompt_from_ids(tokenizer: Any, ids: Sequence[int]) -> Optional[int]:
    """
    Parse distance prompt bin from token IDs.

    Args:
        tokenizer: Tokenizer instance
        ids: Sequence of token IDs

    Returns:
        Bin index if found, None otherwise
    """
    toks = tokenizer.convert_ids_to_tokens(list(ids))
    
    for tok in toks:
        if tok == "<dc>":
            continue
        if tok.startswith("<d_") and tok.endswith(">"):
            # Extract bin index
            core = tok[3:-1]  # Remove "<d_" and ">"
            try:
                return int(core)
            except ValueError:
                continue
    
    return None

