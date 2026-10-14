from __future__ import annotations

"""
Property prompting utilities.

This module is designed to be imported from notebooks/scripts to:
- convert between numeric values and bins
- convert bins to numeric ranges
- build compact special-token prompt prefixes
- parse prompt specs from token IDs

Important: prompts must not include whitespace; the SMILES tokenizer treats whitespace
as unknown tokens and it will corrupt the conditioning prefix.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple


Direction = Literal["up", "down", "any"]
BinningMode = Literal["fixed", "inferred"]
DeltaMode = Literal["exact", "atmost", "atleast"]


@dataclass(frozen=True)
class PropertyBinningConfig:
    """
    Defines how to bin a property value (or delta/magnitude) into discrete bins.

    For mode='fixed', `range` must be provided and stable across runs.
    For mode='inferred', `range` can be None; the caller should supply inferred ranges.
    """

    num_bins: int
    range: Optional[Tuple[float, float]] = None
    mode: BinningMode = "fixed"
    clip: bool = True


@dataclass(frozen=True)
class PropertyPromptingConfig:
    """
    High-level prompting configuration.

    This mirrors Hydra config fields but stays pure-Python for notebook use.
    """

    property_names: Tuple[str, ...]
    num_value_bins: int
    num_delta_bins: int
    value_binning: PropertyBinningConfig
    delta_binning: PropertyBinningConfig
    value_width: int = 2
    delta_width: int = 2


@dataclass(frozen=True)
class ParsedPropertyPrompt:
    property_name: str
    start_bin: Optional[int] = None
    direction: Optional[Direction] = None
    delta_bin: Optional[int] = None
    delta_mode: DeltaMode = "exact"
    signed_delta_bin: Optional[int] = None
    # Pair truth is carried separately from model tokens. In particular, bin 08
    # in a 16-bin signed prompt contains both zero and small positive changes.
    raw_direction: Optional[Direction] = None
    raw_delta: Optional[float] = None
    raw_seed_value: Optional[float] = None
    raw_target_value: Optional[float] = None


def _validate_bins(num_bins: int, bin_idx: int) -> int:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if bin_idx < 0 or bin_idx >= num_bins:
        raise ValueError(f"bin_idx out of range [0,{num_bins-1}], got {bin_idx}")
    return int(bin_idx)


def _validate_range(rng: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = float(rng[0]), float(rng[1])
    if not (hi > lo):
        raise ValueError(f"range must satisfy hi>lo, got {rng!r}")
    return lo, hi


def bin_to_range(*, binning: PropertyBinningConfig, bin_idx: int) -> Tuple[float, float]:
    """
    Convert a bin index into a numeric interval [lo, hi) except the last bin which is [lo, hi].
    """
    _validate_bins(binning.num_bins, bin_idx)
    if binning.range is None:
        raise ValueError("binning.range must be provided for bin_to_range()")
    lo, hi = _validate_range(binning.range)
    width = (hi - lo) / float(binning.num_bins)
    b0 = lo + width * float(bin_idx)
    b1 = lo + width * float(bin_idx + 1)
    if bin_idx == binning.num_bins - 1:
        b1 = hi
    return float(b0), float(b1)


def value_to_bin(*, binning: PropertyBinningConfig, value: float) -> int:
    """
    Convert a numeric value into a bin index.
    """
    if binning.range is None:
        raise ValueError("binning.range must be provided for value_to_bin()")
    lo, hi = _validate_range(binning.range)
    x = float(value)
    if binning.clip:
        if x <= lo:
            return 0
        if x >= hi:
            return binning.num_bins - 1
    else:
        if x < lo or x > hi:
            raise ValueError(f"value {x} out of range {binning.range}")
    width = (hi - lo) / float(binning.num_bins)
    # floor into [0, num_bins-1]
    idx = int((x - lo) / width)
    if idx < 0:
        idx = 0
    if idx >= binning.num_bins:
        idx = binning.num_bins - 1
    return idx


def delta_to_bin(*, binning: PropertyBinningConfig, delta: float, magnitude: bool = True) -> int:
    """
    Convert a delta into a bin index. By default uses magnitude (abs(delta)).
    """
    d = abs(float(delta)) if magnitude else float(delta)
    return value_to_bin(binning=binning, value=d)


def _signed_delta_binning(binning: PropertyBinningConfig) -> PropertyBinningConfig:
    """Interpret the configured magnitude range ``[0, max]`` symmetrically."""
    if binning.range is None:
        raise ValueError("binning.range is required for signed delta bins")
    lo, hi = _validate_range(binning.range)
    if lo != 0.0 or not math.isfinite(hi):
        raise ValueError("signed delta bins require a finite range [0, positive max]")
    _validate_bins(binning.num_bins, 0)
    return replace(binning, range=(-hi, hi))


def signed_delta_to_bin(*, binning: PropertyBinningConfig, delta: float) -> int:
    """Map a finite signed delta onto equal-width bins over ``[-max, +max]``."""
    if not math.isfinite(float(delta)):
        raise ValueError(f"signed delta must be finite, got {delta!r}")
    return value_to_bin(binning=_signed_delta_binning(binning), value=float(delta))


def signed_bin_to_range(*, binning: PropertyBinningConfig, bin_idx: int) -> Tuple[float, float]:
    """Return the interval of a signed bin, using the same limits as encoding."""
    return bin_to_range(binning=_signed_delta_binning(binning), bin_idx=bin_idx)


def build_property_prompt_text(
    prompts: Sequence[ParsedPropertyPrompt],
    *,
    include_start_marker: bool = True,
    value_width: int = 2,
    delta_width: int = 2,
) -> str:
    """
    Build the compact prompt prefix (no whitespace).
    """
    parts: List[str] = []
    if include_start_marker:
        parts.append("<pc>")
    for p in prompts:
        parts.append(f"<p_{p.property_name}>")
        if p.start_bin is not None:
            parts.append(f"<p0_{p.property_name}_{p.start_bin:0{int(value_width)}d}>")
        if p.direction is not None:
            parts.append(f"<pdir_{p.direction}>")
        if p.signed_delta_bin is not None:
            if p.delta_bin is not None or p.delta_mode != "exact":
                raise ValueError("signed delta prompts cannot also specify unsigned or bounded deltas")
            parts.append(f"<psd_{p.property_name}_{p.signed_delta_bin:0{int(delta_width)}d}>")
        elif p.delta_bin is not None:
            if p.delta_mode == "atmost":
                parts.append(f"<pdmax_{p.property_name}_{p.delta_bin:0{int(delta_width)}d}>")
            elif p.delta_mode == "atleast":
                parts.append(f"<pdmin_{p.property_name}_{p.delta_bin:0{int(delta_width)}d}>")
            else:  # exact
                parts.append(f"<pd_{p.property_name}_{p.delta_bin:0{int(delta_width)}d}>")
    return "".join(parts)


def _parse_bin_suffix(token: str) -> int:
    # token ends with _NN> where NN may be 2+ digits
    core = token[:-1]  # strip '>'
    last = core.split("_")[-1]
    return int(last)


def parse_property_prompt_from_ids(tokenizer: Any, ids: Sequence[int]) -> List[ParsedPropertyPrompt]:
    """
    Parse property prompt tokens from token IDs.

    We decode tokens individually for robustness across tokenizers; this is only
    used in validation / analysis paths.
    """
    toks = tokenizer.convert_ids_to_tokens(list(ids))
    out: List[ParsedPropertyPrompt] = []
    cur: Optional[ParsedPropertyPrompt] = None

    for tok in toks:
        if tok == "<pc>":
            continue
        if tok.startswith("<p_") and tok.endswith(">") and not tok.startswith("<p0_") and not tok.startswith("<pd_"):
            # start a new property block
            name = tok[len("<p_") : -1]
            cur = ParsedPropertyPrompt(property_name=name)
            out.append(cur)
            continue
        if cur is None:
            continue
        if tok.startswith(f"<p0_{cur.property_name}_") and tok.endswith(">"):
            b = _parse_bin_suffix(tok)
            cur = ParsedPropertyPrompt(
                property_name=cur.property_name,
                start_bin=b,
                direction=cur.direction,
                delta_bin=cur.delta_bin,
                delta_mode=cur.delta_mode,
                signed_delta_bin=cur.signed_delta_bin,
            )
            out[-1] = cur
            continue
        if tok.startswith("<pdir_") and tok.endswith(">"):
            d = tok[len("<pdir_") : -1]
            if d not in ("up", "down", "any"):
                continue
            cur = ParsedPropertyPrompt(
                property_name=cur.property_name,
                start_bin=cur.start_bin,
                direction=d,  # type: ignore[assignment]
                delta_bin=cur.delta_bin,
                delta_mode=cur.delta_mode,
                signed_delta_bin=cur.signed_delta_bin,
            )
            out[-1] = cur
            continue
        if tok.startswith(f"<psd_{cur.property_name}_") and tok.endswith(">"):
            cur = replace(cur, signed_delta_bin=_parse_bin_suffix(tok))
            out[-1] = cur
            continue
        # Bounded delta: <pdmax_{name}_{bb}>
        if tok.startswith(f"<pdmax_{cur.property_name}_") and tok.endswith(">"):
            b = _parse_bin_suffix(tok)
            cur = ParsedPropertyPrompt(
                property_name=cur.property_name,
                start_bin=cur.start_bin,
                direction=cur.direction,
                delta_bin=b,
                delta_mode="atmost",
                signed_delta_bin=cur.signed_delta_bin,
            )
            out[-1] = cur
            continue
        # Bounded delta: <pdmin_{name}_{bb}>
        if tok.startswith(f"<pdmin_{cur.property_name}_") and tok.endswith(">"):
            b = _parse_bin_suffix(tok)
            cur = ParsedPropertyPrompt(
                property_name=cur.property_name,
                start_bin=cur.start_bin,
                direction=cur.direction,
                delta_bin=b,
                delta_mode="atleast",
                signed_delta_bin=cur.signed_delta_bin,
            )
            out[-1] = cur
            continue
        # Exact delta: <pd_{name}_{bb}>
        if tok.startswith(f"<pd_{cur.property_name}_") and tok.endswith(">"):
            b = _parse_bin_suffix(tok)
            cur = ParsedPropertyPrompt(
                property_name=cur.property_name,
                start_bin=cur.start_bin,
                direction=cur.direction,
                delta_bin=b,
                delta_mode="exact",
                signed_delta_bin=cur.signed_delta_bin,
            )
            out[-1] = cur
            continue

    return out


def format_property_prompt_summary(
    prompts: Sequence[ParsedPropertyPrompt],
    *,
    value_binning: PropertyBinningConfig | None = None,
    delta_binning: PropertyBinningConfig | None = None,
    value_binning_by_property: Mapping[str, PropertyBinningConfig] | None = None,
    delta_binning_by_property: Mapping[str, PropertyBinningConfig] | None = None,
    show_ranges: bool = True,
    decimals: int = 3,
) -> str:
    """
    Render a human-readable summary of a parsed property prompt.

    This is meant for logs/UI. The underlying prompt tokens remain compact and whitespace-free.

    Example (bins only):
      PC[logp start=07 dir=any |Δ|=03][tpsa start=03 dir=any |Δ|=01]

    Example (with ranges, when binning ranges are available):
      PC[logp start=07(-0.25,0.00) dir=any |Δ|=03(0.56,0.75)]...
    """
    if not prompts:
        return "PC[]"

    def _binning_for(prop: str, kind: str) -> PropertyBinningConfig | None:
        if kind == "value":
            if value_binning_by_property and prop in value_binning_by_property:
                return value_binning_by_property[prop]
            return value_binning
        if delta_binning_by_property and prop in delta_binning_by_property:
            return delta_binning_by_property[prop]
        return delta_binning

    def _fmt_rng(rng: Tuple[float, float]) -> str:
        lo, hi = rng
        return f"({lo:.{decimals}f},{hi:.{decimals}f})"

    _DELTA_SYMBOL: dict[DeltaMode, str] = {
        "exact": "|Δ|=",
        "atmost": "|Δ|<=",
        "atleast": "|Δ|>=",
    }

    parts: List[str] = ["PC"]
    for p in prompts:
        seg: List[str] = [p.property_name]

        if p.start_bin is not None:
            seg.append(f"start={int(p.start_bin):02d}")
            if show_ranges:
                vb = _binning_for(p.property_name, "value")
                if vb is not None and vb.range is not None:
                    try:
                        seg[-1] += _fmt_rng(bin_to_range(binning=vb, bin_idx=int(p.start_bin)))
                    except Exception:
                        pass

        if p.direction is not None:
            seg.append(f"dir={p.direction}")

        if p.signed_delta_bin is not None:
            seg.append(f"signed-Δ={int(p.signed_delta_bin):02d}")
            if show_ranges:
                db = _binning_for(p.property_name, "delta")
                if db is not None and db.range is not None:
                    seg[-1] += _fmt_rng(signed_bin_to_range(binning=db, bin_idx=p.signed_delta_bin))

        if p.delta_bin is not None:
            sym = _DELTA_SYMBOL.get(p.delta_mode, "|Δ|=")
            seg.append(f"{sym}{int(p.delta_bin):02d}")
            if show_ranges:
                db = _binning_for(p.property_name, "delta")
                if db is not None and db.range is not None:
                    try:
                        seg[-1] += _fmt_rng(bin_to_range(binning=db, bin_idx=int(p.delta_bin)))
                    except Exception:
                        pass

        parts.append("[" + " ".join(seg) + "]")

    return "".join(parts)
