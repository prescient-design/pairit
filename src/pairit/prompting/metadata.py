"""Row identities and immutable request metadata for prompted generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from numbers import Integral


def stable_prompt_seed(*parts: object) -> int:
    """Process-independent seed; length prefixes disambiguate ordered pairs."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def pair_identifier(dataset: str, seed_smiles: str, target_smiles: str) -> str:
    """Keep the historical ordered-pair identity independent of prompt tokens."""
    return hashlib.sha256("\0".join((dataset, seed_smiles, target_smiles)).encode()).hexdigest()


def generation_pair_seed(seed: int, pair_id: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("generation seed must be an integer in [0, 2**32)")
    # Preserve the producer's seed namespace for existing checkpoint comparisons.
    payload = f"sha256-u32be:lobster_pair_generation_seed_v1\0{seed}\0{pair_id}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def prompt_metadata(seed_smiles, target_smiles, input_prompts, target_prompts) -> str:
    """JSON stays row-local under default Torch collation, even for subsets."""
    if not target_smiles or not target_prompts:
        raise ValueError("Prompt metadata export requires a paired target and at least one property")
    names = [p.property_name for p in target_prompts]
    if len(names) != len(set(names)):
        raise ValueError("Prompt property names must be unique")
    for p in target_prompts:
        values = (p.raw_delta, p.raw_seed_value, p.raw_target_value)
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError(f"Missing finite pair truth for property {p.property_name!r}")
    result = {
        "seed_smiles": seed_smiles,
        "target_smiles": target_smiles,
        "prompt_target_kind": "raw_pair_delta",
        "prompt_target_direction_property_names": names,
        "prompt_target_raw_direction_signs": [{"down": -1, "any": 0, "up": 1}[p.raw_direction]
                                              for p in target_prompts],
        "prompt_target_raw_deltas": [p.raw_delta for p in target_prompts],
        "prompt_target_raw_seed_values": [p.raw_seed_value for p in target_prompts],
        "prompt_target_raw_target_values": [p.raw_target_value for p in target_prompts],
        "prompt_input_properties": [p.property_name for p in input_prompts],
        "prompt_input_signed_delta_bins": [p.signed_delta_bin for p in input_prompts],
        "prompt_target_prompts": [asdict(p) for p in target_prompts],
    }
    return json.dumps(result, allow_nan=False, separators=(",", ":"))
