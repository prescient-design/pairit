"""Scored candidate dataclass and scoring helpers for inference-time compute.

``ScoredCandidate`` is the central data structure for inference-time compute.
Every generated molecule is wrapped in a ``ScoredCandidate`` that carries:

- The raw and canonical SMILES
- The seed molecule that produced it
- All computed molecular properties (logP, QED, SA, etc.)
- The seed's properties (for computing deltas)
- Validity, memorization, and per-property improvement flags
- Tanimoto distance to seed
- Lineage information (parent chain for beam search)
- Selection status and beam rank

The ``to_dict()`` method flattens all fields into a dictionary suitable for
constructing a pandas DataFrame row, including derived columns like
``{prop}_delta``, ``is_improved_{prop}``, and ``is_vin_{prop}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoredCandidate:
    """A generated molecule with all computed scores and metadata.

    Parameters
    ----------
    smiles : str
        Raw generated SMILES string.
    canonical : str | None
        Canonical SMILES (None if invalid).
    seed_smiles : str
        SMILES of the seed molecule that produced this generation.
    seed_idx : int
        Index of the seed in the original seed list.
    is_valid : bool
        Whether the SMILES is chemically valid.
    is_memorized : bool
        Whether the molecule appears in any reference set.
    memorized_by_ref : dict[str, bool]
        Per-reference-set memorization status.
    is_improved : dict[str, bool]
        Per-property improvement status (gen improved over seed).
    properties : dict[str, float | None]
        All computed property values for the generated molecule.
    seed_properties : dict[str, float | None]
        All computed property values for the seed molecule.
    tanimoto_to_seed : float | None
        Tanimoto distance from generated molecule to its seed.
    round_num : int
        Generation round (0 for best-of-N).
    generation_idx : int
        Index within the seed's generations for this round.
    parent_smiles : str | None
        Direct parent molecule (for beam search lineage tracking).
    lineage : list[str] | None
        Full chain of ancestor SMILES from initial seed to parent.
    selected : bool
        Whether this candidate was selected by the strategy.
    beam_rank : int | None
        Rank in beam after selection (beam search only).
    """

    smiles: str
    canonical: str | None
    seed_smiles: str
    seed_idx: int
    is_valid: bool = False
    is_memorized: bool = False
    memorized_by_ref: dict[str, bool] = field(default_factory=dict)
    is_improved: dict[str, bool] = field(default_factory=dict)
    properties: dict[str, float | None] = field(default_factory=dict)
    seed_properties: dict[str, float | None] = field(default_factory=dict)
    tanimoto_to_seed: float | None = None
    round_num: int = 0
    generation_idx: int = 0
    parent_smiles: str | None = None
    lineage: list[str] | None = None
    selected: bool = False
    beam_rank: int | None = None

    def is_vin(self, property_name: str) -> bool:
        """Check if this candidate passes the V,I,N gate for a property."""
        return (
            self.is_valid
            and self.is_improved.get(property_name, False)
            and not self.is_memorized
        )

    def property_delta(self, property_name: str) -> float | None:
        """Compute property delta (generated - seed)."""
        gen_val = self.properties.get(property_name)
        seed_val = self.seed_properties.get(property_name)
        if gen_val is None or seed_val is None:
            return None
        return gen_val - seed_val

    def to_dict(self) -> dict:
        """Convert to a flat dictionary for DataFrame construction."""
        record: dict = {
            "smiles_generated": self.smiles,
            "smiles_canonical": self.canonical,
            "smiles_seed": self.seed_smiles,
            "seed_idx": self.seed_idx,
            "is_valid": self.is_valid,
            "is_memorized": self.is_memorized,
            "tanimoto_to_seed": self.tanimoto_to_seed,
            "round": self.round_num,
            "generation_idx": self.generation_idx,
            "selected": self.selected,
            "parent_smiles": self.parent_smiles,
            "beam_rank": self.beam_rank,
        }

        # Per-reference memorization
        for ref_name, is_mem in self.memorized_by_ref.items():
            record[f"memorized_in_{ref_name}"] = is_mem

        # Properties and deltas
        for prop_name, value in self.properties.items():
            record[prop_name] = value
        for prop_name, value in self.seed_properties.items():
            record[f"{prop_name}_seed"] = value
        for prop_name in self.properties:
            delta = self.property_delta(prop_name)
            if delta is not None:
                record[f"{prop_name}_delta"] = delta

        # Per-property improvement and V,I,N flags
        for prop_name, improved in self.is_improved.items():
            record[f"is_improved_{prop_name}"] = improved
            record[f"is_vin_{prop_name}"] = self.is_vin(prop_name)

        return record
