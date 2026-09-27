"""V,I,N hard-gate filter for inference-time compute.

Algorithm
---------
The V,I,N filter enforces three simultaneous conditions on each candidate:

- **V (Valid)**: the generated SMILES parses to a valid RDKit molecule.
- **I (Improved)**: the ranking property improved over the seed molecule
  (higher for ``increasing`` properties like logP, lower for ``decreasing``
  properties like SA).
- **N (Novel)**: the canonical SMILES does not appear in any reference
  training set (not memorized).

A candidate must satisfy ALL three conditions to pass the hard gate.

Fallback behavior (per-seed mode):
- ``"valid_only"``: if no V,I,N candidates exist for a seed, include all
  valid candidates from that seed (they may be memorized or unimproved).
- ``"none"``: if no V,I,N candidates exist, that seed contributes nothing.

In global mode (used by beam search), the same logic applies but across
the entire candidate pool rather than per-seed.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pairit.inference.scorers import ScoredCandidate

logger = logging.getLogger(__name__)


class VINFilter:
    """Hard-gate filter: only passes candidates that are Valid, Improved, and Novel.

    Parameters
    ----------
    ranking_property : str
        Property name used for the improvement check in the V,I,N gate.
    fallback : str
        What to do when no candidates pass V,I,N for a seed:
        - ``"valid_only"``: fall back to valid (but possibly memorized/unimproved)
          candidates, ranked by property.
        - ``"none"``: return no candidates for that seed.
    """

    def __init__(
        self,
        ranking_property: str,
        fallback: str = "valid_only",
    ):
        if fallback not in ("valid_only", "none"):
            raise ValueError(f"Invalid fallback mode: {fallback!r}. Must be 'valid_only' or 'none'.")
        self.ranking_property = ranking_property
        self.fallback = fallback

    def filter(
        self,
        candidates: list[ScoredCandidate],
        per_seed: bool = True,
    ) -> list[ScoredCandidate]:
        """Apply V,I,N hard gate to candidates.

        Parameters
        ----------
        candidates : list[ScoredCandidate]
            Pool of candidates to filter.
        per_seed : bool
            If True, apply fallback logic per-seed (so each seed gets a chance
            to contribute). If False, apply globally.

        Returns
        -------
        list[ScoredCandidate]
            Candidates that pass V,I,N, plus any fallback candidates.
        """
        if not per_seed:
            return self._filter_global(candidates)
        return self._filter_per_seed(candidates)

    def _filter_global(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        vin = [c for c in candidates if c.is_vin(self.ranking_property)]
        if vin:
            return vin

        logger.warning(
            f"No V,I,N candidates in pool of {len(candidates)}. "
            f"Fallback={self.fallback}."
        )
        if self.fallback == "valid_only":
            return [c for c in candidates if c.is_valid]
        return []

    def _filter_per_seed(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        by_seed: dict[int, list[ScoredCandidate]] = defaultdict(list)
        for c in candidates:
            by_seed[c.seed_idx].append(c)

        result: list[ScoredCandidate] = []
        n_vin_seeds = 0
        n_fallback_seeds = 0
        n_empty_seeds = 0

        for seed_idx in sorted(by_seed):
            seed_candidates = by_seed[seed_idx]
            vin = [c for c in seed_candidates if c.is_vin(self.ranking_property)]

            if vin:
                result.extend(vin)
                n_vin_seeds += 1
            elif self.fallback == "valid_only":
                valid = [c for c in seed_candidates if c.is_valid]
                result.extend(valid)
                if valid:
                    n_fallback_seeds += 1
                else:
                    n_empty_seeds += 1
            else:
                n_empty_seeds += 1

        if n_fallback_seeds > 0 or n_empty_seeds > 0:
            logger.info(
                f"V,I,N filter: {n_vin_seeds} seeds with V,I,N candidates, "
                f"{n_fallback_seeds} seeds fell back to valid-only, "
                f"{n_empty_seeds} seeds produced nothing."
            )

        return result
