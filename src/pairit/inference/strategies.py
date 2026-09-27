"""Selection strategies for inference-time compute.

Algorithm
---------
After V,I,N filtering produces a pool of eligible candidates, a selection
strategy picks the final molecules. Two strategies are provided:

**TopKPerSeed** (used by best-of-N):
    Groups candidates by their seed molecule, sorts each group by the
    ranking property, and takes the top-k from each seed. This guarantees
    every seed with viable candidates contributes to the output.
    Canonical SMILES duplicates are removed globally before grouping.

**GlobalTopK** (used by beam search):
    Pools all candidates regardless of seed, sorts globally by the ranking
    property, and takes the top-k overall. This means only the best
    molecules across all seeds survive — some seeds may contribute multiple
    molecules while others contribute none.
    Each selected candidate is assigned a ``beam_rank`` (0 = best).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from pairit.inference.scorers import ScoredCandidate

logger = logging.getLogger(__name__)


class SelectionStrategy(ABC):
    """Base class for molecule selection strategies."""

    @abstractmethod
    def select(
        self,
        candidates: list[ScoredCandidate],
        k: int,
        property_name: str,
        direction: str,
        deduplicate: bool = True,
    ) -> list[ScoredCandidate]:
        """Select k candidates from the pool.

        Parameters
        ----------
        candidates : list[ScoredCandidate]
            Pool of candidates (already filtered by V,I,N or other gates).
        k : int
            Number of candidates to select.
        property_name : str
            Property to rank by.
        direction : str
            ``"increasing"`` (higher is better) or ``"decreasing"`` (lower is better).
        deduplicate : bool
            Remove canonical SMILES duplicates before selection.

        Returns
        -------
        list[ScoredCandidate]
            Selected candidates with ``selected=True``.
        """
        ...


class TopKPerSeed(SelectionStrategy):
    """Select top-k candidates per seed by property score.

    This matches the existing iterative_generate.py behavior: each seed gets
    up to k selected molecules, ranked by the optimization property.
    """

    def select(
        self,
        candidates: list[ScoredCandidate],
        k: int,
        property_name: str,
        direction: str,
        deduplicate: bool = True,
    ) -> list[ScoredCandidate]:
        reverse = direction == "increasing"

        # Group by seed first — deduplication is per-seed, not global
        by_seed: dict[int, list[ScoredCandidate]] = defaultdict(list)
        for c in candidates:
            if c.properties.get(property_name) is None:
                continue
            by_seed[c.seed_idx].append(c)

        selected: list[ScoredCandidate] = []
        for seed_idx in sorted(by_seed):
            seed_cands = by_seed[seed_idx]

            # Deduplicate within this seed only
            if deduplicate:
                seen: set[str] = set()
                deduped: list[ScoredCandidate] = []
                for c in seed_cands:
                    if c.canonical and c.canonical in seen:
                        continue
                    if c.canonical:
                        seen.add(c.canonical)
                    deduped.append(c)
                seed_cands = deduped

            seed_cands.sort(
                key=lambda c: c.properties.get(property_name, float("-inf")),
                reverse=reverse,
            )
            for c in seed_cands[:k]:
                c.selected = True
                selected.append(c)

        logger.info(
            f"TopKPerSeed: selected {len(selected)} molecules "
            f"from {len(by_seed)} seeds (k={k})"
        )
        return selected


class GlobalTopK(SelectionStrategy):
    """Select top-k candidates globally across all seeds.

    Used by beam search: the best molecules overall form the next beam,
    regardless of which seed produced them.
    """

    def select(
        self,
        candidates: list[ScoredCandidate],
        k: int,
        property_name: str,
        direction: str,
        deduplicate: bool = True,
    ) -> list[ScoredCandidate]:
        reverse = direction == "increasing"

        pool: list[ScoredCandidate] = []
        seen: set[str] = set()

        for c in candidates:
            if c.properties.get(property_name) is None:
                continue
            if deduplicate and c.canonical:
                if c.canonical in seen:
                    continue
                seen.add(c.canonical)
            pool.append(c)

        pool.sort(
            key=lambda c: c.properties.get(property_name, float("-inf")),
            reverse=reverse,
        )

        selected = pool[:k]
        for i, c in enumerate(selected):
            c.selected = True
            c.beam_rank = i

        logger.info(
            f"GlobalTopK: selected {len(selected)} from {len(pool)} candidates (k={k})"
        )
        return selected
