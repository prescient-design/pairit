"""Best-of-N selection.

Selection is a *protocol* decision; evaluation is a *measurement*. They are kept
apart deliberately: :func:`evaluate.evaluate` has the contract "rows in = rows
scored", and ``n_total = len(df)`` is the denominator of `valid`, `v_i`, `v_n`
and `v_n_i`. If selection ran inside it, that one column name
would mean the input budget in one mode and the selected count in the other --
which is precisely how the same generations came to be reported with V,I,N from
5.5 to 63.7.

So the pipeline is::

    load -> [normalize] -> [select] -> evaluate

and the ordering is enforced, not documented. Selection is opt-in; without it
nothing here runs and output is unchanged.

The gate
--------
1. **V,I,N hard gate.** A candidate is eligible only if it is Valid (parses),
   Improved (the selection oracle says so) and Novel (canonical SMILES absent
   from every novelty reference). Ranking on score alone would let memorized or
   unimproved molecules into the top k.
2. **Per-seed canonical dedup.** 64 samples frequently repeat a molecule.
3. **Rank by the selection oracle's score, take the top k.**
4. **Fallback.** If nothing passes the gate, take the best valid candidate; if
   nothing is valid, emit the seed itself, so every seed yields a row.

Seed accounting
---------------
The denominator question this raises is the whole reason the module reports
statistics. Every (method, seed) pair is in exactly one of three states:

``selected``      at least one candidate cleared the gate
``gate_failed``   candidates existed, none cleared the gate
``absent``        the file has no rows for this seed at all

The last two produce the same number and must never be merged: `gate_failed` is
a model that tried and failed, `absent` is a file that is incomplete or was built
on different seeds. Both are counted; both are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
from rdkit import Chem

# Selection modes.
GATED_TOP_K = "gated-top-k"  # the V,I,N gate above -- the default and the one to use
TOP_K_UNGATED = "top-k-ungated"  # rank by score, no gate. Diagnostic: shows what the gate buys
FIRST_K = "first-k"  # first k as generated. Diagnostic baseline
RANDOM_K = "random-k"  # k at random. Diagnostic baseline
SELECTION_MODES = (GATED_TOP_K, TOP_K_UNGATED, FIRST_K, RANDOM_K)

# Denominator policies for the metrics computed after selection.
BUDGET = "budget"  # k * n_seeds -- constant across methods, so rows are comparable
SELECTED = "selected"  # the number of rows that survived -- varies per method
DENOMINATORS = (BUDGET, SELECTED)

# How each surviving row came to be selected. Recorded per row so a fallback can
# never be mistaken for a genuine selection.
REASON_GATED = "gated"
REASON_FALLBACK_VALID = "fallback_valid"
REASON_FALLBACK_SEED = "fallback_seed"


@dataclass
class SelectionConfig:
    """Parameters of a selection pass."""

    mode: str = GATED_TOP_K
    k: int = 5
    denominator: str = BUDGET
    rng_seed: int = 0
    #: Emit the seed itself when a seed has no valid candidate, so every seed
    #: yields a row. Turning this off makes such seeds `absent` instead.
    fallback_to_seed: bool = True

    def __post_init__(self) -> None:
        if self.mode not in SELECTION_MODES:
            raise ValueError(f"mode must be one of {SELECTION_MODES}, got {self.mode!r}")
        if self.denominator not in DENOMINATORS:
            raise ValueError(
                f"denominator must be one of {DENOMINATORS}, got {self.denominator!r}"
            )
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")


@dataclass
class SelectionReport:
    """What one selection pass did. Goes verbatim into the run record."""

    method: str
    dataset: str
    mode: str
    k: int
    denominator: str
    n_seeds: int = 0
    seeds_selected: int = 0
    seeds_gate_failed: int = 0
    seeds_absent: int = 0
    #: Rows emitted as a copy of their own seed, because nothing usable existed.
    #: Cuts across the three seed states, so it is counted separately.
    seeds_fallback_seed: int = 0
    seeds_short_of_k: int = 0
    candidates_in: int = 0
    candidates_out: int = 0
    dedup_removed: int = 0
    n_unscorable: int = 0
    budget: int = 0
    absent_seeds: List[str] = field(default_factory=list)

    @property
    def gate_yield(self) -> Optional[float]:
        """Selected rows as a fraction of the budget.

        The number that explains the gap between the `budget` and `selected`
        denominators. A method at 100% is unaffected by the choice.
        """
        return 100 * self.candidates_out / self.budget if self.budget else None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "method": self.method,
            "dataset": self.dataset,
            "mode": self.mode,
            "k": self.k,
            "denominator": self.denominator,
            "n_seeds": self.n_seeds,
            "seeds_selected": self.seeds_selected,
            "seeds_gate_failed": self.seeds_gate_failed,
            "seeds_absent": self.seeds_absent,
            "seeds_fallback_seed": self.seeds_fallback_seed,
            "seeds_short_of_k": self.seeds_short_of_k,
            "candidates_in": self.candidates_in,
            "candidates_out": self.candidates_out,
            "dedup_removed": self.dedup_removed,
            "n_unscorable": self.n_unscorable,
            "budget": self.budget,
            "gate_yield": self.gate_yield,
        }
        # Keep the record small but not lossy: list absent seeds only when few.
        if self.absent_seeds:
            out["absent_seeds"] = self.absent_seeds[:50]
            out["absent_seeds_truncated"] = len(self.absent_seeds) > 50
        return out


_CANON_CACHE: Dict[str, Optional[str]] = {}


def canonical(smiles: Any) -> Optional[str]:
    """Canonical SMILES, or None if it does not parse. Memoized."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    text = str(smiles)
    if not text or text in ("None", "nan"):
        return None
    if text in _CANON_CACHE:
        return _CANON_CACHE[text]
    mol = Chem.MolFromSmiles(text)
    out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None
    _CANON_CACHE[text] = out
    return out


def resolve_seed_set(
    frames: Sequence[pd.DataFrame],
    policy: str = "intersection",
    reference_seeds: Optional[Iterable[str]] = None,
) -> List[str]:
    """The seed set every method in a table is scored against.

    This must not come from a method's own output. If it did, the denominator
    would be a function of what the method produced -- a method emitting nothing
    for 100 of 135 seeds would be scored on 35 and could top the table by failing
    more.

    Parameters
    ----------
    frames
        One frame per method.
    policy
        ``intersection`` (default) uses seeds present in *every* method, so all
        rows share a denominator and are comparable by construction.
        ``reference`` uses `reference_seeds`, the dataset's canonical test seeds
        -- the only policy under which `absent` is meaningful rather than
        definitional. ``union`` uses every seed any method produced; diagnostic
        only, since it heavily penalizes narrow methods.
    """
    if policy == "reference":
        if reference_seeds is None:
            raise ValueError("seed policy 'reference' requires reference_seeds")
        return sorted({str(s) for s in reference_seeds})

    per_frame = [set(f["seed_smiles"].dropna().astype(str)) for f in frames if len(f)]
    if not per_frame:
        return []
    if policy == "union":
        out: Set[str] = set()
        for s in per_frame:
            out |= s
        return sorted(out)
    if policy == "intersection":
        out = set(per_frame[0])
        for s in per_frame[1:]:
            out &= s
        return sorted(out)
    raise ValueError(f"Unknown seed policy {policy!r}; expected intersection/reference/union")


def select_generations(
    df: pd.DataFrame,
    config: SelectionConfig,
    select_oracle: Optional[Any] = None,
    novelty_reference: Optional[Set[str]] = None,
    seed_set: Optional[Sequence[str]] = None,
    method: str = "",
    dataset: str = "",
) -> "tuple[pd.DataFrame, SelectionReport]":
    """Select up to `k` generations per seed.

    Parameters
    ----------
    df
        Canonical generation frame for ONE (method, dataset) block.
    config
        Mode, k, and denominator policy.
    select_oracle
        Scores and gates candidates. Required for the gated and score-ranked
        modes; ignored by `first-k` and `random-k`.
    novelty_reference
        Canonical SMILES considered memorized. **Must be the same object the
        metric uses.** If the gate and the metric disagree about novelty,
        selection admits molecules the metric then calls memorized and
        `v_n_i` drops for no visible reason.
    seed_set
        The seeds to score against, from :func:`resolve_seed_set`. Seeds in this
        set with no rows in `df` are counted as `absent`.

    Returns
    -------
    (selected, report)
        `selected` carries two extra columns: `selection_reason` and
        `selection_score`.
    """
    method = method or (str(df["method"].iloc[0]) if len(df) and "method" in df else "")
    dataset = dataset or (str(df["dataset"].iloc[0]) if len(df) and "dataset" in df else "")

    if seed_set is None:
        seed_set = sorted(df["seed_smiles"].dropna().astype(str).unique()) if len(df) else []
    seed_set = [str(s) for s in seed_set]

    report = SelectionReport(
        method=method,
        dataset=dataset,
        mode=config.mode,
        k=config.k,
        denominator=config.denominator,
        n_seeds=len(seed_set),
        candidates_in=len(df),
        budget=config.k * len(seed_set),
    )
    if not len(df) or not seed_set:
        report.seeds_absent = len(seed_set)
        report.absent_seeds = list(seed_set)
        return df.iloc[0:0].copy(), report

    if config.k < 1:
        raise ValueError("k must be >= 1")

    work = df.copy()
    work["_seed_key"] = work["seed_smiles"].astype(str)
    work = work[work["_seed_key"].isin(set(seed_set))]
    work["_canon"] = [canonical(s) for s in work["generated_smiles"]]
    # Seed-copy check at gate time (not just post-eval): methods that produce
    # many seed-copies would otherwise consume top-k slots that get filtered
    # out later, leaving `candidates_out=N` but `n_valid=0` after eval.
    # Canonical-SMILES equality is used rather than a fresh tanimoto call —
    # for identical molecules the canonical SMILES are byte-identical, so
    # equality is exactly `tanimoto >= 1.0 - 1e-6`.
    work["_seed_canon"] = [canonical(s) for s in work["seed_smiles"]]
    work["_is_seed_copy"] = (
        work["_canon"].notna()
        & work["_seed_canon"].notna()
        & (work["_canon"] == work["_seed_canon"])
    )

    needs_score = config.mode in (GATED_TOP_K, TOP_K_UNGATED)
    if needs_score:
        if select_oracle is None:
            raise ValueError(
                f"Selection mode {config.mode!r} ranks by oracle score, but no selection "
                f"oracle was given. Pass --select-oracle, or use mode 'first-k'/'random-k'."
            )
        scores = select_oracle.predict(work["generated_smiles"].tolist())
        work["_score"] = [np.nan if s is None else float(s) for s in scores]
        labels = [select_oracle.label(None if pd.isna(s) else s) for s in work["_score"]]
        work["_good"] = [bool(v) for v in labels]
        report.n_unscorable = int(work["_score"].isna().sum())
    else:
        work["_score"] = np.nan
        work["_good"] = True

    # Valid means it parses. Under a scoring mode it must also be scorable --
    # an unscorable molecule cannot be shown to be an improvement.
    valid = work[work["_canon"].notna()]
    if needs_score:
        valid = valid[valid["_score"].notna()]

    ref = novelty_reference or set()
    novel = ~valid["_canon"].isin(ref) if len(valid) else pd.Series(dtype=bool)
    # Seed-copy filter fires at the gate, alongside V+I+N. Post-eval
    # seed-dup filter would remove these anyway, so numeric impact is zero;
    # gating early frees the top-k slot for a real candidate.
    not_seed_copy = ~valid["_is_seed_copy"] if len(valid) else pd.Series(dtype=bool)
    if config.mode == GATED_TOP_K:
        passing = valid[valid["_good"] & novel & not_seed_copy]
    else:
        passing = valid[not_seed_copy] if len(valid) else valid

    rng = np.random.default_rng(config.rng_seed)
    chunks: List[pd.DataFrame] = []
    dedup_removed = 0
    # Presence is judged on the RAW rows, not the valid ones. A seed whose only
    # generation is unparseable is `gate_failed` -- the model tried and produced
    # garbage -- not `absent`, which means the file has no row for it at all.
    # Collapsing those two is exactly what this accounting exists to prevent.
    present = set(work["_seed_key"])
    valid_by_seed = {s: g for s, g in valid.groupby("_seed_key", sort=False)}
    pass_by_seed = {s: g for s, g in passing.groupby("_seed_key", sort=False)}

    for seed in seed_set:
        pool = pass_by_seed.get(seed)
        grp = valid_by_seed.get(seed)
        is_present = seed in present

        if not is_present:
            report.seeds_absent += 1
            report.absent_seeds.append(seed)
        elif pool is not None and len(pool):
            report.seeds_selected += 1
        else:
            report.seeds_gate_failed += 1

        if pool is not None and len(pool):
            reason = REASON_GATED
            take, removed = _pick(pool, config, rng, config.k)
        elif grp is not None and len(grp):
            # Candidates parsed but none cleared the gate: take the best valid one.
            reason = REASON_FALLBACK_VALID
            take, removed = _pick(grp, config, rng, 1)
        else:
            # Nothing usable -- either no rows at all, or every row invalid.
            # Emit the seed itself so the seed still yields a row. That row scores
            # as valid but tanimoto 1.0, so it is a duplicate and not novel: it
            # lowers `valid` and `novel` rather than flattering them.
            if not config.fallback_to_seed:
                continue
            take = _seed_row(df, seed)
            removed = 0
            reason = REASON_FALLBACK_SEED
            report.seeds_fallback_seed += 1

        dedup_removed += removed
        take = take.copy()
        take["selection_reason"] = reason
        chunks.append(take)
        if reason == REASON_GATED and len(take) < config.k:
            report.seeds_short_of_k += 1

    if not chunks:
        return df.iloc[0:0].copy(), report

    out = pd.concat(chunks, ignore_index=True)
    out["selection_score"] = out.get("_score")
    out = out.drop(columns=[c for c in ("_canon", "_good", "_score", "_seed_key") if c in out])
    report.candidates_out = len(out)
    report.dedup_removed = dedup_removed
    return out, report


def _pick(
    group: pd.DataFrame, config: SelectionConfig, rng: np.random.Generator, k: int
) -> "tuple[pd.DataFrame, int]":
    """Take up to k from one seed's pool, deduplicating on canonical SMILES."""
    before = len(group)
    if config.mode == FIRST_K:
        picked = group.drop_duplicates("_canon").head(k)
    elif config.mode == RANDOM_K:
        deduped = group.drop_duplicates("_canon")
        n = min(k, len(deduped))
        idx = rng.choice(len(deduped), size=n, replace=False)
        picked = deduped.iloc[sorted(idx)]
    else:
        picked = (
            group.sort_values("_score", ascending=False, kind="mergesort")
            .drop_duplicates("_canon")
            .head(k)
        )
    deduped_count = len(group.drop_duplicates("_canon"))
    return picked, before - deduped_count


def _seed_row(df: pd.DataFrame, seed: str) -> pd.DataFrame:
    """A one-row frame echoing the seed back as its own generation."""
    template = df.iloc[0:1].copy()
    template["seed_smiles"] = seed
    template["generated_smiles"] = seed
    if "gen_idx" in template:
        template["gen_idx"] = 0
    if "seed_label" in template:
        match = df[df["seed_smiles"].astype(str) == seed]
        template["seed_label"] = match["seed_label"].iloc[0] if len(match) else 0
    template["selection_score"] = np.nan
    return template


def denominator_for(report: SelectionReport) -> int:
    """Rows to divide by, given the report's policy.

    `budget` is k * n_seeds and is constant across methods, so the rows of a
    comparison table share a denominator. `selected` is what survived and varies
    per method; it reproduces older tables but is not comparable across rows.
    """
    return report.budget if report.denominator == BUDGET else report.candidates_out


# ---------------------------------------------------------------------------
# Seed policy as its own stage
#
# Resolving the seed set used to happen inside select_generations, which meant a
# table with no selection step got no seed policy at all: every method kept its
# own seeds and the denominators quietly differed. On the MoleculeNet table that
# showed up as one method scored over 1150 generations and the rest over 1155.
#
# It is now a stage in its own right, so `load -> [normalize] -> seeds ->
# [select] -> evaluate` holds whether or not selection runs.
# ---------------------------------------------------------------------------


@dataclass
class SeedPolicyReport:
    """What restricting to a common seed set did, per method."""

    policy: str
    n_seeds_resolved: int = 0
    per_method: List[Dict[str, Any]] = field(default_factory=list)
    dropped_seeds: List[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return sum(m["rows_dropped"] for m in self.per_method)

    @property
    def denominators_equal(self) -> bool:
        """Whether every method ended up with the same seed count.

        This is the property the policy exists to establish; if it is False the
        table's rows are still not comparable and the run says so.

        On its own this is NOT sufficient: an intersection that collapses to zero
        seeds also makes every count equal. Always read it with
        :attr:`retention`.
        """
        counts = {m["seeds_after"] for m in self.per_method}
        return len(counts) <= 1

    @property
    def retention(self) -> Optional[float]:
        """Smallest fraction of any method's seeds that survived, 0-1.

        The number that distinguishes "dropped a couple of stragglers" from
        "these methods were built on different splits and almost nothing is
        shared". A near-empty intersection is otherwise indistinguishable from a
        clean one: both report equal denominators and a row count.
        """
        fracs = [
            m["seeds_after"] / m["seeds_before"]
            for m in self.per_method
            if m["seeds_before"]
        ]
        return min(fracs) if fracs else None

    @property
    def collapsed(self) -> bool:
        """True when the shared seed set is too small to trust.

        Below half of some method's seeds, the table is no longer measuring what
        it appears to: every row is scored on a small corner of the test set.
        """
        r = self.retention
        return r is not None and r < 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "n_seeds_resolved": self.n_seeds_resolved,
            "rows_dropped": self.rows_dropped,
            "denominators_equal": self.denominators_equal,
            "retention": self.retention,
            "collapsed": self.collapsed,
            "per_method": self.per_method,
            "dropped_seeds": self.dropped_seeds[:50],
            "dropped_seeds_truncated": len(self.dropped_seeds) > 50,
        }


@dataclass
class SeedFilterReport:
    """What a `seed_filter` removed, per dataset. Never summarised away."""

    dataset: str
    filter_name: str
    n_before: int
    n_after: int
    excluded_smiles: List[str]

    @property
    def n_excluded(self) -> int:
        return self.n_before - self.n_after

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "filter": self.filter_name,
            "starting_molecules_before": self.n_before,
            "starting_molecules_after": self.n_after,
            "excluded": self.n_excluded,
            # The actual structures, not just a count. An exclusion nobody can inspect is
            # how 22 starting molecules went missing unnoticed in the first place.
            "excluded_smiles": list(self.excluded_smiles),
        }


def _is_connected(smiles: str) -> bool:
    """True when the SMILES describes a single covalently connected component.

    Tested on the string rather than by parsing: a "." at the top level of a SMILES is
    the component separator, and RDKit round-trips it faithfully, so parsing adds cost
    and a second failure mode without adding information.
    """
    return "." not in str(smiles)


def apply_seed_filter(
    df: pd.DataFrame,
    filter_name: str,
    reference_seeds: Optional[Union[Iterable[str], Mapping[str, Iterable[str]]]] = None,
    group_cols: Sequence[str] = ("dataset",),
) -> "tuple[pd.DataFrame, List[SeedFilterReport], Optional[dict]]":
    """Drop starting molecules the table does not score, symmetrically across methods.

    Runs BEFORE selection, so no method spends its candidate budget on a molecule that
    is about to be excluded, and before the seed policy, so an excluded molecule cannot
    shrink an intersection.

    `connected` drops multi-fragment starting molecules. The justification is specific:
    matched molecular pair construction cannot fragment a disconnected structure, so no
    pair-trained method has ever seen one -- across ZINC-250K pretraining and all three
    MoleculeNet fine-tuning sets, zero training pairs involve one. Scoring them measures
    nothing about the method. It is deliberately NOT extended to molecules our tokenizer
    cannot represent: that is a property of one model's vocabulary, and excluding cases
    where a specific model fails while its baselines do not would flatter that model.

    Returns the filtered frame, one report per dataset, and the reference seeds with the
    same filter applied (so `seed_policy: reference` does not reinstate what was removed).
    """
    if not filter_name or not len(df):
        return df, [], reference_seeds

    if filter_name != "connected":
        raise ValueError(f"unknown seed_filter {filter_name!r}")

    reports: List[SeedFilterReport] = []
    keep_mask = pd.Series(True, index=df.index)
    for keys, block in df.groupby(list(group_cols), dropna=False):
        dataset = str(keys[0] if isinstance(keys, tuple) else keys)
        seeds = block["seed_smiles"].astype(str)
        bad = sorted({s for s in seeds.unique() if not _is_connected(s)})
        if bad:
            keep_mask.loc[block.index] &= ~seeds.isin(bad).values
        reports.append(
            SeedFilterReport(
                dataset=dataset,
                filter_name=filter_name,
                n_before=int(seeds.nunique()),
                n_after=int(seeds.nunique() - len(bad)),
                excluded_smiles=bad,
            )
        )

    filtered_refs = reference_seeds
    if reference_seeds is not None:
        if isinstance(reference_seeds, Mapping):
            filtered_refs = {
                k: [s for s in v if _is_connected(s)] for k, v in reference_seeds.items()
            }
        else:
            filtered_refs = [s for s in reference_seeds if _is_connected(s)]

    return df[keep_mask], reports, filtered_refs


def apply_seed_policy(
    df: pd.DataFrame,
    policy: str = "intersection",
    reference_seeds: Optional[Union[Iterable[str], Mapping[str, Iterable[str]]]] = None,
    group_cols: Sequence[str] = ("dataset",),
) -> "tuple[pd.DataFrame, List[SeedPolicyReport]]":
    """Restrict every method to a common seed set, one report per dataset.

    `union` is a no-op that only reports -- it is the historical behaviour, where
    each method keeps whatever seeds it happens to have.

    `reference_seeds` may be a flat iterable, applied to every dataset, or a
    ``{dataset: seeds}`` mapping. The mapping form exists because a canonical seed
    set is a property of a dataset, so a table spanning two datasets under
    `reference` needs two of them -- passing one flat list would silently restrict
    the second dataset to the first one's seeds and report it as a clean run.

    Returns the restricted frame and the reports.
    """
    if not len(df) or policy == "union":
        reports = []
        for keys, block in df.groupby(list(group_cols), sort=False) if len(df) else []:
            reports.append(_seed_report(block, policy, sorted(set(block["seed_smiles"].astype(str)))))
        return df, reports

    kept, reports = [], []
    for keys, block in df.groupby(list(group_cols), sort=False):
        frames = [g for _, g in block.groupby("method", sort=False)]
        refs = reference_seeds
        if isinstance(reference_seeds, Mapping):
            # groupby yields a scalar for one column and a tuple for several.
            key = keys[0] if isinstance(keys, tuple) else keys
            refs = reference_seeds.get(str(key))
            if policy == "reference" and refs is None:
                raise ValueError(
                    f"seed policy 'reference' has no reference seeds for dataset "
                    f"{str(key)!r}. Register a row with role='seed_source' for it, or "
                    f"use seed_policy 'intersection'."
                )
        seeds = resolve_seed_set(frames, policy=policy, reference_seeds=refs)
        seed_set = set(seeds)
        restricted = block[block["seed_smiles"].astype(str).isin(seed_set)]
        reports.append(_seed_report(block, policy, seeds, restricted))
        kept.append(restricted)

    out = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0]
    return out, reports


def _seed_report(
    block: pd.DataFrame,
    policy: str,
    seeds: Sequence[str],
    restricted: Optional[pd.DataFrame] = None,
) -> SeedPolicyReport:
    restricted = block if restricted is None else restricted
    report = SeedPolicyReport(policy=policy, n_seeds_resolved=len(seeds))
    all_seeds = set()
    for method, grp in block.groupby("method", sort=False):
        before = set(grp["seed_smiles"].astype(str))
        all_seeds |= before
        after = set(restricted[restricted["method"] == method]["seed_smiles"].astype(str))
        report.per_method.append(
            {
                "dataset": str(grp["dataset"].iloc[0]) if "dataset" in grp else "",
                "method": str(method),
                "seeds_before": len(before),
                "seeds_after": len(after),
                "seeds_dropped": len(before - after),
                "rows_before": len(grp),
                "rows_dropped": len(grp) - len(restricted[restricted["method"] == method]),
            }
        )
    report.dropped_seeds = sorted(all_seeds - set(seeds))
    return report
