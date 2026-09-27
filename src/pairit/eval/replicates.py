"""Collapse training-replicate runs into one table row, with a spread.

A configuration is often trained more than once at different **random seeds**, and the
published table reports one row per configuration as ``mean ± sd`` over those runs. Each
run is registered separately -- keeping the provenance of every file -- and rows that
declare the same ``replicate_group`` are combined here.

"Replicate" is a random seed. It is **not** a seed molecule: the registry's ``n_seeds``
counts seed molecules, and conflating the two is easy enough that both senses appear in
this codebase's history.

Aggregation is **rates first, then mean** -- each replicate's rate is computed
independently by ``evaluate`` and this module averages those rates. It does *not* pool the
underlying counts. Pooling weights a replicate by how many seed molecules it happened to
cover, so a run that lost seeds would silently count for less; averaging rates treats each
run as one observation of the configuration, which is what a ± over runs means. The two
agree only when every replicate covers the same number of seed molecules.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

#: Columns that identify a row rather than measure it, so they are carried through
#: rather than averaged.
_IDENTITY = ("method", "dataset")

#: Added to the aggregated frame so a downstream formatter can tell a real spread
#: from a single run, and so the record can show what was combined.
N_REPLICATES = "n_replicates"
REPLICATE_MEMBERS = "replicate_members"


def aggregate_replicates(
    average: pd.DataFrame,
    group_map: Mapping[Tuple[str, str], str],
    *,
    label_map: Mapping[str, str] | None = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Collapse replicate rows of ``average`` into one row per configuration.

    Parameters
    ----------
    average
        One row per ``(method, dataset)``, as produced by :func:`evaluate`.
    group_map
        ``(dataset, method) -> replicate group``. A method absent from the map, or
        mapped to its own name, is a group of one and passes through unchanged.
    label_map
        Optional ``group -> display method name`` for the collapsed row. Defaults to
        the group key itself.

    Returns
    -------
    (aggregated frame, per-group report dicts)

    Every numeric column becomes its mean across replicates, and gains a ``<col>_sd``
    companion (population sd, ddof=0 -- these are the runs performed, not a sample
    from a larger pool). Non-numeric columns take the first value, which is why
    ``method`` is rewritten from the group rather than inherited.
    """
    if average is None or average.empty:
        return average, []

    work = average.copy()
    work["_group"] = [
        group_map.get((str(d), str(m)), str(m))
        for m, d in zip(work["method"], work["dataset"])
    ]

    numeric = [
        c for c in work.columns
        if c not in _IDENTITY and c != "_group" and pd.api.types.is_numeric_dtype(work[c])
    ]
    other = [c for c in work.columns if c not in numeric and c not in ("_group",) + _IDENTITY]

    rows: List[dict] = []
    reports: List[dict] = []
    for (dataset, group), sub in work.groupby(["dataset", "_group"], sort=False):
        members = [str(m) for m in sub["method"]]
        rec: dict = {"method": (label_map or {}).get(group, group), "dataset": dataset}
        for c in numeric:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            rec[c] = float(vals.mean()) if len(vals) else float("nan")
            # ddof=0: the replicates are the population of runs actually done.
            rec[f"{c}_sd"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        for c in other:
            rec[c] = sub[c].iloc[0]
        rec[N_REPLICATES] = int(len(sub))
        rec[REPLICATE_MEMBERS] = ",".join(members)
        rows.append(rec)
        reports.append({
            "dataset": str(dataset),
            "group": str(group),
            "n_replicates": int(len(sub)),
            "members": members,
        })

    out = pd.DataFrame(rows)
    # Preserve the input's column order where possible; sd columns follow their parent.
    ordered: List[str] = []
    for c in average.columns:
        if c in out.columns:
            ordered.append(c)
        if f"{c}_sd" in out.columns:
            ordered.append(f"{c}_sd")
    ordered += [c for c in (N_REPLICATES, REPLICATE_MEMBERS) if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered], reports


def has_replicates(reports: Sequence[dict]) -> bool:
    """True if any group combined more than one run."""
    return any(r.get("n_replicates", 1) > 1 for r in reports)


def format_cell(value: float, sd: float | None, n: int, decimals: int = 2) -> str:
    """Render one table cell.

    A single run prints bare -- ``50.40``, not ``50.40 ± 0.00`` -- because a zero
    spread over one observation is not a measurement, and printing it invites the
    reader to compare it against a real spread.
    """
    if value is None or pd.isna(value):
        return "-"
    if n > 1 and sd is not None and not pd.isna(sd):
        return f"{value:.{decimals}f} ± {sd:.{decimals}f}"
    return f"{value:.{decimals}f}"
