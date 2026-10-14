"""Evaluate every raw attempt against its original matched-pair request.

Descriptor calculation, canonicalization and activity prediction use the existing
evaluation utilities. Only the per-request eligibility and reduction are new.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
import pandas as pd

from . import property_metrics
from .evaluate import compute_tanimoto
from .oracles import Oracle
from .select import canonical

NAMES = "prompt_target_direction_property_names"
SIGNS = "prompt_target_raw_direction_signs"
DELTAS = "prompt_target_raw_deltas"
SEED_VALUES = "prompt_target_raw_seed_values"
TARGET_VALUES = "prompt_target_raw_target_values"
TRUTH_COLUMNS = [NAMES, SIGNS, DELTAS, SEED_VALUES, TARGET_VALUES]
IDENTITY_COLUMNS = ["pair_id", "seed_smiles", "target_smiles"]
DESCRIPTORS = (
    "logp", "qed", "sa", "mw", "tpsa", "hbd", "hba", "rotatable_bonds", "frac_sp3_c",
)


@dataclass
class PromptedEvaluationResult:
    per_generation: pd.DataFrame
    summary: dict[str, Any]


def _list(value: Any) -> list:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("Request metadata must contain lists or JSON arrays")
    return list(value)


def _validate_inputs(
    generations: pd.DataFrame, requests: pd.DataFrame, attempts_per_request: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if isinstance(attempts_per_request, bool) or not isinstance(attempts_per_request, int) or attempts_per_request < 1:
        raise ValueError("attempts_per_request must be a positive integer")
    frames = []
    for name, original, extra in (
        ("requests", requests, []),
        ("generations", generations, ["gen_idx", "generated_smiles"]),
    ):
        required = IDENTITY_COLUMNS + TRUTH_COLUMNS + extra
        missing = sorted(set(required) - set(original.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        frame = original[required].copy()
        if frame[IDENTITY_COLUMNS].isna().any().any():
            raise ValueError(f"{name} has missing request identity")
        frame["pair_id"] = frame["pair_id"].astype(str)
        if (frame["pair_id"].str.len() == 0).any():
            raise ValueError(f"{name} has empty pair_id")
        for column in TRUTH_COLUMNS:
            frame[column] = frame[column].map(_list)
        frames.append(frame)
    requests, generations = frames
    if requests.empty or requests["pair_id"].duplicated().any():
        raise ValueError("Expected request roster must be nonempty with unique pair_id")
    names = requests.iloc[0][NAMES]
    if not names or any(not isinstance(p, str) for p in names) or len(set(names)) != len(names):
        raise ValueError("Property names must be nonempty, unique strings")
    for row in requests.to_dict("records"):
        if row[NAMES] != names or any(len(row[c]) != len(names) for c in TRUTH_COLUMNS):
            raise ValueError(f"Inconsistent property arrays for pair {row['pair_id']}")
        values = np.asarray([row[c] for c in (SIGNS, DELTAS, SEED_VALUES, TARGET_VALUES)], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite original request truth for pair {row['pair_id']}")
        signs, deltas, seeds, targets = values
        if not np.array_equal(signs, np.sign(deltas)) or not np.allclose(deltas, targets - seeds, atol=1e-10, rtol=1e-8):
            raise ValueError(f"Inconsistent raw signs/deltas/values for pair {row['pair_id']}")
        if canonical(row["seed_smiles"]) is None or canonical(row["target_smiles"]) is None:
            raise ValueError(f"Invalid original seed/target SMILES for pair {row['pair_id']}")
    expected = set(requests["pair_id"])
    observed = set(generations["pair_id"])
    if expected != observed:
        raise ValueError(f"Request coverage mismatch: {len(expected - observed)} missing, {len(observed - expected)} unexpected pairs")
    indices = pd.to_numeric(generations["gen_idx"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
        raise ValueError("gen_idx must contain finite integers")
    generations["gen_idx"] = indices.astype(int)
    if generations.duplicated(["pair_id", "gen_idx"]).any():
        raise ValueError("Duplicate (pair_id, gen_idx) attempts")
    bad = sum(set(group["gen_idx"]) != set(range(attempts_per_request)) for _, group in generations.groupby("pair_id"))
    if bad:
        raise ValueError(f"Incomplete or unexpected attempt indices for {bad} requests; expected 0..{attempts_per_request - 1}")
    roster = requests.set_index("pair_id")
    for row in generations.to_dict("records"):
        original = roster.loc[row["pair_id"]]
        if any(row[c] != original[c] for c in IDENTITY_COLUMNS[1:] + TRUTH_COLUMNS):
            raise ValueError(f"Generation changed original request identity/truth for pair {row['pair_id']}")
    return generations, requests, names


def _rate(hits: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    numerator = int((hits & eligible).sum())
    denominator = int(eligible.sum())
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}


def evaluate_prompted_generations(
    generations: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    training_smiles: Collection[str],
    attempts_per_request: int,
    activity_property: str | None = None,
    activity_oracle: Oracle | None = None,
) -> PromptedEvaluationResult:
    """Score a complete raw export against an independent expected request roster.

    Requests and generations carry the same immutable identity and raw truth.
    An activity property selects measured inactive-seed / active-target pairs;
    supply its name explicitly (the original exports call this ``target``).
    The caller must supply a training-only scoring oracle and novelty reference.
    Random-token controls keep the original request truth in these columns.
    All rates are fractions, not percentages; repeated molecules remain attempts.
    """
    rows, requests, names = _validate_inputs(generations, requests, attempts_per_request)
    if activity_property is not None and activity_property not in names:
        raise ValueError(f"Activity property {activity_property!r} is absent from request truth")
    if activity_property in DESCRIPTORS:
        raise ValueError("A descriptor cannot be used as the activity-label property")
    if (activity_property is None) != (activity_oracle is None):
        raise ValueError("Supply both activity_property and activity_oracle, or neither")
    if activity_oracle is not None and (
        activity_oracle.direction != "increasing" or activity_oracle.threshold != 0.5 or activity_oracle.comparison != ">"
    ):
        raise ValueError("Prompted activity evaluation requires an increasing oracle with strict score > 0.5")
    descriptors = [p for p in names if p != activity_property]
    if any(p not in DESCRIPTORS for p in descriptors):
        raise ValueError("Non-descriptor request properties require an explicit activity_property and oracle")
    specs = property_metrics.get_property_specs(descriptors)
    if "sa" in descriptors and property_metrics._sascorer_mod is None:
        raise RuntimeError("SA scoring is unavailable; install an RDKit build including Contrib.SA_Score")
    reference = [canonical(s) for s in training_smiles]
    if not reference or any(s is None for s in reference):
        raise ValueError(f"Novelty reference must be nonempty and valid; found {sum(s is None for s in reference)} invalid entries")
    train_set = set(reference)
    n_requests = len(requests)
    if activity_property is not None:
        j = names.index(activity_property)
        labels = np.asarray([[row[SEED_VALUES][j], row[TARGET_VALUES][j]] for row in requests.to_dict("records")])
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Original activity seed/target values must be measured binary labels")
        selected = (labels[:, 0] == 0) & (labels[:, 1] == 1)
        requests = requests.loc[selected]
        rows = rows.loc[rows["pair_id"].isin(requests["pair_id"])]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No eligible requests remain from {n_requests} original requests")
    canon = rows["generated_smiles"].map(canonical)
    valid = canon.notna().to_numpy()
    novel = (canon.notna() & ~canon.isin(train_set)).to_numpy()
    rows["is_valid"], rows["is_novel"] = valid, novel
    # Cache repeated molecules while retaining every occurrence in denominators.
    seed_canon = rows["seed_smiles"].map(canonical)
    unique_smiles = set(canon[valid]) | set(seed_canon)
    values = {s: property_metrics.compute_properties(s, specs) for s in unique_smiles}
    similarities = {
        (seed, gen): compute_tanimoto(seed, gen)
        for seed, gen in set(zip(rows.loc[valid, "seed_smiles"], canon[valid]))
    }
    rows["tanimoto"] = [similarities.get((seed, gen)) for seed, gen in zip(rows["seed_smiles"], canon)]
    activity_scores: dict[str, float | None] = {}
    activity_labels: dict[str, int | None] = {}
    if activity_oracle is not None:
        unique = list(canon[valid].unique())
        scores = activity_oracle.predict_labels(unique)
        if len(scores) != len(unique):
            raise ValueError("Activity oracle returned the wrong number of predictions")
        activity_scores = dict(zip(unique, (p for p, _ in scores)))
        activity_labels = dict(zip(unique, (label for _, label in scores)))
    all_attempts = np.ones(len(rows), dtype=bool)
    summary: dict[str, Any] = {
        "coverage": {
            "expected_requests": n_requests, "selected_requests": len(requests),
            "excluded_requests": n_requests - len(requests), "attempts_per_request": attempts_per_request,
            "expected_attempts": n_requests * attempts_per_request, "selected_attempts": len(rows),
        },
        "novelty_reference": {"n_entries": len(reference), "n_unique": len(train_set)},
        "validity": _rate(valid, all_attempts), "novelty": _rate(novel, valid),
        "properties": {},
    }
    any_applicable = np.zeros(len(rows), dtype=bool)
    all_hit = np.ones(len(rows), dtype=bool)
    for j, prop in enumerate(names):
        sign = np.asarray([v[j] for v in rows[SIGNS]], dtype=float)
        target_delta = np.asarray([v[j] for v in rows[DELTAS]], dtype=float)
        seed_value = np.asarray([v[j] for v in rows[SEED_VALUES]], dtype=float)
        if prop == activity_property:
            observed = np.asarray([activity_scores.get(s) for s in canon], dtype=float)
            scored = valid & np.isfinite(observed)
            hit = scored & np.asarray([activity_labels.get(s) == 1 for s in canon])
            realized = observed - seed_value
            rows["activity_probability"] = observed
        else:
            observed = np.asarray([values.get(s, {}).get(prop) for s in canon], dtype=float)
            baseline = np.asarray([values[s].get(prop) for s in seed_canon], dtype=float)
            realized = observed - baseline
            scored = valid & np.isfinite(realized)
            hit = scored & (np.sign(realized) == sign)
        applicable = sign != 0
        any_applicable |= applicable
        all_hit &= ~applicable | hit
        rows[f"applicable/{prop}"] = applicable
        rows[f"hit/{prop}"] = hit & applicable
        rows[f"realized_delta/{prop}"] = realized
        metrics = {
            "vin": _rate(valid & novel & hit, applicable),
            "direction_adherence": _rate(hit, valid & applicable),
            "n_scored": int((scored & applicable).sum()),
            "n_unscorable_valid": int((valid & applicable & ~scored).sum()),
        }
        if not metrics["n_scored"]:
            metrics["direction_adherence"]["rate"] = None
        if prop != activity_property:
            finite = scored & applicable
            metrics["mse"] = {
                "value": float(np.mean((realized[finite] - target_delta[finite]) ** 2)) if finite.any() else None,
                "n_scored": int(finite.sum()),
                "n_unscorable_valid": metrics["n_unscorable_valid"],
            }
        summary["properties"][prop] = metrics
    rows["combined_applicable"] = any_applicable
    rows["combined_hit"] = all_hit & any_applicable
    summary["combined_vin"] = _rate(valid & novel & all_hit, any_applicable)
    unavailable = [p for p in DESCRIPTORS if p not in summary["properties"] or summary["properties"][p]["direction_adherence"]["rate"] is None]
    summary["descriptor_macro_adherence"] = {
        "value": None if unavailable else float(np.mean([summary["properties"][p]["direction_adherence"]["rate"] for p in DESCRIPTORS])),
        "n_required_properties": len(DESCRIPTORS), "n_scored_properties": len(DESCRIPTORS) - len(unavailable),
        "unavailable_properties": unavailable,
    }
    similarity = rows["tanimoto"].to_numpy(dtype=float)
    finite = valid & np.isfinite(similarity)
    summary["similarity"] = {"mean": float(similarity[finite].mean()) if finite.any() else None, "n_scored": int(finite.sum()), "n_unscorable_valid": int((valid & ~finite).sum())}
    summary["activity_oracle"] = activity_oracle.provenance() if activity_oracle is not None else None
    return PromptedEvaluationResult(rows, summary)
