"""Scientific denominator, provenance and coverage checks for prompting."""

import json

import numpy as np
import pandas as pd
import pytest

from pairit.eval import property_metrics
from pairit.eval.oracles import CallableOracle
from pairit.eval.prompted import (
    DELTAS, DESCRIPTORS, NAMES, SEED_VALUES, SIGNS, TARGET_VALUES, TRUTH_COLUMNS,
    evaluate_prompted_generations,
)
from pairit.eval.scripts.evaluate_prompted import main


def request(pair_id, names, deltas, seeds=None):
    seeds = seeds if seeds is not None else [0.0] * len(names)
    return {
        "pair_id": pair_id, "seed_smiles": "N", "target_smiles": "O",
        NAMES: list(names), SIGNS: np.sign(deltas).tolist(), DELTAS: list(deltas),
        SEED_VALUES: seeds, TARGET_VALUES: (np.asarray(seeds) + deltas).tolist(),
    }


def attempts(requests, molecules):
    return pd.DataFrame([
        {**row, "gen_idx": j, "generated_smiles": smi}
        for row, group in zip(requests, molecules) for j, smi in enumerate(group)
    ])


@pytest.fixture
def two_properties(monkeypatch):
    values = {"CC": {"qed": 0.5, "logp": 5}, "C": {"qed": 0.5, "logp": 5},
              "CCC": {"qed": 0, "logp": -2}, "CCO": {"qed": 0, "logp": 0},
              "N": {"qed": 0, "logp": 0}}
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: values[s])
    rows = [request("a", ["qed", "logp"], [1, 0]), request("b", ["qed", "logp"], [0, -1])]
    generations = attempts(rows, [["CC", "CC", "C", "invalid"], ["CCC", "CCO", "invalid", None]])
    return pd.DataFrame(rows), generations


def test_raw_attempt_union_repeats_and_nonzero_mse(two_properties):
    requests, generations = two_properties
    result = evaluate_prompted_generations(generations, requests, training_smiles=["C"], attempts_per_request=4)
    summary = result.summary
    assert summary["validity"] == {"numerator": 5, "denominator": 8, "rate": 5 / 8}
    assert summary["novelty"] == {"numerator": 4, "denominator": 5, "rate": 4 / 5}
    assert summary["combined_vin"] == {"numerator": 3, "denominator": 8, "rate": 3 / 8}
    assert summary["properties"]["qed"]["vin"]["numerator"] == 2
    assert summary["properties"]["qed"]["vin"]["denominator"] == 4
    assert summary["properties"]["qed"]["mse"] == {"value": 0.25, "n_scored": 3, "n_unscorable_valid": 0}
    assert summary["properties"]["logp"]["mse"]["value"] == 1
    assert summary["properties"]["logp"]["direction_adherence"]["rate"] == 0.5
    assert len(result.per_generation) == 8  # same seed, two distinct targets/requests; no dedup


@pytest.mark.parametrize("corruption, message", [
    ("missing_pair", "1 missing"), ("missing_attempt", "1 requests"),
    ("duplicate", "Duplicate"), ("changed_truth", "changed original"),
    ("fractional_index", "finite integers"),
])
def test_coverage_and_immutable_truth(two_properties, corruption, message):
    requests, generations = two_properties
    if corruption == "missing_pair":
        generations = generations[generations.pair_id == "a"]
    elif corruption == "missing_attempt":
        generations = generations.iloc[:-1]
    elif corruption == "duplicate":
        generations = pd.concat([generations, generations.iloc[:1]])
    elif corruption == "changed_truth":
        generations.at[0, DELTAS] = [2, 0]
    else:
        generations["gen_idx"] = generations.gen_idx.astype(float)
        generations.at[0, "gen_idx"] = 0.5
    with pytest.raises(ValueError, match=message):
        evaluate_prompted_generations(generations, requests, training_smiles=["C"], attempts_per_request=4)


def test_tiny_nonzero_request_and_single_value_mse(monkeypatch):
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: {"qed": 0 if s == "N" else 0.02})
    rows = [request("small", ["qed"], [0.01])]
    result = evaluate_prompted_generations(attempts(rows, [["CC"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=1)
    metric = result.summary["properties"]["qed"]
    assert metric["vin"]["rate"] == 1  # positive raw delta even if its signed bin is 08
    assert metric["mse"]["value"] == pytest.approx(0.0001)


def test_zero_only_request_is_inapplicable(monkeypatch):
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: {"qed": 0 if s == "N" else 0.2})
    rows = [request("zero", ["qed"], [0])]
    result = evaluate_prompted_generations(attempts(rows, [["CC"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=1)
    assert result.summary["combined_vin"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert result.summary["properties"]["qed"]["mse"]["n_scored"] == 0


def test_nonfinite_score_never_counts_as_zero_mse(monkeypatch):
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: {"qed": 0 if s == "N" else 0.5 if s == "CC" else None})
    rows = [request("finite", ["qed"], [1])]
    result = evaluate_prompted_generations(attempts(rows, [["CC", "CCC"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=2)
    metric = result.summary["properties"]["qed"]
    assert metric["mse"] == {"value": 0.25, "n_scored": 1, "n_unscorable_valid": 1}
    assert metric["vin"]["rate"] == 0.5


def test_activity_uses_original_population_and_explicit_strict_oracle(monkeypatch):
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: {"qed": 0 if s == "N" else 0.2})
    rows = [request("up", ["qed", "target"], [0.1, 1]),
            request("down", ["qed", "target"], [0.1, -1], [0, 1])]
    generations = attempts(rows, [["CC", "CCC", "O"], ["CC", "CCC", "O"]])
    generations["diagnostic_activity"] = 1.0  # deliberately not used
    oracle = CallableOracle(lambda smiles: [dict(CC=0.5, CCC=0.6, O=None)[s] for s in smiles], comparison=">")
    result = evaluate_prompted_generations(generations, pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=3,
                                           activity_property="target", activity_oracle=oracle)
    assert result.summary["coverage"]["excluded_requests"] == 1
    activity = result.summary["properties"]["target"]
    assert activity["vin"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert activity["n_unscorable_valid"] == 1
    assert "mse" not in activity
    oracle.comparison = ">="
    with pytest.raises(ValueError, match="strict score"):
        evaluate_prompted_generations(generations, pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=3,
                                      activity_property="target", activity_oracle=oracle)


def test_missing_descriptor_is_counted_and_macro_does_not_average_eight(monkeypatch):
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: {p: 0 if s == "N" else None if p == "tpsa" else 0.5 for p in DESCRIPTORS})
    rows = [request("nine", DESCRIPTORS, [1] * 9)]
    result = evaluate_prompted_generations(attempts(rows, [["CC"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=1)
    assert result.summary["properties"]["tpsa"]["mse"] == {"value": None, "n_scored": 0, "n_unscorable_valid": 1}
    assert result.summary["descriptor_macro_adherence"]["value"] is None
    assert result.summary["descriptor_macro_adherence"]["unavailable_properties"] == ["tpsa"]


def test_cli_reads_json_truth_and_records_explicit_inputs(two_properties, tmp_path):
    requests, generations = two_properties
    for frame, filename in ((requests, "requests.csv"), (generations, "generations.csv")):
        for column in TRUTH_COLUMNS:
            frame[column] = frame[column].map(json.dumps)
        frame.to_csv(tmp_path / filename, index=False)
    pd.DataFrame({"smiles": ["C"]}).to_csv(tmp_path / "train.csv", index=False)
    out = tmp_path / "scores"
    main(["--generations", str(tmp_path / "generations.csv"), "--requests", str(tmp_path / "requests.csv"),
          "--training-reference", str(tmp_path / "train.csv"), "--training-smiles-column", "smiles",
          "--attempts-per-request", "4", "--output-dir", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["combined_vin"]["rate"] == 3 / 8
    assert summary["inputs"]["requests"]["path"] == str(tmp_path / "requests.csv")
    assert len(pd.read_csv(out / "per_generation.csv")) == 8


def test_rdkit_sa_descriptor_is_available():
    value = property_metrics.compute_properties("CCO", property_metrics.get_property_specs(["sa"]))["sa"]
    assert value is not None and np.isfinite(value)


def test_seed_copy_has_zero_realized_delta_even_when_raw_seed_value_differs(monkeypatch):
    monkeypatch.setattr(property_metrics, "compute_properties", lambda s, specs: {"qed": 0.2})
    rows = [request("copy", ["qed"], [0.1])]
    result = evaluate_prompted_generations(attempts(rows, [["N"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=1)
    assert result.per_generation.iloc[0]["is_novel"]  # membership only, no special seed-copy novelty rule
    assert result.per_generation.iloc[0]["realized_delta/qed"] == 0
    assert result.summary["properties"]["qed"]["vin"]["rate"] == 0


def test_nine_descriptor_macro_is_unweighted(monkeypatch):
    def values(s, specs):
        return {p: 0 if s == "N" else -1 if s == "CC" and p == "qed" else 1 for p in DESCRIPTORS}
    monkeypatch.setattr(property_metrics, "compute_properties", values)
    qed_only = [int(p == "qed") for p in DESCRIPTORS]
    rows = [request("all", DESCRIPTORS, [1] * 9), request("qed", DESCRIPTORS, qed_only)]
    result = evaluate_prompted_generations(attempts(rows, [["CC"], ["CCC"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=1)
    assert result.summary["descriptor_macro_adherence"]["value"] == pytest.approx(17 / 18)


def test_generic_descriptor_proxy_cannot_be_scored_as_activity():
    rows = [request("proxy", ["target"], [0.2], [2.3])]
    oracle = CallableOracle(lambda smiles: [0.7] * len(smiles), comparison=">")
    with pytest.raises(ValueError, match="measured binary labels"):
        evaluate_prompted_generations(attempts(rows, [["CC"]]), pd.DataFrame(rows), training_smiles=["C"], attempts_per_request=1,
                                      activity_property="target", activity_oracle=oracle)


@pytest.mark.parametrize("reference", [[], ["C", "invalid"]])
def test_reference_cannot_silently_drop_missing_or_invalid_molecules(two_properties, reference):
    requests, generations = two_properties
    with pytest.raises(ValueError, match="Novelty reference"):
        evaluate_prompted_generations(generations, requests, training_smiles=reference, attempts_per_request=4)
