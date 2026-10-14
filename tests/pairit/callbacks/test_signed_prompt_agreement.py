"""Existing agreement metrics handle signed prompts without inventing zero signs."""
from unittest.mock import MagicMock

import pytest
import torch

from pairit.callbacks.molecule_validation import conditional_metrics
from pairit.eval import PropertySpec
from pairit.prompting import ParsedPropertyPrompt, PropertyBinningConfig


def _process(monkeypatch, prompt, realized_delta, *, default_only=False):
    values = {"C": {"score": 0.0}, "CC": {"score": realized_delta}}
    monkeypatch.setattr(conditional_metrics, "compute_properties", lambda smiles, specs: values[smiles])
    computer = conditional_metrics.ConditionalMetricsComputer(
        [PropertySpec(name="score", fn=lambda mol: 0.0)], expensive_metrics_to_compute=[],
    )
    module = MagicMock()
    module.device = torch.device("cpu")
    binning = PropertyBinningConfig(num_bins=16, range=(0.0, 4.0))
    result = computer.process_sample(
        module, seed_smiles="C", gen_smiles="CC", is_valid=True,
        canonical_gen="CC", parsed_prompts=[prompt],
        delta_binning_by_property=None if default_only else {"score": binning},
        delta_binning_default=binning,
    )
    return computer, result.prompt_agreement["score"]


@pytest.mark.parametrize(
    "actual,bin_idx,raw_direction,direction_match,bin_match",
    [
        (2.0, 12, None, True, True),
        (-0.01, 7, "down", True, True),
        (0.01, 8, "up", True, True),
        (-0.01, 8, "up", False, False),
        (0.0, 8, "up", False, True),
        (0.0, 8, "any", None, True),
        (0.01, 8, None, None, True),
        (-0.01, 7, None, True, True),
        (2.0, 8, "up", True, False),
    ],
)
def test_signed_direction_uses_raw_truth_before_quantized_interval(
    monkeypatch, actual, bin_idx, raw_direction, direction_match, bin_match,
):
    computer, agreement = _process(
        monkeypatch, ParsedPropertyPrompt("score", signed_delta_bin=bin_idx, raw_direction=raw_direction), actual,
    )
    assert agreement == {"direction_match": direction_match, "magnitude_match": bin_match}
    assert computer.epoch_prompt_magnitude_agreement["score"] == [bin_match]
    if direction_match is None:
        assert "score" not in computer.epoch_prompt_direction_agreement
    else:
        assert computer.epoch_prompt_direction_agreement_strict["score"] == [direction_match]


def test_default_binning_also_scores_signed_prompts(monkeypatch):
    _, agreement = _process(monkeypatch, ParsedPropertyPrompt("score", signed_delta_bin=12), 2.0, default_only=True)
    assert agreement == {"direction_match": True, "magnitude_match": True}


def test_existing_bounded_unsigned_agreement_remains_available(monkeypatch):
    _, agreement = _process(
        monkeypatch, ParsedPropertyPrompt("score", direction="up", delta_bin=9, delta_mode="atmost"), 2.0,
    )
    assert agreement == {"direction_match": True, "magnitude_match": True}
