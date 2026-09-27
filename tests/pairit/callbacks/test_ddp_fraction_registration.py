"""
Tests that all ranks register identical fraction sets, preventing NCCL deadlocks.

The core invariant: every code path in conditional_metrics, unconditional_metrics,
and the main callback must register the SAME set of fraction names regardless of
how much data each rank has. This test simulates rank-divergent data conditions
and verifies fraction name consistency.
"""
import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from pairit.callbacks.molecule_validation.printer import MetricRegistry
from pairit.callbacks.molecule_validation.timing import TimingTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conditional(n_valid, n_direction, n_memorized_checked,
                      property_names=None, ref_names=None):
    """Create a ConditionalMetricsComputer with controlled data amounts."""
    from pairit.callbacks.molecule_validation.conditional_metrics import (
        ConditionalMetricsComputer,
    )

    if property_names is None:
        property_names = ["logp"]
    if ref_names is None:
        ref_names = ["zinc_train"]

    # Build PropertySpec-like objects
    specs = []
    for p in property_names:
        spec = MagicMock()
        spec.name = p
        specs.append(spec)

    ref_canon = {r: {"CCO"} for r in ref_names}
    ref_smiles = {r: ["CCO", "CCN"] for r in ref_names}

    cm = ConditionalMetricsComputer(
        property_specs=specs,
        distance_name="tanimoto",
        reference_canon_sets_by_name=ref_canon,
        default_reference_name=ref_names[0],
        expensive_metrics_to_compute=["memorization"],
        timing_tracker=TimingTracker(),
        reference_smiles_by_name=ref_smiles,
        fcd_min_required=10,
    )
    cm.reset()

    # Populate epoch accumulators to simulate data
    # Use underlying attributes (the public names are read-only properties)
    cm.epoch_valid_canon_gens = [f"mol_{i}" for i in range(n_valid)]
    cm.epoch_seeds = [f"seed_{i}" for i in range(max(n_valid, 1))] if n_valid > 0 else []
    cm.epoch_gens = [f"gen_{i}" for i in range(max(n_valid, 1))] if n_valid > 0 else []
    cm.epoch_cond_gt_lists = []

    # Direction agreement
    cm.epoch_prompt_direction_agreement = {
        p: [True] * n_direction for p in property_names
    }
    cm.epoch_prompt_direction_memorized = {
        p: [False] * n_direction for p in property_names
    }
    cm.epoch_prompt_direction_memorized_by_ref = {
        r: {p: [False] * n_direction for p in property_names}
        for r in ref_names
    }
    cm.epoch_prompt_direction_agreement_strict = {
        p: [True] * n_direction for p in property_names
    }
    cm.epoch_prompt_direction_memorized_strict = {
        p: [False] * n_direction for p in property_names
    }

    # Magnitude agreement
    cm.epoch_prompt_magnitude_agreement = {
        p: [True] * n_direction for p in property_names
    }
    cm.epoch_prompt_magnitude_memorized = {
        p: [False] * n_direction for p in property_names
    }
    cm.epoch_prompt_magnitude_memorized_by_ref = {
        r: {p: [False] * n_direction for p in property_names}
        for r in ref_names
    }
    cm.epoch_prompt_magnitude_bin_errors = {
        p: [0.0] * n_direction for p in property_names
    }
    cm.epoch_prompt_magnitude_bin_errors_memorized = {
        p: [False] * n_direction for p in property_names
    }
    cm.epoch_prompt_magnitude_bin_errors_memorized_by_ref = {
        r: {p: [False] * n_direction for p in property_names}
        for r in ref_names
    }
    cm.epoch_prompt_actual_bins = {
        p: list(range(max(n_direction, 1))) if n_direction > 0 else []
        for p in property_names
    }
    cm.epoch_prompt_prompted_bins = {
        p: list(range(max(n_direction, 1))) if n_direction > 0 else []
        for p in property_names
    }

    # Distance agreement
    cm.epoch_distance_prompt_agreement = [True] * n_direction
    cm.epoch_distance_prompt_memorized = [False] * n_direction
    cm.epoch_distance_prompt_memorized_by_ref = {
        r: [False] * n_direction for r in ref_names
    }

    # All-properties agreement
    cm.epoch_all_properties_direction_agree = [True] * n_direction
    cm.epoch_all_properties_direction_memorized = [False] * n_direction
    cm.epoch_all_properties_direction_memorized_by_ref = {
        r: [False] * n_direction for r in ref_names
    }
    cm.epoch_all_properties_magnitude_agree = [True] * n_direction
    cm.epoch_all_properties_magnitude_memorized = [False] * n_direction
    cm.epoch_all_properties_magnitude_memorized_by_ref = {
        r: [False] * n_direction for r in ref_names
    }

    # Memorization
    cm.epoch_memorized_checked = n_memorized_checked
    cm.epoch_memorized_count = 0
    cm.epoch_memorized_checked_by_ref = {r: n_memorized_checked for r in ref_names}
    cm.epoch_memorized_count_by_ref = {r: 0 for r in ref_names}

    # Distance metrics (means, not fractions)
    cm.epoch_cond_distances = []
    cm.epoch_cond_distance_percentiles = []

    return cm


def _make_unconditional(n_checked, n_first_valid, n_second_valid, ref_names=None):
    """Create an UnconditionalMetricsComputer with controlled data amounts."""
    from pairit.callbacks.molecule_validation.unconditional_metrics import (
        UnconditionalMetricsComputer,
    )

    if ref_names is None:
        ref_names = ["zinc_train"]

    ref_canon = {r: {"CCO"} for r in ref_names}
    ref_smiles = {r: ["CCO", "CCN"] for r in ref_names}

    um = UnconditionalMetricsComputer(
        reference_smiles_by_name=ref_smiles,
        default_reference_name=ref_names[0],
        fcd_min_required=10,
        expensive_metrics_to_compute=["memorization"],
        reference_canon_sets_by_name=ref_canon,
        timing_tracker=TimingTracker(),
    )
    um.reset()

    um.epoch_total_checked = n_checked
    um.epoch_first_valid_count = n_first_valid
    um.epoch_second_valid_count = n_second_valid
    um.epoch_uncond_valid_gen_smiles = [f"mol_{i}" for i in range(n_first_valid)]

    um.epoch_first_mol_checked = n_first_valid
    um.epoch_first_mol_memorized = 0
    um.epoch_second_mol_checked = n_second_valid
    um.epoch_second_mol_memorized = 0
    um.epoch_first_mol_checked_by_ref = {r: n_first_valid for r in ref_names}
    um.epoch_first_mol_memorized_by_ref = {r: 0 for r in ref_names}
    um.epoch_second_mol_checked_by_ref = {r: n_second_valid for r in ref_names}
    um.epoch_second_mol_memorized_by_ref = {r: 0 for r in ref_names}

    return um


# ---------------------------------------------------------------------------
# 1. MetricRegistry basics: (0,0) → NaN, additive identity
# ---------------------------------------------------------------------------

class TestMetricRegistryFractions:
    """Verify MetricRegistry handles (0,0) fractions correctly."""

    def test_zero_zero_fraction_produces_nan(self):
        reg = MetricRegistry()
        reg.register_fraction("test_metric", 0, 0, "test")
        assert math.isnan(reg.get("test_metric"))

    def test_normal_fraction(self):
        reg = MetricRegistry()
        reg.register_fraction("test_metric", 3, 5, "test")
        assert reg.get("test_metric") == pytest.approx(0.6)

    def test_zero_zero_is_additive_identity(self):
        """Simulating DDP sum: (3,5)+(0,0) should give 3/5."""
        global_num = 3.0 + 0.0
        global_den = 5.0 + 0.0
        result = global_num / global_den if global_den > 0 else float('nan')
        assert result == pytest.approx(0.6)

    def test_all_ranks_zero_gives_nan(self):
        global_num = 0.0 + 0.0
        global_den = 0.0 + 0.0
        result = global_num / global_den if global_den > 0 else float('nan')
        assert math.isnan(result)

    def test_log_all_without_dist(self):
        """log_all works in non-DDP mode with (0,0) fractions."""
        reg = MetricRegistry()
        reg.register_fraction("metric_a", 0, 0, "test")
        reg.register_fraction("metric_b", 3, 5, "test")

        pl_module = MagicMock()
        pl_module.device = torch.device("cpu")
        pl_module._trainer = MagicMock()

        with patch("pairit.callbacks.molecule_validation.printer.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            reg.log_all(pl_module, prefix="val")

        assert pl_module.log.call_count == 2

    def test_mixed_fractions_resolve_correctly(self):
        reg = MetricRegistry()
        reg.register_fraction("no_data", 0, 0, "")
        reg.register_fraction("has_data", 7, 10, "")

        pl_module = MagicMock()
        pl_module.device = torch.device("cpu")
        pl_module._trainer = MagicMock()

        with patch("pairit.callbacks.molecule_validation.printer.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            reg.log_all(pl_module, prefix="val")

        logged = {call[0][0]: call[0][1].item() for call in pl_module.log.call_args_list}
        assert math.isnan(logged["val_no_data"])
        assert logged["val_has_data"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 2. Conditional metrics: fraction names are identical regardless of data
# ---------------------------------------------------------------------------

class TestConditionalMetricsFractionConsistency:
    """
    Simulate two ranks with different data volumes and verify they register
    the same fraction names in ConditionalMetricsComputer.compute_and_log().
    """

    def test_fraction_names_match_across_ranks(self):
        """Two 'ranks' with different data must register the same fraction names."""
        # Rank 0: has data
        cm_rank0 = _make_conditional(n_valid=10, n_direction=5, n_memorized_checked=10)
        reg0 = MetricRegistry()
        cm_rank0.compute_and_log(MagicMock(), reg0)

        # Rank 1: NO data
        cm_rank1 = _make_conditional(n_valid=0, n_direction=0, n_memorized_checked=0)
        reg1 = MetricRegistry()
        cm_rank1.compute_and_log(MagicMock(), reg1)

        frac_names_0 = sorted(reg0.fractions.keys())
        frac_names_1 = sorted(reg1.fractions.keys())
        assert frac_names_0 == frac_names_1, (
            f"Fraction name mismatch!\n"
            f"Only in rank0: {set(frac_names_0) - set(frac_names_1)}\n"
            f"Only in rank1: {set(frac_names_1) - set(frac_names_0)}"
        )

    def test_zero_data_fractions_are_zero_zero(self):
        """With no data, all fractions should have denominator 0."""
        cm = _make_conditional(n_valid=0, n_direction=0, n_memorized_checked=0)
        reg = MetricRegistry()
        cm.compute_and_log(MagicMock(), reg)

        for name, (num, denom) in reg.fractions.items():
            assert denom == 0.0, f"Fraction '{name}' has denom={denom}, expected 0"
            assert num == 0.0, f"Fraction '{name}' has num={num}, expected 0"

    def test_data_fractions_are_non_nan(self):
        """With data, all fractions should resolve to non-NaN."""
        cm = _make_conditional(n_valid=10, n_direction=5, n_memorized_checked=10)
        reg = MetricRegistry()
        cm.compute_and_log(MagicMock(), reg)

        for name, (num, denom) in reg.fractions.items():
            val = num / denom if denom > 0 else float('nan')
            assert not math.isnan(val), f"Fraction '{name}' is NaN with data present"


# ---------------------------------------------------------------------------
# 3. Unconditional metrics: fraction names are identical regardless of data
# ---------------------------------------------------------------------------

class TestUnconditionalMetricsFractionConsistency:
    """Verify unconditional metrics register identical fractions across ranks."""

    def test_fraction_names_match_across_ranks(self):
        um_rank0 = _make_unconditional(n_checked=10, n_first_valid=5, n_second_valid=5)
        reg0 = MetricRegistry()
        um_rank0.compute_and_log(reg0)

        um_rank1 = _make_unconditional(n_checked=0, n_first_valid=0, n_second_valid=0)
        reg1 = MetricRegistry()
        um_rank1.compute_and_log(reg1)

        frac_names_0 = sorted(reg0.fractions.keys())
        frac_names_1 = sorted(reg1.fractions.keys())
        assert frac_names_0 == frac_names_1, (
            f"Fraction name mismatch!\n"
            f"Only in rank0: {set(frac_names_0) - set(frac_names_1)}\n"
            f"Only in rank1: {set(frac_names_1) - set(frac_names_0)}"
        )

    def test_zero_data_fractions_are_zero_zero(self):
        um = _make_unconditional(n_checked=0, n_first_valid=0, n_second_valid=0)
        reg = MetricRegistry()
        um.compute_and_log(reg)

        for name, (num, denom) in reg.fractions.items():
            assert denom == 0.0, f"Fraction '{name}' has denom={denom}, expected 0"
            assert num == 0.0, f"Fraction '{name}' has num={num}, expected 0"
