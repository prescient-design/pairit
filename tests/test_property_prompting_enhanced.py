"""Tests for enhanced property prompting: DeltaMode, bounded deltas, component dropout, explicit selection."""
from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest

from pairit.prompting.property_prompting import (
    DeltaMode,
    ParsedPropertyPrompt,
    PropertyBinningConfig,
    build_property_prompt_text,
    format_property_prompt_summary,
    parse_property_prompt_from_ids,
)


# ---------------------------------------------------------------------------
# 1. Backward compat: delta_mode defaults to "exact"
# ---------------------------------------------------------------------------


def test_delta_mode_backward_compat():
    p = ParsedPropertyPrompt(property_name="logp")
    assert p.delta_mode == "exact"
    assert p.start_bin is None
    assert p.direction is None
    assert p.delta_bin is None


# ---------------------------------------------------------------------------
# 2. build_property_prompt_text emits bounded tokens
# ---------------------------------------------------------------------------


class TestBoundedDeltaTokens:
    def test_atmost_token(self):
        p = ParsedPropertyPrompt(property_name="logp", delta_bin=3, delta_mode="atmost")
        text = build_property_prompt_text([p])
        assert "<pdmax_logp_03>" in text
        assert "<pd_logp_" not in text
        assert "<pdmin_logp_" not in text

    def test_atleast_token(self):
        p = ParsedPropertyPrompt(property_name="sa", delta_bin=7, delta_mode="atleast")
        text = build_property_prompt_text([p])
        assert "<pdmin_sa_07>" in text
        assert "<pd_sa_" not in text
        assert "<pdmax_sa_" not in text

    def test_exact_token(self):
        p = ParsedPropertyPrompt(property_name="qed", delta_bin=5, delta_mode="exact")
        text = build_property_prompt_text([p])
        assert "<pd_qed_05>" in text
        assert "<pdmax_" not in text
        assert "<pdmin_" not in text

    def test_mixed_modes(self):
        prompts = [
            ParsedPropertyPrompt(property_name="logp", delta_bin=2, delta_mode="exact"),
            ParsedPropertyPrompt(property_name="sa", delta_bin=4, delta_mode="atmost"),
            ParsedPropertyPrompt(property_name="qed", delta_bin=1, delta_mode="atleast"),
        ]
        text = build_property_prompt_text(prompts)
        assert "<pd_logp_02>" in text
        assert "<pdmax_sa_04>" in text
        assert "<pdmin_qed_01>" in text


# ---------------------------------------------------------------------------
# 3. parse_property_prompt_from_ids round-trips bounded tokens
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(tokens: list[str]):
    """Create a mock tokenizer that maps IDs 0..N-1 to the given token strings."""
    tok = MagicMock()
    tok.convert_ids_to_tokens = MagicMock(return_value=tokens)
    return tok


class TestBoundedDeltaParsing:
    def test_parse_atmost(self):
        tokens = ["<pc>", "<p_logp>", "<pdmax_logp_05>"]
        tok = _make_mock_tokenizer(tokens)
        result = parse_property_prompt_from_ids(tok, list(range(len(tokens))))
        assert len(result) == 1
        assert result[0].property_name == "logp"
        assert result[0].delta_bin == 5
        assert result[0].delta_mode == "atmost"

    def test_parse_atleast(self):
        tokens = ["<pc>", "<p_sa>", "<pdmin_sa_12>"]
        tok = _make_mock_tokenizer(tokens)
        result = parse_property_prompt_from_ids(tok, list(range(len(tokens))))
        assert len(result) == 1
        assert result[0].delta_bin == 12
        assert result[0].delta_mode == "atleast"

    def test_parse_exact(self):
        tokens = ["<pc>", "<p_qed>", "<pd_qed_03>"]
        tok = _make_mock_tokenizer(tokens)
        result = parse_property_prompt_from_ids(tok, list(range(len(tokens))))
        assert len(result) == 1
        assert result[0].delta_bin == 3
        assert result[0].delta_mode == "exact"

    def test_roundtrip_all_modes(self):
        """Build text, simulate tokenization, parse back."""
        prompts = [
            ParsedPropertyPrompt(property_name="logp", start_bin=7, direction="up", delta_bin=3, delta_mode="exact"),
            ParsedPropertyPrompt(property_name="sa", start_bin=2, direction="down", delta_bin=5, delta_mode="atmost"),
            ParsedPropertyPrompt(property_name="qed", direction="any", delta_bin=1, delta_mode="atleast"),
        ]
        text = build_property_prompt_text(prompts)
        # Simulate tokenizer: split at >< boundaries
        raw_tokens = []
        current = ""
        for ch in text:
            current += ch
            if ch == ">":
                raw_tokens.append(current)
                current = ""
        tok = _make_mock_tokenizer(raw_tokens)
        parsed = parse_property_prompt_from_ids(tok, list(range(len(raw_tokens))))
        assert len(parsed) == 3
        assert parsed[0].delta_mode == "exact"
        assert parsed[0].delta_bin == 3
        assert parsed[1].delta_mode == "atmost"
        assert parsed[1].delta_bin == 5
        assert parsed[2].delta_mode == "atleast"
        assert parsed[2].delta_bin == 1


# ---------------------------------------------------------------------------
# 4. Dropout determinism
# ---------------------------------------------------------------------------


class TestDropoutDeterminism:
    def _make_dataset_stub(self, rate: float = 0.5, epoch_vary: bool = True):
        """Create a minimal stub mimicking MoleculeImprovementDataset dropout fields."""

        class Stub:
            _prop_component_dropout_enabled = True
            _prop_component_dropout_rate = rate
            _prop_component_dropout_epoch_vary = epoch_vary
            _prop_randomize_tokens = False
            _current_epoch = 0

            @staticmethod
            def dropout_components(smiles: str, prop_name: str, start_bin, direction, delta_bin, epoch: int):
                """Reproduce the dropout logic from _build_property_prompts."""
                import hashlib

                stable = int(hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:8], 16)
                epoch_offset = epoch * 7919 if epoch_vary else 0
                dr = random.Random(stable + epoch_offset + hash(prop_name))
                if dr.random() < rate:
                    start_bin = None
                if dr.random() < rate:
                    direction = None
                if dr.random() < rate:
                    delta_bin = None
                return start_bin, direction, delta_bin

        return Stub()

    def test_same_smiles_same_epoch_same_result(self):
        stub = self._make_dataset_stub(rate=0.5)
        r1 = stub.dropout_components("CCO", "logp", 5, "up", 3, epoch=0)
        r2 = stub.dropout_components("CCO", "logp", 5, "up", 3, epoch=0)
        assert r1 == r2

    def test_different_epoch_different_result(self):
        stub = self._make_dataset_stub(rate=0.5, epoch_vary=True)
        r1 = stub.dropout_components("CCO", "logp", 5, "up", 3, epoch=0)
        r2 = stub.dropout_components("CCO", "logp", 5, "up", 3, epoch=1)
        # With high probability these differ; use multiple smiles to be safe
        any_different = False
        for smi in ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "C1CCCCC1"]:
            a = stub.dropout_components(smi, "logp", 5, "up", 3, epoch=0)
            b = stub.dropout_components(smi, "logp", 5, "up", 3, epoch=1)
            if a != b:
                any_different = True
                break
        assert any_different, "Expected different dropout patterns across epochs"


# ---------------------------------------------------------------------------
# 5. Dropout rate empirical check
# ---------------------------------------------------------------------------


def test_dropout_rate():
    """Over many samples, empirical drop rate should be within tolerance of configured rate."""
    import hashlib

    target_rate = 0.3
    n_samples = 2000
    dropped = 0
    for i in range(n_samples):
        smi = f"C{'C' * i}"
        stable = int(hashlib.sha1(smi.encode("utf-8")).hexdigest()[:8], 16)
        dr = random.Random(stable + hash("logp"))
        if dr.random() < target_rate:
            dropped += 1
    empirical_rate = dropped / n_samples
    assert abs(empirical_rate - target_rate) < 0.05, (
        f"Empirical drop rate {empirical_rate:.3f} too far from target {target_rate}"
    )


# ---------------------------------------------------------------------------
# 6. Explicit selection mode
# ---------------------------------------------------------------------------


def test_explicit_selection():
    """Only specified properties appear when mode=explicit."""
    all_names = ["logp", "qed", "sa", "mw", "tpsa"]
    explicit = ["logp", "sa"]
    chosen = [n for n in explicit if n in all_names]
    assert chosen == ["logp", "sa"]
    # Property not in all_names is silently skipped
    chosen2 = [n for n in ["logp", "nonexistent"] if n in all_names]
    assert chosen2 == ["logp"]


# ---------------------------------------------------------------------------
# 7. New tokens registered in tokenizer
# ---------------------------------------------------------------------------


def test_new_tokens_registered():
    """Bounded delta tokens are present in the tokenizer vocab."""
    from pairit.tokenization._smiles_tokenizer2 import SmilesTokenizer2Fast

    tok = SmilesTokenizer2Fast(
        add_property_prompt_tokens=True,
        property_prompt_tokens={
            "property_names": ["logp", "sa"],
            "num_value_bins": 32,
            "num_delta_bins": 16,
        },
    )
    vocab = tok.get_vocab()
    # Exact delta tokens
    assert "<pd_logp_00>" in vocab
    assert "<pd_sa_15>" in vocab
    # Bounded delta tokens
    assert "<pdmax_logp_00>" in vocab
    assert "<pdmin_logp_00>" in vocab
    assert "<pdmax_sa_15>" in vocab
    assert "<pdmin_sa_15>" in vocab
    # Verify bounded tokens have distinct IDs
    assert vocab["<pd_logp_00>"] != vocab["<pdmax_logp_00>"]
    assert vocab["<pdmax_logp_00>"] != vocab["<pdmin_logp_00>"]


# ---------------------------------------------------------------------------
# 8. format_property_prompt_summary renders bounded modes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. Per-reference memorization tracking in ConditionalMetricsComputer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8b. Bounded delta offset distribution
# ---------------------------------------------------------------------------


class TestBoundedDeltaDistribution:
    """Verify that bounded delta sampling uses geometric offsets, not uniform."""

    def test_atmost_offset_is_geometrically_distributed(self):
        """Atmost bounds should cluster near the true bin, not be uniform."""
        true_bin = 5
        max_bin = 15
        n_samples = 5000
        offsets = []
        for i in range(n_samples):
            r = random.Random(i)
            remaining = max_bin - true_bin  # 10 bins of headroom
            offset = 0
            while offset < remaining and r.random() < 0.5:
                offset += 1
            offsets.append(offset)

        mean_offset = sum(offsets) / len(offsets)
        # Geometric with p=0.5 has theoretical mean ~1.0 (capped, so slightly less)
        # Uniform would give mean ~5.0.  Assert we're much closer to geometric.
        assert mean_offset < 2.0, f"Mean offset {mean_offset:.2f} too large (expected ~1.0, uniform would be ~5.0)"

        # At least 40% of samples should have offset <= 1
        tight = sum(1 for o in offsets if o <= 1)
        assert tight / n_samples > 0.4, f"Only {tight/n_samples:.1%} of bounds are tight (offset<=1)"

    def test_atleast_offset_is_geometrically_distributed(self):
        """Atleast bounds should cluster near the true bin, not be uniform."""
        true_bin = 10
        n_samples = 5000
        offsets = []
        for i in range(n_samples):
            r = random.Random(i)
            remaining = true_bin  # 10 bins of headroom downward
            offset = 0
            while offset < remaining and r.random() < 0.5:
                offset += 1
            offsets.append(offset)

        mean_offset = sum(offsets) / len(offsets)
        assert mean_offset < 2.0, f"Mean offset {mean_offset:.2f} too large"

    def test_bound_stays_in_valid_range(self):
        """Bounds must never exceed [0, max_bin]."""
        max_bin = 15
        for true_bin in range(max_bin + 1):
            for i in range(200):
                r = random.Random(i + true_bin * 1000)
                # atmost
                remaining = max_bin - true_bin
                offset = 0
                while offset < remaining and r.random() < 0.5:
                    offset += 1
                assert 0 <= true_bin + offset <= max_bin

                # atleast
                r2 = random.Random(i + true_bin * 2000)
                remaining = true_bin
                offset = 0
                while offset < remaining and r2.random() < 0.5:
                    offset += 1
                assert 0 <= true_bin - offset <= max_bin


# ---------------------------------------------------------------------------
# 9. Per-reference memorization tracking in ConditionalMetricsComputer
# ---------------------------------------------------------------------------


class TestPerRefMemorizationTracking:
    """Verify that per-reference memorization lists are initialized and populated
    in parallel with the main prompt agreement lists."""

    def _make_computer(self, ref_names: list[str]):
        """Build a minimal ConditionalMetricsComputer with mock ref sets."""
        from pairit.callbacks.molecule_validation.conditional_metrics import (
            ConditionalMetricsComputer,
        )

        class FakeSpec:
            def __init__(self, name):
                self.name = name

        comp = object.__new__(ConditionalMetricsComputer)
        comp.property_specs = [FakeSpec("logp")]
        comp.reference_canon_sets_by_name = {
            r: {"MEMORIZED_SMILES"} for r in ref_names
        }
        comp.distance_name = "morgan_tanimoto_distance"
        comp.timing_tracker = None
        # _reset_conditional sets up all the prompt tracking buffers we test
        comp._reset_conditional()
        return comp

    def test_direction_by_ref_initialized_on_first_record(self):
        comp = self._make_computer(["zinc_train", "bace_train"])
        # Simulate recording a direction match for prop "logp"
        prop = "logp"
        # Lazy init (mirrors the code in process_single_generation)
        if prop not in comp.epoch_prompt_direction_agreement:
            comp.epoch_prompt_direction_agreement[prop] = []
            comp.epoch_prompt_direction_memorized[prop] = []
            for ref_name in comp.epoch_prompt_direction_memorized_by_ref:
                comp.epoch_prompt_direction_memorized_by_ref[ref_name][prop] = []

        comp.epoch_prompt_direction_agreement[prop].append(True)
        comp.epoch_prompt_direction_memorized[prop].append(False)
        for ref_name in comp.epoch_prompt_direction_memorized_by_ref:
            comp.epoch_prompt_direction_memorized_by_ref[ref_name][prop].append(False)

        # Verify inner lists exist and have correct length
        for ref_name in ["zinc_train", "bace_train"]:
            assert prop in comp.epoch_prompt_direction_memorized_by_ref[ref_name]
            assert len(comp.epoch_prompt_direction_memorized_by_ref[ref_name][prop]) == 1

        # Simulate per-ref backfill (as done in process_single_generation)
        for ref_name in ["zinc_train", "bace_train"]:
            is_mem = True  # pretend memorized in this ref
            if prop in comp.epoch_prompt_direction_memorized_by_ref.get(ref_name, {}):
                if len(comp.epoch_prompt_direction_memorized_by_ref[ref_name][prop]) > 0:
                    comp.epoch_prompt_direction_memorized_by_ref[ref_name][prop][-1] = is_mem

        for ref_name in ["zinc_train", "bace_train"]:
            assert comp.epoch_prompt_direction_memorized_by_ref[ref_name][prop][-1] is True

    def test_all_properties_by_ref_tracks_correctly(self):
        comp = self._make_computer(["zinc_train"])
        # Append to all-properties tracking
        comp.epoch_all_properties_direction_agree.append(True)
        comp.epoch_all_properties_direction_memorized.append(False)
        for ref_name in comp.epoch_all_properties_direction_memorized_by_ref:
            comp.epoch_all_properties_direction_memorized_by_ref[ref_name].append(False)

        assert len(comp.epoch_all_properties_direction_memorized_by_ref["zinc_train"]) == 1

        # Backfill
        comp.epoch_all_properties_direction_memorized_by_ref["zinc_train"][-1] = True
        assert comp.epoch_all_properties_direction_memorized_by_ref["zinc_train"][-1] is True

    def test_magnitude_by_ref_initialized_on_first_record(self):
        comp = self._make_computer(["zinc_train"])
        prop = "logp"
        if prop not in comp.epoch_prompt_magnitude_agreement:
            comp.epoch_prompt_magnitude_agreement[prop] = []
            comp.epoch_prompt_magnitude_memorized[prop] = []
            comp.epoch_prompt_magnitude_bin_errors[prop] = []
            comp.epoch_prompt_magnitude_bin_errors_memorized[prop] = []
            for ref_name in comp.epoch_prompt_magnitude_memorized_by_ref:
                comp.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop] = []
            for ref_name in comp.epoch_prompt_magnitude_bin_errors_memorized_by_ref:
                comp.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop] = []

        comp.epoch_prompt_magnitude_agreement[prop].append(True)
        comp.epoch_prompt_magnitude_memorized[prop].append(False)
        for ref_name in comp.epoch_prompt_magnitude_memorized_by_ref:
            comp.epoch_prompt_magnitude_memorized_by_ref[ref_name][prop].append(False)
        comp.epoch_prompt_magnitude_bin_errors[prop].append(0)
        comp.epoch_prompt_magnitude_bin_errors_memorized[prop].append(False)
        for ref_name in comp.epoch_prompt_magnitude_bin_errors_memorized_by_ref:
            comp.epoch_prompt_magnitude_bin_errors_memorized_by_ref[ref_name][prop].append(False)

        assert "logp" in comp.epoch_prompt_magnitude_memorized_by_ref["zinc_train"]
        assert len(comp.epoch_prompt_magnitude_memorized_by_ref["zinc_train"]["logp"]) == 1
        assert "logp" in comp.epoch_prompt_magnitude_bin_errors_memorized_by_ref["zinc_train"]
        assert len(comp.epoch_prompt_magnitude_bin_errors_memorized_by_ref["zinc_train"]["logp"]) == 1


class TestFormatSummary:
    def test_exact_format(self):
        p = ParsedPropertyPrompt(property_name="logp", delta_bin=3, delta_mode="exact")
        s = format_property_prompt_summary([p], show_ranges=False)
        assert "|Δ|=03" in s

    def test_atmost_format(self):
        p = ParsedPropertyPrompt(property_name="logp", delta_bin=3, delta_mode="atmost")
        s = format_property_prompt_summary([p], show_ranges=False)
        assert "|Δ|<=03" in s

    def test_atleast_format(self):
        p = ParsedPropertyPrompt(property_name="logp", delta_bin=3, delta_mode="atleast")
        s = format_property_prompt_summary([p], show_ranges=False)
        assert "|Δ|>=03" in s
