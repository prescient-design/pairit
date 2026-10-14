"""Signed prompting keeps legacy encodings and raw pair truth distinct."""
from dataclasses import replace

import pytest

from pairit.prompting import (
    ParsedPropertyPrompt,
    PropertyBinningConfig,
    build_property_prompt_text,
    format_property_prompt_summary,
    parse_property_prompt_from_ids,
    signed_bin_to_range,
    signed_delta_to_bin,
)
from pairit.tokenization import SmilesTokenizer2Fast


BINNING = PropertyBinningConfig(num_bins=16, range=(0.0, 4.0))


@pytest.mark.parametrize("delta,expected", [(-8, 0), (-4, 0), (-0.01, 7), (0, 8), (0.01, 8), (4, 15), (8, 15)])
def test_signed_bins_and_clipping(delta, expected):
    assert signed_delta_to_bin(binning=BINNING, delta=delta) == expected


def test_zero_containing_interval_is_not_an_exact_zero_token():
    assert signed_bin_to_range(binning=BINNING, bin_idx=7) == (-0.5, 0.0)
    assert signed_bin_to_range(binning=BINNING, bin_idx=8) == (0.0, 0.5)
    assert signed_bin_to_range(binning=BINNING, bin_idx=15) == (3.5, 4.0)
    with pytest.raises(ValueError, match="out of range"):
        signed_delta_to_bin(binning=replace(BINNING, clip=False), delta=5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_delta_cannot_be_encoded(value):
    with pytest.raises(ValueError, match="finite"):
        signed_delta_to_bin(binning=BINNING, delta=value)


def _tokenizer(signed=False, inherited=None):
    properties = {"property_names": ["qed", "logp"], "num_value_bins": 32, "num_delta_bins": 16}
    if signed:
        properties["num_signed_delta_bins"] = 16
    return SmilesTokenizer2Fast(
        inherit_special_tokens=inherited,
        add_dataset_tokens=True,
        max_dataset_id=2,
        add_property_prompt_tokens=True,
        property_prompt_tokens=properties,
        add_distance_prompt_tokens=True,
        distance_prompt_tokens={"num_bins": 20},
    )


def test_signed_tokens_preserve_all_legacy_ids_including_distance():
    legacy = _tokenizer()
    signed = _tokenizer(signed=True)
    assert all(signed.get_vocab()[token] == idx for token, idx in legacy.get_vocab().items())
    assert signed.convert_tokens_to_ids("<psd_logp_00>") > max(legacy.get_vocab().values())
    assert len(signed) == len(legacy) + 32
    assert set(legacy.additional_special_tokens) <= set(signed.additional_special_tokens)
    assert {"<p_logp>", "<d_00>", "<psd_logp_00>"} <= set(signed.additional_special_tokens)


def test_inherited_signed_ids_remain_stable():
    original = _tokenizer(signed=True)
    inherited = sorted(original.additional_special_tokens, key=original.convert_tokens_to_ids)
    restored = _tokenizer(signed=True, inherited=inherited)
    assert restored.get_vocab() == original.get_vocab()
    assert set(restored.additional_special_tokens) == set(original.additional_special_tokens)


@pytest.mark.parametrize("start_bin", [None, 3])
def test_signed_token_roundtrip_with_optional_seed_value(start_bin):
    tokenizer = _tokenizer(signed=True)
    prompt = ParsedPropertyPrompt(
        "logp", start_bin=start_bin, signed_delta_bin=8,
        raw_direction="up", raw_delta=0.01, raw_seed_value=1.0, raw_target_value=1.01,
    )
    text = build_property_prompt_text([prompt])
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.unk_token_id not in ids
    parsed = parse_property_prompt_from_ids(tokenizer, ids)
    assert parsed == [ParsedPropertyPrompt("logp", start_bin=start_bin, signed_delta_bin=8)]
    assert "0.01" not in text and "<pdir_" not in text
    summary = format_property_prompt_summary(parsed, delta_binning=BINNING)
    assert "signed-Δ=08(0.000,0.500)" in summary


def test_unsigned_bounded_prompt_roundtrip_is_unchanged():
    tokenizer = _tokenizer(signed=True)
    prompt = ParsedPropertyPrompt("qed", start_bin=4, direction="down", delta_bin=2, delta_mode="atmost")
    text = build_property_prompt_text([prompt])
    assert text == "<pc><p_qed><p0_qed_04><pdir_down><pdmax_qed_02>"
    assert parse_property_prompt_from_ids(tokenizer, tokenizer.encode(text, add_special_tokens=False)) == [prompt]


def test_signed_and_unsigned_magnitudes_cannot_conflict():
    with pytest.raises(ValueError, match="unsigned or bounded"):
        build_property_prompt_text([ParsedPropertyPrompt("logp", signed_delta_bin=8, delta_bin=0)])


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_signed_token_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive int"):
        SmilesTokenizer2Fast(
            add_property_prompt_tokens=True,
            property_prompt_tokens={
                "property_names": ["logp"], "num_value_bins": 32,
                "num_delta_bins": 16, "num_signed_delta_bins": count,
            },
        )
