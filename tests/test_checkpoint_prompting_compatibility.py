"""Recover historical prompting token IDs without the former Lobster package."""
import importlib

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from tokenizers import AddedToken

from pairit.cmdline._tokenizer_inheritance import (
    compute_bin_widths_from_config,
    extract_checkpoint_tokenizer_info,
    prepare_tokenizer_inheritance,
    validate_bin_width_compatibility,
)
from pairit.tokenization import SmilesTokenizer2Fast
from pairit.transforms import TokenizerTransform


def _tokenizer():
    return SmilesTokenizer2Fast(
        add_property_prompt_tokens=True,
        property_prompt_tokens={
            "property_names": ["qed", "logp"],
            "num_value_bins": 4,
            "num_delta_bins": 16,
            "num_signed_delta_bins": 16,
        },
        add_distance_prompt_tokens=True,
        distance_prompt_tokens={"num_bins": 4},
    )


def _checkpoint(tokenizer):
    return {
        "hyper_parameters": {"transform_fn": TokenizerTransform(tokenizer)},
        "state_dict": {
            "model.model.embed_tokens.weight": torch.zeros(len(tokenizer), 2),
            "model.lm_head.weight": torch.zeros(len(tokenizer), 2),
        },
    }


def _config(properties=("qed", "logp")):
    return OmegaConf.create({"data": {"transform_fn": {"tokenizer": {
        "add_property_prompt_tokens": True,
        "property_prompt_tokens": {
            "property_names": list(properties), "num_value_bins": 4,
            "num_delta_bins": 16, "num_signed_delta_bins": 16,
        },
        "add_distance_prompt_tokens": True,
        "distance_prompt_tokens": {"num_bins": 4},
    }}}})


def test_all_backend_added_tokens_survive_lost_special_metadata(tmp_path):
    original = _tokenizer()
    original.add_special_tokens({"additional_special_tokens": ["<last_control>"]})
    original.add_tokens(["custom_normal_token"])
    assert "<psd_logp_08>" not in original.additional_special_tokens
    path = tmp_path / "old.ckpt"
    torch.save(_checkpoint(original), path)

    info = extract_checkpoint_tokenizer_info(str(path))
    assert info["success"], info["error"]
    assert info["added_tokens"] == sorted(original.get_added_vocab(), key=original.get_vocab().get)
    assert "<psd_logp_08>" in info["additional_special_tokens"]
    assert "custom_normal_token" not in info["additional_special_tokens"]
    assert info["signed_delta_bin_widths"] == {2}
    assert info["embedding_rows"] == {
        "model.model.embed_tokens.weight": len(original), "model.lm_head.weight": len(original),
    }

    # New property controls can append, but every saved ID must retain its row.
    cfg = _config(properties=("logp", "qed", "tpsa"))
    summary = prepare_tokenizer_inheritance(cfg, str(path), verbose=False)
    restored = SmilesTokenizer2Fast(**OmegaConf.to_container(cfg.data.transform_fn.tokenizer))
    assert summary["inherited"]
    assert all(restored.get_vocab()[tok] == idx for tok, idx in original.get_vocab().items())
    assert restored.convert_tokens_to_ids("<p_tpsa>") >= len(original)
    assert "custom_normal_token" not in restored.all_special_tokens
    prompt = "<pc><p_logp><psd_logp_08><dc><d_01>"
    assert restored.encode(prompt) == original.encode(prompt)
    assert "<psd_logp_08>" in restored.additional_special_tokens


@pytest.mark.parametrize("key", ["model.model.embed_tokens.weight", "model.lm_head.weight"])
def test_checkpoint_embedding_must_cover_every_saved_id(tmp_path, key):
    tokenizer = _tokenizer()
    checkpoint = _checkpoint(tokenizer)
    checkpoint["state_dict"][key] = torch.zeros(len(tokenizer) - 1, 2)
    path = tmp_path / "undersized.ckpt"
    torch.save(checkpoint, path)
    info = extract_checkpoint_tokenizer_info(str(path))
    assert not info["success"]
    assert key in info["error"] and "cannot represent" in info["error"]
    with pytest.raises(ValueError, match="unverified token IDs"):
        prepare_tokenizer_inheritance(_config(), str(path), verbose=False)


def test_padded_embeddings_can_preserve_all_saved_ids(tmp_path):
    tokenizer = _tokenizer()
    checkpoint = _checkpoint(tokenizer)
    checkpoint["state_dict"] = {
        key: torch.zeros(len(tokenizer) + 8, 2) for key in checkpoint["state_dict"]
    }
    path = tmp_path / "padded.ckpt"
    torch.save(checkpoint, path)
    assert extract_checkpoint_tokenizer_info(str(path))["success"]


def test_requested_checkpoint_may_not_silently_skip_inheritance(tmp_path):
    with pytest.raises(ValueError, match="Checkpoint not found"):
        prepare_tokenizer_inheritance(_config(), str(tmp_path / "missing.ckpt"), verbose=False)
    assert not prepare_tokenizer_inheritance(_config(), None, verbose=False)["inherited"]


def test_evaluation_datamodule_alias_preserves_inherited_ids(tmp_path):
    tokenizer = _tokenizer()
    path = tmp_path / "evaluate.ckpt"
    torch.save(_checkpoint(tokenizer), path)
    cfg = OmegaConf.create({"datamodule": _config().data})
    # This is the existing evaluate.py alias/copy-back sequence.
    with open_dict(cfg):
        cfg.data = cfg.datamodule
    prepare_tokenizer_inheritance(cfg, str(path), verbose=False)
    with open_dict(cfg):
        cfg.datamodule.transform_fn = cfg.data.transform_fn
        del cfg.data
    restored = SmilesTokenizer2Fast(**OmegaConf.to_container(cfg.datamodule.transform_fn.tokenizer))
    assert restored.get_vocab() == tokenizer.get_vocab()


def test_checkpoint_without_vocabulary_matrices_is_unsupported(tmp_path):
    checkpoint = _checkpoint(_tokenizer())
    del checkpoint["state_dict"]
    path = tmp_path / "metadata_only.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="missing CLM input/output vocabulary matrices"):
        prepare_tokenizer_inheritance(_config(), str(path), verbose=False)


def test_base_token_id_drift_is_rejected():
    vocab = _tokenizer().get_vocab()
    vocab["<cls>"], vocab["<eos>"] = vocab["<eos>"], vocab["<cls>"]
    with pytest.raises(ValueError, match="IDs changed for 2 tokens"):
        SmilesTokenizer2Fast(inherit_token_ids=vocab)


def test_noncontiguous_inherited_ids_are_rejected():
    vocab = _tokenizer().get_vocab()
    vocab["<cls>"] = len(vocab)
    with pytest.raises(ValueError, match="unique and contiguous"):
        SmilesTokenizer2Fast(inherit_token_ids=vocab)


def test_custom_added_token_matching_is_reported(tmp_path):
    tokenizer = _tokenizer()
    tokenizer.add_tokens([AddedToken("custom", lstrip=True)])
    path = tmp_path / "custom.ckpt"
    torch.save(_checkpoint(tokenizer), path)
    info = extract_checkpoint_tokenizer_info(str(path))
    assert not info["success"]
    assert "custom added-token matching" in info["error"]


def test_signed_bin_width_compatibility():
    widths = compute_bin_widths_from_config(_config())
    assert widths["signed_delta_width"] == 2
    with pytest.raises(ValueError, match="Signed delta bin width mismatch"):
        validate_bin_width_compatibility({"signed_delta_bin_widths": {3}}, widths)


@pytest.mark.parametrize("weights_only", [False, True])
def test_historical_lobster_class_paths_load_as_pairit(tmp_path, monkeypatch, weights_only):
    # Write genuine legacy GLOBAL references, then restore the present class
    # paths before loading. The compatibility module supplies only two aliases.
    tokenizer = _tokenizer()
    checkpoint = _checkpoint(tokenizer)
    path = tmp_path / "lobster.ckpt"
    with monkeypatch.context() as patch:
        patch.setattr(SmilesTokenizer2Fast, "__module__", "lobster.tokenization._smiles_tokenizer2")
        patch.setattr(TokenizerTransform, "__module__", "lobster.transforms._tokenizer_transform")
        torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=weights_only)
    transform = loaded["hyper_parameters"]["transform_fn"]
    assert type(transform) is TokenizerTransform
    assert type(transform.tokenizer) is SmilesTokenizer2Fast
    assert transform.tokenizer.get_vocab() == tokenizer.get_vocab()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lobster.unregistered_checkpoint_module")
