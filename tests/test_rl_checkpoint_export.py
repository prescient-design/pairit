"""Regression checks for exporting trained LoRA weights as plain checkpoints."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from pairit.rl import checkpoint_utils


def test_adapter_export_preserves_logits_without_temporary_base(tmp_path, monkeypatch):
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    base = LlamaForCausalLM(config).eval()
    original_base = copy.deepcopy(base)
    adapted = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )).eval()
    # Nonzero trained adapters: an untouched initialization could hide lost weights.
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    tokens = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        expected = adapted(tokens).logits
        assert not torch.allclose(expected, original_base(tokens).logits)

    adapter_dir = tmp_path / "adapter"
    adapted.save_pretrained(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text())
    adapter_config["base_model_name_or_path"] = "/missing/training-worker/base"
    config_path.write_text(json.dumps(adapter_config))
    monkeypatch.setattr(
        checkpoint_utils, "lightning_to_hf",
        lambda path: (copy.deepcopy(original_base), None),
    )
    merged = checkpoint_utils.load_hf_for_export(str(adapter_dir), "original.ckpt").eval()
    with torch.no_grad():
        torch.testing.assert_close(merged(tokens).logits, expected, atol=1e-6, rtol=1e-5)

    original_path = tmp_path / "original.ckpt"
    tokenizer = SimpleNamespace(padding_side="left")
    torch.save({"hyper_parameters": {"transform_fn": SimpleNamespace(tokenizer=tokenizer)},
                "state_dict": {}}, original_path)
    output = tmp_path / "export.ckpt"
    checkpoint_utils.hf_to_lightning(merged, tokenizer, str(original_path), str(output))
    exported = torch.load(output, weights_only=False)
    restored = LlamaForCausalLM(config).eval()
    restored.load_state_dict({k.removeprefix("model."): v for k, v in exported["state_dict"].items()}, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens).logits, expected, atol=1e-6, rtol=1e-5)
    assert exported["hyper_parameters"]["transform_fn"].tokenizer.padding_side == "right"


def test_unmerged_export_fails_before_writing(tmp_path):
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )), LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    output = tmp_path / "invalid.ckpt"
    with pytest.raises(ValueError, match="Merge adapters"):
        checkpoint_utils.hf_to_lightning(model, None, "unused.ckpt", str(output))
    assert not output.exists()


def test_full_hf_checkpoint_needs_no_base(tmp_path):
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))
    model.save_pretrained(tmp_path)
    restored = checkpoint_utils.load_hf_for_export(str(tmp_path), "unused.ckpt")
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])
