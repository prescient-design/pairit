"""Verify that train/eval wiring preserves the same prompting protocol."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "pairit" / "hydra_config"


def _compose(name, overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        return compose(config_name=name, overrides=overrides)


@pytest.mark.parametrize(
    "dataset,activity,logp_limit",
    [
        ("zinc", None, 9.88),
        ("bace", "Class", 2.468295),
        ("bbbp", "p_np", 2.642859),
        ("hiv", "HIV_active", 4.0425691),
    ],
)
def test_signed_train_eval_share_profiles(dataset, activity, logp_limit):
    data_profile = "zinc_molecule_improvement" if dataset == "zinc" else f"{dataset}_real_molecule_improvement"
    eval_profile = "signed_zinc_eval" if dataset == "zinc" else f"signed_{dataset}"
    train = _compose(
        "train_molecule_improvement",
        ["experiment=property_prompting_signed", f"prompting=signed_{dataset}", f"data={data_profile}"],
    )
    evaluation = _compose(
        "prompted_pair_evaluate",
        [f"prompting@datamodule.prompting={eval_profile}", f"data@datamodule={data_profile}"],
    )
    assert "data" not in evaluation
    training_properties = train.data.prompting.properties
    eval_properties = evaluation.datamodule.prompting.properties
    expected_eval_names = list(training_properties.names)
    if dataset == "zinc":
        expected_eval_names.remove("target")
    assert list(eval_properties.names) == expected_eval_names
    assert training_properties.delta_binning == eval_properties.delta_binning
    assert training_properties.source == eval_properties.source
    assert training_properties.source.column_map.target == activity
    assert training_properties.source.column_map.sa == "sa_rdkit_score"
    assert training_properties.delta_binning.per_property.logp.range[1] == logp_limit
    assert training_properties.include.start_value == "off"
    assert training_properties.include.direction is False
    assert training_properties.include.magnitude == "signed_bin"
    assert training_properties.include.property_dropout.enabled is True
    assert training_properties.include.property_dropout.rate == 0.3
    assert training_properties.eval_override.enabled is False
    assert eval_properties.include.property_dropout.enabled is True
    assert eval_properties.eval_override.enabled is True
    assert eval_properties.eval_override.start_bin is None
    assert eval_properties.eval_override.direction is None
    assert eval_properties.eval_override.delta_bin is None
    assert eval_properties.export_metadata is True
    assert train.data.keep_original_ordering is True
    assert evaluation.datamodule.keep_original_ordering is True
    assert train.data.utility_key is evaluation.datamodule.utility_key is None
    assert train.data.epsilon == evaluation.datamodule.epsilon == 0
    assert OmegaConf.is_missing(evaluation, "output_dir")
    assert OmegaConf.is_missing(evaluation, "dataset")
    assert OmegaConf.is_missing(evaluation.model, "ckpt_path")
    for tokenizer, properties in (
        (train.data.transform_fn.tokenizer, training_properties),
        (evaluation.datamodule.transform_fn.tokenizer, eval_properties),
    ):
        assert tokenizer.property_prompt_tokens.property_names == properties.names
        assert tokenizer.property_prompt_tokens.num_signed_delta_bins == 16


def test_recipe_requires_explicit_data_and_output_roots(monkeypatch):
    monkeypatch.delenv("PAIRIT_PROMPT_DATA_ROOT", raising=False)
    monkeypatch.delenv("PAIRIT_PROMPT_OUTPUT_ROOT", raising=False)
    train = _compose("train_molecule_improvement", ["experiment=property_prompting_signed"])
    evaluation = _compose("prompted_pair_evaluate", [])
    with pytest.raises(InterpolationResolutionError, match="PAIRIT_PROMPT_DATA_ROOT"):
        _ = train.data.root
    with pytest.raises(InterpolationResolutionError, match="PAIRIT_PROMPT_OUTPUT_ROOT"):
        _ = train.paths.root_dir
    with pytest.raises(InterpolationResolutionError, match="PAIRIT_PROMPT_DATA_ROOT"):
        _ = evaluation.datamodule.root


def test_subset_and_randomized_overrides_follow_tokenizer_and_callback():
    cfg = _compose(
        "prompted_pair_evaluate",
        [
            "prompting@datamodule.prompting=signed_bace",
            "datamodule.prompting.properties.names=[qed,target]",
            "datamodule.prompting.properties.randomize_tokens=true",
            "dataset=bace_Class",
        ],
    )
    assert list(cfg.datamodule.transform_fn.tokenizer.property_prompt_tokens.property_names) == ["qed", "target"]
    callback = cfg.callbacks[0]
    assert callback.prompting == cfg.datamodule.prompting
    assert callback.prompting.properties.randomize_tokens is True
    assert callback.num_generations_per_seed == 64
    assert callback.num_conditional_metric_samples is None
    assert callback.max_samples_to_save == 0
    assert callback.save_generated_samples is True
    assert callback.export_prompt_metadata is True
    assert callback.export_pairit_csv is True
    assert callback.pairit_dataset_name == "bace_Class"
    assert callback.pairit_generation_seed == 0
    assert callback.generation_only is True
    assert cfg.pairit_eval.enabled is False


def test_training_dropout_override_and_legacy_defaults_remain_available():
    train = _compose(
        "train_molecule_improvement",
        [
            "experiment=property_prompting_signed",
            "data.prompting.properties.include.property_dropout.rate=0.2",
        ],
    )
    assert train.data.prompting.properties.include.property_dropout.enabled is True
    assert train.data.prompting.properties.include.property_dropout.rate == 0.2
    legacy = _compose(
        "train_molecule_improvement", ["experiment=zinc_property_prompting_all_props"]
    )
    assert legacy.data.prompting.properties.include.start_value == "bin"
    assert legacy.data.prompting.properties.include.direction is True
    assert legacy.data.prompting.properties.include.magnitude == "bin"
