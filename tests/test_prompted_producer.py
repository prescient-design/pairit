"""Exercise the prompted producer with tiny real tables and tokenization."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from pairit.callbacks import MoleculeValidationCallback
from pairit.callbacks.molecule_validation.base_metrics import SampleResult
from pairit.callbacks.molecule_validation.generation import MoleculeGenerator
from pairit.callbacks.molecule_validation.smiles_utils import SmilesUtils
from pairit.data._molecule_improvement_datamodule import (
    MoleculeImprovementDataset,
    MoleculeImprovementLightningDataModule,
)
from pairit.eval.prompted import evaluate_prompted_generations
from pairit.prompting.metadata import generation_pair_seed, pair_identifier
from pairit.prompting.property_prompting import parse_property_prompt_from_ids
from pairit.tokenization import SmilesTokenizer2Fast
from pairit.transforms import TokenizerTransform


@pytest.fixture
def dataset_factory(tmp_path):
    pd.DataFrame({
        "smiles": ["CC", "CCO", "CCN"],
        "qed": [0.50, 0.51, 0.80],
        "logp": [1.0, 1.3, 2.0],
        "Class": [0, 1, 0],
    }).to_parquet(tmp_path / "molecules.parquet", index=False)
    pd.DataFrame({
        "molecule_a_idx": [0, 0, 1],
        "molecule_b_idx": [1, 2, 0],
    }).to_parquet(tmp_path / "pairs.parquet", index=False)

    def make(names=("qed", "logp"), *, randomized=False, mapped_target=None, dropout=False, export=True):
        tokenizer = SmilesTokenizer2Fast(
            add_property_prompt_tokens=True,
            property_prompt_tokens={
                "property_names": list(names), "num_value_bins": 32,
                "num_delta_bins": 16, "num_signed_delta_bins": 16,
            },
        )
        prompting = {
            "properties": {
                "enabled": True,
                "names": list(names),
                "selection": {"mode": "all", "seed": 0},
                "include": {
                    "type": True, "start_value": "off", "direction": False, "magnitude": "signed_bin",
                    "property_dropout": {"enabled": dropout, "rate": 0.5},
                },
                "randomize_tokens": randomized,
                "export_metadata": export,
                "eval_override": {"enabled": not dropout},
                "source": {
                    "prefer_columns": True, "compute_if_missing": False,
                    "column_map": {"target": mapped_target},
                },
                "value_binning": {"mode": "fixed", "default": {"num_bins": 32, "range": [0, 10]}},
                "delta_binning": {
                    "mode": "fixed", "default": {"num_bins": 16, "range": [0, 1]},
                    "per_property": {
                        "qed": {"range": [0, 1]}, "logp": {"range": [0, 8]},
                        # Deliberately distinct: an unmapped proxy must inherit its descriptor's range.
                        "target": {"range": [0, 1.05 if mapped_target else 0.001]},
                    },
                },
            },
            "distance": {"enabled": False},
        }
        return MoleculeImprovementDataset(
            root=str(tmp_path), utility_key=None, epsilon=0, delta=None,
            keep_original_ordering=True, use_cache=False, cache_dir=str(tmp_path / "cache"),
            transform_fn=TokenizerTransform(tokenizer, padding="max_length", truncation=True, max_length=96),
            prompting=prompting,
        )

    return make


def _metadata(item):
    return json.loads(item["prompt_metadata"])


def _input_bins(dataset, item):
    return [
        p.signed_delta_bin
        for p in parse_property_prompt_from_ids(dataset.transform.tokenizer, item["input_ids"].reshape(-1).tolist())
    ]


def test_randomized_prompts_are_pair_specific_and_keep_collated_truth(dataset_factory):
    prompted = dataset_factory()
    randomized = dataset_factory(randomized=True)
    items = [randomized[i] for i in range(len(randomized))]
    assert _input_bins(randomized, items[0]) != _input_bins(randomized, items[1])
    batch = default_collate(items)
    assert len(batch["prompt_metadata"]) == 3
    for i, item in enumerate(items):
        actual = _metadata(item)
        expected = _metadata(prompted[i])
        assert actual == json.loads(batch["prompt_metadata"][i])
        assert actual["prompt_input_signed_delta_bins"] == _input_bins(randomized, item)
        assert _input_bins(randomized, randomized[i]) == _input_bins(randomized, item)
        for key in expected:
            if key.startswith("prompt_target_"):
                assert actual[key] == expected[key]
    first = _metadata(prompted[0])
    assert first["prompt_target_raw_deltas"][0] == pytest.approx(0.01)
    assert first["prompt_target_raw_direction_signs"][0] == 1
    assert first["prompt_target_prompts"][0]["signed_delta_bin"] == 8


def test_reverse_pairs_are_preserved_in_dataset_and_metadata(dataset_factory):
    dataset = dataset_factory()
    assert [(row[0], row[1]) for row in dataset.training_pairs] == [("CC", "CCO"), ("CC", "CCN"), ("CCO", "CC")]
    forward, reverse = _metadata(dataset[0]), _metadata(dataset[2])
    assert forward["seed_smiles"] == reverse["target_smiles"]
    assert forward["target_smiles"] == reverse["seed_smiles"]
    assert reverse["prompt_target_raw_deltas"] == pytest.approx([-d for d in forward["prompt_target_raw_deltas"]])
    assert reverse["prompt_target_raw_direction_signs"] == [-1, -1]


def test_unmapped_target_mirrors_values_and_limits_without_randomization_drift(dataset_factory):
    prompted = dataset_factory(names=("qed", "logp", "target"))
    randomized = dataset_factory(names=("qed", "logp", "target"), randomized=True)
    for i in range(len(prompted)):
        truth = _metadata(prompted[i])["prompt_target_prompts"]
        target = truth[-1]
        mirrors = [p for p in truth[:-1] if p["raw_delta"] == target["raw_delta"]]
        assert len(mirrors) == 1
        mirror = mirrors[0]
        for field in ("signed_delta_bin", "raw_seed_value", "raw_target_value", "raw_direction"):
            assert target[field] == mirror[field]
        # Its own deliberately narrow target range would clip these nonzero deltas.
        assert target["signed_delta_bin"] not in (0, 15)
        assert _metadata(randomized[i])["prompt_target_prompts"] == truth


def test_mapped_target_uses_measured_assay_column(dataset_factory):
    dataset = dataset_factory(names=("qed", "logp", "target"), mapped_target="Class")
    target = _metadata(dataset[0])["prompt_target_prompts"][-1]
    assert target["raw_seed_value"] == 0
    assert target["raw_target_value"] == 1
    assert target["raw_delta"] == 1
    assert target["signed_delta_bin"] == 15
    assert _metadata(dataset[1])["prompt_target_prompts"][-1]["raw_delta"] == 0


def test_dropout_can_fill_property_cache_after_a_different_subset(dataset_factory):
    dataset = dataset_factory(dropout=True, export=False)
    seen = set()
    initial_subset = None
    for epoch in range(64):
        dataset.set_epoch(epoch)
        prompts = dataset._build_property_prompts(smiles1="CC", smiles2="CCO")
        assert prompts == dataset._build_property_prompts(smiles1="CC", smiles2="CCO")
        names = {p.property_name for p in prompts}
        if initial_subset is None and len(names) == 1:
            initial_subset = names
            # Replay this one-property start with an empty cache, as a fresh worker would.
            dataset._prop_cache.clear()
            prompts = dataset._build_property_prompts(smiles1="CC", smiles2="CCO")
            seen.clear()
        if initial_subset is not None:
            seen.update(names)
            for prompt in prompts:
                assert prompt.raw_delta == pytest.approx({"qed": 0.01, "logp": 0.3}[prompt.property_name])
            if seen == {"qed", "logp"}:
                break
    assert initial_subset is not None
    assert seen == {"qed", "logp"}
    assert set(dataset._prop_cache["CC"]) == seen
    assert dataset._get_properties_for_smiles("CC", ["qed", "logp"]) == {"qed": 0.5, "logp": 1.0}


def test_dropout_epoch_reaches_persistent_spawn_worker(dataset_factory):
    dataset = dataset_factory(dropout=True, export=False)
    loader = DataLoader(
        dataset, batch_size=len(dataset), num_workers=1,
        multiprocessing_context="spawn", persistent_workers=True, timeout=60,
    )
    observed = []
    iterator = None
    try:
        for epoch in range(4):
            dataset.set_epoch(epoch)
            expected = default_collate([dataset[i] for i in range(len(dataset))])
            iterator = iter(loader)
            actual = next(iterator)
            assert torch.equal(actual["input_ids"], expected["input_ids"])
            observed.append(actual["input_ids"].clone())
        assert any(not torch.equal(observed[0], row) for row in observed[1:])
    finally:
        if iterator is not None:
            iterator._shutdown_workers()


def test_datamodule_limits_dropout_to_training_and_updates_subset_epoch(dataset_factory):
    prototype = dataset_factory(dropout=True, export=False)
    prototype._prompting["properties"]["include"]["property_dropout"]["rate"] = 1.0
    module = MoleculeImprovementLightningDataModule(
        root=str(prototype.root), train_pair_filename="pairs.parquet", val_pair_filename="pairs.parquet",
        test_pair_filename="pairs.parquet", transform_fn=prototype.transform,
        prompting=prototype._prompting, utility_key=None, epsilon=0, delta=None,
        keep_original_ordering=True, use_cache=False, cache_dir=str(prototype.root / "module_cache"),
        max_train_samples=2, num_workers=0,
    )
    module.setup("fit")
    module.set_epoch(9)
    assert module._train_dataset.dataset._current_epoch == 9
    assert not _input_bins(prototype, module._train_dataset[0])
    for dataset in (module._val_dataset, module._test_dataset):
        assert dataset._prop_property_dropout_enabled is True
        assert dataset._prop_property_dropout_rate == 1.0
        assert len(_input_bins(dataset, dataset[0])) == 2


class _SamplingModel:
    device = torch.device("cpu")

    def __init__(self):
        self.contexts = []

    def generate(self, input_ids, **kwargs):
        self.contexts.append(input_ids.clone())
        draws = torch.randint(10_000, 20_000, (len(input_ids), 3))
        return torch.cat((input_ids, draws), dim=1)


def _module(tokenizer, model=None):
    return SimpleNamespace(
        tokenizer=tokenizer, model=model or _SamplingModel(), _max_length=16, trainer=None, eval=lambda: None,
    )


def test_pair_local_sampling_survives_batch_order_and_restores_cpu_rng(dataset_factory):
    dataset = dataset_factory(randomized=True)
    batch = default_collate([dataset[0], dataset[1]])
    tokenizer = dataset.transform.tokenizer
    model = _SamplingModel()
    module = _module(tokenizer, model)
    seeds = [generation_pair_seed(0, pair_identifier("bace_Class", "CC", target)) for target in ("CCO", "CCN")]
    before = torch.random.get_rng_state().clone()

    def generate(indices):
        return MoleculeGenerator.batch_generate_conditional(
            module, batch["input_ids"][indices], batch["attention_mask"][indices],
            num_generations_per_seed=3, pair_seeds=[seeds[i] for i in indices],
        )

    together = generate([0, 1])
    separate = generate([0]) + generate([1])
    reversed_batch = generate([1, 0])
    assert all(torch.equal(a, b) for a, b in zip(together, separate))
    assert all(torch.equal(a, b) for a, b in zip(together, reversed_batch[3:] + reversed_batch[:3]))
    assert torch.equal(before, torch.random.get_rng_state())
    for contexts in model.contexts:
        for context in contexts:
            assert context[-1] == tokenizer.sep_token_id
            assert SmilesUtils.tokens_to_smiles(tokenizer, context) == ("CC", "")
    with pytest.raises(ValueError, match="One generation seed"):
        MoleculeGenerator.batch_generate_conditional(
            module, batch["input_ids"], batch["attention_mask"], pair_seeds=[seeds[0]],
        )


def test_empty_context_keeps_all_attempt_slots_and_restores_rng(dataset_factory):
    dataset = dataset_factory()
    item = dataset[0]
    model = _SamplingModel()
    before = torch.random.get_rng_state().clone()
    result = MoleculeGenerator.batch_generate_conditional(
        _module(dataset.transform.tokenizer, model), item["input_ids"], torch.zeros_like(item["attention_mask"]),
        num_generations_per_seed=4, pair_seeds=[13],
    )
    assert len(result) == 4 and all(row.numel() == 0 for row in result)
    assert not model.contexts
    assert torch.equal(before, torch.random.get_rng_state())


def test_callback_exports_invalid_empty_and_duplicate_attempts(dataset_factory, tmp_path, monkeypatch):
    dataset = dataset_factory()
    item = dataset[0]
    callback = MoleculeValidationCallback(
        num_unconditional_generations=0, num_conditional_generations=1, num_generations_per_seed=4,
        save_dir=str(tmp_path), max_samples_to_save=0, save_pdfs=False, generation_only=True,
        property_names=["qed"], export_pairit_csv=True, export_prompt_metadata=True,
        pairit_generation_seed=0, pairit_dataset_name="bace_Class",
    )
    metadata = callback._prompt_batch_metadata(default_collate([item]), 1)[0]
    module = _module(dataset.transform.tokenizer)
    metrics = Mock()
    metrics.process_sample.return_value = SampleResult()
    sequences = [
        torch.tensor(module.tokenizer.encode("CC<sep>CCO")),
        torch.tensor(module.tokenizer.encode("CC<sep>CCO")),
        torch.tensor(module.tokenizer.encode("CC<sep>C1")),
        torch.empty(0, dtype=torch.long),
    ]
    for index, sequence in enumerate(sequences):
        callback._process_conditional_generation_sample_post_generation(
            module, item["input_ids"], item["attention_mask"], sequence, 0, 0,
            generation_idx=index, metrics_computer=metrics, prompt_metadata=metadata,
        )
    saved = pd.DataFrame(callback._generated_samples)
    assert len(saved) == 4
    assert saved.generation_index.tolist() == [0, 1, 2, 3]
    assert saved.is_valid.tolist() == [True, True, False, False]
    callback._export_pairit_csv(saved, tmp_path / "conditional.parquet")
    exported = pd.read_csv(tmp_path / "conditional_pairit.csv", keep_default_na=False)
    requests = pd.read_csv(tmp_path / "conditional_pairit_requests.csv", keep_default_na=False)
    assert exported.generated_smiles.tolist() == ["CCO", "CCO", "C1", ""]
    assert exported.gen_idx.tolist() == [0, 1, 2, 3]
    assert exported.pair_id.nunique() == len(requests) == 1
    assert exported.seed_smiles.tolist() == ["CC"] * 4
    assert exported.target_smiles.tolist() == ["CCO"] * 4
    assert json.loads(requests.prompt_target_raw_direction_signs.iloc[0]) == [1, 1]
    result = evaluate_prompted_generations(
        exported, requests, training_smiles=["CC"], attempts_per_request=4,
    )
    assert len(result.per_generation) == 4
    assert result.summary["validity"] == {"numerator": 2, "denominator": 4, "rate": 0.5}
    assert result.summary["coverage"]["expected_requests"] == 1
    assert result.summary["coverage"]["expected_attempts"] == 4
    with pytest.raises(ValueError, match="Incomplete or duplicated generation slots"):
        callback._export_pairit_csv(saved.iloc[:-1], tmp_path / "incomplete.parquet")
    with pytest.raises(ValueError, match="Repeated input pair"):
        callback._prompt_batch_metadata(default_collate([item]), 1)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(ValueError, match="single-process"):
        callback.on_validation_epoch_start(SimpleNamespace(world_size=1), module)
    with pytest.raises(ValueError, match="single-process"):
        callback.evaluate(module, [])


def test_callback_caps_before_generation_and_reports_per_attempt_validity(dataset_factory, tmp_path, monkeypatch, capsys):
    dataset = dataset_factory()
    module = _module(dataset.transform.tokenizer)
    callback = MoleculeValidationCallback(
        num_unconditional_generations=0, num_conditional_generations=2, num_generations_per_seed=4,
        save_dir=str(tmp_path), max_samples_to_save=0, save_pdfs=False, generation_only=True,
        property_names=["qed"], export_pairit_csv=True, export_prompt_metadata=True,
        pairit_generation_seed=0, pairit_dataset_name="bace_Class",
    )
    monkeypatch.setattr(callback.conditional_metrics, "process_sample", lambda **kwargs: SampleResult())
    calls = []

    def generate(pl_module, input_ids, attention_mask, **kwargs):
        calls.append((input_ids.clone(), kwargs))
        assert len(input_ids) == 1
        assert kwargs["num_generations_per_seed"] == 4
        assert len(kwargs["pair_seeds"]) == 1
        return [
            torch.tensor(pl_module.tokenizer.encode("CC<sep>CCO")),
            torch.tensor(pl_module.tokenizer.encode("CC<sep>CCO")),
            torch.tensor(pl_module.tokenizer.encode("CC<sep>C1")),
            torch.empty(0, dtype=torch.long),
        ]

    monkeypatch.setattr(MoleculeGenerator, "batch_generate_conditional", generate)
    result = callback.evaluate(module, DataLoader(dataset, batch_size=2), max_evals=1)
    assert len(calls) == 1
    assert result["val_molecule_validity_conditional"] == 0.5
    assert result["num_conditional_samples"] == 1
    exports = list(tmp_path.glob("*_pairit.csv"))
    assert len(exports) == 1
    exported = pd.read_csv(exports[0], keep_default_na=False)
    assert len(exported) == 4
    assert exported.gen_idx.tolist() == [0, 1, 2, 3]
    assert exported.target_smiles.tolist() == ["CCO"] * 4
    assert "Failed to generate molecule grid" not in capsys.readouterr().out
