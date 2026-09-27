"""Training and evaluation configurations for smoke tests.

This is the single source of truth for experiment configurations used by both
the smoke tests (``test_training_smoke.py``, ``test_evaluation_smoke.py``) and
documented in ``README.md``.

To change an experiment configuration, edit it here.  The smoke tests and docs
will pick up the change automatically.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

# Set these to your local copies before running the slow smoke tests. An unset
# variable leaves a "<unset:NAME>" placeholder in every derived path, so the
# "Required data file not found" failure names the variable to set.
ZINC_DATA_ROOT = os.environ.get("PAIRIT_SMOKE_ZINC_DATA_ROOT", "<unset:PAIRIT_SMOKE_ZINC_DATA_ROOT>")
MOLECULENET_ROOT = os.environ.get("PAIRIT_SMOKE_MOLECULENET_ROOT", "<unset:PAIRIT_SMOKE_MOLECULENET_ROOT>")
ZINC_CKPT_DIR = os.environ.get("PAIRIT_SMOKE_ZINC_CKPT_DIR", "<unset:PAIRIT_SMOKE_ZINC_CKPT_DIR>")

# ---------------------------------------------------------------------------
# Training configurations
# ---------------------------------------------------------------------------

# Each config is a dict with:
#   id:            Short identifier used in test names and parametrize IDs
#   description:   Human-readable description
#   required_data: List of paths that must exist for the test to run
#   hydra_args:    List of Hydra CLI arguments (paths.root_dir and
#                  trainer.devices are injected by the test harness)

TRAINING_CONFIGS: list[dict] = [
    {
        "id": "zinc_baseline",
        "description": "ZINC baseline: matched-pair CLM without property prompting",
        "required_data": [f"{ZINC_DATA_ROOT}/pairs_train.parquet"],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "data=zinc_molecule_improvement",
            "++data.epsilon=1",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
        ],
    },
    {
        "id": "zinc_logp_prompting",
        "description": "ZINC with single-property (logp) prompting",
        "required_data": [f"{ZINC_DATA_ROOT}/pairs_train.parquet"],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "experiment=zinc_property_prompting_all_props",
            "data=zinc_molecule_improvement",
            "++data.epsilon=1",
            "++data.delta=null",
            "++data.prompting.properties.names=[logp]",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
        ],
    },
    {
        "id": "zinc_all_props_prompting",
        "description": "ZINC with all-properties prompting",
        "required_data": [f"{ZINC_DATA_ROOT}/pairs_train.parquet"],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "experiment=zinc_property_prompting_all_props",
            "data=zinc_molecule_improvement",
            "++data.epsilon=1",
            "++data.delta=null",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
        ],
    },
    {
        "id": "bace_real",
        "description": "BACE real-pair fine-tuning (random init, no prompting)",
        "required_data": [f"{MOLECULENET_ROOT}/bace/train/pairs_bace_real.parquet"],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "data=bace_real_molecule_improvement",
            "++callbacks.molecule_validation.visualization_property_name=Class",
            "++callbacks.molecule_validation.property_names=[Class]",
            "++data.prompting.distance.enabled=false",
            "++data.epsilon=0.5",
            "++data.prompting.properties.enabled=false",
            "++data.transform_fn.tokenizer.add_property_prompt_tokens=false",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
        ],
    },
    {
        "id": "bbbp_real",
        "description": "BBBP real-pair fine-tuning (random init, no prompting)",
        "required_data": [f"{MOLECULENET_ROOT}/bbbp/train/pairs_bbbp_real.parquet"],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "data=bbbp_real_molecule_improvement",
            "++callbacks.molecule_validation.visualization_property_name=p_np",
            "++callbacks.molecule_validation.property_names=[p_np]",
            "++data.prompting.distance.enabled=false",
            "++data.epsilon=0.1",
            "++data.prompting.properties.enabled=false",
            "++data.transform_fn.tokenizer.add_property_prompt_tokens=false",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
        ],
    },
    {
        "id": "hiv_real",
        "description": "HIV real-pair fine-tuning (random init, no prompting)",
        "required_data": [f"{MOLECULENET_ROOT}/hiv/train/pairs_hiv_real.parquet"],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "data=hiv_real_molecule_improvement",
            "++callbacks.molecule_validation.visualization_property_name=HIV_active",
            "++callbacks.molecule_validation.property_names=[HIV_active]",
            "++data.prompting.distance.enabled=false",
            "++data.epsilon=0.1",
            "++data.prompting.properties.enabled=false",
            "++data.transform_fn.tokenizer.add_property_prompt_tokens=false",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
        ],
    },
    {
        "id": "mixed_zinc_bace",
        "description": "Mixed ZINC+BACE training with dataset identity tokens",
        "required_data": [
            f"{ZINC_DATA_ROOT}/pairs_train.parquet",
            f"{MOLECULENET_ROOT}/bace/train/pairs_bace_real.parquet",
        ],
        "hydra_args": [
            "--config-name=train_molecule_improvement",
            "data=mixed_zinc_bace",
            "++data.prompting.properties.enabled=false",
            "++data.prompting.distance.enabled=false",
            "++model.max_length=300",
            '++model.model_name="CLM_68M"',
            # Cap samples for speed
            "++data.train_max_samples.zinc=500",
            "++data.train_max_samples.bace=500",
            "++data.val_max_samples.zinc=100",
            "++data.val_max_samples.bace=100",
        ],
    },
]

# ---------------------------------------------------------------------------
# Evaluation configurations
# ---------------------------------------------------------------------------

EVAL_CONFIGS: list[dict] = [
    {
        "id": "zinc_eval_no_prompting",
        "description": "Evaluate ZINC checkpoint without prompting on test split",
        "required_data": [f"{ZINC_DATA_ROOT}/pairs_test.parquet"],
        "requires_checkpoint": ZINC_CKPT_DIR,
        "hydra_args": [
            "--config-name=zinc_pair_evaluate",
            "model.run_name=smoke_test",
            "model.extra_name=no_prompt",
            "split=test",
            "++datamodule.test_pair_filename=pairs_test.parquet",
            "++datamodule.epsilon=null",
            "++datamodule.prompting.properties.enabled=false",
            "++datamodule.transform_fn.tokenizer.add_property_prompt_tokens=false",
            "++callbacks.0.num_generations_per_seed=1",
            "++callbacks.0.temperature=0.95",
        ],
    },
]
