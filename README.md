# PairIT: Property-Guided Generation of Small Molecules with Matched Pairs

PairIT learns property-improving molecular edits from matched pairs of SMILES.
This repository contains the training, sampling, and evaluation code for the paper
*Property-Guided Generation of Small Molecules with Matched Pairs*.

**Code only.** Model checkpoints and trained RF/MiniMol predictors are not
redistributed. Train them locally using the commands below. Public dataset
sources are listed at the end of this README.

## Setup

Use Python 3.11 or 3.12 and a CUDA GPU for model training.

```bash
git clone https://github.com/prescient-design/pairit.git
cd pairit
uv sync --frozen --extra trl --extra data --extra eval --extra test
source .venv/bin/activate
export PAIRIT_ASSETS_DIR="$PWD/assets"
export PAIRIT_RUNS="$PWD/runs"
export PAIRIT_OUTPUT_ROOT="$PWD/outputs"
```

Keep the dependency lock: the GRPO code uses TRL's pre-tokenized input interface.
MiniMol runs in a separate environment described below.

## 1. Data preprocessing and pair creation

Download the source files and prepare MoleculeNet tables and pairs:

```bash
python scripts/download_data.py --dest data/raw
python scripts/prepare_molecules.py --raw data/raw --out assets/data \
  --dataset bace --dataset bbbp --dataset hiv
python scripts/build_pairs.py --molecules assets/data \
  --dataset bace --dataset bbbp --dataset hiv
```

The layout is `assets/data/moleculenet/<dataset>/<split>/{molecules,pairs}.parquet`.
Pairs involving multi-fragment molecules are excluded. Verified train/test pair
counts are BACE **2546/217**, BBBP **512/40**, and HIV **1009/165**.

ZINC preparation is available through the same scripts with `--dataset zinc`.
Its full pair rebuild is slow and has not been independently validated for this
release. Pretraining expects `assets/data/zinc/{train,valid}/` containing molecule
and pair tables, with the molecule property column named `logp`.

### Train the property predictors

RF predictors rank candidates and provide RL rewards. The independent MiniMol
predictors score the final selected molecules. Both are fit on training data only.

```bash
python scripts/train_rf_predictors.py
```

This writes `assets/oracles/moleculenet/evaluators/{bace,bbbp,hiv}_b.joblib`.

MiniMol uses its public pretrained encoder and fits a logistic-regression head.
Create a separate Python 3.11 environment; the following CPU installation matches
the scoring setup on Linux:

```bash
uv venv --python 3.11 .venv-minimol
uv pip install --python .venv-minimol/bin/python 'setuptools<81'
uv pip install --python .venv-minimol/bin/python \
  'torch==2.3.0+cpu' 'numpy<2' 'scipy<1.13' \
  'torch-scatter==2.1.2+pt23cpu' 'torch-sparse==0.6.18+pt23cpu' \
  'torch-cluster==1.6.3+pt23cpu' 'minimol==1.3.5' pyarrow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html
export PAIRIT_MINIMOL_PYTHON="$PWD/.venv-minimol/bin/python"
"$PAIRIT_MINIMOL_PYTHON" scripts/train_minimol_predictors.py
```

MiniMol downloads its encoder from its upstream provider. This repository contains
no encoder or predictor weights. Heads are saved under
`assets/oracles/moleculenet/minimol/{BACE,BBBP,HIV}/head.joblib`.
Both predictor scripts accept `--dataset`, `--molecules`, and `--out`.

## 2. Pretrain on ZINC

With prepared ZINC tables:

```bash
pairit_train --config-name=train_molecule_improvement \
  data=zinc_molecule_improvement model=clm2_smiles \
  model.model_name=CLM_68M model.max_length=300 seed=0 logger=csv \
  data.test_pair_filename=null \
  trainer.val_check_interval=500 trainer.limit_val_batches=10 \
  ++trainer.max_epochs=1 trainer.max_time=00:01:30:00 \
  callbacks.degradation_stopping.threshold=0.1 \
  paths.root_dir="$PAIRIT_RUNS/zinc_pretrain"
```

The 90-minute limit is a wall-clock budget and depends on hardware. The original
checkpoints record sequence length 300; Table 4 lists 1024. These commands use
300. Checkpoints are saved beneath a timestamped run directory. Set
`PRETRAIN_CKPT` to the complete path of the checkpoint selected by validation loss:

```bash
export PRETRAIN_CKPT=/absolute/path/to/the/pretraining/checkpoint.ckpt
```

## 3. Fine-tune on BACE, BBBP, and HIV

Run once per dataset:

```bash
DATASET=bace
LABEL=Class
pairit_train --config-name=train_molecule_improvement \
  data="${DATASET}_real_molecule_improvement" model=clm2_smiles \
  model.model_name=CLM_68M model.max_length=300 seed=0 logger=csv \
  "model.ckpt_path='$PRETRAIN_CKPT'" \
  ++trainer.max_epochs=200 callbacks.degradation_stopping.threshold=0.1 \
  callbacks.molecule_validation.visualization_property_name="$LABEL" \
  "callbacks.molecule_validation.property_names=[$LABEL]" \
  paths.root_dir="$PAIRIT_RUNS/${DATASET}_sft"
```

| Dataset | `DATASET` | `LABEL` |
|---|---|---|
| BACE | `bace` | `Class` |
| BBBP | `bbbp` | `p_np` |
| HIV | `hiv` | `HIV_active` |

The historical BBBP/HIV configurations use the test split for validation and
checkpoint selection. That behavior is preserved for reproduction; use separate
validation files for new experiments and report that change.

## 4. Sample with PairIT and PairIT-Beam

Set `SFT_CKPT` to the full saved checkpoint path for the selected dataset.
Checkpoints can be stored anywhere; no particular parent directory is required.
The `DATASET` and `LABEL` variables are defined in section 3.

```bash
export SFT_CKPT=/absolute/path/to/the/finetuned/checkpoint.ckpt

pairit_best_of_n "model.ckpt_path='$SFT_CKPT'" \
  "data@datamodule=${DATASET}_real_molecule_improvement" \
  "eval_config=${DATASET}_real_pair_evaluate" \
  ++datamodule.ignore_pairs=true ++datamodule.seeds_only_conditional=true \
  ++datamodule.filter_column="$LABEL" ++datamodule.filter_value=0 \
  best_of_n.n=64 generation.temperature=0.5 seed=42 \
  output.dir="$PAIRIT_OUTPUT_ROOT/$DATASET/pairit"

pairit_beam_search "model.ckpt_path='$SFT_CKPT'" \
  "data@datamodule=${DATASET}_real_molecule_improvement" \
  "eval_config=${DATASET}_real_pair_evaluate" \
  ++datamodule.ignore_pairs=true ++datamodule.seeds_only_conditional=true \
  ++datamodule.filter_column="$LABEL" ++datamodule.filter_value=0 \
  beam_search.beam_width=4 beam_search.expansions_per_candidate=8 \
  beam_search.num_rounds=3 generation.temperature=0.5 seed=42 \
  output.dir="$PAIRIT_OUTPUT_ROOT/$DATASET/pairit-beam"
```

Each command writes `all_generations.parquet`. Preserve these raw candidates for
evaluation; the sampler's selected summary is not the paper's evaluation input.

## 5. Evaluate generations

```bash
python scripts/evaluate_generations.py --dataset "$DATASET" \
  --pairit "$PAIRIT_OUTPUT_ROOT/$DATASET/pairit/all_generations.parquet" \
  --pairit-beam "$PAIRIT_OUTPUT_ROOT/$DATASET/pairit-beam/all_generations.parquet" \
  --output-dir "results/$DATASET"
```

This uses RF selection, within-seed canonical deduplication, top five of the
first 64 candidates, and independent MiniMol scoring. Missing candidates count
against the full test-seed budget. Use a new output directory for each evaluation.
RF and MiniMol predictors must be trained first, and `PAIRIT_MINIMOL_PYTHON` must
point to the MiniMol environment.

## 6. Train and sample with PairIT-RL

RL uses real training molecules. The recorded runs select `split=train`, which
excludes rows labeled `split=synthetic` in the original combined files. The real
seed pools contain 798 BACE, 517 BBBP, and 1,420 HIV molecules before the internal
90/10 RL train/evaluation split.

```bash
export RL_SEEDS="$PAIRIT_ASSETS_DIR/data/moleculenet/$DATASET/train/molecules.parquet"
SIM_WEIGHT=1  # BACE; use 5 for BBBP and HIV
export WANDB_MODE=offline
python -m pairit.rl.grpo_train --ckpt-path "$SFT_CKPT" \
  --dataset-name "$DATASET" --seed-molecules "$RL_SEEDS" --split-vals train \
  --reference-files "zinc_train:$PAIRIT_ASSETS_DIR/data/zinc/train/molecules.parquet" \
                    "${DATASET}_train:$RL_SEEDS" \
  --output-dir "$PAIRIT_RUNS/${DATASET}_rl" --run-name "${DATASET}_rl" \
  --reward-mode gated --gated-property-mode threshold_bonus \
  --gated-active-bonus 2 --gated-similarity-weight "$SIM_WEIGHT" \
  --gated-similarity-threshold 0 --temperature 1 --beta 0.01 \
  --learning-rate 5e-5 --per-device-batch-size 32 \
  --gradient-accumulation-steps 1 --num-generations 8 \
  --lora-r 16 --num-train-epochs 10 --seed 0

pairit_eval --config-name="${DATASET}_real_pair_evaluate" \
  "model.ckpt_path='$PAIRIT_RUNS/${DATASET}_rl/model_last.ckpt'" \
  ++datamodule.ignore_pairs=true ++datamodule.seeds_only_conditional=true \
  ++datamodule.filter_column="$LABEL" ++datamodule.filter_value=0 \
  ++datamodule.epsilon=null ++callbacks.0.num_generations_per_seed=64 \
  ++callbacks.0.temperature=0.5 ++callbacks.0.export_pairit_csv=true \
  ++callbacks.0.pairit_dataset_name="$DATASET" \
  ++callbacks.0.pairit_method_name=pairit-rl \
  output_dir="$PAIRIT_OUTPUT_ROOT/$DATASET/pairit-rl"
```

Training writes merged `model_last.ckpt` and `model_best.ckpt` files. The paper
used the last checkpoint. Evaluate the exported `*generated_molecules_pairit.csv`
with `scripts/evaluate_generations.py --pairit-rl <csv> --dataset <dataset>
--output-dir <new-directory>`.

## 7. Property prompting

Prompting code is included, but the signed-prompt data preparation and full paper
recipe have not been validated for this release. See the
[prompting documentation](src/pairit/hydra_config/prompting/README.md).

## Validation status

MoleculeNet pair sets match the reference artifacts in all available splits.
Evaluation of saved paper generations reproduces reported validity, with VIN
within 0.23 percentage points. Fresh ZINC-to-SFT training comparisons are ongoing;
exact numerical reproduction of newly trained models is not claimed.

```bash
pytest -q -m 'not slow'
```

## License

[Genentech Non-Commercial Software License, Version 1.0](LICENSE).
Upstream datasets and dependencies retain their own terms.

## Dataset sources

The downloader fetches files directly from the upstream repositories:

- [Graph-DiT](https://github.com/liugangcode/Graph-DiT): BACE, BBBP, HIV, and scaffold splits.
- [MOOD](https://github.com/SeulLee05/MOOD): ZINC-250K and validation indices.
