# PairIT: Property-Guided Generation of Small Molecules with Matched Pairs

Reference implementation for **PairIT**, a matched-pair SMILES autoregressive
transformer for seeded molecular optimization on small, experimentally
labeled datasets.

Given a seed molecule, PairIT generates local, property-improving edits by
learning from matched pairs of similar molecules with differing chemical
properties. This repository contains training, inference, and evaluation
code for the paper *Property-Guided Generation of Small Molecules with
Matched Pairs*.

## Contents

- [Environment setup](#environment-setup)
- [Data](#data)
- [Pretrain on ZINC](#pretrain-on-zinc)
- [Fine-tune on MoleculeNet](#fine-tune-on-moleculenet)
- [Generate: PairIT and PairIT-Beam](#generate-pairit-and-pairit-beam)
- [Evaluate](#evaluate)
- [PairIT-RL (GRPO)](#pairit-rl-grpo)
- [Property prompting](#property-prompting)
- [Unit tests](#unit-tests)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## Environment setup

Requires Python 3.11 or 3.12. Recommended installer: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/prescient-design/pairit.git
cd pairit
uv sync
```

If `uv sync` completes with fewer than the expected packages, run it once
more; the second call finishes in seconds and produces a complete
environment.

Optional extras:

```bash
uv sync --extra flash   # flash-attn for faster attention on supported GPUs
uv sync --extra trl     # TRL / accelerate for GRPO (PairIT-RL) training
uv pip install mmpdb    # only needed to rebuild MoleculeNet pairs yourself
```

Environment variables (defaults shown):

```bash
export PAIRIT_ASSETS_DIR="$PWD/assets"   # data, oracles, checkpoints
export PAIRIT_RUNS="$PWD/runs"           # training run outputs
export PAIRIT_OUTPUT_ROOT="$PWD/outputs" # generation outputs
export PAIRIT_MINIMOL_PYTHON=/path/to/minimol/venv/bin/python  # MiniMol scoring only
```

## Data

No datasets, model weights, or generations are distributed with this
repository. Everything is rebuilt from public sources.

Both upstream datasets are published by other projects:

| Dataset | Source |
|---|---|
| BACE / BBBP / HIV, with splits | [Graph-DiT](https://github.com/liugangcode/Graph-DiT) |
| ZINC-250K, with validation index | [MOOD](https://github.com/SeulLee05/MOOD) |

Three steps, each writing into the layout the configs expect:

```bash
python scripts/download_data.py   --dest data/raw
python scripts/prepare_molecules.py --raw data/raw --out assets/data
python scripts/build_pairs.py     --molecules assets/data
```

This produces:

```
assets/data/
├── moleculenet/<dataset>/<split>/{molecules,pairs}.parquet
└── zinc/<split>/{molecules,pairs}.parquet
```

`prepare_molecules.py` canonicalises SMILES with RDKit, applies the
upstream splits, and computes the descriptor columns. `build_pairs.py`
builds matched molecular pairs with mmpdb for MoleculeNet and Morgan
fingerprint neighbours for ZINC, matching the paper. Pairs are purely
structural: the improvement direction and the `epsilon` margin are
applied at training time by the data module, so the stored tables
contain both improving and non-improving pairs.

**Fidelity.** Molecule tables reproduce the paper's exactly. BACE pairs
reproduce exactly, pair for pair (2546 train / 217 test). BBBP and HIV
come out 8–12% over the published counts (554 vs 512, 1128 vs 1009 on
train), most likely an mmpdb version difference; close enough to
retrain on, but not identical.

**Cost.** MoleculeNet is small and builds in seconds. ZINC pairing is an
all-pairs fingerprint comparison over ~225k molecules and produces tens
of millions of pairs — budget hours and plenty of disk, or pass
`--dataset bace` to work on MoleculeNet alone while getting set up.

## Pretrain on ZINC

```bash
pairit_train \
  --config-name=train_molecule_improvement \
  data=zinc_molecule_improvement \
  model=clm2_smiles \
  ++model.model_name=CLM_68M \
  logger=csv \
  paths.root_dir='${oc.env:PAIRIT_RUNS,./runs}/pairit_zinc_pretrain'
```

`logger=csv` avoids the Weights & Biases login prompt; drop it to use
W&B. `CLM_68M` is the paper's model size — without it you get the
config default of `CLM_150M`.

## Fine-tune on MoleculeNet

```bash
pairit_train \
  --config-name=train_molecule_improvement \
  data=bace_real_molecule_improvement \
  model=clm2_smiles \
  ++model.model_name=CLM_68M \
  logger=csv \
  "model.ckpt_path='$PAIRIT_RUNS/pairit_zinc_pretrain/last.ckpt'" \
  paths.root_dir="$PAIRIT_RUNS/pairit_bace_finetune"
```

Swap `bace` for `bbbp` or `hiv`. Checkpoint paths contain `=`, so keep
the inner quoting on `model.ckpt_path`.

## Generate: PairIT and PairIT-Beam

PairIT-Beam is an inference-time variant — the same checkpoint, a
different search procedure. Neither command needs separate training.

Best-of-N (the paper's default, N=64):

```bash
pairit_best_of_n \
  model.ckpt_path="$PAIRIT_RUNS/pairit_bace_finetune/last.ckpt" \
  data@datamodule=bace_real_molecule_improvement \
  eval_config=bace_real_pair_evaluate \
  ++best_of_n.n=64 \
  ++generation.temperature=0.5
```

Iterative beam search:

```bash
pairit_beam_search \
  model.ckpt_path="$PAIRIT_RUNS/pairit_bace_finetune/last.ckpt" \
  data@datamodule=bace_real_molecule_improvement \
  eval_config=bace_real_pair_evaluate \
  ++beam_search.beam_width=32 \
  ++beam_search.num_rounds=5
```

Note the override keys: the dataset is `data@datamodule=...` (not
`data=...`), and generation settings live under `best_of_n.*` /
`beam_search.*` and `generation.*`. Hydra's `++` silently *creates*
unknown keys, so a mistyped override is ignored rather than reported.

`eval_config` pulls the dataset's ranking property and reference files
from the matching eval config; without it both commands rank by logP,
which is the ZINC default rather than the MoleculeNet one.

Output goes to `$PAIRIT_OUTPUT_ROOT/<run>/<mode>/.../all_generations.parquet`.

## Evaluate

```python
import pandas as pd
from pairit.eval import evaluate
from pairit.eval.cli_utils import build_eval_config
from pairit.eval.raw_loaders import load_generations

# Sampler output uses the generator's own column names; load_generations
# maps them onto the seven columns evaluate() requires.
df = load_generations(
    "outputs/.../all_generations.parquet",
    fmt="parquet",
    method="pairit",
    dataset="bace",
)

result = evaluate(df, config=build_eval_config(score_oracle="minimol"))
print(result.average[["method", "dataset", "valid", "improved", "novel", "v_n_i"]])
```

`v_n_i` is the paper's aggregate VIN metric (valid AND improved AND
novel, over all generations). Other reported metrics: `sim_seed`,
`unique`, `qed`, `internal_diversity`.

Passing a raw generation file straight to `evaluate()` will fail — it
requires `method`, `dataset`, `seed_idx`, `seed_smiles`, `seed_label`,
`gen_idx` and `generated_smiles`, which is what `load_generations`
produces.

Scoring with `score_oracle="minimol"` needs `PAIRIT_MINIMOL_PYTHON` set;
use `score_oracle=None` to score with the bundled random-forest oracles
instead.

## PairIT-RL (GRPO)

GRPO post-training is a standalone script rather than a Hydra entry
point:

```bash
uv sync --extra trl

python -m pairit.rl.grpo_train \
  --ckpt-path "$PAIRIT_RUNS/pairit_bace_finetune/last.ckpt" \
  --dataset-name bace \
  --seed-molecules assets/data/moleculenet/bace/train/molecules.parquet \
  --output-dir "$PAIRIT_RUNS/pairit_rl_bace" \
  --run-name pairit_rl_bace
```

`--help` lists the reward and KL options (`--reward-mode`, `--beta`,
`--gated-similarity-threshold`, the `--w-*` reward weights).

There is no separate RL inference path: generate from an RL checkpoint
with `pairit_best_of_n` by pointing `model.ckpt_path` at it.

## Property prompting

Single- and multi-property prompting conditions generation on requested
property changes.

```bash
# Multi-property, unsigned
pairit_train \
  --config-name=train_molecule_improvement \
  data=bace_real_molecule_improvement \
  model=clm2_smiles_conditional \
  logger=csv \
  experiment=bace_property_prompting_all_and_dist

# Signed magnitude prompts
pairit_train \
  --config-name=train_molecule_improvement \
  data=bace_real_molecule_improvement \
  model=clm2_smiles_conditional \
  logger=csv \
  experiment=property_prompting_signed
```

Recipes live in `src/pairit/hydra_config/prompting/`; see its
[README](src/pairit/hydra_config/prompting/README.md) for the prompt
grammar and bin settings, and
[`src/pairit/prompting/README.md`](src/pairit/prompting/README.md) for
token compatibility with existing checkpoints.

## Unit tests

```bash
pytest -vv -s tests/ -m "not slow" --disable-warnings
```

This excludes the slow training and evaluation smoke tests, which need a
populated `assets/` tree. Collect the whole directory rather than
individual files: single files can miss import interactions, including
ones that only surface in spawned DataLoader workers.

## Repository layout

```
pairit/
├── src/pairit/          # core library
│   ├── model/           # PairITCLM2 (SMILES autoregressive transformer)
│   ├── data/            # matched-pair datamodules
│   ├── tokenization/    # SMILES tokenizer + regex pre-tokenizer
│   ├── callbacks/       # training-time metrics + validation
│   ├── inference/       # best-of-N, beam search, refilter
│   ├── rl/              # GRPO training
│   ├── prompting/       # property prompting
│   ├── eval/            # metrics (VIN, distribution metrics, oracles)
│   └── hydra_config/    # all Hydra configs
├── pair_gen/            # fingerprint pair construction utilities
├── scripts/             # data download, preparation, pair building
├── tests/               # unit + smoke tests
└── assets/              # (not in git) data, oracles, checkpoints
```

## Citation

```bibtex
@article{pairit_2026,
  title  = {Property-Guided Generation of Small Molecules with Matched Pairs},
  year   = {2026},
  note   = {Add venue + full author list on acceptance.}
}
```

## License

Genentech Non-Commercial Software License Version 1.0 (see [LICENSE](LICENSE)).
The license reuses most of the text of the Apache License, Version 2.0, with an
added restriction against commercial use.
