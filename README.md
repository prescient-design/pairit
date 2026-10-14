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
- [Assets](#assets)
- [Reproduce paper evaluation from shipped generations](#reproduce-paper-evaluation-from-shipped-generations)
- [Train a PairIT variant](#train-a-pairit-variant)
- [Generate from a trained checkpoint](#generate-from-a-trained-checkpoint)
- [Signed property prompting](#signed-property-prompting)
- [Unit tests](#unit-tests)
- [Evaluate your own generations](#evaluate-your-own-generations)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## Environment setup

Requires Python 3.11 or 3.12. Recommended installer: [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo> pairit
cd pairit
uv sync
```

If `uv sync` completes with fewer than the expected packages (a known caching
edge case on some networked filesystems), run it once more. The second call
finishes in seconds and produces a complete environment.

Optional extras:

```bash
uv sync --extra flash   # flash-attn for faster attention on supported GPUs
uv sync --extra trl     # TRL / accelerate for GRPO (PairIT-RL) training
```

Environment variables (default values shown; set only if you want to move
things off the repo tree):

```bash
export PAIRIT_ASSETS_DIR="$PWD/assets"                    # bundled data, oracles, checkpoints
export PAIRIT_RUNS="$PWD/runs"                            # training run outputs
export PAIRIT_MINIMOL_PYTHON=/path/to/minimol/venv/bin/python  # only needed for MiniMol scoring
```

## Assets

Datasets, oracles, generation caches, and trained checkpoints live under
`assets/`:

```
assets/
├── data/              # MoleculeNet real-pair splits (BACE/BBBP/HIV) + hERG + ZINC-250K
├── oracles/           # train-only RF classifiers + MiniMol heads
├── generations/       # per-method per-dataset generation files used in the paper tables
└── checkpoints/       # trained PairIT / PairIT-Beam / PairIT-RL checkpoints
```

The `assets/` tree is not committed to git (large binary payload). Instructions
for downloading it will be added here when the external hosting is finalized.
In the meantime, obtain the tree separately and point `PAIRIT_ASSETS_DIR` at
it, or place it alongside `src/` at the repo root.

## Reproduce paper evaluation from shipped generations

The fastest path to a published number. Uses the generation CSV/parquet
files under `assets/generations/` — no training required.

```python
from pathlib import Path
import pandas as pd
from pairit.eval import evaluate
from pairit.eval.cli_utils import build_eval_config

# Any file under assets/generations/ works; example: BACE PairIT generations
df = pd.read_parquet("assets/generations/pairit/moleculenet/bace/all_generations.parquet")

config = build_eval_config(score_oracle="minimol")  # or score_oracle=None for RF
result = evaluate(df, config=config)

print(result.average[["method", "dataset", "valid", "improved", "novel", "v_n_i"]])
```

`v_n_i` is the paper's aggregate VIN metric (Valid AND Improved AND Novel over
all generations).

The command-line equivalent:

```bash
pairit_eval --config-name=bace_seedsonly_evaluate \
    "model.ckpt_path='assets/checkpoints/pairit/moleculenet/bace/finetuned.ckpt'"
```

## Train a PairIT variant

Pick one model and train it. All variants use the same
`--config-name=train_molecule_improvement` entry point with different
`model=` / `experiment=` overlays.

### PairIT (base — matched-pair supervision only)

```bash
pairit_train \
  --config-name=train_molecule_improvement \
  data=zinc_molecule_improvement \
  model=clm2_smiles \
  ++model.model_name=CLM_68M \
  paths.root_dir='${oc.env:PAIRIT_RUNS,./runs}/pairit_zinc_pretrain'
```

For MoleculeNet fine-tuning after ZINC pretraining, swap the data config and
point at your pretrained checkpoint:

```bash
pairit_train \
  --config-name=train_molecule_improvement \
  data=bace_real_molecule_improvement \
  model=clm2_smiles \
  "model.ckpt_path='${PAIRIT_RUNS}/pairit_zinc_pretrain/last.ckpt'" \
  paths.root_dir='${PAIRIT_RUNS}/pairit_bace_finetune'
```

Save the resulting checkpoint under
`assets/checkpoints/pairit/moleculenet/<dataset>/finetuned.ckpt` if you want
it discoverable by the inference / eval commands below.

### PairIT-Beam (iterative re-seeding at inference)

PairIT-Beam is an inference-time variant of PairIT — no separate training
required. See the [Generate](#generate-from-a-trained-checkpoint) section.

### PairIT-RL (GRPO post-training)

```bash
uv sync --extra trl
pairit_train \
  --config-name=train_molecule_improvement \
  data=bace_real_molecule_improvement \
  model=clm2_smiles \
  "model.ckpt_path='<your PairIT finetune ckpt>'" \
  experiment=grpo_bace \
  paths.root_dir='${PAIRIT_RUNS}/pairit_rl_bace'
```

`experiment=grpo_bace` sets the reward, KL constraint, and similarity-gate
defaults used in the paper. Swap for `grpo_bbbp` / `grpo_hiv` / `grpo_herg`
as needed.

### PairIT with property prompting (multi-property)

```bash
pairit_train \
  --config-name=train_molecule_improvement \
  data=bace_real_molecule_improvement \
  model=clm2_smiles_conditional \
  experiment=bace_property_prompting_all_and_dist
```

## Generate from a trained checkpoint

Best-of-N (the paper's default; N=64):

```bash
pairit_best_of_n \
  "model.ckpt_path='assets/checkpoints/pairit/moleculenet/bace/finetuned.ckpt'" \
  data=bace_real_molecule_improvement \
  ++inference.n_generations=64 \
  ++inference.temperature=0.5 \
  ++inference.output_dir='./generations/pairit_bace'
```

PairIT-Beam (iterative re-seeding; use after training a PairIT base model):

```bash
pairit_beam_search \
  --config-name=beam_search_bace \
  "model.ckpt_path='assets/checkpoints/pairit/moleculenet/bace/finetuned.ckpt'" \
  ++inference.output_dir='./generations/pairit_beam_bace'
```

Both commands write per-generation CSVs / parquets in the canonical schema
consumed by `pairit_eval`.

## Signed property prompting

The explicit `property_prompting_signed` training recipe and
`prompted_pair_evaluate` evaluation recipe support signed magnitude tokens,
descriptor subsets, the generic `target` slot, and randomized prompt controls.
The training recipe enables configurable property dropout at rate 0.3.
Existing unsigned/start-value/direction recipes retain their defaults.

See the [recipe guide](src/pairit/hydra_config/prompting/README.md) for required
prepared data, saved bin limits, checkpoint arguments and output paths. The
[prompt format](src/pairit/prompting/README.md) documents token compatibility.
Evaluation retains every attempt, including invalid and empty outputs, with
an ordered-pair ID, original raw target values and actual input bins. Its
companion request file records inputs visited before generation; compare it
with an independently prepared cohort when checking complete dataset coverage.
Prompt export currently requires one dataset and one process per evaluation.

The [prompting evaluator](src/pairit/eval/README.md) reuses existing molecular
calculations and adds the raw-attempt, per-request reductions. It requires an
explicit expected request roster and training novelty reference; activity
evaluation additionally requires an explicit training-only oracle. It reports
missing-score counts and uses finite-only MSE. Existing general evaluation
and selection metrics are unchanged.

## Evaluate your own generations

Point `pairit_eval` at your generation output:

```python
from pairit.eval import evaluate
from pairit.eval.cli_utils import build_eval_config
import pandas as pd

df = pd.read_csv("./generations/pairit_bace/generations.csv")
config = build_eval_config(score_oracle="minimol")
result = evaluate(df, config=config)
print(result.average)
```

Metrics reported: `valid`, `improved`, `novel`, `v_n_i` (VIN), `sim_seed`,
`unique`, `qed`, `internal_diversity`, and distribution metrics (FCD,
SNN, fragment/scaffold cosine, coverage) if the `--distribution` flag is
passed to the CLI.

## Unit tests

Run the test selection CI runs, from the repository root:

```bash
pytest -vv -s tests/ -m "not slow" --disable-warnings
```

This excludes the slow training and evaluation smoke tests (which need the
`assets/` bundle) but still imports their modules. Collect the whole test
directory rather than individual files: single files can miss import
interactions, including ones that only surface in spawned DataLoader workers.
Shared test helpers use package-relative imports and must not prepend `tests/`
to `sys.path` — doing so makes `tests/pairit/` shadow the installed package.

## Repository layout

```
pairit/
├── src/pairit/          # core library
│   ├── model/           # PairITCLM2 (SMILES autoregressive transformer)
│   ├── data/            # matched-pair datamodules
│   ├── tokenization/    # SMILES tokenizer + regex pre-tokenizer
│   ├── callbacks/       # training-time metrics + validation
│   ├── inference/       # best-of-N, beam-search, refilter
│   ├── rl/              # GRPO training loop
│   ├── prompting/       # property prompting for multi-property control
│   ├── eval/            # metrics library (VIN, distribution metrics, oracles)
│   └── hydra_config/    # all Hydra configs (data/, model/, experiment/, ...)
├── pair_gen/            # pair construction + molecular distance utilities
├── scripts/             # dataset preprocessing + pair building
├── examples/            # generate a synthetic dataset for testing
├── tests/               # unit + smoke tests
└── assets/              # (not in git) data, oracles, generations, checkpoints
```

## Citation

If you use this code, please cite:

```bibtex
@article{pairit_2026,
  title  = {Property-Guided Generation of Small Molecules with Matched Pairs},
  year   = {2026},
  note   = {Add venue + full author list on acceptance.}
}
```

## License

Apache License 2.0 (see [LICENSE](LICENSE)).
