# Property prompting recipes

The `signed_*` profiles add the signed-delta grammar to the existing training
and evaluation entry points. They emit `<p_PROPERTY><psd_PROPERTY_BIN>` with
16 bins, omit starting-value and separate direction tokens, and retain the
configured property order. Existing `zinc_all_props` and assay profiles keep
their legacy starting-value/direction/unsigned-magnitude grammar.

| Profile | `target` meaning | Delta ranges |
| --- | --- | --- |
| `signed_zinc` | A sampled descriptor, including its bin limits | Existing ZINC profile |
| `signed_bace` | `Class` | Saved BACE train-pair limits |
| `signed_bbbp` | `p_np` | Saved BBBP train-pair limits |
| `signed_hiv` | `HIV_active` | Saved HIV train-pair limits |

Standalone evaluation defaults to `signed_zinc_eval`, which inherits the
ZINC grammar and limits but selects only the nine descriptors. The generic
training proxy has no single descriptor identity for scoring, so it is
excluded from this evaluation profile. Assay evaluation profiles retain
all ten properties and score the mapped `target` with the explicit oracle.

The assay profiles use six continuous limits from the September prepared
forward training pairs: the linear 99th percentile of absolute deltas,
serialized to 12 significant digits. Integer descriptor maxima are 8 and
the activity maximum is 1.05. Their source is the matching dataset's
`streamlined_table2_data_v2/<dataset>/binning.env`, used by producer revision
`e6de07e441349d3fabf840505cff3acd670992a2`. The inherited ZINC limits have a
different provenance; they are not claimed to be train-only p99 estimates.
Both training and evaluation load these same profile files to avoid drifting
bin definitions. The `sa: sa_rdkit_score` mapping inherited from the ZINC
profile deliberately forces RDKit SA scoring when that column is absent.

Property dropout is enabled at rate 0.3 in the new training profiles and is
configurable under `properties.include.property_dropout`. This preserves the
enabled-dropout experiment family; it is not an exact replay of the twenty
audited September configs, which disabled dropout. Standalone evaluation
sets `eval_override.enabled=true` with null numerical overrides: it keeps
the pair-derived prompt and suppresses training dropout. The randomized arm
changes input bins while retaining the original pair's raw scoring targets.
Dropout epoch updates reach persistent training workers through a shared
epoch counter. Validation and test datasets suppress training augmentation
without changing the configured dropout setting.

## Prepared inputs and training

These are prompting recipes, not a complete reconstruction of training data,
checkpoint selection, or every historical experiment. They expect existing,
split-isolated molecule and pair tables. For the assay profiles, use the
prepared signed bidirectional pairs and their matching molecule tables; the
overlay preserves their ordering and disables utility/delta filtering. It
does not manufacture reverse pairs or recompute bin limits. Preserve the
checkpoint's property vocabulary and architecture when fine-tuning.

For example, with `DATASET=bace`, an existing prepared dataset at `DATA_ROOT`,
an absolute output directory at `RUN_ROOT`, and a compatible checkpoint at
`CHECKPOINT`, run the following inside a compute job:

```bash
pairit_train --config-name=train_molecule_improvement \
  experiment=property_prompting_signed \
  data="${DATASET}_real_molecule_improvement" \
  prompting="signed_${DATASET}" \
  data.root="$DATA_ROOT" \
  data.molecule_filename="train/molecules_${DATASET}.parquet" \
  data.val_molecule_filename="valid/molecules_${DATASET}.parquet" \
  data.test_molecule_filename="test/molecules_${DATASET}.parquet" \
  data.train_pair_filename="train/pairs_${DATASET}_signed_bidirectional.parquet" \
  data.val_pair_filename="valid/pairs_${DATASET}_signed_bidirectional.parquet" \
  data.test_pair_filename="test/pairs_${DATASET}_signed_bidirectional.parquet" \
  "model.ckpt_path='$CHECKPOINT'" paths.root_dir="$RUN_ROOT"
```

The overlay's architecture is `CLM_24M`; set `model.model_name=CLM_68M` for
the compatible 68M family. Training schedule and checkpoint selection remain
explicit experiment choices. For ZINC pretraining use
`data=zinc_molecule_improvement prompting=signed_zinc`, no checkpoint, and
the actual ZINC split filenames. Data and output roots are required rather
than borrowed from an unrelated experiment.
The shown CLI path overrides suffice; alternatively the data root can come
from `PAIRIT_PROMPT_DATA_ROOT` and the training run root from
`PAIRIT_PROMPT_OUTPUT_ROOT`. Neither environment variable has a fallback.
Hydra's own logs and config snapshots also go under the chosen output root.

Select a subset using, for example,
`data.prompting.properties.names='[qed,logp]'` or `'[qed,target]'`; the
tokenizer's property list follows it. Use
`data.prompting.properties.randomize_tokens=true` for randomized input.
Changing the list for an existing checkpoint must retain its inherited
vocabulary; the selected input list is not permission to renumber tokens.

## Exporting all attempts

Use the same prepared split files and profile as training. `OUTPUT_ROOT`
must point to an absolute data/output location; `CHECKPOINT` is required.

```bash
pairit_eval --config-name=prompted_pair_evaluate \
  dataset="$DATASET" \
  data@datamodule="${DATASET}_real_molecule_improvement" \
  prompting@datamodule.prompting="signed_${DATASET}" \
  datamodule.root="$DATA_ROOT" \
  datamodule.molecule_filename="train/molecules_${DATASET}.parquet" \
  datamodule.val_molecule_filename="valid/molecules_${DATASET}.parquet" \
  datamodule.test_molecule_filename="test/molecules_${DATASET}.parquet" \
  datamodule.train_pair_filename="train/pairs_${DATASET}_signed_bidirectional.parquet" \
  datamodule.val_pair_filename="valid/pairs_${DATASET}_signed_bidirectional.parquet" \
  datamodule.test_pair_filename="test/pairs_${DATASET}_signed_bidirectional.parquet" \
  "model.ckpt_path='$CHECKPOINT'" output_dir="$OUTPUT_ROOT"
```

The evaluation recipe exports 64 attempts per ordered pair at temperature
0.5, with pair-local generation seed 0, immutable target metadata and no
metric subsampling. It keeps invalid attempts and repeats in the exported
population. It does not automatically run the general evaluator's
deduplicated/top-k reductions. Prompting scoring requires the explicit
fine-tune training reference and, for activity, the corresponding train-only
oracle. Use `datamodule.prompting.properties.randomize_tokens=true` for the
randomized-input comparison and override the property list under
`datamodule.prompting.properties.names` for the matching subset model.
`max_samples_to_save: 0` disables molecule-grid rendering in this recipe;
metadata export bypasses sampling and still writes every attempt.
The required `dataset` value is also the pair-identity/RNG namespace. For
replay comparisons use the original producer's exact value, for example
`bace_Class`, rather than an abbreviated name.

Legacy single-property sweeps remain supported by the existing legacy
profiles and explicit bin/direction overrides. Their midpoint-based MSE
and historical missing-value behavior are not the pair-delta scoring
definition of this signed-pair recipe.
