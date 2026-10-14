# Evaluation of original prompting requests

`prompted.evaluate_prompted_generations` evaluates every raw generation attempt
against the original matched-pair request. It uses the existing descriptor
registry, canonicalization, fingerprint similarity and oracle interfaces. The
ordinary evaluator's fixed task improvement, selection and deduplication rules
do not express this per-request question.

Supply one arm and dataset at a time, with three explicit inputs:

* A generation table with `pair_id`, zero-based `gen_idx`, `generated_smiles`,
  and the original request fields below. Keep invalid/empty generations and
  repeated molecules as rows.
* An independent expected request roster, one row per `pair_id`, prepared before
  generation. Do not derive this roster from successful outputs: that would hide
  requests for which every attempt was lost.
* The fine-tuning training molecules used as the novelty reference. No corpus is
  selected automatically. Invalid reference entries fail evaluation.

The callback's companion `_requests.csv` records input requests before generating
their attempts. It verifies all visited inputs, but cannot detect a dataloader
that omitted an input entirely. To claim full dataset coverage, verify this
roster against the intended evaluation split or supply an independent full roster.

Both request and generation tables carry `seed_smiles`, `target_smiles` and these
aligned list columns (JSON arrays in CSV, list columns in Parquet):

| Column | Meaning |
| --- | --- |
| `prompt_target_direction_property_names` | Original requested properties in fixed order |
| `prompt_target_raw_direction_signs` | Sign of each original, unbinned target-minus-seed delta |
| `prompt_target_raw_deltas` | Original target-minus-seed deltas |
| `prompt_target_raw_seed_values` | Original seed descriptor values or measured activity labels |
| `prompt_target_raw_target_values` | Original partner descriptor values or measured activity labels |

The evaluator requires identical repeated identity/truth, finite consistent raw
values, and exactly the supplied attempt budget for every expected pair. Two
requests sharing a seed remain separate. Random-token controls retain the
**original** request truth; their randomized input tokens are not scoring targets.
Missing requests, missing attempt rows and unexpected/duplicate indices fail
before population filtering.

The scientific reduction has one definition:

* A descriptor criterion applies only when its original raw delta is nonzero.
  Realized deltas recompute both seed and generation through the same descriptor
  registry. A hit requires a finite realized delta with the same sign. Unchanged outcomes
  fail nonzero requests. Signed token bin 08 does not determine eligibility.
* Per-property VIN is the fraction of applicable **raw attempts** that are valid,
  novel to the supplied training reference and direction hits. Repeated outputs
  count; no top-k selection or duplicate removal is performed.
* Combined VIN uses the union of applicable criteria as its denominator. Every
  applicable criterion must hit; a zero target adds no condition.
* Descriptor direction adherence uses valid, applicable attempts, without a
  novelty filter. Unscorable valid attempts cannot be hits and are counted
  explicitly. An entirely unscorable criterion has an unavailable rate. The
  nine-descriptor macro is the equal mean over all nine descriptors and is
  unavailable if any descriptor rate is unavailable or absent.
* Descriptor MSE uses valid, finite, nonzero-target attempts and the original raw
  delta, including repeated and memorized outputs. It remains defined for a
  single observation or constant values. Missing scores are excluded from the
  MSE denominator and counted explicitly; they never become zero error.
* With an explicit activity property and scoring oracle, all reported metrics
  use measured inactive-seed / active-target pairs. Excluded request counts are
  reported. Activity hits use a freshly evaluated probability strictly greater
  than 0.5. Stored diagnostic predictions are ignored. The caller must supply
  the training-only oracle for the correct assay; its training provenance cannot
  be inferred from a joblib file. Original exports name this property `target`.

Supported scoring targets are the nine molecular descriptors and an optional
binary activity property. ZINC's training-only generic `target` token mirrors a
descriptor and is not an activity label. Select the nine molecular descriptors
for ZINC evaluation; the reducer does not infer which descriptor a proxy token
represented. Every request in one evaluation must have the same ordered property
list. Fixed subset arms can report their per-property and Combined VIN, while the
nine-descriptor macro remains unavailable if any descriptor is absent.

All rates are fractions. JSON results include each numerator and denominator,
scored/unscorable counts, selected/excluded request counts, reference size, and
oracle provenance. `per_generation.csv` retains selected attempts with the
original truth, realized deltas, applicability and hit flags. Input file facts
are recorded by the CLI. Standard validity, conditional novelty and similarity
summaries come from these same attempts. Molecular calculations and oracle calls
are cached by canonical molecule; metric counts still count every raw attempt.

For example, with `$RUN` the directory holding a generation run and `$DATA`
the dataset root:

```bash
python -m pairit.eval.scripts.evaluate_prompted \
  --generations "$RUN/generations.csv" \
  --requests "$RUN/expected_requests.csv" \
  --training-reference "$DATA/train.parquet" \
  --training-smiles-column smiles \
  --attempts-per-request 64 \
  --activity-property target \
  --activity-oracle "$DATA/oracles/bace_train_only.joblib" \
  --output-dir "$RUN/prompted_scores"
```

Omit the two activity arguments for descriptor-only requests. The output
directory must be new. Historical forced-bin sweeps require their own explicit
request reconstruction; this CLI never infers those requests from directory
names or reproduces historical missing-value denominator behavior.

Focused validation: `python -m pytest tests/test_prompted_evaluation.py`. The
tests cover incomplete coverage, changed truth, repeated attempts, zero and
down requests, union denominators, constant/single-observation MSE, activity
threshold ties, missing descriptor scores, and CSV metadata round trips.
