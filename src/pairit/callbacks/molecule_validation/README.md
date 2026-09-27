# Molecule Validation Subpackage

Modular components extracted from `MoleculeValidationCallback` for computing evaluation metrics during training and standalone evaluation of molecule improvement models.

## Architecture

```
MoleculeValidationCallback (../molecule_validation_callback.py)
│
├── MoleculeGenerator (generation.py)
│   ├── generate_conditional()        - Single conditional generation
│   ├── batch_generate_conditional()  - Batched conditional generation
│   └── generate_unconditional_batch() - Unconditional from CLS token
│
├── ConditionalMetricsComputer (conditional_metrics.py)
│   ├── Property tracking (input vs generated vs delta)
│   ├── Distance to ground truth
│   ├── Neighborhood alignment
│   ├── Memorization checking (per-reference)
│   └── Prompt direction agreement
│
├── UnconditionalMetricsComputer (unconditional_metrics.py)
│   ├── SNN (single nearest neighbor)
│   ├── FCD (Frechet ChemNet Distance)
│   ├── Novelty / Uniqueness
│   └── Coverage and scaffold similarity
│
├── Distance Utilities (distance_utils.py)
│   └── Fingerprint-cached Tanimoto computation
│
├── Property Registry (property_registry.py)
│   ├── Built-in: logp, qed, sa, mw, tpsa, hbd, hba, rotatable_bonds, frac_sp3_c
│   └── ML-based: Class (BACE), p_np (BBBP), HIV_active
│
├── PropertyEvaluator (evaluate_properties.py)
│   └── Loads pretrained RF classifiers for BACE/BBBP/HIV
│
├── SmilesUtils (smiles_utils.py)
│   └── Token stripping, canonicalization, validation
│
├── ValidationPrinter / MetricRegistry (printer.py)
│   └── Metric grouping, safe logging, display formatting
│
└── TimingTracker (timing.py)
    └── Context-manager-based performance profiling
```

## Conditional vs Unconditional Validation

### Conditional Generation
Given an input molecule (molecule_a from a matched pair), generate molecule_b conditioned on molecule_a. The model sees the input SMILES (possibly with property/distance prompt tokens) and generates a continuation.

**Metrics computed**:
- Validity of generated SMILES
- Property deltas (generated - input) for each tracked property
- Fraction of valid-and-improved molecules
- Tanimoto distance to ground truth molecule_b
- Neighborhood alignment (how close generated molecules are to the true neighborhood)

### Unconditional Generation
Generate molecules from scratch using only a CLS/BOS token (no conditioning input). Tests the model's learned distribution.

**Metrics computed**:
- Validity, uniqueness, novelty
- SNN (single nearest neighbor similarity to reference set)
- FCD (Frechet ChemNet Distance to reference set)
- Internal diversity among generated molecules
- Coverage of heavy atom types, fragments, and scaffolds

## Metric Naming Conventions

Metric names follow a structured pattern for WandB compatibility:

```
val_{metric_name}[_{context}][_non_memorized][/{reference_name}]
```

**Components**:
- `val_` prefix: All validation metrics
- `{metric_name}`: The metric being computed (e.g., `molecule_validity`, `tanimoto_distance`, `logp`)
- `{context}`: `conditional` or omitted for unconditional
- `_non_memorized`: Suffix indicating metric excludes memorized molecules
- `/{reference_name}`: Suffix for per-reference-set metrics (e.g., `/zinc_train`, `/bace_train`)

**Examples**:
```
val_molecule_validity_conditional          # Conditional validity rate
val_molecule_validity                      # Unconditional validity rate
val_tanimoto_distance_conditional          # Distance to GT (conditional)
val_logp_prompt_direction_agreement_rate_non_memorized/zinc_train
memorization_rate_conditional/zinc_train   # Per-reference memorization
memorization_rate_conditional/bace_train
```

**Important**: These names are consumed by WandB dashboards and analysis scripts. Do not rename without updating all consumers.

## Memorization Checking

The system tracks whether generated molecules are memorized from training data using exact canonical SMILES matching.

### Per-Reference Memorization
When multiple reference sets are provided (e.g., `bace_train` + `zinc_train`), memorization is checked against each independently:
- `memorization_rate_conditional/bace_train` - Fraction of generations found in BACE training data
- `memorization_rate_conditional/zinc_train` - Fraction found in ZINC training data

### Memorization from Any Reference
A molecule is considered "memorized" if it appears in **any** reference set. This is used to compute `_non_memorized` metric variants that exclude memorized molecules.

### How It Works
1. Reference SMILES are loaded and canonicalized into sets at callback initialization
2. Each generated SMILES is canonicalized and checked for membership
3. Non-memorized subsets are used to compute "clean" agreement/improvement rates

## Property Prompt Agreement

Measures whether the model follows the direction specified in the property prompt.

**Computation**:
1. During training, each pair has a property delta direction (increase/decrease)
2. The prompt encodes this direction as binned property tokens
3. At validation, we generate molecules conditioned on these prompts
4. We compute the actual property delta (generated - input)
5. Agreement = fraction where actual delta direction matches prompt direction

**Metric**: `val_{property}_prompt_direction_agreement_rate_non_memorized/{ref_name}`

The `_non_memorized` variant excludes memorized molecules because a memorized molecule trivially "agrees" with the prompt if the training pair had that direction.

### Bounded Delta Agreement

When bounded delta tokens are used (`atmost` / `atleast`), the magnitude agreement check
uses inequality rather than strict equality:
- **exact** (`<pd_...>`): `actual_bin == prompted_bin`
- **atmost** (`<pdmax_...>`): `actual_bin <= prompted_bin`
- **atleast** (`<pdmin_...>`): `actual_bin >= prompted_bin`

This correctly reflects the semantics of bounded prompts: a model that generates a smaller
delta than the upper bound is still "following" an atmost constraint.

### Strict vs Inclusive Direction Agreement

The `direction_agreement_rate` metric includes samples with `<pdir_any>`, which trivially
count as agreement (any direction matches "any"). This can inflate the metric when many
pairs have zero delta.

The `direction_agreement_rate_strict` metric excludes "any" samples and only counts up/down
prompts — this is the more informative metric for measuring whether the model actually
learned to control property direction.

Both metrics also report a `_count` (sample count) so you can see the denominator.

## Reference Sets

Reference sets serve dual purposes:
1. **Memorization detection**: Check if generated molecules exist in training data
2. **Distribution comparison**: Compute SNN, FCD, coverage against reference molecules

Configured via `callbacks.molecule_validation.reference_smiles_files`:
```yaml
reference_smiles_files:
  - name: zinc_train
    path: /path/to/zinc250k_full_train/molecules.parquet
    default: true  # Used for backward-compatible metric names (no /ref_name suffix)
  - name: bace_train
    path: /path/to/bace/train/molecules_bace_real.parquet
```

The `default: true` reference set's metrics are also logged without the `/{ref_name}` suffix for backward compatibility with existing WandB dashboards.

## File-by-File Overview

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports all public classes and functions |
| `conditional_metrics.py` | `ConditionalMetricsComputer` - all conditional generation metrics |
| `unconditional_metrics.py` | `UnconditionalMetricsComputer` - all unconditional generation metrics |
| `generation.py` | `MoleculeGenerator` - wraps `pl_module.model.generate()` calls |
| `property_registry.py` | Property function registry (RDKit + ML-based evaluators) |
| `evaluate_properties.py` | `PropertyEvaluator` - loads pretrained RF classifiers for MoleculeNet tasks |
| `distance_utils.py` | Fingerprint-cached Tanimoto distance computation |
| `smiles_utils.py` | `SmilesUtils` - SMILES validation, canonicalization, token stripping |
| `printer.py` | `ValidationPrinter`, `MetricRegistry` - logging and display |
| `timing.py` | `TimingTracker` - context-manager performance profiling |
| `FIXES_APPLIED.md` | Documents critical bug fixes during refactoring |
