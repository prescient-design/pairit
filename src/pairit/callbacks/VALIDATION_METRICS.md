## Molecule Validation and First Molecule Callback Metrics

This document lists all metrics currently logged to W&B by the molecule validation callback and the first molecule callback, with brief explanations. Metric names below include the exact keys as they appear in W&B.

Notes
- Many metrics are logged with the "val_" prefix; the names below include that prefix.
- Some metric names contain placeholders:
  - {spec.name}: property name from `property_specs` (defaults: `logp`, `qed`, `sa`)
  - {distance_name}: distance metric label (default: `tanimoto`)
  - {feat_name}: RDKit feature name from `FEATURE_NAMES` (includes `logp`, `tpsa`, etc.)

### Basic validity (unconditional generation)
- val_molecule_validity: Fraction of valid molecules among both positions in the pair.
- val_molecule_validity_first: Fraction of valid first molecules.
- val_molecule_validity_second: Fraction of valid second molecules.

### Conditional validity and improvements (per-batch, aggregated per-epoch)
- val_molecule_validity_conditional: Fraction of valid conditional generations in the processed batch(es).
- val_fraction_valid_and_improved_conditional/{spec.name}: Fraction of batch samples that are valid and improve the given property beyond its `min_delta`.
- val_fraction_valid_and_improved_conditional/any: Fraction of batch samples that improve any configured property while valid.

### Per-sample conditional property values and deltas (aggregated per-epoch)
- val_{spec.name}_conditional_input: Property value for the input (seed) molecule.
- val_{spec.name}_conditional_generated: Property value for the generated molecule.
- val_{spec.name}_delta_conditional: Difference `generated − input` for the property.

### Per-sample conditional distances to ground truth (aggregated per-epoch)
- val_{distance_name}_distance_to_gt_conditional: Min distance of generated molecule to provided GT set.
- val_{distance_name}_distance_input_to_gt_conditional: Min distance of input molecule to GT set.
- val_{distance_name}_gt_closest_delta_conditional: Difference `min_dist(gen, GT) − min_dist(input, GT)`.

### Per-sample conditional heavy atom and normalized distance metrics (aggregated per-epoch)
- val_heavy_atom_delta_conditional: Change in number of heavy atoms `generated − seed`.
- val_tanimoto_distance_per_seed_heavy_atom_conditional: Tanimoto distance divided by the number of heavy atoms in the seed molecule.

### Conditional aggregate metrics (epoch-level via registry)
- val_{distance_name}_distance_to_gt_conditional_mean: Mean of the per-sample min distances to GT.
- val_{distance_name}_neighborhood_distance_median_conditional: Median distance to neighborhood (if neighbors available).
- val_{distance_name}_neighborhood_distance_p75_conditional: 75th percentile of distance to neighborhood.
- val_{distance_name}_nearest_neighbor_distance_conditional: Nearest (min) distance to neighborhood.
- val_w1_delta_conditional/{feat_name}: Neighborhood alignment proxy; mean signed W1 delta per feature (lower is better alignment).
- val_kde_ll_in_conditional: Mean KDE log-likelihood of input within neighborhood feature distributions.
- val_kde_ll_gen_conditional: Mean KDE log-likelihood of generation within neighborhood feature distributions.
- val_kde_ll_delta_conditional: `kde_ll_gen − kde_ll_in`.
- val_{distance_name}_gt_closest_delta_conditional_mean: Mean of `min_dist(gen, GT) − min_dist(input, GT)`.
- val_uniqueness_conditional_valid: Uniqueness fraction among valid conditional generations (canonicalized).
- val_novelty_conditional_valid: Novelty fraction vs. training canonical set (if provided).
- val_similarity_to_seed_mean: Mean Tanimoto similarity between seed and generated molecules.
- val_max_similarity_to_gt_gen_mean: Mean of max similarity of gens to GT per sample.
- val_max_similarity_to_gt_seed_mean: Mean of max similarity of seeds to GT per sample.
- val_max_similarity_to_gt_delta_mean: Mean difference `(gen_max_sim − seed_max_sim)`.
- val_property_delta_mean/{spec.name}: Mean property delta `generated − input`.
- val_property_delta_median/{spec.name}: Median property delta.
- val_property_delta_std/{spec.name}: Std of property delta.
- val_property_frac_improved/{spec.name}: Fraction of pairs with property improvement (per `direction`/`min_delta`).
- val_property_preservation_mad_macro: Macro mean of |delta| across non-target properties.
- val_property_preservation_mad/{spec.name}: Mean absolute difference |`generated − input`| for the property.
- val_delta_under_sim/{spec.name}: Mean property delta restricted to seed–gen similarity band (default [0.4, 0.6]).
- val_frac_improved_under_sim/{spec.name}: Fraction improved under the similarity constraint.
- val_topk_best/{spec.name}_best: Best property value among valid conditional generations (deduped).
- val_topk_best/{spec.name}_mean_topk: Mean of top‑k (default k=3) property values among valid conditionals.
- val_validity_fraction_conditional: Validity fraction over all conditional generations (including invalid).
- val_uniqueness_conditional_valid: Uniqueness fraction among valid canonicalized conditional generations.
- val_novelty_conditional_valid: Novelty fraction vs. training set among valid canonicalized conditionals.
- val_cond_pairs_count: Number of conditional pairs processed.
- val_cond_valid_count: Number of valid conditional generations (canonicalized).

### Unconditional aggregate metrics (epoch-level via registry)
- val_snn_mean_unconditional: Mean single-nearest-neighbor similarity to the reference set.
- val_snn_median_unconditional: Median SNN similarity to the reference set.
- val_diversity_unconditional: Internal diversity = `1 − mean_pairwise_tanimoto(ECFP4)` among valid gens (deduped).
- val_diversity_n_pairs_unconditional: Number of pairs used for internal diversity.
- val_coverage_ratio_unconditional: Coverage ratio of heavy-atom types vs. reference set.
- val_fragment_cosine_unconditional: Cosine similarity of BRICS fragment frequencies (gens vs refs).
- val_scaffold_cosine_unconditional: Cosine similarity of Bemis–Murcko scaffold frequencies (gens vs refs).
- val_feature_w1_macro_unconditional: Macro mean W1 distance across RDKit features between gens and refs.
- val_feature_w1_norm_macro_unconditional: Macro mean normalized W1 across features.
- val_feature_kde_delta_macro_unconditional: Macro mean KDE log-likelihood delta across features.
- val_fcd_unconditional: Fréchet ChemNet Distance between gen/ref ChemNet embeddings (if enough samples).
- val_uncond_valid_count: Number of valid unconditional generations (canonicalized) used for metrics.

### First Molecule Callback (validation batches)
- val_first_molecule_loss: Average cross-entropy loss for predicting the first molecule in each pair.
- val_first_molecule_perplexity: Perplexity (exp(loss)) for the first-molecule prediction.
- val_first_molecule_samples: Number of valid samples used to compute the first-molecule metrics.

### Defaults and configuration
- Properties: defaults are `logp` (higher better), `qed` (higher better), and `sa` (lower better).
- Distance name: default is `tanimoto`; names are included in metric keys where relevant.
- Feature names (for W1/KDE): include `num_rings`, `num_aromatic_rings`, `num_heterocycles`, `heavy_atom_count`, `frac_sp3_c`, `hba`, `hbd`, `tpsa`, `logp`.


