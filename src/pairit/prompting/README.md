# Property prompts

Property prompts support both the existing seed-value/direction/magnitude grammar
and signed delta tokens. A signed block is `<p_logp><psd_logp_08>`; an optional
seed-value token can precede the signed token for checkpoints trained with it.
The prefix starts with `<pc>` and contains no whitespace.

Set `include.magnitude: signed_bin` and register the same number of
`property_prompt_tokens.num_signed_delta_bins` in the tokenizer. The paper's
signed recipes use 16 bins. Existing unsigned prompts, bounded magnitudes, and
dropout settings retain their behavior; signed prompting does not disable
property dropout. The consumed dropout path is
`prompting.properties.include.property_dropout.{enabled,rate}`.

`PropertyBinningConfig.range` stores the magnitude limits `[0, max]`.
`signed_delta_to_bin` uses these limits symmetrically over `[-max, max]`, with
the existing clipping policy. `signed_bin_to_range` uses the same arithmetic.
Use the saved experiment's property-specific limits. Generic evaluator defaults
are not a replacement for fitted training limits.

For 16 bins, bin 07 is `[-max/8, 0)` and bin 08 is `[0, max/8)`.
Bin 08 therefore encodes both exact zero and small positive changes. Preserve
raw pair values, delta and direction separately using `ParsedPropertyPrompt`'s
`raw_*` fields. These fields never enter the token prefix and cannot be recovered
by decoding it. Randomized input bins must not replace the original scoring
target.

The existing callback agreement metrics accept signed prompts. Direction uses
raw truth when supplied, including small positive bin-08 requests; exact zero
has no requested direction. Without raw truth, only intervals wholly above or
below zero supply a direction, while bin agreement still includes the
zero-containing interval. Thus a token-only direction rate has a narrower
population than raw-target paper adherence. Supply the original target prompt
to score randomized controls against the same task.

Signed tokens append after all existing token families, including distance
tokens. Existing vocabulary IDs and special-token registrations are retained.
When loading a checkpoint, inherit its complete vocabulary ordering before
adding any new controls; changing property lists must not remap learned tokens.
The existing training and evaluation entry points recover all backend added
tokens, including historical controls missing from `additional_special_tokens`,
and verify the full saved ID map before appending new tokens. The checkpoint's
CLM input and output vocabulary matrices must cover every saved ID; unused
padding rows are allowed. A requested checkpoint with missing tokenizer/matrix
metadata, incompatible IDs, or custom added-token matching flags fails explicitly.
The two historical Lobster pickle paths for `TokenizerTransform` and
`SmilesTokenizer2Fast` resolve to their PairIT equivalents through
`pairit.cmdline._checkpoint_compat`; other Lobster objects are unsupported.

Run the focused regressions from the repository root:

```bash
PYTHONPATH=src pytest -q tests/test_signed_property_prompting.py \
  tests/test_checkpoint_prompting_compatibility.py \
  tests/test_property_prompting_enhanced.py \
  tests/pairit/callbacks/test_signed_prompt_agreement.py \
  tests/pairit/callbacks/test_ddp_fraction_registration.py
```
