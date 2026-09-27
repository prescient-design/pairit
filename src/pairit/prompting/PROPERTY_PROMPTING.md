## Property prompting (medchem) overview

This repo supports **property prompting** by prefixing sequences with **special tokens** (no whitespace) right after `<cls>`.

### Token scheme

- **Start marker**: `<pc>`
- **Property token**: `<p_{name}>` where `{name}` matches the configured property key (e.g. `logp`, `qed`, `sa`, `mw`, `tpsa`, `hbd`, `hba`, `rotatable_bonds`, `frac_sp3_c`)
- **Start-value bin**: `<p0_{name}_{bb}>` (zero-padded bin index)
- **Direction**: `<pdir_up>` / `<pdir_down>` / `<pdir_any>`
- **Magnitude (abs delta) bin**: `<pd_{name}_{bb}>`

Example: “increase logp from start-bin 07 with magnitude-bin 03”

```text
<pc><p_logp><p0_logp_07><pdir_up><pd_logp_03>
```

### Bin/range utilities (notebook-friendly)

```python
from pairit.prompting.property_prompting import PropertyBinningConfig, bin_to_range, value_to_bin, delta_to_bin

vb = PropertyBinningConfig(num_bins=32, range=(-2.0, 6.0), mode="fixed")
db = PropertyBinningConfig(num_bins=16, range=(0.0, 4.0), mode="fixed")

# value -> bin
value_bin = value_to_bin(binning=vb, value=2.3)

# bin -> numeric interval
lo, hi = bin_to_range(binning=vb, bin_idx=value_bin)

# delta magnitude -> bin
delta_bin = delta_to_bin(binning=db, delta=-1.2, magnitude=True)
```

### Parsing prompt tokens from token IDs (validation / analysis)

```python
from pairit.prompting.property_prompting import parse_property_prompt_from_ids

# ids should be the conditioning prefix token IDs (e.g. up to <sep>), not necessarily the whole sequence
parsed = parse_property_prompt_from_ids(tokenizer, ids)
for p in parsed:
    print(p.property_name, p.start_bin, p.direction, p.delta_bin)
```


