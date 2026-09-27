#!/usr/bin/env python3
"""
Smoke-test + demo for property prompting tokens.

This script demonstrates:
- How prompt tokens are appended to the tokenizer vocab (IDs are higher / appended)
- Building a prompt prefix (<pc>...) and concatenating with SMILES sequences
- Tokenization + detokenization + prompt parsing from token IDs
- Computing medchem properties for the molecules and comparing against the prompt

Run (recommended):
  python -m pairit.prompting.demo_property_prompting
"""

from __future__ import annotations

import sys
from dataclasses import asdict
import re
from pathlib import Path


def _try_import_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def compute_properties(smiles: str, prop_names: list[str]) -> dict[str, float | None]:
    """
    Compute a subset of medchem properties using RDKit (+ local SA scorer when available).
    Mirrors the minimal set used in dataset prompting.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, QED
    except Exception as e:
        raise RuntimeError(f"RDKit required: {e}") from e

    # Optional SA scorer (avoid importing pairit.callbacks package which can depend on lightning).
    sascorer = None
    try:
        import importlib.util

        root = Path(__file__).resolve().parents[2]  # .../src/pairit
        sascorer_path = root / "callbacks" / "sascorer.py"
        if sascorer_path.exists():
            spec = importlib.util.spec_from_file_location("_pairit_sascorer", str(sascorer_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                sascorer = mod
    except Exception:
        sascorer = None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {p: None for p in prop_names}

    out: dict[str, float | None] = {}
    for p in prop_names:
        try:
            if p == "logp":
                out[p] = float(Crippen.MolLogP(mol))
            elif p == "qed":
                out[p] = float(QED.qed(mol))
            elif p == "sa":
                out[p] = None if sascorer is None else float(sascorer.calculateScore(mol))
            elif p == "mw":
                out[p] = float(Descriptors.MolWt(mol))
            elif p == "tpsa":
                out[p] = float(Descriptors.TPSA(mol))
            elif p == "hbd":
                out[p] = float(Descriptors.NumHDonors(mol))
            elif p == "hba":
                out[p] = float(Descriptors.NumHAcceptors(mol))
            elif p == "rotatable_bonds":
                out[p] = float(Descriptors.NumRotatableBonds(mol))
            elif p == "frac_sp3_c":
                out[p] = float(Descriptors.FractionCSP3(mol))
            else:
                out[p] = None
        except Exception:
            out[p] = None
    return out


def tokens_to_smiles_stripping_prompts(tokenizer, token_ids: list[int]) -> tuple[str, str | None]:
    """
    Minimal detokenizer for this demo:
    - decode ids to string
    - strip <cls>/<eos>/<pad>
    - strip all tokenizer.additional_special_tokens except <sep>
    - split on <sep>
    """
    decoded = tokenizer.decode(token_ids)
    for token in [tokenizer.cls_token, tokenizer.eos_token, tokenizer.pad_token]:
        if token and token in decoded:
            decoded = decoded.replace(token, "")

    sep_tok = getattr(tokenizer, "sep_token", "<sep>")
    extra = getattr(tokenizer, "additional_special_tokens", None) or []
    for tok in extra:
        if tok == sep_tok:
            continue
        if tok and tok in decoded:
            decoded = decoded.replace(tok, "")

    if sep_tok in decoded:
        a, b = decoded.split(sep_tok, 1)
        return ("".join(a.split()).strip(), "".join(b.split()).strip())
    return ("".join(decoded.split()).strip(), None)


def _ensure_1d_token_ids(ids) -> list[int]:
    """
    TokenizerTransform commonly returns a batch-shaped tensor (e.g. [1, seq_len]).
    Normalize to a flat list[int] for decode/processing.
    """
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    # ids may be list[int] or list[list[int]]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return [int(x) for x in ids[0]]
    if isinstance(ids, list):
        return [int(x) for x in ids]
    return []


def main() -> int:
    # We can always test prompt build/parse + binning without tokenizer dependencies.
    have_tokenizer_deps = True
    try:
        import tokenizers  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        have_tokenizer_deps = False
        print(
            "NOTE: tokenizers/transformers not installed; skipping tokenizer tokenization section.\n"
            "To enable it:\n"
            "  pip install tokenizers transformers\n"
            f"Import error: {e}\n",
            file=sys.stderr,
        )

    have_rdkit = _try_import_rdkit()
    if not have_rdkit:
        print(
            "NOTE: RDKit not installed; skipping property computation section.\n"
            "To enable it, install RDKit in your env.\n",
            file=sys.stderr,
        )

    from pairit.prompting.property_prompting import (
        PropertyBinningConfig,
        ParsedPropertyPrompt,
        bin_to_range,
        build_property_prompt_text,
        format_property_prompt_summary,
        parse_property_prompt_from_ids,
        value_to_bin,
        delta_to_bin,
    )

    # --- configuration for the demo ---
    prop_names = ["logp", "qed", "sa", "mw", "tpsa", "hbd", "hba", "rotatable_bonds", "frac_sp3_c"]

    # Fixed ranges for demonstration (adjust as desired).
    value_ranges = {
        "logp": (-2.0, 6.0),
        "qed": (0.0, 1.0),
        "sa": (1.0, 10.0),
        "mw": (100.0, 700.0),
        "tpsa": (0.0, 200.0),
        "hbd": (0.0, 10.0),
        "hba": (0.0, 20.0),
        "rotatable_bonds": (0.0, 20.0),
        "frac_sp3_c": (0.0, 1.0),
    }

    # Example molecule pair (seed -> improved)
    smiles1 = "CCO"  # ethanol
    smiles2 = "CCOC"  # ethyl methyl ether

    # Build a prompt from computed properties, for a couple properties only (logp + tpsa).
    if have_rdkit:
        props1 = compute_properties(smiles1, ["logp", "tpsa"])
        props2 = compute_properties(smiles2, ["logp", "tpsa"])
    else:
        # Dummy values just to exercise binning + prompt build/parse
        props1 = {"logp": 0.0, "tpsa": 10.0}
        props2 = {"logp": 0.5, "tpsa": 15.0}

    vb_logp = PropertyBinningConfig(num_bins=32, range=value_ranges["logp"], mode="fixed")
    vb_tpsa = PropertyBinningConfig(num_bins=32, range=value_ranges["tpsa"], mode="fixed")
    db_logp = PropertyBinningConfig(num_bins=16, range=(0.0, 3.0), mode="fixed")
    db_tpsa = PropertyBinningConfig(num_bins=16, range=(0.0, 100.0), mode="fixed")

    logp_start_bin = value_to_bin(binning=vb_logp, value=float(props1["logp"]))
    logp_delta_bin = delta_to_bin(binning=db_logp, delta=float(props2["logp"]) - float(props1["logp"]), magnitude=True)
    tpsa_start_bin = value_to_bin(binning=vb_tpsa, value=float(props1["tpsa"]))
    tpsa_delta_bin = delta_to_bin(binning=db_tpsa, delta=float(props2["tpsa"]) - float(props1["tpsa"]), magnitude=True)

    prompt_spec = [
        ParsedPropertyPrompt(property_name="logp", start_bin=logp_start_bin, direction="any", delta_bin=logp_delta_bin),
        ParsedPropertyPrompt(property_name="tpsa", start_bin=tpsa_start_bin, direction="any", delta_bin=tpsa_delta_bin),
    ]
    prompt_text = build_property_prompt_text(prompt_spec, include_start_marker=True, value_width=2, delta_width=2)

    # --- always-on: prompt build -> parse roundtrip (using a tiny fake tokenizer) ---
    class _FakeTokenizer:
        def __init__(self, tokens: list[str]):
            self._id_to_tok = list(tokens)

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [self._id_to_tok[i] for i in ids]

    prompt_tokens = re.findall(r"<[^>]+>", prompt_text)
    fake_tok = _FakeTokenizer(prompt_tokens)
    fake_ids = list(range(len(prompt_tokens)))
    parsed = parse_property_prompt_from_ids(fake_tok, fake_ids)
    print("\n=== Prompt parsing ===")
    for p in parsed:
        print(asdict(p))

    print("\n=== Prompt summary (human-readable) ===")
    print(
        format_property_prompt_summary(
            parsed,
            value_binning_by_property={"logp": vb_logp, "tpsa": vb_tpsa},
            delta_binning_by_property={"logp": db_logp, "tpsa": db_tpsa},
            show_ranges=True,
            decimals=3,
        )
    )

    print("\n=== Bin ranges (script-specified) ===")
    for p in parsed:
        if p.property_name == "logp" and p.start_bin is not None:
            lo, hi = bin_to_range(binning=vb_logp, bin_idx=int(p.start_bin))
            dlo, dhi = bin_to_range(binning=db_logp, bin_idx=int(p.delta_bin)) if p.delta_bin is not None else (None, None)
            print(f"logp start_bin={p.start_bin} -> [{lo:.3f}, {hi:.3f}], delta_bin={p.delta_bin} -> [{dlo:.3f}, {dhi:.3f}]")
        if p.property_name == "tpsa" and p.start_bin is not None:
            lo, hi = bin_to_range(binning=vb_tpsa, bin_idx=int(p.start_bin))
            dlo, dhi = bin_to_range(binning=db_tpsa, bin_idx=int(p.delta_bin)) if p.delta_bin is not None else (None, None)
            print(f"tpsa start_bin={p.start_bin} -> [{lo:.3f}, {hi:.3f}], delta_bin={p.delta_bin} -> [{dlo:.3f}, {dhi:.3f}]")

    if have_rdkit:
        print("\n=== RDKit properties (for prompted-config properties) ===")
        p1 = compute_properties(smiles1, prop_names)
        p2 = compute_properties(smiles2, prop_names)
        for k in prop_names:
            v1, v2 = p1.get(k), p2.get(k)
            d = None if (v1 is None or v2 is None) else (v2 - v1)
            print(f"{k:16s}: seed={v1!r:>10}  gen={v2!r:>10}  delta={d!r}")

    # --- optional: tokenizer tokenization/detokenization section ---
    if have_tokenizer_deps:
        from pairit.tokenization import SmilesTokenizer2Fast
        from pairit.transforms import TokenizerTransform

        tokenizer = SmilesTokenizer2Fast(
            add_dataset_tokens=False,
            add_property_prompt_tokens=True,
            property_prompt_tokens={
                "property_names": prop_names,
                "num_value_bins": 32,
                "num_delta_bins": 16,
            },
        )
        transform = TokenizerTransform(tokenizer, padding="max_length", truncation=True, max_length=128)

        print("\n=== Tokenizer / vocab ===")
        print(f"vocab_size_total: {len(tokenizer.get_vocab())}")
        for tok in ["<pc>", "<p_logp>", "<p0_logp_00>", "<pd_logp_00>", "<pdir_up>", "<pdir_any>"]:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            print(f"token {tok} -> id {tok_id}")

        sequence = f"{prompt_text}{smiles1}<sep>{smiles2}"
        toks = transform(sequence)
        ids = _ensure_1d_token_ids(toks["input_ids"])

        print("\n=== Prompt + sequence ===")
        print(f"prompt_text: {prompt_text}")
        print(f"sequence: {sequence}")
        print(f"decoded_full: {tokenizer.decode(ids)}")

        out1, out2 = tokens_to_smiles_stripping_prompts(tokenizer, ids)
        print("\n=== Detokenization ===")
        print(f"tokens_to_smiles -> smiles1={out1!r}, smiles2={out2!r}")
        assert out1 == smiles1, f"Expected {smiles1!r}, got {out1!r}"
        assert out2 == smiles2, f"Expected {smiles2!r}, got {out2!r}"

    print("\nOK: demo completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


