"""Turn the downloaded source data into PairIT molecule tables.

Reads what ``scripts/download_data.py`` fetched and writes one parquet per
split, in the layout the training and evaluation configs expect::

    <out>/moleculenet/<dataset>/<split>/molecules_<dataset>_real.parquet
    <out>/zinc/<split>/molecules.parquet

SMILES are canonicalised with RDKit and eight descriptors are computed
(logP, SAS, TPSA, MW, HBD, HBA, QED, fracCSP3). MoleculeNet splits come
from the Graph-DiT ``*_split.pt`` index lists; the ZINC split comes from
MOOD's ``valid_idx_zinc250k.json``.

Usage::

    python scripts/prepare_molecules.py --raw data/raw --out data/prepared
    python scripts/prepare_molecules.py --raw data/raw --out data/prepared --dataset bace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

try:
    import sascorer
except ImportError:  # sascorer ships inside RDKit's Contrib tree
    import os

    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

MOLECULENET = ("bace", "bbbp", "hiv")
DESCRIPTORS = ("logP", "SAS", "TPSA", "MW", "HBD", "HBA", "QED", "fracCSP3")


def describe(smiles: str) -> dict | None:
    """Canonicalise one SMILES and compute its descriptors; None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "smiles": Chem.MolToSmiles(mol),
        "logP": Crippen.MolLogP(mol),
        "SAS": sascorer.calculateScore(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MW": Descriptors.MolWt(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        # Lipinski definition, not rdMolDescriptors.CalcNumHBA -- the two
        # disagree on roughly one molecule in six.
        "HBA": Descriptors.NumHAcceptors(mol),
        "QED": QED.qed(mol),
        "fracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def build_table(rows: list[dict], split: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["split"] = split
    frame["molecule_type"] = "real"
    frame["molecule_idx"] = range(len(frame))
    columns = ["smiles", "original_id", "Class", "split", "molecule_type", "molecule_idx"]
    return frame[columns + list(DESCRIPTORS)]


def prepare_moleculenet(dataset: str, raw: Path, out: Path) -> None:
    import torch  # imported lazily: only the MoleculeNet path needs it

    source = pd.read_csv(raw / "moleculenet" / f"{dataset}_b.csv.gz")
    split_index = torch.load(raw / "moleculenet" / f"{dataset}_b_split.pt", weights_only=False)

    for split, key in (("train", "train_index"), ("valid", "val_index"), ("test", "test_index")):
        rows = []
        skipped = 0
        # Sorted so molecule_idx follows source-file order, which is what the
        # pair tables index into.
        for position in sorted(split_index[key]):
            record = source.iloc[position]
            described = describe(record["smiles"])
            if described is None:
                skipped += 1
                continue
            described["original_id"] = record["mol_id"]
            described["Class"] = float(record["Class"])
            rows.append(described)

        table = build_table(rows, split)
        target = out / "moleculenet" / dataset / split / f"molecules_{dataset}_real.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)
        note = f" ({skipped} unparseable skipped)" if skipped else ""
        print(f"  {target}  [{len(table)} molecules]{note}")


def prepare_zinc(raw: Path, out: Path) -> None:
    source = pd.read_csv(raw / "zinc" / "zinc250k.csv")
    valid_positions = set(json.loads((raw / "zinc" / "valid_idx_zinc250k.json").read_text()))

    assignments = ["valid" if i in valid_positions else "train" for i in range(len(source))]
    source = source.assign(_split=assignments)

    for split in ("train", "valid"):
        subset = source[source["_split"] == split]
        rows = []
        skipped = 0
        for position, record in enumerate(subset.itertuples(index=False)):
            described = describe(record.smiles)
            if described is None:
                skipped += 1
                continue
            described["original_id"] = f"ZINC_{position}"
            described["Class"] = float("nan")
            rows.append(described)

        table = build_table(rows, split)
        target = out / "zinc" / split / "molecules.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)
        note = f" ({skipped} unparseable skipped)" if skipped else ""
        print(f"  {target}  [{len(table)} molecules]{note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="download_data.py output")
    parser.add_argument("--out", type=Path, default=Path("data/prepared"), help="where to write tables")
    parser.add_argument(
        "--dataset",
        choices=[*MOLECULENET, "zinc"],
        action="append",
        help="limit to one dataset; repeatable (default: all)",
    )
    args = parser.parse_args(argv)

    datasets = args.dataset or [*MOLECULENET, "zinc"]
    for name in datasets:
        print(f"{name}:")
        if name == "zinc":
            prepare_zinc(args.raw, args.out)
        else:
            prepare_moleculenet(name, args.raw, args.out)

    print(f"\nDone. Next: python scripts/build_pairs.py --molecules {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
