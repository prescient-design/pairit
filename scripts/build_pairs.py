"""Build the training pair tables from prepared molecule tables.

The two dataset families are paired differently, following the paper:

* **MoleculeNet (BACE/BBBP/HIV)** uses matched molecular pairs. Each split
  is fragmented and indexed independently with mmpdb, then reduced to one
  row per unordered molecule pair.
* **ZINC-250K** uses Morgan fingerprint neighbours, keeping the closest
  partners per molecule.

Pairs are purely structural in both cases: no property-improvement filter
is applied here. Direction and the improvement margin (``epsilon``) are
applied at training time by the data module, which is why the stored
tables contain both improving and non-improving pairs.

Writes next to each molecule table::

    <molecules>/moleculenet/<dataset>/<split>/pairs.parquet
    <molecules>/zinc/<split>/pairs.parquet

MoleculeNet requires mmpdb (``pip install mmpdb``).

Checked against the pair counts in Table 5 of the paper. BACE reproduces
exactly and pair-for-pair (2546 train, 217 test). BBBP and HIV come out
8-12% over the published counts (554 vs 512 and 1128 vs 1009 on train)
despite identical molecule tables, most likely a difference in mmpdb
version rather than in the recipe.

Usage::

    python scripts/build_pairs.py --molecules data/prepared
    python scripts/build_pairs.py --molecules data/prepared --dataset bace
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MOLECULENET = ("bace", "bbbp", "hiv")

# Raised from the mmpdb defaults of 10 and 100: MoleculeNet molecules are
# large drug-like compounds, and at the defaults the biggest ones are
# dropped before fragmentation.
FRAGMENT_OPTIONS = ["--max-rotatable-bonds", "20", "--max-heavies", "200"]
INDEX_OPTIONS = ["--min-variable-heavies", "0", "--max-variable-heavies", "10"]

_FINGERPRINTS = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(
        _FINGERPRINTS.GetFingerprint(mol_a), _FINGERPRINTS.GetFingerprint(mol_b)
    )


def run_mmpdb(arguments: list[str]) -> None:
    command = [sys.executable, "-m", "mmpdblib", "-q", *arguments]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "mmpdb failed (is it installed? `pip install mmpdb`)\n"
            f"  command: {' '.join(command)}\n{result.stderr[-2000:]}"
        )


def mmp_pairs(molecules: pd.DataFrame) -> pd.DataFrame:
    """Fragment + index one split with mmpdb; one row per unordered pair."""
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        smiles_file = workspace / "input.smi"
        # mmpdb wants SMILES first, identifier second.
        smiles_file.write_text(
            "".join(f"{s}\t{i}\n" for s, i in zip(molecules["smiles"], molecules["molecule_idx"]))
        )

        fragments = workspace / "fragments.fragdb"
        database = workspace / "pairs.mmpdb"
        run_mmpdb(["fragment", str(smiles_file), "-o", str(fragments), *FRAGMENT_OPTIONS])
        run_mmpdb(["index", str(fragments), "-o", str(database), *INDEX_OPTIONS])

        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                """
                SELECT c1.public_id, c2.public_id, p.rule_environment_id, p.constant_id
                FROM pair p
                JOIN compound c1 ON p.compound1_id = c1.id
                JOIN compound c2 ON p.compound2_id = c2.id
                """
            ).fetchall()

    pairs = pd.DataFrame(rows, columns=["a", "b", "rule_environment_id", "constant_id"])
    if pairs.empty:
        return pd.DataFrame(
            columns=["molecule_a_idx", "molecule_b_idx", "similarity", "rule_environment_id", "constant_id", "pair_type"]
        )

    pairs[["a", "b"]] = pairs[["a", "b"]].astype(int)
    pairs["molecule_a_idx"] = pairs[["a", "b"]].min(axis=1)
    pairs["molecule_b_idx"] = pairs[["a", "b"]].max(axis=1)
    pairs = pairs.drop_duplicates(["molecule_a_idx", "molecule_b_idx"]).reset_index(drop=True)

    smiles = dict(zip(molecules["molecule_idx"], molecules["smiles"]))
    pairs["similarity"] = [
        tanimoto(smiles[a], smiles[b]) for a, b in zip(pairs["molecule_a_idx"], pairs["molecule_b_idx"])
    ]
    pairs["constant_id"] = pairs["constant_id"].fillna(-1).astype(int)
    pairs["pair_type"] = "real_mmp"
    return pairs[
        ["molecule_a_idx", "molecule_b_idx", "similarity", "rule_environment_id", "constant_id", "pair_type"]
    ]


def tanimoto_pairs(molecules: pd.DataFrame, max_per_molecule: int) -> pd.DataFrame:
    """Closest fingerprint neighbours per molecule, as a distance column."""
    fingerprints = []
    for smile in molecules["smiles"]:
        mol = Chem.MolFromSmiles(smile)
        fingerprints.append(None if mol is None else _FINGERPRINTS.GetFingerprint(mol))

    rows: list[tuple[int, int, float]] = []
    for i, reference in enumerate(fingerprints):
        if reference is None:
            continue
        others = [(j, fp) for j, fp in enumerate(fingerprints) if j != i and fp is not None]
        if not others:
            continue
        scores = DataStructs.BulkTanimotoSimilarity(reference, [fp for _, fp in others])
        ranked = sorted(zip((j for j, _ in others), scores), key=lambda pair: pair[1], reverse=True)
        for j, score in ranked[:max_per_molecule]:
            rows.append((i, j, 1.0 - float(score)))

    return pd.DataFrame(rows, columns=["molecule_a_idx", "molecule_b_idx", "morgan_tanimoto_distance"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", type=Path, default=Path("data/prepared"))
    parser.add_argument("--dataset", choices=[*MOLECULENET, "zinc"], action="append")
    parser.add_argument(
        "--max-pairs-per-molecule",
        type=int,
        default=100,
        help="ZINC only: closest neighbours kept per molecule (default: 100)",
    )
    args = parser.parse_args(argv)

    for name in args.dataset or [*MOLECULENET, "zinc"]:
        print(f"{name}:")
        splits = ("train", "valid") if name == "zinc" else ("train", "valid", "test")
        for split in splits:
            directory = (
                args.molecules / "zinc" / split
                if name == "zinc"
                else args.molecules / "moleculenet" / name / split
            )
            source = directory / "molecules.parquet"
            if not source.exists():
                print(f"  skip (not prepared)  {directory}")
                continue

            molecules = pd.read_parquet(source)
            if name == "zinc":
                pairs = tanimoto_pairs(molecules, args.max_pairs_per_molecule)
            else:
                pairs = mmp_pairs(molecules)

            target = directory / "pairs.parquet"
            pairs.to_parquet(target, index=False)
            print(f"  {target}  [{len(pairs)} pairs from {len(molecules)} molecules]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
