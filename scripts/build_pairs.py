"""Build matched pairs from prepared molecule tables.

Pairs are structural only: two molecules are paired when their Morgan
fingerprints are at least ``--min-similarity`` apart. Direction and the
property-improvement margin are applied later, at training time, by the
data module -- the pair file itself is unordered.

The paper builds pairs with mmpdb (matched molecular pairs), which is why
the shipped pair tables carry ``rule_environment_id`` / ``constant_id``
columns. Those columns are not read by the data module, and fingerprint
similarity is a close enough substitute to retrain on; use this script
unless you specifically need to reproduce the mmpdb pair set.

The default threshold is calibrated against the paper's BACE training
pairs: it yields 2915 pairs over 798 molecules at median similarity
0.714, against 2546 at 0.684 for the mmpdb set, with the same median of
three partners per molecule.

Writes, next to each molecule table::

    <molecules>/moleculenet/<dataset>/<split>/pairs_<dataset>_real.parquet
    <molecules>/zinc/<split>/pairs.parquet

Usage::

    python scripts/build_pairs.py --molecules data/prepared
    python scripts/build_pairs.py --molecules data/prepared --dataset bace --min-similarity 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MOLECULENET = ("bace", "bbbp", "hiv")


def build_pairs(
    smiles: list[str],
    min_similarity: float,
    max_per_molecule: int,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """All unordered pairs above a Tanimoto threshold, capped per molecule."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        fingerprints.append(None if mol is None else generator.GetFingerprint(mol))

    rows: list[tuple[int, int, float]] = []
    for i, reference in enumerate(fingerprints):
        if reference is None:
            continue
        others = [(j, fingerprints[j]) for j in range(i + 1, len(fingerprints)) if fingerprints[j] is not None]
        if not others:
            continue
        indices = [j for j, _ in others]
        scores = DataStructs.BulkTanimotoSimilarity(reference, [fp for _, fp in others])

        hits = [(j, s) for j, s in zip(indices, scores) if s >= min_similarity]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        for j, score in hits[:max_per_molecule]:
            rows.append((i, j, float(score)))

    frame = pd.DataFrame(rows, columns=["molecule_a_idx", "molecule_b_idx", "similarity"])
    frame["pair_type"] = "real_tanimoto"
    return frame


def process(molecules_path: Path, pairs_path: Path, min_similarity: float, max_per_molecule: int) -> None:
    molecules = pd.read_parquet(molecules_path)
    pairs = build_pairs(molecules["smiles"].tolist(), min_similarity, max_per_molecule)
    pairs.to_parquet(pairs_path, index=False)

    if len(pairs):
        per_molecule = pairs.groupby("molecule_a_idx").size()
        summary = (
            f"median similarity {pairs.similarity.median():.3f}, "
            f"median {per_molecule.median():.0f} per molecule"
        )
    else:
        summary = "no pairs above threshold"
    print(f"  {pairs_path}  [{len(pairs)} pairs from {len(molecules)} molecules; {summary}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecules",
        type=Path,
        default=Path("data/prepared"),
        help="prepare_molecules.py output (default: data/prepared)",
    )
    parser.add_argument(
        "--dataset",
        choices=[*MOLECULENET, "zinc"],
        action="append",
        help="limit to one dataset; repeatable (default: all)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.65,
        help="Tanimoto threshold (default: 0.65)",
    )
    parser.add_argument(
        "--max-pairs-per-molecule",
        type=int,
        default=50,
        help="keep only this many most-similar partners per molecule (default: 50)",
    )
    args = parser.parse_args(argv)

    for name in args.dataset or [*MOLECULENET, "zinc"]:
        print(f"{name}:")
        if name == "zinc":
            for split in ("train", "valid"):
                directory = args.molecules / "zinc" / split
                if not (directory / "molecules.parquet").exists():
                    print(f"  skip (not prepared)  {directory}")
                    continue
                process(
                    directory / "molecules.parquet",
                    directory / "pairs.parquet",
                    args.min_similarity,
                    args.max_pairs_per_molecule,
                )
        else:
            for split in ("train", "valid", "test"):
                directory = args.molecules / "moleculenet" / name / split
                source = directory / f"molecules_{name}_real.parquet"
                if not source.exists():
                    print(f"  skip (not prepared)  {directory}")
                    continue
                process(
                    source,
                    directory / f"pairs_{name}_real.parquet",
                    args.min_similarity,
                    args.max_pairs_per_molecule,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
