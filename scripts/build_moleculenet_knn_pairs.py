"""Build Tanimoto k-nearest-neighbour pairs for the MoleculeNet datasets.

MoleculeNet has only ever been paired by MMP in this project — the ZINC neighbour sweep
varies pairing at *pretraining*, but every arm then fine-tunes on the same MMP pairs. This
script supplies the missing kNN arm so pairing method can be compared downstream.

Molecules come from the **real-only** tables (`real_pairs_by_split`), not `*_both`: the
synthetic molecules in `both` were themselves produced by MMP enumeration, so building kNN
over them would not isolate the pairing method. Real-only also has a like-for-like MMP
baseline already measured (BACE 2,546 train pairs, BBBP 512, HIV 1,009).

Validation splits: BACE has a real one. **BBBP and HIV do not** — their `valid/`
directories are empty, and the existing `*_both` configs work around this by pointing
`val_molecule_filename` at `test/`, i.e. selecting checkpoints on the test set. This script
instead carves a molecule-disjoint validation holdout out of train (connected components of
the pair graph), so the kNN arm never sees test during training.

Set ``PAIRIT_MOLECULENET_PAIRS_SRC`` to the directory holding the real-only
``<dataset>/<split>/`` molecule tables.

Usage::

    python scripts/build_moleculenet_knn_pairs.py --k 5 20
    python scripts/build_moleculenet_knn_pairs.py --k 10 --dataset bace --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tanimoto_knn_pairs import (  # noqa: E402
    descriptor_columns,
    fingerprints,
    knn_pairs,
)
# NOTE: the TDC builder's connected-component splitter is NOT reusable here. It assumes a
# fragmented pair graph (true for MMP), but a kNN graph is almost always a single connected
# component, so the first component taken swallows every pair and train ends up empty. We
# split on molecules instead — see split_pairs_by_molecule below.

SRC_ROOT = os.environ.get(
    "PAIRIT_MOLECULENET_PAIRS_SRC",
    "<unset:PAIRIT_MOLECULENET_PAIRS_SRC>",
)
DST_ROOT = os.environ.get(
    "PAIRIT_MOLECULENET_KNN_ROOT",
    "datasets/moleculenet_knn",
)
SPLITS = ("train", "valid", "test")
VAL_HOLDOUT_FRACTION = 0.15
VAL_HOLDOUT_SEED = 0

# dataset -> (target column, improvement direction, molecule filename)
DATASETS: dict[str, tuple[str, str, str]] = {
    "bace": ("Class", "increasing", "molecules_bace_real.parquet"),
    "bbbp": ("p_np", "increasing", "molecules_bbbp_real.parquet"),
    "hiv": ("HIV_active", "increasing", "molecules_hiv_real.parquet"),
}


def split_pairs_by_molecule(
    pairs: pd.DataFrame, fraction: float, seed: int, boundary_pairs: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a molecule-disjoint validation pair set by holding out molecules.

    Parameters
    ----------
    pairs
        kNN pair table for the train split.
    fraction
        Fraction of molecules to reserve for validation.
    seed
        RNG seed for the molecule partition.
    boundary_pairs
        If False (default, and the original behaviour) a validation pair needs BOTH
        endpoints held out, and pairs straddling the boundary are discarded. If True, a
        pair with EITHER endpoint held out becomes a validation pair. Training pairs
        require neither endpoint held out in both modes, so no training pair ever touches
        a validation molecule; only the validation set becomes more permissive. Use this
        on sparse pair graphs (MMP), where the strict rule can leave <10 validation pairs.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train_pairs, val_pairs)``, sharing no molecule. Pairs straddling the boundary
        are dropped, which is the price of disjointness on a densely connected kNN graph.

    Notes
    -----
    A connected-component split cannot be used: kNN graphs are typically a single
    component, so component-based partitioning assigns every pair to one side.
    """
    mols = np.unique(np.concatenate([pairs["molecule_a_idx"].to_numpy(),
                                     pairs["molecule_b_idx"].to_numpy()]))
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(len(mols) * fraction)))
    held = set(rng.choice(mols, size=n_hold, replace=False).tolist())

    a_in = pairs["molecule_a_idx"].isin(held)
    b_in = pairs["molecule_b_idx"].isin(held)
    val = pairs[(a_in | b_in)] if boundary_pairs else pairs[a_in & b_in]
    train = pairs[~a_in & ~b_in]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def load_molecules(dataset: str, split: str) -> pd.DataFrame | None:
    """Read one split's molecule table, or None if it is absent/empty."""
    _, _, mol_file = DATASETS[dataset]
    path = os.path.join(SRC_ROOT, dataset, split, mol_file)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df if len(df) else None


def build(dataset: str, k: int, dry_run: bool, boundary_pairs: bool = False) -> None:
    """Build and write kNN pairs for every split of one MoleculeNet dataset."""
    target, direction, _ = DATASETS[dataset]
    print("=" * 96)
    print(f"{dataset}   k={k}   target={target}   direction={direction}")
    print("=" * 96)

    pending_val_pairs: pd.DataFrame | None = None
    pending_val_mols: pd.DataFrame | None = None

    for split in SPLITS:
        mols = load_molecules(dataset, split)

        # BBBP/HIV have no valid split; use the holdout carved from train rather than
        # falling back to test the way the existing *_both configs do.
        if split == "valid" and mols is None:
            if pending_val_pairs is None:
                print(f"  [{split}] no valid split and no train holdout available; skipping")
                continue
            mols = pending_val_mols
            pairs = pending_val_pairs
            print(f"  [{split}] synthesised from a train holdout: {len(pairs)} pairs "
                  f"(molecule-disjoint from train)")
            if not dry_run:
                dst = os.path.join(DST_ROOT, f"k{k}", dataset, split)
                os.makedirs(dst, exist_ok=True)
                mols.to_parquet(os.path.join(dst, f"molecules_{dataset}.parquet"), index=False)
                pairs.to_parquet(os.path.join(dst, f"pairs_{dataset}.parquet"), index=False)
            continue

        if mols is None:
            print(f"  [{split}] missing, skipping")
            continue

        # Drop multi-component (salt) records before pairing; reset molecule_idx so it
        # keeps matching the row position the pairs index into.
        disc = mols["smiles"].str.contains(r"\.")
        if int(disc.sum()):
            print(f"  [{split}] dropped {int(disc.sum())} disconnected (salt) records")
        mols = mols[~disc].reset_index(drop=True)

        fps, valid = fingerprints(mols["smiles"].tolist())
        pairs_list = knn_pairs(fps, k=k, min_similarity=None)

        y = mols[target].to_numpy()
        a = np.array([p[0] for p in pairs_list])
        b = np.array([p[1] for p in pairs_list])
        sim = np.array([p[2] for p in pairs_list])
        out = pd.DataFrame({
            "molecule_a_idx": a,
            "molecule_b_idx": b,
            "similarity": sim,
            "morgan_tanimoto_distance": 1.0 - sim,
            "pair_type": "real_tan_knn",
            "seed_y": y[a],
            "target_y": y[b],
            "delta_y": y[b] - y[a],
        })

        # For datasets lacking a real valid split, reserve part of train now.
        if split == "train" and load_molecules(dataset, "valid") is None:
            out, val_pairs = split_pairs_by_molecule(
                out, VAL_HOLDOUT_FRACTION, VAL_HOLDOUT_SEED, boundary_pairs
            )
            pending_val_pairs, pending_val_mols = val_pairs, mols.copy()
            print(f"  [{split}] reserved {len(val_pairs)} pairs for validation")

        covered = len(set(a.tolist()) | set(b.tolist()))
        n_valid = int(valid.sum())
        print(f"  [{split}] molecules={len(mols):<5} pairs={len(out):<6} "
              f"coverage={covered}/{n_valid} ({100 * covered / max(n_valid, 1):.1f}%)  "
              f"median_sim={np.median(sim):.3f}")

        if not dry_run:
            # The MoleculeNet tables already carry RDKit descriptors; only compute them
            # when absent, otherwise descriptor_columns() appends duplicate column names
            # and the parquet write fails.
            DESC = ["logP", "SAS", "TPSA", "MW", "HBD", "HBA", "QED", "fracCSP3"]
            mols_out = mols if all(c in mols.columns for c in DESC) else descriptor_columns(mols)
            mols_out = mols_out.copy()
            mols_out["molecule_idx"] = range(len(mols_out))
            dst = os.path.join(DST_ROOT, f"k{k}", dataset, split)
            os.makedirs(dst, exist_ok=True)
            mols_out.to_parquet(os.path.join(dst, f"molecules_{dataset}.parquet"), index=False)
            out.to_parquet(os.path.join(dst, f"pairs_{dataset}.parquet"), index=False)

    if not dry_run:
        print(f"  wrote -> {os.path.join(DST_ROOT, f'k{k}', dataset)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 20],
                        help="neighbour counts to build (default: 5 20)")
    parser.add_argument("--dataset", default=None, help="build only this dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--val-boundary-pairs",
        action="store_true",
        help="validation pairs need only ONE held-out endpoint (default: both). Training "
             "stays molecule-disjoint from validation either way.",
    )
    args = parser.parse_args()

    for k in args.k:
        for ds in DATASETS:
            if args.dataset and ds != args.dataset:
                continue
            build(ds, k, args.dry_run, args.val_boundary_pairs)


if __name__ == "__main__":
    main()
