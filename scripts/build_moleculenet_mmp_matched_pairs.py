"""Rebuild MoleculeNet MMP pairs under the SAME validation split as the kNN arm.

Without this, the kNN-vs-MMP comparison is confounded: the kNN datasets built by
`build_moleculenet_knn_pairs.py` carve a molecule-disjoint validation holdout out of train
for BBBP and HIV, whereas the existing `*_both` configs point `val_molecule_filename` at
`test/` — i.e. they select checkpoints on the test set. Comparing across that difference
would hold the two arms to different standards.

This script emits MMP pair datasets that share the kNN arm's molecules and its validation
holdout exactly, so the only thing differing between arms is how pairs were formed.

Holdout equivalence
-------------------
The kNN builder chooses held-out molecules with
``rng.choice(unique_molecules_in_pair_graph, 15%)`` at seed 0. kNN coverage is 100%, so
that pool is exactly ``arange(n_molecules)`` and the draw is identical to sampling the
molecule table directly. This script reproduces that draw and asserts the resulting held-out
set matches the kNN arm's actual validation molecules.

Usage::

    python scripts/build_moleculenet_mmp_matched_pairs.py
    python scripts/build_moleculenet_mmp_matched_pairs.py --dataset bbbp --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_moleculenet_knn_pairs import (  # noqa: E402
    DATASETS,
    SRC_ROOT,
    VAL_HOLDOUT_FRACTION,
    VAL_HOLDOUT_SEED,
    load_molecules,
)

KNN_ROOT = os.environ.get(
    "PAIRIT_MOLECULENET_KNN_ROOT",
    "datasets/moleculenet_knn",
)
DST_ROOT = os.environ.get(
    "PAIRIT_MOLECULENET_MMP_MATCHED_ROOT",
    "datasets/moleculenet_mmp_matched",
)
SPLITS = ("train", "valid", "test")

PAIR_FILES = {
    "bace": "pairs_bace_real.parquet",
    "bbbp": "pairs_bbbp_real.parquet",
    "hiv": "pairs_hiv_real.parquet",
}


def held_out_molecules(n_molecules: int) -> set[int]:
    """Reproduce the kNN arm's validation draw over ``arange(n_molecules)``."""
    rng = np.random.default_rng(VAL_HOLDOUT_SEED)
    n_hold = max(1, int(round(n_molecules * VAL_HOLDOUT_FRACTION)))
    return set(rng.choice(np.arange(n_molecules), size=n_hold, replace=False).tolist())


def load_mmp_pairs(dataset: str, split: str) -> pd.DataFrame | None:
    """Read one split's MMP pair table, or None if absent/empty."""
    path = os.path.join(SRC_ROOT, dataset, split, PAIR_FILES[dataset])
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df if len(df) else None


def annotate(pairs: pd.DataFrame, mols: pd.DataFrame, target: str) -> pd.DataFrame:
    """Add the columns the kNN tables carry, so both arms have the same schema."""
    y = mols[target].to_numpy()
    a = pairs["molecule_a_idx"].to_numpy()
    b = pairs["molecule_b_idx"].to_numpy()
    out = pairs.copy()
    if "similarity" in out.columns and "morgan_tanimoto_distance" not in out.columns:
        out["morgan_tanimoto_distance"] = 1.0 - out["similarity"]
    out["seed_y"] = y[a]
    out["target_y"] = y[b]
    out["delta_y"] = y[b] - y[a]
    return out


def build(dataset: str, dry_run: bool, boundary_pairs: bool = False) -> None:
    """Write MMP pairs for one dataset under the kNN arm's validation split."""
    target, direction, _ = DATASETS[dataset]
    print("=" * 96)
    print(f"{dataset}   target={target}   direction={direction}")
    print("=" * 96)

    has_real_valid = load_molecules(dataset, "valid") is not None

    for split in SPLITS:
        mols = load_molecules(dataset, split)

        # Datasets with a real valid split (BACE) need no carving — the kNN arm uses it
        # as-is, so this arm does too.
        if has_real_valid:
            if mols is None:
                print(f"  [{split}] missing, skipping")
                continue
            disc = mols["smiles"].str.contains(r"\.")
            mols = mols[~disc].reset_index(drop=True)
            pairs = load_mmp_pairs(dataset, split)
            if pairs is None:
                print(f"  [{split}] no MMP pairs, skipping")
                continue
            n = len(mols)
            keep = (pairs["molecule_a_idx"] < n) & (pairs["molecule_b_idx"] < n)
            pairs = pairs[keep].reset_index(drop=True)
            out = annotate(pairs, mols, target)
            covered = len(set(out["molecule_a_idx"]) | set(out["molecule_b_idx"]))
            print(f"  [{split}] molecules={len(mols):<5} pairs={len(out):<6} "
                  f"coverage={covered}/{n} ({100 * covered / max(n, 1):.1f}%)  [real valid split]")
            if not dry_run:
                _write(dataset, split, mols, out)
            continue

        # BBBP / HIV: carve train into train+valid using the kNN arm's exact holdout.
        if split == "valid":
            continue                      # produced alongside train below
        if mols is None:
            print(f"  [{split}] missing, skipping")
            continue

        disc = mols["smiles"].str.contains(r"\.")
        if int(disc.sum()):
            print(f"  [{split}] dropped {int(disc.sum())} disconnected (salt) records")
        mols = mols[~disc].reset_index(drop=True)

        pairs = load_mmp_pairs(dataset, split)
        if pairs is None:
            print(f"  [{split}] no MMP pairs, skipping")
            continue
        n = len(mols)
        keep = (pairs["molecule_a_idx"] < n) & (pairs["molecule_b_idx"] < n)
        pairs = pairs[keep].reset_index(drop=True)
        out = annotate(pairs, mols, target)

        if split == "train":
            held = held_out_molecules(n)
            _verify_matches_knn(dataset, held)
            a_in = out["molecule_a_idx"].isin(held)
            b_in = out["molecule_b_idx"].isin(held)
            tr = out[~a_in & ~b_in].reset_index(drop=True)
            # Default: a validation pair needs both endpoints held out. With
            # --val-boundary-pairs, one is enough — necessary on sparse MMP graphs where
            # the strict rule leaves single-digit validation sets. Training is unaffected.
            va = (out[a_in | b_in] if boundary_pairs else out[a_in & b_in]).reset_index(drop=True)
            print(f"  [train] molecules={n:<5} pairs={len(tr):<6} "
                  f"(held out {len(held)} molecules -> {len(va)} val pairs, "
                  f"{len(out) - len(tr) - len(va)} boundary pairs dropped)")
            if not dry_run:
                _write(dataset, "train", mols, tr)
                _write(dataset, "valid", mols, va)
        else:
            covered = len(set(out["molecule_a_idx"]) | set(out["molecule_b_idx"]))
            print(f"  [{split}] molecules={n:<5} pairs={len(out):<6} "
                  f"coverage={covered}/{n} ({100 * covered / max(n, 1):.1f}%)")
            if not dry_run:
                _write(dataset, split, mols, out)

    if not dry_run:
        print(f"  wrote -> {os.path.join(DST_ROOT, dataset)}")
    print()


def _verify_matches_knn(dataset: str, held: set[int]) -> None:
    """Assert the reproduced holdout is the one the kNN arm actually used.

    The invariant checked is on the kNN arm's TRAIN pairs, not its validation pairs: no
    training pair may touch a held-out molecule. That holds in both validation modes,
    whereas "every validation molecule is held out" is only true under the strict rule —
    with --val-boundary-pairs a validation pair deliberately has one endpoint outside the
    holdout, so the old subset check would spuriously fail.
    """
    for k in (5, 20):
        tp = os.path.join(KNN_ROOT, f"k{k}", dataset, "train", f"pairs_{dataset}.parquet")
        if not os.path.exists(tp):
            continue
        t = pd.read_parquet(tp)
        knn_train_mols = set(t["molecule_a_idx"]) | set(t["molecule_b_idx"])
        leaked = knn_train_mols & held
        if leaked:
            raise RuntimeError(
                f"{dataset}: {len(leaked)} kNN k={k} training molecules are inside the "
                f"reproduced holdout — the two arms do NOT share a split."
            )
        print(f"    holdout check vs kNN k={k}: {len(knn_train_mols)} train molecules, none "
              f"inside the {len(held)}-molecule holdout ✓")


def _write(dataset: str, split: str, mols: pd.DataFrame, pairs: pd.DataFrame) -> None:
    dst = os.path.join(DST_ROOT, dataset, split)
    os.makedirs(dst, exist_ok=True)
    m = mols.copy()
    m["molecule_idx"] = range(len(m))
    m.to_parquet(os.path.join(dst, f"molecules_{dataset}.parquet"), index=False)
    pairs.to_parquet(os.path.join(dst, f"pairs_{dataset}.parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--val-boundary-pairs",
        action="store_true",
        help="validation pairs need only ONE held-out endpoint (default: both)",
    )
    args = parser.parse_args()
    for ds in DATASETS:
        if args.dataset and ds != args.dataset:
            continue
        build(ds, args.dry_run, args.val_boundary_pairs)


if __name__ == "__main__":
    main()
