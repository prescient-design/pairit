"""Build PairIT-consumable copies of the TDC matched-pair datasets.

The upstream files (set via ``PAIRIT_TDC_SRC_ROOT``) are correct but not directly
trainable, for three reasons:

1. **Pair rows are massively duplicated.** There is one row per MMP *rule
   environment*, so the same ``(molecule_a_idx, molecule_b_idx)`` appears 6-618
   times with identical ``similarity``/``delta_y`` and differing
   ``rule_environment_id``. Inflation is 16-58x overall, and it is *uneven*
   across pairs, so training would silently upweight pairs that happen to have
   many rule environments.
2. **Both orderings are present.** 19-48% of unique ordered pairs also appear
   reversed. ``MoleculeImprovementLightningDataModule`` reorients every pair
   worse->better via ``utility_direction``, so ``(a, b)`` and ``(b, a)`` collapse
   to the *same* training example -- another 1.1-1.3x duplication.
3. **No distance column.** The pair files carry ``similarity``, but the
   datamodule's ``delta`` filter and distance prompting read a *distance*
   column. Passing ``similarity`` as ``distance_type`` would invert the filter.

This script writes deduplicated copies (unique *unordered* pairs), adds
``morgan_tanimoto_distance = 1 - similarity``, records the collapsed row count as
``n_mmp_rules``, and adds a dataset-specific alias for the ``Y`` target column so
the three datasets can coexist in one property registry / reference set without
colliding on the generic name ``Y``.

It also repairs the **empty Caco2_Wang validation pair file** by holding out a
molecule-disjoint slice of the train pairs (see ``VAL_HOLDOUT_FRACTION``). The 72
real Caco2 valid molecules have no pairs upstream and are left unused.

Run::

    python scripts/build_tdc_pair_datasets.py

Outputs land under ``$PAIRIT_TDC_DST_ROOT/<dataset>/<split>/`` (default
``datasets/tdc/<dataset>/<split>/``) mirroring the upstream filenames.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SRC_ROOT = os.environ.get(
    "PAIRIT_TDC_SRC_ROOT",
    "datasets/tdc_raw",
)
DST_ROOT = os.environ.get(
    "PAIRIT_TDC_DST_ROOT",
    "datasets/tdc",
)
SPLITS = ("train", "valid", "test")

# Fraction of train *seed molecules* held out to build a validation pair set for
# datasets whose upstream valid pair file is empty.
VAL_HOLDOUT_FRACTION = 0.15
VAL_HOLDOUT_SEED = 0


@dataclass
class DatasetSpec:
    """Per-dataset target semantics.

    Parameters
    ----------
    name
        Directory / filename stem used upstream (e.g. ``"Caco2_Wang"``).
    target_alias
        Dataset-specific duplicate of the generic ``Y`` column. Keeps the three
        TDC targets distinguishable in ``PROPERTY_REGISTRY``, reference SMILES
        files, and mixed-dataset training.
    utility_direction
        ``"increasing"`` if a higher target value is the improvement,
        ``"decreasing"`` if lower is.
    rationale
        Why that direction is the chemically sensible one. Printed by the script.
    needs_synthetic_val
        Whether the upstream valid pair file is empty and must be rebuilt from a
        train holdout.
    """

    name: str
    target_alias: str
    utility_direction: str
    rationale: str
    needs_synthetic_val: bool = False


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="Caco2_Wang",
        target_alias="caco2_papp",
        utility_direction="increasing",
        rationale=(
            "Y is log10 Caco-2 apparent permeability (cm/s). Higher = more "
            "permeable = better oral absorption."
        ),
        needs_synthetic_val=True,
    ),
    DatasetSpec(
        name="FreeSolv",
        target_alias="freesolv_dg",
        utility_direction="decreasing",
        rationale=(
            "Y is experimental hydration free energy (kcal/mol). More negative = "
            "more favourable hydration = more water-soluble."
        ),
    ),
    DatasetSpec(
        name="hERG",
        target_alias="herg_block",
        utility_direction="decreasing",
        rationale=(
            "Y is a binary hERG-blocker label (1 = blocker, cardiotoxic). Lower = "
            "safer, so the model should learn blocker -> non-blocker."
        ),
    ),
]


def _paths(root: str, ds: str, split: str) -> tuple[str, str]:
    """Return ``(molecules_path, pairs_path)`` for one dataset split."""
    d = os.path.join(root, ds, split)
    return (
        os.path.join(d, f"molecules_{ds}.parquet"),
        os.path.join(d, f"pairs_{ds}.parquet"),
    )


def dedupe_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Collapse MMP rule-environment rows to unique unordered molecule pairs.

    Parameters
    ----------
    pairs
        Raw upstream pair table. Must contain ``molecule_a_idx`` and
        ``molecule_b_idx``.

    Returns
    -------
    pandas.DataFrame
        One row per unordered ``{a, b}`` pair, with ``n_mmp_rules`` counting the
        collapsed rows and ``morgan_tanimoto_distance`` derived from
        ``similarity``. Column order otherwise preserved.

    Notes
    -----
    All duplicate rows of a given ``(a, b)`` carry identical ``similarity``,
    ``seed_y``, ``target_y`` and ``delta_y`` (verified on all three datasets), so
    keeping the first row loses nothing but the rule-environment identifiers.
    """
    if pairs.empty or "molecule_a_idx" not in pairs.columns:
        return pairs

    key = [frozenset((a, b)) for a, b in zip(pairs["molecule_a_idx"], pairs["molecule_b_idx"])]
    out = pairs.assign(_key=key)
    counts = out.groupby("_key", sort=False).size().rename("n_mmp_rules")
    out = out.drop_duplicates(subset="_key").merge(counts, left_on="_key", right_index=True)
    out = out.drop(columns="_key").reset_index(drop=True)

    if "similarity" in out.columns:
        out["morgan_tanimoto_distance"] = 1.0 - out["similarity"]
    return out


def split_train_pairs_for_val(
    pairs: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a molecule-disjoint validation pair set out of the train pairs.

    Parameters
    ----------
    pairs
        Deduplicated train pair table.
    fraction
        Target fraction of *pairs* to move into the validation set.
    seed
        RNG seed for the component shuffle.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train_pairs, val_pairs)``, sharing no molecule.

    Notes
    -----
    Splits on **connected components** of the pair graph rather than on
    individual molecules. Because components are disjoint by construction, no
    pair straddles the boundary and none has to be discarded -- a plain random
    molecule holdout drops every pair with one endpoint on each side, which on
    these sparse graphs costs ~28% of the training pairs.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in zip(pairs["molecule_a_idx"], pairs["molecule_b_idx"]):
        union(int(a), int(b))

    comp = np.array([find(int(a)) for a in pairs["molecule_a_idx"]])
    comp_ids = np.unique(comp)
    rng = np.random.default_rng(seed)
    rng.shuffle(comp_ids)

    target = int(round(len(pairs) * fraction))
    val_comps: set[int] = set()
    taken = 0
    for cid in comp_ids:
        if taken >= target:
            break
        val_comps.add(int(cid))
        taken += int((comp == cid).sum())

    val_mask = np.isin(comp, list(val_comps))
    return pairs[~val_mask].reset_index(drop=True), pairs[val_mask].reset_index(drop=True)


def assert_no_molecules_lost(spec: DatasetSpec) -> None:
    """Fail if any upstream molecule is absent from the written copy.

    This is a conservation check, not a correctness check. It exists because the
    Caco2 validation workaround below *replaces* the valid molecule table with the
    train one (the carved val pairs index into train), and an earlier version of
    this script let that replacement silently drop the 72 real upstream valid
    molecules. Verifying that the datamodule loads cannot catch a loss like that --
    only comparing the input and output row sets can.

    Raises
    ------
    RuntimeError
        If any upstream SMILES is missing from every output file for the dataset.
    """
    upstream: set[str] = set()
    for split in SPLITS:
        src_mol, _ = _paths(SRC_ROOT, spec.name, split)
        if os.path.exists(src_mol):
            upstream |= set(pd.read_parquet(src_mol)["smiles"])

    written: set[str] = set()
    dst_root = os.path.join(DST_ROOT, spec.name)
    for split in SPLITS:
        split_dir = os.path.join(dst_root, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in os.listdir(split_dir):
            if fn.startswith("molecules_") and fn.endswith(".parquet"):
                written |= set(pd.read_parquet(os.path.join(split_dir, fn))["smiles"])

    missing = upstream - written
    if missing:
        raise RuntimeError(
            f"{spec.name}: {len(missing)} upstream molecules are absent from the written "
            f"dataset. Writing must never discard source rows. Example: {sorted(missing)[0]}"
        )
    print(f"  conservation check: all {len(upstream)} upstream molecules present in output")


def build(spec: DatasetSpec, dry_run: bool = False) -> None:
    """Write the deduplicated, alias-annotated copy of one TDC dataset."""
    print("=" * 92)
    print(f"{spec.name}  ->  utility_direction={spec.utility_direction}")
    print(f"  {spec.rationale}")
    print("=" * 92)

    pending_val: pd.DataFrame | None = None

    for split in SPLITS:
        src_mol, src_pair = _paths(SRC_ROOT, spec.name, split)
        mols = pd.read_parquet(src_mol)
        pairs = pd.read_parquet(src_pair)

        # Dataset-specific alias for the generic `Y` target column.
        mols[spec.target_alias] = mols["Y"]

        n_raw = len(pairs)
        pairs = dedupe_pairs(pairs)

        if split == "train" and spec.needs_synthetic_val:
            pairs, pending_val = split_train_pairs_for_val(
                pairs, VAL_HOLDOUT_FRACTION, VAL_HOLDOUT_SEED
            )
            print(
                f"  [{split}] carved val holdout: {len(pending_val)} val pairs, "
                f"{len(pairs)} train pairs remain"
            )

        if split == "valid" and spec.needs_synthetic_val:
            assert pending_val is not None, "train split must be processed before valid"
            # Preserve the upstream valid molecules before overwriting the table below.
            # They have no pairs so they cannot be used for pair training, but they are
            # still the dataset's real validation split and are needed for seeds-only
            # evaluation -- do not let the holdout workaround silently discard them.
            if not dry_run:
                orphan_dir = os.path.join(DST_ROOT, spec.name, "valid")
                os.makedirs(orphan_dir, exist_ok=True)
                mols.to_parquet(
                    os.path.join(orphan_dir, f"molecules_{spec.name}_upstream_valid.parquet"),
                    index=False,
                )
            print(f"  [{split}] preserved {len(mols)} upstream valid molecules (no pairs) "
                  f"as molecules_{spec.name}_upstream_valid.parquet")

            pairs = pending_val
            # Val pairs index into the TRAIN molecule table, so ship that table.
            mols = pd.read_parquet(_paths(SRC_ROOT, spec.name, "train")[0])
            mols[spec.target_alias] = mols["Y"]
            print(f"  [{split}] upstream pair file empty -> using train holdout ({len(pairs)} pairs)")

        n_out = len(pairs)
        ratio = f"{n_raw / n_out:.1f}x" if n_out else "n/a"
        print(f"  [{split}] molecules={len(mols)}  pairs {n_raw} -> {n_out} (deduped {ratio})")

        if not dry_run:
            dst_dir = os.path.join(DST_ROOT, spec.name, split)
            os.makedirs(dst_dir, exist_ok=True)
            dst_mol, dst_pair = _paths(DST_ROOT, spec.name, split)
            mols.to_parquet(dst_mol, index=False)
            pairs.to_parquet(dst_pair, index=False)

    if not dry_run:
        assert_no_molecules_lost(spec)
        print(f"  wrote -> {os.path.join(DST_ROOT, spec.name)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing")
    parser.add_argument("--dataset", default=None, help="only build this dataset")
    args = parser.parse_args()

    for spec in DATASETS:
        if args.dataset and spec.name != args.dataset:
            continue
        build(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
