"""Build Tanimoto k-nearest-neighbour molecular pairs for the TDC datasets.

Motivation
----------
The existing pair sets leave most molecules unpaired:

- **MMP** pairs only exist where a valid single-fragment transformation is found
  (40-76% molecule coverage).
- **Threshold Tanimoto** pairs (``similarity > 0.4``) miss whole datasets whose
  molecules are small: FreeSolv's median pair similarity is 0.14, so a 0.4 cutoff
  finds almost nothing there.

A **k-nearest-neighbour** construction instead gives every molecule its k closest
neighbours regardless of absolute similarity, so *every* molecule is represented and
coverage is 100% by construction. This mirrors the canonical ZINC pair recipe
(``filter_by: morgan_tanimoto``, ``max_pairs_per_mol: 100``).

Pairs are built **within a split** so no pair straddles train/valid/test.

Usage
-----
::

    python scripts/build_tanimoto_knn_pairs.py --k 10
    python scripts/build_tanimoto_knn_pairs.py --k 5 --dataset hERG
    python scripts/build_tanimoto_knn_pairs.py --k 10 --min-similarity 0.2

Input molecule tables are read from ``$PAIRIT_TDC_PAIRS_SRC/<dataset>/<split>/``.
Output goes to ``<DST_ROOT>/<dataset>/<split>/`` with the same filenames the
datamodule expects, plus recomputed RDKit descriptor columns (the upstream molecule
tables dropped them) and a dataset-specific alias for the ``Y`` target.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED

from pairit._assets import ASSETS
from pairit.callbacks import sascorer

RDLogger.DisableLog("rdApp.*")

SRC_ROOT = os.environ.get(
    "PAIRIT_TDC_PAIRS_SRC",
    "<unset:PAIRIT_TDC_PAIRS_SRC>",
)
DST_ROOT = os.environ.get(
    "PAIRIT_TDC_KNN_ROOT",
    "datasets/tdc_knn",
)
SPLITS = ("train", "valid", "test")

# dataset -> (target alias column, improvement direction, oracle joblib)
DATASETS: dict[str, tuple[str, str, str]] = {
    "Caco2_Wang": ("caco2_papp", "increasing", "caco2_wang.joblib"),
    "FreeSolv": ("freesolv_dg", "decreasing", "freesolv.joblib"),
    "hERG": ("herg_block", "decreasing", "herg_b.joblib"),
}

ORACLE_DIR = os.environ.get(
    "TDC_EVALUATOR_DIR",
    str(ASSETS / "oracles" / "tdc" / "evaluators"),
)

RADIUS = 2
NBITS = 2048


def strip_salts(smiles: str) -> str | None:
    """Reduce a multi-component SMILES to its largest fragment.

    A minority of TDC records are salts stored as dot-separated components — e.g.
    hERG's CLOFILIUM PHOSPHATE is three clofilium cations plus a phosphate anion in
    one record, which RDKit renders as several disconnected molecules and which
    inflates its whole-record descriptors to MW 1111.9 / logP 16.64 against the
    active species' 339.0 / 6.49.

    Parameters
    ----------
    smiles
        Possibly multi-component SMILES.

    Returns
    -------
    str or None
        Canonical SMILES of the largest fragment by heavy-atom count, or ``None``
        if the input does not parse. Single-component inputs are returned
        canonicalised and otherwise unchanged.

    Notes
    -----
    The measured ``Y`` is kept as-is: it was determined for the salt as dosed, but
    the property is attributed to the active species. Note also that the RF oracles
    were fit on the *unstripped* SMILES, so stripping shifts what the oracle sees for
    these records.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return Chem.MolToSmiles(mol)
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)


def fingerprints(smiles: list[str]) -> tuple[list, np.ndarray]:
    """Compute ECFP4 bit vectors.

    Parameters
    ----------
    smiles
        SMILES strings, one per molecule.

    Returns
    -------
    tuple
        ``(fps, valid)`` where ``fps`` holds an RDKit bit vector (or ``None``) per
        input and ``valid`` is a boolean mask of parseable molecules.
    """
    fps, valid = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(None)
            valid.append(False)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS))
            valid.append(True)
    return fps, np.array(valid)


def knn_pairs(fps: list, k: int, min_similarity: float | None) -> list[tuple[int, int, float]]:
    """Return unique unordered pairs joining each molecule to its ``k`` nearest others.

    Parameters
    ----------
    fps
        ECFP4 bit vectors, ``None`` for unparseable molecules.
    k
        Neighbours to keep per molecule.
    min_similarity
        Optional floor. Neighbours below it are dropped, which can leave a molecule
        with fewer than ``k`` partners (or none) — set to ``None`` to guarantee full
        coverage.

    Returns
    -------
    list of tuple
        ``(idx_a, idx_b, similarity)`` with ``idx_a < idx_b``, deduplicated. The kNN
        relation is asymmetric, so taking the union of both directions is what makes
        every molecule appear at least once.

    Notes
    -----
    Brute-force O(n^2) similarity. The TDC splits are at most a few hundred
    molecules, so this takes well under a second and avoids a neighbour-index
    dependency.
    """
    idx = [i for i, f in enumerate(fps) if f is not None]
    out: dict[tuple[int, int], float] = {}

    for i in idx:
        others = [j for j in idx if j != i]
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in others])
        order = np.argsort(sims)[::-1][:k]
        for pos in order:
            j, sim = others[pos], float(sims[pos])
            if min_similarity is not None and sim < min_similarity:
                continue
            key = (i, j) if i < j else (j, i)
            # identical pair may arrive from either endpoint; similarity is symmetric
            out[key] = sim

    return [(a, b, s) for (a, b), s in out.items()]


def descriptor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the RDKit descriptors the upstream molecule tables dropped."""
    rows = []
    for s in df["smiles"]:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append({})
            continue
        rows.append(
            {
                "logP": Crippen.MolLogP(mol),
                "SAS": sascorer.calculateScore(mol),
                "TPSA": Descriptors.TPSA(mol),
                "MW": Descriptors.MolWt(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "QED": QED.qed(mol),
                "fracCSP3": Descriptors.FractionCSP3(mol),
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def oracle_probabilities(smiles: list[str], joblib_name: str) -> np.ndarray | None:
    """Score molecules with the dataset's RF oracle, returning None if unavailable.

    Provides a *continuous* signal for datasets whose measured target is binary
    (hERG). The oracle was fit on the full source CSV, so these values are
    memorisation-inflated on molecules from that CSV — useful for ranking within a
    pair, not as a substitute for a measured value.
    """
    path = os.path.join(ORACLE_DIR, joblib_name)
    if not os.path.exists(path):
        print(f"    (oracle {joblib_name} not found; skipping probability column)")
        return None
    from joblib import load

    model = load(path)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(bv, arr)
        fps.append(arr)
    X = np.array(fps)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def build(dataset: str, k: int, min_similarity: float | None, dry_run: bool,
          salt_strip: bool = False, drop_disconnected: bool = False,
          train_median: float | None = None,
          dst_root: str = DST_ROOT) -> float | None:
    """Build and write kNN pairs for every split of one dataset.

    Returns the train median of ``Y``, used to threshold the binary representation
    consistently across all splits.
    """
    alias, direction, joblib_name = DATASETS[dataset]
    print("=" * 92)
    print(f"{dataset}   k={k}   min_similarity={min_similarity}   direction={direction}")
    print("=" * 92)

    for split in SPLITS:
        mol_path = os.path.join(SRC_ROOT, dataset, split, f"molecules_{dataset}.parquet")
        if not os.path.exists(mol_path):
            print(f"  [{split}] no molecule file, skipping")
            continue

        mols = pd.read_parquet(mol_path)
        if drop_disconnected:
            # Drop multi-component records (salts, counterions) outright. Indices are
            # reset before any pairing, so positional molecule_a_idx / molecule_b_idx
            # in the emitted pairs refer to the surviving table.
            disc = mols["smiles"].str.contains(r"\.")
            if int(disc.sum()):
                print(f"  [{split}] dropped {int(disc.sum())} disconnected records: "
                      + ", ".join(str(x) for x in mols.loc[disc, "original_id"].head(4)))
            mols = mols[~disc].reset_index(drop=True)
        if salt_strip:
            n_multi = int(mols["smiles"].str.contains(r"\.").sum())
            stripped = [strip_salts(s) for s in mols["smiles"]]
            mols = mols.assign(smiles_original=mols["smiles"], smiles=stripped)
            mols = mols[mols["smiles"].notna()].reset_index(drop=True)
            if n_multi:
                print(f"  [{split}] salt-stripped {n_multi} multi-component records")
        fps, valid = fingerprints(mols["smiles"].tolist())
        pairs = knn_pairs(fps, k=k, min_similarity=min_similarity)

        y = mols["Y"].to_numpy()
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        sim = np.array([p[2] for p in pairs])

        out = pd.DataFrame(
            {
                "molecule_a_idx": a,
                "molecule_b_idx": b,
                "similarity": sim,
                "morgan_tanimoto_distance": 1.0 - sim,
                "pair_type": "real_tan_knn",
                "seed_y": y[a],
                "target_y": y[b],
                "delta_y": y[b] - y[a],
            }
        )

        covered = len(set(a.tolist()) | set(b.tolist()))
        n_valid = int(valid.sum())
        print(
            f"  [{split}] molecules={len(mols)} (valid {n_valid})  pairs={len(out)}  "
            f"coverage={covered}/{n_valid} ({100 * covered / max(n_valid, 1):.1f}%)  "
            f"pairs/mol={len(out) / max(n_valid, 1):.1f}  "
            f"median_sim={np.median(sim):.3f}"
        )

        if split == "train":
            train_median = float(np.median(mols["Y"]))

        if not dry_run:
            mols_out = descriptor_columns(mols)
            # Rows were dropped above, so the inherited molecule_idx is stale. The
            # MoleculeNet PairIT tables keep molecule_idx == positional index, and the
            # datamodule resolves pairs positionally, so re-derive it to match.
            mols_out["molecule_idx"] = range(len(mols_out))
            # --- both target representations, always ---
            # continuous: measured Y (or the oracle probability where Y is binary)
            mols_out[alias] = mols_out["Y"]
            # binary: thresholded at the TRAIN median, same direction semantics as Y,
            # so `utility_direction` applies unchanged to either column.
            mols_out[f"{alias}_binary"] = (mols_out["Y"] >= train_median).astype(float)
            probs = oracle_probabilities(mols_out["smiles"].tolist(), joblib_name)
            if probs is not None:
                mols_out[f"{alias}_prob"] = probs
            dst = os.path.join(dst_root, dataset, split)
            os.makedirs(dst, exist_ok=True)
            mols_out.to_parquet(os.path.join(dst, f"molecules_{dataset}.parquet"), index=False)
            out.to_parquet(os.path.join(dst, f"pairs_{dataset}.parquet"), index=False)

    if not dry_run:
        print(f"  wrote -> {os.path.join(dst_root, dataset)}")
    print()
    return train_median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="neighbours per molecule (default 10)")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="optional similarity floor; omit to guarantee 100%% coverage",
    )
    parser.add_argument("--dataset", default=None, help="build only this dataset")
    parser.add_argument(
        "--dst-root",
        default=DST_ROOT,
        help="output root; point elsewhere when sweeping k so the shipped k=10 tree is kept",
    )
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing")
    parser.add_argument(
        "--strip-salts",
        action="store_true",
        help="reduce multi-component (salt) records to their largest fragment",
    )
    parser.add_argument(
        "--drop-disconnected",
        action="store_true",
        help="drop multi-component (salt) records entirely; indices are reset before pairing",
    )
    args = parser.parse_args()

    for ds in DATASETS:
        if args.dataset and ds != args.dataset:
            continue
        build(ds, args.k, args.min_similarity, args.dry_run, args.strip_salts,
              args.drop_disconnected, dst_root=args.dst_root)


if __name__ == "__main__":
    main()
