"""
Predict BACE, BBBP, HIV activity on zinc250k train molecules using .joblib classifiers.

Saves assets/data/zinc/train/zinc_train_predicted_labels.parquet with columns:
  smiles, logp, Class_pred, Class_prob, p_np_pred, p_np_prob, HIV_active_pred, HIV_active_prob

Run once before build_nn_lookup.py when using the zinc-augmented lookup variant
(nn-real-only-zinc).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from pairit._assets import ASSETS

DEFAULT_ZINC_TRAIN = ASSETS / "data" / "zinc" / "train" / "molecules.parquet"
DEFAULT_EVALUATOR_DIR = ASSETS / "oracles" / "moleculenet" / "evaluators"
DEFAULT_OUT = ASSETS / "data" / "zinc" / "train" / "zinc_train_predicted_labels.parquet"

TASKS = [
    ("bace_b", "Class"),
    ("bbbp_b", "p_np"),
    ("hiv_b", "HIV_active"),
]
NBITS = 2048
RADIUS = 2
THRESHOLD = 0.5


def _morgan_fps(smiles_list: list[str]) -> tuple[list[int], np.ndarray]:
    valid_idx, arrs = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
        arr = np.zeros(NBITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        valid_idx.append(i)
        arrs.append(arr)
    return valid_idx, np.vstack(arrs) if arrs else np.empty((0, NBITS), dtype=np.float32)


def predict(smiles_list: list[str], evaluator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, labels) arrays aligned to smiles_list, NaN for invalid."""
    probs = np.full(len(smiles_list), np.nan)
    labels = np.full(len(smiles_list), np.nan)
    valid_idx, fps = _morgan_fps(smiles_list)
    if len(valid_idx) == 0:
        return probs, labels
    p = evaluator.predict_proba(fps)[:, 1]
    for j, i in enumerate(valid_idx):
        probs[i] = p[j]
        labels[i] = 1.0 if p[j] >= THRESHOLD else 0.0
    return probs, labels


def main():
    parser = argparse.ArgumentParser(description="Predict BACE/BBBP/HIV on zinc train molecules.")
    parser.add_argument("--zinc-train", default=str(DEFAULT_ZINC_TRAIN))
    parser.add_argument("--evaluator-dir", default=str(DEFAULT_EVALUATOR_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    zinc_path = Path(args.zinc_train)
    eval_dir = Path(args.evaluator_dir)
    out_path = Path(args.out)

    print(f"Loading zinc train from {zinc_path} ...")
    df = pd.read_parquet(zinc_path)
    train = df[df["split"] == "train"].copy().reset_index(drop=True)
    print(f"  {len(train)} train molecules")

    smiles_list = train["smiles"].tolist()
    result = train[["smiles", "logp"]].copy()

    for task_name, label_col in TASKS:
        jlib_path = eval_dir / f"{task_name}.joblib"
        print(f"Running {jlib_path} ...")
        evaluator = joblib.load(jlib_path)
        probs, labels = predict(smiles_list, evaluator)
        result[f"{label_col}_prob"] = probs
        result[f"{label_col}_pred"] = labels
        n_pos = int((labels == 1.0).sum())
        print(f"  {label_col}: {n_pos} / {len(train)} predicted class 1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
