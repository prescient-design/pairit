"""Train the paper's MoleculeNet random-forest predictors on train splits only.

Uses the existing RF oracle recipe: Morgan radius 2, 2048 bits, default sklearn
RandomForestClassifier with random_state=0. No trained weights are distributed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

LABELS = {"bace": "Class", "bbbp": "p_np", "hiv": "HIV_active"}


def ecfp4(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid prepared SMILES: {smiles}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    array = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def main():
    assets = Path(os.environ.get("PAIRIT_ASSETS_DIR", "assets"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", type=Path, default=assets / "data/moleculenet")
    parser.add_argument("--out", type=Path, default=assets / "oracles/moleculenet/evaluators")
    parser.add_argument("--dataset", choices=sorted(LABELS), action="append")
    args = parser.parse_args()
    datasets = args.dataset or list(LABELS)
    for dataset in datasets:
        if (args.out / f"{dataset}_b.joblib").exists():
            parser.error(f"Existing predictor for {dataset}; choose a new --out directory")
    RDLogger.DisableLog("rdApp.warning")
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        splits = {}
        for split in ("train", "valid", "test"):
            path = args.molecules / dataset / split / "molecules.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if "molecule_type" in frame:
                frame = frame[frame.molecule_type == "real"]
            frame = frame.dropna(subset=["smiles", LABELS[dataset]])
            splits[split] = (
                np.stack([ecfp4(smiles) for smiles in frame.smiles]),
                frame[LABELS[dataset]].to_numpy(dtype=int),
            )
        model = RandomForestClassifier(random_state=0, n_jobs=-1)
        model.fit(*splits["train"])
        metrics = {}
        for split, (features, labels) in splits.items():
            scores = model.predict_proba(features)[:, 1]
            metrics[split] = {
                "n": len(labels),
                "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                "acc_at_0.5": float(accuracy_score(labels, scores >= 0.5)),
            }
        joblib.dump(model, args.out / f"{dataset}_b.joblib")
        (args.out / f"{dataset}_b_metrics.yaml").write_text(yaml.safe_dump({
            "dataset": dataset, "sklearn_version": sklearn.__version__,
            "training_convention": "train only; ECFP4 2048 bits; RF(random_state=0)",
            "parameters": model.get_params(), "metrics": metrics,
        }))
        print(dataset, metrics, flush=True)


if __name__ == "__main__":
    main()
