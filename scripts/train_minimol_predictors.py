"""Train MoleculeNet MiniMol heads on training molecules only.

Adapted from the existing predictor-training workflow. MiniMol supplies the
public pretrained encoder; this script fits a logistic-regression head.
Run in the separate MiniMol environment. No trained weights are distributed.
"""
from __future__ import annotations
import os
import argparse
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

RDLogger.DisableLog("rdApp.*")

LABELS = {"bace": "Class", "bbbp": "p_np", "hiv": "HIV_active"}


def canon(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def load_dataset(path, y_col):
    frames = []
    for split in ("train", "valid", "test"):
        source = path / split / "molecules.parquet"
        if source.exists():
            frame = pd.read_parquet(source)
            if "molecule_type" in frame:
                frame = frame[frame["molecule_type"] == "real"]
            frame = frame[["smiles", y_col]].copy()
            frame["split"] = split
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No molecule tables under {path}")
    df = pd.concat(frames, ignore_index=True).dropna()
    df.columns = ["smi", "y", "split"]
    df["smiles"] = df["smi"].map(canon)
    return df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)


def main():
    assets = Path(os.environ.get("PAIRIT_ASSETS_DIR", "assets"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", type=Path, default=assets / "data/moleculenet")
    parser.add_argument("--out", type=Path, default=assets / "oracles/moleculenet/minimol")
    parser.add_argument("--dataset", choices=sorted(LABELS), action="append")
    args = parser.parse_args()
    datasets = args.dataset or list(LABELS)
    for dataset in datasets:
        if (args.out / dataset.upper() / "head.joblib").exists():
            parser.error(f"Existing predictor for {dataset}; choose a new --out directory")
    # Import here so any import errors are visible per-run rather than at module load
    print("Importing MiniMol...", flush=True)
    from minimol import Minimol
    model = Minimol()
    print("MiniMol loaded.", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        name, kind = dataset.upper(), "binary"
        path, y_col = args.molecules / dataset, LABELS[dataset]
        t0 = time.time()
        print(f"\n=== {name} ({kind}) ===", flush=True)
        df = load_dataset(path, y_col)
        splits = df["split"].value_counts().to_dict()
        print(f"  loaded {len(df)} unique canonical SMILES (splits: {splits})", flush=True)

        # Encode in batches to control memory; MiniMol accepts list of SMILES
        smis = df["smiles"].tolist()
        y = df["y"].astype(float).to_numpy()
        splits_arr = df["split"].to_numpy()

        BATCH = 64
        emb_list = []
        for i in range(0, len(smis), BATCH):
            batch = smis[i:i + BATCH]
            out = model(batch)
            # MiniMol returns a list of 1-D tensors OR a 2-D tensor; normalize
            if isinstance(out, list):
                arr = np.stack([e.detach().cpu().numpy() for e in out])
            else:
                arr = out.detach().cpu().numpy()
            emb_list.append(arr)
            if (i // BATCH) % 5 == 0:
                print(f"    encoded {i + len(batch)}/{len(smis)}  "
                      f"({(time.time()-t0):.1f}s)", flush=True)
        emb = np.vstack(emb_list).astype(np.float32)
        print(f"  encoded all; embedding matrix {emb.shape}  "
              f"({(time.time()-t0):.1f}s total)", flush=True)

        # Fit head on train, evaluate on valid/test
        train_mask = splits_arr == "train"
        Xtr, ytr = emb[train_mask], y[train_mask]

        metrics = {}
        if kind == "binary":
            head = LogisticRegression(max_iter=2000, C=1.0)
            head.fit(Xtr, ytr.astype(int))
        else:
            head = Ridge(alpha=1.0)
            head.fit(Xtr, ytr)

        for sp in ("train", "valid", "test"):
            m = splits_arr == sp
            if m.sum() == 0:
                continue
            X, yy = emb[m], y[m]
            if kind == "binary":
                p = head.predict_proba(X)[:, 1]
                try:
                    auc = float(roc_auc_score(yy.astype(int), p))
                except ValueError:
                    auc = None
                metrics[sp] = {"roc_auc": auc, "n": int(m.sum())}
            else:
                yhat = head.predict(X)
                rmse = float(np.sqrt(mean_squared_error(yy, yhat)))
                mae = float(mean_absolute_error(yy, yhat))
                metrics[sp] = {"rmse": rmse, "mae": mae, "n": int(m.sum())}

        # Save artifacts
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(head, out_dir / "head.joblib")
        np.save(out_dir / "train_embeddings.npy", Xtr)
        np.save(out_dir / "train_y.npy", ytr)
        with open(out_dir / "train_smiles.txt", "w") as f:
            f.write("\n".join([smis[i] for i in range(len(smis)) if train_mask[i]]))
        # save full embeddings + all-split labels so downstream can reuse without re-encoding
        np.save(out_dir / "all_embeddings.npy", emb)
        np.save(out_dir / "all_y.npy", y)
        with open(out_dir / "all_smiles.txt", "w") as f:
            f.write("\n".join(smis))
        with open(out_dir / "all_splits.txt", "w") as f:
            f.write("\n".join(splits_arr.tolist()))
        config = {
            "dataset": name,
            "task_kind": kind,
            "y_col": y_col,
            "source_type": "prepared_moleculenet_parquet",
            "source_path": str(path),
            "n_total": int(len(df)),
            "splits": splits,
            "head_type": type(head).__name__,
            "minimol_version": version("minimol"),
            "sklearn_version": version("scikit-learn"),
            "head_parameters": head.get_params(),
            "minimol_checkpoint": "minimol_v1",
            "embedding_dim": int(emb.shape[1]),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with open(out_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        print(f"  head fit + saved. Metrics: {metrics}", flush=True)
        print(f"  wrote {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
