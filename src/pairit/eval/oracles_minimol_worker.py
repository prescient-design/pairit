"""Score SMILES with MiniMol, in a process of its own.

MiniMol cannot be imported alongside `pairit_eval`: graphium pins torch 2.3 and
numpy<2, ``Minimol()`` calls ``hydra.initialize`` without clearing global state so
it cannot be constructed twice, and `pairit_eval` itself depends on hydra-core.

So this module is never imported by `pairit_eval`. It is *executed*, by
:class:`pairit_eval.oracles.SubprocessOracle`, under the interpreter of the
MiniMol virtualenv:

    <minimol-venv>/bin/python -m pairit_eval.oracles_minimol_worker \\
        --smiles-file todo.txt --out preds.csv --dataset bace

It reads one SMILES per line, appends ``smiles,minimol_pred`` rows to ``--out``
for the ones it can embed, and leaves existing rows alone. Molecules that fail to
embed are simply absent from the output -- the caller treats an absent molecule as
unscorable and reports it, which is the honest outcome.

Run standalone to warm a cache:

    python -m pairit_eval.oracles_minimol_worker \\
        --smiles-file all.txt --out /path/to/bace.csv --dataset bace
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Where the per-dataset MiniMol heads live. Overridable so this is not welded to
# one filesystem. Defaults to assets/oracles/{moleculenet,tdc}/minimol/
# resolved relative to this file (parents[3] = the repository root), so the worker keeps
# working when pairit is not installed in the MiniMol venv.
_ASSETS_ROOT = Path(__file__).resolve().parents[3] / "assets" / "oracles"
_DEFAULT_HEAD_DIRS = [
    str(_ASSETS_ROOT / "moleculenet" / "minimol"),
    str(_ASSETS_ROOT / "tdc" / "minimol"),
]

# Dataset names differ between pairit_eval and the MiniMol head directories.
_HEAD_ALIASES = {
    "caco2_wang": ["Caco2_Wang", "caco2_wang"],
    "freesolv": ["FreeSolv", "freesolv"],
    "herg": ["hERG", "herg"],
    "bace": ["bace", "BACE"],
    "bbbp": ["bbbp", "BBBP"],
    "hiv": ["hiv", "HIV"],
}

_BATCH = 256


def head_dirs() -> List[str]:
    env = os.environ.get("PAIRIT_MINIMOL_HEAD_DIRS")
    return env.split(os.pathsep) if env else list(_DEFAULT_HEAD_DIRS)


def find_head(dataset: str) -> Path:
    """Locate a dataset's MiniMol head, trying both naming conventions."""
    names = _HEAD_ALIASES.get(dataset.lower(), [dataset])
    tried = []
    for root in head_dirs():
        for name in names:
            candidate = Path(root) / name / "head.joblib"
            tried.append(str(candidate))
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No MiniMol head for dataset {dataset!r}. Looked in:\n  " + "\n  ".join(tried)
        + "\nSet $PAIRIT_MINIMOL_HEAD_DIRS to point elsewhere."
    )


def embed(smiles: List[str]) -> Dict[str, "object"]:
    """Embed each SMILES. Failures are omitted rather than raising.

    Batched for speed, with a per-molecule retry so one unparseable molecule
    cannot take out its whole batch.
    """
    from minimol import Minimol

    model = Minimol()
    vecs: Dict[str, object] = {}
    for i in range(0, len(smiles), _BATCH):
        chunk = smiles[i : i + _BATCH]
        try:
            for s, v in zip(chunk, model(chunk)):
                vecs[s] = v
        except Exception:
            for s in chunk:
                try:
                    vecs[s] = model([s])[0]
                except Exception:
                    pass  # unembeddable -> stays absent -> reported as unscorable
        print(f"  embedded {min(i + _BATCH, len(smiles))}/{len(smiles)}", flush=True)
    return vecs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles-file", required=True, type=Path, help="one SMILES per line")
    ap.add_argument("--out", required=True, type=Path, help="smiles,minimol_pred CSV to extend")
    ap.add_argument("--dataset", required=True, help="selects which MiniMol head to use")
    ap.add_argument("--pred-col", default="minimol_pred")
    args = ap.parse_args(argv)

    import joblib
    import numpy as np
    import pandas as pd

    todo = [ln.strip() for ln in args.smiles_file.read_text().splitlines() if ln.strip()]
    if not todo:
        print("nothing to score")
        return 0

    # Never rescore what is already cached: this worker is called incrementally.
    existing = pd.DataFrame(columns=["smiles", args.pred_col])
    if args.out.exists():
        existing = pd.read_csv(args.out)
        known = set(existing["smiles"].astype(str))
        todo = [s for s in todo if s not in known]
        if not todo:
            print("all requested molecules are already cached")
            return 0

    print(f"{args.dataset}: scoring {len(todo)} new molecules")
    head = joblib.load(find_head(args.dataset))
    vecs = embed(todo)

    keys = [s for s in todo if s in vecs]
    if not keys:
        print("nothing could be embedded", file=sys.stderr)
        return 0
    X = np.array([np.asarray(vecs[s], dtype=float) for s in keys])
    pred = head.predict_proba(X)[:, 1] if hasattr(head, "predict_proba") else head.predict(X)

    fresh = pd.DataFrame({"smiles": keys, args.pred_col: np.asarray(pred, dtype=float)})
    combined = pd.concat([existing, fresh], ignore_index=True).drop_duplicates(
        subset="smiles", keep="last"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"-> {args.out}  (+{len(keys)} of {len(todo)}, {len(combined)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
