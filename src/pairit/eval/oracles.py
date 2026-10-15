"""Property oracles.

An oracle answers "is this molecule good?". `pairit_eval` historically had exactly
one oracle slot, which made two things impossible:

1. **Filtering with one model and scoring with another.** The protocol we now use
   selects candidates with a random forest and judges them with MiniMol. With a
   single slot there is no way to say that.
2. **Datasets where improvement means a *lower* value.** hERG and FreeSolv are
   `decreasing` -- for hERG, improvement is blocker -> non-blocker. The binary
   improvement path hardcoded "predicted label 1 is good", which scores those
   datasets backwards.

So an oracle here is not just a model: it carries the `direction`, the
`threshold`, and the comparison operator used at that threshold, and it can
report its own provenance for the run record.

Implementations
---------------
FingerprintJoblibOracle
    A scikit-learn model on Morgan fingerprints, loaded from a ``.joblib``. This
    is the historical behaviour and delegates to
    :func:`property_metrics.batch_predict_binary_property` so the single-oracle
    path stays bit-identical.
PrecomputedTableOracle
    A ``smiles,prediction`` lookup table. For models that cannot be imported into
    this process.
SubprocessOracle
    A precomputed table that fills its own gaps by invoking a scorer under a
    different Python interpreter. This is how MiniMol works without a manual
    pre-pass.
CallableOracle
    Wraps any function. Mainly for tests.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from rdkit import Chem

from pairit._assets import ASSETS
from . import property_metrics
from .config import DATASET_CONFIG

INCREASING = "increasing"
DECREASING = "decreasing"
_DIRECTIONS = (INCREASING, DECREASING)

# `higher`/`lower` is the vocabulary PropertySpec already uses for continuous
# properties. Accept both spellings so a config written either way works.
_DIRECTION_ALIASES = {
    "increasing": INCREASING,
    "higher": INCREASING,
    "max": INCREASING,
    "decreasing": DECREASING,
    "lower": DECREASING,
    "min": DECREASING,
}


# Canonicalization is the hot path when a lookup table is large, and the same
# SMILES recurs across seeds; memoize it.
_CANON_CACHE: Dict[str, str] = {}


def normalize_direction(direction: Optional[str]) -> str:
    """Map any accepted spelling of a direction onto `increasing`/`decreasing`."""
    if direction is None:
        return INCREASING
    key = str(direction).strip().lower()
    if key not in _DIRECTION_ALIASES:
        raise ValueError(
            f"Unknown direction {direction!r}. Expected one of "
            f"{sorted(set(_DIRECTION_ALIASES))}."
        )
    return _DIRECTION_ALIASES[key]


def _sha256(path: Union[str, Path], limit: Optional[int] = None) -> Optional[str]:
    """Hash a file. `limit` caps bytes read, for files too large to hash cheaply."""
    try:
        h = hashlib.sha256()
        read = 0
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                read += len(chunk)
                if limit is not None and read >= limit:
                    break
        return h.hexdigest()
    except OSError:
        return None


class Oracle:
    """Base class. Subclasses implement :meth:`predict`.

    Parameters
    ----------
    name
        Short identifier used in column suffixes and the run record.
    direction
        `increasing` (higher prediction is better) or `decreasing`.
    threshold
        Decision boundary separating "good" from "not good".
    comparison
        ``">="`` or ``">"``. Only matters at an exact tie, which is not rare with
        a finite forest -- 500 trees splitting 250/250 gives exactly 0.5.
    """

    def __init__(
        self,
        name: str,
        direction: str = INCREASING,
        threshold: float = 0.5,
        comparison: str = ">=",
    ) -> None:
        if comparison not in (">=", ">"):
            raise ValueError(f"comparison must be '>=' or '>', got {comparison!r}")
        self.name = name
        self.direction = normalize_direction(direction)
        self.threshold = float(threshold)
        self.comparison = comparison
        self.n_scored = 0
        self.n_unscorable = 0

    # -- subclass hook ----------------------------------------------------
    def predict(self, smiles: Sequence[str]) -> List[Optional[float]]:
        """Return one score per input, or None where the molecule is unscorable."""
        raise NotImplementedError

    # -- shared behaviour -------------------------------------------------
    def label(self, score: Optional[float]) -> Optional[int]:
        """Turn a score into a raw class label (1 if score >= threshold else 0).

        This returns the RAW class prediction, NOT a direction-adjusted "is it good"
        label. Direction handling lives in ``compute_improvement`` (which knows the
        seed class comes from raw parquet, also un-adjusted). If this method ever
        inverts for `decreasing`, both `predicted_label` and `improvement` end up
        inverted for hERG relative to seed_label and the MMP baseline convention,
        and every hERG method's V,I,N score flips (blockers get counted as
        improvements).
        """
        if score is None or pd.isna(score):
            return None
        hit = score >= self.threshold if self.comparison == ">=" else score > self.threshold
        return 1 if hit else 0

    def predict_labels(self, smiles: Sequence[str]) -> List[Tuple[Optional[float], Optional[int]]]:
        """`(score, label)` per input -- the shape `_evaluate_one_block` consumes."""
        scores = self.predict(smiles)
        out: List[Tuple[Optional[float], Optional[int]]] = []
        n_scored = n_unscorable = 0
        for s in scores:
            if s is None or pd.isna(s):
                out.append((None, None))
                n_unscorable += 1
            else:
                out.append((float(s), self.label(s)))
                n_scored += 1
        self.n_scored += n_scored
        self.n_unscorable += n_unscorable
        return out

    def provenance(self) -> Dict[str, Any]:
        """What went into the run record. Subclasses extend, never replace."""
        return {
            "name": self.name,
            "kind": type(self).__name__,
            "direction": self.direction,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "n_scored": self.n_scored,
            "n_unscorable": self.n_unscorable,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} {self.direction} @{self.threshold}>"


class FingerprintJoblibOracle(Oracle):
    """A scikit-learn model over Morgan fingerprints, loaded from a joblib.

    Delegates scoring to :func:`property_metrics.batch_predict_binary_property`
    so that a run with a single oracle produces byte-identical output to the
    pre-oracle code.
    """

    def __init__(
        self,
        model: Any,
        name: str,
        path: Optional[Union[str, Path]] = None,
        direction: str = INCREASING,
        threshold: float = 0.5,
        comparison: str = ">=",
        nbits: int = 2048,
        radius: int = 2,
    ) -> None:
        super().__init__(name, direction=direction, threshold=threshold, comparison=comparison)
        self.model = model
        self.path = Path(path) if path is not None else None
        self.nbits = nbits
        self.radius = radius

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "FingerprintJoblibOracle":
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Oracle not found: {path}")
        return cls(joblib.load(path), name=name or path.stem, path=path, **kwargs)

    def predict(self, smiles: Sequence[str]) -> List[Optional[float]]:
        # Binary oracles expose predict_proba and we want P(active); regression oracles
        # (the continuous TDC tasks -- FreeSolv hydration free energy, Caco2 permeability)
        # do not, and calling the binary path on them raises AttributeError. Dispatch on
        # the fitted estimator rather than on config, so a joblib that disagrees with
        # DATASET_CONFIG fails loudly at load rather than silently scoring the wrong thing.
        if hasattr(self.model, "predict_proba"):
            pairs = property_metrics.batch_predict_binary_property(
                list(smiles), self.model, nbits=self.nbits, radius=self.radius,
                threshold=self.threshold,
            )
            return [p for p, _ in pairs]
        return self._predict_continuous(list(smiles))

    def _predict_continuous(self, smiles: List[str]) -> List[Optional[float]]:
        """Regression oracle: the predicted property value, None where RDKit fails.

        Unparseable SMILES yield None rather than a sentinel number -- a fabricated value
        would be selected on and would move the reported rate.
        """
        import numpy as np
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.nbits)
        out: List[Optional[float]] = [None] * len(smiles)
        fps, where = [], []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            # Use DataStructs.ConvertToNumpyArray, never np.asarray(fingerprint):
            # passing an RDKit ExplicitBitVect through np.asarray corrupts memory
            # under numpy 2.x, segfaulting later with no traceback rather than raising.
            arr = np.zeros((self.nbits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
            fps.append(arr)
            where.append(i)
        if fps:
            for i, value in zip(where, self.model.predict(np.vstack(fps))):
                out[i] = float(value)
        return out

    def provenance(self) -> Dict[str, Any]:
        info = super().provenance()
        info.update(
            {
                "path": str(self.path) if self.path else None,
                "sha256": _sha256(self.path) if self.path else None,
                "model_class": type(self.model).__name__,
                "featurizer": f"morgan_r{self.radius}_{self.nbits}bits",
            }
        )
        n_features = getattr(self.model, "n_features_in_", None)
        if n_features is not None:
            info["n_features_in"] = int(n_features)
        return info


class PrecomputedTableOracle(Oracle):
    """Scores looked up from a ``smiles,prediction`` table.

    For models that cannot run in this process. A molecule missing from the table
    is **unscorable**, and stays visibly so -- it is counted in `n_unscorable` and
    reported, never silently turned into a passing or failing molecule.
    """

    def __init__(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        smiles_col: str = "smiles",
        pred_col: Optional[str] = None,
        direction: str = INCREASING,
        threshold: float = 0.5,
        comparison: str = ">=",
        canonicalize: bool = True,
    ) -> None:
        super().__init__(
            name or Path(path).stem, direction=direction, threshold=threshold, comparison=comparison
        )
        self.path = Path(path)
        self.smiles_col = smiles_col
        self.pred_col = pred_col
        self.canonicalize = canonicalize
        self._table: Optional[Dict[str, float]] = None

    # -- table handling ---------------------------------------------------
    def _key(self, smiles: str) -> Optional[str]:
        """Look-up key. Canonicalizing means a table built elsewhere still hits.

        Falls back to the raw string when RDKit cannot parse the molecule, so an
        unparseable entry is a cache miss rather than an exception.
        """
        if smiles is None or pd.isna(smiles) or smiles in ("", "None"):
            return None
        text = str(smiles)
        if not self.canonicalize:
            return text
        cached = _CANON_CACHE.get(text)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(text)
        key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else text
        _CANON_CACHE[text] = key
        return key

    def _load(self) -> Dict[str, float]:
        if self._table is not None:
            return self._table
        if not self.path.exists():
            self._table = {}
            return self._table
        df = pd.read_parquet(self.path) if self.path.suffix == ".parquet" else pd.read_csv(self.path)
        pred_col = self.pred_col
        if pred_col is None:
            candidates = [c for c in df.columns if c != self.smiles_col]
            if len(candidates) != 1:
                raise ValueError(
                    f"{self.path}: cannot infer the prediction column from {list(df.columns)}. "
                    f"Pass pred_col explicitly."
                )
            pred_col = candidates[0]
            self.pred_col = pred_col
        table: Dict[str, float] = {}
        for smi, val in zip(df[self.smiles_col], df[pred_col]):
            key = self._key(smi)
            if key is not None and not pd.isna(val):
                table[key] = float(val)
        self._table = table
        return table

    def reload(self) -> None:
        self._table = None

    def missing(self, smiles: Sequence[str]) -> List[str]:
        """Inputs with no entry in the table, deduplicated, original spelling."""
        table = self._load()
        seen, out = set(), []
        for s in smiles:
            key = self._key(s)
            if key is None or key in table or s in seen:
                continue
            seen.add(s)
            out.append(str(s))
        return out

    def predict(self, smiles: Sequence[str]) -> List[Optional[float]]:
        table = self._load()
        out: List[Optional[float]] = []
        for s in smiles:
            key = self._key(s)
            out.append(table.get(key) if key is not None else None)
        return out

    def provenance(self) -> Dict[str, Any]:
        info = super().provenance()
        table = self._load()
        stat = self.path.stat() if self.path.exists() else None
        info.update(
            {
                "path": str(self.path),
                "exists": self.path.exists(),
                "n_entries": len(table),
                "smiles_col": self.smiles_col,
                "pred_col": self.pred_col,
                "canonicalized_keys": self.canonicalize,
                "size_bytes": stat.st_size if stat else None,
                "mtime": stat.st_mtime if stat else None,
            }
        )
        return info


class SubprocessOracle(PrecomputedTableOracle):
    """A precomputed table that fills its own gaps.

    On :meth:`predict`, any molecule missing from the cache is scored by running
    a worker under a different Python interpreter, and the cache is re-read. If
    nothing is missing, no subprocess is spawned, so repeat runs cost a file read.

    This exists because MiniMol cannot be imported here: graphium pins torch 2.3
    and numpy<2, ``Minimol()`` calls ``hydra.initialize`` and cannot be
    constructed twice, and this package already depends on hydra-core. The venv
    boundary is real, but it should not become a manual step for the caller.
    """

    def __init__(
        self,
        path: Union[str, Path],
        worker: str,
        python: Optional[Union[str, Path]] = None,
        worker_args: Optional[Sequence[str]] = None,
        offline: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(path, **kwargs)
        self.worker = worker
        self.python = str(python) if python else None
        self.worker_args = list(worker_args or [])
        self.offline = offline
        self.timeout = timeout
        self.n_filled = 0
        self.n_subprocess_calls = 0

    def resolve_python(self) -> Optional[str]:
        """Interpreter to score with: explicit, then env var, then this one."""
        return self.python or os.environ.get("PAIRIT_MINIMOL_PYTHON") or None

    def fill(self, smiles: Sequence[str]) -> int:
        """Score anything missing and extend the cache. Returns how many were added."""
        missing = self.missing(smiles)
        if not missing:
            return 0
        if self.offline:
            raise RuntimeError(
                f"Oracle {self.name!r} is missing predictions for {len(missing)} molecules and "
                f"--offline-oracles is set, so they cannot be scored. Either drop the flag or "
                f"extend {self.path}."
            )
        python = self.resolve_python()
        if python is None:
            raise RuntimeError(
                f"Oracle {self.name!r} is missing predictions for {len(missing)} molecules and no "
                f"interpreter was given to score them with. Pass --minimol-python "
                f"/path/to/venv/bin/python, or set $PAIRIT_MINIMOL_PYTHON, or pass "
                f"--offline-oracles to fail instead of scoring."
            )
        if not Path(python).exists():
            raise RuntimeError(f"Oracle {self.name!r}: interpreter not found: {python}")

        tmpdir = tempfile.mkdtemp(prefix="pairit_oracle_")
        try:
            smiles_file = Path(tmpdir) / "todo.txt"
            smiles_file.write_text("\n".join(missing) + "\n")
            cmd = [
                python,
                "-m",
                self.worker,
                "--smiles-file",
                str(smiles_file),
                "--out",
                str(self.path),
                *self.worker_args,
            ]
            self.n_subprocess_calls += 1
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
                raise RuntimeError(
                    f"Oracle {self.name!r}: scoring worker failed (exit {proc.returncode}).\n"
                    f"  command: {' '.join(cmd)}\n  " + "\n  ".join(tail)
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.reload()
        still_missing = self.missing(smiles)
        filled = len(missing) - len(still_missing)
        self.n_filled += filled
        return filled

    def predict(self, smiles: Sequence[str]) -> List[Optional[float]]:
        self.fill(smiles)
        return super().predict(smiles)

    def provenance(self) -> Dict[str, Any]:
        info = super().provenance()
        info.update(
            {
                "worker": self.worker,
                "python": self.resolve_python(),
                "offline": self.offline,
                "n_filled_this_run": self.n_filled,
                "n_subprocess_calls": self.n_subprocess_calls,
            }
        )
        return info


class CallableOracle(Oracle):
    """Wraps a function from SMILES to score. Mainly an injection point for tests."""

    def __init__(
        self,
        fn: Callable[[Sequence[str]], Sequence[Optional[float]]],
        name: str = "callable",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.fn = fn

    def predict(self, smiles: Sequence[str]) -> List[Optional[float]]:
        return list(self.fn(list(smiles)))


class PropertySpecOracle(Oracle):
    """Adapts a continuous :class:`PropertySpec` to the oracle interface."""

    def __init__(
        self,
        spec: Any,
        name: Optional[str] = None,
        threshold: float = 0.0,
        comparison: str = ">=",
    ) -> None:
        super().__init__(
            name or getattr(spec, "name", "property"),
            direction=getattr(spec, "direction", INCREASING),
            threshold=threshold,
            comparison=comparison,
        )
        self.spec = spec

    def predict(self, smiles: Sequence[str]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for s in smiles:
            try:
                out.append(self.spec.fn(s))
            except Exception:
                out.append(None)
        return out


def resolve_oracle(
    spec: Union[str, Path, Oracle, None],
    dataset: str,
    dataset_config: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Optional[Oracle]:
    """Build an oracle from a user-supplied spec.

    Accepted forms, dispatched on shape:

    ``Oracle``           returned unchanged
    directory            ``{task_name}.joblib`` inside it (the historical convention)
    ``*.joblib``         a fingerprint model
    ``*.csv``/``*.parquet``  a precomputed prediction table
    ``"minimol"``        the MiniMol subprocess oracle for this dataset
    ``None``             no oracle

    `direction` and `threshold` come from the dataset config when it has them, so
    a `decreasing` dataset is scored correctly without the caller restating it.
    """
    if spec is None:
        return None
    if isinstance(spec, Oracle):
        return spec

    config = dataset_config if dataset_config is not None else DATASET_CONFIG
    ds_config = (config or {}).get(dataset, {}) or {}
    defaults: Dict[str, Any] = {}
    if ds_config.get("direction") is not None:
        defaults["direction"] = ds_config["direction"]
    if ds_config.get("threshold") is not None:
        defaults["threshold"] = ds_config["threshold"]
    defaults.update(kwargs)

    text = str(spec)
    if text == "minimol":
        pred_dir = Path(
            os.environ.get(
                "PAIRIT_MINIMOL_PRED_DIR", str(ASSETS / "oracles" / "minimol_preds")
            )
        )
        return SubprocessOracle(
            pred_dir / f"{dataset}.csv",
            worker="pairit.eval.oracles_minimol_worker",
            worker_args=["--dataset", dataset],
            name=f"minimol:{dataset}",
            pred_col="minimol_pred",
            **defaults,
        )

    path = Path(text)
    if path.is_dir():
        task_name = ds_config.get("task_name", dataset)
        return FingerprintJoblibOracle.from_path(
            path / f"{task_name}.joblib", name=f"{task_name}@{path.name}", **defaults
        )
    if path.suffix == ".joblib":
        return FingerprintJoblibOracle.from_path(path, **defaults)
    if path.suffix in (".csv", ".parquet"):
        return PrecomputedTableOracle(path, **defaults)
    raise ValueError(
        f"Cannot interpret oracle spec {spec!r}. Expected a directory, a .joblib, a "
        f"prediction table (.csv/.parquet), 'minimol', or an Oracle instance."
    )


def resolve_oracle_pair(
    select_spec: Union[str, Path, Oracle, None],
    score_spec: Union[str, Path, Oracle, None],
    fallback: Union[str, Path, Oracle, None],
    dataset: str,
    dataset_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Optional[Oracle], Optional[Oracle]]:
    """Resolve the (selection, scoring) pair.

    Fallback semantics (paper protocol):
      - score  = score_spec  → else fallback  (evaluator_dir, typically train-only RF)
      - select = select_spec → else fallback  (evaluator_dir) — NOT score
        This is critical: when only score_spec='minimol' is given (paper
        protocol) and evaluator_dir points at RF joblibs, `select` MUST resolve
        to RF, NOT to MiniMol. Otherwise selection uses the same oracle as
        scoring — reward-hacking bug — because MiniMol both filters and judges.
      - If neither select_spec nor fallback resolves, select degrades to score
        (single-oracle mode, historical behaviour).

    Result: any config that names two distinct oracles (or one oracle + a
    fallback evaluator_dir) gives the correct two-oracle protocol without
    the caller needing to know the ordering.
    """
    resolved_fallback = resolve_oracle(fallback, dataset, dataset_config)
    score = resolve_oracle(score_spec, dataset, dataset_config) or resolved_fallback
    select = (
        resolve_oracle(select_spec, dataset, dataset_config)
        or resolved_fallback
        or score
    )
    return select, score
