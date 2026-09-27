"""The per-run decision record.

Every run writes ``{prefix}_decisions.json`` and prints a short summary. It
answers the question that repeatedly could not be answered about past numbers:
*which file produced this cell, scored by which oracle, over which denominator?*

The registry says where the data is. This says what actually ran. It is generated
rather than hand-written, so it cannot drift from the run it describes -- which is
why it replaces the hand-asserted `protocol` and `headline` fields that used to
live alongside the paths, one of which reads `pairit_eval_commit: unknown`.

This is the per-run machine record. ``DECISIONS.md`` is the cross-run narrative
for humans. Neither replaces the other.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Union

_HASH_LIMIT = 64 << 20  # don't hash more than 64MB of any one file by default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_state(path: Union[str, Path]) -> Dict[str, Any]:
    """Commit and dirty flag for the repo containing `path`.

    This is the field that fixes 'pairit_eval_commit: unknown / NOT RECOVERABLE'
    on previously-recorded results.
    """
    out: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        root = Path(path).resolve().parent
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if rev.returncode == 0:
            out["commit"] = rev.stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            )
            out["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def file_facts(path: Union[str, Path], hash_it: bool = True) -> Dict[str, Any]:
    """Identify a file: path, size, mtime, and optionally a content hash.

    Size and mtime are cheap and nearly always sufficient. Hashing is capped
    because reference corpora run to hundreds of megabytes on a shared
    filesystem and hashing them on every run buys little over (size, mtime).
    """
    p = Path(path)
    facts: Dict[str, Any] = {"path": str(p), "exists": False}
    try:
        st = p.stat()
    except OSError as e:
        facts["error"] = f"{type(e).__name__}: {e}"
        return facts
    facts.update(
        exists=True,
        size_bytes=st.st_size,
        mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    )
    if hash_it and st.st_size <= _HASH_LIMIT:
        h = hashlib.sha256()
        try:
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            facts["sha256"] = h.hexdigest()
        except OSError:
            pass
    elif hash_it:
        facts["sha256"] = None
        facts["sha256_skipped"] = "file larger than the hash limit"
    return facts


class DecisionRecord:
    """Accumulates what a run decided, then writes it out.

    Methods are additive and order-independent; anything not recorded is simply
    absent rather than guessed at.
    """

    def __init__(self, argv: Optional[Sequence[str]] = None) -> None:
        self.data: Dict[str, Any] = {
            "schema_version": 1,
            "run": {
                "timestamp_utc": _utc_now(),
                "argv": list(argv if argv is not None else sys.argv),
                "cwd": str(Path.cwd()),
                "user": _safe(getpass.getuser),
                "host": _safe(socket.gethostname),
            },
            "versions": self._versions(),
            "inputs": [],
            "oracles": {},
            "novelty_references": [],
            "selection": [],
            "seed_policy": [],
            "normalization": [],
            "denominators": {},
            "column_aliases": {},
            "outputs": {},
            "warnings": [],
        }

    # -- capture ----------------------------------------------------------
    @staticmethod
    def _versions() -> Dict[str, Any]:
        import pairit.eval as pairit_eval  # local alias preserves downstream dict keys ("pairit_eval", "pairit_eval_git") for backwards-compat

        versions: Dict[str, Any] = {
            "python": platform.python_version(),
            "pairit_eval": getattr(pairit_eval, "__version__", None),
            "pairit_eval_git": _git_state(pairit_eval.__file__),
        }
        for mod in ("pandas", "numpy", "rdkit", "sklearn", "joblib"):
            try:
                versions[mod] = __import__(mod).__version__
            except Exception:
                versions[mod] = None
        return versions

    def add_input(
        self,
        dataset: str,
        method: str,
        path: Optional[Union[str, Path]] = None,
        *,
        n_rows: Optional[int] = None,
        n_seeds: Optional[int] = None,
        gens_per_seed: Optional[float] = None,
        loader: Optional[str] = None,
        fmt: Optional[str] = None,
        source: str = "cli",
        registry_file: Optional[str] = None,
        registry_line: Optional[int] = None,
        description: Optional[str] = None,
        notes: Optional[str] = None,
        hash_it: bool = True,
    ) -> None:
        """Record which file produced one (dataset, method) cell.

        Keyed the same way a table cell is, so a number is traceable to a file on
        disk without knowing how the run was invoked.
        """
        entry: Dict[str, Any] = {"dataset": dataset, "method": method, "source": source}
        if path is not None:
            entry.update(file_facts(path, hash_it=hash_it))
        entry.update(
            {
                "n_rows": n_rows,
                "n_seeds": n_seeds,
                "gens_per_seed": gens_per_seed,
                "loader": loader,
                "format": fmt,
            }
        )
        if registry_file:
            entry["registry"] = {
                "file": registry_file,
                "line": registry_line,
                "description": description,
                "notes": notes,
            }
        self.data["inputs"].append(entry)

    def add_oracle(self, role: str, oracle: Any, dataset: str = "_") -> None:
        """Record an oracle under its role (`select` or `score`), per dataset.

        Keyed by dataset because oracles are resolved per dataset -- BACE is
        scored by bace_b, HIV by hiv_b. A single slot would keep only whichever
        dataset was processed last, silently omitting the oracle that scored
        every other one.
        """
        block = self.data["oracles"].setdefault(dataset, {})
        block[role] = oracle.provenance() if oracle is not None else None
        sel, sco = block.get("select"), block.get("score")
        if sel is not None and sco is not None:
            # Whether filtering and judging were actually separated is the first
            # thing a reader of a dual-oracle table wants to know.
            block["identical"] = sel.get("name") == sco.get("name")

    def add_novelty_reference(
        self, dataset: str, path: Optional[Union[str, Path]], n_molecules: Optional[int] = None
    ) -> None:
        entry: Dict[str, Any] = {"dataset": dataset, "n_molecules": n_molecules}
        if path is not None:
            # Reference corpora are large and live on a shared filesystem; size
            # and mtime identify them adequately.
            entry.update(file_facts(path, hash_it=False))
        self.data["novelty_references"].append(entry)

    def add_seed_filter(self, report) -> None:
        """Record what a `seed_filter` excluded, with the structures themselves.

        Kept distinct from the seed-policy record: a policy decides which molecules the
        methods share, a filter decides which the TABLE scores at all. Merging them
        would make an exclusion look like an artifact of method coverage.
        """
        self.data.setdefault("seed_filters", []).append(report.as_dict())

    def add_seed_policy(self, report: Any) -> None:
        """Record how the seed set was resolved, per dataset."""
        self.data.setdefault("seed_policy", []).append(
            report.to_dict() if hasattr(report, "to_dict") else dict(report)
        )

    def add_replicate_group(self, report: Any) -> None:
        """Record which runs were combined into one table row.

        A reader looking at ``68.30 ± 1.40`` needs to know it came from five files and
        which ones, since the aggregated row's `method` is the configuration name and
        no longer matches any single registry row.
        """
        self.data.setdefault("replicate_groups", []).append(
            report.to_dict() if hasattr(report, "to_dict") else dict(report)
        )

    def add_selection(self, report: Any) -> None:
        self.data["selection"].append(
            report.to_dict() if hasattr(report, "to_dict") else dict(report)
        )

    def add_normalization(self, dataset: str, **stats: Any) -> None:
        self.data["normalization"].append({"dataset": dataset, **stats})

    def set_config(self, config: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        resolved: Dict[str, Any] = {}
        for attr in (
            "evaluator_dir", "moleculenet_train_dir", "moleculenet_test_dir",
            "zinc250k_train_molecules", "zinc250k_test_csv",
            "select_oracle", "score_oracle", "minimol_python", "offline_oracles",
        ):
            value = getattr(config, attr, None)
            resolved[attr] = str(value) if isinstance(value, Path) else _jsonable(value)
        if extra:
            resolved.update({k: _jsonable(v) for k, v in extra.items()})
        self.data["config"] = resolved

    def set_denominators(self, table: Dict[str, Any]) -> None:
        """Record which denominator each reported rate used.

        Declarative, so a reader never has to infer it from a column name.
        """
        self.data["denominators"] = table

    def set_column_aliases(self, aliases: Dict[str, str]) -> None:
        self.data["column_aliases"] = aliases

    def add_output(self, kind: str, path: Union[str, Path]) -> None:
        self.data["outputs"][kind] = str(path)

    def warn(self, message: str) -> None:
        self.data["warnings"].append(message)

    def add_metrics(self, average_rows: Any) -> None:
        """Attach the metric values this run produced."""
        try:
            self.data["metrics"] = json.loads(average_rows.to_json(orient="records"))
        except Exception:
            self.data["metrics"] = _jsonable(average_rows)

    # -- output -----------------------------------------------------------
    def write(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.data, indent=2, default=str))
        return p

    def summary(self) -> str:
        """A short human-readable version, printed on every run."""
        d = self.data
        lines: List[str] = ["", "=" * 60, "Decisions", "=" * 60]

        git = d["versions"].get("pairit_eval_git") or {}
        commit = (git.get("commit") or "unknown")[:10]
        dirty = " (dirty)" if git.get("dirty") else ""
        lines.append(f"pairit_eval {commit}{dirty}")

        if d["inputs"]:
            lines.append("inputs:")
            for i in d["inputs"]:
                bits = [f"  {i['dataset']}/{i['method']}"]
                if i.get("path"):
                    bits.append(str(i["path"]))
                counts = []
                if i.get("n_rows") is not None:
                    counts.append(f"{i['n_rows']} rows")
                if i.get("n_seeds") is not None:
                    counts.append(f"{i['n_seeds']} seeds")
                if i.get("gens_per_seed") is not None:
                    counts.append(f"{i['gens_per_seed']:g} gens/seed")
                if counts:
                    bits.append(f"[{', '.join(counts)}]")
                lines.append(" ".join(bits))

        for dataset, block in d["oracles"].items():
            if not isinstance(block, dict):
                continue
            label = "" if dataset == "_" else f"{dataset} "
            for role in ("select", "score"):
                o = block.get(role)
                if not o:
                    continue
                unscorable = (
                    f", {o['n_unscorable']} unscorable" if o.get("n_unscorable") else ""
                )
                lines.append(
                    f"{label}{role} oracle: {o.get('name')} "
                    f"({o.get('direction')} @ {o.get('threshold')}{unscorable})"
                )
            if block.get("identical") is False:
                lines.append(f"  -> {label}filtering and judging used DIFFERENT oracles")

        for sp in d.get("seed_policy", []):
            note = "" if sp["denominators_equal"] else "  <- NOT equal across methods"
            if sp["rows_dropped"] or not sp["denominators_equal"]:
                lines.append(
                    f"seed policy {sp['policy']}: {sp['n_seeds_resolved']} shared seeds, "
                    f"{sp['rows_dropped']} rows dropped{note}"
                )

        for s in d["selection"]:
            yld = f"{s['gate_yield']:.1f}%" if s.get("gate_yield") is not None else "n/a"
            lines.append(
                f"selection {s['dataset']}/{s['method']}: {s['mode']} k={s['k']} "
                f"-> {s['candidates_out']}/{s['budget']} rows (yield {yld}); "
                f"seeds selected {s['seeds_selected']}, gate-failed "
                f"{s['seeds_gate_failed']}, absent {s['seeds_absent']}"
            )
            if s.get("seeds_absent"):
                lines.append(
                    f"  WARNING: {s['seeds_absent']} seeds have no rows in this file"
                )

        for w in d["warnings"]:
            lines.append(f"WARNING: {w}")
        if d["outputs"].get("decisions"):
            lines.append(f"full record: {d['outputs']['decisions']}")
        return "\n".join(lines)


def _safe(fn: Any) -> Optional[str]:
    try:
        return fn()
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# Which denominator each reported rate divides by. Declarative so the record
# cannot drift from the code, and so the inconsistencies are visible rather than
# buried -- see DECISIONS.md section 2.
METRIC_DENOMINATORS: Dict[str, Dict[str, str]] = {
    "valid": {
        "numerator": "n_valid",
        "denominator": "n_gen",
        "note": "n_valid removes generations identical to their seed but KEEPS intra-seed repeats",
    },
    "v_i": {"numerator": "n_v_i", "denominator": "n_gen", "note": ""},
    "v_n": {"numerator": "n_v_n", "denominator": "n_gen", "note": ""},
    "v_n_i": {
        "numerator": "n_v_n_i",
        "denominator": "n_gen",
        "note": "VIN = Valid AND Novel AND Improved. Numerator counted over valid_all.",
    },
    "novel": {
        "numerator": "n_novel",
        "denominator": "n_valid_incl_seed_dups",
        "note": "the only rate on this denominator - known inconsistency, left unchanged",
    },
    "improved": {
        "numerator": "n_improved",
        "denominator": "n_uniq_valid",
        "note": "",
    },
    "maintained": {"numerator": "n_maintained", "denominator": "n_uniq_valid", "note": "dead 1->1 metric kept for output-schema compatibility; always 0"},
    "processed": {"numerator": "n_processed", "denominator": "n_seeds", "note": ""},
    "sim_seed": {
        "numerator": "mean tanimoto",
        "denominator": "n_uniq_valid",
        "note": "excludes generations identical to their seed",
    },
}

DENOMINATOR_GLOSSARY: Dict[str, str] = {
    "n_gen": "every generation scored, valid or not",
    "n_budget": "k * n_seeds, when selection is on",
    "n_valid_incl_seed_dups": "valid generations, including those identical to their seed",
    "n_valid": "valid, seed-duplicates removed, intra-seed repeats kept",
    "n_uniq_valid": "valid, seed-duplicates AND intra-seed repeats removed",
    "n_seeds": "seeds in the seed set",
}


# Explicit column names: every rate carries the denominator it divides by, so a
# cell cannot be misread and two conventions cannot be pasted into one column.
# Emitted alongside the legacy names, which stay as aliases for one release --
# summarize_results.py, plot_summary_metrics.py, minimol_rescore.py and the
# registry checker all read these CSVs by name.
COLUMN_ALIASES: Dict[str, str] = {
    "valid": "valid/n_gen",
    "v_i": "vin_i/n_gen",
    "v_n": "vin_n/n_gen",
    "v_n_i": "vin/n_gen",
    "novel": "novel/n_valid_incl_seed_dups",
    "improved": "improved/n_uniq_valid",
    "maintained": "maintained/n_uniq_valid",
    "processed": "processed/n_seeds",
    "sim_seed": "sim_seed/n_uniq_valid",
    "sim_seed_v_n_i": "sim_seed/n_vin",
    # Both are computed over the valid, seed-duplicate-removed generations -- the same
    # population as `sim_seed` -- so they carry the same denominator name. `qed` is a
    # mean rather than a rate, but naming its population matters just as much: a QED
    # averaged over all generations and one averaged over the valid ones are different
    # numbers, and nothing else in the row would say which you have.
    "unique": "unique/n_uniq_valid",
    "qed": "qed/n_uniq_valid",
}

#: Under selection the composites divide by k * n_seeds instead of the row count.
COLUMN_ALIASES_SELECTED: Dict[str, str] = {
    "v_i": "vin_i/n_budget",
    "v_n": "vin_n/n_budget",
    "v_n_i": "vin/n_budget",
    "valid": "valid/n_budget",
}


#: Rates whose denominator is the generation count, and therefore the ones the
#: budget denominator actually changes. Everything else is conditional on
#: validity and is unaffected by how many generations were attempted.
_BUDGET_RESCALED = {
    "valid": "n_valid",
    "v_i": "n_v_i",
    "v_n": "n_v_n",
    "v_n_i": "n_v_n_i",
    "v_n_i": "n_v_n_i",
}


def budget_columns(df: "Any", budgets: "Dict[Any, int]") -> "Any":
    """Add `{metric}/n_budget` columns computed against k x n_seeds.

    NOT a rename. The budget denominator only means anything if the rate is
    actually divided by the budget: after selection `n_total` is the number of
    rows that survived, which is smaller and varies per method. Relabelling
    without recomputing produced a column that said `n_budget` and carried the
    selected denominator -- on one HIV table that read 94.82 where the budget
    figure is 73.22.

    `budgets` maps (dataset, method) to k * n_seeds. A pair with no budget gets
    no budget column rather than a wrong one.
    """
    out = df.copy()
    for metric, count_col in _BUDGET_RESCALED.items():
        if metric not in out.columns or count_col not in out.columns:
            continue
        target = COLUMN_ALIASES_SELECTED.get(metric)
        if not target:
            continue
        values = []
        for _, row in out.iterrows():
            budget = budgets.get((str(row.get("dataset")), str(row.get("method"))))
            count = row.get(count_col)
            values.append(
                100 * count / budget if budget and pd.notna(count) else float("nan")
            )
        out[target] = values
    return out


def explicit_columns(df: "Any", denominator: str = "n_gen") -> "Any":
    """Add a `{metric}/{denominator}` column beside each legacy rate column.

    Adds, never renames: downstream readers keep working untouched. The explicit
    name is the one to cite; the bare name is retained for one release.

    Parameters
    ----------
    denominator
        ``"n_gen"`` (default) or ``"n_budget"`` when selection ran with the
        budget denominator, which changes what the composites divide by.
    """
    # Deliberately does NOT apply COLUMN_ALIASES_SELECTED: those names mean
    # "divided by the budget", which requires recomputation, not relabelling.
    # Use budget_columns() for them.
    aliases = dict(COLUMN_ALIASES)
    if denominator == "n_budget":
        # Under a budget policy the n_gen names would be misleading too, since
        # n_total is the selected count; emit only the validity-conditional ones.
        aliases = {k: v for k, v in aliases.items() if "/n_gen" not in v}
    out = df.copy()
    for legacy, explicit in aliases.items():
        if legacy in out.columns and explicit not in out.columns:
            out[explicit] = out[legacy]
    return out


def active_aliases(denominator: str = "n_gen") -> Dict[str, str]:
    """The legacy -> explicit map in force for a given denominator policy."""
    aliases = dict(COLUMN_ALIASES)
    if denominator == "n_budget":
        aliases.update(COLUMN_ALIASES_SELECTED)
    return aliases
