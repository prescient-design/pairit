"""The generation registry: which file backs which analysis.

**Selective by design.** One row per generation set that actually backs a table or
figure -- not every file ever produced. `use_for` is the admission criterion: if
nothing cites a run, it is not registered. A sweep of forty configurations
contributes the one row that was chosen, not forty.

One tab-separated row per generation source. It replaces a 3,898-line YAML whose
weight was the reason it stopped being maintained -- ~33 lines per entry, the
same warning paragraph pasted into eight protocols, two mutually exclusive ways
to write a path, and an append-here marker that existed because appending in the
wrong place silently put an entry in a list nothing read.

TSV because it is greppable, diffable line by line, opens in a spreadsheet,
needs no parser, keeps merge conflicts local, and can be appended to anywhere.

The file is pure data -- header row, then rows, no comment block -- so any reader
handles it without special flags. How to use it is in ``internal/docs/REGISTRY.md``.

Division of labour
------------------
This file says **where the data is**. ``table_registry.tsv`` says **what protocol
a table uses**. The per-run ``{prefix}_decisions.json`` records **what actually
ran**. Three files, no overlap, and the third is generated so it cannot drift
from the first two.

Columns
-------
dataset, method, role, path, n_seeds, gens_per_seed, format, status,
description, notes

`n_seeds` and `gens_per_seed` are **derived**, not authored: ``check`` opens each
file and fills them in. They are the two facts that catch a mismatched source
before it reaches a table.

`description` and `notes` are two columns on purpose. A description is what the
run *is*, and every row has one; a note is what is *wrong with it*, and most rows
should be empty. Fusing them is how the YAML ended up with the same ten-line
warning in eight places.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

COLUMNS = [
    "dataset",
    "method",
    "use_for",
    "label",
    "role",
    "path",
    "n_seeds",
    "gens_per_seed",
    "format",
    "status",
    "description",
    "notes",
    "replicate_group",
]

#: Columns a file may omit from its header and still load. Everything in COLUMNS
#: minus these must be present.
#:
#: `replicate_group` is optional so that a registry written before it existed keeps
#: working untouched -- a missing header column reads as blank, which means
#: "standalone", which is exactly the old behaviour. Files are upgraded in place the
#: next time `check_registry --fix` writes them, since the writer emits all COLUMNS.
OPTIONAL_COLUMNS = ("replicate_group",)

DERIVED_COLUMNS = ("n_seeds", "gens_per_seed")

ROLES = ("generations", "oracle", "novelty_ref", "seed_source", "output")
STATUSES = ("ok", "caution", "blocked")

# The registry file is PURE DATA: a header row, then rows. No comment block.
#
# A leading "#" block is unsafe here. pandas.read_csv(sep="\t") without
# comment="#" raises, which is survivable, but csv.DictReader silently takes the
# first comment line as the header and mis-parses every row -- 71 rows instead of
# 44, with no error. Guidance therefore lives in internal/docs/REGISTRY.md, and the file
# stays readable by pandas, csv, awk, cut and Excel with no special handling.
HEADER_COMMENT = ""


@dataclass
class RegistryRow:
    """One registered source."""

    dataset: str
    method: str
    #: Comma-separated table/figure ids this row backs. This is the admission
    #: criterion, not metadata: a row with no `use_for` does not belong here.
    use_for: str = ""
    #: Optional short name for display -- a few words, no punctuation needed.
    #: `description` stays the full prose; this is what fits in a table.
    label: str = ""
    role: str = "generations"
    path: str = ""
    n_seeds: Optional[int] = None
    gens_per_seed: Optional[float] = None
    format: str = "pairit_csv"
    status: str = "ok"
    description: str = ""
    notes: str = ""
    #: Rows sharing a value are training-replicate runs of ONE configuration, and a
    #: table aggregates them into a single row as mean +/- sd. Blank means standalone.
    #:
    #: "Replicate" here is a *random seed*, not a seed molecule -- `n_seeds` above
    #: counts seed molecules, and the two senses are easy to confuse.
    #:
    #: Optional and blank-by-default, so every pre-existing row keeps its current
    #: behaviour: `group_key()` falls back to `method`, which is unique per
    #: (dataset, role), so an un-annotated row is a group of one.
    replicate_group: str = ""
    line_no: Optional[int] = None
    source_file: Optional[str] = None

    def group_key(self) -> str:
        """Configuration this row belongs to. Blank `replicate_group` -> itself."""
        return self.replicate_group or self.method

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.dataset, self.method, self.role)

    def to_row(self) -> Dict[str, str]:
        def fmt(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:g}"
            return str(value)

        return {c: fmt(getattr(self, c)) for c in COLUMNS}


class RegistryError(ValueError):
    """A registry file that cannot be parsed at all."""


def load_registry(path: Union[str, Path]) -> List[RegistryRow]:
    """Parse a registry TSV.

    Uses :mod:`csv` with a tab delimiter rather than pandas: pandas mangles a
    `notes` field containing quotes and silently type-coerces `n_seeds`, both of
    which matter here.
    """
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"Registry not found: {path}")

    rows: List[RegistryRow] = []
    with open(path, newline="", encoding="utf-8") as fh:
        # Comment lines are tolerated so a file written by an older version still
        # loads, but none are written any more -- see HEADER_COMMENT.
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return rows

    reader = csv.DictReader(lines, delimiter="\t")
    missing = [
        c for c in COLUMNS
        if c not in (reader.fieldnames or []) and c not in OPTIONAL_COLUMNS
    ]
    if missing:
        raise RegistryError(
            f"{path}: header is missing column(s) {missing}.\nExpected exactly:\n  "
            + "\t".join(COLUMNS)
        )

    for i, raw in enumerate(reader, start=2):
        rows.append(
            RegistryRow(
                dataset=(raw.get("dataset") or "").strip(),
                method=(raw.get("method") or "").strip(),
                use_for=(raw.get("use_for") or "").strip(),
                label=(raw.get("label") or "").strip(),
                role=(raw.get("role") or "generations").strip(),
                path=(raw.get("path") or "").strip(),
                n_seeds=_as_int(raw.get("n_seeds")),
                gens_per_seed=_as_float(raw.get("gens_per_seed")),
                format=(raw.get("format") or "pairit_csv").strip(),
                status=(raw.get("status") or "ok").strip(),
                description=(raw.get("description") or "").strip(),
                notes=(raw.get("notes") or "").strip(),
                # `.get` (not `raw["..."]`) is what lets a file written before this
                # column existed still load: DictReader yields None for the missing
                # field rather than raising, so old rows read as standalone.
                replicate_group=(raw.get("replicate_group") or "").strip(),
                line_no=i,
                source_file=str(path),
            )
        )
    return rows


def write_registry(rows: Sequence[RegistryRow], path: Union[str, Path]) -> Path:
    """Write rows back, preserving the header comment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return path


def select_rows(
    rows: Iterable[RegistryRow],
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    role: Optional[str] = "generations",
    include_blocked: bool = False,
    use_for: Optional[str] = None,
) -> List[RegistryRow]:
    """Filter registry rows. `methods=None` or ``["*"]`` means every method.

    `use_for` selects the rows backing one table or figure.
    """
    out = []
    wildcard = methods is None or list(methods) == ["*"]
    for row in rows:
        if role is not None and row.role != role:
            continue
        if use_for and use_for not in [u.strip() for u in row.use_for.split(",")]:
            continue
        if datasets and row.dataset not in datasets:
            continue
        if not wildcard and row.method not in (methods or []):
            continue
        if row.status == "blocked" and not include_blocked:
            continue
        out.append(row)
    return out


@dataclass
class RowCheck:
    """Outcome of checking one row."""

    row: RegistryRow
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    observed_n_seeds: Optional[int] = None
    observed_gens_per_seed: Optional[float] = None

    @property
    def ok(self) -> bool:
        return not self.errors


def probe(path: Union[str, Path]) -> Tuple[bool, str]:
    """Does this path exist, and if not, why not?

    ``Path.exists()`` returns False both for a missing file and for one inside a
    directory the caller cannot traverse, and it raises PermissionError on some
    filesystems. Those are different problems with different fixes -- a missing
    file needs regenerating, an unreadable one needs a permission change -- so
    they are distinguished here.
    """
    p = Path(path)
    try:
        if p.exists():
            return True, "ok"
    except PermissionError:
        return False, "permission denied on an ancestor directory"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"

    for parent in p.parents:
        try:
            if parent.exists():
                break
        except PermissionError:
            return False, f"permission denied at {parent}"
        except OSError:
            return False, "unreadable"
    return False, "not found"


def check_row(row: RegistryRow, derive: bool = True) -> RowCheck:
    """Validate one row and, if asked, derive `n_seeds` / `gens_per_seed`."""
    result = RowCheck(row=row)

    if not row.dataset:
        result.errors.append("dataset is empty")
    if not row.method:
        result.errors.append("method is empty")
    if row.role not in ROLES:
        result.errors.append(f"role {row.role!r} is not one of {ROLES}")
    if row.status not in STATUSES:
        result.errors.append(f"status {row.status!r} is not one of {STATUSES}")
    if not row.description:
        result.warnings.append("no description -- say what this run is")
    if row.role == "generations" and not row.use_for:
        result.warnings.append(
            "no use_for -- this registry lists runs that back a table or figure; "
            "if nothing cites this run, it does not belong here"
        )

    for column in ("dataset", "method", "label", "description", "notes"):
        if "\t" in str(getattr(row, column)):
            result.errors.append(f"{column} contains a tab, which breaks the file format")

    if not row.path:
        result.errors.append("path is empty")
        return result

    exists, why = probe(row.path)
    if not exists:
        (result.warnings if row.status == "blocked" else result.errors).append(
            f"path unreadable ({why}): {row.path}"
        )
        return result

    if row.role != "generations" or not derive:
        return result

    from .raw_loaders import FORMATS, load_generations

    if row.format not in FORMATS:
        result.errors.append(f"format {row.format!r} is not one of {FORMATS}")
        return result

    try:
        df = load_generations(row.path, row.format, dataset=row.dataset, method=row.method)
    except Exception as e:
        result.errors.append(f"could not load: {type(e).__name__}: {e}")
        return result

    if not len(df):
        result.warnings.append("file loaded but contains no rows")
        return result

    # Count distinct seed SMILES, never seed_idx: seed_idx wraps at the batch
    # size and reports the same value for files with very different seed counts.
    n_seeds = int(df["seed_smiles"].nunique())
    result.observed_n_seeds = n_seeds
    result.observed_gens_per_seed = round(len(df) / n_seeds, 3) if n_seeds else None

    if row.n_seeds is not None and row.n_seeds != n_seeds:
        result.warnings.append(
            f"n_seeds says {row.n_seeds}, file has {n_seeds}"
        )
    if (
        row.gens_per_seed is not None
        and result.observed_gens_per_seed is not None
        and abs(row.gens_per_seed - result.observed_gens_per_seed) > 0.01
    ):
        result.warnings.append(
            f"gens_per_seed says {row.gens_per_seed:g}, file has "
            f"{result.observed_gens_per_seed:g}"
        )
    return result


def check_registry(
    path: Union[str, Path], derive: bool = True
) -> Tuple[List[RowCheck], List[str]]:
    """Check every row. Returns per-row results and file-level errors."""
    rows = load_registry(path)
    results = [check_row(r, derive=derive) for r in rows]

    file_errors: List[str] = []
    seen: Dict[Tuple[str, str, str], int] = {}
    for row in rows:
        if row.key in seen:
            file_errors.append(
                f"duplicate entry for {row.key} at lines {seen[row.key]} and {row.line_no}"
            )
        else:
            seen[row.key] = row.line_no or 0

    file_errors.extend(_check_mixed_selection_stage(results))
    return results, file_errors


#: How far `gens_per_seed` may spread within one table before it looks like a
#: post-selection file and a raw candidate pool have been mixed. Post-selection rows
#: cluster near the table's k (often 1.5-5 -- seeds that clear the gate yield k rows,
#: seeds that fail yield one); a raw pool reads 20, 52 or 64. A 4x spread separates
#: those two regimes without firing on the ordinary shortfall.
SELECTION_STAGE_RATIO = 4.0


def _check_mixed_selection_stage(results: Sequence["RowCheck"]) -> List[str]:
    """Warn when one table's rows look like different selection stages.

    Nothing in the schema records whether a file is pre- or post-selection, and
    `evaluate.py` will happily score a 64-candidate pool alongside already-selected
    top-5 files: the result is plausible and roughly an order of magnitude wrong.
    `gens_per_seed` is the only available signal, so compare it within a `use_for`.

    This is a warning, not an error -- mixing stages deliberately is legitimate.
    `fig_inference_time_compute` registers the same run before and after selection to
    show the difference; its description says they must never be averaged together.
    """
    by_table: Dict[str, List[Tuple[str, str, float]]] = {}
    for r in results:
        row = r.row
        gps = r.observed_gens_per_seed if r.observed_gens_per_seed is not None else row.gens_per_seed
        if gps is None or gps <= 0 or row.role != "generations":
            continue
        for table in (u.strip() for u in row.use_for.split(",") if u.strip()):
            by_table.setdefault(table, []).append((row.dataset, row.method, float(gps)))

    out: List[str] = []
    for table, entries in sorted(by_table.items()):
        if len(entries) < 2:
            continue
        lo = min(e[2] for e in entries)
        hi = max(e[2] for e in entries)
        if lo <= 0 or hi / lo < SELECTION_STAGE_RATIO:
            continue
        biggest = max(entries, key=lambda e: e[2])
        smallest = min(entries, key=lambda e: e[2])
        out.append(
            f"{table}: generations-per-seed spans {lo:g} to {hi:g} "
            f"({smallest[0]}/{smallest[1]} vs {biggest[0]}/{biggest[1]}). "
            f"If one of these is a raw candidate pool and the others are already "
            f"top-k selected, the table scores them at different stages. Register the "
            f"post-selection file, or set select/k on the table spec. Ignore this if "
            f"the table deliberately compares before and after selection."
        )
    return out


def apply_derived(results: Sequence[RowCheck]) -> List[RegistryRow]:
    """Write observed counts back into the rows."""
    out = []
    for r in results:
        row = r.row
        if r.observed_n_seeds is not None:
            row.n_seeds = r.observed_n_seeds
        if r.observed_gens_per_seed is not None:
            row.gens_per_seed = r.observed_gens_per_seed
        out.append(row)
    return out


def _as_int(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    try:
        return int(float(text)) if text else None
    except ValueError:
        return None


def _as_float(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    try:
        return float(text) if text else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Table registry
#
# The generation registry says WHERE the data is. This says WHAT PROTOCOL a
# published table uses. Without it, "add a method to the existing table" has no
# answer: the protocol lives only in whatever flags someone typed last time.
# ---------------------------------------------------------------------------

#: Columns a table registry file may omit and still load. `seed_filter` postdates the
#: shipped registries, and requiring it would break every existing file at load time.
OPTIONAL_TABLE_COLUMNS = ("seed_filter",)

TABLE_COLUMNS = [
    "table_id",
    "datasets",
    "methods",
    "select",
    "k",
    "select_oracle",
    "score_oracle",
    "seed_policy",
    "denominator",
    "metrics",
    "seed_filter",
    "output",
    "description",
]

TABLE_HEADER_COMMENT = ""  # pure data; see internal/docs/REGISTRY.md


@dataclass
class TableSpec:
    """One published table's protocol."""

    table_id: str
    datasets: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=lambda: ["*"])
    select: Optional[str] = None
    k: int = 5
    select_oracle: Optional[str] = None
    score_oracle: Optional[str] = None
    seed_policy: str = "intersection"
    denominator: str = "budget"
    metrics: List[str] = field(default_factory=list)
    #: Which starting molecules the table scores. Empty = every one in the seed set.
    #: "connected" drops multi-fragment molecules (salts, hydrates, counterions) from
    #: the reference set AND from every method, symmetrically. It exists because matched
    #: molecular pair construction cannot produce a pair from a disconnected structure,
    #: so no pair-trained method can learn them -- scoring them measures nothing about
    #: the method. Applied at the seed-policy stage, before selection, so no method
    #: spends its candidate budget on a molecule that is about to be excluded.
    seed_filter: str = ""
    output: str = ""
    description: str = ""
    line_no: Optional[int] = None
    source_file: Optional[str] = None

    @property
    def all_methods(self) -> bool:
        return not self.methods or self.methods == ["*"]

    def to_row(self) -> Dict[str, str]:
        return {
            "table_id": self.table_id,
            "datasets": ",".join(self.datasets),
            "methods": ",".join(self.methods),
            "select": self.select or "-",
            "k": str(self.k),
            "select_oracle": self.select_oracle or "-",
            "score_oracle": self.score_oracle or "-",
            "seed_policy": self.seed_policy,
            "denominator": self.denominator,
            "metrics": ",".join(self.metrics),
            "seed_filter": self.seed_filter,
            "output": self.output,
            "description": self.description,
        }


def _split(value: Optional[str]) -> List[str]:
    text = (value or "").strip()
    if not text or text == "-":
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def _dash_to_none(value: Optional[str]) -> Optional[str]:
    text = (value or "").strip()
    return None if text in ("", "-") else text


#: Closed vocabulary. A typo here would silently score a different set of starting
#: molecules, so an unknown value is an error rather than a no-op.
SEED_FILTERS = ("", "connected")


def _seed_filter(value: Optional[str], path: Path, line_no: int) -> str:
    text = (value or "").strip()
    if text == "-":
        text = ""
    if text not in SEED_FILTERS:
        raise RegistryError(
            f"{path}:{line_no}: seed_filter {text!r} is not one of {SEED_FILTERS}. "
            f"An unrecognised filter would silently score a different set of starting "
            f"molecules than intended."
        )
    return text


def load_table_registry(path: Union[str, Path]) -> List[TableSpec]:
    """Parse a table registry TSV."""
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"Table registry not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []

    reader = csv.DictReader(lines, delimiter="\t")
    missing = [c for c in TABLE_COLUMNS
               if c not in (reader.fieldnames or []) and c not in OPTIONAL_TABLE_COLUMNS]
    if missing:
        raise RegistryError(
            f"{path}: header is missing column(s) {missing}.\nExpected exactly:\n  "
            + "\t".join(TABLE_COLUMNS)
        )

    specs = []
    for i, raw in enumerate(reader, start=2):
        specs.append(
            TableSpec(
                table_id=(raw.get("table_id") or "").strip(),
                datasets=_split(raw.get("datasets")),
                methods=_split(raw.get("methods")) or ["*"],
                select=_dash_to_none(raw.get("select")),
                k=_as_int(raw.get("k")) or 5,
                select_oracle=_dash_to_none(raw.get("select_oracle")),
                score_oracle=_dash_to_none(raw.get("score_oracle")),
                seed_policy=(raw.get("seed_policy") or "intersection").strip(),
                denominator=(raw.get("denominator") or "budget").strip(),
                metrics=_split(raw.get("metrics")),
                seed_filter=_seed_filter(raw.get("seed_filter"), path, i),
                output=(raw.get("output") or "").strip(),
                description=(raw.get("description") or "").strip(),
                line_no=i,
                source_file=str(path),
            )
        )
    return specs


def write_table_registry(specs: Sequence[TableSpec], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for spec in specs:
            writer.writerow(spec.to_row())
    return path


def find_table(specs: Iterable[TableSpec], table_id: str) -> TableSpec:
    for spec in specs:
        if spec.table_id == table_id:
            return spec
    known = ", ".join(sorted(s.table_id for s in specs)) or "(none)"
    raise RegistryError(f"No table {table_id!r} in the table registry. Known: {known}")
