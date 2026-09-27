#!/usr/bin/env python3
"""
Unified evaluation CLI for pairit_eval.

Two modes:
  File mode (default):  evaluate CSV/Parquet files directly
  Method mode:          load registered baselines from raw outputs

Examples:
  python scripts/evaluate.py generations.csv
  python scripts/evaluate.py lobster:generations.parquet --no-normalize
  python scripts/evaluate.py method_a:a.csv method_b:b.csv --distribution
  python scripts/evaluate.py --methods nfbo ddsbm-bridge --dataset bace
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

_REPO = Path(__file__).resolve().parent.parent
if _REPO not in sys.path:
    sys.path.insert(0, str(_REPO))

from pairit.eval.cli_utils import (
    DEFAULT_EVALUATOR_DIR,
    DEFAULT_MOLECULENET_TEST_DIR,
    DEFAULT_MOLECULENET_TRAIN_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_ZINC250K_TEST_CSV,
    DEFAULT_ZINC250K_TRAIN_MOLECULES,
    build_eval_config,
    load_and_tag_inputs,
    n_generations_map,
    parse_named_inputs,
)
from pairit.eval.evaluate import evaluate
from pairit.eval.normalize import normalize_generation_table
from pairit.eval import select as select_mod
from pairit.eval.decisions import (
    DENOMINATOR_GLOSSARY,
    METRIC_DENOMINATORS,
    DecisionRecord,
    active_aliases,
    budget_columns,
    explicit_columns,
)
from pairit.eval.pipeline import summarize_average
from pairit.eval.replicates import (
    N_REPLICATES,
    aggregate_replicates,
    format_cell,
    has_replicates,
)
from pairit.eval.raw_loaders import load_method_all


def _parse_extra_train_smiles(raw: List[str]) -> Dict[str, Set[str]]:
    """Parse NAME:PATH pairs into {name: set_of_canonical_smiles}."""
    from rdkit import Chem

    result: Dict[str, Set[str]] = {}
    for item in raw:
        if ":" not in item:
            raise ValueError(f"--extra-train-smiles must be NAME:PATH, got: {item}")
        name, path_str = item.split(":", 1)
        path = Path(path_str.strip())
        if not path.exists():
            raise FileNotFoundError(f"Extra train SMILES file not found: {path}")
        if path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
        smiles_set: Set[str] = set()
        for s in df[smiles_col].dropna():
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                smiles_set.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        result[name.strip()] = smiles_set
    return result


def _save_result(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified evaluation CLI for pairit_eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input mode ---
    p.add_argument(
        "inputs", nargs="*", metavar="INPUT",
        help="CSV/Parquet paths. Optional name:path syntax for method naming.",
    )
    p.add_argument(
        "--methods", nargs="+", metavar="METHOD",
        help="Registered method names (triggers raw loading instead of file input).",
    )

    # --- Output ---
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    p.add_argument("--output-prefix", default=None, help="Filename prefix for outputs.")
    p.add_argument("--output-format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--no-save", action="store_true", help="Suppress file output; print summary only.")

    # --- Filtering ---
    p.add_argument("--dataset", nargs="+", metavar="DS", help="Filter to specific dataset(s).")

    # --- Normalization ---
    p.add_argument("--n-generations", type=int, default=5, help="Expected generations per seed.")
    p.add_argument("--table", default=None,
                   help="Regenerate a table by id from --table-registry. Supplies the whole "
                        "protocol (datasets, methods, selection, oracles, seed policy, "
                        "denominator), so the table cannot drift on remembered flags.")
    p.add_argument("--table-registry", type=Path, default=None,
                   help="Table registry TSV. Defaults to table_registry.tsv beside --registry.")
    p.add_argument("--registry", type=Path, default=None,
                   help="Generation registry TSV. Rows are resolved to paths and formats, so a "
                        "table is reproduced from the registry rather than from remembered "
                        "flags. Use with --dataset and --method.")
    p.add_argument("--method", nargs="*", default=None,
                   help="Methods to take from --registry. Omit or pass '*' for every method "
                        "registered for the selected datasets.")
    p.add_argument("--no-normalize", action="store_true", help="Skip normalization; use input as-is.")
    p.add_argument("--select", default=None, choices=list(select_mod.SELECTION_MODES),
                   help="Run best-of-N selection before evaluating. 'gated-top-k' keeps only "
                        "candidates that are Valid, Improved and Novel, then ranks by the "
                        "selection oracle and takes the top --k per seed. Omit for no selection.")
    p.add_argument("--k", type=int, default=5, help="Generations kept per seed when --select is on.")
    p.add_argument("--select-from-n", type=int, default=None,
                   help="Candidate budget per seed before selection. Required with --select "
                        "when normalization is on, so best-of-N cannot silently become best-of-5.")
    p.add_argument("--selection-denominator", default=select_mod.BUDGET,
                   choices=list(select_mod.DENOMINATORS),
                   help="Denominator for rates after selection. 'budget' is k * n_seeds and is "
                        "constant across methods, so table rows are comparable; 'selected' is "
                        "the surviving row count and reproduces older tables.")
    p.add_argument("--seed-policy", default="intersection",
                   choices=["intersection", "reference", "union"],
                   help="Which seeds every method is scored against. 'intersection' (default) "
                        "uses seeds present in every method so budgets match.")
    p.add_argument("--selection-rng-seed", type=int, default=0,
                   help="Seed for --select random-k.")

    # --- Evaluation ---
    p.add_argument("--distribution", action="store_true", help="Enable FCD, SNN, etc.")
    p.add_argument("--property-metrics", action="store_true", help="Enable continuous property stats.")
    p.add_argument("--property-names", default=None, help="Comma-separated property names.")
    p.add_argument("--min-property-delta", type=float, default=1.0)
    p.add_argument("--reuse-precomputed", action="store_true", help="Reuse precomputed columns if present.")
    p.add_argument("--no-verbose", action="store_true", help="Suppress progress bars.")

    # --- Extra train SMILES ---
    p.add_argument(
        "--extra-train-smiles", nargs="+", metavar="NAME:PATH",
        help="Additional reference sets for memorization checking.",
    )

    # --- Data path overrides ---
    p.add_argument("--evaluator-dir", type=Path, default=DEFAULT_EVALUATOR_DIR,
                   help="Oracle used for both selection and scoring unless one of "
                        "--select-oracle / --score-oracle overrides it.")
    p.add_argument("--score-oracle", default=None,
                   help="Oracle that judges molecules (drives improvement and the "
                        "composite metrics). A directory, a .joblib, a smiles/prediction "
                        "table (.csv/.parquet), or 'minimol'. Defaults to --evaluator-dir.")
    p.add_argument("--select-oracle", default=None,
                   help="Oracle that filters candidates during selection. Same forms as "
                        "--score-oracle. Defaults to the scoring oracle, i.e. no split.")
    p.add_argument("--minimol-python", type=Path, default=None,
                   help="Interpreter used to score molecules an out-of-process oracle is "
                        "missing. Falls back to $PAIRIT_MINIMOL_PYTHON.")
    p.add_argument("--offline-oracles", action="store_true",
                   help="Fail instead of scoring when an out-of-process oracle has cache "
                        "misses. For environments without the scoring venv.")
    p.add_argument("--moleculenet-train-dir", type=Path, default=DEFAULT_MOLECULENET_TRAIN_DIR)
    p.add_argument("--moleculenet-test-dir", type=Path, default=DEFAULT_MOLECULENET_TEST_DIR)
    p.add_argument("--zinc250k-train-molecules", type=Path, default=DEFAULT_ZINC250K_TRAIN_MOLECULES)
    p.add_argument("--zinc250k-test-csv", type=Path, default=DEFAULT_ZINC250K_TEST_CSV)

    return p


def _apply_table_spec(args) -> int:
    """Fill args from a table spec, so `--table X` is the whole protocol.

    Explicit flags still win: a spec is a default, not a lock, so a one-off
    variation does not require editing the registry. Whatever ends up in effect
    is recorded either way.
    """
    from pairit.eval.registry import RegistryError, find_table, load_table_registry

    table_registry = args.table_registry
    if table_registry is None:
        if args.registry is None:
            print(
                "ERROR: --table needs --table-registry, or --registry so its sibling "
                "table_registry.tsv can be found.",
                file=sys.stderr,
            )
            return 1
        table_registry = Path(args.registry).parent / "table_registry.tsv"

    try:
        spec = find_table(load_table_registry(table_registry), args.table)
    except RegistryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.registry is None:
        args.registry = Path(table_registry).parent / "generation_registry.tsv"
    if not args.dataset:
        args.dataset = spec.datasets
    if args.method is None:
        args.method = spec.methods
    if args.select is None:
        args.select = spec.select
        args.k = spec.k
    if args.select_oracle is None:
        args.select_oracle = spec.select_oracle
    if args.score_oracle is None:
        args.score_oracle = spec.score_oracle
    args.seed_policy = spec.seed_policy
    args.selection_denominator = spec.denominator
    args.seed_filter = spec.seed_filter
    if args.output_prefix is None:
        args.output_prefix = spec.table_id
    if args.output_dir is None and spec.output:
        # A relative output in the table registry is relative to the LOBSTER
        # repo, not to the caller's cwd -- the markdown writer already resolves
        # it that way, and having the CSVs land somewhere else split one table's
        # artifacts across two directories.
        out = Path(spec.output)
        if not out.is_absolute():
            out = Path(os.environ.get("PAIRIT_TABLE_OUTPUT_ROOT", "./pairit_tables")) / out
        args.output_dir = out.parent

    args._table_spec = spec
    print(f"Table {spec.table_id!r}: {spec.description}")
    print(
        f"  datasets={','.join(spec.datasets)} methods={','.join(spec.methods)} "
        f"select={spec.select} k={spec.k} seeds={spec.seed_policy} "
        f"denominator={spec.denominator}"
        + (f" seed_filter={spec.seed_filter}" if spec.seed_filter else "")
    )
    return 0


def _load_from_registry(args, record, verbose):
    """Resolve registry rows to loaded generations.

    Each row carries its own `format` and `gens_per_seed`, so a table can mix
    per-dataset generation budgets -- which a single global --n-generations
    cannot express.
    """
    from pairit.eval.raw_loaders import load_generations
    from pairit.eval.registry import RegistryError, load_registry, select_rows

    try:
        rows = load_registry(args.registry)
    except RegistryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None, None, None

    chosen = select_rows(
        rows, datasets=args.dataset, methods=args.method, use_for=getattr(args, "table", None)
    )
    if not chosen:
        print(
            f"ERROR: no rows in {args.registry} matched dataset={args.dataset} "
            f"method={args.method}.",
            file=sys.stderr,
        )
        return None, None, None

    # Resolved BEFORE the generations, because `ddsbm_dir` needs it to name its starting
    # molecules -- those files carry only an `idx`, and without the reference list the
    # whole directory loads as one starting molecule.
    seed_rows = select_rows(
        rows, datasets=args.dataset, methods=None, role="seed_source",
        use_for=getattr(args, "table", None),
    )
    args._reference_seeds = _load_seed_sources(seed_rows, record, verbose)

    # Pool size per (dataset, method) from the registry, so --select-from-n need not
    # be remembered on the command line. Recorded values are measured, not guessed:
    # see docs/handoff/inventory_raw_pools.py in the lobster repo.
    args._registry_pool_n = {
        (r.dataset, r.method): float(r.gens_per_seed)
        for r in chosen
        if getattr(r, 'gens_per_seed', None)
    }


    frames = []
    for row in chosen:
        try:
            block = load_generations(
                row.path, row.format, dataset=row.dataset, method=row.method,
                ddsbm_reference_seeds=args._reference_seeds.get(row.dataset),
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {row.dataset}/{row.method} (line {row.line_no}): {e}", file=sys.stderr)
            return None, None, None
        if not len(block):
            print(f"WARNING: {row.dataset}/{row.method} loaded no rows", file=sys.stderr)
            continue
        frames.append(block)
        n_seeds = int(block["seed_smiles"].nunique())
        record.add_input(
            dataset=row.dataset,
            method=row.method,
            path=row.path,
            n_rows=len(block),
            n_seeds=n_seeds,
            gens_per_seed=round(len(block) / n_seeds, 3) if n_seeds else None,
            fmt=row.format,
            loader="load_generations",
            source="registry",
            registry_file=row.source_file,
            registry_line=row.line_no,
            description=row.description,
            notes=row.notes,
        )
        if row.status == "caution" and row.notes:
            record.warn(f"{row.dataset}/{row.method} is marked caution: {row.notes}")
        if verbose:
            print(f"  {row.dataset}/{row.method}: {len(block)} rows, {n_seeds} seeds")

    if not frames:
        print("ERROR: registry matched rows but none loaded any data.", file=sys.stderr)
        return None, None, None

    df = pd.concat(frames, ignore_index=True)
    prefix = "_".join(sorted({r.dataset for r in chosen}))
    # (dataset, method) -> configuration, for collapsing training replicates after
    # evaluation. Blank `replicate_group` maps a row to itself, i.e. a group of one,
    # which is why a registry written before the column existed is unaffected.
    args._replicate_groups = {(r.dataset, r.method): r.group_key() for r in chosen}
    args._replicate_labels = {
        r.group_key(): r.label for r in chosen if r.replicate_group and r.label
    }

    return df, f"registry_{prefix}", Path(args.registry).parent


def _load_seed_sources(seed_rows, record, verbose) -> dict:
    """``{dataset -> [starting molecule SMILES]}`` from rows with ``role='seed_source'``.

    Serves two consumers with different needs, which is why **file order is preserved**:

    * `seed_policy: reference` only needs membership, so order is irrelevant to it.
    * `ddsbm_dir` resolves its ``idx`` column *positionally* against this list. Sorting
      would silently pair every DDSBM generation with the wrong starting molecule -- a
      corruption that produces a full-looking table of meaningless numbers.

    An explicit row beats deriving the set from the dataset's test split: a table is often
    scored on a SUBSET of test (hERG uses the 85 blockers out of 111), and which subset is
    a property of the table, not of the dataset.
    """
    from pairit.eval.raw_loaders import _pick

    out: dict = {}
    for row in seed_rows:
        path = Path(row.path)
        if not path.exists():
            print(f"WARNING: seed_source {row.dataset}/{row.method}: missing {path}",
                  file=sys.stderr)
            continue
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        # REF-SMI is the TDC seed_lookup convention (REF-SMI = starting molecule,
        # PRB-SMI = its matched probe); without it those files load with no seed column
        # and DDSBM idx resolution silently collapses to one starting molecule.
        col = _pick(raw.columns, "seed_smiles", "smiles", "ref-smi")
        if col is None:
            print(f"WARNING: seed_source {row.dataset}/{row.method}: no smiles column "
                  f"among {list(raw.columns)}", file=sys.stderr)
            continue
        # dict.fromkeys dedups while keeping first-seen order.
        # Restrict to the STARTING class. A seeded improvement task starts from the
        # undesirable class and generates toward the desirable one, so the canonical seed
        # set is the undesirable half of the test split -- not the whole split. Without
        # this, BACE resolves 267 reference molecules instead of 135 (135 class-0 plus 132
        # class-1 that no method ever generates for), every method reports 132 seeds
        # "absent", and `denominator: budget` roughly doubles, halving every rate.
        #
        # Which label is the starting class follows `direction`, the same convention
        # `improvement` uses: `increasing` starts at 0 (BACE/BBBP/HIV), `decreasing`
        # starts at 1. Datasets with no label column (the continuous TDC tasks) are
        # untouched -- their seed sources are already the exact starting set.
        from pairit.eval.config import DATASET_CONFIG

        ds_cfg = DATASET_CONFIG.get(row.dataset, {})
        label_col = ds_cfg.get("label_col")
        if label_col and label_col in raw.columns:
            start_label = 1 if ds_cfg.get("direction") == "decreasing" else 0
            before = len(raw)
            raw = raw[raw[label_col] == start_label]
            if verbose:
                print(f"  seed_source {row.dataset}: {before} -> {len(raw)} rows "
                      f"({label_col}=={start_label}, the starting class)")

        seeds = list(dict.fromkeys(str(s) for s in raw[col].dropna()))
        out.setdefault(row.dataset, []).extend(seeds)
        record.add_input(
            dataset=row.dataset, method=row.method, path=row.path,
            n_rows=len(raw), n_seeds=len(seeds), gens_per_seed=None,
            fmt=row.format, loader="seed_source", source="registry",
            registry_file=row.source_file, registry_line=row.line_no,
            description=row.description, notes=row.notes,
        )
        if verbose:
            print(f"  seed_source {row.dataset}: {len(seeds)} starting molecules")
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def _write_markdown_table(spec, average, registry_path, record, note_column=True):
    """Emit the table's markdown, with and without a Note column.

    The table registry declares an `output` path; before this it declared one
    nothing wrote, so the .md files named there never existed and the tables
    were still assembled by hand.

    Labels come from the registry so a table reads with the names a person uses
    ("PairIT (real pairs)"), not the slugs.
    """
    from pairit.eval.registry import load_registry

    labels = {}
    if registry_path and Path(registry_path).exists():
        try:
            for row in load_registry(registry_path):
                if row.label:
                    labels[(row.dataset, row.method)] = row.label
        except Exception:
            pass

    metrics = [m for m in (spec.metrics or []) if m in average.columns]
    if not metrics:
        # Fall back to whatever explicit columns exist, so a spec with a stale
        # metric list still produces something rather than an empty table.
        metrics = [c for c in average.columns if "/" in c][:4]

    out_path = Path(spec.output)
    if not out_path.is_absolute():
        root = os.environ.get("PAIRIT_TABLE_OUTPUT_ROOT", "./pairit_tables")
        out_path = Path(root) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def render(with_note):
        lines = [
            f"# {spec.table_id}", "",
            spec.description, "",
            f"Generated by `pairit_eval --table {spec.table_id}`. "
            f"Seed policy `{spec.seed_policy}`, denominator `{spec.denominator}`"
            + (f", selection `{spec.select}` k={spec.k}" if spec.select else ", no selection")
            + ".", "",
        ]
        head = ["Dataset", "Method"] + metrics + (["Note"] if with_note else [])
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for _, r in average.sort_values(["dataset", "method"]).iterrows():
            key = (str(r["dataset"]), str(r["method"]))
            cells = [str(r["dataset"]), labels.get(key, str(r["method"]))]
            # An aggregated row carries `<metric>_sd` and `n_replicates`; a plain one
            # has neither, and then this is the same `{v:.2f}` as before.
            n_rep = int(r[N_REPLICATES]) if N_REPLICATES in average.columns else 1
            for m in metrics:
                v = r[m]
                sd = r.get(f"{m}_sd") if f"{m}_sd" in average.columns else None
                if pd.isna(v):
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(format_cell(v, sd, n_rep))
                else:
                    cells.append(str(v))
            if with_note:
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "Every number above is reproducible with the command named at the top; "
                  "the run's `_decisions.json` records which file produced each cell, which "
                  "oracles scored it, and which denominator each rate used."]
        return "\n".join(lines) + "\n"

    out_path.write_text(render(with_note=True))
    plain = out_path.with_name(out_path.stem + "_nonotes" + out_path.suffix)
    plain.write_text(render(with_note=False))
    record.add_output("table_markdown", out_path)
    record.add_output("table_markdown_nonotes", plain)
    return out_path, plain


def _record_inputs(record, df, path_by_method, source="cli"):
    """One record entry per (dataset, method) -- the same key as a table cell.

    n_seeds counts distinct seed SMILES, never seed_idx: seed_idx wraps at the
    batch size and reports the same number for files with very different seed
    counts.
    """
    if not len(df):
        return
    for (dataset, method), block in df.groupby(["dataset", "method"], sort=False):
        n_seeds = int(block["seed_smiles"].nunique()) if "seed_smiles" in block else None
        record.add_input(
            dataset=str(dataset),
            method=str(method),
            path=path_by_method.get(str(method)),
            n_rows=len(block),
            n_seeds=n_seeds,
            gens_per_seed=round(len(block) / n_seeds, 3) if n_seeds else None,
            source=source,
        )


def _run_selection(df, args, config, verbose, record=None):
    """Selection stage: one pass per (dataset, method), sharing one seed set.

    The novelty reference passed to the gate is the SAME set the metric uses. If
    they differed, selection would admit molecules the metric then calls
    memorized, and v_n_i would drop for no visible reason.
    """
    from pairit.eval import oracles as oracles_mod
    from pairit.eval.evaluate import _load_training_smiles

    cfg = select_mod.SelectionConfig(
        mode=args.select,
        k=args.k,
        denominator=args.selection_denominator,
        rng_seed=args.selection_rng_seed,
    )
    out_frames, reports = [], []
    for dataset, ds_block in df.groupby("dataset", sort=False):
        select_oracle = None
        if args.select in (select_mod.GATED_TOP_K, select_mod.TOP_K_UNGATED):
            spec = args.select_oracle or args.score_oracle or args.evaluator_dir
            select_oracle = oracles_mod.resolve_oracle(
                spec, dataset, dataset_config=config.dataset_config if config else None
            )
            if select_oracle is None:
                print(
                    f"ERROR: --select {args.select} ranks by oracle score but no oracle "
                    f"could be resolved for dataset {dataset!r}. Pass --select-oracle.",
                    file=sys.stderr,
                )
                raise SystemExit(1)

        novelty = _load_training_smiles(dataset, config) if config else None
        if config:
            # Record WHICH corpora novelty was checked against. add_novelty_reference
            # existed but had no caller, so every run reported novelty_references: []
            # and the reference was invisible -- which is exactly how it went unnoticed
            # that MoleculeNet novelty was checking one split and not the pretraining set.
            from pairit.eval.evaluate import training_smiles_sources

            if record is not None:
                # Report per-source molecule counts, not the deduplicated
                # combined total: passing `len(novelty)` for every entry would
                # make every source look like it contributed the whole
                # reference set. Read each source's own smiles column and
                # count independently.
                from pairit.eval.evaluate import _read_smiles_from_parquet
                for src in training_smiles_sources(dataset, config):
                    per_source_n = None
                    if src.exists():
                        try:
                            per_source_n = int(len(_read_smiles_from_parquet(src)))
                        except Exception:
                            per_source_n = None
                    record.add_novelty_reference(
                        dataset, src if src.exists() else None,
                        n_molecules=per_source_n)

        # The seed policy already ran as its own stage, so the frame is
        # restricted; take the seed set from it rather than resolving twice.
        seeds = sorted(set(ds_block["seed_smiles"].astype(str)))
        if verbose:
            print(f"  {dataset}: {len(seeds)} seeds after the seed policy")

        for method, block in ds_block.groupby("method", sort=False):
            picked, report = select_mod.select_generations(
                block,
                cfg,
                select_oracle=select_oracle,
                novelty_reference=novelty,
                seed_set=seeds,
                method=str(method),
                dataset=str(dataset),
            )
            out_frames.append(picked)
            reports.append(report)
            if verbose:
                yld = f"{report.gate_yield:.1f}%" if report.gate_yield is not None else "n/a"
                print(
                    f"    {method}: {report.candidates_in} -> {report.candidates_out} rows "
                    f"| selected {report.seeds_selected}, gate-failed "
                    f"{report.seeds_gate_failed}, absent {report.seeds_absent} | yield {yld}"
                )

    combined = pd.concat(out_frames, ignore_index=True) if out_frames else df.iloc[0:0]
    return combined, reports


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # --- Validate input mode ---
    if args.table:
        rc = _apply_table_spec(args)
        if rc:
            return rc

    # --reuse-precomputed short-circuits scoring entirely: it keeps the columns
    # already in the file. Combined with a newly-specified oracle that produces a
    # table which LOOKS dual-scored and is not -- the worst kind of wrong, since
    # nothing about the output says so.
    if args.reuse_precomputed and (args.score_oracle or args.select_oracle):
        print(
            "ERROR: --reuse-precomputed keeps the scores already in the file, so "
            "--score-oracle / --select-oracle would have no effect and the output would "
            "silently be scored by whichever oracle produced the file.\n"
            "       Drop --reuse-precomputed to score with the oracle you asked for.",
            file=sys.stderr,
        )
        return 1

    modes = [bool(args.inputs), bool(args.methods), bool(args.registry)]
    if sum(modes) > 1:
        print(
            "ERROR: choose one input mode -- positional files, --methods, or --registry.",
            file=sys.stderr,
        )
        return 1
    if not any(modes):
        print("ERROR: Provide input file(s), --methods, or --registry.", file=sys.stderr)
        return 1

    # --- Build config ---
    config = build_eval_config(
        evaluator_dir=args.evaluator_dir,
        select_oracle=args.select_oracle,
        score_oracle=args.score_oracle,
        minimol_python=args.minimol_python,
        offline_oracles=args.offline_oracles,
        moleculenet_train_dir=args.moleculenet_train_dir,
        moleculenet_test_dir=args.moleculenet_test_dir,
        zinc250k_train_molecules=args.zinc250k_train_molecules,
        zinc250k_test_csv=args.zinc250k_test_csv,
    )

    verbose = not args.no_verbose

    # Every run records what it decided. Generated, not hand-written, so it
    # cannot drift from the run it describes.
    record = DecisionRecord(argv=["evaluate.py"] + list(argv or sys.argv[1:]))
    record.set_config(config, extra={
        "select": args.select,
        "k": args.k if args.select else None,
        "select_from_n": args.select_from_n,
        "selection_denominator": args.selection_denominator if args.select else None,
        "seed_policy": args.seed_policy if args.select else None,
        "normalize": not args.no_normalize,
        "n_generations": args.n_generations,
        "min_property_delta": args.min_property_delta,
        "reuse_precomputed": args.reuse_precomputed,
    })
    record.set_denominators({"metrics": METRIC_DENOMINATORS, "glossary": DENOMINATOR_GLOSSARY})

    # --- Load data ---
    if args.registry:
        df, default_prefix, default_output_dir = _load_from_registry(args, record, verbose)
        if df is None:
            return 1
    elif args.methods:
        # Method mode: load from registered raw loaders
        frames = []
        for method in args.methods:
            raw = load_method_all(method)
            if args.dataset:
                raw = raw[raw["dataset"].isin(args.dataset)].copy()
            if len(raw) == 0:
                print(f"WARNING: No data for method '{method}'", file=sys.stderr)
                continue
            frames.append(raw)
        if not frames:
            print("ERROR: No data loaded for any method.", file=sys.stderr)
            return 1
        df = pd.concat(frames, ignore_index=True)
        _record_inputs(record, df, {}, source="method-registry")
        default_prefix = "_".join(args.methods)
        default_output_dir = Path(DEFAULT_RESULTS_DIR) / "evaluate"
    else:
        # File mode: load CSV/Parquet files
        parsed = parse_named_inputs(args.inputs)
        df = load_and_tag_inputs(parsed)
        if args.dataset:
            df = df[df["dataset"].isin(args.dataset)].copy()
        _record_inputs(record, df, {name: path for name, path in parsed}, source="cli")
        first_path = parsed[0][1]
        default_prefix = first_path.stem
        default_output_dir = first_path.parent

    output_dir = args.output_dir or default_output_dir
    output_prefix = args.output_prefix or default_prefix

    # --- Normalize ---
    # Ordering is load -> [normalize] -> [select] -> evaluate, and it is enforced.
    # Normalizing AFTER selection would pad variable-length selections back up to
    # k and score the padding as invalid. Normalizing BEFORE selection with the
    # default --n-generations would truncate 64 candidates to 5, silently turning
    # best-of-64 into best-of-5 with no visible symptom.
    # The pool size is a property of the generations, not of the invocation, so take it
    # from the registry rather than making the caller remember a flag. Without this a
    # table spec is not the whole protocol: running one "by the spec" would truncate to
    # --n-generations and silently report best-of-5.
    if args.select and not args.no_normalize and args.select_from_n is None:
        pool = getattr(args, "_registry_pool_n", None)
        if pool:
            lo, hi = min(pool.values()), max(pool.values())
            # Normalizing every method up to the largest pool pads the smaller ones with
            # empty slots and scores the padding as invalid, so a method that legitimately
            # generates fewer candidates is punished for it. Inferring is only safe when
            # the pools are comparable; when they are not, the budget is a decision about
            # what the table is comparing and must be made explicitly.
            if hi > 2 * lo:
                by_size = sorted(((v, f"{d}/{m}") for (d, m), v in pool.items()), reverse=True)
                print(
                    f"ERROR: candidate pools differ too much to infer --select-from-n "
                    f"({lo:g} to {hi:g}). Normalizing to {hi:g} would pad the smaller pools "
                    f"and score the padding as invalid.\n       Largest: "
                    + ", ".join(f"{n}={v:g}" for v, n in by_size[:3])
                    + "\n       Smallest: "
                    + ", ".join(f"{n}={v:g}" for v, n in by_size[-3:])
                    + "\n       Pass --select-from-n explicitly to state the budget this "
                      "table compares at.",
                    file=sys.stderr,
                )
                return 1
            args.select_from_n = int(hi)
            print(f"  --select-from-n {args.select_from_n} (inferred from registry gens_per_seed)")

    if args.select and not args.no_normalize and args.select_from_n is None:
        print(
            f"ERROR: --select with normalization needs --select-from-n N, the number of "
            f"candidates per seed to select from. Without it, --n-generations "
            f"({args.n_generations}) truncates the candidate pool first and "
            f"'best-of-N' silently becomes 'best-of-{args.n_generations}'.\n"
            f"       Pass --select-from-n 64, record gens_per_seed on the registry rows "
            f"so it can be inferred, or --no-normalize to skip normalization entirely.",
            file=sys.stderr,
        )
        return 1

    if not args.no_normalize:
        if verbose:
            print(f"Normalizing ({len(df)} rows) ...")
        n_per_seed = args.select_from_n if args.select else args.n_generations
        n_gen_map = n_generations_map(df, n_per_seed)
        df = normalize_generation_table(df, config=config, n_generations_by_dataset=n_gen_map)
        if verbose:
            print(f"  -> {len(df)} rows after normalization")

    # --- Seed policy ---
    # Runs whether or not selection does. Previously seed resolution lived inside
    # the selection stage, so a table with no --select got no seed policy at all
    # and each method kept its own seeds -- on MoleculeNet that meant one method
    # scored over 1150 generations and the rest over 1155.
    # `reference` needs a canonical seed set; without one it raised. Registered
    # `seed_source` rows supply it, and a table using `reference` with none is a
    # configuration error worth naming rather than a traceback.
    ref_seeds = getattr(args, "_reference_seeds", None) or None
    if args.seed_policy == "reference" and not ref_seeds:
        print(
            "ERROR: seed policy 'reference' needs a canonical seed set, and no row with "
            "role='seed_source' is registered for this table.\n"
            "  Add one (role=seed_source, path=<the file listing the seeds>, use_for=<table id>), "
            "or set seed_policy to 'intersection'.",
            file=sys.stderr,
        )
        return 1
    # Before the seed policy on purpose: a molecule the table does not score must not
    # shrink an intersection, and must not consume a method's candidate budget.
    filter_name = getattr(args, "seed_filter", "") or ""
    if filter_name:
        df, filter_reports, ref_seeds = select_mod.apply_seed_filter(
            df, filter_name, reference_seeds=ref_seeds
        )
        for rep in filter_reports:
            record.add_seed_filter(rep)
            if rep.n_excluded:
                print(
                    f"  seed_filter '{rep.filter_name}' on {rep.dataset}: "
                    f"{rep.n_before} -> {rep.n_after} starting molecules "
                    f"({rep.n_excluded} excluded, applied to every method)"
                )

    df, seed_reports = select_mod.apply_seed_policy(
        df, policy=args.seed_policy, reference_seeds=ref_seeds
    )
    for rep in seed_reports:
        record.add_seed_policy(rep)
        if verbose and rep.rows_dropped:
            print(
                f"  seed policy '{rep.policy}': {rep.n_seeds_resolved} shared seeds, "
                f"{rep.rows_dropped} rows dropped"
            )
        if rep.collapsed:
            pct = 100 * (rep.retention or 0)
            worst = min(rep.per_method, key=lambda m: m["seeds_after"] / max(m["seeds_before"], 1))
            print(
                f"ERROR: seed policy '{rep.policy}' kept only {rep.n_seeds_resolved} shared "
                f"seeds -- {worst['method']} retains {pct:.0f}% of its {worst['seeds_before']}. "
                f"These methods were probably built on different seed sets, and every row would "
                f"be scored on a small corner of the test set.\n"
                f"       Use --seed-policy union to keep each method's own seeds, or "
                f"--seed-policy reference to score against the canonical test seeds.",
                file=sys.stderr,
            )
            return 1
        if not rep.denominators_equal:
            counts = {m["method"]: m["seeds_after"] for m in rep.per_method}
            record.warn(
                f"seed policy '{rep.policy}' left methods with different seed counts "
                f"({counts}); rows of this table are not directly comparable"
            )
    if len(df) == 0:
        print("ERROR: the seed policy left no rows.", file=sys.stderr)
        return 1

    # --- Select ---
    selection_reports = []
    if args.select:
        df, selection_reports = _run_selection(df, args, config, verbose, record=record)
        for rep in selection_reports:
            record.add_selection(rep)
            if rep.seeds_absent:
                record.warn(
                    f"{rep.dataset}/{rep.method}: {rep.seeds_absent} seeds have no rows in "
                    f"this file (distinct from the {rep.seeds_gate_failed} that were "
                    f"attempted and failed the gate)"
                )
        if len(df) == 0:
            print("ERROR: selection produced no rows.", file=sys.stderr)
            return 1

    # --- Extra train SMILES ---
    extra_train: Optional[Dict[str, Set[str]]] = None
    if args.extra_train_smiles:
        extra_train = _parse_extra_train_smiles(args.extra_train_smiles)

    # --- Property names ---
    property_names = None
    if args.property_names:
        property_names = [n.strip() for n in args.property_names.split(",")]

    # --- Evaluate ---
    if verbose:
        print(f"Evaluating ({len(df)} rows) ...")
    result = evaluate(
        df,
        config=config,
        datasets=args.dataset,
        verbose=verbose,
        compute_distribution_metrics=args.distribution,
        compute_property_metrics=args.property_metrics,
        property_names=property_names,
        min_property_delta=args.min_property_delta,
        reuse_precomputed=args.reuse_precomputed,
        extra_train_smiles=extra_train,
    )

    # --- Record what actually ran ---
    for ds, res in (result.resources or {}).items():
        for role in ("select", "score"):
            oracle = res.get(f"{role}_oracle")
            if oracle is not None:
                record.add_oracle(role, oracle, dataset=ds)
                if getattr(oracle, "n_unscorable", 0):
                    record.warn(
                        f"{ds}: {role} oracle {oracle.name!r} could not score "
                        f"{oracle.n_unscorable} molecules; they count as failures, not as "
                        f"absent from the denominator"
                    )
    # --- Explicit column names ---
    # Every rate gets a `{metric}/{denominator}` column beside the legacy one, so
    # a number cannot travel without its denominator attached. Added, never
    # renamed, so existing readers are unaffected.
    denom_token = (
        "n_budget"
        if args.select and args.selection_denominator == select_mod.BUDGET
        else "n_gen"
    )
    result.average = explicit_columns(result.average, denom_token)
    result.per_seed = explicit_columns(result.per_seed, denom_token)
    if denom_token == "n_budget":
        # Recompute against k * n_seeds rather than relabelling n_total, which
        # after selection is the surviving row count.
        budgets = {(r.dataset, r.method): r.budget for r in selection_reports}
        result.average = budget_columns(result.average, budgets)
    record.set_column_aliases(active_aliases(denom_token))

    # --- Collapse training replicates -------------------------------------------
    # Runs of one configuration at different random seeds are registered separately
    # (so every file keeps its provenance) and become one row here, mean +/- sd.
    # Runs on the SAME random seed are unaffected: without a `replicate_group` every
    # row is its own group, so this is a no-op for every pre-existing table.
    replicate_reports: list = []
    groups = getattr(args, "_replicate_groups", None)
    if groups:
        result.average, replicate_reports = aggregate_replicates(
            result.average, groups, label_map=getattr(args, "_replicate_labels", None)
        )
        if verbose and has_replicates(replicate_reports):
            n = sum(1 for r in replicate_reports if r["n_replicates"] > 1)
            print(f"  collapsed {n} replicate group(s) into single rows")

    # Recorded AFTER the column work, so the record carries the same numbers the
    # table reports. Recording first left the run record without the very
    # columns the table was built from -- vin/n_budget was simply absent.
    record.add_metrics(result.average)
    for rep in replicate_reports:
        record.add_replicate_group(rep)

    # --- Emit the table's markdown ---
    spec = getattr(args, "_table_spec", None)
    if spec is not None and spec.output and not args.no_save:
        md, plain = _write_markdown_table(
            spec, result.average, args.registry, record, note_column=True
        )
        if verbose:
            print(f"\nTable written:\n  {md}\n  {plain}")

    # --- Save ---
    if not args.no_save:
        fmt = args.output_format
        pg_path = _save_result(result.per_generation, output_dir / f"{output_prefix}_per_generation", fmt)
        ps_path = _save_result(result.per_seed, output_dir / f"{output_prefix}_per_seed", fmt)
        av_path = _save_result(result.average, output_dir / f"{output_prefix}_average", fmt)
        record.add_output("per_generation", pg_path)
        record.add_output("per_seed", ps_path)
        record.add_output("average", av_path)
        dec_path = record.write(output_dir / f"{output_prefix}_decisions.json")
        record.add_output("decisions", dec_path)
        if verbose:
            print(f"\nSaved:")
            print(f"  decisions:      {dec_path}")
            print(f"  per_generation: {pg_path}")
            print(f"  per_seed:       {ps_path}")
            print(f"  average:        {av_path}")

    # --- Print summary ---
    summary = summarize_average(result.average)
    print(f"\n{'='*60}")
    print("Summary (average)")
    print(f"{'='*60}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 20)
    print(summary.to_string(index=False))

    # The decisions summary goes last, so the thing a reader most needs in order
    # to interpret the table above is the last thing on screen.
    print(record.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
