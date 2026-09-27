#!/usr/bin/env python
"""Pretty-print the generation registry.

    python scripts/show_registry.py <registry.tsv> --tables        # LIST EVERY TABLE + protocol
    python scripts/show_registry.py <registry.tsv>                 # every run, grouped by table
    python scripts/show_registry.py <registry.tsv> --table X       # one table's runs
    python scripts/show_registry.py <registry.tsv> --long           # full descriptions
    python scripts/show_registry.py <registry.tsv> --paths          # with paths

Default view is one line per run: label, seeds x gens/seed, and status. That is
usually the question ("what is actually in here?"), and full prose descriptions
make it unreadable, so those need --long.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairit.eval.registry import RegistryError, load_registry  # noqa: E402

MARK = {"ok": " ", "caution": "!", "blocked": "x"}



def _show_tables(args, rows) -> int:
    """List every registered table with the protocol that defines it.

    The protocol -- not the paths -- is what makes a number reproducible: the same
    generations scored at a different k, seed policy or oracle give a different answer.
    So this prints those fields rather than a bare list of ids.
    """
    from pairit.eval.registry import load_table_registry

    path = args.table_registry or args.registry.parent / "table_registry.tsv"
    if not path.exists():
        print(f"ERROR: no table registry at {path}; pass --table-registry", file=sys.stderr)
        return 1
    specs = load_table_registry(path)

    counts = defaultdict(int)
    for r in rows:
        for use in (r.use_for.split(",") if r.use_for else []):
            counts[use.strip()] += 1

    w = max(len(s.table_id) for s in specs) + 2
    print(f"{len(specs)} registered tables in {path}\n")
    print(f"  {'table_id':{w}s} {'rows':>5}  {'select':<13} {'k':>3}  "
          f"{'seed policy':<13} {'denominator':<11} score oracle")
    print("  " + "─" * (w + 62))
    for spec in sorted(specs, key=lambda s: s.table_id):
        oracle = (spec.score_oracle or "-").split("/")[-1] or "-"
        print(f"  {spec.table_id:{w}s} {counts.get(spec.table_id, 0):>5}  "
              f"{(spec.select or '-'):<13} {spec.k or '-':>3}  "
              f"{spec.seed_policy:<13} {spec.denominator:<11} {oracle}")
    orphan = sorted(t for t in counts if t not in {s.table_id for s in specs})
    if orphan:
        print(f"\n  {len(orphan)} use_for value(s) with NO table spec: {', '.join(orphan)}")
    empty = sorted(s.table_id for s in specs if not counts.get(s.table_id))
    if empty:
        print(f"  {len(empty)} table(s) with NO rows backing them: {', '.join(empty)}")
    print(f"\n  Regenerate one with:\n"
          f"    python scripts/evaluate.py --table <table_id> \\\n"
          f"        --registry {args.registry} --output-dir <dir>")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("registry", type=Path)
    ap.add_argument("--table", default=None, help="Show only rows backing this table/figure id.")
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--long", action="store_true", help="Show the full description of each run.")
    ap.add_argument("--paths", action="store_true", help="Show the file path of each run.")
    ap.add_argument("--role", default="generations", help="Role to show; 'all' for every role.")
    ap.add_argument("--tables", action="store_true",
                    help="List every registered table with its protocol and row count, "
                         "instead of listing runs. This is the 'what can I regenerate?' view.")
    ap.add_argument("--table-registry", type=Path, default=None,
                    help="Table registry TSV. Defaults to table_registry.tsv beside the "
                         "generation registry.")
    args = ap.parse_args(argv)

    try:
        rows = load_registry(args.registry)
    except RegistryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.tables:
        return _show_tables(args, rows)

    if args.role != "all":
        rows = [r for r in rows if r.role == args.role]
    if args.dataset:
        rows = [r for r in rows if r.dataset in args.dataset]
    if args.table:
        rows = [r for r in rows if args.table in [u.strip() for u in r.use_for.split(",")]]
    if not rows:
        print("No matching rows.")
        return 0

    by_table = defaultdict(list)
    for r in rows:
        for use in (r.use_for.split(",") if r.use_for else ["(uncited)"]):
            by_table[use.strip() or "(uncited)"].append(r)

    width = max(len((r.label or r.method)) for r in rows) + 2
    for table in sorted(by_table):
        members = sorted(by_table[table], key=lambda r: (r.dataset, r.label or r.method))
        print(f"\n{table}  ({len(members)} runs)")
        print("─" * (len(table) + 14))
        for r in members:
            name = r.label or r.method
            shape = (
                f"{r.n_seeds} seeds x {r.gens_per_seed:g}"
                if r.n_seeds and r.gens_per_seed
                else "not measured"
            )
            print(f"  {MARK.get(r.status, '?')} {r.dataset:11s} {name:{width}s} {shape}")
            if args.long and r.description:
                print(textwrap.fill(r.description, 96,
                                    initial_indent=" " * 6, subsequent_indent=" " * 6))
            if r.notes:
                # Notes are often several sentences of caveat. In the compact view
                # one line is enough to flag that a caveat exists; --long has it all.
                note = r.notes if args.long else textwrap.shorten(r.notes, 88, placeholder=" …")
                print(textwrap.fill(f"note: {note}", 96,
                                    initial_indent=" " * 6, subsequent_indent=" " * 12))
            if args.paths:
                print(f"      {r.path}")

    n_caution = sum(1 for r in rows if r.status == "caution")
    n_blocked = sum(1 for r in rows if r.status == "blocked")
    print(f"\n{len(rows)} runs" + (f", {n_caution} caution (!)" if n_caution else "")
          + (f", {n_blocked} blocked (x)" if n_blocked else ""))
    if not args.long:
        print("Run with --long for full descriptions, --paths for file locations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
