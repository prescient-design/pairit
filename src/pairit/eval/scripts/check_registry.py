#!/usr/bin/env python
"""Validate a generation registry, and fill in its derived columns.

    python scripts/check_registry.py docs/generation_registry.tsv
    python scripts/check_registry.py docs/generation_registry.tsv --fix
    python scripts/check_registry.py docs/generation_registry.tsv --strict   # CI

Advisory by default: it reports problems and exits 0, so a half-known path can
still be registered. An unrecorded path is the failure this file exists to
prevent, and refusing the entry would push people back to not recording it.
Use --strict before pulling numbers into a table.

--fix opens every registered file, counts distinct seeds and generations per
seed, and writes them back. Those two columns are derived, never hand-authored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairit.eval.registry import (  # noqa: E402
    RegistryError,
    apply_derived,
    check_registry,
    write_registry,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registry", type=Path)
    ap.add_argument("--fix", action="store_true",
                    help="Write derived n_seeds / gens_per_seed back into the file.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if anything is wrong. For CI and before publishing.")
    ap.add_argument("--no-derive", action="store_true",
                    help="Only check paths; do not open files. Much faster.")
    args = ap.parse_args(argv)

    try:
        results, file_errors = check_registry(args.registry, derive=not args.no_derive)
    except RegistryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # A use_for naming no table is a dangling citation: the row claims to back
    # something that cannot be regenerated. Cheap to check, and it catches the
    # common case of registering a run before writing its table spec.
    known_tables = set()
    table_path = args.registry.parent / "table_registry.tsv"
    if table_path.exists():
        try:
            from pairit.eval.registry import load_table_registry

            known_tables = {t.table_id for t in load_table_registry(table_path)}
        except RegistryError:
            pass

    n_err = n_warn = 0
    for err in file_errors:
        print(f"ERROR  {err}")
        n_err += 1

    for r in results:
        where = f"line {r.row.line_no}  {r.row.dataset}/{r.row.method}"
        for e in r.errors:
            print(f"ERROR  {where}: {e}")
            n_err += 1
        for w in r.warnings:
            print(f"WARN   {where}: {w}")
            n_warn += 1
        if known_tables and r.row.use_for:
            dangling = [
                u.strip() for u in r.row.use_for.split(",")
                if u.strip() and u.strip() not in known_tables
            ]
            if dangling:
                print(
                    f"WARN   {where}: use_for names no table in table_registry.tsv: "
                    f"{dangling}. The row claims to back something that cannot be "
                    f"regenerated; add the table spec or fix the id."
                )
                n_warn += 1

    checked = len(results)
    derived = sum(1 for r in results if r.observed_n_seeds is not None)
    print(
        f"\n{checked} rows checked, {derived} opened and measured, "
        f"{n_err} error(s), {n_warn} warning(s)"
    )

    if args.fix:
        rows = apply_derived(results)
        write_registry(rows, args.registry)
        print(f"Wrote derived columns back to {args.registry}")

    if n_err and not args.strict:
        print(
            "\nAdvisory mode: exiting 0 so an incomplete entry can still be recorded.\n"
            "Run with --strict before using these numbers in a table."
        )
    return 1 if (args.strict and (n_err or n_warn)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
