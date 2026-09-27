"""Refilter: apply V,I,N filtering and selection to pre-existing generation parquets.

Algorithm
---------
This script operates on molecules that have ALREADY been generated (by
``pairit_eval``, ``pairit_best_of_n``, ``pairit_beam_search``, or
``pairit_iterative_gen``). It does NOT run the model. Instead it:

1. **Load** a parquet of generated molecules (must contain ``smiles_generated``
   and ``smiles_seed`` columns at minimum).
2. **Validate** each SMILES with RDKit (if ``is_valid`` column is missing).
3. **Compute properties** for each molecule (if property columns are missing,
   or ``--recompute`` is set).
4. **Check memorization** against reference SMILES files (if provided).
5. **Compute improvement** flags per property (gen > seed for increasing, etc.).
6. **Apply V,I,N hard-gate** — keep only Valid, Improved, Novel molecules.
7. **Select top-k** per seed (or globally) by a ranking property.
8. **Save** the enriched parquet with all flags, plus a summary parquet of
   selected molecules.

This is useful when you want to:
- Apply stricter filtering to an existing ``pairit_eval`` output.
- Change the ranking property or direction without re-generating.
- Add memorization checking after the fact.
- Compare different selection strategies on the same generation set.

Usage
-----
::

    python -m pairit.inference.refilter \\
        --input /path/to/all_generations.parquet \\
        --output /path/to/output_dir/ \\
        --ranking-property logp \\
        --ranking-direction increasing \\
        --reference /path/to/zinc_train.parquet \\
        --reference /path/to/bace_train.parquet \\
        --top-k 1

Also works with ``pairit_eval`` conditional parquets::

    python -m pairit.inference.refilter \\
        --input /path/to/conditional_*_generated_molecules.parquet \\
        --ranking-property Class \\
        --ranking-direction increasing
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pairit.callbacks.molecule_validation.property_registry import PROPERTY_REGISTRY
from pairit.callbacks.molecule_validation.smiles_utils import SmilesUtils
from pairit.inference.filters import VINFilter
from pairit.inference.scorers import ScoredCandidate
from pairit.inference.strategies import GlobalTopK, TopKPerSeed
from pairit.inference.utils import (
    check_improvement,
    check_memorization_batch,
    compute_properties,
    compute_tanimoto_distances,
    load_reference_smiles_files,
)

logger = logging.getLogger(__name__)


def load_and_normalize(input_path: str) -> pd.DataFrame:
    """Load a parquet and normalize column names to the inference schema.

    Handles both inference output format (``smiles_generated``,
    ``smiles_seed``) and ``pairit_eval`` format (``smiles_generated``,
    ``smiles_conditioning``).

    Parameters
    ----------
    input_path : str
        Path to a parquet file.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with at least ``smiles_generated`` and
        ``smiles_seed`` columns.
    """
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Normalize column names
    if "smiles_conditioning" in df.columns and "smiles_seed" not in df.columns:
        df = df.rename(columns={"smiles_conditioning": "smiles_seed"})

    if "smiles_generated" not in df.columns:
        raise ValueError(
            f"Input parquet must have 'smiles_generated' column. "
            f"Found: {list(df.columns)}"
        )
    if "smiles_seed" not in df.columns:
        raise ValueError(
            f"Input parquet must have 'smiles_seed' (or 'smiles_conditioning') column. "
            f"Found: {list(df.columns)}"
        )

    return df


def enrich_dataframe(
    df: pd.DataFrame,
    property_names: list[str],
    ranking_property: str,
    ranking_direction: str,
    reference_canon_sets: dict[str, set[str]],
    recompute: bool = False,
) -> pd.DataFrame:
    """Add validity, properties, memorization, improvement, and V,I,N flags.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``smiles_generated`` and ``smiles_seed`` columns.
    property_names : list[str]
        Properties to compute.
    ranking_property : str
        Property for V,I,N improvement check.
    ranking_direction : str
        ``"increasing"`` or ``"decreasing"``.
    reference_canon_sets : dict[str, set[str]]
        Reference sets for memorization.
    recompute : bool
        If True, recompute even if columns already exist.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all flags.
    """
    n = len(df)

    # Validity
    if "is_valid" not in df.columns or recompute:
        logger.info("Computing validity...")
        df["is_valid"] = [SmilesUtils.is_valid(s) for s in df["smiles_generated"]]
    n_valid = df["is_valid"].sum()
    logger.info(f"  Valid: {n_valid}/{n} ({100 * n_valid / max(1, n):.1f}%)")

    # Canonical SMILES
    if "smiles_canonical" not in df.columns or recompute:
        df["smiles_canonical"] = [
            SmilesUtils.canonicalize(s) if v else None
            for s, v in zip(df["smiles_generated"], df["is_valid"])
        ]

    # Properties for generated molecules
    for prop in property_names:
        if prop not in df.columns or recompute:
            logger.info(f"Computing {prop} for generated molecules...")
            props = compute_properties(df["smiles_generated"].tolist(), [prop])
            df[prop] = props[prop]

    # Seed properties
    for prop in property_names:
        seed_col = f"{prop}_seed"
        if seed_col not in df.columns or recompute:
            logger.info(f"Computing {prop} for seed molecules...")
            seed_props = compute_properties(df["smiles_seed"].tolist(), [prop])
            df[seed_col] = seed_props[prop]

    # Deltas
    for prop in property_names:
        delta_col = f"{prop}_delta"
        if delta_col not in df.columns or recompute:
            df[delta_col] = df[prop] - df[f"{prop}_seed"]

    # Tanimoto to seed
    if "tanimoto_to_seed" not in df.columns or recompute:
        logger.info("Computing Tanimoto distances...")
        seeds = df["smiles_seed"].tolist()
        unique_seeds = list(dict.fromkeys(seeds))
        seed_to_idx = {s: i for i, s in enumerate(unique_seeds)}
        seed_indices = [seed_to_idx[s] for s in seeds]
        df["tanimoto_to_seed"] = compute_tanimoto_distances(
            df["smiles_generated"].tolist(), unique_seeds, seed_indices
        )

    # Memorization
    if reference_canon_sets:
        logger.info("Checking memorization...")
        is_mem, mem_by_ref = check_memorization_batch(
            df["smiles_generated"].tolist(),
            df["is_valid"].tolist(),
            reference_canon_sets,
        )
        df["is_memorized"] = is_mem
        for ref_name in reference_canon_sets:
            df[f"memorized_in_{ref_name}"] = [m.get(ref_name, False) for m in mem_by_ref]
        logger.info(f"  Memorized: {sum(is_mem)}/{n_valid} valid")
    elif "is_memorized" not in df.columns:
        df["is_memorized"] = False

    # Direction map
    direction_map: dict[str, str] = {}
    for pname in property_names:
        if pname == ranking_property:
            direction_map[pname] = ranking_direction
        elif pname in PROPERTY_REGISTRY:
            direction_map[pname] = PROPERTY_REGISTRY[pname][1]
        else:
            direction_map[pname] = "increasing"

    # Improvement and V,I,N flags
    for prop in property_names:
        imp_col = f"is_improved_{prop}"
        vin_col = f"is_vin_{prop}"
        direction = direction_map[prop]

        df[imp_col] = [
            check_improvement(g, s, direction)
            for g, s in zip(df[prop], df[f"{prop}_seed"])
        ]
        df[vin_col] = df["is_valid"] & df[imp_col] & ~df["is_memorized"]

    return df


def select_from_dataframe(
    df: pd.DataFrame,
    ranking_property: str,
    ranking_direction: str,
    top_k: int = 1,
    vin_hard_gate: bool = True,
    vin_fallback: str = "valid_only",
    mode: str = "per_seed",
) -> pd.DataFrame:
    """Apply V,I,N filtering and top-k selection to an enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame with V,I,N flags.
    ranking_property : str
        Property to rank by.
    ranking_direction : str
        ``"increasing"`` or ``"decreasing"``.
    top_k : int
        Number of molecules to select per seed (or globally).
    vin_hard_gate : bool
        Whether to apply V,I,N hard gate.
    vin_fallback : str
        Fallback mode: ``"valid_only"`` or ``"none"``.
    mode : str
        ``"per_seed"`` or ``"global"``.

    Returns
    -------
    pd.DataFrame
        Selected molecules.
    """
    # Build ScoredCandidates from DataFrame
    candidates: list[ScoredCandidate] = []
    for i, row in df.iterrows():
        props = {p: row.get(p) for p in df.columns
                 if p in PROPERTY_REGISTRY or p == ranking_property}
        seed_props = {p.replace("_seed", ""): row.get(p) for p in df.columns
                      if p.endswith("_seed") and not p.startswith("smiles")}
        improved = {p.replace("is_improved_", ""): row.get(p, False)
                    for p in df.columns if p.startswith("is_improved_")}

        c = ScoredCandidate(
            smiles=row.get("smiles_generated", ""),
            canonical=row.get("smiles_canonical"),
            seed_smiles=row.get("smiles_seed", ""),
            seed_idx=row.get("seed_idx", 0),
            is_valid=row.get("is_valid", False),
            is_memorized=row.get("is_memorized", False),
            is_improved=improved,
            properties=props,
            seed_properties=seed_props,
            tanimoto_to_seed=row.get("tanimoto_to_seed"),
            round_num=row.get("round", 0),
        )
        candidates.append(c)

    # Filter
    if vin_hard_gate:
        f = VINFilter(ranking_property=ranking_property, fallback=vin_fallback)
        per_seed = mode == "per_seed"
        filtered = f.filter(candidates, per_seed=per_seed)
    else:
        filtered = [c for c in candidates if c.is_valid]

    # Select
    if mode == "per_seed":
        strategy = TopKPerSeed()
    else:
        strategy = GlobalTopK()

    selected = strategy.select(
        filtered, k=top_k,
        property_name=ranking_property,
        direction=ranking_direction,
    )

    # Mark selected in original DataFrame
    selected_canonical = {c.canonical for c in selected if c.canonical}
    df = df.copy()
    df["selected"] = df["smiles_canonical"].isin(selected_canonical)

    return df[df["selected"]].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Refilter: apply V,I,N filtering to pre-existing generation parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input parquet (all_generations.parquet or pairit_eval output)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: {input_dir}/refiltered/)",
    )
    parser.add_argument(
        "--ranking-property", "-p", default="logp",
        help="Property to rank by (default: logp)",
    )
    parser.add_argument(
        "--ranking-direction", "-d", default=None,
        help="Direction: 'increasing' or 'decreasing' (default: from PROPERTY_REGISTRY)",
    )
    parser.add_argument(
        "--reference", "-r", action="append", default=[],
        help="Reference parquet for memorization (can specify multiple times)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=1,
        help="Number of molecules to select per seed (default: 1)",
    )
    parser.add_argument(
        "--mode", choices=["per_seed", "global"], default="per_seed",
        help="Selection mode (default: per_seed)",
    )
    parser.add_argument(
        "--no-vin", action="store_true",
        help="Disable V,I,N hard gate (select from all valid)",
    )
    parser.add_argument(
        "--vin-fallback", choices=["valid_only", "none"], default="valid_only",
        help="Fallback when no V,I,N candidates (default: valid_only)",
    )
    parser.add_argument(
        "--properties", nargs="+", default=None,
        help="Properties to compute (default: [ranking_property, logp, qed, sa])",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute properties even if columns exist",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Resolve direction
    direction = args.ranking_direction
    if direction is None:
        if args.ranking_property in PROPERTY_REGISTRY:
            direction = PROPERTY_REGISTRY[args.ranking_property][1]
        else:
            direction = "increasing"

    # Property names
    prop_names = args.properties or list(dict.fromkeys(
        [args.ranking_property, "logp", "qed", "sa"]
    ))

    # Load references
    ref_sets: dict[str, set[str]] = {}
    if args.reference:
        ref_specs = [
            {"name": Path(p).stem, "path": p}
            for p in args.reference
        ]
        ref_sets, _ = load_reference_smiles_files(ref_specs)

    # Load input
    df = load_and_normalize(args.input)

    # Assign seed_idx if missing
    if "seed_idx" not in df.columns:
        unique_seeds = list(dict.fromkeys(df["smiles_seed"].tolist()))
        seed_to_idx = {s: i for i, s in enumerate(unique_seeds)}
        df["seed_idx"] = [seed_to_idx[s] for s in df["smiles_seed"]]

    # Enrich
    logger.info("\nEnriching DataFrame...")
    df = enrich_dataframe(
        df, prop_names, args.ranking_property, direction, ref_sets, args.recompute
    )

    # Select
    logger.info("\nSelecting molecules...")
    selected_df = select_from_dataframe(
        df,
        ranking_property=args.ranking_property,
        ranking_direction=direction,
        top_k=args.top_k,
        vin_hard_gate=not args.no_vin,
        vin_fallback=args.vin_fallback,
        mode=args.mode,
    )

    # Save
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.input).parent / "refiltered")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    enriched_path = Path(output_dir) / "all_generations_enriched.parquet"
    df.to_parquet(enriched_path)
    logger.info(f"Saved enriched: {enriched_path} ({len(df)} rows)")

    summary_path = Path(output_dir) / "summary.parquet"
    selected_df.to_parquet(summary_path)
    logger.info(f"Saved selected: {summary_path} ({len(selected_df)} rows)")

    # Print summary
    vin_col = f"is_vin_{args.ranking_property}"
    print(f"\n{'=' * 50}")
    print(f"REFILTER SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Input: {args.input} ({len(df)} molecules)")
    print(f"  Valid: {df['is_valid'].sum()}")
    if vin_col in df.columns:
        print(f"  V,I,N ({args.ranking_property}): {df[vin_col].sum()}")
    print(f"  Selected: {len(selected_df)} (top-{args.top_k} {args.mode})")
    if len(selected_df) > 0 and args.ranking_property in selected_df.columns:
        vals = selected_df[args.ranking_property].dropna()
        if len(vals) > 0:
            print(f"  {args.ranking_property}: mean={vals.mean():.3f}, "
                  f"best={vals.max():.3f}" if direction == "increasing"
                  else f"  {args.ranking_property}: mean={vals.mean():.3f}, "
                  f"best={vals.min():.3f}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
