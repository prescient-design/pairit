#!/usr/bin/env python3
"""
Script to pre-compute and add molecular properties to parquet files.

This script reads a molecules.parquet file, computes properties that are in the
prompting config but missing from the parquet file, and saves them back to the file.
This avoids recomputing properties during training, especially with multiple workers.

Usage:
    python scripts/add_properties_to_parquet.py \
        --input molecules.parquet \
        --output molecules.parquet \
        --properties logp qed sa mw tpsa hbd hba rotatable_bonds frac_sp3_c Class \
        --smiles-column smiles \
        --batch-size 1000 \
        --overwrite-existing
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict

# Try to import parquet reading libraries (try polars first, then pandas, then pyarrow)
try:
    import polars as pl
    HAS_POLARS = True
    HAS_PANDAS = False
except ImportError:
    HAS_POLARS = False
    try:
        import pandas as pd
        HAS_PANDAS = True
    except ImportError:
        HAS_PANDAS = False
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            HAS_PYARROW = True
        except ImportError:
            HAS_PYARROW = False
            print("Error: Need at least one of: polars, pandas, or pyarrow")
            print("Install with: pip install polars  OR  pip install pandas pyarrow")
            sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc="", unit=""):
        return iterable

from rdkit import Chem

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairit.callbacks.molecule_validation.property_registry import (
    PROPERTY_REGISTRY,
    get_property_specs,
    compute_all_properties_for_smiles,
)


def compute_properties_batch(
    smiles_list: List[str],
    property_names: List[str],
    batch_size: int = 1000,
) -> Dict[str, List[Optional[float]]]:
    """
    Compute properties for a batch of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        property_names: List of property names to compute
        batch_size: Batch size for progress tracking (not used for computation)
    
    Returns:
        Dictionary mapping property names to lists of computed values
    """
    # Get property specs
    property_specs = get_property_specs(property_names)
    
    # Initialize results
    results = {name: [] for name in property_names}
    
    # Compute properties for each SMILES
    for smiles in tqdm(smiles_list, desc="Computing properties", unit="molecules"):
        # Handle None, NaN, or empty strings
        if not smiles or (isinstance(smiles, float) and str(smiles).lower() == "nan"):
            # Add None for all properties
            for name in property_names:
                results[name].append(None)
            continue
        
        # Ensure smiles is a string
        smiles = str(smiles).strip()
        if not smiles:
            for name in property_names:
                results[name].append(None)
            continue
        
        # Compute all properties for this SMILES
        props = compute_all_properties_for_smiles(smiles, property_specs)
        
        # Add to results
        for name in property_names:
            results[name].append(props.get(name, None))
    
    return results


def add_properties_to_parquet(
    input_path: Path,
    output_path: Path,
    property_names: List[str],
    smiles_column: str = "smiles",
    batch_size: int = 1000,
    overwrite_existing: bool = False,
    backup: bool = True,
):
    """
    Add computed properties to a parquet file.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (can be same as input)
        property_names: List of property names to compute and add
        smiles_column: Name of the column containing SMILES strings
        batch_size: Batch size for processing (for progress tracking)
        overwrite_existing: If True, recompute properties even if columns exist
        backup: If True, create backup of original file when overwriting
    """
    print(f"Reading parquet file: {input_path}")
    
    # Read parquet file using available library
    if HAS_POLARS:
        df = pl.read_parquet(input_path)
        df_columns = df.columns
        df_len = len(df)
        smiles_list = df[smiles_column].to_list()
    elif HAS_PANDAS:
        df = pd.read_parquet(input_path)
        df_columns = df.columns.tolist()
        df_len = len(df)
        smiles_list = df[smiles_column].tolist()
    else:  # HAS_PYARROW
        table = pq.read_table(input_path)
        df = table.to_pandas()  # Convert to pandas for easier manipulation
        df_columns = df.columns.tolist()
        df_len = len(df)
        smiles_list = df[smiles_column].tolist()
    
    if smiles_column not in df_columns:
        raise ValueError(
            f"Column '{smiles_column}' not found in parquet file. "
            f"Available columns: {df_columns}"
        )
    
    print(f"Loaded {df_len} molecules")
    
    # Check which properties are already present
    existing_props = [p for p in property_names if p in df_columns]
    missing_props = [p for p in property_names if p not in df_columns]
    
    # Check for properties with None values that need recomputation
    props_with_none = []
    props_to_skip = []
    
    if existing_props and not overwrite_existing:
        # Check each existing property for None values
        for prop in existing_props:
            if HAS_POLARS:
                none_count = df.filter(pl.col(prop).is_null()).height
            elif HAS_PANDAS:
                none_count = df[prop].isna().sum()
            else:  # pyarrow converted to pandas
                none_count = df[prop].isna().sum()
            
            if none_count > 0:
                props_with_none.append(prop)
                print(f"  {prop}: {none_count}/{df_len} values are None - will recompute")
            else:
                props_to_skip.append(prop)
                print(f"  {prop}: all values present - will skip")
    
    # Determine which properties to compute
    if overwrite_existing:
        # Compute all properties
        props_to_compute = property_names
        if existing_props:
            print(f"Properties already present: {existing_props}")
            print(f"Will recompute all (--overwrite-existing enabled)")
    else:
        # Compute missing properties + properties with None values
        props_to_compute = missing_props + props_with_none
        if props_to_skip:
            print(f"\nProperties with valid values (skipping): {props_to_skip}")
        if props_with_none:
            print(f"Properties with None values (will recompute): {props_with_none}")
        if missing_props:
            print(f"Missing properties (will compute): {missing_props}")
    
    if not props_to_compute:
        print("\nAll properties already present with valid values. Nothing to do.")
        return
    
    print(f"\nWill compute {len(props_to_compute)} properties: {props_to_compute}")
    
    # Compute properties
    print(f"\nComputing {len(props_to_compute)} properties for {len(smiles_list)} molecules...")
    results = compute_properties_batch(smiles_list, props_to_compute, batch_size)
    
    # Add computed properties to dataframe
    # For properties with None values, only update the None entries
    if HAS_POLARS:
        for prop_name in props_to_compute:
            if prop_name in props_with_none and not overwrite_existing:
                # Only update None values, keep existing valid values
                new_values = pl.Series(results[prop_name])
                df = df.with_columns(
                    pl.when(pl.col(prop_name).is_null())
                    .then(new_values)
                    .otherwise(pl.col(prop_name))
                    .alias(prop_name)
                )
            else:
                # Replace entire column (missing property or overwrite mode)
                df = df.with_columns(pl.Series(prop_name, results[prop_name]))
    else:  # pandas or pyarrow (converted to pandas)
        for prop_name in props_to_compute:
            if prop_name in props_with_none and not overwrite_existing:
                # Only update None values, keep existing valid values
                mask = df[prop_name].isna()
                new_values = pd.Series(results[prop_name], index=df.index)
                df.loc[mask, prop_name] = new_values.loc[mask]
            else:
                # Replace entire column (missing property or overwrite mode)
                df[prop_name] = results[prop_name]
    
    # Create backup if overwriting and backup is enabled
    if output_path == input_path and backup:
        backup_path = input_path.with_suffix(f".backup{input_path.suffix}")
        print(f"\nCreating backup: {backup_path}")
        if HAS_POLARS:
            pl.read_parquet(input_path).write_parquet(backup_path)
        elif HAS_PANDAS:
            pd.read_parquet(input_path).to_parquet(backup_path, index=False)
        else:  # pyarrow
            pq.read_table(input_path).to_pandas().to_parquet(backup_path, index=False)
    
    # Write output
    print(f"\nWriting output to: {output_path}")
    if HAS_POLARS:
        df.write_parquet(output_path)
    elif HAS_PANDAS:
        df.to_parquet(output_path, index=False)
    else:  # pyarrow
        pa.Table.from_pandas(df).to_pandas().to_parquet(output_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total molecules: {len(df)}")
    for prop_name in props_to_compute:
        non_null = sum(1 for v in results[prop_name] if v is not None)
        null = len(results[prop_name]) - non_null
        print(f"  {prop_name}: {non_null} computed, {null} failed/None")
    if props_to_skip:
        print(f"\n  Skipped (already have valid values): {', '.join(props_to_skip)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Add molecular properties to parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add all properties from config to molecules.parquet
  python scripts/add_properties_to_parquet.py \\
      --input data/molecules.parquet \\
      --properties logp qed sa mw tpsa hbd hba rotatable_bonds frac_sp3_c Class

  # Only compute missing properties (don't overwrite existing)
  python scripts/add_properties_to_parquet.py \\
      --input data/molecules.parquet \\
      --properties logp qed sa

  # Recompute all properties (overwrite existing)
  python scripts/add_properties_to_parquet.py \\
      --input data/molecules.parquet \\
      --properties logp qed sa \\
      --overwrite-existing
        """,
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output parquet file path (default: same as input, overwrites)",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        required=True,
        help="Property names to compute (e.g., logp qed sa mw tpsa hbd hba rotatable_bonds frac_sp3_c Class)",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Name of the column containing SMILES strings (default: smiles)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for progress tracking (default: 1000)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute properties even if columns already exist",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup when overwriting input file",
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path
    output_path = args.output if args.output else args.input
    
    # Validate property names
    available_props = list(PROPERTY_REGISTRY.keys())
    invalid_props = [p for p in args.properties if p not in available_props]
    if invalid_props:
        print(f"Error: Invalid property names: {invalid_props}")
        print(f"Available properties: {', '.join(available_props)}")
        sys.exit(1)
    
    # Run
    try:
        add_properties_to_parquet(
            input_path=args.input,
            output_path=output_path,
            property_names=args.properties,
            smiles_column=args.smiles_column,
            batch_size=args.batch_size,
            overwrite_existing=args.overwrite_existing,
            backup=not args.no_backup,
        )
        print("\n✅ Success!")
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

