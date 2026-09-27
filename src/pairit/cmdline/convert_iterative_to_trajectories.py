"""
Convert iterative generation output to trajectory format.

Takes the output parquet from iterative_generate.py and converts it into a
trajectory format where each row represents a complete path through the
iterative generation process.

Usage:
    python -m pairit.cmdline.convert_iterative_to_trajectories input.parquet output.parquet
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_trajectory_dataframe(
    df: pd.DataFrame,
    property_columns: list[str] | None = None,
    include_all_generations: bool = False,
) -> pd.DataFrame:
    """
    Convert iterative generation results to trajectory format.

    Each row in the output represents a complete trajectory from an initial seed
    through all rounds of generation. For k>1 (multiple selections per seed),
    trajectories branch, resulting in multiple rows per initial seed.

    Parameters
    ----------
    df : pd.DataFrame
        Output from iterative_generate.py with columns:
        - round: iteration round (0-indexed)
        - seed_idx: index of seed within that round
        - smiles_seed: the seed SMILES used for generation
        - smiles_generated: the generated SMILES
        - smiles_canonical: canonical form of generated SMILES
        - is_valid: validity flag
        - is_memorized: memorization flag (optional)
        - score: optimization property value
        - selected: whether selected for next round
        - plus any property columns (logp, qed, sa, etc.)

    property_columns : list[str] | None
        List of property column names to include. If None, auto-detects
        common properties (logp, qed, sa, mw, tpsa, etc.).

    include_all_generations : bool
        If True, include all generations (not just selected ones) as separate
        trajectories. If False (default), only trace through selected molecules.

    Returns
    -------
    pd.DataFrame
        Trajectory dataframe where each row is a complete trajectory with columns:
        - trajectory_id: unique identifier for the trajectory
        - initial_seed: the starting seed SMILES
        - num_rounds: number of rounds in this trajectory
        - final_smiles: the last generated SMILES in the trajectory
        - round_{N}_seed: seed SMILES for round N
        - round_{N}_generated: generated SMILES for round N
        - round_{N}_valid: validity for round N
        - round_{N}_memorized: memorization status for round N (if available)
        - round_{N}_score: score for round N
        - round_{N}_{prop}: property value for round N (for each property)
        - round_{N}_selected: whether selected for next round
    """
    if df.empty:
        return pd.DataFrame()

    # Auto-detect property columns if not specified
    if property_columns is None:
        known_props = {'logp', 'qed', 'sa', 'mw', 'tpsa', 'hbd', 'hba', 'rotatable_bonds',
                       'num_rings', 'num_aromatic_rings', 'frac_sp3_c', 'Class'}
        property_columns = [c for c in df.columns if c.lower() in {p.lower() for p in known_props}]

    # Get all rounds
    rounds = sorted(df['round'].unique())
    max_round = max(rounds)

    # Check if memorization data exists
    has_memorization = 'is_memorized' in df.columns

    # Build index of round data
    # For each round, map seed SMILES -> list of (generated, valid, memorized, score, selected, props)
    round_data: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        rnd = row['round']
        seed = row['smiles_seed']

        entry = {
            'generated': row.get('smiles_canonical') or row.get('smiles_generated', ''),
            'generated_raw': row.get('smiles_generated', ''),
            'valid': row.get('is_valid', False),
            'memorized': row.get('is_memorized') if has_memorization else None,
            'score': row.get('score'),
            'selected': row.get('selected', False),
            'distance': row.get('tanimoto_distance'),
        }

        # Add property values
        for prop in property_columns:
            if prop in row:
                entry[prop] = row[prop]

        round_data[rnd][seed].append(entry)

    # Get initial seeds (seeds in round 0)
    initial_seeds = list(round_data[0].keys())

    # Build trajectories by tracing through rounds
    trajectories = []
    trajectory_id = 0

    def trace_trajectory(current_seed: str, current_round: int, path: list[dict]):
        """Recursively trace a trajectory through rounds."""
        nonlocal trajectory_id

        if current_round > max_round:
            # End of trajectory - save it
            if path:
                traj = build_trajectory_record(trajectory_id, path, property_columns, has_memorization)
                trajectories.append(traj)
                trajectory_id += 1
            return

        # Get generations for this seed in this round
        generations = round_data[current_round].get(current_seed, [])

        if not generations:
            # No generations found - end trajectory here
            if path:
                traj = build_trajectory_record(trajectory_id, path, property_columns, has_memorization)
                trajectories.append(traj)
                trajectory_id += 1
            return

        # Find selected generations (or all if include_all_generations)
        if include_all_generations:
            next_steps = generations
        else:
            # Filter to selected and deduplicate by canonical SMILES
            # (same molecule generated multiple times should only create one trajectory)
            selected_gens = [g for g in generations if g['selected']]
            seen_smiles = set()
            next_steps = []
            for g in selected_gens:
                smiles = g['generated']
                if smiles and smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    next_steps.append(g)
            if not next_steps:
                # No selected generations - end trajectory here
                # But first, add the best generation from this round to show where it ended
                best_gen = max(generations, key=lambda x: x['score'] if x['score'] is not None else float('-inf'))
                new_path = path + [{'seed': current_seed, 'round': current_round, **best_gen}]
                traj = build_trajectory_record(trajectory_id, new_path, property_columns, has_memorization)
                trajectories.append(traj)
                trajectory_id += 1
                return

        # Continue trajectory for each selected generation
        for gen in next_steps:
            new_path = path + [{'seed': current_seed, 'round': current_round, **gen}]
            next_seed = gen['generated']

            if next_seed:
                trace_trajectory(next_seed, current_round + 1, new_path)
            else:
                # Invalid/empty generation - end trajectory
                traj = build_trajectory_record(trajectory_id, new_path, property_columns, has_memorization)
                trajectories.append(traj)
                trajectory_id += 1

    # Trace from each initial seed
    for seed in initial_seeds:
        trace_trajectory(seed, 0, [])

    if not trajectories:
        return pd.DataFrame()

    return pd.DataFrame(trajectories)


def build_trajectory_record(
    trajectory_id: int,
    path: list[dict],
    property_columns: list[str],
    has_memorization: bool,
) -> dict:
    """Build a single trajectory record from a path."""
    record = {
        'trajectory_id': trajectory_id,
        'initial_seed': path[0]['seed'] if path else None,
        'num_rounds': len(path),
    }

    # Add per-round data
    for step in path:
        rnd = step['round']
        prefix = f'round_{rnd}_'

        record[f'{prefix}seed'] = step['seed']
        record[f'{prefix}generated'] = step['generated']
        record[f'{prefix}valid'] = step['valid']
        record[f'{prefix}score'] = step['score']
        record[f'{prefix}selected'] = step['selected']
        record[f'{prefix}distance'] = step.get('distance')

        if has_memorization:
            record[f'{prefix}memorized'] = step['memorized']

        for prop in property_columns:
            if prop in step:
                record[f'{prefix}{prop}'] = step[prop]

    # Add final state
    if path:
        final_step = path[-1]
        record['final_smiles'] = final_step['generated']
        record['final_score'] = final_step['score']
        record['final_valid'] = final_step['valid']
        if has_memorization:
            record['final_memorized'] = final_step['memorized']

    return record


def compute_trajectory_statistics(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trajectory-level statistics to the trajectory dataframe.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Trajectory dataframe from build_trajectory_dataframe().

    Returns
    -------
    pd.DataFrame
        Trajectory dataframe with additional columns:
        - score_improvement: final_score - initial_score
        - score_trajectory: list of scores through rounds
        - all_valid: whether all generations in trajectory were valid
        - any_memorized: whether any generation was memorized
        - num_memorized: count of memorized generations
    """
    if traj_df.empty:
        return traj_df

    df = traj_df.copy()

    # Find score columns
    score_cols = [c for c in df.columns if c.endswith('_score') and c.startswith('round_')]
    score_cols = sorted(score_cols, key=lambda x: int(x.split('_')[1]))

    # Compute score improvement
    if score_cols and 'final_score' in df.columns:
        initial_scores = df[score_cols[0]] if score_cols else None
        if initial_scores is not None:
            df['score_improvement'] = df['final_score'] - initial_scores

    # Check if all valid
    valid_cols = [c for c in df.columns if c.endswith('_valid') and c.startswith('round_')]
    if valid_cols:
        df['all_valid'] = df[valid_cols].all(axis=1)

    # Check memorization
    memorized_cols = [c for c in df.columns if c.endswith('_memorized') and c.startswith('round_')]
    if memorized_cols:
        df['any_memorized'] = df[memorized_cols].any(axis=1)
        df['num_memorized'] = df[memorized_cols].sum(axis=1)
        df['all_novel'] = ~df['any_memorized']

    return df


def convert_to_trajectories(
    input_path: str,
    output_path: str | None = None,
    property_columns: list[str] | None = None,
    include_all_generations: bool = False,
    add_statistics: bool = True,
) -> pd.DataFrame:
    """
    Convert iterative generation parquet to trajectory format.

    Parameters
    ----------
    input_path : str
        Path to input parquet from iterative_generate.py.
    output_path : str | None
        Path to save output parquet. If None, doesn't save.
    property_columns : list[str] | None
        Property columns to include. If None, auto-detects.
    include_all_generations : bool
        If True, include all generations as separate trajectories.
    add_statistics : bool
        If True, add trajectory-level statistics.

    Returns
    -------
    pd.DataFrame
        Trajectory dataframe.
    """
    logger.info(f"Loading input from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows, {df['round'].nunique()} rounds")

    logger.info("Building trajectories...")
    traj_df = build_trajectory_dataframe(
        df,
        property_columns=property_columns,
        include_all_generations=include_all_generations,
    )
    logger.info(f"Built {len(traj_df)} trajectories")

    if add_statistics and not traj_df.empty:
        logger.info("Computing trajectory statistics...")
        traj_df = compute_trajectory_statistics(traj_df)

    if output_path:
        logger.info(f"Saving to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_parquet(output_path)

    return traj_df


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert iterative generation output to trajectory format."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input parquet from iterative_generate.py",
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to save output parquet (default: input_trajectories.parquet)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all generations, not just selected ones",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't add trajectory-level statistics",
    )
    parser.add_argument(
        "--properties",
        type=str,
        nargs="+",
        default=None,
        help="Property columns to include (default: auto-detect)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Default output path
    if args.output_path is None:
        input_p = Path(args.input_path)
        args.output_path = str(input_p.parent / f"{input_p.stem}_trajectories.parquet")

    traj_df = convert_to_trajectories(
        args.input_path,
        args.output_path,
        property_columns=args.properties,
        include_all_generations=args.include_all,
        add_statistics=not args.no_stats,
    )

    # Print summary
    print(f"\nTrajectory Summary:")
    print(f"  Total trajectories: {len(traj_df)}")
    if 'num_rounds' in traj_df.columns:
        print(f"  Rounds per trajectory: {traj_df['num_rounds'].min()}-{traj_df['num_rounds'].max()}")
    if 'final_valid' in traj_df.columns:
        print(f"  Final valid: {traj_df['final_valid'].sum()}/{len(traj_df)}")
    if 'all_valid' in traj_df.columns:
        print(f"  All-valid trajectories: {traj_df['all_valid'].sum()}/{len(traj_df)}")
    if 'score_improvement' in traj_df.columns:
        improvements = traj_df['score_improvement'].dropna()
        if len(improvements) > 0:
            print(f"  Score improvement: mean={improvements.mean():.3f}, max={improvements.max():.3f}")
    if 'all_novel' in traj_df.columns:
        print(f"  All-novel trajectories: {traj_df['all_novel'].sum()}/{len(traj_df)}")

    print(f"\nSaved to: {args.output_path}")


if __name__ == "__main__":
    main()
