"""Evaluate raw PairIT candidates with the paper's RF-select/MiniMol-score protocol.

Example:
    python scripts/evaluate_generations.py --dataset bace \
        --pairit outputs/bace/pairit/all_generations.parquet \
        --pairit-beam outputs/bace/beam/all_generations.parquet \
        --output-dir results/bace
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['bace', 'bbbp', 'hiv'])
    for method in ('pairit', 'pairit-beam', 'pairit-rl'):
        parser.add_argument(f'--{method}', type=Path, help='Raw candidates: Parquet or callback CSV')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    inputs = {method: getattr(args, method.replace('-', '_'))
              for method in ('pairit', 'pairit-beam', 'pairit-rl')}
    inputs = {method: path for method, path in inputs.items() if path is not None}
    if not inputs:
        parser.error('provide at least one of --pairit, --pairit-beam, --pairit-rl')
    for path in inputs.values():
        if not path.is_file():
            parser.error(f'input does not exist: {path}')
        if path.suffix not in ('.csv', '.parquet'):
            parser.error(f'expected .csv or .parquet: {path}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; use a new directory for each evaluation')

    from pairit._assets import ASSETS
    from pairit.eval.cli_utils import build_eval_config
    from pairit.eval.comparison_matrix import (
        SourceSpec, materialize_inputs, evaluate_final_inputs, build_summary_tables,
    )
    config = build_eval_config(
        moleculenet_train_dir=ASSETS/'data/moleculenet',
        moleculenet_test_dir=ASSETS/'data/moleculenet',
        score_oracle='minimol',
    )
    specs = [SourceSpec(
        base_method=method, display_base=method, family=method, source_group='pairit',
        dataset=args.dataset, train_regime='real-only', novelty_reference='real',
        source_kind='pairit_allgens_parquet' if path.suffix=='.parquet' else 'canonical_csv',
        path=path.resolve(), source_n_generations=64,
    ) for method, path in inputs.items()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'dataset': args.dataset,
        'protocol': 'RF selection, within-seed canonical deduplication, top 5 of first 64; MiniMol scoring; full test-seed budget',
        'inputs': {method: {'path': str(path.resolve()),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for method, path in inputs.items()},
    }
    (args.output_dir/'inputs.json').write_text(json.dumps(manifest, indent=2)+'\n')
    final, metadata, selection = materialize_inputs(
        specs, output_root=args.output_dir, config=config, verbose=True,
    )
    selection.to_csv(args.output_dir/'selection_summary.csv', index=False)
    outputs = evaluate_final_inputs(
        final, specs, output_root=args.output_dir, config=config,
        method_meta=metadata, verbose=True,
    )
    summary = build_summary_tables(outputs, args.output_dir)['regular']
    print(summary[summary.selection_mode=='best64'].to_string(index=False))


if __name__ == '__main__':
    main()
