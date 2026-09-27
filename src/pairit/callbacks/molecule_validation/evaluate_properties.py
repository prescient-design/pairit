#!/usr/bin/env python3
"""
Property evaluation script for MoleculeNet datasets (BACE, BBBP, HIV).
Uses pretrained Random Forest classifiers to score molecules.

Usage:
    python evaluate_properties.py --dataset bbbp --eval-accuracy
    python evaluate_properties.py --dataset bbbp
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from joblib import load

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


# Task metadata
TASK_INFO = {
    'bace_b': {
        'type': 'classification',
        'description': 'BACE-1 inhibitor classification',
    },
    'bbbp_b': {
        'type': 'classification',
        'description': 'Blood-brain barrier penetration',
    },
    'hiv_b': {
        'type': 'classification',
        'description': 'HIV replication inhibition',
    },
    # --- TDC tasks ---
    # Task names must match the shipped joblib basenames in the evaluator directory.
    'caco2_wang': {
        'type': 'regression',
        'description': 'Caco-2 apparent permeability, log10 cm/s',
    },
    'freesolv': {
        'type': 'regression',
        'description': 'Hydration free energy, kcal/mol',
    },
    'herg_b': {
        'type': 'classification',
        'description': 'hERG channel blockade (1 = blocker / cardiotoxic)',
    },
    # --- CYP substrate tasks (multi-property; one molecule table, three endpoints) ---
    # Names match the joblib basenames written by scripts/train_cyp_oracles.py.
    'cyp2c9_substrate': {
        'type': 'classification',
        'description': 'CYP2C9 substrate (1 = metabolised by CYP2C9)',
    },
    'cyp2d6_substrate': {
        'type': 'classification',
        'description': 'CYP2D6 substrate (1 = metabolised by CYP2D6)',
    },
    'cyp3a4_substrate': {
        'type': 'classification',
        'description': 'CYP3A4 substrate (1 = metabolised by CYP3A4)',
    },
}


class PropertyEvaluator:
    """
    Evaluator for molecular properties using pretrained Random Forest models.
    Uses ECFP4 (Morgan) fingerprints for molecule representation.
    """
    
    def __init__(self, model_path: str, task_name: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the .joblib model file
            task_name: Task identifier (e.g., 'bace_b', 'bbbp_b', 'hiv_b')
        """
        self.task_name = task_name
        self.model_path = model_path
        
        if task_name not in TASK_INFO:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_INFO.keys())}")
        
        self.task_type = TASK_INFO[task_name]['type']
        
        # Load the pretrained model
        try:
            self.model = load(model_path)
            print(f"✓ Loaded {task_name} evaluator from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluator from {model_path}: {e}")
    
    @staticmethod
    def smiles_to_ecfp4(smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES to ECFP4 fingerprint (Morgan radius=2, 2048 bits).
        
        Args:
            smiles: SMILES string
            
        Returns:
            Numpy array of shape (1, 2048) or None if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate ECFP4 fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        
        # Convert to numpy array
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1, -1)
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Predict properties for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Numpy array of predictions (probabilities for classification)
        """
        fps = []
        valid_mask = []
        
        for smiles in smiles_list:
            fp = self.smiles_to_ecfp4(smiles)
            if fp is not None:
                fps.append(fp)
                valid_mask.append(1)
            else:
                fps.append(np.zeros((1, 2048)))
                valid_mask.append(0)
        
        fps = np.concatenate(fps, axis=0)
        
        # Predict
        if self.task_type == 'classification':
            # Return probability of positive class
            scores = self.model.predict_proba(fps)[:, 1]
        else:
            scores = self.model.predict(fps)
        
        # Mask invalid molecules with 0
        scores = scores * np.array(valid_mask)
        
        return scores.astype(np.float32)
    
    def predict_with_labels(self, smiles_list: List[str], threshold: float = 0.5) -> tuple:
        """
        Predict properties and convert to binary labels (for classification).
        
        Args:
            smiles_list: List of SMILES strings
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Tuple of (probabilities, labels, valid_mask)
        """
        probs = self.predict(smiles_list)
        labels = (probs >= threshold).astype(int)
        valid_mask = np.array([1 if Chem.MolFromSmiles(s) is not None else 0 for s in smiles_list])

        return probs, labels, valid_mask


class LazyPropertyEvaluator:
    """Defer loading a :class:`PropertyEvaluator` until its first prediction.

    ``PROPERTY_REGISTRY`` builds every evaluator at module import time, so
    registering a task whose ``.joblib`` does not exist yet would break *every*
    import of pairit. This wrapper holds only the path, loads on first use, and
    raises an actionable error if the model is still missing.

    Parameters
    ----------
    model_path
        Path to the ``.joblib`` model. Need not exist at construction time.
    task_name
        Task identifier; must be a key of :data:`TASK_INFO`.

    Raises
    ------
    FileNotFoundError
        On first :meth:`predict` call, if ``model_path`` does not exist.

    Examples
    --------
    >>> ev = LazyPropertyEvaluator("/path/to/herg_b.joblib", "herg_b")
    >>> ev.is_available()
    False

    Notes
    -----
    For regression tasks :meth:`PropertyEvaluator.predict` masks invalid SMILES to
    ``0.0``, which is a legitimate value on some scales (e.g. FreeSolv dG). Callers
    that care about the distinction should check validity separately.

    See Also
    --------
    PropertyEvaluator : the eagerly-loaded evaluator this wraps.
    """

    def __init__(self, model_path: str, task_name: str) -> None:
        self.model_path = model_path
        self.task_name = task_name
        self._evaluator: Optional[PropertyEvaluator] = None

    def is_available(self) -> bool:
        """Return whether the backing ``.joblib`` file exists on disk."""
        return Path(self.model_path).exists()

    @property
    def evaluator(self) -> PropertyEvaluator:
        """The underlying evaluator, loading it on first access."""
        if self._evaluator is None:
            if not self.is_available():
                raise FileNotFoundError(
                    f"Oracle for task '{self.task_name}' is not trained yet: no file at "
                    f"{self.model_path}. Train it and save the sklearn model with joblib, "
                    f"matching the interface of the MoleculeNet evaluators "
                    f"(ECFP4 radius=2, nBits=2048 input)."
                )
            self._evaluator = PropertyEvaluator(self.model_path, self.task_name)
        return self._evaluator

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Predict for a list of SMILES, loading the model if needed."""
        return self.evaluator.predict(smiles_list)

    def predict_with_labels(self, smiles_list: List[str], threshold: float = 0.5) -> tuple:
        """Predict probabilities and binary labels, loading the model if needed."""
        return self.evaluator.predict_with_labels(smiles_list, threshold=threshold)


def evaluate_from_file(
    dataset: str,
    input_file: Path,
    evaluator_dir: Path,
    output_file: Optional[Path] = None,
    smiles_column: str = 'smiles',
):
    """
    Evaluate properties for molecules from a file.

    Rows are branched on the ``molecule_type`` column ('real' vs 'generated' /
    legacy 'synthetic'). Reference rows keep their ground-truth labels; model
    generations get oracle predictions filled in.

    Args:
        dataset: Dataset name ('bace', 'bbbp', or 'hiv')
        input_file: Path to input file (CSV, TSV, or Parquet)
        evaluator_dir: Directory containing evaluator .joblib files
        output_file: Optional output file path (defaults to input_file for in-place update)
        smiles_column: Column name containing SMILES
    """
    # Standardize dataset name
    task_name = f"{dataset.lower()}_b"
    
    if task_name not in TASK_INFO:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: bace, bbbp, hiv")
    
    # Load evaluator
    model_path = evaluator_dir / f"{task_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Evaluator not found: {model_path}")
    
    evaluator = PropertyEvaluator(str(model_path), task_name)
    
    # Load input file
    print(f"\nLoading molecules from {input_file}...")
    if input_file.suffix == '.parquet':
        df = pd.read_parquet(input_file)
    elif input_file.suffix in ['.csv', '.txt']:
        # Try comma, then tab, then space
        try:
            df = pd.read_csv(input_file)
        except:
            try:
                df = pd.read_csv(input_file, sep='\t')
            except:
                df = pd.read_csv(input_file, sep=' ', header=None, names=[smiles_column])
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {df.columns.tolist()}")
    
    print(f"Loaded {len(df)} molecules")
    
    # Predict on ALL molecules to get probabilities
    # For synthetic: add prob + pred
    # For real: add prob only (keep ground truth intact)
    print(f"\nEvaluating {task_name} property on all {len(df)} molecules...")
    smiles_list = df[smiles_column].tolist()
    probs, labels, valid_mask = evaluator.predict_with_labels(smiles_list)
    
    # Initialize oracle columns with NaN for all rows
    df[f'{task_name}_oracle_prob'] = float('nan')
    df[f'{task_name}_oracle_pred'] = float('nan')
    
    # Fill in probabilities for ALL molecules
    df[f'{task_name}_oracle_prob'] = probs
    
    # Fill in predictions only for generated molecules (or all if no molecule_type column).
    # NOTE: the parquet schema still writes 'synthetic' as the molecule_type value for
    # model-generated rows (legacy naming; unrelated to the removed real+synthetic pair
    # augmentation). Keep the literal here so on-disk data continues to load.
    if 'molecule_type' in df.columns:
        generated_mask = df['molecule_type'] == 'synthetic'
        real_mask = df['molecule_type'] == 'real'
        n_generated = generated_mask.sum()
        n_real = real_mask.sum()

        print(f"  - {n_real} real molecules (adding oracle_prob only, keeping ground truth)")
        print(f"  - {n_generated} generated molecules (adding oracle_prob + oracle_pred)")

        # Only fill predictions for generated molecules
        df.loc[generated_mask, f'{task_name}_oracle_pred'] = labels[generated_mask]
        # Real molecules: oracle_pred stays NaN (ground truth is in original column)
    else:
        # No molecule_type column: treat all as generated
        print(f"  No 'molecule_type' column found, adding predictions for all molecules")
        df[f'{task_name}_oracle_pred'] = labels
    
    # Print statistics
    n_valid = valid_mask.sum()
    n_invalid = len(valid_mask) - n_valid
    n_positive = (labels[valid_mask == 1] == 1).sum()
    n_negative = (labels[valid_mask == 1] == 0).sum()
    
    print(f"\n{'='*60}")
    print(f"RESULTS for {dataset.upper()} ({TASK_INFO[task_name]['description']})")
    print(f"{'='*60}")
    print(f"Evaluated molecules:  {len(smiles_list)}")
    print(f"Valid molecules:      {n_valid} ({n_valid/len(smiles_list)*100:.1f}%)")
    print(f"Invalid molecules:    {n_invalid}")
    print(f"\nPredictions (valid molecules only):")
    print(f"  Positive (≥0.5):    {n_positive} ({n_positive/n_valid*100:.1f}%)")
    print(f"  Negative (<0.5):    {n_negative} ({n_negative/n_valid*100:.1f}%)")
    print(f"\nProbability statistics (valid molecules):")
    valid_probs = probs[valid_mask == 1]
    print(f"  Mean:               {valid_probs.mean():.3f}")
    print(f"  Std:                {valid_probs.std():.3f}")
    print(f"  Min:                {valid_probs.min():.3f}")
    print(f"  Max:                {valid_probs.max():.3f}")
    print(f"{'='*60}\n")
    
    # Save output (default to in-place update if no output specified)
    if output_file is None:
        output_file = input_file
        print(f"Updating original file in-place: {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.suffix == '.parquet':
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"✓ Results saved to {output_file}")
    
    return df


def evaluate_oracle_accuracy(
    dataset: str,
    input_file: Path,
    evaluator_dir: Path,
    smiles_column: str = 'smiles',
):
    """
    Evaluate oracle accuracy on real molecules with ground truth.
    Does NOT modify the parquet file - only prints metrics.
    
    Args:
        dataset: Dataset name ('bace', 'bbbp', or 'hiv')
        input_file: Path to input file (CSV, TSV, or Parquet)
        evaluator_dir: Directory containing evaluator .joblib files
        smiles_column: Column name containing SMILES
    """
    # Standardize dataset name
    task_name = f"{dataset.lower()}_b"
    
    if task_name not in TASK_INFO:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: bace, bbbp, hiv")
    
    # Map task to ground truth column
    ground_truth_columns = {
        'bace_b': 'Class',
        'bbbp_b': 'p_np',
        'hiv_b': 'HIV_active',
    }
    gt_col = ground_truth_columns[task_name]
    
    # Load evaluator
    model_path = evaluator_dir / f"{task_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Evaluator not found: {model_path}")
    
    evaluator = PropertyEvaluator(str(model_path), task_name)
    
    # Load input file
    print(f"\n{'='*70}")
    print(f"ORACLE ACCURACY EVALUATION: {dataset.upper()}")
    print(f"{'='*70}")
    print(f"\nLoading molecules from {input_file}...")
    
    if input_file.suffix == '.parquet':
        df = pd.read_parquet(input_file)
    elif input_file.suffix in ['.csv', '.txt']:
        try:
            df = pd.read_csv(input_file)
        except:
            try:
                df = pd.read_csv(input_file, sep='\t')
            except:
                df = pd.read_csv(input_file, sep=' ', header=None, names=[smiles_column])
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {df.columns.tolist()}")
    
    if gt_col not in df.columns:
        raise ValueError(f"Ground truth column '{gt_col}' not found. Available: {df.columns.tolist()}")
    
    print(f"Total molecules in file: {len(df)}")
    
    # Filter to real molecules only
    if 'molecule_type' in df.columns:
        real_mask = df['molecule_type'] == 'real'
        df_real = df[real_mask]
        print(f"  - Real molecules: {len(df_real)}")
        print(f"  - Synthetic molecules: {(~real_mask).sum()} (skipping)")
    else:
        df_real = df
        print(f"  - No 'molecule_type' column found, assuming all molecules are real")
    
    if len(df_real) == 0:
        print("No real molecules found. Cannot evaluate accuracy.")
        return
    
    # Filter out molecules with missing ground truth
    valid_gt_mask = df_real[gt_col].notna() & (df_real[gt_col] != 'null')
    df_real = df_real[valid_gt_mask]
    print(f"  - With ground truth labels: {len(df_real)}")
    
    if len(df_real) == 0:
        print("No molecules with ground truth found. Cannot evaluate accuracy.")
        return
    
    # Get SMILES and ground truth
    smiles_list = df_real[smiles_column].tolist()
    ground_truth = df_real[gt_col].astype(int).values
    
    # Get split information if available
    has_split = 'split' in df_real.columns
    if has_split:
        splits = df_real['split'].values
        unique_splits = np.unique(splits)
        print(f"  - Splits found: {', '.join(unique_splits)}")
    
    # Predict
    print(f"\nRunning oracle predictions...")
    probs = evaluator.predict(smiles_list)
    preds = (probs >= 0.5).astype(int)
    
    # Overall accuracy
    valid_mask = np.array([1 if Chem.MolFromSmiles(s) is not None else 0 for s in smiles_list])
    n_valid = valid_mask.sum()
    n_invalid = len(valid_mask) - n_valid
    
    # Calculate metrics on valid molecules only
    valid_indices = valid_mask == 1
    valid_gt = ground_truth[valid_indices]
    valid_preds = preds[valid_indices]
    valid_probs = probs[valid_indices]
    
    accuracy = (valid_preds == valid_gt).sum() / len(valid_gt)
    
    # Per-class metrics
    n_positive_gt = (valid_gt == 1).sum()
    n_negative_gt = (valid_gt == 0).sum()
    
    # True positives, false positives, etc.
    tp = ((valid_preds == 1) & (valid_gt == 1)).sum()
    fp = ((valid_preds == 1) & (valid_gt == 0)).sum()
    tn = ((valid_preds == 0) & (valid_gt == 0)).sum()
    fn = ((valid_preds == 0) & (valid_gt == 1)).sum()
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC-AUC (if scikit-learn is available)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(valid_gt, valid_probs)
    except:
        auc = None
    
    # Print results
    print(f"\n{'='*70}")
    print(f"ORACLE PERFORMANCE: {dataset.upper()} ({TASK_INFO[task_name]['description']})")
    print(f"{'='*70}")
    print(f"\nDataset Statistics:")
    print(f"  Total evaluated:       {len(smiles_list)}")
    print(f"  Valid SMILES:          {n_valid} ({n_valid/len(smiles_list)*100:.1f}%)")
    print(f"  Invalid SMILES:        {n_invalid}")
    
    print(f"\nGround Truth Distribution:")
    print(f"  Positive (class 1):    {n_positive_gt} ({n_positive_gt/len(valid_gt)*100:.1f}%)")
    print(f"  Negative (class 0):    {n_negative_gt} ({n_negative_gt/len(valid_gt)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"ACCURACY METRICS (threshold = 0.5):")
    print(f"{'='*70}")
    print(f"  Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:             {precision:.4f}")
    print(f"  Recall (Sensitivity):  {recall:.4f}")
    print(f"  F1 Score:              {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC:               {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Neg    Pos")
    print(f"  Actual   Neg    {tn:6d} {fp:6d}")
    print(f"           Pos    {fn:6d} {tp:6d}")
    
    print(f"\nProbability Statistics:")
    print(f"  Mean:                  {valid_probs.mean():.4f}")
    print(f"  Std:                   {valid_probs.std():.4f}")
    print(f"  Min:                   {valid_probs.min():.4f}")
    print(f"  Max:                   {valid_probs.max():.4f}")
    
    # Per-split accuracy if available
    if has_split:
        print(f"\n{'='*70}")
        print(f"PER-SPLIT ACCURACY:")
        print(f"{'='*70}")
        for split_name in unique_splits:
            split_mask = (splits == split_name) & valid_indices
            if split_mask.sum() > 0:
                split_gt = ground_truth[split_mask]
                split_preds = preds[split_mask]
                split_acc = (split_preds == split_gt).sum() / len(split_gt)
                n_pos = (split_gt == 1).sum()
                n_neg = (split_gt == 0).sum()
                print(f"  {split_name.upper():8s}  Accuracy: {split_acc:.4f} ({split_acc*100:.2f}%)  |  N={len(split_gt)} (Pos:{n_pos}, Neg:{n_neg})")
    
    print(f"{'='*70}\n")
    print("✓ Evaluation complete (no files were modified)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate molecular properties using pretrained classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate BACE from parquet file
  python evaluate_properties.py --dataset bace --input molecules_bace.parquet
  
  # Evaluate BBBP from CSV with custom column name
  python evaluate_properties.py --dataset bbbp --input data.csv --smiles-col SMILES
  
  # Save results to file
  python evaluate_properties.py --dataset hiv --input molecules.csv --output results.csv
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['bace', 'bbbp', 'hiv'],
        help='Dataset to evaluate (bace, bbbp, or hiv)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        help='Input file containing SMILES (default: auto-detect from dataset name in --data-dir)'
    )
    parser.add_argument(
        '--evaluator-dir',
        type=Path,
        default=Path(__file__).parent / 'evaluators',
        help='Directory containing evaluator .joblib files (default: ./evaluators/)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: update input file in-place)'
    )
    parser.add_argument(
        '--eval-accuracy',
        action='store_true',
        help='Evaluate oracle accuracy on real molecules with ground truth (does not modify files)'
    )
    from pairit._assets import ASSETS
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=ASSETS / 'data' / 'moleculenet',
        help='Directory containing per-dataset molecule parquet files (bace/, bbbp/, hiv/).'
    )
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='smiles',
        help='Column name containing SMILES (default: smiles)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect input file if not provided
    if args.input is None:
        args.input = args.data_dir / f"molecules_{args.dataset}.parquet"
        print(f"Auto-detected input file: {args.input}")
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Run accuracy evaluation mode (does not modify files)
    if args.eval_accuracy:
        evaluate_oracle_accuracy(
            dataset=args.dataset,
            input_file=args.input,
            evaluator_dir=args.evaluator_dir,
            smiles_column=args.smiles_col,
        )
    else:
        # Run normal evaluation (writes predictions to file)
        evaluate_from_file(
            dataset=args.dataset,
            input_file=args.input,
            evaluator_dir=args.evaluator_dir,
            output_file=args.output,
            smiles_column=args.smiles_col,
        )


if __name__ == '__main__':
    main()
