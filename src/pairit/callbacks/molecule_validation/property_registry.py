"""
Property Registry for Molecular Validation
===========================================

Provides predefined molecular properties and utilities for adding custom ones.
Makes it easy to configure which properties to compute during validation.
"""

from typing import Dict, Callable, Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import Crippen, QED, Descriptors
from pairit._assets import ASSETS
from pairit.eval import PropertySpec
from pairit.callbacks import sascorer
from pairit.callbacks.molecule_validation.evaluate_properties import (
    LazyPropertyEvaluator,
    PropertyEvaluator,
)
import os
import pandas as pd


def _safe_compute(fn: Callable, mol) -> Optional[float]:
    """Safely compute a property, returning None on failure."""
    try:
        return float(fn(mol))
    except Exception:
        return None

# MoleculeNet activity oracles.
#
# These MUST be the train-split-only evaluators. The previous directory,
# ``.../moleculenet/evaluators`` (byte-identical to
# ``pairit_eval/moleculenet_property_evaluators``), holds models fit on the FULL
# dataset: they score ROC-AUC 1.000 on the held-out test split because they had seen
# it. Confirmed 2026-07-30 by training an identically-configured RF, which reaches
# 0.92/0.91/0.80 on the train split and reproduces 1.000 exactly on all data.
#
# This registry is not only a metrics helper -- it feeds the GRPO reward
# (``rl/rewards.py``), beam-search pruning and best-of-N selection
# (``inference/``), and training-time validation. A leaked oracle here means models
# are OPTIMISED against a contaminated signal, which no amount of re-scoring can
# undo. See docs/oracle_comparison_moleculenet.md.
_MOLECULENET_EVALUATOR_DIR = str(ASSETS / "oracles" / "moleculenet" / "evaluators")

bace_evaluator_dir = _MOLECULENET_EVALUATOR_DIR
bace_evaluator = LazyPropertyEvaluator(os.path.join(bace_evaluator_dir, "bace_b.joblib"), "bace_b")

bbbp_evaluator_dir = _MOLECULENET_EVALUATOR_DIR
bbbp_evaluator = LazyPropertyEvaluator(os.path.join(bbbp_evaluator_dir, "bbbp_b.joblib"), "bbbp_b")

hiv_evaluator_dir = _MOLECULENET_EVALUATOR_DIR
hiv_evaluator = LazyPropertyEvaluator(os.path.join(hiv_evaluator_dir, "hiv_b.joblib"), "hiv_b")

def compute_Class(mol) -> Optional[float]:
    try:
        smiles = Chem.MolToSmiles(mol)
        return bace_evaluator.predict([smiles])[0]
    except Exception as e:
        return None

def compute_p_np(mol) -> Optional[float]:
    try:
        smiles = Chem.MolToSmiles(mol)
        return bbbp_evaluator.predict([smiles])[0]
    except Exception as e:
        return None

def compute_HIV_active(mol) -> Optional[float]:
    try:
        smiles = Chem.MolToSmiles(mol)
        return hiv_evaluator.predict([smiles])[0]
    except Exception as e:
        return None


# --- TDC oracles ---------------------------------------------------------------
# RF(random_state=0) on ECFP4 (Morgan r=2, 2048 bits), retrained 2026-07-28 to fit the
# TDC `train` split only (scaffold split, seed 42). Honest test error: caco2 RMSE 0.492,
# freesolv RMSE 2.370, herg ROC-AUC 0.826 -- see the shipped <task>_metrics.yaml.
#
# The filenames here MUST match the joblib basenames in the evaluator directory, and the
# task names must be keys of TASK_INFO. Getting either wrong fails SILENTLY: compute_*
# swallows the exception and returns None, so property metrics vanish rather than error.
# Loaded lazily so a missing/moved evaluator directory cannot break imports of pairit.
tdc_evaluator_dir = os.environ.get(
    "TDC_EVALUATOR_DIR",
    str(ASSETS / "oracles" / "tdc" / "evaluators"),
)
caco2_evaluator = LazyPropertyEvaluator(
    os.path.join(tdc_evaluator_dir, "caco2_wang.joblib"), "caco2_wang"
)
freesolv_evaluator = LazyPropertyEvaluator(
    os.path.join(tdc_evaluator_dir, "freesolv.joblib"), "freesolv"
)
herg_evaluator = LazyPropertyEvaluator(
    os.path.join(tdc_evaluator_dir, "herg_b.joblib"), "herg_b"
)


def compute_caco2_papp(mol) -> Optional[float]:
    """Predict Caco-2 apparent permeability (log10 cm/s) via the TDC oracle."""
    try:
        return float(caco2_evaluator.predict([Chem.MolToSmiles(mol)])[0])
    except Exception:
        return None


def compute_freesolv_dg(mol) -> Optional[float]:
    """Predict hydration free energy (kcal/mol) via the TDC oracle."""
    try:
        return float(freesolv_evaluator.predict([Chem.MolToSmiles(mol)])[0])
    except Exception:
        return None


def compute_herg_block(mol) -> Optional[float]:
    """Predict hERG blockade probability via the TDC oracle."""
    try:
        return float(herg_evaluator.predict([Chem.MolToSmiles(mol)])[0])
    except Exception:
        return None


# --- CYP substrate oracles (multi-property) -------------------------------------
# Three binary endpoints on ONE molecule table, so a single generation is scored
# against all three at once. Trained by scripts/train_cyp_oracles.py on the
# CYP_Substrate *train* split only (Bemis-Murcko scaffold split, seed 42).
#
# HONEST TEST ROC-AUC IS MODEST -- 0.587 / 0.771 / 0.645 for 2C9 / 2D6 / 3A4. CYP2C9
# is barely above chance, so oracle-derived improvement on that endpoint is close to
# noise. This is a property of the task (654 molecules, scaffold split), not a bug,
# but it means:
#   * do NOT report a single-oracle V,I,N for CYP2C9 without the MiniMol second
#     oracle alongside it (scripts/train_cyp_minimol_oracles.py) -- see
#     docs/cyp_substrate_dataset.md and the hERG dual-oracle precedent; and
#   * prefer per-isoform reporting over the aggregate, which inherits 2C9's noise.
# The measured labels in the parquets are reliable; only *predictions on generated
# molecules* carry this error.
cyp_evaluator_dir = os.environ.get(
    "CYP_EVALUATOR_DIR",
    str(ASSETS / "oracles" / "cyp" / "evaluators"),
)
cyp2c9_evaluator = LazyPropertyEvaluator(
    os.path.join(cyp_evaluator_dir, "cyp2c9_substrate.joblib"), "cyp2c9_substrate"
)
cyp2d6_evaluator = LazyPropertyEvaluator(
    os.path.join(cyp_evaluator_dir, "cyp2d6_substrate.joblib"), "cyp2d6_substrate"
)
cyp3a4_evaluator = LazyPropertyEvaluator(
    os.path.join(cyp_evaluator_dir, "cyp3a4_substrate.joblib"), "cyp3a4_substrate"
)


def compute_cyp2c9_substrate(mol) -> Optional[float]:
    """Predict P(molecule is a CYP2C9 substrate). Weak oracle -- see the note above."""
    try:
        return float(cyp2c9_evaluator.predict([Chem.MolToSmiles(mol)])[0])
    except Exception:
        return None


def compute_cyp2d6_substrate(mol) -> Optional[float]:
    """Predict P(molecule is a CYP2D6 substrate)."""
    try:
        return float(cyp2d6_evaluator.predict([Chem.MolToSmiles(mol)])[0])
    except Exception:
        return None


def compute_cyp3a4_substrate(mol) -> Optional[float]:
    """Predict P(molecule is a CYP3A4 substrate)."""
    try:
        return float(cyp3a4_evaluator.predict([Chem.MolToSmiles(mol)])[0])
    except Exception:
        return None


def compute_cyp_n_substrate(mol) -> Optional[float]:
    """Expected number of the three CYPs that metabolise the molecule, in [0, 3].

    The continuous relaxation of the measured ``cyp_n_substrate`` count: the sum of
    the three predicted probabilities rather than a separately fitted regressor, so
    it stays consistent with the per-isoform oracles by construction.

    Returns ``None`` if any of the three oracles fails, since a partial sum is not
    comparable with a full one and silently mixing the two would corrupt any ranking
    built on this value.
    """
    parts = [
        compute_cyp2c9_substrate(mol),
        compute_cyp2d6_substrate(mol),
        compute_cyp3a4_substrate(mol),
    ]
    if any(p is None for p in parts):
        return None
    return float(sum(parts))

# Property computation functions (take RDKit mol object)
def compute_logp(mol) -> Optional[float]:
    """Compute LogP (lipophilicity)."""
    return _safe_compute(Crippen.MolLogP, mol)


def compute_qed(mol) -> Optional[float]:
    """Compute QED (Quantitative Estimate of Drug-likeness)."""
    return _safe_compute(QED.qed, mol)


def compute_sa(mol) -> Optional[float]:
    """Compute SA score (Synthetic Accessibility)."""
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def compute_mw(mol) -> Optional[float]:
    """Compute molecular weight."""
    return _safe_compute(Descriptors.MolWt, mol)


def compute_tpsa(mol) -> Optional[float]:
    """Compute topological polar surface area."""
    return _safe_compute(Descriptors.TPSA, mol)


def compute_hbd(mol) -> Optional[float]:
    """Compute number of hydrogen bond donors."""
    return _safe_compute(Descriptors.NumHDonors, mol)


def compute_hba(mol) -> Optional[float]:
    """Compute number of hydrogen bond acceptors."""
    return _safe_compute(Descriptors.NumHAcceptors, mol)


def compute_rotatable_bonds(mol) -> Optional[float]:
    """Compute number of rotatable bonds."""
    return _safe_compute(Descriptors.NumRotatableBonds, mol)


def compute_frac_sp3_c(mol) -> Optional[float]:
    """Compute fraction of sp3-hybridized carbons (RDKit FractionCSP3)."""
    return _safe_compute(Descriptors.FractionCSP3, mol)


# Property registry: name -> (function, direction, description)
# Direction is "increasing" if higher values are better, "decreasing" if lower is better
PROPERTY_REGISTRY: Dict[str, tuple] = {
    "logp": (
        compute_logp, 
        "increasing", 
        "LogP (lipophilicity) - typically want moderate values (0-5) for drug-likeness"
    ),
    "qed": (
        compute_qed, 
        "increasing", 
        "QED (drug-likeness) - higher is better, range [0,1]"
    ),
    "sa": (
        compute_sa, 
        "decreasing", 
        "SA score (synthetic accessibility) - lower is easier to synthesize, range [1,10]"
    ),
    "mw": (
        compute_mw, 
        "increasing", 
        "Molecular weight - depends on application, typical drug range 150-500 Da"
    ),
    "tpsa": (
        compute_tpsa, 
        "decreasing", 
        "Topological polar surface area - lower values improve membrane permeability"
    ),
    "hbd": (
        compute_hbd, 
        "decreasing", 
        "Hydrogen bond donors - lower values improve oral bioavailability"
    ),
    "hba": (
        compute_hba, 
        "decreasing", 
        "Hydrogen bond acceptors - lower values improve oral bioavailability"
    ),
    "rotatable_bonds": (
        compute_rotatable_bonds, 
        "decreasing", 
        "Number of rotatable bonds - lower values improve oral bioavailability"
    ),
    "frac_sp3_c": (
        compute_frac_sp3_c,
        "increasing",
        "Fraction of sp3 carbons (RDKit FractionCSP3) - higher values can indicate 3D character"
    ),
    "Class": (
        compute_Class,
        "increasing",
        "BACE-B classification prediction"
    ),
    "p_np": (
        compute_p_np,
        "increasing",
        "BBBP classification prediction"
    ),
    "HIV_active": (
        compute_HIV_active,
        "increasing",
        "HIV active classification prediction"
    ),
    "caco2_papp": (
        compute_caco2_papp,
        "increasing",
        "TDC Caco2_Wang permeability, log10 cm/s - higher is more permeable"
    ),
    "freesolv_dg": (
        compute_freesolv_dg,
        "decreasing",
        "TDC FreeSolv hydration free energy, kcal/mol - more negative is better hydrated"
    ),
    "herg_block": (
        compute_herg_block,
        "decreasing",
        "TDC hERG blockade probability - lower is safer / less cardiotoxic"
    ),
    # --- CYP substrate (multi-property) ---
    # `decreasing` encodes "fewer metabolising CYPs is better". That is a DEFENSIBLE
    # DEFAULT, NOT A SETTLED FACT: metabolic redundancy is often desirable, since a
    # drug cleared by several CYPs is buffered against one pathway being inhibited or
    # genetically absent (CYP2D6 poor metabolisers are ~7% of Europeans). Same status
    # as FreeSolv's direction. For the multi-property prompted setup the direction
    # here is largely inert anyway -- the per-property prompt token supplies the
    # direction per example, and pairs are not pre-oriented.
    "cyp2c9_substrate": (
        compute_cyp2c9_substrate,
        "decreasing",
        "P(CYP2C9 substrate) - WEAK ORACLE (test ROC-AUC 0.587); do not report alone"
    ),
    "cyp2d6_substrate": (
        compute_cyp2d6_substrate,
        "decreasing",
        "P(CYP2D6 substrate) - test ROC-AUC 0.771"
    ),
    "cyp3a4_substrate": (
        compute_cyp3a4_substrate,
        "decreasing",
        "P(CYP3A4 substrate) - test ROC-AUC 0.645"
    ),
    "cyp_n_substrate": (
        compute_cyp_n_substrate,
        "decreasing",
        "Expected number of CYP2C9/2D6/3A4 metabolising the molecule, [0,3] - "
        "inherits CYP2C9's oracle noise; prefer per-isoform reporting"
    ),
}


# Default property set for backward compatibility
DEFAULT_PROPERTIES = ["logp", "qed", "sa"]


def get_property_specs(property_names: List[str]) -> List[PropertySpec]:
    """
    Get PropertySpec objects for given property names.
    
    Args:
        property_names: List of property names (e.g., ["logp", "qed", "sa"])
    
    Returns:
        List of PropertySpec objects
    
    Raises:
        ValueError: If any property name is not in registry
        
    Example:
        >>> specs = get_property_specs(["logp", "qed"])
        >>> # Use specs for validation callback
    """
    specs = []
    for name in property_names:
        if name not in PROPERTY_REGISTRY:
            available = ", ".join(sorted(PROPERTY_REGISTRY.keys()))
            raise ValueError(
                f"Unknown property '{name}'. Available properties: {available}"
            )
        
        fn, direction, description = PROPERTY_REGISTRY[name]
        # Normalize direction to internal format
        internal_direction = "higher" if direction in ["increasing", "higher", "maximize"] else "lower"
        specs.append(PropertySpec(name, fn, direction=internal_direction, min_delta=0.0))
    
    return specs


def register_custom_property(
    name: str, 
    compute_fn: Callable, 
    direction: str, 
    description: str
):
    """
    Register a custom property for use in validation.
    
    Args:
        name: Property name (must be unique)
        compute_fn: Function that takes RDKit mol and returns float
        direction: "increasing" or "decreasing"
        description: Human-readable description
    
    Raises:
        ValueError: If property name already registered
    
    Example:
        >>> def compute_rings(mol):
        ...     return Descriptors.RingCount(mol)
        >>> register_custom_property("num_rings", compute_rings, "decreasing", 
        ...                          "Number of rings - fewer is simpler")
    """
    if name in PROPERTY_REGISTRY:
        raise ValueError(f"Property '{name}' already registered")
    
    PROPERTY_REGISTRY[name] = (compute_fn, direction, description)


def get_available_properties() -> List[str]:
    """Get list of all available property names."""
    return sorted(PROPERTY_REGISTRY.keys())


def get_property_description(name: str) -> str:
    """Get description for a property."""
    if name not in PROPERTY_REGISTRY:
        raise ValueError(f"Unknown property '{name}'")
    return PROPERTY_REGISTRY[name][2]


def compute_all_properties_for_smiles(smiles: str, property_specs: Optional[List[PropertySpec]] = None) -> Dict[str, Optional[float]]:
    """
    Compute all specified properties for a single SMILES string.
    
    Useful for adding property values to saved parquet files or for quick property computation.
    
    Args:
        smiles: SMILES string to compute properties for
        property_specs: List of PropertySpec objects. If None, uses default properties (logp, qed, sa)
    
    Returns:
        Dictionary mapping property names to computed values (None if computation failed)
        
    Example:
    >>> from pairit.eval import PropertySpec
        >>> from rdkit.Chem import Crippen, QED
        >>> specs = [PropertySpec("logp", lambda m: Crippen.MolLogP(m), "higher")]
        >>> props = compute_all_properties_for_smiles("CCO", specs)
        >>> print(props)
        {'logp': -0.0014}
    """
    if property_specs is None:
        # Use default properties
        from pairit.eval import PropertySpec
        property_specs = [
            PropertySpec("logp", compute_logp, "higher"),
            PropertySpec("qed", compute_qed, "higher"),
            PropertySpec("sa", compute_sa, "lower"),
        ]
    
    results = {}
    
    if not smiles:
        return {spec.name: None for spec in property_specs}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {spec.name: None for spec in property_specs}
        
        for spec in property_specs:
            try:
                value = spec.fn(mol) # why was this func?
                results[spec.name] = float(value) if value is not None else None
            except Exception as e:
                print(f"Error computing {spec.name}: {e}")
                results[spec.name] = None
    except Exception:
        return {spec.name: None for spec in property_specs}
    
    return results


# ============================================================================
# RELEVANCE CHECK FUNCTIONS
# ============================================================================

def get_max_alkyl_chain_length(mol: Chem.Mol) -> int:
    """
    Find the maximum length of consecutive saturated alkyl chains.
    Only counts carbons with H only attached (CH2/CH3), not in rings.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        Maximum chain length found (0 if none or mol is None)
    """
    if mol is None:
        return 0
    
    # SMARTS patterns for consecutive sp3 carbons with only H attached
    patterns = [
        ('[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]', 10),
        ('[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]', 9),
        ('[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]', 8),
        ('[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]', 7),
        ('[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]', 6),
        ('[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]-[CH2,CH3;!R]', 5),
    ]
    
    for pattern_str, length in patterns:
        pattern = Chem.MolFromSmarts(pattern_str)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return length
    
    return 0


def get_max_ring_size(mol: Chem.Mol) -> int:
    """
    Get the maximum ring size in the molecule.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        Maximum ring size (0 if no rings or mol is None)
    """
    if mol is None:
        return 0
    
    ri = mol.GetRingInfo()
    max_size = 0
    for ring in ri.AtomRings():
        max_size = max(max_size, len(ring))
    return max_size


def get_seed_mol_stats(smiles: str) -> Dict[str, float]:
    """
    Get molecular statistics for a seed molecule.
    
    Args:
        smiles: SMILES string of the seed molecule
    
    Returns:
        Dictionary with keys: 'mw', 'max_ring', 'max_chain'
        Values are 0 if molecule cannot be parsed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'mw': 0.0, 'max_ring': 0, 'max_chain': 0}
    
    return {
        'mw': float(Descriptors.MolWt(mol)),
        'max_ring': get_max_ring_size(mol),
        'max_chain': get_max_alkyl_chain_length(mol),
    }


def load_dataset_statistics_from_files(
    file_paths: str | List[str],
    smiles_column: str = "smiles",
    max_samples: int = 10000,
    split_column: Optional[str] = None,
    split_value: Optional[str] = None,
) -> Dict[str, int]:
    """
    Load dataset statistics (max ring size and max alkyl chain) from one or more files.
    
    Adapted from unified_eval_zinc.py to work with any dataset files.
    Samples up to max_samples molecules for performance.
    
    Args:
        file_paths: Path to a single file or list of paths to multiple files
        smiles_column: Name of column containing SMILES strings (for CSV/parquet)
        max_samples: Maximum number of molecules to sample for statistics (default: 10000)
        split_column: Optional column name for filtering by split (e.g., "split")
        split_value: Optional value in split_column to filter for (e.g., "train")
    
    Returns:
        Dictionary with keys 'max_ring_size' and 'max_alkyl_chain'
        
    Example:
        >>> stats = load_dataset_statistics_from_files("data/train.csv", smiles_column="smiles")
        >>> print(stats)
        {'max_ring_size': 8, 'max_alkyl_chain': 5}
    """
    import pandas as pd
    from pathlib import Path
    
    max_ring = 0
    max_chain = 0
    
    # Normalize to list for uniform processing
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    total_processed = 0
    
    for file_path in file_paths:
        if total_processed >= max_samples:
            break
            
        path = Path(file_path)
        if not path.exists():
            continue
        
        try:
            # Load file based on extension
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix in ['.csv', '.txt']:
                # Try comma, then tab separator
                try:
                    df = pd.read_csv(path)
                except Exception:
                    try:
                        df = pd.read_csv(path, sep='\t')
                    except Exception:
                        # Assume space-separated .smi format
                        df = pd.read_csv(path, sep=' ', header=None, names=[smiles_column])
            elif path.suffix == '.smi':
                # .smi file: one SMILES per line
                smiles_list = []
                with open(path, 'r') as f:
                    for line in f:
                        if total_processed >= max_samples:
                            break
                        smiles = line.strip().split()[0]  # Take first token
                        if smiles:
                            smiles_list.append(smiles)
                            total_processed += 1
                
                # Process SMILES from .smi file
                for smiles in smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        max_ring = max(max_ring, get_max_ring_size(mol))
                        max_chain = max(max_chain, get_max_alkyl_chain_length(mol))
                continue
            else:
                # Unknown format, skip
                continue
            
            # Get SMILES column
            if smiles_column not in df.columns:
                # Try common column names
                for col in ['smiles', 'SMILES', 'canonical_smiles']:
                    if col in df.columns:
                        smiles_column = col
                        break
                else:
                    continue  # No SMILES column found
            
            # Filter by split if specified
            if split_column is not None and split_value is not None:
                if split_column in df.columns:
                    df = df[df[split_column] == split_value]
            
            # Sample up to max_samples
            remaining = max_samples - total_processed
            if len(df) > remaining:
                df = df.head(remaining)
            
            # Process SMILES
            for smiles in df[smiles_column]:
                if total_processed >= max_samples:
                    break
                if pd.isna(smiles) or not isinstance(smiles, str):
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    max_ring = max(max_ring, get_max_ring_size(mol))
                    max_chain = max(max_chain, get_max_alkyl_chain_length(mol))
                    total_processed += 1
                    
        except Exception as e:
            # Skip file on error
            continue
    
    return {
        'max_ring_size': max_ring,
        'max_alkyl_chain': max_chain,
    }

