"""
Distance Computation Utilities
================================

Provides fingerprint-cached distance computation to avoid redundant parsing
and fingerprinting when computing distances from one molecule to many.
"""

from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
import numpy as np


def parse_and_fingerprint(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[DataStructs.ExplicitBitVect]]:
    """
    Parse SMILES and compute Morgan fingerprint.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Tuple of (mol_object, fingerprint) or (None, None) if invalid
        
    Example:
        >>> mol, fp = parse_and_fingerprint("CCO")
        >>> if fp is not None:
        ...     # Use fingerprint for distance computation
        ...     pass
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = gen.GetFingerprint(mol)
        return mol, fp
    except Exception:
        return None, None


def compute_distances_with_fingerprint_caching(
    query_smiles: str,
    reference_smiles_list: List[str]
) -> List[float]:
    """
    Compute Tanimoto distances from one query to many references.
    
    Caches the query fingerprint to avoid redundant parsing when computing
    distances to multiple reference molecules. This is much more efficient
    than calling a distance function repeatedly with the same query.
    
    Args:
        query_smiles: Query SMILES string
        reference_smiles_list: List of reference SMILES
    
    Returns:
        List of Tanimoto distances (1.0 - similarity), where 0.0 = identical
        
    Example:
        >>> query = "CCO"
        >>> refs = ["CC", "CCC", "CCCC"]
        >>> distances = compute_distances_with_fingerprint_caching(query, refs)
        >>> # Query is only parsed once, not 3 times
    """
    # Parse and fingerprint query ONCE
    query_mol, query_fp = parse_and_fingerprint(query_smiles)
    if query_fp is None:
        return [1.0] * len(reference_smiles_list)  # Max distance for invalid
    
    # Compute distances to all references (reusing query fingerprint)
    distances = []
    for ref_smiles in reference_smiles_list:
        ref_mol, ref_fp = parse_and_fingerprint(ref_smiles)
        if ref_fp is None:
            distances.append(1.0)  # Max distance for invalid
        else:
            similarity = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            distances.append(1.0 - similarity)
    
    return distances


def compute_min_distance_to_set(
    query_smiles: str,
    reference_smiles_list: List[str]
) -> float:
    """
    Compute minimum distance from query to a reference set.
    
    Useful for finding the closest molecule in a ground truth set.
    
    Args:
        query_smiles: Query SMILES
        reference_smiles_list: Reference set
    
    Returns:
        Minimum distance (float), or 1.0 if all references are invalid
        
    Example:
        >>> query = "CCO"
        >>> gt_set = ["CC", "CCC", "CCCC"]
        >>> min_dist = compute_min_distance_to_set(query, gt_set)
        >>> # Returns distance to closest molecule in gt_set
    """
    distances = compute_distances_with_fingerprint_caching(
        query_smiles, 
        reference_smiles_list
    )
    return float(np.min(distances)) if distances else 1.0

