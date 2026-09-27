import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem, rdFMCS
import selfies as sf
import numpy as np
import py3Dmol
import pickle
import os
from tqdm import tqdm
from atomic_datasets.utils.rdkit import is_molecule_sane
from typing import Optional

import itertools
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import polars as pl
#from rdkit.Chem.rdmolfiles import SDMolSupplier

import time
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
import pandas as pd

import numpy as np
import matplotlib
#matplotlib.use("Agg")
matplotlib.rcParams['savefig.bbox'] = 'standard'     # important
matplotlib.rcParams['savefig.transparent'] = False
import matplotlib.pyplot as plt
from itertools import cycle
from typing import Iterable, Optional, List, Literal

def is_valency_ok(mol):
    pt = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()

        # Current valence: from the atom itself (new API)
        try:
            current_valence = atom.GetTotalValence() #atom.GetValence(which=rdchem.AtomValence.ALL)
        except Exception as e:
            print(f"Warning: cannot compute valence for atom {atomic_num}: {e}")
            continue

        # Allowed max valence
        try:
            max_valence = max([v for v in pt.GetValenceList(atomic_num)]) #pt.GetValence(atomic_num, which=Chem.GetPeriodicTable().ALL)
        except Exception as e:
            print(f"Warning: cannot get max valence for atom {atomic_num}: {e}")
            continue

        if current_valence > max_valence:
            # get partial or formal charge. if current valence > max_valence + current charge, then it's not ok
            print(f'valence for atom {pt.GetElementSymbol(atomic_num)} is {current_valence} but max is {max_valence}')
            return False
    return True

def visualize_mol(mol):
    view = py3Dmol.view(data=Chem.MolToMolBlock(mol),style={'stick':{},'sphere':{'scale':0.3}})
    view.zoomTo()
    view.show()

def human_readable_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

# mol = dataset[0]['properties']['rdkit_mol']
# mol.GetNumConformers()
# np.linalg.norm(mol.GetConformer().GetPositions() - dataset[0]['nodes']['positions'])

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

def my_mols_to_3d_grid(mol_pairs, rows=2, cols=4, spacing=5.0, col_titles=None, width=800, height=600):
    """
    Show a grid of molecules with their 3D conformers using py3Dmol.
    Each molecule is translated to a unique grid location in 3D.

    Args:   
        mol_pairs: list of (mol_i, mol_j) tuples (RDKit Mol objects)
        rows: number of rows in the grid
        cols: number of columns in the grid
        spacing: spacing between molecules in the grid
        col_titles: list of strings for column titles (length should be >= cols)
        width: width of the view in pixels (default: 800)
        height: height of the view in pixels (default: 600)
    """
    view = py3Dmol.view(width=width, height=height)
    count = 0

    for row in range(rows):
        for col in range(cols):
            if count >= len(mol_pairs)*2:
                break
            mol = mol_pairs[col][row]
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol)

            # Translate molecule to grid position
            conf = mol.GetConformer()
            #dx, dy = j * spacing, -i * spacing

            # Check if molecule is centered (zero center of mass) and center it if needed
            positions = []
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                positions.append([pos.x, pos.y, pos.z])
            
            positions = np.array(positions)
            center_of_mass = np.mean(positions, axis=0)
            
            # Check if center of mass is close to zero (within tolerance)
            tolerance = 1e-6
            if np.any(np.abs(center_of_mass) > tolerance):
                
                # Center the molecule by subtracting the center of mass from all positions
                for k in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(k)
                    centered_pos = pos - Chem.rdGeometry.Point3D(*center_of_mass)
                    conf.SetAtomPosition(k, centered_pos)

            offset = np.array([col * spacing, -row * spacing, 0.0])
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                conf.SetAtomPosition(k, pos + Chem.rdGeometry.Point3D(*offset)) #(pos.x + dx, pos.y + dy, pos.z))

            # Add to viewer
            mb = Chem.MolToMolBlock(mol)
            view.addModel(mb, 'mol')
            view.setStyle({'model': count}, {'stick': {}, 'sphere': {'scale': 0.3}})
            count += 1

    # Add column titles as text labels if provided
    if col_titles and len(col_titles) >= cols:
        title_height = spacing * 0.8  # Height above the grid for titles (proportional to spacing)
        for col in range(cols):
            if col < len(col_titles):
                title = col_titles[col]
                # Position title above the center of each column
                title_x = col * spacing 
                title_y = title_height
                title_z = 0
                
                # Add text label
                view.addLabel(
                    title,
                    {
                        'position': {'x': title_x, 'y': title_y, 'z': title_z},
                        'fontSize': 16,
                        'fontColor': 'black',
                        'backgroundColor': 'white',
                        'borderThickness': 1,
                        'borderColor': 'black'
                    }
                )
    
    view.zoomTo()
    view.show()  # Auto-display the view
    return view

def visualize_mol_pairs_grid(
    mol_pairs,
    mols_per_row=2,
    legends=None,
    col_titles=None,
    size=(300, 300),
    title_fontsize=24
):
    """
    Visualizes a grid of 3D conformers from molecule pairs, with optional column titles.
    
    Args:
        mol_pairs: list of (mol_i, mol_j) tuples (RDKit Mol objects)
        mols_per_row: number of molecules per row (should be 2 for pairs)
        legends: list of string legends, length should be 2 * len(mol_pairs)
        col_titles: list of column titles, e.g., ["Mol A", "Mol B"]
        size: size of each sub-image (w, h)
        title_fontsize: font size for the column titles
    """
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import MolsToGridImage

    # Flatten mols
    mols = [mol for pair in mol_pairs for mol in pair]

    # Compute 2D coords if needed
    # for mol in mols:
    #     rdDepictor.Compute2DCoords(mol)

    # Generate placeholder legends if not provided
    if legends is None:
        legends = [f"Mol {i}" for i in range(len(mols))]

    # Generate main image
    img = MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=size,
        legends=legends,
        useMolBlock=False,
    )

    # Add column titles above image (optional)
    if col_titles:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", title_fontsize)
        except IOError:
            font = ImageFont.load_default()

        spacing = 10
        title_y = spacing
        for i, title in enumerate(col_titles):
            x = i * size[0] + size[0] // 2
            text_width, _ = draw.textsize(title, font=font)
            draw.text((x - text_width // 2, title_y), title, fill="black", font=font)

    return img

def display_mols_3d_grid(mol_pairs, rows=2, cols=4, spacing=5.0, col_titles=None, width=800, height=600):
    """
    Display a grid of molecules with their 3D conformers using py3Dmol.
    This function automatically shows the visualization.
    
    Args:   
        mol_pairs: list of (mol_i, mol_j) tuples (RDKit Mol objects)
        rows: number of rows in the grid
        cols: number of columns in the grid
        spacing: spacing between molecules in the grid
        col_titles: list of strings for column titles (length should be >= cols)
        width: width of the view in pixels (default: 800)
        height: height of the view in pixels (default: 600)
    """
    view = my_mols_to_3d_grid(mol_pairs, rows, cols, spacing, col_titles, width, height)
    view.show()
    return view

def get_shape_tanimoto(molA, molB, return_extra=True, verbose=True):
    
    start_time = time.time()

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    mcs_result = rdFMCS.FindMCS([molA, molB], parameters=mcs_params)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    # Find matching atoms
    matchA = molA.GetSubstructMatch(mcs_mol)
    matchB = molB.GetSubstructMatch(mcs_mol)
    matching_time = time.time() - start_time

    try:
        # Align molecules
        align_start = time.time()
        rmsd = AllChem.AlignMol(molB, molA, atomMap=list(zip(matchB, matchA)))
        align_time = time.time() - align_start

        # Calculate shape Tanimoto
        tani_start = time.time() 
        shape_tani_dist = ShapeTanimotoDist(molA, molB)
        tani_time = time.time() - tani_start
        
        if verbose:
            elapsed = time.time() - start_time
            if elapsed > 10:
                print(f"Mol A atoms: {molA.GetNumAtoms()} Mol B atoms: {molB.GetNumAtoms()} | Find matching atoms: {matching_time:.3f}s | Alignment: {align_time:.3f}s | Shape tani: {tani_time:.3f}s | Total: {time.time() - start_time:.3f}s")
                print(f" rmsd: {rmsd:.3f} | shape tani: {shape_tani_dist:.3f}")
            
        if return_extra:
            return shape_tani_dist, matchA, matchB, rmsd, matching_time
        else:
            return shape_tani_dist
    except Exception as e:
        if verbose:
            elapsed = time.time() - start_time
            print(f"    (threw exception) Elapsed time: {elapsed:.2f} seconds")
            
        if return_extra:
            return 100, [], [], -1, -1
        else:
            return 100


def visualize_mol_grid(mols, titles=None, num_rows=None, num_cols=None, spacing=5.0, width=800, height=600, title_height=None):
    """
    Visualize a grid of molecules using py3Dmol.
    
    Args:
        mols: list of RDKit Mol objects to visualize
        titles: optional list of strings for titles above each molecule
        num_rows: optional number of rows (auto-calculated if None)
        num_cols: optional number of columns (auto-calculated if None)
        spacing: spacing between molecules in the grid
        width: width of the view in pixels
        height: height of the view in pixels
        title_height: optional float, controls vertical space for titles/row spacing (default: spacing*0.8 if titles, else 0)
    
    Returns:
        py3Dmol view object
    """
    if not mols:
        raise ValueError("mols list cannot be empty")
    
    # Auto-calculate grid dimensions if not provided
    if num_rows is None and num_cols is None:
        n_mols = len(mols)
        num_cols = int(np.ceil(np.sqrt(n_mols)))
        num_rows = int(np.ceil(n_mols / num_cols))
        if num_rows > num_cols:
            num_cols, num_rows = num_rows, num_cols
    elif num_rows is None:
        num_rows = int(np.ceil(len(mols) / num_cols))
    elif num_cols is None:
        num_cols = int(np.ceil(len(mols) / num_rows))
    
    view = py3Dmol.view(width=width, height=height)
    count = 0
    
    # Calculate title spacing if titles are provided
    if title_height is not None:
        row_title_spacing = title_height if titles else 0
    else:
        row_title_spacing = spacing * 0.8 if titles else 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            mol_idx = row * num_cols + col
            if mol_idx >= len(mols):
                break
            mol = mols[mol_idx]
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol)
            conf = mol.GetConformer()
            positions = []
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                positions.append([pos.x, pos.y, pos.z])
            positions = np.array(positions)
            center_of_mass = np.mean(positions, axis=0)
            tolerance = 1e-6
            if np.any(np.abs(center_of_mass) > tolerance):
                for k in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(k)
                    centered_pos = pos - Chem.rdGeometry.Point3D(*center_of_mass)
                    conf.SetAtomPosition(k, centered_pos)
            # Position molecule in grid
            offset = np.array([col * spacing, -(row * spacing + row_title_spacing), 0.0])
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                conf.SetAtomPosition(k, pos + Chem.rdGeometry.Point3D(*offset))
            mb = Chem.MolToMolBlock(mol)
            view.addModel(mb, 'mol')
            view.setStyle({'model': count}, {'stick': {}, 'sphere': {'scale': 0.3}})
            count += 1
    # Add titles if provided
    if titles:
        for row in range(num_rows):
            for col in range(num_cols):
                mol_idx = row * num_cols + col
                if mol_idx >= len(mols):
                    break
                if mol_idx < len(titles) and titles[mol_idx]:
                    title = titles[mol_idx]
                    title_x = col * spacing
                    title_y = -(row * spacing + row_title_spacing * 0.5)  # Position above molecule
                    view.addLabel(
                        title,
                        {
                            'position': {'x': title_x, 'y': title_y, 'z': 0},
                            'fontSize': 14,
                            'fontColor': 'black',
                            'backgroundColor': 'white',
                            'borderThickness': 1,
                            'borderColor': 'black'
                        }
                    )
    view.zoomTo()
    view.show()
    return view

def get_tanimoto_distance(smiles1: str, smiles2: str, verbose: bool = True) -> float:
    """
    Compute Tanimoto distance between two SMILES strings using Morgan fingerprints.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        float: Tanimoto distance (1.0 - similarity), where 0.0 = identical, 1.0 = completely different
    """
    try:
    
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            if verbose:
                print(f"Invalid molecule: {smiles1} or {smiles2}")
            return 1.0  # Maximum distance for invalid molecules
            
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = gen.GetFingerprint(mol1)
        fp2 = gen.GetFingerprint(mol2)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return 1.0 - similarity  # Convert to distance
        
    except Exception as e:
        if verbose:
            print(f"Error computing Tanimoto distance: {e}")
        return 1.0  # Maximum distance on error


def compute_single_molecule_snn(
    smiles: str,
    reference_smiles: List[str],
    *,
    radius: int = 2,
    nbits: int = 2048,
):
    """
    Compute the single-nearest-neighbor (SNN) in a reference pool by Tanimoto similarity.

    Returns:
        (closest_ref_smiles, closest_ref_similarity, closest_ref_index)

    Notes:
    - This is a lightweight helper used by validation for logging the closest reference molecule.
    - Invalid molecules are skipped.
    """
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
        if mol is None:
            return None, None, None

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(nbits))
        fp = gen.GetFingerprint(mol)

        best_sim = None
        best_idx = None
        best_smi = None
        for i, rs in enumerate(reference_smiles or []):
            rm = Chem.MolFromSmiles(rs) if rs is not None else None
            if rm is None:
                continue
            rfp = gen.GetFingerprint(rm)
            sim = float(DataStructs.TanimotoSimilarity(fp, rfp))
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_idx = int(i)
                best_smi = rs

        return best_smi, best_sim, best_idx
    except Exception:
        return None, None, None


def get_shape_tanimoto_distance(smiles1: str, smiles2: str, verbose: bool = True) -> float:
    """
    Compute shape Tanimoto distance between two SMILES strings.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        verbose: Whether to print verbose output
        
    Returns:
        float: Shape Tanimoto distance (lower = more similar shapes)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 100.0  # Maximum distance for invalid molecules
            
        # Generate 3D conformers if needed
        if mol1.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol1)
        if mol2.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol2)
            
        # Use the existing get_shape_tanimoto function
        shape_dist = get_shape_tanimoto(mol1, mol2, return_extra=False, verbose=verbose)
        return shape_dist
        
    except Exception as e:
        if verbose:
            print(f"Error computing shape Tanimoto distance: {e}")
        return 100.0  # Maximum distance on error

def filter_top_pairs_per_molecule(df: pd.DataFrame, property_key: Optional[str] = 'shape_tanimoto_distance', percentile: Optional[float] = None, num_pairs: Optional[int] = None, groupby_name="molecule_a_idx") -> pd.DataFrame:
    """
    For each unique molecule_b_idx, keep only the top pairs with the lowest shape_tanimoto_distance.
    Can filter by either percentile or concrete number of pairs per molecule.

    Args:
        df: DataFrame with columns ['molecule_a_idx', 'molecule_b_idx', 'shape_tanimoto_distance', ...]
        percentile: float in (0, 100), e.g., 5.0 for top 5% (closest pairs). If provided, num_pairs is ignored.
        num_pairs: int > 0, e.g., 10 for top 10 closest pairs per molecule. Used only if percentile is None.

    Returns:
        Filtered DataFrame with only the top pairs per molecule_b_idx.
    """
    print(f"groupby_name: {groupby_name}")
    if percentile is not None and num_pairs is not None:
        raise ValueError("Cannot specify both percentile and num_pairs. Use one or the other.")
    
    if percentile is None and num_pairs is None:
        raise ValueError("Must specify either percentile or num_pairs.")
    
    if percentile is not None:
        if percentile <= 0 or percentile > 100:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
        
        def filter_group_percentile(group):
            cutoff = group[property_key].quantile(percentile / 100.0)
            return group[group[property_key] <= cutoff]
        
        filtered = df.groupby(groupby_name, group_keys=False).apply(filter_group_percentile) # OH: should these be grouping by molecule_a_idx or molecule_b_idx?
    
    else:  # num_pairs is not None
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        
        def filter_group_num_pairs(group):
            # Sort by property_key, e.g. shape_tanimoto_distance, and take top num_pairs
            return group.nsmallest(num_pairs, property_key)
        
        filtered = df.groupby(groupby_name, group_keys=False).apply(filter_group_num_pairs)
    
    return filtered.reset_index(drop=True)

def ensure_conformer(mol, force_3d=True):
    """
    Ensure the molecule has at least one conformer.
    If not, generate one (3D with ETKDG or 2D with Compute2DCoords).
    Returns the (possibly modified) molecule and the conformer id.
    """
    if mol.GetNumConformers() == 0:
        if force_3d:
            mol = Chem.AddHs(mol)  # needed for good 3D embedding
            cid = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if cid == -1:
                raise ValueError("3D embedding failed")
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
            return mol, cid
        else:
            Chem.rdDepictor.Compute2DCoords(mol)
            return mol, 0
    else:
        return mol, 0  # already had at least one conformer


# Optional deps
try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_SND = True
except Exception:
    _HAS_SND = False


def overlapping_histograms(
    data_list: Iterable[Iterable[float]],
    labels: Optional[List[str]] = None,
    density: bool = True,
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.35,
    edgecolor: str = "black",
    linewidth: float = 2.0,
    curve: Literal["none", "fast", "kde"] = "fast",
    kde_subsample: int = 5000,
    kde_grid: int = 500,
    show_curves_only: bool = False,
):
    """
    Fast overlapping histograms with optional smooth curves.

    Speed tricks:
    - curve="fast": draw a smooth curve by Gaussian-smoothing the histogram
      (very fast, no O(n*m) cost). Requires SciPy ndimage; otherwise skipped.
    - curve="kde": true KDE; subsamples to `kde_subsample` for speed.
    - curve="none": just histograms.

    Parameters
    ----------
    data_list : iterable of iterables
        Each element is 1D-like. Coerced to np.ndarray and cleaned of NaN/inf.
    labels : list[str], optional
        If None, no labels are created (no "Series" fallback).
    density : bool, default True
        If True, area under each histogram ≈ 1 (PDF). Else show counts.
    bins : int or sequence, default 50
        Common bin edges for all datasets.
    ax : plt.Axes, optional
    alpha : float, default 0.35
        Bar transparency.
    edgecolor : str, default "black"
    linewidth : float, default 2.0
        Curve line width.
    curve : {"none", "fast", "kde"}, default "fast"
        Smooth line mode.
    kde_subsample : int, default 5000
        Max samples used for KDE (ignored for curve="fast"/"none").
    kde_grid : int, default 500
        Number of x-points for curve plotting.
    show_curves_only : bool, default False
        If True, only show curves (no histogram bars).

    Returns
    -------
    ax, hist_handles, curve_lines, bin_edges
    """
    ax = ax if ax is not None else plt.gca()

    # Coerce & clean
    arrays = []
    for arr in data_list:
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        arrays.append(a)

    if labels is None:
        labels = [None] * len(arrays)
    else:
        if len(labels) < len(arrays):
            labels = labels + [None] * (len(arrays) - len(labels))
        elif len(labels) > len(arrays):
            labels = labels[:len(arrays)]

    # Determine common bin edges
    if np.isscalar(bins):
        non_empty = [a for a in arrays if a.size > 0]
        if not non_empty:
            raise ValueError("All input arrays are empty after cleaning.")
        dmin = min(a.min() for a in non_empty)
        dmax = max(a.max() for a in non_empty)
        if dmin == dmax:
            dmin -= 0.5
            dmax += 0.5
        bin_edges = np.linspace(dmin, dmax, int(bins) + 1)
    else:
        bin_edges = np.asarray(bins)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("`bins` must be 1D with at least two edges.")

    # Colors (version-compatible)
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = [next(color_cycle) for _ in arrays]

    # Precompute histograms with numpy (faster than multiple plt.hist calls)
    hist_handles = []
    counts_list = []
    for a, label, color in zip(arrays, labels, colors):
        if a.size == 0:
            counts_list.append(None)
            continue
        counts, _ = np.histogram(a, bins=bin_edges)
        bin_widths = np.diff(bin_edges)
        if density:
            # Convert counts to PDF per bin
            counts = counts / (counts.sum() * bin_widths)
        counts_list.append(counts)
        # Draw filled histogram quickly with stepfilled-like effect
        # Use bar with shared edges to minimize artists
        if not show_curves_only:
            ax.bar(
                bin_edges[:-1],
                counts,
                width=bin_widths,
                align="edge",
                alpha=alpha,
                edgecolor=edgecolor,
                color=color,
                label=label,
            )
        hist_handles.append(ax)

    # Curves
    curve_lines = []
    if curve == "kde":
        if not _HAS_SCIPY:
            # Fallback text if SciPy isn't available
            ax.text(0.99, 0.99, "KDE unavailable (SciPy missing)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, alpha=0.7)
        else:
            x_min, x_max = bin_edges[0], bin_edges[-1]
            x = np.linspace(x_min, x_max, int(kde_grid))
            mean_bin_width = np.mean(np.diff(bin_edges))
            for a, color in zip(arrays, colors):
                if a.size < 2:
                    continue
                # Subsample for speed
                if a.size > kde_subsample:
                    idx = np.random.default_rng(0).choice(a.size, size=kde_subsample, replace=False)
                    a_use = a[idx]
                else:
                    a_use = a
                kde = gaussian_kde(a_use)
                y = kde(x)
                if not density:
                    # Scale KDE to counts scale: integrate to n, not 1
                    y = y * (a.size * mean_bin_width)
                line = ax.plot(x, y, color=color, linewidth=linewidth, label=label)
                curve_lines.extend(line)

    elif curve == "fast":
        # Smooth the histogram with a Gaussian filter (very fast)
        if not _HAS_SND:
            # If ndimage missing, silently skip smoothing
            pass
        else:
            # Compute sigma (in bins) ~ Scott's rule bandwidth / bin_width
            # Scott’s h = 1.06 * std * n^{-1/5}
            bin_width = np.mean(np.diff(bin_edges))
            for a, counts, color in zip(arrays, counts_list, colors):
                if counts is None or a.size < 2:
                    continue
                std = np.std(a)
                if std == 0:
                    continue
                n = a.size
                h = 1.06 * std * (n ** (-1/5))  # bandwidth in data units
                sigma_bins = max(h / bin_width, 0.5)  # floor to avoid zero/very sharp
                smooth = gaussian_filter1d(counts.astype(float), sigma=sigma_bins, mode="nearest")
                # Plot as a line centered at bin centers
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                line = ax.plot(centers, smooth, linewidth=linewidth, color=color, label=label)
                curve_lines.extend(line)

    # Labels and legend
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    # Only show legend if there are non-None labels
    if any(label is not None for label in labels):
        ax.legend()

    plt.show()

    # Example usage:
    # ax, hist_handles, curve_lines, edges = overlapping_histograms(data_list)
    # ax.set_title("My Histogram")  # Customize the plot using returned ax
    # ax.set_xlabel("X Label")
    # ax.set_ylabel("Y Label") 
    # plt.show()  # Display the plot
    return ax, hist_handles, curve_lines, bin_edges


def smiles_num_atoms(smiles: str) -> int:
    """
    Given a SMILES string, return the number of atoms in the molecule.
    Returns None if the SMILES string cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else -1

def save_fig_as_png(fig, out_path, dpi=300, background="white", drop_alpha=True):
    """
    Rasterize a Matplotlib figure exactly as drawn and save as PNG.
    Avoids bbox_inches/tight second draw issues that can crop to the first axes.
    """
    import numpy as np
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    # Solid background (important if you later save JPEGs)
    fig.patch.set_facecolor(background)
    for ax in fig.axes:
        ax.set_facecolor(background)

    # Ensure Agg canvas and desired DPI
    fig.set_dpi(dpi)
    canvas = FigureCanvas(fig)
    canvas.draw()

    try:
        # Newer/portable API
        buf, (w, h) = canvas.print_to_buffer()
        rgba = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
    except Exception:
        # Fallback for canvases that expose tostring_argb() instead
        w, h = canvas.get_width_height()
        argb = canvas.tostring_argb()
        rgba = np.frombuffer(argb, np.uint8).reshape(h, w, 4)
        rgba = rgba[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA

    img = rgba[:, :, :3] if drop_alpha else rgba  # drop alpha if you want opaque PNG
    Image.fromarray(img).save(out_path, format="PNG", compress_level=6)


def generate_smiles_grid_rdkit(input_smiles_list, n_cols=4, save_dir=None, save_name='grid.png', titles=None):
    """
    Generate a grid visualization of 2D molecular structures from SMILES strings using RDKit's MolsToGridImage.
    Optimized for PNG output.
    
    Args:
        input_smiles_list (List[str]): List of SMILES strings to visualize
        n_cols (int): Number of columns in the grid. Default 4.
        save_dir (str): Directory to save the visualization
        save_name (str): Name of file to save the visualization (without directory)
        titles (List[str], optional): List of titles for each molecule. Must match length of smiles_list.
        
    Returns:
        PIL.Image: The generated grid image
    """
    from rdkit.Chem import Draw
    
    smiles_list = []
    for elt in input_smiles_list:
        if elt is not None:
            smiles_list.append(elt)
        else:
            smiles_list.append("")
    
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles_list]
    
    # Debug information
    valid_mols = sum(1 for m in mols if m is not None)
    print(f"Generated {len(mols)} molecules from SMILES, {valid_mols} valid")
    
    if all(m is None for m in mols):
        print("Warning: No valid molecules found in SMILES list")
        print(f"Input SMILES: {smiles_list[:5]}...")  # Show first 5 for debugging
    
    # Prepare molecules and legends without dropping invalid entries
    # Invalid entries become empty RDKit molecules (blank cells)
    prepared_mols = []
    prepared_legends = []
    for i, mol in enumerate(mols):
        if mol is None:
            mol = Chem.Mol()  # empty molecule renders as blank
        prepared_mols.append(mol)
        if titles is not None and i < len(titles):
            prepared_legends.append(titles[i] if titles[i] is not None else "")
        else:
            prepared_legends.append("")
    
    if not prepared_mols:
        print("No valid molecules to display")
        return None
    
    # Generate grid image using RDKit
    img = Draw.MolsToGridImage(
        prepared_mols,
        molsPerRow=n_cols,
        subImgSize=(300, 300),
        legends=prepared_legends,
        #useMolBlock=False,
    )
    
    # Save if path provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, save_name)
        img.save(full_path)
        print(f"Saved RDKit grid to: {full_path}")
    
    return img

def generate_smiles_grid(
    input_smiles_list, 
    n_cols=None, 
    save_dir=None, 
    save_name='grid.pdf', 
    titles=None,
    row_labels=None,
    col_labels=None,
    property_deltas=None,
    property_name=None,
    memorization_flags=None,
    is_duplicate=None,
    extra_labels=None,
    title_fontsize=10,
    annotation_fontsize=None,
    row_label_fontsize=12,
    col_label_fontsize=12,
    row_spacing=4,
):
    """
    Generate a grid visualization of 2D molecular structures from SMILES strings.
    
    Args:
        input_smiles_list: List of SMILES strings OR list of lists (where each inner list is a row).
                          If list of lists, n_cols will be set to max length of any inner list.
        n_cols (int, optional): Number of columns in the grid. If None and input is list of lists,
                                will be set to max length. Default 4 for flat list.
        save_dir (str, optional): Directory to save the visualization
        save_name (str): Name of file to save the visualization (without directory)
        titles (List[str], optional): List of titles for each molecule. Must match length of flattened smiles_list.
        row_labels (List[str], optional): List of labels for each row. Length should match number of rows.
        col_labels (List[str], optional): List of labels for each column. Length should match n_cols.
        property_deltas (List[Optional[float]] | List[List[Optional[float]]], optional): Property deltas to display 
                                                           below each molecule. Can be a flat list or list of lists 
                                                           (matching input_smiles_list structure). Formatted as 
                                                           "Δ{property_name}={value:.3f}".
        property_name (str, optional): Name of the property for delta display. Defaults to "prop".
        memorization_flags (List[Optional[bool]] | List[List[Optional[bool]]], optional): Memorization flags. 
                                                           Can be a flat list or list of lists (matching 
                                                           input_smiles_list structure). If True, adds a red box 
                                                           outline around the molecule.
        is_duplicate (List[Optional[bool]] | List[List[Optional[bool]]], optional): Duplicate flags. 
                                                           Can be a flat list or list of lists (matching 
                                                           input_smiles_list structure). If True, adds a blue box 
                                                           outline around the molecule.
        extra_labels (List[str] | List[List[str]], optional): Additional labels to add below each molecule 
                                           visualization, in addition to memorization and property delta annotations.
                                           Can be a flat list or list of lists (matching input_smiles_list structure).
        title_fontsize (int, optional): Font size for explicit titles. Default 10.
        annotation_fontsize (int, optional): Font size for per-image labels (memorization and property delta 
                                           annotations). If None, uses title_fontsize. Default None.
        row_label_fontsize (int, optional): Font size for row labels. Default 12.
        col_label_fontsize (int, optional): Font size for column labels. Default 12.
        row_spacing (float, optional): Spacing between rows. If None, uses default spacing (4.0 per row).
                                      Larger values increase spacing, smaller values decrease spacing. Default None.
        
    Returns:
        matplotlib Figure object
    """
    from rdkit.Chem import Draw
    
    # Helper function to flatten a list or list of lists, matching the SMILES structure
    def flatten_to_match_smiles(data, smiles_rows, n_cols, n_rows):
        """Flatten data to match the flattened SMILES structure."""
        if data is None:
            return None
        # Check if data is a list of lists
        if data and isinstance(data[0], list):
            # Flatten list of lists, padding rows to match n_cols
            flattened = []
            for row in data:
                padded_row = row + [None] * (n_cols - len(row))
                flattened.extend(padded_row[:n_cols])
            return flattened
        else:
            # Already a flat list
            return data
    
    # Handle input as list or list of lists
    if input_smiles_list and isinstance(input_smiles_list[0], list):
        # Input is list of lists (each list is a row)
        smiles_rows = input_smiles_list
        n_rows = len(smiles_rows)
        if n_cols is None:
            n_cols = max(len(row) for row in smiles_rows) if smiles_rows else 4
        # Flatten to a single list, padding rows with None to match n_cols
        smiles_list = []
        for row in smiles_rows:
            padded_row = row + [None] * (n_cols - len(row))
            smiles_list.extend(padded_row[:n_cols])
        
        # Flatten property_deltas, memorization_flags, is_duplicate, and extra_labels if they're lists of lists
        property_deltas = flatten_to_match_smiles(property_deltas, smiles_rows, n_cols, n_rows)
        memorization_flags = flatten_to_match_smiles(memorization_flags, smiles_rows, n_cols, n_rows)
        is_duplicate = flatten_to_match_smiles(is_duplicate, smiles_rows, n_cols, n_rows)
        extra_labels = flatten_to_match_smiles(extra_labels, smiles_rows, n_cols, n_rows)
    else:
        # Input is a flat list
        smiles_list = []
        for elt in input_smiles_list:
            if elt is not None:
                smiles_list.append(elt)
            else:
                smiles_list.append("")
        if n_cols is None:
            n_cols = 4
        n_rows = (len(smiles_list) + n_cols - 1) // n_cols
    
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(str(s)) if s is not None and s != "" else None for s in smiles_list]
    
    # Debug information
    valid_mols = sum(1 for m in mols if m is not None)
    print(f"Generated {len(mols)} molecules from SMILES, {valid_mols} valid")
    
    if all(m is None for m in mols):
        print("Warning: No valid molecules found in SMILES list")
        print(f"Input SMILES: {smiles_list[:5]}...")  # Show first 5 for debugging
    
    # Build titles from property_deltas, memorization_flags, and extra_labels
    titles_auto_generated = False
    if titles is None and (property_deltas is not None or memorization_flags is not None or extra_labels is not None):
        titles = []
        titles_auto_generated = True
        for idx in range(len(mols)):
            title_parts = []
            
            # Add extra label if provided
            if extra_labels is not None and idx < len(extra_labels) and extra_labels[idx] is not None:
                title_parts.append(str(extra_labels[idx]))
            
            # Add property delta if provided
            if property_deltas is not None and idx < len(property_deltas) and property_deltas[idx] is not None:
                pname = property_name or "prop"
                title_parts.append(f"Δ{pname}={float(property_deltas[idx]):.3f}")
            
            # Note: Memorization is indicated by red box outline, not text
            
            # Join all parts with spaces
            titles.append(" ".join(title_parts) if title_parts else "")
    
    # Determine font size to use for titles
    if annotation_fontsize is None:
        annotation_fontsize = title_fontsize
    # Use annotation_fontsize for auto-generated titles, title_fontsize for explicit titles
    effective_title_fontsize = annotation_fontsize if titles_auto_generated else title_fontsize
    
    # Validate titles if provided
    if titles is not None and len(titles) != len(mols):
        raise ValueError(f"Number of titles ({len(titles)}) must match number of molecules ({len(mols)})")
    
    # Validate row_labels
    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError(f"Number of row_labels ({len(row_labels)}) must match number of rows ({n_rows})")
    
    # Validate col_labels
    if col_labels is not None and len(col_labels) != n_cols:
        raise ValueError(f"Number of col_labels ({len(col_labels)}) must match number of columns ({n_cols})")
    
    # Calculate grid dimensions
    n_mols = len(mols)
    
    # Create grid of molecule drawings using subplots
    # Add extra space for row/column labels if needed
    fig_width = 4 * n_cols
    # Use row_spacing if provided, otherwise default to 4.0 per row
    height_per_row = row_spacing if row_spacing is not None else 4.0
    fig_height = height_per_row * n_rows
    if row_labels is not None:
        fig_width += 0.5  # Extra space for row labels
    if col_labels is not None:
        fig_height += 0.5  # Extra space for column labels
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                             constrained_layout=True)
    fig.set_facecolor('white')
    
    # Normalize axes to a flat list for easy indexing
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]

    for idx, mol in enumerate(mols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes_list[idx]
        ax.set_facecolor('white')
        
        if mol is not None:
            try:
                img = Draw.MolToImage(mol, size=(300, 300))
                ax.imshow(np.asarray(img))
            except Exception as e:
                print(f"Error rendering molecule {idx}: {e}")
                ax.text(0.5, 0.5, f"Error\n{smiles_list[idx][:20] if idx < len(smiles_list) else 'Unknown'}...", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            # Handle invalid molecules
            invalid_smiles = smiles_list[idx] if idx < len(smiles_list) else "Unknown"
            # ax.text(0.5, 0.5, f"Invalid SMILES\n{invalid_smiles[:20]}...", 
            #        ha='center', va='center', transform=ax.transAxes, 
            #        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            print(f'invalid smiles: {invalid_smiles}')
            ax.text(0.5, 0.5, f"Invalid SMILES", 
                   ha='center', va='center', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax.axis('off')
        
        # Add red box outline for memorized molecules
        if memorization_flags is not None and idx < len(memorization_flags) and memorization_flags[idx]:
            from matplotlib.patches import Rectangle
            # Add red rectangle border around the entire axes
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                           fill=False, edgecolor='red', linewidth=3, zorder=10)
            ax.add_patch(rect)
        
        # Add blue box outline for duplicate molecules
        if is_duplicate is not None and idx < len(is_duplicate) and is_duplicate[idx]:
            from matplotlib.patches import Rectangle
            # Add blue rectangle border around the entire axes
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                           fill=False, edgecolor='blue', linewidth=3, zorder=11)
            ax.add_patch(rect)
        
        # Add title (which may include property deltas, memorization flags, extra_labels)
        if titles is not None and idx < len(titles):
            if titles_auto_generated and property_deltas is not None and idx < len(property_deltas) and property_deltas[idx] is not None:
                # Render title with colored property delta using ax.text for better control
                # Build parts
                parts = []
                if extra_labels is not None and idx < len(extra_labels) and extra_labels[idx] is not None:
                    parts.append((str(extra_labels[idx]), 'black'))
                
                # Property delta with color (green if positive, red if negative)
                pname = property_name or "prop"
                delta_value = float(property_deltas[idx])
                delta_text = f"Δ{pname}={delta_value:.3f}"
                delta_color = 'green' if delta_value > 0 else 'red' if delta_value < 0 else 'black'
                parts.append((delta_text, delta_color))
                
                # Note: Memorization is indicated by red box outline, not text
                
                # Render using ax.text positioned where title normally goes
                # Title is typically at y=-0.05 in axes coordinates
                title_y = -0.05
                full_text = " ".join([p[0] for p in parts])
                
                # Render each part separately, centered
                # Simple character-based width estimation for positioning
                char_width = 0.006  # Approximate character width in axes coordinates
                total_chars = sum(len(p[0]) + 1 for p in parts) - 1  # Total chars including spaces
                x_start = 0.5 - (total_chars * char_width) / 2
                
                current_x = x_start
                for part_text, part_color in parts:
                    part_chars = len(part_text)
                    part_center = current_x + (part_chars * char_width) / 2
                    ax.text(part_center, title_y, part_text, ha='center', va='top',
                           transform=ax.transAxes, fontsize=effective_title_fontsize, color=part_color)
                    current_x += part_chars * char_width + char_width  # Add space after
            else:
                # Use normal title rendering for explicit titles or when no property delta
                ax.set_title(titles[idx], fontsize=effective_title_fontsize)
        
        # Add row label on the left side of the first column
        if row_labels is not None and col == 0:
            ax.text(-0.1, 0.5, row_labels[row], 
                   ha='right', va='center', transform=ax.transAxes,
                   fontsize=row_label_fontsize, fontweight='bold')
        
        # Add column label at the top of the first row
        if col_labels is not None and row == 0:
            ax.text(0.5, 1.1, col_labels[col],
                   ha='center', va='bottom', transform=ax.transAxes,
                   fontsize=col_label_fontsize, fontweight='bold')
    
    # Turn off any remaining empty axes
    for j in range(len(mols), len(axes_list)):
        axes_list[j].axis('off')

    # Save if path provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, save_name)
        
        # Handle different file formats appropriately
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext in ['.png']:
            # For PNG, use facecolor='white' to avoid transparency issues
            save_fig_as_png(fig, full_path, dpi=300)
        else:
            plt.tight_layout()
            # For other formats (PDF, PNG), use default settings
            fig.savefig(full_path, bbox_inches='tight', dpi=300)
        
        print(f"Saved grid to: {full_path}")
    
    return fig

def generate_paired_smiles_grid(smiles_list1: List[str], smiles_list2: List[str], n_cols: int = 4,
                              titles: Optional[List[str]] = None, save_dir: Optional[str] = None,
                              save_name: Optional[str] = None, use_rdkit: bool = False,
                              property_deltas: Optional[List[Optional[float]]] = None,
                              memorization_flags_list1: Optional[List[Optional[bool]]] = None,
                              memorization_flags_list2: Optional[List[Optional[bool]]] = None,
                              property_name: Optional[str] = None) -> None:
    """
    Generate a grid visualization of paired SMILES strings in alternating rows.
    
    Args:
        smiles_list1: First list of SMILES strings
        smiles_list2: Second list of SMILES strings
        n_cols: Number of columns in the grid (default: 4)
        titles: Optional list of titles for each molecule
        save_dir: Directory to save the visualization (optional)
        save_name: Name of the saved file (optional)
        use_rdkit: If True, use RDKit for rendering (returns PIL.Image), else matplotlib (returns Figure)
        property_deltas: Optional list of property deltas to display in titles for list2
        memorization_flags_list1: Optional list of memorization flags for list1
        memorization_flags_list2: Optional list of memorization flags for list2
        property_name: Optional name of the property for delta display
        
    Returns:
        PIL.Image if use_rdkit=True, matplotlib Figure otherwise, or None if save_dir/save_name provided
    """
    if len(smiles_list1) != len(smiles_list2):
        raise ValueError("Both SMILES lists must have the same length")
        
    # Interleave the two lists based on n_cols
    combined_smiles = []
    combined_titles: Optional[List[str]] = [] if (titles is not None or property_deltas is not None or 
                                                   memorization_flags_list1 is not None or 
                                                   memorization_flags_list2 is not None) else None
    
    N = len(smiles_list1)
    for i in range(0, N, n_cols):
        # Add chunk from first list
        chunk1 = smiles_list1[i:i + n_cols]
        combined_smiles.extend(chunk1)
        # Add chunk from second list
        chunk2 = smiles_list2[i:i + n_cols]
        combined_smiles.extend(chunk2)
        
        # Generate titles if needed
        if combined_titles is not None:
            # Titles for list1 row
            for j in range(len(chunk1)):
                k = i + j
                t = "mol1"
                if memorization_flags_list1 is not None and k < len(memorization_flags_list1) and memorization_flags_list1[k]:
                    t += " MEM"
                combined_titles.append(t)
            # Titles for list2 row (with property deltas and memorization)
            for j in range(len(chunk2)):
                k = i + j
                t = "mol2"
                if property_deltas is not None and k < len(property_deltas) and property_deltas[k] is not None:
                    pname = property_name or "prop"
                    t += f" Δ{pname}={float(property_deltas[k]):.3f}"
                if memorization_flags_list2 is not None and k < len(memorization_flags_list2) and memorization_flags_list2[k]:
                    t += " MEM"
                combined_titles.append(t)
        
    # If explicit titles were provided, prefer them (must match combined length)
    if titles is not None:
        if len(titles) != len(combined_smiles):
            raise ValueError("titles length must match number of grid cells")
        combined_titles = titles
        
    # Dispatch to RDKit-based or Matplotlib-based generator
    if use_rdkit:
        return generate_smiles_grid_rdkit(
            combined_smiles,
            n_cols=n_cols,
            save_dir=save_dir,
            save_name=save_name if save_name is not None else 'paired_grid.png',
            titles=combined_titles,
        )
    else:
        return generate_smiles_grid(
            combined_smiles,
            n_cols=n_cols,
            titles=combined_titles,
            save_dir=save_dir,
            save_name=save_name if save_name is not None else 'paired_grid.pdf',
        )


def generate_ground_truth_grid(
    *,
    conditioning_smiles: List[str],
    generated_smiles: List[str],
    gt_smiles: List[str],
    n_cols: int = 4,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    titles: Optional[List[str]] = None,
    property_deltas: Optional[List[Optional[float]]] = None,
    memorization_flags: Optional[List[Optional[bool]]] = None,
    property_name: Optional[str] = None,
    use_rdkit: bool = False,
):
    """
    Visualize conditional generations with ground truth.

    Creates a grid with 3 rows per chunk:
      conditioning (seed), generated, ground truth

    - If use_rdkit=True: returns a PIL.Image (and optionally saves PNG)
    - Else: returns a matplotlib Figure (and optionally saves PDF)
    """
    if not (len(conditioning_smiles) == len(generated_smiles) == len(gt_smiles)):
        raise ValueError(
            "conditioning_smiles, generated_smiles, and gt_smiles must have the same length "
            f"({len(conditioning_smiles)}, {len(generated_smiles)}, {len(gt_smiles)})"
        )

    # Interleave in blocks of n_cols: seeds row, gens row, gts row
    combined_smiles: List[str] = []
    combined_titles: Optional[List[str]] = [] if (titles is not None or property_deltas is not None or memorization_flags is not None) else None

    N = len(conditioning_smiles)
    for i in range(0, N, n_cols):
        seed_chunk = conditioning_smiles[i : i + n_cols]
        gen_chunk = generated_smiles[i : i + n_cols]
        gt_chunk = gt_smiles[i : i + n_cols]

        combined_smiles.extend(seed_chunk)
        combined_smiles.extend(gen_chunk)
        combined_smiles.extend(gt_chunk)

        if combined_titles is not None:
            # Seeds row titles
            for j in range(len(seed_chunk)):
                combined_titles.append("seed")
            # Generated row titles (optionally show deltas/memorization)
            for j in range(len(gen_chunk)):
                k = i + j
                t = "gen"
                if property_deltas is not None and k < len(property_deltas) and property_deltas[k] is not None:
                    pname = property_name or "prop"
                    t += f" Δ{pname}={float(property_deltas[k]):.3f}"
                if memorization_flags is not None and k < len(memorization_flags) and memorization_flags[k]:
                    t += " MEM"
                combined_titles.append(t)
            # Ground truth row titles
            for j in range(len(gt_chunk)):
                combined_titles.append("gt")

    # If explicit titles were provided, prefer them (must match combined length)
    if titles is not None:
        if len(titles) != len(combined_smiles):
            raise ValueError("titles length must match number of grid cells")
        combined_titles = titles

    if use_rdkit:
        return generate_smiles_grid_rdkit(
            combined_smiles,
            n_cols=int(n_cols),
            save_dir=save_dir,
            save_name=save_name if save_name is not None else "ground_truth_grid.png",
            titles=combined_titles,
        )
    return generate_smiles_grid(
        combined_smiles,
        n_cols=int(n_cols),
        save_dir=save_dir,
        save_name=save_name if save_name is not None else "ground_truth_grid.pdf",
        titles=combined_titles,
        )
