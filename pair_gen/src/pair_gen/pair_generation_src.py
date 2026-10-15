#!/usr/bin/env python3
"""
Molecule Pair Generation Module for MoleculeNet and Zinc250K Datasets

This module provides functionality to generate parquet files containing molecule pairs
based on Tanimoto distance filtering. It supports MoleculeNet (BBBP/BACE/HIV) and Zinc
250k datasets and can generate pairs for training, validation, and test splits.

Key Features:
- Loads molecular datasets and computes Morgan fingerprints
- Computes Tanimoto distances between all molecule pairs
- Filters pairs based on percentile threshold or maximum pairs per molecule
- Computes shape Tanimoto distances for filtered pairs
- Generates parquet files with molecule pairs, distances, and shape information
- Supports streaming writes for large datasets
- Includes timeout handling for shape computation
- Caches preprocessed data for efficiency

Main Classes:
- PairGenerator: Main class for generating molecule pairs
- CustomArrowStreamWriter: Handles streaming writes to Arrow IPC format
- TimeoutError: Custom exception for timeout handling

Configuration:
The module uses Hydra configuration for all parameters including:
- Dataset selection (BBBP, BACE, HIV, Zinc 250k)
- Split configuration (train, val, test)
- Filtering parameters (percentile_threshold or max_pairs_per_mol)
- Output directory and file paths
- Timeout settings for shape computation
"""

import os
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import time
import signal
import random
from datetime import datetime, timedelta
import sys
from omegaconf import DictConfig, OmegaConf, ListConfig

# RDKit and molecule-utility imports
from atomic_datasets.utils.rdkit import is_molecule_sane
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import selfies as sf
import pandas as pd

# Import our utility functions
from pair_gen.utils_mol import get_shape_tanimoto

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Only show the message, no timestamp/level/logger name
    stream=sys.stdout,  # Explicitly set to stdout
    force=True          # Overwrites any previous logging config
)
logger = logging.getLogger()

ZINC_DIR = os.environ.get("PAIRIT_ZINC250K_DIR", "<unset:PAIRIT_ZINC250K_DIR>")
MOLECULENET_SDF_DIR = os.environ.get("PAIRIT_MOLECULENET_SDF_DIR", "<unset:PAIRIT_MOLECULENET_SDF_DIR>")
ZINC_TRAIN_FILE = os.path.join(ZINC_DIR, "zinc250k_train.csv")
ZINC_VAL_FILE = os.path.join(ZINC_DIR, "zinc250k_valid.csv")

# Dataset property names - centralized definitions
# These constants define the molecular properties available for each dataset type
# and are used throughout the codebase for consistency
MOLECULENET_PROPERTY_NAMES = ['SA', 'SC']
ZINC250K_PROPERTY_NAMES = ['qed', 'sa', 'parp1', 'fa7', '5h1tb', 'braf', 'jak2', 'logp']


class TimeoutError(Exception):
    """
    Custom timeout exception raised when operations exceed the specified time limit.
    
    This exception is used to handle cases where shape Tanimoto computation
    takes too long and needs to be terminated gracefully.
    
    Attributes:
        message: Description of the timeout error
    """
    pass


def timeout_after(seconds: float):
    """
    Context manager and decorator for timeout functionality using signal.SIGALRM.
    
    This function provides two ways to implement timeouts:
    1. As a context manager for use with 'with' statements
    2. As a decorator for functions
    
    The timeout mechanism uses Unix signal.SIGALRM to interrupt long-running
    operations, particularly useful for shape Tanimoto computations that can
    be computationally expensive.
    
    Args:
        seconds: Timeout duration in seconds
        
    Returns:
        TimeoutContext: Context manager for timeout functionality
        
    Example:
        # As context manager:
        with timeout_after(10.0):
            result = expensive_computation()
            
        # As decorator:
        @timeout_after(10.0)
        def expensive_function():
            return some_computation()
    
    Raises:
        TimeoutError: When the operation exceeds the specified time limit
        
    Note:
        This implementation is Unix-specific and uses signal.SIGALRM.
        The timeout handler is automatically restored after use.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    
    class TimeoutContext:
        def __enter__(self):
            # Set the signal handler
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore the old handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)
            if exc_type == TimeoutError:
                return True  # Suppress the exception
            return False
    
    return TimeoutContext()


class CustomArrowStreamWriter:
    """
    Custom Arrow IPC stream writer for molecule pairs with specific schema.
    
    This class handles streaming writes of molecule pair data to Arrow IPC format,
    using record batches for true streaming writes and reads. It's designed to
    efficiently handle large datasets without loading everything into memory.
    
    The writer expects data in a specific format containing molecule indices,
    distance metrics, and shape matching information.
    
    Attributes:
        path: File path where the Arrow IPC stream will be written
        schema: PyArrow schema defining the data structure
        row_count: Total number of rows written to the stream
        writer: PyArrow IPC stream writer instance
        
    Schema Fields:
        - molecule_a_idx: Index of the first molecule in the pair (int32)
        - molecule_b_idx: Index of the second molecule in the pair (int32)
        - split: Dataset split name (string)
        - morgan_tanimoto_distance: Morgan fingerprint Tanimoto distance (float32)
        - shape_tanimoto_distance: Shape Tanimoto distance (float32, optional)
        - matchA: Atom matching indices for molecule A (list of int32, optional)
        - matchB: Atom matching indices for molecule B (list of int32, optional)
        - rmsd: Root mean square deviation (float32, optional)
        - matching_time: Time taken for shape matching (float32, optional)
        
        Note: Optional fields are only included in the schema when the corresponding
        distance computation is enabled in the configuration.
    """
    
    def __init__(self, path: str, schema: pa.Schema):
        """
        Initialize the Arrow IPC stream writer.
        
        Args:
            path: File path where the Arrow IPC stream will be written
            schema: PyArrow schema defining the data structure
            
        Note:
            Creates the parent directory if it doesn't exist.
            Initializes the PyArrow IPC stream writer for record batches.
        """
        self.path = path
        self.schema = schema
        self.row_count = 0
        
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create Arrow IPC stream writer for record batches
        self.writer = pa.ipc.new_stream(path, schema)

    def write_chunk(self, chunk: List[Tuple]):
        """
        Write a chunk of molecule pair data to the Arrow IPC stream.
        
        Args:
            chunk: List of tuples containing molecule pair data. Each tuple should contain:
                - molecule_a_idx: Index of first molecule (int)
                - molecule_b_idx: Index of second molecule (int)
                - split: Dataset split name (str)
                - morgan_tanimoto_distance: Morgan fingerprint distance (float)
                - shape_tanimoto_distance: Shape distance (float, optional)
                - matchA: Atom matching for molecule A (list of int, optional)
                - matchB: Atom matching for molecule B (list of int, optional)
                - rmsd: Root mean square deviation (float, optional)
                - matching_time: Shape matching time (float, optional)
                
        Note:
            The chunk data is converted to a PyArrow record batch and written
            to the stream. The row_count is updated accordingly.
            The presence of optional fields depends on the schema configuration.
        """
        batch_data = {
            "molecule_a_idx": [r[0] for r in chunk],
            "molecule_b_idx": [r[1] for r in chunk],
            "split": [r[2] for r in chunk],
            "morgan_tanimoto_distance": [r[3] for r in chunk],
        }
        
        # Add shape_tanimoto fields if they exist in the chunk data
        if len(chunk[0]) > 4:  # Check if shape_tanimoto fields are present
            batch_data["shape_tanimoto_distance"] = [r[4] for r in chunk]
            batch_data["matchA"] = [r[5] for r in chunk]
            batch_data["matchB"] = [r[6] for r in chunk]
            
            # Add extra metrics if they exist in the chunk data
            if len(chunk[0]) > 7:  # Check if extra metrics are present
                batch_data["rmsd"] = [r[7] for r in chunk]
                batch_data["matching_time"] = [r[8] for r in chunk]
        
        # Create record batch (not table) and write to stream
        batch = pa.record_batch(batch_data, schema=self.schema)
        self.writer.write_batch(batch)  # Use write_batch, not write
        self.row_count += batch.num_rows

    def close(self):
        """
        Close the Arrow IPC stream writer and log final statistics.
        
        This method finalizes the stream and logs the total number of rows written.
        If an error occurs during closing, it attempts to clean up any corrupted files.
        
        Raises:
            Exception: If there's an error during stream closing
            
        Note:
            The method includes error handling to remove corrupted files
            if the stream cannot be properly closed.
        """
        try:
            self.writer.close()
            logger.info(f"Finalized {self.row_count} rows written to {self.path}")
        except Exception as e:
            logger.error(f"Error closing stream writer: {e}")
            # Try to clean up the corrupted file
            if Path(self.path).exists():
                try:
                    Path(self.path).unlink()
                    logger.info(f"Removed corrupted file: {self.path}")
                except Exception as cleanup_error:
                    logger.error(f"Could not remove corrupted file: {cleanup_error}")
            raise


class PairGenerator:
    """
    Generates parquet files for molecule pairs based on Tanimoto distance filtering.

    This class is the main interface for generating molecule pair datasets from
    molecular datasets like MoleculeNet and Zinc250K. It supports multiple filtering
    strategies and can handle large datasets efficiently through streaming writes.

    The class processes molecular datasets by:
    1. Loading and preprocessing molecular data (SMILES, SELFIES, fingerprints)
    2. Computing Tanimoto distances between all molecule pairs
    3. Filtering pairs based on distance thresholds or pair limits
    4. Computing shape Tanimoto distances for filtered pairs
    5. Generating parquet files with the pair data

    Key Features:
    - Supports MoleculeNet datasets (BBBP, BACE, HIV) and Zinc 250k
    - Configurable filtering (percentile-based or max pairs per molecule)
    - Streaming writes for memory efficiency
    - Caching of preprocessed data
    - Timeout handling for shape computations
    - Support for multiple dataset splits

    Attributes:
        config: Hydra configuration object containing all parameters
        data_dir: Path to the data directory
        output_dir: Path to the output directory
        dataset_name: Name of the dataset (bbbp, bace, hiv, zinc250k)
        percentile_threshold: Percentage threshold for pair filtering
        max_molecules: Maximum number of molecules to process per split
        max_pairs_per_mol: Maximum pairs per molecule (alternative to percentile)
        test_mode: Whether to run in test mode with reduced dataset sizes
        splits: List of dataset splits to process (train, val, test)
        split_data: Dictionary containing processed data for each split
        molecules_file: Path to the molecules parquet file
        pairs_file: Path to the pairs Arrow IPC file
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the pair generator with configuration.
        
        This method sets up the PairGenerator with all necessary configuration
        parameters, validates the configuration, creates output directories,
        and initializes data structures for processing.
        
        Args:
            config: Hydra configuration object containing all parameters
                Required config fields:
                - data_dir: Path to the data directory
                - output_dir: Path to the output directory  
                - dataset_name: Name of the dataset (bbbp, bace, hiv, zinc250k)
                - percentile_threshold: Percentage for pair filtering (0-100)
                - splits: List of splits to process (train, val, test)
                
                Optional config fields:
                - max_molecules: Maximum molecules per split
                - max_pairs_per_mol: Alternative to percentile_threshold
                - test_mode: Run with reduced dataset sizes
                - test_sizes: Dataset sizes for test mode (train, val, test)
                - overwrite_molecules_parquet: Force regeneration of molecules file
                - overwrite_pairs_parquet: Force regeneration of pairs file
                - overwrite_ds: Force regeneration of cached dataset data
                - train_only: Process only training split
                - recompute_dataset: Force recomputation of dataset
                - use_second_half_of_train: Use only second half of training data
                - log_extra_metrics: Include RMSD and matching time in output
                - save_as_parquet: Convert Arrow IPC to Parquet format
                - shape_tanimoto_timeout: Timeout for shape computations (seconds)
                - write_every: Chunk size for streaming writes
                
        Raises:
            ValueError: If configuration is invalid or missing required fields
            FileNotFoundError: If data directory doesn't exist
            
        Note:
            The configuration is validated and saved to the output directory.
            Output directories are created if they don't exist.
            File paths are set up for molecules.parquet and pairs.arrow files.
        """
        self.config = config
        
        # Log the configuration
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
        
        # Save configuration to output directory
        
        # Extract configuration parameters
        self.cfg = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.dataset_name = config.dataset_name
        self.percentile_threshold = config.percentile_threshold
        self.max_molecules = config.get('max_molecules', None)
        self.max_pairs_per_mol = config.get('max_pairs_per_mol', None)
        self.test_mode = config.get('test_mode', False)
        self.test_sizes = config.get('test_sizes', (500, 100, 100))
        self.overwrite_molecules_parquet = config.get('overwrite_molecules_parquet', False)
        self.overwrite_pairs_parquet = config.get('overwrite_pairs_parquet', False)
        self.overwrite_ds = config.get('overwrite_ds', False)
        self.train_only = config.get('train_only', False)
        self.recompute_dataset = config.get('recompute_dataset', False)
        self.use_second_half_of_train = config.get('use_second_half_of_train', False)
        self.log_extra_metrics = config.get('log_extra_metrics', False)  # New config option for logging RMSD and matching time
        self.save_as_parquet = config.get('save_as_parquet', True)  # Whether to convert Arrow IPC to Parquet at the end            

                # Extract splits configuration with validation
        self.splits = config.get('splits', ["train", "val", "test"])
        self._validate_splits()
        
        # Validate pair generation parameters
        self._validate_pair_generation_params()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_file = os.path.join(self.output_dir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(OmegaConf.to_yaml(config))
        logger.info(f"Configuration record saved to: {config_file}")
        
        # File paths
        self.molecules_file = self.output_dir / "molecules.parquet"
        self.pairs_file = self.output_dir / "pairs.arrow"  # Use Arrow IPC format for streaming reads
        (Path(self.data_dir) / "cache").mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.split_data = {}
        
        # Statistics tracking
        self.selfies_failures = 0
        self.insane_molecules = 0
        self.valency_issues = 0
        self.total_molecules_processed = 0
        
    def load_dataset(self):
        """
        Load dataset based on configuration.
        
        This method dispatches to the appropriate dataset loading method based on
        the dataset_name configuration parameter. Currently supports MoleculeNet
        datasets (BBBP, BACE, HIV) and Zinc 250k.

        The loaded data is stored in self.split_data dictionary, with each split
        containing processed molecular data including SMILES, SELFIES, fingerprints,
        and molecular properties.

        Raises:
            ValueError: If the dataset_name is not supported

        Note:
            This method calls the appropriate dataset loading method based on
            the dataset_name configuration.
        """
        if self.dataset_name.lower() in ["bbbp", "bace", "hiv"]:
            self.load_molecule_net_dataset()
        elif self.dataset_name.lower() == "zinc250k":
            self.load_zinc_250k_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Available datasets: bbbp, bace, hiv, zinc250k")
        
    def load_zinc_250k_dataset(self):
        """
        Load Zinc 250k dataset for each split and compute fingerprints, with caching.
        
        This method loads the Zinc 250k dataset from CSV files, processes the molecular data,
        and caches the results for efficiency. For each molecule, it extracts SMILES,
        computes Morgan fingerprints, generates SELFIES representations, and extracts
        Zinc 250k molecular properties.
        
        The processed data is stored in self.split_data with the following structure:
        - smiles: List of SMILES strings
        - selfies: List of SELFIES representations (None for failed encodings)
        - fingerprints: List of Morgan fingerprints (radius=2, size=2048)
        - molecules: List of RDKit molecule objects
        - dataset: Original dataset object (DataFrame)
        - qed, sa, parp1, fa7, 5h1tb, braf, jak2, logp: Zinc 250k properties (defined in ZINC250K_PROPERTY_NAMES)
        - invalid_indices: Set of indices where SELFIES encoding failed
        - selfies_failures: Count of SELFIES encoding failures
        
        Uses self attributes:
        - data_dir: Base directory for data files
        - dataset_splits: List of dataset splits to process
        - test_mode: Whether to use reduced dataset sizes
        - test_sizes: Dataset sizes for test mode
        - max_molecules: Maximum molecules to process per split
        - overwrite_ds: Whether to force regeneration of cached data
        - use_second_half_of_train: Whether to use only second half of training data
        
        Note:
            Zinc 250k dataset files are expected to be located at:
            - Train: $PAIRIT_ZINC250K_DIR/zinc250k_train.csv
            - Validation: $PAIRIT_ZINC250K_DIR/zinc250k_valid.csv
            Cached data is stored in {data_dir}/cache/zinc250k_ds_caches/
            Properties are extracted from CSV columns and NaN values are replaced with 0.0.
        """
        self.split_data = {}
        ds_cache_dir = Path(self.data_dir) / "cache" / "zinc250k_ds_caches"
        ds_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the CSV files
        train_df = pd.read_csv(ZINC_TRAIN_FILE) #, sep=" ", header=None)
        val_df = pd.read_csv(ZINC_VAL_FILE) #, sep=" ", header=None)
        
        # Set column names based on Zinc 250k format
        column_names = ['mol_id', 'smiles', 'qed', 'sa', 'parp1', 'fa7', '5h1tb', 'braf', 'jak2', 'logp']
        # train_df.columns = column_names
        # val_df.columns = column_names
        
        # Create a mapping of split names to dataframes
        split_dataframes = {
            'train': train_df,
            'val': val_df
        }
        
        for idx, split in enumerate(self.dataset_splits):
            logger.info(f"Loading Zinc 250k split '{split}' from {self.data_dir}")
            
            # For Zinc 250k, we only have train and val splits
            if split == 'test':
                logger.warning(f"Zinc 250k dataset does not have a test split. Skipping '{split}'.")
                continue
                
            cache_path = ds_cache_dir / f"zinc250k_preprocessed_{split}.pkl"
            
            if cache_path.exists() and not self.overwrite_ds:
                logger.info(f"Loading split '{split}' data from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                split_dict = cache_data['split_data']
                
                # Take random subset if in test mode
                if self.test_mode:
                    n = self.test_sizes[idx]
                    indices = list(range(len(split_dict['smiles'])))
                    random_indices = random.sample(indices, n)
                    for key in split_dict:
                        if isinstance(split_dict[key], list):
                            split_dict[key] = [split_dict[key][i] for i in random_indices]
                
                # Compute invalid_indices and selfies_failures from loaded data
                invalid_indices = set(i for i, selfies in enumerate(split_dict['selfies']) if selfies is None)
                selfies_failures = sum(1 for selfies in split_dict['selfies'] if selfies is None)
            else:
                # Get the dataframe for this split
                df = split_dataframes[split]
                
                # Apply size limits
                if self.test_mode:
                    n = self.test_sizes[idx]
                    df = df.head(n)
                elif self.max_molecules is not None:
                    df = df.head(self.max_molecules)
                    logger.info(f'Maxing out molecules on split {split}')
                
                split_dict = {
                    'smiles': [],
                    'selfies': [],
                    'fingerprints': [],
                    'molecules': [],
                    'dataset': df
                }
                
                # Initialize property lists
                property_names = ZINC250K_PROPERTY_NAMES
                for prop_name in property_names:
                    split_dict[prop_name] = []
                
                invalid_indices = set()
                selfies_failures = 0
                
                for i, row in enumerate(df.itertuples()):
                    smiles = row.smiles
                    
                    # Convert SMILES to RDKit molecule
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                        continue
                    
                    # Compute fingerprint
                    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
                    
                    # Generate SELFIES
                    try:
                        selfies = sf.encoder(smiles)
                    except Exception as e:
                        selfies = None
                        selfies_failures += 1
                        invalid_indices.add(i)
                    
                    # Extract properties
                    for prop_name in property_names:
                        value = getattr(row, prop_name, 0.0)
                        if pd.isna(value):
                            value = 0.0
                        split_dict[prop_name].append(float(value))
                    
                    split_dict['smiles'].append(smiles)
                    split_dict['selfies'].append(selfies)
                    split_dict['fingerprints'].append(fingerprint)
                    split_dict['molecules'].append(mol)
                
                # Save to cache
                cache_data = {'split_data': split_dict}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved split '{split}' data to {cache_path}")
            
            # Add computed fields to split_dict
            split_dict['invalid_indices'] = invalid_indices
            split_dict['selfies_failures'] = selfies_failures
            
            self.split_data[split] = split_dict
            logger.info(f"Split '{split}': {len(split_dict['smiles'])} molecules, {selfies_failures} SELFIES failures, {len(invalid_indices)} invalid.")
        
        if self.use_second_half_of_train:
            np.random.seed(42)
            len_train_dset = len(self.split_data['train']['dataset'])
            fixed_perm = np.random.permutation(len_train_dset)
            self.second_half_train_indices = fixed_perm[len_train_dset//2:]
        
    def load_molecule_net_dataset(self):
        """
        Load MoleculeNet dataset for each split and compute fingerprints, with caching.
        
        This method loads MoleculeNet datasets (BBBP, BACE, HIV) from SDF files,
        processes the molecular data, and caches the results for efficiency.
        For each molecule, it extracts SMILES, computes Morgan fingerprints,
        generates SELFIES representations, and extracts molecular properties.
        
        The processed data is stored in self.split_data with the following structure:
        - smiles: List of SMILES strings
        - selfies: List of SELFIES representations (None for failed encodings)
        - fingerprints: List of Morgan fingerprints (radius=2, size=2048)
        - molecules: List of RDKit molecule objects
        - dataset: Original dataset object
        - SA: List of SA (Synthetic Accessibility) scores
        - SC: List of SC (Synthetic Complexity) scores (defined in MOLECULENET_PROPERTY_NAMES)
        - invalid_indices: Set of indices where SELFIES encoding failed
        - selfies_failures: Count of SELFIES encoding failures
        
        Uses self attributes:
        - data_dir: Base directory for data files
        - dataset_splits: List of dataset splits to process
        - test_mode: Whether to use reduced dataset sizes
        - test_sizes: Dataset sizes for test mode
        - max_molecules: Maximum molecules to process per split
        - overwrite_ds: Whether to force regeneration of cached data
        
        Note:
            SDF files are expected to be located at:
            $PAIRIT_MOLECULENET_SDF_DIR/{dataset_name}_b_{split}_best1.sdf
            Cached data is stored in {data_dir}/cache/moleculenet_ds_caches/
        """
        self.split_data = {}
        ds_cache_dir = Path(self.data_dir) / "cache" / "moleculenet_ds_caches"
        ds_cache_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, split in enumerate(self.dataset_splits):
            logger.info(f"Loading MoleculeNet split '{split}' from {self.data_dir}")
            sdf_path = os.path.join(MOLECULENET_SDF_DIR, f'{self.dataset_name}_b_{split}_best1.sdf')
            supplier = Chem.SDMolSupplier(sdf_path)
            # Convert to list (filter out invalid molecules)
            molecules = [mol for mol in supplier if mol is not None]
            
            cache_path = ds_cache_dir / f"{self.dataset_name}_preprocessed_{split}.pkl"
            
            if cache_path.exists() and not self.overwrite_ds:
                logger.info(f"Loading split '{split}' data from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                split_dict = cache_data['split_data']
                
                # Take random subset if in test mode
                if self.test_mode:
                    n = self.test_sizes[idx]
                    indices = list(range(len(split_dict['smiles'])))
                    random_indices = random.sample(indices, n)
                    for key in split_dict:
                        if isinstance(split_dict[key], list):
                            split_dict[key] = [split_dict[key][i] for i in random_indices]
                
                # Compute invalid_indices and selfies_failures from loaded data
                invalid_indices = set(i for i, selfies in enumerate(split_dict['selfies']) if selfies is None)
                selfies_failures = sum(1 for selfies in split_dict['selfies'] if selfies is None)
            else:
                if self.test_mode:
                    n = self.test_sizes[idx]
                    molecules = molecules[:n]
                elif self.max_molecules is not None:
                    molecules = molecules[:self.max_molecules]
                    print('maxing out molecules on split', split)
                
                split_dict = {
                    'smiles': [],
                    'selfies': [],
                    'fingerprints': [],
                    'molecules': [],
                    'dataset': molecules
                }

                # Initialize property lists
                property_names = MOLECULENET_PROPERTY_NAMES
                for prop_name in property_names:
                    split_dict[prop_name] = []

                invalid_indices = set()
                selfies_failures = 0
                for i, mol in enumerate(molecules):
                    smiles = Chem.MolToSmiles(mol)
                    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
                    try:
                        selfies = sf.encoder(smiles)
                    except Exception as e:
                        selfies = None
                        selfies_failures += 1
                        invalid_indices.add(i)

                    split_dict['smiles'].append(smiles)
                    split_dict['selfies'].append(selfies)
                    split_dict['fingerprints'].append(fingerprint)
                    split_dict['molecules'].append(mol)
                    
                    # Get SA and SC properties
                    split_dict['SA'].append(float(mol.GetProp('SA')))
                    split_dict['SC'].append(float(mol.GetProp('SC')))

                # Save to cache
                cache_data = {'split_data': split_dict}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved split '{split}' data to {cache_path}")

            # Add computed fields to split_dict
            split_dict['invalid_indices'] = invalid_indices
            split_dict['selfies_failures'] = selfies_failures
            
            self.split_data[split] = split_dict
            logger.info(f"Split '{split}': {len(split_dict['smiles'])} molecules, {selfies_failures} SELFIES failures, {len(invalid_indices)} invalid.")
        
        if self.use_second_half_of_train:
            np.random.seed(42)
            len_train_dset = len(self.split_data['train']['dataset'])
            fixed_perm = np.random.permutation(len_train_dset)
            self.second_half_train_indices = fixed_perm[len_train_dset//2:]
        
    def _validate_splits(self):
        """
        Validate the splits configuration and create dataset split mapping.
        
        This method validates that the splits configuration is properly formatted
        and contains valid split names. It creates a mapping between user-friendly
        split names and the actual dataset split names used by the underlying
        dataset libraries.
        
        The method updates self.splits and self.dataset_splits attributes with
        the validated configuration.
        
        Uses self attributes:
        - splits: List of user-specified splits to validate
            
        Raises:
            ValueError: If splits configuration is invalid or contains unsupported split names
            
        Note:
            Currently supports train, val, and test splits.
            The method converts ListConfig to regular list for easier handling.
        """
        # Map user-friendly split names to dataset split names
        split_mapping = {
            "train": "train",
            "val": "val",  
            "test": "test"
        }
        
        
        
        if not (isinstance(self.splits, list) or isinstance(self.splits, ListConfig)):
            raise ValueError(f"splits must be a list, got {type(self.splits)}")
        
        # Convert to regular list if it's a ListConfig
        splits_list = list(self.splits) if isinstance(self.splits, ListConfig) else self.splits
        
        for split in splits_list:
            if split not in split_mapping:
                raise ValueError(f"Invalid split '{split}'. Valid splits are: {list(split_mapping.keys())}")
        
        if not splits_list:
            raise ValueError("splits list cannot be empty")
        
        print("splits_list", splits_list)
        
        # Update self.splits to be a regular list and create dataset splits mapping
        self.splits = splits_list
        self.dataset_splits = [split_mapping[split] for split in splits_list]
        
        logger.info(f"Using splits: {self.splits} (maps to dataset splits: {self.dataset_splits})")
        
    def _format_time_estimate(self, seconds: float) -> str:
        """
        Format time estimate in human readable format.
        
        Converts seconds to the most appropriate time unit (seconds, minutes, hours, days)
        and returns a formatted string with the value and unit.
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            str: Formatted time string with appropriate unit
            
        Example:
            >>> _format_time_estimate(65.5)
            "1.1 minutes"
            >>> _format_time_estimate(3600)
            "1.0 hours"
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"
        
    def _validate_pair_generation_params(self):
        """
        Validate that only one of percentile_threshold or max_pairs_per_mol is specified.
            Uses self attributes:
            - percentile_threshold: Percentage threshold for pair filtering
            - max_pairs_per_mol: Maximum pairs per molecule
                
            Raises:
                ValueError: If both or neither parameter is specified, or if values are invalid
                
            Note:
                The method logs the validated configuration for user confirmation.
                percentile_threshold must be between 0 and 100.
                max_pairs_per_mol must be positive.
        """
        percentile_specified = self.percentile_threshold is not None
        max_pairs_specified = self.max_pairs_per_mol is not None
        
        if percentile_specified and max_pairs_specified:
            raise ValueError(
                f"Both percentile_threshold ({self.percentile_threshold}) and max_pairs_per_mol ({self.max_pairs_per_mol}) "
                f"are specified. Please specify only one of them."
            )
        
        if not percentile_specified and not max_pairs_specified:
            raise ValueError(
                "Neither percentile_threshold nor max_pairs_per_mol is specified. "
                "Please specify one of them."
            )
        
        if percentile_specified:
            if self.percentile_threshold <= 0 or self.percentile_threshold > 100:
                raise ValueError(f"percentile_threshold must be between 0 and 100, got {self.percentile_threshold}")
            logger.info(f"Using percentile_threshold: {self.percentile_threshold}%")
        
        if max_pairs_specified:
            if self.max_pairs_per_mol <= 0:
                raise ValueError(f"max_pairs_per_mol must be positive, got {self.max_pairs_per_mol}")
            logger.info(f"Using max_pairs_per_mol: {self.max_pairs_per_mol}")

    def compute_distances_and_pairs(self):
        """
        Compute distances and generate pairs for all specified splits.
        
        This is the core method that generates molecule pairs based on distance filtering.
        The filtering method is determined by the 'filter_by' configuration parameter,
        and additional distances are computed based on the 'also_compute' configuration.
        
        For each molecule in each split, it computes the primary distance to all other
        molecules in the same split, filters to the closest pairs, and optionally computes
        additional distances for those pairs.
        
        The method uses streaming writes to handle large datasets efficiently and includes
        timeout handling for shape computations. It processes data in chunks and provides
        periodic status updates.
        
        Output Data Structure:
        Each pair entry contains:
        - molecule_a_idx: Global index of first molecule (int32)
        - molecule_b_idx: Global index of second molecule (int32)
        - split: Dataset split name (string)
        - morgan_tanimoto_distance: Morgan fingerprint Tanimoto distance (float32)
        - shape_tanimoto_distance: Shape Tanimoto distance (float32, only if 'shape_tanimoto' in also_compute)
        - matchA: Atom matching indices for molecule A (list of int32, only if shape_tanimoto is computed)
        - matchB: Atom matching indices for molecule B (list of int32, only if shape_tanimoto is computed)
        - rmsd: Root mean square deviation (float32, optional if log_extra_metrics=True and shape_tanimoto is computed)
        - matching_time: Time taken for shape matching (float32, optional if log_extra_metrics=True and shape_tanimoto is computed)
        
        Uses self attributes:
        - percentile_threshold: Percentage threshold for pair filtering
        - max_pairs_per_mol: Maximum pairs per molecule (alternative to percentile)
        - splits: List of user-specified splits to process
        - dataset_splits: List of dataset split names
        - split_data: Dictionary containing processed molecular data
        - split_local_indices: Dictionary mapping split names to valid molecule indices
        - split_local_indices_to_global_idx: Dictionary mapping original indices to global item_ids
        - log_extra_metrics: Whether to include RMSD and matching time
        - config: Configuration object containing filter_by, also_compute, timeout and chunk settings
        - pairs_file: Output file path for the pairs Arrow IPC file
        - use_second_half_of_train: Whether to use only second half of training data
        
        Creates:
        - Arrow IPC file at self.pairs_file containing all molecule pairs
        
        Raises:
            ValueError: If valid indices are not found (generate_molecules_parquet must be run first)
            ValueError: If filter_by configuration is not supported
            TimeoutError: If shape computation times out (handled gracefully)
            Exception: If there's an error during pair generation
            
        Note:
            The method uses streaming writes with configurable chunk sizes (write_every).
            Shape computations have configurable timeouts (shape_tanimoto_timeout).
            Status updates are printed every 60 seconds during processing.
            The method handles timeouts gracefully by setting shape distance to -1.0.
            Processing is done within splits only (no cross-split pairs).
            Additional distances are only computed if specified in the also_compute configuration.
        """
        
        # Check filter_by configuration
        filter_by = self.config.get('filter_by', 'morgan_tanimoto')
        if filter_by != 'morgan_tanimoto':
            raise ValueError(f"Unsupported filter_by: {filter_by}. Currently only 'morgan_tanimoto' is supported.")
        
        # Check also_compute configuration
        also_compute = self.config.get('also_compute', [])
        compute_shape_tanimoto = 'shape_tanimoto' in also_compute
        
        # Determine the method and log it
        if self.percentile_threshold is not None:
            logger.info(f"Computing {filter_by} distances and filtering to {self.percentile_threshold}% closest pairs for splits: {self.splits}")
        else:
            logger.info(f"Computing {filter_by} distances and keeping {self.max_pairs_per_mol} closest pairs per molecule for splits: {self.splits}")
        
        if compute_shape_tanimoto:
            logger.info(f"Also computing shape_tanimoto distances")
        
        # Define schema for the pairs parquet file (including split column)
        schema_fields = [
            ("molecule_a_idx", pa.int32()),
            ("molecule_b_idx", pa.int32()),
            ("split", pa.string()),  # Add split column
            ("morgan_tanimoto_distance", pa.float32()),
        ]
        
        # Add shape_tanimoto fields if enabled
        if compute_shape_tanimoto:
            schema_fields.extend([
                ("shape_tanimoto_distance", pa.float32()),
                ("matchA", pa.list_(pa.int32())),
                ("matchB", pa.list_(pa.int32())),
            ])
        
        # Add extra metrics fields if enabled and shape_tanimoto is computed
        if self.log_extra_metrics and compute_shape_tanimoto:
            schema_fields.extend([
                ("rmsd", pa.float32()),
                ("matching_time", pa.float32()),
            ])
        
        schema = pa.schema(schema_fields)
        
        # Initialize stream writer
        stream_writer = None
        try:
            stream_writer = CustomArrowStreamWriter(str(self.pairs_file), schema)
            chunk_results = []
            write_every = self.cfg.get('write_every', 10000)
            
            # Time tracking variables
            start_time = time.time()
            last_status_time = start_time
            status_interval = 60
            total_pairs_processed = 0
            
            # Process each split
            for user_split, dataset_split in zip(self.splits, self.dataset_splits):
                logger.info(f"Processing split: {user_split} (dataset: {dataset_split})")
                
                # Get data for this split
                split_data = self.split_data[dataset_split]
                smiles_list = split_data['smiles']
                fingerprints = split_data['fingerprints']
                molecules = split_data['molecules']
                
                # Use the valid indices that were computed in generate_molecules_parquet
                if not hasattr(self, 'split_local_indices'):
                    raise ValueError("Valid indices not found. Run generate_molecules_parquet() first.")
                
                valid_indices = self.split_local_indices[dataset_split]
                
                # Apply second half of train filtering if enabled
                if self.use_second_half_of_train and dataset_split == "train":
                    if hasattr(self, 'second_half_train_indices'):
                        # Filter valid_indices to only include molecules in the second half
                        valid_indices = [idx for idx in valid_indices if idx in self.second_half_train_indices]
                        logger.info(f"Using second half of train split: {len(valid_indices)} molecules")
                    else:
                        logger.warning("second_half_train_indices not found, using all valid molecules")
                
                total_molecules = len(valid_indices)
                
                # Calculate pairs per molecule based on the specified method
                if self.percentile_threshold is not None:
                    # Use percentile method
                    percentile_decimal = self.percentile_threshold / 100.0
                    pairs_per_molecule = max(1, int((total_molecules - 1) * percentile_decimal))
                else:
                    # Use max_pairs_per_mol method
                    pairs_per_molecule = min(self.max_pairs_per_mol, total_molecules - 1)
                
                total_expected_pairs = total_molecules * pairs_per_molecule
                
                logger.info(f"Split '{user_split}': {total_expected_pairs} expected pairs, {pairs_per_molecule} pairs per molecule")
                logger.info(f"Processing {len(valid_indices)} valid molecules in split '{dataset_split}'")
                
                # Process each molecule in this split
                for idx, original_idx in enumerate(tqdm(valid_indices, desc=f"Processing molecule pairs in split '{user_split}'")):
                    mol_a = molecules[original_idx]
                    fp_a = fingerprints[original_idx]
                    
                    # Create list of other valid molecules for comparison (same split only)
                    other_fps = []
                    other_indices = []
                    other_mols = []
                    
                    for other_original_idx in valid_indices:
                        if other_original_idx != original_idx:  # Skip self-comparison
                            other_fps.append(fingerprints[other_original_idx])
                            other_indices.append(other_original_idx)
                            other_mols.append(molecules[other_original_idx])
                    
                    # Compute Tanimoto similarities in bulk
                    tanimoto_similarities = DataStructs.BulkTanimotoSimilarity(fp_a, other_fps)
                    
                    # Convert to distances and create distance list
                    distances = []
                    for j_idx, similarity in enumerate(tanimoto_similarities):
                        tanimoto_distance = 1.0 - similarity
                        distances.append((other_indices[j_idx], tanimoto_distance, other_mols[j_idx]))
                    
                    # Sort by distance (closest first)
                    distances.sort(key=lambda x: x[1])
                    
                    # Keep only the closest pairs based on the specified method
                    closest_pairs = distances[:pairs_per_molecule]
                    
                    # Process the closest pairs
                    for other_original_idx, morgan_distance, mol_b in closest_pairs:
                        # Use the global indices from molecules parquet
                        molecule_a_idx = self.split_local_indices_to_global_idx[dataset_split][original_idx]
                        molecule_b_idx = self.split_local_indices_to_global_idx[dataset_split][other_original_idx]
                        
                        # Create base pair entry with split information
                        pair_entry = (
                            molecule_a_idx,
                            molecule_b_idx,
                            user_split,  # Add split column
                            float(morgan_distance),
                        )
                        
                        # Compute shape Tanimoto distance if enabled
                        if compute_shape_tanimoto:
                            try:
                                timeout = self.config.get('shape_tanimoto_timeout', 10.0)
                                with timeout_after(timeout):
                                    shape_distance, matchA, matchB, rmsd, matching_time = get_shape_tanimoto(mol_a, mol_b, return_extra=True)
                            except TimeoutError:
                                logger.warning(f"Shape Tanimoto computation timed out after {timeout}s")
                                shape_distance, matchA, matchB, rmsd, matching_time = -1.0, [], [], -1.0, -1.0
                            
                            # Add shape_tanimoto fields
                            pair_entry += (float(shape_distance), matchA, matchB)
                            
                            # Add extra metrics if enabled
                            if self.log_extra_metrics:
                                pair_entry += (float(rmsd), float(matching_time))
                        
                        chunk_results.append(pair_entry)
                    
                    # Update total pairs processed
                    total_pairs_processed += len(closest_pairs)
                    
                    # Write chunks periodically
                    if len(chunk_results) >= write_every:
                        stream_writer.write_chunk(chunk_results)
                        chunk_results = []
                    
                    # Print status periodically
                    current_time = time.time()
                    if current_time - last_status_time >= status_interval:
                        elapsed_time = current_time - start_time
                        
                        if total_pairs_processed > 0:
                            pairs_per_second = total_pairs_processed / elapsed_time
                            
                            status_msg = (
                                f"STATUS UPDATE - "
                                f"Total pairs: {total_pairs_processed:,} | "
                                f"Rate: {pairs_per_second:.1f} pairs/sec | "
                                f"Elapsed: {self._format_time_estimate(elapsed_time)}"
                            )
                        else:
                            status_msg = (
                                f"STATUS UPDATE - "
                                f"Starting pair generation... | "
                                f"Elapsed: {self._format_time_estimate(elapsed_time)}"
                            )
                        
                        logger.info(status_msg)
                        last_status_time = current_time
            
            # Write any remaining chunks
            if chunk_results:
                stream_writer.write_chunk(chunk_results)
            
            # Close the stream writer
            stream_writer.close()
            
            logger.info(f"Pair generation completed! Total pairs: {total_pairs_processed:,}")
            
        except Exception as e:
            logger.error(f"Error during pair generation: {e}")
            if stream_writer:
                stream_writer.close()
            raise e

    def generate_pairs_parquet(self):
        """
        Generate the pairs Arrow IPC file using stream writing.
        
        This method orchestrates the pair generation process by calling the core
        compute_distances_and_pairs method and then reading the generated
        file to provide statistics and validation.
        
        The method creates an Arrow IPC file containing molecule pairs with distance
        metrics and shape information. It then reads the file to verify the output
        and provide summary statistics.
        
        Uses self attributes:
        - pairs_file: Output file path for the pairs Arrow IPC file
            
        Creates:
        - Arrow IPC file at self.pairs_file containing molecule pairs
            
        Note:
            This method calls compute_distances_and_pairs() to do the actual work.
            After generation, it reads the file to provide statistics on the output.
            If the file cannot be read, it logs a warning but doesn't fail.
        """
        logger.info("Generating pairs Arrow IPC file...")
        
        # Compute distances and pairs using stream writing
        self.compute_distances_and_pairs()
        
        logger.info(f"Saved pairs to {self.pairs_file}")
        
        # Read the file to get statistics
        try:
            # Try reading as Arrow IPC format            
            reader = pa.ipc.RecordBatchStreamReader(self.pairs_file)
            table = reader.read_all()
            pairs_df = pl.from_arrow(table)
        except Exception as e:
            logger.warning(f"Could not read Arrow IPC file: {e}")
            logger.info("File may still be in progress or in a different format")
            return
            
        logger.info(f"Pairs dataframe shape: {pairs_df.shape}")
        logger.info(f"Columns: {pairs_df.columns}")
        
        # Print some statistics
        if len(pairs_df) > 0:
            morgan_distances = pairs_df['morgan_tanimoto_distance'].to_list()
            if 'shape_tanimoto_distance' in pairs_df.columns:
                shape_distances = [d for d in pairs_df['shape_tanimoto_distance'].to_list() if not np.isnan(d)]
            else:
                shape_distances = None
            
            logger.info(f"Morgan Tanimoto distance stats:")
            logger.info(f"  Mean: {np.mean(morgan_distances):.4f}")
            logger.info(f"  Std: {np.std(morgan_distances):.4f}")
            logger.info(f"  Min: {np.min(morgan_distances):.4f}")
            logger.info(f"  Max: {np.max(morgan_distances):.4f}")
            
            if shape_distances:
                logger.info(f"Shape Tanimoto distance stats:")
                logger.info(f"  Mean: {np.mean(shape_distances):.4f}")
                logger.info(f"  Std: {np.std(shape_distances):.4f}")
                logger.info(f"  Min: {np.min(shape_distances):.4f}")
                logger.info(f"  Max: {np.max(shape_distances):.4f}")
        
    def read_pairs_file_streaming(self):
        """
        Read the pairs Arrow IPC file while it's being written.
        
        This method attempts to read the Arrow IPC file and return it as a Polars DataFrame.
        It's designed to work even while the file is still being written, making it useful
        for monitoring progress during pair generation.
        
        Returns:
            Optional[pl.DataFrame]: Polars DataFrame containing the pairs data if successful,
                                   None otherwise.
                                   
        Note:
            The method handles various error conditions gracefully:
            - File doesn't exist yet
            - File is empty or incomplete
            - File is in an unexpected format
            All errors are logged at debug level and the method returns None.
        """
        try:
            # Try reading as Arrow IPC format
            pairs_df = pl.read_ipc(self.pairs_file)
            return pairs_df
        except Exception as e:
            # Check if file exists but is empty or incomplete
            if self.pairs_file.exists():
                file_size = self.pairs_file.stat().st_size
                if file_size == 0:
                    logger.debug("Arrow IPC file exists but is empty")
                else:
                    logger.debug(f"Arrow IPC file exists (size: {file_size}) but cannot be read: {e}")
            else:
                logger.debug(f"Arrow IPC file does not exist yet: {e}")
            return None
    
    def get_pairs_file_stats(self):
        """
        Get statistics about the pairs file if it can be read.
        
        This method attempts to read the pairs file and compute summary statistics
        on the distance metrics. It's useful for monitoring the quality and distribution
        of the generated pairs.
        
        Returns:
            Optional[Dict]: Dictionary containing statistics if successful, None otherwise.
                           The dictionary includes:
                           - total_rows: Total number of pairs
                           - columns: List of column names
                           - morgan_distance: Dict with mean, std, min, max of Morgan distances
                           - shape_distance: Dict with mean, std, min, max of shape distances (if available)
                           
        Note:
            The method handles cases where the file cannot be read gracefully.
            Shape distance statistics are only included if valid shape distances exist
            (excluding NaN and negative values from timeouts).
        """
        pairs_df = self.read_pairs_file_streaming()
        if pairs_df is None:
            return None
            
        stats = {
            'total_rows': len(pairs_df),
            'columns': pairs_df.columns,
        }
        
        if len(pairs_df) > 0:
            morgan_distances = pairs_df['morgan_tanimoto_distance'].to_list()
            shape_distances = [d for d in pairs_df['shape_tanimoto_distance'].to_list() if not np.isnan(d)]
            
            stats['morgan_distance'] = {
                'mean': float(np.mean(morgan_distances)),
                'std': float(np.std(morgan_distances)),
                'min': float(np.min(morgan_distances)),
                'max': float(np.max(morgan_distances)),
            }
            
            if shape_distances:
                stats['shape_distance'] = {
                    'mean': float(np.mean(shape_distances)),
                    'std': float(np.std(shape_distances)),
                    'min': float(np.min(shape_distances)),
                    'max': float(np.max(shape_distances)),
                }
        
        return stats
        
    def convert_to_parquet(self):
        """
        Convert the Arrow IPC file to Parquet format for compatibility.
        
        This method reads the Arrow IPC file and converts it to Parquet format,
        which is more widely supported by data processing tools. The conversion
        preserves all data and schema information.
        
        Uses self attributes:
        - pairs_file: Path to the source Arrow IPC file
        - output_dir: Directory where the parquet file will be saved
            
        Creates:
        - pairs.parquet file in the output directory
            
        Note:
            The method includes error handling and fallback conversion methods.
            If the primary conversion method fails, it attempts an alternative
            approach using Polars directly.
            If conversion fails completely, the method logs errors but doesn't
            raise exceptions to avoid disrupting the main workflow.
        """
        if not self.pairs_file.exists():
            logger.warning(f"Arrow IPC file {self.pairs_file} does not exist")
            return
            
        parquet_file = self.output_dir / "pairs.parquet"
        try:
            # Try reading with PyArrow first, then convert to Polars
            with pa.ipc.open_stream(self.pairs_file) as reader:
                batches = []
                try:
                    while True:
                        batch = reader.read_next_batch()
                        if batch is None:
                            break
                        batches.append(batch)
                except StopIteration:
                    pass
                
                if batches:
                    # Convert record batches to table and concatenate
                    # Use pa.Table.from_batches() instead of individual to_table() calls
                    try:
                        #breakpoint()
                        table = pa.Table.from_batches(batches)
                        # Convert to Polars DataFrame
                        pairs_df = pl.from_arrow(table)
                        pairs_df.write_parquet(parquet_file)
                        logger.info(f"Converted Arrow IPC to Parquet: {parquet_file}")
                    except Exception as e:
                        logger.error(f"Error converting Arrow IPC to Parquet: {e}")
                        raise e
                else:
                    logger.warning("No data found in Arrow IPC file")
                    
        except Exception as e:
            logger.error(f"Failed to convert Arrow IPC to Parquet: {e}")
            # Try alternative method using Polars directly
            try:
                pairs_df = pl.read_ipc(self.pairs_file)
                pairs_df.write_parquet(parquet_file)
                logger.info(f"Converted Arrow IPC to Parquet (alternative method): {parquet_file}")
            except Exception as e2:
                logger.error(f"Alternative conversion method also failed: {e2}")
    
    def generate_parquet_files(self):
        """
        Generate all parquet files for the molecule pair dataset.
        
        This is the main orchestration method that runs the complete pipeline
        for generating molecule pair datasets. It loads the dataset, generates
        the molecules parquet file, generates the pairs Arrow IPC file, and
        optionally converts to Parquet format.
        
        The method follows this sequence:
        1. Check if files already exist (unless overwrite is enabled)
        2. Load the molecular dataset with preprocessing and caching
        3. Generate the molecules parquet file with global indexing
        4. Generate the pairs Arrow IPC file with distance metrics
        5. Optionally convert Arrow IPC to Parquet format
        
        Uses self attributes:
        - overwrite_pairs_parquet: Whether to force regeneration of existing files
        - molecules_file: Path to the molecules parquet file
        - pairs_file: Path to the pairs Arrow IPC file
        - save_as_parquet: Whether to convert Arrow IPC to Parquet format
            
        Note:
            The method includes checks to avoid regenerating existing files unless
            explicitly requested. It provides comprehensive logging throughout
            the process and handles errors gracefully.
        """
        logger.info("Starting parquet file generation...")
        
        # Check if files already exist
        if not self.overwrite_pairs_parquet and (self.molecules_file.exists() or self.pairs_file.exists()):
            logger.warning("Files already exist. Use overwrite_parquet=True to regenerate.")
            return  
        
        # Load dataset
        self.load_dataset()
        
        # Generate parquet files
        self.generate_molecules_parquet()
        self.generate_pairs_parquet()  # No longer needs splits parameter
        
        # Convert to Parquet format if requested
        if self.save_as_parquet:
            self.convert_to_parquet()
        
        logger.info("Parquet file generation completed!")

    def generate_molecules_parquet(self):
        """
        Generate the molecules parquet file for all specified splits.
        
        This method creates a consolidated parquet file containing all valid molecules
        from the specified dataset splits. It filters out molecules with failed SELFIES
        encodings and creates a global indexing scheme across all splits.
        
        The output parquet file contains the following columns:
        - item_id: Global sequential index across all splits (int)
        - smiles: SMILES representation of the molecule (str)
        - selfies: SELFIES representation of the molecule (str, None for failed encodings)
        - split: Dataset split name (train, val, test) (str)
        - [property_names]: Dataset-specific molecular properties (float)
        
        For MoleculeNet datasets, properties include: SA, SC
        For Zinc 250k datasets, properties include: qed, sa, parp1, fa7, 5h1tb, braf, jak2, logp

        Note: Property names are defined as global constants
        (MOLECULENET_PROPERTY_NAMES, ZINC250K_PROPERTY_NAMES) for consistency.
        
        Uses self attributes:
        - molecules_file: Output file path for the molecules parquet
        - overwrite_molecules_parquet: Whether to force regeneration
        - splits: List of user-specified splits to process
        - dataset_splits: List of dataset split names
        - split_data: Dictionary containing processed molecular data
        - dataset_name: Name of the dataset for property selection
        
        Creates self attributes:
        - split_local_indices: Dictionary mapping split names to valid molecule indices
        - split_local_indices_to_global_idx: Dictionary mapping original indices to global item_ids
        
        Raises:
            ValueError: If a required split is not found in the loaded data
            
        Note:
            The method skips generation if the file already exists and overwrite_molecules_parquet is False.
            Invalid molecules (those with failed SELFIES encodings) are excluded from the output.
            Global item_id indexing starts from 0 and is sequential across all splits.
        """
        # Check if file exists and recomputation not forced
        if self.molecules_file.exists() and not self.overwrite_molecules_parquet:
            logger.info(f"Molecules file {self.molecules_file} already exists. Use overwrite_molecules_parquet=True to regenerate.")
            return
            
        logger.info(f"Generating molecules parquet file for splits: {self.splits}")
        
        # Determine property names based on dataset
        if self.dataset_name.lower() == "zinc250k":
            property_names = ZINC250K_PROPERTY_NAMES
        else:  # MoleculeNet datasets
            property_names = MOLECULENET_PROPERTY_NAMES
        
        # Create molecules data for all specified splits
        all_data = []
        self.split_local_indices = {}  # Store valid indices for each split
        self.split_local_indices_to_global_idx = {}  # Store mapping for each split
        
        global_item_id = 0  # Global counter across all splits
        
        for user_split, dataset_split in zip(self.splits, self.dataset_splits):
            if dataset_split not in self.split_data:
                raise ValueError(f"Split '{dataset_split}' not found in loaded data. Available splits: {list(self.split_data.keys())}")
            
            split_data = self.split_data[dataset_split]
            
            # Get valid molecules (exclude invalid ones)
            valid_indices = [i for i in range(len(split_data['smiles'])) if i not in split_data['invalid_indices']] # local index
            logger.info(f"Split '{dataset_split}': {len(valid_indices)} valid molecules, {len(split_data['invalid_indices'])} invalid.")
            
            # Store valid indices for this split
            self.split_local_indices[dataset_split] = valid_indices
            
            # Create mapping for this split
            split_mapping = {}
            
            # Add molecules for this split
            for original_idx in valid_indices:
                row = {
                    'item_id': global_item_id,  # Global sequential indexing
                    'smiles': split_data['smiles'][original_idx],
                    'selfies': split_data['selfies'][original_idx],
                    'split': user_split  # Use user-friendly split name
                }
                for prop_name in property_names:
                    if prop_name in split_data:
                        row[prop_name] = split_data[prop_name][original_idx]
                    else:
                        row[prop_name] = 0.0  # Default value if property not found
                all_data.append(row)
                
                # Store mapping from original index to global item_id
                split_mapping[original_idx] = global_item_id
                global_item_id += 1
            
            self.split_local_indices_to_global_idx[dataset_split] = split_mapping
        
        molecules_df = pl.DataFrame(all_data)
        molecules_df.write_parquet(self.molecules_file)
        logger.info(f"Saved molecules to {self.molecules_file} (shape: {molecules_df.shape})")


def main():
    """
    Legacy main function for backward compatibility.
    
    This function provides a helpful message directing users to the updated
    script that uses Hydra configuration. The original script has been
    refactored to use a more robust configuration system.
    
    Note:
        This function is kept for backward compatibility but should not be used
        for new development. Use the generate_pairs.py script instead.
    """
    print("This script has been updated to use Hydra configuration.")
    print("Please use the new generate_pairs.py script instead:")
    print("python generate_pairs.py")
    print("\nOr run with custom configuration:")
    print("python generate_pairs.py dataset_name=bace percentile_threshold=10 test_mode=true")


if __name__ == "__main__":
    main() 

# to-do's (do not delete this comment)
# fix splits to use anderson splits - now done
# track for how many the selfies computation fails (or the molecule is not sane) and print a statistic on this
