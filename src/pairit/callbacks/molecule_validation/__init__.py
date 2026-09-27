"""
Molecule Validation Module
===========================

Organized validation metrics and generation for molecule improvement models.

This module splits the large MoleculeValidationCallback into focused components:
- printer: Metric logging and printing utilities
- smiles_utils: SMILES validation and conversion utilities
- generation: Molecule generation (conditional/unconditional)
- conditional_metrics: Conditional generation metrics computation
- unconditional_metrics: Unconditional generation metrics computation
- timing: Timing infrastructure for performance tracking
- distance_utils: Fingerprint-cached distance computation
- property_registry: Configurable molecular properties
"""

from .printer import ValidationPrinter, MetricRegistry, shorten_str
from .smiles_utils import SmilesUtils
from .generation import MoleculeGenerator
from .base_metrics import MoleculeMetricsComputer, SampleResult
from .conditional_metrics import ConditionalMetricsComputer
from .unconditional_metrics import UnconditionalMetricsComputer
from pairit.callbacks.molecule_validation.evaluate_properties import PropertyEvaluator
from .timing import TimingTracker
from .distance_utils import (
    compute_distances_with_fingerprint_caching,
    compute_min_distance_to_set,
    parse_and_fingerprint,
)
from .property_registry import (
    get_property_specs,
    register_custom_property,
    PROPERTY_REGISTRY,
    DEFAULT_PROPERTIES,
    get_available_properties,
    get_seed_mol_stats,
    get_max_ring_size,
    get_max_alkyl_chain_length,
    load_dataset_statistics_from_files,
)

__all__ = [
    'ValidationPrinter',
    'MetricRegistry',
    'SmilesUtils',
    'MoleculeGenerator',
    'MoleculeMetricsComputer',
    'SampleResult',
    'ConditionalMetricsComputer',
    'UnconditionalMetricsComputer',
    'PropertyEvaluator',
    'TimingTracker',
    'compute_distances_with_fingerprint_caching',
    'compute_min_distance_to_set',
    'parse_and_fingerprint',
    'get_property_specs',
    'register_custom_property',
    'PROPERTY_REGISTRY',
    'DEFAULT_PROPERTIES',
    'get_available_properties',
    'get_seed_mol_stats',
    'get_max_ring_size',
    'get_max_alkyl_chain_length',
    'load_dataset_statistics_from_files',
]

