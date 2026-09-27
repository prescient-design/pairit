"""
Validation Printer and Metric Registry
=======================================

Provides consistent, organized printing and metric logging for validation callbacks.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.distributed as dist


def _has_trainer(pl_module) -> bool:
    """Check if a Lightning module has an attached trainer."""
    return hasattr(pl_module, '_trainer') and pl_module._trainer is not None


def _safe_log(pl_module, name: str, value, **kwargs):
    """Log to pl_module only if a trainer is attached."""
    if _has_trainer(pl_module):
        pl_module.log(name, value, **kwargs)

def shorten_str(possible_smiles: str, n: int = 300) -> str:
    """
    Truncate a string to a specified length for better printing.
    
    This function takes a (e.g. SMILES) string and returns a truncated version that is more 
    suitable for printing. If the string is longer than n characters, it truncates 
    to the first n characters and adds an ellipsis. It also removes spaces from the 
    input string before processing.
    
    Parameters
    ----------
    possible_smiles : str
        The string to be truncated. Can be any string, but typically a 
        molecular SMILES representation.
    n : int, default=300
        The maximum number of characters to include in the output string.
        
    Returns
    -------
    str
        The truncated SMILES string. If the input is longer than n characters,
        returns the first n characters followed by '...'. If the input is not
        a string, returns the input unchanged.
        
    Examples
    --------
    >>> shorten_str("CCO", 5)
    'CCO'
    >>> shorten_str("CC(C)(C)C1=CC=C(C=C1)C(C)C(=O)O", 10)
    'CC(C)(C)C...'
    """
    if type(possible_smiles) != str:
        return possible_smiles
    return ''.join(possible_smiles.split(' '))[:n] + '...' if len(possible_smiles) > n else possible_smiles

@dataclass
class MetricGroup:
    """A group of related metrics for organized printing."""
    title: str
    metrics: List[Tuple[str, float, str]]  # (name, value, description)


class ValidationPrinter:
    """
    Consistent, structured printing for validation metrics.
    
    Provides formatted output with headers, metric lines, and summaries.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the printer.
        
        Args:
            verbose: Whether to print output (can be disabled for silent operation)
        """
        self.verbose = verbose
    
    def print_header(self, title: str, width: int = 80):
        """
        Print a section header.
        
        Args:
            title: Header title text
            width: Total width of header line
        """
        if not self.verbose:
            return
        print(f"\n{'='*width}")
        print(f"  {title}")
        print(f"{'='*width}")
    
    def print_subheader(self, title: str, width: int = 80):
        """
        Print a subsection header.
        
        Args:
            title: Subheader title text
            width: Total width of header line
        """
        if not self.verbose:
            return
        print(f"\n  {title}")
        print(f"  {'-'*(width-4)}")
    
    def print_metric(self, name: str, value: float, unit: str = "", 
                    description: str = "", indent: int = 4):
        """
        Print a single metric in consistent format.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit string (e.g., "%", "ms")
            description: Optional description
            indent: Number of spaces to indent
        """
        if not self.verbose:
            return
        
        # Format value with unit
        if isinstance(value, (int, bool)):
            value_str = f"{value}{unit}"
        else:
            value_str = f"{value:.4f}{unit}"
        
        # Build description suffix
        desc_str = f"  # {description}" if description else ""
        
        # Print with dot leaders
        indent_str = " " * indent
        print(f"{indent_str}{name:.<50} {value_str:>10}{desc_str}")
    
    def print_group(self, group: MetricGroup, width: int = 80):
        """
        Print a group of metrics.
        
        Args:
            group: MetricGroup to print
            width: Total width of output
        """
        if not self.verbose:
            return
        
        self.print_subheader(group.title, width)
        for name, value, desc in group.metrics:
            self.print_metric(name, value, description=desc)
    
    def print_summary(self, groups: List[MetricGroup], title: str = "Validation Metrics", 
                     width: int = 80):
        """
        Print a complete summary of multiple metric groups.
        
        Args:
            groups: List of MetricGroup objects
            title: Overall summary title
            width: Total width of output
        """
        if not self.verbose:
            return
        
        self.print_header(title, width)
        for group in groups:
            self.print_group(group, width)
        print(f"{'='*width}\n")

    # Convenience wrappers used by callbacks
    def print_section(self, title: str, width: int = 80):
        """Alias for print_header for semantic clarity at call sites."""
        self.print_header(title, width)

    def print_info(self, message: str):
        if not self.verbose:
            return
        print(f"[Info] {message}")

    def print_warning(self, message: str):
        if not self.verbose:
            return
        print(f"[Warning] {message}")

    def print_timing(self, message: str):
        if not self.verbose:
            return
        print(f"[Timing] {message}")

    def print_progress(self, message: str):
        if not self.verbose:
            return
        print(f"[Progress] {message}")

    def print_debug(self, message: str):
        if not self.verbose:
            return
        print(f"[Debug] {message}")


class MetricRegistry:
    """
    Central registry for collecting and logging validation metrics.
    
    Collects metrics during computation, then logs them all at once with
    organized printing. Ensures consistency in metric naming, device placement,
    and logging flags.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self.metrics: Dict[str, float] = {}
        self.fractions: Dict[str, Tuple[float, float]] = {}  # name -> (numerator, denominator)
        self.metadata: Dict[str, Dict[str, str]] = {}
        self._categories_order: List[str] = []
    
    def register(self, name: str, value: float, 
                 description: str = "", 
                 category: str = "General",
                 unit: str = ""):
        """
        Register a metric to be logged later.
        
        Args:
            name: Metric name (will be prefixed with 'val_' when logged)
            value: Metric value
            description: Human-readable description
            category: Category for grouping in printed output
            unit: Optional unit string
        """
        self.metrics[name] = float(value)
        self.metadata[name] = {
            'description': description,
            'category': category,
            'unit': unit
        }
        
        # Track category order for consistent printing
        if category not in self._categories_order:
            self._categories_order.append(category)
    
    def register_multiple(self, metrics: Dict[str, float], category: str = "General",
                         descriptions: Optional[Dict[str, str]] = None):
        """
        Register multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric_name -> value
            category: Category for all metrics
            descriptions: Optional dictionary of metric_name -> description
        """
        descriptions = descriptions or {}
        for name, value in metrics.items():
            desc = descriptions.get(name, "")
            self.register(name, value, description=desc, category=category)
    
    def register_fraction(self, name: str, numerator: float, denominator: float,
                          description: str = "", category: str = "General",
                          unit: str = ""):
        """
        Register a fraction metric (numerator/denominator) for DDP-correct aggregation.

        In DDP, numerators and denominators are summed across ranks before division,
        ensuring correct global fractions (``sum(num) / sum(denom)``).

        Parameters
        ----------
        name : str
            Metric name (will be prefixed with 'val_' when logged).
        numerator : float
            Numerator value.
        denominator : float
            Denominator value.
        description : str
            Human-readable description.
        category : str
            Category for grouping in printed output.
        unit : str
            Optional unit string.
        """
        self.fractions[name] = (float(numerator), float(denominator))
        self.metadata[name] = {
            'description': description,
            'category': category,
            'unit': unit,
        }
        if category not in self._categories_order:
            self._categories_order.append(category)

    def get_by_category(self) -> Dict[str, List[Tuple[str, float, Dict[str, str]]]]:
        """
        Get all metrics grouped by category.

        Returns:
            Dictionary mapping category -> list of (name, value, metadata) tuples
        """
        by_category: Dict[str, List[Tuple[str, float, Dict[str, str]]]] = {}

        # Regular metrics
        for name, value in self.metrics.items():
            cat = self.metadata[name]['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((name, value, self.metadata[name]))

        # Resolved fractions (use local num/denom; DDP reduction happens in log_all)
        for name, (num, denom) in self.fractions.items():
            value = float(num) / float(denom) if denom > 0 else float('nan')
            cat = self.metadata[name]['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((name, value, self.metadata[name]))

        # Return in consistent order
        return {cat: by_category[cat] for cat in self._categories_order if cat in by_category}
    
    def log_all(self, pl_module, printer: Optional[ValidationPrinter] = None,
                prefix: str = "val", echo: bool = False):
        """
        Log all registered metrics to Lightning and optionally print summary.

        Fraction metrics are aggregated via all_reduce (SUM) across DDP ranks
        before computing the ratio, so the global fraction is correct.  Regular
        (non-fraction) metrics are logged with ``sync_dist=True`` which averages
        across ranks — appropriate for means computed over similar sample sizes.

        Parameters
        ----------
        pl_module : LightningModule
            PyTorch Lightning module with ``.log()`` method.
        printer : ValidationPrinter, optional
            Optional printer for formatted console output.
        prefix : str
            Prefix for metric names (default: ``"val"``).
        echo : bool
            If True, also print each metric as a simple name/value line.
        """
        # --- DDP reduce fractions ---------------------------------------------------
        resolved_fractions: Dict[str, float] = {}
        if self.fractions:
            frac_names = sorted(self.fractions.keys())
            nums = torch.tensor([self.fractions[n][0] for n in frac_names],
                                dtype=torch.float64, device=pl_module.device)
            denoms = torch.tensor([self.fractions[n][1] for n in frac_names],
                                  dtype=torch.float64, device=pl_module.device)
            if dist.is_initialized():
                rank = dist.get_rank()
                print(
                    f"[DDP MetricRegistry] rank={rank}, "
                    f"num_fractions={len(frac_names)}, "
                    f"fraction_names_hash={hash(tuple(frac_names))}"
                )
                # Safety check: verify all ranks have same number of fractions
                local_count = torch.tensor(
                    [len(frac_names)], dtype=torch.long, device=pl_module.device
                )
                counts = [
                    torch.zeros_like(local_count)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(counts, local_count)
                if not all(c.item() == local_count.item() for c in counts):
                    print(
                        f"[DDP ERROR] rank={rank} has {local_count.item()} fractions, "
                        f"other ranks have {[c.item() for c in counts]}. "
                        f"Skipping all_reduce to prevent deadlock. "
                        f"Fraction names: {frac_names}"
                    )
                    # Fall through WITHOUT all_reduce — metrics will be per-rank only
                else:
                    dist.all_reduce(nums, op=dist.ReduceOp.SUM)
                    dist.all_reduce(denoms, op=dist.ReduceOp.SUM)
            for i, name in enumerate(frac_names):
                d = denoms[i].item()
                resolved_fractions[name] = (nums[i].item() / d) if d > 0 else float('nan')

        # --- Group by category (for printing) ----------------------------------------
        by_category = self.get_by_category()
        # Overlay DDP-resolved fraction values for printing
        if resolved_fractions:
            for cat, items in by_category.items():
                by_category[cat] = [
                    (n, resolved_fractions[n], m) if n in resolved_fractions else (n, v, m)
                    for n, v, m in items
                ]

        # Print organized summary
        if printer and printer.verbose:
            groups = []
            for category, metrics in by_category.items():
                metric_tuples = [
                    (name, value, meta['description'])
                    for name, value, meta in metrics
                ]
                groups.append(MetricGroup(category, metric_tuples))

            printer.print_summary(groups, title="Validation Metrics Summary")

            # Optional echo mode
            if echo:
                for category, metrics in by_category.items():
                    for name, value, meta in metrics:
                        sep = "/" if "/" in prefix else "_"
                        metric_name = f"{prefix}{sep}{name}"
                        printer.print_metric(metric_name, value, description=meta.get('description', ''))

        sep = "/" if "/" in prefix else "_"

        # --- Log regular (non-fraction) metrics with sync_dist=True -----------------
        for name, value in self.metrics.items():
            tensor_value = torch.tensor(value, device=pl_module.device)
            _safe_log(pl_module,
                f"{prefix}{sep}{name}",
                tensor_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True
            )

        # --- Log fraction metrics with sync_dist=False (already reduced) ------------
        for name, value in resolved_fractions.items():
            tensor_value = torch.tensor(value, device=pl_module.device)
            _safe_log(pl_module,
                f"{prefix}{sep}{name}",
                tensor_value,
                sync_dist=False,
                on_step=False,
                on_epoch=True
            )

        # --- Direct logger fallback for epoch-end metrics ---------------------------
        # Lightning's pl_module.log() can silently drop metrics logged from
        # on_validation_epoch_end if the key wasn't registered during
        # on_validation_batch_end. Log directly to the logger as a fallback.
        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "log_metrics"):
            step = getattr(pl_module, "global_step", None)
            direct_metrics = {}
            for name, value in self.metrics.items():
                v = float(value)
                if not (math.isnan(v) or math.isinf(v)):
                    direct_metrics[f"{prefix}{sep}{name}"] = v
            for name, value in resolved_fractions.items():
                v = float(value)
                if not (math.isnan(v) or math.isinf(v)):
                    direct_metrics[f"{prefix}{sep}{name}"] = v
            if direct_metrics:
                logger.log_metrics(direct_metrics, step=step)

        # Clear registry for next use
        self.clear()
    
    def clear(self):
        """Clear all registered metrics."""
        self.metrics = {}
        self.fractions = {}
        self.metadata = {}
        self._categories_order = []

    def __len__(self) -> int:
        """Return number of registered metrics (regular + fraction)."""
        return len(self.metrics) + len(self.fractions)

    def __contains__(self, name: str) -> bool:
        """Check if a metric is registered (regular or fraction)."""
        return name in self.metrics or name in self.fractions

    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get a registered metric value (fractions resolved locally)."""
        if name in self.metrics:
            return self.metrics[name]
        if name in self.fractions:
            num, denom = self.fractions[name]
            return (num / denom) if denom > 0 else float('nan')
        return default

