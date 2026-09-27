"""
Timing Infrastructure for Molecule Validation
==============================================

Provides simple timing tracking for validation operations.
Prints a single summary table at the end of each validation epoch.
"""

import time
from contextlib import contextmanager
from typing import Dict
import torch


def _has_trainer(pl_module) -> bool:
    """Check if a Lightning module has an attached trainer."""
    return hasattr(pl_module, '_trainer') and pl_module._trainer is not None


def _safe_log(pl_module, name: str, value, **kwargs):
    """Log to pl_module only if a trainer is attached."""
    if _has_trainer(pl_module):
        pl_module.log(name, value, **kwargs)


def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    if seconds < 3600.0:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{s:05.2f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s"


class TimingTracker:
    """
    Track and log timing metrics for validation operations.

    Provides context managers for timing operations and automatic logging
    to PyTorch Lightning with proper device placement and DDP support.

    All operations are accumulated (additive) so that repeated calls
    (e.g., per-batch) are summed into a single entry.

    Call ``print_summary()`` at the end of each validation epoch to get
    a single, readable table of all timed operations.
    """

    def __init__(self):
        """Initialize empty timing dictionary and call-count tracker."""
        self.timings: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._epoch_wall_start: float | None = None

    def start_epoch(self) -> None:
        """Mark the wall-clock start of a validation epoch."""
        self._epoch_wall_start = time.time()

    @contextmanager
    def time_operation(self, name: str):
        """Context manager to time an operation and accumulate result."""
        start = time.time()
        yield
        elapsed = time.time() - start
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def record(self, name: str, elapsed: float) -> None:
        """Manually record an elapsed time (useful when context manager isn't practical)."""
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------
    def print_summary(self, epoch: int | None = None) -> None:
        """Print a compact timing summary table to stdout."""
        wall_total = time.time() - self._epoch_wall_start if self._epoch_wall_start else None

        header = "Validation Timing"
        if epoch is not None:
            header += f" — Epoch {epoch}"

        W = 72
        print()
        print("=" * W)
        print(f"  {header}")
        print("=" * W)

        if not self.timings:
            print("  (no operations timed)")
            print("=" * W)
            return

        # Sort by hierarchy: group by first path component, then alphabetical
        sorted_keys = sorted(self.timings.keys())
        accounted = 0.0

        name_w = max(len(k) for k in sorted_keys)
        name_w = min(max(name_w, 30), 50)

        prev_group = None
        for key in sorted_keys:
            group = key.split("/")[0] if "/" in key else ""
            if group != prev_group and group:
                print(f"  [{group}]")
                prev_group = group

            elapsed = self.timings[key]
            count = self._counts.get(key, 1)
            accounted += elapsed

            label = key
            time_str = _fmt_time(elapsed)

            count_str = f"  ({count}x)" if count > 1 else ""
            pct_str = ""
            if wall_total and wall_total > 0:
                pct_str = f"  {elapsed / wall_total * 100:5.1f}%"

            print(f"  {label:<{name_w}}  {time_str:>12}{count_str}{pct_str}")

        print("-" * W)
        if wall_total is not None:
            unaccounted = wall_total - accounted
            print(f"  {'accounted':<{name_w}}  {_fmt_time(accounted):>12}  {accounted / wall_total * 100:5.1f}%")
            if unaccounted > 0.5:
                print(f"  {'unaccounted':<{name_w}}  {_fmt_time(unaccounted):>12}  {unaccounted / wall_total * 100:5.1f}%")
            print(f"  {'WALL TOTAL':<{name_w}}  {_fmt_time(wall_total):>12}  100.0%")
        else:
            print(f"  {'TOTAL (accounted)':<{name_w}}  {_fmt_time(accounted):>12}")
        print("=" * W)
        print()

    # ------------------------------------------------------------------
    # WandB / Lightning logging
    # ------------------------------------------------------------------
    def log_all(self, pl_module, prefix: str = "timing", required_keys: list = None):
        """Log all timing metrics to wandb under timing/ namespace."""
        if required_keys is not None:
            for key in required_keys:
                elapsed = self.timings.get(key, 0.0)
                _safe_log(pl_module,
                    f"{prefix}/{key}",
                    torch.tensor(elapsed, device=pl_module.device),
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True
                )
        else:
            for name, elapsed in self.timings.items():
                _safe_log(pl_module,
                    f"{prefix}/{name}",
                    torch.tensor(elapsed, device=pl_module.device),
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True
                )
        # Also log wall total if available
        if self._epoch_wall_start is not None:
            wall_total = time.time() - self._epoch_wall_start
            _safe_log(pl_module,
                f"{prefix}/wall_total",
                torch.tensor(wall_total, device=pl_module.device),
                sync_dist=False,
                on_step=False,
                on_epoch=True
            )
        self.timings.clear()
        self._counts.clear()

    def get_timing(self, name: str) -> float:
        """Get timing for a specific operation."""
        return self.timings.get(name, 0.0)

    def clear(self):
        """Clear all timing data."""
        self.timings.clear()
        self._counts.clear()
        self._epoch_wall_start = None
