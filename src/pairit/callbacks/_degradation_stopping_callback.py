"""
Callback for stopping training when a monitored metric degrades by a threshold amount.

This callback monitors a metric (e.g., val_loss) and stops training gracefully when
the metric has increased (degraded) by a specified threshold compared to the best
value seen so far. Works with both single and multi-GPU training.
"""

import logging
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class DegradationStoppingCallback(Callback):
    """
    Stops training when a monitored metric degrades by a threshold amount.
    
    This callback tracks the best value of a metric and stops training when
    the current value has increased (for loss metrics) or decreased (for 
    accuracy metrics) by a threshold amount compared to the best value.
    
    Parameters
    ----------
    monitor : str, default="val_loss"
        Metric name to monitor. Can be any logged metric (e.g., "val_loss", 
        "val/accuracy", etc.).
    threshold : float, default=0.1
        Threshold for stopping. Interpretation depends on threshold_mode.
        Requires patience consecutive checks before stopping (unless immediate_threshold is exceeded).
    threshold_mode : str, default="absolute"
        How to interpret threshold:
        - "absolute": Threshold is an absolute difference.
          For loss (mode='min'): stops when current >= best + threshold
          For accuracy (mode='max'): stops when current <= best - threshold
        - "proportional": Threshold is a multiplicative factor.
          For loss (mode='min'): stops when current >= best * (1 + threshold)
          For accuracy (mode='max'): stops when current <= best * (1 - threshold)
    patience : int, default=3
        Number of consecutive validation checks where metric is degraded beyond threshold
        before stopping. Prevents stopping due to noise/fluctuation. Set to 1 for immediate
        stopping when threshold is exceeded.
    immediate_threshold : float, optional
        If set, stop immediately (ignoring patience) when degradation exceeds this value.
        Must be larger than threshold. Useful for catching severe degradation.
    mode : str, default="min"
        Whether the metric should be minimized ("min") or maximized ("max").
        For loss metrics, use "min". For accuracy metrics, use "max".
    min_epochs : int, default=0
        Minimum number of epochs to train before stopping is allowed.
    min_steps : int, default=0
        Minimum number of training steps before stopping is allowed.
    verbose : bool, default=True
        Whether to log when stopping occurs.
    
    Examples
    --------
    >>> # Stop when val_loss increases by 0.1 (absolute) from best value
    >>> callback = DegradationStoppingCallback(
    ...     monitor="val_loss",
    ...     threshold=0.1,
    ...     threshold_mode="absolute",
    ...     mode="min"
    ... )
    
    >>> # Stop when val_loss is 10% worse (proportional) than best value
    >>> callback = DegradationStoppingCallback(
    ...     monitor="val_loss",
    ...     threshold=0.1,  # 10% = 0.1
    ...     threshold_mode="proportional",
    ...     mode="min"
    ... )
    
    >>> # Stop when accuracy decreases by 0.05 (absolute) from best value
    >>> callback = DegradationStoppingCallback(
    ...     monitor="val/accuracy",
    ...     threshold=0.05,
    ...     threshold_mode="absolute",
    ...     mode="max"
    ... )
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        threshold: float = 0.1,
        threshold_mode: str = "absolute",
        patience: int = 3,
        immediate_threshold: Optional[float] = None,
        mode: str = "min",
        min_epochs: int = 0,
        min_steps: int = 0,
        verbose: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.patience = patience
        self.immediate_threshold = immediate_threshold
        self.mode = mode
        self.min_epochs = min_epochs
        self.min_steps = min_steps
        self.verbose = verbose
        
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if threshold_mode not in ("absolute", "proportional"):
            raise ValueError(f"threshold_mode must be 'absolute' or 'proportional', got {threshold_mode}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if immediate_threshold is not None and immediate_threshold <= threshold:
            raise ValueError(f"immediate_threshold ({immediate_threshold}) must be > threshold ({threshold})")
        
        # Track best value seen so far
        self.best_value: Optional[float] = None
        self.best_epoch: int = -1
        self.best_step: int = -1
        
        # Track consecutive degradation checks
        self.degradation_count: int = 0
        
    def _get_metric_value(self, trainer: pl.Trainer) -> Optional[float]:
        """Extract the monitored metric value from trainer."""
        # Try callback_metrics first (metrics from validation)
        if self.monitor in trainer.callback_metrics:
            metric = trainer.callback_metrics[self.monitor]
            # Handle tensor metrics
            if hasattr(metric, "item"):
                return float(metric.item())
            return float(metric)
        
        # Fallback to logged_metrics
        if self.monitor in trainer.logged_metrics:
            metric = trainer.logged_metrics[self.monitor]
            if hasattr(metric, "item"):
                return float(metric.item())
            return float(metric)
        
        return None
    
    def _check_degradation(self, current_value: float, trainer: pl.Trainer) -> tuple[bool, bool, float, bool]:
        """
        Check if current value is degraded beyond threshold.
        
        Returns
        -------
        tuple[bool, bool, float, bool]
            (is_degraded, is_immediate_stop, degradation_amount, is_new_best)
            - is_degraded: True if degradation >= threshold
            - is_immediate_stop: True if degradation >= immediate_threshold (if set)
            - degradation_amount: The absolute degradation value
            - is_new_best: True if current value is better than best (best updated)
        """
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = trainer.current_epoch
            self.best_step = trainer.global_step
            self.degradation_count = 0
            return False, False, 0.0, True
        
        if self.mode == "min":
            # For loss metrics: lower is better
            if current_value < self.best_value:
                # New best value - reset counter
                self.best_value = current_value
                self.best_epoch = trainer.current_epoch
                self.best_step = trainer.global_step
                self.degradation_count = 0
                return False, False, 0.0, True
            
            degradation = current_value - self.best_value
            
            # Check thresholds
            if self.threshold_mode == "absolute":
                is_degraded = degradation >= self.threshold
                if self.immediate_threshold is not None:
                    is_immediate = degradation >= self.immediate_threshold
                else:
                    is_immediate = False
            else:  # proportional
                degradation_pct = degradation / self.best_value if self.best_value != 0 else 0
                is_degraded = degradation_pct >= self.threshold
                if self.immediate_threshold is not None:
                    is_immediate = degradation_pct >= self.immediate_threshold
                else:
                    is_immediate = False
        else:  # mode == "max"
            # For accuracy metrics: higher is better
            if current_value > self.best_value:
                # New best value - reset counter
                self.best_value = current_value
                self.best_epoch = trainer.current_epoch
                self.best_step = trainer.global_step
                self.degradation_count = 0
                return False, False, 0.0, True
            
            degradation = self.best_value - current_value
            
            # Check thresholds
            if self.threshold_mode == "absolute":
                is_degraded = degradation >= self.threshold
                if self.immediate_threshold is not None:
                    is_immediate = degradation >= self.immediate_threshold
                else:
                    is_immediate = False
            else:  # proportional
                degradation_pct = degradation / self.best_value if self.best_value != 0 else 0
                is_degraded = degradation_pct >= self.threshold
                if self.immediate_threshold is not None:
                    is_immediate = degradation_pct >= self.immediate_threshold
                else:
                    is_immediate = False
        
        return is_degraded, is_immediate, degradation, False
    
    def _should_stop(self, current_value: float, trainer: pl.Trainer) -> bool:
        """Determine if training should stop based on degradation threshold and patience."""
        # Check minimum epochs/steps requirements
        if trainer.current_epoch < self.min_epochs:
            return False
        if trainer.global_step < self.min_steps:
            return False
        
        # Check if degraded (also handles initializing best_value)
        is_degraded, is_immediate, degradation, is_new_best = self._check_degradation(current_value, trainer)
        
        # If we found a new best value, don't stop
        if is_new_best:
            return False
        
        # If metric is not degraded, reset counter
        if not is_degraded:
            self.degradation_count = 0
            return False
        
        # If immediate threshold is exceeded, stop immediately
        if is_immediate:
            if self.verbose and trainer.global_rank == 0:
                logger.info(
                    f"DegradationStoppingCallback: Immediate stop triggered - degradation ({degradation:.6f}) "
                    f"exceeds immediate_threshold ({self.immediate_threshold:.6f})"
                )
            return True
        
        # Increment degradation counter
        self.degradation_count += 1
        
        # Stop if we've seen degradation for patience consecutive checks
        if self.degradation_count >= self.patience:
            return True
        
        return False
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check metric after each validation epoch."""
        current_value = self._get_metric_value(trainer)
        
        if current_value is None:
            if self.verbose and trainer.global_rank == 0:
                logger.warning(
                    f"DegradationStoppingCallback: Metric '{self.monitor}' not found. "
                    f"Available metrics: {list(trainer.callback_metrics.keys()) + list(trainer.logged_metrics.keys())}"
                )
            return
        
        if self._should_stop(current_value, trainer):
            if self.verbose and trainer.global_rank == 0:
                degradation = abs(current_value - self.best_value)
                stop_reason = "immediate threshold" if self.degradation_count == 1 and self.immediate_threshold is not None else f"patience ({self.patience} consecutive checks)"
                
                if self.threshold_mode == "proportional":
                    degradation_pct = (degradation / self.best_value) * 100 if self.best_value != 0 else 0
                    logger.info(
                        f"DegradationStoppingCallback: Stopping training due to metric degradation.\n"
                        f"  Metric: {self.monitor}\n"
                        f"  Best value: {self.best_value:.6f} (epoch {self.best_epoch}, step {self.best_step})\n"
                        f"  Current value: {current_value:.6f} (epoch {trainer.current_epoch}, step {trainer.global_step})\n"
                        f"  Degradation: {degradation:.6f} ({degradation_pct:.2f}%) (threshold: {self.threshold*100:.2f}%)\n"
                        f"  Stop reason: {stop_reason}"
                    )
                else:
                    logger.info(
                        f"DegradationStoppingCallback: Stopping training due to metric degradation.\n"
                        f"  Metric: {self.monitor}\n"
                        f"  Best value: {self.best_value:.6f} (epoch {self.best_epoch}, step {self.best_step})\n"
                        f"  Current value: {current_value:.6f} (epoch {trainer.current_epoch}, step {trainer.global_step})\n"
                        f"  Degradation: {degradation:.6f} (threshold: {self.threshold:.6f})\n"
                        f"  Stop reason: {stop_reason}"
                    )
            
            # Stop training gracefully
            trainer.should_stop = True
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Also check after training epoch in case metric is logged there."""
        # This is a fallback - validation metrics are typically checked in on_validation_epoch_end
        # But some metrics might be logged during training
        if self.monitor.startswith("train/"):
            current_value = self._get_metric_value(trainer)
            if current_value is not None:
                if self._should_stop(current_value, trainer):
                    if self.verbose and trainer.global_rank == 0:
                        degradation = abs(current_value - self.best_value)
                        stop_reason = "immediate threshold" if self.degradation_count == 1 and self.immediate_threshold is not None else f"patience ({self.patience} consecutive checks)"
                        
                        if self.threshold_mode == "proportional":
                            degradation_pct = (degradation / self.best_value) * 100 if self.best_value != 0 else 0
                            logger.info(
                                f"DegradationStoppingCallback: Stopping training due to metric degradation.\n"
                                f"  Metric: {self.monitor}\n"
                                f"  Best value: {self.best_value:.6f} (epoch {self.best_epoch}, step {self.best_step})\n"
                                f"  Current value: {current_value:.6f} (epoch {trainer.current_epoch}, step {trainer.global_step})\n"
                                f"  Degradation: {degradation:.6f} ({degradation_pct:.2f}%) (threshold: {self.threshold*100:.2f}%)\n"
                                f"  Stop reason: {stop_reason}"
                            )
                        else:
                            logger.info(
                                f"DegradationStoppingCallback: Stopping training due to metric degradation.\n"
                                f"  Metric: {self.monitor}\n"
                                f"  Best value: {self.best_value:.6f} (epoch {self.best_epoch}, step {self.best_step})\n"
                                f"  Current value: {current_value:.6f} (epoch {trainer.current_epoch}, step {trainer.global_step})\n"
                                f"  Degradation: {degradation:.6f} (threshold: {self.threshold:.6f})\n"
                                f"  Stop reason: {stop_reason}"
                            )
                    trainer.should_stop = True


