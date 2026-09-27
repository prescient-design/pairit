"""
Preemption detection and logging callback.

This callback detects when preemption occurs and logs it to W&B for better tracking.
"""

import os
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class PreemptionCallback(Callback):
    """
    Callback to detect and log preemption events.
    
    This callback:
    1. Detects when training is interrupted (preemption)
    2. Logs preemption events to W&B
    3. Tracks preemption count and timing
    4. Provides better visibility into preemption behavior
    """
    
    def __init__(self, log_preemption_events: bool = True):
        """
        Initialize the preemption callback.
        
        Args:
            log_preemption_events: Whether to log preemption events to W&B
        """
        super().__init__()
        self.log_preemption_events = log_preemption_events
        self.preemption_count = 0
        self.last_epoch = 0
        self.last_global_step = 0
        self.training_start_time = None
        
    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        if rank_zero_only.rank == 0:
            import time
            self.training_start_time = time.time()
            
            # Log training start event
            if self.log_preemption_events and hasattr(trainer.logger, 'experiment'):
                try:
                    trainer.logger.experiment.log({
                        "preemption/event": "training_start",
                        "preemption/count": self.preemption_count,
                        "preemption/timestamp": self.training_start_time
                    })
                    print("Logged training start event to W&B")
                except Exception as e:
                    print(f"Warning: Could not log training start to W&B: {e}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        if rank_zero_only.rank == 0:
            current_epoch = trainer.current_epoch
            current_step = trainer.global_step
            
            # Check for potential preemption (epoch/step regression)
            if current_epoch < self.last_epoch or current_step < self.last_global_step:
                self.preemption_count += 1
                print(f"WARNING: Detected potential preemption regression!")
                print(f"  Previous: epoch={self.last_epoch}, step={self.last_global_step}")
                print(f"  Current:  epoch={current_epoch}, step={current_step}")
                
                # Log preemption regression event
                if self.log_preemption_events and hasattr(trainer.logger, 'experiment'):
                    try:
                        trainer.logger.experiment.log({
                            "preemption/event": "regression_detected",
                            "preemption/count": self.preemption_count,
                            "preemption/previous_epoch": self.last_epoch,
                            "preemption/previous_step": self.last_global_step,
                            "preemption/current_epoch": current_epoch,
                            "preemption/current_step": current_step,
                            "preemption/regression_reason": "epoch_or_step_decreased"
                        })
                        print("Logged preemption regression to W&B")
                    except Exception as e:
                        print(f"Warning: Could not log preemption regression to W&B: {e}")
            
            # Update tracking variables
            self.last_epoch = current_epoch
            self.last_global_step = current_step
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends (normally or due to preemption)."""
        if rank_zero_only.rank == 0:
            import time
            training_end_time = time.time()
            training_duration = training_end_time - self.training_start_time if self.training_start_time else 0
            
            # Log training end event
            if self.log_preemption_events and hasattr(trainer.logger, 'experiment'):
                try:
                    trainer.logger.experiment.log({
                        "preemption/event": "training_end",
                        "preemption/count": self.preemption_count,
                        "preemption/training_duration": training_duration,
                        "preemption/final_epoch": trainer.current_epoch,
                        "preemption/final_step": trainer.global_step
                    })
                    print(f"Logged training end event to W&B (duration: {training_duration:.2f}s)")
                except Exception as e:
                    print(f"Warning: Could not log training end to W&B: {e}")
    
    def on_exception(self, trainer, pl_module, exception):
        """Called when an exception occurs during training."""
        if rank_zero_only.rank == 0:
            # This might indicate preemption
            if self.log_preemption_events and hasattr(trainer.logger, 'experiment'):
                try:
                    trainer.logger.experiment.log({
                        "preemption/event": "exception_occurred",
                        "preemption/count": self.preemption_count,
                        "preemption/exception_type": type(exception).__name__,
                        "preemption/exception_message": str(exception),
                        "preemption/current_epoch": trainer.current_epoch,
                        "preemption/current_step": trainer.global_step
                    })
                    print(f"Logged exception event to W&B: {type(exception).__name__}")
                except Exception as e:
                    print(f"Warning: Could not log exception to W&B: {e}")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when a checkpoint is saved."""
        if rank_zero_only.rank == 0:
            # Store preemption tracking info in checkpoint
            checkpoint['preemption_callback'] = {
                'preemption_count': self.preemption_count,
                'last_epoch': self.last_epoch,
                'last_global_step': self.last_global_step,
                'training_start_time': self.training_start_time
            }
            
            # Store W&B run ID for preemption support
            if hasattr(trainer.logger, 'experiment') and trainer.logger.experiment:
                try:
                    wandb_run_id = trainer.logger.experiment.id
                    checkpoint['preemption_callback']['wandb_run_id'] = wandb_run_id
                    print(f"Stored W&B run ID in checkpoint: {wandb_run_id}")
                except Exception as e:
                    print(f"Warning: Could not store W&B run ID in checkpoint: {e}")
            
            # Log checkpoint save event
            if self.log_preemption_events and hasattr(trainer.logger, 'experiment'):
                try:
                    trainer.logger.experiment.log({
                        "preemption/event": "checkpoint_saved",
                        "preemption/count": self.preemption_count,
                        "preemption/checkpoint_epoch": trainer.current_epoch,
                        "preemption/checkpoint_step": trainer.global_step
                    })
                except Exception as e:
                    print(f"Warning: Could not log checkpoint save to W&B: {e}")
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when a checkpoint is loaded."""
        if rank_zero_only.rank == 0:
            # Restore preemption tracking info from checkpoint
            if 'preemption_callback' in checkpoint:
                callback_data = checkpoint['preemption_callback']
                self.preemption_count = callback_data.get('preemption_count', 0)
                self.last_epoch = callback_data.get('last_epoch', 0)
                self.last_global_step = callback_data.get('last_global_step', 0)
                self.training_start_time = callback_data.get('training_start_time', None)
                
                print(f"Restored preemption tracking: count={self.preemption_count}, "
                      f"last_epoch={self.last_epoch}, last_step={self.last_global_step}")
                
                # Log checkpoint load event
                if self.log_preemption_events and hasattr(trainer.logger, 'experiment'):
                    try:
                        trainer.logger.experiment.log({
                            "preemption/event": "checkpoint_loaded",
                            "preemption/count": self.preemption_count,
                            "preemption/loaded_epoch": trainer.current_epoch,
                            "preemption/loaded_step": trainer.global_step,
                            "preemption/resume_reason": "preemption_detected"
                        })
                        print("Logged checkpoint load event to W&B")
                    except Exception as e:
                        print(f"Warning: Could not log checkpoint load to W&B: {e}")


