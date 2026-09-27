"""
LoRA-specific callback for logging LoRA parameters and metrics during training.
"""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class LoRACallback(Callback):
    """
    Callback for logging LoRA-specific metrics during training.
    
    This callback logs LoRA configuration parameters and training metrics
    to help monitor LoRA fine-tuning performance.
    
    Parameters
    ----------
    log_every_n_steps : int, default=100
        How often to log LoRA metrics during training.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        """
        Log LoRA-specific metrics after each training batch.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module (model).
        outputs : dict
            The outputs from the training step.
        batch : dict
            The current batch.
        batch_idx : int
            The index of the current batch.
        """
        if batch_idx % self.log_every_n_steps == 0:
            # Log LoRA-specific metrics if LoRA is enabled
            if hasattr(pl_module, 'use_lora') and pl_module.use_lora:
                # Log LoRA configuration parameters
                pl_module.log("lora/rank", float(pl_module.lora_rank), sync_dist=True)
                pl_module.log("lora/alpha", float(pl_module.lora_alpha), sync_dist=True)
                pl_module.log("lora/dropout", float(pl_module.lora_dropout), sync_dist=True)
                
                # Log trainable parameter count
                trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in pl_module.parameters())
                pl_module.log("lora/trainable_parameters", float(trainable_params), sync_dist=True)
                pl_module.log("lora/total_parameters", float(total_params), sync_dist=True)
                pl_module.log("lora/trainable_ratio", float(trainable_params) / float(total_params), sync_dist=True)
                
                # Log LoRA target modules count
                pl_module.log("lora/target_modules_count", float(len(pl_module.target_modules)), sync_dist=True)
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log LoRA configuration at the start of training.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module (model).
        """
        if hasattr(pl_module, 'use_lora') and pl_module.use_lora:
            print(f"LoRA Configuration:")
            print(f"  Rank: {pl_module.lora_rank}")
            print(f"  Alpha: {pl_module.lora_alpha}")
            print(f"  Dropout: {pl_module.lora_dropout}")
            print(f"  Target Modules: {pl_module.target_modules}")
            print(f"  Bias: {pl_module.lora_bias}")
            print(f"  Task Type: {pl_module.lora_task_type}")
            
            # Log initial parameter counts
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in pl_module.parameters())
            print(f"  Trainable Parameters: {trainable_params:,}")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Ratio: {trainable_params/total_params:.4f}")
