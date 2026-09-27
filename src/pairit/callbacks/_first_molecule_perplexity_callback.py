import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
import lightning.pytorch as pl


def _has_trainer(pl_module) -> bool:
    """
    Safely check if pl_module has an attached trainer.
    
    The trainer property raises RuntimeError if not attached, so we need
    to check the private _trainer attribute or use try-except.
    """
    try:
        # Check if _trainer attribute exists and is not None
        return hasattr(pl_module, '_trainer') and pl_module._trainer is not None
    except Exception:
        return False


class FirstMoleculePerplexityCallback(Callback):
    """
    PyTorch Lightning callback for computing perplexity of just the first molecule in each pair.
    
    This callback extracts the first molecule from each pair in the validation batch
    and computes its perplexity separately from the full pair perplexity. This is useful
    for understanding how well the model performs on individual molecules vs. pairs.
    
    The callback operates by:
    1. Extracting the first molecule (up to the separator token) from each pair
    2. Computing the loss and perplexity for just that first molecule
    3. Logging the results to track first molecule performance separately
    """
    
    def __init__(self):
        """Initialize the FirstMoleculePerplexityCallback."""
        super().__init__()
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Called after each validation batch to compute perplexity of first molecules.
        
        This method extracts the first molecule from each pair in the batch and
        computes its perplexity separately from the full pair perplexity.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing the model and tokenizer.
        outputs
            Outputs from the validation step (unused in this callback).
        batch : dict
            Dictionary containing the batch data with keys:
            - "input_ids": torch.LongTensor, shape [batch_size, seq_len]
            - "attention_mask": torch.LongTensor, shape [batch_size, seq_len]
        batch_idx : int
            Index of the current validation batch.
        dataloader_idx : int
            Index of the validation dataloader when multiple are returned.
        """

        batch_size = batch["input_ids"].shape[0]
        total_first_mol_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # Extract first molecule (up to separator token)
            first_mol_ids, first_mol_mask = self._extract_first_molecule(
                batch["input_ids"][i, 0], 
                batch["attention_mask"][i, 0], 
                pl_module.tokenizer.sep_token_id
            )
            
            if first_mol_ids is not None and first_mol_ids.numel() > 0:
                # Compute loss for just the first molecule
                loss = self._compute_first_molecule_loss(pl_module, first_mol_ids, first_mol_mask)
                if loss is not None:
                    total_first_mol_loss += loss.item()
                    valid_samples += 1
        
        # Log the metrics only if trainer is attached
        # IMPORTANT: Always log on all ranks, even if valid_samples=0, to prevent DDP deadlock
        # Use NaN or 0 for invalid cases so sync_dist=True works correctly
        if _has_trainer(pl_module):
            if valid_samples > 0:
                avg_first_mol_loss = total_first_mol_loss / valid_samples
                first_mol_perplexity = torch.exp(torch.tensor(avg_first_mol_loss))
            else:
                # No valid samples: use NaN to indicate invalid, but still log to keep ranks aligned
                avg_first_mol_loss = float('nan')
                first_mol_perplexity = torch.tensor(float('nan'))
            
            # Move to GPU before logging for sync_dist
            avg_first_mol_loss_tensor = torch.tensor(avg_first_mol_loss, device=pl_module.device)
            first_mol_perplexity_tensor = first_mol_perplexity.to(pl_module.device)
            valid_samples_tensor = torch.tensor(valid_samples, device=pl_module.device)
            
            # Keep legacy metric names stable across multi-val-dataloader runs
            pl_module.log("val_first_molecule_loss", avg_first_mol_loss_tensor, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=True) # changed from False
            pl_module.log("val_first_molecule_perplexity", first_mol_perplexity_tensor, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=True)
            pl_module.log("val_first_molecule_samples", valid_samples_tensor.float(), sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=True)
    
    def _extract_first_molecule(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sep_token_id: int) -> tuple:
        """
        Extract the first molecule from a pair sequence.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs for a single sequence.
        attention_mask : torch.Tensor
            Attention mask for a single sequence.
        sep_token_id : int
            Token ID of the separator token.
            
        Returns
        -------
        tuple
            (first_mol_ids, first_mol_mask) or (None, None) if no separator found
        """
        # Find separator position
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        
        if len(sep_positions) == 0:
            # No separator found, return None
            return None, None
        
        sep_pos = sep_positions[0]
        
        # Extract up to (but not including) the separator
        first_mol_ids = input_ids[:sep_pos]
        first_mol_mask = attention_mask[:sep_pos]
        
        return first_mol_ids, first_mol_mask
    
    def _compute_first_molecule_loss(self, pl_module, first_mol_ids: torch.Tensor, first_mol_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for just the first molecule using the model's _compute_loss function.
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            The PyTorch Lightning module containing the model.
        first_mol_ids : torch.Tensor
            Token IDs for the first molecule.
        first_mol_mask : torch.Tensor
            Attention mask for the first molecule.
            
        Returns
        -------
        torch.Tensor or None
            The computed loss, or None if computation fails.
        """
        try:
            first_mol_ids = first_mol_ids.unsqueeze(0).to(pl_module.model.device)
            first_mol_mask = first_mol_mask.unsqueeze(0).to(pl_module.model.device)
            
            # Create a batch dictionary in the format expected by _compute_loss
            first_mol_batch = {
                "input_ids": first_mol_ids,
                "attention_mask": first_mol_mask
            }
            
            # Use the model's existing _compute_loss function
            with torch.no_grad():
                loss, _, _ = pl_module._compute_loss(first_mol_batch, compute_split_loss=False)
            
            return loss
            
        except Exception as e:
            print(f"Error computing first molecule loss: {e}")
            return None

    def evaluate(
        self,
        module: pl.LightningModule,
        dataloader: DataLoader,
        max_evals: int | None = 32,
    ) -> dict[str, float]:
        """
        Standalone evaluation that iterates the validation dataloader and computes
        average first-molecule loss and perplexity.

        Parameters
        ----------
        module : pl.LightningModule
            Model providing `_compute_loss` and `tokenizer`.
        trainer : Optional[pl.Trainer]
            If provided, the trainer's datamodule will be used to obtain the
            validation dataloader and metrics will be logged.

        Returns
        -------
        Dict[str, float]
            Dictionary containing average loss/perplexity and sample count.
        """
        if dataloader is None:
            print(f"Error in FirstMoleculePerplexityCallback's evaluate method: Dataloader is None, cannot evaluate")
            return {"val_first_molecule_loss": 0.0, "val_first_molecule_perplexity": 0.0, "val_first_molecule_samples": 0}

        module.eval()

        total_loss = 0.0
        valid_samples = 0
        processed = 0

        for batch in dataloader:
            if isinstance(max_evals, int) and processed >= max_evals:
                break
            input_ids = batch["input_ids"].to(module.model.device)
            attention_mask = batch["attention_mask"].to(module.model.device)

            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                if isinstance(max_evals, int) and processed >= max_evals:
                    break
                # Support both [B, S] and [B, 2, S] layouts
                ids_i = input_ids[i]
                mask_i = attention_mask[i]
                if ids_i.dim() == 2:
                    ids_i = ids_i[0]
                    mask_i = mask_i[0]

                first_ids, first_mask = self._extract_first_molecule(ids_i, mask_i, module.tokenizer.sep_token_id)
                if first_ids is None or first_ids.numel() == 0:
                    continue

                try:
                    decoded_first = module.tokenizer.decode(first_ids.reshape(-1).tolist())
                    decoded_first_clean = ''.join(decoded_first.split(' ')).strip()
                    print(f"[FirstMoleculePerplexityCallback] First molecule (decoded): {decoded_first_clean}")
                except Exception as e:
                    print(f"[FirstMoleculePerplexityCallback] Warning: failed to decode first molecule: {e}")

                loss = self._compute_first_molecule_loss(module, first_ids, first_mask)
                if loss is not None:
                    try:
                        sample_ppl = float(torch.exp(loss).item())
                        print(f"[FirstMoleculePerplexityCallback] First molecule perplexity: {sample_ppl:.4f}")
                    except Exception as e:
                        print(f"[FirstMoleculePerplexityCallback] Warning: failed to compute perplexity: {e}")
                    total_loss += float(loss.item())
                    valid_samples += 1
                processed += 1

        avg_loss = (total_loss / valid_samples) if valid_samples > 0 else 0.0
        perplexity = float(torch.exp(torch.tensor(avg_loss))) if valid_samples > 0 else 0.0

        metrics = {
            "val_first_molecule_loss": avg_loss,
            "val_first_molecule_perplexity": perplexity,
            "val_first_molecule_samples": valid_samples,
        }

        return metrics

