import faulthandler, signal, sys
faulthandler.enable(all_threads=True, file=sys.stderr)
faulthandler.register(signal.SIGUSR1, all_threads=True, file=sys.stderr)

import os
print("os.cpu_count()", os.cpu_count(), flush=True)
print("affinity", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, flush=True)

import time
import dotenv
import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import torch

# Register safe globals for PyTorch 2.6+ checkpoint loading
import pairit.cmdline._checkpoint_compat  # noqa: F401
import wandb
from pairit.cmdline._utils import (
    instantiate_callbacks,
    load_checkpoint_weights_only,
    find_latest_checkpoint_robust,
    log_preemption_event,
    configure_wandb_for_preemption
)
from pairit.cmdline._tokenizer_inheritance import (
    prepare_tokenizer_inheritance,
    get_checkpoint_path_for_inheritance,
)

dotenv.load_dotenv(".env")

def _normalize_ckpt_path(p):
    """Make ckpt paths robust to Hydra/bash quoting like '\"/abs/path.ckpt\"'."""
    if p is None:
        return None
    if not isinstance(p, str):
        return p
    s = p.strip()
    # Strip one layer of matching quotes (common when passing model.ckpt_path='\"...\"')
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _print_tokenizer_special_tokens(tokenizer):
    """Print tokenizer special tokens for debugging purposes."""
    print("\n" + "=" * 70)
    print("TOKENIZER SPECIAL TOKENS DEBUG INFO")
    print("=" * 70)
    
    try:
        # Basic special tokens
        print(f"Basic special tokens:")
        print(f"  cls_token: {getattr(tokenizer, 'cls_token', None)}")
        print(f"  eos_token: {getattr(tokenizer, 'eos_token', None)}")
        print(f"  pad_token: {getattr(tokenizer, 'pad_token', None)}")
        print(f"  sep_token: {getattr(tokenizer, 'sep_token', None)}")
        print(f"  unk_token: {getattr(tokenizer, 'unk_token', None)}")
        print(f"  mask_token: {getattr(tokenizer, 'mask_token', None)}")
        
        # Additional special tokens
        additional = getattr(tokenizer, "additional_special_tokens", None)
        if additional:
            print(f"\nAdditional special tokens ({len(additional)} total):")
            # Group by type for readability
            dataset_tokens = [t for t in additional if t.startswith("<ds")]
            prop_tokens = [t for t in additional if t.startswith("<p")]
            dist_tokens = [t for t in additional if t.startswith("<d") and not t.startswith("<ds")]
            other_tokens = [t for t in additional if not (t.startswith("<ds") or t.startswith("<p") or (t.startswith("<d") and not t.startswith("<ds")))]
            
            if dataset_tokens:
                print(f"  Dataset tokens ({len(dataset_tokens)}): {sorted(dataset_tokens)}")
            if prop_tokens:
                print(f"  Property prompt tokens ({len(prop_tokens)}): {sorted(prop_tokens)[:10]}{'...' if len(prop_tokens) > 10 else ''}")
            if dist_tokens:
                print(f"  Distance prompt tokens ({len(dist_tokens)}): {sorted(dist_tokens)}")
            if other_tokens:
                print(f"  Other tokens ({len(other_tokens)}): {sorted(other_tokens)[:10]}{'...' if len(other_tokens) > 10 else ''}")
        else:
            print(f"\nAdditional special tokens: None (empty list)")
        
        # Vocab size
        vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else "unknown"
        print(f"\nVocabulary size: {vocab_size}")
        
    except Exception as e:
        print(f"Error printing tokenizer info: {e}")
    
    print("=" * 70 + "\n")


def _print_logging_frequency_info(trainer, datamodule, cfg):
    """Print info about logging frequencies to help debug WandB alignment issues."""
    print("\n" + "=" * 70)
    print("LOGGING FREQUENCY INFO")
    print("=" * 70)
    
    # Get train dataset size
    try:
        datamodule.setup("fit")
        train_dataset = datamodule._train_dataset
        train_size = len(train_dataset) if train_dataset else "unknown"
    except Exception:
        train_size = "unknown"
    
    # Get batch size from config
    batch_size = cfg.data.get("batch_size", "unknown")
    
    # Calculate steps per epoch
    if train_size != "unknown" and batch_size != "unknown":
        # Account for drop_last
        drop_last = cfg.data.get("drop_last", True)
        if drop_last:
            steps_per_epoch = train_size // batch_size
        else:
            steps_per_epoch = (train_size + batch_size - 1) // batch_size
    else:
        steps_per_epoch = "unknown"
    
    # Get trainer config values
    val_check_interval = trainer.val_check_interval
    limit_val_batches = trainer.limit_val_batches
    log_every_n_steps = trainer.log_every_n_steps
    accumulate_grad_batches = trainer.accumulate_grad_batches
    num_devices = trainer.num_devices
    num_nodes = trainer.num_nodes
    
    print(f"\nDataset & Batching:")
    print(f"  Train dataset size: {train_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Drop last: {cfg.data.get('drop_last', True)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    print(f"\nDistributed Training:")
    print(f"  Num devices: {num_devices}")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Accumulate grad batches: {accumulate_grad_batches}")
    
    print(f"\nLogging Frequencies:")
    print(f"  log_every_n_steps: {log_every_n_steps}")
    print(f"    -> train_loss logged to WandB every {log_every_n_steps} training steps")
    
    print(f"\nValidation Schedule:")
    print(f"  val_check_interval: {val_check_interval}")
    print(f"  limit_val_batches: {limit_val_batches}")
    
    # Calculate validation frequency
    if steps_per_epoch != "unknown":
        if isinstance(val_check_interval, float) and val_check_interval <= 1.0:
            val_every_n_steps = int(steps_per_epoch * val_check_interval)
            validations_per_epoch = int(1.0 / val_check_interval) if val_check_interval > 0 else 0
            print(f"    -> val_check_interval={val_check_interval} means {validations_per_epoch} validation runs per epoch")
            print(f"    -> validation runs every ~{val_every_n_steps} training steps")
        elif isinstance(val_check_interval, int):
            val_every_n_steps = val_check_interval
            validations_per_epoch = steps_per_epoch // val_check_interval if val_check_interval > 0 else 0
            print(f"    -> val_check_interval={val_check_interval} means validation every {val_check_interval} steps")
            print(f"    -> ~{validations_per_epoch} validation runs per epoch")
        else:
            val_every_n_steps = "unknown"
            print(f"    -> Could not calculate validation frequency")
        
        print(f"  limit_val_batches: {limit_val_batches}")
        print(f"    -> Each validation run processes {limit_val_batches} batches")
    
    print(f"\nExpected WandB behavior:")
    print(f"  - train_loss: logged every {log_every_n_steps} steps")
    if steps_per_epoch != "unknown" and isinstance(val_check_interval, (int, float)):
        if isinstance(val_check_interval, float) and val_check_interval <= 1.0:
            val_every_n_steps = int(steps_per_epoch * val_check_interval)
        else:
            val_every_n_steps = val_check_interval
        print(f"  - val_loss_step: {limit_val_batches} points clustered at each validation (every ~{val_every_n_steps} steps)")
        print(f"  - val_loss_epoch: 1 point per validation run (every ~{val_every_n_steps} steps)")
    print(f"  - Note: WandB may downsample dense data for display")
    
    print("=" * 70 + "\n")


@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def train(cfg: DictConfig) -> tuple[pl.LightningModule, pl.LightningDataModule, list[pl.Callback]]:
    print('here at the very beginning')
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    print('now after log_cfg')

    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)
    print('now after instantiate setup')

    # Tokenizer inheritance: extract special tokens from checkpoint and inject into config
    # This MUST happen before datamodule/model instantiation so the tokenizer gets the right tokens
    ckpt_path_for_inheritance = get_checkpoint_path_for_inheritance(cfg)
    if ckpt_path_for_inheritance:
        # All ranks need to do the inheritance (config is local to each process)
        # Only rank 0 prints diagnostics
        is_rank_zero = (rank_zero_only.rank == 0)
        inheritance_summary = prepare_tokenizer_inheritance(
            cfg, ckpt_path_for_inheritance, verbose=is_rank_zero
        )
        if is_rank_zero and inheritance_summary["inherited"]:
            print(f"[Tokenizer Inheritance] Will inherit {inheritance_summary['num_tokens_inherited']} tokens from checkpoint")
    print('now after tokenizer inheritance check')

    datamodule = hydra.utils.instantiate(cfg.data)
    print('now after datamodule')

    model = hydra.utils.instantiate(cfg.model, _recursive_=True) #False)
    print('now after model')

    # Print tokenizer special tokens for debugging (only on rank 0)
    if rank_zero_only.rank == 0 and hasattr(model, 'tokenizer') and model.tokenizer is not None:
        _print_tokenizer_special_tokens(model.tokenizer)

    print(f'Num model parameters: {model.num_trainable_parameters}')
    
    # LoRA-specific setup
    if cfg.model.get("use_lora", False):
        print(f"LoRA enabled: {model.model.num_parameters():,} total parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"LoRA rank: {cfg.model.get('lora_rank', 16)}")
        print(f"LoRA alpha: {cfg.model.get('lora_alpha', 32)}")
        print(f"LoRA target modules: {cfg.model.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj'])}")

    if cfg.compile:
        model.compile()

    # PyTorch 2.6 defaults torch.load(weights_only=True), which cannot unpickle full training
    # state (tokenizers.AddedToken, omegaconf globals, optimizer) when Lightning resumes from a
    # preemption checkpoint -> UnpicklingError on requeue. We trust our own checkpoints, so
    # force weights_only=False for all torch.load in this process (idempotent).
    import torch as _torch
    if not getattr(_torch.load, "_pairit_wo_patched", False):
        _orig_torch_load = _torch.load

        def _torch_load_wo_false(*args, **kwargs):
            # FORCE weights_only=False (not setdefault): Lightning's resume path
            # (pl_load -> torch.load(..., weights_only=True)) passes it EXPLICITLY, so
            # setdefault was a no-op and the requeue still crashed on tokenizers.AddedToken.
            # We trust our own checkpoints, so always override.
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        _torch_load_wo_false._pairit_wo_patched = True
        _torch.load = _torch_load_wo_false

    # Handle preemption support - PROPER DISTRIBUTED TRAINING
    ckpt_path = _normalize_ckpt_path(cfg.model.ckpt_path)
    is_preemption_checkpoint = False  # Track if this is a preemption checkpoint
    wandb_run_id = None
    checkpoint_info = {}
    
    # --- Synchronize output_dir across ranks (DDP not yet initialized) ---
    # Hydra resolves ${now:...} independently per process, so each rank gets a
    # slightly different microsecond timestamp.  Use a temp file to let rank 0
    # share its output_dir with other ranks on the same node.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", 1)))
    print(f"[output_dir sync] LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, "
          f"env WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE')}, "
          f"output_dir={cfg.paths.output_dir}")
    if world_size > 1:
        import tempfile
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", os.environ.get("SLURM_JOB_ID", "default"))
        sync_dir = os.path.join(tempfile.gettempdir(), f"lobster_sync_{run_id}")
        os.makedirs(sync_dir, exist_ok=True)
        sync_file = os.path.join(sync_dir, "output_dir.txt")
        if local_rank == 0:
            with open(sync_file, "w") as f:
                f.write(cfg.paths.output_dir)
            print(f"Rank 0: wrote output_dir to {sync_file}")
        else:
            # Wait for rank 0 to write the file (should be near-instant on shared /tmp)
            for _ in range(300):  # up to 30s
                if os.path.exists(sync_file) and os.path.getsize(sync_file) > 0:
                    break
                time.sleep(0.1)
            with open(sync_file, "r") as f:
                cfg.paths.output_dir = f.read().strip()
        print(f"Rank {local_rank}: output_dir synced to {cfg.paths.output_dir}")

    if cfg.get("preempt_friendly", False):
        output_dir = cfg.paths.output_dir

        # removes timestamp from output_dir; this is bad if paths doesn't set the timestamp, but we assume here that it does
        without_time_output_dir = os.path.dirname(output_dir)
        print(f"Searching for checkpoints in: {without_time_output_dir}")

        # All ranks discover the checkpoint independently (read-only filesystem walks).
        # DDP is not yet initialized at this point, so broadcast is not possible.
        latest_ckpt, extracted_wandb_run_id, checkpoint_info = find_latest_checkpoint_robust(without_time_output_dir, cfg)

        print(f"latest_ckpt found in {without_time_output_dir}: {latest_ckpt}")
        print(f"extracted_wandb_run_id: {extracted_wandb_run_id}")

        if latest_ckpt:
            if 'epoch' in checkpoint_info and 'global_step' in checkpoint_info:
                epoch = checkpoint_info['epoch']
                step = checkpoint_info['global_step']
                print(f"Preemption enabled: Found latest checkpoint at {latest_ckpt}")
                print(f"  Checkpoint info: epoch={epoch}, step={step}")
            else:
                print(f"Warning: Checkpoint {latest_ckpt} missing epoch/step info")

            cfg = configure_wandb_for_preemption(cfg, extracted_wandb_run_id, checkpoint_info)

            if extracted_wandb_run_id:
                print(f"Preemption enabled: Using W&B run ID {extracted_wandb_run_id} for resumption")
            else:
                print("Preemption enabled: No W&B run ID found, will create new run")

            cfg.paths.output_dir = os.path.dirname(latest_ckpt)
            print(f"Preemption enabled: Reset output_dir to {cfg.paths.output_dir}")

            ckpt_path = latest_ckpt
            is_preemption_checkpoint = True
            wandb_run_id = extracted_wandb_run_id
        else:
            print("Preemption enabled: However, no existing checkpoint found, so starting fresh")

        print(f"Preemption state: ckpt_path={ckpt_path}, is_preemption={is_preemption_checkpoint}, wandb_id={wandb_run_id}, output_dir={cfg.paths.output_dir}")

    # Debug: check what ModelCheckpoint dirpath resolves to
    if cfg.get("callbacks") and cfg.callbacks.get("model_checkpoint"):
        print(f"[debug] ModelCheckpoint dirpath will resolve to: {cfg.callbacks.model_checkpoint.get('dirpath', 'NOT SET')}")
        print(f"[debug] cfg.paths.output_dir is currently: {cfg.paths.output_dir}")

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    if cfg.get("preempt_friendly", False):
        from pairit.callbacks import PreemptionCallback
        if not any(isinstance(cb, PreemptionCallback) for cb in callbacks):
            print("WARNING: preempt_friendly=True but PreemptionCallback not in callbacks. "
                  "W&B run ID will not be stored in checkpoints.")

    # Optional convenience flag: allow `++trainer.use_wandb=false` without passing it to Lightning Trainer.
    use_wandb = True
    try:
        if "use_wandb" in cfg.trainer:
            use_wandb = bool(cfg.trainer.get("use_wandb"))
            with open_dict(cfg.trainer):
                cfg.trainer.pop("use_wandb")
    except Exception:
        pass

    # Initialize logger AFTER preemption logic has modified cfg.logger
    # Ensure output_dir exists before WandbLogger tries to write to it
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    if (not cfg.dryrun) and (rank_zero_only.rank == 0) and use_wandb:
        logger = hydra.utils.instantiate(cfg.logger)
        # Don't call wandb.init() here - let WandbLogger handle it
        # this is a change from original pairit code
    else:
        logger = None

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    print('trainer strategy: ', trainer.strategy)
    print('trainer devices: ', trainer.device_ids)

    # Print logging frequency info for debugging
    if rank_zero_only.rank == 0:
        _print_logging_frequency_info(trainer, datamodule, cfg)

    # Update W&B config if logger is available
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        try:
            trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)
        except Exception as e:
            print(f"Warning: Could not update W&B config: {e}")
        
        # Log preemption event if this is a preemption checkpoint
        if is_preemption_checkpoint and checkpoint_info:
            log_preemption_event(trainer.logger, checkpoint_info)

    if not cfg.dryrun:
        # Print checkpoint info for debugging
        if ckpt_path:
            print(f"Resuming training from checkpoint: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if 'epoch' in checkpoint:
                    print(f"Checkpoint epoch: {checkpoint['epoch']}")
                if 'global_step' in checkpoint:
                    print(f"Checkpoint global_step: {checkpoint['global_step']}")
            except Exception as e:
                print(f"Warning: Could not read checkpoint info: {e}")
        
        # For LoRA models, don't pass ckpt_path to trainer since we load it manually in the model
        if cfg.model.get("use_lora", False) and cfg.model.get("ckpt_path") is not None:
            trainer.fit(model, datamodule=datamodule, ckpt_path=None)
        else:
            # Handle weights_only option for checkpoint loading
            # For preemption checkpoints, always load full state (including optimizer)
            # For regular checkpoints, respect the weights_only setting
            if ckpt_path and cfg.get("weights_only", True) and not is_preemption_checkpoint:
                # Load only model weights using the utility function
                load_checkpoint_weights_only(model, ckpt_path)
                # Train without passing ckpt_path since we already loaded the weights
                trainer.fit(model, datamodule=datamodule, ckpt_path=None)
            else:
                if is_preemption_checkpoint:
                    print(f"Preemption checkpoint: Loading full state (including optimizer) from {ckpt_path}")
                # PyTorch 2.6+ defaults torch.load to weights_only=True, which cannot
                # unpickle the full training state (optimizer/omegaconf globals) when
                # resuming. This breaks preemption requeue with UnpicklingError. Force
                # weights_only=False on the resume path; fall back for older Lightning
                # versions whose fit() lacks the weights_only kwarg.
                try:
                    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
                except TypeError:
                    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

        if cfg.run_test:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        wandb.finish()

    return model, datamodule, callbacks


if __name__ == "__main__": # HL because torchrun doesn't like pairit_train shortcut, so calling this script directly
    train()