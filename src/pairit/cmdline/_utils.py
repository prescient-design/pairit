import glob
import os
import hydra
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from omegaconf import DictConfig

# Register safe globals for PyTorch 2.6+ checkpoint loading
import pairit.cmdline._checkpoint_compat  # noqa: F401


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        print("[instantiate_callbacks] No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("[instantiate_callbacks] Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"[instantiate_callbacks] Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_plugins(plugins_cfg: DictConfig) -> list[Callback]:
    """Instantiates plugins from config."""
    plugins: list[PrecisionPlugin] = []

    if not plugins_cfg:
        print("[instantiate_plugins] No plugin configs found! Skipping..")
        return plugins

    if not isinstance(plugins_cfg, DictConfig):
        raise TypeError("[instantiate_plugins] Plugins config must be a DictConfig!")

    for _, cb_conf in plugins_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"[instantiate_plugins] Instantiating plugin <{cb_conf._target_}>")
            plugins.append(hydra.utils.instantiate(cb_conf))

    return plugins


def load_checkpoint_weights_only(model, checkpoint_path: str) -> None:
    """
    Load only model weights from a checkpoint, ignoring optimizer state.
    
    This is a utility function that can be used with any PyTorch Lightning model
    to load only the model weights without the optimizer state.
    
    Args:
        model: The PyTorch Lightning model to load weights into
        checkpoint_path: Path to the checkpoint file
    """
    print(f"Loading only model weights from checkpoint: {checkpoint_path}")
    
    # Load checkpoint manually to control what gets loaded
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Model weights loaded successfully (optimizer state ignored)")
    else:
        print("Warning: No state_dict found in checkpoint")
        raise ValueError(f"No state_dict found in checkpoint: {checkpoint_path}")


def load_model_from_checkpoint(
    model_class,
    checkpoint_path: str,
    weights_only: bool = False,
    **model_kwargs
):
    """
    Load a model from checkpoint with optional weights_only loading.
    
    This is a more pythonic way to handle checkpoint loading that works with
    any PyTorch Lightning model class.
    
    Args:
        model_class: The model class to instantiate
        checkpoint_path: Path to the checkpoint file
        weights_only: If True, load only model weights, not optimizer state
        **model_kwargs: Additional keyword arguments to pass to model_class
        
    Returns:
        The loaded model instance
    """
    if weights_only:
        # First instantiate the model with the provided kwargs
        model = model_class(**model_kwargs)
        # Then load only the weights
        load_checkpoint_weights_only(model, checkpoint_path)
        return model
    else:
        # Use PyTorch Lightning's built-in load_from_checkpoint
        return model_class.load_from_checkpoint(checkpoint_path, **model_kwargs)


def find_latest_checkpoint_robust(output_dir: str, cfg: DictConfig = None) -> tuple[str | None, str | None, dict]:
    """
    Robust checkpoint discovery with validation and W&B run ID extraction.
    
    This function searches for PyTorch Lightning checkpoint files in the specified directory,
    validates their integrity, and extracts W&B run IDs for preemption support. It uses
    multiple fallback methods to find W&B run IDs stored in different parts of the checkpoint.
    
    The function is designed for preemption scenarios where you need to resume training
    with the same W&B run ID to maintain continuity in your experiment tracking.
    
    Args:
        output_dir: Directory to search for checkpoint files (e.g., "outputs/experiment")
        
    Returns:
        Tuple containing:
        - checkpoint_path: Path to the latest valid checkpoint, or None if none found
        - wandb_run_id: Extracted W&B run ID from the checkpoint, or None if not found
        - checkpoint_info: Dictionary containing checkpoint metadata (epoch, global_step, etc.)
        
    Behavior:
        1. Scans directory for *.ckpt files
        2. Validates each checkpoint by loading it and checking for required fields
        3. Sorts checkpoints by global_step (most reliable ordering)
        4. Extracts W&B run ID using multiple fallback methods:
           - Method 1: From callbacks data (most common)
           - Method 2: From logger states (fallback)
           - Method 3: From hyperparameters (last resort)
        5. Returns the latest valid checkpoint with its W&B run ID
        
    Example:
        >>> ckpt_path, run_id, info = find_latest_checkpoint_robust("outputs/my_experiment")
        >>> if ckpt_path:
        ...     print(f"Resuming from {ckpt_path} with W&B run {run_id}")
        ...     print(f"Checkpoint epoch: {info['epoch']}, step: {info['global_step']}")
    """
    if not os.path.exists(output_dir):
        return None, None, {}
    
    # Find all checkpoint files recursively with validation
    ckpt_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_file = os.path.join(root, file)
                try:
                    # Validate checkpoint integrity
                    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                    if 'epoch' in checkpoint and 'global_step' in checkpoint:
                        ckpt_files.append((ckpt_file, checkpoint))
                        print(f"Found valid checkpoint: {ckpt_file} (epoch={checkpoint.get('epoch', 0)}, step={checkpoint.get('global_step', 0)})")
                except Exception as e:
                    print(f"Warning: Skipping corrupted checkpoint {ckpt_file}: {e}")
    
    if not ckpt_files:
        return None, None, {}
    
    # Sort by global_step (more reliable than mtime)
    ckpt_files.sort(key=lambda x: x[1].get('global_step', 0), reverse=True)
    latest_ckpt, latest_checkpoint = ckpt_files[0]
    
    # Extract W&B run ID with multiple fallback methods
    wandb_run_id = None
    
    # Method 1: From preemption_callback (most reliable for new checkpoints)
    if 'preemption_callback' in latest_checkpoint:
        preemption_data = latest_checkpoint['preemption_callback']
        if 'wandb_run_id' in preemption_data:
            wandb_run_id = preemption_data['wandb_run_id']
            print(f"Found W&B run ID in preemption_callback: {wandb_run_id}")
    
    # Method 2: From callbacks (legacy location)
    if not wandb_run_id and 'callbacks' in latest_checkpoint:
        for callback_data in latest_checkpoint['callbacks'].values():
            if 'wandb_run_id' in callback_data:
                wandb_run_id = callback_data['wandb_run_id']
                print(f"Found W&B run ID in callbacks: {wandb_run_id}")
                break
    
    # Method 3: From logger states (fallback for older checkpoints)
    if not wandb_run_id and 'logger_states' in latest_checkpoint:
        for logger_state in latest_checkpoint['logger_states'].values():
            if 'wandb_run_id' in logger_state:
                wandb_run_id = logger_state['wandb_run_id']
                print(f"Found W&B run ID in logger states: {wandb_run_id}")
                break
    
    # Method 4: From hyperparameters (last resort)
    if not wandb_run_id and 'hyper_parameters' in latest_checkpoint:
        hyperparams = latest_checkpoint['hyper_parameters']
        if 'wandb_run_id' in hyperparams:
            wandb_run_id = hyperparams['wandb_run_id']
            print(f"Found W&B run ID in hyperparameters: {wandb_run_id}")
    
    if not wandb_run_id:
        print("Warning: Could not extract W&B run ID from checkpoint")
        print("Attempting to find most recent W&B run with same name...")
        
        # Fallback: Try to find the most recent W&B run with the same name
        if cfg is not None:
            try:
                wandb_run_id = find_most_recent_wandb_run_by_name(cfg)
                if wandb_run_id:
                    print(f"Found most recent W&B run with same name: {wandb_run_id}")
                else:
                    print("No W&B run found with matching name")
            except Exception as e:
                print(f"Warning: Could not search for W&B run by name: {e}")
        else:
            print("Warning: No config provided for W&B run name fallback search")
    
    return latest_ckpt, wandb_run_id, latest_checkpoint


def find_latest_checkpoint(output_dir: str, cfg: DictConfig = None) -> tuple[str | None, str | None]:
    """
    Legacy function for backward compatibility.
    
    This is a simplified wrapper around find_latest_checkpoint_robust that only
    returns the checkpoint path and W&B run ID, without the full checkpoint info.
    
    Args:
        output_dir: Directory to search for checkpoint files
        cfg: Optional config for W&B run name fallback
        
    Returns:
        Tuple of (checkpoint_path, wandb_run_id) or (None, None) if no checkpoints found
    """
    ckpt_path, wandb_run_id, _ = find_latest_checkpoint_robust(output_dir, cfg)
    return ckpt_path, wandb_run_id


def log_preemption_event(wandb_logger, checkpoint_info: dict, preemption_count: int = 0):
    """
    Log preemption events directly to W&B for better experiment tracking.
    
    This function logs preemption events to W&B with detailed metadata about
    the checkpoint being resumed from. This helps track preemption patterns
    and provides visibility into training interruptions.
    
    Args:
        wandb_logger: W&B logger instance (must have 'experiment' attribute)
        checkpoint_info: Dictionary containing checkpoint metadata (epoch, global_step, etc.)
        preemption_count: Number of preemptions that have occurred (for tracking)
        
    Behavior:
        - Logs preemption event with timestamp
        - Records checkpoint epoch and global step
        - Tracks preemption count for analysis
        - Adds resume reason for context
        - Handles errors gracefully (prints warning if logging fails)
        
    Example:
        >>> log_preemption_event(wandb_logger, {"epoch": 5, "global_step": 1000}, 2)
        # Logs: preemption/event="resume", preemption/count=2, etc.
    """
    if wandb_logger and hasattr(wandb_logger, 'experiment'):
        try:
            wandb_logger.experiment.log({
                "preemption/event": "resume",
                "preemption/count": preemption_count,
                "preemption/checkpoint_epoch": checkpoint_info.get('epoch', 0),
                "preemption/checkpoint_global_step": checkpoint_info.get('global_step', 0),
                "preemption/checkpoint_timestamp": checkpoint_info.get('timestamp', ''),
                "preemption/resume_reason": "automatic_checkpoint_found"
            })
            print(f"Logged preemption event to W&B: count={preemption_count}, epoch={checkpoint_info.get('epoch', 0)}, step={checkpoint_info.get('global_step', 0)}")
        except Exception as e:
            print(f"Warning: Could not log preemption event to W&B: {e}")


def configure_wandb_for_preemption(cfg: DictConfig, wandb_run_id: str, checkpoint_info: dict):
    """
    Configure W&B logger settings for preemption scenarios.
    
    This function modifies the W&B logger configuration to properly handle
    preemption scenarios. It sets the appropriate resume mode and run ID
    to ensure W&B continues logging to the same run after preemption.
    
    Args:
        cfg: Hydra configuration object (will be modified in-place)
        wandb_run_id: W&B run ID to resume (if available)
        checkpoint_info: Dictionary containing checkpoint metadata
        
    Behavior:
        - If wandb_run_id is provided: Sets resume="must" and id=wandb_run_id
        - If no wandb_run_id: Sets resume=None and id=None (creates new run)
        - Adds "preemption_resumed" tag to distinguish resumed runs
        - Sets detailed notes about the preemption (epoch, step info)
        - Modifies cfg.logger in-place (no return value needed)
        
    Example:
        >>> configure_wandb_for_preemption(cfg, "abc123", {"epoch": 5, "global_step": 1000})
        # cfg.logger now has: resume="must", id="abc123", tags=["preemption_resumed"], etc.
    """
    if "logger" not in cfg:
        cfg.logger = {}
    
    # Set resume mode - only resume if we have a specific run ID
    if wandb_run_id:
        cfg.logger.resume = "must"  # Must resume the specific run ID
        cfg.logger.id = wandb_run_id
    else:
        cfg.logger.resume = None  # Don't resume if no run ID
        cfg.logger.id = None
    
    # Add preemption metadata
    existing_tags = cfg.logger.get("tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = []
    cfg.logger.tags = existing_tags + ["preemption_resumed"]
    
    # Add detailed notes about the preemption
    epoch = checkpoint_info.get('epoch', 0)
    step = checkpoint_info.get('global_step', 0)
    cfg.logger.notes = f"Resumed from preemption. Checkpoint: epoch={epoch}, step={step}"
    
    return cfg


def find_most_recent_wandb_run_by_name(cfg: DictConfig) -> str | None:
    """
    Find the most recent W&B run with the same name as the current run.
    
    This function serves as a fallback when the W&B run ID cannot be extracted
    from the checkpoint. It searches for the most recent W&B run that has the
    same name as the current run configuration.
    
    Returns:
        W&B run ID of the most recent matching run, or None if not found
        
    Behavior:
        1. Gets the current W&B run name from environment or config
        2. Searches W&B API for runs with the same name
        3. Returns the most recent run ID (by creation time)
        4. Handles cases where W&B is not initialized or API is unavailable
        
    Example:
        >>> run_id = find_most_recent_wandb_run_by_name()
        >>> if run_id:
        ...     print(f"Found recent W&B run: {run_id}")
        ... else:
        ...     print("No matching W&B run found")
    """
    print(f"Searching for W&B runs by name.")
    print(f"Config: {cfg}")
    try:
        # Get the run name from the config
        run_name = None
        
        # Try to get run name from logger config
        if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'name') and cfg.logger.name:
            run_name = cfg.logger.name
        elif hasattr(cfg, 'logger') and hasattr(cfg.logger, 'run_name') and cfg.logger.run_name:
            run_name = cfg.logger.run_name
        elif type(cfg) == str:
            run_name = cfg
        
        if not run_name:
            print("Warning: Could not determine W&B run name from config for fallback search")
            return None
        
        print(f"Searching for W&B runs with name: {run_name}")
        
        # Search for runs with the same name
        api = wandb.Api()
        
        # Get entity and project with multiple fallback methods
        entity = None
        project = None
        
        # Method 1: From config (if explicitly set)
        if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'entity') and cfg.logger.entity:
            entity = cfg.logger.entity
        if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'project') and cfg.logger.project:
            project = cfg.logger.project
        
        # Method 2: From environment variables (if config is None or not set)
        if not entity:
            entity = os.environ.get('WANDB_ENTITY', None)
        if not project:
            project = os.environ.get('WANDB_PROJECT', None)
        
        # Method 3: From W&B settings (stored during login)
        if not entity or not project:
            try:
                settings = wandb.api.settings()
                if not entity:
                    entity = settings.get('entity', None)
                if not project:
                    project = settings.get('project', None)
            except Exception as e:
                print(f"Warning: Could not get W&B settings: {e}")

        if not entity or not project:
            print(f"Warning: Could not determine W&B entity/project. Entity: {entity}, Project: {project}")
            return None

        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        
        if not runs:
            print(f"No W&B runs found with name: {run_name}")
            return None
        
        # Sort by creation time and get the most recent
        runs = sorted(runs, key=lambda x: x.created_at, reverse=True)
        most_recent_run = runs[0]
        
        print(f"Found {len(runs)} W&B runs with name '{run_name}'")
        print(f"Most recent run: {most_recent_run.id} (created: {most_recent_run.created_at})")
        
        return most_recent_run.id
        
    except Exception as e:
        print(f"Warning: Could not search W&B runs by name: {e}")
        return None
