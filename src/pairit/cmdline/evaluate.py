"""Standalone evaluation entry point for molecule improvement models.

Loads a trained checkpoint, instantiates a datamodule and evaluation callbacks
via Hydra, then runs the model through the specified data split.  The evaluation
loop delegates to ``pairit.evaluation.evaluate_model_with_callbacks`` which
iterates over the dataloader and invokes each callback (e.g.
``MoleculeValidationCallback``) to compute generation and property metrics.

Typical invocation (via the ``pairit_eval`` CLI or directly)::

    pairit_eval --config-name=zinc_pair_evaluate \\
        model.ckpt_path='/path/to/checkpoint.ckpt' \\
        split=test

The Hydra config stack resolves model, datamodule, and callback settings.
Tokenizer inheritance ensures that special tokens added during training
(e.g. property prompt tokens) are recovered from the checkpoint before the
datamodule or model are instantiated.
"""

# Register safe globals for PyTorch 2.6+ checkpoint loading FIRST
# This must happen before any torch.load calls (including in Lightning)
# NOTE: This import is intentionally unused in code but required for side effects
import pairit.cmdline._checkpoint_compat  # noqa: F401  # Imported for side effects

import logging
import subprocess
from pathlib import Path
from typing import Any

import hydra
import hydra.utils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from pairit.evaluation import evaluate_model_with_callbacks
from pairit.cmdline._tokenizer_inheritance import (
    prepare_tokenizer_inheritance,
    get_checkpoint_path_for_inheritance,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_checkpoint_metadata(ckpt_path: str) -> dict[str, Any]:
    """Extract training metadata from a Lightning checkpoint.

    Performs a lightweight load of the checkpoint to read ``global_step``
    and ``epoch`` without loading the full ``state_dict``.

    Parameters
    ----------
    ckpt_path : str
        Path to a ``.ckpt`` file saved by PyTorch Lightning.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``global_step`` and ``epoch`` keys.
        Returns zeros for both if the checkpoint cannot be read.
    """
    try:
        # Load only top-level keys (skip state_dict for speed)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return {
            "global_step": checkpoint.get("global_step", 0),
            "epoch": checkpoint.get("epoch", 0),
        }
    except Exception as e:
        logger.warning(f"Could not extract metadata from checkpoint: {e}")
        return {"global_step": 0, "epoch": 0}


def extract_run_name_from_path(ckpt_path: str) -> str:
    """Extract the run name from a checkpoint path.

    Parses paths of the form
    ``…/lobster_runs/{run_name}/{timestamp}/{checkpoint_file}`` and returns
    the ``{run_name}`` component.  Used as an OmegaConf custom resolver
    (``${extract_run_name:…}``) so that evaluation configs can automatically
    derive the run name for output directory organisation.

    Parameters
    ----------
    ckpt_path : str
        Absolute path to a ``.ckpt`` file.

    Returns
    -------
    str
        The run name directory immediately below ``lobster_runs/``.

    Raises
    ------
    ValueError
        If the path is empty, does not contain ``lobster_runs``, or has
        no directory component after ``lobster_runs``.
    """
    if not ckpt_path:
        raise ValueError("ckpt_path is empty or None")

    path = Path(ckpt_path)
    parts = path.parts

    # Accept both the current "runs" directory and the pre-rename "lobster_runs",
    # so checkpoint paths from older runs still resolve to a run_name.
    for marker in ("runs", "lobster_runs"):
        if marker not in parts:
            continue
        runs_idx = parts.index(marker)
        if runs_idx + 1 < len(parts):
            return parts[runs_idx + 1]
        raise ValueError(f"Could not extract run_name from path: {ckpt_path} (no directory after {marker!r})")

    raise ValueError(f"Could not find a 'runs' or 'lobster_runs' directory in path: {ckpt_path}")


def _run_pairit_eval(output_dir: str, pairit_cfg: DictConfig) -> None:
    """Run pairit_eval's evaluate.py on exported CSV files.

    Builds a CLI command from the Hydra config, mapping each config key to
    the corresponding ``scripts/evaluate.py`` CLI argument.  PairIT exports
    use canonical ordering, so ``--no-normalize`` is always passed.

    Parameters
    ----------
    output_dir : str
        Directory containing ``*_pairit.csv`` files produced by the callback.
    pairit_cfg : DictConfig
        Configuration block from ``cfg.pairit_eval`` with keys matching
        pairit_eval CLI arguments.
    """
    csv_files = sorted(Path(output_dir).rglob("*_pairit.csv"))
    if not csv_files:
        logger.warning("No *_pairit.csv files found in %s; skipping pairit_eval", output_dir)
        return

    script_path = str(getattr(pairit_cfg, "script_path", "evaluate.py"))

    for csv_file in csv_files:
        cmd = ["python", script_path, str(csv_file)]

        # PairIT exports are already in canonical ordering
        cmd.append("--no-normalize")

        # Dataset filter
        if getattr(pairit_cfg, "dataset", None):
            cmd.extend(["--dataset", str(pairit_cfg.dataset)])

        # Relevance mode

        # Number of generations per seed
        if getattr(pairit_cfg, "n_generations", None):
            cmd.extend(["--n-generations", str(pairit_cfg.n_generations)])

        # Property metrics
        if getattr(pairit_cfg, "property_metrics", False):
            cmd.append("--property-metrics")
        if getattr(pairit_cfg, "property_names", None):
            cmd.extend(["--property-names", str(pairit_cfg.property_names)])
        if getattr(pairit_cfg, "min_property_delta", None):
            cmd.extend(["--min-property-delta", str(pairit_cfg.min_property_delta)])

        # Distribution metrics
        if getattr(pairit_cfg, "distribution", False):
            cmd.append("--distribution")

        # Output options
        cmd.extend(["--output-dir", str(csv_file.parent)])
        if getattr(pairit_cfg, "save", True) is False:
            cmd.append("--no-save")
        if getattr(pairit_cfg, "output_format", None):
            cmd.extend(["--output-format", str(pairit_cfg.output_format)])

        # Reuse precomputed columns
        if getattr(pairit_cfg, "reuse_precomputed", False):
            cmd.append("--reuse-precomputed")

        # Suppress progress bars
        if getattr(pairit_cfg, "no_verbose", False):
            cmd.append("--no-verbose")

        # Extra training SMILES for multi-reference memorization (NAME:PATH format)
        extra_train = getattr(pairit_cfg, "extra_train_smiles", None)
        if extra_train:
            extra_args = []
            if isinstance(extra_train, (list, tuple)):
                extra_args = [str(s) for s in extra_train if s]
            elif isinstance(extra_train, dict):
                extra_args = [f"{k}:{v}" for k, v in extra_train.items() if k and v]
            if extra_args:  # only add flag if there are actual arguments
                cmd.append("--extra-train-smiles")
                cmd.extend(extra_args)

        # Data path overrides
        if getattr(pairit_cfg, "evaluator_dir", None):
            cmd.extend(["--evaluator-dir", str(pairit_cfg.evaluator_dir)])
        if getattr(pairit_cfg, "moleculenet_train_dir", None):
            cmd.extend(["--moleculenet-train-dir", str(pairit_cfg.moleculenet_train_dir)])
        if getattr(pairit_cfg, "moleculenet_test_dir", None):
            cmd.extend(["--moleculenet-test-dir", str(pairit_cfg.moleculenet_test_dir)])
        if getattr(pairit_cfg, "zinc250k_train_molecules", None):
            cmd.extend(["--zinc250k-train-molecules", str(pairit_cfg.zinc250k_train_molecules)])
        if getattr(pairit_cfg, "zinc250k_test_csv", None):
            cmd.extend(["--zinc250k-test-csv", str(pairit_cfg.zinc250k_test_csv)])

        logger.info("Running pairit_eval: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.stdout:
                logger.info("pairit_eval stdout:\n%s", result.stdout)
            if result.returncode != 0:
                logger.warning(
                    "pairit_eval exited %d\nstderr: %s",
                    result.returncode,
                    result.stderr,
                )
            else:
                logger.info("pairit_eval completed successfully for %s", csv_file.name)
        except Exception as e:
            logger.warning("Failed to run pairit_eval on %s: %s", csv_file, e)


# Register custom resolver for extracting run_name from checkpoint path
OmegaConf.register_new_resolver("extract_run_name", extract_run_name_from_path, replace=True)


@hydra.main(config_path="../hydra_config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """Run model evaluation with specified callbacks.

    Orchestrates the full evaluation pipeline:

    1. **Tokenizer inheritance** -- Extracts special tokens that were added
       during training (property prompt tokens, distance bins, etc.) from
       the checkpoint and injects them into the datamodule config *before*
       instantiation so the tokenizer vocabulary matches the saved model.
    2. **Callback instantiation** -- Each callback in ``cfg.callbacks``
       (typically ``MoleculeValidationCallback``) is created via Hydra.
    3. **Model loading** -- Loads from ``cfg.model.ckpt_path`` using
       Lightning's ``load_from_checkpoint``, with a manual fallback for
       PyTorch 2.6+ ``weights_only`` restrictions.
    4. **Datamodule setup** -- Instantiates and prepares the datamodule,
       then selects the requested split (train / val / test).
    5. **Evaluation loop** -- Delegates to
       ``evaluate_model_with_callbacks`` which iterates over the dataloader
       and invokes each callback.

    Parameters
    ----------
    cfg : DictConfig
        Resolved Hydra configuration.  Expected top-level keys:
        ``model``, ``datamodule``, ``callbacks``, ``split``,
        ``output_dir``.
    """

    import time as _time
    _phase_times: dict[str, float] = {}
    _eval_t0 = _time.time()

    logger.info("Starting model evaluation with callbacks")
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))

    # --- Extract checkpoint metadata and compute datapoints seen ---
    checkpoint_info: dict[str, Any] = {}
    ckpt_path = getattr(cfg.model, "ckpt_path", None) if hasattr(cfg, "model") else None
    if ckpt_path:
        ckpt_meta = extract_checkpoint_metadata(str(ckpt_path))
        global_step = ckpt_meta["global_step"]
        epoch = ckpt_meta["epoch"]

        # Determine batch size for datapoints calculation
        training_batch_size = getattr(cfg, "training_batch_size", None)
        if training_batch_size is None and hasattr(cfg, "datamodule"):
            training_batch_size = getattr(cfg.datamodule, "batch_size", None)
            if training_batch_size is not None:
                logger.warning(
                    "training_batch_size not set; falling back to datamodule.batch_size=%s "
                    "(may not match actual training batch size)",
                    training_batch_size,
                )

        accumulate_grad_batches = getattr(cfg, "training_accumulate_grad_batches", 1) or 1
        num_devices = getattr(cfg, "training_num_devices", 1) or 1

        datapoints_seen = None
        if training_batch_size is not None:
            datapoints_seen = int(global_step * training_batch_size * accumulate_grad_batches * num_devices)

        checkpoint_info = {
            "checkpoint_global_step": global_step,
            "checkpoint_epoch": epoch,
            "datapoints_seen": datapoints_seen,
            "training_batch_size": training_batch_size,
            "training_accumulate_grad_batches": accumulate_grad_batches,
            "training_num_devices": num_devices,
        }

        logger.info("=== Checkpoint Info ===")
        logger.info(f"  global_step: {global_step}")
        logger.info(f"  epoch: {epoch}")
        logger.info(f"  training_batch_size: {training_batch_size}")
        logger.info(f"  accumulate_grad_batches: {accumulate_grad_batches}")
        logger.info(f"  num_devices: {num_devices}")
        if datapoints_seen is not None:
            logger.info(f"  datapoints_seen: {datapoints_seen:,}")
        else:
            logger.info("  datapoints_seen: unknown (batch_size not available)")
        logger.info("======================")

    # Tokenizer inheritance: extract special tokens from checkpoint and inject into config
    # This MUST happen before datamodule/model instantiation so the tokenizer gets the right tokens
    # Handle both cfg.data (training) and cfg.datamodule (evaluation) config structures
    ckpt_path_for_inheritance = get_checkpoint_path_for_inheritance(cfg)
    if ckpt_path_for_inheritance:
        # For evaluation configs, datamodule is aliased as 'datamodule' (via data@datamodule)
        # The inheritance function expects cfg.data, so we need to adapt the config
        # Temporarily create cfg.data if it doesn't exist (pointing to datamodule)
        needs_cleanup = False
        if "data" not in cfg and "datamodule" in cfg:
            with open_dict(cfg):
                cfg.data = cfg.datamodule
            needs_cleanup = True

        try:
            inheritance_summary = prepare_tokenizer_inheritance(cfg, ckpt_path_for_inheritance, verbose=True)
            if inheritance_summary["inherited"]:
                logger.info(
                    f"[Tokenizer Inheritance] Inherited {inheritance_summary['num_tokens_inherited']} tokens from checkpoint"
                )

            # If we created cfg.data temporarily, copy the updated transform_fn back to datamodule
            if needs_cleanup and "transform_fn" in cfg.data:
                with open_dict(cfg):
                    cfg.datamodule.transform_fn = cfg.data.transform_fn
        finally:
            # Clean up temporary cfg.data if we created it
            if needs_cleanup and "data" in cfg:
                with open_dict(cfg):
                    del cfg.data

    _t0 = _time.time()
    logger.info("Instantiating callbacks...")
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]

    # Propagate pairit dataset name to callbacks so exported CSVs have the
    # correct ``dataset`` column (required by pairit_eval for filtering).
    pairit_dataset = None
    pairit_cfg_early = getattr(cfg, "pairit_eval", None)
    if pairit_cfg_early and getattr(pairit_cfg_early, "enabled", False):
        pairit_dataset = getattr(pairit_cfg_early, "dataset", None)
    if pairit_dataset:
        for cb in callbacks:
            if hasattr(cb, "pairit_dataset_name") and not cb.pairit_dataset_name:
                cb.pairit_dataset_name = str(pairit_dataset)
                logger.info(
                    "Set callback pairit_dataset_name='%s' from pairit_eval.dataset",
                    pairit_dataset,
                )

    _phase_times["Callback instantiation"] = _time.time() - _t0

    _t0 = _time.time()
    logger.info("Instantiating model...")
    if hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        logger.info(f"✓ Loading model from checkpoint: {cfg.model.ckpt_path}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        try:
            # Try loading with Lightning's method first (uses safe globals if registered)
            model = model_cls.load_from_checkpoint(cfg.model.ckpt_path)
            logger.info("✓ Successfully loaded model from checkpoint")
        except Exception as e:
            if "weights_only" in str(e) or "Unsupported global" in str(e) or "WeightsUnpickler" in str(e):
                # Fallback: manually load checkpoint with weights_only=False
                logger.warning(f"Failed to load with weights_only=True, trying manual load: {e}")
                checkpoint = torch.load(cfg.model.ckpt_path, map_location="cpu", weights_only=False)
                # Create a copy of model config without evaluation-specific keys
                # run_name and extra_name are evaluation metadata, not model constructor arguments
                # ckpt_path is a valid model param, but we set it to None since we're loading manually
                model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
                # Remove evaluation-specific keys that aren't model init args
                for key in ["run_name", "extra_name"]:
                    if key in model_cfg:
                        del model_cfg[key]
                # Set ckpt_path to None since we're handling checkpoint loading manually
                if "ckpt_path" in model_cfg:
                    model_cfg.ckpt_path = None
                # Instantiate model with filtered config
                model = hydra.utils.instantiate(model_cfg)
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                logger.info("✓ Successfully loaded model from checkpoint (manual load)")
            else:
                raise
    else:
        logger.info("⚠️  No checkpoint path found, instantiating fresh model")
        model = hydra.utils.instantiate(cfg.model)
        logger.info("⚠️  Using randomly initialized model weights")

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _phase_times["Model loading"] = _time.time() - _t0

    # ── Tokenizer sync: use the checkpoint's tokenizer for the datamodule ──
    # The checkpoint stores the exact tokenizer used during training (including
    # property prompt tokens with correct IDs). If the eval config builds a
    # tokenizer with different property_names or bin counts, token IDs will
    # mismatch, causing the model to receive garbled prompt inputs.
    # Fix: extract the checkpoint tokenizer and inject it into the datamodule.
    ckpt_tokenizer = None
    # Try from model object first
    ckpt_tokenizer = getattr(getattr(model, "transform_fn", None), "tokenizer", None)
    # Fallback: load from raw checkpoint (handles weights_only issues)
    if ckpt_tokenizer is None and hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        try:
            ckpt_path = str(cfg.model.ckpt_path).strip("'\"")
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            hp = raw_ckpt.get("hyper_parameters", {})
            tf = hp.get("transform_fn", None)
            if tf is not None and hasattr(tf, "tokenizer"):
                ckpt_tokenizer = tf.tokenizer
        except Exception as e:
            logger.warning("[Tokenizer Sync] Failed to extract tokenizer from checkpoint: %s", e)
    if ckpt_tokenizer is not None:
        logger.info(
            "[Tokenizer Sync] Extracted checkpoint tokenizer (vocab=%d)",
            len(ckpt_tokenizer),
        )
    else:
        logger.info("[Tokenizer Sync] No tokenizer found in checkpoint, using config tokenizer")

    _t0 = _time.time()
    logger.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Patch the datamodule's tokenizer with the checkpoint's tokenizer BEFORE setup().
    # setup() creates Dataset objects that tokenize data — they must use the correct tokenizer.
    if ckpt_tokenizer is not None:
        patched = False
        # The LightningDataModule stores its transform as _transform_fn
        tf = getattr(datamodule, "_transform_fn", None)
        if tf is not None and hasattr(tf, "tokenizer"):
            old_vocab = len(tf.tokenizer)
            tf.tokenizer = ckpt_tokenizer
            logger.info(
                "[Tokenizer Sync] Replaced datamodule._transform_fn.tokenizer "
                "(was %d tokens, now %d tokens)",
                old_vocab, len(ckpt_tokenizer),
            )
            patched = True
        if not patched:
            logger.warning(
                "[Tokenizer Sync] Could not find datamodule._transform_fn.tokenizer to patch"
            )
    datamodule.prepare_data()
    datamodule.setup()

    if cfg.split == "val":
        dataloader = datamodule.val_dataloader()
    elif cfg.split == "test":
        dataloader = datamodule.test_dataloader()
    elif cfg.split == "train":
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f"Invalid split: {cfg.split}")
    _phase_times["Data setup"] = _time.time() - _t0

    # Build metadata including checkpoint info and setup timing
    metadata = OmegaConf.to_container(cfg, resolve=True)
    if checkpoint_info:
        metadata["checkpoint_info"] = checkpoint_info
    metadata["setup_timing"] = _phase_times
    logger.info("Setup timing: %s", {k: f"{v:.1f}s" for k, v in _phase_times.items()})

    callback_results, report_path = evaluate_model_with_callbacks(
        callbacks=callbacks,
        model=model,
        dataloader=dataloader,
        output_dir=cfg.output_dir,
        metadata=metadata,
    )

    # --- Optional pairit_eval post-processing ---
    pairit_cfg = getattr(cfg, "pairit_eval", None)
    if pairit_cfg and getattr(pairit_cfg, "enabled", False):
        # Auto-populate pairit_eval config from callback settings if not explicitly set
        with open_dict(cfg):
            for cb in callbacks:
                if hasattr(cb, "property_names") and cb.property_names:
                    if not getattr(pairit_cfg, "property_names", None):
                        cfg.pairit_eval.property_names = ",".join(cb.property_names)
                    if not getattr(pairit_cfg, "property_metrics", False):
                        cfg.pairit_eval.property_metrics = True
                if hasattr(cb, "pairit_dataset_name") and cb.pairit_dataset_name:
                    if not getattr(pairit_cfg, "dataset", None):
                        cfg.pairit_eval.dataset = cb.pairit_dataset_name
        _t0 = _time.time()
        _run_pairit_eval(cfg.output_dir, pairit_cfg)
        pairit_elapsed = _time.time() - _t0
        _phase_times["pairit_eval"] = pairit_elapsed
        logger.info("pairit_eval completed in %.1fs", pairit_elapsed)

        # Append pairit_eval timing to the evaluation report
        if report_path and Path(report_path).exists():
            with open(report_path, "a") as f:
                if pairit_elapsed >= 3600:
                    dur = f"{pairit_elapsed / 3600:.1f}h"
                elif pairit_elapsed >= 60:
                    dur = f"{pairit_elapsed / 60:.1f}m"
                else:
                    dur = f"{pairit_elapsed:.1f}s"
                f.write(f"\n## pairit_eval Post-processing\n\n")
                f.write(f"| Phase | Duration |\n|-------|----------|\n")
                f.write(f"| pairit_eval | {dur} |\n\n")

    # --- Optional wandb logging ---
    wandb_project = getattr(cfg, "wandb_project", None)
    if wandb_project:
        try:
            import wandb

            wandb_entity = getattr(cfg, "wandb_entity", None)
            run_name = getattr(cfg.model, "run_name", None)
            extra_name = getattr(cfg.model, "extra_name", None)
            wandb_run_name = f"eval_{run_name}" if run_name else "eval"
            if extra_name:
                wandb_run_name = f"{wandb_run_name}_{extra_name}"

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=metadata,
            )

            # Collect metrics from successful callbacks
            wandb_metrics: dict[str, Any] = {}
            for cb_result in callback_results:
                if cb_result.success and isinstance(cb_result.results, dict):
                    wandb_metrics.update(cb_result.results)

            # Add checkpoint info as metrics
            wandb_metrics.update(checkpoint_info)

            wandb.log(wandb_metrics)
            wandb.finish()
            logger.info(f"Logged evaluation results to wandb project '{wandb_project}'")
        except ImportError:
            logger.warning("wandb not installed; skipping wandb logging")
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    _total_wall = _time.time() - _eval_t0
    logger.info(
        "Evaluation finished. Total wall time: %.1fs (%.1fm). "
        "Setup: %.1fs, Callbacks: %.1fs",
        _total_wall,
        _total_wall / 60,
        sum(_phase_times.values()),
        _total_wall - sum(_phase_times.values()),
    )
