#!/usr/bin/env python
import inspect
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import yaml
from torch.utils.data import DataLoader
from upath import UPath

logger = logging.getLogger(__name__)


@dataclass
class CallbackResult:
    """Structured result from a callback evaluation."""

    class_name: str
    results: Any
    success: bool
    error_message: str | None = None


def _convert_to_yaml_friendly(obj: Any) -> Any:
    """Convert objects to YAML-friendly types for nice display.
    Test if YAML displays it nicely by dumping and checking the result

    Parameters
    ----------
    obj : Any
        Object that may not display nicely in YAML

    Returns
    -------
    Any
        Object with non-YAML-friendly types converted to strings
    """
    try:
        yaml_str = yaml.dump(obj, default_flow_style=False)

        # If it contains binary data or complex object markers, convert to string
        if any(marker in yaml_str for marker in ["!!binary", "!!python/object", "!!python/tuple"]):
            return str(obj)

        return obj

    except (yaml.representer.RepresenterError, TypeError):
        return str(obj)


def _format_results_for_markdown(results: Any) -> str:
    """Format results nicely for markdown display.

    Parameters
    ----------
    results : Any
        The results to format

    Returns
    -------
    str
        Formatted markdown string
    """
    # Convert non-YAML-friendly objects to strings first
    results = _convert_to_yaml_friendly(results)

    if isinstance(results, dict):
        # Use YAML formatting for dictionaries
        return f"```yaml\n{yaml.dump(results, default_flow_style=False, sort_keys=False)}```"
    elif isinstance(results, (list, tuple)):
        # Use YAML formatting for lists/tuples
        return f"```yaml\n{yaml.dump(results, default_flow_style=False, sort_keys=False)}```"
    else:
        # For other types, use regular code block
        return f"```\n{results}\n```"


def _extract_key_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Extract the most important metrics for the report summary.

    Pulls out validity, memorization, improvement, V&I&NM, and tanimoto
    metrics from the full results dict.

    Parameters
    ----------
    results : dict[str, Any]
        Full metrics dictionary from a callback.

    Returns
    -------
    dict[str, Any]
        Subset of metrics most useful for a quick summary.
    """
    if not isinstance(results, dict):
        return {}

    key_patterns = [
        "validity_fraction_conditional",
        "memorization_rate_conditional",
        "property_frac_improved/",
        "V_N_I_rate/",
        "V_N_I_count/",
        "V_N_I_total/",
        "tanimoto_distance_conditional",
        "similarity_to_seed_mean",
        "cond_pairs_count",
        "cond_valid_count",
        "prompt_direction_agreement_rate_non_memorized",
    ]
    summary = {}
    for k, v in results.items():
        for pattern in key_patterns:
            if pattern in k:
                summary[k] = v
                break
    return summary


def _generate_evaluation_report(
    callback_results: list[CallbackResult],
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
) -> Path:
    """Generate a markdown report from evaluation results.

    Parameters
    ----------
    callback_results : list[CallbackResult]
        List of callback evaluation results
    output_dir : str | Path
        Directory to save the report
    metadata : dict[str, Any] | None
        Arbitrary metadata to include in the report (e.g., config information)
    timing : dict[str, float] | None
        Timing information (seconds) for each phase of evaluation

    Returns
    -------
    Path
        Path to the generated report
    """
    markdown_report = "# Model Evaluation Report\n\n"
    markdown_report += f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add timing section if provided
    if timing:
        markdown_report += "## Timing\n\n"
        markdown_report += "| Phase | Duration |\n|-------|----------|\n"
        for phase, seconds in timing.items():
            if seconds >= 3600:
                duration_str = f"{seconds / 3600:.1f}h"
            elif seconds >= 60:
                duration_str = f"{seconds / 60:.1f}m"
            else:
                duration_str = f"{seconds:.1f}s"
            markdown_report += f"| {phase} | {duration_str} |\n"
        markdown_report += "\n"

    # Add key metrics summary
    for i, result in enumerate(callback_results):
        if result.success and isinstance(result.results, dict):
            key_metrics = _extract_key_metrics(result.results)
            if key_metrics:
                markdown_report += "## Key Metrics Summary\n\n"
                markdown_report += "| Metric | Value |\n|--------|-------|\n"
                for k, v in sorted(key_metrics.items()):
                    if isinstance(v, float):
                        markdown_report += f"| {k} | {v:.4f} |\n"
                    else:
                        markdown_report += f"| {k} | {v} |\n"
                markdown_report += "\n"
                break  # Only first successful callback

    # Add metadata section if provided
    if metadata:
        markdown_report += "## Metadata\n\n"
        # Format metadata nicely using the same function as results
        markdown_report += _format_results_for_markdown(metadata) + "\n\n"

    # Add section for each callback's results
    markdown_report += "## Evaluation Results\n\n"

    for i, result in enumerate(callback_results):
        # Use class name with index for the section header
        markdown_report += f"### {result.class_name} #{i + 1}\n\n"

        # Add results or error message
        if result.success:
            # If result is a path, assume it's an image and include it in the markdown
            if isinstance(result.results, str | Path | UPath) and Path(result.results).exists():
                markdown_report += f"![{result.class_name} Visualization]({result.results})\n\n"
            else:
                # Format results nicely
                markdown_report += _format_results_for_markdown(result.results) + "\n\n"
        else:
            markdown_report += f"**Error:** {result.error_message}\n\n"

    report_path = output_dir / "evaluation_report.md"

    logger.info(f"Writing evaluation report to {report_path}")

    with open(report_path, "w") as f:
        f.write(markdown_report)

    return report_path


def evaluate_model_with_callbacks(
    callbacks: Sequence[L.Callback],
    model: L.LightningModule,
    dataloader: DataLoader | None = None,
    output_dir: str | Path | UPath = "evaluation_results",
    metadata: dict[str, Any] | None = None,
) -> tuple[list[CallbackResult], Path]:
    """Evaluate a model with various callbacks and generate a markdown report.

    Callbacks are expected to have an `evaluate` method that takes a model and optionally a dataloader.
    If the callback evaluation method requires a dataloader, it must be provided.

    Parameters
    ----------
    callbacks : Sequence[L.Callback]
        List of callbacks to evaluate
    model : L.LightningModule
        The model to evaluate
    dataloader : DataLoader | None
        The dataloader to use for evaluation, required for some callbacks
    output_dir : str | Path
        Directory to save evaluation results
    metadata : dict[str, Any] | None
        Arbitrary metadata to include in the report (e.g., config information)

    Returns
    -------
    tuple[list[CallbackResult], Path]
        List of callback results and path to the generated report
    """
    eval_start = time.time()
    logger.info("Starting model evaluation with callbacks")
    if str(output_dir).startswith("s3://"):
        raise NotImplementedError("S3 output is not supported yet")

    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    callback_results = []
    timing: dict[str, float] = {}

    for callback in callbacks:
        try:
            class_name = callback.__class__.__name__

            logger.info(f"Evaluating with callback: {class_name}")

            if not hasattr(callback, "evaluate") or not callable(callback.evaluate):
                error_msg = f"Callback {class_name} does not have an evaluate method"
                logger.error(error_msg)
                callback_results.append(
                    CallbackResult(class_name=class_name, results=None, success=False, error_message=error_msg)
                )
                continue

            # Inspect signature if the callback evaluation method requires a dataloader
            callback_signature = inspect.signature(callback.evaluate)

            cb_start = time.time()
            if "dataloader" in callback_signature.parameters:
                if dataloader is None:
                    error_msg = f"Callback {class_name} requires a dataloader but none was provided"
                    logger.error(error_msg)
                    callback_results.append(
                        CallbackResult(class_name=class_name, results=None, success=False, error_message=error_msg)
                    )
                    continue

                results = callback.evaluate(model, dataloader=dataloader)
            else:
                results = callback.evaluate(model)

            cb_elapsed = time.time() - cb_start
            timing[class_name] = cb_elapsed
            logger.info(f"Successfully evaluated with {class_name} in {cb_elapsed:.1f}s")
            logger.info(f"Callback results: {results}")

            callback_results.append(CallbackResult(class_name=class_name, results=results, success=True))

        except Exception as e:
            cb_elapsed = time.time() - cb_start
            timing[class_name] = cb_elapsed
            logger.exception(f"Error in {class_name}: {e}")
            callback_results.append(
                CallbackResult(class_name=class_name, results=None, success=False, error_message=str(e))
            )

    total_elapsed = time.time() - eval_start
    timing["Total"] = total_elapsed
    logger.info(f"All callbacks completed in {total_elapsed:.1f}s")

    # Generate markdown report
    logger.info("Generating evaluation report")
    report_path = _generate_evaluation_report(callback_results, output_dir, metadata, timing=timing)

    logger.info(f"Evaluation complete, report saved to {report_path}")

    return callback_results, report_path
