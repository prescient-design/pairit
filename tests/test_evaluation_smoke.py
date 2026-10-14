"""Smoke tests for molecule improvement evaluation.

Runs ``pairit_eval`` via subprocess to verify the full evaluation pipeline
(Hydra config resolution, model loading, datamodule setup, callback execution).
Tests fail (not skip) when required data or checkpoints are missing.

Experiment configurations live in ``tests/smoke_test_configs.py`` (single
source of truth shared with README.md).

Usage::

    python -m pytest tests/test_evaluation_smoke.py -m slow
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .smoke_test_configs import EVAL_CONFIGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON = sys.executable
EVAL_SCRIPT = str(Path(__file__).resolve().parents[1] / "src" / "pairit" / "cmdline" / "evaluate.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_best_checkpoint(run_dir: str) -> str | None:
    """Find the best (non-last) checkpoint under a pairit run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None
    timestamp_dirs = [d for d in run_path.iterdir() if d.is_dir() and d.name != "generated"]
    if not timestamp_dirs:
        return None
    ts_dir = sorted(timestamp_dirs)[0]
    ckpts = [f for f in ts_dir.iterdir() if f.suffix == ".ckpt" and f.name != "last.ckpt"]
    if not ckpts:
        return None
    return str(sorted(ckpts)[0])


def _assert_data_exists(*paths: str) -> None:
    """Assert all data paths exist, with a clear error message if not."""
    for p in paths:
        assert Path(p).exists(), f"Required data file not found: {p}"


def _run_eval(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an evaluation command and return the result."""
    env = {
        **os.environ,
        "HYDRA_FULL_ERROR": "1",
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [PYTHON, EVAL_SCRIPT] + args
    return subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Parametrized evaluation smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    EVAL_CONFIGS,
    ids=[c["id"] for c in EVAL_CONFIGS],
)
def test_eval_config(config, tmp_path):
    """Smoke-test an evaluation configuration end-to-end."""
    _assert_data_exists(*config["required_data"])

    ckpt_dir = config.get("requires_checkpoint")
    if ckpt_dir:
        ckpt_path = _find_best_checkpoint(ckpt_dir)
        assert ckpt_path is not None, f"No checkpoint found under {ckpt_dir}"
    else:
        ckpt_path = None

    args = list(config["hydra_args"])
    if ckpt_path:
        args.append(f"model.ckpt_path='{ckpt_path}'")

    result = _run_eval(args)

    assert result.returncode == 0, (
        f"{config['id']} evaluation failed.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )
