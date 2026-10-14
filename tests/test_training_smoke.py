"""Smoke tests for molecule improvement training configurations.

Each test runs ``pairit_train`` (via subprocess) with minimal overrides so the
full Hydra config stack is exercised end-to-end.

Tests fail (not skip) when required data files are missing -- the data is
expected to be present on the cluster.

Experiment configurations live in ``tests/smoke_test_configs.py`` (single
source of truth shared with README.md).

Usage::

    # Fast mode -- no molecule generation (~1 min per test):
    python -m pytest tests/test_training_smoke.py -m slow --no-generation

    # Full mode -- with generation (~7 min per test):
    python -m pytest tests/test_training_smoke.py -m slow

    # Multi-GPU (distributed):
    python -m pytest tests/test_training_smoke.py -m slow --num-devices 2

    # Run a single config:
    python -m pytest tests/test_training_smoke.py -m slow -k zinc_baseline
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from .smoke_test_configs import TRAINING_CONFIGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON = sys.executable
TRAIN_SCRIPT = str(Path(__file__).resolve().parents[1] / "src" / "pairit" / "cmdline" / "train.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def num_devices(request):
    """Number of trainer devices requested via --num-devices."""
    n = int(request.config.getoption("--num-devices"))
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n > available:
        pytest.skip(f"Requested {n} GPUs but only {available} available")
    return n


@pytest.fixture
def fast_overrides(request):
    """Hydra overrides for keeping smoke tests fast.

    With ``--no-generation`` the molecule validation callback is neutered
    (zero generations), cutting per-test runtime from ~7 min to ~1 min.
    """
    overrides = [
        "++trainer.max_steps=2",
        "++trainer.limit_val_batches=2",
        "++callbacks.molecule_validation.save_generated_samples=false",
        "++callbacks.molecule_validation.expensive_metrics_to_compute=[]",
        "++trainer.num_sanity_val_steps=0",
        "logger=csv",
    ]
    if request.config.getoption("--no-generation"):
        overrides += [
            "++callbacks.molecule_validation.num_unconditional_generations=0",
            "++callbacks.molecule_validation.num_conditional_generations=0",
        ]
    else:
        overrides += [
            "++callbacks.molecule_validation.num_unconditional_generations=2",
            "++callbacks.molecule_validation.num_conditional_generations=2",
        ]
    return overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_data_exists(*paths: str) -> None:
    """Assert all data paths exist, with a clear error message if not."""
    for p in paths:
        assert Path(p).exists(), f"Required data file not found: {p}"


def _run_training(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a training command and return the result."""
    env = {
        **os.environ,
        "HYDRA_FULL_ERROR": "1",
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [PYTHON, TRAIN_SCRIPT] + args
    return subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Parametrized training smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    TRAINING_CONFIGS,
    ids=[c["id"] for c in TRAINING_CONFIGS],
)
def test_training_config(config, tmp_path, num_devices, fast_overrides):
    """Smoke-test a training configuration end-to-end."""
    _assert_data_exists(*config["required_data"])

    args = (
        config["hydra_args"]
        + [
            f"paths.root_dir={tmp_path}",
            f"++trainer.devices={num_devices}",
        ]
        + fast_overrides
    )
    result = _run_training(args)

    assert result.returncode == 0, (
        f"{config['id']} training failed.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )
