"""Shared pytest configuration for smoke tests."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--num-devices",
        action="store",
        default="1",
        help="Number of GPUs to use for training smoke tests (default: 1)",
    )
    parser.addoption(
        "--no-generation",
        action="store_true",
        default=False,
        help="Skip molecule generation in validation callback (much faster)",
    )
