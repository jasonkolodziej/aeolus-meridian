"""Shared fixtures and torch skip handling."""

from datetime import UTC, datetime

import pytest

from aeolus.data.availability import LatencyOracle
from aeolus.models.base import torch_available

TARGET = datetime(2026, 8, 6, 6, tzinfo=UTC)


def pytest_collection_modifyitems(config, items):
    """Skip torch-marked tests when the optional extra is not installed."""
    if torch_available():
        return
    skip = pytest.mark.skip(reason="torch extra not installed (uv sync --extra torch)")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def target_time():
    return TARGET


@pytest.fixture
def oracle():
    """Nominal feed timing: everything arrives at its typical latency."""
    return LatencyOracle()


@pytest.fixture
def worst_case_oracle():
    return LatencyOracle(use_max_latency=True)
