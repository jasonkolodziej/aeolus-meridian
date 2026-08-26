"""Feature and validation-loss drift (Scope v2.1 §4.6.3, §5.5)."""

import numpy as np
import pytest

from aeolus.data.features import FEATURE_NAMES
from aeolus.data.sources import Flavor
from aeolus.monitoring.drift import (
    DriftError,
    ReferenceDistribution,
    ValidationLossMonitor,
    detect_feature_drift,
)

N_FEATURES = len(FEATURE_NAMES)


def reference(seed=0, flavor=Flavor.GDAS_FINETUNE):
    rng = np.random.default_rng(seed)
    return ReferenceDistribution.fit(rng.normal(0, 1, (500, N_FEATURES)), flavor)


def test_reference_must_be_the_stage_b_distribution():
    """Measuring drift against ERA5 statistics would show a permanent offset."""
    rng = np.random.default_rng(0)
    with pytest.raises(DriftError, match="Stage B"):
        ReferenceDistribution.fit(rng.normal(0, 1, (100, N_FEATURES)), Flavor.ERA5_PRETRAIN)


def test_matching_distribution_shows_no_drift():
    rng = np.random.default_rng(1)
    report = detect_feature_drift(reference(), rng.normal(0, 1, (200, N_FEATURES)))
    assert not report.alert


def test_shifted_mean_is_detected():
    rng = np.random.default_rng(2)
    live = rng.normal(0, 1, (200, N_FEATURES))
    live[:, 3] += 4.0
    report = detect_feature_drift(reference(), live)
    assert report.alert
    assert FEATURE_NAMES[3] in [f.name for f in report.drifted]


def test_collapsed_variance_is_detected():
    rng = np.random.default_rng(3)
    live = rng.normal(0, 1, (200, N_FEATURES))
    live[:, 5] = 0.01 * live[:, 5]
    assert detect_feature_drift(reference(), live).alert


def test_too_few_live_samples_raises():
    rng = np.random.default_rng(4)
    with pytest.raises(DriftError, match="at least"):
        detect_feature_drift(reference(), rng.normal(0, 1, (5, N_FEATURES)))


def test_feature_count_mismatch_raises():
    rng = np.random.default_rng(5)
    with pytest.raises(DriftError, match="feature count"):
        detect_feature_drift(reference(), rng.normal(0, 1, (100, N_FEATURES - 1)))


def test_summary_names_the_worst_feature():
    rng = np.random.default_rng(6)
    live = rng.normal(0, 1, (200, N_FEATURES))
    live[:, 0] += 6.0
    assert FEATURE_NAMES[0] in detect_feature_drift(reference(), live).summary()


# --- validation loss monitor ----------------------------------------------


def test_first_observation_only_sets_the_baseline():
    monitor = ValidationLossMonitor()
    assert not monitor.observe(1.0)
    assert monitor.baseline == 1.0


def test_trigger_requires_three_consecutive_breaches():
    monitor = ValidationLossMonitor()
    monitor.observe(1.0)
    assert not monitor.observe(1.20)
    assert not monitor.observe(1.21)
    assert monitor.observe(1.22)


def test_a_recovery_resets_the_consecutive_count():
    monitor = ValidationLossMonitor()
    monitor.observe(1.0)
    monitor.observe(1.20)
    monitor.observe(1.05)
    assert not monitor.observe(1.25)


def test_increases_below_the_threshold_never_trigger():
    monitor = ValidationLossMonitor()
    monitor.observe(1.0)
    for _ in range(5):
        assert not monitor.observe(1.10)
