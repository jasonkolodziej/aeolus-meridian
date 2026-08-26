"""Source roles enforce the train/serve policy (Scope v2.1 §4.1, §4.6)."""

import pytest

from aeolus.data import sources
from aeolus.data.sources import OperationalUseError, Role


def test_era5_is_pretrain_only():
    assert sources.get("era5").role is Role.PRETRAIN_ONLY


def test_final_besttrack_is_labels_only():
    assert sources.get("besttrack_final").role is Role.LABELS_ONLY


def test_working_besttrack_is_operational():
    assert sources.get("besttrack_working").role is Role.OPERATIONAL


def test_reading_era5_in_the_inference_path_is_refused():
    with pytest.raises(OperationalUseError, match="gdas_gfs"):
        sources.assert_not_operational("era5")


def test_reading_final_besttrack_in_the_inference_path_is_refused():
    with pytest.raises(OperationalUseError, match="besttrack_working"):
        sources.assert_not_operational("besttrack_final")


def test_operational_sources_pass_the_guard():
    for src in sources.operational_sources():
        sources.assert_not_operational(src.key)


def test_gdas_latency_exceeds_the_v2_forty_five_minute_assumption():
    """The fact that invalidated v2's schedule, asserted so it cannot regress."""
    latency_hours = sources.get("gdas_gfs").typical_latency.total_seconds() / 3600
    assert latency_hours > 3.0


def test_era5_latency_is_days_not_hours():
    assert sources.get("era5").typical_latency.days >= 5


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        sources.get("nonexistent")
