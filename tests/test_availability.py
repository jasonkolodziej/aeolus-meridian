"""Input availability at cycle time t (Scope v2.1 §6.2.1)."""

from datetime import UTC, datetime, timedelta

import pytest

from aeolus.data.availability import (
    LatencyOracle,
    earliest_ready_time,
    resolve_cycle_inputs,
)

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def test_same_cycle_gfs_has_not_published_at_cycle_time():
    """v2 assumed it had. It has not."""
    oracle = LatencyOracle()
    published = oracle.published_at("gdas_gfs", T)
    assert published > T + timedelta(hours=3)


def test_resolved_cycle_uses_the_previous_nwp_cycle():
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=45), LatencyOracle())
    assert inputs.nwp.nwp_cycle == T - timedelta(hours=6)
    assert inputs.nwp.lag_hours == 6


def test_cycle_is_ready_once_vitals_and_staged_nwp_are_in():
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=45), LatencyOracle())
    assert inputs.ready
    assert not inputs.blocking


def test_cycle_is_not_ready_before_vitals_arrive():
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=10), LatencyOracle())
    assert not inputs.ready
    assert [s.source_key for s in inputs.blocking] == ["besttrack_working"]


def test_missing_t_minus_6_degrades_to_t_minus_12():
    oracle = LatencyOracle()
    oracle.set_missing("gdas_gfs", T - timedelta(hours=6))
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=45), oracle)
    assert inputs.nwp.lag_hours == 12
    assert inputs.degraded


def test_all_nwp_missing_marks_the_required_input_unavailable():
    oracle = LatencyOracle()
    for lag in (6, 12):
        oracle.set_missing("gdas_gfs", T - timedelta(hours=lag))
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=45), oracle)
    assert not inputs.ready
    assert any(s.source_key == "gdas_gfs" for s in inputs.blocking)


def test_opportunistic_absence_is_not_a_degradation():
    oracle = LatencyOracle()
    oracle.set_missing("dropsonde", T)
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=45), oracle)
    assert inputs.ready
    assert not inputs.degraded
    assert inputs.status("dropsonde").opportunistic


def test_missing_optional_source_degrades_but_does_not_block():
    oracle = LatencyOracle()
    oracle.set_missing("goes", T)
    inputs = resolve_cycle_inputs(T, T + timedelta(minutes=45), oracle)
    assert inputs.ready
    assert inputs.degraded
    assert not inputs.status("goes").available


def test_ensemble_perturbations_use_the_previous_cycle():
    inputs = resolve_cycle_inputs(T, T + timedelta(hours=1), LatencyOracle())
    assert inputs.status("ensemble_perturbations").valid_time == T - timedelta(hours=6)


def test_sst_is_persisted_from_the_previous_day():
    inputs = resolve_cycle_inputs(T, T + timedelta(hours=1), LatencyOracle())
    assert inputs.status("sst_ohc").valid_time == T - timedelta(days=1)


def test_earliest_ready_time_is_driven_by_the_vitals_arrival():
    ready = earliest_ready_time(T, LatencyOracle())
    assert ready == T + timedelta(minutes=45)


def test_earliest_ready_time_is_none_when_vitals_never_arrive():
    oracle = LatencyOracle()
    oracle.set_missing("besttrack_working", T)
    assert earliest_ready_time(T, oracle) is None


def test_only_operational_sources_are_ever_resolved():
    inputs = resolve_cycle_inputs(T, T + timedelta(hours=1), LatencyOracle())
    keys = {s.source_key for s in inputs.statuses}
    assert "era5" not in keys
    assert "besttrack_final" not in keys
