"""Synoptic arithmetic and the t-6 NWP selection rule (Scope v2.1 §6.2.1)."""

from datetime import UTC, datetime, timedelta

import pytest

from aeolus.time_utils import (
    NWPUnavailableError,
    advisory_time,
    cycle_label,
    floor_synoptic,
    is_synoptic,
    next_synoptic,
    parse_cycle_label,
    select_nwp_cycle,
    synoptic_range,
)

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def test_is_synoptic_accepts_the_four_cycle_hours():
    for hour in (0, 6, 12, 18):
        assert is_synoptic(datetime(2026, 8, 6, hour, tzinfo=UTC))


def test_is_synoptic_rejects_off_hours_and_stray_minutes():
    assert not is_synoptic(datetime(2026, 8, 6, 7, tzinfo=UTC))
    assert not is_synoptic(datetime(2026, 8, 6, 6, 30, tzinfo=UTC))


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError, match="UTC"):
        is_synoptic(datetime(2026, 8, 6, 6))


def test_floor_and_next_synoptic():
    assert floor_synoptic(datetime(2026, 8, 6, 7, 45, tzinfo=UTC)) == T
    assert next_synoptic(datetime(2026, 8, 6, 7, 45, tzinfo=UTC)) == T + timedelta(hours=6)
    assert next_synoptic(T) == T + timedelta(hours=6)


def test_synoptic_range_is_inclusive_and_ordered():
    times = synoptic_range(T, T + timedelta(hours=18))
    assert times == [T + timedelta(hours=6 * i) for i in range(4)]


def test_cycle_label_round_trips():
    assert cycle_label(T) == "20260806_06Z"
    assert parse_cycle_label("20260806_06Z") == T


def test_advisory_is_three_hours_after_synoptic_time():
    assert advisory_time(T) == datetime(2026, 8, 6, 9, tzinfo=UTC)


# --- the v2.1 correction ---------------------------------------------------


def test_same_cycle_nwp_is_never_selected_even_when_offered():
    """The t cycle does not exist at t; offering it must not change the answer."""
    available = {T, T - timedelta(hours=6)}
    sel = select_nwp_cycle(T, available)
    assert sel.nwp_cycle == T - timedelta(hours=6)
    assert sel.lag_hours == 6
    assert not sel.degraded


def test_falls_back_to_t_minus_12_when_t_minus_6_is_missing():
    sel = select_nwp_cycle(T, {T - timedelta(hours=12)})
    assert sel.lag_hours == 12
    assert sel.degraded
    assert sel.tag == "12h"


def test_raises_when_nothing_within_the_staleness_budget():
    with pytest.raises(NWPUnavailableError, match="climatology"):
        select_nwp_cycle(T, {T - timedelta(hours=18)}, max_lag_hours=12)


def test_forecast_hour_matches_the_lag():
    """The t-6 cycle's 6h forecast is what is valid at t."""
    sel = select_nwp_cycle(T, {T - timedelta(hours=6)})
    assert sel.forecast_hour == 6
