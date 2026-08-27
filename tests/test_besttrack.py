"""Working vs final best-track, and the Stage B noise emulator (§4.6.2)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aeolus.data.besttrack import (
    Fix,
    Track,
    TrackQuality,
    WorkingTrackNoise,
    assert_input_safe,
    emulate_working_fix,
    emulate_working_track,
    recalibrate_from_pairs,
)
from aeolus.geo import haversine_nm

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def make_fix(hour_offset=0, quality=TrackQuality.FINAL, **kw):
    params = dict(
        storm_id="AL092026",
        valid_time=T + timedelta(hours=hour_offset),
        lat=22.4,
        lon=-72.1,
        max_wind_kt=85.0,
        min_pressure_mb=968.0,
        quality=quality,
    )
    params.update(kw)
    return Fix(**params)


def make_track(n=6, quality=TrackQuality.FINAL):
    return Track(
        storm_id="AL092026",
        fixes=tuple(
            make_fix(6 * i, quality, lat=22.4 + 0.5 * i, lon=-72.1 - 0.6 * i)
            for i in range(n)
        ),
    )


def test_fix_rejects_non_synoptic_times():
    with pytest.raises(ValueError, match="synoptic"):
        Fix("AL092026", datetime(2026, 8, 6, 7, tzinfo=UTC), 22.0, -72.0, 85.0, 968.0,
            TrackQuality.WORKING)


def test_fix_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError, match="lat"):
        make_fix(lat=95.0)


def test_final_fixes_are_not_model_input_safe():
    assert not make_fix(quality=TrackQuality.FINAL).is_model_input_safe
    assert make_fix(quality=TrackQuality.WORKING).is_model_input_safe
    assert make_fix(quality=TrackQuality.EMULATED).is_model_input_safe


def test_assert_input_safe_blocks_final_quality():
    with pytest.raises(ValueError, match="labels-only|FINAL"):
        assert_input_safe([make_fix(quality=TrackQuality.FINAL)])


def test_assert_input_safe_allows_working_quality():
    assert_input_safe([make_fix(quality=TrackQuality.WORKING)])


def test_track_rejects_mixed_quality():
    with pytest.raises(ValueError, match="mixed-quality"):
        Track("AL092026", (make_fix(0, TrackQuality.FINAL), make_fix(6, TrackQuality.WORKING)))


def test_track_rejects_unordered_fixes():
    with pytest.raises(ValueError, match="ascending"):
        Track("AL092026", (make_fix(6), make_fix(0)))


def test_window_ending_returns_none_rather_than_padding():
    track = make_track(6)
    assert track.window_ending(track.fixes[1].valid_time, 4) is None
    window = track.window_ending(track.fixes[4].valid_time, 4)
    assert len(window) == 4
    assert window[-1].valid_time == track.fixes[4].valid_time


# --- the emulator ----------------------------------------------------------


def test_emulated_fix_is_marked_and_differs_from_final():
    rng = np.random.default_rng(0)
    final = make_fix()
    working = emulate_working_fix(final, rng, WorkingTrackNoise(quantise=False))
    assert working.quality is TrackQuality.EMULATED
    assert (working.lat, working.lon) != (final.lat, final.lon)


def test_quantisation_can_leave_a_position_unchanged():
    """0.1-degree rounding means a small perturbation is sometimes invisible."""
    rng = np.random.default_rng(0)
    final = make_fix()
    positions = {
        (f.lat, f.lon)
        for f in (emulate_working_fix(final, rng) for _ in range(200))
    }
    assert len(positions) > 5


def test_emulator_refuses_non_final_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="FINAL"):
        emulate_working_fix(make_fix(quality=TrackQuality.WORKING), rng)


def test_emulated_position_error_matches_the_requested_rms():
    rng = np.random.default_rng(42)
    final = make_fix()
    noise = WorkingTrackNoise(position_rms_nm=15.0, quantise=False)
    errors = [
        haversine_nm(final.lat, final.lon, *_latlon(emulate_working_fix(final, rng, noise)))
        for _ in range(2000)
    ]
    assert 13.0 < float(np.sqrt(np.mean(np.square(errors)))) < 17.0


def _latlon(fix):
    return fix.lat, fix.lon


def test_emulated_winds_are_quantised_to_five_knots():
    rng = np.random.default_rng(1)
    for _ in range(50):
        wind = emulate_working_fix(make_fix(), rng).max_wind_kt
        assert wind % 5.0 == pytest.approx(0.0, abs=1e-9)


def test_emulated_track_keeps_length_and_ordering():
    rng = np.random.default_rng(7)
    final = make_track(8)
    working = emulate_working_track(final, rng)
    assert len(working.fixes) == 8
    assert working.quality is TrackQuality.EMULATED
    assert [f.valid_time for f in working.fixes] == [f.valid_time for f in final.fixes]


def test_recalibration_recovers_the_injected_error_scale():
    rng = np.random.default_rng(3)
    noise = WorkingTrackNoise(position_rms_nm=20.0, intensity_rms_kt=8.0, quantise=False)
    finals = [make_fix(6 * i) for i in range(400)]
    pairs = [(emulate_working_fix(f, rng, noise), f) for f in finals]
    measured = recalibrate_from_pairs(pairs)
    assert measured.position_rms_nm == pytest.approx(20.0, rel=0.2)
    assert measured.intensity_rms_kt == pytest.approx(8.0, rel=0.25)


def test_literature_noise_is_wider_than_the_scope_defaults():
    """Published estimates put intensity and pressure error above §4.6.2's."""
    scope, lit = WorkingTrackNoise(), WorkingTrackNoise.from_literature()
    assert lit.intensity_rms_kt > scope.intensity_rms_kt
    assert lit.pressure_rms_mb > scope.pressure_rms_mb


def test_defaults_are_unchanged_by_the_literature_alternative():
    """from_literature() is opt-in; it must not move Stage B silently."""
    assert WorkingTrackNoise().intensity_rms_kt == 5.0
    assert WorkingTrackNoise().pressure_rms_mb == 3.0


def test_recalibration_rejects_mismatched_pairs():
    with pytest.raises(ValueError, match="valid_time"):
        recalibrate_from_pairs([(make_fix(0, TrackQuality.WORKING), make_fix(6))])
