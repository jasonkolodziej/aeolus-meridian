"""ERA5T paired-input skew audit (Scope v2.1 §4.6.3)."""

from datetime import UTC, datetime, timedelta

from aeolus.monitoring.skew import SkewMonitor, SkewSample, audit_due

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def sample(days_ago=0, track_offset_deg=0.1, wind_delta=0.0, lead=48):
    when = T - timedelta(days=days_ago)
    when = when.replace(hour=6, minute=0, second=0, microsecond=0)
    return SkewSample(
        target_time=when,
        lead_hours=lead,
        operational_lat=25.0 + track_offset_deg,
        operational_lon=-70.0,
        operational_wind_kt=90.0 + wind_delta,
        era5t_lat=25.0,
        era5t_lon=-70.0,
        era5t_wind_kt=90.0,
    )


def test_audit_is_not_due_immediately_after_the_cycle():
    assert not audit_due(T, T + timedelta(hours=6))


def test_audit_becomes_due_once_era5t_lands():
    assert audit_due(T, T + timedelta(days=6))


def test_track_delta_is_a_great_circle_distance():
    s = sample(track_offset_deg=1.0)
    assert 55.0 < s.track_delta_nm < 65.0


def test_intensity_delta_is_signed_toward_the_operational_run():
    assert sample(wind_delta=5.0).intensity_delta_kt == 5.0


def test_report_is_withheld_below_the_sample_minimum():
    monitor = SkewMonitor(min_samples=8)
    for i in range(4):
        monitor.record(sample(days_ago=i))
    assert monitor.report(T) is None


def test_small_deltas_do_not_raise_an_alert():
    monitor = SkewMonitor(min_samples=4)
    for i in range(6):
        monitor.record(sample(days_ago=i, track_offset_deg=0.05))
    report = monitor.report(T)
    assert report is not None
    assert not report.alert


def test_large_track_delta_raises_an_alert():
    monitor = SkewMonitor(min_samples=4)
    for i in range(6):
        monitor.record(sample(days_ago=i, track_offset_deg=0.6))
    report = monitor.report(T)
    assert report.alert
    assert report.mean_track_delta_nm > 15.0


def test_intensity_skew_alone_can_raise_an_alert():
    monitor = SkewMonitor(min_samples=4)
    for i in range(6):
        monitor.record(sample(days_ago=i, track_offset_deg=0.01, wind_delta=9.0))
    assert monitor.report(T).alert


def test_samples_outside_the_window_are_excluded():
    monitor = SkewMonitor(min_samples=2)
    for i in range(4):
        monitor.record(sample(days_ago=i, track_offset_deg=0.05))
    for i in range(30, 36):
        monitor.record(sample(days_ago=i, track_offset_deg=2.0))
    report = monitor.report(T)
    assert report.n == 4
    assert not report.alert


def test_metric_names_match_the_scope_additions():
    monitor = SkewMonitor(min_samples=2)
    for i in range(4):
        monitor.record(sample(days_ago=i))
    keys = monitor.report(T).metric_dict()
    assert "skew_track_delta_nm_48h" in keys
    assert "skew_intensity_delta_kt_48h" in keys


def test_pruning_drops_ancient_samples():
    monitor = SkewMonitor(min_samples=2)
    for i in (0, 1, 40, 41):
        monitor.record(sample(days_ago=i))
    assert monitor.prune(T) == 2
