"""Verification metrics (Scope v2.1 §7.3, Appendix B)."""

import numpy as np
import pytest

from aeolus.geo import haversine_nm
from aeolus.metrics.probabilistic import (
    brier_score,
    crps_ensemble,
    rank_histogram,
    reliability,
    spread_skill,
)
from aeolus.metrics.track import (
    ForecastPoint,
    VerificationPair,
    beat_rate,
    diebold_mariano,
    to_metric_dict,
    verify,
)


def pair(lead=48, dlat=0.0, dwind=0.0, bearing=None):
    return VerificationPair(
        forecast=ForecastPoint(lead, 25.0 + dlat, -70.0, 90.0 + dwind),
        obs_lat=25.0,
        obs_lon=-70.0,
        obs_wind_kt=90.0,
        obs_motion_bearing=bearing,
    )


def test_perfect_forecast_has_zero_error():
    p = pair()
    assert p.track_error_nm == pytest.approx(0.0, abs=1e-6)
    assert p.intensity_error_kt == 0.0


def test_one_degree_of_latitude_is_about_sixty_nautical_miles():
    assert pair(dlat=1.0).track_error_nm == pytest.approx(60.0, rel=0.02)


def test_intensity_error_sign_indicates_over_forecast():
    assert pair(dwind=10.0).intensity_error_kt == 10.0
    assert pair(dwind=-10.0).intensity_error_kt == -10.0


def test_verify_groups_by_lead_time():
    stats = verify([pair(48, 1.0), pair(48, -1.0), pair(72, 2.0)])
    assert set(stats) == {48, 72}
    assert stats[48].n == 2


def test_track_bias_cancels_but_mae_does_not():
    stats = verify([pair(48, 1.0), pair(48, -1.0)])
    assert stats[48].track_mae_nm > 50.0
    assert stats[48].intensity_bias_kt == 0.0


def test_cross_track_decomposition_separates_right_of_track_error():
    """A storm moving north with the forecast to its east is a right bias."""
    p = VerificationPair(
        forecast=ForecastPoint(48, 25.0, -69.0, 90.0),
        obs_lat=25.0,
        obs_lon=-70.0,
        obs_wind_kt=90.0,
        obs_motion_bearing=0.0,
    )
    cross, along = p.decomposed_nm()
    assert cross > 40.0
    assert abs(along) < 5.0


def test_metric_dict_uses_the_scope_names():
    keys = to_metric_dict(verify([pair(48, 1.0)]))
    assert "track_error_48h_nm" in keys
    assert "intensity_error_48h_kt" in keys


def test_beat_rate_counts_ties_as_losses():
    assert beat_rate(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5


def test_beat_rate_rejects_empty_input():
    with pytest.raises(ValueError, match="zero cases"):
        beat_rate(np.array([]), np.array([]))


def test_diebold_mariano_favours_a_clearly_better_candidate():
    rng = np.random.default_rng(0)
    baseline = np.abs(rng.normal(80, 20, 200))
    candidate = baseline * 0.7
    stat, p = diebold_mariano(candidate, baseline)
    assert stat < 0
    assert p < 0.05


def test_diebold_mariano_finds_no_difference_between_equals():
    rng = np.random.default_rng(1)
    errors = np.abs(rng.normal(80, 20, 200))
    _, p = diebold_mariano(errors, errors.copy())
    assert p == 1.0


def test_diebold_mariano_needs_a_minimum_sample():
    with pytest.raises(ValueError, match="at least 8"):
        diebold_mariano(np.ones(4), np.ones(4))


# --- probabilistic ---------------------------------------------------------


def test_crps_of_a_single_member_is_absolute_error():
    assert crps_ensemble(np.array([5.0]), 8.0) == pytest.approx(3.0)


def test_crps_rewards_a_well_placed_ensemble():
    rng = np.random.default_rng(0)
    good = rng.normal(10, 2, 500)
    bad = rng.normal(25, 2, 500)
    assert crps_ensemble(good, 10.0) < crps_ensemble(bad, 10.0)


def test_brier_score_is_zero_for_perfect_probabilities():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_brier_score_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        brier_score(np.array([1.5]), np.array([1.0]))


def test_calibrated_ensemble_has_a_spread_skill_ratio_near_one():
    """Truth and members drawn from the same predictive distribution."""
    rng = np.random.default_rng(2)
    signal = rng.normal(0, 5, 600)
    members = signal[:, None] + rng.normal(0, 2, (600, 30))
    truth = signal + rng.normal(0, 2, 600)
    ss = spread_skill(members, truth)
    assert ss.ratio == pytest.approx(1.0, rel=0.15)
    assert not ss.underdispersed
    assert not ss.overdispersed


def test_underdispersion_is_flagged():
    """The characteristic failure of an ensemble anchored to a deterministic guess."""
    rng = np.random.default_rng(3)
    truth = rng.normal(0, 5, 300)
    members = truth[:, None] + rng.normal(0, 5, (300, 20)) * 0.1 + 5.0
    assert spread_skill(members, truth).underdispersed


def test_rank_histogram_has_one_more_bin_than_members():
    rng = np.random.default_rng(4)
    members = rng.normal(0, 1, (100, 9))
    counts = rank_histogram(members, rng.normal(0, 1, 100))
    assert counts.size == 10
    assert counts.sum() == 100


def test_reliability_returns_bin_centres_and_counts():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(float)
    centres, freq, counts = reliability(p, y, bins=5)
    assert centres.size == freq.size == counts.size == 5
    assert counts.sum() == 500


def test_haversine_is_symmetric():
    a = haversine_nm(25.0, -70.0, 30.0, -65.0)
    b = haversine_nm(30.0, -65.0, 25.0, -70.0)
    assert a == pytest.approx(b)
