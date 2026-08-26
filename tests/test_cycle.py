"""End-to-end cycle execution and degraded modes (Scope v2.1 §6.1, §10.1)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aeolus.data.availability import LatencyOracle
from aeolus.data.besttrack import Fix, TrackQuality
from aeolus.data.sources import Flavor
from aeolus.inference.cycle import (
    CycleError,
    DeterministicForecast,
    assert_operational_flavor,
    climatological_ensemble,
    extrapolate_fix,
    fusion_weights,
    run_cycle,
)
from aeolus.inference.postprocess import EnsembleMember
from aeolus.inference.scheduler import plan_cycle

T = datetime(2026, 8, 6, 6, tzinfo=UTC)
LEADS = (12, 24, 36, 48, 72, 96, 120)


def make_fix(quality=TrackQuality.WORKING, valid_time=T, hours=0):
    return Fix(
        storm_id="AL092026",
        valid_time=valid_time + timedelta(hours=hours),
        lat=22.4,
        lon=-72.1,
        max_wind_kt=85.0,
        min_pressure_mb=968.0,
        quality=quality,
    )


def deterministic_fn(plan, fix):
    steps = np.arange(1, len(LEADS) + 1, dtype=float)
    return DeterministicForecast(
        target_time=plan.target_time,
        lead_hours=LEADS,
        lats=fix.lat + 0.8 * steps,
        lons=fix.lon - 1.0 * steps,
        winds_kt=np.clip(fix.max_wind_kt + 3.0 * steps - 0.3 * steps**2, 20.0, None),
        contributors={"lstm": 0.3, "transformer": 0.4, "gnn": 0.3},
    )


def good_ensemble(det):
    """A credibly-dispersed ensemble: spread grows with lead time."""
    rng = np.random.default_rng(0)
    growth = 1.0 + np.arange(len(LEADS), dtype=float)
    return [
        EnsembleMember(
            member_id=i,
            lead_hours=LEADS,
            lats=det.lats + rng.normal(0, 0.6, len(LEADS)) * growth,
            lons=det.lons + rng.normal(0, 0.6, len(LEADS)) * growth,
            winds_kt=np.clip(det.winds_kt + rng.normal(0, 8, len(LEADS)), 0, None),
        )
        for i in range(20)
    ]


def test_nominal_cycle_produces_products_before_the_advisory():
    plan = plan_cycle(T, LatencyOracle())
    out = run_cycle(plan, make_fix(), deterministic_fn, good_ensemble)
    assert out.on_time
    assert out.products.ensemble_size == 20
    assert not out.flags


def test_final_quality_fix_is_refused_at_the_cycle_boundary():
    plan = plan_cycle(T, LatencyOracle())
    with pytest.raises(ValueError, match="labels-only|FINAL"):
        run_cycle(plan, make_fix(TrackQuality.FINAL), deterministic_fn, good_ensemble)


def test_fix_must_match_the_cycle_time():
    plan = plan_cycle(T, LatencyOracle())
    with pytest.raises(CycleError, match="does not match"):
        run_cycle(plan, make_fix(hours=6), deterministic_fn, good_ensemble)


def test_diffusion_crash_falls_back_to_a_climatological_ensemble():
    def crashing(_det):
        raise RuntimeError("CUDA out of memory")

    plan = plan_cycle(T, LatencyOracle())
    out = run_cycle(plan, make_fix(), deterministic_fn, crashing)
    assert out.products.ensemble_size == 20
    assert any(f.startswith("meridian_fallback") for f in out.flags)
    assert out.degraded


def test_empty_ensemble_also_triggers_the_fallback():
    plan = plan_cycle(T, LatencyOracle())
    out = run_cycle(plan, make_fix(), deterministic_fn, lambda d: [])
    assert out.products.ensemble_size > 0
    assert out.degraded


def test_stale_nwp_is_flagged_on_the_payload():
    oracle = LatencyOracle()
    oracle.set_missing("gdas_gfs", T - timedelta(hours=6))
    plan = plan_cycle(T, oracle)
    out = run_cycle(plan, make_fix(), deterministic_fn, good_ensemble)
    assert "nwp_stale=12h" in out.flags
    assert out.payload()["nwp_cycle_lag_hours"] == 12


def test_estimated_vitals_are_flagged_and_require_an_estimated_fix():
    oracle = LatencyOracle()
    oracle.set_missing("besttrack_working", T)
    plan = plan_cycle(T, oracle)
    with pytest.raises(CycleError, match="extrapolated"):
        run_cycle(plan, make_fix(TrackQuality.WORKING), deterministic_fn, good_ensemble)
    out = run_cycle(plan, make_fix(TrackQuality.ESTIMATED), deterministic_fn, good_ensemble)
    assert "vitals=estimated" in out.flags
    assert out.payload()["vitals"] == "estimated"


def test_missing_optional_source_is_flagged_but_the_cycle_runs():
    oracle = LatencyOracle()
    oracle.set_missing("goes", T)
    plan = plan_cycle(T, oracle)
    out = run_cycle(plan, make_fix(), deterministic_fn, good_ensemble)
    assert "missing:goes" in out.flags


def test_absent_opportunistic_feeds_do_not_flag_the_cycle():
    """Recon and microwave are absent from most cycles; flagging them is noise."""
    oracle = LatencyOracle()
    for key in ("dropsonde", "microwave"):
        oracle.set_missing(key, T)
    plan = plan_cycle(T, oracle)
    out = run_cycle(plan, make_fix(), deterministic_fn, good_ensemble)
    assert not out.flags
    assert not out.degraded


def test_payload_is_json_shaped_and_carries_the_cone():
    plan = plan_cycle(T, LatencyOracle())
    payload = run_cycle(plan, make_fix(), deterministic_fn, good_ensemble).payload()
    assert payload["cycle"] == "20260806_06Z"
    assert len(payload["cone"]) == len(LEADS)
    assert {"lead_hours", "lat", "lon", "radius_nm", "basis"} <= set(payload["cone"][0])


# --- helpers ---------------------------------------------------------------


def test_extrapolation_continues_the_last_motion_vector():
    history = (
        Fix("AL092026", T - timedelta(hours=12), 20.0, -70.0, 80.0, 975.0, TrackQuality.WORKING),
        Fix("AL092026", T - timedelta(hours=6), 21.0, -71.0, 85.0, 970.0, TrackQuality.WORKING),
    )
    fix = extrapolate_fix(history, T)
    assert fix.quality is TrackQuality.ESTIMATED
    assert fix.lat == pytest.approx(22.0)
    assert fix.lon == pytest.approx(-72.0)
    assert fix.max_wind_kt == 85.0


def test_extrapolation_needs_two_prior_fixes():
    history = (Fix("AL092026", T - timedelta(hours=6), 21.0, -71.0, 85.0, 970.0,
                   TrackQuality.WORKING),)
    with pytest.raises(CycleError, match="two prior"):
        extrapolate_fix(history, T)


def test_fusion_weights_favour_the_lower_error_model():
    weights = fusion_weights({"lstm": 100.0, "transformer": 50.0})
    assert weights["transformer"] > weights["lstm"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_fusion_weights_never_zero_a_model_out():
    weights = fusion_weights({"lstm": 1000.0, "transformer": 10.0, "gnn": 12.0})
    assert min(weights.values()) > 0.0


def test_climatological_fallback_spread_grows_with_lead_time():
    det = deterministic_fn(plan_cycle(T, LatencyOracle()), make_fix())
    members = climatological_ensemble(det, n_members=200, seed=1)
    early = np.std([m.at(12)[0] for m in members])
    late = np.std([m.at(120)[0] for m in members])
    assert late > early


def test_operational_flavor_guard_rejects_pretrain_features():
    with pytest.raises(CycleError, match="production runs on"):
        assert_operational_flavor(Flavor.ERA5_PRETRAIN)
