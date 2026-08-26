"""Cycle timeline, vitals gating and the advisory deadline (Scope v2.1 §6.2.2)."""

from datetime import UTC, datetime, timedelta

import pytest

from aeolus.data.availability import LatencyOracle
from aeolus.inference.scheduler import (
    ADVISORY_MARGIN,
    CycleAbandoned,
    DEFAULT_BUDGETS,
    REDUCED_BUDGETS,
    Stage,
    VITALS_TIMEOUT,
    derive_vitals_timeout,
    plan_cycle,
    plan_day,
    total_budget,
)

T = datetime(2026, 8, 6, 6, tzinfo=UTC)


def test_cycle_starts_when_vitals_land_not_at_synoptic_time():
    plan = plan_cycle(T, LatencyOracle())
    assert plan.cycle_start == T + timedelta(minutes=45)


def test_v2_delivery_claim_is_not_reproduced():
    """v2 promised AEOLUS output at t+0:45. Nothing can deliver that."""
    plan = plan_cycle(T, LatencyOracle())
    assert plan.aeolus_ready_target > T + timedelta(minutes=45)


def test_products_land_before_the_advisory_with_margin():
    plan = plan_cycle(T, LatencyOracle())
    assert plan.meridian_ready_target < plan.advisory_deadline
    assert plan.margin_target >= ADVISORY_MARGIN


def test_nominal_start_does_not_need_load_shedding():
    plan = plan_cycle(T, LatencyOracle())
    assert not plan.load_shed
    assert plan.meets_advisory_deadline


def test_late_start_sheds_ensemble_load_to_hold_the_advisory():
    """A t+1:30 start plus the full profile would breach the margin, so shed."""
    oracle = LatencyOracle()
    oracle.set_arrival("besttrack_working", T, T + timedelta(hours=3))
    plan = plan_cycle(T, oracle)
    assert plan.cycle_start == T + VITALS_TIMEOUT
    assert plan.load_shed
    assert plan.meets_advisory_deadline
    assert plan.degraded


def test_load_shedding_can_be_disabled_and_then_the_deadline_is_missed():
    """Documents the trade explicitly: without shedding, a late fix misses."""
    oracle = LatencyOracle()
    oracle.set_arrival("besttrack_working", T, T + timedelta(hours=3))
    plan = plan_cycle(T, oracle, allow_load_shedding=False)
    assert not plan.load_shed
    assert not plan.meets_advisory_deadline


def test_worst_case_feed_latency_still_clears_the_advisory():
    plan = plan_cycle(T, LatencyOracle(use_max_latency=True))
    assert plan.meets_advisory_deadline


def test_target_total_budget_is_about_fifty_five_minutes():
    assert total_budget() == timedelta(minutes=55)


def test_worst_case_total_budget_exceeds_the_target_by_design():
    assert total_budget(worst_case=True) == timedelta(minutes=102)


def test_reduced_profile_is_materially_faster_than_the_standard_one():
    assert total_budget(REDUCED_BUDGETS, worst_case=True) < total_budget(worst_case=True)


def test_vitals_timeout_is_derived_not_asserted():
    """v2.1 wrote t+1:30; the arithmetic gives t+1:20."""
    assert VITALS_TIMEOUT == derive_vitals_timeout()
    assert VITALS_TIMEOUT == timedelta(minutes=80)


def test_a_cycle_started_at_the_timeout_still_clears_the_margin():
    latest_finish = VITALS_TIMEOUT + total_budget(REDUCED_BUDGETS, worst_case=True)
    assert latest_finish + ADVISORY_MARGIN <= timedelta(hours=3)


def test_stage_order_matches_the_scope_table():
    plan = plan_cycle(T, LatencyOracle())
    assert [s.stage for s in plan.stages] == [b.stage for b in DEFAULT_BUDGETS]
    assert plan.stages[0].stage is Stage.ASSEMBLY
    assert plan.stages[-1].stage is Stage.POSTPROCESS


def test_aeolus_is_ready_before_meridian():
    plan = plan_cycle(T, LatencyOracle())
    assert plan.aeolus_ready_target < plan.meridian_ready_target


def test_stages_are_contiguous():
    plan = plan_cycle(T, LatencyOracle())
    for earlier, later in zip(plan.stages, plan.stages[1:]):
        assert later.start == earlier.end_target


# --- degraded modes --------------------------------------------------------


def test_late_vitals_triggers_estimation_at_the_timeout():
    oracle = LatencyOracle()
    oracle.set_arrival("besttrack_working", T, T + timedelta(hours=3))
    plan = plan_cycle(T, oracle)
    assert plan.vitals_estimated
    assert plan.cycle_start == T + VITALS_TIMEOUT
    assert plan.degraded


def test_missing_vitals_can_be_abandoned_when_estimation_is_disabled():
    oracle = LatencyOracle()
    oracle.set_missing("besttrack_working", T)
    with pytest.raises(CycleAbandoned, match="working fix"):
        plan_cycle(T, oracle, allow_estimated_vitals=False)


def test_missing_vitals_still_produces_a_plan_by_default():
    oracle = LatencyOracle()
    oracle.set_missing("besttrack_working", T)
    plan = plan_cycle(T, oracle)
    assert plan.vitals_estimated


def test_stale_nwp_marks_the_plan_degraded():
    oracle = LatencyOracle()
    oracle.set_missing("gdas_gfs", T - timedelta(hours=6))
    plan = plan_cycle(T, oracle)
    assert plan.inputs.nwp.lag_hours == 12
    assert plan.degraded


def test_total_nwp_outage_abandons_the_cycle():
    oracle = LatencyOracle()
    for lag in (6, 12):
        oracle.set_missing("gdas_gfs", T - timedelta(hours=lag))
    with pytest.raises(CycleAbandoned, match="climatology"):
        plan_cycle(T, oracle)


def test_all_four_daily_cycles_plan_and_meet_their_deadlines():
    day = datetime(2026, 8, 6, 0, tzinfo=UTC)
    plans = plan_day(day, LatencyOracle())
    assert len(plans) == 4
    assert [p.target_time.hour for p in plans] == [0, 6, 12, 18]
    assert all(p.meets_advisory_deadline for p in plans)


def test_describe_mentions_the_nwp_lag():
    assert "t-6h" in plan_cycle(T, LatencyOracle()).describe()
