"""Forecast cycle scheduling.

Scope v2.1 §6.2.2 and §8.2. The cycle is *gated on the working fix*, not on the
synoptic time: prefetch runs from t+0:00 against already-staged t-6 NWP fields,
but the cycle formally starts when TC-Vitals lands (nominally t+0:45, timeout
t+1:30). The deliverable deadline is the NHC advisory at t+3:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..data.availability import (
    AvailabilityOracle,
    CycleInputs,
    resolve_cycle_inputs,
)
from ..time_utils import ADVISORY_OFFSET, advisory_time, cycle_label, require_synoptic

#: Nominal working-fix arrival.
NOMINAL_VITALS_OFFSET = timedelta(minutes=45)

#: Safety margin required between product delivery and the advisory.
ADVISORY_MARGIN = timedelta(minutes=30)


class Stage(str, Enum):
    ASSEMBLY = "assembly"
    PREPROCESS = "preprocess"
    DETERMINISTIC = "deterministic"
    FUSION = "fusion"
    DIFFUSION = "diffusion"
    POSTPROCESS = "postprocess"


@dataclass(frozen=True, slots=True)
class StageBudget:
    stage: Stage
    target: timedelta
    maximum: timedelta

    def __post_init__(self) -> None:
        if self.maximum < self.target:
            raise ValueError(f"{self.stage.value}: max budget below target")


#: The §6.2.2 table, as code. Durations are measured from cycle start.
DEFAULT_BUDGETS: tuple[StageBudget, ...] = (
    StageBudget(Stage.ASSEMBLY, timedelta(minutes=10), timedelta(minutes=20)),
    StageBudget(Stage.PREPROCESS, timedelta(minutes=15), timedelta(minutes=25)),
    StageBudget(Stage.DETERMINISTIC, timedelta(minutes=10), timedelta(minutes=18)),
    StageBudget(Stage.FUSION, timedelta(minutes=2), timedelta(minutes=4)),
    StageBudget(Stage.DIFFUSION, timedelta(minutes=13), timedelta(minutes=25)),
    StageBudget(Stage.POSTPROCESS, timedelta(minutes=5), timedelta(minutes=10)),
)


#: Load-shedding profile. Used when the standard profile would not clear the
#: advisory -- typically after a late working fix. The saving comes almost
#: entirely from MERIDIAN: a 10-member ensemble instead of 20-50. A smaller
#: ensemble is a real loss of tail resolution, but it is a smaller loss than
#: delivering nothing before the advisory goes out.
REDUCED_BUDGETS: tuple[StageBudget, ...] = (
    StageBudget(Stage.ASSEMBLY, timedelta(minutes=8), timedelta(minutes=14)),
    StageBudget(Stage.PREPROCESS, timedelta(minutes=12), timedelta(minutes=18)),
    StageBudget(Stage.DETERMINISTIC, timedelta(minutes=10), timedelta(minutes=16)),
    StageBudget(Stage.FUSION, timedelta(minutes=2), timedelta(minutes=4)),
    StageBudget(Stage.DIFFUSION, timedelta(minutes=6), timedelta(minutes=10)),
    StageBudget(Stage.POSTPROCESS, timedelta(minutes=4), timedelta(minutes=8)),
)


def total_budget(budgets=DEFAULT_BUDGETS, *, worst_case: bool = False) -> timedelta:
    return sum(
        (b.maximum if worst_case else b.target for b in budgets),
        timedelta(),
    )


def derive_vitals_timeout(
    budgets: tuple[StageBudget, ...] = REDUCED_BUDGETS,
    *,
    margin: timedelta = ADVISORY_MARGIN,
) -> timedelta:
    """Latest cycle start that still clears the advisory on the worst-case path.

    Scope v2.1 §6.2.2 wrote the timeout as a flat t+1:30. That number does not
    survive the arithmetic: with the reduced profile's 70-minute worst case and
    a 30-minute advisory margin, a 1:30 start finishes at t+2:40 and leaves only
    20 minutes. Deriving the timeout from the budget keeps the three numbers
    consistent when any one of them is retuned, rather than leaving a latent
    10-minute shortfall in a table.
    """
    return ADVISORY_OFFSET - total_budget(budgets, worst_case=True) - margin


#: The point past which we stop waiting for the working fix (t+1:20).
VITALS_TIMEOUT = ADVISORY_OFFSET - timedelta(minutes=70) - ADVISORY_MARGIN


@dataclass(frozen=True, slots=True)
class ScheduledStage:
    stage: Stage
    start: datetime
    end_target: datetime
    end_max: datetime


@dataclass(frozen=True, slots=True)
class CyclePlan:
    """A concrete plan for one forecast cycle."""

    target_time: datetime
    cycle_start: datetime
    stages: tuple[ScheduledStage, ...]
    inputs: CycleInputs
    vitals_estimated: bool
    #: True when the reduced-ensemble profile was selected to hold the deadline.
    load_shed: bool = False

    @property
    def label(self) -> str:
        return cycle_label(self.target_time)

    @property
    def aeolus_ready_target(self) -> datetime:
        return self._stage(Stage.FUSION).end_target

    @property
    def aeolus_ready_max(self) -> datetime:
        return self._stage(Stage.FUSION).end_max

    @property
    def meridian_ready_target(self) -> datetime:
        return self.stages[-1].end_target

    @property
    def meridian_ready_max(self) -> datetime:
        return self.stages[-1].end_max

    @property
    def advisory_deadline(self) -> datetime:
        return advisory_time(self.target_time)

    @property
    def margin_target(self) -> timedelta:
        return self.advisory_deadline - self.meridian_ready_target

    @property
    def margin_max(self) -> timedelta:
        return self.advisory_deadline - self.meridian_ready_max

    @property
    def meets_advisory_deadline(self) -> bool:
        """True if even the worst-case path clears the advisory with margin."""
        return self.margin_max >= ADVISORY_MARGIN

    @property
    def degraded(self) -> bool:
        return self.vitals_estimated or self.load_shed or self.inputs.degraded

    def _stage(self, stage: Stage) -> ScheduledStage:
        for s in self.stages:
            if s.stage is stage:
                return s
        raise KeyError(stage)

    def describe(self) -> str:
        lines = [
            f"cycle {self.label}  start {self.cycle_start:%H:%M}Z"
            f"  nwp t-{self.inputs.nwp.lag_hours}h"
            + ("  [DEGRADED]" if self.degraded else "")
        ]
        for s in self.stages:
            lines.append(
                f"  {s.stage.value:<14} {s.start:%H:%M} -> "
                f"{s.end_target:%H:%M} (max {s.end_max:%H:%M})"
            )
        lines.append(
            f"  advisory {self.advisory_deadline:%H:%M}Z  "
            f"margin {int(self.margin_max.total_seconds() // 60)} min worst case"
        )
        return "\n".join(lines)


class CycleAbandoned(RuntimeError):
    """The cycle cannot run: a required input never arrived."""


def plan_cycle(
    t: datetime,
    oracle: AvailabilityOracle,
    *,
    budgets: tuple[StageBudget, ...] = DEFAULT_BUDGETS,
    vitals_timeout: timedelta = VITALS_TIMEOUT,
    allow_estimated_vitals: bool = True,
    allow_load_shedding: bool = True,
    max_nwp_lag_hours: int = 12,
) -> CyclePlan:
    """Build the plan for forecast cycle ``t``.

    Cycle start is the later of the working-fix arrival and the NWP staging
    time, capped at ``t + vitals_timeout``. Past the cap the cycle runs on an
    extrapolated vitals estimate and the product carries ``vitals=estimated``
    (§6.2.2 degraded modes) -- late is worse than approximate, because an
    advisory published without our guidance is worth nothing.
    """
    require_synoptic(t)
    timeout_at = t + vitals_timeout

    vitals_pub = oracle.published_at("besttrack_working", t)
    vitals_estimated = vitals_pub is None or vitals_pub > timeout_at
    if vitals_estimated and not allow_estimated_vitals:
        raise CycleAbandoned(
            f"working fix for {cycle_label(t)} not available by "
            f"{timeout_at:%H:%M}Z and estimation is disabled"
        )

    cycle_start = timeout_at if vitals_estimated else max(vitals_pub, t)
    inputs = resolve_cycle_inputs(
        t, cycle_start, oracle, max_nwp_lag_hours=max_nwp_lag_hours
    )

    nwp_status = inputs.status("gdas_gfs")
    if not nwp_status.available:
        raise CycleAbandoned(
            f"no NWP cycle within {max_nwp_lag_hours}h for {cycle_label(t)}; "
            "fall back to LSTM + climatology mode (Scope v2.1 §4.6.4)"
        )

    deadline = advisory_time(t) - ADVISORY_MARGIN
    load_shed = False
    if allow_load_shedding and cycle_start + total_budget(budgets, worst_case=True) > deadline:
        # The standard profile cannot hold the advisory from this start time.
        # Shed ensemble members rather than deliver late (§6.2.2 degraded modes).
        budgets = REDUCED_BUDGETS
        load_shed = True

    return CyclePlan(
        target_time=t,
        cycle_start=cycle_start,
        stages=_schedule_stages(cycle_start, budgets),
        inputs=inputs,
        vitals_estimated=vitals_estimated,
        load_shed=load_shed,
    )


def _schedule_stages(
    cycle_start: datetime, budgets: tuple[StageBudget, ...]
) -> tuple[ScheduledStage, ...]:
    stages: list[ScheduledStage] = []
    target_cursor = max_cursor = cycle_start
    for budget in budgets:
        stages.append(
            ScheduledStage(
                stage=budget.stage,
                start=target_cursor,
                end_target=target_cursor + budget.target,
                end_max=max_cursor + budget.maximum,
            )
        )
        target_cursor += budget.target
        max_cursor += budget.maximum
    return tuple(stages)


def plan_day(
    day_start: datetime,
    oracle: AvailabilityOracle,
    **kwargs,
) -> list[CyclePlan]:
    """Plan all four cycles of a day, skipping any that must be abandoned."""
    require_synoptic(day_start)
    plans: list[CyclePlan] = []
    for i in range(4):
        t = day_start + timedelta(hours=6 * i)
        try:
            plans.append(plan_cycle(t, oracle, **kwargs))
        except CycleAbandoned:
            continue
    return plans
