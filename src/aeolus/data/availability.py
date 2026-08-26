"""What is actually available at cycle time ``t``.

Scope v2.1 §6.2.1. The v2 schedule assumed same-cycle GFS at t+0:45; GFS 0.25°
publishes ~3.5-4h after its cycle time, so that schedule could not have run.
This module makes availability an explicit, queryable model so the scheduler
plans against real latencies instead of assumed ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from ..time_utils import NWPCycleSelection, require_synoptic, select_nwp_cycle
from . import sources
from .sources import Role


class AvailabilityOracle(Protocol):
    """Answers 'was source X, valid at time V, published by wall-clock N?'"""

    def published_at(self, source_key: str, valid_time: datetime) -> datetime | None:
        """Wall-clock publication time, or None if it never arrives."""
        ...


@dataclass(slots=True)
class LatencyOracle:
    """Default oracle: publication time = valid time + the registry latency.

    ``overrides`` lets a test or a replay harness pin a specific arrival, and
    ``missing`` simulates an outage (the source never publishes for that time).
    """

    use_max_latency: bool = False
    overrides: dict[tuple[str, datetime], datetime] = field(default_factory=dict)
    missing: set[tuple[str, datetime]] = field(default_factory=set)

    def published_at(self, source_key: str, valid_time: datetime) -> datetime | None:
        key = (source_key, valid_time)
        if key in self.missing:
            return None
        if key in self.overrides:
            return self.overrides[key]
        src = sources.get(source_key)
        latency = src.max_latency if self.use_max_latency else src.typical_latency
        return valid_time + latency

    def set_arrival(self, source_key: str, valid_time: datetime, at: datetime) -> None:
        self.overrides[(source_key, valid_time)] = at

    def set_missing(self, source_key: str, valid_time: datetime) -> None:
        self.missing.add((source_key, valid_time))


@dataclass(frozen=True, slots=True)
class InputStatus:
    source_key: str
    valid_time: datetime
    available: bool
    published_at: datetime | None
    required: bool
    #: Event-driven feeds that are absent from most cycles by nature. Their
    #: absence is normal and must not raise a degradation flag, or every cycle
    #: ships degraded and the flag stops carrying information.
    opportunistic: bool = False

    @property
    def blocking(self) -> bool:
        return self.required and not self.available

    @property
    def unexpectedly_missing(self) -> bool:
        return not self.available and not self.required and not self.opportunistic


@dataclass(frozen=True, slots=True)
class CycleInputs:
    """Resolved input set for one forecast cycle."""

    target_time: datetime
    as_of: datetime
    nwp: NWPCycleSelection
    statuses: tuple[InputStatus, ...]

    @property
    def ready(self) -> bool:
        return not any(s.blocking for s in self.statuses)

    @property
    def blocking(self) -> tuple[InputStatus, ...]:
        return tuple(s for s in self.statuses if s.blocking)

    @property
    def degraded(self) -> bool:
        """True if the cycle runs on stale NWP or is missing optional inputs."""
        return self.nwp.degraded or any(s.unexpectedly_missing for s in self.statuses)

    def available_keys(self) -> tuple[str, ...]:
        return tuple(s.source_key for s in self.statuses if s.available)

    def status(self, source_key: str) -> InputStatus:
        for s in self.statuses:
            if s.source_key == source_key:
                return s
        raise KeyError(source_key)


#: Inputs without which a cycle cannot run at all.
REQUIRED_SOURCES: tuple[str, ...] = ("besttrack_working", "gdas_gfs")

#: Inputs that improve the forecast but whose absence only degrades it.
OPTIONAL_SOURCES: tuple[str, ...] = (
    "goes",
    "ndbc",
    "sst_ohc",
    "ensemble_perturbations",
)

#: Event- and orbit-driven feeds. Present for a minority of cycles by nature:
#: aircraft reconnaissance only flies for threatening storms, and a microwave
#: overpass either happened in the window or did not.
OPPORTUNISTIC_SOURCES: tuple[str, ...] = ("dropsonde", "microwave")


def resolve_cycle_inputs(
    t: datetime,
    as_of: datetime,
    oracle: AvailabilityOracle,
    *,
    max_nwp_lag_hours: int = 12,
) -> CycleInputs:
    """Determine the input set for forecast cycle ``t`` as of wall-clock ``as_of``.

    The NWP selection deliberately searches only cycles at or before ``t-6``
    (see :func:`aeolus.time_utils.select_nwp_cycle`), and additionally requires
    that the chosen cycle has *published* by ``as_of`` -- a cycle that exists in
    the archive but has not yet been transferred is not usable.
    """
    require_synoptic(t)

    available_nwp = {
        cycle
        for lag in range(6, max_nwp_lag_hours + 1, 6)
        if (cycle := t - timedelta(hours=lag))
        and (pub := oracle.published_at("gdas_gfs", cycle)) is not None
        and pub <= as_of
    }
    try:
        nwp = select_nwp_cycle(t, available_nwp, max_lag_hours=max_nwp_lag_hours)
        nwp_available = True
    except Exception:
        nwp = NWPCycleSelection(
            target_time=t,
            nwp_cycle=t - timedelta(hours=max_nwp_lag_hours),
            lag_hours=max_nwp_lag_hours,
            forecast_hour=max_nwp_lag_hours,
            degraded=True,
        )
        nwp_available = False

    statuses: list[InputStatus] = []
    for key in REQUIRED_SOURCES + OPTIONAL_SOURCES + OPPORTUNISTIC_SOURCES:
        src = sources.get(key)
        if src.role is not Role.OPERATIONAL:  # pragma: no cover - registry invariant
            raise sources.OperationalUseError(f"{key} is not operational")

        if key == "gdas_gfs":
            pub = oracle.published_at(key, nwp.nwp_cycle)
            statuses.append(
                InputStatus(key, nwp.nwp_cycle, nwp_available, pub, required=True)
            )
            continue

        valid_time = _valid_time_for(key, t)
        pub = oracle.published_at(key, valid_time)
        statuses.append(
            InputStatus(
                source_key=key,
                valid_time=valid_time,
                available=pub is not None and pub <= as_of,
                published_at=pub,
                required=key in REQUIRED_SOURCES,
                opportunistic=key in OPPORTUNISTIC_SOURCES,
            )
        )

    return CycleInputs(target_time=t, as_of=as_of, nwp=nwp, statuses=tuple(statuses))


def _valid_time_for(source_key: str, t: datetime) -> datetime:
    """Valid time of the observation a cycle at ``t`` would use.

    Most observational feeds are current to ``t``. Ensemble perturbations share
    the NWP lag; SST/OHC is a daily product persisted from the previous day.
    """
    if source_key == "ensemble_perturbations":
        return t - timedelta(hours=6)
    if source_key == "sst_ohc":
        return t - timedelta(days=1)
    return t


def earliest_ready_time(
    t: datetime,
    oracle: AvailabilityOracle,
    *,
    max_nwp_lag_hours: int = 12,
) -> datetime | None:
    """Wall-clock time at which every *required* input for cycle ``t`` has landed.

    Returns None if a required input never arrives. This is what the scheduler
    uses to place the cycle start, replacing v2's fixed "t+0:00" assumption.
    """
    require_synoptic(t)
    times: list[datetime] = []

    vitals = oracle.published_at("besttrack_working", t)
    if vitals is None:
        return None
    times.append(vitals)

    nwp_pubs = [
        pub
        for lag in range(6, max_nwp_lag_hours + 1, 6)
        if (pub := oracle.published_at("gdas_gfs", t - timedelta(hours=lag))) is not None
    ]
    if not nwp_pubs:
        return None
    times.append(min(nwp_pubs))

    return max(times)
