"""Synoptic-time arithmetic for the 6-hourly forecast cycle.

Scope v2.1 §6.2. Every operational time in this system is anchored to a
synoptic time ``t`` in {00Z, 06Z, 12Z, 18Z}. The single most important fact
encoded here is that the NWP cycle *named* ``t`` is not available at ``t`` --
it publishes roughly 3.5-4 hours later -- so cycle ``t`` consumes the ``t-6``
NWP cycle's fields valid at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

SYNOPTIC_HOURS: tuple[int, ...] = (0, 6, 12, 18)
CYCLE_INTERVAL = timedelta(hours=6)

#: NHC public advisories are issued at 03/09/15/21Z, i.e. t + 3h.
ADVISORY_OFFSET = timedelta(hours=3)


def is_synoptic(ts: datetime) -> bool:
    """True if ``ts`` falls exactly on a 00/06/12/18Z synoptic time."""
    _require_utc(ts)
    return (
        ts.hour in SYNOPTIC_HOURS
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
    )


def floor_synoptic(ts: datetime) -> datetime:
    """Most recent synoptic time at or before ``ts``."""
    _require_utc(ts)
    hour = max(h for h in SYNOPTIC_HOURS if h <= ts.hour)
    return ts.replace(hour=hour, minute=0, second=0, microsecond=0)


def next_synoptic(ts: datetime) -> datetime:
    """First synoptic time strictly after ``ts``."""
    _require_utc(ts)
    floor = floor_synoptic(ts)
    return floor if floor > ts else floor + CYCLE_INTERVAL


def synoptic_range(start: datetime, end: datetime) -> list[datetime]:
    """All synoptic times in the inclusive interval ``[start, end]``."""
    _require_utc(start)
    _require_utc(end)
    if end < start:
        raise ValueError("end must not precede start")
    out: list[datetime] = []
    cursor = start if is_synoptic(start) else next_synoptic(start)
    while cursor <= end:
        out.append(cursor)
        cursor += CYCLE_INTERVAL
    return out


def advisory_time(t: datetime) -> datetime:
    """Public advisory deadline for synoptic time ``t`` (t + 3h)."""
    require_synoptic(t)
    return t + ADVISORY_OFFSET


def cycle_label(t: datetime) -> str:
    """Canonical archive label, e.g. ``20260806_06Z`` (Scope v2.1 §9.1)."""
    require_synoptic(t)
    return f"{t:%Y%m%d}_{t.hour:02d}Z"


def parse_cycle_label(label: str) -> datetime:
    """Inverse of :func:`cycle_label`."""
    try:
        stamp, zed = label.split("_")
        if not zed.endswith("Z"):
            raise ValueError
        hour = int(zed[:-1])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"malformed cycle label: {label!r}") from exc
    parsed = datetime.strptime(stamp, "%Y%m%d").replace(hour=hour, tzinfo=UTC)
    require_synoptic(parsed)
    return parsed


def require_synoptic(ts: datetime) -> None:
    """Raise unless ``ts`` is an exact synoptic time."""
    if not is_synoptic(ts):
        raise ValueError(f"{ts!r} is not a 00/06/12/18Z synoptic time")


def _require_utc(ts: datetime) -> None:
    if ts.tzinfo is None or ts.utcoffset() != timedelta(0):
        raise ValueError(f"{ts!r} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class NWPCycleSelection:
    """Which NWP cycle a forecast cycle actually consumes.

    Scope v2.1 §6.2.1. ``lag_hours`` is 6 in the nominal case and 12 when the
    t-6 cycle is missing (§4.6.4 NOMADS-outage fallback).
    """

    target_time: datetime
    nwp_cycle: datetime
    lag_hours: int
    forecast_hour: int
    degraded: bool

    @property
    def tag(self) -> str:
        """Value for the MLflow ``nwp_cycle_lag`` run tag (§7.2)."""
        return f"{self.lag_hours}h"


def select_nwp_cycle(
    t: datetime,
    available_cycles: set[datetime] | frozenset[datetime],
    *,
    max_lag_hours: int = 12,
) -> NWPCycleSelection:
    """Pick the NWP cycle to drive forecast cycle ``t``.

    The cycle named ``t`` is deliberately never selected even if it appears in
    ``available_cycles``: at ``t`` it does not exist yet, and silently using it
    is exactly the timing error Scope v2 contained. Selection walks back in
    6-hour steps from ``t-6`` to ``t - max_lag_hours``.

    Raises:
        NWPUnavailableError: if nothing within ``max_lag_hours`` is available.
    """
    require_synoptic(t)
    for lag in range(6, max_lag_hours + 1, 6):
        candidate = t - timedelta(hours=lag)
        if candidate in available_cycles:
            return NWPCycleSelection(
                target_time=t,
                nwp_cycle=candidate,
                lag_hours=lag,
                forecast_hour=lag,
                degraded=lag > 6,
            )
    raise NWPUnavailableError(
        f"no NWP cycle within {max_lag_hours}h of {cycle_label(t)}; "
        "degrade to LSTM + climatology per Scope v2.1 §4.6.4"
    )


class NWPUnavailableError(RuntimeError):
    """No NWP cycle within the staleness budget (Scope v2.1 §4.6.4)."""
