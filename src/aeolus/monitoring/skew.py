"""Train/serve skew monitoring.

Scope v2.1 §4.6.3. Roughly five days after each cycle, ERA5T becomes available
for that synoptic time. The audit re-runs the deterministic stack on ERA5T
inputs and compares the result to what the operational (GDAS-driven) run
produced. A growing gap means the fine-tuning stage has gone stale relative to
whatever NCEP changed in the operational system.

This is the measurement that makes the v2.1 policy self-checking. Without it,
train/serve skew is a thing we asserted we fixed in a document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..data.sources import get as get_source
from ..geo import haversine_nm
from ..time_utils import require_synoptic

#: Alert threshold from §4.6.3: rolling 2-week mean delta above this at 48h.
SKEW_ALERT_TRACK_NM = 15.0
SKEW_ALERT_INTENSITY_KT = 4.0
SKEW_WINDOW = timedelta(days=14)

#: Audit lead: how long after a cycle ERA5T is expected to be usable.
AUDIT_DELAY = timedelta(days=5)


@dataclass(frozen=True, slots=True)
class SkewSample:
    """One paired comparison at one lead time."""

    target_time: datetime
    lead_hours: int
    operational_lat: float
    operational_lon: float
    operational_wind_kt: float
    era5t_lat: float
    era5t_lon: float
    era5t_wind_kt: float

    def __post_init__(self) -> None:
        require_synoptic(self.target_time)
        if self.lead_hours <= 0:
            raise ValueError("lead_hours must be positive")

    @property
    def track_delta_nm(self) -> float:
        return float(
            haversine_nm(
                self.era5t_lat, self.era5t_lon, self.operational_lat, self.operational_lon
            )
        )

    @property
    def intensity_delta_kt(self) -> float:
        """Signed: positive means the operational run was stronger."""
        return float(self.operational_wind_kt - self.era5t_wind_kt)


@dataclass(frozen=True, slots=True)
class SkewReport:
    lead_hours: int
    n: int
    window_start: datetime
    window_end: datetime
    mean_track_delta_nm: float
    mean_abs_intensity_delta_kt: float
    intensity_bias_kt: float
    alert: bool
    reasons: tuple[str, ...] = ()

    def metric_dict(self) -> dict[str, float]:
        """The §7.3 metric names added in v2.1."""
        return {
            f"skew_track_delta_nm_{self.lead_hours}h": self.mean_track_delta_nm,
            f"skew_intensity_delta_kt_{self.lead_hours}h": self.mean_abs_intensity_delta_kt,
        }


def audit_due(target_time: datetime, now: datetime) -> bool:
    """True once ERA5T should be available for ``target_time`` (§4.6.3)."""
    require_synoptic(target_time)
    era5_latency = get_source("era5").typical_latency
    return now >= target_time + max(AUDIT_DELAY, era5_latency)


@dataclass(slots=True)
class SkewMonitor:
    """Rolling store of paired samples, with the §4.6.3 alert rule."""

    window: timedelta = SKEW_WINDOW
    track_threshold_nm: float = SKEW_ALERT_TRACK_NM
    intensity_threshold_kt: float = SKEW_ALERT_INTENSITY_KT
    min_samples: int = 8
    samples: list[SkewSample] = field(default_factory=list)

    def record(self, sample: SkewSample) -> None:
        self.samples.append(sample)

    def in_window(self, lead_hours: int, now: datetime) -> list[SkewSample]:
        start = now - self.window
        return [
            s
            for s in self.samples
            if s.lead_hours == lead_hours and start <= s.target_time <= now
        ]

    def report(self, now: datetime, lead_hours: int = 48) -> SkewReport | None:
        """Summarise the rolling window, or None if too few samples.

        Returning None rather than a noisy estimate matters: with a handful of
        storms a fortnight, a two-sample "mean" would fire the retrain trigger on
        one unusual case.
        """
        window = self.in_window(lead_hours, now)
        if len(window) < self.min_samples:
            return None

        track = np.array([s.track_delta_nm for s in window])
        intensity = np.array([s.intensity_delta_kt for s in window])
        mean_track = float(track.mean())
        mean_abs_intensity = float(np.abs(intensity).mean())

        reasons: list[str] = []
        if mean_track > self.track_threshold_nm:
            reasons.append(
                f"mean {lead_hours}h track delta {mean_track:.1f} nm exceeds "
                f"{self.track_threshold_nm:.1f} nm"
            )
        if mean_abs_intensity > self.intensity_threshold_kt:
            reasons.append(
                f"mean {lead_hours}h intensity delta {mean_abs_intensity:.1f} kt exceeds "
                f"{self.intensity_threshold_kt:.1f} kt"
            )

        return SkewReport(
            lead_hours=lead_hours,
            n=len(window),
            window_start=min(s.target_time for s in window),
            window_end=max(s.target_time for s in window),
            mean_track_delta_nm=mean_track,
            mean_abs_intensity_delta_kt=mean_abs_intensity,
            intensity_bias_kt=float(intensity.mean()),
            alert=bool(reasons),
            reasons=tuple(reasons),
        )

    def prune(self, now: datetime) -> int:
        """Drop samples older than twice the window. Returns the number removed."""
        cutoff = now - 2 * self.window
        before = len(self.samples)
        self.samples = [s for s in self.samples if s.target_time >= cutoff]
        return before - len(self.samples)
