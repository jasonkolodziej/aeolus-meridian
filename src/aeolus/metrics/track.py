"""Deterministic forecast verification.

Scope v2.1 §7.3 and Appendix B. Track error is great-circle distance in nautical
miles; intensity error is signed in knots so that bias is recoverable. Errors
are reported per lead time because a model can be excellent at 24h and useless
at 96h, and a single averaged number hides exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import bearing_deg, cross_along_track_nm, haversine_nm

#: Lead times verified, in hours (§7.3).
LEAD_TIMES: tuple[int, ...] = (12, 24, 36, 48, 72, 96, 120)


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    lead_hours: int
    lat: float
    lon: float
    max_wind_kt: float


@dataclass(frozen=True, slots=True)
class VerificationPair:
    """One forecast and the observed truth it is verified against."""

    forecast: ForecastPoint
    obs_lat: float
    obs_lon: float
    obs_wind_kt: float
    #: Observed storm motion bearing, for cross/along-track decomposition.
    obs_motion_bearing: float | None = None

    def __post_init__(self) -> None:
        if self.forecast.lead_hours <= 0:
            raise ValueError("lead_hours must be positive")

    @property
    def track_error_nm(self) -> float:
        return float(
            haversine_nm(self.obs_lat, self.obs_lon, self.forecast.lat, self.forecast.lon)
        )

    @property
    def intensity_error_kt(self) -> float:
        """Signed: positive means the forecast was too strong."""
        return float(self.forecast.max_wind_kt - self.obs_wind_kt)

    def decomposed_nm(self) -> tuple[float, float]:
        if self.obs_motion_bearing is None:
            raise ValueError("obs_motion_bearing required for cross/along decomposition")
        return cross_along_track_nm(
            self.forecast.lat,
            self.forecast.lon,
            self.obs_lat,
            self.obs_lon,
            self.obs_motion_bearing,
        )


def motion_bearing(prev_lat: float, prev_lon: float, lat: float, lon: float) -> float:
    return bearing_deg(prev_lat, prev_lon, lat, lon)


@dataclass(frozen=True, slots=True)
class LeadTimeStats:
    lead_hours: int
    n: int
    track_mae_nm: float
    track_rmse_nm: float
    intensity_mae_kt: float
    intensity_bias_kt: float
    cross_track_bias_nm: float | None = None
    along_track_bias_nm: float | None = None


def verify(pairs: list[VerificationPair]) -> dict[int, LeadTimeStats]:
    """Aggregate verification pairs into per-lead-time statistics."""
    by_lead: dict[int, list[VerificationPair]] = {}
    for p in pairs:
        by_lead.setdefault(p.forecast.lead_hours, []).append(p)

    out: dict[int, LeadTimeStats] = {}
    for lead, group in sorted(by_lead.items()):
        track = np.array([p.track_error_nm for p in group])
        intensity = np.array([p.intensity_error_kt for p in group])
        decomposable = [p for p in group if p.obs_motion_bearing is not None]
        cross = along = None
        if decomposable:
            pairs_xy = np.array([p.decomposed_nm() for p in decomposable])
            cross = float(pairs_xy[:, 0].mean())
            along = float(pairs_xy[:, 1].mean())
        out[lead] = LeadTimeStats(
            lead_hours=lead,
            n=len(group),
            track_mae_nm=float(track.mean()),
            track_rmse_nm=float(np.sqrt((track**2).mean())),
            intensity_mae_kt=float(np.abs(intensity).mean()),
            intensity_bias_kt=float(intensity.mean()),
            cross_track_bias_nm=cross,
            along_track_bias_nm=along,
        )
    return out


def to_metric_dict(stats: dict[int, LeadTimeStats]) -> dict[str, float]:
    """Flatten to the §7.3 MLflow metric names."""
    out: dict[str, float] = {}
    for lead, s in stats.items():
        out[f"track_error_{lead}h_nm"] = s.track_mae_nm
        out[f"track_rmse_{lead}h_nm"] = s.track_rmse_nm
        out[f"intensity_error_{lead}h_kt"] = s.intensity_mae_kt
        out[f"intensity_bias_{lead}h_kt"] = s.intensity_bias_kt
        if s.cross_track_bias_nm is not None:
            out[f"cross_track_bias_{lead}h_nm"] = s.cross_track_bias_nm
            out[f"along_track_bias_{lead}h_nm"] = s.along_track_bias_nm
    return out


def beat_rate(candidate_errors: np.ndarray, baseline_errors: np.ndarray) -> float:
    """Fraction of cases where the candidate error is strictly smaller.

    Ties count as losses: a model that matches NHC has not beaten it.
    """
    candidate_errors = np.asarray(candidate_errors, dtype=float)
    baseline_errors = np.asarray(baseline_errors, dtype=float)
    if candidate_errors.shape != baseline_errors.shape:
        raise ValueError("candidate and baseline arrays must have the same shape")
    if candidate_errors.size == 0:
        raise ValueError("cannot compute a beat rate over zero cases")
    return float(np.mean(candidate_errors < baseline_errors))


def diebold_mariano(
    candidate_errors: np.ndarray,
    baseline_errors: np.ndarray,
    *,
    power: int = 2,
) -> tuple[float, float]:
    """Diebold-Mariano test statistic and two-sided p-value (§5.3 Stage 5).

    Uses a lag-0 variance estimate, which is adequate for the storm-wise
    independent samples this is applied to; serially-correlated 6-hourly fixes
    from a single storm should be aggregated to one case per storm first, or the
    test will be anti-conservative.

    Returns ``(statistic, p_value)``. A negative statistic favours the candidate.
    """
    cand = np.asarray(candidate_errors, dtype=float)
    base = np.asarray(baseline_errors, dtype=float)
    if cand.shape != base.shape:
        raise ValueError("candidate and baseline arrays must have the same shape")
    n = cand.size
    if n < 8:
        raise ValueError(f"Diebold-Mariano needs at least 8 cases, got {n}")

    d = cand**power - base**power
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d <= 0:
        return 0.0, 1.0
    stat = float(mean_d / np.sqrt(var_d / n))
    p = float(2.0 * (1.0 - _normal_cdf(abs(stat))))
    return stat, p


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
