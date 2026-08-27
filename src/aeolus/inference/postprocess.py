"""Post-processing of the MERIDIAN ensemble into forecast products.

Scope v2.1 §6.1. Cone of uncertainty, intensity PDF, landfall probability and
the rapid-intensification flag.

Note on the cone: NHC's official cone is built from *historical* official-forecast
error percentiles, not from the current ensemble spread. A cone drawn from a
model's own spread is only honest if that spread is calibrated, which is exactly
what :func:`aeolus.metrics.probabilistic.spread_skill` measures. This module
therefore computes both and refuses to emit an ensemble-derived cone when the
ensemble is underdispersed -- falling back to the climatological radii instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import haversine_nm
from ..metrics.probabilistic import ensemble_percentiles

#: NHC-style 2/3-probability circle radii (nm) by lead time.
#:
#: NHC sizes each circle so that two-thirds of official forecast errors over the
#: previous five years fall within it, and re-derives the table annually
#: (https://www.nhc.noaa.gov/aboutcone.shtml). The values below are placeholders
#: close to the published 2023 Atlantic radii; replace with the current season's
#: table rather than assuming these stay reasonable.
#:
#: Note for the 2026 season NHC is trialling a cone built from **ellipses** rather
#: than circles, separating the along-track (speed) and cross-track (directional)
#: error components. That is the decomposition metrics.track.cross_along_track_nm
#: already computes, so an elliptical ConeSegment -- semi-major, semi-minor and
#: orientation in place of a single radius -- is a natural extension here.
CLIMATOLOGICAL_CONE_NM: dict[int, float] = {
    12: 26.0,
    24: 41.0,
    36: 55.0,
    48: 70.0,
    72: 102.0,
    96: 151.0,
    120: 198.0,
}

#: Rapid intensification: >=30 kt increase in 24 h.
#:
#: Kaplan and DeMaria (2003, doi:10.1175/1520-0434(2003)018<1093:LCORIT>2.0.CO;2)
#: define RI as roughly the 95th percentile of over-water 24 h intensity changes
#: for Atlantic storms developing 1989-2000, which works out to 30 kt. It is a
#: percentile of a continuous distribution, not a physical discontinuity -- later
#: work finds no gap in the distribution near 30 kt. Treat both this and the
#: alert probability in build_products() as tunable conventions, not physics.
RI_THRESHOLD_KT = 30.0
RI_WINDOW_HOURS = 24


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """One member's track and intensity at a set of lead times."""

    member_id: int
    lead_hours: tuple[int, ...]
    lats: np.ndarray
    lons: np.ndarray
    winds_kt: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.lead_hours)
        if not (self.lats.size == self.lons.size == self.winds_kt.size == n):
            raise ValueError("member arrays must match lead_hours length")

    def at(self, lead: int) -> tuple[float, float, float]:
        i = self.lead_hours.index(lead)
        return float(self.lats[i]), float(self.lons[i]), float(self.winds_kt[i])


@dataclass(frozen=True, slots=True)
class ConeSegment:
    lead_hours: int
    center_lat: float
    center_lon: float
    radius_nm: float
    #: "ensemble" when derived from spread, "climatology" when fallen back.
    basis: str


@dataclass(frozen=True, slots=True)
class ForecastProducts:
    cone: tuple[ConeSegment, ...]
    intensity_pdf: dict[int, dict[int, float]]
    landfall_probability: float | None
    rapid_intensification: bool
    ri_probability: float
    ensemble_size: int
    degraded: bool = False
    notes: tuple[str, ...] = ()


def deterministic_center(members: list[EnsembleMember], lead: int) -> tuple[float, float]:
    """Ensemble-mean position at a lead time."""
    pts = np.array([m.at(lead)[:2] for m in members], dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def cone_radius_from_spread(
    members: list[EnsembleMember], lead: int, percentile: float = 67.0
) -> float:
    """Radius containing ``percentile``% of members around the ensemble mean."""
    clat, clon = deterministic_center(members, lead)
    dists = np.array(
        [haversine_nm(clat, clon, *m.at(lead)[:2]) for m in members], dtype=float
    )
    return float(np.percentile(dists, percentile))


def build_cone(
    members: list[EnsembleMember],
    lead_times: tuple[int, ...] | None = None,
    *,
    min_members: int = 10,
    underdispersion_fraction: float = 0.5,
) -> tuple[tuple[ConeSegment, ...], tuple[str, ...]]:
    """Build the cone, falling back to climatology when spread is untrustworthy.

    At forecast time there is no observation to verify against, so calibration
    cannot be measured directly -- :func:`spread_skill` is the post-hoc check,
    run in verification. The real-time proxy used here is the ensemble radius
    against the climatological radius at the longest lead: an ensemble
    dramatically tighter than climatology five days out is displaying the
    underdispersion signature of a diffusion model over-anchored to its
    deterministic conditioning, and a cone drawn from it would understate risk.
    """
    if not members:
        raise ValueError("cannot build a cone from an empty ensemble")
    leads = lead_times or members[0].lead_hours
    notes: list[str] = []

    use_ensemble = len(members) >= min_members
    if not use_ensemble:
        notes.append(
            f"ensemble of {len(members)} below the {min_members}-member minimum; "
            "cone drawn from climatological radii"
        )

    if use_ensemble:
        long_lead = leads[-1]
        ensemble_radius = cone_radius_from_spread(members, long_lead)
        climo_radius = CLIMATOLOGICAL_CONE_NM.get(long_lead, _interpolate_climo(long_lead))
        if ensemble_radius < underdispersion_fraction * climo_radius:
            use_ensemble = False
            notes.append(
                f"ensemble radius at {long_lead}h ({ensemble_radius:.0f} nm) is under "
                f"{underdispersion_fraction:.0%} of climatology ({climo_radius:.0f} nm); "
                "cone drawn from climatological radii instead"
            )

    segments: list[ConeSegment] = []
    for lead in leads:
        clat, clon = deterministic_center(members, lead)
        if use_ensemble:
            radius = cone_radius_from_spread(members, lead)
            basis = "ensemble"
        else:
            radius = CLIMATOLOGICAL_CONE_NM.get(lead, _interpolate_climo(lead))
            basis = "climatology"
        segments.append(ConeSegment(lead, clat, clon, radius, basis))
    return tuple(segments), tuple(notes)


def _interpolate_climo(lead: int) -> float:
    leads = np.array(sorted(CLIMATOLOGICAL_CONE_NM))
    radii = np.array([CLIMATOLOGICAL_CONE_NM[int(k)] for k in leads], dtype=float)
    return float(np.interp(lead, leads, radii))


def intensity_pdf(
    members: list[EnsembleMember], lead_times: tuple[int, ...] | None = None
) -> dict[int, dict[int, float]]:
    """Per-lead-time intensity percentiles (10/25/50/75/90)."""
    leads = lead_times or members[0].lead_hours
    return {
        lead: ensemble_percentiles(
            np.array([m.at(lead)[2] for m in members]), percentiles=(10, 25, 50, 75, 90)
        )
        for lead in leads
    }


def landfall_probability(
    members: list[EnsembleMember],
    coastline_lat: float,
    coastline_lon: float,
    threshold_nm: float = 60.0,
    lead: int | None = None,
) -> float:
    """Fraction of members passing within ``threshold_nm`` of a coastal point."""
    if not members:
        raise ValueError("empty ensemble")
    leads = (lead,) if lead is not None else members[0].lead_hours
    hits = 0
    for m in members:
        closest = min(
            haversine_nm(coastline_lat, coastline_lon, *m.at(l)[:2]) for l in leads
        )
        hits += int(closest <= threshold_nm)
    return hits / len(members)


def rapid_intensification_probability(
    members: list[EnsembleMember],
    *,
    threshold_kt: float = RI_THRESHOLD_KT,
    window_hours: int = RI_WINDOW_HOURS,
) -> float:
    """Fraction of members showing a >=threshold intensity gain in any window."""
    if not members:
        raise ValueError("empty ensemble")
    count = 0
    for m in members:
        leads = np.array(m.lead_hours, dtype=float)
        winds = np.asarray(m.winds_kt, dtype=float)
        gained = False
        for i, lead_i in enumerate(leads):
            j = np.searchsorted(leads, lead_i + window_hours)
            if j < len(leads) and winds[j] - winds[i] >= threshold_kt:
                gained = True
                break
        count += int(gained)
    return count / len(members)


def build_products(
    members: list[EnsembleMember],
    *,
    coastline: tuple[float, float] | None = None,
    ri_alert_threshold: float = 0.3,
    degraded: bool = False,
) -> ForecastProducts:
    """Assemble the §6.1 product suite from an ensemble."""
    cone, notes = build_cone(members)
    ri_prob = rapid_intensification_probability(members)
    return ForecastProducts(
        cone=cone,
        intensity_pdf=intensity_pdf(members),
        landfall_probability=(
            landfall_probability(members, *coastline) if coastline else None
        ),
        rapid_intensification=ri_prob >= ri_alert_threshold,
        ri_probability=ri_prob,
        ensemble_size=len(members),
        degraded=degraded or any("climatological" in n for n in notes),
        notes=notes,
    )
