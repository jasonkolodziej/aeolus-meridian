"""Geodesy helpers.

Track errors in this system are reported in nautical miles because that is the
unit NHC verification uses and the unit Scope v2.1 Appendix B states targets in.
Internally everything is great-circle on a spherical Earth, which is accurate to
well under a nautical mile at hurricane-track distances -- far below the ~50 nm
error scale we are trying to measure.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_NM = 1.852
NM_PER_DEG_LAT = 60.0


def km_to_nm(km: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(km) / KM_PER_NM if isinstance(km, np.ndarray) else km / KM_PER_NM


def nm_to_km(nm: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(nm) * KM_PER_NM if isinstance(nm, np.ndarray) else nm * KM_PER_NM


def haversine_nm(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle distance in nautical miles. Broadcasts over arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    out = dist_km / KM_PER_NM
    return float(out) if np.isscalar(lat1) or out.ndim == 0 else out


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def offset_position(lat: float, lon: float, distance_nm: float, bearing: float) -> tuple[float, float]:
    """Displace a position by ``distance_nm`` along ``bearing`` (deg from north)."""
    ang = nm_to_km(distance_nm) / EARTH_RADIUS_KM
    phi1, lam1 = np.radians(lat), np.radians(lon)
    theta = np.radians(bearing)
    phi2 = np.arcsin(np.sin(phi1) * np.cos(ang) + np.cos(phi1) * np.sin(ang) * np.cos(theta))
    lam2 = lam1 + np.arctan2(
        np.sin(theta) * np.sin(ang) * np.cos(phi1),
        np.cos(ang) - np.sin(phi1) * np.sin(phi2),
    )
    return float(np.degrees(phi2)), float(wrap_longitude(np.degrees(lam2)))


def wrap_longitude(lon: float | np.ndarray) -> float | np.ndarray:
    """Wrap longitude into [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def cross_along_track_nm(
    fcst_lat: float,
    fcst_lon: float,
    obs_lat: float,
    obs_lon: float,
    motion_bearing: float,
) -> tuple[float, float]:
    """Decompose a track error into (cross-track, along-track) nm.

    Positive cross-track means the forecast lies to the right of the observed
    storm motion; positive along-track means the forecast is ahead of it. This
    decomposition is what makes a bias diagnosable (a fast bias and a rightward
    bias have completely different causes).
    """
    dist = haversine_nm(obs_lat, obs_lon, fcst_lat, fcst_lon)
    brg = bearing_deg(obs_lat, obs_lon, fcst_lat, fcst_lon)
    delta = np.radians(brg - motion_bearing)
    return float(dist * np.sin(delta)), float(dist * np.cos(delta))
