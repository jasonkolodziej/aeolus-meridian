"""Synthetic data generator.

The real system reads HURDAT2, ERA5, GDAS and GOES. None of those are available
in a test environment, so this module produces data with the same *shape and
statistics* -- including, deliberately, an ERA5-vs-GDAS offset. Without that
offset the train/serve machinery would be untestable: the skew audit would
always report zero and the fine-tuning stage would look pointless.

Everything is seeded and reproducible. This is a stand-in for the ingestion
layer, not a simulation of the atmosphere -- storms follow a plausible
recurving climatology with realistic intensity behaviour, and that is all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from ..geo import wrap_longitude
from .besttrack import Fix, Track, TrackQuality
from .features import GriddedFields
from .sources import Flavor

#: Systematic offsets of GDAS analyses relative to ERA5, applied to synthetic
#: fields so Stage B has something real to correct. Loosely representative of
#: analysis-system differences; not measured from data.
GDAS_BIAS = {
    "wind": 1.5,      # kt, deep-layer
    "rh": -2.0,       # percent
    "sst": -0.15,     # degrees C
    "noise_scale": 1.6,  # GDAS analyses are noisier than reanalysis
}


@dataclass(frozen=True, slots=True)
class StormSpec:
    storm_id: str
    genesis: datetime
    genesis_lat: float
    genesis_lon: float
    duration_hours: int
    peak_wind_kt: float
    recurve_hour: int


def make_storm_spec(rng: np.random.Generator, season: int, index: int) -> StormSpec:
    """Draw a plausible Atlantic storm."""
    genesis_day = int(rng.integers(150, 300))  # June through late October
    genesis = datetime(season, 1, 1, tzinfo=UTC) + timedelta(days=genesis_day)
    genesis = genesis.replace(hour=int(rng.choice([0, 6, 12, 18])), minute=0, second=0, microsecond=0)
    duration = int(rng.integers(8, 40)) * 6  # 2 to 10 days
    return StormSpec(
        storm_id=f"AL{index:02d}{season}",
        genesis=genesis,
        genesis_lat=float(rng.uniform(9.0, 20.0)),
        genesis_lon=float(rng.uniform(-58.0, -25.0)),
        duration_hours=duration,
        peak_wind_kt=float(rng.uniform(45.0, 150.0)),
        recurve_hour=int(duration * rng.uniform(0.4, 0.8)),
    )


def generate_track(spec: StormSpec, rng: np.random.Generator) -> Track:
    """Generate a FINAL-quality track: smooth, as post-season reanalysis is."""
    fixes: list[Fix] = []
    lat, lon = spec.genesis_lat, spec.genesis_lon
    heading = 285.0  # west-northwest initially
    n_steps = spec.duration_hours // 6

    for step in range(n_steps + 1):
        hour = step * 6
        # Recurvature: heading rotates toward the northeast after recurve_hour.
        if hour > spec.recurve_hour:
            heading += rng.uniform(2.0, 6.0)
        else:
            heading += rng.uniform(-1.0, 1.5)
        heading = float(np.clip(heading, 240.0, 80.0 + 360.0)) % 360.0

        speed_kt = 8.0 + 6.0 * (hour / max(spec.duration_hours, 1))
        dist_nm = speed_kt * 6.0
        d_lat = dist_nm * np.cos(np.radians(heading)) / 60.0
        d_lon = dist_nm * np.sin(np.radians(heading)) / (60.0 * max(np.cos(np.radians(lat)), 0.2))
        lat = float(np.clip(lat + d_lat, 5.0, 60.0))
        lon = float(wrap_longitude(lon + d_lon))

        # Intensity: rise to peak near mid-life, then decay.
        phase = hour / max(spec.duration_hours, 1)
        envelope = np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 0.7
        wind = float(np.clip(25.0 + (spec.peak_wind_kt - 25.0) * envelope, 20.0, 185.0))
        if lat > 35.0:  # extratropical decay
            wind *= 0.85
        pressure = float(1010.0 - 0.9 * (wind - 25.0) - 0.002 * (wind - 25.0) ** 2)

        fixes.append(
            Fix(
                storm_id=spec.storm_id,
                valid_time=spec.genesis + timedelta(hours=hour),
                lat=round(lat, 2),
                lon=round(lon, 2),
                max_wind_kt=round(wind, 1),
                min_pressure_mb=round(pressure, 1),
                quality=TrackQuality.FINAL,
            )
        )
    return Track(storm_id=spec.storm_id, fixes=tuple(fixes))


def generate_season(
    season: int, n_storms: int = 12, seed: int | None = None
) -> list[Track]:
    """Generate one season of FINAL-quality tracks."""
    rng = np.random.default_rng(seed if seed is not None else season)
    return [
        generate_track(make_storm_spec(rng, season, i + 1), rng)
        for i in range(n_storms)
    ]


def generate_archive(
    start_season: int = 1980,
    end_season: int = 2025,
    storms_per_season: int = 12,
    seed: int = 20260806,
) -> list[Track]:
    """Generate a multi-decade archive spanning the §4.4 split boundaries."""
    if end_season < start_season:
        raise ValueError("end_season precedes start_season")
    rng = np.random.default_rng(seed)
    tracks: list[Track] = []
    for season in range(start_season, end_season + 1):
        n = max(int(rng.normal(storms_per_season, 3)), 4)
        tracks.extend(generate_season(season, n_storms=n, seed=int(rng.integers(1 << 31))))
    return tracks


def generate_fields(
    valid_time: datetime,
    flavor: Flavor,
    *,
    shape: tuple[int, int] = (41, 41),
    seed: int | None = None,
    shear_kt: float = 12.0,
    sst_c: float = 28.5,
) -> GriddedFields:
    """Generate storm-centred environmental fields in one flavor.

    The GDAS flavor is the ERA5 field plus a systematic bias and extra noise --
    the analysis-system difference Stage B exists to absorb.
    """
    rng = np.random.default_rng(seed if seed is not None else int(valid_time.timestamp()))
    nlat, nlon = shape
    y, x = np.mgrid[0:nlat, 0:nlon]
    cy, cx = nlat / 2.0, nlon / 2.0
    r = np.hypot(y - cy, x - cx) / max(cy, 1.0)

    is_gdas = flavor is Flavor.GDAS_FINETUNE
    bias_wind = GDAS_BIAS["wind"] if is_gdas else 0.0
    noise = GDAS_BIAS["noise_scale"] if is_gdas else 1.0

    def field(base: float, amplitude: float, sigma: float) -> np.ndarray:
        return base + amplitude * np.exp(-(r**2)) + rng.normal(0.0, sigma * noise, shape)

    u850 = field(-8.0, -14.0, 1.2) + bias_wind
    v850 = field(2.0, 9.0, 1.2) + bias_wind
    u200 = u850 + shear_kt + rng.normal(0.0, 1.5 * noise, shape)
    v200 = v850 + rng.normal(0.0, 1.5 * noise, shape)

    return GriddedFields(
        valid_time=valid_time,
        flavor=flavor,
        u200=u200,
        v200=v200,
        u850=u850,
        v850=v850,
        z500=field(5860.0, -60.0, 6.0),
        rh700=np.clip(
            field(62.0, 22.0, 3.0) + (GDAS_BIAS["rh"] if is_gdas else 0.0), 0.0, 100.0
        ),
        t700=field(281.0, -3.0, 0.6),
        mslp=field(1012.0, -45.0, 1.5),
        sst=field(sst_c, 0.8, 0.25) + (GDAS_BIAS["sst"] if is_gdas else 0.0),
        ohc=np.clip(field(60.0, 40.0, 5.0), 0.0, None),
    )


def paired_fields(
    valid_time: datetime, *, seed: int | None = None, **kwargs
) -> tuple[GriddedFields, GriddedFields]:
    """ERA5 and GDAS fields for the same time, for skew-audit testing."""
    return (
        generate_fields(valid_time, Flavor.ERA5_PRETRAIN, seed=seed, **kwargs),
        generate_fields(valid_time, Flavor.GDAS_FINETUNE, seed=seed, **kwargs),
    )
