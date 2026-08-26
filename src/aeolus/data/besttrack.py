"""Best-track handling: the working/final split and the Stage B noise emulator.

Scope v2.1 §4.1 and §4.6.2.

Two products share the name "best track" and conflating them is the leakage bug
this module exists to prevent:

* **Working** (a-/b-deck, TC-Vitals) -- issued within ~45-90 min of synoptic
  time, from whatever data was on hand. Noisy. This is what a model can
  actually see in production, so it is what a model must be *trained* to see.
* **Final** (HURDAT2) -- reanalysed after the season with aircraft, scatterometer
  and satellite data unavailable in real time, then smoothed. Systematically
  cleaner than anything operationally obtainable. Legitimate as a *label*, never
  as an input.

A model trained on final-quality history and served working-quality history has
been trained on a signal that does not exist at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

import numpy as np

from ..geo import offset_position
from ..time_utils import require_synoptic


class TrackQuality(str, Enum):
    WORKING = "working"
    FINAL = "final"
    #: Working-quality values emulated from final by adding measured error.
    EMULATED = "emulated"
    #: Extrapolated because the real fix was late (§6.2.2 degraded mode).
    ESTIMATED = "estimated"


@dataclass(frozen=True, slots=True)
class Fix:
    """A single storm fix at one synoptic time."""

    storm_id: str
    valid_time: datetime
    lat: float
    lon: float
    max_wind_kt: float
    min_pressure_mb: float
    quality: TrackQuality

    def __post_init__(self) -> None:
        require_synoptic(self.valid_time)
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat out of range: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon out of range: {self.lon}")
        if self.max_wind_kt < 0:
            raise ValueError("max_wind_kt must be non-negative")

    @property
    def is_model_input_safe(self) -> bool:
        """False for FINAL fixes -- they may only be used as labels."""
        return self.quality is not TrackQuality.FINAL


@dataclass(frozen=True, slots=True)
class Track:
    """An ordered sequence of fixes for one storm, all of one quality."""

    storm_id: str
    fixes: tuple[Fix, ...]

    def __post_init__(self) -> None:
        if not self.fixes:
            raise ValueError("a Track needs at least one fix")
        if any(f.storm_id != self.storm_id for f in self.fixes):
            raise ValueError("fix storm_id does not match track storm_id")
        times = [f.valid_time for f in self.fixes]
        if times != sorted(times):
            raise ValueError("fixes must be in ascending time order")
        if len(set(times)) != len(times):
            raise ValueError("duplicate valid_time in track")
        qualities = {f.quality for f in self.fixes}
        if len(qualities) > 1:
            raise ValueError(f"mixed-quality track: {sorted(q.value for q in qualities)}")

    @property
    def quality(self) -> TrackQuality:
        return self.fixes[0].quality

    @property
    def start(self) -> datetime:
        return self.fixes[0].valid_time

    @property
    def end(self) -> datetime:
        return self.fixes[-1].valid_time

    @property
    def season(self) -> int:
        return self.start.year

    @property
    def peak_wind_kt(self) -> float:
        return max(f.max_wind_kt for f in self.fixes)

    def at(self, ts: datetime) -> Fix | None:
        for f in self.fixes:
            if f.valid_time == ts:
                return f
        return None

    def window_ending(self, ts: datetime, length: int) -> tuple[Fix, ...] | None:
        """The ``length`` consecutive fixes ending at ``ts``, or None if short.

        Used to build LSTM input sequences (§4.3 Stage 2). Returns None rather
        than padding: a partially-observed storm history is a different regime
        and should be handled explicitly, not silently zero-filled.
        """
        idx = next((i for i, f in enumerate(self.fixes) if f.valid_time == ts), None)
        if idx is None or idx + 1 < length:
            return None
        return self.fixes[idx + 1 - length : idx + 1]

    def as_array(self) -> np.ndarray:
        """(n, 4) array of lat, lon, wind, pressure."""
        return np.array(
            [[f.lat, f.lon, f.max_wind_kt, f.min_pressure_mb] for f in self.fixes],
            dtype=float,
        )


# Empirical working-vs-final error, Scope v2.1 §4.6.2. These are the documented
# starting values; recalibrate_from_pairs() replaces them with numbers measured
# from our own archive as soon as paired working/final data exists.
DEFAULT_POSITION_RMS_NM = 15.0
DEFAULT_INTENSITY_RMS_KT = 5.0
DEFAULT_PRESSURE_RMS_MB = 3.0


@dataclass(frozen=True, slots=True)
class WorkingTrackNoise:
    """Measured error characteristics of working fixes relative to final."""

    position_rms_nm: float = DEFAULT_POSITION_RMS_NM
    intensity_rms_kt: float = DEFAULT_INTENSITY_RMS_KT
    pressure_rms_mb: float = DEFAULT_PRESSURE_RMS_MB
    #: Working fixes are quantised: positions to 0.1 deg, winds to 5 kt.
    quantise: bool = True

    def __post_init__(self) -> None:
        for name in ("position_rms_nm", "intensity_rms_kt", "pressure_rms_mb"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def emulate_working_fix(
    final: Fix,
    rng: np.random.Generator,
    noise: WorkingTrackNoise | None = None,
) -> Fix:
    """Degrade a FINAL fix into an EMULATED working-quality fix.

    Position error is applied as an isotropic 2-D displacement whose magnitude
    has the requested RMS (Rayleigh-distributed, so the per-axis sigma is
    ``rms / sqrt(2)``), rather than as independent lat/lon jitter -- the latter
    would understate error near the poles and produce a preferred axis.
    """
    if final.quality is not TrackQuality.FINAL:
        raise ValueError(f"expected a FINAL fix, got {final.quality.value}")
    noise = noise or WorkingTrackNoise()

    sigma = noise.position_rms_nm / np.sqrt(2.0)
    dx, dy = rng.normal(0.0, sigma, size=2)
    magnitude = float(np.hypot(dx, dy))
    bearing = float(np.degrees(np.arctan2(dx, dy)) % 360.0)
    lat, lon = offset_position(final.lat, final.lon, magnitude, bearing)

    wind = final.max_wind_kt + rng.normal(0.0, noise.intensity_rms_kt)
    pressure = final.min_pressure_mb + rng.normal(0.0, noise.pressure_rms_mb)

    if noise.quantise:
        lat = round(lat, 1)
        lon = round(lon, 1)
        wind = 5.0 * round(wind / 5.0)

    return replace(
        final,
        lat=float(lat),
        lon=float(lon),
        max_wind_kt=float(max(wind, 0.0)),
        min_pressure_mb=float(pressure),
        quality=TrackQuality.EMULATED,
    )


def emulate_working_track(
    final: Track,
    rng: np.random.Generator,
    noise: WorkingTrackNoise | None = None,
) -> Track:
    """Emulate a whole working-quality track (§4.6.2 Stage B input construction)."""
    return Track(
        storm_id=final.storm_id,
        fixes=tuple(emulate_working_fix(f, rng, noise) for f in final.fixes),
    )


def recalibrate_from_pairs(pairs: list[tuple[Fix, Fix]]) -> WorkingTrackNoise:
    """Measure working-vs-final error from real paired fixes.

    ``pairs`` is a list of ``(working, final)`` fixes at identical valid times.
    Scope v2.1 §4.6.2 requires the emulator constants be replaced with values
    measured from our own archive; this is that measurement.
    """
    if not pairs:
        raise ValueError("need at least one (working, final) pair")
    pos, wind, pres = [], [], []
    for working, final in pairs:
        if working.valid_time != final.valid_time:
            raise ValueError("paired fixes must share a valid_time")
        if working.storm_id != final.storm_id:
            raise ValueError("paired fixes must share a storm_id")
        from ..geo import haversine_nm  # local import keeps module import cheap

        pos.append(haversine_nm(final.lat, final.lon, working.lat, working.lon))
        wind.append(working.max_wind_kt - final.max_wind_kt)
        pres.append(working.min_pressure_mb - final.min_pressure_mb)

    def _rms(values: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    return WorkingTrackNoise(
        position_rms_nm=_rms(pos),
        intensity_rms_kt=_rms(wind),
        pressure_rms_mb=_rms(pres),
    )


def assert_input_safe(fixes: tuple[Fix, ...] | list[Fix]) -> None:
    """Guard: refuse FINAL-quality fixes on the model-input path.

    Called by the Stage B dataset builder and by the inference cycle. Together
    with :func:`aeolus.data.sources.assert_not_operational` this makes the
    train/serve policy a runtime invariant rather than a convention.
    """
    bad = [f for f in fixes if not f.is_model_input_safe]
    if bad:
        raise ValueError(
            f"{len(bad)} FINAL-quality fix(es) reached a model-input path "
            f"(first: {bad[0].storm_id} @ {bad[0].valid_time:%Y-%m-%d %HZ}). "
            "Final best-track is labels-only -- see Scope v2.1 §4.1/§4.6.2."
        )
