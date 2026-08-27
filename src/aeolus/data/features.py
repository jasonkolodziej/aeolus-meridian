"""Derived environmental features, computed through one code path per flavor.

Scope v2.1 §4.3 Stage 3 and §4.6.1. Every derived field exists in an ERA5
flavor (Stage A) and a GDAS flavor (Stage B / production). The requirement that
matters is that the *same function* computes both: if shear is computed one way
for pretraining and another for production, the fine-tuning stage is correcting
for a code difference as well as a data difference, and neither is diagnosable.

:class:`FeatureSet` therefore carries its flavor, and :func:`assert_flavor`
is called at every boundary where a flavor mismatch would be silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .sources import Flavor

# Two structural gaps in this feature set, both affecting intensity rather than
# track, and both worth naming here rather than discovering during a backtest.
#
# 1. No inner-core moisture. Emanuel and Zhang (2017,
#    doi:10.1175/JAS-D-17-0008.1) find intensity error growth is at least as
#    sensitive to inner-core moisture specification as to the wind field.
#    ``rh700_pct`` below is an area mean over the storm-relative box -- an
#    environmental quantity, not an inner-core one.
#
# 2. No ocean feedback. ``sst_c`` and ``ohc_kj_cm2`` are static area means from a
#    daily product persisted from the previous day (see data.availability). A
#    storm's own cold wake -- the upwelling and mixing it induces, which then
#    limits its own intensification -- is nowhere in this system. For intensity
#    that is a first-order feedback, not a refinement. A minimum viable version
#    makes SST/OHC vary along the forecast track with a wake parameterisation
#    rather than freezing them at t-1 day.

#: Names produced by :func:`compute_environment_features`, in output order.
FEATURE_NAMES: tuple[str, ...] = (
    "shear_magnitude_kt",
    "shear_direction_deg",
    "steering_u_kt",
    "steering_v_kt",
    "sst_c",
    "ohc_kj_cm2",
    "rh700_pct",
    "vorticity850_1e5s",
    "potential_intensity_kt",
    "ivt_kg_ms",
)


@dataclass(frozen=True, slots=True)
class GriddedFields:
    """Storm-centred environmental fields on a common 0.25 degree grid.

    A deliberately small subset of §4.3 Stage 1 output: enough to compute the
    derived fields the deterministic models consume, without pretending to be a
    full NWP state vector.
    """

    valid_time: datetime
    flavor: Flavor
    #: All arrays share shape (nlat, nlon) over the storm-relative box.
    u200: np.ndarray
    v200: np.ndarray
    u850: np.ndarray
    v850: np.ndarray
    z500: np.ndarray
    rh700: np.ndarray
    t700: np.ndarray
    mslp: np.ndarray
    sst: np.ndarray
    ohc: np.ndarray

    def __post_init__(self) -> None:
        shapes = {a.shape for a in self._arrays()}
        if len(shapes) != 1:
            raise ValueError(f"ragged field shapes: {sorted(shapes)}")
        if len(next(iter(shapes))) != 2:
            raise ValueError("fields must be 2-D (nlat, nlon)")

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.u200, self.v200, self.u850, self.v850, self.z500,
            self.rh700, self.t700, self.mslp, self.sst, self.ohc,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.u200.shape


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Derived scalar features for one storm at one time, tagged with flavor."""

    valid_time: datetime
    flavor: Flavor
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(FEATURE_NAMES),):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features, got {self.values.shape}"
            )
        if not np.all(np.isfinite(self.values)):
            raise ValueError("non-finite feature value")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, (float(v) for v in self.values), strict=True))

    def __getitem__(self, name: str) -> float:
        return float(self.values[FEATURE_NAMES.index(name)])


class FlavorMismatchError(ValueError):
    """A feature computed from one input distribution reached the other path."""


def assert_flavor(obj: FeatureSet | GriddedFields, expected: Flavor) -> None:
    """Guard against mixing ERA5-derived and GDAS-derived features.

    Called by the Stage A/B dataset builders and by the inference path. In
    production ``expected`` is always ``GDAS_FINETUNE``.
    """
    if obj.flavor is not expected:
        raise FlavorMismatchError(
            f"expected {expected.value} features, got {obj.flavor.value} "
            "(Scope v2.1 §4.6.1: pretrain and operational flavors must not mix)"
        )


def area_mean(field: np.ndarray, radius_frac: float = 0.5) -> float:
    """Mean over a centred sub-box, used for environmental sampling (§4.3)."""
    if not 0.0 < radius_frac <= 1.0:
        raise ValueError("radius_frac must be in (0, 1]")
    nlat, nlon = field.shape
    hlat = max(int(nlat * radius_frac / 2), 1)
    hlon = max(int(nlon * radius_frac / 2), 1)
    clat, clon = nlat // 2, nlon // 2
    box = field[
        max(clat - hlat, 0) : clat + hlat + 1,
        max(clon - hlon, 0) : clon + hlon + 1,
    ]
    return float(np.mean(box))


def deep_layer_shear(fields: GriddedFields) -> tuple[float, float]:
    """200-850 mb shear magnitude (kt) and direction (deg from north).

    Annular sampling (the 200-800 km ring) is the operational convention; here
    we use the centred box mean, which is adequate for the synthetic data this
    reference implementation ships with and is the single place to change when
    real fields are wired in.
    """
    du = area_mean(fields.u200) - area_mean(fields.u850)
    dv = area_mean(fields.v200) - area_mean(fields.v850)
    magnitude = float(np.hypot(du, dv))
    direction = float((np.degrees(np.arctan2(du, dv)) + 360.0) % 360.0)
    return magnitude, direction


def steering_flow(fields: GriddedFields) -> tuple[float, float]:
    """Deep-layer-mean steering wind (u, v) in knots.

    A 850/200 mb mass-weighted mean stands in for the full 850-250 mb integral.
    """
    u = 0.7 * area_mean(fields.u850) + 0.3 * area_mean(fields.u200)
    v = 0.7 * area_mean(fields.v850) + 0.3 * area_mean(fields.v200)
    return float(u), float(v)


def relative_vorticity_850(fields: GriddedFields, dx_km: float = 27.75) -> float:
    """Centred-difference relative vorticity at 850 mb, in 1e-5 s^-1.

    ``dx_km`` defaults to the meridional spacing of a 0.25 degree grid.
    """
    dvdx = np.gradient(fields.v850, axis=1) / (dx_km * 1000.0)
    dudy = np.gradient(fields.u850, axis=0) / (dx_km * 1000.0)
    kt_to_ms = 0.514444
    return float(area_mean((dvdx - dudy) * kt_to_ms) * 1e5)


def potential_intensity(sst_c: float, ohc: float, shear_kt: float) -> float:
    """Empirical potential-intensity proxy in knots.

    An SST/OHC/shear regression standing in for the real algorithm named in §4.3,
    which needs a full thermodynamic sounding. The interface is the one the real
    implementation will keep; swapping it out changes this function only.

    The algorithm to swap in is Emanuel (1995,
    doi:10.1175/1520-0469(1995)052<3969:SOTCTS>2.0.CO;2) as extended by Bister and
    Emanuel (1998, doi:10.1007/BF01030791) -- the open-cycle formulation with
    dissipative heating, which is what the operational ``pcmin`` code implements.
    Citing the 1995 paper alone understates what is actually used.
    """
    base = 15.0 * max(sst_c - 26.0, 0.0)
    ohc_bonus = 0.25 * max(ohc, 0.0)
    shear_penalty = 1.8 * max(shear_kt - 10.0, 0.0)
    return float(np.clip(35.0 + base + ohc_bonus - shear_penalty, 0.0, 200.0))


def integrated_vapor_transport(fields: GriddedFields) -> float:
    """IVT proxy (kg m^-1 s^-1) from 850 mb wind and 700 mb humidity."""
    speed_ms = np.hypot(fields.u850, fields.v850) * 0.514444
    q = np.clip(fields.rh700, 0.0, 100.0) / 100.0 * 0.018
    return float(area_mean(speed_ms * q) * 9806.65 / 9.81)


def compute_environment_features(fields: GriddedFields) -> FeatureSet:
    """The single code path used for both Stage A and Stage B (§4.6.1).

    The only thing that differs between flavors is the ``fields`` handed in;
    the arithmetic is identical, so any train/serve gap that shows up in the
    skew audit is attributable to the data rather than to the code.
    """
    shear_mag, shear_dir = deep_layer_shear(fields)
    steer_u, steer_v = steering_flow(fields)
    sst_c = area_mean(fields.sst)
    ohc = area_mean(fields.ohc)
    values = np.array(
        [
            shear_mag,
            shear_dir,
            steer_u,
            steer_v,
            sst_c,
            ohc,
            area_mean(fields.rh700),
            relative_vorticity_850(fields),
            potential_intensity(sst_c, ohc, shear_mag),
            integrated_vapor_transport(fields),
        ],
        dtype=float,
    )
    return FeatureSet(
        valid_time=fields.valid_time, flavor=fields.flavor, values=values
    )


@dataclass(frozen=True, slots=True)
class Normalizer:
    """Per-feature standardisation fitted on the training split only.

    Fitted statistics are flavor-specific: GDAS analyses have a different mean
    and spread from ERA5, so a normaliser fitted on Stage A must not be reused
    at inference. The flavor travels with the object and is checked on apply.
    """

    flavor: Flavor
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feature_sets: list[FeatureSet]) -> Normalizer:
        if not feature_sets:
            raise ValueError("cannot fit a normalizer on an empty set")
        flavors = {fs.flavor for fs in feature_sets}
        if len(flavors) > 1:
            raise FlavorMismatchError(
                f"cannot fit across mixed flavors: {sorted(f.value for f in flavors)}"
            )
        stacked = np.stack([fs.values for fs in feature_sets])
        std = stacked.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(flavor=flavors.pop(), mean=stacked.mean(axis=0), std=std)

    def apply(self, fs: FeatureSet) -> np.ndarray:
        assert_flavor(fs, self.flavor)
        return (fs.values - self.mean) / self.std
