"""Data source registry.

Scope v2.1 §4.1. Each source carries an explicit real-time latency and a
*role*. The role is load-bearing, not documentation: :func:`operational_sources`
is what the inference scheduler is allowed to read, and
:func:`assert_not_operational` is the guard that keeps reanalysis out of the
production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum


class Role(str, Enum):
    """What a source is permitted to be used for."""

    #: Read live during a forecast cycle, and used for Stage B fine-tuning.
    OPERATIONAL = "operational"
    #: Stage A pretraining only. Never read during a forecast cycle.
    PRETRAIN_ONLY = "pretrain_only"
    #: Supervision targets and verification. Never a model *input*.
    LABELS_ONLY = "labels_only"


class Flavor(str, Enum):
    """Input distribution a feature or dataset was computed from (§4.6.1)."""

    ERA5_PRETRAIN = "era5_pretrain"
    GDAS_FINETUNE = "gdas_finetune"


@dataclass(frozen=True, slots=True)
class DataSource:
    key: str
    provider: str
    role: Role
    #: Typical wall-clock delay between valid time and availability.
    typical_latency: timedelta
    #: Worst-case delay used for scheduling budgets.
    max_latency: timedelta
    fmt: str
    retention: str
    notes: str = ""

    @property
    def is_operational(self) -> bool:
        return self.role is Role.OPERATIONAL


_H = timedelta(hours=1)
_M = timedelta(minutes=1)
_D = timedelta(days=1)

#: The §4.1 table, as code.
REGISTRY: dict[str, DataSource] = {
    src.key: src
    for src in (
        DataSource(
            key="besttrack_working",
            provider="NHC ATCF (a-/b-deck, TC-Vitals)",
            role=Role.OPERATIONAL,
            typical_latency=45 * _M,
            max_latency=90 * _M,
            fmt="ATCF text",
            retention="permanent",
            notes="Gating input for the cycle. Noisy real-time fixes.",
        ),
        DataSource(
            key="besttrack_final",
            provider="NHC / NOAA (HURDAT2)",
            role=Role.LABELS_ONLY,
            typical_latency=180 * _D,
            max_latency=365 * _D,
            fmt="CSV / ATCF",
            retention="permanent",
            notes="Post-season reanalysed. Labels and verification only.",
        ),
        DataSource(
            key="gdas_gfs",
            provider="NOAA NOMADS",
            role=Role.OPERATIONAL,
            typical_latency=int(3.5 * 60) * _M,
            max_latency=4 * _H,
            fmt="GRIB2",
            retention="2015-present archived; rolling 2yr hot",
            notes="Operational gridded input; cycle t consumes the t-6 cycle.",
        ),
        DataSource(
            key="era5",
            provider="Copernicus CDS",
            role=Role.PRETRAIN_ONLY,
            typical_latency=5 * _D,
            max_latency=90 * _D,
            fmt="GRIB / NetCDF",
            retention="rolling 10yr + permanent storm-relative extracts",
            notes="Stage A pretraining and the ERA5T skew audit only.",
        ),
        DataSource(
            key="goes",
            provider="NOAA AWS / GCS (GOES-18/19)",
            role=Role.OPERATIONAL,
            typical_latency=5 * _M,
            max_latency=20 * _M,
            fmt="NetCDF / Zarr",
            retention="rolling 1yr raw; permanent storm-relative crops",
        ),
        DataSource(
            key="ndbc",
            provider="NDBC buoy / C-MAN",
            role=Role.OPERATIONAL,
            typical_latency=15 * _M,
            max_latency=1 * _H,
            fmt="JSON / NetCDF",
            retention="rolling 3yr",
        ),
        DataSource(
            key="dropsonde",
            provider="NOAA / AFRC",
            role=Role.OPERATIONAL,
            typical_latency=1 * _H,
            max_latency=3 * _H,
            fmt="BUFR / NetCDF",
            retention="permanent",
            notes="Event-driven; absent for most cycles.",
        ),
        DataSource(
            key="microwave",
            provider="RSS / CIMSS",
            role=Role.OPERATIONAL,
            typical_latency=1 * _H,
            max_latency=4 * _H,
            fmt="HDF5",
            retention="rolling 2yr + permanent storm crops",
            notes="Orbit-dependent; opportunistic.",
        ),
        DataSource(
            key="sst_ohc",
            provider="NOAA / Copernicus",
            role=Role.OPERATIONAL,
            typical_latency=1 * _D,
            max_latency=2 * _D,
            fmt="NetCDF",
            retention="rolling 5yr",
            notes="Persisted from the previous day.",
        ),
        DataSource(
            key="ensemble_perturbations",
            provider="GEFS / ECMWF EPS",
            role=Role.OPERATIONAL,
            typical_latency=4 * _H,
            max_latency=6 * _H,
            fmt="GRIB2",
            retention="rolling 1yr",
            notes="MERIDIAN conditioning; t-6 members valid at t.",
        ),
    )
}


def get(key: str) -> DataSource:
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown data source {key!r}") from exc


def operational_sources() -> list[DataSource]:
    """Sources the inference path may read."""
    return [s for s in REGISTRY.values() if s.role is Role.OPERATIONAL]


def pretrain_only_sources() -> list[DataSource]:
    return [s for s in REGISTRY.values() if s.role is Role.PRETRAIN_ONLY]


class OperationalUseError(RuntimeError):
    """A non-operational source was requested inside the inference path."""


def assert_not_operational(key: str) -> None:
    """Guard used by the inference path (Scope v2.1 §3.2 "hard rule").

    Raises :class:`OperationalUseError` when code attempts to read a
    pretrain-only or labels-only source during a forecast cycle. This is the
    mechanical enforcement of the train/serve consistency policy -- without it,
    an ERA5 read is a one-line change that silently reintroduces the skew.
    """
    src = get(key)
    if src.role is Role.OPERATIONAL:
        return
    raise OperationalUseError(
        f"source {key!r} has role {src.role.value!r} and must not be read during "
        "a forecast cycle (Scope v2.1 §4.6). Use the operational equivalent: "
        f"{_operational_substitute(key)}"
    )


def _operational_substitute(key: str) -> str:
    return {
        "era5": "gdas_gfs",
        "besttrack_final": "besttrack_working",
    }.get(key, "see §4.1 role column")
