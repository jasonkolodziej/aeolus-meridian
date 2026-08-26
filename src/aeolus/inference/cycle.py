"""Forecast cycle execution.

Scope v2.1 §6.1 and §10.1. The cycle is written around its failure modes,
because a hurricane forecast that arrives late or not at all is a worse outcome
than a degraded one that arrives on time:

* diffusion crash -> AEOLUS deterministic plus a climatological spread ensemble
* stale NWP      -> run on t-12 with a staleness flag on the payload
* late vitals    -> extrapolated fix, flagged ``vitals=estimated``

Every degradation is recorded on :class:`CycleOutput` so downstream consumers
can see what they are being handed. Silent degradation is the failure mode that
loses trust.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..data import sources
from ..data.besttrack import Fix, TrackQuality, assert_input_safe
from ..data.sources import Flavor
from ..geo import offset_position
from ..time_utils import cycle_label
from .postprocess import CLIMATOLOGICAL_CONE_NM, EnsembleMember, ForecastProducts, build_products
from .scheduler import CyclePlan


class CycleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeterministicForecast:
    """AEOLUS output: one track and intensity series."""

    target_time: datetime
    lead_hours: tuple[int, ...]
    lats: np.ndarray
    lons: np.ndarray
    winds_kt: np.ndarray
    #: Which models contributed, and their fusion weights.
    contributors: dict[str, float] = field(default_factory=dict)

    def at(self, lead: int) -> tuple[float, float, float]:
        i = self.lead_hours.index(lead)
        return float(self.lats[i]), float(self.lons[i]), float(self.winds_kt[i])


@dataclass(frozen=True, slots=True)
class CycleOutput:
    plan: CyclePlan
    deterministic: DeterministicForecast
    products: ForecastProducts
    flags: tuple[str, ...]
    completed_at: datetime

    @property
    def label(self) -> str:
        return cycle_label(self.plan.target_time)

    @property
    def degraded(self) -> bool:
        return bool(self.flags)

    @property
    def on_time(self) -> bool:
        return self.completed_at <= self.plan.advisory_deadline

    def payload(self) -> dict:
        """JSON-shaped dissemination payload (§6.1 output dissemination)."""
        return {
            "cycle": self.label,
            "issued_at": self.completed_at.isoformat(),
            "advisory_deadline": self.plan.advisory_deadline.isoformat(),
            "nwp_cycle_lag_hours": self.plan.inputs.nwp.lag_hours,
            "vitals": "estimated" if self.plan.vitals_estimated else "observed",
            "ensemble_size": self.products.ensemble_size,
            "rapid_intensification": self.products.rapid_intensification,
            "ri_probability": round(self.products.ri_probability, 3),
            "cone": [
                {
                    "lead_hours": c.lead_hours,
                    "lat": round(c.center_lat, 3),
                    "lon": round(c.center_lon, 3),
                    "radius_nm": round(c.radius_nm, 1),
                    "basis": c.basis,
                }
                for c in self.products.cone
            ],
            "flags": list(self.flags),
        }


DeterministicFn = Callable[[CyclePlan, Fix], DeterministicForecast]
EnsembleFn = Callable[[DeterministicForecast], list[EnsembleMember]]


def climatological_ensemble(
    deterministic: DeterministicForecast,
    n_members: int = 20,
    seed: int = 0,
) -> list[EnsembleMember]:
    """Fallback ensemble when MERIDIAN is unavailable (§10.1).

    Members are drawn around the deterministic track using the climatological
    cone radii, so the resulting spread is honest about being climatology rather
    than a learned distribution. It is emphatically not a substitute for the
    diffusion ensemble -- it carries no flow-dependent structure -- but it keeps
    a probabilistic product on the wire.
    """
    rng = np.random.default_rng(seed)
    members: list[EnsembleMember] = []
    for i in range(n_members):
        lats, lons, winds = [], [], []
        for k, lead in enumerate(deterministic.lead_hours):
            radius = CLIMATOLOGICAL_CONE_NM.get(lead, 60.0)
            # 67th-percentile radius -> sigma via the Rayleigh quantile.
            sigma = radius / np.sqrt(-2.0 * np.log(1.0 - 0.67))
            dist = float(abs(rng.normal(0.0, sigma)))
            bearing = float(rng.uniform(0.0, 360.0))
            lat, lon = offset_position(
                float(deterministic.lats[k]), float(deterministic.lons[k]), dist, bearing
            )
            lats.append(lat)
            lons.append(lon)
            winds.append(float(deterministic.winds_kt[k] + rng.normal(0.0, 8.0)))
        members.append(
            EnsembleMember(
                member_id=i,
                lead_hours=deterministic.lead_hours,
                lats=np.array(lats),
                lons=np.array(lons),
                winds_kt=np.clip(np.array(winds), 0.0, None),
            )
        )
    return members


def run_cycle(
    plan: CyclePlan,
    initial_fix: Fix,
    deterministic_fn: DeterministicFn,
    ensemble_fn: EnsembleFn,
    *,
    now: datetime | None = None,
    coastline: tuple[float, float] | None = None,
    ensemble_seed: int = 0,
) -> CycleOutput:
    """Run one forecast cycle end to end.

    Guardrails applied before any model runs:

    * the initial fix must not be FINAL-quality (§4.6.2 -- final best-track is a
      label, and in real time it does not exist yet anyway);
    * every source the plan resolved must be operational-role (§4.6).
    """
    flags: list[str] = []

    assert_input_safe([initial_fix])
    if initial_fix.valid_time != plan.target_time:
        raise CycleError(
            f"initial fix valid {initial_fix.valid_time:%Y-%m-%d %HZ} does not match "
            f"cycle {plan.label}"
        )
    for key in plan.inputs.available_keys():
        sources.assert_not_operational(key)

    if plan.vitals_estimated:
        flags.append("vitals=estimated")
        if initial_fix.quality is not TrackQuality.ESTIMATED:
            raise CycleError(
                "plan expects an extrapolated fix but the supplied fix is "
                f"{initial_fix.quality.value}"
            )
    if plan.inputs.nwp.degraded:
        flags.append(f"nwp_stale={plan.inputs.nwp.lag_hours}h")
    for status in plan.inputs.statuses:
        if status.unexpectedly_missing:
            flags.append(f"missing:{status.source_key}")

    deterministic = deterministic_fn(plan, initial_fix)

    try:
        members = ensemble_fn(deterministic)
        if not members:
            raise CycleError("ensemble generator returned no members")
    except Exception as exc:  # noqa: BLE001 - any MERIDIAN failure degrades, never blocks
        flags.append(f"meridian_fallback:{type(exc).__name__}")
        members = climatological_ensemble(deterministic, seed=ensemble_seed)

    products = build_products(
        members, coastline=coastline, degraded=bool(flags)
    )
    return CycleOutput(
        plan=plan,
        deterministic=deterministic,
        products=products,
        flags=tuple(flags),
        completed_at=now or plan.meridian_ready_target,
    )


def extrapolate_fix(history: tuple[Fix, ...], target_time: datetime) -> Fix:
    """Extrapolate a fix when TC-Vitals is late (§6.2.2 degraded mode).

    Linear extrapolation of the last motion vector, with intensity held
    constant. Persistence is the right choice here: over the ~45 minutes we are
    bridging, any cleverer intensity model adds variance without adding skill.
    """
    if len(history) < 2:
        raise CycleError("need at least two prior fixes to extrapolate")
    prev, last = history[-2], history[-1]
    dt_prior = (last.valid_time - prev.valid_time).total_seconds() / 3600.0
    dt_target = (target_time - last.valid_time).total_seconds() / 3600.0
    if dt_prior <= 0 or dt_target <= 0:
        raise CycleError("fix history must precede the target time")

    scale = dt_target / dt_prior
    return Fix(
        storm_id=last.storm_id,
        valid_time=target_time,
        lat=last.lat + (last.lat - prev.lat) * scale,
        lon=last.lon + (last.lon - prev.lon) * scale,
        max_wind_kt=last.max_wind_kt,
        min_pressure_mb=last.min_pressure_mb,
        quality=TrackQuality.ESTIMATED,
    )


def fusion_weights(
    recent_errors: dict[str, float], *, floor: float = 0.02
) -> dict[str, float]:
    """Inverse-error fusion weights (§6.1 fusion layer, §10.1 divergence).

    Weights are inversely proportional to recent validation error, with a floor
    so no model is fully zeroed on a short noisy record -- mid-season there may
    be only a handful of verifying storms, and a model discarded on three cases
    is unrecoverable for the rest of the season.
    """
    if not recent_errors:
        raise CycleError("no models to weight")
    if any(v <= 0 for v in recent_errors.values()):
        raise CycleError("recent errors must be positive")
    raw = {k: 1.0 / v for k, v in recent_errors.items()}
    total = sum(raw.values())
    weights = {k: max(v / total, floor) for k, v in raw.items()}
    norm = sum(weights.values())
    return {k: v / norm for k, v in weights.items()}


def assert_operational_flavor(flavor: Flavor) -> None:
    """Inference may only use operational-flavor features (§4.6.1)."""
    if flavor is not Flavor.GDAS_FINETUNE:
        raise CycleError(
            f"inference received {flavor.value} features; production runs on "
            f"{Flavor.GDAS_FINETUNE.value} only"
        )
