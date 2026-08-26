"""Retraining triggers.

Scope v2.1 §5.5, with the §5.7 dependency correction. v2's drift trigger said
"retrain the affected model only", which is wrong for any Group 1 model: the
diffusion and fusion models are trained on latents extracted from Group 1
checkpoints, so retraining an LSTM in isolation leaves MERIDIAN conditioned on
representations that no longer exist. :func:`expand_jobs` applies the cascade so
the invalidation is automatic rather than remembered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from ..tracking.registry import DERIVED_MODELS, GROUP1_MODELS, invalidated_by
from ..tracking.tags import ExecutionMode, Trigger


class Reason(str, Enum):
    SCHEDULED_MONTHLY = "scheduled_monthly"
    PRESEASON = "preseason"
    DRIFT = "drift"
    SKEW = "skew"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"
    NIGHTLY_LATENT = "nightly_latent"
    CASCADE = "cascade"


_REASON_TO_TAG: dict[Reason, Trigger] = {
    Reason.SCHEDULED_MONTHLY: Trigger.SCHEDULED_MONTHLY,
    Reason.PRESEASON: Trigger.SEASONAL,
    Reason.DRIFT: Trigger.DRIFT_DETECTED,
    Reason.SKEW: Trigger.SKEW_DETECTED,
    Reason.DATA_VOLUME: Trigger.DATA_VOLUME,
    Reason.MANUAL: Trigger.MANUAL_SWEEP,
    Reason.NIGHTLY_LATENT: Trigger.NIGHTLY_LATENT,
    Reason.CASCADE: Trigger.DRIFT_DETECTED,
}


@dataclass(frozen=True, slots=True)
class RetrainJob:
    model: str
    reason: Reason
    mode: ExecutionMode
    #: True when this job exists only because an upstream model was retrained.
    cascaded: bool = False
    note: str = ""

    @property
    def trigger_tag(self) -> Trigger:
        return _REASON_TO_TAG[self.reason]


def expand_jobs(jobs: list[RetrainJob]) -> list[RetrainJob]:
    """Add the derived-model retrains implied by any Group 1 retrain (§5.7).

    Ordering is guaranteed: Group 1 first, then latent regeneration consumers
    (diffusion, fusion). Duplicate derived jobs are collapsed.
    """
    out: list[RetrainJob] = []
    seen: set[str] = set()
    for job in jobs:
        if job.model not in seen:
            out.append(job)
            seen.add(job.model)

    cascade_needed: set[str] = set()
    for job in jobs:
        cascade_needed.update(invalidated_by(job.model))

    for model in DERIVED_MODELS:
        if model in cascade_needed and model not in seen:
            out.append(
                RetrainJob(
                    model=model,
                    reason=Reason.CASCADE,
                    mode=ExecutionMode.PARALLEL_GROUP2
                    if model == "diffusion"
                    else ExecutionMode.PARALLEL_GROUP3,
                    cascaded=True,
                    note="latents invalidated by an upstream Group 1 retrain (§5.7)",
                )
            )
            seen.add(model)

    order = {name: i for i, name in enumerate(GROUP1_MODELS + DERIVED_MODELS)}
    return sorted(out, key=lambda j: order.get(j.model, 99))


# ---- individual trigger evaluators ---------------------------------------


def scheduled_monthly(today: date) -> list[RetrainJob]:
    """Full parallel retrain on the 1st of each month (§5.5)."""
    if today.day != 1:
        return []
    return [
        RetrainJob(m, Reason.SCHEDULED_MONTHLY, ExecutionMode.PARALLEL_GROUP1)
        for m in GROUP1_MODELS
    ]


def preseason(today: date) -> list[RetrainJob]:
    """May 1 Atlantic pre-season full refresh (§5.5, §8.1)."""
    if (today.month, today.day) != (5, 1):
        return []
    return [
        RetrainJob(m, Reason.PRESEASON, ExecutionMode.PARALLEL_GROUP1) for m in GROUP1_MODELS
    ]


def on_drift(model: str) -> list[RetrainJob]:
    """Isolated sequential retrain of a drifted model, plus its cascade."""
    return expand_jobs([RetrainJob(model, Reason.DRIFT, ExecutionMode.SEQUENTIAL)])


def on_skew(model: str) -> list[RetrainJob]:
    """Stage B re-fine-tune triggered by the §4.6.3 skew alert.

    Only Stage B reruns: the pretrained representation is still valid, what has
    moved is the operational distribution.
    """
    return expand_jobs([RetrainJob(model, Reason.SKEW, ExecutionMode.SEQUENTIAL,
                                   note="Stage B re-fine-tune only (§4.6.3)")])


def on_data_volume(new_synoptic_times: int, threshold: int = 500) -> list[RetrainJob]:
    """Warm-start retrain once enough new synoptic times have accumulated (§5.5)."""
    if new_synoptic_times < threshold:
        return []
    return [
        RetrainJob(m, Reason.DATA_VOLUME, ExecutionMode.PARALLEL_GROUP1) for m in GROUP1_MODELS
    ]


@dataclass(frozen=True, slots=True)
class SeasonState:
    """Whether a storm is currently active, and the model-set pin protecting it."""

    active_storms: tuple[str, ...] = ()
    pinned_label: str | None = None

    @property
    def storms_active(self) -> bool:
        return bool(self.active_storms)


def nightly_latent(now: datetime, state: SeasonState) -> list[RetrainJob]:
    """02:00 UTC diffusion refresh, suppressed while storms are active.

    v2 ran this unconditionally. Swapping the ensemble generator underneath a
    live storm means consecutive advisories carry forecasts from different
    models, and any spread change is then uninterpretable -- so while a storm is
    active the pinned model set holds and the refresh waits.
    """
    if now.hour != 2:
        return []
    if state.storms_active:
        return []
    return [
        RetrainJob(
            "diffusion",
            Reason.NIGHTLY_LATENT,
            ExecutionMode.PARALLEL_GROUP2,
            note="nightly latent refresh",
        )
    ]


def evaluate_all(
    now: datetime,
    state: SeasonState,
    *,
    drifted_models: tuple[str, ...] = (),
    skewed_models: tuple[str, ...] = (),
    new_synoptic_times: int = 0,
) -> list[RetrainJob]:
    """Run every trigger and return the expanded, de-duplicated job list."""
    today = now.date()
    jobs: list[RetrainJob] = []
    jobs += scheduled_monthly(today)
    jobs += preseason(today)
    jobs += on_data_volume(new_synoptic_times)
    for m in drifted_models:
        jobs += [RetrainJob(m, Reason.DRIFT, ExecutionMode.SEQUENTIAL)]
    for m in skewed_models:
        jobs += [RetrainJob(m, Reason.SKEW, ExecutionMode.SEQUENTIAL)]
    jobs += nightly_latent(now, state)
    return expand_jobs(jobs)
