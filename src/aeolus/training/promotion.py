"""Promotion gates.

Scope v2.1 §5.3 Stage 6, revised. Two changes from v2, both about not lying to
ourselves:

1. **Promotion is decided on validation, not test.** v2 promoted on test metrics.
   Selecting repeatedly against a held-out set converts it into a validation set
   and the "unbiased evaluation vs NHC consensus" it was reserved for stops
   being unbiased. The test set is touched on an explicit budget
   (:class:`TestSetBudget`) and never as a promotion criterion.
2. **Metrics must come from the operational flavor.** A candidate evaluated on
   ERA5 inputs is not evidence about production behaviour (§4.6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from ..data.sources import Flavor
from .curriculum import CurriculumRun, assert_deployable


class PromotionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Evaluation results for one candidate on one split."""

    split: str
    flavor: Flavor
    values: dict[str, float]

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split {self.split!r}")
        if not self.values:
            raise ValueError("empty metric set")

    def get(self, key: str) -> float:
        try:
            return self.values[key]
        except KeyError as exc:
            raise PromotionError(f"metric {key!r} was not computed") from exc


@dataclass(frozen=True, slots=True)
class Threshold:
    """One promotion criterion."""

    metric: str
    limit: float
    #: True when lower values are better (errors); False for rates.
    lower_is_better: bool = True

    def passes(self, value: float) -> bool:
        return value <= self.limit if self.lower_is_better else value >= self.limit

    def describe(self, value: float) -> str:
        op = "<=" if self.lower_is_better else ">="
        verdict = "pass" if self.passes(value) else "FAIL"
        return f"{self.metric}={value:.3f} {op} {self.limit:.3f} [{verdict}]"


#: Appendix B production thresholds.
#:
#: The absolute track/intensity numbers do not survive contact with the NHC
#: verification record and must be re-derived before anyone treats them as a
#: bar. Official 48 h Atlantic track error was 45.4 n mi in 2024 (a record at
#: every lead time) and 53.4 n mi in 2025; NOAA's own GPRA target for 2026 is
#: 51.0 n mi. Against a 45-55 nm baseline, the 90 nm production threshold below
#: would admit a system roughly twice as bad as the incumbent.
#: See https://www.nhc.noaa.gov/verification/
#:
#: The beat-rate gate is therefore not a supplement to the absolute limits -- it
#: is the only thing standing between this table and that outcome, because it is
#: measured against the live baseline rather than a number written down in 2026.
#:
#: Track and intensity should not share this table's structure either. Intensity
#: skill has improved far more slowly than track skill (Emanuel and Zhang 2016,
#: doi:10.1175/JAS-D-16-0100.1; DeMaria et al. 2014,
#: doi:10.1175/BAMS-D-12-00240.1), so a beat rate on intensity is a claim about
#: a nearly static baseline near an intrinsic predictability limit, while the
#: same number on track is a claim about a rapidly improving one. They are not
#: equivalent achievements and the intensity limits deserve their own derivation.
DEFAULT_THRESHOLDS: tuple[Threshold, ...] = (
    Threshold("track_error_48h_nm", 90.0),
    Threshold("track_error_72h_nm", 150.0),
    Threshold("track_error_120h_nm", 250.0),
    Threshold("intensity_error_48h_kt", 15.0),
    Threshold("intensity_error_72h_kt", 20.0),
    Threshold("nhc_consensus_beat_rate_48h", 0.50, lower_is_better=False),
)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    model_name: str
    promote_to_staging: bool
    promote_to_production: bool
    reasons: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return not self.promote_to_staging


def evaluate_promotion(
    model_name: str,
    run: CurriculumRun,
    val_metrics: MetricSet,
    *,
    incumbent_val_metrics: MetricSet | None = None,
    thresholds: tuple[Threshold, ...] = DEFAULT_THRESHOLDS,
    primary_metric: str = "track_error_48h_nm",
    require_manual_gate: bool = True,
) -> PromotionDecision:
    """Decide staging and production promotion for a candidate.

    Staging requires the candidate to beat the incumbent on ``primary_metric``.
    Production additionally requires every threshold to pass -- and, unless
    ``require_manual_gate`` is cleared, a human. §5.3 kept a manual gate on
    production and this preserves it: an automated pipeline that can put a model
    in front of an emergency manager without anyone looking at it is a hazard,
    not a feature.
    """
    reasons: list[str] = []

    assert_deployable(run)

    if val_metrics.split != "val":
        raise PromotionError(
            f"promotion must be decided on the validation split, got {val_metrics.split!r} "
            "(Scope v2.1 §5.3: the test set is not a promotion criterion)"
        )
    if val_metrics.flavor is not Flavor.GDAS_FINETUNE:
        raise PromotionError(
            f"promotion metrics have flavor {val_metrics.flavor.value}; only "
            "operational-flavor evaluation gates promotion (§4.6.1)"
        )

    if incumbent_val_metrics is None:
        beats_incumbent = True
        reasons.append("no incumbent; candidate is the first registered version")
    else:
        cand = val_metrics.get(primary_metric)
        inc = incumbent_val_metrics.get(primary_metric)
        beats_incumbent = cand < inc
        reasons.append(
            f"{primary_metric}: candidate {cand:.2f} vs incumbent {inc:.2f} "
            f"[{'better' if beats_incumbent else 'not better'}]"
        )

    threshold_results = []
    for th in thresholds:
        value = val_metrics.values.get(th.metric)
        if value is None:
            reasons.append(f"{th.metric}: missing [FAIL]")
            threshold_results.append(False)
            continue
        threshold_results.append(th.passes(value))
        reasons.append(th.describe(value))

    all_thresholds_pass = all(threshold_results)
    promote_production = beats_incumbent and all_thresholds_pass and not require_manual_gate
    if beats_incumbent and all_thresholds_pass and require_manual_gate:
        reasons.append("production promotion awaiting manual gate (§5.3 Stage 6)")

    return PromotionDecision(
        model_name=model_name,
        promote_to_staging=beats_incumbent,
        promote_to_production=promote_production,
        reasons=tuple(reasons),
    )


@dataclass(slots=True)
class TestSetBudget:
    """Rate-limits access to the held-out test set.

    The test set is a finite resource: every look spends a little of its power
    to say anything unbiased. §4.4 reserves 2023-2025 for an unbiased comparison
    against NHC consensus, so evaluations are metered and logged rather than run
    on every candidate.
    """

    max_evaluations_per_season: int = 4
    log: list[tuple[date, str, str]] = field(default_factory=list)

    def evaluations_in(self, season: int) -> int:
        return sum(1 for when, _, _ in self.log if when.year == season)

    def remaining(self, season: int) -> int:
        return max(self.max_evaluations_per_season - self.evaluations_in(season), 0)

    def spend(self, model_name: str, justification: str, when: date | None = None) -> None:
        when = when or datetime.now(UTC).date()
        if not justification.strip():
            raise PromotionError("test-set evaluation requires a written justification")
        if self.remaining(when.year) == 0:
            raise PromotionError(
                f"test-set evaluation budget for {when.year} is exhausted "
                f"({self.max_evaluations_per_season} used). Evaluate on validation, "
                "or raise the budget deliberately and record why."
            )
        self.log.append((when, model_name, justification.strip()))
