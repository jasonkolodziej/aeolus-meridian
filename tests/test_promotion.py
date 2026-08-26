"""Promotion gates on validation, operational flavor only (Scope v2.1 §5.3)."""

from datetime import date

import pytest

from aeolus.data.sources import Flavor
from aeolus.training.curriculum import Curriculum, CurriculumRun, StageResult
from aeolus.training.promotion import (
    MetricSet,
    PromotionError,
    TestSetBudget,
    Threshold,
    evaluate_promotion,
)


def good_metrics(split="val", flavor=Flavor.GDAS_FINETUNE, **overrides):
    values = {
        "track_error_48h_nm": 70.0,
        "track_error_72h_nm": 120.0,
        "track_error_120h_nm": 210.0,
        "intensity_error_48h_kt": 11.0,
        "intensity_error_72h_kt": 16.0,
        "nhc_consensus_beat_rate_48h": 0.56,
    }
    values.update(overrides)
    return MetricSet(split=split, flavor=flavor, values=values)


def complete_run():
    run = CurriculumRun(Curriculum.standard("lstm"))
    for name, flavor in (("A", Flavor.ERA5_PRETRAIN), ("B", Flavor.GDAS_FINETUNE)):
        run.record(
            StageResult(name, flavor, 10, 0.4, 0.5, f"s3://ckpt/{name}")
        )
    return run


def test_first_candidate_is_staged():
    decision = evaluate_promotion("lstm", complete_run(), good_metrics())
    assert decision.promote_to_staging


def test_production_requires_a_manual_gate_by_default():
    decision = evaluate_promotion("lstm", complete_run(), good_metrics())
    assert not decision.promote_to_production
    assert any("manual gate" in r for r in decision.reasons)


def test_production_promotes_when_the_manual_gate_is_cleared():
    decision = evaluate_promotion(
        "lstm", complete_run(), good_metrics(), require_manual_gate=False
    )
    assert decision.promote_to_production


def test_test_split_metrics_cannot_gate_promotion():
    with pytest.raises(PromotionError, match="validation split"):
        evaluate_promotion("lstm", complete_run(), good_metrics(split="test"))


def test_era5_flavor_metrics_cannot_gate_promotion():
    with pytest.raises(PromotionError, match="operational-flavor"):
        evaluate_promotion(
            "lstm", complete_run(), good_metrics(flavor=Flavor.ERA5_PRETRAIN)
        )


def test_incomplete_curriculum_blocks_promotion():
    run = CurriculumRun(Curriculum.standard("lstm"))
    run.record(StageResult("A", Flavor.ERA5_PRETRAIN, 10, 0.4, 0.5, "s3://a"))
    with pytest.raises(Exception, match="incomplete"):
        evaluate_promotion("lstm", run, good_metrics())


def test_candidate_worse_than_incumbent_is_not_staged():
    incumbent = good_metrics(track_error_48h_nm=60.0)
    decision = evaluate_promotion(
        "lstm", complete_run(), good_metrics(), incumbent_val_metrics=incumbent
    )
    assert not decision.promote_to_staging
    assert decision.blocked


def test_threshold_failure_blocks_production_but_not_staging():
    metrics = good_metrics(track_error_48h_nm=95.0)
    decision = evaluate_promotion(
        "lstm", complete_run(), metrics, require_manual_gate=False
    )
    assert decision.promote_to_staging
    assert not decision.promote_to_production


def test_missing_metric_is_reported_as_a_failure():
    values = dict(good_metrics().values)
    values.pop("intensity_error_48h_kt")
    metrics = MetricSet("val", Flavor.GDAS_FINETUNE, values)
    decision = evaluate_promotion(
        "lstm", complete_run(), metrics, require_manual_gate=False
    )
    assert not decision.promote_to_production
    assert any("missing" in r for r in decision.reasons)


def test_beat_rate_threshold_is_higher_is_better():
    th = Threshold("nhc_consensus_beat_rate_48h", 0.5, lower_is_better=False)
    assert th.passes(0.55)
    assert not th.passes(0.45)


# --- test-set budget -------------------------------------------------------


def test_budget_allows_a_limited_number_of_looks():
    budget = TestSetBudget(max_evaluations_per_season=2)
    budget.spend("lstm", "pre-season verification", date(2026, 5, 1))
    budget.spend("gnn", "pre-season verification", date(2026, 5, 1))
    assert budget.remaining(2026) == 0


def test_budget_exhaustion_raises():
    budget = TestSetBudget(max_evaluations_per_season=1)
    budget.spend("lstm", "annual check", date(2026, 5, 1))
    with pytest.raises(PromotionError, match="budget"):
        budget.spend("gnn", "another look", date(2026, 6, 1))


def test_budget_requires_a_justification():
    with pytest.raises(PromotionError, match="justification"):
        TestSetBudget().spend("lstm", "   ", date(2026, 5, 1))


def test_budget_resets_across_seasons():
    budget = TestSetBudget(max_evaluations_per_season=1)
    budget.spend("lstm", "2026 check", date(2026, 5, 1))
    budget.spend("lstm", "2027 check", date(2027, 5, 1))
    assert budget.evaluations_in(2027) == 1
