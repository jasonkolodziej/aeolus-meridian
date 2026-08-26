"""Two-stage pretrain/fine-tune curriculum (Scope v2.1 §4.6.1)."""

import pytest

from aeolus.data.sources import Flavor
from aeolus.training.curriculum import (
    Curriculum,
    CurriculumError,
    CurriculumRun,
    StageResult,
    assert_deployable,
    stage_a,
    stage_b,
)


def result(stage="B", flavor=Flavor.GDAS_FINETUNE, **kw):
    params = dict(
        stage_name=stage,
        flavor=flavor,
        epochs_completed=10,
        final_train_loss=0.4,
        final_val_loss=0.5,
        checkpoint_uri=f"s3://ckpt/{stage}",
    )
    params.update(kw)
    return StageResult(**params)


def test_stage_a_uses_era5_and_stage_b_uses_gdas():
    assert "era5" in stage_a().input_sources
    assert "gdas_gfs" in stage_b().input_sources
    assert "era5" not in stage_b().input_sources


def test_both_stages_label_from_final_besttrack():
    assert stage_a().label_source == "besttrack_final"
    assert stage_b().label_source == "besttrack_final"


def test_stage_b_learning_rate_is_lower_than_stage_a():
    assert stage_b().learning_rate < stage_a().learning_rate


def test_final_besttrack_cannot_be_a_stage_input():
    with pytest.raises(CurriculumError, match="labels-only"):
        stage_a().__class__(
            name="X",
            flavor=Flavor.ERA5_PRETRAIN,
            input_sources=("besttrack_final",),
            label_source="besttrack_final",
            season_range=(1980, 2000),
            epochs=1,
            learning_rate=1e-3,
        )


def test_stage_b_cannot_read_a_pretrain_only_source():
    with pytest.raises(CurriculumError, match="operational"):
        stage_b().__class__(
            name="B",
            flavor=Flavor.GDAS_FINETUNE,
            input_sources=("era5",),
            label_source="besttrack_final",
            season_range=(2015, 2019),
            epochs=1,
            learning_rate=1e-4,
        )


def test_curriculum_must_end_on_the_operational_flavor():
    with pytest.raises(CurriculumError, match="serving distribution"):
        Curriculum(model_name="lstm", stages=(stage_a(),))


def test_standard_curriculum_is_two_stages_ending_in_b():
    cur = Curriculum.standard("lstm")
    assert [s.name for s in cur.stages] == ["A", "B"]
    assert cur.deployable_stage.flavor is Flavor.GDAS_FINETUNE


def test_stages_must_be_recorded_in_order():
    run = CurriculumRun(Curriculum.standard("lstm"))
    with pytest.raises(CurriculumError, match="out of order"):
        run.record(result("B"))


def test_stage_flavor_mismatch_is_caught():
    run = CurriculumRun(Curriculum.standard("lstm"))
    with pytest.raises(CurriculumError, match="flavor"):
        run.record(result("A", flavor=Flavor.GDAS_FINETUNE))


def test_pretrain_only_run_is_not_deployable():
    run = CurriculumRun(Curriculum.standard("lstm"))
    run.record(result("A", flavor=Flavor.ERA5_PRETRAIN))
    with pytest.raises(CurriculumError, match="incomplete"):
        assert_deployable(run)


def test_completed_two_stage_run_is_deployable():
    run = CurriculumRun(Curriculum.standard("lstm"))
    run.record(result("A", flavor=Flavor.ERA5_PRETRAIN))
    run.record(result("B", flavor=Flavor.GDAS_FINETUNE))
    assert run.complete
    assert_deployable(run)
