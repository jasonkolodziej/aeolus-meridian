"""Run tagging convention (Scope v2.1 §7.2)."""

import pytest

from aeolus.data.sources import Flavor
from aeolus.tracking.tags import (
    REQUIRED_TAGS,
    ExecutionMode,
    RunTags,
    Trigger,
    validate,
)


def make_tags(**overrides):
    params = dict(
        model_type="lstm",
        git_commit="a3f7d2e",
        dvc_version="v2.4.1-storms-1980-2022",
        storm_split="train-1980-2019_val-2020-2022",
        trigger=Trigger.SCHEDULED_MONTHLY,
        gpu_type="A100-80GB",
        execution_mode=ExecutionMode.PARALLEL_GROUP1,
        input_flavor=Flavor.GDAS_FINETUNE,
    )
    params.update(overrides)
    return RunTags(**params)


def test_v2_1_adds_input_flavor_and_nwp_cycle_lag():
    tags = make_tags().to_dict()
    assert tags["input_flavor"] == "gdas_finetune"
    assert tags["nwp_cycle_lag"] == "6h"


def test_input_flavor_is_required():
    assert "input_flavor" in REQUIRED_TAGS


def test_malformed_git_commit_is_rejected():
    with pytest.raises(ValueError, match="hex sha"):
        make_tags(git_commit="not-a-sha")


def test_malformed_dvc_version_is_rejected():
    with pytest.raises(ValueError, match="dvc_version"):
        make_tags(dvc_version="latest")


def test_malformed_nwp_cycle_lag_is_rejected():
    with pytest.raises(ValueError, match="nwp_cycle_lag"):
        make_tags(nwp_cycle_lag="six hours")


def test_pretrain_flavor_tags_are_not_operational():
    assert not make_tags(input_flavor=Flavor.ERA5_PRETRAIN).is_operational_flavor


def test_none_values_are_dropped_from_the_dict():
    assert "baseline_beaten" not in make_tags().to_dict()
    assert make_tags(baseline_beaten=True).to_dict()["baseline_beaten"] == "true"


def test_validate_accepts_a_complete_tag_set():
    validate(make_tags().to_dict())


def test_validate_reports_missing_tags():
    tags = make_tags().to_dict()
    del tags["storm_split"]
    with pytest.raises(ValueError, match="storm_split"):
        validate(tags)


def test_validate_rejects_an_unknown_flavor():
    tags = make_tags().to_dict()
    tags["input_flavor"] = "reanalysis"
    with pytest.raises(ValueError, match="input_flavor"):
        validate(tags)
