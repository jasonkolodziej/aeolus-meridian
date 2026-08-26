"""Registry, model-set pinning and rollback (Scope v2.1 §7.1, §5.7, §10.2)."""

import pytest

from aeolus.data.sources import Flavor
from aeolus.tracking.registry import (
    ALL_MODELS,
    GROUP1_MODELS,
    ModelRegistry,
    RegistryError,
    Stage,
    invalidated_by,
    latent_signature,
)

METRICS = {"track_error_48h_nm": 70.0}


def register_group1(reg, version_bump=()):
    versions = {}
    for name in GROUP1_MODELS:
        v = reg.register(name, run_id=f"run-{name}", input_flavor=Flavor.GDAS_FINETUNE,
                         metrics=METRICS)
        versions[name] = v.version
    for name in version_bump:
        v = reg.register(name, run_id=f"run-{name}-2", input_flavor=Flavor.GDAS_FINETUNE,
                         metrics=METRICS)
        versions[name] = v.version
    return versions


def full_set(reg, tmp_signature=None):
    versions = register_group1(reg)
    sig = tmp_signature or latent_signature(versions)
    for name in ("diffusion", "fusion"):
        v = reg.register(name, run_id=f"run-{name}", input_flavor=Flavor.GDAS_FINETUNE,
                         metrics=METRICS, latent_signature=sig)
        versions[name] = v.version
    return versions


def test_era5_flavor_weights_cannot_be_registered(tmp_path):
    reg = ModelRegistry(tmp_path)
    with pytest.raises(RegistryError, match="Stage B"):
        reg.register("lstm", run_id="r", input_flavor=Flavor.ERA5_PRETRAIN, metrics=METRICS)


def test_derived_models_require_a_latent_signature(tmp_path):
    reg = ModelRegistry(tmp_path)
    with pytest.raises(RegistryError, match="latent_signature"):
        reg.register("diffusion", run_id="r", input_flavor=Flavor.GDAS_FINETUNE,
                     metrics=METRICS)


def test_versions_increment(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.register("lstm", run_id="a", input_flavor=Flavor.GDAS_FINETUNE,
                        metrics=METRICS).version == 1
    assert reg.register("lstm", run_id="b", input_flavor=Flavor.GDAS_FINETUNE,
                        metrics=METRICS).version == 2


def test_registry_persists_across_instances(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("lstm", run_id="a", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    assert len(ModelRegistry(tmp_path).versions("lstm")) == 1


def test_pinning_a_full_coherent_set_succeeds(tmp_path):
    reg = ModelRegistry(tmp_path)
    versions = full_set(reg)
    pin = reg.pin_set("2026-preseason", versions)
    assert set(pin["members"]) == set(ALL_MODELS)
    assert reg.production("diffusion").version == versions["diffusion"]


def test_pinning_an_incomplete_set_is_refused(tmp_path):
    reg = ModelRegistry(tmp_path)
    versions = register_group1(reg)
    with pytest.raises(RegistryError, match="missing members"):
        reg.pin_set("partial", versions)


def test_retraining_one_group1_model_invalidates_the_derived_models(tmp_path):
    """The v2 'retrain the affected model only' bug, caught mechanically."""
    reg = ModelRegistry(tmp_path)
    versions = full_set(reg)
    new_lstm = reg.register("lstm", run_id="lstm-v2", input_flavor=Flavor.GDAS_FINETUNE,
                            metrics=METRICS)
    versions["lstm"] = new_lstm.version
    with pytest.raises(RegistryError, match="latent signature"):
        reg.pin_set("broken", versions)


def test_signature_changes_when_any_group1_version_changes():
    base = {m: 1 for m in GROUP1_MODELS}
    bumped = dict(base, gnn=2)
    assert latent_signature(base) != latent_signature(bumped)


def test_signature_requires_every_group1_model():
    with pytest.raises(RegistryError, match="missing"):
        latent_signature({"lstm": 1})


def test_invalidation_map_covers_group1_only():
    assert invalidated_by("lstm") == ("diffusion", "fusion")
    assert invalidated_by("diffusion") == ()


def test_promoting_a_new_production_archives_the_incumbent(tmp_path):
    reg = ModelRegistry(tmp_path)
    v1 = reg.register("lstm", run_id="a", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    v2 = reg.register("lstm", run_id="b", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    reg.transition("lstm", v1.version, Stage.PRODUCTION)
    reg.transition("lstm", v2.version, Stage.PRODUCTION)
    assert reg.get("lstm", v1.version).stage is Stage.ARCHIVED
    assert reg.production("lstm").version == v2.version


def test_rollback_restores_the_previous_production_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    v1 = reg.register("lstm", run_id="a", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    v2 = reg.register("lstm", run_id="b", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    reg.transition("lstm", v1.version, Stage.PRODUCTION)
    reg.transition("lstm", v2.version, Stage.PRODUCTION)
    assert reg.rollback("lstm").version == v1.version


def test_rollback_without_history_raises(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("lstm", run_id="a", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    with pytest.raises(RegistryError, match="roll back"):
        reg.rollback("lstm")


def test_mlflow_failure_degrades_to_local_only(tmp_path):
    class BrokenClient:
        def create_model_version(self, *a):
            raise RuntimeError("mlflow unreachable")

    reg = ModelRegistry(tmp_path, mlflow_client=BrokenClient())
    reg.register("lstm", run_id="a", input_flavor=Flavor.GDAS_FINETUNE, metrics=METRICS)
    assert not reg.mlflow_available
    assert len(reg.versions("lstm")) == 1
