"""Model registry.

Scope v2.1 §7.1 and §10.1. MLflow is the intended backend; when it is absent or
unreachable the registry degrades to a local JSON store and syncs later, which
is the §10.1 "MLflow server down" mitigation made real rather than aspirational.

Beyond plain versioning this module implements **model-set pinning**. MERIDIAN's
diffusion model and the fusion model are trained on latents extracted from
specific Group 1 checkpoints. Promoting a new LSTM on its own therefore does not
produce a better system -- it produces a diffusion model conditioned on latents
that no longer exist. Production pins a coherent *set*, and a set is only
promotable when every member agrees on the latent signature it was built from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from ..data.sources import Flavor

GROUP1_MODELS: tuple[str, ...] = ("lstm", "cnn", "transformer", "gnn", "pinn")
DERIVED_MODELS: tuple[str, ...] = ("diffusion", "fusion")
ALL_MODELS: tuple[str, ...] = GROUP1_MODELS + DERIVED_MODELS


class Stage(str, Enum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class RegistryError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelVersion:
    name: str
    version: int
    stage: Stage
    run_id: str
    input_flavor: Flavor
    metrics: dict[str, float]
    tags: dict[str, str]
    #: Identifies the Group 1 checkpoint set the latents were extracted from.
    #: None for Group 1 models themselves.
    latent_signature: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> dict:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["input_flavor"] = self.input_flavor.value
        return data

    @classmethod
    def from_json(cls, data: dict) -> ModelVersion:
        data = dict(data)
        data["stage"] = Stage(data["stage"])
        data["input_flavor"] = Flavor(data["input_flavor"])
        return cls(**data)


class ModelRegistry:
    """Local-first registry with an optional MLflow mirror.

    ``mlflow_client`` is any object exposing MLflow's model-registry surface; it
    is duck-typed so tests can pass a stub and so an MLflow outage degrades to
    the local store instead of failing the run.
    """

    def __init__(self, root: Path | str, mlflow_client=None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "registry.json"
        self._mlflow = mlflow_client
        self._versions: dict[str, list[ModelVersion]] = {}
        self._pins: dict[str, dict] = {}
        self._load()

    # ---- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text())
        self._versions = {
            name: [ModelVersion.from_json(v) for v in versions]
            for name, versions in raw.get("versions", {}).items()
        }
        self._pins = raw.get("pins", {})

    def _save(self) -> None:
        payload = {
            "versions": {
                name: [v.to_json() for v in versions]
                for name, versions in self._versions.items()
            },
            "pins": self._pins,
        }
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @property
    def mlflow_available(self) -> bool:
        return self._mlflow is not None

    # ---- registration ----------------------------------------------------

    def register(
        self,
        name: str,
        *,
        run_id: str,
        input_flavor: Flavor,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        latent_signature: str | None = None,
    ) -> ModelVersion:
        """Register a new version.

        Rejects non-operational-flavor weights outright: per §4.6.1 only Stage B
        output is a candidate for anything, so an ERA5-flavor artifact has no
        business in the registry at all.
        """
        if name not in ALL_MODELS:
            raise RegistryError(f"unknown model {name!r}; expected one of {ALL_MODELS}")
        if input_flavor is not Flavor.GDAS_FINETUNE:
            raise RegistryError(
                f"{name}: refusing to register {input_flavor.value} weights -- only "
                "Stage B (operational-flavor) artifacts are registrable (§4.6.1)"
            )
        if name in DERIVED_MODELS and not latent_signature:
            raise RegistryError(
                f"{name}: derived models must record the latent_signature of the "
                "Group 1 set they were trained against (§5.7)"
            )

        versions = self._versions.setdefault(name, [])
        version = ModelVersion(
            name=name,
            version=len(versions) + 1,
            stage=Stage.NONE,
            run_id=run_id,
            input_flavor=input_flavor,
            metrics=dict(metrics),
            tags=dict(tags or {}),
            latent_signature=latent_signature,
        )
        versions.append(version)
        self._save()
        self._mirror("create_model_version", name, version.version)
        return version

    def get(self, name: str, version: int) -> ModelVersion:
        for v in self._versions.get(name, []):
            if v.version == version:
                return v
        raise RegistryError(f"{name} v{version} not found")

    def versions(self, name: str) -> list[ModelVersion]:
        return list(self._versions.get(name, []))

    def latest(self, name: str) -> ModelVersion:
        versions = self._versions.get(name)
        if not versions:
            raise RegistryError(f"no versions registered for {name!r}")
        return versions[-1]

    def in_stage(self, name: str, stage: Stage) -> ModelVersion | None:
        for v in reversed(self._versions.get(name, [])):
            if v.stage is stage:
                return v
        return None

    def production(self, name: str) -> ModelVersion | None:
        return self.in_stage(name, Stage.PRODUCTION)

    # ---- transitions -----------------------------------------------------

    def transition(self, name: str, version: int, stage: Stage) -> ModelVersion:
        """Move a version to a stage, archiving any incumbent in that stage."""
        target = self.get(name, version)
        if stage is Stage.PRODUCTION:
            incumbent = self.production(name)
            if incumbent is not None and incumbent.version != version:
                incumbent.stage = Stage.ARCHIVED
        target.stage = stage
        self._save()
        self._mirror("transition_model_version_stage", name, version, stage.value)
        return target

    def rollback(self, name: str) -> ModelVersion:
        """Revert production to the most recent previously-archived version (§10.2)."""
        current = self.production(name)
        candidates = [
            v
            for v in self._versions.get(name, [])
            if v.stage is Stage.ARCHIVED and (current is None or v.version != current.version)
        ]
        if not candidates:
            raise RegistryError(f"{name}: no archived version to roll back to")
        previous = max(candidates, key=lambda v: v.version)
        if current is not None:
            current.stage = Stage.ARCHIVED
        previous.stage = Stage.PRODUCTION
        self._save()
        return previous

    # ---- model-set pinning ----------------------------------------------

    def pin_set(self, label: str, members: dict[str, int]) -> dict:
        """Pin a coherent set of versions as the production system.

        Every model in :data:`ALL_MODELS` must be present, all members must be
        operational-flavor, and every derived model's ``latent_signature`` must
        match the signature computed from the pinned Group 1 versions. This is
        what stops a single-model retrain from silently invalidating MERIDIAN.
        """
        missing = [m for m in ALL_MODELS if m not in members]
        if missing:
            raise RegistryError(f"model set is missing members: {missing}")

        resolved = {name: self.get(name, ver) for name, ver in members.items()}
        signature = latent_signature({n: resolved[n].version for n in GROUP1_MODELS})

        for name in DERIVED_MODELS:
            got = resolved[name].latent_signature
            if got != signature:
                raise RegistryError(
                    f"{name} v{resolved[name].version} was trained against latent "
                    f"signature {got!r}, but the pinned Group 1 set is {signature!r}. "
                    "Regenerate latents and retrain the derived models (§5.7)."
                )

        pin = {
            "label": label,
            "members": {n: v.version for n, v in resolved.items()},
            "latent_signature": signature,
            "pinned_at": datetime.now(UTC).isoformat(),
        }
        self._pins[label] = pin
        for name, v in resolved.items():
            self.transition(name, v.version, Stage.PRODUCTION)
        self._save()
        return pin

    def active_pin(self) -> dict | None:
        if not self._pins:
            return None
        return max(self._pins.values(), key=lambda p: p["pinned_at"])

    # ---- mlflow mirror ---------------------------------------------------

    def _mirror(self, method: str, *args) -> None:
        """Best-effort mirror to MLflow; failures degrade to local-only (§10.1)."""
        if self._mlflow is None:
            return
        fn = getattr(self._mlflow, method, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - tracking must never fail a training run
            self._mlflow = None


def latent_signature(group1_versions: dict[str, int]) -> str:
    """Deterministic signature of a Group 1 checkpoint set.

    Any change to any Group 1 version changes the signature, which is precisely
    what invalidates downstream diffusion and fusion models.
    """
    missing = [m for m in GROUP1_MODELS if m not in group1_versions]
    if missing:
        raise RegistryError(f"latent signature needs all Group 1 models; missing {missing}")
    parts = [f"{name}v{group1_versions[name]}" for name in GROUP1_MODELS]
    return "-".join(parts)


def invalidated_by(model_name: str) -> tuple[str, ...]:
    """Which models must be retrained when ``model_name`` is retrained (§5.7).

    Used by the drift trigger so that "retrain the affected model only" cannot
    leave the derived models conditioned on latents that no longer exist.
    """
    if model_name in GROUP1_MODELS:
        return DERIVED_MODELS
    return ()
