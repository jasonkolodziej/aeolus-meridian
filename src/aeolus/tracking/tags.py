"""Run tagging convention.

Scope v2.1 §7.2. Tags are validated rather than free-form: an untagged or
mistagged run is one that cannot be found again when a forecast goes wrong, and
the post-mortem requirement in §10.3 depends on every run being locatable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

from ..data.sources import Flavor


class Trigger(str, Enum):
    SCHEDULED_MONTHLY = "scheduled_monthly"
    SEASONAL = "seasonal_preseason"
    DRIFT_DETECTED = "drift_detected"
    SKEW_DETECTED = "skew_detected"
    DATA_VOLUME = "data_volume"
    MANUAL_SWEEP = "manual_sweep"
    NIGHTLY_LATENT = "nightly_latent"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL_GROUP1 = "parallel_group1"
    PARALLEL_GROUP2 = "parallel_group2"
    PARALLEL_GROUP3 = "parallel_group3"


_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_DVC_VERSION = re.compile(r"^v\d+\.\d+\.\d+(-[A-Za-z0-9._-]+)?$")

REQUIRED_TAGS: tuple[str, ...] = (
    "model_type",
    "git_commit",
    "dvc_version",
    "storm_split",
    "trigger",
    "gpu_type",
    "execution_mode",
    "input_flavor",
)


@dataclass(frozen=True, slots=True)
class RunTags:
    model_type: str
    git_commit: str
    dvc_version: str
    storm_split: str
    trigger: Trigger
    gpu_type: str
    execution_mode: ExecutionMode
    #: v2.1: which input distribution this run's weights were fitted on.
    input_flavor: Flavor
    #: v2.1: NWP cycle offset assumed at inference, e.g. "6h".
    nwp_cycle_lag: str = "6h"
    baseline_beaten: bool | None = None
    stage_name: str | None = None

    def __post_init__(self) -> None:
        if not _GIT_SHA.match(self.git_commit):
            raise ValueError(f"git_commit {self.git_commit!r} is not a hex sha")
        if not _DVC_VERSION.match(self.dvc_version):
            raise ValueError(
                f"dvc_version {self.dvc_version!r} must look like v2.4.1-storms-1980-2022"
            )
        if not re.fullmatch(r"\d+h", self.nwp_cycle_lag):
            raise ValueError(f"nwp_cycle_lag {self.nwp_cycle_lag!r} must look like '6h'")
        if not self.model_type:
            raise ValueError("model_type is required")

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, Enum):
                out[key] = value.value
            elif isinstance(value, bool):
                out[key] = "true" if value else "false"
            else:
                out[key] = str(value)
        return out

    @property
    def is_operational_flavor(self) -> bool:
        return self.input_flavor is Flavor.GDAS_FINETUNE


def validate(tags: dict[str, str]) -> None:
    """Check a raw tag dict carries every required key (§7.2)."""
    missing = [k for k in REQUIRED_TAGS if not tags.get(k)]
    if missing:
        raise ValueError(f"run is missing required tags: {missing}")
    flavors = {f.value for f in Flavor}
    if tags["input_flavor"] not in flavors:
        raise ValueError(
            f"input_flavor {tags['input_flavor']!r} must be one of {sorted(flavors)}"
        )
