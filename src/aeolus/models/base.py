"""Shared plumbing for the torch-backed model implementations.

Torch is an optional dependency (``pip install aeolus-meridian[torch]``). The
operational layers -- scheduling, availability, curriculum, promotion, metrics --
carry the v2.1 policy and must remain importable and testable without it, so the
import is deferred and failure is a clear message rather than a stack trace at
module load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TORCH_HINT = (
    "torch is required for model construction; install the optional extra:\n"
    "    uv sync --extra torch"
)


def require_torch() -> Any:
    """Import torch or raise a message that says how to get it."""
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(_TORCH_HINT) from exc
    return torch


def torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Architecture-independent description of a model's I/O contract.

    Carried alongside every model so the fusion layer and the latent extractor
    can be written against the contract rather than against each architecture.
    """

    name: str
    input_dim: int
    latent_dim: int
    #: Forecast lead times the head emits, in hours.
    lead_hours: tuple[int, ...]
    #: Outputs per lead time: (delta_lat, delta_lon, wind).
    outputs_per_lead: int = 3

    @property
    def output_dim(self) -> int:
        return len(self.lead_hours) * self.outputs_per_lead


DEFAULT_LEADS: tuple[int, ...] = (12, 24, 36, 48, 72, 96, 120)
