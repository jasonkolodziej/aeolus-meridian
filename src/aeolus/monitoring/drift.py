"""Data and performance drift detection.

Scope v2.1 §4.6.3 and §8.3. The v2.1-specific requirement is the reference
distribution: drift is measured against the **Stage B fine-tune** distribution,
not the ERA5 pretrain distribution. Comparing live GDAS features to ERA5
statistics would show a large constant offset that is not drift at all, and
would either fire permanently or be tuned until it never fires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.features import FEATURE_NAMES
from ..data.sources import Flavor


class DriftError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceDistribution:
    """Summary statistics of the distribution a model was fine-tuned on."""

    flavor: Flavor
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    n: int

    def __post_init__(self) -> None:
        if self.flavor is not Flavor.GDAS_FINETUNE:
            raise DriftError(
                f"drift reference must be the Stage B ({Flavor.GDAS_FINETUNE.value}) "
                f"distribution, got {self.flavor.value} (§4.6.3)"
            )
        if not (len(self.feature_names) == self.mean.size == self.std.size):
            raise DriftError("feature name/statistic length mismatch")
        if np.any(self.std <= 0):
            raise DriftError("reference std must be positive")

    @classmethod
    def fit(cls, samples: np.ndarray, flavor: Flavor, names=FEATURE_NAMES) -> ReferenceDistribution:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise DriftError("samples must be 2-D (n_samples, n_features)")
        std = samples.std(axis=0, ddof=1)
        std[std < 1e-8] = 1e-8
        return cls(
            flavor=flavor,
            feature_names=tuple(names),
            mean=samples.mean(axis=0),
            std=std,
            n=samples.shape[0],
        )


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    name: str
    standardized_shift: float
    variance_ratio: float
    drifted: bool


@dataclass(frozen=True, slots=True)
class DriftReport:
    features: tuple[FeatureDrift, ...]
    n_live: int

    @property
    def drifted(self) -> tuple[FeatureDrift, ...]:
        return tuple(f for f in self.features if f.drifted)

    @property
    def alert(self) -> bool:
        return bool(self.drifted)

    def summary(self) -> str:
        if not self.alert:
            return f"no feature drift over {self.n_live} live samples"
        worst = max(self.drifted, key=lambda f: abs(f.standardized_shift))
        return (
            f"{len(self.drifted)} drifted feature(s) over {self.n_live} samples; "
            f"worst {worst.name} shift {worst.standardized_shift:+.2f} sigma"
        )


def detect_feature_drift(
    reference: ReferenceDistribution,
    live: np.ndarray,
    *,
    shift_sigma: float = 2.0,
    variance_ratio_limit: float = 3.0,
    min_samples: int = 30,
) -> DriftReport:
    """Compare live features to the Stage B reference.

    Two independent signals: a shift in the mean (expressed in reference sigmas)
    and a change in spread. A model can be broken by either -- a shifted mean
    moves it off its training manifold, while a collapsed variance usually means
    an upstream feed has started returning a constant.
    """
    live = np.asarray(live, dtype=float)
    if live.ndim != 2:
        raise DriftError("live must be 2-D (n_samples, n_features)")
    if live.shape[1] != reference.mean.size:
        raise DriftError("live feature count does not match reference")
    if live.shape[0] < min_samples:
        raise DriftError(
            f"need at least {min_samples} live samples to assess drift, got {live.shape[0]}"
        )

    shift = (live.mean(axis=0) - reference.mean) / reference.std
    live_std = live.std(axis=0, ddof=1)
    live_std[live_std < 1e-8] = 1e-8
    ratio = (live_std / reference.std) ** 2

    features = tuple(
        FeatureDrift(
            name=name,
            standardized_shift=float(s),
            variance_ratio=float(r),
            drifted=bool(abs(s) > shift_sigma or r > variance_ratio_limit or r < 1 / variance_ratio_limit),
        )
        for name, s, r in zip(reference.feature_names, shift, ratio, strict=True)
    )
    return DriftReport(features=features, n_live=live.shape[0])


@dataclass(slots=True)
class ValidationLossMonitor:
    """The §5.5 drift trigger: validation loss up >15% over 3 consecutive checks."""

    threshold_fraction: float = 0.15
    consecutive_required: int = 3
    baseline: float | None = None
    history: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []
        if self.threshold_fraction <= 0:
            raise DriftError("threshold_fraction must be positive")

    def observe(self, val_loss: float) -> bool:
        """Record a check; return True when the retrain trigger fires."""
        if self.baseline is None:
            self.baseline = val_loss
            return False
        self.history.append(val_loss)
        limit = self.baseline * (1.0 + self.threshold_fraction)
        recent = self.history[-self.consecutive_required :]
        return (
            len(recent) == self.consecutive_required
            and all(v > limit for v in recent)
        )

    def reset(self, new_baseline: float) -> None:
        self.baseline = new_baseline
        self.history = []
