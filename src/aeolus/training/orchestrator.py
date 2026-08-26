"""Training orchestration.

Scope v2.1 §5.1, §5.4 and §5.7. Two execution modes over the same pipeline:

* **Sequential** -- one model at a time, LSTM -> CNN -> Transformer -> GNN ->
  PINN -> latents -> Diffusion -> Fusion. v2's sequential order omitted the
  fusion model; it is included here.
* **Parallel** -- Group 1 concurrently, then latent generation, then Group 2
  (diffusion) and Group 3 (fusion), which are independent of each other and run
  together.

The orchestrator is deliberately backend-agnostic: it computes the schedule and
calls a ``runner`` callable. That keeps the dependency graph testable without
Kubernetes, Slurm, or a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ..tracking.registry import DERIVED_MODELS, GROUP1_MODELS, latent_signature
from ..tracking.tags import ExecutionMode


class Mode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class OrchestrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Per-model resource envelope from the §5.4 tables."""

    model: str
    gpu_memory_gb: int
    hours_low: float
    hours_high: float

    @property
    def mean_hours(self) -> float:
        return (self.hours_low + self.hours_high) / 2.0


PROFILES: dict[str, ResourceProfile] = {
    p.model: p
    for p in (
        ResourceProfile("lstm", 8, 4, 6),
        ResourceProfile("cnn", 16, 8, 12),
        ResourceProfile("transformer", 40, 18, 24),
        ResourceProfile("gnn", 24, 12, 16),
        ResourceProfile("pinn", 16, 6, 10),
        ResourceProfile("diffusion", 48, 24, 36),
        ResourceProfile("fusion", 8, 2, 4),
    )
}

#: Latent generation itself takes time and GPU; v2 left it implicit.
LATENT_GENERATION_HOURS = (1.5, 3.0)


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    kind: str  # "train" | "latents"
    depends_on: tuple[str, ...]
    execution_mode: ExecutionMode
    gpu_memory_gb: int
    hours_low: float
    hours_high: float


@dataclass(frozen=True, slots=True)
class Wave:
    """A set of tasks that may run concurrently."""

    index: int
    tasks: tuple[Task, ...]

    @property
    def gpus_required(self) -> int:
        return len(self.tasks)

    @property
    def hours_low(self) -> float:
        return max(t.hours_low for t in self.tasks)

    @property
    def hours_high(self) -> float:
        return max(t.hours_high for t in self.tasks)


@dataclass(frozen=True, slots=True)
class Schedule:
    mode: Mode
    waves: tuple[Wave, ...]

    @property
    def peak_gpus(self) -> int:
        return max(w.gpus_required for w in self.waves)

    @property
    def total_hours_low(self) -> float:
        return sum(w.hours_low for w in self.waves)

    @property
    def total_hours_high(self) -> float:
        return sum(w.hours_high for w in self.waves)

    def task_names(self) -> tuple[str, ...]:
        return tuple(t.name for w in self.waves for t in w.tasks)

    def wave_of(self, task_name: str) -> int:
        for w in self.waves:
            if any(t.name == task_name for t in w.tasks):
                return w.index
        raise KeyError(task_name)


def _train_task(model: str, mode: ExecutionMode, depends: tuple[str, ...]) -> Task:
    p = PROFILES[model]
    return Task(
        name=model,
        kind="train",
        depends_on=depends,
        execution_mode=mode,
        gpu_memory_gb=p.gpu_memory_gb,
        hours_low=p.hours_low,
        hours_high=p.hours_high,
    )


def _latent_task(depends: tuple[str, ...], mode: ExecutionMode) -> Task:
    return Task(
        name="latents",
        kind="latents",
        depends_on=depends,
        execution_mode=mode,
        gpu_memory_gb=24,
        hours_low=LATENT_GENERATION_HOURS[0],
        hours_high=LATENT_GENERATION_HOURS[1],
    )


def build_schedule(mode: Mode, models: tuple[str, ...] = GROUP1_MODELS) -> Schedule:
    """Build the wave schedule for a full retraining cycle."""
    unknown = [m for m in models if m not in PROFILES]
    if unknown:
        raise OrchestrationError(f"unknown models: {unknown}")
    if not models:
        raise OrchestrationError("no models to train")

    if mode is Mode.SEQUENTIAL:
        waves: list[Wave] = []
        prev: tuple[str, ...] = ()
        # Deterministic order: cheap models first so failures surface early.
        ordered = tuple(m for m in ("lstm", "cnn", "transformer", "gnn", "pinn") if m in models)
        for i, model in enumerate(ordered):
            task = _train_task(model, ExecutionMode.SEQUENTIAL, prev)
            waves.append(Wave(index=i, tasks=(task,)))
            prev = (model,)
        waves.append(Wave(index=len(waves), tasks=(_latent_task(ordered, ExecutionMode.SEQUENTIAL),)))
        for model in DERIVED_MODELS:
            waves.append(
                Wave(
                    index=len(waves),
                    tasks=(_train_task(model, ExecutionMode.SEQUENTIAL, ("latents",)),),
                )
            )
        return Schedule(mode=mode, waves=tuple(waves))

    group1 = tuple(
        _train_task(m, ExecutionMode.PARALLEL_GROUP1, ()) for m in models
    )
    latents = _latent_task(models, ExecutionMode.PARALLEL_GROUP1)
    derived = (
        _train_task("diffusion", ExecutionMode.PARALLEL_GROUP2, ("latents",)),
        _train_task("fusion", ExecutionMode.PARALLEL_GROUP3, ("latents",)),
    )
    return Schedule(
        mode=mode,
        waves=(
            Wave(0, group1),
            Wave(1, (latents,)),
            Wave(2, derived),
        ),
    )


def validate_schedule(schedule: Schedule) -> None:
    """Every dependency must appear in an earlier wave."""
    completed: set[str] = set()
    for wave in schedule.waves:
        for task in wave.tasks:
            missing = [d for d in task.depends_on if d not in completed]
            if missing:
                raise OrchestrationError(
                    f"task {task.name!r} in wave {wave.index} depends on {missing}, "
                    "which has not completed"
                )
        completed.update(t.name for t in wave.tasks)


@dataclass(slots=True)
class RunOutcome:
    task: str
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class OrchestrationResult:
    mode: Mode
    outcomes: list[RunOutcome] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    latent_signature: str | None = None

    @property
    def failed(self) -> list[str]:
        return [o.task for o in self.outcomes if not o.ok]

    @property
    def succeeded(self) -> list[str]:
        return [o.task for o in self.outcomes if o.ok]

    @property
    def complete(self) -> bool:
        return not self.failed and not self.skipped


def run_schedule(
    schedule: Schedule,
    runner: Callable[[Task], RunOutcome],
    *,
    group1_versions: dict[str, int] | None = None,
    fail_fast: bool = False,
) -> OrchestrationResult:
    """Execute a schedule, honouring dependencies on failure.

    A Group 1 failure does not stop its siblings (§10.1: "one model fails,
    others continue"), but it does skip everything downstream of it -- latent
    generation against an incomplete checkpoint set would produce a diffusion
    model trained on a set that will never be pinned.
    """
    result = OrchestrationResult(mode=schedule.mode)
    done: set[str] = set()

    for wave in schedule.waves:
        for task in wave.tasks:
            unmet = [d for d in task.depends_on if d not in done]
            if unmet:
                result.skipped.append(task.name)
                continue
            outcome = runner(task)
            result.outcomes.append(outcome)
            if outcome.ok:
                done.add(task.name)
            elif fail_fast:
                return result

    if group1_versions and not result.failed:
        try:
            result.latent_signature = latent_signature(group1_versions)
        except Exception as exc:  # noqa: BLE001
            result.skipped.append(f"latent_signature: {exc}")
    return result
