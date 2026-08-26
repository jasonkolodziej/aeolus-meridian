"""Sequential and parallel schedules (Scope v2.1 §5.1, §5.4, §5.7)."""

import pytest

from aeolus.training.orchestrator import (
    Mode,
    OrchestrationError,
    RunOutcome,
    build_schedule,
    run_schedule,
    validate_schedule,
)


def ok_runner(task):
    return RunOutcome(task=task.name, ok=True)


def failing_runner(*failures):
    def runner(task):
        return RunOutcome(task=task.name, ok=task.name not in failures)

    return runner


def test_parallel_schedule_has_three_waves():
    schedule = build_schedule(Mode.PARALLEL)
    assert len(schedule.waves) == 3
    assert schedule.waves[0].gpus_required == 5


def test_parallel_latent_stage_sits_between_group1_and_derived():
    schedule = build_schedule(Mode.PARALLEL)
    assert schedule.wave_of("lstm") < schedule.wave_of("latents")
    assert schedule.wave_of("latents") < schedule.wave_of("diffusion")


def test_diffusion_and_fusion_run_concurrently_in_parallel_mode():
    schedule = build_schedule(Mode.PARALLEL)
    assert schedule.wave_of("diffusion") == schedule.wave_of("fusion")


def test_sequential_schedule_includes_fusion():
    """v2's sequential order omitted the fusion model entirely."""
    assert "fusion" in build_schedule(Mode.SEQUENTIAL).task_names()


def test_sequential_schedule_uses_one_gpu_at_a_time():
    assert build_schedule(Mode.SEQUENTIAL).peak_gpus == 1


def test_sequential_is_slower_than_parallel():
    seq = build_schedule(Mode.SEQUENTIAL)
    par = build_schedule(Mode.PARALLEL)
    assert seq.total_hours_high > par.total_hours_high


def test_parallel_cycle_fits_the_appendix_b_target():
    assert build_schedule(Mode.PARALLEL).total_hours_high <= 72


def test_both_schedules_validate():
    for mode in (Mode.SEQUENTIAL, Mode.PARALLEL):
        validate_schedule(build_schedule(mode))


def test_unknown_model_is_rejected():
    with pytest.raises(OrchestrationError, match="unknown"):
        build_schedule(Mode.PARALLEL, models=("nonexistent",))


def test_successful_run_completes_every_task():
    schedule = build_schedule(Mode.PARALLEL)
    result = run_schedule(schedule, ok_runner)
    assert result.complete
    assert len(result.succeeded) == len(schedule.task_names())


def test_group1_failure_lets_siblings_finish_but_blocks_downstream():
    schedule = build_schedule(Mode.PARALLEL)
    result = run_schedule(schedule, failing_runner("gnn"))
    assert "lstm" in result.succeeded
    assert "gnn" in result.failed
    assert "latents" in result.skipped
    assert "diffusion" in result.skipped


def test_latent_signature_is_recorded_only_on_a_clean_run():
    schedule = build_schedule(Mode.PARALLEL)
    versions = {m: 1 for m in ("lstm", "cnn", "transformer", "gnn", "pinn")}
    clean = run_schedule(schedule, ok_runner, group1_versions=versions)
    assert clean.latent_signature
    dirty = run_schedule(schedule, failing_runner("cnn"), group1_versions=versions)
    assert dirty.latent_signature is None


def test_fail_fast_stops_at_the_first_failure():
    schedule = build_schedule(Mode.SEQUENTIAL)
    result = run_schedule(schedule, failing_runner("lstm"), fail_fast=True)
    assert len(result.outcomes) == 1
