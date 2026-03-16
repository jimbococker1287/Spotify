from __future__ import annotations

from spotify.tuning import _resolve_optuna_worker_plan


def test_resolve_optuna_worker_plan_keeps_trial_parallelism_for_single_model() -> None:
    model_workers, trial_jobs = _resolve_optuna_worker_plan(
        model_count=1,
        requested_trial_jobs=3,
        requested_model_workers=2,
    )

    assert model_workers == 1
    assert trial_jobs == 3


def test_resolve_optuna_worker_plan_splits_jobs_across_parallel_studies() -> None:
    model_workers, trial_jobs = _resolve_optuna_worker_plan(
        model_count=4,
        requested_trial_jobs=4,
        requested_model_workers=2,
    )

    assert model_workers == 2
    assert trial_jobs == 2


def test_resolve_optuna_worker_plan_floors_to_one_trial_job() -> None:
    model_workers, trial_jobs = _resolve_optuna_worker_plan(
        model_count=4,
        requested_trial_jobs=2,
        requested_model_workers=3,
    )

    assert model_workers == 3
    assert trial_jobs == 1
