from __future__ import annotations

import numpy as np

from spotify.tuning import (
    _build_optuna_cache_payload,
    _parse_model_timeout_overrides,
    _resolve_effective_fidelity_schedule,
    _resolve_optuna_worker_plan,
    _resolve_tuning_timeout_reason,
    _sample_aligned_rows,
)


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


def test_resolve_optuna_worker_plan_caps_model_workers_to_total_job_budget() -> None:
    model_workers, trial_jobs = _resolve_optuna_worker_plan(
        model_count=4,
        requested_trial_jobs=2,
        requested_model_workers=3,
    )

    assert model_workers == 2
    assert trial_jobs == 1


def test_sample_aligned_rows_reuses_full_arrays_without_copies() -> None:
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)

    sampled_X, sampled_y = _sample_aligned_rows(
        (X, y),
        max_rows=6,
        rng=np.random.default_rng(42),
    )

    assert sampled_X is X
    assert sampled_y is y


def test_sample_aligned_rows_keeps_sampled_arrays_aligned() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    sampled_X, sampled_y = _sample_aligned_rows(
        (X, y),
        max_rows=4,
        rng=np.random.default_rng(42),
    )

    assert len(sampled_X) == 4
    assert np.array_equal(sampled_X[:, 0] // 2, sampled_y)


def test_pruning_disabled_uses_only_full_fidelity() -> None:
    assert _resolve_effective_fidelity_schedule((0.25, 0.6, 1.0), "none") == (1.0,)
    assert _resolve_effective_fidelity_schedule((0.25, 0.6, 1.0), "median") == (0.25, 0.6, 1.0)


def test_tuning_timeout_reason_checks_trial_and_model_deadlines() -> None:
    assert (
        _resolve_tuning_timeout_reason(
            now=15.0,
            trial_started=4.0,
            study_started=0.0,
            per_trial_timeout_seconds=10,
            model_timeout_seconds=30,
        )
        == "trial timeout exceeded (11.0s)"
    )
    assert (
        _resolve_tuning_timeout_reason(
            now=31.0,
            trial_started=25.0,
            study_started=0.0,
            per_trial_timeout_seconds=10,
            model_timeout_seconds=30,
        )
        == "model timeout exceeded (31.0s)"
    )


def test_zero_timeout_and_pruner_overrides_are_preserved(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_OPTUNA_STARTUP_TRIALS", "0")
    monkeypatch.setenv("SPOTIFY_OPTUNA_WARMUP_STEPS", "0")

    payload = _build_optuna_cache_payload(
        cache_fingerprint="prepared123",
        model_name="logreg",
        random_seed=42,
        trials=8,
        max_train_samples=10_000,
        max_eval_samples=4_000,
        model_timeout_seconds=0,
        per_trial_timeout_seconds=0,
        fidelity_schedule=(0.25, 0.6, 1.0),
        pruner_name="median",
    )

    assert _parse_model_timeout_overrides("logreg=0,mlp=120") == {"logreg": 0, "mlp": 120}
    assert payload["model_timeout_seconds"] == 0
    assert payload["per_trial_timeout_seconds"] == 0
    assert payload["median_startup_trials"] == 0
    assert payload["median_warmup_steps"] == 0
