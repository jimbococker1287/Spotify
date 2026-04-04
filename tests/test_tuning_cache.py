from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from spotify.probability_bundles import save_prediction_bundle
import spotify.tuning as tuning


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_build_optuna_cache_key_changes_with_budget() -> None:
    base_payload = {
        "schema_version": tuning.OPTUNA_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": "abc123",
        "model_name": "logreg",
        "random_seed": 42,
        "trials": 10,
    }
    changed_payload = dict(base_payload)
    changed_payload["trials"] = 12

    assert tuning._build_optuna_cache_key(base_payload) != tuning._build_optuna_cache_key(changed_payload)


def test_run_optuna_tuning_reuses_cached_result_without_loading_optuna(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPOTIFY_OPTUNA_PRUNER", "median")
    monkeypatch.setenv("SPOTIFY_OPTUNA_PRUNING_FIDELITIES", "0.25,0.6,1.0")
    monkeypatch.setenv("SPOTIFY_OPTUNA_TRIAL_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("SPOTIFY_OPTUNA_STARTUP_TRIALS", "5")
    monkeypatch.setenv("SPOTIFY_OPTUNA_WARMUP_STEPS", "1")
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "run" / "optuna"
    cache_fingerprint = "prepared123"
    model_name = "logreg"
    cache_payload = tuning._build_optuna_cache_payload(
        cache_fingerprint=cache_fingerprint,
        model_name=model_name,
        random_seed=42,
        trials=8,
        max_train_samples=10_000,
        max_eval_samples=4_000,
        model_timeout_seconds=300,
        per_trial_timeout_seconds=120,
        fidelity_schedule=(0.25, 0.6, 1.0),
        pruner_name="median",
    )
    cache_key = tuning._build_optuna_cache_key(cache_payload)
    cache_paths = tuning._resolve_optuna_model_cache_paths(
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        model_name=model_name,
        cache_key=cache_key,
    )
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths.trial_log_path.write_text(
        "trial,state,value,duration_s,params_json\n0,COMPLETE,0.81,0.12,{}\n",
        encoding="utf-8",
    )
    cache_paths.history_plot_path.write_bytes(b"png")
    cache_paths.estimator_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    cache_paths.estimator_artifact_path.write_bytes(b"joblib")
    save_prediction_bundle(
        cache_paths.prediction_bundle_path,
        val_proba=np.asarray([[0.8, 0.2]], dtype="float32"),
        test_proba=np.asarray([[0.7, 0.3]], dtype="float32"),
    )
    tuning.write_json(
        cache_paths.result_path,
        {
            "cache_schema_version": tuning.OPTUNA_CACHE_SCHEMA_VERSION,
            "result": {
                "model_name": "logreg_optuna",
                "base_model_name": "logreg",
                "model_family": "linear",
                "fit_seconds": 1.25,
                "val_top1": 0.81,
                "val_top5": 0.95,
                "val_ndcg_at5": 0.84,
                "val_mrr_at5": 0.82,
                "val_coverage_at5": 0.44,
                "val_diversity_at5": 0.30,
                "test_top1": 0.79,
                "test_top5": 0.93,
                "test_ndcg_at5": 0.8,
                "test_mrr_at5": 0.78,
                "test_coverage_at5": 0.41,
                "test_diversity_at5": 0.28,
                "n_trials": 8,
                "best_params": {"C": 1.0, "max_iter": 300},
                "prediction_bundle_path": "",
                "estimator_artifact_path": "",
            },
        },
    )
    tuning.write_json(cache_paths.metadata_path, cache_payload)

    def _unexpected_optuna_load():
        raise AssertionError("Optuna should not be loaded on a full cache hit")

    monkeypatch.setattr(tuning, "_load_optuna", _unexpected_optuna_load)
    cache_stats: dict[str, object] = {}

    results = tuning.run_optuna_tuning(
        data=object(),  # type: ignore[arg-type]
        output_dir=output_dir,
        selected_models=(model_name,),
        random_seed=42,
        trials=8,
        timeout_seconds=300,
        max_train_samples=10_000,
        max_eval_samples=4_000,
        logger=_logger("spotify.test.tuning.cache"),
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_stats_out=cache_stats,
    )

    assert len(results) == 1
    assert results[0].base_model_name == "logreg"
    assert results[0].prediction_bundle_path == str(output_dir / "prediction_bundles" / "classical_tuned_logreg.npz")
    assert results[0].estimator_artifact_path == str(output_dir / "estimators" / "classical_tuned_logreg.joblib")
    assert (output_dir / "prediction_bundles" / "classical_tuned_logreg.npz").exists()
    assert (output_dir / "estimators" / "classical_tuned_logreg.joblib").exists()
    assert (output_dir / "optuna_trials_logreg.csv").exists()
    assert (output_dir / "optuna_history_logreg.png").exists()
    assert (output_dir / "optuna_results.json").exists()
    assert cache_stats == {
        "enabled": True,
        "fingerprint": cache_fingerprint,
        "hit_model_names": ["logreg"],
        "miss_model_names": [],
    }
