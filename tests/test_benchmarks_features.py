from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import spotify.benchmarks as benchmarks
from spotify.benchmarks import _sequence_feature_block, build_classical_feature_bundle, build_tabular_features, run_classical_benchmarks
from spotify.data import PreparedData
from spotify.probability_bundles import save_prediction_bundle


def _make_prepared_data() -> PreparedData:
    return PreparedData(
        df=pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=10, freq="h"),
                "artist_label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                "master_metadata_album_artist_name": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            }
        ),
        context_features=["hour", "dayofweek"],
        X_seq_train=np.array([[0, 1], [1, 2], [2, 0], [0, 1]], dtype="int32"),
        X_seq_val=np.array([[1, 2], [2, 0]], dtype="int32"),
        X_seq_test=np.array([[0, 1], [1, 2]], dtype="int32"),
        X_ctx_train=np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]], dtype="float32"),
        X_ctx_val=np.array([[0.2, 0.8], [0.8, 0.2]], dtype="float32"),
        X_ctx_test=np.array([[0.1, 0.9], [0.9, 0.1]], dtype="float32"),
        y_train=np.array([1, 2, 0, 1], dtype="int32"),
        y_val=np.array([2, 0], dtype="int32"),
        y_test=np.array([1, 2], dtype="int32"),
        y_skip_train=np.zeros(4, dtype="float32"),
        y_skip_val=np.zeros(2, dtype="float32"),
        y_skip_test=np.zeros(2, dtype="float32"),
        num_artists=3,
        num_ctx=2,
    )


def test_sequence_feature_block_matches_expected_values() -> None:
    seq = np.array(
        [
            [1, 1, 2, 2],
            [3, 4, 5, 6],
            [7, 7, 7, 7],
        ],
        dtype="int32",
    )

    features = _sequence_feature_block(seq)

    expected = np.array(
        [
            [2.0, 2.0, 1.0, 1.5, 0.5, 0.5, 3.0, 1.0, 0.5],
            [6.0, 5.0, 3.0, 4.5, np.std([3.0, 4.0, 5.0, 6.0]), 1.0, 2.0, 0.0, 1.0],
            [7.0, 7.0, 7.0, 7.0, 0.0, 0.25, 4.0, 1.0, 0.25],
        ],
        dtype="float32",
    )

    assert features.shape == expected.shape
    assert np.allclose(features, expected)


def test_sequence_feature_block_handles_single_step_sequences() -> None:
    seq = np.array([[9], [2]], dtype="int32")

    features = _sequence_feature_block(seq)

    expected = np.array(
        [
            [9.0, 9.0, 9.0, 9.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype="float32",
    )

    assert features.shape == expected.shape
    assert np.allclose(features, expected)


def test_build_classical_feature_bundle_matches_manual_features() -> None:
    data = _make_prepared_data()

    bundle = build_classical_feature_bundle(data)
    X_train, X_val, X_test = build_tabular_features(data)

    assert np.allclose(bundle.X_train, X_train)
    assert np.allclose(bundle.X_val, X_val)
    assert np.allclose(bundle.X_test, X_test)
    assert np.array_equal(bundle.X_train_seq, data.X_seq_train)
    assert np.array_equal(bundle.X_val_seq, data.X_seq_val)
    assert np.array_equal(bundle.X_test_seq, data.X_seq_test)
    assert np.array_equal(bundle.y_train, data.y_train)
    assert np.array_equal(bundle.y_val, data.y_val)
    assert np.array_equal(bundle.y_test, data.y_test)


def test_run_classical_benchmarks_reuses_provided_feature_bundle(tmp_path, monkeypatch) -> None:
    data = _make_prepared_data()
    bundle = build_classical_feature_bundle(data)
    logger = logging.getLogger("spotify.test.benchmarks")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    class _TinyEstimator:
        def fit(self, X: np.ndarray, y: np.ndarray):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), self.classes_[0], dtype=self.classes_.dtype)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            proba = np.zeros((len(X), len(self.classes_)), dtype="float32")
            proba[:, 0] = 1.0
            return proba

    def _fail_build(_data):
        raise AssertionError("bundle should be reused instead of rebuilding classical features")

    monkeypatch.setattr(benchmarks, "build_classical_feature_bundle", _fail_build)
    monkeypatch.setattr(benchmarks, "validate_classical_models", lambda selected_models, random_seed: None)
    monkeypatch.setattr(benchmarks, "resolve_classical_parallelism", lambda: (1, 1))
    monkeypatch.setattr(
        benchmarks,
        "build_classical_estimator",
        lambda model_name, random_seed, params=None, estimator_n_jobs=-1: ("dummy", _TinyEstimator()),
    )

    results = run_classical_benchmarks(
        data=data,
        output_dir=tmp_path,
        selected_models=("dummy",),
        random_seed=7,
        max_train_samples=0,
        max_eval_samples=0,
        logger=logger,
        feature_bundle=bundle,
    )

    assert len(results) == 1
    assert results[0].model_name == "dummy"


def test_run_classical_benchmarks_reuses_cached_results_without_refitting(tmp_path: Path, monkeypatch) -> None:
    data = _make_prepared_data()
    logger = logging.getLogger("spotify.test.benchmarks.cache")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "run"
    cache_fingerprint = "prepared123"
    model_name = "logreg"
    cache_payload = benchmarks._build_classical_cache_payload(
        cache_fingerprint=cache_fingerprint,
        model_name=model_name,
        random_seed=7,
        max_train_samples=0,
        max_eval_samples=0,
        sequence_length=int(data.X_seq_train.shape[1]),
        num_artists=int(data.num_artists),
        num_ctx=int(data.num_ctx),
    )
    cache_key = benchmarks._build_classical_cache_key(cache_payload)
    cache_paths = benchmarks._resolve_classical_model_cache_paths(
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        model_name=model_name,
        cache_key=cache_key,
    )
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths.estimator_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    cache_paths.estimator_artifact_path.write_bytes(b"classical-estimator")
    save_prediction_bundle(
        cache_paths.prediction_bundle_path,
        val_proba=np.asarray([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype="float32"),
        test_proba=np.asarray([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]], dtype="float32"),
    )
    benchmarks.write_json(
        cache_paths.result_path,
        {
            "cache_schema_version": benchmarks.CLASSICAL_BENCHMARK_CACHE_SCHEMA_VERSION,
            "result": {
                "model_name": model_name,
                "model_family": "linear",
                "fit_seconds": 12.5,
                "val_top1": 0.5,
                "val_top5": 1.0,
                "val_ndcg_at5": 0.6,
                "val_mrr_at5": 0.55,
                "val_coverage_at5": 1.0,
                "val_diversity_at5": 1.0,
                "test_top1": 0.4,
                "test_top5": 1.0,
                "test_ndcg_at5": 0.5,
                "test_mrr_at5": 0.45,
                "test_coverage_at5": 1.0,
                "test_diversity_at5": 1.0,
                "prediction_bundle_path": "cached-bundle",
                "estimator_artifact_path": "cached-estimator",
            },
        },
        sort_keys=True,
    )
    benchmarks.write_json(cache_paths.metadata_path, cache_payload, sort_keys=True)

    monkeypatch.setenv("SPOTIFY_CACHE_CLASSICAL", "1")
    monkeypatch.setattr(benchmarks, "build_classical_feature_bundle", lambda _data: (_ for _ in ()).throw(AssertionError("features should not rebuild on a full cache hit")))
    monkeypatch.setattr(benchmarks, "build_classical_estimator", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("estimator should not fit on a full cache hit")))
    cache_stats: dict[str, object] = {}

    results = run_classical_benchmarks(
        data=data,
        output_dir=output_dir,
        selected_models=(model_name,),
        random_seed=7,
        max_train_samples=0,
        max_eval_samples=0,
        logger=logger,
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_stats_out=cache_stats,
    )

    assert len(results) == 1
    assert results[0].model_name == model_name
    assert results[0].fit_seconds == 12.5
    assert results[0].val_top1 == 0.5
    assert (output_dir / "estimators" / "classical_logreg.joblib").read_bytes() == b"classical-estimator"
    assert (output_dir / "prediction_bundles" / "classical_logreg.npz").exists()
    assert (output_dir / "estimators" / "classical_logreg.joblib").samefile(cache_paths.estimator_artifact_path)
    assert (output_dir / "prediction_bundles" / "classical_logreg.npz").samefile(cache_paths.prediction_bundle_path)
    assert (output_dir / "classical_results.json").exists()
    assert cache_stats == {
        "enabled": True,
        "fingerprint": cache_fingerprint,
        "hit_model_names": [model_name],
        "miss_model_names": [],
    }


def test_resolve_classical_parallelism_avoids_nested_jobs_by_default(monkeypatch) -> None:
    monkeypatch.setattr(benchmarks.os, "cpu_count", lambda: 12)
    monkeypatch.setenv("SPOTIFY_CLASSICAL_MODEL_WORKERS", "3")
    monkeypatch.setenv("SPOTIFY_MAX_CLASSICAL_WORKERS", "3")
    monkeypatch.delenv("SPOTIFY_SKLEARN_NJOBS", raising=False)

    workers, estimator_n_jobs = benchmarks.resolve_classical_parallelism()

    assert workers == 3
    assert estimator_n_jobs == 1


def test_resolve_classical_parallelism_respects_explicit_estimator_jobs(monkeypatch) -> None:
    monkeypatch.setattr(benchmarks.os, "cpu_count", lambda: 12)
    monkeypatch.setenv("SPOTIFY_CLASSICAL_MODEL_WORKERS", "3")
    monkeypatch.setenv("SPOTIFY_MAX_CLASSICAL_WORKERS", "3")
    monkeypatch.setenv("SPOTIFY_SKLEARN_NJOBS", "4")

    workers, estimator_n_jobs = benchmarks.resolve_classical_parallelism()

    assert workers == 3
    assert estimator_n_jobs == 4
