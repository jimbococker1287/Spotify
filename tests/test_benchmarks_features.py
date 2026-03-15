from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import spotify.benchmarks as benchmarks
from spotify.benchmarks import _sequence_feature_block, build_classical_feature_bundle, build_tabular_features, run_classical_benchmarks
from spotify.data import PreparedData


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
