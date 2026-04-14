from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from spotify.data import PreparedData
from spotify.training import (
    _resolve_tensorflow_input_mode,
    _weighted_top1_accuracy_from_proba,
    _weighted_topk_accuracy_from_proba,
    compute_baselines,
    compute_sample_weights,
)


def _logger() -> logging.Logger:
    logger = logging.getLogger("spotify.test.training")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _prepared_data() -> PreparedData:
    return PreparedData(
        df=pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=8, freq="h"),
                "artist_label": [10, 10, 20, 30, 10, 20, 30, 10],
                "master_metadata_album_artist_name": ["A", "A", "B", "C", "A", "B", "C", "A"],
            }
        ),
        context_features=["hour"],
        X_seq_train=np.array([[0, 1], [1, 1], [1, 2]], dtype="int32"),
        X_seq_val=np.array([[0, 1], [1, 2]], dtype="int32"),
        X_seq_test=np.array([[1, 1], [2, 2]], dtype="int32"),
        X_ctx_train=np.zeros((3, 1), dtype="float32"),
        X_ctx_val=np.zeros((2, 1), dtype="float32"),
        X_ctx_test=np.zeros((2, 1), dtype="float32"),
        y_train=np.array([1, 1, 2], dtype="int32"),
        y_val=np.array([1, 2], dtype="int32"),
        y_test=np.array([1, 2], dtype="int32"),
        y_skip_train=np.array([0, 1, 1], dtype="float32"),
        y_skip_val=np.array([1, 0], dtype="float32"),
        y_skip_test=np.array([0, 1], dtype="float32"),
        num_artists=3,
        num_ctx=1,
    )


def test_compute_baselines_matches_expected_metrics() -> None:
    metrics = compute_baselines(_prepared_data(), _logger())

    assert np.isclose(metrics["majority_top1"], 0.5)
    assert np.isclose(metrics["last_artist_top1"], 1.0)
    assert np.isclose(metrics["markov_top1"], 1.0)


def test_compute_sample_weights_supports_non_contiguous_labels() -> None:
    data = _prepared_data()
    data = PreparedData(
        df=data.df,
        context_features=data.context_features,
        X_seq_train=data.X_seq_train,
        X_seq_val=data.X_seq_val,
        X_seq_test=data.X_seq_test,
        X_ctx_train=data.X_ctx_train,
        X_ctx_val=data.X_ctx_val,
        X_ctx_test=data.X_ctx_test,
        y_train=np.array([10, 10, 20], dtype="int32"),
        y_val=np.array([20, 30], dtype="int32"),
        y_test=np.array([30, 10], dtype="int32"),
        y_skip_train=data.y_skip_train,
        y_skip_val=data.y_skip_val,
        y_skip_test=data.y_skip_test,
        num_artists=data.num_artists,
        num_ctx=data.num_ctx,
    )

    weights = compute_sample_weights(data)

    assert np.allclose(weights.artist_train, np.array([0.75, 0.75, 1.5], dtype="float32"))
    assert np.allclose(weights.artist_val, np.array([1.5, 1.0], dtype="float32"))
    assert np.allclose(weights.artist_test, np.array([1.0, 0.75], dtype="float32"))
    assert np.all(np.isfinite(weights.skip_train))
    assert np.all(np.isfinite(weights.skip_val))
    assert np.all(np.isfinite(weights.skip_test))


def test_weighted_accuracy_helpers_match_expected_scores() -> None:
    proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.3, 0.5],
            [0.2, 0.7, 0.1],
        ],
        dtype="float32",
    )
    y_true = np.array([0, 1, 2], dtype="int32")
    weights = np.array([1.0, 2.0, 3.0], dtype="float32")

    top1 = _weighted_top1_accuracy_from_proba(proba, y_true, weights)
    top2 = _weighted_topk_accuracy_from_proba(proba, y_true, weights, k=2)

    assert np.isclose(top1, 1.0 / 6.0)
    assert np.isclose(top2, 0.5)


class _DummyTFConfig:
    def __init__(self, gpu_count: int):
        self._gpu_count = int(gpu_count)

    def list_logical_devices(self, device_type: str):
        if device_type != "GPU":
            return []
        return [object() for _ in range(self._gpu_count)]


class _DummyTF:
    def __init__(self, gpu_count: int):
        self.config = _DummyTFConfig(gpu_count)


class _DummyStrategy:
    def __init__(self, replicas: int):
        self.num_replicas_in_sync = int(replicas)


def test_resolve_tensorflow_input_mode_prefers_arrays_on_cpu_only_darwin(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_TF_INPUT_MODE", raising=False)
    monkeypatch.setattr("spotify.training.sys.platform", "darwin")

    mode, reason = _resolve_tensorflow_input_mode(tf=_DummyTF(gpu_count=0), strategy=_DummyStrategy(replicas=1))

    assert mode == "arrays"
    assert reason == "auto(darwin_cpu_single_device)"


def test_resolve_tensorflow_input_mode_honors_forced_dataset(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_TF_INPUT_MODE", "dataset")
    monkeypatch.setattr("spotify.training.sys.platform", "darwin")

    mode, reason = _resolve_tensorflow_input_mode(tf=_DummyTF(gpu_count=0), strategy=_DummyStrategy(replicas=1))

    assert mode == "dataset"
    assert reason == "forced_dataset"
