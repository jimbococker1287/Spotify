from __future__ import annotations

import builtins
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.data import PreparedData
from spotify.probability_bundles import save_prediction_bundle
import spotify.training as training


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _prepared_data() -> PreparedData:
    return PreparedData(
        df=pd.DataFrame({"artist_label": [0, 1, 2, 1]}),
        context_features=["hour"],
        X_seq_train=np.array([[0, 1], [1, 2]], dtype="int32"),
        X_seq_val=np.array([[1, 2]], dtype="int32"),
        X_seq_test=np.array([[0, 1]], dtype="int32"),
        X_ctx_train=np.zeros((2, 1), dtype="float32"),
        X_ctx_val=np.zeros((1, 1), dtype="float32"),
        X_ctx_test=np.zeros((1, 1), dtype="float32"),
        y_train=np.array([1, 2], dtype="int32"),
        y_val=np.array([2], dtype="int32"),
        y_test=np.array([1], dtype="int32"),
        y_skip_train=np.array([0, 1], dtype="int32"),
        y_skip_val=np.array([1], dtype="int32"),
        y_skip_test=np.array([0], dtype="int32"),
        num_artists=3,
        num_ctx=1,
    )


def test_build_deep_cache_key_changes_with_epochs() -> None:
    base_payload = {
        "schema_version": training.DEEP_TRAINING_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": "abc123",
        "model_name": "dense",
        "epochs": 4,
    }
    changed_payload = dict(base_payload)
    changed_payload["epochs"] = 8

    assert training._build_deep_cache_key(base_payload) != training._build_deep_cache_key(changed_payload)


def test_train_and_evaluate_models_reuses_cached_deep_artifacts_without_tensorflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPOTIFY_CACHE_DEEP", "1")
    monkeypatch.setenv("SPOTIFY_RUN_EAGER", "0")
    monkeypatch.setenv("SPOTIFY_STEPS_PER_EXECUTION", "64")
    monkeypatch.setenv("SPOTIFY_MIXED_PRECISION", "off")
    monkeypatch.setenv("SPOTIFY_DISTRIBUTION_STRATEGY", "none")
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "1")

    data = _prepared_data()
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "run"
    cache_fingerprint = "prepared123"
    model_name = "dense"
    cache_payload = training._build_deep_cache_payload(
        cache_fingerprint=cache_fingerprint,
        model_name=model_name,
        random_seed=42,
        batch_size=32,
        epochs=2,
        sequence_length=int(data.X_seq_train.shape[1]),
        num_artists=int(data.num_artists),
        num_ctx=int(data.num_ctx),
    )
    cache_key = training._build_deep_cache_key(cache_payload)
    cache_paths = training._resolve_deep_model_cache_paths(
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        model_name=model_name,
        cache_key=cache_key,
    )
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths.checkpoint_path.write_bytes(b"keras")
    save_prediction_bundle(
        cache_paths.prediction_bundle_path,
        val_proba=np.asarray([[0.2, 0.7, 0.1]], dtype="float32"),
        test_proba=np.asarray([[0.6, 0.3, 0.1]], dtype="float32"),
    )
    training.write_json(
        cache_paths.result_path,
        {
            "cache_schema_version": training.DEEP_TRAINING_CACHE_SCHEMA_VERSION,
            "result": {
                "history": {
                    "loss": [1.0, 0.7],
                    "val_loss": [1.1, 0.8],
                    "val_sparse_categorical_accuracy": [0.4, 0.6],
                    "val_top_5": [0.8, 0.9],
                },
                "val_metrics": {"top1": 0.6, "top5": 0.9},
                "test_metrics": {"top1": 0.5, "top5": 0.8},
                "fit_seconds": 12.5,
            },
        },
    )
    training.write_json(cache_paths.metadata_path, cache_payload)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise AssertionError("TensorFlow should not be imported on a deep cache hit")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cache_stats: dict[str, object] = {}

    artifacts = training.train_and_evaluate_models(
        data=data,
        model_builders=[(model_name, lambda: None)],
        batch_size=32,
        epochs=2,
        output_dir=output_dir,
        strategy=None,
        logger=_logger("spotify.test.training.cache"),
        random_seed=42,
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
        cache_stats_out=cache_stats,
    )

    assert list(artifacts.histories) == [model_name]
    assert artifacts.histories[model_name].history["val_sparse_categorical_accuracy"][-1] == 0.6
    assert artifacts.val_metrics[model_name]["top1"] == 0.6
    assert artifacts.test_metrics[model_name]["top5"] == 0.8
    assert artifacts.fit_seconds[model_name] == 12.5
    assert artifacts.prediction_bundle_paths[model_name] == str(output_dir / "prediction_bundles" / "deep_dense.npz")
    assert (output_dir / "best_dense.keras").exists()
    assert (output_dir / "prediction_bundles" / "deep_dense.npz").exists()
    assert cache_stats == {
        "enabled": True,
        "fingerprint": cache_fingerprint,
        "hit_model_names": ["dense"],
        "miss_model_names": [],
    }
