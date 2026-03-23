from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.cli import build_parser
from spotify.config import build_config
from spotify.data import PreparedData
from spotify.ensemble import build_probability_ensemble
from spotify.probability_bundles import align_proba_to_num_classes, save_prediction_bundle


def test_core_and_experimental_profiles_are_available() -> None:
    parser = build_parser()

    core_config = build_config(parser.parse_args(["--profile", "core"]))
    experimental_config = build_config(parser.parse_args(["--profile", "experimental"]))

    assert core_config.profile == "core"
    assert "catboost" in core_config.classical_model_names
    assert "gru_artist" in core_config.model_names

    assert experimental_config.profile == "experimental"
    assert "hist_gbm" in experimental_config.classical_model_names
    assert "transformer_xl" in experimental_config.model_names
    assert experimental_config.enable_moonshot_lab is True


def test_align_proba_to_num_classes_places_sparse_class_columns_correctly() -> None:
    proba = np.array([[0.2, 0.8], [0.7, 0.3]], dtype="float32")
    classes = np.array([1, 3], dtype="int64")

    aligned = align_proba_to_num_classes(proba, classes, num_classes=5)

    assert aligned.shape == (2, 5)
    assert np.allclose(aligned[:, 0], 0.0)
    assert np.allclose(aligned[:, 1], [0.2, 0.7])
    assert np.allclose(aligned[:, 3], [0.8, 0.3])
    assert np.allclose(aligned.sum(axis=1), 1.0)


def test_build_probability_ensemble_creates_blended_result(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.ensemble")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    data = PreparedData(
        df=pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=9, freq="h"),
                "artist_label": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "master_metadata_album_artist_name": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "hour": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "dayofweek": [3] * 9,
                "session_position": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "is_artist_repeat_from_prev": [0] * 9,
                "skipped": [0] * 9,
            }
        ),
        context_features=["hour"],
        X_seq_train=np.zeros((3, 2), dtype="int32"),
        X_seq_val=np.zeros((2, 2), dtype="int32"),
        X_seq_test=np.zeros((2, 2), dtype="int32"),
        X_ctx_train=np.zeros((3, 1), dtype="float32"),
        X_ctx_val=np.zeros((2, 1), dtype="float32"),
        X_ctx_test=np.zeros((2, 1), dtype="float32"),
        y_train=np.array([0, 1, 2], dtype="int32"),
        y_val=np.array([0, 1], dtype="int32"),
        y_test=np.array([0, 1], dtype="int32"),
        y_skip_train=np.zeros(3, dtype="float32"),
        y_skip_val=np.zeros(2, dtype="float32"),
        y_skip_test=np.zeros(2, dtype="float32"),
        num_artists=3,
        num_ctx=1,
    )

    bundle_dir = tmp_path / "bundles"
    bundle_a = save_prediction_bundle(
        bundle_dir / "a.npz",
        val_proba=np.array([[0.75, 0.20, 0.05], [0.20, 0.70, 0.10]], dtype="float32"),
        test_proba=np.array([[0.60, 0.30, 0.10], [0.25, 0.60, 0.15]], dtype="float32"),
    )
    bundle_b = save_prediction_bundle(
        bundle_dir / "b.npz",
        val_proba=np.array([[0.55, 0.35, 0.10], [0.10, 0.80, 0.10]], dtype="float32"),
        test_proba=np.array([[0.50, 0.40, 0.10], [0.15, 0.70, 0.15]], dtype="float32"),
    )

    results = [
        {
            "model_name": "mlp_optuna",
            "base_model_name": "mlp",
            "model_type": "classical_tuned",
            "val_top1": 0.36,
            "prediction_bundle_path": str(bundle_a),
        },
        {
            "model_name": "gru_artist",
            "model_type": "deep",
            "val_top1": 0.28,
            "prediction_bundle_path": str(bundle_b),
        },
    ]

    built = build_probability_ensemble(
        data=data,
        results=results,
        sequence_length=2,
        run_dir=tmp_path,
        logger=logger,
    )

    assert built is not None
    assert built.row["model_name"] == "blended_ensemble"
    assert built.row["model_type"] == "ensemble"
    assert Path(str(built.row["prediction_bundle_path"])).exists()
    assert len(built.row["ensemble_members"]) >= 2
