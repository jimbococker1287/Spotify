from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from spotify.analytics_db import refresh_analytics_database
from spotify.governance import evaluate_champion_gate
from spotify.serving import load_predictor, resolve_model_row
from spotify.session_knn import SessionKNNClassifier


def test_session_knn_predict_proba_returns_valid_distribution() -> None:
    X_train = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [1, 2, 4],
            [3, 4, 5],
        ],
        dtype="int32",
    )
    y_train = np.array([4, 5, 4, 6], dtype="int32")
    model = SessionKNNClassifier(n_neighbors=2, candidate_cap=4, smoothing=0.5).fit(X_train, y_train)

    proba = model.predict_proba(np.array([[1, 2, 3], [3, 4, 5]], dtype="int32"))

    assert proba.shape == (2, len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_significance_gate_blocks_non_significant_lift(tmp_path: Path) -> None:
    history_csv = tmp_path / "history.csv"
    history_csv.write_text("run_id,model_name,val_top1\n", encoding="utf-8")
    backtest_history_csv = tmp_path / "backtest_history.csv"
    backtest_history_csv.write_text(
        "run_id,profile,model_name,top1\n"
        "run_a,full,mlp,0.250\n"
        "run_a,full,mlp,0.251\n"
        "run_a,full,mlp,0.249\n",
        encoding="utf-8",
    )

    result = evaluate_champion_gate(
        history_csv=history_csv,
        backtest_history_csv=backtest_history_csv,
        current_run_id="run_b",
        current_results=[{"model_name": "ensemble", "val_top1": 0.5}],
        current_backtest_rows=[
            {"model_name": "ensemble", "top1": 0.2515},
            {"model_name": "ensemble", "top1": 0.2505},
            {"model_name": "ensemble", "top1": 0.2510},
        ],
        regression_threshold=0.01,
        metric_source="backtest_top1",
        current_profile="full",
        require_significant_lift=True,
        significance_z=1.96,
    )

    assert result["promoted"] is False
    assert result["status"] == "fail_not_significant"


def test_serving_can_resolve_and_load_classical_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    estimator_dir = run_dir / "estimators"
    estimator_dir.mkdir(parents=True)

    model = SessionKNNClassifier(n_neighbors=2, candidate_cap=4).fit(
        np.array([[1, 2, 3], [2, 3, 4], [1, 2, 4]], dtype="int32"),
        np.array([4, 5, 4], dtype="int32"),
    )
    estimator_path = estimator_dir / "classical_session_knn.joblib"
    joblib.dump(model, estimator_path)

    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": [f"artist_{idx}" for idx in range(8)], "sequence_length": 3}),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "session_knn",
                    "model_type": "classical",
                    "val_top1": 0.2,
                    "estimator_artifact_path": str(estimator_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    row = resolve_model_row(run_dir, explicit_model_name=None, alias_model_name=None)
    predictor = load_predictor(
        run_dir=run_dir,
        row=row,
        artist_labels=[f"artist_{idx}" for idx in range(8)],
    )
    proba = predictor.predict_proba(
        np.array([[1, 2, 3]], dtype="int32"),
        np.zeros((1, 2), dtype="float32"),
    )

    assert predictor.model_type == "classical"
    assert proba.shape == (1, 8)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_analytics_db_refresh_builds_duckdb(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.analytics")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "Streaming_History_Audio_2024_0.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2026-01-01T00:00:00Z",
                    "master_metadata_album_artist_name": "A",
                    "platform": "ios",
                    "reason_start": "trackdone",
                    "reason_end": "trackdone",
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs"
    (output_dir / "history").mkdir(parents=True)
    (output_dir / "runs" / "run_a").mkdir(parents=True)
    pd.DataFrame([{"run_id": "run_a", "profile": "core", "model_name": "mlp", "val_top1": 0.3, "test_top1": 0.2}]).to_csv(
        output_dir / "history" / "experiment_history.csv",
        index=False,
    )
    pd.DataFrame([{"run_id": "run_a", "profile": "core", "model_name": "mlp", "top1": 0.25}]).to_csv(
        output_dir / "history" / "backtest_history.csv",
        index=False,
    )
    (output_dir / "runs" / "run_a" / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "profile": "core",
                "timestamp": "2026-01-01T00:00:00",
                "champion_gate": {"promoted": True, "metric_source": "backtest_top1"},
                "champion_alias": {"model_name": "mlp", "model_type": "classical"},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "runs" / "run_a" / "run_results.json").write_text(
        json.dumps([{"model_name": "mlp", "model_type": "classical", "val_top1": 0.3, "test_top1": 0.2}]),
        encoding="utf-8",
    )

    db_path = refresh_analytics_database(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )

    assert db_path is not None
    assert Path(db_path).exists()


def test_analytics_db_refresh_can_reuse_preloaded_raw_df(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.analytics.raw_df")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    (output_dir / "history").mkdir(parents=True)
    (output_dir / "runs" / "run_a").mkdir(parents=True)
    pd.DataFrame([{"run_id": "run_a", "profile": "core", "model_name": "mlp", "val_top1": 0.3, "test_top1": 0.2}]).to_csv(
        output_dir / "history" / "experiment_history.csv",
        index=False,
    )
    (output_dir / "runs" / "run_a" / "run_manifest.json").write_text(
        json.dumps({"run_id": "run_a", "profile": "core", "timestamp": "2026-01-01T00:00:00"}),
        encoding="utf-8",
    )
    (output_dir / "runs" / "run_a" / "run_results.json").write_text(
        json.dumps([{"model_name": "mlp", "model_type": "classical", "val_top1": 0.3, "test_top1": 0.2}]),
        encoding="utf-8",
    )

    raw_df = pd.DataFrame(
        [
            {
                "ts": "2026-01-01T00:00:00Z",
                "master_metadata_album_artist_name": "A",
                "platform": "ios",
                "reason_start": "trackdone",
                "reason_end": "trackdone",
            }
        ]
    )

    db_path = refresh_analytics_database(
        data_dir=tmp_path / "missing-data-dir",
        output_dir=output_dir,
        include_video=False,
        logger=logger,
        raw_df=raw_df,
    )

    assert db_path is not None
    assert Path(db_path).exists()
