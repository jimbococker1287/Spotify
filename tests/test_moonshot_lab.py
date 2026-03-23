from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from spotify.data import PreparedData
from spotify.moonshot_lab import run_moonshot_lab
from spotify.reporting import write_run_report


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _prepared_data() -> PreparedData:
    timestamps = pd.date_range("2026-03-01", periods=12, freq="h")
    hours = np.arange(12, dtype="float32")
    return PreparedData(
        df=pd.DataFrame(
            {
                "ts": timestamps,
                "artist_label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "master_metadata_album_artist_name": ["A", "B", "C"] * 4,
                "hour": hours,
                "hour_sin": np.sin((hours / 24.0) * (2.0 * np.pi)),
                "hour_cos": np.cos((hours / 24.0) * (2.0 * np.pi)),
                "dayofweek": [6] * 12,
                "session_position": [0, 1, 2] * 4,
                "session_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                "skipped": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                "offline": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "tech_playback_errors_24h": [0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 1, 0],
                "energy": [0.40, 0.62, 0.55, 0.45, 0.65, 0.58, 0.43, 0.66, 0.60, 0.44, 0.63, 0.57],
                "danceability": [0.50, 0.72, 0.64, 0.52, 0.74, 0.66, 0.51, 0.73, 0.67, 0.53, 0.71, 0.65],
                "tempo": [105.0, 122.0, 115.0, 108.0, 124.0, 117.0, 106.0, 123.0, 118.0, 107.0, 121.0, 116.0],
            }
        ),
        context_features=["hour", "tech_playback_errors_24h", "offline"],
        X_seq_train=np.array([[0, 1], [1, 2], [2, 0], [0, 1], [1, 2]], dtype="int32"),
        X_seq_val=np.array([[2, 1], [0, 2]], dtype="int32"),
        X_seq_test=np.array([[1, 0], [2, 1]], dtype="int32"),
        X_ctx_train=np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 1.0], [4.0, 0.0, 0.0]],
            dtype="float32",
        ),
        X_ctx_val=np.array([[6.0, 0.0, 0.0], [7.0, 2.0, 1.0]], dtype="float32"),
        X_ctx_test=np.array([[8.0, 0.0, 0.0], [9.0, 3.0, 1.0]], dtype="float32"),
        y_train=np.array([2, 0, 1, 2, 0], dtype="int32"),
        y_val=np.array([0, 1], dtype="int32"),
        y_test=np.array([2, 0], dtype="int32"),
        y_skip_train=np.array([0, 1, 0, 0, 1], dtype="float32"),
        y_skip_val=np.array([0, 1], dtype="float32"),
        y_skip_test=np.array([0, 1], dtype="float32"),
        num_artists=3,
        num_ctx=3,
    )


def test_moonshot_lab_writes_component_artifacts_and_root_summary(tmp_path: Path) -> None:
    data = _prepared_data()
    run_dir = tmp_path / "run_a"
    retrieval_artifact_path = tmp_path / "retrieval.joblib"
    joblib.dump(SimpleNamespace(artist_embeddings=np.eye(3, dtype="float32")), retrieval_artifact_path)

    artifact_paths = run_moonshot_lab(
        data=data,
        results=[
            {
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "retrieval_artifact_path": str(retrieval_artifact_path),
            }
        ],
        run_dir=run_dir,
        sequence_length=2,
        artist_labels=["A", "B", "C"],
        random_seed=42,
        logger=_logger("spotify.test.moonshot"),
    )

    moonshot_summary = run_dir / "analysis" / "moonshot_summary.json"
    assert moonshot_summary.exists()
    assert moonshot_summary in artifact_paths
    assert (run_dir / "analysis" / "multimodal" / "multimodal_artist_space_summary.json").exists()
    assert (run_dir / "analysis" / "causal" / "causal_skip_summary.json").exists()
    assert (run_dir / "analysis" / "digital_twin" / "listener_digital_twin_summary.json").exists()
    assert (run_dir / "analysis" / "journey_planner" / "journey_plans_summary.json").exists()
    assert (run_dir / "analysis" / "safe_policy" / "safe_bandit_policy_summary.json").exists()
    assert (run_dir / "analysis" / "group_auto_dj" / "group_auto_dj_summary.json").exists()
    assert (run_dir / "analysis" / "stress_test" / "stress_test_summary.json").exists()

    payload = json.loads(moonshot_summary.read_text(encoding="utf-8"))
    assert payload["multimodal_retrieval_fusion_enabled"] is True
    assert payload["multimodal_embedding_dim"] >= 2
    assert payload["journey_seed_count"] >= 1
    assert payload["safe_policy_bucket_count"] >= 1
    assert payload["group_auto_dj_scenario_count"] == 4
    assert 0.0 <= payload["group_auto_dj_mean_safe_route_rate"] <= 1.0
    assert payload["group_auto_dj_mean_fairness"] > 0.0
    assert payload["stress_scenario_count"] == 5


def test_run_report_lists_nested_moonshot_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    history_csv = history_dir / "experiment_history.csv"
    history_csv.write_text("run_id,model_name,val_top1\n", encoding="utf-8")

    (run_dir / "analysis" / "multimodal").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "group_auto_dj").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "moonshot_summary.json").write_text(json.dumps({"journey_seed_count": 2}), encoding="utf-8")
    (run_dir / "analysis" / "multimodal" / "multimodal_artist_space_summary.json").write_text(
        json.dumps({"embedding_dim": 8}),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "group_auto_dj" / "group_auto_dj_summary.json").write_text(
        json.dumps([{"scenario": "party", "safe_route_rate": 0.5}]),
        encoding="utf-8",
    )

    report_path = write_run_report(
        run_dir=run_dir,
        history_dir=history_dir,
        manifest={"run_id": "run_a", "profile": "full", "data_records": 12},
        results=[{"model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.6, "fit_seconds": 1.0}],
        champion_gate={"status": "pass", "promoted": True, "metric_source": "val_top1", "threshold": 0.01, "regression": 0.0},
        history_csv=history_csv,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "analysis/moonshot_summary.json" in content
    assert "analysis/multimodal/multimodal_artist_space_summary.json" in content
    assert "analysis/group_auto_dj/group_auto_dj_summary.json" in content
