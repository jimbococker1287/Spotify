from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

import spotify.predict_service as predict_service
import spotify.taste_os_service as taste_os_service
from spotify.deployment_registry import publish_deployment_release
from spotify.digital_twin import ListenerDigitalTwinArtifact
from spotify.multimodal import MultimodalArtistSpace
from spotify.predict_next import PredictionInputContext
from spotify.production_smoke import HISTORY_COLUMNS, build_production_smoke, main
from spotify.run_artifacts import safe_read_json, write_csv_rows
from spotify.safe_policy import SafeBanditPolicyArtifact


class _PredictStub:
    model_name = "retrieval_reranker"
    model_type = "retrieval_reranker"

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        _ = (seq_batch, ctx_batch)
        return np.asarray([[0.10, 0.18, 0.52, 0.20]], dtype="float32")


class _StubEndEstimator:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        _ = features
        return np.asarray([[0.72, 0.28]], dtype="float32")


def _prediction_context() -> PredictionInputContext:
    return PredictionInputContext(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        artist_to_label={"Artist A": 0, "Artist B": 1, "Artist C": 2, "Artist D": 3},
        sequence_length=2,
        latest_sequence_labels=np.array([0, 1], dtype="int32"),
        latest_sequence_names=["Artist A", "Artist B"],
        context_scaled=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
        context_raw=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
        context_features=["hour", "tech_playback_errors_24h", "offline"],
    )


def _space() -> MultimodalArtistSpace:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype="float32",
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return MultimodalArtistSpace(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        feature_names=["f0", "f1"],
        raw_features=np.zeros((4, 2), dtype="float32"),
        embeddings=embeddings.astype("float32"),
        popularity=np.asarray([0.40, 0.32, 0.18, 0.10], dtype="float32"),
        energy=np.asarray([0.45, 0.55, 0.76, 0.88], dtype="float32"),
        danceability=np.asarray([0.40, 0.50, 0.72, 0.82], dtype="float32"),
        tempo=np.asarray([102.0, 110.0, 128.0, 134.0], dtype="float32"),
    )


def _twin() -> ListenerDigitalTwinArtifact:
    return ListenerDigitalTwinArtifact(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        transition_matrix=np.asarray(
            [
                [0.10, 0.45, 0.35, 0.10],
                [0.08, 0.12, 0.60, 0.20],
                [0.05, 0.15, 0.20, 0.60],
                [0.12, 0.18, 0.46, 0.24],
            ],
            dtype="float32",
        ),
        end_estimator=_StubEndEstimator(),
        context_features=["hour", "tech_playback_errors_24h", "offline"],
        average_track_seconds=180.0,
    )


def _safe_policy() -> SafeBanditPolicyArtifact:
    return SafeBanditPolicyArtifact(
        policy_map={
            "high_friction": {"transition": 0.7, "continuity": 0.4, "novelty": 0.1, "repeat": 0.9},
            "normal_friction": {"transition": 0.9, "continuity": 0.3, "novelty": 0.2, "repeat": 0.7},
        },
        global_policy={"transition": 0.8, "continuity": 0.3, "novelty": 0.2, "repeat": 0.8},
        reward_metric="reward",
    )


def _write_project_deploy_templates(project_root: Path) -> None:
    k8s_root = project_root / "deploy" / "kubernetes"
    ecs_root = project_root / "deploy" / "ecs"
    k8s_root.mkdir(parents=True, exist_ok=True)
    ecs_root.mkdir(parents=True, exist_ok=True)
    for name in ("predict-deployment.yaml", "taste-os-deployment.yaml"):
        (k8s_root / name).write_text("readinessProbe:\n  httpGet:\n    path: /readyz\n", encoding="utf-8")
    for name in ("predict-task-definition.json", "taste-os-task-definition.json"):
        (ecs_root / name).write_text(
            json.dumps({"mount": "/app/outputs/deployments/registry/channels/stable"}),
            encoding="utf-8",
        )


def _write_run_artifacts(run_dir: Path) -> None:
    _write_project_deploy_templates(run_dir.parents[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["Artist A", "Artist B", "Artist C", "Artist D"], "sequence_length": 2}),
        encoding="utf-8",
    )
    (run_dir / "context_scaler.joblib").write_bytes(b"scaler")
    serving_dir = run_dir / "analysis" / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = serving_dir / "prediction_input_context_audio.joblib"
    bundle_path.write_bytes(b"bundle")
    (serving_dir / "prediction_input_context_audio.manifest.json").write_text(
        json.dumps({"signature": "abc"}),
        encoding="utf-8",
    )
    multimodal_dir = run_dir / "analysis" / "multimodal"
    digital_twin_dir = run_dir / "analysis" / "digital_twin"
    safe_policy_dir = run_dir / "analysis" / "safe_policy"
    multimodal_dir.mkdir(parents=True, exist_ok=True)
    digital_twin_dir.mkdir(parents=True, exist_ok=True)
    safe_policy_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(_space(), multimodal_dir / "multimodal_artist_space.joblib")
    joblib.dump(_twin(), digital_twin_dir / "listener_digital_twin.joblib")
    joblib.dump(_safe_policy(), safe_policy_dir / "safe_bandit_policy.joblib")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "champion_alias": {
                    "model_name": "retrieval_reranker",
                    "model_type": "retrieval_reranker",
                },
                "champion_gate": {"promoted": True, "status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "champion_gate.json").write_text(json.dumps({"promoted": True, "status": "pass"}), encoding="utf-8")
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "retrieval_reranker",
                    "model_type": "retrieval_reranker",
                    "val_top1": 0.44,
                    "retrieval_artifact_path": str(bundle_path),
                }
            ]
        ),
        encoding="utf-8",
    )


def test_production_smoke_exercises_both_asgi_services(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    outputs_dir = project_root / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_demo_run"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_run_artifacts(run_dir)
    write_csv_rows(
        outputs_dir / "history" / "production_smoke_history.csv",
        [
            {
                "generated_at": "2026-05-01T00:00:00+00:00",
                "release_id": "20260501_old_run",
                "run_dir": str(outputs_dir / "runs" / "20260501_old_run"),
                "requested_run_dir": str(registry_root / "channels" / "stable"),
                "model_name": "retrieval_reranker",
                "status": "pass",
                "production_ready": True,
                "check_count": 12,
                "pass_count": 12,
                "warning_count": 0,
                "fail_count": 0,
                "request_count": 6,
                "successful_request_count": 6,
                "max_latency_ms": 25.0,
                "average_latency_ms": 8.0,
                "predict_readyz_status": "pass",
                "predict_metrics_status": "pass",
                "taste_os_readyz_status": "pass",
                "taste_os_metrics_status": "pass",
                "blocker_count": 0,
            }
        ],
        fieldnames=HISTORY_COLUMNS,
    )
    publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
    )

    monkeypatch.setattr(
        predict_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker"},
    )
    monkeypatch.setattr(predict_service, "load_predictor", lambda run_dir, row, artist_labels: _PredictStub())
    monkeypatch.setattr(
        predict_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger, **kwargs: _prediction_context(),
    )
    monkeypatch.setattr(
        taste_os_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker"},
    )
    monkeypatch.setattr(taste_os_service, "load_predictor", lambda run_dir, row, artist_labels: _PredictStub())
    monkeypatch.setattr(
        taste_os_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger, **kwargs: _prediction_context(),
    )

    logger = logging.getLogger("spotify.test.production_smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    payload = build_production_smoke(
        project_root=project_root,
        outputs_dir=outputs_dir,
        run_dir=registry_root / "channels" / "stable",
        top_k=3,
        scenario="skip_recovery",
        require_serving_bundle=True,
        logger=logger,
    )

    assert payload["summary"]["status"] == "pass"  # type: ignore[index]
    assert payload["summary"]["successful_request_count"] == 6  # type: ignore[index]
    endpoints = {str(row["endpoint"]) for row in payload["requests"]}  # type: ignore[index]
    assert endpoints == {"/v1/readyz", "/v1/metrics", "/v1/predict", "/v1/taste-os/session"}
    assert all(str(row["request_id"]) for row in payload["requests"])  # type: ignore[index]
    check_keys = {str(row["check_key"]) for row in payload["checks"]}  # type: ignore[index]
    assert "readyz_deployment_registry_release_model_matches_service" in check_keys
    assert "readyz_deployment_registry_channel_alias_matches_service_run" in check_keys
    assert (outputs_dir / "analysis" / "production_smoke" / "production_smoke.json").exists()
    assert (outputs_dir / "history" / "production_smoke_history.csv").exists()
    trend = safe_read_json(outputs_dir / "analysis" / "production_smoke" / "production_smoke_trend.json", default={})
    assert trend["history_run_count"] == 2
    assert trend["latest"]["release_id"] == "20260502_demo_run"
    assert trend["previous"]["release_id"] == "20260501_old_run"
    assert payload["trend_summary"]["history_run_count"] == 2  # type: ignore[index]
    assert payload["paths"]["history_csv"].endswith("production_smoke_history.csv")  # type: ignore[index]
    trend_report = (outputs_dir / "analysis" / "production_smoke" / "production_smoke_trend.md").read_text(encoding="utf-8")
    assert "Production Smoke Trend" in trend_report
    report = (outputs_dir / "analysis" / "production_smoke" / "production_smoke.md").read_text(encoding="utf-8")
    assert "Production Smoke" in report
    assert "History runs tracked: 2" in report


def test_production_smoke_cli_writes_failure_artifact_for_missing_channel(tmp_path: Path, capsys) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    outputs_dir = project_root / "outputs"

    result = main(
        [
            "--project-root",
            str(project_root),
            "--outputs-dir",
            str(outputs_dir),
            "--run-dir",
            str(outputs_dir / "deployments" / "registry" / "channels" / "stable"),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    payload = safe_read_json(outputs_dir / "analysis" / "production_smoke" / "production_smoke.json", default={})
    assert result == 1
    assert "production_smoke_status=fail" in captured.out
    assert "production_smoke_history=" in captured.out
    assert payload["summary"]["fail_count"] >= 1
