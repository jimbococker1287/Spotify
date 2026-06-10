from __future__ import annotations

import json
from pathlib import Path
import sys
import types

from spotify.champion_alias import resolve_prediction_run_dir, write_champion_alias
from spotify.deployment_registry import publish_deployment_release
from spotify.run_artifacts import safe_read_json


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


def _write_run_artifacts(run_dir: Path, *, promoted: bool) -> None:
    _write_project_deploy_templates(run_dir.parents[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["A", "B", "C"], "sequence_length": 2}),
        encoding="utf-8",
    )
    (run_dir / "context_scaler.joblib").write_bytes(b"scaler")
    (run_dir / "analysis" / "serving").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "serving" / "prediction_input_context_audio.joblib").write_bytes(b"bundle")
    (run_dir / "analysis" / "serving" / "prediction_input_context_audio.manifest.json").write_text(
        json.dumps({"signature": "abc"}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "champion_alias": {
                    "model_name": "retrieval_reranker",
                    "model_type": "retrieval_reranker",
                },
                "champion_gate": {"promoted": promoted, "status": "pass" if promoted else "fail"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "champion_gate.json").write_text(
        json.dumps({"promoted": promoted, "status": "pass" if promoted else "fail"}),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "retrieval_reranker",
                    "model_type": "retrieval_reranker",
                    "val_top1": 0.44,
                    "retrieval_artifact_path": str(run_dir / "analysis" / "serving" / "prediction_input_context_audio.joblib"),
                }
            ]
        ),
        encoding="utf-8",
    )


def test_publish_deployment_release_creates_channel_alias_and_manifest(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_demo_run"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_run_artifacts(run_dir, promoted=True)

    payload = publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
        artifact_base_uri="s3://spotify-demo-registry/releases",
    )

    assert payload["release_id"] == run_dir.name
    assert payload["promoted"] is True

    channel_dir = registry_root / "channels" / "stable"
    release_dir = registry_root / "releases" / run_dir.name
    assert (channel_dir / "alias.json").exists()
    assert (channel_dir / "deployment_channel.json").exists()
    assert (release_dir / "deployment_release.json").exists()
    assert payload["release_readiness_status"] == "pass"

    resolved_run_dir, alias_model_name = resolve_prediction_run_dir(str(channel_dir))
    assert resolved_run_dir == run_dir.resolve()
    assert alias_model_name == "retrieval_reranker"

    release_manifest = safe_read_json(release_dir / "deployment_release.json", default={})
    assert release_manifest["available_serving_bundles"][0]["artifact_uri"].startswith("s3://spotify-demo-registry/releases")
    assert release_manifest["release_readiness"]["pre_activation"]["status"] == "pass"
    assert release_manifest["release_readiness"]["post_activation"]["status"] == "pass"


def test_publish_deployment_release_records_previous_release_for_rollback(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    registry_root = outputs_dir / "deployments" / "registry"
    first_run_dir = outputs_dir / "runs" / "20260502_first"
    second_run_dir = outputs_dir / "runs" / "20260502_second"
    _write_run_artifacts(first_run_dir, promoted=True)
    _write_run_artifacts(second_run_dir, promoted=True)

    first = publish_deployment_release(
        run_dir=first_run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
    )
    second = publish_deployment_release(
        run_dir=second_run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
    )

    channel_manifest = safe_read_json(registry_root / "channels" / "stable" / "deployment_channel.json", default={})
    assert first["release_id"] == "20260502_first"
    assert second["release_id"] == "20260502_second"
    assert channel_manifest["current_release_id"] == "20260502_second"
    assert channel_manifest["rollback_release_id"] == "20260502_first"


def test_publish_deployment_release_blocks_channel_activation_when_champion_alias_drifts(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    registry_root = outputs_dir / "deployments" / "registry"
    first_run_dir = outputs_dir / "runs" / "20260502_first"
    second_run_dir = outputs_dir / "runs" / "20260502_second"
    _write_run_artifacts(first_run_dir, promoted=True)
    _write_run_artifacts(second_run_dir, promoted=True)
    write_champion_alias(
        output_dir=outputs_dir,
        run_id=first_run_dir.name,
        run_dir=first_run_dir,
        model_name="retrieval_reranker",
        model_type="retrieval_reranker",
    )
    publish_deployment_release(
        run_dir=first_run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
    )

    try:
        publish_deployment_release(
            run_dir=second_run_dir,
            outputs_dir=outputs_dir,
            registry_root=registry_root,
            channel="stable",
        )
    except ValueError as exc:
        assert "pre-activation" in str(exc)
        assert "Global champion alias resolves to a different run" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected release readiness to block champion alias drift.")

    channel_manifest = safe_read_json(registry_root / "channels" / "stable" / "deployment_channel.json", default={})
    assert channel_manifest["current_release_id"] == first_run_dir.name


def test_publish_deployment_release_rejects_unpromoted_runs_by_default(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_unpromoted"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_run_artifacts(run_dir, promoted=False)

    try:
        publish_deployment_release(
            run_dir=run_dir,
            outputs_dir=outputs_dir,
            registry_root=registry_root,
            channel="canary",
        )
    except ValueError as exc:
        assert "not promoted" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected publish_deployment_release to reject unpromoted runs.")


def test_publish_deployment_release_can_publish_artifacts_to_file_uri(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_file_publish"
    registry_root = outputs_dir / "deployments" / "registry"
    remote_root = tmp_path / "remote-artifacts"
    _write_run_artifacts(run_dir, promoted=True)

    payload = publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
        artifact_base_uri=remote_root.resolve().as_uri(),
        publish_artifacts=True,
    )

    assert payload["published_artifact_count"] > 0
    release_manifest = safe_read_json(registry_root / "releases" / run_dir.name / "deployment_release.json", default={})
    assert release_manifest["artifact_publish_enabled"] is True
    assert release_manifest["published_artifact_count"] == payload["published_artifact_count"]
    artifact_entry = release_manifest["artifacts"][0]
    assert artifact_entry["artifact_published"] is True
    assert str(artifact_entry["artifact_uri"]).startswith("file://")

    expected_remote = remote_root / run_dir.name / "run_manifest.json"
    assert expected_remote.exists()


def test_publish_deployment_release_can_publish_artifacts_to_s3(tmp_path: Path, monkeypatch) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_s3_publish"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_run_artifacts(run_dir, promoted=True)

    uploads: list[tuple[str, str, str]] = []

    class _FakeS3Client:
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            uploads.append((filename, bucket, key))

    fake_boto3 = types.SimpleNamespace(client=lambda service_name: _FakeS3Client() if service_name == "s3" else None)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    payload = publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
        artifact_base_uri="s3://spotify-demo-registry/releases",
        publish_artifacts=True,
    )

    assert payload["published_artifact_count"] > 0
    assert uploads
    assert uploads[0][1] == "spotify-demo-registry"
    assert uploads[0][2].startswith(f"releases/{run_dir.name}/")


def test_publish_deployment_release_can_publish_artifacts_to_gcs(tmp_path: Path, monkeypatch) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_gcs_publish"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_run_artifacts(run_dir, promoted=True)

    uploads: list[tuple[str, str]] = []

    class _FakeBlob:
        def __init__(self, blob_name: str) -> None:
            self.blob_name = blob_name

        def upload_from_filename(self, filename: str) -> None:
            uploads.append((self.blob_name, filename))

    class _FakeBucket:
        def __init__(self, bucket_name: str) -> None:
            self.bucket_name = bucket_name

        def blob(self, blob_name: str) -> _FakeBlob:
            return _FakeBlob(blob_name)

    class _FakeStorageClient:
        def bucket(self, bucket_name: str) -> _FakeBucket:
            return _FakeBucket(bucket_name)

    fake_google = types.ModuleType("google")
    fake_google_cloud = types.ModuleType("google.cloud")
    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = lambda: _FakeStorageClient()
    fake_google_cloud.storage = fake_storage
    fake_google.cloud = fake_google_cloud  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    payload = publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
        artifact_base_uri="gs://spotify-demo-registry/releases",
        publish_artifacts=True,
    )

    assert payload["published_artifact_count"] > 0
    assert uploads
    assert uploads[0][0].startswith(f"releases/{run_dir.name}/")
