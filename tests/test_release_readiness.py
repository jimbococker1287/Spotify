from __future__ import annotations

import json
from pathlib import Path

from spotify.champion_alias import write_champion_alias
from spotify.deployment_registry import publish_deployment_release
from spotify.release_readiness import build_release_readiness_smoke, main
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


def _write_run_artifacts(run_dir: Path, *, promoted: bool = True, serving_manifest: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["A", "B", "C"], "sequence_length": 2}),
        encoding="utf-8",
    )
    (run_dir / "context_scaler.joblib").write_bytes(b"scaler")
    serving_dir = run_dir / "analysis" / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = serving_dir / "prediction_input_context_audio.joblib"
    bundle_path.write_bytes(b"bundle")
    if serving_manifest:
        (serving_dir / "prediction_input_context_audio.manifest.json").write_text(
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
                    "retrieval_artifact_path": str(bundle_path),
                }
            ]
        ),
        encoding="utf-8",
    )


def test_release_readiness_passes_for_published_registry_channel(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    outputs_dir = project_root / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_demo_run"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_project_deploy_templates(project_root)
    _write_run_artifacts(run_dir)
    write_champion_alias(
        output_dir=outputs_dir,
        run_id=run_dir.name,
        run_dir=run_dir,
        model_name="retrieval_reranker",
        model_type="retrieval_reranker",
    )
    publish_deployment_release(
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
    )

    payload = build_release_readiness_smoke(
        project_root=project_root,
        outputs_dir=outputs_dir,
        run_dir=str(registry_root / "channels" / "stable"),
        registry_root=registry_root,
        channel="stable",
        require_registry=True,
    )

    assert payload["summary"]["status"] == "pass"  # type: ignore[index]
    assert payload["summary"]["release_ready"] is True  # type: ignore[index]
    assert payload["available_serving_bundle_count"] == 1
    assert (outputs_dir / "analysis" / "release_readiness" / "release_readiness_smoke.json").exists()
    report = (outputs_dir / "analysis" / "release_readiness" / "release_readiness_smoke.md").read_text(encoding="utf-8")
    assert "Release Readiness Smoke" in report


def test_release_readiness_fails_when_registry_channel_points_to_different_run(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    outputs_dir = project_root / "outputs"
    first_run = outputs_dir / "runs" / "20260502_first"
    second_run = outputs_dir / "runs" / "20260502_second"
    registry_root = outputs_dir / "deployments" / "registry"
    _write_project_deploy_templates(project_root)
    _write_run_artifacts(first_run)
    _write_run_artifacts(second_run)
    publish_deployment_release(
        run_dir=first_run,
        outputs_dir=outputs_dir,
        registry_root=registry_root,
        channel="stable",
    )

    payload = build_release_readiness_smoke(
        project_root=project_root,
        outputs_dir=outputs_dir,
        run_dir=str(second_run),
        registry_root=registry_root,
        channel="stable",
        require_registry=True,
    )

    checks = {str(row["check_key"]): row for row in payload["checks"]}  # type: ignore[index]
    assert payload["summary"]["status"] == "fail"  # type: ignore[index]
    assert checks["registry_release_matches_resolved_run"]["status"] == "fail"
    assert checks["registry_channel_alias_matches_run"]["status"] == "fail"


def test_release_readiness_cli_writes_report_and_honors_strict(tmp_path: Path, capsys) -> None:
    project_root = tmp_path / "repo"
    outputs_dir = project_root / "outputs"
    run_dir = outputs_dir / "runs" / "20260502_missing_bundle_manifest"
    _write_project_deploy_templates(project_root)
    _write_run_artifacts(run_dir, serving_manifest=False)

    result = main(
        [
            "--project-root",
            str(project_root),
            "--outputs-dir",
            str(outputs_dir),
            "--run-dir",
            str(run_dir),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    payload = safe_read_json(outputs_dir / "analysis" / "release_readiness" / "release_readiness_smoke.json", default={})
    assert result == 1
    assert "release_readiness_status=fail" in captured.out
    assert payload["summary"]["fail_count"] >= 1
