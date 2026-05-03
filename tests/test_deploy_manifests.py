from __future__ import annotations

from pathlib import Path


def test_kubernetes_deploy_manifests_exist_and_reference_outputs_volume() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    k8s_root = repo_root / "deploy" / "kubernetes"

    expected_files = [
        "README.md",
        "predict-configmap.example.yaml",
        "predict-secret.example.yaml",
        "taste-os-configmap.example.yaml",
        "taste-os-secret.example.yaml",
        "outputs-pvc.yaml",
        "predict-deployment.yaml",
        "predict-service.yaml",
        "taste-os-deployment.yaml",
        "taste-os-service.yaml",
    ]
    for filename in expected_files:
        assert (k8s_root / filename).exists(), filename

    predict_deployment = (k8s_root / "predict-deployment.yaml").read_text(encoding="utf-8")
    taste_os_deployment = (k8s_root / "taste-os-deployment.yaml").read_text(encoding="utf-8")
    assert "/app/outputs" in predict_deployment
    assert "spotify-outputs-pvc" in predict_deployment
    assert "/app/outputs" in taste_os_deployment
    assert "spotify-outputs-pvc" in taste_os_deployment


def test_ecs_task_definitions_exist_and_reference_registry_channel() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ecs_root = repo_root / "deploy" / "ecs"

    expected_files = [
        "README.md",
        "predict-task-definition.json",
        "taste-os-task-definition.json",
    ]
    for filename in expected_files:
        assert (ecs_root / filename).exists(), filename

    predict_task = (ecs_root / "predict-task-definition.json").read_text(encoding="utf-8")
    taste_os_task = (ecs_root / "taste-os-task-definition.json").read_text(encoding="utf-8")
    assert "/app/outputs/deployments/registry/channels/stable" in predict_task
    assert "\"SERVICE_MODE\", \"value\": \"predict\"" in predict_task
    assert "/app/outputs/deployments/registry/channels/stable" in taste_os_task
    assert "\"SERVICE_MODE\", \"value\": \"taste-os\"" in taste_os_task
