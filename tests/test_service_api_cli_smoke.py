from __future__ import annotations

import subprocess
import sys


def test_service_api_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.service_api", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.service_api" in result.stdout
    assert "--app" in result.stdout
    assert "--request-rate-limit" in result.stdout
    assert "--rate-limit-backend" in result.stdout
    assert "--state-db-url" in result.stdout
    assert "--auth-mode" in result.stdout
    assert "--jwks-url" in result.stdout
    assert "--require-deployment-registry" in result.stdout


def test_serving_bundle_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.serving_bundle", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.serving_bundle" in result.stdout
    assert "--data-dir" in result.stdout
    assert "--all-contexts" in result.stdout


def test_deployment_registry_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.deployment_registry", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.deployment_registry" in result.stdout
    assert "--registry-root" in result.stdout
    assert "--channel" in result.stdout
    assert "--publish-artifacts" in result.stdout
    assert "--skip-release-readiness" in result.stdout
