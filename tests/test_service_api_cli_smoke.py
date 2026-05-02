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
