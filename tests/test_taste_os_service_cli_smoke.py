from __future__ import annotations

import subprocess
import sys


def test_taste_os_service_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.taste_os_service", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.taste_os_service" in result.stdout
    assert "--run-dir" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout
