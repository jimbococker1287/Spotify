from __future__ import annotations

import subprocess
import sys


def test_predict_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.predict_next", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.predict_next" in result.stdout
    assert "--run-dir" in result.stdout
    assert "--model-name" in result.stdout
    assert "--top-k" in result.stdout
    assert "--spotify-public-metadata" in result.stdout
