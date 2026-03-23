from __future__ import annotations

import subprocess
import sys


def test_compare_public_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.compare_public", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.compare_public" in result.stdout
    assert "--lookback-days" in result.stdout
    assert "--scope" in result.stdout
