from __future__ import annotations

import subprocess
import sys


def test_taste_os_showcase_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.taste_os_showcase", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "showcase" in result.stdout.lower()
    assert "--output-dir" in result.stdout
    assert "--recent-artists" in result.stdout
