from __future__ import annotations

import subprocess
import sys


def test_outward_package_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.outward_package", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "outward" in result.stdout.lower()

