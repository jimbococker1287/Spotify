from __future__ import annotations

import subprocess
import sys


def test_control_room_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.control_room", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "control room" in result.stdout.lower()
    assert "--output-dir" in result.stdout
    assert "--top-n" in result.stdout
