from __future__ import annotations

import subprocess
import sys


def test_research_claims_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.research_claims", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "research-claim pack" in result.stdout.lower()
    assert "--benchmark-manifest" in result.stdout
    assert "--run-dir" in result.stdout
