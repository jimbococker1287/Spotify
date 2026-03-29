from __future__ import annotations

import subprocess
import sys


def test_branch_portfolio_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.branch_portfolio", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "branch" in result.stdout.lower()

