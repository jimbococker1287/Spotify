from __future__ import annotations

import subprocess
import sys


def test_claim_to_demo_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.claim_to_demo", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "claim-to-demo" in result.stdout.lower()

