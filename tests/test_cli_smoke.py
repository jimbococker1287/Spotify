from __future__ import annotations

import subprocess
import sys


def test_python_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify" in result.stdout
    assert "--batch" in result.stdout
    assert "--profile" in result.stdout
    assert "--classical-only" in result.stdout
    assert "--mlflow" in result.stdout
    assert "--optuna-trials" in result.stdout
    assert "--temporal-backtest" in result.stdout
