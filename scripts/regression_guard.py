#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression guard: run a bounded smoke pipeline and assert key artifacts/metrics exist.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Command timeout.")
    parser.add_argument("--profile", type=str, default="dev", help="Pipeline profile for the guard run.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs override.")
    parser.add_argument("--deep-models", type=str, default="dense", help="Deep model subset.")
    parser.add_argument("--classical-models", type=str, default="logreg", help="Classical model subset.")
    return parser.parse_args()


def _extract_run_dir(output_text: str) -> Path:
    match = re.search(r"Run output directory:\s*(.+)", output_text)
    if not match:
        raise RuntimeError("Could not parse run output directory from pipeline logs.")
    return Path(match.group(1).strip()).expanduser().resolve()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main() -> int:
    args = _parse_args()
    run_name = f"regression-guard-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    cmd = [
        sys.executable,
        "-m",
        "spotify",
        "--profile",
        args.profile,
        "--run-name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--models",
        args.deep_models,
        "--classical-models",
        args.classical_models,
        "--no-optuna",
        "--no-temporal-backtest",
        "--no-shap",
        "--no-mlflow",
    ]
    env = os.environ.copy()
    env.setdefault("SPOTIFY_FORCE_CPU", "1")
    env.setdefault("SPOTIFY_RUN_EAGER", "0")
    env.setdefault("SPOTIFY_DISABLE_MONITOR", "1")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Regression guard timed out after {args.timeout_seconds}s. "
            "This indicates a potential hang."
        ) from exc

    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            "Regression guard pipeline failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {result.returncode}\n"
            f"Output:\n{combined_output}"
        )

    run_dir = _extract_run_dir(combined_output)
    _require_file(run_dir / "train.log")
    _require_file(run_dir / "run_manifest.json")
    _require_file(run_dir / "run_results.json")
    _require_file(run_dir / "run_leaderboard.png")

    log_text = (run_dir / "train.log").read_text(encoding="utf-8")
    required_snippets = [
        "[dense] Epoch progress: step=1",
        "[TEST]",
        "Pipeline completed successfully",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in log_text]
    if missing:
        raise RuntimeError(f"Regression guard log validation failed. Missing snippets: {missing}")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("data_records", 0)) <= 0:
        raise RuntimeError("Regression guard manifest has non-positive data_records.")

    run_results = json.loads((run_dir / "run_results.json").read_text(encoding="utf-8"))
    if not run_results:
        raise RuntimeError("Regression guard run_results.json is empty.")
    if not any(str(row.get("model_type")) == "deep" for row in run_results):
        raise RuntimeError("Regression guard expected at least one deep model result.")
    if not any(str(row.get("model_type")) == "classical" for row in run_results):
        raise RuntimeError("Regression guard expected at least one classical model result.")

    print(f"Regression guard passed. run_name={run_name}")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
