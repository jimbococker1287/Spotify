#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check champion_gate.json from a run and emit optional alerts on regression.",
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit outputs/runs/<run_id> directory.")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs root when resolving latest run.")
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Optional webhook URL. Falls back to SPOTIFY_ALERT_WEBHOOK_URL.",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Always return 0 even when champion gate fails.",
    )
    parser.add_argument(
        "--notify-macos",
        action="store_true",
        help="Send macOS Notification Center alert via osascript.",
    )
    return parser.parse_args()


def _find_latest_run(outputs_dir: Path) -> Path:
    runs_dir = outputs_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Run directory root not found: {runs_dir}")
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _post_webhook(url: str, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10):  # noqa: S310
        pass


def _notify_macos(title: str, message: str) -> None:
    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "{message}" with title "{title}"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _find_latest_run(outputs_dir)

    gate_path = run_dir / "champion_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"champion_gate.json not found: {gate_path}")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    promoted = bool(gate.get("promoted", False))
    status = str(gate.get("status", ""))
    metric_source = str(gate.get("metric_source", "val_top1"))
    regression = gate.get("regression", "")
    threshold = gate.get("threshold", "")
    model_name = str(gate.get("challenger_model_name", ""))

    message = (
        f"run={run_dir.name} promoted={promoted} status={status} "
        f"metric={metric_source} regression={regression} threshold={threshold} challenger={model_name}"
    )
    print(message)

    webhook_url = args.webhook_url or os.getenv("SPOTIFY_ALERT_WEBHOOK_URL", "").strip()
    if webhook_url:
        payload = {
            "text": message,
            "run_id": run_dir.name,
            "promoted": promoted,
            "status": status,
            "metric_source": metric_source,
            "regression": regression,
            "threshold": threshold,
            "challenger_model_name": model_name,
        }
        try:
            _post_webhook(webhook_url, payload)
            print("alert=webhook_sent")
        except Exception as exc:
            print(f"alert=webhook_failed error={exc}")

    if args.notify_macos or os.getenv("SPOTIFY_ALERT_NOTIFY_MACOS", "0").strip().lower() in ("1", "true", "yes", "on"):
        title = "Spotify Champion Gate"
        _notify_macos(title, message)

    if (not promoted or status.startswith("fail")) and not args.allow_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
