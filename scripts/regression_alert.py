#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import urllib.request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check champion gate plus control-room review actions from a run and emit optional ops alerts.",
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
    parser.add_argument(
        "--review-threshold",
        type=str,
        default=os.getenv("SPOTIFY_ALERT_REVIEW_THRESHOLD", "high"),
        help="Fail on control-room review actions at or above this priority: off|high|medium|low.",
    )
    parser.add_argument(
        "--max-review-actions",
        type=int,
        default=int(os.getenv("SPOTIFY_ALERT_MAX_REVIEW_ACTIONS", "3")),
        help="Maximum number of review actions to include in stdout and webhook payloads.",
    )
    return parser.parse_args()


def _find_latest_run(outputs_dir: Path) -> Path:
    runs_dir = outputs_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Run directory root not found: {runs_dir}")
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from spotify.run_artifacts import collect_run_manifests

        manifests = collect_run_manifests(outputs_dir)
    except Exception:
        manifests = []

    ranked_manifest_runs: list[tuple[int, str, Path]] = []
    for manifest in manifests:
        run_id = str(manifest.get("run_id", "")).strip()
        if not run_id:
            continue
        run_dir = runs_dir / run_id
        timestamp = str(manifest.get("timestamp", "")).strip()
        ranked_manifest_runs.append(
            (
                int((run_dir / "champion_gate.json").exists()),
                timestamp,
                run_dir,
            )
        )
    if ranked_manifest_runs:
        ranked_manifest_runs.sort(reverse=True)
        return ranked_manifest_runs[0][2]

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    run_dirs.sort(
        key=lambda path: (
            int((path / "champion_gate.json").exists()),
            path.stat().st_mtime,
        ),
        reverse=True,
    )
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


def _read_json(path: Path) -> dict[str, object]:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from spotify.run_artifacts import safe_read_json

        payload = safe_read_json(path, default={})
    except Exception:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return payload if isinstance(payload, dict) else {}


def _priority_value(priority: str) -> int:
    normalized = str(priority).strip().lower()
    if normalized == "high":
        return 3
    if normalized == "medium":
        return 2
    if normalized == "low":
        return 1
    return 0


def _normalize_review_threshold(raw_value: str) -> str:
    normalized = str(raw_value).strip().lower()
    if normalized in {"off", "high", "medium", "low"}:
        return normalized
    return "high"


def _refresh_control_room(outputs_dir: Path) -> Path | None:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from spotify.control_room import write_control_room_report

        json_path, _ = write_control_room_report(outputs_dir)
        return json_path
    except Exception:
        return None


def _load_control_room_payload(outputs_dir: Path) -> dict[str, object]:
    json_path = _refresh_control_room(outputs_dir)
    candidate_path = json_path if json_path is not None else outputs_dir / "analytics" / "control_room.json"
    if not candidate_path.exists():
        return {}
    return _read_json(candidate_path)


def _resolve_run_dir_from_control_room(outputs_dir: Path, control_room: dict[str, object]) -> Path | None:
    latest_run = control_room.get("latest_run", {})
    if not isinstance(latest_run, dict):
        return None
    run_id = str(latest_run.get("run_id", "")).strip()
    if not run_id:
        return None
    run_dir = outputs_dir / "runs" / run_id
    return run_dir if run_dir.exists() else None


def _resolve_default_run_dir(outputs_dir: Path, control_room: dict[str, object]) -> Path:
    selected = _resolve_run_dir_from_control_room(outputs_dir, control_room)
    if selected is not None and (selected / "champion_gate.json").exists():
        return selected
    return _find_latest_run(outputs_dir)


def _review_actions_for_run(
    *,
    control_room: dict[str, object],
    run_id: str,
    max_actions: int,
) -> tuple[list[dict[str, object]], str]:
    latest_run = control_room.get("latest_run", {})
    if not isinstance(latest_run, dict):
        return [], "missing"
    latest_run_id = str(latest_run.get("run_id", "")).strip()
    if not latest_run_id:
        return [], "missing"
    if latest_run_id != str(run_id).strip():
        return [], f"stale:{latest_run_id}"

    actions_raw = control_room.get("review_actions", [])
    if not isinstance(actions_raw, list):
        return [], "missing"
    actions: list[dict[str, object]] = []
    for item in actions_raw:
        if not isinstance(item, dict):
            continue
        actions.append(dict(item))
        if len(actions) >= max(0, int(max_actions)):
            break
    return actions, "ok"


def _highest_review_priority(actions: list[dict[str, object]]) -> str:
    best = ""
    best_value = 0
    for action in actions:
        value = _priority_value(str(action.get("priority", "")))
        if value > best_value:
            best_value = value
            best = str(action.get("priority", "")).strip().lower()
    return best or "none"


def _review_threshold_triggered(*, actions: list[dict[str, object]], threshold: str) -> bool:
    normalized = _normalize_review_threshold(threshold)
    if normalized == "off":
        return False
    return _priority_value(_highest_review_priority(actions)) >= _priority_value(normalized)


def _review_action_summary(action: dict[str, object]) -> str:
    priority = str(action.get("priority", "")).strip().upper() or "UNKNOWN"
    title = str(action.get("title", "")).strip() or "Untitled review action"
    return f"[{priority}] {title}"


def _parse_timestamp(raw_value: object) -> datetime | None:
    value = str(raw_value).strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _run_timestamp(run_dir: Path) -> datetime | None:
    manifest = _read_json(run_dir / "run_manifest.json")
    parsed = _parse_timestamp(manifest.get("timestamp"))
    if parsed is not None:
        return parsed
    try:
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _requested_run_pending_control_room(control_room: dict[str, object], run_dir: Path) -> bool:
    latest_run = control_room.get("latest_run", {})
    if not isinstance(latest_run, dict):
        return False
    latest_run_id = str(latest_run.get("run_id", "")).strip()
    if not latest_run_id or latest_run_id == run_dir.name:
        return False
    latest_timestamp = _parse_timestamp(latest_run.get("timestamp"))
    requested_timestamp = _run_timestamp(run_dir)
    if latest_timestamp is None or requested_timestamp is None:
        return False
    return requested_timestamp >= latest_timestamp


def main() -> int:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    control_room = _load_control_room_payload(outputs_dir)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = _resolve_default_run_dir(outputs_dir, control_room)

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
    review_threshold = _normalize_review_threshold(args.review_threshold)
    if args.run_dir and _requested_run_pending_control_room(control_room, run_dir):
        latest_run = control_room.get("latest_run", {})
        latest_run = latest_run if isinstance(latest_run, dict) else {}
        review_actions = []
        control_room_status = f"pending_control_room:{str(latest_run.get('run_id', '')).strip() or 'none'}"
    else:
        review_actions, control_room_status = _review_actions_for_run(
            control_room=control_room,
            run_id=run_dir.name,
            max_actions=max(0, int(args.max_review_actions)),
        )
    highest_review_priority = _highest_review_priority(review_actions)
    async_handoff = control_room.get("async_handoff", {})
    async_handoff = async_handoff if isinstance(async_handoff, dict) else {}
    handoff_status = str(async_handoff.get("status", "missing")).strip() or "missing"
    next_step = (
        str(async_handoff.get("recommended_run_command", "")).strip()
        or str(async_handoff.get("recommended_review_command", "")).strip()
        or "make control-room"
    )

    message = (
        f"run={run_dir.name} promoted={promoted} status={status} "
        f"metric={metric_source} regression={regression} threshold={threshold} challenger={model_name} "
        f"review_status={control_room_status} review_priority={highest_review_priority} review_count={len(review_actions)} "
        f"handoff={handoff_status} next_step={next_step}"
    )
    print(message)
    for idx, action in enumerate(review_actions, start=1):
        print(f"review_action[{idx}]={_review_action_summary(action)}")

    webhook_url = args.webhook_url or os.getenv("SPOTIFY_ALERT_WEBHOOK_URL", "").strip()
    if webhook_url:
        baseline_summary = control_room.get("baseline_comparison", {})
        baseline_summary = baseline_summary if isinstance(baseline_summary, dict) else {}
        payload = {
            "text": message,
            "run_id": run_dir.name,
            "promoted": promoted,
            "status": status,
            "metric_source": metric_source,
            "regression": regression,
            "threshold": threshold,
            "challenger_model_name": model_name,
            "review_status": control_room_status,
            "review_threshold": review_threshold,
            "review_priority": highest_review_priority,
            "review_actions": review_actions,
            "baseline_summary": baseline_summary.get("summary", []),
            "async_handoff": async_handoff,
            "next_step": next_step,
        }
        try:
            _post_webhook(webhook_url, payload)
            print("alert=webhook_sent")
        except Exception as exc:
            print(f"alert=webhook_failed error={exc}")

    if args.notify_macos or os.getenv("SPOTIFY_ALERT_NOTIFY_MACOS", "0").strip().lower() in ("1", "true", "yes", "on"):
        title = "Spotify Champion Gate"
        if review_actions:
            _notify_macos(title, f"{message} { _review_action_summary(review_actions[0]) }")
        else:
            _notify_macos(title, message)

    if (not promoted or status.startswith("fail")) and not args.allow_fail:
        return 2
    if _review_threshold_triggered(actions=review_actions, threshold=review_threshold) and not args.allow_fail:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
