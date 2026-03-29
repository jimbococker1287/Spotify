#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail scheduled runs when control-room metrics exceed explicit ops thresholds.",
    )
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs root.")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit outputs/runs/<run_id> directory.")
    parser.add_argument(
        "--max-robustness-gap",
        type=str,
        default="off",
        help="Maximum allowed worst robustness gap. Use off to disable.",
    )
    parser.add_argument(
        "--max-stress-skip-risk",
        type=str,
        default="off",
        help="Maximum allowed worst stress skip risk. Use off to disable.",
    )
    parser.add_argument(
        "--max-target-drift-jsd",
        type=str,
        default="off",
        help="Maximum allowed target drift JSD. Use off to disable.",
    )
    parser.add_argument(
        "--max-selective-risk",
        type=str,
        default="off",
        help="Maximum allowed selective risk. Use off to disable.",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Always return 0 even when thresholds are exceeded.",
    )
    return parser.parse_args()


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _format_metric(value) -> str:
    metric = _safe_float(value)
    if not math.isfinite(metric):
        return "n/a"
    return f"{metric:.3f}"


def _parse_threshold(raw_value: str) -> float | None:
    normalized = str(raw_value).strip().lower()
    if normalized in {"", "off", "none", "false"}:
        return None
    value = _safe_float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"Invalid threshold value: {raw_value}")
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _refresh_control_room(outputs_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from spotify.control_room import write_control_room_report

    json_path, _ = write_control_room_report(outputs_dir)
    return json_path


def _violation_row(*, key: str, label: str, current: object, maximum: float | None) -> dict[str, object] | None:
    if maximum is None:
        return None
    current_value = _safe_float(current)
    if not math.isfinite(current_value):
        return {
            "key": key,
            "label": label,
            "current": current_value,
            "threshold": maximum,
            "status": "unknown",
            "message": f"{label} is missing, so the guard could not verify it against `{maximum:.3f}`.",
        }
    if current_value <= maximum + 1e-12:
        return None
    return {
        "key": key,
        "label": label,
        "current": current_value,
        "threshold": maximum,
        "status": "fail",
        "message": f"{label} is `{current_value:.3f}` which exceeds `{maximum:.3f}`.",
    }


def _violation_area(key: str) -> str:
    if key == "robustness_gap":
        return "robustness"
    if key == "stress_skip_risk":
        return "stress_test"
    if key == "target_drift_jsd":
        return "drift"
    if key == "selective_risk":
        return "uncertainty"
    return "review"


def _triage_playbook(area: str) -> dict[str, list[str]]:
    normalized = str(area).strip().lower()
    if normalized == "promotion":
        return {
            "inspect": [
                "Compare the challenger against the last promoted baseline in outputs/analytics/control_room.md.",
                "Inspect run_manifest.json and run_results.json for the latest run to confirm which metric blocked promotion.",
            ],
            "fix": [
                "Decide whether the regression is caused by drift, calibration, or a model-family change before retraining.",
                "If the challenger is only better on validation, tighten the candidate set or promotion metric before another full run.",
            ],
            "rerun": [
                "Run a fast scheduled pass after the fix to confirm the challenger path is stable.",
                "Only schedule another full run after the fast pass clears the control-room review.",
            ],
        }
    if normalized == "robustness":
        return {
            "inspect": [
                "Open analysis/robustness_summary.json and isolate the worst segment and bucket.",
                "Compare the latest run to the promoted baseline on that slice before changing global defaults.",
            ],
            "fix": [
                "Add slice-aware safeguards, reranking constraints, or feature checks for the failing segment.",
                "If one model family is causing the gap, narrow the scheduled profile until the slice is stable again.",
            ],
            "rerun": [
                "Re-run the fast schedule and confirm the worst robustness gap drops under the configured threshold.",
                "Re-run the control-room guard before promoting the fix into the next full cadence.",
            ],
        }
    if normalized == "drift":
        return {
            "inspect": [
                "Open analysis/data_drift_summary.json and confirm whether the shift is target drift, context drift, or segment drift.",
                "Check whether the drifted segment matches a recent product, profile, or data-export change.",
            ],
            "fix": [
                "If the shift is real, retrain on fresher data or adjust profile-matching and baseline rules.",
                "If the shift is accidental, fix the data path or feature generation before another benchmark run.",
            ],
            "rerun": [
                "Run a fast schedule and verify target drift JSD moves back under the threshold.",
                "Use the refreshed control room to confirm that regressions are still meaningful after the drift change.",
            ],
        }
    if normalized == "stress_test":
        return {
            "inspect": [
                "Open analysis/moonshot_summary.json and analysis/stress_test/stress_test_summary.json for the worst scenario.",
                "Check whether the safe policy actually routes early enough under that scenario.",
            ],
            "fix": [
                "Tighten safe-policy routing or reduce novelty in the failing stress regime.",
                "Promote the scenario into a regression check if it is a realistic operating risk.",
            ],
            "rerun": [
                "Re-run the guard and confirm worst stress skip risk drops below the threshold.",
                "Only widen the scheduled cadence once the safe-route behavior looks stable again.",
            ],
        }
    if normalized == "uncertainty":
        return {
            "inspect": [
                "Inspect the latest conformal summary under analysis/*_conformal_summary.json.",
                "Check whether confidence is poorly calibrated or abstention is effectively disabled.",
            ],
            "fix": [
                "Adjust calibration, abstention thresholds, or serving defaults before broadening exposure.",
                "If selective risk is concentrated in one slice, fix that slice before raising coverage again.",
            ],
            "rerun": [
                "Run a fast pass and confirm selective risk falls under the configured ceiling.",
                "Refresh the control room and verify abstention or prediction-set behavior is now visible in the report.",
            ],
        }
    if normalized == "instrumentation":
        return {
            "inspect": [
                "Open outputs/analytics/control_room.md and list which expected summaries are missing from the latest run.",
                "Inspect the latest run directory to confirm whether analysis generation was skipped, failed, or never scheduled.",
            ],
            "fix": [
                "Restore the missing analysis step in the scheduled pipeline before trusting control-room thresholds.",
                "If this was a deliberate smoke run, separate it from the main ops cadence so it does not pollute production review.",
            ],
            "rerun": [
                "Re-run the control room after the missing summaries are present.",
                "Only treat threshold output as authoritative once the coverage section shows the expected artifacts.",
            ],
        }
    return {
        "inspect": [
            "Open outputs/analytics/control_room.md and review the latest run against the promoted baseline.",
        ],
        "fix": [
            "Address the blocking review action before scheduling another full run.",
        ],
        "rerun": [
            "Refresh the control room and guard outputs after the fix.",
        ],
    }


def _build_triage_items(
    *,
    control_room: dict[str, object],
    violations: list[dict[str, object]],
) -> list[dict[str, object]]:
    review_actions = control_room.get("review_actions", [])
    review_actions = review_actions if isinstance(review_actions, list) else []
    violation_by_area = {_violation_area(str(item.get("key", ""))): item for item in violations}
    triage_items: list[dict[str, object]] = []
    seen_areas: set[str] = set()

    for action in review_actions:
        if not isinstance(action, dict):
            continue
        area = str(action.get("area", "review")).strip().lower() or "review"
        playbook = _triage_playbook(area)
        triage_items.append(
            {
                "area": area,
                "priority": str(action.get("priority", "")).strip().lower() or "medium",
                "title": str(action.get("title", "")).strip() or "Untitled review action",
                "trigger": str(action.get("detail", "")).strip(),
                "inspect_files": [str(item) for item in action.get("inspect", [])] if isinstance(action.get("inspect", []), list) else [],
                "threshold_violation": violation_by_area.get(area),
                "inspect_steps": playbook["inspect"],
                "fix_steps": playbook["fix"],
                "rerun_steps": playbook["rerun"],
            }
        )
        seen_areas.add(area)

    for violation in violations:
        area = _violation_area(str(violation.get("key", "")))
        if area in seen_areas:
            continue
        playbook = _triage_playbook(area)
        triage_items.append(
            {
                "area": area,
                "priority": "high",
                "title": str(violation.get("label", "")).strip() or "Threshold violation",
                "trigger": str(violation.get("message", "")).strip(),
                "inspect_files": [],
                "threshold_violation": violation,
                "inspect_steps": playbook["inspect"],
                "fix_steps": playbook["fix"],
                "rerun_steps": playbook["rerun"],
            }
        )
        seen_areas.add(area)

    return triage_items


def _write_triage_artifacts(
    *,
    outputs_dir: Path,
    control_room: dict[str, object],
    status: str,
    thresholds: dict[str, float | None],
    violations: list[dict[str, object]],
) -> tuple[Path, Path]:
    analytics_dir = outputs_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    latest_run = control_room.get("latest_run", {})
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    baseline = control_room.get("baseline_comparison", {})
    baseline = baseline if isinstance(baseline, dict) else {}
    triage_items = _build_triage_items(control_room=control_room, violations=violations)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(outputs_dir),
        "control_room_status": status,
        "run_id": str(latest_run.get("run_id", "")),
        "run_profile": str(latest_run.get("profile", "")),
        "promotion_status": str(latest_run.get("promotion_status", "")),
        "thresholds": thresholds,
        "baseline_summary": baseline.get("summary", []) if isinstance(baseline.get("summary", []), list) else [],
        "violations": violations,
        "triage_items": triage_items,
    }

    json_path = analytics_dir / "control_room_triage.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Control Room Triage",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Run: `{payload['run_id']}` (`{payload['run_profile']}`)",
        f"- Promotion: `{payload['promotion_status']}`",
        f"- Control-room status: `{status}`",
        f"- Threshold violations: `{len(violations)}`",
        "",
        "## Baseline Context",
        "",
    ]
    baseline_summary = payload["baseline_summary"] if isinstance(payload["baseline_summary"], list) else []
    if baseline_summary:
        for item in baseline_summary:
            lines.append(f"- {item}")
    else:
        lines.append("- No baseline comparison summary was available.")

    lines.extend(["", "## Threshold Violations", ""])
    if violations:
        for violation in violations:
            lines.append(f"- {violation['message']}")
    else:
        lines.append("- No configured thresholds were exceeded.")

    lines.extend(["", "## Playbook", ""])
    if triage_items:
        for item in triage_items:
            lines.append("")
            lines.append(f"### {item['title']}")
            lines.append("")
            lines.append(f"- Area: `{item['area']}`")
            lines.append(f"- Priority: `{item['priority']}`")
            lines.append(f"- Trigger: {item['trigger']}")
            for inspect_file in item["inspect_files"]:
                lines.append(f"- Inspect file: `{inspect_file}`")
            for step in item["inspect_steps"]:
                lines.append(f"- Inspect: {step}")
            for step in item["fix_steps"]:
                lines.append(f"- Fix: {step}")
            for step in item["rerun_steps"]:
                lines.append(f"- Rerun: {step}")
    else:
        lines.append("- No triage items were generated.")

    md_path = analytics_dir / "control_room_triage.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None

    control_room_path = _refresh_control_room(outputs_dir)
    control_room = _read_json(control_room_path)
    latest_run = control_room.get("latest_run", {})
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    safety = control_room.get("safety", {})
    safety = safety if isinstance(safety, dict) else {}
    qoe = control_room.get("qoe", {})
    qoe = qoe if isinstance(qoe, dict) else {}

    latest_run_id = str(latest_run.get("run_id", "")).strip()
    requested_run_id = run_dir.name if run_dir is not None else latest_run_id
    status = "ok"
    if requested_run_id and latest_run_id and requested_run_id != latest_run_id:
        status = f"stale:{latest_run_id}"

    max_robustness_gap = _parse_threshold(args.max_robustness_gap)
    max_stress_skip_risk = _parse_threshold(args.max_stress_skip_risk)
    max_target_drift_jsd = _parse_threshold(args.max_target_drift_jsd)
    max_selective_risk = _parse_threshold(args.max_selective_risk)
    thresholds = {
        "max_robustness_gap": max_robustness_gap,
        "max_stress_skip_risk": max_stress_skip_risk,
        "max_target_drift_jsd": max_target_drift_jsd,
        "max_selective_risk": max_selective_risk,
    }

    violations = [
        row
        for row in (
            _violation_row(
                key="robustness_gap",
                label="Worst robustness gap",
                current=safety.get("robustness_max_top1_gap"),
                maximum=max_robustness_gap,
            ),
            _violation_row(
                key="stress_skip_risk",
                label="Worst stress skip risk",
                current=qoe.get("stress_worst_skip_risk"),
                maximum=max_stress_skip_risk,
            ),
            _violation_row(
                key="target_drift_jsd",
                label="Target drift JSD",
                current=safety.get("test_jsd_target_drift"),
                maximum=max_target_drift_jsd,
            ),
            _violation_row(
                key="selective_risk",
                label="Selective risk",
                current=safety.get("test_selective_risk"),
                maximum=max_selective_risk,
            ),
        )
        if row is not None
    ]
    triage_json, triage_md = _write_triage_artifacts(
        outputs_dir=outputs_dir,
        control_room=control_room,
        status=status,
        thresholds=thresholds,
        violations=violations,
    )

    print(
        "run="
        f"{requested_run_id} control_room_status={status} violations={len(violations)} "
        f"robustness_gap={_format_metric(safety.get('robustness_max_top1_gap'))} "
        f"stress_skip_risk={_format_metric(qoe.get('stress_worst_skip_risk'))} "
        f"target_drift_jsd={_format_metric(safety.get('test_jsd_target_drift'))} "
        f"selective_risk={_format_metric(safety.get('test_selective_risk'))}"
    )
    for idx, violation in enumerate(violations, start=1):
        print(f"violation[{idx}]={violation['key']} {violation['message']}")
    print(f"triage_json={triage_json}")
    print(f"triage_md={triage_md}")

    if status.startswith("stale:") and not args.allow_fail:
        return 5
    if violations and not args.allow_fail:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
