from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def _violation_area(key: str) -> str:
    if key in {"robustness_gap", "repeat_from_prev_new_gap"}:
        return "robustness"
    if key in {"stress_skip_risk", "stress_benchmark_skip_risk"}:
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
                "Open analysis/robustness_guardrails.json and analysis/robustness_summary.json to isolate the failing slice.",
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
                "Open analysis/moonshot_summary.json plus analysis/stress_test/stress_test_benchmark.json for the standing benchmark.",
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
    if normalized == "cadence":
        return {
            "inspect": [
                "Open outputs/analytics/control_room_history.csv and confirm when the last fast and full lanes actually ran.",
                "Inspect scripts/run_scheduled.sh and the scheduler entry that triggers it to confirm the intended cadence is still active.",
            ],
            "fix": [
                "Restore the daily fast or weekly full lane before widening the roadmap again.",
                "If smoke or probe runs are polluting review, keep them off the main scheduled cadence or make them complete their analysis pack.",
            ],
            "rerun": [
                "Run the missing scheduled lane and refresh the control room.",
                "Confirm the async handoff and cadence sections now point at the fresh run instead of the stale fallback.",
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


def write_control_room_triage_artifacts(
    *,
    outputs_dir: Path,
    control_room: dict[str, object],
    status: str,
    thresholds: dict[str, float | None] | None = None,
    violations: list[dict[str, object]] | None = None,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    analytics_dir = outputs_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    latest_run = control_room.get("latest_run", {})
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    baseline = control_room.get("baseline_comparison", {})
    baseline = baseline if isinstance(baseline, dict) else {}
    violation_rows = list(violations or [])
    triage_items = _build_triage_items(control_room=control_room, violations=violation_rows)
    generated = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")

    payload = {
        "generated_at": generated,
        "output_dir": str(outputs_dir),
        "control_room_status": str(status),
        "run_id": str(latest_run.get("run_id", "")),
        "run_profile": str(latest_run.get("profile", "")),
        "promotion_status": str(latest_run.get("promotion_status", "")),
        "thresholds": dict(thresholds or {}),
        "baseline_summary": baseline.get("summary", []) if isinstance(baseline.get("summary", []), list) else [],
        "violations": violation_rows,
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
        f"- Threshold violations: `{len(violation_rows)}`",
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
    if violation_rows:
        for violation in violation_rows:
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


__all__ = ["write_control_room_triage_artifacts"]
