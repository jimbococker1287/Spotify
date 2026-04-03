from __future__ import annotations

from datetime import datetime
import math
from typing import Callable


def build_next_bets(
    *,
    portfolio: dict[str, object],
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    operating_rhythm: dict[str, object],
    run_selection: dict[str, object],
    safe_float: Callable[[object], float],
    safe_int: Callable[[object, int], int],
    operating_lane: Callable[[dict[str, object], str], dict[str, object]],
) -> list[str]:
    bets: list[str] = []

    full_lane = operating_lane(operating_rhythm, "full")
    fast_lane = operating_lane(operating_rhythm, "fast")
    if str(full_lane.get("status", "")) in {"missing", "stale"}:
        bets.append("The weekly full lane is stale; restore `make schedule-run MODE=full` before expanding the roadmap.")
    elif str(fast_lane.get("status", "")) in {"missing", "stale", "attention"}:
        bets.append("The daily fast lane needs attention; tighten the recurring cadence before trusting the current operating rhythm.")

    if bool(operating_rhythm.get("selection_gap")):
        bets.append("The freshest observed run is not the review-ready ops run yet; keep smoke/dev runs separate or finish their artifacts before handoff.")

    if not bool(latest_run.get("promoted")):
        bets.append("Stabilize the promotion path so the latest run can graduate cleanly to champion.")

    target_slice_gap = safe_float(safety.get("repeat_from_prev_new_gap"))
    robustness_gap = safe_float(safety.get("robustness_max_top1_gap"))
    worst_segment = str(safety.get("robustness_worst_segment", "")).strip()
    worst_bucket = str(safety.get("robustness_worst_bucket", "")).strip()
    if math.isfinite(target_slice_gap) and target_slice_gap >= 0.15:
        bets.append(
            "The repeat-from-prev new-listener slice is still fragile; make that guardrail a first-class benchmark before expanding the product surface."
        )
    elif math.isfinite(robustness_gap) and robustness_gap >= 0.15:
        bets.append(
            f"Robustness gaps are concentrated in {worst_segment}={worst_bucket}; slice-aware safeguards are the next highest-leverage build."
        )

    friction_delta = safe_float(qoe.get("proxy_test_mean_delta"))
    top_friction_feature = str(qoe.get("top_friction_feature", "")).strip()
    if math.isfinite(friction_delta) and friction_delta >= 0.03:
        bets.append(
            f"Playback friction looks material (mean test delta {friction_delta:.3f}); expand QoE tooling around {top_friction_feature or 'technical friction'}."
        )

    stress_benchmark_skip_risk = safe_float(qoe.get("stress_benchmark_skip_risk"))
    stress_skip_risk = safe_float(qoe.get("stress_worst_skip_risk"))
    if math.isfinite(stress_benchmark_skip_risk) and stress_benchmark_skip_risk >= 0.35:
        bets.append("The standing evening-drift stress benchmark is still too risky; treat it like a core safety metric, not a side lab.")
    elif math.isfinite(stress_skip_risk) and stress_skip_risk >= 0.35:
        bets.append("The moonshot stress lab is surfacing meaningful failure modes; promote one scenario into first-class regression checks.")

    promoted_runs = safe_int(portfolio.get("promoted_runs"), default=0)
    total_runs = safe_int(portfolio.get("total_runs"), default=0)
    if total_runs >= 5 and promoted_runs <= 1:
        bets.append("You have enough history to define a sharper canonical benchmark and a smaller default product profile.")

    if not bets:
        bets.append("The platform baseline looks healthy; the next leap is turning this control room into a recurring workflow.")
    return bets[:5]


def build_operating_rhythm(
    *,
    manifests: list[dict[str, object]],
    latest_run: dict[str, object],
    run_selection: dict[str, object],
    reference_time: datetime,
    build_cadence_lane: Callable[..., dict[str, object]],
    status_rank: Callable[[str], int],
) -> dict[str, object]:
    fast_lane = build_cadence_lane(
        manifests=manifests,
        lane="fast",
        profiles={"fast", "small", "core", "experimental"},
        target_interval_hours=24,
        reference_time=reference_time,
    )
    full_lane = build_cadence_lane(
        manifests=manifests,
        lane="full",
        profiles={"full"},
        target_interval_hours=24 * 7,
        reference_time=reference_time,
    )
    lane_statuses = [str(fast_lane.get("status", "")), str(full_lane.get("status", ""))]
    overall_status = max(lane_statuses, key=status_rank, default="attention")

    latest_observed = run_selection.get("latest_observed_run", {})
    latest_observed = latest_observed if isinstance(latest_observed, dict) else {}
    observed_run_id = str(latest_observed.get("run_id", ""))
    selected_run_id = str(latest_run.get("run_id", ""))
    selection_gap = bool(observed_run_id) and observed_run_id != selected_run_id

    recommended_run_command = ""
    recommended_run_reason = ""
    if str(full_lane.get("status", "")) in {"missing", "stale"}:
        recommended_run_command = str(full_lane.get("recommended_command", ""))
        recommended_run_reason = "The weekly full lane is stale or missing."
    elif str(fast_lane.get("status", "")) in {"missing", "stale", "attention"}:
        recommended_run_command = str(fast_lane.get("recommended_command", ""))
        recommended_run_reason = "The daily fast lane needs attention."

    summary = [
        str(fast_lane.get("summary", "")).strip(),
        str(full_lane.get("summary", "")).strip(),
    ]
    if selection_gap:
        summary.append(
            f"Latest observed run `{observed_run_id}` is newer than the ops-selected run `{selected_run_id}`. "
            f"{run_selection.get('selection_reason', '')}".strip()
        )
    if recommended_run_command:
        summary.append(f"Recommended next scheduled command: `{recommended_run_command}`.")
    else:
        summary.append("Cadence is healthy enough that the next move is review, not an immediate scheduled rerun.")

    return {
        "overall_status": overall_status,
        "reference_time": reference_time.isoformat(timespec="seconds"),
        "lanes": {
            "fast": fast_lane,
            "full": full_lane,
        },
        "selection_gap": selection_gap,
        "recommended_review_command": "make control-room",
        "recommended_run_command": recommended_run_command,
        "recommended_run_reason": recommended_run_reason,
        "summary": [item for item in summary if item][:5],
    }


def build_baseline_comparison(
    *,
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    baseline_manifest: dict[str, object],
    build_run_health_snapshot: Callable[[dict[str, object]], dict[str, dict[str, object]]],
    metric_delta_row: Callable[..., dict[str, object]],
    format_metric: Callable[[object], str],
) -> dict[str, object]:
    if not baseline_manifest:
        return {
            "baseline_available": False,
            "comparison_mode": "latest_vs_last_strong_run",
            "summary": ["No prior promoted run is available yet, so future ops reviews will use the first successful promotion as the baseline."],
            "metric_deltas": [],
        }

    baseline_snapshot = build_run_health_snapshot(baseline_manifest)
    baseline_run = baseline_snapshot["run"]
    baseline_safety = baseline_snapshot["safety"]
    baseline_qoe = baseline_snapshot["qoe"]

    metric_rows = [
        metric_delta_row(
            key="best_model_test_top1",
            label="Best model test top1",
            current=latest_run.get("best_model_test_top1"),
            baseline=baseline_run.get("best_model_test_top1"),
            higher_is_better=True,
        ),
        metric_delta_row(
            key="best_model_val_top1",
            label="Best model val top1",
            current=latest_run.get("best_model_val_top1"),
            baseline=baseline_run.get("best_model_val_top1"),
            higher_is_better=True,
        ),
        metric_delta_row(
            key="target_drift_jsd",
            label="Target drift JSD",
            current=safety.get("test_jsd_target_drift"),
            baseline=baseline_safety.get("test_jsd_target_drift"),
            higher_is_better=False,
        ),
        metric_delta_row(
            key="test_ece",
            label="Test ECE",
            current=safety.get("test_ece"),
            baseline=baseline_safety.get("test_ece"),
            higher_is_better=False,
        ),
        metric_delta_row(
            key="test_selective_risk",
            label="Selective risk",
            current=safety.get("test_selective_risk"),
            baseline=baseline_safety.get("test_selective_risk"),
            higher_is_better=False,
        ),
        metric_delta_row(
            key="robustness_gap",
            label="Worst robustness gap",
            current=safety.get("robustness_max_top1_gap"),
            baseline=baseline_safety.get("robustness_max_top1_gap"),
            higher_is_better=False,
        ),
        metric_delta_row(
            key="stress_skip_risk",
            label="Worst stress skip risk",
            current=qoe.get("stress_worst_skip_risk"),
            baseline=baseline_qoe.get("stress_worst_skip_risk"),
            higher_is_better=False,
        ),
    ]

    summary: list[str] = []
    changed_model = (
        str(latest_run.get("best_model_name", "")).strip()
        and str(baseline_run.get("best_model_name", "")).strip()
        and str(latest_run.get("best_model_name", "")).strip() != str(baseline_run.get("best_model_name", "")).strip()
    )
    if changed_model:
        summary.append(
            f"Best model changed from {baseline_run['best_model_name']} to {latest_run['best_model_name']}."
        )

    for key in ("best_model_test_top1", "target_drift_jsd", "robustness_gap", "stress_skip_risk"):
        row = next((item for item in metric_rows if str(item.get("key")) == key), None)
        if row is None or str(row.get("status")) in ("flat", "unknown"):
            continue
        direction_word = "improved" if str(row.get("status")) == "better" else "worsened"
        summary.append(
            f"{row['label']} {direction_word} from {format_metric(row['baseline'])} to {format_metric(row['current'])} (delta `{format_metric(row['delta'])}`)."
        )

    if not summary:
        summary.append("The latest run is broadly in line with the last promoted baseline across the tracked ops metrics.")

    return {
        "baseline_available": True,
        "comparison_mode": "latest_vs_last_strong_run",
        "baseline_run": {
            "run_id": str(baseline_run.get("run_id", "")),
            "profile": str(baseline_run.get("profile", "")),
            "timestamp": str(baseline_run.get("timestamp", "")),
            "best_model_name": str(baseline_run.get("best_model_name", "")),
            "best_model_type": str(baseline_run.get("best_model_type", "")),
            "promotion_status": str(baseline_run.get("promotion_status", "")),
        },
        "summary": summary[:5],
        "metric_deltas": metric_rows,
    }


def build_review_actions(
    *,
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    ops_coverage: dict[str, object],
    baseline_comparison: dict[str, object],
    run_selection: dict[str, object],
    operating_rhythm: dict[str, object],
    safe_float: Callable[[object], float],
    safe_int: Callable[[object, int], int],
    operating_lane: Callable[[dict[str, object], str], dict[str, object]],
    format_metric: Callable[[object], str],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []

    coverage_ratio = safe_float(ops_coverage.get("coverage_ratio"))
    missing_summaries = ops_coverage.get("missing_summaries", [])
    missing_summaries = [str(item) for item in missing_summaries if str(item).strip()] if isinstance(missing_summaries, list) else []
    if math.isfinite(coverage_ratio) and coverage_ratio < 0.8:
        preview = ", ".join(missing_summaries[:4]) if missing_summaries else "required analysis outputs"
        actions.append(
            {
                "priority": "high",
                "area": "instrumentation",
                "title": "Backfill missing ops artifacts before trusting this run",
                "detail": (
                    f"Latest run only has `{safe_int(ops_coverage.get('available_summary_count'), default=0)}` of "
                    f"`{safe_int(ops_coverage.get('expected_summary_count'), default=0)}` expected summaries. "
                    f"Missing: {preview}."
                ),
                "inspect": missing_summaries or ["outputs/runs/<run_id>/analysis/"],
            }
        )

    cadence_notes: list[str] = []
    fast_lane = operating_lane(operating_rhythm, "fast")
    full_lane = operating_lane(operating_rhythm, "full")
    for lane_payload in (fast_lane, full_lane):
        lane_status = str(lane_payload.get("status", ""))
        if lane_status not in {"missing", "stale"}:
            continue
        cadence_notes.append(str(lane_payload.get("summary", "")).strip())

    latest_observed = run_selection.get("latest_observed_run", {})
    latest_observed = latest_observed if isinstance(latest_observed, dict) else {}
    selection_gap_note = ""
    if bool(operating_rhythm.get("selection_gap")):
        selection_gap_note = (
            f"Latest observed run `{latest_observed.get('run_id', '')}` is newer than the ops-selected run "
            f"`{latest_run.get('run_id', '')}` because {run_selection.get('selection_reason', 'the newer run is not review-ready yet')}."
        )

    if cadence_notes:
        cadence_priority = "high" if str(full_lane.get("status", "")) in {"missing", "stale"} else "medium"
        if selection_gap_note:
            cadence_notes.append(selection_gap_note)
        actions.append(
            {
                "priority": cadence_priority,
                "area": "cadence",
                "title": "Restore the recurring run cadence",
                "detail": " ".join(note for note in cadence_notes if note),
                "inspect": ["outputs/analytics/control_room_history.csv", "scripts/run_scheduled.sh"],
            }
        )
    elif selection_gap_note:
        actions.append(
            {
                "priority": "low",
                "area": "selection",
                "title": "Keep the freshest run legible while the fuller review pack stays anchored",
                "detail": selection_gap_note,
                "inspect": ["outputs/analytics/control_room.md", "outputs/analytics/control_room_history.csv"],
            }
        )

    if not bool(latest_run.get("promoted")):
        regression = safe_float(safety.get("champion_gate_regression"))
        detail = (
            f"Latest run failed promotion on {latest_run.get('promotion_status', 'unknown')} "
            f"with champion-gate regression `{format_metric(regression)}`."
        )
        baseline_run = baseline_comparison.get("baseline_run", {})
        if isinstance(baseline_run, dict) and baseline_run.get("run_id"):
            detail += f" Compare against promoted baseline `{baseline_run['run_id']}` before retraining."
        actions.append(
            {
                "priority": "high",
                "area": "promotion",
                "title": "Recover the champion path",
                "detail": detail,
                "inspect": ["run_manifest.json", "run_results.json"],
            }
        )

    target_slice_gap = safe_float(safety.get("repeat_from_prev_new_gap"))
    target_slice_model = str(safety.get("repeat_from_prev_new_model", "")).strip()
    target_slice_segment = str(safety.get("repeat_from_prev_new_segment", "")).strip()
    target_slice_bucket = str(safety.get("repeat_from_prev_new_bucket", "")).strip()
    robustness_gap = safe_float(safety.get("robustness_max_top1_gap"))
    if math.isfinite(target_slice_gap) and target_slice_gap >= 0.15:
        actions.append(
            {
                "priority": "high",
                "area": "robustness",
                "title": "Harden the worst slice before the next full run",
                "detail": (
                    f"Guardrail slice {target_slice_segment or 'repeat_from_prev'}={target_slice_bucket or 'new'} "
                    f"is still at gap `{format_metric(target_slice_gap)}` on `{target_slice_model or 'unknown_model'}`."
                ),
                "inspect": ["analysis/robustness_guardrails.json", "analysis/robustness_summary.json"],
            }
        )
    elif math.isfinite(robustness_gap) and robustness_gap >= 0.15:
        actions.append(
            {
                "priority": "high",
                "area": "robustness",
                "title": "Harden the worst slice before the next full run",
                "detail": (
                    f"Worst robustness gap is `{format_metric(robustness_gap)}` on "
                    f"{safety.get('robustness_worst_segment', 'segment')}={safety.get('robustness_worst_bucket', 'bucket')}."
                ),
                "inspect": ["analysis/robustness_summary.json"],
            }
        )

    target_drift = safe_float(safety.get("test_jsd_target_drift"))
    if math.isfinite(target_drift) and target_drift >= 0.15:
        actions.append(
            {
                "priority": "medium",
                "area": "drift",
                "title": "Review drift before trusting regressions",
                "detail": (
                    f"Target drift JSD is `{format_metric(target_drift)}` and segment shift peaks at "
                    f"`{format_metric(safety.get('largest_segment_shift_value'))}` for {safety.get('largest_segment_shift_label', 'n/a')}."
                ),
                "inspect": ["analysis/data_drift_summary.json"],
            }
        )

    stress_benchmark_skip_risk = safe_float(qoe.get("stress_benchmark_skip_risk"))
    stress_benchmark_scenario = str(qoe.get("stress_benchmark_scenario", "")).strip()
    stress_benchmark_policy = str(qoe.get("stress_benchmark_policy_name", "")).strip()
    stress_skip_risk = safe_float(qoe.get("stress_worst_skip_risk"))
    if math.isfinite(stress_benchmark_skip_risk) and stress_benchmark_skip_risk >= 0.35:
        actions.append(
            {
                "priority": "medium",
                "area": "stress_test",
                "title": "Promote the worst stress scenario into regression checks",
                "detail": (
                    f"Standing benchmark `{stress_benchmark_scenario or 'unknown'}` with policy "
                    f"`{stress_benchmark_policy or 'unknown'}` reaches skip risk `{format_metric(stress_benchmark_skip_risk)}`."
                ),
                "inspect": [
                    "analysis/moonshot_summary.json",
                    "analysis/stress_test/stress_test_benchmark.json",
                    "analysis/stress_test/stress_test_summary.json",
                ],
            }
        )
    elif math.isfinite(stress_skip_risk) and stress_skip_risk >= 0.35:
        actions.append(
            {
                "priority": "medium",
                "area": "stress_test",
                "title": "Promote the worst stress scenario into regression checks",
                "detail": (
                    f"Scenario `{qoe.get('stress_worst_skip_scenario', 'unknown')}` reaches skip risk "
                    f"`{format_metric(stress_skip_risk)}` under the current safety route."
                ),
                "inspect": ["analysis/moonshot_summary.json", "analysis/stress_test/stress_test_summary.json"],
            }
        )

    selective_risk = safe_float(safety.get("test_selective_risk"))
    abstention_rate = safe_float(safety.get("test_abstention_rate"))
    accepted_rate = safe_float(safety.get("test_accepted_rate"))
    operating_threshold = safe_float(safety.get("conformal_operating_threshold"))
    if math.isfinite(selective_risk) and selective_risk >= 0.50 and (not math.isfinite(abstention_rate) or abstention_rate <= 0.01):
        actions.append(
            {
                "priority": "medium",
                "area": "uncertainty",
                "title": "Inspect abstention settings before serving broadly",
                "detail": (
                    f"Selective risk is `{format_metric(selective_risk)}` while abstention is `{format_metric(abstention_rate)}` "
                    f"(accepted=`{format_metric(accepted_rate)}`, operating_threshold=`{format_metric(operating_threshold)}`)."
                ),
                "inspect": ["analysis/*_conformal_summary.json"],
            }
        )

    if not actions:
        actions.append(
            {
                "priority": "low",
                "area": "review",
                "title": "Run the normal weekly review",
                "detail": "No acute promotion, drift, robustness, or stress issues crossed the current control-room thresholds.",
                "inspect": ["outputs/analytics/control_room.md"],
            }
        )

    return actions[:6]


def build_ops_health(
    *,
    review_actions: list[dict[str, object]],
    operating_rhythm: dict[str, object],
    ops_coverage: dict[str, object],
    split_review_actions: Callable[[list[dict[str, object]]], tuple[list[dict[str, object]], list[dict[str, object]]]],
    safe_float: Callable[[object], float],
) -> dict[str, object]:
    operational_actions, strategic_actions = split_review_actions(review_actions)
    operational_high = sum(
        1
        for action in operational_actions
        if str(action.get("priority", "")).strip().lower() == "high"
    )
    operational_medium = sum(
        1
        for action in operational_actions
        if str(action.get("priority", "")).strip().lower() == "medium"
    )
    strategic_high = sum(
        1
        for action in strategic_actions
        if str(action.get("priority", "")).strip().lower() == "high"
    )
    coverage_ratio = safe_float(ops_coverage.get("coverage_ratio"))
    cadence_status = str(operating_rhythm.get("overall_status", "")).strip().lower()

    if math.isfinite(coverage_ratio) and coverage_ratio < 0.8:
        status = "blocked"
        headline = "Operational review is blocked until the latest run has a complete artifact pack."
    elif cadence_status in {"missing", "stale", "attention"} or operational_high > 0 or operational_medium > 0:
        status = "attention"
        headline = "Operational review is usable, but cadence or instrumentation still needs attention."
    else:
        status = "healthy"
        headline = "Operational review is healthy; remaining findings are strategic rather than tooling or cadence blockers."

    summary: list[str] = []
    if operational_actions:
        summary.append(
            f"Operational blockers: `{len(operational_actions)}` total with `{operational_high}` high and `{operational_medium}` medium priority."
        )
    else:
        summary.append("No cadence or instrumentation blockers are currently open.")
    if strategic_actions:
        summary.append(
            f"Strategic findings still open: `{len(strategic_actions)}` total with `{strategic_high}` high priority."
        )
    cadence_command = str(operating_rhythm.get("recommended_run_command", "")).strip()
    if cadence_command and cadence_status in {"missing", "stale", "attention"}:
        summary.append(f"Next ops move: `{cadence_command}`.")
    elif cadence_status == "healthy":
        summary.append("Cadence is healthy enough that the next move is review and prioritization, not an immediate rerun.")

    return {
        "status": status,
        "headline": headline,
        "operational_action_count": int(len(operational_actions)),
        "operational_high_priority_count": int(operational_high),
        "operational_medium_priority_count": int(operational_medium),
        "strategic_action_count": int(len(strategic_actions)),
        "strategic_high_priority_count": int(strategic_high),
        "cadence_status": cadence_status,
        "operational_areas": sorted(
            {
                str(action.get("area", "")).strip().lower()
                for action in operational_actions
                if str(action.get("area", "")).strip()
            }
        ),
        "strategic_areas": sorted(
            {
                str(action.get("area", "")).strip().lower()
                for action in strategic_actions
                if str(action.get("area", "")).strip()
            }
        ),
        "summary": summary[:4],
    }


def build_async_handoff(
    *,
    latest_run: dict[str, object],
    review_actions: list[dict[str, object]],
    next_bets: list[str],
    operating_rhythm: dict[str, object],
    run_selection: dict[str, object],
    ops_coverage: dict[str, object],
    ops_health: dict[str, object] | None = None,
    safe_float: Callable[[object], float],
    safe_int: Callable[[object, int], int],
) -> dict[str, object]:
    ops_health = ops_health if isinstance(ops_health, dict) else {}
    top_action = review_actions[0] if review_actions and isinstance(review_actions[0], dict) else {}
    top_priority = str(top_action.get("priority", "")).strip().lower()
    coverage_ratio = safe_float(ops_coverage.get("coverage_ratio"))
    recommended_run_command = str(operating_rhythm.get("recommended_run_command", "")).strip()
    recommended_review_command = str(operating_rhythm.get("recommended_review_command", "make control-room")).strip()
    ops_status = str(ops_health.get("status", "")).strip().lower()
    strategic_high = safe_int(ops_health.get("strategic_high_priority_count"), default=0)

    if math.isfinite(coverage_ratio) and coverage_ratio < 0.8 or ops_status == "blocked":
        status = "blocked"
        headline = "Async review is blocked until the missing ops artifacts are backfilled."
    elif ops_status == "attention":
        status = "attention"
        headline = "Async review is usable, but cadence or instrumentation still needs attention."
    elif strategic_high > 0 or top_priority == "high":
        status = "ready"
        headline = "Async review is ready; strategic safety findings remain open before promotion."
    else:
        status = "ready"
        headline = "Async review is ready; the current report is enough for a teammate handoff."

    latest_observed = run_selection.get("latest_observed_run", {})
    latest_observed = latest_observed if isinstance(latest_observed, dict) else {}
    summary = [
        f"Start with `{recommended_review_command}` and review ops-selected run `{latest_run.get('run_id', '')}` (`{latest_run.get('profile', '')}`).",
        f"Promotion is `{latest_run.get('promotion_status', '')}` and the top open action is `{top_action.get('title', 'none')}`.",
    ]
    for item in ops_health.get("summary", []):
        text = str(item).strip()
        if text:
            summary.append(text)
    if bool(operating_rhythm.get("selection_gap")) and latest_observed.get("run_id"):
        summary.append(
            f"Latest observed run `{latest_observed.get('run_id', '')}` is newer than the selected run, so include the run-selection note in the handoff."
        )
    if recommended_run_command:
        summary.append(f"Recommended next scheduled command: `{recommended_run_command}`.")
    else:
        summary.append("No immediate scheduled rerun is required once the current review actions are acknowledged.")
    if next_bets:
        summary.append(f"Primary next bet: {next_bets[0]}")

    share_artifacts = [
        "outputs/analytics/control_room.md",
        "outputs/analytics/control_room_weekly_summary.md",
    ]
    if any(
        isinstance(action, dict)
        and str(action.get("priority", "")).strip().lower() in {"high", "medium"}
        for action in review_actions
    ):
        share_artifacts.append("outputs/analytics/control_room_triage.md")

    return {
        "status": status,
        "headline": headline,
        "recommended_review_command": recommended_review_command,
        "recommended_run_command": recommended_run_command,
        "share_artifacts": share_artifacts,
        "summary": summary[:5],
    }
