from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math

import pandas as pd

from .control_room_rendering import build_weekly_ops_summary_markdown_lines
from .run_artifacts import safe_read_csv as _safe_read_csv
from .run_artifacts import write_json, write_markdown


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_reference_time(reference_time: datetime | None) -> datetime:
    if reference_time is None:
        return datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        return reference_time.replace(tzinfo=timezone.utc)
    return reference_time.astimezone(timezone.utc)


def _operating_lane(operating_rhythm: dict[str, object], lane: str) -> dict[str, object]:
    lanes = operating_rhythm.get("lanes", {})
    lanes = lanes if isinstance(lanes, dict) else {}
    lane_payload = lanes.get(lane, {})
    return lane_payload if isinstance(lane_payload, dict) else {}


def _format_metric(value) -> str:
    metric = _safe_float(value)
    if not math.isfinite(metric):
        return "n/a"
    return f"{metric:.3f}"


def _metric_delta_row(
    *,
    key: str,
    label: str,
    current: object,
    baseline: object,
    higher_is_better: bool,
    epsilon: float = 0.005,
) -> dict[str, object]:
    current_value = _safe_float(current)
    baseline_value = _safe_float(baseline)
    if not math.isfinite(current_value) or not math.isfinite(baseline_value):
        return {
            "key": key,
            "label": label,
            "current": current_value,
            "baseline": baseline_value,
            "delta": float("nan"),
            "status": "unknown",
            "direction": "higher" if higher_is_better else "lower",
        }

    delta = current_value - baseline_value
    if abs(delta) < epsilon:
        status = "flat"
    elif (higher_is_better and delta > 0.0) or ((not higher_is_better) and delta < 0.0):
        status = "better"
    else:
        status = "worse"

    return {
        "key": key,
        "label": label,
        "current": current_value,
        "baseline": baseline_value,
        "delta": delta,
        "status": status,
        "direction": "higher" if higher_is_better else "lower",
    }


def _snapshot_sort_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df.copy()

    frame = history_df.copy()
    for column in (
        "promoted",
        "best_model_val_top1",
        "best_model_test_top1",
        "champion_gate_regression",
        "target_drift_jsd",
        "test_ece",
        "test_selective_risk",
        "test_abstention_rate",
        "robustness_gap",
        "stress_skip_risk",
        "ops_coverage_ratio",
        "available_summary_count",
        "expected_summary_count",
        "review_action_count",
        "high_priority_review_actions",
        "operational_high_priority_review_actions",
        "strategic_high_priority_review_actions",
        "medium_priority_review_actions",
        "next_bet_count",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    run_ts = pd.to_datetime(frame.get("run_timestamp"), errors="coerce", utc=True)
    generated_ts = pd.to_datetime(frame.get("generated_at"), errors="coerce", utc=True)
    frame["_snapshot_ts"] = run_ts.fillna(generated_ts)
    frame["_generated_ts"] = generated_ts
    return frame.sort_values(
        ["_snapshot_ts", "_generated_ts", "run_id"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _control_room_snapshot_row(report: dict[str, object]) -> dict[str, object]:
    latest_run = report.get("latest_run", {})
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    safety = report.get("safety", {})
    safety = safety if isinstance(safety, dict) else {}
    qoe = report.get("qoe", {})
    qoe = qoe if isinstance(qoe, dict) else {}
    ops_coverage = report.get("ops_coverage", {})
    ops_coverage = ops_coverage if isinstance(ops_coverage, dict) else {}
    operating_rhythm = report.get("operating_rhythm", {})
    operating_rhythm = operating_rhythm if isinstance(operating_rhythm, dict) else {}
    async_handoff = report.get("async_handoff", {})
    async_handoff = async_handoff if isinstance(async_handoff, dict) else {}
    ops_health = report.get("ops_health", {})
    ops_health = ops_health if isinstance(ops_health, dict) else {}
    review_actions = report.get("review_actions", [])
    review_actions = review_actions if isinstance(review_actions, list) else []
    baseline = report.get("baseline_comparison", {})
    baseline = baseline if isinstance(baseline, dict) else {}
    baseline_run = baseline.get("baseline_run", {})
    baseline_run = baseline_run if isinstance(baseline_run, dict) else {}

    areas = sorted(
        {
            str(action.get("area", "")).strip().lower()
            for action in review_actions
            if isinstance(action, dict) and str(action.get("area", "")).strip()
        }
    )
    high_count = sum(
        1
        for action in review_actions
        if isinstance(action, dict) and str(action.get("priority", "")).strip().lower() == "high"
    )
    medium_count = sum(
        1
        for action in review_actions
        if isinstance(action, dict) and str(action.get("priority", "")).strip().lower() == "medium"
    )

    return {
        "generated_at": str(report.get("generated_at", "")),
        "run_id": str(latest_run.get("run_id", "")),
        "run_timestamp": str(latest_run.get("timestamp", "")),
        "profile": str(latest_run.get("profile", "")),
        "promoted": int(bool(latest_run.get("promoted"))),
        "promotion_status": str(latest_run.get("promotion_status", "")),
        "best_model_name": str(latest_run.get("best_model_name", "")),
        "best_model_type": str(latest_run.get("best_model_type", "")),
        "best_model_val_top1": _safe_float(latest_run.get("best_model_val_top1")),
        "best_model_test_top1": _safe_float(latest_run.get("best_model_test_top1")),
        "champion_gate_regression": _safe_float(safety.get("champion_gate_regression")),
        "target_drift_jsd": _safe_float(safety.get("test_jsd_target_drift")),
        "test_ece": _safe_float(safety.get("test_ece")),
        "test_selective_risk": _safe_float(safety.get("test_selective_risk")),
        "test_abstention_rate": _safe_float(safety.get("test_abstention_rate")),
        "test_accepted_rate": _safe_float(safety.get("test_accepted_rate")),
        "conformal_operating_threshold": _safe_float(safety.get("conformal_operating_threshold")),
        "robustness_gap": _safe_float(safety.get("robustness_max_top1_gap")),
        "repeat_from_prev_new_gap": _safe_float(safety.get("repeat_from_prev_new_gap")),
        "stress_skip_risk": _safe_float(qoe.get("stress_worst_skip_risk")),
        "stress_benchmark_skip_risk": _safe_float(qoe.get("stress_benchmark_skip_risk")),
        "ops_coverage_ratio": _safe_float(ops_coverage.get("coverage_ratio")),
        "available_summary_count": _safe_int(ops_coverage.get("available_summary_count"), default=0),
        "expected_summary_count": _safe_int(ops_coverage.get("expected_summary_count"), default=0),
        "operating_status": str(operating_rhythm.get("overall_status", "")),
        "ops_health_status": str(ops_health.get("status", "")),
        "fast_cadence_status": str(_operating_lane(operating_rhythm, "fast").get("status", "")),
        "full_cadence_status": str(_operating_lane(operating_rhythm, "full").get("status", "")),
        "async_handoff_status": str(async_handoff.get("status", "")),
        "recommended_run_command": str(operating_rhythm.get("recommended_run_command", "")),
        "review_action_count": int(len(review_actions)),
        "high_priority_review_actions": int(high_count),
        "operational_high_priority_review_actions": _safe_int(
            ops_health.get("operational_high_priority_count"),
            default=0,
        ),
        "strategic_high_priority_review_actions": _safe_int(
            ops_health.get("strategic_high_priority_count"),
            default=0,
        ),
        "medium_priority_review_actions": int(medium_count),
        "review_action_areas": "|".join(areas),
        "baseline_run_id": str(baseline_run.get("run_id", "")),
        "next_bet_count": int(len(report.get("next_bets", []))),
    }


def _write_control_room_history(analytics_dir: Path, report: dict[str, object]) -> tuple[Path, pd.DataFrame]:
    history_path = analytics_dir / "control_room_history.csv"
    existing = _safe_read_csv(history_path)
    row_df = pd.DataFrame([_control_room_snapshot_row(report)])
    combined = pd.concat([existing, row_df], ignore_index=True, sort=False) if not existing.empty else row_df

    if "run_id" in combined.columns:
        generated_ts = pd.to_datetime(combined.get("generated_at"), errors="coerce", utc=True)
        combined["_generated_ts"] = generated_ts
        combined = (
            combined.sort_values(["run_id", "_generated_ts"], ascending=[True, False], na_position="last")
            .drop_duplicates(subset=["run_id"], keep="first")
            .drop(columns=["_generated_ts"])
        )

    sorted_frame = _snapshot_sort_frame(combined)
    for helper_column in ("_snapshot_ts", "_generated_ts"):
        if helper_column in sorted_frame.columns:
            sorted_frame = sorted_frame.drop(columns=[helper_column])
    sorted_frame.to_csv(history_path, index=False)
    return history_path, sorted_frame


def _build_ops_trends(report: dict[str, object], history_df: pd.DataFrame) -> dict[str, object]:
    if history_df.empty:
        return {
            "history_available": False,
            "summary": ["No prior control-room snapshots are available yet."],
            "metric_deltas": [],
        }

    frame = _snapshot_sort_frame(history_df)
    latest_run = report.get("latest_run", {})
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    current_run_id = str(latest_run.get("run_id", ""))
    run_ids = frame.get("run_id", pd.Series(dtype="object")).fillna("").astype(str)
    current_mask = run_ids == current_run_id
    current_records = list(
        frame.loc[
            current_mask,
            [
                "run_id",
                "run_timestamp",
                "profile",
                "promotion_status",
                "best_model_test_top1",
                "robustness_gap",
                "target_drift_jsd",
                "stress_skip_risk",
                "test_selective_risk",
            ],
        ]
        .head(1)
        .itertuples(index=False, name=None)
    )
    previous_records = list(
        frame.loc[
            ~current_mask,
            [
                "run_id",
                "run_timestamp",
                "profile",
                "promotion_status",
                "best_model_test_top1",
                "robustness_gap",
                "target_drift_jsd",
                "stress_skip_risk",
                "test_selective_risk",
            ],
        ]
        .head(1)
        .itertuples(index=False, name=None)
    )
    recent_window = frame.head(min(5, len(frame.index))).copy()

    summary: list[str] = []
    metric_deltas: list[dict[str, object]] = []
    previous_snapshot: dict[str, object] = {}
    if previous_records:
        previous = previous_records[0]
        previous_snapshot = {
            "run_id": str(previous[0]),
            "run_timestamp": str(previous[1]),
            "profile": str(previous[2]),
            "promotion_status": str(previous[3]),
        }
        current = current_records[0] if current_records else ("", "", "", "", None, None, None, None, None)
        metric_deltas = [
            _metric_delta_row(
                key="best_model_test_top1",
                label="Best model test top1",
                current=current[4],
                baseline=previous[4],
                higher_is_better=True,
            ),
            _metric_delta_row(
                key="robustness_gap",
                label="Worst robustness gap",
                current=current[5],
                baseline=previous[5],
                higher_is_better=False,
            ),
            _metric_delta_row(
                key="target_drift_jsd",
                label="Target drift JSD",
                current=current[6],
                baseline=previous[6],
                higher_is_better=False,
            ),
            _metric_delta_row(
                key="stress_skip_risk",
                label="Worst stress skip risk",
                current=current[7],
                baseline=previous[7],
                higher_is_better=False,
            ),
            _metric_delta_row(
                key="selective_risk",
                label="Selective risk",
                current=current[8],
                baseline=previous[8],
                higher_is_better=False,
            ),
        ]
        summary.append(f"Previous snapshot was `{previous_snapshot['run_id']}` at `{previous_snapshot['run_timestamp']}`.")
        for row in metric_deltas:
            if str(row.get("status")) in ("flat", "unknown"):
                continue
            direction_word = "improved" if str(row.get("status")) == "better" else "worsened"
            summary.append(
                f"{row['label']} {direction_word} from {_format_metric(row['baseline'])} to {_format_metric(row['current'])}."
            )
    else:
        summary.append("Only one run snapshot is available, so trend comparisons will start on the next run.")

    promoted_count = int(pd.to_numeric(recent_window.get("promoted"), errors="coerce").fillna(0).sum()) if not recent_window.empty else 0
    failed_promotions = int(len(recent_window.index) - promoted_count)
    high_issue_runs = int(
        (pd.to_numeric(recent_window.get("high_priority_review_actions"), errors="coerce").fillna(0) > 0).sum()
    ) if not recent_window.empty else 0
    operational_issue_runs = int(
        (pd.to_numeric(recent_window.get("operational_high_priority_review_actions"), errors="coerce").fillna(0) > 0).sum()
    ) if not recent_window.empty else 0
    summary.append(
        f"In the last `{len(recent_window.index)}` run snapshots, promotions passed `{promoted_count}` times and failed `{failed_promotions}` times."
    )
    if high_issue_runs > 0:
        summary.append(f"High-priority review actions appeared in `{high_issue_runs}` of the last `{len(recent_window.index)}` snapshots.")
    if operational_issue_runs > 0:
        summary.append(
            f"Operational blockers specifically appeared in `{operational_issue_runs}` of the last `{len(recent_window.index)}` snapshots."
        )

    area_counts: dict[str, int] = {}
    for raw_value in recent_window.get("review_action_areas", pd.Series(dtype="object")).fillna("").astype(str):
        for area in [item for item in raw_value.split("|") if item]:
            area_counts[area] = int(area_counts.get(area, 0) + 1)
    recurring_areas = [
        f"{area} ({count})"
        for area, count in sorted(area_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]
    if recurring_areas:
        summary.append(f"Recurring ops areas across recent runs: {', '.join(recurring_areas[:3])}.")

    async_blocked_count = int(
        recent_window.get("async_handoff_status", pd.Series(dtype="object")).fillna("").astype(str).isin(["blocked"]).sum()
    ) if not recent_window.empty else 0
    fast_issue_count = int(
        recent_window.get("fast_cadence_status", pd.Series(dtype="object")).fillna("").astype(str).isin(["attention", "stale", "missing"]).sum()
    ) if not recent_window.empty else 0
    full_issue_count = int(
        recent_window.get("full_cadence_status", pd.Series(dtype="object")).fillna("").astype(str).isin(["attention", "stale", "missing"]).sum()
    ) if not recent_window.empty else 0
    if async_blocked_count > 0 or fast_issue_count > 0 or full_issue_count > 0:
        summary.append(
            f"Async handoff was blocked in `{async_blocked_count}` recent snapshot(s); fast cadence needed attention in `{fast_issue_count}` and full cadence in `{full_issue_count}`."
        )

    return {
        "history_available": True,
        "snapshot_count": int(len(frame.index)),
        "recent_window_count": int(len(recent_window.index)),
        "previous_snapshot": previous_snapshot,
        "summary": summary[:6],
        "metric_deltas": metric_deltas,
    }


def _write_weekly_ops_summary(
    analytics_dir: Path,
    report: dict[str, object],
    history_df: pd.DataFrame,
    *,
    lookback_days: int = 7,
    generated_at: datetime | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    history_frame = _snapshot_sort_frame(history_df)
    if history_frame.empty:
        window_frame = history_frame.copy()
    else:
        latest_ts = history_frame["_snapshot_ts"].dropna().max()
        if pd.isna(latest_ts):
            window_frame = history_frame.head(min(7, len(history_frame.index))).copy()
        else:
            cutoff = latest_ts - pd.Timedelta(days=max(1, int(lookback_days)))
            window_frame = history_frame[history_frame["_snapshot_ts"] >= cutoff].copy()
            if window_frame.empty:
                window_frame = history_frame.head(min(7, len(history_frame.index))).copy()

    promoted_runs = int(pd.to_numeric(window_frame.get("promoted"), errors="coerce").fillna(0).sum()) if not window_frame.empty else 0
    failed_promotions = int(len(window_frame.index) - promoted_runs)
    avg_test_top1 = _safe_float(pd.to_numeric(window_frame.get("best_model_test_top1"), errors="coerce").mean()) if not window_frame.empty else float("nan")
    worst_robustness_gap = _safe_float(pd.to_numeric(window_frame.get("robustness_gap"), errors="coerce").max()) if not window_frame.empty else float("nan")
    worst_stress_skip_risk = _safe_float(pd.to_numeric(window_frame.get("stress_skip_risk"), errors="coerce").max()) if not window_frame.empty else float("nan")
    worst_selective_risk = _safe_float(pd.to_numeric(window_frame.get("test_selective_risk"), errors="coerce").max()) if not window_frame.empty else float("nan")
    async_blocked_snapshots = int(
        window_frame.get("async_handoff_status", pd.Series(dtype="object")).fillna("").astype(str).isin(["blocked"]).sum()
    ) if not window_frame.empty else 0
    operational_issue_snapshots = int(
        (pd.to_numeric(window_frame.get("operational_high_priority_review_actions"), errors="coerce").fillna(0) > 0).sum()
    ) if not window_frame.empty else 0
    fast_cadence_issue_snapshots = int(
        window_frame.get("fast_cadence_status", pd.Series(dtype="object")).fillna("").astype(str).isin(["attention", "stale", "missing"]).sum()
    ) if not window_frame.empty else 0
    full_cadence_issue_snapshots = int(
        window_frame.get("full_cadence_status", pd.Series(dtype="object")).fillna("").astype(str).isin(["attention", "stale", "missing"]).sum()
    ) if not window_frame.empty else 0

    area_counts: dict[str, int] = {}
    for raw_value in window_frame.get("review_action_areas", pd.Series(dtype="object")).fillna("").astype(str):
        for area in [item for item in raw_value.split("|") if item]:
            area_counts[area] = int(area_counts.get(area, 0) + 1)
    recurring_areas = [
        {"area": area, "count": count}
        for area, count in sorted(area_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]

    review_actions = report.get("review_actions", [])
    review_actions = review_actions if isinstance(review_actions, list) else []
    current_focus = [
        {
            "priority": str(action.get("priority", "")).strip().lower(),
            "area": str(action.get("area", "")).strip().lower(),
            "title": str(action.get("title", "")).strip(),
        }
        for action in review_actions[:5]
        if isinstance(action, dict)
    ]

    summary_lines = [
        f"Runs in window: `{len(window_frame.index)}` with `{promoted_runs}` promotions and `{failed_promotions}` failed promotions.",
        f"Average best-model test top1 across the window is `{_format_metric(avg_test_top1)}`.",
        f"Worst observed robustness gap is `{_format_metric(worst_robustness_gap)}` and worst stress skip risk is `{_format_metric(worst_stress_skip_risk)}`.",
        f"Async handoff was blocked in `{async_blocked_snapshots}` snapshot(s); fast cadence needed attention in `{fast_cadence_issue_snapshots}` and full cadence in `{full_cadence_issue_snapshots}`.",
    ]
    if operational_issue_snapshots > 0:
        summary_lines.append(
            f"Operational blockers appeared in `{operational_issue_snapshots}` snapshot(s) during the weekly window."
        )
    if recurring_areas:
        recurring_labels = [f"{row['area']} ({row['count']})" for row in recurring_areas[:3]]
        summary_lines.append(f"Recurring ops areas this week: {', '.join(recurring_labels)}.")
    else:
        summary_lines.append("No ops area repeated often enough yet to count as a weekly recurring pattern.")

    payload = {
        "generated_at": _normalize_reference_time(generated_at).isoformat(timespec="seconds"),
        "lookback_days": int(max(1, int(lookback_days))),
        "snapshots_considered": int(len(window_frame.index)),
        "promoted_runs": int(promoted_runs),
        "failed_promotions": int(failed_promotions),
        "average_best_model_test_top1": avg_test_top1,
        "worst_robustness_gap": worst_robustness_gap,
        "worst_stress_skip_risk": worst_stress_skip_risk,
        "worst_selective_risk": worst_selective_risk,
        "async_handoff_blocked_snapshots": int(async_blocked_snapshots),
        "operational_issue_snapshots": int(operational_issue_snapshots),
        "fast_cadence_issue_snapshots": int(fast_cadence_issue_snapshots),
        "full_cadence_issue_snapshots": int(full_cadence_issue_snapshots),
        "recurring_areas": recurring_areas,
        "current_focus": current_focus,
        "summary": summary_lines,
        "window_runs": [
            {
                "run_id": str(run_id),
                "run_timestamp": str(run_timestamp),
                "promotion_status": str(promotion_status),
                "best_model_name": str(best_model_name),
                "best_model_test_top1": _safe_float(best_model_test_top1),
                "robustness_gap": _safe_float(robustness_gap),
                "stress_skip_risk": _safe_float(stress_skip_risk),
            }
            for run_id, run_timestamp, promotion_status, best_model_name, best_model_test_top1, robustness_gap, stress_skip_risk in window_frame.reindex(
                columns=[
                    "run_id",
                    "run_timestamp",
                    "promotion_status",
                    "best_model_name",
                    "best_model_test_top1",
                    "robustness_gap",
                    "stress_skip_risk",
                ]
            ).itertuples(index=False, name=None)
        ],
    }

    json_path = write_json(analytics_dir / "control_room_weekly_summary.json", payload)
    md_path = write_markdown(
        analytics_dir / "control_room_weekly_summary.md",
        build_weekly_ops_summary_markdown_lines(payload),
    )
    return json_path, md_path, payload


__all__ = [
    "_build_ops_trends",
    "_format_metric",
    "_metric_delta_row",
    "_normalize_reference_time",
    "_operating_lane",
    "_safe_float",
    "_safe_int",
    "_snapshot_sort_frame",
    "_write_control_room_history",
    "_write_weekly_ops_summary",
]
