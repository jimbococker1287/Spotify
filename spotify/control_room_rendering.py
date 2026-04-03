from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def build_weekly_ops_summary_markdown_lines(payload: Mapping[str, Any]) -> list[str]:
    current_focus = payload.get("current_focus", [])
    current_focus = current_focus if isinstance(current_focus, list) else []

    lines = [
        "# Weekly Ops Summary",
        "",
        f"- Generated: `{payload.get('generated_at', '')}`",
        f"- Lookback days: `{payload.get('lookback_days', '')}`",
        f"- Snapshots considered: `{payload.get('snapshots_considered', '')}`",
        "",
        "## Summary",
        "",
    ]
    for item in payload.get("summary", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Operating Rhythm",
            "",
            f"- Async handoff blocked snapshots: `{payload.get('async_handoff_blocked_snapshots', '')}`",
            f"- Fast cadence issue snapshots: `{payload.get('fast_cadence_issue_snapshots', '')}`",
            f"- Full cadence issue snapshots: `{payload.get('full_cadence_issue_snapshots', '')}`",
        ]
    )
    lines.extend(["", "## Current Focus", ""])
    if current_focus:
        for item in current_focus:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- [{str(item.get('priority', '')).upper()}] {item.get('title', '')} ({item.get('area', '')})"
            )
    else:
        lines.append("- No current review actions are open.")
    return lines


def build_control_room_markdown_lines(
    report: Mapping[str, Any],
    *,
    format_metric: Callable[[Any], str],
    safe_int: Callable[[Any, int], int],
    operating_lane: Callable[[dict[str, object], str], dict[str, object]],
) -> list[str]:
    portfolio = _mapping(report.get("portfolio", {}))
    latest_run = _mapping(report.get("latest_run", {}))
    safety = _mapping(report.get("safety", {}))
    qoe = _mapping(report.get("qoe", {}))
    ops_coverage = _mapping(report.get("ops_coverage", {}))
    run_selection = _mapping(report.get("run_selection", {}))
    operating_rhythm = _mapping(report.get("operating_rhythm", {}))
    ops_health = _mapping(report.get("ops_health", {}))
    async_handoff = _mapping(report.get("async_handoff", {}))
    baseline = _mapping(report.get("baseline_comparison", {}))
    ops_history = _mapping(report.get("ops_history", {}))
    ops_trends = _mapping(report.get("ops_trends", {}))
    weekly_summary = _mapping(report.get("weekly_ops_summary", {}))
    leaderboards = _mapping(report.get("leaderboards", {}))

    review_actions = report.get("review_actions", [])
    review_actions = review_actions if isinstance(review_actions, list) else []
    review_ritual = report.get("review_ritual", [])
    review_ritual = review_ritual if isinstance(review_ritual, list) else []
    next_bets = report.get("next_bets", [])
    next_bets = next_bets if isinstance(next_bets, list) else []

    fast_lane = operating_lane(operating_rhythm, "fast")
    full_lane = operating_lane(operating_rhythm, "full")
    fast_lane_run = _mapping(fast_lane.get("latest_run", {}))
    full_lane_run = _mapping(full_lane.get("latest_run", {}))
    latest_observed_run = _mapping(run_selection.get("latest_observed_run", {}))
    selected_run = _mapping(run_selection.get("selected_run", {}))

    lines = [
        "# Control Room",
        "",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Output root: `{report.get('output_dir', '')}`",
        "",
        "## Portfolio",
        "",
        f"- Runs tracked: `{portfolio.get('total_runs', '')}`",
        f"- Promoted runs: `{portfolio.get('promoted_runs', '')}`",
        f"- Profiles seen: `{', '.join(portfolio.get('profiles_seen', [])) if isinstance(portfolio.get('profiles_seen', []), list) and portfolio.get('profiles_seen', []) else 'n/a'}`",
        f"- Experiment history rows: `{portfolio.get('experiment_history_rows', '')}`",
        f"- Backtest history rows: `{portfolio.get('backtest_history_rows', '')}`",
        "",
        "## Run Selection",
        "",
        f"- Latest observed run: `{latest_observed_run.get('run_id', '')}` (`{latest_observed_run.get('profile', '')}`)",
        f"- Ops-selected run: `{selected_run.get('run_id', '')}` (`{selected_run.get('profile', '')}`)",
        f"- Selection mode: `{run_selection.get('selection_mode', '')}`",
        f"- Selection reason: {run_selection.get('selection_reason', 'n/a')}",
        "",
        "## Operating Rhythm",
        "",
        f"- Overall status: `{operating_rhythm.get('overall_status', '')}`",
        f"- Fast lane: `{fast_lane.get('status', '')}` latest=`{fast_lane_run.get('run_id', '')}` age_h=`{format_metric(fast_lane.get('hours_since_run'))}`",
        f"- Full lane: `{full_lane.get('status', '')}` latest=`{full_lane_run.get('run_id', '')}` age_h=`{format_metric(full_lane.get('hours_since_run'))}`",
        f"- Recommended next run: `{operating_rhythm.get('recommended_run_command', '') or 'none'}`",
        "",
        "## Ops Health",
        "",
        f"- Status: `{ops_health.get('status', '')}`",
        f"- Operational blockers: `{safe_int(ops_health.get('operational_action_count'), default=0)}` total with high=`{safe_int(ops_health.get('operational_high_priority_count'), default=0)}` medium=`{safe_int(ops_health.get('operational_medium_priority_count'), default=0)}`",
        f"- Strategic findings: `{safe_int(ops_health.get('strategic_action_count'), default=0)}` total with high=`{safe_int(ops_health.get('strategic_high_priority_count'), default=0)}`",
        "",
    ]
    for item in ops_health.get("summary", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Latest Run",
            "",
            f"- Run: `{latest_run.get('run_id', '')}` (`{latest_run.get('profile', '')}`)",
            f"- Timestamp: `{latest_run.get('timestamp', '')}`",
            f"- Promotion: `{latest_run.get('promotion_status', '')}`",
            f"- Best model: `{latest_run.get('best_model_name', '')}` [{latest_run.get('best_model_type', '')}] val_top1=`{format_metric(latest_run.get('best_model_val_top1'))}` test_top1=`{format_metric(latest_run.get('best_model_test_top1'))}`",
            f"- Champion alias: `{latest_run.get('champion_model_name', '')}` [{latest_run.get('champion_model_type', '')}]",
            f"- Pipeline time: total=`{format_metric(latest_run.get('pipeline_total_seconds'))}`s measured=`{format_metric(latest_run.get('pipeline_measured_seconds'))}`s overhead=`{format_metric(latest_run.get('pipeline_unmeasured_overhead_seconds'))}`s",
            f"- Slowest phase: `{latest_run.get('pipeline_slowest_phase', '')}` duration=`{format_metric(latest_run.get('pipeline_slowest_phase_seconds'))}`s",
            "",
            "## Ops Coverage",
            "",
            f"- Available summaries: `{safe_int(ops_coverage.get('available_summary_count'), default=0)}` / `{safe_int(ops_coverage.get('expected_summary_count'), default=0)}`",
            f"- Coverage ratio: `{format_metric(ops_coverage.get('coverage_ratio'))}`",
            f"- Missing summaries: `{', '.join(ops_coverage.get('missing_summaries', [])) if isinstance(ops_coverage.get('missing_summaries', []), list) and ops_coverage.get('missing_summaries', []) else 'none'}`",
            "",
            "## Safety",
            "",
            f"- Champion gate metric: `{safety.get('champion_gate_metric_source', '')}` regression=`{format_metric(safety.get('champion_gate_regression'))}`",
            f"- Target drift (train->test JSD): `{format_metric(safety.get('test_jsd_target_drift'))}`",
            f"- Largest context shift: `{safety.get('largest_context_shift_feature', '')}` value=`{format_metric(safety.get('largest_context_shift_value'))}`",
            f"- Largest segment shift: `{safety.get('largest_segment_shift_label', '')}` value=`{format_metric(safety.get('largest_segment_shift_value'))}`",
            f"- Worst robustness gap: `{safety.get('robustness_worst_model', '')}` {safety.get('robustness_worst_segment', '')}={safety.get('robustness_worst_bucket', '')} gap=`{format_metric(safety.get('robustness_max_top1_gap'))}`",
            f"- Guardrail slice: `{safety.get('repeat_from_prev_new_model', '')}` {safety.get('repeat_from_prev_new_segment', '')}={safety.get('repeat_from_prev_new_bucket', '')} gap=`{format_metric(safety.get('repeat_from_prev_new_gap'))}` top1=`{format_metric(safety.get('repeat_from_prev_new_top1'))}`",
            f"- Test ECE: `{format_metric(safety.get('test_ece'))}` selective_risk=`{format_metric(safety.get('test_selective_risk'))}` abstention=`{format_metric(safety.get('test_abstention_rate'))}` accepted=`{format_metric(safety.get('test_accepted_rate'))}` operating_threshold=`{format_metric(safety.get('conformal_operating_threshold'))}`",
            "",
            "## QoE",
            "",
            f"- Friction analysis: `{qoe.get('friction_status', '')}` with `{qoe.get('friction_feature_count', '')}` friction features",
            f"- Mean test skip-risk delta without friction: `{format_metric(qoe.get('proxy_test_mean_delta'))}`",
            f"- Top friction feature: `{qoe.get('top_friction_feature', '')}` delta=`{format_metric(qoe.get('top_friction_mean_risk_delta'))}`",
            f"- Digital twin test AUC: `{format_metric(qoe.get('digital_twin_test_auc'))}` causal test AUC=`{format_metric(qoe.get('causal_test_auc_total'))}`",
            f"- Stress scenario: `{qoe.get('stress_worst_skip_scenario', '')}` skip_risk=`{format_metric(qoe.get('stress_worst_skip_risk'))}`",
            f"- Standing stress benchmark: `{qoe.get('stress_benchmark_scenario', '')}` policy=`{qoe.get('stress_benchmark_policy_name', '')}` skip_risk=`{format_metric(qoe.get('stress_benchmark_skip_risk'))}` delta_vs_reference=`{format_metric(qoe.get('stress_benchmark_skip_delta_vs_reference'))}`",
            "",
            "## Since Last Strong Run",
            "",
        ]
    )

    if bool(baseline.get("baseline_available")):
        baseline_run = _mapping(baseline.get("baseline_run", {}))
        lines.extend(
            [
                f"- Baseline run: `{baseline_run.get('run_id', '')}` (`{baseline_run.get('profile', '')}`) at `{baseline_run.get('timestamp', '')}`",
                f"- Baseline best model: `{baseline_run.get('best_model_name', '')}` [{baseline_run.get('best_model_type', '')}]",
                "",
            ]
        )
        for item in baseline.get("summary", []):
            lines.append(f"- {item}")
    else:
        for item in baseline.get("summary", []):
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Recent Trends",
            "",
            f"- Snapshots tracked: `{safe_int(ops_history.get('snapshot_count'), default=0)}`",
            "- History artifact: `outputs/analytics/control_room_history.csv`",
            "- Weekly summary artifact: `outputs/analytics/control_room_weekly_summary.md`",
            "",
        ]
    )
    for item in ops_trends.get("summary", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Weekly Window",
            "",
            f"- Snapshots considered: `{safe_int(weekly_summary.get('snapshots_considered'), default=0)}` over `{safe_int(weekly_summary.get('lookback_days'), default=7)}` day(s)",
        ]
    )
    for item in weekly_summary.get("summary", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Review Actions",
            "",
        ]
    )
    for action in review_actions:
        if not isinstance(action, dict):
            continue
        detail = str(action.get("detail", "")).strip()
        inspect = action.get("inspect", [])
        inspect_items = inspect if isinstance(inspect, list) else []
        inspect_text = f" Inspect: {', '.join(str(item) for item in inspect_items if item)}." if inspect_items else ""
        lines.append(f"- [{str(action.get('priority', '')).upper()}] {action.get('title', '')}: {detail}{inspect_text}")

    lines.extend(
        [
            "",
            "## Review Ritual",
            "",
        ]
    )
    for step in review_ritual:
        lines.append(f"- {step}")

    lines.extend(
        [
            "",
            "## Async Handoff",
            "",
            f"- Status: `{async_handoff.get('status', '')}`",
            f"- Headline: {async_handoff.get('headline', 'n/a')}",
            f"- Review command: `{async_handoff.get('recommended_review_command', '')}`",
            f"- Next run command: `{async_handoff.get('recommended_run_command', '') or 'none'}`",
            f"- Share artifacts: `{', '.join(async_handoff.get('share_artifacts', [])) if isinstance(async_handoff.get('share_artifacts', []), list) else ''}`",
            "",
        ]
    )
    for item in async_handoff.get("summary", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Leaderboards",
            "",
            "### Experiment Top Models",
            "",
        ]
    )

    for row in leaderboards.get("experiment_top_models", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('model_name', '')}` [{row.get('model_type', '')}] mean_val_top1=`{format_metric(row.get('mean_metric'))}` best=`{format_metric(row.get('best_metric'))}` runs=`{row.get('run_count', '')}`"
        )

    lines.extend(["", "### Backtest Top Models", ""])
    for row in leaderboards.get("backtest_top_models", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('model_name', '')}` [{row.get('model_type', '')}] mean_backtest_top1=`{format_metric(row.get('mean_metric'))}` best=`{format_metric(row.get('best_metric'))}` runs=`{row.get('run_count', '')}`"
        )

    lines.extend(["", "## Next Bets", ""])
    for bet in next_bets:
        lines.append(f"- {bet}")
    return lines
