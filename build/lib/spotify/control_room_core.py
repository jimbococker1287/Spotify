from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import math

import pandas as pd
from .control_room_business import build_async_handoff as _build_async_handoff_payload
from .control_room_business import build_baseline_comparison as _build_baseline_comparison_payload
from .control_room_business import build_next_bets as _build_next_bets_payload
from .control_room_business import build_operating_rhythm as _build_operating_rhythm_payload
from .control_room_business import build_ops_health as _build_ops_health_payload
from .control_room_business import build_review_actions as _build_review_actions_payload
from .control_room_history import (
    _build_ops_trends,
    _format_metric,
    _metric_delta_row,
    _normalize_reference_time,
    _operating_lane,
    _safe_float,
    _safe_int,
    _write_control_room_history,
    _write_weekly_ops_summary,
)
from .control_room_rendering import build_control_room_markdown_lines
from .control_room_triage import write_control_room_triage_artifacts
from .control_room_selection import (
    _build_cadence_lane as _build_cadence_lane_impl,
    _is_promoted_manifest,
    _latest_promoted_manifest,
    _select_latest_control_room_candidate as _select_latest_control_room_candidate_impl,
    _status_rank,
)
from .model_types import analysis_prefix_for_model_type
from .run_artifacts import collect_run_manifests as _collect_run_manifests
from .run_artifacts import safe_read_csv as _safe_read_csv
from .run_artifacts import safe_read_json as _safe_read_json
from .run_artifacts import write_json, write_markdown

_OPERATIONAL_REVIEW_AREAS = frozenset({"instrumentation", "cadence"})


def _analysis_prefix_for_model_type(model_type: str) -> str | None:
    return analysis_prefix_for_model_type(model_type)


def _split_review_actions(review_actions: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    operational: list[dict[str, object]] = []
    strategic: list[dict[str, object]] = []
    for action in review_actions:
        if not isinstance(action, dict):
            continue
        area = str(action.get("area", "")).strip().lower()
        if area in _OPERATIONAL_REVIEW_AREAS:
            operational.append(action)
        else:
            strategic.append(action)
    return operational, strategic


def _freshness_rank(age_hours: float) -> int:
    if not math.isfinite(age_hours):
        return 0
    if age_hours <= 36:
        return 4
    if age_hours <= 24 * 7:
        return 3
    if age_hours <= 24 * 14:
        return 2
    if age_hours <= 24 * 30:
        return 1
    return 0


def _load_run_results(run_dir: Path) -> list[dict[str, object]]:
    payload = _safe_read_json(run_dir / "run_results.json", default=None)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    return []


def _best_result_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            _safe_float(row.get("val_top1")),
            _safe_float(row.get("test_top1")),
        ),
        default={},
    )


def _resolve_confidence_summary(
    *,
    run_dir: Path,
    manifest: dict[str, object],
    results: list[dict[str, object]],
) -> dict[str, object]:
    alias = manifest.get("champion_alias", {})
    target_name = ""
    target_type = ""
    if isinstance(alias, dict):
        target_name = str(alias.get("model_name", "")).strip()
        target_type = str(alias.get("model_type", "")).strip().lower()

    if not target_name:
        best_row = _best_result_row(results)
        target_name = str(best_row.get("model_name", "")).strip()
        target_type = str(best_row.get("model_type", "")).strip().lower()

    if not target_name:
        return {}

    if not target_type:
        for row in results:
            if str(row.get("model_name", "")).strip() == target_name:
                target_type = str(row.get("model_type", "")).strip().lower()
                break

    prefix = _analysis_prefix_for_model_type(target_type)
    if prefix is None:
        return {}

    payload = _safe_read_json(
        run_dir / "analysis" / f"{prefix}_{target_name}_confidence_summary.json",
        default=None,
    )
    return payload if isinstance(payload, dict) else {}


def _resolve_robustness_guardrail(
    *,
    analysis_dir: Path | None,
    robustness_summary: object,
) -> dict[str, object]:
    payload = _safe_read_json(analysis_dir / "robustness_guardrails.json", default=None) if analysis_dir is not None else {}
    if isinstance(payload, dict) and payload:
        return payload
    if not isinstance(robustness_summary, list) or not robustness_summary:
        return {}
    summary_rows = [row for row in robustness_summary if isinstance(row, dict)]
    if not summary_rows:
        return {}
    segment = str(summary_rows[0].get("guardrail_segment", "")).strip()
    bucket = str(summary_rows[0].get("guardrail_bucket", "")).strip()
    if not segment or not bucket:
        return {}
    models: list[dict[str, object]] = []
    for row in summary_rows:
        guardrail_gap = _safe_float(row.get("guardrail_gap"))
        models.append(
            {
                "model_name": str(row.get("model_name", "")),
                "segment": str(row.get("guardrail_segment", segment)),
                "bucket": str(row.get("guardrail_bucket", bucket)),
                "slice_top1": _safe_float(row.get("guardrail_top1")),
                "slice_gap": guardrail_gap,
                "slice_count": _safe_int(row.get("guardrail_bucket_count"), default=0),
                "global_top1": _safe_float(row.get("global_top1")),
            }
        )
    available_models = [row for row in models if math.isfinite(_safe_float(row.get("slice_gap")))]
    available_models.sort(key=lambda row: _safe_float(row.get("slice_gap")), reverse=True)
    worst_row = available_models[0] if available_models else {}
    return {
        "segment": segment,
        "bucket": bucket,
        "model_count": int(len(models)),
        "available_model_count": int(len(available_models)),
        "worst_model_name": str(worst_row.get("model_name", "")),
        "worst_gap": _safe_float(worst_row.get("slice_gap")),
        "worst_top1": _safe_float(worst_row.get("slice_top1")),
        "worst_bucket_count": _safe_int(worst_row.get("slice_count"), default=0),
        "models": models,
    }


def _resolve_stress_benchmark(
    *,
    analysis_dir: Path | None,
    moonshot_summary: object,
) -> dict[str, object]:
    payload = (
        _safe_read_json(analysis_dir / "stress_test" / "stress_test_benchmark.json", default=None)
        if analysis_dir is not None
        else {}
    )
    if isinstance(payload, dict) and payload:
        return payload
    if not isinstance(moonshot_summary, dict) or not moonshot_summary:
        return {}
    benchmark_scenario = str(moonshot_summary.get("stress_benchmark_scenario", "")).strip()
    benchmark_policy = str(moonshot_summary.get("stress_benchmark_policy_name", "")).strip()
    if not benchmark_scenario or not benchmark_policy:
        return {}
    return {
        "benchmark_scenario": benchmark_scenario,
        "benchmark_policy_name": benchmark_policy,
        "reference_policy_name": str(moonshot_summary.get("stress_benchmark_reference_policy_name", "")).strip(),
        "available": True,
        "reference_available": bool(str(moonshot_summary.get("stress_benchmark_reference_policy_name", "")).strip()),
        "skip_risk": _safe_float(moonshot_summary.get("stress_benchmark_skip_risk")),
        "end_risk": _safe_float(moonshot_summary.get("stress_benchmark_end_risk")),
        "reference_skip_risk": float("nan"),
        "reference_end_risk": float("nan"),
        "skip_risk_delta_vs_reference": _safe_float(moonshot_summary.get("stress_benchmark_skip_delta_vs_reference")),
        "end_risk_delta_vs_reference": float("nan"),
        "scenario_rank_by_skip_risk": _safe_int(moonshot_summary.get("stress_benchmark_scenario_rank"), default=0),
        "scenario_count_for_policy": 0,
        "evaluated_sessions": 0,
        "total_test_sessions": 0,
        "sample_fraction": float("nan"),
    }


def _rank_models(history_df: pd.DataFrame, *, metric_column: str, top_n: int) -> list[dict[str, object]]:
    if history_df.empty or "model_name" not in history_df.columns or metric_column not in history_df.columns:
        return []

    frame = history_df.copy()
    frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
    model_name = frame["model_name"].fillna("").astype(str).str.strip()
    valid_rows = model_name.ne("") & frame[metric_column].notna()
    frame = frame.loc[valid_rows].copy()
    if frame.empty:
        return []
    frame["model_name"] = model_name.loc[frame.index]

    if "model_type" not in frame.columns:
        frame["model_type"] = "unknown"

    run_agg_column = "run_id" if "run_id" in frame.columns else "model_name"
    grouped = (
        frame.groupby(["model_name", "model_type"], dropna=False)
        .agg(
            mean_metric=(metric_column, "mean"),
            best_metric=(metric_column, "max"),
            run_count=(run_agg_column, "nunique"),
        )
        .reset_index()
        .sort_values(["mean_metric", "best_metric", "run_count"], ascending=[False, False, False])
    )

    rows: list[dict[str, object]] = []
    for row in grouped.head(max(1, int(top_n))).itertuples(index=False):
        rows.append(
            {
                "model_name": str(row.model_name),
                "model_type": str(row.model_type),
                "mean_metric": float(row.mean_metric),
                "best_metric": float(row.best_metric),
                "run_count": int(row.run_count),
            }
        )
    return rows


def _build_next_bets(
    *,
    portfolio: dict[str, object],
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    operating_rhythm: dict[str, object],
    run_selection: dict[str, object],
) -> list[str]:
    return _build_next_bets_payload(
        portfolio=portfolio,
        latest_run=latest_run,
        safety=safety,
        qoe=qoe,
        operating_rhythm=operating_rhythm,
        run_selection=run_selection,
        safe_float=_safe_float,
        safe_int=_safe_int,
        operating_lane=_operating_lane,
    )


def _build_run_health_snapshot(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    run_dir = Path(str(manifest.get("run_dir", ""))).expanduser() if manifest else None
    results = _load_run_results(run_dir) if run_dir and run_dir.exists() else []
    best_result = _best_result_row(results)
    analysis_dir = run_dir / "analysis" if run_dir and run_dir.exists() else None

    drift_summary = _safe_read_json(analysis_dir / "data_drift_summary.json", default=None) if analysis_dir is not None else {}
    friction_summary = _safe_read_json(analysis_dir / "friction_proxy_summary.json", default=None) if analysis_dir is not None else {}
    moonshot_summary = _safe_read_json(analysis_dir / "moonshot_summary.json", default=None) if analysis_dir is not None else {}
    robustness_summary = _safe_read_json(analysis_dir / "robustness_summary.json", default=None) if analysis_dir is not None else {}
    robustness_guardrail = _resolve_robustness_guardrail(analysis_dir=analysis_dir, robustness_summary=robustness_summary)
    stress_benchmark = _resolve_stress_benchmark(analysis_dir=analysis_dir, moonshot_summary=moonshot_summary)
    confidence_summary = (
        _resolve_confidence_summary(run_dir=run_dir, manifest=manifest, results=results)
        if run_dir is not None and run_dir.exists()
        else {}
    )

    gate = manifest.get("champion_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    alias = manifest.get("champion_alias", {})
    alias = alias if isinstance(alias, dict) else {}
    phase_timings = manifest.get("phase_timings", {})
    phase_timings = phase_timings if isinstance(phase_timings, dict) else {}
    slowest_phase = phase_timings.get("slowest_phase", {})
    slowest_phase = slowest_phase if isinstance(slowest_phase, dict) else {}
    largest_context_shift = drift_summary.get("largest_context_shift", {}) if isinstance(drift_summary, dict) else {}
    largest_segment_shift = drift_summary.get("largest_segment_shift", {}) if isinstance(drift_summary, dict) else {}
    top_friction_rows = friction_summary.get("top_friction_features", []) if isinstance(friction_summary, dict) else []
    top_friction = top_friction_rows[0] if isinstance(top_friction_rows, list) and top_friction_rows else {}
    worst_robustness = robustness_summary[0] if isinstance(robustness_summary, list) and robustness_summary else {}

    run = {
        "run_id": str(manifest.get("run_id", "")),
        "run_name": str(manifest.get("run_name", "") or ""),
        "profile": str(manifest.get("profile", "")),
        "timestamp": str(manifest.get("timestamp", "")),
        "data_records": _safe_int(manifest.get("data_records")),
        "num_artists": _safe_int(manifest.get("num_artists")),
        "num_context_features": _safe_int(manifest.get("num_context_features")),
        "promoted": bool(gate.get("promoted")),
        "promotion_status": str(gate.get("status", "unknown")),
        "champion_model_name": str(alias.get("model_name", "")),
        "champion_model_type": str(alias.get("model_type", "")),
        "best_model_name": str(best_result.get("model_name", "")),
        "best_model_type": str(best_result.get("model_type", "")),
        "best_model_val_top1": _safe_float(best_result.get("val_top1")),
        "best_model_test_top1": _safe_float(best_result.get("test_top1")),
        "pipeline_total_seconds": _safe_float(phase_timings.get("total_seconds")),
        "pipeline_measured_seconds": _safe_float(phase_timings.get("measured_seconds")),
        "pipeline_unmeasured_overhead_seconds": _safe_float(phase_timings.get("unmeasured_overhead_seconds")),
        "pipeline_slowest_phase": str(slowest_phase.get("phase_name", "")),
        "pipeline_slowest_phase_seconds": _safe_float(slowest_phase.get("duration_seconds")),
    }

    repeat_guardrail_model = str(robustness_guardrail.get("operational_worst_model_name", "")).strip() or str(
        robustness_guardrail.get("worst_model_name", "")
    ).strip()
    repeat_guardrail_gap = _safe_float(robustness_guardrail.get("operational_worst_gap"))
    if math.isnan(repeat_guardrail_gap):
        repeat_guardrail_gap = _safe_float(robustness_guardrail.get("worst_gap"))
    repeat_guardrail_top1 = _safe_float(robustness_guardrail.get("operational_worst_top1"))
    if math.isnan(repeat_guardrail_top1):
        repeat_guardrail_top1 = _safe_float(robustness_guardrail.get("worst_top1"))
    repeat_guardrail_count = _safe_int(robustness_guardrail.get("operational_worst_bucket_count"), default=0)
    if repeat_guardrail_count <= 0:
        repeat_guardrail_count = _safe_int(robustness_guardrail.get("worst_bucket_count"), default=0)

    safety = {
        "champion_gate_status": str(gate.get("status", "unknown")),
        "champion_gate_metric_source": str(gate.get("metric_source", "")),
        "champion_gate_regression": _safe_float(gate.get("regression")),
        "largest_context_shift_feature": str(largest_context_shift.get("feature", "")),
        "largest_context_shift_value": _safe_float(largest_context_shift.get("max_abs_std_mean_diff")),
        "largest_segment_shift_label": (
            f"{largest_segment_shift.get('split', '')}:{largest_segment_shift.get('segment', '')}={largest_segment_shift.get('bucket', '')}"
            if largest_segment_shift
            else ""
        ),
        "largest_segment_shift_value": _safe_float(largest_segment_shift.get("abs_share_shift")),
        "test_jsd_target_drift": _safe_float(
            (drift_summary.get("target_drift", {}) if isinstance(drift_summary, dict) else {}).get("train_vs_test_jsd")
        ),
        "test_ece": _safe_float(confidence_summary.get("test_ece")),
        "test_selective_risk": _safe_float(confidence_summary.get("test_selective_risk")),
        "test_abstention_rate": _safe_float(confidence_summary.get("test_abstention_rate")),
        "test_accepted_rate": _safe_float(confidence_summary.get("test_accepted_rate")),
        "conformal_operating_threshold": _safe_float(confidence_summary.get("conformal_operating_threshold")),
        "robustness_worst_model": str(worst_robustness.get("model_name", "")),
        "robustness_worst_segment": str(worst_robustness.get("worst_segment", "")),
        "robustness_worst_bucket": str(worst_robustness.get("worst_bucket", "")),
        "robustness_max_top1_gap": _safe_float(worst_robustness.get("max_top1_gap")),
        "repeat_from_prev_new_segment": str(robustness_guardrail.get("segment", "")),
        "repeat_from_prev_new_bucket": str(robustness_guardrail.get("bucket", "")),
        "repeat_from_prev_new_model": repeat_guardrail_model,
        "repeat_from_prev_new_gap": repeat_guardrail_gap,
        "repeat_from_prev_new_top1": repeat_guardrail_top1,
        "repeat_from_prev_new_count": repeat_guardrail_count,
    }

    qoe = {
        "friction_status": str(friction_summary.get("status", "")) if isinstance(friction_summary, dict) else "",
        "friction_feature_count": _safe_int(friction_summary.get("friction_feature_count")) if isinstance(friction_summary, dict) else 0,
        "proxy_test_mean_delta": _safe_float(
            (friction_summary.get("proxy_counterfactual", {}) if isinstance(friction_summary, dict) else {}).get("test_mean_delta")
        ),
        "top_friction_feature": str(top_friction.get("feature", "")),
        "top_friction_mean_risk_delta": _safe_float(top_friction.get("mean_risk_delta")),
        "digital_twin_test_auc": _safe_float(moonshot_summary.get("digital_twin_test_auc")) if isinstance(moonshot_summary, dict) else float("nan"),
        "causal_test_auc_total": _safe_float(moonshot_summary.get("causal_test_auc_total")) if isinstance(moonshot_summary, dict) else float("nan"),
        "stress_worst_skip_scenario": str(moonshot_summary.get("stress_worst_skip_scenario", "")) if isinstance(moonshot_summary, dict) else "",
        "stress_worst_skip_risk": _safe_float(moonshot_summary.get("stress_worst_skip_risk")) if isinstance(moonshot_summary, dict) else float("nan"),
        "stress_benchmark_scenario": str(stress_benchmark.get("benchmark_scenario", "")),
        "stress_benchmark_policy_name": str(stress_benchmark.get("benchmark_policy_name", "")),
        "stress_benchmark_skip_risk": _safe_float(stress_benchmark.get("skip_risk")),
        "stress_benchmark_skip_delta_vs_reference": _safe_float(stress_benchmark.get("skip_risk_delta_vs_reference")),
    }
    missing_summaries = [
        filename
        for filename, available in (
            ("run_results.json", bool(results)),
            ("analysis/data_drift_summary.json", isinstance(drift_summary, dict) and bool(drift_summary)),
            ("analysis/friction_proxy_summary.json", isinstance(friction_summary, dict) and bool(friction_summary)),
            ("analysis/robustness_summary.json", isinstance(robustness_summary, list) and bool(robustness_summary)),
            ("analysis/moonshot_summary.json", isinstance(moonshot_summary, dict) and bool(moonshot_summary)),
            ("analysis/*_confidence_summary.json", isinstance(confidence_summary, dict) and bool(confidence_summary)),
        )
        if not available
    ]
    available_summary_count = int(6 - len(missing_summaries))
    ops_coverage = {
        "run_dir_exists": bool(run_dir and run_dir.exists()),
        "analysis_dir_exists": bool(analysis_dir and analysis_dir.exists()),
        "available_summary_count": int(available_summary_count),
        "expected_summary_count": 6,
        "coverage_ratio": float(available_summary_count / 6.0),
        "missing_summaries": missing_summaries,
    }
    return {
        "run": run,
        "safety": safety,
        "qoe": qoe,
        "ops_coverage": ops_coverage,
    }


def _select_latest_control_room_candidate(
    manifests: list[dict[str, object]],
    *,
    reference_time: datetime,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    return _select_latest_control_room_candidate_impl(
        manifests,
        reference_time=reference_time,
        build_run_health_snapshot=_build_run_health_snapshot,
    )


def _build_cadence_lane(
    *,
    manifests: list[dict[str, object]],
    lane: str,
    profiles: set[str],
    target_interval_hours: int,
    reference_time: datetime,
) -> dict[str, object]:
    return _build_cadence_lane_impl(
        manifests=manifests,
        lane=lane,
        profiles=profiles,
        target_interval_hours=target_interval_hours,
        reference_time=reference_time,
        build_run_health_snapshot=_build_run_health_snapshot,
    )




def _build_operating_rhythm(
    *,
    manifests: list[dict[str, object]],
    latest_run: dict[str, object],
    run_selection: dict[str, object],
    reference_time: datetime,
) -> dict[str, object]:
    return _build_operating_rhythm_payload(
        manifests=manifests,
        latest_run=latest_run,
        run_selection=run_selection,
        reference_time=reference_time,
        build_cadence_lane=_build_cadence_lane,
        status_rank=_status_rank,
    )


def _build_baseline_comparison(
    *,
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    baseline_manifest: dict[str, object],
) -> dict[str, object]:
    return _build_baseline_comparison_payload(
        latest_run=latest_run,
        safety=safety,
        qoe=qoe,
        baseline_manifest=baseline_manifest,
        build_run_health_snapshot=_build_run_health_snapshot,
        metric_delta_row=_metric_delta_row,
        format_metric=_format_metric,
    )


def _build_review_actions(
    *,
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    ops_coverage: dict[str, object],
    baseline_comparison: dict[str, object],
    run_selection: dict[str, object],
    operating_rhythm: dict[str, object],
) -> list[dict[str, object]]:
    return _build_review_actions_payload(
        latest_run=latest_run,
        safety=safety,
        qoe=qoe,
        ops_coverage=ops_coverage,
        baseline_comparison=baseline_comparison,
        run_selection=run_selection,
        operating_rhythm=operating_rhythm,
        safe_float=_safe_float,
        safe_int=_safe_int,
        operating_lane=_operating_lane,
        format_metric=_format_metric,
    )


def _build_ops_health(
    *,
    review_actions: list[dict[str, object]],
    operating_rhythm: dict[str, object],
    ops_coverage: dict[str, object],
) -> dict[str, object]:
    return _build_ops_health_payload(
        review_actions=review_actions,
        operating_rhythm=operating_rhythm,
        ops_coverage=ops_coverage,
        split_review_actions=_split_review_actions,
        safe_float=_safe_float,
    )


def _build_async_handoff(
    *,
    latest_run: dict[str, object],
    review_actions: list[dict[str, object]],
    next_bets: list[str],
    operating_rhythm: dict[str, object],
    run_selection: dict[str, object],
    ops_coverage: dict[str, object],
    ops_health: dict[str, object] | None = None,
) -> dict[str, object]:
    return _build_async_handoff_payload(
        latest_run=latest_run,
        review_actions=review_actions,
        next_bets=next_bets,
        operating_rhythm=operating_rhythm,
        run_selection=run_selection,
        ops_coverage=ops_coverage,
        ops_health=ops_health,
        safe_float=_safe_float,
        safe_int=_safe_int,
    )


def build_control_room_report(
    output_dir: Path,
    *,
    top_n: int = 5,
    reference_time: datetime | None = None,
) -> dict[str, object]:
    output_root = output_dir.expanduser().resolve()
    now = _normalize_reference_time(reference_time)
    manifests = _collect_run_manifests(output_root)
    latest_manifest, latest_snapshot, run_selection = _select_latest_control_room_candidate(
        manifests,
        reference_time=now,
    )

    experiment_history = _safe_read_csv(output_root / "history" / "experiment_history.csv")
    backtest_history = _safe_read_csv(output_root / "history" / "backtest_history.csv")
    optuna_history = _safe_read_csv(output_root / "history" / "optuna_history.csv")

    promoted_runs = 0
    profile_values: set[str] = set()
    for manifest in manifests:
        profile = str(manifest.get("profile", "")).strip()
        if profile:
            profile_values.add(profile)
        if _is_promoted_manifest(manifest):
            promoted_runs += 1

    latest_run = latest_snapshot["run"]
    safety = latest_snapshot["safety"]
    qoe = latest_snapshot["qoe"]
    ops_coverage = latest_snapshot["ops_coverage"]
    operating_rhythm = _build_operating_rhythm(
        manifests=manifests,
        latest_run=latest_run,
        run_selection=run_selection,
        reference_time=now,
    )

    baseline_manifest = _latest_promoted_manifest(
        manifests,
        exclude_run_id=str(latest_run.get("run_id", "")) if bool(latest_run.get("promoted")) else None,
    )
    baseline_comparison = _build_baseline_comparison(
        latest_run=latest_run,
        safety=safety,
        qoe=qoe,
        baseline_manifest=baseline_manifest,
    )
    review_actions = _build_review_actions(
        latest_run=latest_run,
        safety=safety,
        qoe=qoe,
        ops_coverage=ops_coverage,
        baseline_comparison=baseline_comparison,
        run_selection=run_selection,
        operating_rhythm=operating_rhythm,
    )
    ops_health = _build_ops_health(
        review_actions=review_actions,
        operating_rhythm=operating_rhythm,
        ops_coverage=ops_coverage,
    )

    report = {
        "generated_at": now.isoformat(timespec="seconds"),
        "output_dir": str(output_root),
        "portfolio": {
            "total_runs": int(len(manifests)),
            "promoted_runs": int(promoted_runs),
            "profiles_seen": sorted(profile_values),
            "experiment_history_rows": int(len(experiment_history.index)),
            "backtest_history_rows": int(len(backtest_history.index)),
            "optuna_history_rows": int(len(optuna_history.index)),
            "latest_run_id": str(latest_manifest.get("run_id", "")),
            "latest_profile": str(latest_manifest.get("profile", "")),
            "latest_observed_run_id": str((run_selection.get("latest_observed_run", {}) if isinstance(run_selection, dict) else {}).get("run_id", "")),
            "latest_observed_profile": str((run_selection.get("latest_observed_run", {}) if isinstance(run_selection, dict) else {}).get("profile", "")),
        },
        "latest_run": latest_run,
        "safety": safety,
        "qoe": qoe,
        "ops_coverage": ops_coverage,
        "run_selection": run_selection,
        "operating_rhythm": operating_rhythm,
        "ops_health": ops_health,
        "leaderboards": {
            "experiment_top_models": _rank_models(experiment_history, metric_column="val_top1", top_n=top_n),
            "backtest_top_models": _rank_models(backtest_history, metric_column="top1", top_n=top_n),
        },
        "baseline_comparison": baseline_comparison,
        "review_actions": review_actions,
        "review_ritual": [
            "Open this control room first after every meaningful run.",
            "Compare the latest run to the last promoted baseline before interpreting regressions.",
            "Clear cadence or instrumentation blockers first, then triage any remaining strategic high-priority findings before the next full run.",
            "If only medium and low priorities remain, capture decisions asynchronously and keep the operating cadence moving.",
        ],
    }
    report["next_bets"] = _build_next_bets(
        portfolio=report["portfolio"],
        latest_run=latest_run,
        safety=safety,
        qoe=qoe,
        operating_rhythm=operating_rhythm,
        run_selection=run_selection,
    )
    report["async_handoff"] = _build_async_handoff(
        latest_run=latest_run,
        review_actions=review_actions,
        next_bets=report["next_bets"],
        operating_rhythm=operating_rhythm,
        run_selection=run_selection,
        ops_coverage=ops_coverage,
        ops_health=ops_health,
    )
    return report


def write_control_room_report(
    output_dir: Path,
    *,
    top_n: int = 5,
    reference_time: datetime | None = None,
) -> tuple[Path, Path]:
    output_root = output_dir.expanduser().resolve()
    analytics_dir = output_root / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    now = _normalize_reference_time(reference_time)
    report = build_control_room_report(output_root, top_n=top_n, reference_time=now)
    history_path, history_df = _write_control_room_history(analytics_dir, report)
    ops_trends = _build_ops_trends(report, history_df)
    weekly_json_path, weekly_md_path, weekly_payload = _write_weekly_ops_summary(
        analytics_dir,
        report,
        history_df,
        generated_at=now,
    )
    report["ops_history"] = {
        "csv_path": str(history_path),
        "snapshot_count": int(len(history_df.index)),
    }
    report["ops_trends"] = ops_trends
    report["weekly_ops_summary"] = {
        "json_path": str(weekly_json_path),
        "markdown_path": str(weekly_md_path),
        "generated_at": str(weekly_payload.get("generated_at", "")),
        "lookback_days": _safe_int(weekly_payload.get("lookback_days"), default=7),
        "snapshots_considered": _safe_int(weekly_payload.get("snapshots_considered"), default=0),
        "summary": [
            str(item)
            for item in weekly_payload.get("summary", [])
            if str(item).strip()
        ]
        if isinstance(weekly_payload.get("summary", []), list)
        else [],
        "current_focus": weekly_payload.get("current_focus", []),
        "async_handoff_blocked_snapshots": _safe_int(weekly_payload.get("async_handoff_blocked_snapshots"), default=0),
        "operational_issue_snapshots": _safe_int(weekly_payload.get("operational_issue_snapshots"), default=0),
        "fast_cadence_issue_snapshots": _safe_int(weekly_payload.get("fast_cadence_issue_snapshots"), default=0),
        "full_cadence_issue_snapshots": _safe_int(weekly_payload.get("full_cadence_issue_snapshots"), default=0),
    }
    triage_json_path, triage_md_path = write_control_room_triage_artifacts(
        outputs_dir=output_root,
        control_room=report,
        status="ok",
        thresholds={},
        violations=[],
        generated_at=now,
    )
    report["triage_artifacts"] = {
        "json_path": str(triage_json_path),
        "markdown_path": str(triage_md_path),
    }

    json_path = write_json(analytics_dir / "control_room.json", report)
    md_path = write_markdown(
        analytics_dir / "control_room.md",
        build_control_room_markdown_lines(
            report,
            format_metric=_format_metric,
            safe_int=_safe_int,
            operating_lane=_operating_lane,
        ),
    )
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a product-style control room summary for Spotify project outputs.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing run artifacts.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top models to keep in each leaderboard.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    json_path, md_path = write_control_room_report(output_dir, top_n=max(1, int(args.top_n)))
    report = _safe_read_json(json_path, default={})
    if not report:
        report = build_control_room_report(output_dir, top_n=max(1, int(args.top_n)))

    latest_run = report["latest_run"]
    run_selection = report.get("run_selection", {})
    run_selection = run_selection if isinstance(run_selection, dict) else {}
    latest_observed = run_selection.get("latest_observed_run", {})
    latest_observed = latest_observed if isinstance(latest_observed, dict) else {}
    print(f"Control room written to {json_path}")
    print(f"Markdown summary written to {md_path}")
    history_path = output_dir / "analytics" / "control_room_history.csv"
    weekly_md_path = output_dir / "analytics" / "control_room_weekly_summary.md"
    operating_rhythm = report.get("operating_rhythm", {})
    operating_rhythm = operating_rhythm if isinstance(operating_rhythm, dict) else {}
    async_handoff = report.get("async_handoff", {})
    async_handoff = async_handoff if isinstance(async_handoff, dict) else {}
    if history_path.exists():
        print(f"Ops history written to {history_path}")
    if weekly_md_path.exists():
        print(f"Weekly ops summary written to {weekly_md_path}")
    if str(latest_observed.get("run_id", "")) and str(latest_observed.get("run_id", "")) != str(latest_run.get("run_id", "")):
        print(f"Latest observed run: {latest_observed['run_id']} ({latest_observed.get('profile', '')})")
    print(f"Latest run: {latest_run['run_id']} ({latest_run['profile']})")
    print(f"Best model: {latest_run['best_model_name']} val_top1={_format_metric(latest_run['best_model_val_top1'])}")
    if str(operating_rhythm.get("recommended_run_command", "")):
        print(f"Recommended next run: {operating_rhythm['recommended_run_command']}")
    if str(async_handoff.get("headline", "")):
        print(f"Async handoff: {async_handoff.get('status', '')} | {async_handoff['headline']}")
    print("Next bets:")
    for bet in report["next_bets"]:
        print(f"- {bet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
