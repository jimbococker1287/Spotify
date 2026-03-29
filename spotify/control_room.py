from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import math

import pandas as pd
from .run_artifacts import collect_run_manifests as _collect_run_manifests
from .run_artifacts import safe_read_csv as _safe_read_csv
from .run_artifacts import safe_read_json as _safe_read_json


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


def _analysis_prefix_for_model_type(model_type: str) -> str | None:
    normalized = str(model_type).strip().lower()
    if normalized == "deep":
        return "deep"
    if normalized in ("classical", "classical_tuned"):
        return "classical"
    if normalized in ("retrieval", "retrieval_reranker", "ensemble"):
        return normalized
    return None


def _manifest_sort_key(row: dict[str, object]) -> tuple[str, str]:
    return (
        str(row.get("timestamp", "")),
        str(row.get("run_id", "")),
    )


def _latest_manifest(manifests: list[dict[str, object]]) -> dict[str, object]:
    if not manifests:
        return {}
    return max(manifests, key=_manifest_sort_key)


def _is_promoted_manifest(manifest: dict[str, object]) -> bool:
    gate = manifest.get("champion_gate", {})
    if not isinstance(gate, dict):
        return False
    return bool(gate.get("promoted"))


def _latest_promoted_manifest(
    manifests: list[dict[str, object]],
    *,
    exclude_run_id: str | None = None,
) -> dict[str, object]:
    return max(
        (
            manifest
            for manifest in manifests
            if _is_promoted_manifest(manifest)
            and str(manifest.get("run_id", "")) != str(exclude_run_id or "")
        ),
        key=_manifest_sort_key,
        default={},
    )


def _profile_signal_rank(profile: str) -> int:
    normalized = str(profile).strip().lower()
    if normalized == "full":
        return 5
    if normalized in {"experimental", "core"}:
        return 4
    if normalized in {"small", "fast"}:
        return 3
    if normalized == "dev":
        return 1
    return 2


def _manifest_looks_like_smoke_run(manifest: dict[str, object]) -> bool:
    combined = " ".join(
        [
            str(manifest.get("run_name", "")).strip().lower(),
            str(manifest.get("run_id", "")).strip().lower(),
        ]
    )
    if not combined.strip():
        return False
    smoke_hints = ("check", "smoke", "probe", "debug", "verify")
    return any(hint in combined for hint in smoke_hints)


def _normalize_reference_time(reference_time: datetime | None) -> datetime:
    if reference_time is None:
        return datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        return reference_time.replace(tzinfo=timezone.utc)
    return reference_time.astimezone(timezone.utc)


def _parse_manifest_timestamp(value: object) -> datetime | None:
    raw_value = str(value).strip()
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_hours(timestamp: object, *, reference_time: datetime) -> float:
    parsed = _parse_manifest_timestamp(timestamp)
    if parsed is None:
        return float("nan")
    delta_hours = (reference_time - parsed).total_seconds() / 3600.0
    return max(delta_hours, 0.0)


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
    bets: list[str] = []

    full_lane = _operating_lane(operating_rhythm, "full")
    fast_lane = _operating_lane(operating_rhythm, "fast")
    if str(full_lane.get("status", "")) in {"missing", "stale"}:
        bets.append("The weekly full lane is stale; restore `make schedule-run MODE=full` before expanding the roadmap.")
    elif str(fast_lane.get("status", "")) in {"missing", "stale", "attention"}:
        bets.append("The daily fast lane needs attention; tighten the recurring cadence before trusting the current operating rhythm.")

    if bool(operating_rhythm.get("selection_gap")):
        bets.append("The freshest observed run is not the review-ready ops run yet; keep smoke/dev runs separate or finish their artifacts before handoff.")

    if not bool(latest_run.get("promoted")):
        bets.append("Stabilize the promotion path so the latest run can graduate cleanly to champion.")

    robustness_gap = _safe_float(safety.get("robustness_max_top1_gap"))
    worst_segment = str(safety.get("robustness_worst_segment", "")).strip()
    worst_bucket = str(safety.get("robustness_worst_bucket", "")).strip()
    if math.isfinite(robustness_gap) and robustness_gap >= 0.15:
        bets.append(
            f"Robustness gaps are concentrated in {worst_segment}={worst_bucket}; slice-aware safeguards are the next highest-leverage build."
        )

    friction_delta = _safe_float(qoe.get("proxy_test_mean_delta"))
    top_friction_feature = str(qoe.get("top_friction_feature", "")).strip()
    if math.isfinite(friction_delta) and friction_delta >= 0.03:
        bets.append(
            f"Playback friction looks material (mean test delta {friction_delta:.3f}); expand QoE tooling around {top_friction_feature or 'technical friction'}."
        )

    stress_skip_risk = _safe_float(qoe.get("stress_worst_skip_risk"))
    if math.isfinite(stress_skip_risk) and stress_skip_risk >= 0.35:
        bets.append("The moonshot stress lab is surfacing meaningful failure modes; promote one scenario into first-class regression checks.")

    promoted_runs = _safe_int(portfolio.get("promoted_runs"))
    total_runs = _safe_int(portfolio.get("total_runs"))
    if total_runs >= 5 and promoted_runs <= 1:
        bets.append("You have enough history to define a sharper canonical benchmark and a smaller default product profile.")

    if not bets:
        bets.append("The platform baseline looks healthy; the next leap is turning this control room into a recurring workflow.")
    return bets[:5]


def _build_run_health_snapshot(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    run_dir = Path(str(manifest.get("run_dir", ""))).expanduser() if manifest else None
    results = _load_run_results(run_dir) if run_dir and run_dir.exists() else []
    best_result = _best_result_row(results)
    analysis_dir = run_dir / "analysis" if run_dir and run_dir.exists() else None

    drift_summary = _safe_read_json(analysis_dir / "data_drift_summary.json", default=None) if analysis_dir is not None else {}
    friction_summary = _safe_read_json(analysis_dir / "friction_proxy_summary.json", default=None) if analysis_dir is not None else {}
    moonshot_summary = _safe_read_json(analysis_dir / "moonshot_summary.json", default=None) if analysis_dir is not None else {}
    robustness_summary = _safe_read_json(analysis_dir / "robustness_summary.json", default=None) if analysis_dir is not None else {}
    confidence_summary = (
        _resolve_confidence_summary(run_dir=run_dir, manifest=manifest, results=results)
        if run_dir is not None and run_dir.exists()
        else {}
    )

    gate = manifest.get("champion_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    alias = manifest.get("champion_alias", {})
    alias = alias if isinstance(alias, dict) else {}
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
    }

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
        "robustness_worst_model": str(worst_robustness.get("model_name", "")),
        "robustness_worst_segment": str(worst_robustness.get("worst_segment", "")),
        "robustness_worst_bucket": str(worst_robustness.get("worst_bucket", "")),
        "robustness_max_top1_gap": _safe_float(worst_robustness.get("max_top1_gap")),
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


def _manifest_ops_signal(
    manifest: dict[str, object],
    snapshot: dict[str, dict[str, object]],
    *,
    reference_time: datetime,
) -> dict[str, object]:
    profile = str(manifest.get("profile", "")).strip().lower()
    ops_coverage = snapshot.get("ops_coverage", {})
    ops_coverage = ops_coverage if isinstance(ops_coverage, dict) else {}
    coverage_ratio = _safe_float(ops_coverage.get("coverage_ratio"))
    if not math.isfinite(coverage_ratio):
        coverage_ratio = -1.0
    backtest_rows = _safe_int(manifest.get("backtest_rows"))
    optuna_rows = _safe_int(manifest.get("optuna_rows"))
    smoke_like = _manifest_looks_like_smoke_run(manifest)
    production_profile = profile not in {"", "dev"}
    profile_rank = _profile_signal_rank(profile)
    coverage_ready = coverage_ratio >= 0.80
    age_hours = _age_hours(manifest.get("timestamp", ""), reference_time=reference_time)
    freshness_rank = _freshness_rank(age_hours)
    return {
        "run_id": str(manifest.get("run_id", "")),
        "run_name": str(manifest.get("run_name", "") or ""),
        "profile": profile,
        "timestamp": str(manifest.get("timestamp", "")),
        "smoke_like": bool(smoke_like),
        "production_profile": bool(production_profile),
        "profile_rank": int(profile_rank),
        "backtest_rows": int(backtest_rows),
        "optuna_rows": int(optuna_rows),
        "coverage_ratio": float(coverage_ratio),
        "available_summary_count": _safe_int(ops_coverage.get("available_summary_count"), default=0),
        "expected_summary_count": _safe_int(ops_coverage.get("expected_summary_count"), default=0),
        "coverage_ready": bool(coverage_ready),
        "age_hours": float(age_hours) if math.isfinite(age_hours) else float("nan"),
        "freshness_rank": int(freshness_rank),
        "sort_key": (
            int(coverage_ready),
            int(not smoke_like),
            int(freshness_rank),
            int(production_profile),
            int(backtest_rows > 0),
            int(profile_rank),
            float(coverage_ratio),
            int(backtest_rows),
            int(optuna_rows),
            str(manifest.get("timestamp", "")),
            str(manifest.get("run_id", "")),
        ),
    }


def _build_run_selection_summary(
    *,
    latest_observed_signal: dict[str, object],
    selected_signal: dict[str, object],
) -> dict[str, object]:
    observed_run_id = str(latest_observed_signal.get("run_id", ""))
    selected_run_id = str(selected_signal.get("run_id", ""))
    observed_matches_selected = observed_run_id == selected_run_id

    if observed_matches_selected:
        if bool(selected_signal.get("coverage_ready")) and not bool(selected_signal.get("smoke_like")):
            reason = "Latest observed run already looks like the strongest ops candidate."
        elif not bool(selected_signal.get("smoke_like")) and bool(selected_signal.get("production_profile")):
            reason = "Latest observed run is still the strongest ops candidate, but its ops coverage is incomplete."
        else:
            reason = "No stronger ops candidate was available, so the control room stayed on the latest observed run."
    else:
        reasons: list[str] = []
        if bool(latest_observed_signal.get("smoke_like")):
            reasons.append("the latest observed run looks like a smoke/check run")
        if not bool(latest_observed_signal.get("production_profile")):
            reasons.append(
                f"the latest observed run uses the `{latest_observed_signal.get('profile', 'unknown')}` profile"
            )
        if float(selected_signal.get("coverage_ratio", -1.0)) > float(latest_observed_signal.get("coverage_ratio", -1.0)):
            reasons.append("the selected run has better ops artifact coverage")
        if _safe_int(selected_signal.get("backtest_rows"), default=0) > _safe_int(latest_observed_signal.get("backtest_rows"), default=0):
            reasons.append("the selected run has stronger backtest evidence")
        if not reasons:
            reasons.append("the selected run scored higher on the ops signal ranking")
        reason = (
            "Latest observed run was skipped because "
            + "; ".join(reasons)
            + f". Control room selected `{selected_run_id}` instead."
        )

    return {
        "selection_mode": "ops_signal_ranking",
        "observed_matches_selected": bool(observed_matches_selected),
        "selection_reason": reason,
        "latest_observed_run": latest_observed_signal,
        "selected_run": selected_signal,
    }


def _select_latest_control_room_candidate(
    manifests: list[dict[str, object]],
    *,
    reference_time: datetime,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    if not manifests:
        return {}, {"run": {}, "safety": {}, "qoe": {}, "ops_coverage": {}}, {
            "selection_mode": "ops_signal_ranking",
            "observed_matches_selected": True,
            "selection_reason": "No run manifests were available.",
            "latest_observed_run": {},
            "selected_run": {},
        }

    candidates: list[dict[str, object]] = []
    for manifest in manifests:
        snapshot = _build_run_health_snapshot(manifest)
        signal = _manifest_ops_signal(manifest, snapshot, reference_time=reference_time)
        candidates.append(
            {
                "manifest": manifest,
                "snapshot": snapshot,
                "signal": signal,
            }
        )

    latest_observed = max(candidates, key=lambda item: _manifest_sort_key(item["manifest"]))
    selected = max(candidates, key=lambda item: (item["signal"]["sort_key"], _manifest_sort_key(item["manifest"])))
    selection = _build_run_selection_summary(
        latest_observed_signal=latest_observed["signal"],
        selected_signal=selected["signal"],
    )
    return selected["manifest"], selected["snapshot"], selection


def _status_rank(status: str) -> int:
    normalized = str(status).strip().lower()
    if normalized == "healthy":
        return 0
    if normalized == "attention":
        return 1
    if normalized == "stale":
        return 2
    if normalized == "missing":
        return 3
    if normalized == "blocked":
        return 4
    return 1


def _latest_manifest_for_profiles(
    manifests: list[dict[str, object]],
    *,
    profiles: set[str],
) -> dict[str, object]:
    candidates = [
        manifest
        for manifest in manifests
        if str(manifest.get("profile", "")).strip().lower() in profiles and not _manifest_looks_like_smoke_run(manifest)
    ]
    return max(candidates, key=_manifest_sort_key, default={})


def _build_cadence_lane(
    *,
    manifests: list[dict[str, object]],
    lane: str,
    profiles: set[str],
    target_interval_hours: int,
    reference_time: datetime,
) -> dict[str, object]:
    manifest = _latest_manifest_for_profiles(manifests, profiles=profiles)
    if not manifest:
        return {
            "lane": lane,
            "profiles": sorted(profiles),
            "target_interval_hours": int(target_interval_hours),
            "status": "missing",
            "latest_run": {},
            "hours_since_run": float("nan"),
            "overdue_hours": float("nan"),
            "recommended_command": f"make schedule-run MODE={lane}",
            "summary": f"No recent `{lane}`-lane run was found. Run `make schedule-run MODE={lane}` to restore cadence.",
        }

    snapshot = _build_run_health_snapshot(manifest)
    signal = _manifest_ops_signal(manifest, snapshot, reference_time=reference_time)
    hours_since_run = _safe_float(signal.get("age_hours"))
    overdue_hours = max(hours_since_run - float(target_interval_hours), 0.0) if math.isfinite(hours_since_run) else float("nan")

    if not math.isfinite(hours_since_run):
        status = "attention"
        summary = f"The latest `{lane}`-lane run is missing a readable timestamp, so cadence could not be verified."
    elif hours_since_run > float(target_interval_hours) * 2.0:
        status = "stale"
        summary = (
            f"The `{lane}` lane is stale at `{hours_since_run:.1f}` hours since `{signal.get('run_id', '')}`. "
            f"Restore cadence with `make schedule-run MODE={lane}`."
        )
    elif hours_since_run > float(target_interval_hours):
        status = "attention"
        summary = (
            f"The `{lane}` lane is slipping at `{hours_since_run:.1f}` hours since `{signal.get('run_id', '')}`. "
            f"Plan `make schedule-run MODE={lane}` soon."
        )
    elif float(signal.get("coverage_ratio", 0.0)) < 0.8:
        status = "attention"
        summary = (
            f"The latest `{lane}`-lane run is recent, but ops coverage is only `{_format_metric(signal.get('coverage_ratio'))}`. "
            f"Backfill analysis before treating it as the cadence anchor."
        )
    else:
        status = "healthy"
        summary = (
            f"The `{lane}` lane is healthy with `{signal.get('run_id', '')}` at `{hours_since_run:.1f}` hours old "
            f"and coverage `{_format_metric(signal.get('coverage_ratio'))}`."
        )

    return {
        "lane": lane,
        "profiles": sorted(profiles),
        "target_interval_hours": int(target_interval_hours),
        "status": status,
        "latest_run": {
            "run_id": str(signal.get("run_id", "")),
            "profile": str(signal.get("profile", "")),
            "timestamp": str(signal.get("timestamp", "")),
            "coverage_ratio": _safe_float(signal.get("coverage_ratio")),
        },
        "hours_since_run": hours_since_run,
        "overdue_hours": overdue_hours,
        "recommended_command": f"make schedule-run MODE={lane}",
        "summary": summary,
    }


def _operating_lane(operating_rhythm: dict[str, object], lane: str) -> dict[str, object]:
    lanes = operating_rhythm.get("lanes", {})
    lanes = lanes if isinstance(lanes, dict) else {}
    lane_payload = lanes.get(lane, {})
    return lane_payload if isinstance(lane_payload, dict) else {}


def _build_operating_rhythm(
    *,
    manifests: list[dict[str, object]],
    latest_run: dict[str, object],
    run_selection: dict[str, object],
    reference_time: datetime,
) -> dict[str, object]:
    fast_lane = _build_cadence_lane(
        manifests=manifests,
        lane="fast",
        profiles={"fast", "small", "core", "experimental", "dev"},
        target_interval_hours=24,
        reference_time=reference_time,
    )
    full_lane = _build_cadence_lane(
        manifests=manifests,
        lane="full",
        profiles={"full"},
        target_interval_hours=24 * 7,
        reference_time=reference_time,
    )
    lane_statuses = [str(fast_lane.get("status", "")), str(full_lane.get("status", ""))]
    overall_status = max(lane_statuses, key=_status_rank, default="attention")

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


def _build_baseline_comparison(
    *,
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
    baseline_manifest: dict[str, object],
) -> dict[str, object]:
    if not baseline_manifest:
        return {
            "baseline_available": False,
            "comparison_mode": "latest_vs_last_strong_run",
            "summary": ["No prior promoted run is available yet, so future ops reviews will use the first successful promotion as the baseline."],
            "metric_deltas": [],
        }

    baseline_snapshot = _build_run_health_snapshot(baseline_manifest)
    baseline_run = baseline_snapshot["run"]
    baseline_safety = baseline_snapshot["safety"]
    baseline_qoe = baseline_snapshot["qoe"]

    metric_rows = [
        _metric_delta_row(
            key="best_model_test_top1",
            label="Best model test top1",
            current=latest_run.get("best_model_test_top1"),
            baseline=baseline_run.get("best_model_test_top1"),
            higher_is_better=True,
        ),
        _metric_delta_row(
            key="best_model_val_top1",
            label="Best model val top1",
            current=latest_run.get("best_model_val_top1"),
            baseline=baseline_run.get("best_model_val_top1"),
            higher_is_better=True,
        ),
        _metric_delta_row(
            key="target_drift_jsd",
            label="Target drift JSD",
            current=safety.get("test_jsd_target_drift"),
            baseline=baseline_safety.get("test_jsd_target_drift"),
            higher_is_better=False,
        ),
        _metric_delta_row(
            key="test_ece",
            label="Test ECE",
            current=safety.get("test_ece"),
            baseline=baseline_safety.get("test_ece"),
            higher_is_better=False,
        ),
        _metric_delta_row(
            key="test_selective_risk",
            label="Selective risk",
            current=safety.get("test_selective_risk"),
            baseline=baseline_safety.get("test_selective_risk"),
            higher_is_better=False,
        ),
        _metric_delta_row(
            key="robustness_gap",
            label="Worst robustness gap",
            current=safety.get("robustness_max_top1_gap"),
            baseline=baseline_safety.get("robustness_max_top1_gap"),
            higher_is_better=False,
        ),
        _metric_delta_row(
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
            f"{row['label']} {direction_word} from {_format_metric(row['baseline'])} to {_format_metric(row['current'])} (delta `{_format_metric(row['delta'])}`)."
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
    actions: list[dict[str, object]] = []

    coverage_ratio = _safe_float(ops_coverage.get("coverage_ratio"))
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
                    f"Latest run only has `{_safe_int(ops_coverage.get('available_summary_count'), default=0)}` of "
                    f"`{_safe_int(ops_coverage.get('expected_summary_count'), default=0)}` expected summaries. "
                    f"Missing: {preview}."
                ),
                "inspect": missing_summaries or ["outputs/runs/<run_id>/analysis/"],
            }
        )

    cadence_notes: list[str] = []
    fast_lane = _operating_lane(operating_rhythm, "fast")
    full_lane = _operating_lane(operating_rhythm, "full")
    for lane_payload in (fast_lane, full_lane):
        lane_status = str(lane_payload.get("status", ""))
        if lane_status not in {"missing", "stale"}:
            continue
        cadence_notes.append(str(lane_payload.get("summary", "")).strip())

    latest_observed = run_selection.get("latest_observed_run", {})
    latest_observed = latest_observed if isinstance(latest_observed, dict) else {}
    if bool(operating_rhythm.get("selection_gap")):
        cadence_notes.append(
            f"Latest observed run `{latest_observed.get('run_id', '')}` is newer than the ops-selected run "
            f"`{latest_run.get('run_id', '')}` because {run_selection.get('selection_reason', 'the newer run is not review-ready yet')}."
        )

    if cadence_notes:
        cadence_priority = "high" if str(full_lane.get("status", "")) in {"missing", "stale"} else "medium"
        actions.append(
            {
                "priority": cadence_priority,
                "area": "cadence",
                "title": "Restore the recurring run cadence",
                "detail": " ".join(note for note in cadence_notes if note),
                "inspect": ["outputs/analytics/control_room_history.csv", "scripts/run_scheduled.sh"],
            }
        )

    if not bool(latest_run.get("promoted")):
        regression = _safe_float(safety.get("champion_gate_regression"))
        detail = (
            f"Latest run failed promotion on {latest_run.get('promotion_status', 'unknown')} "
            f"with champion-gate regression `{_format_metric(regression)}`."
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

    robustness_gap = _safe_float(safety.get("robustness_max_top1_gap"))
    if math.isfinite(robustness_gap) and robustness_gap >= 0.15:
        actions.append(
            {
                "priority": "high",
                "area": "robustness",
                "title": "Harden the worst slice before the next full run",
                "detail": (
                    f"Worst robustness gap is `{_format_metric(robustness_gap)}` on "
                    f"{safety.get('robustness_worst_segment', 'segment')}={safety.get('robustness_worst_bucket', 'bucket')}."
                ),
                "inspect": ["analysis/robustness_summary.json"],
            }
        )

    target_drift = _safe_float(safety.get("test_jsd_target_drift"))
    if math.isfinite(target_drift) and target_drift >= 0.15:
        actions.append(
            {
                "priority": "medium",
                "area": "drift",
                "title": "Review drift before trusting regressions",
                "detail": (
                    f"Target drift JSD is `{_format_metric(target_drift)}` and segment shift peaks at "
                    f"`{_format_metric(safety.get('largest_segment_shift_value'))}` for {safety.get('largest_segment_shift_label', 'n/a')}."
                ),
                "inspect": ["analysis/data_drift_summary.json"],
            }
        )

    stress_skip_risk = _safe_float(qoe.get("stress_worst_skip_risk"))
    if math.isfinite(stress_skip_risk) and stress_skip_risk >= 0.35:
        actions.append(
            {
                "priority": "medium",
                "area": "stress_test",
                "title": "Promote the worst stress scenario into regression checks",
                "detail": (
                    f"Scenario `{qoe.get('stress_worst_skip_scenario', 'unknown')}` reaches skip risk "
                    f"`{_format_metric(stress_skip_risk)}` under the current safety route."
                ),
                "inspect": ["analysis/moonshot_summary.json", "analysis/stress_test/stress_test_summary.json"],
            }
        )

    selective_risk = _safe_float(safety.get("test_selective_risk"))
    abstention_rate = _safe_float(safety.get("test_abstention_rate"))
    if math.isfinite(selective_risk) and selective_risk >= 0.50 and (not math.isfinite(abstention_rate) or abstention_rate <= 0.01):
        actions.append(
            {
                "priority": "medium",
                "area": "uncertainty",
                "title": "Inspect abstention settings before serving broadly",
                "detail": (
                    f"Selective risk is `{_format_metric(selective_risk)}` while abstention is `{_format_metric(abstention_rate)}`."
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


def _build_async_handoff(
    *,
    latest_run: dict[str, object],
    review_actions: list[dict[str, object]],
    next_bets: list[str],
    operating_rhythm: dict[str, object],
    run_selection: dict[str, object],
    ops_coverage: dict[str, object],
) -> dict[str, object]:
    top_action = review_actions[0] if review_actions and isinstance(review_actions[0], dict) else {}
    top_priority = str(top_action.get("priority", "")).strip().lower()
    coverage_ratio = _safe_float(ops_coverage.get("coverage_ratio"))
    recommended_run_command = str(operating_rhythm.get("recommended_run_command", "")).strip()
    recommended_review_command = str(operating_rhythm.get("recommended_review_command", "make control-room")).strip()

    if math.isfinite(coverage_ratio) and coverage_ratio < 0.8:
        status = "blocked"
        headline = "Async review is blocked until the missing ops artifacts are backfilled."
    elif top_priority == "high":
        status = "attention"
        headline = "Async review can proceed, but high-priority actions should close before another full run."
    elif str(operating_rhythm.get("overall_status", "")) in {"missing", "stale", "attention"}:
        status = "attention"
        headline = "Async review is usable, but the recurring cadence still needs attention."
    else:
        status = "ready"
        headline = "Async review is ready; the current report is enough for a teammate handoff."

    latest_observed = run_selection.get("latest_observed_run", {})
    latest_observed = latest_observed if isinstance(latest_observed, dict) else {}
    summary = [
        f"Start with `{recommended_review_command}` and review ops-selected run `{latest_run.get('run_id', '')}` (`{latest_run.get('profile', '')}`).",
        f"Promotion is `{latest_run.get('promotion_status', '')}` and the top open action is `{top_action.get('title', 'none')}`.",
    ]
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
    if top_priority in {"high", "medium"}:
        share_artifacts.append("outputs/analytics/control_room_triage.md")

    return {
        "status": status,
        "headline": headline,
        "recommended_review_command": recommended_review_command,
        "recommended_run_command": recommended_run_command,
        "share_artifacts": share_artifacts,
        "summary": summary[:5],
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
        "robustness_gap": _safe_float(safety.get("robustness_max_top1_gap")),
        "stress_skip_risk": _safe_float(qoe.get("stress_worst_skip_risk")),
        "ops_coverage_ratio": _safe_float(ops_coverage.get("coverage_ratio")),
        "available_summary_count": _safe_int(ops_coverage.get("available_summary_count"), default=0),
        "expected_summary_count": _safe_int(ops_coverage.get("expected_summary_count"), default=0),
        "operating_status": str(operating_rhythm.get("overall_status", "")),
        "fast_cadence_status": str(_operating_lane(operating_rhythm, "fast").get("status", "")),
        "full_cadence_status": str(_operating_lane(operating_rhythm, "full").get("status", "")),
        "async_handoff_status": str(async_handoff.get("status", "")),
        "recommended_run_command": str(operating_rhythm.get("recommended_run_command", "")),
        "review_action_count": int(len(review_actions)),
        "high_priority_review_actions": int(high_count),
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
    current_run_id = str((report.get("latest_run", {}) if isinstance(report.get("latest_run", {}), dict) else {}).get("run_id", ""))
    current_row = frame[frame["run_id"].astype(str) == current_run_id].head(1)
    previous_rows = frame[frame["run_id"].astype(str) != current_run_id].head(1)
    recent_window = frame.head(min(5, len(frame.index))).copy()

    summary: list[str] = []
    metric_deltas: list[dict[str, object]] = []
    previous_snapshot: dict[str, object] = {}
    if not previous_rows.empty:
        previous = previous_rows.iloc[0].to_dict()
        previous_snapshot = {
            "run_id": str(previous.get("run_id", "")),
            "run_timestamp": str(previous.get("run_timestamp", "")),
            "profile": str(previous.get("profile", "")),
            "promotion_status": str(previous.get("promotion_status", "")),
        }
        current = current_row.iloc[0].to_dict() if not current_row.empty else {}
        metric_deltas = [
            _metric_delta_row(
                key="best_model_test_top1",
                label="Best model test top1",
                current=current.get("best_model_test_top1"),
                baseline=previous.get("best_model_test_top1"),
                higher_is_better=True,
            ),
            _metric_delta_row(
                key="robustness_gap",
                label="Worst robustness gap",
                current=current.get("robustness_gap"),
                baseline=previous.get("robustness_gap"),
                higher_is_better=False,
            ),
            _metric_delta_row(
                key="target_drift_jsd",
                label="Target drift JSD",
                current=current.get("target_drift_jsd"),
                baseline=previous.get("target_drift_jsd"),
                higher_is_better=False,
            ),
            _metric_delta_row(
                key="stress_skip_risk",
                label="Worst stress skip risk",
                current=current.get("stress_skip_risk"),
                baseline=previous.get("stress_skip_risk"),
                higher_is_better=False,
            ),
            _metric_delta_row(
                key="selective_risk",
                label="Selective risk",
                current=current.get("test_selective_risk"),
                baseline=previous.get("test_selective_risk"),
                higher_is_better=False,
            ),
        ]
        summary.append(
            f"Previous snapshot was `{previous_snapshot['run_id']}` at `{previous_snapshot['run_timestamp']}`."
        )
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
    summary.append(
        f"In the last `{len(recent_window.index)}` run snapshots, promotions passed `{promoted_count}` times and failed `{failed_promotions}` times."
    )
    if high_issue_runs > 0:
        summary.append(f"High-priority review actions appeared in `{high_issue_runs}` of the last `{len(recent_window.index)}` snapshots.")

    area_counts: dict[str, int] = {}
    for raw_value in recent_window.get("review_action_areas", pd.Series(dtype="object")).fillna("").astype(str):
        for area in [item for item in raw_value.split("|") if item]:
            area_counts[area] = int(area_counts.get(area, 0) + 1)
    recurring_areas = [f"{area} ({count})" for area, count in sorted(area_counts.items(), key=lambda item: (-item[1], item[0])) if count >= 2]
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
    if recurring_areas:
        recurring_labels = [f"{row['area']} ({row['count']})" for row in recurring_areas[:3]]
        summary_lines.append(
            f"Recurring ops areas this week: {', '.join(recurring_labels)}."
        )
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
        "fast_cadence_issue_snapshots": int(fast_cadence_issue_snapshots),
        "full_cadence_issue_snapshots": int(full_cadence_issue_snapshots),
        "recurring_areas": recurring_areas,
        "current_focus": current_focus,
        "summary": summary_lines,
        "window_runs": [
            {
                "run_id": str(row.get("run_id", "")),
                "run_timestamp": str(row.get("run_timestamp", "")),
                "promotion_status": str(row.get("promotion_status", "")),
                "best_model_name": str(row.get("best_model_name", "")),
                "best_model_test_top1": _safe_float(row.get("best_model_test_top1")),
                "robustness_gap": _safe_float(row.get("robustness_gap")),
                "stress_skip_risk": _safe_float(row.get("stress_skip_risk")),
            }
            for row in window_frame.to_dict(orient="records")
        ],
    }

    json_path = analytics_dir / "control_room_weekly_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Weekly Ops Summary",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Lookback days: `{payload['lookback_days']}`",
        f"- Snapshots considered: `{payload['snapshots_considered']}`",
        "",
        "## Summary",
        "",
    ]
    for item in payload["summary"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Operating Rhythm",
            "",
            f"- Async handoff blocked snapshots: `{payload['async_handoff_blocked_snapshots']}`",
            f"- Fast cadence issue snapshots: `{payload['fast_cadence_issue_snapshots']}`",
            f"- Full cadence issue snapshots: `{payload['full_cadence_issue_snapshots']}`",
        ]
    )
    lines.extend(["", "## Current Focus", ""])
    if current_focus:
        for item in current_focus:
            lines.append(f"- [{item['priority'].upper()}] {item['title']} ({item['area']})")
    else:
        lines.append("- No current review actions are open.")

    md_path = analytics_dir / "control_room_weekly_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, payload


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
        "leaderboards": {
            "experiment_top_models": _rank_models(experiment_history, metric_column="val_top1", top_n=top_n),
            "backtest_top_models": _rank_models(backtest_history, metric_column="top1", top_n=top_n),
        },
        "baseline_comparison": baseline_comparison,
        "review_actions": review_actions,
        "review_ritual": [
            "Open this control room first after every meaningful run.",
            "Compare the latest run to the last promoted baseline before interpreting regressions.",
            "Work through every high-priority review action before scheduling another full run.",
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
    )
    return report


def _format_metric(value) -> str:
    metric = _safe_float(value)
    if not math.isfinite(metric):
        return "n/a"
    return f"{metric:.3f}"


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
        "fast_cadence_issue_snapshots": _safe_int(weekly_payload.get("fast_cadence_issue_snapshots"), default=0),
        "full_cadence_issue_snapshots": _safe_int(weekly_payload.get("full_cadence_issue_snapshots"), default=0),
    }

    json_path = analytics_dir / "control_room.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    portfolio = report["portfolio"]
    latest_run = report["latest_run"]
    safety = report["safety"]
    qoe = report["qoe"]
    ops_coverage = report.get("ops_coverage", {})
    ops_coverage = ops_coverage if isinstance(ops_coverage, dict) else {}
    run_selection = report.get("run_selection", {})
    run_selection = run_selection if isinstance(run_selection, dict) else {}
    operating_rhythm = report.get("operating_rhythm", {})
    operating_rhythm = operating_rhythm if isinstance(operating_rhythm, dict) else {}
    async_handoff = report.get("async_handoff", {})
    async_handoff = async_handoff if isinstance(async_handoff, dict) else {}
    baseline = report.get("baseline_comparison", {})
    ops_history = report.get("ops_history", {})
    ops_history = ops_history if isinstance(ops_history, dict) else {}
    ops_trends = report.get("ops_trends", {})
    ops_trends = ops_trends if isinstance(ops_trends, dict) else {}
    weekly_summary = report.get("weekly_ops_summary", {})
    weekly_summary = weekly_summary if isinstance(weekly_summary, dict) else {}
    review_actions = report.get("review_actions", [])

    lines = [
        "# Control Room",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Output root: `{report['output_dir']}`",
        "",
        "## Portfolio",
        "",
        f"- Runs tracked: `{portfolio['total_runs']}`",
        f"- Promoted runs: `{portfolio['promoted_runs']}`",
        f"- Profiles seen: `{', '.join(portfolio['profiles_seen']) if portfolio['profiles_seen'] else 'n/a'}`",
        f"- Experiment history rows: `{portfolio['experiment_history_rows']}`",
        f"- Backtest history rows: `{portfolio['backtest_history_rows']}`",
        "",
        "## Run Selection",
        "",
        f"- Latest observed run: `{(run_selection.get('latest_observed_run', {}) if isinstance(run_selection.get('latest_observed_run', {}), dict) else {}).get('run_id', '')}` (`{(run_selection.get('latest_observed_run', {}) if isinstance(run_selection.get('latest_observed_run', {}), dict) else {}).get('profile', '')}`)",
        f"- Ops-selected run: `{(run_selection.get('selected_run', {}) if isinstance(run_selection.get('selected_run', {}), dict) else {}).get('run_id', '')}` (`{(run_selection.get('selected_run', {}) if isinstance(run_selection.get('selected_run', {}), dict) else {}).get('profile', '')}`)",
        f"- Selection mode: `{run_selection.get('selection_mode', '')}`",
        f"- Selection reason: {run_selection.get('selection_reason', 'n/a')}",
        "",
        "## Operating Rhythm",
        "",
        f"- Overall status: `{operating_rhythm.get('overall_status', '')}`",
        f"- Fast lane: `{_operating_lane(operating_rhythm, 'fast').get('status', '')}` latest=`{(_operating_lane(operating_rhythm, 'fast').get('latest_run', {}) if isinstance(_operating_lane(operating_rhythm, 'fast').get('latest_run', {}), dict) else {}).get('run_id', '')}` age_h=`{_format_metric(_operating_lane(operating_rhythm, 'fast').get('hours_since_run'))}`",
        f"- Full lane: `{_operating_lane(operating_rhythm, 'full').get('status', '')}` latest=`{(_operating_lane(operating_rhythm, 'full').get('latest_run', {}) if isinstance(_operating_lane(operating_rhythm, 'full').get('latest_run', {}), dict) else {}).get('run_id', '')}` age_h=`{_format_metric(_operating_lane(operating_rhythm, 'full').get('hours_since_run'))}`",
        f"- Recommended next run: `{operating_rhythm.get('recommended_run_command', '') or 'none'}`",
        "",
        "## Latest Run",
        "",
        f"- Run: `{latest_run['run_id']}` (`{latest_run['profile']}`)",
        f"- Timestamp: `{latest_run['timestamp']}`",
        f"- Promotion: `{latest_run['promotion_status']}`",
        f"- Best model: `{latest_run['best_model_name']}` [{latest_run['best_model_type']}] val_top1=`{_format_metric(latest_run['best_model_val_top1'])}` test_top1=`{_format_metric(latest_run['best_model_test_top1'])}`",
        f"- Champion alias: `{latest_run['champion_model_name']}` [{latest_run['champion_model_type']}]",
        "",
        "## Ops Coverage",
        "",
        f"- Available summaries: `{_safe_int(ops_coverage.get('available_summary_count'), default=0)}` / `{_safe_int(ops_coverage.get('expected_summary_count'), default=0)}`",
        f"- Coverage ratio: `{_format_metric(ops_coverage.get('coverage_ratio'))}`",
        f"- Missing summaries: `{', '.join(ops_coverage.get('missing_summaries', [])) if isinstance(ops_coverage.get('missing_summaries', []), list) and ops_coverage.get('missing_summaries', []) else 'none'}`",
        "",
        "## Safety",
        "",
        f"- Champion gate metric: `{safety['champion_gate_metric_source']}` regression=`{_format_metric(safety['champion_gate_regression'])}`",
        f"- Target drift (train->test JSD): `{_format_metric(safety['test_jsd_target_drift'])}`",
        f"- Largest context shift: `{safety['largest_context_shift_feature']}` value=`{_format_metric(safety['largest_context_shift_value'])}`",
        f"- Largest segment shift: `{safety['largest_segment_shift_label']}` value=`{_format_metric(safety['largest_segment_shift_value'])}`",
        f"- Worst robustness gap: `{safety['robustness_worst_model']}` {safety['robustness_worst_segment']}={safety['robustness_worst_bucket']} gap=`{_format_metric(safety['robustness_max_top1_gap'])}`",
        f"- Test ECE: `{_format_metric(safety['test_ece'])}` selective_risk=`{_format_metric(safety['test_selective_risk'])}` abstention=`{_format_metric(safety['test_abstention_rate'])}`",
        "",
        "## QoE",
        "",
        f"- Friction analysis: `{qoe['friction_status']}` with `{qoe['friction_feature_count']}` friction features",
        f"- Mean test skip-risk delta without friction: `{_format_metric(qoe['proxy_test_mean_delta'])}`",
        f"- Top friction feature: `{qoe['top_friction_feature']}` delta=`{_format_metric(qoe['top_friction_mean_risk_delta'])}`",
        f"- Digital twin test AUC: `{_format_metric(qoe['digital_twin_test_auc'])}` causal test AUC=`{_format_metric(qoe['causal_test_auc_total'])}`",
        f"- Stress scenario: `{qoe['stress_worst_skip_scenario']}` skip_risk=`{_format_metric(qoe['stress_worst_skip_risk'])}`",
        "",
        "## Since Last Strong Run",
        "",
    ]

    if isinstance(baseline, dict) and bool(baseline.get("baseline_available")):
        baseline_run = baseline.get("baseline_run", {})
        baseline_run = baseline_run if isinstance(baseline_run, dict) else {}
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
            f"- Snapshots tracked: `{_safe_int(ops_history.get('snapshot_count'), default=0)}`",
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
            f"- Snapshots considered: `{_safe_int(weekly_summary.get('snapshots_considered'), default=0)}` over `{_safe_int(weekly_summary.get('lookback_days'), default=7)}` day(s)",
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
        lines.append(
            f"- [{str(action.get('priority', '')).upper()}] {action.get('title', '')}: {detail}{inspect_text}"
        )

    lines.extend(
        [
            "",
            "## Review Ritual",
            "",
        ]
    )
    for step in report.get("review_ritual", []):
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

    for row in report["leaderboards"]["experiment_top_models"]:
        lines.append(
            f"- `{row['model_name']}` [{row['model_type']}] mean_val_top1=`{_format_metric(row['mean_metric'])}` best=`{_format_metric(row['best_metric'])}` runs=`{row['run_count']}`"
        )

    lines.extend(["", "### Backtest Top Models", ""])
    for row in report["leaderboards"]["backtest_top_models"]:
        lines.append(
            f"- `{row['model_name']}` [{row['model_type']}] mean_backtest_top1=`{_format_metric(row['mean_metric'])}` best=`{_format_metric(row['best_metric'])}` runs=`{row['run_count']}`"
        )

    lines.extend(["", "## Next Bets", ""])
    for bet in report["next_bets"]:
        lines.append(f"- {bet}")

    md_path = analytics_dir / "control_room.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
