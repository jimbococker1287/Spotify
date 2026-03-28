from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import math

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _safe_read_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _collect_run_manifests(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_manifest.json")):
        payload = _safe_read_json(path)
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row["run_dir"] = str(path.parent.resolve())
        rows.append(row)
    return rows


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
    promoted = sorted(
        (
            manifest
            for manifest in manifests
            if _is_promoted_manifest(manifest)
            and str(manifest.get("run_id", "")) != str(exclude_run_id or "")
        ),
        key=_manifest_sort_key,
        reverse=True,
    )
    return promoted[0] if promoted else {}


def _load_run_results(run_dir: Path) -> list[dict[str, object]]:
    payload = _safe_read_json(run_dir / "run_results.json")
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    return []


def _best_result_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    ranked = sorted(
        rows,
        key=lambda row: (
            _safe_float(row.get("val_top1")),
            _safe_float(row.get("test_top1")),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else {}


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

    payload = _safe_read_json(run_dir / "analysis" / f"{prefix}_{target_name}_confidence_summary.json")
    return payload if isinstance(payload, dict) else {}


def _rank_models(history_df: pd.DataFrame, *, metric_column: str, top_n: int) -> list[dict[str, object]]:
    if history_df.empty or "model_name" not in history_df.columns or metric_column not in history_df.columns:
        return []

    frame = history_df.copy()
    frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
    frame = frame[frame["model_name"].notna()].copy()
    frame = frame[frame["model_name"].astype(str).str.strip() != ""].copy()
    frame = frame[frame[metric_column].notna()].copy()
    if frame.empty:
        return []

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
    for _, row in grouped.head(max(1, int(top_n))).iterrows():
        rows.append(
            {
                "model_name": str(row["model_name"]),
                "model_type": str(row["model_type"]),
                "mean_metric": float(row["mean_metric"]),
                "best_metric": float(row["best_metric"]),
                "run_count": int(row["run_count"]),
            }
        )
    return rows


def _build_next_bets(
    *,
    portfolio: dict[str, object],
    latest_run: dict[str, object],
    safety: dict[str, object],
    qoe: dict[str, object],
) -> list[str]:
    bets: list[str] = []

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

    drift_summary = _safe_read_json(analysis_dir / "data_drift_summary.json") if analysis_dir is not None else {}
    friction_summary = _safe_read_json(analysis_dir / "friction_proxy_summary.json") if analysis_dir is not None else {}
    moonshot_summary = _safe_read_json(analysis_dir / "moonshot_summary.json") if analysis_dir is not None else {}
    robustness_summary = _safe_read_json(analysis_dir / "robustness_summary.json") if analysis_dir is not None else {}
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
    return {
        "run": run,
        "safety": safety,
        "qoe": qoe,
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
    baseline_comparison: dict[str, object],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []

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


def build_control_room_report(output_dir: Path, *, top_n: int = 5) -> dict[str, object]:
    output_root = output_dir.expanduser().resolve()
    manifests = _collect_run_manifests(output_root)
    latest_manifest = _latest_manifest(manifests)
    latest_snapshot = _build_run_health_snapshot(latest_manifest) if latest_manifest else {"run": {}, "safety": {}, "qoe": {}}

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
        baseline_comparison=baseline_comparison,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
        },
        "latest_run": latest_run,
        "safety": safety,
        "qoe": qoe,
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
    )
    return report


def _format_metric(value) -> str:
    metric = _safe_float(value)
    if not math.isfinite(metric):
        return "n/a"
    return f"{metric:.3f}"


def write_control_room_report(output_dir: Path, *, top_n: int = 5) -> tuple[Path, Path]:
    output_root = output_dir.expanduser().resolve()
    analytics_dir = output_root / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    report = build_control_room_report(output_root, top_n=top_n)

    json_path = analytics_dir / "control_room.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    portfolio = report["portfolio"]
    latest_run = report["latest_run"]
    safety = report["safety"]
    qoe = report["qoe"]
    baseline = report.get("baseline_comparison", {})
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
        "## Latest Run",
        "",
        f"- Run: `{latest_run['run_id']}` (`{latest_run['profile']}`)",
        f"- Timestamp: `{latest_run['timestamp']}`",
        f"- Promotion: `{latest_run['promotion_status']}`",
        f"- Best model: `{latest_run['best_model_name']}` [{latest_run['best_model_type']}] val_top1=`{_format_metric(latest_run['best_model_val_top1'])}` test_top1=`{_format_metric(latest_run['best_model_test_top1'])}`",
        f"- Champion alias: `{latest_run['champion_model_name']}` [{latest_run['champion_model_type']}]",
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
    report = build_control_room_report(output_dir, top_n=max(1, int(args.top_n)))
    json_path, md_path = write_control_room_report(output_dir, top_n=max(1, int(args.top_n)))

    latest_run = report["latest_run"]
    print(f"Control room written to {json_path}")
    print(f"Markdown summary written to {md_path}")
    print(f"Latest run: {latest_run['run_id']} ({latest_run['profile']})")
    print(f"Best model: {latest_run['best_model_name']} val_top1={_format_metric(latest_run['best_model_val_top1'])}")
    print("Next bets:")
    for bet in report["next_bets"]:
        print(f"- {bet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
