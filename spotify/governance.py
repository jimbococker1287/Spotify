from __future__ import annotations

from pathlib import Path
import math

from .recommender_safety import evaluate_promotion_gate


def _to_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _map_platform_gate(
    gate: dict[str, object],
    *,
    metric_source: str,
) -> dict[str, object]:
    champion_score = _to_float(gate.get("champion_score"))
    challenger_score = _to_float(gate.get("challenger_score"))
    champion_std = _to_float(gate.get("champion_score_std"))
    challenger_std = _to_float(gate.get("challenger_score_std"))
    champion_count = _to_float(gate.get("champion_score_count"))
    challenger_count = _to_float(gate.get("challenger_score_count"))

    return {
        "status": str(gate.get("status", "")).strip(),
        "promoted": bool(gate.get("promoted", False)),
        "metric_source": metric_source,
        "profile_match": bool(gate.get("profile_match", False)),
        "require_significant_lift": bool(gate.get("require_significant_lift", False)),
        "significance_z": _to_float(gate.get("significance_z")),
        "significance_margin": _to_float(gate.get("significance_margin")),
        "significant_lift": gate.get("significant_lift"),
        "threshold": _to_float(gate.get("threshold")),
        "regression": _to_float(gate.get("regression")),
        "champion_run_id": str(gate.get("champion_run_id", "")).strip(),
        "champion_model_name": str(gate.get("champion_model_name", "")).strip(),
        "champion_score": champion_score,
        "challenger_model_name": str(gate.get("challenger_model_name", "")).strip(),
        "challenger_score": challenger_score,
        "champion_val_top1": (champion_score if metric_source == "val_top1" else float("nan")),
        "challenger_val_top1": (challenger_score if metric_source == "val_top1" else float("nan")),
        "champion_backtest_top1": (champion_score if metric_source == "backtest_top1" else float("nan")),
        "challenger_backtest_top1": (challenger_score if metric_source == "backtest_top1" else float("nan")),
        "champion_backtest_std": (champion_std if metric_source == "backtest_top1" else float("nan")),
        "challenger_backtest_std": (challenger_std if metric_source == "backtest_top1" else float("nan")),
        "champion_backtest_count": (champion_count if metric_source == "backtest_top1" else 0.0),
        "challenger_backtest_count": (challenger_count if metric_source == "backtest_top1" else 0.0),
        "max_selective_risk": _to_float(gate.get("max_selective_risk")),
        "max_abstention_rate": _to_float(gate.get("max_abstention_rate")),
        "challenger_selective_risk": _to_float(gate.get("challenger_selective_risk")),
        "challenger_abstention_rate": _to_float(gate.get("challenger_abstention_rate")),
        "selected_candidate_rank": int(_to_float(gate.get("selected_candidate_rank")) or 0),
        "eligible_candidate_count": int(_to_float(gate.get("eligible_candidate_count")) or 0),
        "challenger_selection_reason": str(gate.get("challenger_selection_reason", "")).strip(),
        "top_candidate_model_name": str(gate.get("top_candidate_model_name", "")).strip(),
        "top_candidate_score": _to_float(gate.get("top_candidate_score")),
        "top_candidate_risk_blockers": list(gate.get("top_candidate_risk_blockers", []))
        if isinstance(gate.get("top_candidate_risk_blockers", []), list)
        else [],
    }


def evaluate_champion_gate(
    *,
    history_csv: Path,
    current_run_id: str,
    current_results: list[dict[str, object]],
    regression_threshold: float,
    backtest_history_csv: Path | None = None,
    current_backtest_rows: list[dict[str, object]] | None = None,
    metric_source: str = "backtest_top1",
    current_profile: str | None = None,
    require_profile_match: bool = True,
    require_significant_lift: bool = False,
    significance_z: float = 1.96,
    current_risk_metrics: dict[str, dict[str, float]] | None = None,
    max_selective_risk: float | None = None,
    max_abstention_rate: float | None = None,
) -> dict[str, object]:
    threshold = max(0.0, float(regression_threshold))
    source = str(metric_source).strip().lower()
    if source not in ("backtest_top1", "val_top1"):
        source = "backtest_top1"

    effective_source = source
    status_suffix = ""

    if source == "backtest_top1":
        gate = evaluate_promotion_gate(
            history_csv=backtest_history_csv,
            current_run_id=current_run_id,
            current_rows=list(current_backtest_rows or []),
            metric_name="top1",
            regression_threshold=threshold,
            current_profile=current_profile,
            require_profile_match=require_profile_match,
            require_significant_lift=require_significant_lift,
            significance_z=significance_z,
            current_risk_metrics=current_risk_metrics,
            max_selective_risk=max_selective_risk,
            max_abstention_rate=max_abstention_rate,
        )
        if str(gate.get("status", "")).strip() == "no_current_results":
            effective_source = "val_top1"
            status_suffix = "_fallback_to_val"
        else:
            mapped = _map_platform_gate(gate, metric_source="backtest_top1")
            return mapped

    gate = evaluate_promotion_gate(
        history_csv=history_csv,
        current_run_id=current_run_id,
        current_rows=current_results,
        metric_name="val_top1",
        regression_threshold=threshold,
        current_profile=current_profile,
        require_profile_match=require_profile_match,
        require_significant_lift=require_significant_lift,
        significance_z=significance_z,
        current_risk_metrics=current_risk_metrics,
        max_selective_risk=max_selective_risk,
        max_abstention_rate=max_abstention_rate,
    )
    mapped = _map_platform_gate(gate, metric_source=effective_source)
    if status_suffix:
        mapped["status"] = str(mapped.get("status", "")).strip() + status_suffix
    return mapped
