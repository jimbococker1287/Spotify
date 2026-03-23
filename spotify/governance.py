from __future__ import annotations

from pathlib import Path
import csv
import math


def _to_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _best_row_by_metric(
    rows: list[dict[str, object]],
    metric_name: str,
) -> tuple[str, float]:
    best_model = ""
    best_score = float("-inf")
    for row in rows:
        model_name = str(row.get("model_name", "")).strip()
        if not model_name:
            continue
        score = _to_float(row.get(metric_name))
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_model = model_name
    return best_model, best_score


def _best_row_from_backtest_rows(
    rows: list[dict[str, object]],
) -> tuple[str, float]:
    by_model: dict[str, list[float]] = {}
    for row in rows:
        model_name = str(row.get("model_name", "")).strip()
        if not model_name:
            continue
        score = _to_float(row.get("top1"))
        if math.isnan(score):
            continue
        by_model.setdefault(model_name, []).append(score)

    best_model = ""
    best_score = float("-inf")
    for model_name, scores in by_model.items():
        if not scores:
            continue
        mean_score = float(sum(scores) / float(len(scores)))
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_name
    return best_model, best_score


def _scores_from_backtest_rows(rows: list[dict[str, object]]) -> dict[str, list[float]]:
    by_model: dict[str, list[float]] = {}
    for row in rows:
        model_name = str(row.get("model_name", "")).strip()
        if not model_name:
            continue
        score = _to_float(row.get("top1"))
        if math.isnan(score):
            continue
        by_model.setdefault(model_name, []).append(score)
    return by_model


def _best_prior_from_experiment_history(
    history_csv: Path,
    current_run_id: str,
    metric_name: str,
    *,
    target_profile: str | None,
    require_profile_match: bool,
) -> tuple[str, str, float]:
    champion_run_id = ""
    champion_model_name = ""
    champion_score = float("-inf")

    if not history_csv.exists():
        return champion_run_id, champion_model_name, champion_score

    with history_csv.open("r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            run_id = str(row.get("run_id", "")).strip()
            if not run_id or run_id == current_run_id:
                continue
            if require_profile_match and target_profile:
                profile = str(row.get("profile", "")).strip().lower()
                if profile and profile != str(target_profile).strip().lower():
                    continue
            model_name = str(row.get("model_name", "")).strip()
            if not model_name:
                continue
            score = _to_float(row.get(metric_name))
            if math.isnan(score):
                continue
            if score > champion_score:
                champion_score = score
                champion_run_id = run_id
                champion_model_name = model_name

    return champion_run_id, champion_model_name, champion_score


def _best_prior_from_backtest_history(
    history_csv: Path | None,
    current_run_id: str,
    *,
    target_profile: str | None,
    require_profile_match: bool,
) -> tuple[str, str, float]:
    if history_csv is None or not history_csv.exists():
        return "", "", float("-inf")

    by_run_model: dict[tuple[str, str], list[float]] = {}
    with history_csv.open("r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            run_id = str(row.get("run_id", "")).strip()
            model_name = str(row.get("model_name", "")).strip()
            if not run_id or run_id == current_run_id or not model_name:
                continue
            if require_profile_match and target_profile:
                profile = str(row.get("profile", "")).strip().lower()
                if profile and profile != str(target_profile).strip().lower():
                    continue
            score = _to_float(row.get("top1"))
            if math.isnan(score):
                continue
            by_run_model.setdefault((run_id, model_name), []).append(score)

    champion_run_id = ""
    champion_model_name = ""
    champion_score = float("-inf")
    for (run_id, model_name), scores in by_run_model.items():
        if not scores:
            continue
        mean_score = float(sum(scores) / float(len(scores)))
        if mean_score > champion_score:
            champion_score = mean_score
            champion_run_id = run_id
            champion_model_name = model_name

    return champion_run_id, champion_model_name, champion_score


def _scores_for_backtest_run_model(
    history_csv: Path | None,
    *,
    run_id: str,
    model_name: str,
    target_profile: str | None,
    require_profile_match: bool,
) -> list[float]:
    if history_csv is None or not history_csv.exists() or not run_id or not model_name:
        return []

    scores: list[float] = []
    with history_csv.open("r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            row_run_id = str(row.get("run_id", "")).strip()
            row_model_name = str(row.get("model_name", "")).strip()
            if row_run_id != run_id or row_model_name != model_name:
                continue
            if require_profile_match and target_profile:
                profile = str(row.get("profile", "")).strip().lower()
                if profile and profile != str(target_profile).strip().lower():
                    continue
            score = _to_float(row.get("top1"))
            if not math.isnan(score):
                scores.append(score)
    return scores


def _sample_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": float("nan"), "std": float("nan"), "count": 0.0, "stderr": float("nan")}
    count = float(len(scores))
    mean = float(sum(scores) / count)
    if len(scores) < 2:
        std = 0.0
    else:
        variance = sum((score - mean) ** 2 for score in scores) / float(len(scores) - 1)
        std = math.sqrt(max(0.0, variance))
    stderr = float(std / math.sqrt(count)) if count > 0 else float("nan")
    return {"mean": mean, "std": float(std), "count": count, "stderr": stderr}


def _no_current_result_payload(
    threshold: float,
    metric_source: str,
    *,
    profile_match: bool,
    require_significant_lift: bool = False,
    significance_z: float = 1.96,
    max_selective_risk: float | None = None,
    max_abstention_rate: float | None = None,
) -> dict[str, object]:
    return {
        "status": "no_current_results",
        "promoted": False,
        "metric_source": metric_source,
        "profile_match": bool(profile_match),
        "require_significant_lift": bool(require_significant_lift),
        "significance_z": float(significance_z),
        "significance_margin": float("nan"),
        "significant_lift": None,
        "threshold": threshold,
        "regression": float("nan"),
        "champion_run_id": "",
        "champion_model_name": "",
        "champion_score": float("nan"),
        "challenger_model_name": "",
        "challenger_score": float("nan"),
        "champion_val_top1": float("nan"),
        "challenger_val_top1": float("nan"),
        "champion_backtest_top1": float("nan"),
        "challenger_backtest_top1": float("nan"),
        "champion_backtest_std": float("nan"),
        "challenger_backtest_std": float("nan"),
        "champion_backtest_count": 0.0,
        "challenger_backtest_count": 0.0,
        "max_selective_risk": (float(max_selective_risk) if max_selective_risk is not None else float("nan")),
        "max_abstention_rate": (float(max_abstention_rate) if max_abstention_rate is not None else float("nan")),
        "challenger_selective_risk": float("nan"),
        "challenger_abstention_rate": float("nan"),
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

    challenger_model_name = ""
    challenger_score = float("-inf")
    champion_run_id = ""
    champion_model_name = ""
    champion_score = float("-inf")
    effective_source = source
    status_suffix = ""

    if source == "backtest_top1":
        backtest_rows = current_backtest_rows or []
        current_score_map = _scores_from_backtest_rows(backtest_rows)
        challenger_model_name, challenger_score = _best_row_from_backtest_rows(backtest_rows)
        champion_run_id, champion_model_name, champion_score = _best_prior_from_backtest_history(
            backtest_history_csv,
            current_run_id,
            target_profile=current_profile,
            require_profile_match=require_profile_match,
        )
        if challenger_model_name == "" or challenger_score == float("-inf"):
            # Fallback to val_top1 when backtest is unavailable in the current run.
            effective_source = "val_top1"
            status_suffix = "_fallback_to_val"

    if effective_source == "val_top1":
        challenger_model_name, challenger_score = _best_row_by_metric(current_results, "val_top1")
        champion_run_id, champion_model_name, champion_score = _best_prior_from_experiment_history(
            history_csv,
            current_run_id,
            "val_top1",
            target_profile=current_profile,
            require_profile_match=require_profile_match,
        )

    if challenger_model_name == "" or challenger_score == float("-inf"):
        return _no_current_result_payload(
            threshold,
            effective_source,
            profile_match=require_profile_match,
            require_significant_lift=require_significant_lift,
            significance_z=significance_z,
            max_selective_risk=max_selective_risk,
            max_abstention_rate=max_abstention_rate,
        )

    challenger_risk = (current_risk_metrics or {}).get(challenger_model_name, {})
    challenger_selective_risk = _to_float(challenger_risk.get("val_selective_risk"))
    challenger_abstention_rate = _to_float(challenger_risk.get("val_abstention_rate"))

    if champion_score == float("-inf"):
        status = "no_prior_champion" + status_suffix
        promoted = True
        if max_selective_risk is not None and not math.isnan(challenger_selective_risk) and challenger_selective_risk > float(max_selective_risk):
            promoted = False
            status = "fail_selective_risk" + status_suffix
        if max_abstention_rate is not None and not math.isnan(challenger_abstention_rate) and challenger_abstention_rate > float(max_abstention_rate):
            promoted = False
            status = "fail_abstention_rate" + status_suffix
        return {
            "status": status,
            "promoted": promoted,
            "metric_source": effective_source,
            "profile_match": bool(require_profile_match),
            "require_significant_lift": bool(require_significant_lift),
            "significance_z": float(significance_z),
            "significance_margin": float("nan"),
            "significant_lift": None,
            "threshold": threshold,
            "regression": 0.0,
            "champion_run_id": "",
            "champion_model_name": "",
            "champion_score": float("nan"),
            "challenger_model_name": challenger_model_name,
            "challenger_score": challenger_score,
            "champion_val_top1": (float("nan") if effective_source != "val_top1" else float("nan")),
            "challenger_val_top1": (challenger_score if effective_source == "val_top1" else float("nan")),
            "champion_backtest_top1": (float("nan") if effective_source != "backtest_top1" else float("nan")),
            "challenger_backtest_top1": (challenger_score if effective_source == "backtest_top1" else float("nan")),
            "champion_backtest_std": float("nan"),
            "challenger_backtest_std": float("nan"),
            "champion_backtest_count": 0.0,
            "challenger_backtest_count": 0.0,
            "max_selective_risk": (float(max_selective_risk) if max_selective_risk is not None else float("nan")),
            "max_abstention_rate": (float(max_abstention_rate) if max_abstention_rate is not None else float("nan")),
            "challenger_selective_risk": challenger_selective_risk,
            "challenger_abstention_rate": challenger_abstention_rate,
        }

    regression = float(champion_score - challenger_score)
    promoted = regression <= threshold
    challenger_stats = {"mean": float("nan"), "std": float("nan"), "count": 0.0, "stderr": float("nan")}
    champion_stats = {"mean": float("nan"), "std": float("nan"), "count": 0.0, "stderr": float("nan")}
    significance_margin = float("nan")
    significant_lift = None
    status = ("pass" if promoted else "fail") + status_suffix

    if effective_source == "backtest_top1":
        challenger_scores = current_score_map.get(challenger_model_name, []) if "current_score_map" in locals() else []
        champion_scores = _scores_for_backtest_run_model(
            backtest_history_csv,
            run_id=champion_run_id,
            model_name=champion_model_name,
            target_profile=current_profile,
            require_profile_match=require_profile_match,
        )
        challenger_stats = _sample_stats(challenger_scores)
        champion_stats = _sample_stats(champion_scores)
        if require_significant_lift and not math.isnan(challenger_stats["stderr"]) and not math.isnan(champion_stats["stderr"]):
            combined_stderr = math.sqrt(
                max(0.0, (challenger_stats["stderr"] ** 2) + (champion_stats["stderr"] ** 2))
            )
            significance_margin = float(max(0.0, significance_z) * combined_stderr)
            lift = challenger_score - champion_score
            significant_lift = bool(lift > significance_margin)
            if promoted and lift > 0 and not significant_lift:
                promoted = False
                status = "fail_not_significant" + status_suffix

    if promoted and max_selective_risk is not None and not math.isnan(challenger_selective_risk):
        if challenger_selective_risk > float(max_selective_risk):
            promoted = False
            status = "fail_selective_risk" + status_suffix
    if promoted and max_abstention_rate is not None and not math.isnan(challenger_abstention_rate):
        if challenger_abstention_rate > float(max_abstention_rate):
            promoted = False
            status = "fail_abstention_rate" + status_suffix

    return {
        "status": status,
        "promoted": promoted,
        "metric_source": effective_source,
        "profile_match": bool(require_profile_match),
        "require_significant_lift": bool(require_significant_lift),
        "significance_z": float(significance_z),
        "significance_margin": significance_margin,
        "significant_lift": significant_lift,
        "threshold": threshold,
        "regression": regression,
        "champion_run_id": champion_run_id,
        "champion_model_name": champion_model_name,
        "champion_score": champion_score,
        "challenger_model_name": challenger_model_name,
        "challenger_score": challenger_score,
        "champion_val_top1": (champion_score if effective_source == "val_top1" else float("nan")),
        "challenger_val_top1": (challenger_score if effective_source == "val_top1" else float("nan")),
        "champion_backtest_top1": (champion_score if effective_source == "backtest_top1" else float("nan")),
        "challenger_backtest_top1": (challenger_score if effective_source == "backtest_top1" else float("nan")),
        "champion_backtest_std": champion_stats["std"],
        "challenger_backtest_std": challenger_stats["std"],
        "champion_backtest_count": champion_stats["count"],
        "challenger_backtest_count": challenger_stats["count"],
        "max_selective_risk": (float(max_selective_risk) if max_selective_risk is not None else float("nan")),
        "max_abstention_rate": (float(max_abstention_rate) if max_abstention_rate is not None else float("nan")),
        "challenger_selective_risk": challenger_selective_risk,
        "challenger_abstention_rate": challenger_abstention_rate,
    }
