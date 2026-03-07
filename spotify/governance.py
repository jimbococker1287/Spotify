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


def _best_prior_from_experiment_history(
    history_csv: Path,
    current_run_id: str,
    metric_name: str,
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


def _no_current_result_payload(threshold: float, metric_source: str) -> dict[str, object]:
    return {
        "status": "no_current_results",
        "promoted": False,
        "metric_source": metric_source,
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
        challenger_model_name, challenger_score = _best_row_from_backtest_rows(backtest_rows)
        champion_run_id, champion_model_name, champion_score = _best_prior_from_backtest_history(
            backtest_history_csv,
            current_run_id,
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
        )

    if challenger_model_name == "" or challenger_score == float("-inf"):
        return _no_current_result_payload(threshold, effective_source)

    if champion_score == float("-inf"):
        return {
            "status": "no_prior_champion" + status_suffix,
            "promoted": True,
            "metric_source": effective_source,
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
        }

    regression = float(champion_score - challenger_score)
    promoted = regression <= threshold
    return {
        "status": ("pass" if promoted else "fail") + status_suffix,
        "promoted": promoted,
        "metric_source": effective_source,
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
    }
