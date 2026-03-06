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


def evaluate_champion_gate(
    *,
    history_csv: Path,
    current_run_id: str,
    current_results: list[dict[str, object]],
    regression_threshold: float,
) -> dict[str, object]:
    threshold = max(0.0, float(regression_threshold))

    current_best_row: dict[str, object] | None = None
    current_best = float("-inf")
    for row in current_results:
        score = _to_float(row.get("val_top1"))
        if math.isnan(score):
            continue
        if score > current_best:
            current_best = score
            current_best_row = row

    if current_best_row is None:
        return {
            "status": "no_current_results",
            "promoted": False,
            "threshold": threshold,
            "regression": float("nan"),
            "champion_run_id": "",
            "champion_model_name": "",
            "champion_val_top1": float("nan"),
            "challenger_model_name": "",
            "challenger_val_top1": float("nan"),
        }

    champion_run_id = ""
    champion_model_name = ""
    champion_score = float("-inf")
    if history_csv.exists():
        with history_csv.open("r", encoding="utf-8") as infile:
            for row in csv.DictReader(infile):
                run_id = str(row.get("run_id", "")).strip()
                if not run_id or run_id == current_run_id:
                    continue
                score = _to_float(row.get("val_top1"))
                if math.isnan(score):
                    continue
                if score > champion_score:
                    champion_score = score
                    champion_run_id = run_id
                    champion_model_name = str(row.get("model_name", ""))

    challenger_score = _to_float(current_best_row.get("val_top1"))
    challenger_model_name = str(current_best_row.get("model_name", ""))

    if champion_score == float("-inf"):
        return {
            "status": "no_prior_champion",
            "promoted": True,
            "threshold": threshold,
            "regression": 0.0,
            "champion_run_id": "",
            "champion_model_name": "",
            "champion_val_top1": float("nan"),
            "challenger_model_name": challenger_model_name,
            "challenger_val_top1": challenger_score,
        }

    regression = float(champion_score - challenger_score)
    promoted = regression <= threshold
    return {
        "status": ("pass" if promoted else "fail"),
        "promoted": promoted,
        "threshold": threshold,
        "regression": regression,
        "champion_run_id": champion_run_id,
        "champion_model_name": champion_model_name,
        "champion_val_top1": champion_score,
        "challenger_model_name": challenger_model_name,
        "challenger_val_top1": challenger_score,
    }
