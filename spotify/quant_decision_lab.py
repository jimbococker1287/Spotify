from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .run_artifacts import latest_manifest_run_dir
from .run_artifacts import safe_read_json
from .run_artifacts import write_csv_rows
from .run_artifacts import write_json
from .run_artifacts import write_markdown


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _rows_for_columns(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    trimmed = frame.copy()
    for column in columns:
        if column not in trimmed.columns:
            trimmed[column] = None
    return trimmed[columns].to_dict(orient="records")


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _native_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _native_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _normalize_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.where(numeric.notna(), 0.5, np.nan), index=series.index, dtype="float64")
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.where(numeric.notna(), 0.5, np.nan), index=series.index, dtype="float64")
    scaled = (numeric - min_value) / (max_value - min_value)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.astype("float64", copy=False)


def _bounded_unit_score(value: object, *, default: float = 0.0) -> float:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return default
    return min(1.0, max(0.0, numeric))


def _join_unique(items: list[object] | tuple[object, ...] | set[object]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        values.append(text)
        seen.add(text)
    return " | ".join(values)


def _extract_selected_run_id(output_dir: Path) -> str:
    control_room = safe_read_json(output_dir / "analytics" / "control_room.json", default={})
    if isinstance(control_room, dict):
        run_selection = control_room.get("run_selection", {})
        if isinstance(run_selection, dict):
            selected = run_selection.get("selected_run")
            if isinstance(selected, dict):
                run_id = str(selected.get("run_id", "")).strip()
                if run_id:
                    return run_id
            run_id = str(selected or "").strip()
            if run_id:
                return run_id
    latest_dir = latest_manifest_run_dir(output_dir)
    return latest_dir.name if latest_dir is not None else ""


def _resolve_run_dir(*, output_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    selected_run_id = _extract_selected_run_id(output_dir)
    if selected_run_id:
        candidate = (output_dir / "runs" / selected_run_id).resolve()
        if candidate.exists():
            return candidate
    latest_dir = latest_manifest_run_dir(output_dir)
    if latest_dir is not None:
        return latest_dir
    raise FileNotFoundError("No completed run could be resolved for quant decision analysis.")


def _collect_conformal_metrics(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((run_dir / "analysis").glob("*_conformal_summary.json")):
        payload = safe_read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        test = payload.get("test", {}) if isinstance(payload.get("test"), dict) else {}
        val = payload.get("val", {}) if isinstance(payload.get("val"), dict) else {}
        operating_point = payload.get("operating_point", {}) if isinstance(payload.get("operating_point"), dict) else {}
        tag = str(payload.get("tag", "")).strip()
        model_name = tag
        if model_name.startswith("ensemble_"):
            model_name = model_name.removeprefix("ensemble_")
        elif model_name.startswith("classical_"):
            model_name = model_name.removeprefix("classical_")
        elif model_name.startswith("deep_"):
            model_name = model_name.removeprefix("deep_")
        elif model_name.startswith("retrieval_") and "_retrieval_" in model_name:
            model_name = model_name.split("_", 1)[1]
        if model_name.endswith("_retrieval_reranker"):
            model_name = "retrieval_reranker"
        rows.append(
            {
                "model_name": model_name,
                "conformal_tag": tag,
                "test_selective_risk": _safe_float(test.get("selective_risk")),
                "test_abstention_rate": _safe_float(test.get("abstention_rate")),
                "test_accepted_rate": _safe_float(test.get("accepted_rate")),
                "val_selective_risk": _safe_float(val.get("selective_risk")),
                "val_abstention_rate": _safe_float(val.get("abstention_rate")),
                "operating_abstention_threshold": _safe_float(operating_point.get("abstention_threshold")),
                "target_selective_risk": _safe_float(operating_point.get("target_selective_risk")),
                "min_accepted_rate": _safe_float(operating_point.get("min_accepted_rate")),
            }
        )
    return pd.DataFrame(rows)


def _build_model_frontier(run_dir: Path) -> pd.DataFrame:
    run_results = safe_read_json(run_dir / "run_results.json", default=[])
    policy_rows = safe_read_json(run_dir / "analysis" / "policy_simulation_summary.json", default=[])
    if not isinstance(run_results, list):
        run_results = []
    if not isinstance(policy_rows, list):
        policy_rows = []
    results_df = pd.DataFrame(run_results)
    policy_df = pd.DataFrame(policy_rows)
    if results_df.empty:
        return pd.DataFrame()
    conformal_df = _collect_conformal_metrics(run_dir)

    model_frame = results_df.copy()
    policy_columns = ["model_name", "test_hit_at_k", "test_discounted_reward", "test_expected_utility_mass"]
    if not policy_df.empty:
        for column in policy_columns:
            if column not in policy_df.columns:
                policy_df[column] = None
        model_frame = model_frame.merge(policy_df[policy_columns], on="model_name", how="left")
    if not conformal_df.empty:
        model_frame = model_frame.merge(conformal_df, on="model_name", how="left")

    numeric_columns = [
        "test_top1",
        "val_top1",
        "fit_seconds",
        "test_hit_at_k",
        "test_discounted_reward",
        "test_expected_utility_mass",
        "test_selective_risk",
        "test_abstention_rate",
        "test_accepted_rate",
    ]
    for column in numeric_columns:
        model_frame[column] = pd.to_numeric(model_frame.get(column), errors="coerce")

    model_frame["quality_norm"] = _normalize_series(model_frame["test_top1"], higher_is_better=True)
    model_frame["reward_norm"] = _normalize_series(model_frame["test_discounted_reward"], higher_is_better=True)
    model_frame["hit_norm"] = _normalize_series(model_frame["test_hit_at_k"], higher_is_better=True)
    model_frame["accepted_norm"] = _normalize_series(model_frame["test_accepted_rate"], higher_is_better=True)
    model_frame["risk_norm"] = _normalize_series(model_frame["test_selective_risk"], higher_is_better=False)
    model_frame["speed_norm"] = _normalize_series(model_frame["fit_seconds"], higher_is_better=False)
    score_columns = ["quality_norm", "reward_norm", "hit_norm", "accepted_norm", "risk_norm", "speed_norm"]
    model_frame["score_inputs"] = model_frame[score_columns].notna().sum(axis=1)
    model_frame["utility_score"] = (
        (0.30 * model_frame["quality_norm"].fillna(0.0))
        + (0.25 * model_frame["reward_norm"].fillna(0.0))
        + (0.15 * model_frame["hit_norm"].fillna(0.0))
        + (0.10 * model_frame["accepted_norm"].fillna(0.0))
        + (0.15 * model_frame["risk_norm"].fillna(0.0))
        + (0.05 * model_frame["speed_norm"].fillna(0.0))
    )
    model_frame["utility_score"] = np.where(
        model_frame["score_inputs"] > 0,
        model_frame["utility_score"] * (model_frame["score_inputs"] / len(score_columns)),
        np.nan,
    )
    model_frame["risk_complete"] = model_frame["test_selective_risk"].notna() & model_frame["test_accepted_rate"].notna()

    valid = model_frame[
        ["test_top1", "test_discounted_reward", "test_accepted_rate", "test_selective_risk", "fit_seconds"]
    ].notna().all(axis=1)
    frontier_flags = []
    dominance_counts = []
    for idx, row in model_frame.iterrows():
        if not bool(valid.loc[idx]):
            frontier_flags.append(False)
            dominance_counts.append(0)
            continue
        dominated = False
        dominates = 0
        for other_idx, other in model_frame.loc[valid].iterrows():
            if other_idx == idx:
                continue
            other_better_or_equal = (
                other["test_top1"] >= row["test_top1"]
                and other["test_discounted_reward"] >= row["test_discounted_reward"]
                and other["test_accepted_rate"] >= row["test_accepted_rate"]
                and other["test_selective_risk"] <= row["test_selective_risk"]
                and other["fit_seconds"] <= row["fit_seconds"]
            )
            other_strict = (
                other["test_top1"] > row["test_top1"]
                or other["test_discounted_reward"] > row["test_discounted_reward"]
                or other["test_accepted_rate"] > row["test_accepted_rate"]
                or other["test_selective_risk"] < row["test_selective_risk"]
                or other["fit_seconds"] < row["fit_seconds"]
            )
            if other_better_or_equal and other_strict:
                dominated = True
            row_better_or_equal = (
                row["test_top1"] >= other["test_top1"]
                and row["test_discounted_reward"] >= other["test_discounted_reward"]
                and row["test_accepted_rate"] >= other["test_accepted_rate"]
                and row["test_selective_risk"] <= other["test_selective_risk"]
                and row["fit_seconds"] <= other["fit_seconds"]
            )
            row_strict = (
                row["test_top1"] > other["test_top1"]
                or row["test_discounted_reward"] > other["test_discounted_reward"]
                or row["test_accepted_rate"] > other["test_accepted_rate"]
                or row["test_selective_risk"] < other["test_selective_risk"]
                or row["fit_seconds"] < other["fit_seconds"]
            )
            if row_better_or_equal and row_strict:
                dominates += 1
        frontier_flags.append(not dominated)
        dominance_counts.append(dominates)
    model_frame["is_pareto_efficient"] = frontier_flags
    model_frame["dominates_count"] = dominance_counts
    model_frame["frontier_status"] = np.where(model_frame["is_pareto_efficient"], "efficient", "dominated")
    return model_frame.sort_values(
        ["is_pareto_efficient", "utility_score", "test_top1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_policy_frontier(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = safe_read_json(run_dir / "analysis" / "stress_test" / "stress_test_summary.json", default=[])
    benchmark = safe_read_json(run_dir / "analysis" / "stress_test" / "stress_test_benchmark.json", default={})
    if not isinstance(rows, list):
        rows = []
    if not isinstance(benchmark, dict):
        benchmark = {}
    stress_df = pd.DataFrame(rows)
    if stress_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for column in ["mean_session_length", "mean_skip_risk", "mean_end_risk"]:
        stress_df[column] = pd.to_numeric(stress_df.get(column), errors="coerce")

    policy_frontier = (
        stress_df.groupby(["policy_name", "policy_family"], dropna=False)
        .agg(
            scenario_count=("scenario", "count"),
            mean_session_length=("mean_session_length", "mean"),
            mean_skip_risk=("mean_skip_risk", "mean"),
            worst_skip_risk=("mean_skip_risk", "max"),
            mean_end_risk=("mean_end_risk", "mean"),
        )
        .reset_index()
    )
    policy_frontier["length_norm"] = _normalize_series(policy_frontier["mean_session_length"], higher_is_better=True)
    policy_frontier["skip_norm"] = _normalize_series(policy_frontier["mean_skip_risk"], higher_is_better=False)
    policy_frontier["worst_skip_norm"] = _normalize_series(policy_frontier["worst_skip_risk"], higher_is_better=False)
    policy_frontier["end_norm"] = _normalize_series(policy_frontier["mean_end_risk"], higher_is_better=False)
    policy_frontier["utility_score"] = (
        (0.30 * policy_frontier["length_norm"].fillna(0.0))
        + (0.30 * policy_frontier["skip_norm"].fillna(0.0))
        + (0.25 * policy_frontier["worst_skip_norm"].fillna(0.0))
        + (0.15 * policy_frontier["end_norm"].fillna(0.0))
    )
    policy_frontier["is_benchmark_selected"] = (
        policy_frontier["policy_name"].astype(str)
        == str(benchmark.get("benchmark_selected_policy_name", "")).strip()
    )
    policy_frontier["is_benchmark_canonical"] = (
        policy_frontier["policy_name"].astype(str)
        == str(benchmark.get("benchmark_policy_name", "")).strip()
    )

    frontier_flags = []
    for idx, row in policy_frontier.iterrows():
        dominated = False
        for other_idx, other in policy_frontier.iterrows():
            if other_idx == idx:
                continue
            better_or_equal = (
                other["mean_session_length"] >= row["mean_session_length"]
                and other["mean_skip_risk"] <= row["mean_skip_risk"]
                and other["worst_skip_risk"] <= row["worst_skip_risk"]
                and other["mean_end_risk"] <= row["mean_end_risk"]
            )
            strict = (
                other["mean_session_length"] > row["mean_session_length"]
                or other["mean_skip_risk"] < row["mean_skip_risk"]
                or other["worst_skip_risk"] < row["worst_skip_risk"]
                or other["mean_end_risk"] < row["mean_end_risk"]
            )
            if better_or_equal and strict:
                dominated = True
                break
        frontier_flags.append(not dominated)
    policy_frontier["is_pareto_efficient"] = frontier_flags
    policy_frontier["frontier_status"] = np.where(policy_frontier["is_pareto_efficient"], "efficient", "dominated")
    policy_frontier = policy_frontier.sort_values(
        ["is_pareto_efficient", "utility_score", "mean_skip_risk"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    baseline_rows = stress_df.loc[stress_df["policy_name"].astype(str) == "baseline_exploit"].copy()
    safe_rows = stress_df.loc[stress_df["policy_family"].astype(str) == "safe"].copy()
    baseline_reference = (
        baseline_rows.loc[baseline_rows["scenario"].astype(str) == "baseline"].iloc[0].to_dict()
        if not baseline_rows.loc[baseline_rows["scenario"].astype(str) == "baseline"].empty
        else {}
    )
    safe_baseline_reference = (
        safe_rows.sort_values(["scenario", "mean_skip_risk"]).loc[safe_rows["scenario"].astype(str) == "baseline"].iloc[0].to_dict()
        if not safe_rows.loc[safe_rows["scenario"].astype(str) == "baseline"].empty
        else {}
    )
    sensitivity_rows: list[dict[str, object]] = []
    for scenario_name in sorted(stress_df["scenario"].dropna().astype(str).unique()):
        baseline_row = (
            baseline_rows.loc[baseline_rows["scenario"].astype(str) == scenario_name].iloc[0].to_dict()
            if not baseline_rows.loc[baseline_rows["scenario"].astype(str) == scenario_name].empty
            else {}
        )
        safe_slice = safe_rows.loc[safe_rows["scenario"].astype(str) == scenario_name].copy()
        best_safe_row = (
            safe_slice.sort_values(["mean_skip_risk", "mean_end_risk"]).iloc[0].to_dict() if not safe_slice.empty else {}
        )
        baseline_skip = _safe_float(baseline_row.get("mean_skip_risk"))
        baseline_end = _safe_float(baseline_row.get("mean_end_risk"))
        best_safe_skip = _safe_float(best_safe_row.get("mean_skip_risk"))
        best_safe_end = _safe_float(best_safe_row.get("mean_end_risk"))
        baseline_reference_skip = _safe_float(baseline_reference.get("mean_skip_risk"))
        baseline_reference_end = _safe_float(baseline_reference.get("mean_end_risk"))
        safe_reference_skip = _safe_float(safe_baseline_reference.get("mean_skip_risk"))
        pressure_score = 0.0
        if math.isfinite(baseline_skip) and math.isfinite(baseline_reference_skip):
            pressure_score += max(0.0, baseline_skip - baseline_reference_skip)
        if math.isfinite(baseline_end) and math.isfinite(baseline_reference_end):
            pressure_score += 0.5 * max(0.0, baseline_end - baseline_reference_end)
        if math.isfinite(best_safe_skip) and math.isfinite(baseline_skip):
            pressure_score += max(0.0, best_safe_skip - baseline_skip)
        if math.isfinite(best_safe_skip) and math.isfinite(safe_reference_skip):
            pressure_score += 0.75 * max(0.0, best_safe_skip - safe_reference_skip)
        sensitivity_rows.append(
            {
                "scenario": scenario_name,
                "baseline_skip_risk": baseline_skip,
                "baseline_end_risk": baseline_end,
                "baseline_skip_delta_vs_baseline_scenario": (
                    baseline_skip - baseline_reference_skip
                    if math.isfinite(baseline_skip) and math.isfinite(baseline_reference_skip)
                    else float("nan")
                ),
                "baseline_end_delta_vs_baseline_scenario": (
                    baseline_end - baseline_reference_end
                    if math.isfinite(baseline_end) and math.isfinite(baseline_reference_end)
                    else float("nan")
                ),
                "best_safe_policy_name": best_safe_row.get("policy_name"),
                "best_safe_skip_risk": best_safe_skip,
                "best_safe_end_risk": best_safe_end,
                "safe_skip_improvement_vs_baseline": (
                    baseline_skip - best_safe_skip
                    if math.isfinite(baseline_skip) and math.isfinite(best_safe_skip)
                    else float("nan")
                ),
                "pressure_score": pressure_score,
            }
        )
    scenario_sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["pressure_score", "baseline_skip_risk"], ascending=[False, False]
    ).reset_index(drop=True)
    return policy_frontier, scenario_sensitivity


def _listener_archetype_context(output_dir: Path) -> dict[str, Any]:
    output_root = output_dir / "analysis" / "listener_archetypes"
    brief = safe_read_json(output_root / "taste_state_brief.json", default={})
    summary = safe_read_json(output_root / "listener_archetype_summary.json", default={})
    evolution = safe_read_json(output_root / "taste_evolution_brief.json", default={})
    regime_shifts = safe_read_json(output_root / "taste_evolution_regime_shifts.json", default=[])
    seasonal = safe_read_json(output_root / "listener_archetype_seasonal.json", default=[])
    brief_payload = brief if isinstance(brief, dict) else {}
    summary_payload = summary if isinstance(summary, dict) else {}
    evolution_payload = evolution if isinstance(evolution, dict) else {}
    archetype_rows = summary_payload.get("archetypes", [])
    archetype_lookup: dict[str, dict[str, object]] = {}
    if isinstance(archetype_rows, list):
        for row in archetype_rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("archetype_label", "")).strip()
            if label:
                archetype_lookup[label] = _native_value(row)

    dominant_label = str(brief_payload.get("dominant_archetype", "")).strip()
    highest_skip_label = str(brief_payload.get("highest_skip_archetype", "")).strip()
    exploratory_label = str(brief_payload.get("highest_exploration_archetype", "")).strip()
    regime_rows = [_native_value(row) for row in regime_shifts if isinstance(row, dict)] if isinstance(regime_shifts, list) else []
    seasonal_rows = [_native_value(row) for row in seasonal if isinstance(row, dict)] if isinstance(seasonal, list) else []
    lifecycle_available = bool(
        (str(evolution_payload.get("status", "")).strip().lower() == "ok")
        or regime_rows
        or seasonal_rows
    )
    available = bool(dominant_label or highest_skip_label or exploratory_label)
    return {
        "status": "ok" if available else "missing",
        "available": available,
        "brief": brief_payload,
        "summary": summary_payload,
        "archetype_lookup": archetype_lookup,
        "dominant_archetype": dominant_label,
        "highest_skip_archetype": highest_skip_label,
        "highest_exploration_archetype": exploratory_label,
        "lifecycle": {
            "available": lifecycle_available,
            "evolution_brief": evolution_payload,
            "regime_shifts": regime_rows,
            "seasonal": seasonal_rows,
        },
    }


def _prefer_efficient_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "is_pareto_efficient" not in frame.columns:
        return frame.copy()
    efficient_mask = frame["is_pareto_efficient"].fillna(False).astype(bool)
    if bool(efficient_mask.any()):
        return frame.loc[efficient_mask].copy()
    return frame.copy()


def _compact_record(row: dict[str, object], columns: list[str], *, selection_basis: str) -> dict[str, object]:
    payload = {column: _native_value(row.get(column)) for column in columns}
    payload["selection_basis"] = selection_basis
    return payload


def _archetype_lifecycle_profile(archetype_label: str, listener_context: dict[str, Any]) -> dict[str, object]:
    label = str(archetype_label).strip()
    lifecycle = listener_context.get("lifecycle", {})
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
    if not label or not bool(lifecycle.get("available")):
        return {
            "available": False,
            "archetype_label": label,
            "regime_shift_months": [],
            "dominant_months": [],
            "top_share_gain_months": [],
            "top_share_loss_months": [],
            "biggest_regime_shift_month": "",
            "biggest_regime_shift_role": "",
            "is_most_seasonal_archetype": False,
            "peak_season": "",
            "peak_season_label": "",
            "peak_season_share": None,
            "seasonality_gap": None,
            "active_seasons": [],
            "top_cross_state_transition_label": "",
            "top_cross_state_transition_involved": False,
        }

    evolution_brief = lifecycle.get("evolution_brief", {})
    evolution_brief = evolution_brief if isinstance(evolution_brief, dict) else {}
    regime_rows = lifecycle.get("regime_shifts", [])
    regime_rows = regime_rows if isinstance(regime_rows, list) else []
    seasonal_rows = lifecycle.get("seasonal", [])
    seasonal_rows = seasonal_rows if isinstance(seasonal_rows, list) else []

    regime_shift_months: list[str] = []
    dominant_months: list[str] = []
    top_share_gain_months: list[str] = []
    top_share_loss_months: list[str] = []
    for row in regime_rows:
        if not isinstance(row, dict):
            continue
        month = str(row.get("month", "")).strip()
        if not month:
            continue
        score = _safe_float(row.get("regime_shift_score"))
        dominant = str(row.get("dominant_archetype", "")).strip()
        previous = str(row.get("previous_dominant_archetype", "")).strip()
        gain = str(row.get("top_share_gain_archetype", "")).strip()
        loss = str(row.get("top_share_loss_archetype", "")).strip()
        if dominant == label:
            dominant_months.append(month)
        if gain == label:
            top_share_gain_months.append(month)
        if loss == label:
            top_share_loss_months.append(month)
        if math.isfinite(score) and score > 0 and label in {dominant, previous, gain, loss}:
            regime_shift_months.append(month)

    biggest_regime_shift = evolution_brief.get("biggest_regime_shift", {})
    biggest_regime_shift = biggest_regime_shift if isinstance(biggest_regime_shift, dict) else {}
    biggest_regime_shift_month = str(biggest_regime_shift.get("month", "")).strip()
    biggest_regime_shift_role = ""
    if str(biggest_regime_shift.get("dominant_archetype", "")).strip() == label:
        biggest_regime_shift_role = "dominant_archetype"
    elif str(biggest_regime_shift.get("previous_dominant_archetype", "")).strip() == label:
        biggest_regime_shift_role = "previous_dominant_archetype"
    elif str(biggest_regime_shift.get("top_share_gain_archetype", "")).strip() == label:
        biggest_regime_shift_role = "top_share_gain_archetype"
    elif str(biggest_regime_shift.get("top_share_loss_archetype", "")).strip() == label:
        biggest_regime_shift_role = "top_share_loss_archetype"

    most_seasonal = evolution_brief.get("most_seasonal_archetype", {})
    most_seasonal = most_seasonal if isinstance(most_seasonal, dict) else {}
    is_most_seasonal = str(most_seasonal.get("archetype_label", "")).strip() == label

    seasonal_slice = pd.DataFrame(
        [row for row in seasonal_rows if isinstance(row, dict) and str(row.get("archetype_label", "")).strip() == label]
    )
    peak_season = ""
    peak_season_label = ""
    peak_season_share = None
    seasonality_gap = None
    active_seasons: list[str] = []
    if not seasonal_slice.empty:
        seasonal_slice["archetype_share"] = pd.to_numeric(seasonal_slice.get("archetype_share"), errors="coerce")
        season_profile = (
            seasonal_slice.groupby("season", dropna=False)["archetype_share"]
            .mean()
            .reset_index(name="mean_share")
            .sort_values(["mean_share", "season"], ascending=[False, True])
            .reset_index(drop=True)
        )
        season_label_profile = (
            seasonal_slice.groupby(["season_label", "season"], dropna=False)["archetype_share"]
            .mean()
            .reset_index(name="mean_share")
            .sort_values(["mean_share", "season_label"], ascending=[False, True])
            .reset_index(drop=True)
        )
        active_seasons = [str(item) for item in season_profile["season"].astype(str).tolist()]
        if not season_profile.empty:
            peak_row = season_profile.iloc[0]
            peak_season = str(peak_row["season"])
            peak_season_share = _native_value(peak_row["mean_share"])
            if len(season_profile.index) > 1:
                seasonality_gap = _native_value(float(season_profile.iloc[0]["mean_share"] - season_profile.iloc[-1]["mean_share"]))
            else:
                seasonality_gap = 0.0
        if not season_label_profile.empty:
            peak_season_label = str(season_label_profile.iloc[0]["season_label"])

    top_cross_state = evolution_brief.get("top_cross_state_transition", {})
    top_cross_state = top_cross_state if isinstance(top_cross_state, dict) else {}
    from_label = str(top_cross_state.get("from_archetype", "")).strip()
    to_label = str(top_cross_state.get("to_archetype", "")).strip()
    top_cross_state_label = f"{from_label} -> {to_label}" if from_label or to_label else ""

    return _native_value(
        {
            "available": True,
            "archetype_label": label,
            "regime_shift_months": sorted(dict.fromkeys(regime_shift_months)),
            "dominant_months": sorted(dict.fromkeys(dominant_months)),
            "top_share_gain_months": sorted(dict.fromkeys(top_share_gain_months)),
            "top_share_loss_months": sorted(dict.fromkeys(top_share_loss_months)),
            "biggest_regime_shift_month": biggest_regime_shift_month,
            "biggest_regime_shift_role": biggest_regime_shift_role,
            "is_most_seasonal_archetype": is_most_seasonal,
            "peak_season": peak_season,
            "peak_season_label": peak_season_label,
            "peak_season_share": peak_season_share,
            "seasonality_gap": seasonality_gap,
            "active_seasons": active_seasons,
            "top_cross_state_transition_label": top_cross_state_label,
            "top_cross_state_transition_involved": label in {from_label, to_label} and bool(top_cross_state_label),
        }
    )


def _annotate_policy_or_scenario(
    record: dict[str, object],
    *,
    lifecycle_profile: dict[str, object],
    target: str,
) -> dict[str, object]:
    if not record:
        return record
    profile = lifecycle_profile if isinstance(lifecycle_profile, dict) else {}
    if not bool(profile.get("available")):
        record["lifecycle_signals"] = []
        record["lifecycle_annotation"] = ""
        return record

    label = str(profile.get("archetype_label", "")).strip()
    biggest_shift_month = str(profile.get("biggest_regime_shift_month", "")).strip()
    biggest_shift_role = str(profile.get("biggest_regime_shift_role", "")).strip()
    regime_shift_months = [str(item) for item in profile.get("regime_shift_months", []) if str(item).strip()]
    peak_season = str(profile.get("peak_season", "")).strip()
    peak_season_label = str(profile.get("peak_season_label", "")).strip()
    cross_state_label = str(profile.get("top_cross_state_transition_label", "")).strip()

    signals: list[str] = []
    notes: list[str] = []
    if biggest_shift_role and biggest_shift_month:
        if biggest_shift_role in {"dominant_archetype", "top_share_gain_archetype"}:
            signals.append("biggest_regime_shift")
            notes.append(f"`{label}` is directly involved in the biggest regime shift around `{biggest_shift_month}`.")
        else:
            signals.append("regime_shift_active")
            notes.append(f"`{label}` is part of the biggest regime-shift handoff around `{biggest_shift_month}`.")
    elif regime_shift_months:
        signals.append("regime_shift_active")
        preview = ", ".join(regime_shift_months[:3])
        notes.append(f"`{label}` shows meaningful month-over-month movement in `{preview}`.")

    if bool(profile.get("is_most_seasonal_archetype")) and peak_season:
        signals.append("most_seasonal_archetype")
        if peak_season_label:
            notes.append(f"`{label}` is the most seasonal archetype and peaks in `{peak_season_label}`.")
        else:
            notes.append(f"`{label}` is the most seasonal archetype and peaks in `{peak_season}`.")
    elif peak_season:
        signals.append("seasonal_peak")
        if peak_season_label:
            notes.append(f"`{label}` peaks in `{peak_season_label}`.")
        else:
            notes.append(f"`{label}` peaks in `{peak_season}`.")

    if bool(profile.get("top_cross_state_transition_involved")) and cross_state_label:
        signals.append("cross_state_transition")
        notes.append(f"It participates in the largest cross-state transition `{cross_state_label}`.")

    if target == "policy":
        if "biggest_regime_shift" in signals or "regime_shift_active" in signals:
            notes.append("Re-check this policy first when taste mix shifts instead of treating the lane as static.")
        if "most_seasonal_archetype" in signals or "seasonal_peak" in signals:
            notes.append("Replay this policy near the seasonal peak before assuming year-round stability.")
    else:
        if "biggest_regime_shift" in signals or "regime_shift_active" in signals:
            notes.append("Use this scenario first during regime-shift windows.")
        if "most_seasonal_archetype" in signals or "seasonal_peak" in signals:
            notes.append("Treat this as a calendar-aware scenario review, not only a generic stress slice.")

    record["lifecycle_signals"] = signals
    record["lifecycle_annotation"] = " ".join(notes)
    return record


def _select_model_recommendation(model_frontier: pd.DataFrame, role: str) -> dict[str, object]:
    if model_frontier.empty:
        return {}
    ranked = _prefer_efficient_rows(model_frontier)
    for column in [
        "utility_score",
        "test_top1",
        "test_discounted_reward",
        "test_hit_at_k",
        "test_expected_utility_mass",
        "test_selective_risk",
        "test_accepted_rate",
    ]:
        ranked[column] = pd.to_numeric(ranked.get(column), errors="coerce")
    if role == "high_skip":
        risk_complete = ranked.loc[ranked.get("risk_complete", False).fillna(False).astype(bool)].copy()
        if not risk_complete.empty:
            ranked = risk_complete
        selected = ranked.sort_values(
            ["test_selective_risk", "test_accepted_rate", "utility_score"],
            ascending=[True, False, False],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Lowest selective risk with complete uncertainty metrics."
    elif role == "exploratory":
        selected = ranked.sort_values(
            ["test_expected_utility_mass", "test_hit_at_k", "utility_score", "test_top1"],
            ascending=[False, False, False, False],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Highest expected utility mass and hit-rate proxy for discovery breadth."
    else:
        selected = ranked.sort_values(
            ["utility_score", "test_top1", "test_discounted_reward"],
            ascending=[False, False, False],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Highest utility score on the efficient model frontier."
    return _compact_record(
        selected,
        [
            "model_name",
            "model_type",
            "model_family",
            "frontier_status",
            "is_pareto_efficient",
            "utility_score",
            "test_top1",
            "test_hit_at_k",
            "test_discounted_reward",
            "test_expected_utility_mass",
            "test_selective_risk",
            "test_accepted_rate",
        ],
        selection_basis=basis,
    )


def _select_policy_recommendation(policy_frontier: pd.DataFrame, role: str) -> dict[str, object]:
    if policy_frontier.empty:
        return {}
    ranked = _prefer_efficient_rows(policy_frontier)
    for column in ["utility_score", "mean_session_length", "mean_skip_risk", "worst_skip_risk", "mean_end_risk"]:
        ranked[column] = pd.to_numeric(ranked.get(column), errors="coerce")
    safe_ranked = ranked.loc[ranked["policy_family"].astype(str) == "safe"].copy()
    if role == "high_skip":
        if not safe_ranked.empty:
            ranked = safe_ranked
        selected = ranked.sort_values(
            ["mean_skip_risk", "worst_skip_risk", "mean_end_risk", "utility_score"],
            ascending=[True, True, True, False],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Lowest skip-risk safe policy on the stress-tested frontier."
    elif role == "exploratory":
        if not safe_ranked.empty:
            ranked = safe_ranked
        selected = ranked.sort_values(
            ["mean_session_length", "utility_score", "mean_skip_risk"],
            ascending=[False, False, True],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Longest safe-policy session length while preserving frontier utility."
    else:
        selected = ranked.sort_values(
            ["utility_score", "mean_skip_risk", "mean_end_risk"],
            ascending=[False, True, True],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Highest utility score on the policy frontier."
    return _compact_record(
        selected,
        [
            "policy_name",
            "policy_family",
            "frontier_status",
            "is_pareto_efficient",
            "utility_score",
            "mean_session_length",
            "mean_skip_risk",
            "worst_skip_risk",
            "mean_end_risk",
            "scenario_count",
        ],
        selection_basis=basis,
    )


def _select_scenario_focus(scenario_sensitivity: pd.DataFrame, role: str) -> dict[str, object]:
    if scenario_sensitivity.empty:
        return {}
    ranked = scenario_sensitivity.copy()
    for column in [
        "baseline_skip_risk",
        "baseline_end_risk",
        "best_safe_skip_risk",
        "best_safe_end_risk",
        "safe_skip_improvement_vs_baseline",
        "pressure_score",
    ]:
        ranked[column] = pd.to_numeric(ranked.get(column), errors="coerce")
    if role == "dominant":
        baseline = ranked.loc[ranked["scenario"].astype(str) == "baseline"].copy()
        if not baseline.empty:
            selected = baseline.iloc[0].to_dict()
        else:
            selected = ranked.sort_values(
                ["pressure_score", "baseline_skip_risk"],
                ascending=[True, True],
                na_position="last",
            ).iloc[0].to_dict()
        basis = "Default operating slice anchored on the baseline scenario."
    elif role == "high_skip":
        selected = ranked.sort_values(
            ["pressure_score", "baseline_skip_risk", "safe_skip_improvement_vs_baseline"],
            ascending=[False, False, False],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Highest pressure scenario where skip-risk control matters most."
    else:
        selected = ranked.sort_values(
            ["safe_skip_improvement_vs_baseline", "pressure_score", "best_safe_skip_risk"],
            ascending=[False, False, True],
            na_position="last",
        ).iloc[0].to_dict()
        basis = "Largest safe skip-improvement window for discovery-style routing experiments."
    return _compact_record(
        selected,
        [
            "scenario",
            "baseline_skip_risk",
            "baseline_end_risk",
            "best_safe_policy_name",
            "best_safe_skip_risk",
            "best_safe_end_risk",
            "safe_skip_improvement_vs_baseline",
            "pressure_score",
        ],
        selection_basis=basis,
    )


def _build_archetype_decision_bridge(
    *,
    output_dir: Path,
    run_dir: Path,
    model_frontier: pd.DataFrame,
    policy_frontier: pd.DataFrame,
    scenario_sensitivity: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    listener_context = _listener_archetype_context(output_dir)
    lifecycle_context = listener_context.get("lifecycle", {})
    lifecycle_context = lifecycle_context if isinstance(lifecycle_context, dict) else {}
    evolution_brief = lifecycle_context.get("evolution_brief", {})
    evolution_brief = evolution_brief if isinstance(evolution_brief, dict) else {}
    summary = [f"Latest quant decision anchor is `{run_dir.name}`."]
    if not listener_context["available"]:
        summary.append("Listener archetype outputs were not available, so archetype-specific bridge recommendations were skipped.")
        actions = [
            "Build `make listener-archetypes` before relying on archetype-specific model and policy slices.",
            "Use the existing quant frontier artifacts directly until archetype summaries are present.",
        ]
        payload = {
            "status": "listener_archetypes_missing",
            "run_id": run_dir.name,
            "listener_archetypes_available": False,
            "listener_archetype_source": {
                "status": listener_context["status"],
                "dominant_archetype": listener_context["dominant_archetype"],
                "highest_skip_archetype": listener_context["highest_skip_archetype"],
                "highest_exploration_archetype": listener_context["highest_exploration_archetype"],
                "lifecycle_available": False,
            },
            "archetype_recommendations": [],
            "summary": summary,
            "actions": actions,
        }
        markdown = [
            "# Archetype Decision Bridge",
            "",
            *[f"- {line}" for line in summary],
            "",
            "## Suggested Uses",
            "",
            *[f"- {line}" for line in actions],
        ]
        return payload, markdown

    brief = listener_context["brief"]
    lookup = listener_context["archetype_lookup"]
    if bool(lifecycle_context.get("available")):
        biggest_shift = evolution_brief.get("biggest_regime_shift", {})
        biggest_shift = biggest_shift if isinstance(biggest_shift, dict) else {}
        most_seasonal = evolution_brief.get("most_seasonal_archetype", {})
        most_seasonal = most_seasonal if isinstance(most_seasonal, dict) else {}
        biggest_shift_month = str(biggest_shift.get("month", "")).strip()
        if biggest_shift_month:
            summary.append(f"Listener lifecycle shows the largest regime shift in `{biggest_shift_month}`.")
        most_seasonal_label = str(most_seasonal.get("archetype_label", "")).strip()
        peak_season = str(most_seasonal.get("peak_season", "")).strip()
        if most_seasonal_label and peak_season:
            summary.append(f"Most seasonal archetype is `{most_seasonal_label}`, peaking in `{peak_season}`.")
    else:
        summary.append("Listener lifecycle outputs were not available, so the bridge stays on steady-state archetype summaries only.")
    roles = [
        {
            "role": "dominant",
            "title": "Dominant Archetype",
            "archetype_label": listener_context["dominant_archetype"],
            "metric_name": "days",
            "metric_value": brief.get("dominant_archetype_days"),
            "archetype_basis": "Most common listener state across observed days.",
        },
        {
            "role": "high_skip",
            "title": "High-Skip Archetype",
            "archetype_label": listener_context["highest_skip_archetype"],
            "metric_name": "skip_rate",
            "metric_value": brief.get("highest_skip_rate"),
            "archetype_basis": "Highest skip-rate listener state from listener archetypes.",
        },
        {
            "role": "exploratory",
            "title": "Exploratory Archetype",
            "archetype_label": listener_context["highest_exploration_archetype"],
            "metric_name": "exploration_ratio",
            "metric_value": brief.get("highest_exploration_ratio"),
            "archetype_basis": "Most exploratory listener state from archetype summaries.",
        },
    ]

    recommendations: list[dict[str, object]] = []
    for role in roles:
        label = str(role["archetype_label"]).strip()
        if not label:
            continue
        lifecycle_profile = _archetype_lifecycle_profile(label, listener_context)
        model = _select_model_recommendation(model_frontier, str(role["role"]))
        policy = _annotate_policy_or_scenario(
            _select_policy_recommendation(policy_frontier, str(role["role"])),
            lifecycle_profile=lifecycle_profile,
            target="policy",
        )
        scenario = _annotate_policy_or_scenario(
            _select_scenario_focus(scenario_sensitivity, str(role["role"])),
            lifecycle_profile=lifecycle_profile,
            target="scenario",
        )
        recommendation = {
            "role": role["role"],
            "title": role["title"],
            "archetype_label": label,
            "archetype_basis": role["archetype_basis"],
            "archetype_metric_name": role["metric_name"],
            "archetype_metric_value": _native_value(role["metric_value"]),
            "archetype_details": lookup.get(label, {}),
            "lifecycle_context": lifecycle_profile,
            "recommended_model": model,
            "recommended_policy": policy,
            "scenario_focus": scenario,
        }
        recommendations.append(recommendation)
        summary.append(
            f"{role['title']} `{label}` maps to model `{model.get('model_name', '')}`, policy `{policy.get('policy_name', '')}`, and scenario `{scenario.get('scenario', '')}`."
        )

    actions = [
        "Use the dominant archetype bridge as the default evaluation slice for everyday Taste OS demos and benchmark checks.",
        "Start calibration and recovery experiments with the high-skip bridge because it couples the safest model with the lowest skip-risk policy.",
        "Use the exploratory bridge when pressure-testing discovery routing so scenario focus and model breadth stay aligned.",
    ]
    payload = {
        "status": "ok",
        "run_id": run_dir.name,
        "listener_archetypes_available": True,
        "listener_archetype_source": {
            "status": listener_context["status"],
            "dominant_archetype": listener_context["dominant_archetype"],
            "highest_skip_archetype": listener_context["highest_skip_archetype"],
            "highest_exploration_archetype": listener_context["highest_exploration_archetype"],
            "summary": _native_value(brief.get("summary", [])),
            "actions": _native_value(brief.get("actions", [])),
            "archetype_count": len(lookup),
            "lifecycle_available": bool(lifecycle_context.get("available")),
            "taste_evolution_summary": _native_value(evolution_brief.get("summary", [])),
            "taste_evolution_actions": _native_value(evolution_brief.get("actions", [])),
            "biggest_regime_shift": _native_value(evolution_brief.get("biggest_regime_shift", {})),
            "most_seasonal_archetype": _native_value(evolution_brief.get("most_seasonal_archetype", {})),
        },
        "archetype_recommendations": recommendations,
        "summary": summary,
        "actions": actions,
    }

    markdown = [
        "# Archetype Decision Bridge",
        "",
        *[f"- {line}" for line in summary],
        "",
        "## Recommended Lanes",
        "",
    ]
    for recommendation in recommendations:
        model = recommendation.get("recommended_model", {})
        policy = recommendation.get("recommended_policy", {})
        scenario = recommendation.get("scenario_focus", {})
        markdown.extend(
            [
                f"### {recommendation.get('title', '')}",
                "",
                f"- Archetype: `{recommendation.get('archetype_label', '')}` ({recommendation.get('archetype_metric_name', '')}: `{recommendation.get('archetype_metric_value', '')}`)",
                f"- Model: `{model.get('model_name', '')}`. {model.get('selection_basis', '')}",
                f"- Policy: `{policy.get('policy_name', '')}`. {policy.get('selection_basis', '')}",
                (
                    f"- Policy lifecycle: {policy.get('lifecycle_annotation', '')}"
                    if str(policy.get("lifecycle_annotation", "")).strip()
                    else ""
                ),
                f"- Scenario focus: `{scenario.get('scenario', '')}`. {scenario.get('selection_basis', '')}",
                (
                    f"- Scenario lifecycle: {scenario.get('lifecycle_annotation', '')}"
                    if str(scenario.get("lifecycle_annotation", "")).strip()
                    else ""
                ),
                "",
            ]
        )
    markdown.extend(
        [
            "## Suggested Uses",
            "",
            *[f"- {line}" for line in actions],
        ]
    )
    return payload, markdown


def _bridge_alignment_for_combo(
    *,
    model_name: str,
    policy_name: str,
    scenario: str,
    bridge_payload: dict[str, Any],
) -> dict[str, object]:
    recommendations = bridge_payload.get("archetype_recommendations", []) if isinstance(bridge_payload, dict) else []
    if not isinstance(recommendations, list):
        recommendations = []

    best_score = 0.0
    matched_roles: list[str] = []
    matched_labels: list[str] = []
    lifecycle_signals: list[str] = []
    high_skip_context = False
    high_skip_label = ""
    high_skip_rate = float("nan")

    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        role = str(recommendation.get("role", "")).strip()
        label = str(recommendation.get("archetype_label", "")).strip()
        recommended_model = recommendation.get("recommended_model", {})
        recommended_policy = recommendation.get("recommended_policy", {})
        scenario_focus = recommendation.get("scenario_focus", {})
        recommended_model = recommended_model if isinstance(recommended_model, dict) else {}
        recommended_policy = recommended_policy if isinstance(recommended_policy, dict) else {}
        scenario_focus = scenario_focus if isinstance(scenario_focus, dict) else {}

        model_match = model_name == str(recommended_model.get("model_name", "")).strip()
        policy_match = policy_name == str(recommended_policy.get("policy_name", "")).strip()
        scenario_match = scenario == str(scenario_focus.get("scenario", "")).strip()
        role_score = (0.30 if model_match else 0.0) + (0.35 if policy_match else 0.0) + (0.35 if scenario_match else 0.0)
        best_score = max(best_score, role_score)
        if role_score >= 0.65:
            matched_roles.append(role)
            matched_labels.append(label)

        context_match = role_score >= 0.85 or scenario_match
        if context_match:
            for signal_source in [recommended_policy.get("lifecycle_signals", []), scenario_focus.get("lifecycle_signals", [])]:
                if isinstance(signal_source, list):
                    lifecycle_signals.extend(str(signal) for signal in signal_source)
            if role == "high_skip":
                high_skip_context = True
                high_skip_label = label
                high_skip_rate = _safe_float(recommendation.get("archetype_metric_value"))

    lifecycle_drift_signals = {"biggest_regime_shift", "regime_shift_active", "cross_state_transition"}
    return {
        "bridge_alignment_score": best_score,
        "matched_roles": matched_roles,
        "matched_labels": matched_labels,
        "lifecycle_signals": lifecycle_signals,
        "lifecycle_drift_context": bool(lifecycle_drift_signals.intersection(set(lifecycle_signals))),
        "high_skip_context": high_skip_context,
        "high_skip_label": high_skip_label,
        "high_skip_rate": high_skip_rate,
    }


def _build_scenario_utility_simulation(
    *,
    run_dir: Path,
    model_frontier: pd.DataFrame,
    policy_frontier: pd.DataFrame,
    scenario_sensitivity: pd.DataFrame,
    bridge_payload: dict[str, Any],
    drift_summary: dict[str, object],
) -> tuple[dict[str, Any], list[dict[str, object]], list[str], list[str]]:
    columns = [
        "rank",
        "model_name",
        "policy_name",
        "scenario",
        "utility_score",
        "model_utility_component",
        "policy_utility_component",
        "scenario_resilience_component",
        "safe_improvement_component",
        "bridge_alignment_component",
        "risk_penalty",
        "model_frontier_status",
        "policy_frontier_status",
        "policy_family",
        "model_test_top1",
        "model_selective_risk",
        "model_accepted_rate",
        "policy_mean_skip_risk",
        "policy_worst_skip_risk",
        "scenario_pressure_score",
        "scenario_baseline_skip_risk",
        "scenario_best_safe_skip_risk",
        "scenario_safe_skip_improvement",
        "drift_jsd",
        "archetype_roles",
        "archetype_labels",
        "lifecycle_signals",
        "high_skip_context",
        "high_drift_context",
        "notes",
    ]
    weights = {
        "model_frontier_utility": 0.38,
        "policy_frontier_utility": 0.32,
        "scenario_resilience": 0.15,
        "archetype_bridge_alignment": 0.10,
        "safe_skip_improvement": 0.05,
    }
    penalties = {
        "non_safe_high_skip_policy": 0.05,
        "non_safe_high_pressure_scenario": 0.04,
        "incomplete_risk_under_drift": 0.03,
        "elevated_selective_risk_under_drift": 0.03,
    }
    missing_inputs: list[str] = []
    if model_frontier.empty:
        missing_inputs.append("model_frontier")
    if policy_frontier.empty:
        missing_inputs.append("policy_frontier")
    if scenario_sensitivity.empty:
        missing_inputs.append("scenario_sensitivity")

    target_drift = drift_summary.get("target_drift", {}) if isinstance(drift_summary.get("target_drift"), dict) else {}
    drift_jsd = _safe_float(target_drift.get("train_vs_test_jsd"))
    high_global_drift = math.isfinite(drift_jsd) and drift_jsd >= 0.20
    score_formula = {
        "description": "utility_score = weighted model utility + policy utility + scenario resilience + bridge alignment + safe-skip improvement - risk penalties",
        "weights": weights,
        "penalties": penalties,
        "high_drift_jsd_threshold": 0.20,
        "high_skip_thresholds": {
            "policy_mean_skip_risk": 0.60,
            "policy_worst_skip_risk": 0.65,
            "scenario_baseline_skip_risk": 0.65,
            "scenario_best_safe_skip_risk": 0.60,
        },
    }

    if missing_inputs:
        summary = [
            f"Latest quant decision anchor is `{run_dir.name}`.",
            f"Scenario utility simulation was skipped because these inputs were missing: `{', '.join(missing_inputs)}`.",
        ]
        payload = {
            "status": "insufficient_inputs",
            "run_id": run_dir.name,
            "missing_inputs": missing_inputs,
            "score_formula": score_formula,
            "drift_jsd": _native_value(drift_jsd),
            "row_count": 0,
            "top_combinations": [],
            "combinations": [],
            "summary": summary,
        }
        markdown = [
            "# Scenario Utility Simulation",
            "",
            *[f"- {line}" for line in summary],
            "",
            "## Formula",
            "",
            f"- {score_formula['description']}",
        ]
        return payload, [], markdown, columns

    model_candidates = _prefer_efficient_rows(model_frontier).copy()
    policy_candidates = _prefer_efficient_rows(policy_frontier).copy()
    scenarios = scenario_sensitivity.copy()

    model_candidate_source = (
        "pareto_efficient" if "is_pareto_efficient" in model_frontier.columns and bool(model_frontier["is_pareto_efficient"].fillna(False).astype(bool).any()) else "all_available"
    )
    policy_candidate_source = (
        "pareto_efficient" if "is_pareto_efficient" in policy_frontier.columns and bool(policy_frontier["is_pareto_efficient"].fillna(False).astype(bool).any()) else "all_available"
    )

    for column in ["utility_score", "test_top1", "test_selective_risk", "test_accepted_rate"]:
        if column not in model_candidates.columns:
            model_candidates[column] = np.nan
        model_candidates[column] = pd.to_numeric(model_candidates[column], errors="coerce")
    for column in ["utility_score", "mean_skip_risk", "worst_skip_risk", "mean_end_risk"]:
        if column not in policy_candidates.columns:
            policy_candidates[column] = np.nan
        policy_candidates[column] = pd.to_numeric(policy_candidates[column], errors="coerce")
    for column in [
        "pressure_score",
        "baseline_skip_risk",
        "best_safe_skip_risk",
        "safe_skip_improvement_vs_baseline",
    ]:
        if column not in scenarios.columns:
            scenarios[column] = np.nan
        scenarios[column] = pd.to_numeric(scenarios[column], errors="coerce")

    scenarios["scenario_resilience_component"] = _normalize_series(scenarios["pressure_score"], higher_is_better=False).fillna(0.0)
    scenarios["safe_improvement_component"] = _normalize_series(
        scenarios["safe_skip_improvement_vs_baseline"],
        higher_is_better=True,
    ).fillna(0.0)
    pressure_values = pd.to_numeric(scenarios["pressure_score"], errors="coerce")
    high_pressure_threshold = float(pressure_values.quantile(0.75)) if pressure_values.notna().any() else float("nan")

    rows: list[dict[str, object]] = []
    model_candidates = model_candidates.sort_values(["utility_score", "test_top1"], ascending=[False, False], na_position="last")
    policy_candidates = policy_candidates.sort_values(
        ["utility_score", "mean_skip_risk", "mean_end_risk"],
        ascending=[False, True, True],
        na_position="last",
    )
    scenarios = scenarios.sort_values(["pressure_score", "baseline_skip_risk"], ascending=[False, False], na_position="last")

    for _, model in model_candidates.iterrows():
        model_name = str(model.get("model_name", "")).strip()
        model_component = _bounded_unit_score(model.get("utility_score"))
        model_selective_risk = _safe_float(model.get("test_selective_risk"))
        model_risk_complete = bool(model.get("risk_complete", False))
        for _, policy in policy_candidates.iterrows():
            policy_name = str(policy.get("policy_name", "")).strip()
            policy_family = str(policy.get("policy_family", "")).strip()
            policy_component = _bounded_unit_score(policy.get("utility_score"))
            policy_skip = _safe_float(policy.get("mean_skip_risk"))
            policy_worst_skip = _safe_float(policy.get("worst_skip_risk"))
            for _, scenario_row in scenarios.iterrows():
                scenario_name = str(scenario_row.get("scenario", "")).strip()
                scenario_resilience = _bounded_unit_score(scenario_row.get("scenario_resilience_component"))
                safe_improvement = _bounded_unit_score(scenario_row.get("safe_improvement_component"))
                pressure_score = _safe_float(scenario_row.get("pressure_score"))
                baseline_skip = _safe_float(scenario_row.get("baseline_skip_risk"))
                best_safe_skip = _safe_float(scenario_row.get("best_safe_skip_risk"))
                scenario_high_pressure = (
                    math.isfinite(pressure_score)
                    and math.isfinite(high_pressure_threshold)
                    and pressure_score >= high_pressure_threshold
                    and pressure_score > 0.0
                )
                alignment = _bridge_alignment_for_combo(
                    model_name=model_name,
                    policy_name=policy_name,
                    scenario=scenario_name,
                    bridge_payload=bridge_payload,
                )
                bridge_component = _bounded_unit_score(alignment.get("bridge_alignment_score"))
                lifecycle_signals = [
                    str(signal)
                    for signal in alignment.get("lifecycle_signals", [])
                    if str(signal).strip()
                ]
                high_skip_context = bool(alignment.get("high_skip_context")) or (
                    math.isfinite(policy_skip) and policy_skip >= 0.60
                ) or (
                    math.isfinite(policy_worst_skip) and policy_worst_skip >= 0.65
                ) or (
                    math.isfinite(baseline_skip) and baseline_skip >= 0.65
                ) or (
                    math.isfinite(best_safe_skip) and best_safe_skip >= 0.60
                )
                lifecycle_drift_context = bool(alignment.get("lifecycle_drift_context"))
                high_drift_context = high_global_drift or lifecycle_drift_context

                risk_penalty = 0.0
                if high_skip_context and policy_family != "safe":
                    risk_penalty += penalties["non_safe_high_skip_policy"]
                if scenario_high_pressure and policy_family != "safe":
                    risk_penalty += penalties["non_safe_high_pressure_scenario"]
                if high_drift_context and not model_risk_complete:
                    risk_penalty += penalties["incomplete_risk_under_drift"]
                if high_drift_context and math.isfinite(model_selective_risk) and model_selective_risk > 0.45:
                    risk_penalty += penalties["elevated_selective_risk_under_drift"]

                utility_score = (
                    weights["model_frontier_utility"] * model_component
                    + weights["policy_frontier_utility"] * policy_component
                    + weights["scenario_resilience"] * scenario_resilience
                    + weights["archetype_bridge_alignment"] * bridge_component
                    + weights["safe_skip_improvement"] * safe_improvement
                    - risk_penalty
                )
                utility_score = min(1.0, max(0.0, utility_score))

                notes: list[str] = []
                matched_roles = [str(role) for role in alignment.get("matched_roles", []) if str(role).strip()]
                matched_labels = [str(label) for label in alignment.get("matched_labels", []) if str(label).strip()]
                if matched_roles:
                    notes.append(f"Bridge alignment with `{_join_unique(matched_roles)}` archetype lane(s).")
                if high_skip_context:
                    skip_label = str(alignment.get("high_skip_label", "")).strip()
                    skip_rate = _safe_float(alignment.get("high_skip_rate"))
                    if skip_label and math.isfinite(skip_rate):
                        notes.append(f"High-skip bridge context: `{skip_label}` skip rate `{skip_rate:.3f}`; keep skip-risk guardrails visible.")
                    else:
                        notes.append("High-skip operating context: skip risk is elevated in the policy or scenario slice.")
                if high_global_drift and math.isfinite(drift_jsd):
                    notes.append(f"High-drift context: target drift JSD `{drift_jsd:.3f}`; refresh calibration before promotion.")
                if lifecycle_drift_context:
                    notes.append(f"Lifecycle drift context: `{_join_unique(lifecycle_signals)}` signal(s) from the archetype bridge.")
                if scenario_high_pressure:
                    notes.append("Scenario pressure is in the top quartile; review stress losses before rollout.")
                if risk_penalty > 0:
                    notes.append(f"Risk penalty `{risk_penalty:.2f}` applied for non-safe policy or drift-sensitive model risk.")

                rows.append(
                    {
                        "rank": 0,
                        "model_name": model_name,
                        "policy_name": policy_name,
                        "scenario": scenario_name,
                        "utility_score": round(utility_score, 6),
                        "model_utility_component": round(model_component, 6),
                        "policy_utility_component": round(policy_component, 6),
                        "scenario_resilience_component": round(scenario_resilience, 6),
                        "safe_improvement_component": round(safe_improvement, 6),
                        "bridge_alignment_component": round(bridge_component, 6),
                        "risk_penalty": round(risk_penalty, 6),
                        "model_frontier_status": _native_value(model.get("frontier_status")),
                        "policy_frontier_status": _native_value(policy.get("frontier_status")),
                        "policy_family": policy_family,
                        "model_test_top1": _native_value(model.get("test_top1")),
                        "model_selective_risk": _native_value(model.get("test_selective_risk")),
                        "model_accepted_rate": _native_value(model.get("test_accepted_rate")),
                        "policy_mean_skip_risk": _native_value(policy.get("mean_skip_risk")),
                        "policy_worst_skip_risk": _native_value(policy.get("worst_skip_risk")),
                        "scenario_pressure_score": _native_value(scenario_row.get("pressure_score")),
                        "scenario_baseline_skip_risk": _native_value(scenario_row.get("baseline_skip_risk")),
                        "scenario_best_safe_skip_risk": _native_value(scenario_row.get("best_safe_skip_risk")),
                        "scenario_safe_skip_improvement": _native_value(scenario_row.get("safe_skip_improvement_vs_baseline")),
                        "drift_jsd": _native_value(drift_jsd),
                        "archetype_roles": _join_unique(matched_roles),
                        "archetype_labels": _join_unique(matched_labels),
                        "lifecycle_signals": _join_unique(lifecycle_signals),
                        "high_skip_context": high_skip_context,
                        "high_drift_context": high_drift_context,
                        "notes": " ".join(notes),
                    }
                )

    simulation_frame = pd.DataFrame(rows)
    if not simulation_frame.empty:
        simulation_frame = simulation_frame.sort_values(
            [
                "utility_score",
                "bridge_alignment_component",
                "model_utility_component",
                "policy_utility_component",
                "scenario_resilience_component",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        simulation_frame["rank"] = np.arange(1, len(simulation_frame.index) + 1)
        rows = [_native_value(row) for row in simulation_frame[columns].to_dict(orient="records")]

    top_rows = rows[:10]
    top = top_rows[0] if top_rows else {}
    summary = [
        f"Latest quant decision anchor is `{run_dir.name}`.",
        f"Simulated `{len(rows)}` model / policy / scenario combinations using `{model_candidate_source}` model rows and `{policy_candidate_source}` policy rows.",
    ]
    if top:
        summary.append(
            f"Top simulated combo is model `{top.get('model_name', '')}`, policy `{top.get('policy_name', '')}`, scenario `{top.get('scenario', '')}` at utility `{_safe_float(top.get('utility_score')):.3f}`."
        )
    if high_global_drift and math.isfinite(drift_jsd):
        summary.append(f"High target drift is active at JSD `{drift_jsd:.3f}`, so drift-sensitive notes are attached to candidate rows.")

    payload = {
        "status": "ok",
        "run_id": run_dir.name,
        "candidate_strategy": {
            "models": model_candidate_source,
            "policies": policy_candidate_source,
            "scenarios": "all_sensitivity_rows",
        },
        "score_formula": score_formula,
        "drift_jsd": _native_value(drift_jsd),
        "row_count": len(rows),
        "top_combinations": top_rows,
        "combinations": rows,
        "summary": summary,
    }

    markdown = [
        "# Scenario Utility Simulation",
        "",
        *[f"- {line}" for line in summary],
        "",
        "## Formula",
        "",
        f"- {score_formula['description']}",
        f"- Weights: model `{weights['model_frontier_utility']:.2f}`, policy `{weights['policy_frontier_utility']:.2f}`, scenario resilience `{weights['scenario_resilience']:.2f}`, bridge alignment `{weights['archetype_bridge_alignment']:.2f}`, safe skip improvement `{weights['safe_skip_improvement']:.2f}`.",
        "",
        "## Top Combinations",
        "",
        "| Rank | Model | Policy | Scenario | Utility | Context Notes |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in top_rows[:8]:
        notes = str(row.get("notes", "")).replace("|", "/")
        markdown.append(
            f"| {row.get('rank', '')} | `{row.get('model_name', '')}` | `{row.get('policy_name', '')}` | `{row.get('scenario', '')}` | `{_safe_float(row.get('utility_score')):.3f}` | {notes} |"
        )
    if not top_rows:
        markdown.append("|  |  |  |  |  | No combinations were available. |")
    return payload, rows, markdown, columns


def _brief_lines(
    *,
    run_dir: Path,
    model_frontier: pd.DataFrame,
    policy_frontier: pd.DataFrame,
    scenario_sensitivity: pd.DataFrame,
    drift_summary: dict[str, object],
) -> tuple[dict[str, Any], list[str]]:
    best_model = model_frontier.iloc[0].to_dict() if not model_frontier.empty else {}
    safest_model = (
        model_frontier.sort_values(["test_selective_risk", "utility_score"], ascending=[True, False]).iloc[0].to_dict()
        if not model_frontier.empty and model_frontier["test_selective_risk"].notna().any()
        else {}
    )
    best_policy = policy_frontier.iloc[0].to_dict() if not policy_frontier.empty else {}
    hardest_scenario = scenario_sensitivity.iloc[0].to_dict() if not scenario_sensitivity.empty else {}
    target_drift = drift_summary.get("target_drift", {}) if isinstance(drift_summary.get("target_drift"), dict) else {}
    drift_jsd = _safe_float(target_drift.get("train_vs_test_jsd"))

    summary = [
        f"Latest quant decision anchor is `{run_dir.name}`.",
    ]
    if best_model:
        summary.append(
            f"Top model by utility frontier is `{best_model.get('model_name', '')}` with utility score `{_safe_float(best_model.get('utility_score')):.3f}` and test top-1 `{_safe_float(best_model.get('test_top1')):.3f}`."
        )
    if safest_model:
        summary.append(
            f"Safest model with complete uncertainty metrics is `{safest_model.get('model_name', '')}` at selective risk `{_safe_float(safest_model.get('test_selective_risk')):.3f}`."
        )
    if best_policy:
        summary.append(
            f"Best policy on the stress frontier is `{best_policy.get('policy_name', '')}` with mean skip risk `{_safe_float(best_policy.get('mean_skip_risk')):.3f}`."
        )
    if hardest_scenario:
        summary.append(
            f"Highest-pressure scenario is `{hardest_scenario.get('scenario', '')}` with pressure score `{_safe_float(hardest_scenario.get('pressure_score')):.3f}`."
        )
    if math.isfinite(drift_jsd):
        summary.append(f"Current target drift JSD remains `{drift_jsd:.3f}`.")
    actions = [
        "Use the efficient model frontier instead of raw top-1 alone when choosing challenger lanes.",
        "Treat the pressure-ranked scenarios as the first stress tests for new Taste OS or safety-policy changes.",
        "Use the safest complete-risk model as the calibration baseline before introducing a new challenger family.",
    ]
    payload = {
        "run_id": run_dir.name,
        "best_model": best_model,
        "safest_model": safest_model,
        "best_policy": best_policy,
        "hardest_scenario": hardest_scenario,
        "drift_jsd": drift_jsd,
        "summary": summary,
        "actions": actions,
    }
    markdown = [
        "# Quant Decision Brief",
        "",
        *[f"- {line}" for line in summary],
        "",
        "## Suggested Uses",
        "",
        *[f"- {line}" for line in actions],
    ]
    return payload, markdown


def build_quant_decision_lab(
    *,
    output_dir: Path,
    run_dir: Path | None,
    logger,
) -> list[Path]:
    resolved_run_dir = _resolve_run_dir(output_dir=output_dir, run_dir=run_dir)
    model_frontier = _build_model_frontier(resolved_run_dir)
    policy_frontier, scenario_sensitivity = _build_policy_frontier(resolved_run_dir)
    drift_summary = safe_read_json(resolved_run_dir / "analysis" / "data_drift_summary.json", default={})
    if model_frontier.empty and policy_frontier.empty:
        return []

    output_root = output_dir / "analysis" / "quant_decision_lab"
    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not model_frontier.empty:
        model_columns = [
            "model_name",
            "model_type",
            "model_family",
            "test_top1",
            "test_discounted_reward",
            "test_hit_at_k",
            "test_selective_risk",
            "test_abstention_rate",
            "test_accepted_rate",
            "fit_seconds",
            "utility_score",
            "frontier_status",
            "is_pareto_efficient",
            "dominates_count",
            "risk_complete",
        ]
        path = _write_csv(
            output_root / "model_decision_frontier.csv",
            _rows_for_columns(model_frontier, model_columns),
            model_columns,
        )
        paths.append(path)
        paths.append(write_json(output_root / "model_decision_frontier.json", model_frontier.to_dict(orient="records")))
    if not policy_frontier.empty:
        policy_columns = [
            "policy_name",
            "policy_family",
            "scenario_count",
            "mean_session_length",
            "mean_skip_risk",
            "worst_skip_risk",
            "mean_end_risk",
            "utility_score",
            "frontier_status",
            "is_pareto_efficient",
            "is_benchmark_selected",
            "is_benchmark_canonical",
        ]
        path = _write_csv(
            output_root / "policy_decision_frontier.csv",
            _rows_for_columns(policy_frontier, policy_columns),
            policy_columns,
        )
        paths.append(path)
        paths.append(write_json(output_root / "policy_decision_frontier.json", policy_frontier.to_dict(orient="records")))
    if not scenario_sensitivity.empty:
        sensitivity_columns = [
            "scenario",
            "baseline_skip_risk",
            "baseline_end_risk",
            "baseline_skip_delta_vs_baseline_scenario",
            "baseline_end_delta_vs_baseline_scenario",
            "best_safe_policy_name",
            "best_safe_skip_risk",
            "best_safe_end_risk",
            "safe_skip_improvement_vs_baseline",
            "pressure_score",
        ]
        path = _write_csv(
            output_root / "scenario_sensitivity.csv",
            _rows_for_columns(scenario_sensitivity, sensitivity_columns),
            sensitivity_columns,
        )
        paths.append(path)
        paths.append(write_json(output_root / "scenario_sensitivity.json", scenario_sensitivity.to_dict(orient="records")))

    bridge_payload, bridge_markdown = _build_archetype_decision_bridge(
        output_dir=output_dir,
        run_dir=resolved_run_dir,
        model_frontier=model_frontier,
        policy_frontier=policy_frontier,
        scenario_sensitivity=scenario_sensitivity,
    )
    paths.append(write_json(output_root / "archetype_decision_bridge.json", bridge_payload))
    paths.append(write_markdown(output_root / "archetype_decision_bridge.md", bridge_markdown))

    simulation_payload, simulation_rows, simulation_markdown, simulation_columns = _build_scenario_utility_simulation(
        run_dir=resolved_run_dir,
        model_frontier=model_frontier,
        policy_frontier=policy_frontier,
        scenario_sensitivity=scenario_sensitivity,
        bridge_payload=bridge_payload,
        drift_summary=drift_summary if isinstance(drift_summary, dict) else {},
    )
    paths.append(
        _write_csv(
            output_root / "scenario_utility_simulation.csv",
            simulation_rows,
            simulation_columns,
        )
    )
    paths.append(write_json(output_root / "scenario_utility_simulation.json", simulation_payload))
    paths.append(write_markdown(output_root / "scenario_utility_simulation.md", simulation_markdown))

    brief_payload, brief_markdown = _brief_lines(
        run_dir=resolved_run_dir,
        model_frontier=model_frontier,
        policy_frontier=policy_frontier,
        scenario_sensitivity=scenario_sensitivity,
        drift_summary=drift_summary if isinstance(drift_summary, dict) else {},
    )
    paths.append(write_json(output_root / "quant_decision_brief.json", brief_payload))
    paths.append(write_markdown(output_root / "quant_decision_brief.md", brief_markdown))
    logger.info(
        "Built quant decision lab for run %s with %d model rows and %d policy rows.",
        resolved_run_dir.name,
        len(model_frontier.index),
        len(policy_frontier.index),
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build quant decision-lab artifacts for the latest completed run.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing run artifacts.")
    parser.add_argument("--run-dir", type=str, default="", help="Optional explicit run directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.quant_decision_lab")
    run_dir = Path(args.run_dir).expanduser().resolve() if str(args.run_dir).strip() else None
    paths = build_quant_decision_lab(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        run_dir=run_dir,
        logger=logger,
    )
    if not paths:
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
