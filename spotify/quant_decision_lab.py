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
