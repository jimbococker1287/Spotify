from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype

from .aws_athena_export import _prepare_raw_streaming_history
from .aws_athena_export import _prepare_run_manifests
from .aws_athena_export import _prepare_run_results
from .data import load_streaming_history
from .run_artifacts import collect_run_analysis_rows
from .run_artifacts import rows_to_frame
from .run_artifacts import safe_read_csv
from .run_artifacts import safe_read_json
from .run_artifacts import write_json
from .run_artifacts import write_markdown


@dataclass
class AnalyticsWarehouseBundle:
    root: Path
    bronze: dict[str, pd.DataFrame]
    silver: dict[str, pd.DataFrame]
    gold: dict[str, pd.DataFrame]
    manifest: dict[str, Any]

    def tables(self) -> dict[str, pd.DataFrame]:
        return {**self.bronze, **self.silver, **self.gold}


def _empty_frame(columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _json_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return str(value)


def _series_mode(series: pd.Series) -> object | None:
    if series.empty:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    mode = cleaned.mode(dropna=True)
    if mode.empty:
        return cleaned.iloc[0]
    return mode.iloc[0]


def _extract_run_id(value: object) -> object:
    if isinstance(value, dict):
        run_id = value.get("run_id")
        if run_id is not None:
            return run_id
    return value


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool_fraction(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64")
    normalized = series.astype(str).str.strip().str.lower()
    normalized = normalized.map(
        {
            "true": 1.0,
            "1": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "false": 0.0,
            "0": 0.0,
            "no": 0.0,
            "n": 0.0,
        }
    )
    return pd.to_numeric(normalized, errors="coerce")


def _count_truthy(values: pd.Series) -> int:
    count = 0
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                count += 1
            continue
        if bool(value):
            count += 1
    return count


def _normalize_frame_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in frame.columns:
        series = frame[column]
        if not is_object_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        numeric_candidate = pd.to_numeric(non_null.replace("", None), errors="coerce")
        if numeric_candidate.notna().sum() == len(non_null):
            frame[column] = pd.to_numeric(series.replace("", None), errors="coerce")
            continue
        sample_text = non_null.astype(str).head(5)
        if sample_text.str.contains(r"\d{4}-\d{2}-\d{2}|T\d{2}:\d{2}", regex=True).all():
            datetime_candidate = pd.to_datetime(non_null, errors="coerce", format="mixed")
            if datetime_candidate.notna().sum() == len(non_null):
                frame[column] = pd.to_datetime(series, errors="coerce", format="mixed")
                continue
        frame[column] = series.map(
            lambda value: (
                _json_string(value)
                if isinstance(value, (dict, list, tuple, set))
                else (None if pd.isna(value) else str(value))
            )
        )
    return frame


def _reindex_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _prepare_experiment_history_frame(path: Path) -> pd.DataFrame:
    columns = [
        "event_timestamp",
        "run_id",
        "run_name",
        "profile",
        "model_name",
        "model_type",
        "model_family",
        "val_top1",
        "val_top5",
        "test_top1",
        "test_top5",
        "fit_seconds",
        "epochs",
        "data_records",
    ]
    df = safe_read_csv(path)
    if df.empty:
        return _empty_frame(columns)
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        df["event_timestamp"] = None
    return _reindex_frame(df, columns)


def _prepare_backtest_history_frame(path: Path) -> pd.DataFrame:
    columns = [
        "event_timestamp",
        "run_id",
        "run_name",
        "profile",
        "model_name",
        "model_family",
        "fold",
        "train_rows",
        "test_rows",
        "fit_seconds",
        "top1",
        "top5",
    ]
    df = safe_read_csv(path)
    if df.empty:
        return _empty_frame(columns)
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        df["event_timestamp"] = None
    return _reindex_frame(df, columns)


def _prepare_optuna_history_frame(path: Path) -> pd.DataFrame:
    columns = [
        "event_timestamp",
        "run_id",
        "run_name",
        "profile",
        "model_name",
        "base_model_name",
        "n_trials",
        "val_top1",
        "test_top1",
        "fit_seconds",
        "best_params_json",
    ]
    df = safe_read_csv(path)
    if df.empty:
        return _empty_frame(columns)
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        df["event_timestamp"] = None
    return _reindex_frame(df, columns)


def _prepare_benchmark_history_frame(path: Path) -> pd.DataFrame:
    columns = [
        "event_timestamp",
        "benchmark_id",
        "model_name",
        "model_type",
        "model_family",
        "runs",
        "val_top1_mean",
        "val_top1_std",
        "val_top1_ci95",
        "test_top1_mean",
        "test_top1_std",
        "test_top1_ci95",
    ]
    df = safe_read_csv(path)
    if df.empty:
        return _empty_frame(columns)
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        df["event_timestamp"] = None
    return _reindex_frame(df, columns)


def _control_room_snapshot_frames(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot_columns = [
        "generated_at",
        "selected_run_id",
        "selected_run_reason",
        "latest_run_id",
        "latest_run_profile",
        "latest_run_best_model_name",
        "latest_run_best_model_type",
        "latest_run_best_model_test_top1",
        "latest_run_serving_model_name",
        "latest_run_serving_model_type",
        "latest_run_promoted",
        "ops_health_status",
        "ops_health_headline",
        "operating_rhythm_status",
        "operating_rhythm_recommended_run_command",
        "operating_rhythm_recommended_run_reason",
        "target_drift_jsd",
        "selective_risk",
        "abstention_rate",
        "accepted_rate",
        "repeat_from_prev_new_gap",
        "stress_benchmark_skip_risk",
        "stress_benchmark_scenario",
        "stress_benchmark_policy_name",
        "review_action_count",
        "next_bet_count",
        "raw_json",
    ]
    action_columns = [
        "generated_at",
        "selected_run_id",
        "action_order",
        "priority",
        "area",
        "title",
        "detail",
        "inspect_targets",
        "raw_json",
    ]
    payload = safe_read_json(output_dir / "analytics" / "control_room.json", default={})
    if not isinstance(payload, dict):
        return _empty_frame(snapshot_columns), _empty_frame(action_columns)

    latest_run = payload.get("latest_run", {}) if isinstance(payload.get("latest_run"), dict) else {}
    ops_health = payload.get("ops_health", {}) if isinstance(payload.get("ops_health"), dict) else {}
    operating_rhythm = (
        payload.get("operating_rhythm", {}) if isinstance(payload.get("operating_rhythm"), dict) else {}
    )
    safety = payload.get("safety", {}) if isinstance(payload.get("safety"), dict) else {}
    qoe = payload.get("qoe", {}) if isinstance(payload.get("qoe"), dict) else {}
    run_selection = payload.get("run_selection", {}) if isinstance(payload.get("run_selection"), dict) else {}
    review_actions = payload.get("review_actions", [])
    next_bets = payload.get("next_bets", [])
    generated_at = str(payload.get("generated_at", "") or "")

    snapshot_row = {
        "generated_at": generated_at,
        "selected_run_id": _extract_run_id(run_selection.get("selected_run")),
        "selected_run_reason": run_selection.get("selection_reason"),
        "latest_run_id": latest_run.get("run_id"),
        "latest_run_profile": latest_run.get("profile"),
        "latest_run_best_model_name": latest_run.get("best_model_name"),
        "latest_run_best_model_type": latest_run.get("best_model_type"),
        "latest_run_best_model_test_top1": latest_run.get("best_model_test_top1"),
        "latest_run_serving_model_name": latest_run.get("serving_model_name"),
        "latest_run_serving_model_type": latest_run.get("serving_model_type"),
        "latest_run_promoted": latest_run.get("promoted"),
        "ops_health_status": ops_health.get("status"),
        "ops_health_headline": ops_health.get("headline"),
        "operating_rhythm_status": operating_rhythm.get("overall_status"),
        "operating_rhythm_recommended_run_command": operating_rhythm.get("recommended_run_command"),
        "operating_rhythm_recommended_run_reason": operating_rhythm.get("recommended_run_reason"),
        "target_drift_jsd": safety.get("test_jsd_target_drift"),
        "selective_risk": safety.get("test_selective_risk"),
        "abstention_rate": safety.get("test_abstention_rate"),
        "accepted_rate": safety.get("test_accepted_rate"),
        "repeat_from_prev_new_gap": safety.get("repeat_from_prev_new_gap"),
        "stress_benchmark_skip_risk": qoe.get("stress_benchmark_skip_risk"),
        "stress_benchmark_scenario": qoe.get("stress_benchmark_scenario"),
        "stress_benchmark_policy_name": qoe.get("stress_benchmark_policy_name"),
        "review_action_count": len(review_actions) if isinstance(review_actions, list) else 0,
        "next_bet_count": len(next_bets) if isinstance(next_bets, list) else 0,
        "raw_json": _json_string(payload),
    }
    action_rows: list[dict[str, object]] = []
    if isinstance(review_actions, list):
        for action_order, action in enumerate(review_actions, start=1):
            if not isinstance(action, dict):
                continue
            inspect_targets = action.get("inspect", [])
            action_rows.append(
                {
                    "generated_at": generated_at,
                    "selected_run_id": _extract_run_id(run_selection.get("selected_run")),
                    "action_order": action_order,
                    "priority": action.get("priority"),
                    "area": action.get("area"),
                    "title": action.get("title"),
                    "detail": action.get("detail"),
                    "inspect_targets": "|".join(
                        str(item).strip() for item in inspect_targets if str(item).strip()
                    )
                    if isinstance(inspect_targets, list)
                    else None,
                    "raw_json": _json_string(action),
                }
            )
    return pd.DataFrame([snapshot_row], columns=snapshot_columns), pd.DataFrame(action_rows, columns=action_columns)


def _load_creator_report_family_assets(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_columns = [
        "report_family_id",
        "seed_group_slug",
        "primary_report_path",
        "artifact_index_path",
        "ranking_markdown_path",
        "ranking_csv_path",
        "scene_markdown_path",
        "scene_csv_path",
        "scene_seed_markdown_path",
        "scene_seed_csv_path",
        "seed_markdown_path",
        "seed_csv_path",
        "backfilled_artifact_index_at",
    ]
    base_dir = output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"
    family_rows: list[dict[str, object]] = []
    ranking_frames: list[pd.DataFrame] = []
    scene_frames: list[pd.DataFrame] = []
    scene_seed_frames: list[pd.DataFrame] = []
    for path in sorted(base_dir.glob("*_report_family.json")):
        payload = safe_read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        report_family_id = path.stem.removesuffix("_report_family")
        seed_group_slug = report_family_id.removeprefix("creator_label_intelligence_")
        comparison_md = (
            payload.get("comparison_view_markdown", {})
            if isinstance(payload.get("comparison_view_markdown"), dict)
            else {}
        )
        comparison_csv = (
            payload.get("comparison_view_csv", {}) if isinstance(payload.get("comparison_view_csv"), dict) else {}
        )
        family_rows.append(
            {
                "report_family_id": report_family_id,
                "seed_group_slug": seed_group_slug,
                "primary_report_path": payload.get("primary_report"),
                "artifact_index_path": payload.get("artifact_index_markdown"),
                "ranking_markdown_path": comparison_md.get("ranking_comparison"),
                "ranking_csv_path": comparison_csv.get("ranking_comparison"),
                "scene_markdown_path": comparison_md.get("scene_comparison"),
                "scene_csv_path": comparison_csv.get("scene_comparison"),
                "scene_seed_markdown_path": comparison_md.get("scene_seed_comparison"),
                "scene_seed_csv_path": comparison_csv.get("scene_seed_comparison"),
                "seed_markdown_path": comparison_md.get("seed_comparison"),
                "seed_csv_path": comparison_csv.get("seed_comparison"),
                "backfilled_artifact_index_at": payload.get("backfilled_artifact_index_at"),
            }
        )

        ranking_path = base_dir / f"{report_family_id}_ranking_comparison.csv"
        ranking_df = safe_read_csv(ranking_path)
        if not ranking_df.empty:
            ranking_df = ranking_df.copy()
            ranking_df.insert(0, "report_family_id", report_family_id)
            ranking_df.insert(1, "seed_group_slug", seed_group_slug)
            ranking_frames.append(ranking_df)

        scene_path = base_dir / f"{report_family_id}_scene_comparison.csv"
        scene_df = safe_read_csv(scene_path)
        if not scene_df.empty:
            scene_df = scene_df.copy()
            scene_df.insert(0, "report_family_id", report_family_id)
            scene_df.insert(1, "seed_group_slug", seed_group_slug)
            scene_frames.append(scene_df)

        scene_seed_path = base_dir / f"{report_family_id}_scene_seed_comparison.csv"
        scene_seed_df = safe_read_csv(scene_seed_path)
        if not scene_seed_df.empty:
            scene_seed_df = scene_seed_df.copy()
            scene_seed_df.insert(0, "report_family_id", report_family_id)
            scene_seed_df.insert(1, "seed_group_slug", seed_group_slug)
            scene_seed_frames.append(scene_seed_df)

    family_df = pd.DataFrame(family_rows, columns=family_columns)
    ranking_df = pd.concat(ranking_frames, ignore_index=True, sort=False) if ranking_frames else pd.DataFrame()
    scene_df = pd.concat(scene_frames, ignore_index=True, sort=False) if scene_frames else pd.DataFrame()
    scene_seed_df = (
        pd.concat(scene_seed_frames, ignore_index=True, sort=False) if scene_seed_frames else pd.DataFrame()
    )
    return family_df, ranking_df, scene_df, scene_seed_df


def _build_listener_daily_activity(raw_streaming_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "played_date",
        "total_streams",
        "total_ms_played",
        "unique_artists",
        "unique_tracks",
        "skip_rate",
        "shuffle_rate",
        "offline_rate",
        "primary_platform",
        "track_stream_share",
    ]
    if raw_streaming_history.empty or "played_at" not in raw_streaming_history.columns:
        return _empty_frame(columns)
    frame = raw_streaming_history.copy()
    frame["played_at"] = pd.to_datetime(frame["played_at"], errors="coerce")
    frame = frame.loc[frame["played_at"].notna()].copy()
    if frame.empty:
        return _empty_frame(columns)
    frame["played_date"] = frame["played_at"].dt.date.astype(str)
    frame["ms_played"] = _to_numeric(frame.get("ms_played", pd.Series(index=frame.index, dtype="float64")))
    frame["skipped_metric"] = _to_bool_fraction(frame.get("skipped", pd.Series(index=frame.index)))
    frame["shuffle_metric"] = _to_bool_fraction(frame.get("shuffle", pd.Series(index=frame.index)))
    frame["offline_metric"] = _to_bool_fraction(frame.get("offline", pd.Series(index=frame.index)))
    frame["content_type"] = frame.get("content_type", pd.Series("unknown", index=frame.index)).fillna("unknown")
    grouped = (
        frame.groupby("played_date", dropna=False)
        .agg(
            total_streams=("played_date", "size"),
            total_ms_played=("ms_played", "sum"),
            unique_artists=("master_metadata_album_artist_name", "nunique"),
            unique_tracks=("master_metadata_track_name", "nunique"),
            skip_rate=("skipped_metric", "mean"),
            shuffle_rate=("shuffle_metric", "mean"),
            offline_rate=("offline_metric", "mean"),
        )
        .reset_index()
    )
    platform_lookup = (
        frame.groupby("played_date", dropna=False)["platform"].agg(_series_mode).reset_index(name="primary_platform")
    )
    track_share = (
        frame.assign(track_flag=frame["content_type"].astype(str).eq("track").astype("float64"))
        .groupby("played_date", dropna=False)["track_flag"]
        .mean()
        .reset_index(name="track_stream_share")
    )
    merged = grouped.merge(platform_lookup, on="played_date", how="left").merge(track_share, on="played_date", how="left")
    return merged[columns].sort_values("played_date").reset_index(drop=True)


def _build_model_run_summary(
    run_manifests: pd.DataFrame,
    run_results: pd.DataFrame,
    backtest_history: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "run_name",
        "profile",
        "run_timestamp",
        "model_name",
        "model_type",
        "model_family",
        "base_model_name",
        "val_top1",
        "test_top1",
        "fit_seconds",
        "epochs",
        "mean_backtest_top1",
        "backtest_folds",
        "promoted",
        "champion_gate_status",
        "champion_gate_metric_source",
        "champion_alias_model_name",
        "champion_alias_model_type",
        "is_serving_alias",
        "val_rank_within_run",
        "test_rank_within_run",
        "data_records",
    ]
    if run_results.empty:
        return _empty_frame(columns)
    model_summary = run_results.copy()
    manifest_columns = [
        column
        for column in [
            "run_id",
            "run_name",
            "profile",
            "run_timestamp",
            "data_records",
            "champion_gate_promoted",
            "champion_gate_status",
            "champion_gate_metric_source",
            "champion_alias_model_name",
            "champion_alias_model_type",
        ]
        if column in run_manifests.columns
    ]
    if manifest_columns:
        manifest_slice = run_manifests[manifest_columns].copy()
        manifest_slice = manifest_slice.rename(columns={"champion_gate_promoted": "promoted"})
        model_summary = model_summary.merge(manifest_slice, on="run_id", how="left")
    if not backtest_history.empty and {"run_id", "model_name", "top1"}.issubset(backtest_history.columns):
        backtest_summary = (
            backtest_history.assign(top1=_to_numeric(backtest_history["top1"]))
            .groupby(["run_id", "model_name"], dropna=False)
            .agg(mean_backtest_top1=("top1", "mean"), backtest_folds=("top1", "count"))
            .reset_index()
        )
        model_summary = model_summary.merge(backtest_summary, on=["run_id", "model_name"], how="left")
    else:
        model_summary["mean_backtest_top1"] = None
        model_summary["backtest_folds"] = None

    model_summary["val_top1"] = _to_numeric(model_summary.get("val_top1", pd.Series(index=model_summary.index)))
    model_summary["test_top1"] = _to_numeric(model_summary.get("test_top1", pd.Series(index=model_summary.index)))
    model_summary["val_rank_within_run"] = (
        model_summary.groupby("run_id")["val_top1"].rank(method="dense", ascending=False)
    )
    model_summary["test_rank_within_run"] = (
        model_summary.groupby("run_id")["test_top1"].rank(method="dense", ascending=False)
    )
    model_summary["is_serving_alias"] = (
        model_summary.get("model_name", pd.Series(index=model_summary.index)).astype(str)
        == model_summary.get("champion_alias_model_name", pd.Series(index=model_summary.index)).astype(str)
    )
    for column in columns:
        if column not in model_summary.columns:
            model_summary[column] = None
    return model_summary[columns].sort_values(["run_timestamp", "run_id", "val_rank_within_run"]).reset_index(drop=True)


def _build_ops_review_snapshot(
    control_room_snapshot: pd.DataFrame,
    control_room_history: pd.DataFrame,
    control_room_review_actions: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "generated_at",
        "selected_run_id",
        "latest_run_id",
        "latest_run_best_model_name",
        "latest_run_serving_model_name",
        "ops_health_status",
        "operating_rhythm_status",
        "recommended_run_command",
        "review_action_count",
        "high_priority_actions",
        "medium_priority_actions",
        "history_points",
        "current_target_drift_jsd",
        "current_stress_benchmark_skip_risk",
        "current_selective_risk",
        "latest_fast_cadence_status",
        "latest_full_cadence_status",
    ]
    if control_room_snapshot.empty:
        return _empty_frame(columns)
    snapshot = control_room_snapshot.iloc[0]
    action_count = int(len(control_room_review_actions.index))
    high_priority_actions = int(
        (control_room_review_actions.get("priority", pd.Series(dtype="object")).astype(str) == "high").sum()
    )
    medium_priority_actions = int(
        (control_room_review_actions.get("priority", pd.Series(dtype="object")).astype(str) == "medium").sum()
    )
    history_points = int(len(control_room_history.index))
    latest_fast_status = None
    latest_full_status = None
    if not control_room_history.empty:
        history = control_room_history.copy()
        if "generated_at" in history.columns:
            history["generated_at"] = pd.to_datetime(history["generated_at"], errors="coerce")
            history = history.sort_values("generated_at", ascending=False)
        latest_history = history.iloc[0]
        latest_fast_status = latest_history.get("fast_cadence_status")
        latest_full_status = latest_history.get("full_cadence_status")
    row = {
        "generated_at": snapshot.get("generated_at"),
        "selected_run_id": snapshot.get("selected_run_id"),
        "latest_run_id": snapshot.get("latest_run_id"),
        "latest_run_best_model_name": snapshot.get("latest_run_best_model_name"),
        "latest_run_serving_model_name": snapshot.get("latest_run_serving_model_name"),
        "ops_health_status": snapshot.get("ops_health_status"),
        "operating_rhythm_status": snapshot.get("operating_rhythm_status"),
        "recommended_run_command": snapshot.get("operating_rhythm_recommended_run_command"),
        "review_action_count": action_count,
        "high_priority_actions": high_priority_actions,
        "medium_priority_actions": medium_priority_actions,
        "history_points": history_points,
        "current_target_drift_jsd": snapshot.get("target_drift_jsd"),
        "current_stress_benchmark_skip_risk": snapshot.get("stress_benchmark_skip_risk"),
        "current_selective_risk": snapshot.get("selective_risk"),
        "latest_fast_cadence_status": latest_fast_status,
        "latest_full_cadence_status": latest_full_status,
    }
    return pd.DataFrame([row], columns=columns)


def _build_creator_report_family_summary(
    creator_report_families: pd.DataFrame,
    creator_ranking_opportunities: pd.DataFrame,
    creator_scene_summary: pd.DataFrame,
    creator_scene_seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "report_family_id",
        "seed_group_slug",
        "ranking_rows",
        "scene_rows",
        "scene_seed_rows",
        "priority_now_count",
        "avg_opportunity_score",
        "top_opportunity_artist",
        "top_scene_name",
        "primary_report_path",
        "artifact_index_path",
        "backfilled_artifact_index_at",
    ]
    if creator_report_families.empty:
        return _empty_frame(columns)
    family_summary = creator_report_families.copy()
    if not creator_ranking_opportunities.empty and "report_family_id" in creator_ranking_opportunities.columns:
        ranking = creator_ranking_opportunities.copy()
        ranking["opportunity_score"] = _to_numeric(ranking.get("opportunity_score", pd.Series(index=ranking.index)))
        ranking_counts = (
            ranking.groupby("report_family_id", dropna=False)
            .agg(
                ranking_rows=("artist_name", "count"),
                priority_now_count=(
                    "opportunity_band",
                    lambda values: sum(str(value) == "priority_now" for value in values),
                ),
                avg_opportunity_score=("opportunity_score", "mean"),
            )
            .reset_index()
        )
        ranking_top = (
            ranking.sort_values(["report_family_id", "opportunity_score"], ascending=[True, False])
            .drop_duplicates(subset=["report_family_id"], keep="first")[["report_family_id", "artist_name"]]
            .rename(columns={"artist_name": "top_opportunity_artist"})
        )
        family_summary = family_summary.merge(ranking_counts, on="report_family_id", how="left")
        family_summary = family_summary.merge(ranking_top, on="report_family_id", how="left")
    if not creator_scene_summary.empty and "report_family_id" in creator_scene_summary.columns:
        scene_counts = (
            creator_scene_summary.groupby("report_family_id", dropna=False)
            .agg(scene_rows=("scene_name", "count"))
            .reset_index()
        )
        scene_top = (
            creator_scene_summary.assign(
                avg_opportunity_score=_to_numeric(
                    creator_scene_summary.get("avg_opportunity_score", pd.Series(index=creator_scene_summary.index))
                )
            )
            .sort_values(["report_family_id", "avg_opportunity_score"], ascending=[True, False])
            .drop_duplicates(subset=["report_family_id"], keep="first")[["report_family_id", "scene_name"]]
            .rename(columns={"scene_name": "top_scene_name"})
        )
        family_summary = family_summary.merge(scene_counts, on="report_family_id", how="left")
        family_summary = family_summary.merge(scene_top, on="report_family_id", how="left")
    if not creator_scene_seed_summary.empty and "report_family_id" in creator_scene_seed_summary.columns:
        scene_seed_counts = (
            creator_scene_seed_summary.groupby("report_family_id", dropna=False)
            .agg(scene_seed_rows=("seed_artist", "count"))
            .reset_index()
        )
        family_summary = family_summary.merge(scene_seed_counts, on="report_family_id", how="left")
    for column in columns:
        if column not in family_summary.columns:
            family_summary[column] = None
    return family_summary[columns].sort_values("report_family_id").reset_index(drop=True)


def _build_mart_run_quality(model_run_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "run_name",
        "profile",
        "run_timestamp",
        "promoted",
        "champion_gate_status",
        "champion_gate_metric_source",
        "best_model_name",
        "best_model_type",
        "best_val_top1",
        "best_test_top1",
        "best_mean_backtest_top1",
        "serving_model_name",
        "serving_model_type",
        "models_evaluated",
        "data_records",
    ]
    if model_run_summary.empty:
        return _empty_frame(columns)
    ranked = model_run_summary.copy()
    ranked["test_top1"] = _to_numeric(ranked.get("test_top1", pd.Series(index=ranked.index)))
    ranked["val_top1"] = _to_numeric(ranked.get("val_top1", pd.Series(index=ranked.index)))
    ranked["mean_backtest_top1"] = _to_numeric(ranked.get("mean_backtest_top1", pd.Series(index=ranked.index)))
    best_rows = (
        ranked.sort_values(["run_id", "test_top1", "val_top1"], ascending=[True, False, False])
        .drop_duplicates(subset=["run_id"], keep="first")
        .rename(
            columns={
                "model_name": "best_model_name",
                "model_type": "best_model_type",
                "val_top1": "best_val_top1",
                "test_top1": "best_test_top1",
                "mean_backtest_top1": "best_mean_backtest_top1",
                "champion_alias_model_name": "serving_model_name",
                "champion_alias_model_type": "serving_model_type",
            }
        )
    )
    run_counts = (
        ranked.groupby("run_id", dropna=False)
        .agg(models_evaluated=("model_name", "count"))
        .reset_index()
    )
    merged = best_rows.merge(run_counts, on="run_id", how="left")
    for column in columns:
        if column not in merged.columns:
            merged[column] = None
    return merged[columns].sort_values("run_timestamp").reset_index(drop=True)


def _build_mart_model_registry(model_run_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "model_type",
        "model_family",
        "latest_run_id",
        "latest_profile",
        "latest_run_timestamp",
        "runs",
        "promoted_runs",
        "mean_val_top1",
        "mean_test_top1",
        "max_test_top1",
        "mean_backtest_top1",
        "mean_fit_seconds",
    ]
    if model_run_summary.empty:
        return _empty_frame(columns)
    registry = model_run_summary.copy()
    registry["val_top1"] = _to_numeric(registry.get("val_top1", pd.Series(index=registry.index)))
    registry["test_top1"] = _to_numeric(registry.get("test_top1", pd.Series(index=registry.index)))
    registry["mean_backtest_top1"] = _to_numeric(registry.get("mean_backtest_top1", pd.Series(index=registry.index)))
    registry["fit_seconds"] = _to_numeric(registry.get("fit_seconds", pd.Series(index=registry.index)))
    grouped = (
        registry.groupby(["model_name", "model_type"], dropna=False)
        .agg(
            model_family=("model_family", _series_mode),
            runs=("run_id", "nunique"),
            promoted_runs=("promoted", _count_truthy),
            mean_val_top1=("val_top1", "mean"),
            mean_test_top1=("test_top1", "mean"),
            max_test_top1=("test_top1", "max"),
            mean_backtest_top1=("mean_backtest_top1", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    latest_rows = (
        registry.sort_values(["run_timestamp", "run_id"], ascending=[False, False])
        .drop_duplicates(subset=["model_name", "model_type"], keep="first")[
            ["model_name", "model_type", "run_id", "profile", "run_timestamp"]
        ]
        .rename(
            columns={
                "run_id": "latest_run_id",
                "profile": "latest_profile",
                "run_timestamp": "latest_run_timestamp",
            }
        )
    )
    merged = grouped.merge(latest_rows, on=["model_name", "model_type"], how="left")
    for column in columns:
        if column not in merged.columns:
            merged[column] = None
    return merged[columns].sort_values(["mean_test_top1", "mean_val_top1"], ascending=[False, False]).reset_index(
        drop=True
    )


def _build_mart_ops_overview(
    ops_review_snapshot: pd.DataFrame,
    control_room_history: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "selected_run_id",
        "latest_run_id",
        "latest_run_best_model_name",
        "latest_run_serving_model_name",
        "ops_health_status",
        "operating_rhythm_status",
        "review_action_count",
        "high_priority_actions",
        "medium_priority_actions",
        "history_points",
        "history_window_start",
        "history_window_end",
        "mean_target_drift_jsd",
        "mean_stress_benchmark_skip_risk",
        "mean_selective_risk",
        "recommended_run_command",
    ]
    if ops_review_snapshot.empty:
        return _empty_frame(columns)
    snapshot = ops_review_snapshot.iloc[0].to_dict()
    history_window_start = None
    history_window_end = None
    mean_target_drift_jsd = None
    mean_stress_skip = None
    mean_selective_risk = None
    if not control_room_history.empty:
        history = control_room_history.copy()
        if "generated_at" in history.columns:
            history["generated_at"] = pd.to_datetime(history["generated_at"], errors="coerce")
            history_window_start = history["generated_at"].min()
            history_window_end = history["generated_at"].max()
        if "target_drift_jsd" in history.columns:
            mean_target_drift_jsd = _to_numeric(history["target_drift_jsd"]).mean()
        if "stress_benchmark_skip_risk" in history.columns:
            mean_stress_skip = _to_numeric(history["stress_benchmark_skip_risk"]).mean()
        if "test_selective_risk" in history.columns:
            mean_selective_risk = _to_numeric(history["test_selective_risk"]).mean()
    row = {
        "selected_run_id": snapshot.get("selected_run_id"),
        "latest_run_id": snapshot.get("latest_run_id"),
        "latest_run_best_model_name": snapshot.get("latest_run_best_model_name"),
        "latest_run_serving_model_name": snapshot.get("latest_run_serving_model_name"),
        "ops_health_status": snapshot.get("ops_health_status"),
        "operating_rhythm_status": snapshot.get("operating_rhythm_status"),
        "review_action_count": snapshot.get("review_action_count"),
        "high_priority_actions": snapshot.get("high_priority_actions"),
        "medium_priority_actions": snapshot.get("medium_priority_actions"),
        "history_points": snapshot.get("history_points"),
        "history_window_start": history_window_start,
        "history_window_end": history_window_end,
        "mean_target_drift_jsd": mean_target_drift_jsd,
        "mean_stress_benchmark_skip_risk": mean_stress_skip,
        "mean_selective_risk": mean_selective_risk,
        "recommended_run_command": snapshot.get("recommended_run_command"),
    }
    return pd.DataFrame([row], columns=columns)


def _build_mart_creator_opportunities(creator_ranking_opportunities: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "artist_name",
        "appearance_count",
        "family_count",
        "priority_now_count",
        "mean_opportunity_score",
        "max_opportunity_score",
        "best_opportunity_rank",
        "top_scene_name",
        "top_primary_driver",
        "example_seed_bridges",
        "example_why_now",
    ]
    if creator_ranking_opportunities.empty:
        return _empty_frame(columns)
    ranking = creator_ranking_opportunities.copy()
    ranking["opportunity_score"] = _to_numeric(ranking.get("opportunity_score", pd.Series(index=ranking.index)))
    ranking["opportunity_rank"] = _to_numeric(ranking.get("opportunity_rank", pd.Series(index=ranking.index)))
    grouped = (
        ranking.groupby("artist_name", dropna=False)
        .agg(
            appearance_count=("artist_name", "count"),
            family_count=("report_family_id", "nunique"),
            priority_now_count=(
                "opportunity_band",
                lambda values: sum(str(value) == "priority_now" for value in values),
            ),
            mean_opportunity_score=("opportunity_score", "mean"),
            max_opportunity_score=("opportunity_score", "max"),
            best_opportunity_rank=("opportunity_rank", "min"),
        )
        .reset_index()
    )
    top_rows = (
        ranking.sort_values(["artist_name", "opportunity_score"], ascending=[True, False])
        .drop_duplicates(subset=["artist_name"], keep="first")[
            ["artist_name", "scene_name", "primary_driver", "seed_bridges", "why_now"]
        ]
        .rename(
            columns={
                "scene_name": "top_scene_name",
                "primary_driver": "top_primary_driver",
                "seed_bridges": "example_seed_bridges",
                "why_now": "example_why_now",
            }
        )
    )
    merged = grouped.merge(top_rows, on="artist_name", how="left")
    for column in columns:
        if column not in merged.columns:
            merged[column] = None
    return merged[columns].sort_values(["max_opportunity_score", "appearance_count"], ascending=[False, False]).reset_index(
        drop=True
    )


def _build_mart_creator_scene_pressure(creator_scene_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scene_name",
        "family_count",
        "total_priority_now_count",
        "mean_opportunity_score",
        "max_top_opportunity_score",
        "mean_scene_local_play_share",
        "mean_scene_label_concentration",
        "mean_scene_release_pressure",
        "top_opportunity_artist_example",
        "top_seed_artists_example",
    ]
    if creator_scene_summary.empty:
        return _empty_frame(columns)
    scene = creator_scene_summary.copy()
    scene["avg_opportunity_score"] = _to_numeric(scene.get("avg_opportunity_score", pd.Series(index=scene.index)))
    scene["top_opportunity_score"] = _to_numeric(scene.get("top_opportunity_score", pd.Series(index=scene.index)))
    scene["scene_local_play_share"] = _to_numeric(scene.get("scene_local_play_share", pd.Series(index=scene.index)))
    scene["scene_label_concentration"] = _to_numeric(
        scene.get("scene_label_concentration", pd.Series(index=scene.index))
    )
    scene["scene_release_pressure"] = _to_numeric(
        scene.get("scene_release_pressure", pd.Series(index=scene.index))
    )
    grouped = (
        scene.groupby("scene_name", dropna=False)
        .agg(
            family_count=("report_family_id", "nunique"),
            total_priority_now_count=("priority_now_count", "sum"),
            mean_opportunity_score=("avg_opportunity_score", "mean"),
            max_top_opportunity_score=("top_opportunity_score", "max"),
            mean_scene_local_play_share=("scene_local_play_share", "mean"),
            mean_scene_label_concentration=("scene_label_concentration", "mean"),
            mean_scene_release_pressure=("scene_release_pressure", "mean"),
        )
        .reset_index()
    )
    top_rows = (
        scene.sort_values(["scene_name", "top_opportunity_score"], ascending=[True, False])
        .drop_duplicates(subset=["scene_name"], keep="first")[
            ["scene_name", "top_opportunity_artist", "top_seed_artists"]
        ]
        .rename(
            columns={
                "top_opportunity_artist": "top_opportunity_artist_example",
                "top_seed_artists": "top_seed_artists_example",
            }
        )
    )
    merged = grouped.merge(top_rows, on="scene_name", how="left")
    for column in columns:
        if column not in merged.columns:
            merged[column] = None
    return merged[columns].sort_values(
        ["mean_opportunity_score", "family_count"], ascending=[False, False]
    ).reset_index(drop=True)


def build_analytics_warehouse_bundle(
    *,
    data_dir: Path,
    output_dir: Path,
    include_video: bool,
    logger,
    raw_df: pd.DataFrame | None = None,
) -> AnalyticsWarehouseBundle:
    if raw_df is None:
        raw_df = load_streaming_history(data_dir, include_video=include_video, logger=logger)

    bronze_raw_streaming = _prepare_raw_streaming_history(raw_df)
    bronze_experiment = _prepare_experiment_history_frame(output_dir / "history" / "experiment_history.csv")
    bronze_backtest = _prepare_backtest_history_frame(output_dir / "history" / "backtest_history.csv")
    bronze_optuna = _prepare_optuna_history_frame(output_dir / "history" / "optuna_history.csv")
    bronze_benchmark = _prepare_benchmark_history_frame(output_dir / "history" / "benchmark_history.csv")
    bronze_run_manifests = _prepare_run_manifests(output_dir)
    bronze_run_results = _prepare_run_results(output_dir)
    bronze_robustness = rows_to_frame(collect_run_analysis_rows(output_dir, "robustness_summary.json"))
    bronze_policy = rows_to_frame(collect_run_analysis_rows(output_dir, "policy_simulation_summary.json"))
    bronze_moonshot = rows_to_frame(collect_run_analysis_rows(output_dir, "moonshot_summary.json"))
    bronze_control_room, bronze_review_actions = _control_room_snapshot_frames(output_dir)
    bronze_control_room_history = safe_read_csv(output_dir / "analytics" / "control_room_history.csv")
    (
        bronze_creator_report_families,
        bronze_creator_ranking,
        bronze_creator_scene,
        bronze_creator_scene_seed,
    ) = _load_creator_report_family_assets(output_dir)

    bronze = {
        "raw_streaming_history": bronze_raw_streaming,
        "experiment_history": bronze_experiment,
        "backtest_history": bronze_backtest,
        "benchmark_history": bronze_benchmark,
        "optuna_history": bronze_optuna,
        "run_manifests": bronze_run_manifests,
        "run_results": bronze_run_results,
        "robustness_summary": bronze_robustness,
        "policy_summary": bronze_policy,
        "moonshot_summary": bronze_moonshot,
        "control_room_snapshot": bronze_control_room,
        "control_room_review_actions": bronze_review_actions,
        "control_room_history": bronze_control_room_history,
        "creator_report_families": bronze_creator_report_families,
        "creator_ranking_opportunities": bronze_creator_ranking,
        "creator_scene_summary": bronze_creator_scene,
        "creator_scene_seed_summary": bronze_creator_scene_seed,
    }

    silver_listener_daily = _build_listener_daily_activity(bronze_raw_streaming)
    silver_model_run_summary = _build_model_run_summary(
        bronze_run_manifests,
        bronze_run_results,
        bronze_backtest,
    )
    silver_ops_review_snapshot = _build_ops_review_snapshot(
        bronze_control_room,
        bronze_control_room_history,
        bronze_review_actions,
    )
    silver_creator_report_summary = _build_creator_report_family_summary(
        bronze_creator_report_families,
        bronze_creator_ranking,
        bronze_creator_scene,
        bronze_creator_scene_seed,
    )

    silver = {
        "listener_daily_activity": silver_listener_daily,
        "model_run_summary": silver_model_run_summary,
        "ops_review_snapshot": silver_ops_review_snapshot,
        "creator_report_family_summary": silver_creator_report_summary,
    }

    gold_run_quality = _build_mart_run_quality(silver_model_run_summary)
    gold_model_registry = _build_mart_model_registry(silver_model_run_summary)
    gold_ops_overview = _build_mart_ops_overview(silver_ops_review_snapshot, bronze_control_room_history)
    gold_creator_opportunities = _build_mart_creator_opportunities(bronze_creator_ranking)
    gold_creator_scene_pressure = _build_mart_creator_scene_pressure(bronze_creator_scene)

    gold = {
        "mart_run_quality": gold_run_quality,
        "mart_model_registry": gold_model_registry,
        "mart_ops_overview": gold_ops_overview,
        "mart_creator_opportunities": gold_creator_opportunities,
        "mart_creator_scene_pressure": gold_creator_scene_pressure,
    }

    warehouse_root = output_dir / "analytics" / "warehouse"
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "warehouse_root": str(warehouse_root.resolve()),
        "summary": {
            "bronze_assets": len(bronze),
            "silver_assets": len(silver),
            "gold_assets": len(gold),
            "creator_report_family_count": int(len(bronze_creator_report_families.index)),
            "latest_control_room_run_id": (
                bronze_control_room.iloc[0]["latest_run_id"] if not bronze_control_room.empty else None
            ),
        },
        "lineage": [
            "bronze captures locally prepared raw history, run artifacts, control-room snapshots, and creator report-family exports.",
            "silver standardizes listener behavior, per-model run summaries, ops review status, and creator family summaries.",
            "gold exposes analytics marts for run quality, model registry, ops overview, and creator opportunity pressure.",
        ],
    }
    return AnalyticsWarehouseBundle(
        root=warehouse_root,
        bronze=bronze,
        silver=silver,
        gold=gold,
        manifest=manifest,
    )


def write_analytics_warehouse(bundle: AnalyticsWarehouseBundle, *, logger) -> Path:
    bundle.root.mkdir(parents=True, exist_ok=True)
    layer_payloads: dict[str, list[dict[str, object]]] = {}
    for layer_name, assets in [
        ("bronze", bundle.bronze),
        ("silver", bundle.silver),
        ("gold", bundle.gold),
    ]:
        layer_dir = bundle.root / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_entries: list[dict[str, object]] = []
        for asset_name, df in assets.items():
            asset_path = layer_dir / f"{asset_name}.parquet"
            storage_df = _normalize_frame_for_storage(df)
            storage_df.to_parquet(asset_path, index=False)
            layer_entries.append(
                {
                    "name": asset_name,
                    "path": str(asset_path.resolve()),
                    "rows": int(len(storage_df.index)),
                    "column_count": int(len(storage_df.columns)),
                    "columns": [str(column) for column in storage_df.columns],
                }
            )
        layer_payloads[layer_name] = layer_entries

    manifest = dict(bundle.manifest)
    manifest["layers"] = layer_payloads
    write_json(bundle.root / "warehouse_manifest.json", manifest, sort_keys=False)

    markdown_lines = [
        "# Analytics Warehouse",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Data dir: `{manifest['data_dir']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        "",
        "## Lineage",
        "",
    ]
    for line in manifest.get("lineage", []):
        markdown_lines.append(f"- {line}")
    for layer_name in ("bronze", "silver", "gold"):
        markdown_lines.extend(
            [
                "",
                f"## {layer_name.title()}",
                "",
                "| Asset | Rows | Columns | Path |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for asset in layer_payloads.get(layer_name, []):
            markdown_lines.append(
                f"| `{asset['name']}` | {asset['rows']} | {asset['column_count']} | `{Path(str(asset['path'])).name}` |"
            )
    write_markdown(bundle.root / "warehouse_manifest.md", markdown_lines)
    logger.info("Analytics warehouse refreshed: %s", bundle.root)
    return bundle.root


def build_analytics_warehouse(
    *,
    data_dir: Path,
    output_dir: Path,
    include_video: bool,
    logger,
    raw_df: pd.DataFrame | None = None,
) -> Path:
    bundle = build_analytics_warehouse_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=include_video,
        logger=logger,
        raw_df=raw_df,
    )
    return write_analytics_warehouse(bundle, logger=logger)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local analytics warehouse for the Spotify project.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing run artifacts.")
    parser.add_argument("--no-video", action="store_true", help="Exclude video history files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.analytics_warehouse")
    warehouse_root = build_analytics_warehouse(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        include_video=not bool(args.no_video),
        logger=logger,
    )
    print(warehouse_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
