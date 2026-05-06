from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype
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


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="boolean")
    normalized = series.map(
        lambda value: (
            True
            if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "y"}
            else (
                False
                if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", "n"}
                else (None if pd.isna(value) else bool(value))
            )
        )
    )
    return normalized.astype("boolean")


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
            frame[column] = series.map(lambda value: None if pd.isna(value) else str(value)).astype("string")
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
        ).astype("string")
    return frame


def _logical_type_from_dtype(dtype: object) -> str:
    if is_bool_dtype(dtype):
        return "boolean"
    if is_integer_dtype(dtype):
        return "integer"
    if is_float_dtype(dtype):
        return "float"
    if is_datetime64_any_dtype(dtype):
        return "timestamp"
    return "string"


def _schema_records(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for position, column in enumerate(df.columns, start=1):
        dtype = df[column].dtype
        records.append(
            {
                "name": str(column),
                "position": position,
                "dtype": str(dtype),
                "logical_type": _logical_type_from_dtype(dtype),
            }
        )
    return records


def _reindex_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _load_analysis_table_frame(
    path: Path,
    columns: list[str],
    *,
    datetime_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return _empty_frame(columns)
    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
    return _reindex_frame(df, columns)


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


def _creator_market_snapshot_frames(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scene_columns = [
        "scene_name",
        "family_count",
        "avg_scene_local_play_share",
        "avg_opportunity_score",
        "total_priority_now",
        "total_watchlist",
        "avg_release_pressure",
        "avg_label_concentration",
        "avg_inbound_target_share",
        "avg_outbound_source_share",
        "avg_seed_bridge_count",
        "dominant_driver",
        "top_opportunity_artist",
        "top_migration_route",
        "strategy_posture",
        "momentum_score",
    ]
    lane_columns = [
        "scene_name",
        "primary_driver",
        "family_count",
        "artist_count",
        "opportunity_count",
        "priority_now_count",
        "watchlist_count",
        "avg_opportunity_score",
        "avg_scene_local_play_share",
        "avg_scene_release_pressure",
        "avg_scene_label_concentration",
        "avg_seed_bridge_count",
        "avg_fan_migration_score",
        "avg_release_whitespace_score",
        "avg_local_gap_score",
        "avg_scene_momentum_score",
        "representative_artist",
        "lane_posture",
        "lane_attractiveness_score",
    ]
    migration_columns = [
        "source_artist",
        "target_artist",
        "family_count",
        "route_mentions",
        "total_transition_count",
        "avg_source_out_share",
        "avg_target_in_share",
        "source_scene_name",
        "target_scene_name",
        "route_strength_score",
    ]
    bridge_columns = [
        "scene_name",
        "seed_artist",
        "family_count",
        "opportunity_count",
        "avg_opportunity_score",
        "avg_bridge_artist_count",
        "avg_scene_local_play_share",
        "avg_scene_release_pressure",
        "avg_scene_label_concentration",
        "top_opportunity_artist",
        "dominant_driver",
        "bridge_score",
    ]
    whitespace_columns = [
        "artist_name",
        "scene_name",
        "family_count",
        "avg_opportunity_score",
        "avg_release_whitespace_score",
        "max_days_since_latest_release",
        "avg_seed_bridge_count",
        "dominant_labels",
        "primary_driver",
        "whitespace_signal_score",
    ]
    brief_columns = [
        "report_family_count",
        "top_scene_json",
        "top_lane_json",
        "top_route_json",
        "top_bridge_json",
        "top_whitespace_json",
        "summary_json",
        "actions_json",
        "raw_json",
    ]
    manifest_columns = [
        "report_family_count",
        "manifest_backed_report_family_count",
        "asset_backed_report_family_count",
        "complete_report_family_count",
        "partial_report_family_count",
        "partial_report_family_ids_json",
        "table_count",
        "artifact_root",
        "raw_json",
    ]
    base_dir = output_dir / "analysis" / "creator_market_intelligence"
    scene_market_pulse = _load_analysis_table_frame(base_dir / "scene_market_pulse.csv", scene_columns)
    opportunity_lane_atlas = _load_analysis_table_frame(base_dir / "opportunity_lane_atlas.csv", lane_columns)
    market_migration_network = _load_analysis_table_frame(base_dir / "market_migration_network.csv", migration_columns)
    seed_scene_bridge_atlas = _load_analysis_table_frame(base_dir / "seed_scene_bridge_atlas.csv", bridge_columns)
    release_whitespace_atlas = _load_analysis_table_frame(base_dir / "release_whitespace_atlas.csv", whitespace_columns)

    brief_payload = safe_read_json(base_dir / "creator_market_brief.json", default={})
    if isinstance(brief_payload, dict):
        brief_snapshot = pd.DataFrame(
            [
                {
                    "report_family_count": brief_payload.get("report_family_count"),
                    "top_scene_json": _json_string(brief_payload.get("top_scene")),
                    "top_lane_json": _json_string(brief_payload.get("top_lane")),
                    "top_route_json": _json_string(brief_payload.get("top_route")),
                    "top_bridge_json": _json_string(brief_payload.get("top_bridge")),
                    "top_whitespace_json": _json_string(brief_payload.get("top_whitespace")),
                    "summary_json": _json_string(brief_payload.get("summary")),
                    "actions_json": _json_string(brief_payload.get("actions")),
                    "raw_json": _json_string(brief_payload),
                }
            ],
            columns=brief_columns,
        )
    else:
        brief_snapshot = _empty_frame(brief_columns)

    manifest_payload = safe_read_json(base_dir / "creator_market_manifest.json", default={})
    if isinstance(manifest_payload, dict):
        tables_payload = manifest_payload.get("tables", {})
        table_count = len(tables_payload) if isinstance(tables_payload, dict) else 0
        manifest_snapshot = pd.DataFrame(
            [
                {
                    "report_family_count": manifest_payload.get("report_family_count"),
                    "manifest_backed_report_family_count": manifest_payload.get("manifest_backed_report_family_count"),
                    "asset_backed_report_family_count": manifest_payload.get("asset_backed_report_family_count"),
                    "complete_report_family_count": manifest_payload.get("complete_report_family_count"),
                    "partial_report_family_count": manifest_payload.get("partial_report_family_count"),
                    "partial_report_family_ids_json": _json_string(manifest_payload.get("partial_report_family_ids")),
                    "table_count": table_count,
                    "artifact_root": manifest_payload.get("artifact_root"),
                    "raw_json": _json_string(manifest_payload),
                }
            ],
            columns=manifest_columns,
        )
    else:
        manifest_snapshot = _empty_frame(manifest_columns)

    return (
        scene_market_pulse,
        opportunity_lane_atlas,
        market_migration_network,
        seed_scene_bridge_atlas,
        release_whitespace_atlas,
        brief_snapshot,
        manifest_snapshot,
    )


def _research_platform_snapshot_frames(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_registry_columns = [
        "run_id",
        "profile",
        "timestamp",
        "promoted",
        "champion_gate_status",
        "benchmark_protocol_present",
        "safety_platform_contract_present",
        "conformal_summary_count",
        "backtest_model_count",
        "benchmark_contract_version",
        "benchmark_comparison_mode",
        "safety_api_group_count",
        "spotify_wrapper_count",
        "portability_note_count",
        "portability_signal_status",
        "research_artifact_ratio",
        "research_stage",
        "claim_pack_attached",
        "claim_pack_path",
        "claim_pack_freshness_status",
        "claim_pack_stale_source_path",
        "claim_pack_stale_source_count",
        "run_manifest_path",
        "run_manifest_timestamp",
        "run_manifest_age_hours",
        "benchmark_protocol_path",
        "safety_platform_contract_path",
        "target_drift_jsd",
        "test_selective_risk",
        "test_abstention_rate",
        "robustness_gap",
        "stress_skip_risk",
        "ops_coverage_ratio",
    ]
    benchmark_columns = [
        "benchmark_id",
        "canonical_profile",
        "comparison_mode",
        "comparison_ready",
        "comparison_status",
        "run_count",
        "model_count",
        "present_artifact_count",
        "required_artifact_count",
        "required_artifact_ratio",
        "significant_pair_count",
        "comparison_blocker_count",
        "top_comparison_blocker",
        "comparison_blockers_json",
        "comparator_guard_status",
        "deep_comparator_ready",
        "observed_model_classes_json",
        "best_model_name",
        "best_model_type",
        "best_val_top1_mean",
        "best_test_top1_mean",
        "top_significant_pair",
        "top_significant_margin",
        "manifest_freshness_status",
        "manifest_stale_source_path",
        "manifest_stale_source_count",
        "manifest_age_hours",
        "summary_path",
        "significance_path",
        "benchmark_strength_score",
        "manifest_path",
    ]
    claim_columns = [
        "claim_key",
        "title",
        "role",
        "status",
        "claim_readiness_status",
        "summary",
        "live_signal_status",
        "benchmark_evidence_status",
        "repeated_evidence_status",
        "slice_evidence_status",
        "risk_evidence_status",
        "artifact_pack_status",
        "supporting_artifact_count",
        "existing_supporting_artifact_count",
        "missing_supporting_artifact_count",
        "stale_supporting_artifact_count",
        "supporting_artifact_path_status",
        "supporting_artifact_freshness_status",
        "missing_supporting_artifact_path",
        "stale_supporting_artifact_path",
        "missing_check_count",
        "blocked",
        "next_gate",
        "target_drift_jsd",
        "selective_risk",
        "stress_skip_risk",
        "live_test_top1_lift_vs_deep",
        "benchmark_comparison_ready",
        "benchmark_significant_lift",
        "claims_path",
        "metrics_json",
        "missing_checks_json",
    ]
    maturity_columns = [
        "anchor_run_id",
        "anchor_run_json",
        "strongest_benchmark_lock_json",
        "strongest_benchmark_id",
        "claim_ready_count",
        "claim_blocked_count",
        "claim_total_count",
        "incomplete_benchmark_lock_count",
        "stale_benchmark_manifest_count",
        "stale_claim_artifact_count",
        "submission_status",
        "ready_for_external_review",
        "blockers_json",
        "top_blocker",
        "top_next_gate",
        "summary_json",
        "actions_json",
        "raw_json",
    ]
    manifest_columns = [
        "anchor_run_id",
        "artifact_root",
        "table_count",
        "raw_json",
    ]
    base_dir = output_dir / "analysis" / "research_platform_lab"
    run_registry = _load_analysis_table_frame(
        base_dir / "run_research_registry.csv",
        run_registry_columns,
        datetime_columns=("timestamp", "run_manifest_timestamp"),
    )
    benchmark_lock_atlas = _load_analysis_table_frame(base_dir / "benchmark_lock_atlas.csv", benchmark_columns)
    claim_registry = _load_analysis_table_frame(base_dir / "research_claim_registry.csv", claim_columns)

    maturity_payload = safe_read_json(base_dir / "research_platform_maturity.json", default={})
    if isinstance(maturity_payload, dict):
        strongest_benchmark = (
            maturity_payload.get("strongest_benchmark_lock", {})
            if isinstance(maturity_payload.get("strongest_benchmark_lock"), dict)
            else {}
        )
        blockers = maturity_payload.get("blockers", [])
        top_blocker = None
        if isinstance(blockers, list):
            for blocker in blockers:
                text = str(blocker).strip()
                if text:
                    top_blocker = text
                    break
        maturity_snapshot = pd.DataFrame(
            [
                {
                    "anchor_run_id": maturity_payload.get("anchor_run_id"),
                    "anchor_run_json": _json_string(maturity_payload.get("anchor_run")),
                    "strongest_benchmark_lock_json": _json_string(maturity_payload.get("strongest_benchmark_lock")),
                    "strongest_benchmark_id": strongest_benchmark.get("benchmark_id"),
                    "claim_ready_count": maturity_payload.get("claim_ready_count"),
                    "claim_blocked_count": maturity_payload.get("claim_blocked_count"),
                    "claim_total_count": maturity_payload.get("claim_total_count"),
                    "incomplete_benchmark_lock_count": maturity_payload.get("incomplete_benchmark_lock_count"),
                    "stale_benchmark_manifest_count": maturity_payload.get("stale_benchmark_manifest_count"),
                    "stale_claim_artifact_count": maturity_payload.get("stale_claim_artifact_count"),
                    "submission_status": maturity_payload.get("submission_status"),
                    "ready_for_external_review": maturity_payload.get("ready_for_external_review"),
                    "blockers_json": _json_string(blockers),
                    "top_blocker": top_blocker,
                    "top_next_gate": maturity_payload.get("top_next_gate"),
                    "summary_json": _json_string(maturity_payload.get("summary")),
                    "actions_json": _json_string(maturity_payload.get("actions")),
                    "raw_json": _json_string(maturity_payload),
                }
            ],
            columns=maturity_columns,
        )
    else:
        maturity_snapshot = _empty_frame(maturity_columns)

    manifest_payload = safe_read_json(base_dir / "research_platform_manifest.json", default={})
    if isinstance(manifest_payload, dict):
        tables_payload = manifest_payload.get("tables", {})
        table_count = len(tables_payload) if isinstance(tables_payload, dict) else 0
        manifest_snapshot = pd.DataFrame(
            [
                {
                    "anchor_run_id": manifest_payload.get("anchor_run_id"),
                    "artifact_root": manifest_payload.get("artifact_root"),
                    "table_count": table_count,
                    "raw_json": _json_string(manifest_payload),
                }
            ],
            columns=manifest_columns,
        )
    else:
        manifest_snapshot = _empty_frame(manifest_columns)

    return run_registry, benchmark_lock_atlas, claim_registry, maturity_snapshot, manifest_snapshot


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


def _build_creator_market_scene_summary(
    scene_market_pulse: pd.DataFrame,
    opportunity_lane_atlas: pd.DataFrame,
    creator_market_brief_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scene_name",
        "family_count",
        "lane_count",
        "artist_count",
        "priority_now_count",
        "watchlist_count",
        "avg_scene_local_play_share",
        "avg_opportunity_score",
        "avg_release_pressure",
        "avg_label_concentration",
        "avg_inbound_target_share",
        "avg_seed_bridge_count",
        "dominant_driver",
        "representative_artist",
        "top_opportunity_artist",
        "top_migration_route",
        "strategy_posture",
        "lane_posture",
        "momentum_score",
        "lane_attractiveness_score",
        "report_family_count",
    ]
    if scene_market_pulse.empty:
        return _empty_frame(columns)
    summary = scene_market_pulse.copy().rename(
        columns={
            "total_priority_now": "priority_now_count",
            "total_watchlist": "watchlist_count",
        }
    )
    for column in [
        "family_count",
        "priority_now_count",
        "watchlist_count",
        "avg_scene_local_play_share",
        "avg_opportunity_score",
        "avg_release_pressure",
        "avg_label_concentration",
        "avg_inbound_target_share",
        "avg_seed_bridge_count",
        "momentum_score",
    ]:
        if column in summary.columns:
            summary[column] = _to_numeric(summary[column])

    if not opportunity_lane_atlas.empty and "scene_name" in opportunity_lane_atlas.columns:
        lane = opportunity_lane_atlas.copy()
        lane["artist_count"] = _to_numeric(lane.get("artist_count", pd.Series(index=lane.index)))
        lane["lane_attractiveness_score"] = _to_numeric(
            lane.get("lane_attractiveness_score", pd.Series(index=lane.index))
        )
        lane_grouped = (
            lane.groupby("scene_name", dropna=False)
            .agg(
                lane_count=("primary_driver", "count"),
                artist_count=("artist_count", "sum"),
                lane_attractiveness_score=("lane_attractiveness_score", "max"),
            )
            .reset_index()
        )
        lane_top = (
            lane.sort_values(["scene_name", "lane_attractiveness_score"], ascending=[True, False])
            .drop_duplicates(subset=["scene_name"], keep="first")[["scene_name", "representative_artist", "lane_posture"]]
        )
        summary = summary.merge(lane_grouped, on="scene_name", how="left").merge(lane_top, on="scene_name", how="left")

    report_family_count = None
    if not creator_market_brief_snapshot.empty:
        report_family_count = creator_market_brief_snapshot.iloc[0].get("report_family_count")
    summary["report_family_count"] = report_family_count
    for column in columns:
        if column not in summary.columns:
            summary[column] = None
    return summary[columns].sort_values(
        ["momentum_score", "priority_now_count", "avg_opportunity_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_research_platform_status_summary(
    run_research_registry: pd.DataFrame,
    benchmark_lock_atlas: pd.DataFrame,
    research_claim_registry: pd.DataFrame,
    research_platform_maturity_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "anchor_run_id",
        "anchor_profile",
        "anchor_timestamp",
        "anchor_research_stage",
        "submission_status",
        "ready_for_external_review",
        "claim_ready_count",
        "claim_blocked_count",
        "claim_total_count",
        "blocked_live_signal_count",
        "blocked_benchmark_gap_count",
        "incomplete_benchmark_lock_count",
        "comparison_ready_benchmark_count",
        "stale_benchmark_manifest_count",
        "stale_claim_artifact_count",
        "strongest_benchmark_id",
        "strongest_benchmark_score",
        "top_next_gate",
        "top_blocker",
    ]
    if (
        run_research_registry.empty
        and benchmark_lock_atlas.empty
        and research_claim_registry.empty
        and research_platform_maturity_snapshot.empty
    ):
        return _empty_frame(columns)

    maturity = research_platform_maturity_snapshot.iloc[0].to_dict() if not research_platform_maturity_snapshot.empty else {}
    anchor_run_id = str(maturity.get("anchor_run_id", "") or "").strip()

    anchor_profile = None
    anchor_timestamp = None
    anchor_research_stage = None
    if not run_research_registry.empty:
        run_registry = run_research_registry.copy()
        if "timestamp" in run_registry.columns:
            run_registry["timestamp"] = pd.to_datetime(run_registry["timestamp"], errors="coerce", format="mixed")
            run_registry = run_registry.sort_values(["timestamp", "run_id"], ascending=[False, False])
        anchor_row = run_registry.iloc[0]
        if anchor_run_id and "run_id" in run_registry.columns:
            matching = run_registry.loc[run_registry["run_id"].astype(str) == anchor_run_id]
            if not matching.empty:
                anchor_row = matching.iloc[0]
        anchor_profile = anchor_row.get("profile")
        anchor_timestamp = anchor_row.get("timestamp")
        anchor_research_stage = anchor_row.get("research_stage")

    blocked_live_signal_count = 0
    blocked_benchmark_gap_count = 0
    claim_ready_count = maturity.get("claim_ready_count")
    claim_blocked_count = maturity.get("claim_blocked_count")
    claim_total_count = maturity.get("claim_total_count")
    if not research_claim_registry.empty:
        claims = research_claim_registry.copy()
        claims["blocked"] = _to_bool_series(claims.get("blocked", pd.Series(index=claims.index)))
        claims["benchmark_comparison_ready"] = _to_bool_series(
            claims.get("benchmark_comparison_ready", pd.Series(index=claims.index))
        )
        live_status = claims.get("live_signal_status", pd.Series(index=claims.index, dtype="object")).astype(str).str.lower()
        blocked_live_signal_count = int(
            (claims["blocked"].fillna(False) & live_status.isin({"ready", "live", "pass", "supported"})).sum()
        )
        blocked_benchmark_gap_count = int(
            (claims["blocked"].fillna(False) & ~claims["benchmark_comparison_ready"].fillna(False)).sum()
        )
        if claim_ready_count is None:
            claim_ready_count = int(
                claims.get("claim_readiness_status", pd.Series(index=claims.index, dtype="object")).astype(str).str.lower().isin({"ready", "pass"}).sum()
            )
        if claim_blocked_count is None:
            claim_blocked_count = int(claims["blocked"].fillna(False).sum())
        if claim_total_count is None:
            claim_total_count = int(len(claims.index))

    comparison_ready_benchmark_count = 0
    strongest_benchmark_id = maturity.get("strongest_benchmark_id")
    strongest_benchmark_score = None
    if not benchmark_lock_atlas.empty:
        benchmarks = benchmark_lock_atlas.copy()
        benchmarks["comparison_ready"] = _to_bool_series(
            benchmarks.get("comparison_ready", pd.Series(index=benchmarks.index))
        )
        benchmarks["benchmark_strength_score"] = _to_numeric(
            benchmarks.get("benchmark_strength_score", pd.Series(index=benchmarks.index))
        )
        comparison_ready_benchmark_count = int(benchmarks["comparison_ready"].fillna(False).sum())
        strongest = benchmarks.sort_values(["benchmark_strength_score", "benchmark_id"], ascending=[False, True]).iloc[0]
        if not strongest_benchmark_id:
            strongest_benchmark_id = strongest.get("benchmark_id")
        strongest_benchmark_score = strongest.get("benchmark_strength_score")

    row = {
        "anchor_run_id": anchor_run_id or None,
        "anchor_profile": anchor_profile,
        "anchor_timestamp": anchor_timestamp,
        "anchor_research_stage": anchor_research_stage,
        "submission_status": maturity.get("submission_status"),
        "ready_for_external_review": maturity.get("ready_for_external_review"),
        "claim_ready_count": claim_ready_count,
        "claim_blocked_count": claim_blocked_count,
        "claim_total_count": claim_total_count,
        "blocked_live_signal_count": blocked_live_signal_count,
        "blocked_benchmark_gap_count": blocked_benchmark_gap_count,
        "incomplete_benchmark_lock_count": maturity.get("incomplete_benchmark_lock_count"),
        "comparison_ready_benchmark_count": comparison_ready_benchmark_count,
        "stale_benchmark_manifest_count": maturity.get("stale_benchmark_manifest_count"),
        "stale_claim_artifact_count": maturity.get("stale_claim_artifact_count"),
        "strongest_benchmark_id": strongest_benchmark_id,
        "strongest_benchmark_score": strongest_benchmark_score,
        "top_next_gate": maturity.get("top_next_gate"),
        "top_blocker": maturity.get("top_blocker"),
    }
    return pd.DataFrame([row], columns=columns)


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


def _build_mart_creator_market_watchlist(
    release_whitespace_atlas: pd.DataFrame,
    creator_market_scene_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "artist_name",
        "scene_name",
        "family_count",
        "avg_opportunity_score",
        "avg_release_whitespace_score",
        "whitespace_signal_score",
        "avg_seed_bridge_count",
        "max_days_since_latest_release",
        "dominant_labels",
        "primary_driver",
        "scene_priority_now_count",
        "scene_momentum_score",
        "scene_strategy_posture",
        "scene_dominant_driver",
        "market_priority_score",
    ]
    if release_whitespace_atlas.empty:
        return _empty_frame(columns)
    watchlist = release_whitespace_atlas.copy()
    for column in [
        "family_count",
        "avg_opportunity_score",
        "avg_release_whitespace_score",
        "whitespace_signal_score",
        "avg_seed_bridge_count",
        "max_days_since_latest_release",
    ]:
        if column in watchlist.columns:
            watchlist[column] = _to_numeric(watchlist[column])

    if not creator_market_scene_summary.empty and "scene_name" in creator_market_scene_summary.columns:
        scene_summary = creator_market_scene_summary.rename(
            columns={
                "priority_now_count": "scene_priority_now_count",
                "momentum_score": "scene_momentum_score",
                "strategy_posture": "scene_strategy_posture",
                "dominant_driver": "scene_dominant_driver",
            }
        )[
            [
                "scene_name",
                "scene_priority_now_count",
                "scene_momentum_score",
                "scene_strategy_posture",
                "scene_dominant_driver",
            ]
        ]
        watchlist = watchlist.merge(scene_summary, on="scene_name", how="left")

    watchlist["market_priority_score"] = (
        _to_numeric(watchlist.get("whitespace_signal_score", pd.Series(index=watchlist.index))).fillna(0.0) * 0.7
        + _to_numeric(watchlist.get("scene_momentum_score", pd.Series(index=watchlist.index))).fillna(0.0) * 0.3
        + _to_numeric(watchlist.get("scene_priority_now_count", pd.Series(index=watchlist.index))).fillna(0.0) * 0.05
    )
    for column in columns:
        if column not in watchlist.columns:
            watchlist[column] = None
    return watchlist[columns].sort_values(
        ["market_priority_score", "avg_opportunity_score", "max_days_since_latest_release"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_mart_research_platform_status(research_platform_status_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "anchor_run_id",
        "anchor_profile",
        "anchor_timestamp",
        "anchor_research_stage",
        "submission_status",
        "ready_for_external_review",
        "status_posture",
        "claim_ready_count",
        "claim_blocked_count",
        "claim_total_count",
        "incomplete_benchmark_lock_count",
        "comparison_ready_benchmark_count",
        "stale_benchmark_manifest_count",
        "stale_claim_artifact_count",
        "strongest_benchmark_id",
        "strongest_benchmark_score",
        "top_next_gate",
        "top_blocker",
    ]
    if research_platform_status_summary.empty:
        return _empty_frame(columns)
    status = research_platform_status_summary.copy()
    ready = _to_bool_series(status.get("ready_for_external_review", pd.Series(index=status.index))).fillna(False)
    blocked_claims = _to_numeric(status.get("claim_blocked_count", pd.Series(index=status.index))).fillna(0.0)
    incomplete_locks = _to_numeric(status.get("incomplete_benchmark_lock_count", pd.Series(index=status.index))).fillna(0.0)
    stale_artifacts = (
        _to_numeric(status.get("stale_claim_artifact_count", pd.Series(index=status.index))).fillna(0.0)
        + _to_numeric(status.get("stale_benchmark_manifest_count", pd.Series(index=status.index))).fillna(0.0)
    )
    status["status_posture"] = "attention"
    status.loc[(blocked_claims > 0) | (incomplete_locks > 0), "status_posture"] = "blocked"
    status.loc[ready & (blocked_claims <= 0) & (incomplete_locks <= 0) & (stale_artifacts <= 0), "status_posture"] = "ready"
    for column in columns:
        if column not in status.columns:
            status[column] = None
    return status[columns].reset_index(drop=True)


def _build_mart_research_claim_watchlist(
    research_claim_registry: pd.DataFrame,
    research_platform_status_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "claim_key",
        "title",
        "role",
        "status",
        "claim_readiness_status",
        "blocked",
        "missing_check_count",
        "next_gate",
        "live_signal_status",
        "benchmark_evidence_status",
        "benchmark_comparison_ready",
        "benchmark_significant_lift",
        "artifact_pack_status",
        "supporting_artifact_count",
        "missing_supporting_artifact_count",
        "stale_supporting_artifact_count",
        "supporting_artifact_freshness_status",
        "target_drift_jsd",
        "selective_risk",
        "stress_skip_risk",
        "live_test_top1_lift_vs_deep",
        "submission_status",
        "ready_for_external_review",
        "watchlist_score",
    ]
    if research_claim_registry.empty:
        return _empty_frame(columns)
    watchlist = research_claim_registry.copy()
    watchlist["blocked"] = _to_bool_series(watchlist.get("blocked", pd.Series(index=watchlist.index)))
    watchlist["benchmark_comparison_ready"] = _to_bool_series(
        watchlist.get("benchmark_comparison_ready", pd.Series(index=watchlist.index))
    )
    watchlist["benchmark_significant_lift"] = _to_bool_series(
        watchlist.get("benchmark_significant_lift", pd.Series(index=watchlist.index))
    )
    for column in [
        "missing_check_count",
        "supporting_artifact_count",
        "missing_supporting_artifact_count",
        "stale_supporting_artifact_count",
        "target_drift_jsd",
        "selective_risk",
        "stress_skip_risk",
        "live_test_top1_lift_vs_deep",
    ]:
        if column in watchlist.columns:
            watchlist[column] = _to_numeric(watchlist[column])

    if not research_platform_status_summary.empty:
        summary_row = research_platform_status_summary.iloc[0]
        watchlist["submission_status"] = summary_row.get("submission_status")
        watchlist["ready_for_external_review"] = summary_row.get("ready_for_external_review")

    watchlist["watchlist_score"] = (
        watchlist["blocked"].fillna(False).astype("int64") * 100.0
        + watchlist.get("missing_check_count", pd.Series(index=watchlist.index)).fillna(0.0) * 10.0
        + watchlist.get("stale_supporting_artifact_count", pd.Series(index=watchlist.index)).fillna(0.0) * 5.0
        + watchlist.get("selective_risk", pd.Series(index=watchlist.index)).fillna(0.0) * 10.0
        + watchlist.get("stress_skip_risk", pd.Series(index=watchlist.index)).fillna(0.0) * 10.0
    )
    for column in columns:
        if column not in watchlist.columns:
            watchlist[column] = None
    return watchlist[columns].sort_values(
        ["watchlist_score", "missing_check_count", "claim_key"],
        ascending=[False, False, True],
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
    (
        bronze_creator_market_scene_pulse,
        bronze_creator_market_opportunity_lane_atlas,
        bronze_creator_market_migration_network,
        bronze_creator_market_seed_bridge_atlas,
        bronze_creator_market_release_whitespace_atlas,
        bronze_creator_market_brief_snapshot,
        bronze_creator_market_manifest_snapshot,
    ) = _creator_market_snapshot_frames(output_dir)
    (
        bronze_research_platform_run_registry,
        bronze_research_platform_benchmark_lock_atlas,
        bronze_research_platform_claim_registry,
        bronze_research_platform_maturity_snapshot,
        bronze_research_platform_manifest_snapshot,
    ) = _research_platform_snapshot_frames(output_dir)

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
        "creator_market_scene_pulse": bronze_creator_market_scene_pulse,
        "creator_market_opportunity_lane_atlas": bronze_creator_market_opportunity_lane_atlas,
        "creator_market_migration_network": bronze_creator_market_migration_network,
        "creator_market_seed_bridge_atlas": bronze_creator_market_seed_bridge_atlas,
        "creator_market_release_whitespace_atlas": bronze_creator_market_release_whitespace_atlas,
        "creator_market_brief_snapshot": bronze_creator_market_brief_snapshot,
        "creator_market_manifest_snapshot": bronze_creator_market_manifest_snapshot,
        "research_platform_run_registry": bronze_research_platform_run_registry,
        "research_platform_benchmark_lock_atlas": bronze_research_platform_benchmark_lock_atlas,
        "research_platform_claim_registry": bronze_research_platform_claim_registry,
        "research_platform_maturity_snapshot": bronze_research_platform_maturity_snapshot,
        "research_platform_manifest_snapshot": bronze_research_platform_manifest_snapshot,
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
    silver_creator_market_scene_summary = _build_creator_market_scene_summary(
        bronze_creator_market_scene_pulse,
        bronze_creator_market_opportunity_lane_atlas,
        bronze_creator_market_brief_snapshot,
    )
    silver_research_platform_status_summary = _build_research_platform_status_summary(
        bronze_research_platform_run_registry,
        bronze_research_platform_benchmark_lock_atlas,
        bronze_research_platform_claim_registry,
        bronze_research_platform_maturity_snapshot,
    )

    silver = {
        "listener_daily_activity": silver_listener_daily,
        "model_run_summary": silver_model_run_summary,
        "ops_review_snapshot": silver_ops_review_snapshot,
        "creator_report_family_summary": silver_creator_report_summary,
        "creator_market_scene_summary": silver_creator_market_scene_summary,
        "research_platform_status_summary": silver_research_platform_status_summary,
    }

    gold_run_quality = _build_mart_run_quality(silver_model_run_summary)
    gold_model_registry = _build_mart_model_registry(silver_model_run_summary)
    gold_ops_overview = _build_mart_ops_overview(silver_ops_review_snapshot, bronze_control_room_history)
    gold_creator_opportunities = _build_mart_creator_opportunities(bronze_creator_ranking)
    gold_creator_scene_pressure = _build_mart_creator_scene_pressure(bronze_creator_scene)
    gold_creator_market_watchlist = _build_mart_creator_market_watchlist(
        bronze_creator_market_release_whitespace_atlas,
        silver_creator_market_scene_summary,
    )
    gold_research_platform_status = _build_mart_research_platform_status(
        silver_research_platform_status_summary
    )
    gold_research_claim_watchlist = _build_mart_research_claim_watchlist(
        bronze_research_platform_claim_registry,
        silver_research_platform_status_summary,
    )

    gold = {
        "mart_run_quality": gold_run_quality,
        "mart_model_registry": gold_model_registry,
        "mart_ops_overview": gold_ops_overview,
        "mart_creator_opportunities": gold_creator_opportunities,
        "mart_creator_scene_pressure": gold_creator_scene_pressure,
        "mart_creator_market_watchlist": gold_creator_market_watchlist,
        "mart_research_platform_status": gold_research_platform_status,
        "mart_research_claim_watchlist": gold_research_claim_watchlist,
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
            "creator_market_report_family_count": (
                int(bronze_creator_market_brief_snapshot.iloc[0]["report_family_count"])
                if not bronze_creator_market_brief_snapshot.empty
                and pd.notna(bronze_creator_market_brief_snapshot.iloc[0]["report_family_count"])
                else 0
            ),
            "research_platform_anchor_run_id": (
                bronze_research_platform_maturity_snapshot.iloc[0]["anchor_run_id"]
                if not bronze_research_platform_maturity_snapshot.empty
                else None
            ),
            "latest_control_room_run_id": (
                bronze_control_room.iloc[0]["latest_run_id"] if not bronze_control_room.empty else None
            ),
        },
        "lineage": [
            "bronze captures locally prepared raw history, run artifacts, control-room snapshots, creator report-family exports, creator-market branch outputs, and research-platform branch outputs.",
            "silver standardizes listener behavior, per-model run summaries, ops review status, creator family summaries, creator-market scene rollups, and research-platform status rollups.",
            "gold exposes analytics marts for run quality, model registry, ops overview, creator opportunity pressure, creator-market watchlists, and research-platform status tracking.",
        ],
    }
    return AnalyticsWarehouseBundle(
        root=warehouse_root,
        bronze=bronze,
        silver=silver,
        gold=gold,
        manifest=manifest,
    )


def load_analytics_warehouse_manifest(warehouse_root: Path) -> dict[str, Any]:
    manifest = safe_read_json(warehouse_root / "warehouse_manifest.json", default={})
    return manifest if isinstance(manifest, dict) else {}


def warehouse_manifest_frames(warehouse_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_columns = [
        "manifest_generated_at",
        "layer",
        "asset_name",
        "object_name",
        "parquet_path",
        "expected_rows",
        "expected_column_count",
        "expected_columns_json",
        "schema_json",
    ]
    column_columns = [
        "manifest_generated_at",
        "layer",
        "asset_name",
        "object_name",
        "column_position",
        "column_name",
        "dtype",
        "logical_type",
    ]
    manifest = load_analytics_warehouse_manifest(warehouse_root)
    generated_at = str(manifest.get("generated_at", "") or "")
    layers = manifest.get("layers", {})
    asset_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []

    if not isinstance(layers, dict):
        return pd.DataFrame(columns=asset_columns), pd.DataFrame(columns=column_columns)

    for layer_name, assets in layers.items():
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            asset_name = str(asset.get("name", "") or "")
            schema = asset.get("schema", [])
            schema_records = [record for record in schema if isinstance(record, dict)] if isinstance(schema, list) else []
            expected_columns = [str(record.get("name", "") or "") for record in schema_records]
            asset_rows.append(
                {
                    "manifest_generated_at": generated_at,
                    "layer": str(layer_name),
                    "asset_name": asset_name,
                    "object_name": asset_name,
                    "parquet_path": str(asset.get("path", "") or ""),
                    "expected_rows": int(asset.get("rows", 0) or 0),
                    "expected_column_count": int(asset.get("column_count", 0) or 0),
                    "expected_columns_json": json.dumps(expected_columns),
                    "schema_json": json.dumps(schema_records),
                }
            )
            for record in schema_records:
                column_rows.append(
                    {
                        "manifest_generated_at": generated_at,
                        "layer": str(layer_name),
                        "asset_name": asset_name,
                        "object_name": asset_name,
                        "column_position": int(record.get("position", 0) or 0),
                        "column_name": str(record.get("name", "") or ""),
                        "dtype": str(record.get("dtype", "") or ""),
                        "logical_type": str(record.get("logical_type", "") or ""),
                    }
                )

    return pd.DataFrame(asset_rows, columns=asset_columns), pd.DataFrame(column_rows, columns=column_columns)


def verify_analytics_warehouse_artifacts(warehouse_root: Path, *, logger) -> dict[str, Any]:
    asset_manifest, _ = warehouse_manifest_frames(warehouse_root)
    results: list[dict[str, object]] = []
    failures: list[str] = []

    for asset in asset_manifest.to_dict(orient="records"):
        asset_name = str(asset.get("asset_name", "") or "")
        parquet_path = Path(str(asset.get("parquet_path", "") or ""))
        expected_rows = int(asset.get("expected_rows", 0) or 0)
        expected_columns = json.loads(str(asset.get("expected_columns_json", "[]") or "[]"))
        expected_schema = json.loads(str(asset.get("schema_json", "[]") or "[]"))
        expected_logical_types = [str(record.get("logical_type", "") or "") for record in expected_schema]
        actual_rows: int | None = None
        actual_columns: list[str] = []
        actual_logical_types: list[str] = []
        error_message = ""
        row_count_match = False
        column_match = False
        logical_type_match = False

        try:
            frame = pd.read_parquet(parquet_path)
            actual_rows = int(len(frame.index))
            actual_columns = [str(column) for column in frame.columns]
            actual_logical_types = [str(record["logical_type"]) for record in _schema_records(frame)]
            row_count_match = actual_rows == expected_rows
            column_match = actual_columns == expected_columns
            logical_type_match = actual_logical_types == expected_logical_types
        except Exception as exc:
            error_message = str(exc)

        status = "pass" if row_count_match and column_match and logical_type_match and not error_message else "fail"
        results.append(
            {
                "layer": str(asset.get("layer", "") or ""),
                "asset_name": asset_name,
                "parquet_path": str(parquet_path),
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "row_count_match": row_count_match,
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "column_match": column_match,
                "expected_logical_types": expected_logical_types,
                "actual_logical_types": actual_logical_types,
                "logical_type_match": logical_type_match,
                "status": status,
                "error": error_message,
            }
        )
        if status != "pass":
            failures.append(asset_name)

    verification_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "warehouse_root": str(warehouse_root.resolve()),
        "status": "pass" if not failures else "fail",
        "summary": {
            "checked_assets": int(len(results)),
            "passed_assets": int(sum(1 for row in results if row["status"] == "pass")),
            "failed_assets": int(sum(1 for row in results if row["status"] != "pass")),
        },
        "results": results,
    }
    write_json(warehouse_root / "warehouse_verification.json", verification_payload, sort_keys=False)

    markdown_lines = [
        "# Analytics Warehouse Verification",
        "",
        f"- Generated at: `{verification_payload['generated_at']}`",
        f"- Warehouse root: `{verification_payload['warehouse_root']}`",
        f"- Status: `{verification_payload['status']}`",
        "",
        "| Asset | Rows | Columns | Logical types | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in results:
        markdown_lines.append(
            "| `{asset_name}` | {row_count_match} | {column_match} | {logical_type_match} | `{status}` |".format(
                asset_name=row["asset_name"],
                row_count_match="pass" if row["row_count_match"] else "fail",
                column_match="pass" if row["column_match"] else "fail",
                logical_type_match="pass" if row["logical_type_match"] else "fail",
                status=row["status"],
            )
        )
        if row["error"]:
            markdown_lines.append(f"| `{row['asset_name']}` error | `{row['error']}` |  |  |  |")
    write_markdown(warehouse_root / "warehouse_verification.md", markdown_lines)

    if failures:
        raise ValueError(
            "Analytics warehouse verification failed for assets: " + ", ".join(sorted(set(failures)))
        )

    logger.info(
        "Analytics warehouse artifacts verified: %s (%d assets)",
        warehouse_root,
        len(results),
    )
    return verification_payload


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
            schema = _schema_records(storage_df)
            layer_entries.append(
                {
                    "name": asset_name,
                    "path": str(asset_path.resolve()),
                    "rows": int(len(storage_df.index)),
                    "column_count": int(len(storage_df.columns)),
                    "columns": [str(column) for column in storage_df.columns],
                    "schema": schema,
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
    verify_analytics_warehouse_artifacts(bundle.root, logger=logger)
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
