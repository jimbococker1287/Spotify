from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .run_artifacts import safe_read_csv
from .run_artifacts import safe_read_json
from .run_artifacts import write_csv_rows
from .run_artifacts import write_json
from .run_artifacts import write_markdown


_ASSET_SUFFIXES = (
    "scene_comparison",
    "ranking_comparison",
    "opportunities",
    "migration_watch",
    "scene_seed_comparison",
    "seed_comparison",
)
_ASSET_SUFFIXES_BY_LENGTH = tuple(sorted(_ASSET_SUFFIXES, key=len, reverse=True))
_TREND_DELTA_COLUMNS = [
    "signal_type",
    "signal_key",
    "category_rank",
    "scene_name",
    "primary_driver",
    "source_artist",
    "target_artist",
    "family_count",
    "repeat_count",
    "first_report_family_id",
    "latest_report_family_id",
    "first_value",
    "latest_value",
    "delta_value",
    "coverage_ratio",
    "metadata_row_count",
    "opportunity_row_count",
    "trend_score",
    "severity",
    "comparison_basis",
    "trend_note",
    "action_hint",
]
_STALE_RELEASE_DAYS = 180.0


@dataclass
class _ReportFamilyRecord:
    has_manifest: bool = False
    manifest_path: Path | None = None
    family_timestamp: str = ""
    timestamp_source: str = ""
    asset_paths: dict[str, Path] = field(default_factory=dict)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _normalize_timestamp(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return ""
    normalized = text.removesuffix("Z") + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).isoformat()
    except Exception:
        return text


def _path_modified_timestamp(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _manifest_family_timestamp(payload: object, path: Path) -> tuple[str, str]:
    if isinstance(payload, dict):
        packaging = payload.get("packaging_metadata", {})
        packaging = packaging if isinstance(packaging, dict) else {}
        for key in [
            "normalized_at",
            "refreshed_at",
            "packaged_at",
            "generated_at",
            "created_at",
            "timestamp",
        ]:
            timestamp = _normalize_timestamp(packaging.get(key))
            if timestamp:
                return timestamp, f"manifest_packaging_metadata.{key}"
        for key in ["generated_at", "created_at", "timestamp"]:
            timestamp = _normalize_timestamp(payload.get(key))
            if timestamp:
                return timestamp, f"manifest.{key}"
    timestamp = _path_modified_timestamp(path)
    return (timestamp, "manifest_mtime") if timestamp else ("", "")


def _set_family_timestamp(record: _ReportFamilyRecord, timestamp: str, source: str, *, prefer: bool = False) -> None:
    if not timestamp:
        return
    if prefer or not record.family_timestamp:
        record.family_timestamp = timestamp
        record.timestamp_source = source


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _rows_for_columns(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    trimmed = frame.copy()
    for column in columns:
        if column not in trimmed.columns:
            trimmed[column] = None
    return trimmed[columns].to_dict(orient="records")


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


def _mode_text(series: pd.Series) -> str:
    normalized = [str(value).strip() for value in series if str(value).strip()]
    if not normalized:
        return ""
    counts = pd.Series(normalized).value_counts()
    return str(counts.index[0]) if not counts.empty else ""


def _parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    return [piece.strip() for piece in text.split("|") if piece.strip()]


def _series_or_default(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame.index), index=frame.index, dtype="object")


def _asset_root(output_dir: Path) -> Path:
    return output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"


def _parse_asset_path(path: Path) -> tuple[str, str] | None:
    if path.suffix.casefold() != ".csv":
        return None
    for suffix in _ASSET_SUFFIXES_BY_LENGTH:
        marker = f"_{suffix}"
        if not path.stem.endswith(marker):
            continue
        family_id = path.stem[: -len(marker)]
        if family_id:
            return family_id, suffix
    return None


def _collect_report_family_inventory(base_dir: Path) -> dict[str, _ReportFamilyRecord]:
    inventory: dict[str, _ReportFamilyRecord] = {}
    if not base_dir.exists():
        return inventory
    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith("_report_family.json"):
            family_id = path.stem.removesuffix("_report_family")
            if family_id:
                record = inventory.setdefault(family_id, _ReportFamilyRecord())
                record.has_manifest = True
                record.manifest_path = path
                timestamp, source = _manifest_family_timestamp(safe_read_json(path, default={}), path)
                _set_family_timestamp(record, timestamp, source, prefer=True)
            continue
        parsed = _parse_asset_path(path)
        if parsed is None:
            continue
        family_id, suffix = parsed
        record = inventory.setdefault(family_id, _ReportFamilyRecord())
        record.asset_paths.setdefault(suffix, path)
        _set_family_timestamp(record, _path_modified_timestamp(path), f"{suffix}_mtime")
    return inventory


def _report_family_manifest_counts(inventory: dict[str, _ReportFamilyRecord]) -> dict[str, object]:
    report_family_ids = sorted(inventory)
    manifest_backed_family_ids = sorted(family_id for family_id, record in inventory.items() if record.has_manifest)
    asset_backed_family_ids = sorted(family_id for family_id, record in inventory.items() if record.asset_paths)
    complete_family_ids = sorted(
        family_id
        for family_id, record in inventory.items()
        if record.has_manifest and all(suffix in record.asset_paths for suffix in _ASSET_SUFFIXES)
    )
    partial_family_ids = sorted(set(report_family_ids) - set(complete_family_ids))
    return {
        "report_family_ids": report_family_ids,
        "report_family_count": len(report_family_ids),
        "manifest_backed_report_family_count": len(manifest_backed_family_ids),
        "asset_backed_report_family_count": len(asset_backed_family_ids),
        "complete_report_family_count": len(complete_family_ids),
        "partial_report_family_count": len(partial_family_ids),
        "partial_report_family_ids": partial_family_ids,
    }


def _discover_report_family_ids(base_dir: Path) -> list[str]:
    return sorted(_collect_report_family_inventory(base_dir))


def _report_family_order_context(
    family_inventory: dict[str, _ReportFamilyRecord],
) -> tuple[dict[str, int], str]:
    if not family_inventory:
        return {}, "report_family_id"
    family_ids = sorted(family_inventory)
    if all(str(family_inventory[family_id].family_timestamp).strip() for family_id in family_ids):
        ordered = sorted(
            family_ids,
            key=lambda family_id: (
                str(family_inventory[family_id].family_timestamp),
                family_id,
            ),
        )
        return {family_id: index for index, family_id in enumerate(ordered)}, "family_timestamp"
    return {family_id: index for index, family_id in enumerate(family_ids)}, "report_family_id"


def _load_creator_assets(
    output_dir: Path,
    *,
    family_inventory: dict[str, _ReportFamilyRecord] | None = None,
) -> dict[str, pd.DataFrame]:
    base_dir = _asset_root(output_dir)
    frames: dict[str, pd.DataFrame] = {}
    if not base_dir.exists():
        return {suffix: pd.DataFrame() for suffix in _ASSET_SUFFIXES}
    inventory = family_inventory if family_inventory is not None else _collect_report_family_inventory(base_dir)
    for suffix in _ASSET_SUFFIXES:
        buckets: list[pd.DataFrame] = []
        for family_id in sorted(inventory):
            path = inventory[family_id].asset_paths.get(suffix)
            if path is None:
                continue
            frame = safe_read_csv(path)
            if frame.empty:
                continue
            frame.insert(0, "report_family_id", family_id)
            frame.insert(1, "seed_group_slug", family_id.removeprefix("creator_label_intelligence_"))
            buckets.append(frame)
        frames[suffix] = pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame()
    return frames


def _scene_posture(row: pd.Series, *, thresholds: dict[str, float]) -> str:
    inbound = _safe_float(row.get("avg_inbound_target_share"))
    release_pressure = _safe_float(row.get("avg_release_pressure"))
    concentration = _safe_float(row.get("avg_label_concentration"))
    play_share = _safe_float(row.get("avg_scene_local_play_share"))
    opportunity = _safe_float(row.get("avg_opportunity_score"))
    if math.isfinite(concentration) and concentration >= thresholds["concentration_high"]:
        return "label_risk_watch"
    if (
        math.isfinite(inbound)
        and inbound >= thresholds["inbound_high"]
        and math.isfinite(opportunity)
        and opportunity >= thresholds["opportunity_mid"]
    ):
        return "migration_capture"
    if (
        math.isfinite(release_pressure)
        and release_pressure >= thresholds["release_high"]
        and math.isfinite(play_share)
        and play_share >= thresholds["play_mid"]
    ):
        return "timing_window_open"
    if math.isfinite(play_share) and play_share >= thresholds["play_high"]:
        return "scale_scene_now"
    return "cultivate_selectively"


def _lane_posture(row: pd.Series) -> str:
    whitespace = _safe_float(row.get("avg_release_whitespace_score"))
    migration = _safe_float(row.get("avg_fan_migration_score"))
    seed_bridges = _safe_float(row.get("avg_seed_bridge_count"))
    scene_momentum = _safe_float(row.get("avg_scene_momentum_score"))
    local_gap = _safe_float(row.get("avg_local_gap_score"))
    primary_driver = str(row.get("primary_driver", "")).strip()

    candidates = [
        ("release_window", whitespace),
        ("migration_capture", migration),
        ("seed_bridge_expand", seed_bridges),
        ("scene_momentum", scene_momentum),
        ("gap_capture", local_gap),
    ]
    best_label, best_score = max(candidates, key=lambda item: item[1] if math.isfinite(item[1]) else -1.0)
    if primary_driver == "seed_adjacency" and math.isfinite(seed_bridges) and seed_bridges >= 1.5:
        return "seed_bridge_expand"
    if math.isfinite(best_score) and best_score > 0.0:
        return best_label
    return primary_driver or "balanced_watch"


def _build_scene_market_pulse(
    *,
    scene_df: pd.DataFrame,
    opportunities_df: pd.DataFrame,
    migration_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
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
    if scene_df.empty:
        return pd.DataFrame(columns=columns)

    scene = scene_df.copy()
    for column in [
        "scene_id",
        "scene_local_play_share",
        "avg_opportunity_score",
        "priority_now_count",
        "watchlist_count",
        "scene_release_pressure",
        "scene_label_concentration",
        "top_opportunity_score",
    ]:
        scene[column] = pd.to_numeric(scene.get(column), errors="coerce")

    grouped = (
        scene.groupby("scene_name", dropna=False)
        .agg(
            family_count=("report_family_id", "nunique"),
            avg_scene_local_play_share=("scene_local_play_share", "mean"),
            avg_opportunity_score=("avg_opportunity_score", "mean"),
            total_priority_now=("priority_now_count", "sum"),
            total_watchlist=("watchlist_count", "sum"),
            avg_release_pressure=("scene_release_pressure", "mean"),
            avg_label_concentration=("scene_label_concentration", "mean"),
        )
        .reset_index()
    )
    top_scene_artist = (
        scene.sort_values(["scene_name", "top_opportunity_score"], ascending=[True, False])
        .drop_duplicates(subset=["scene_name"], keep="first")[["scene_name", "top_opportunity_artist"]]
    )
    grouped = grouped.merge(top_scene_artist, on="scene_name", how="left")

    if not opportunities_df.empty:
        opportunities = opportunities_df.copy()
        for column in [
            "opportunity_score",
            "fan_migration_score",
            "release_whitespace_score",
            "scene_momentum_score",
            "local_gap_score",
        ]:
            opportunities[column] = pd.to_numeric(opportunities.get(column), errors="coerce")
        opportunities["seed_bridge_count"] = _series_or_default(opportunities, "connected_seed_artists", []).map(_parse_json_list).map(len)
        opportunity_rollup = (
            opportunities.groupby("scene_name", dropna=False)
            .agg(
                avg_seed_bridge_count=("seed_bridge_count", "mean"),
                dominant_driver=("primary_driver", _mode_text),
            )
            .reset_index()
        )
        grouped = grouped.merge(opportunity_rollup, on="scene_name", how="left")
    else:
        grouped["avg_seed_bridge_count"] = np.nan
        grouped["dominant_driver"] = ""

    if not migration_df.empty and "scene_id" in scene.columns:
        lookup = (
            scene[["report_family_id", "scene_id", "scene_name"]]
            .dropna(subset=["scene_id"])
            .drop_duplicates()
        )
        migration = migration_df.copy()
        for column in ["source_scene_id", "target_scene_id", "source_out_share", "target_in_share", "transition_count"]:
            migration[column] = pd.to_numeric(migration.get(column), errors="coerce")
        migration = migration.merge(
            lookup.rename(columns={"scene_id": "target_scene_id", "scene_name": "target_scene_name"}),
            on=["report_family_id", "target_scene_id"],
            how="left",
        )
        migration["route_name"] = (
            migration.get("source_artist", "").astype(str).str.strip()
            + " -> "
            + migration.get("target_artist", "").astype(str).str.strip()
        )
        migration_by_scene = (
            migration.loc[migration["target_scene_name"].astype(str).str.strip().ne("")]
            .groupby("target_scene_name", dropna=False)
            .agg(
                avg_inbound_target_share=("target_in_share", "mean"),
                avg_outbound_source_share=("source_out_share", "mean"),
            )
            .reset_index()
            .rename(columns={"target_scene_name": "scene_name"})
        )
        top_routes = (
            migration.loc[migration["target_scene_name"].astype(str).str.strip().ne("")]
            .sort_values(["target_scene_name", "source_out_share", "transition_count"], ascending=[True, False, False])
            .drop_duplicates(subset=["target_scene_name"], keep="first")[["target_scene_name", "route_name"]]
            .rename(columns={"target_scene_name": "scene_name", "route_name": "top_migration_route"})
        )
        grouped = grouped.merge(migration_by_scene, on="scene_name", how="left")
        grouped = grouped.merge(top_routes, on="scene_name", how="left")
    else:
        grouped["avg_inbound_target_share"] = np.nan
        grouped["avg_outbound_source_share"] = np.nan
        grouped["top_migration_route"] = ""

    for column in [
        "avg_scene_local_play_share",
        "avg_opportunity_score",
        "total_priority_now",
        "avg_inbound_target_share",
        "avg_label_concentration",
    ]:
        grouped[column] = pd.to_numeric(grouped.get(column), errors="coerce")

    grouped["momentum_score"] = (
        0.30 * _normalize_series(grouped["avg_opportunity_score"], higher_is_better=True).fillna(0.0)
        + 0.25 * _normalize_series(grouped["avg_scene_local_play_share"], higher_is_better=True).fillna(0.0)
        + 0.20 * _normalize_series(grouped["total_priority_now"], higher_is_better=True).fillna(0.0)
        + 0.15 * _normalize_series(grouped["avg_inbound_target_share"], higher_is_better=True).fillna(0.0)
        + 0.10 * _normalize_series(grouped["avg_label_concentration"], higher_is_better=False).fillna(0.0)
    )

    thresholds = {
        "concentration_high": float(grouped["avg_label_concentration"].quantile(0.75)) if grouped["avg_label_concentration"].notna().any() else 0.0,
        "inbound_high": float(grouped["avg_inbound_target_share"].quantile(0.75)) if grouped["avg_inbound_target_share"].notna().any() else 0.0,
        "release_high": float(grouped["avg_release_pressure"].quantile(0.75)) if grouped["avg_release_pressure"].notna().any() else 0.0,
        "play_high": float(grouped["avg_scene_local_play_share"].quantile(0.75)) if grouped["avg_scene_local_play_share"].notna().any() else 0.0,
        "play_mid": float(grouped["avg_scene_local_play_share"].quantile(0.50)) if grouped["avg_scene_local_play_share"].notna().any() else 0.0,
        "opportunity_mid": float(grouped["avg_opportunity_score"].quantile(0.50)) if grouped["avg_opportunity_score"].notna().any() else 0.0,
    }
    grouped["strategy_posture"] = grouped.apply(_scene_posture, axis=1, thresholds=thresholds)
    return grouped[columns].sort_values(["momentum_score", "avg_opportunity_score"], ascending=[False, False]).reset_index(drop=True)


def _build_opportunity_lane_atlas(opportunities_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    if opportunities_df.empty:
        return pd.DataFrame(columns=columns)

    opportunities = opportunities_df.copy()
    for column in [
        "opportunity_score",
        "scene_local_play_share",
        "scene_release_pressure",
        "scene_label_concentration",
        "fan_migration_score",
        "release_whitespace_score",
        "local_gap_score",
        "scene_momentum_score",
    ]:
        opportunities[column] = pd.to_numeric(opportunities.get(column), errors="coerce")
    opportunities["seed_bridge_count"] = _series_or_default(opportunities, "connected_seed_artists", []).map(_parse_json_list).map(len)
    opportunities["is_priority_now"] = _series_or_default(opportunities, "opportunity_band", "").astype(str).eq("priority_now").astype(int)
    opportunities["is_watchlist"] = _series_or_default(opportunities, "opportunity_band", "").astype(str).eq("watchlist").astype(int)

    grouped = (
        opportunities.groupby(["scene_name", "primary_driver"], dropna=False)
        .agg(
            family_count=("report_family_id", "nunique"),
            artist_count=("artist_name", "nunique"),
            opportunity_count=("artist_name", "count"),
            priority_now_count=("is_priority_now", "sum"),
            watchlist_count=("is_watchlist", "sum"),
            avg_opportunity_score=("opportunity_score", "mean"),
            avg_scene_local_play_share=("scene_local_play_share", "mean"),
            avg_scene_release_pressure=("scene_release_pressure", "mean"),
            avg_scene_label_concentration=("scene_label_concentration", "mean"),
            avg_seed_bridge_count=("seed_bridge_count", "mean"),
            avg_fan_migration_score=("fan_migration_score", "mean"),
            avg_release_whitespace_score=("release_whitespace_score", "mean"),
            avg_local_gap_score=("local_gap_score", "mean"),
            avg_scene_momentum_score=("scene_momentum_score", "mean"),
        )
        .reset_index()
    )
    representative_artist = (
        opportunities.sort_values(
            ["scene_name", "primary_driver", "opportunity_score"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["scene_name", "primary_driver"], keep="first")[
            ["scene_name", "primary_driver", "artist_name"]
        ]
        .rename(columns={"artist_name": "representative_artist"})
    )
    grouped = grouped.merge(representative_artist, on=["scene_name", "primary_driver"], how="left")
    grouped["lane_attractiveness_score"] = (
        0.25 * _normalize_series(grouped["avg_opportunity_score"], higher_is_better=True).fillna(0.0)
        + 0.20 * _normalize_series(grouped["priority_now_count"], higher_is_better=True).fillna(0.0)
        + 0.15 * _normalize_series(grouped["avg_scene_local_play_share"], higher_is_better=True).fillna(0.0)
        + 0.15 * _normalize_series(grouped["avg_fan_migration_score"], higher_is_better=True).fillna(0.0)
        + 0.10 * _normalize_series(grouped["avg_seed_bridge_count"], higher_is_better=True).fillna(0.0)
        + 0.10 * _normalize_series(grouped["avg_local_gap_score"], higher_is_better=True).fillna(0.0)
        + 0.05 * _normalize_series(grouped["avg_release_whitespace_score"], higher_is_better=True).fillna(0.0)
    )
    grouped["lane_posture"] = grouped.apply(_lane_posture, axis=1)
    return grouped[columns].sort_values(
        ["lane_attractiveness_score", "avg_opportunity_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _build_market_migration_network(scene_df: pd.DataFrame, migration_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    if migration_df.empty:
        return pd.DataFrame(columns=columns)

    migration = migration_df.copy()
    for column in ["source_scene_id", "target_scene_id", "source_out_share", "target_in_share", "transition_count"]:
        migration[column] = pd.to_numeric(migration.get(column), errors="coerce")

    if not scene_df.empty and "scene_id" in scene_df.columns:
        lookup = (
            scene_df[["report_family_id", "scene_id", "scene_name"]]
            .dropna(subset=["scene_id"])
            .drop_duplicates()
        )
        migration = migration.merge(
            lookup.rename(columns={"scene_id": "source_scene_id", "scene_name": "source_scene_name"}),
            on=["report_family_id", "source_scene_id"],
            how="left",
        )
        migration = migration.merge(
            lookup.rename(columns={"scene_id": "target_scene_id", "scene_name": "target_scene_name"}),
            on=["report_family_id", "target_scene_id"],
            how="left",
        )
    else:
        migration["source_scene_name"] = ""
        migration["target_scene_name"] = ""

    grouped = (
        migration.groupby(["source_artist", "target_artist"], dropna=False)
        .agg(
            family_count=("report_family_id", "nunique"),
            route_mentions=("target_artist", "count"),
            total_transition_count=("transition_count", "sum"),
            avg_source_out_share=("source_out_share", "mean"),
            avg_target_in_share=("target_in_share", "mean"),
            source_scene_name=("source_scene_name", _mode_text),
            target_scene_name=("target_scene_name", _mode_text),
        )
        .reset_index()
    )
    grouped["route_strength_score"] = (
        0.45 * _normalize_series(grouped["avg_source_out_share"], higher_is_better=True).fillna(0.0)
        + 0.25 * _normalize_series(grouped["avg_target_in_share"], higher_is_better=True).fillna(0.0)
        + 0.30 * _normalize_series(grouped["total_transition_count"], higher_is_better=True).fillna(0.0)
    )
    return grouped[columns].sort_values(
        ["route_strength_score", "total_transition_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _build_seed_scene_bridge_atlas(seed_frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
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
    non_empty_seed_frames = [item for item in seed_frames if not item.empty]
    frame = pd.concat(non_empty_seed_frames, ignore_index=True) if non_empty_seed_frames else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "avg_opportunity_score",
        "bridge_artist_count",
        "opportunity_count",
        "scene_local_play_share",
        "scene_release_pressure",
        "scene_label_concentration",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.drop_duplicates().reset_index(drop=True)
    grouped = (
        frame.groupby(["scene_name", "seed_artist"], dropna=False)
        .agg(
            family_count=("report_family_id", "nunique"),
            opportunity_count=("opportunity_count", "sum"),
            avg_opportunity_score=("avg_opportunity_score", "mean"),
            avg_bridge_artist_count=("bridge_artist_count", "mean"),
            avg_scene_local_play_share=("scene_local_play_share", "mean"),
            avg_scene_release_pressure=("scene_release_pressure", "mean"),
            avg_scene_label_concentration=("scene_label_concentration", "mean"),
            top_opportunity_artist=("top_opportunity_artist", _mode_text),
            dominant_driver=("top_driver", _mode_text),
        )
        .reset_index()
    )
    grouped["bridge_score"] = (
        0.35 * _normalize_series(grouped["avg_opportunity_score"], higher_is_better=True).fillna(0.0)
        + 0.25 * _normalize_series(grouped["avg_bridge_artist_count"], higher_is_better=True).fillna(0.0)
        + 0.20 * _normalize_series(grouped["avg_scene_local_play_share"], higher_is_better=True).fillna(0.0)
        + 0.10 * _normalize_series(grouped["avg_scene_release_pressure"], higher_is_better=True).fillna(0.0)
        + 0.10 * _normalize_series(grouped["avg_scene_label_concentration"], higher_is_better=False).fillna(0.0)
    )
    return grouped[columns].sort_values(["bridge_score", "avg_opportunity_score"], ascending=[False, False]).reset_index(drop=True)


def _build_release_whitespace_atlas(opportunities_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    if opportunities_df.empty:
        return pd.DataFrame(columns=columns)

    opportunities = opportunities_df.copy()
    for column in ["opportunity_score", "release_whitespace_score", "days_since_latest_release"]:
        opportunities[column] = pd.to_numeric(opportunities.get(column), errors="coerce")
    opportunities["seed_bridge_count"] = _series_or_default(opportunities, "connected_seed_artists", []).map(_parse_json_list).map(len)
    opportunities["dominant_release_labels"] = _series_or_default(opportunities, "dominant_release_labels", []).map(_parse_json_list)
    whitespace = opportunities.loc[
        opportunities["release_whitespace_score"].fillna(0.0).gt(0.0)
        | opportunities["days_since_latest_release"].fillna(0.0).gt(0.0)
        | opportunities["dominant_release_labels"].map(bool)
    ].copy()
    if whitespace.empty:
        return pd.DataFrame(columns=columns)

    whitespace["dominant_label_text"] = whitespace["dominant_release_labels"].map(lambda items: "|".join(items[:3]))
    grouped = (
        whitespace.groupby(["artist_name", "scene_name"], dropna=False)
        .agg(
            family_count=("report_family_id", "nunique"),
            avg_opportunity_score=("opportunity_score", "mean"),
            avg_release_whitespace_score=("release_whitespace_score", "mean"),
            max_days_since_latest_release=("days_since_latest_release", "max"),
            avg_seed_bridge_count=("seed_bridge_count", "mean"),
            dominant_labels=("dominant_label_text", _mode_text),
            primary_driver=("primary_driver", _mode_text),
        )
        .reset_index()
    )
    grouped["whitespace_signal_score"] = (
        0.45 * _normalize_series(grouped["avg_release_whitespace_score"], higher_is_better=True).fillna(0.0)
        + 0.30 * _normalize_series(grouped["avg_opportunity_score"], higher_is_better=True).fillna(0.0)
        + 0.15 * _normalize_series(grouped["max_days_since_latest_release"], higher_is_better=True).fillna(0.0)
        + 0.10 * _normalize_series(grouped["avg_seed_bridge_count"], higher_is_better=True).fillna(0.0)
    )
    return grouped[columns].sort_values(
        ["whitespace_signal_score", "avg_release_whitespace_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _trend_number(value: object) -> str:
    numeric = _safe_float(value)
    return "n/a" if not math.isfinite(numeric) else f"{numeric:.3f}"


def _trend_severity(value: float, *, high: float, medium: float) -> str:
    if math.isfinite(value) and value >= high:
        return "high"
    if math.isfinite(value) and value >= medium:
        return "medium"
    return "low"


def _repeat_signal_sort_key(item: dict[str, object]) -> tuple[int, float, float]:
    return (
        int(item.get("family_count", 0) or 0),
        _safe_float(item.get("trend_score")),
        _safe_float(item.get("delta_value")),
    )


def _ordered_family_rollup(frame: pd.DataFrame, family_order: dict[str, int]) -> pd.DataFrame:
    rollup = frame.copy()
    rollup["family_order"] = rollup["report_family_id"].map(family_order).fillna(len(family_order)).astype(int)
    return rollup.sort_values(["family_order", "report_family_id"]).reset_index(drop=True)


def _build_trend_delta_markdown(
    trend_deltas: pd.DataFrame,
    *,
    report_family_count: int,
    comparison_basis: str,
) -> list[str]:
    lines = [
        "# Creator Market Trend Deltas",
        "",
        f"- Report families compared: `{report_family_count}`",
        f"- Comparison basis: `{comparison_basis}`",
        f"- Highlight rows: `{len(trend_deltas.index)}`",
        "",
    ]
    sections = [
        ("Rising Scenes", "rising_scene", "No rising scene deltas are available yet."),
        ("Repeated Opportunity Lanes", "repeated_opportunity_lane", "No repeated opportunity lanes are available yet."),
        ("Repeated Migration Routes", "repeated_migration_route", "No repeated migration routes are available yet."),
        (
            "Release-Whitespace Coverage",
            "release_whitespace",
            "Release-whitespace coverage did not cross stale or sparse thresholds.",
        ),
    ]
    for title, signal_filter, empty_text in sections:
        lines.extend([f"## {title}", ""])
        if signal_filter == "release_whitespace":
            rows = trend_deltas.loc[
                trend_deltas["signal_type"].isin(["stale_release_whitespace", "sparse_release_whitespace_coverage"])
            ].head(5)
        else:
            rows = trend_deltas.loc[trend_deltas["signal_type"].eq(signal_filter)].head(5)
        if rows.empty:
            lines.extend([f"- {empty_text}", ""])
            continue
        for row in rows.to_dict(orient="records"):
            signal_key = str(row.get("signal_key", "")).strip()
            note = str(row.get("trend_note", "")).strip()
            action = str(row.get("action_hint", "")).strip()
            delta = _trend_number(row.get("delta_value"))
            coverage = _trend_number(row.get("coverage_ratio"))
            if row.get("signal_type") == "sparse_release_whitespace_coverage":
                lines.append(f"- `{signal_key}`: coverage `{coverage}`. {note} {action}".strip())
            else:
                lines.append(f"- `{signal_key}`: delta `{delta}`. {note} {action}".strip())
        lines.append("")
    return lines


def _build_creator_market_trend_deltas(
    *,
    scene_df: pd.DataFrame,
    opportunities_df: pd.DataFrame,
    migration_df: pd.DataFrame,
    family_inventory: dict[str, _ReportFamilyRecord],
) -> tuple[pd.DataFrame, list[str]]:
    family_order, comparison_basis = _report_family_order_context(family_inventory)
    report_family_count = len(family_order)
    rows: list[dict[str, object]] = []

    if report_family_count >= 2 and not scene_df.empty and {"report_family_id", "scene_name"}.issubset(scene_df.columns):
        scene = scene_df.copy()
        scene["report_family_id"] = _series_or_default(scene, "report_family_id", "").astype(str).str.strip()
        scene["scene_name"] = _series_or_default(scene, "scene_name", "").astype(str).str.strip()
        scene = scene.loc[scene["report_family_id"].ne("") & scene["scene_name"].ne("")].copy()
        for column in [
            "scene_local_play_share",
            "avg_opportunity_score",
            "priority_now_count",
            "watchlist_count",
            "scene_release_pressure",
        ]:
            scene[column] = pd.to_numeric(_series_or_default(scene, column, np.nan), errors="coerce")
        if not scene.empty:
            scene_rollup = (
                scene.groupby(["report_family_id", "scene_name"], dropna=False)
                .agg(
                    avg_scene_local_play_share=("scene_local_play_share", "mean"),
                    avg_opportunity_score=("avg_opportunity_score", "mean"),
                    priority_now_count=("priority_now_count", "sum"),
                    watchlist_count=("watchlist_count", "sum"),
                    avg_release_pressure=("scene_release_pressure", "mean"),
                )
                .reset_index()
            )
            scene_rollup["signal_value"] = (
                scene_rollup["avg_opportunity_score"].fillna(0.0)
                + scene_rollup["avg_scene_local_play_share"].fillna(0.0)
                + 0.05 * scene_rollup["priority_now_count"].fillna(0.0)
                + 0.02 * scene_rollup["watchlist_count"].fillna(0.0)
                + 0.10 * scene_rollup["avg_release_pressure"].fillna(0.0)
            )
            scene_rollup = _ordered_family_rollup(scene_rollup, family_order)
            candidates: list[dict[str, object]] = []
            for scene_name, group in scene_rollup.groupby("scene_name", dropna=False):
                if group["report_family_id"].nunique() < 2:
                    continue
                first = group.iloc[0]
                latest = group.iloc[-1]
                delta_value = _safe_float(latest.get("signal_value")) - _safe_float(first.get("signal_value"))
                if not math.isfinite(delta_value) or delta_value <= 0.0:
                    continue
                candidates.append(
                    {
                        "signal_type": "rising_scene",
                        "signal_key": str(scene_name),
                        "scene_name": str(scene_name),
                        "primary_driver": "",
                        "source_artist": "",
                        "target_artist": "",
                        "family_count": int(group["report_family_id"].nunique()),
                        "repeat_count": int(len(group.index)),
                        "first_report_family_id": str(first.get("report_family_id", "")),
                        "latest_report_family_id": str(latest.get("report_family_id", "")),
                        "first_value": _safe_float(first.get("signal_value")),
                        "latest_value": _safe_float(latest.get("signal_value")),
                        "delta_value": delta_value,
                        "coverage_ratio": np.nan,
                        "metadata_row_count": "",
                        "opportunity_row_count": "",
                        "trend_score": delta_value,
                        "severity": _trend_severity(delta_value, high=0.25, medium=0.05),
                        "comparison_basis": comparison_basis,
                        "trend_note": (
                            f"Scene rose from `{first.get('report_family_id', '')}` to `{latest.get('report_family_id', '')}` "
                            f"as play share reached `{_trend_number(latest.get('avg_scene_local_play_share'))}` and priority count `{_trend_number(latest.get('priority_now_count'))}`."
                        ),
                        "action_hint": "Use it as a next-family seed or strategy-deck lead scene.",
                    }
                )
            for rank, row in enumerate(sorted(candidates, key=lambda item: _safe_float(item["trend_score"]), reverse=True)[:10], start=1):
                row["category_rank"] = rank
                rows.append(row)

    if report_family_count >= 2 and not opportunities_df.empty and {"report_family_id", "scene_name"}.issubset(opportunities_df.columns):
        opportunities = opportunities_df.copy()
        opportunities["report_family_id"] = _series_or_default(opportunities, "report_family_id", "").astype(str).str.strip()
        opportunities["scene_name"] = _series_or_default(opportunities, "scene_name", "").astype(str).str.strip()
        opportunities["primary_driver"] = _series_or_default(opportunities, "primary_driver", "").astype(str).str.strip()
        opportunities["primary_driver"] = opportunities["primary_driver"].replace("", "unknown_driver")
        opportunities = opportunities.loc[opportunities["report_family_id"].ne("") & opportunities["scene_name"].ne("")].copy()
        for column in ["opportunity_score", "release_whitespace_score", "fan_migration_score"]:
            opportunities[column] = pd.to_numeric(_series_or_default(opportunities, column, np.nan), errors="coerce")
        opportunities["is_priority_now"] = _series_or_default(opportunities, "opportunity_band", "").astype(str).eq("priority_now").astype(int)
        if not opportunities.empty:
            lane_rollup = (
                opportunities.groupby(["report_family_id", "scene_name", "primary_driver"], dropna=False)
                .agg(
                    opportunity_count=("artist_name", "count"),
                    priority_now_count=("is_priority_now", "sum"),
                    avg_opportunity_score=("opportunity_score", "mean"),
                    avg_release_whitespace_score=("release_whitespace_score", "mean"),
                    avg_fan_migration_score=("fan_migration_score", "mean"),
                )
                .reset_index()
            )
            lane_rollup["signal_value"] = (
                lane_rollup["avg_opportunity_score"].fillna(0.0)
                + 0.04 * lane_rollup["priority_now_count"].fillna(0.0)
                + 0.02 * lane_rollup["opportunity_count"].fillna(0.0)
                + 0.10 * lane_rollup["avg_fan_migration_score"].fillna(0.0)
                + 0.05 * lane_rollup["avg_release_whitespace_score"].fillna(0.0)
            )
            lane_rollup = _ordered_family_rollup(lane_rollup, family_order)
            candidates = []
            for (scene_name, primary_driver), group in lane_rollup.groupby(["scene_name", "primary_driver"], dropna=False):
                if group["report_family_id"].nunique() < 2:
                    continue
                first = group.iloc[0]
                latest = group.iloc[-1]
                delta_value = _safe_float(latest.get("signal_value")) - _safe_float(first.get("signal_value"))
                total_opportunities = int(pd.to_numeric(group["opportunity_count"], errors="coerce").fillna(0).sum())
                trend_score = _safe_float(latest.get("signal_value")) + 0.10 * max(0, int(group["report_family_id"].nunique()) - 1)
                candidates.append(
                    {
                        "signal_type": "repeated_opportunity_lane",
                        "signal_key": f"{scene_name} / {primary_driver}",
                        "scene_name": str(scene_name),
                        "primary_driver": str(primary_driver),
                        "source_artist": "",
                        "target_artist": "",
                        "family_count": int(group["report_family_id"].nunique()),
                        "repeat_count": total_opportunities,
                        "first_report_family_id": str(first.get("report_family_id", "")),
                        "latest_report_family_id": str(latest.get("report_family_id", "")),
                        "first_value": _safe_float(first.get("signal_value")),
                        "latest_value": _safe_float(latest.get("signal_value")),
                        "delta_value": delta_value,
                        "coverage_ratio": np.nan,
                        "metadata_row_count": "",
                        "opportunity_row_count": total_opportunities,
                        "trend_score": trend_score,
                        "severity": "high" if int(group["report_family_id"].nunique()) >= 3 or trend_score >= 0.60 else "medium",
                        "comparison_basis": comparison_basis,
                        "trend_note": (
                            f"Lane repeats across `{int(group['report_family_id'].nunique())}` families with `{total_opportunities}` opportunity rows."
                        ),
                        "action_hint": "Treat repeated lanes as reusable market plays instead of one-off artist recommendations.",
                    }
                )
            for rank, row in enumerate(sorted(candidates, key=_repeat_signal_sort_key, reverse=True)[:10], start=1):
                row["category_rank"] = rank
                rows.append(row)

    if report_family_count >= 2 and not migration_df.empty and {"report_family_id", "source_artist", "target_artist"}.issubset(migration_df.columns):
        migration = migration_df.copy()
        migration["report_family_id"] = _series_or_default(migration, "report_family_id", "").astype(str).str.strip()
        migration["source_artist"] = _series_or_default(migration, "source_artist", "").astype(str).str.strip()
        migration["target_artist"] = _series_or_default(migration, "target_artist", "").astype(str).str.strip()
        migration = migration.loc[
            migration["report_family_id"].ne("") & migration["source_artist"].ne("") & migration["target_artist"].ne("")
        ].copy()
        for column in ["source_scene_id", "target_scene_id", "source_out_share", "target_in_share", "transition_count"]:
            migration[column] = pd.to_numeric(_series_or_default(migration, column, np.nan), errors="coerce")
        if not migration.empty:
            if not scene_df.empty and {"report_family_id", "scene_id", "scene_name"}.issubset(scene_df.columns):
                lookup = scene_df[["report_family_id", "scene_id", "scene_name"]].copy()
                lookup["report_family_id"] = lookup["report_family_id"].astype(str).str.strip()
                lookup["scene_id"] = pd.to_numeric(lookup["scene_id"], errors="coerce")
                lookup["scene_name"] = lookup["scene_name"].astype(str).str.strip()
                lookup = lookup.dropna(subset=["scene_id"]).drop_duplicates()
                migration = migration.merge(
                    lookup.rename(columns={"scene_id": "source_scene_id", "scene_name": "source_scene_name"}),
                    on=["report_family_id", "source_scene_id"],
                    how="left",
                )
                migration = migration.merge(
                    lookup.rename(columns={"scene_id": "target_scene_id", "scene_name": "target_scene_name"}),
                    on=["report_family_id", "target_scene_id"],
                    how="left",
                )
            else:
                migration["source_scene_name"] = ""
                migration["target_scene_name"] = ""
            for column in ["source_scene_name", "target_scene_name"]:
                migration[column] = _series_or_default(migration, column, "").fillna("").astype(str).replace("nan", "")
            route_rollup = (
                migration.groupby(["report_family_id", "source_artist", "target_artist"], dropna=False)
                .agg(
                    route_mentions=("target_artist", "count"),
                    total_transition_count=("transition_count", "sum"),
                    avg_source_out_share=("source_out_share", "mean"),
                    avg_target_in_share=("target_in_share", "mean"),
                    source_scene_name=("source_scene_name", _mode_text),
                    target_scene_name=("target_scene_name", _mode_text),
                )
                .reset_index()
            )
            route_rollup["signal_value"] = (
                route_rollup["avg_source_out_share"].fillna(0.0)
                + route_rollup["avg_target_in_share"].fillna(0.0)
                + route_rollup["total_transition_count"].fillna(0.0) / 1000.0
            )
            route_rollup = _ordered_family_rollup(route_rollup, family_order)
            candidates = []
            for (source_artist, target_artist), group in route_rollup.groupby(["source_artist", "target_artist"], dropna=False):
                if group["report_family_id"].nunique() < 2:
                    continue
                first = group.iloc[0]
                latest = group.iloc[-1]
                delta_value = _safe_float(latest.get("signal_value")) - _safe_float(first.get("signal_value"))
                repeat_count = int(pd.to_numeric(group["route_mentions"], errors="coerce").fillna(0).sum())
                trend_score = _safe_float(latest.get("signal_value")) + 0.10 * max(0, int(group["report_family_id"].nunique()) - 1)
                candidates.append(
                    {
                        "signal_type": "repeated_migration_route",
                        "signal_key": f"{source_artist} -> {target_artist}",
                        "scene_name": str(latest.get("target_scene_name", "")),
                        "primary_driver": "migration_capture",
                        "source_artist": str(source_artist),
                        "target_artist": str(target_artist),
                        "family_count": int(group["report_family_id"].nunique()),
                        "repeat_count": repeat_count,
                        "first_report_family_id": str(first.get("report_family_id", "")),
                        "latest_report_family_id": str(latest.get("report_family_id", "")),
                        "first_value": _safe_float(first.get("signal_value")),
                        "latest_value": _safe_float(latest.get("signal_value")),
                        "delta_value": delta_value,
                        "coverage_ratio": np.nan,
                        "metadata_row_count": "",
                        "opportunity_row_count": "",
                        "trend_score": trend_score,
                        "severity": "high" if int(group["report_family_id"].nunique()) >= 3 or trend_score >= 0.50 else "medium",
                        "comparison_basis": comparison_basis,
                        "trend_note": (
                            f"Route repeats across `{int(group['report_family_id'].nunique())}` families and most recently lands in `{latest.get('target_scene_name', '')}`."
                        ),
                        "action_hint": "Use repeated routes as migration-capture candidates for next creator briefs.",
                    }
                )
            for rank, row in enumerate(sorted(candidates, key=_repeat_signal_sort_key, reverse=True)[:10], start=1):
                row["category_rank"] = rank
                rows.append(row)

    if not opportunities_df.empty and "report_family_id" in opportunities_df.columns:
        opportunities = opportunities_df.copy()
        opportunities["report_family_id"] = _series_or_default(opportunities, "report_family_id", "").astype(str).str.strip()
        opportunities["artist_name"] = _series_or_default(opportunities, "artist_name", "").astype(str).str.strip()
        opportunities["scene_name"] = _series_or_default(opportunities, "scene_name", "").astype(str).str.strip()
        opportunities = opportunities.loc[opportunities["report_family_id"].ne("")].copy()
        opportunities["days_since_latest_release"] = pd.to_numeric(
            _series_or_default(opportunities, "days_since_latest_release", np.nan),
            errors="coerce",
        )
        opportunities["release_whitespace_score"] = pd.to_numeric(
            _series_or_default(opportunities, "release_whitespace_score", np.nan),
            errors="coerce",
        )
        opportunities["opportunity_score"] = pd.to_numeric(
            _series_or_default(opportunities, "opportunity_score", np.nan),
            errors="coerce",
        )
        release_labels = _series_or_default(opportunities, "dominant_release_labels", []).map(_parse_json_list)
        opportunities["_has_release_metadata"] = (
            opportunities["days_since_latest_release"].notna()
            | opportunities["release_whitespace_score"].fillna(0.0).gt(0.0)
            | release_labels.map(bool)
        )
        opportunity_row_count = int(len(opportunities.index))
        metadata_row_count = int(opportunities["_has_release_metadata"].sum())
        coverage_ratio = float(metadata_row_count / opportunity_row_count) if opportunity_row_count else float("nan")
        if opportunity_row_count and coverage_ratio < 0.50:
            family_coverage = (
                opportunities.groupby("report_family_id", dropna=False)
                .agg(
                    metadata_row_count=("_has_release_metadata", "sum"),
                    opportunity_row_count=("_has_release_metadata", "count"),
                )
                .reset_index()
            )
            family_coverage["coverage_ratio"] = (
                family_coverage["metadata_row_count"] / family_coverage["opportunity_row_count"].replace(0, np.nan)
            )
            family_coverage = _ordered_family_rollup(family_coverage, family_order)
            first = family_coverage.iloc[0]
            latest = family_coverage.iloc[-1]
            delta_value = _safe_float(latest.get("coverage_ratio")) - _safe_float(first.get("coverage_ratio"))
            rows.append(
                {
                    "signal_type": "sparse_release_whitespace_coverage",
                    "signal_key": "release_metadata_coverage",
                    "category_rank": 1,
                    "scene_name": "",
                    "primary_driver": "release_whitespace",
                    "source_artist": "",
                    "target_artist": "",
                    "family_count": int(opportunities["report_family_id"].nunique()),
                    "repeat_count": metadata_row_count,
                    "first_report_family_id": str(first.get("report_family_id", "")),
                    "latest_report_family_id": str(latest.get("report_family_id", "")),
                    "first_value": _safe_float(first.get("coverage_ratio")),
                    "latest_value": _safe_float(latest.get("coverage_ratio")),
                    "delta_value": delta_value,
                    "coverage_ratio": coverage_ratio,
                    "metadata_row_count": metadata_row_count,
                    "opportunity_row_count": opportunity_row_count,
                    "trend_score": 1.0 - coverage_ratio,
                    "severity": "high" if coverage_ratio < 0.25 else "medium",
                    "comparison_basis": comparison_basis,
                    "trend_note": (
                        f"Only `{metadata_row_count}` of `{opportunity_row_count}` opportunity rows carry release dates, whitespace scores, or label metadata."
                    ),
                    "action_hint": "Backfill public release dates and label metadata before over-weighting whitespace trends.",
                }
            )
        stale = opportunities.loc[
            opportunities["artist_name"].ne("")
            & opportunities["days_since_latest_release"].ge(_STALE_RELEASE_DAYS)
        ].copy()
        if not stale.empty:
            stale = _ordered_family_rollup(stale, family_order)
            stale_rollup = (
                stale.groupby(["artist_name", "scene_name"], dropna=False)
                .agg(
                    family_count=("report_family_id", "nunique"),
                    repeat_count=("artist_name", "count"),
                    max_days_since_latest_release=("days_since_latest_release", "max"),
                    avg_release_whitespace_score=("release_whitespace_score", "mean"),
                    avg_opportunity_score=("opportunity_score", "mean"),
                    first_report_family_id=("report_family_id", "first"),
                    latest_report_family_id=("report_family_id", "last"),
                )
                .reset_index()
            )
            stale_rollup["trend_score"] = (
                stale_rollup["max_days_since_latest_release"].fillna(0.0) / _STALE_RELEASE_DAYS
                + stale_rollup["avg_release_whitespace_score"].fillna(0.0)
                + 0.25 * stale_rollup["avg_opportunity_score"].fillna(0.0)
            )
            stale_rollup = stale_rollup.sort_values(
                ["trend_score", "max_days_since_latest_release"],
                ascending=[False, False],
            )
            for rank, row in enumerate(stale_rollup.head(10).to_dict(orient="records"), start=1):
                max_days = _safe_float(row.get("max_days_since_latest_release"))
                rows.append(
                    {
                        "signal_type": "stale_release_whitespace",
                        "signal_key": str(row.get("artist_name", "")),
                        "category_rank": rank,
                        "scene_name": str(row.get("scene_name", "")),
                        "primary_driver": "release_whitespace",
                        "source_artist": "",
                        "target_artist": "",
                        "family_count": int(row.get("family_count", 0) or 0),
                        "repeat_count": int(row.get("repeat_count", 0) or 0),
                        "first_report_family_id": str(row.get("first_report_family_id", "")),
                        "latest_report_family_id": str(row.get("latest_report_family_id", "")),
                        "first_value": max_days,
                        "latest_value": max_days,
                        "delta_value": 0.0,
                        "coverage_ratio": coverage_ratio,
                        "metadata_row_count": metadata_row_count,
                        "opportunity_row_count": opportunity_row_count,
                        "trend_score": _safe_float(row.get("trend_score")),
                        "severity": "high" if math.isfinite(max_days) and max_days >= 365.0 else "medium",
                        "comparison_basis": comparison_basis,
                        "trend_note": f"Latest known release age is `{_trend_number(max_days)}` days.",
                        "action_hint": "Validate whether the gap is a true release window before prioritizing outreach.",
                    }
                )

    if not rows:
        empty = pd.DataFrame(columns=_TREND_DELTA_COLUMNS)
        return empty, _build_trend_delta_markdown(
            empty,
            report_family_count=report_family_count,
            comparison_basis=comparison_basis,
        )

    trend_deltas = pd.DataFrame(rows)
    for column in _TREND_DELTA_COLUMNS:
        if column not in trend_deltas.columns:
            trend_deltas[column] = np.nan
    for column in [
        "category_rank",
        "family_count",
        "repeat_count",
        "first_value",
        "latest_value",
        "delta_value",
        "coverage_ratio",
        "metadata_row_count",
        "opportunity_row_count",
        "trend_score",
    ]:
        trend_deltas[column] = pd.to_numeric(trend_deltas[column], errors="coerce")
    signal_order = {
        "rising_scene": 0,
        "repeated_opportunity_lane": 1,
        "repeated_migration_route": 2,
        "sparse_release_whitespace_coverage": 3,
        "stale_release_whitespace": 4,
    }
    trend_deltas["_signal_order"] = trend_deltas["signal_type"].map(signal_order).fillna(99).astype(int)
    trend_deltas = trend_deltas.sort_values(
        ["_signal_order", "category_rank", "trend_score"],
        ascending=[True, True, False],
    ).drop(columns=["_signal_order"])
    trend_deltas = trend_deltas[_TREND_DELTA_COLUMNS].reset_index(drop=True)
    return trend_deltas, _build_trend_delta_markdown(
        trend_deltas,
        report_family_count=report_family_count,
        comparison_basis=comparison_basis,
    )


def _build_market_brief(
    *,
    report_family_count: int,
    scene_market_pulse: pd.DataFrame,
    lane_atlas: pd.DataFrame,
    migration_network: pd.DataFrame,
    seed_bridge_atlas: pd.DataFrame,
    whitespace_atlas: pd.DataFrame,
    trend_deltas: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    top_scene = scene_market_pulse.iloc[0].to_dict() if not scene_market_pulse.empty else {}
    top_lane = lane_atlas.iloc[0].to_dict() if not lane_atlas.empty else {}
    top_route = migration_network.iloc[0].to_dict() if not migration_network.empty else {}
    top_bridge = seed_bridge_atlas.iloc[0].to_dict() if not seed_bridge_atlas.empty else {}
    top_whitespace = whitespace_atlas.iloc[0].to_dict() if not whitespace_atlas.empty else {}
    top_trend = trend_deltas.iloc[0].to_dict() if not trend_deltas.empty else {}
    trend_delta_counts = (
        {str(key): int(value) for key, value in trend_deltas["signal_type"].value_counts().to_dict().items()}
        if not trend_deltas.empty and "signal_type" in trend_deltas.columns
        else {}
    )

    summary = [
        f"Creator market lab is aggregating `{report_family_count}` creator report families.",
    ]
    if top_scene:
        summary.append(
            f"Strongest market scene is `{top_scene.get('scene_name', '')}` with momentum score `{_safe_float(top_scene.get('momentum_score')):.3f}` and posture `{top_scene.get('strategy_posture', '')}`."
        )
    if top_lane:
        summary.append(
            f"Most attractive opportunity lane is `{top_lane.get('scene_name', '')} / {top_lane.get('primary_driver', '')}` at score `{_safe_float(top_lane.get('lane_attractiveness_score')):.3f}`."
        )
    if top_route:
        summary.append(
            f"Most concentrated migration route is `{top_route.get('source_artist', '')} -> {top_route.get('target_artist', '')}` with route score `{_safe_float(top_route.get('route_strength_score')):.3f}`."
        )
    if top_bridge:
        summary.append(
            f"Strongest scene-seed bridge is `{top_bridge.get('scene_name', '')} x {top_bridge.get('seed_artist', '')}` at bridge score `{_safe_float(top_bridge.get('bridge_score')):.3f}`."
        )
    if top_whitespace:
        summary.append(
            f"Top release-whitespace watch is `{top_whitespace.get('artist_name', '')}` in `{top_whitespace.get('scene_name', '')}` with score `{_safe_float(top_whitespace.get('whitespace_signal_score')):.3f}`."
        )
    else:
        summary.append("Release-whitespace metadata is still sparse in the saved creator families, so that lane remains mostly a future-facing surface.")
    if top_trend:
        summary.append(
            f"Trend deltas add `{len(trend_deltas.index)}` cross-family highlights, led by `{top_trend.get('signal_type', '')}` for `{top_trend.get('signal_key', '')}`."
        )

    actions = [
        "Use the scene market pulse to pick the next creator brief or strategy deck seed set.",
        "Use the lane atlas when deciding whether to expand through adjacency, migration, or whitespace rather than treating every opportunity row the same.",
        "Use the migration network and scene-seed bridges as the default handoff into creator strategy and cultural-analysis work.",
        "Use trend deltas to separate repeated market patterns from signals that only appear in one report family.",
    ]
    payload = {
        "report_family_count": int(report_family_count),
        "top_scene": top_scene,
        "top_lane": top_lane,
        "top_route": top_route,
        "top_bridge": top_bridge,
        "top_whitespace": top_whitespace,
        "top_trend_delta": top_trend,
        "trend_delta_counts": trend_delta_counts,
        "summary": summary,
        "actions": actions,
    }
    markdown_lines = [
        "# Creator Market Brief",
        "",
        *[f"- {line}" for line in summary],
        "",
        "## Suggested Uses",
        "",
        *[f"- {line}" for line in actions],
    ]
    return payload, markdown_lines


def build_creator_market_intelligence(*, output_dir: Path, logger) -> list[Path]:
    base_dir = _asset_root(output_dir)
    family_inventory = _collect_report_family_inventory(base_dir)
    family_counts = _report_family_manifest_counts(family_inventory)
    report_family_ids = list(family_counts["report_family_ids"])
    assets = _load_creator_assets(output_dir, family_inventory=family_inventory)
    scene_df = assets.get("scene_comparison", pd.DataFrame()).copy()
    opportunities_df = assets.get("opportunities", pd.DataFrame()).copy()
    migration_df = assets.get("migration_watch", pd.DataFrame()).copy()
    seed_df = assets.get("seed_comparison", pd.DataFrame()).copy()
    scene_seed_df = assets.get("scene_seed_comparison", pd.DataFrame()).copy()

    if not report_family_ids and scene_df.empty and opportunities_df.empty:
        return []

    scene_market_pulse = _build_scene_market_pulse(
        scene_df=scene_df,
        opportunities_df=opportunities_df,
        migration_df=migration_df,
    )
    lane_atlas = _build_opportunity_lane_atlas(opportunities_df)
    migration_network = _build_market_migration_network(scene_df, migration_df)
    seed_bridge_atlas = _build_seed_scene_bridge_atlas([seed_df, scene_seed_df])
    whitespace_atlas = _build_release_whitespace_atlas(opportunities_df)
    trend_deltas, trend_delta_markdown = _build_creator_market_trend_deltas(
        scene_df=scene_df,
        opportunities_df=opportunities_df,
        migration_df=migration_df,
        family_inventory=family_inventory,
    )
    brief_payload, brief_markdown = _build_market_brief(
        report_family_count=int(family_counts["report_family_count"]),
        scene_market_pulse=scene_market_pulse,
        lane_atlas=lane_atlas,
        migration_network=migration_network,
        seed_bridge_atlas=seed_bridge_atlas,
        whitespace_atlas=whitespace_atlas,
        trend_deltas=trend_deltas,
    )

    output_root = output_dir / "analysis" / "creator_market_intelligence"
    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    tables = {
        "scene_market_pulse": (
            scene_market_pulse,
            [
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
            ],
        ),
        "opportunity_lane_atlas": (
            lane_atlas,
            [
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
            ],
        ),
        "market_migration_network": (
            migration_network,
            [
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
            ],
        ),
        "seed_scene_bridge_atlas": (
            seed_bridge_atlas,
            [
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
            ],
        ),
        "release_whitespace_atlas": (
            whitespace_atlas,
            [
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
            ],
        ),
        "creator_market_trend_deltas": (
            trend_deltas,
            _TREND_DELTA_COLUMNS,
        ),
    }

    manifest_payload = {
        "report_family_count": int(brief_payload["report_family_count"]),
        "manifest_backed_report_family_count": int(family_counts["manifest_backed_report_family_count"]),
        "asset_backed_report_family_count": int(family_counts["asset_backed_report_family_count"]),
        "complete_report_family_count": int(family_counts["complete_report_family_count"]),
        "partial_report_family_count": int(family_counts["partial_report_family_count"]),
        "partial_report_family_ids": list(family_counts["partial_report_family_ids"]),
        "artifact_root": str(output_root),
        "tables": {},
    }
    for stem, (frame, columns) in tables.items():
        csv_path = _write_csv(output_root / f"{stem}.csv", _rows_for_columns(frame, columns), columns)
        json_path = write_json(output_root / f"{stem}.json", frame.to_dict(orient="records"))
        manifest_payload["tables"][stem] = {
            "row_count": int(len(frame.index)),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }
        paths.extend([csv_path, json_path])

    trend_delta_md = write_markdown(output_root / "creator_market_trend_deltas.md", trend_delta_markdown)
    if "creator_market_trend_deltas" in manifest_payload["tables"]:
        manifest_payload["tables"]["creator_market_trend_deltas"]["markdown_path"] = str(trend_delta_md)
    paths.append(trend_delta_md)

    brief_json = write_json(output_root / "creator_market_brief.json", brief_payload)
    brief_md = write_markdown(output_root / "creator_market_brief.md", brief_markdown)
    manifest_json = write_json(output_root / "creator_market_manifest.json", manifest_payload)
    paths.extend([brief_json, brief_md, manifest_json])
    logger.info(
        "Built creator market intelligence with %d report families and %d scene rows.",
        int(brief_payload["report_family_count"]),
        len(scene_market_pulse.index),
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build creator / market-intelligence branch artifacts from saved creator report families.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing public-insights artifacts.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.creator_market_intelligence")
    paths = build_creator_market_intelligence(output_dir=Path(args.output_dir).expanduser().resolve(), logger=logger)
    if not paths:
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
