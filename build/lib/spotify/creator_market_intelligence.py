from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .run_artifacts import safe_read_csv
from .run_artifacts import write_csv_rows
from .run_artifacts import write_json
from .run_artifacts import write_markdown


_ASSET_SUFFIXES = (
    "scene_comparison",
    "ranking_comparison",
    "opportunities",
    "migration_watch",
    "seed_comparison",
    "scene_seed_comparison",
)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


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


def _family_id_from_path(path: Path, suffix: str) -> str:
    return path.stem.removesuffix(f"_{suffix}")


def _discover_report_family_ids(base_dir: Path) -> list[str]:
    family_ids: set[str] = set()
    for path in sorted(base_dir.glob("*_report_family.json")):
        family_ids.add(path.stem.removesuffix("_report_family"))
    for suffix in _ASSET_SUFFIXES:
        for path in sorted(base_dir.glob(f"*_{suffix}.csv")):
            family_ids.add(_family_id_from_path(path, suffix))
    return sorted(family_ids)


def _load_creator_assets(output_dir: Path) -> dict[str, pd.DataFrame]:
    base_dir = _asset_root(output_dir)
    frames: dict[str, pd.DataFrame] = {}
    if not base_dir.exists():
        return {suffix: pd.DataFrame() for suffix in _ASSET_SUFFIXES}
    for suffix in _ASSET_SUFFIXES:
        buckets: list[pd.DataFrame] = []
        for path in sorted(base_dir.glob(f"*_{suffix}.csv")):
            frame = safe_read_csv(path)
            if frame.empty:
                continue
            family_id = _family_id_from_path(path, suffix)
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
    frame = pd.concat([item for item in seed_frames if not item.empty], ignore_index=True) if seed_frames else pd.DataFrame()
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


def _build_market_brief(
    *,
    report_family_count: int,
    scene_market_pulse: pd.DataFrame,
    lane_atlas: pd.DataFrame,
    migration_network: pd.DataFrame,
    seed_bridge_atlas: pd.DataFrame,
    whitespace_atlas: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    top_scene = scene_market_pulse.iloc[0].to_dict() if not scene_market_pulse.empty else {}
    top_lane = lane_atlas.iloc[0].to_dict() if not lane_atlas.empty else {}
    top_route = migration_network.iloc[0].to_dict() if not migration_network.empty else {}
    top_bridge = seed_bridge_atlas.iloc[0].to_dict() if not seed_bridge_atlas.empty else {}
    top_whitespace = whitespace_atlas.iloc[0].to_dict() if not whitespace_atlas.empty else {}

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

    actions = [
        "Use the scene market pulse to pick the next creator brief or strategy deck seed set.",
        "Use the lane atlas when deciding whether to expand through adjacency, migration, or whitespace rather than treating every opportunity row the same.",
        "Use the migration network and scene-seed bridges as the default handoff into creator strategy and cultural-analysis work.",
    ]
    payload = {
        "report_family_count": int(report_family_count),
        "top_scene": top_scene,
        "top_lane": top_lane,
        "top_route": top_route,
        "top_bridge": top_bridge,
        "top_whitespace": top_whitespace,
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
    report_family_ids = _discover_report_family_ids(base_dir)
    assets = _load_creator_assets(output_dir)
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
    brief_payload, brief_markdown = _build_market_brief(
        report_family_count=max(len(report_family_ids), int(scene_df.get("report_family_id", pd.Series(dtype="object")).nunique() or 0)),
        scene_market_pulse=scene_market_pulse,
        lane_atlas=lane_atlas,
        migration_network=migration_network,
        seed_bridge_atlas=seed_bridge_atlas,
        whitespace_atlas=whitespace_atlas,
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
    }

    manifest_payload = {
        "report_family_count": int(brief_payload["report_family_count"]),
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
