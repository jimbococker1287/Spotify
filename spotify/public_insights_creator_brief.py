from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable

import joblib
import pandas as pd

from .creator_label_intelligence import build_creator_label_intelligence, prepare_creator_intelligence_inputs
from .run_artifacts import write_csv_rows as _shared_write_csv_rows
from .run_artifacts import write_json, write_markdown


_CREATOR_REPORT_FAMILY_VIEW_SPECS: dict[str, dict[str, object]] = {
    "ranking_comparison": {
        "label": "Ranking Comparison",
        "view_group": "comparison",
        "markdown_filename": "ranking_view.md",
        "legacy_markdown_filenames": ("ranking_comparison.md",),
        "csv_filename": "ranking_comparison.csv",
    },
    "scene_comparison": {
        "label": "Scene Comparison",
        "view_group": "comparison",
        "markdown_filename": "scene_view.md",
        "legacy_markdown_filenames": ("scene_comparison.md",),
        "csv_filename": "scene_comparison.csv",
    },
    "seed_comparison": {
        "label": "Seed Comparison",
        "view_group": "comparison",
        "markdown_filename": "seed_view.md",
        "legacy_markdown_filenames": ("seed_comparison.md",),
        "csv_filename": "seed_comparison.csv",
    },
    "scene_seed_comparison": {
        "label": "Scene Vs Seed Comparison",
        "view_group": "comparison",
        "markdown_filename": "scene_seed_view.md",
        "legacy_markdown_filenames": ("scene_seed_comparison.md",),
        "csv_filename": "scene_seed_comparison.csv",
    },
    "opportunity_lane_comparison": {
        "label": "Opportunity-Lane Comparison",
        "view_group": "comparison",
        "markdown_filename": "opportunity_lane_view.md",
        "legacy_markdown_filenames": ("opportunity_lane_comparison.md",),
        "csv_filename": "opportunity_lane_comparison.csv",
    },
    "priority_shortlist": {
        "label": "Priority Shortlist",
        "view_group": "brief",
        "markdown_filename": "",
        "legacy_markdown_filenames": (),
        "csv_filename": "priority_shortlist.csv",
    },
    "migration_watch": {
        "label": "Migration Watch",
        "view_group": "brief",
        "markdown_filename": "",
        "legacy_markdown_filenames": (),
        "csv_filename": "migration_watch.csv",
    },
    "release_watch": {
        "label": "Release Watch",
        "view_group": "brief",
        "markdown_filename": "",
        "legacy_markdown_filenames": (),
        "csv_filename": "release_watch.csv",
    },
    "scene_strategy_watch": {
        "label": "Scene Strategy Watch",
        "view_group": "brief",
        "markdown_filename": "scene_strategy_watch.md",
        "legacy_markdown_filenames": (),
        "csv_filename": "scene_strategy_watch.csv",
    },
}
_CREATOR_REPORT_FAMILY_READING_ORDER = (
    "primary_report",
    "ranking_comparison",
    "opportunity_lane_comparison",
    "scene_comparison",
    "scene_strategy_watch",
    "seed_comparison",
    "scene_seed_comparison",
    "migration_watch",
    "release_watch",
)
_CREATOR_REPORT_FAMILY_REFRESH_ANCHORS = {
    "opportunity_lane": "opportunity_lane_comparison",
    "scene_strategy": "scene_strategy_watch",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "report"


def _write_report(
    *,
    output_dir: Path,
    category: str,
    stem: str,
    payload: dict[str, Any],
    markdown_lines: list[str],
) -> tuple[Path, Path]:
    report_dir = output_dir / "analysis" / "public_spotify" / category
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(report_dir / f"{stem}.json", payload)
    md_path = write_markdown(report_dir / f"{stem}.md", markdown_lines)
    return json_path, md_path


def _write_markdown_lines(path: Path, markdown_lines: list[str]) -> Path:
    return write_markdown(path, markdown_lines)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    return _shared_write_csv_rows(
        path,
        rows,
        fieldnames=fieldnames,
        value_serializer=lambda value: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value,
    )


def _coerce_manifest_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw_value in value.items():
        key_text = str(key).strip()
        value_text = str(raw_value).strip()
        if key_text and value_text:
            out[key_text] = value_text
    return out


def _existing_report_path(report_dir: Path, raw_path: object) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = report_dir / path
    return path.resolve() if path.exists() else None


def _artifact_modified_at(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _creator_report_family_artifact_candidates(
    report_dir: Path,
    stem: str,
    view_key: str,
    *,
    artifact_kind: str,
) -> list[Path]:
    spec = _CREATOR_REPORT_FAMILY_VIEW_SPECS.get(view_key, {})
    if artifact_kind == "markdown":
        names = [
            str(spec.get("markdown_filename", "")).strip(),
            *[str(item).strip() for item in spec.get("legacy_markdown_filenames", ()) if str(item).strip()],
        ]
    else:
        names = [str(spec.get("csv_filename", "")).strip()]
    return [report_dir / f"{stem}_{name}" for name in names if name]


def _resolve_report_artifact_path(
    report_dir: Path,
    raw_path: object,
    candidates: list[Path],
) -> str | None:
    existing_path = _existing_report_path(report_dir, raw_path)
    if existing_path is not None:
        return str(existing_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def normalize_creator_report_family_manifest(
    manifest: dict[str, Any],
    *,
    report_dir: Path,
    stem: str,
    refreshed_at: str | None = None,
    refresh_source: str | None = None,
) -> dict[str, Any]:
    normalized = dict(manifest)
    known_comparison_keys = {
        key
        for key, spec in _CREATOR_REPORT_FAMILY_VIEW_SPECS.items()
        if str(spec.get("view_group", "")) == "comparison"
    }
    known_brief_keys = {
        key
        for key, spec in _CREATOR_REPORT_FAMILY_VIEW_SPECS.items()
        if str(spec.get("view_group", "")) == "brief"
    }
    comparison_md = _coerce_manifest_map(manifest.get("comparison_view_markdown"))
    comparison_csv = _coerce_manifest_map(manifest.get("comparison_view_csv"))
    brief_md = _coerce_manifest_map(manifest.get("brief_view_markdown"))
    brief_csv = _coerce_manifest_map(manifest.get("brief_view_csv"))

    primary_report_path = _resolve_report_artifact_path(
        report_dir,
        manifest.get("primary_report"),
        [report_dir / f"{stem}.md"],
    )
    primary_report_json = _resolve_report_artifact_path(
        report_dir,
        manifest.get("primary_report_json"),
        [report_dir / f"{stem}.json"],
    )
    if primary_report_path:
        normalized["primary_report"] = primary_report_path
    else:
        normalized.pop("primary_report", None)
    if primary_report_json:
        normalized["primary_report_json"] = primary_report_json
    else:
        normalized.pop("primary_report_json", None)

    normalized_comparison_md = {key: value for key, value in comparison_md.items() if key not in known_comparison_keys}
    normalized_comparison_csv = {key: value for key, value in comparison_csv.items() if key not in known_comparison_keys}
    normalized_brief_md = {key: value for key, value in brief_md.items() if key not in known_brief_keys}
    normalized_brief_csv = {key: value for key, value in brief_csv.items() if key not in known_brief_keys}
    view_inventory: dict[str, dict[str, object]] = {}

    for view_key, spec in _CREATOR_REPORT_FAMILY_VIEW_SPECS.items():
        view_group = str(spec.get("view_group", "")).strip()
        markdown_lookup = comparison_md if view_group == "comparison" else brief_md
        csv_lookup = comparison_csv if view_group == "comparison" else brief_csv
        markdown_path = _resolve_report_artifact_path(
            report_dir,
            markdown_lookup.get(view_key),
            _creator_report_family_artifact_candidates(report_dir, stem, view_key, artifact_kind="markdown"),
        )
        csv_path = _resolve_report_artifact_path(
            report_dir,
            csv_lookup.get(view_key),
            _creator_report_family_artifact_candidates(report_dir, stem, view_key, artifact_kind="csv"),
        )
        if markdown_path:
            if view_group == "comparison":
                normalized_comparison_md[view_key] = markdown_path
            else:
                normalized_brief_md[view_key] = markdown_path
        if csv_path:
            if view_group == "comparison":
                normalized_comparison_csv[view_key] = csv_path
            else:
                normalized_brief_csv[view_key] = csv_path

        markdown_resolved = _existing_report_path(report_dir, markdown_path)
        csv_resolved = _existing_report_path(report_dir, csv_path)
        view_inventory[view_key] = {
            "label": str(spec.get("label", view_key.replace("_", " ").title())),
            "view_group": view_group,
            "markdown_path": markdown_path,
            "csv_path": csv_path,
            "markdown_modified_at": _artifact_modified_at(markdown_resolved),
            "csv_modified_at": _artifact_modified_at(csv_resolved),
            "ready": bool(markdown_path and csv_path),
            "canonical_markdown_filename": (
                f"{stem}_{str(spec.get('markdown_filename', '')).strip()}"
                if str(spec.get("markdown_filename", "")).strip()
                else None
            ),
            "canonical_csv_filename": f"{stem}_{str(spec.get('csv_filename', '')).strip()}",
        }
        legacy_filenames = [str(item).strip() for item in spec.get("legacy_markdown_filenames", ()) if str(item).strip()]
        if legacy_filenames:
            view_inventory[view_key]["legacy_markdown_filenames"] = [f"{stem}_{name}" for name in legacy_filenames]

    normalized["comparison_view_markdown"] = normalized_comparison_md
    normalized["comparison_view_csv"] = normalized_comparison_csv
    normalized["brief_view_markdown"] = normalized_brief_md
    normalized["brief_view_csv"] = normalized_brief_csv

    existing_order = manifest.get("reading_order")
    ordered_candidates = (
        [str(item).strip() for item in existing_order if str(item).strip()]
        if isinstance(existing_order, list)
        else list(_CREATOR_REPORT_FAMILY_READING_ORDER)
    )
    available_markdown_keys = set(normalized_comparison_md) | set(normalized_brief_md)
    reading_order: list[str] = []
    for item in ordered_candidates:
        if item == "primary_report":
            if primary_report_path and item not in reading_order:
                reading_order.append(item)
            continue
        if item in _CREATOR_REPORT_FAMILY_VIEW_SPECS:
            if item in available_markdown_keys and item not in reading_order:
                reading_order.append(item)
            continue
        if item not in reading_order:
            reading_order.append(item)
    for item in _CREATOR_REPORT_FAMILY_READING_ORDER:
        if item == "primary_report":
            if primary_report_path and item not in reading_order:
                reading_order.append(item)
            continue
        if item in available_markdown_keys and item not in reading_order:
            reading_order.append(item)
    normalized["reading_order"] = reading_order

    anchor_views: dict[str, dict[str, object]] = {}
    for anchor_name, view_key in _CREATOR_REPORT_FAMILY_REFRESH_ANCHORS.items():
        inventory = view_inventory.get(view_key, {})
        anchor_views[anchor_name] = {
            "view_key": view_key,
            "label": inventory.get("label"),
            "view_group": inventory.get("view_group"),
            "markdown_path": inventory.get("markdown_path"),
            "csv_path": inventory.get("csv_path"),
            "markdown_modified_at": inventory.get("markdown_modified_at"),
            "csv_modified_at": inventory.get("csv_modified_at"),
            "ready": bool(inventory.get("ready")),
        }

    existing_packaging = manifest.get("packaging_metadata", {})
    packaging_metadata = dict(existing_packaging) if isinstance(existing_packaging, dict) else {}
    packaging_metadata.update(
        {
            "version": 2,
            "primary_report_stem": stem,
            "reading_order": list(reading_order),
            "comparison_view_markdown_count": len(normalized_comparison_md),
            "comparison_view_csv_count": len(normalized_comparison_csv),
            "brief_view_markdown_count": len(normalized_brief_md),
            "brief_view_csv_count": len(normalized_brief_csv),
            "refresh_anchor_keys": list(_CREATOR_REPORT_FAMILY_REFRESH_ANCHORS.values()),
            "refresh_anchor_ready": all(bool(row.get("ready")) for row in anchor_views.values()),
            "anchor_views": anchor_views,
            "view_inventory": view_inventory,
        }
    )
    if refreshed_at:
        packaging_metadata["normalized_at"] = refreshed_at
    elif str(packaging_metadata.get("normalized_at", "")).strip():
        packaging_metadata["normalized_at"] = str(packaging_metadata.get("normalized_at")).strip()
    else:
        packaging_metadata.pop("normalized_at", None)
    if refresh_source:
        packaging_metadata["refresh_source"] = refresh_source
    elif str(packaging_metadata.get("refresh_source", "")).strip():
        packaging_metadata["refresh_source"] = str(packaging_metadata.get("refresh_source")).strip()
    else:
        packaging_metadata.pop("refresh_source", None)
    normalized["packaging_metadata"] = packaging_metadata
    return normalized


def _resolve_multimodal_artifact(
    *,
    output_dir: Path,
    run_dir: str | None,
    multimodal_artifact: str | None,
) -> Path | None:
    if multimodal_artifact:
        path = Path(multimodal_artifact).expanduser().resolve()
        return path if path.exists() else None

    if run_dir:
        candidate = Path(run_dir).expanduser().resolve()
        if candidate.is_file():
            return candidate if candidate.exists() else None
        artifact_path = candidate / "analysis" / "multimodal" / "multimodal_artist_space.joblib"
        return artifact_path if artifact_path.exists() else None

    paths = sorted(output_dir.glob("runs/*/analysis/multimodal/multimodal_artist_space.joblib"))
    return paths[-1] if paths else None


def _creator_brief_scene_comparison(payload: dict[str, Any]) -> list[dict[str, object]]:
    scenes = payload.get("scenes", [])
    scenes = scenes if isinstance(scenes, list) else []
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    fan_migration = payload.get("fan_migration", [])
    fan_migration = fan_migration if isinstance(fan_migration, list) else []

    top_opportunity_by_scene: dict[int, dict[str, object]] = {}
    opportunity_count_by_scene: Counter[int] = Counter()
    priority_count_by_scene: Counter[int] = Counter()
    watchlist_count_by_scene: Counter[int] = Counter()
    scene_seed_coverage: dict[int, set[str]] = {}
    opportunity_rows_by_scene: dict[int, list[dict[str, object]]] = {}
    for row in opportunities:
        if not isinstance(row, dict):
            continue
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, int):
            continue
        opportunity_count_by_scene[int(scene_id)] += 1
        opportunity_rows_by_scene.setdefault(int(scene_id), []).append(row)
        if str(row.get("opportunity_band", "")) == "priority_now":
            priority_count_by_scene[int(scene_id)] += 1
        elif str(row.get("opportunity_band", "")) == "watchlist":
            watchlist_count_by_scene[int(scene_id)] += 1
        seed_bucket = scene_seed_coverage.setdefault(int(scene_id), set())
        connected_seed_artists = row.get("connected_seed_artists", [])
        if isinstance(connected_seed_artists, list):
            seed_bucket.update(str(item) for item in connected_seed_artists if str(item).strip())
        current_best = top_opportunity_by_scene.get(int(scene_id))
        if current_best is None or float(row.get("opportunity_score", 0.0) or 0.0) > float(
            current_best.get("opportunity_score", 0.0) or 0.0
        ):
            top_opportunity_by_scene[int(scene_id)] = row

    top_migration_by_scene: dict[int, dict[str, object]] = {}
    for row in fan_migration:
        if not isinstance(row, dict):
            continue
        target_scene_id = row.get("target_scene_id")
        if not isinstance(target_scene_id, int):
            continue
        current_best = top_migration_by_scene.get(int(target_scene_id))
        if current_best is None or float(row.get("source_out_share", 0.0) or 0.0) > float(
            current_best.get("source_out_share", 0.0) or 0.0
        ):
            top_migration_by_scene[int(target_scene_id)] = row

    rows: list[dict[str, object]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = int(scene.get("scene_id", -1))
        top_opportunity = top_opportunity_by_scene.get(scene_id, {})
        scene_opportunities = opportunity_rows_by_scene.get(scene_id, [])
        avg_opportunity_score = (
            sum(float(item.get("opportunity_score", 0.0) or 0.0) for item in scene_opportunities)
            / max(len(scene_opportunities), 1)
        )
        top_migration = top_migration_by_scene.get(scene_id, {})
        rows.append(
            {
                "scene_id": scene_id,
                "scene_name": str(scene.get("scene_name", "")),
                "scene_local_play_share": float(scene.get("scene_local_play_share", 0.0) or 0.0),
                "scene_release_pressure": float(scene.get("scene_release_pressure", 0.0) or 0.0),
                "scene_label_concentration": float(scene.get("scene_label_concentration", 0.0) or 0.0),
                "seed_count": int(scene.get("seed_count", 0) or 0),
                "artist_count": int(scene.get("artist_count", 0) or 0),
                "opportunity_count": int(opportunity_count_by_scene.get(scene_id, 0)),
                "priority_now_count": int(priority_count_by_scene.get(scene_id, 0)),
                "watchlist_count": int(watchlist_count_by_scene.get(scene_id, 0)),
                "avg_opportunity_score": round(float(avg_opportunity_score), 4) if scene_opportunities else 0.0,
                "top_opportunity_artist": str(top_opportunity.get("artist_name", "")),
                "top_opportunity_score": float(top_opportunity.get("opportunity_score", 0.0) or 0.0),
                "top_seed_artists": sorted(scene_seed_coverage.get(scene_id, set())),
                "top_migration_route": (
                    f"{top_migration.get('source_artist', '')} -> {top_migration.get('target_artist', '')}"
                    if top_migration
                    else ""
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["scene_local_play_share"]),
            float(row["avg_opportunity_score"]),
            int(row["opportunity_count"]),
            float(row["top_opportunity_score"]),
        ),
        reverse=True,
    )
    return rows


def _creator_brief_seed_comparison(payload: dict[str, Any]) -> list[dict[str, object]]:
    adjacency = payload.get("artist_adjacency", [])
    adjacency = adjacency if isinstance(adjacency, list) else []
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    scenes = payload.get("scenes", [])
    scenes = scenes if isinstance(scenes, list) else []

    opportunity_by_artist = {
        str(row.get("artist_name", "")): float(row.get("opportunity_score", 0.0) or 0.0)
        for row in opportunities
        if isinstance(row, dict)
    }
    scene_share_lookup = {
        str(row.get("scene_name", "")): float(row.get("scene_local_play_share", 0.0) or 0.0)
        for row in scenes
        if isinstance(row, dict)
    }
    opportunity_rows_by_seed: dict[str, list[dict[str, object]]] = {}
    for row in opportunities:
        if not isinstance(row, dict):
            continue
        connected_seed_artists = row.get("connected_seed_artists", [])
        if not isinstance(connected_seed_artists, list):
            continue
        for seed_artist in [str(item) for item in connected_seed_artists if str(item).strip()]:
            opportunity_rows_by_seed.setdefault(seed_artist, []).append(row)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in adjacency:
        if not isinstance(row, dict):
            continue
        source_artist = str(row.get("source_artist", "")).strip()
        if not source_artist:
            continue
        grouped.setdefault(source_artist, []).append(row)

    rows: list[dict[str, object]] = []
    for source_artist, source_rows in grouped.items():
        ordered = sorted(
            source_rows,
            key=lambda row: (
                float(row.get("hybrid_score", 0.0) or 0.0),
                float(row.get("transition_share", 0.0) or 0.0),
            ),
            reverse=True,
        )
        best = ordered[0]
        best_target = str(best.get("target_artist", ""))
        seed_opportunities = opportunity_rows_by_seed.get(source_artist, [])
        scene_groups: dict[str, list[dict[str, object]]] = {}
        for opportunity_row in seed_opportunities:
            scene_name = str(opportunity_row.get("scene_name", "")).strip() or "unmapped"
            scene_groups.setdefault(scene_name, []).append(opportunity_row)
        top_scene_name = ""
        top_scene_avg_score = 0.0
        if scene_groups:
            top_scene_name, top_scene_rows = max(
                scene_groups.items(),
                key=lambda item: (
                    sum(float(row.get("opportunity_score", 0.0) or 0.0) for row in item[1]) / max(len(item[1]), 1),
                    len(item[1]),
                ),
            )
            top_scene_avg_score = sum(float(row.get("opportunity_score", 0.0) or 0.0) for row in top_scene_rows) / max(
                len(top_scene_rows),
                1,
            )
        top_seed_opportunity = max(
            seed_opportunities,
            key=lambda row: float(row.get("opportunity_score", 0.0) or 0.0),
            default={},
        )
        rows.append(
            {
                "seed_artist": source_artist,
                "top_adjacent_artist": best_target,
                "top_hybrid_score": float(best.get("hybrid_score", 0.0) or 0.0),
                "top_transition_share": float(best.get("transition_share", 0.0) or 0.0),
                "top_target_opportunity_score": float(opportunity_by_artist.get(best_target, 0.0)),
                "adjacent_artist_count": int(len(ordered)),
                "scene_coverage_count": int(len(scene_groups)),
                "priority_now_count": int(
                    sum(1 for row in seed_opportunities if str(row.get("opportunity_band", "")) == "priority_now")
                ),
                "watchlist_count": int(
                    sum(1 for row in seed_opportunities if str(row.get("opportunity_band", "")) == "watchlist")
                ),
                "top_scene_name": top_scene_name,
                "top_scene_avg_opportunity_score": round(float(top_scene_avg_score), 4) if top_scene_name else 0.0,
                "top_scene_play_share": float(scene_share_lookup.get(top_scene_name, 0.0)),
                "top_opportunity_artist": str(top_seed_opportunity.get("artist_name", "")),
                "top_opportunity_score": float(top_seed_opportunity.get("opportunity_score", 0.0) or 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["top_hybrid_score"]),
            float(row["top_opportunity_score"]),
            float(row["top_target_opportunity_score"]),
        ),
        reverse=True,
    )
    return rows


def _creator_brief_ranking_comparison(payload: dict[str, Any]) -> list[dict[str, object]]:
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    rows: list[dict[str, object]] = []
    for row in opportunities[:10]:
        if not isinstance(row, dict):
            continue
        connected_seed_artists = row.get("connected_seed_artists", [])
        seed_bridges = (
            [str(item) for item in connected_seed_artists if str(item).strip()]
            if isinstance(connected_seed_artists, list)
            else []
        )
        release_component = float(row.get("freshness_component", 0.0) or 0.0) + float(
            row.get("whitespace_component", 0.0) or 0.0
        )
        scene_component = float(row.get("scene_momentum_component", 0.0) or 0.0) + float(
            row.get("label_concentration_component", 0.0) or 0.0
        )
        gap_component = float(row.get("local_gap_component", 0.0) or 0.0) + float(
            row.get("popularity_tail_component", 0.0) or 0.0
        )
        rows.append(
            {
                "opportunity_rank": int(row.get("opportunity_rank", len(rows) + 1) or len(rows) + 1),
                "artist_name": str(row.get("artist_name", "")),
                "scene_name": str(row.get("scene_name", "")),
                "seed_bridges": seed_bridges,
                "opportunity_band": str(row.get("opportunity_band", "")),
                "primary_driver": str(row.get("primary_driver", "")),
                "opportunity_score": float(row.get("opportunity_score", 0.0) or 0.0),
                "adjacency_component": float(row.get("adjacency_component", 0.0) or 0.0),
                "migration_component": float(row.get("migration_component", 0.0) or 0.0),
                "release_component": round(float(release_component), 4),
                "scene_component": round(float(scene_component), 4),
                "gap_component": round(float(gap_component), 4),
                "why_now": str(row.get("why_now", "")),
            }
        )
    return rows


def _creator_brief_opportunity_lane_comparison(payload: dict[str, Any]) -> list[dict[str, object]]:
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    scenes = payload.get("scenes", [])
    scenes = scenes if isinstance(scenes, list) else []
    release_whitespace = payload.get("release_whitespace", [])
    release_whitespace = release_whitespace if isinstance(release_whitespace, list) else []
    fan_migration = payload.get("fan_migration", [])
    fan_migration = fan_migration if isinstance(fan_migration, list) else []

    scene_lookup = {
        str(row.get("scene_name", "")).strip(): row
        for row in scenes
        if isinstance(row, dict) and str(row.get("scene_name", "")).strip()
    }
    artist_scene_lookup = {
        str(row.get("artist_name", "")).strip(): str(row.get("scene_name", "")).strip()
        for row in opportunities
        if isinstance(row, dict)
        and str(row.get("artist_name", "")).strip()
        and str(row.get("scene_name", "")).strip()
    }

    whitespace_by_scene: dict[str, dict[str, object]] = {}
    for row in release_whitespace:
        if not isinstance(row, dict):
            continue
        scene_name = str(row.get("scene_name", "")).strip()
        if not scene_name:
            continue
        current_best = whitespace_by_scene.get(scene_name)
        if current_best is None or float(row.get("release_whitespace_score", 0.0) or 0.0) > float(
            current_best.get("release_whitespace_score", 0.0) or 0.0
        ):
            whitespace_by_scene[scene_name] = row

    incoming_migration_by_scene: dict[str, float] = {}
    for row in fan_migration:
        if not isinstance(row, dict):
            continue
        scene_name = ""
        if isinstance(row.get("target_scene_id"), int):
            scene_name = next(
                (
                    str(scene.get("scene_name", "")).strip()
                    for scene in scenes
                    if isinstance(scene, dict) and int(scene.get("scene_id", -1)) == int(row.get("target_scene_id"))
                ),
                "",
            )
        if not scene_name:
            scene_name = artist_scene_lookup.get(str(row.get("target_artist", "")).strip(), "")
        if not scene_name:
            continue
        share = float(row.get("source_out_share", 0.0) or 0.0)
        incoming_migration_by_scene[scene_name] = max(incoming_migration_by_scene.get(scene_name, 0.0), share)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in opportunities:
        if not isinstance(row, dict):
            continue
        scene_name = str(row.get("scene_name", "")).strip() or "unmapped"
        primary_driver = str(row.get("primary_driver", "")).strip() or "unassigned"
        grouped.setdefault((scene_name, primary_driver), []).append(row)

    rows: list[dict[str, object]] = []
    for (scene_name, primary_driver), lane_rows in grouped.items():
        scene_row = scene_lookup.get(scene_name, {})
        best = max(lane_rows, key=lambda row: float(row.get("opportunity_score", 0.0) or 0.0))
        whitespace_row = whitespace_by_scene.get(scene_name, {})
        avg_score = sum(float(row.get("opportunity_score", 0.0) or 0.0) for row in lane_rows) / max(len(lane_rows), 1)
        priority_now_count = sum(1 for row in lane_rows if str(row.get("opportunity_band", "")) == "priority_now")
        watchlist_count = sum(1 for row in lane_rows if str(row.get("opportunity_band", "")) == "watchlist")
        seed_bridge_count = len(
            {
                str(seed_artist).strip()
                for lane_row in lane_rows
                for seed_artist in (
                    lane_row.get("connected_seed_artists", [])
                    if isinstance(lane_row.get("connected_seed_artists", []), list)
                    else []
                )
                if str(seed_artist).strip()
            }
        )
        incoming_migration_share = float(incoming_migration_by_scene.get(scene_name, 0.0))
        whitespace_score = float(whitespace_row.get("release_whitespace_score", 0.0) or 0.0)
        release_pressure = float(scene_row.get("scene_release_pressure", 0.0) or 0.0)
        label_concentration = float(scene_row.get("scene_label_concentration", 0.0) or 0.0)
        if primary_driver == "release_whitespace" and whitespace_score >= 1.0:
            lane_posture = "cadence_capture"
        elif primary_driver == "fan_migration" and incoming_migration_share >= 0.20:
            lane_posture = "migration_capture"
        elif label_concentration >= 0.80 and release_pressure >= 0.80:
            lane_posture = "competitive_scene"
        elif primary_driver == "seed_adjacency":
            lane_posture = "adjacency_expansion"
        else:
            lane_posture = "watch"
        rows.append(
            {
                "scene_name": scene_name,
                "primary_driver": primary_driver,
                "lane_posture": lane_posture,
                "opportunity_count": int(len(lane_rows)),
                "priority_now_count": int(priority_now_count),
                "watchlist_count": int(watchlist_count),
                "avg_opportunity_score": round(float(avg_score), 4),
                "top_opportunity_artist": str(best.get("artist_name", "")),
                "top_opportunity_score": float(best.get("opportunity_score", 0.0) or 0.0),
                "seed_bridge_count": int(seed_bridge_count),
                "incoming_migration_share": incoming_migration_share,
                "release_whitespace_score": whitespace_score,
                "scene_local_play_share": float(scene_row.get("scene_local_play_share", 0.0) or 0.0),
                "scene_release_pressure": release_pressure,
                "scene_label_concentration": label_concentration,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["priority_now_count"]),
            float(row["avg_opportunity_score"]),
            float(row["incoming_migration_share"]),
            float(row["release_whitespace_score"]),
        ),
        reverse=True,
    )
    return rows


def _creator_brief_scene_seed_comparison(payload: dict[str, Any]) -> list[dict[str, object]]:
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    scenes = payload.get("scenes", [])
    scenes = scenes if isinstance(scenes, list) else []
    scene_lookup = {
        str(row.get("scene_name", "")): row
        for row in scenes
        if isinstance(row, dict)
    }

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in opportunities:
        if not isinstance(row, dict):
            continue
        scene_name = str(row.get("scene_name", "")).strip() or "unmapped"
        connected_seed_artists = row.get("connected_seed_artists", [])
        seed_artists = (
            [str(item) for item in connected_seed_artists if str(item).strip()]
            if isinstance(connected_seed_artists, list) and connected_seed_artists
            else ["unassigned"]
        )
        for seed_artist in seed_artists:
            grouped.setdefault((scene_name, seed_artist), []).append(row)

    rows: list[dict[str, object]] = []
    for (scene_name, seed_artist), group_rows in grouped.items():
        best = max(group_rows, key=lambda row: float(row.get("opportunity_score", 0.0) or 0.0))
        top_driver_counter: Counter[str] = Counter(str(row.get("primary_driver", "")) for row in group_rows)
        scene_row = scene_lookup.get(scene_name, {})
        rows.append(
            {
                "scene_name": scene_name,
                "seed_artist": seed_artist,
                "bridge_artist_count": int(len({str(row.get('artist_name', '')) for row in group_rows})),
                "opportunity_count": int(len(group_rows)),
                "avg_opportunity_score": round(
                    sum(float(row.get("opportunity_score", 0.0) or 0.0) for row in group_rows) / max(len(group_rows), 1),
                    4,
                ),
                "top_opportunity_artist": str(best.get("artist_name", "")),
                "top_opportunity_score": float(best.get("opportunity_score", 0.0) or 0.0),
                "top_driver": top_driver_counter.most_common(1)[0][0] if top_driver_counter else "",
                "scene_local_play_share": float(scene_row.get("scene_local_play_share", 0.0) or 0.0),
                "scene_release_pressure": float(scene_row.get("scene_release_pressure", 0.0) or 0.0),
                "scene_label_concentration": float(scene_row.get("scene_label_concentration", 0.0) or 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["avg_opportunity_score"]),
            int(row["opportunity_count"]),
            float(row["scene_local_play_share"]),
        ),
        reverse=True,
    )
    return rows


def _creator_brief_scene_strategy_watch(payload: dict[str, Any]) -> list[dict[str, object]]:
    scenes = payload.get("scenes", [])
    scenes = scenes if isinstance(scenes, list) else []
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    release_whitespace = payload.get("release_whitespace", [])
    release_whitespace = release_whitespace if isinstance(release_whitespace, list) else []
    fan_migration = payload.get("fan_migration", [])
    fan_migration = fan_migration if isinstance(fan_migration, list) else []

    artist_scene_lookup = {
        str(row.get("artist_name", "")).strip(): str(row.get("scene_name", "")).strip()
        for row in opportunities
        if isinstance(row, dict)
        and str(row.get("artist_name", "")).strip()
        and str(row.get("scene_name", "")).strip()
    }
    opportunity_rows_by_scene: dict[str, list[dict[str, object]]] = {}
    for row in opportunities:
        if not isinstance(row, dict):
            continue
        scene_name = str(row.get("scene_name", "")).strip() or "unmapped"
        opportunity_rows_by_scene.setdefault(scene_name, []).append(row)

    whitespace_by_scene: dict[str, dict[str, object]] = {}
    for row in release_whitespace:
        if not isinstance(row, dict):
            continue
        scene_name = str(row.get("scene_name", "")).strip()
        if not scene_name:
            continue
        current_best = whitespace_by_scene.get(scene_name)
        if current_best is None or float(row.get("release_whitespace_score", 0.0) or 0.0) > float(
            current_best.get("release_whitespace_score", 0.0) or 0.0
        ):
            whitespace_by_scene[scene_name] = row

    incoming_migration_by_scene: dict[str, float] = {}
    outgoing_migration_by_scene: dict[str, float] = {}
    for row in fan_migration:
        if not isinstance(row, dict):
            continue
        target_scene_name = ""
        if isinstance(row.get("target_scene_id"), int):
            target_scene_name = next(
                (
                    str(scene.get("scene_name", "")).strip()
                    for scene in scenes
                    if isinstance(scene, dict) and int(scene.get("scene_id", -1)) == int(row.get("target_scene_id"))
                ),
                "",
            )
        if not target_scene_name:
            target_scene_name = artist_scene_lookup.get(str(row.get("target_artist", "")).strip(), "")
        source_scene_name = ""
        if isinstance(row.get("source_scene_id"), int):
            source_scene_name = next(
                (
                    str(scene.get("scene_name", "")).strip()
                    for scene in scenes
                    if isinstance(scene, dict) and int(scene.get("scene_id", -1)) == int(row.get("source_scene_id"))
                ),
                "",
            )
        if not source_scene_name:
            source_scene_name = artist_scene_lookup.get(str(row.get("source_artist", "")).strip(), "")
        share = float(row.get("source_out_share", 0.0) or 0.0)
        if target_scene_name:
            incoming_migration_by_scene[target_scene_name] = max(incoming_migration_by_scene.get(target_scene_name, 0.0), share)
        if source_scene_name:
            outgoing_migration_by_scene[source_scene_name] = max(outgoing_migration_by_scene.get(source_scene_name, 0.0), share)

    lane_rows = _creator_brief_opportunity_lane_comparison(payload)
    top_lane_by_scene: dict[str, dict[str, object]] = {}
    for row in lane_rows:
        scene_name = str(row.get("scene_name", "")).strip()
        if scene_name and scene_name not in top_lane_by_scene:
            top_lane_by_scene[scene_name] = row

    rows: list[dict[str, object]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_name = str(scene.get("scene_name", "")).strip()
        scene_opportunities = opportunity_rows_by_scene.get(scene_name, [])
        priority_now_count = sum(
            1 for row in scene_opportunities if str(row.get("opportunity_band", "")).strip() == "priority_now"
        )
        whitespace_row = whitespace_by_scene.get(scene_name, {})
        top_lane = top_lane_by_scene.get(scene_name, {})
        incoming_share = float(incoming_migration_by_scene.get(scene_name, 0.0))
        outgoing_share = float(outgoing_migration_by_scene.get(scene_name, 0.0))
        release_pressure = float(scene.get("scene_release_pressure", 0.0) or 0.0)
        label_concentration = float(scene.get("scene_label_concentration", 0.0) or 0.0)
        whitespace_score = float(whitespace_row.get("release_whitespace_score", 0.0) or 0.0)
        if incoming_share >= 0.20 and whitespace_score >= 1.0:
            posture = "accelerate_capture"
            action_note = "Inbound listener movement and release whitespace are both present, so this scene can support a deliberate push now."
        elif label_concentration >= 0.80 and release_pressure >= 0.80:
            posture = "protect_window"
            action_note = "Label concentration and release pressure are both high, so any launch here needs sharper timing and partner awareness."
        elif outgoing_share >= 0.20:
            posture = "defend_audience"
            action_note = "Listener outflow is visible, so use this scene to stabilize audience retention before expanding elsewhere."
        elif release_pressure >= 0.80:
            posture = "refresh_catalog"
            action_note = "Release pressure is elevated without strong inflow yet, so catalog freshness is the main lever."
        else:
            posture = "steady_watch"
            action_note = "The scene is viable but not urgent; keep it in the brief as a supporting lane."
        rows.append(
            {
                "scene_name": scene_name,
                "strategy_posture": posture,
                "action_note": action_note,
                "scene_local_play_share": float(scene.get("scene_local_play_share", 0.0) or 0.0),
                "scene_release_pressure": release_pressure,
                "scene_label_concentration": label_concentration,
                "incoming_migration_share": incoming_share,
                "outgoing_migration_share": outgoing_share,
                "release_whitespace_anchor_artist": str(whitespace_row.get("artist_name", "")),
                "release_whitespace_score": whitespace_score,
                "opportunity_count": int(len(scene_opportunities)),
                "priority_now_count": int(priority_now_count),
                "top_lane_driver": str(top_lane.get("primary_driver", "")),
                "top_lane_posture": str(top_lane.get("lane_posture", "")),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["priority_now_count"]),
            float(row["incoming_migration_share"]),
            float(row["scene_release_pressure"]),
            float(row["scene_local_play_share"]),
        ),
        reverse=True,
    )
    return rows


def _creator_brief_executive_summary(payload: dict[str, Any]) -> list[str]:
    summary = payload.get("graph_summary", {})
    summary = summary if isinstance(summary, dict) else {}
    scene_comparison = _creator_brief_scene_comparison(payload)
    seed_comparison = _creator_brief_seed_comparison(payload)
    opportunity_lanes = _creator_brief_opportunity_lane_comparison(payload)
    scene_strategy_watch = _creator_brief_scene_strategy_watch(payload)
    release_whitespace = payload.get("release_whitespace", [])
    release_whitespace = release_whitespace if isinstance(release_whitespace, list) else []
    fan_migration = payload.get("fan_migration", [])
    fan_migration = fan_migration if isinstance(fan_migration, list) else []
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []

    lines = [
        (
            f"The graph covers `{summary.get('node_count', 0)}` artists across `{summary.get('scene_count', 0)}` scenes, "
            f"with `{summary.get('opportunity_count', 0)}` ranked opportunity lanes."
        )
    ]
    if scene_comparison:
        top_scene = scene_comparison[0]
        lines.append(
            f"The strongest current scene is `{top_scene['scene_name']}` with local play share `{top_scene['scene_local_play_share']:.3f}` and `{top_scene['opportunity_count']}` mapped opportunities."
        )
    if opportunity_lanes:
        top_lane = opportunity_lanes[0]
        lines.append(
            f"The strongest opportunity lane is `{top_lane['scene_name']} / {top_lane['primary_driver']}` with average score `{top_lane['avg_opportunity_score']:.3f}` across `{top_lane['opportunity_count']}` candidates."
        )
    if opportunities:
        top_opportunity = opportunities[0]
        lines.append(
            f"The clearest near-term opportunity is `{top_opportunity.get('artist_name', '')}` in `{top_opportunity.get('scene_name', 'unmapped')}` with score `{float(top_opportunity.get('opportunity_score', 0.0) or 0.0):.3f}`."
        )
    if release_whitespace:
        top_whitespace = release_whitespace[0]
        lines.append(
            f"Release whitespace is strongest around `{top_whitespace.get('artist_name', '')}`, whose cadence score is `{float(top_whitespace.get('release_whitespace_score', 0.0) or 0.0):.3f}`."
        )
    elif seed_comparison:
        top_seed = seed_comparison[0]
        lines.append(
            f"`{top_seed['seed_artist']}` is best positioned to bridge into `{top_seed['top_adjacent_artist']}` with hybrid score `{top_seed['top_hybrid_score']:.3f}`."
        )
    if fan_migration:
        meaningful_migrations = [
            row
            for row in fan_migration
            if isinstance(row, dict) and str(row.get("source_artist", "")) != str(row.get("target_artist", ""))
        ]
        top_migration = max(
            meaningful_migrations or [row for row in fan_migration if isinstance(row, dict)],
            key=lambda row: float(row.get("source_out_share", 0.0) or 0.0),
            default={},
        )
        if top_migration:
            lines.append(
                f"The strongest fan-migration route is `{top_migration.get('source_artist', '')} -> {top_migration.get('target_artist', '')}` at share `{float(top_migration.get('source_out_share', 0.0) or 0.0):.3f}`."
            )
    if scene_strategy_watch:
        top_scene_strategy = scene_strategy_watch[0]
        lines.append(
            f"Strategically, `{top_scene_strategy['scene_name']}` is in `{top_scene_strategy['strategy_posture']}` mode because release pressure is `{top_scene_strategy['scene_release_pressure']:.3f}` and inbound migration is `{top_scene_strategy['incoming_migration_share']:.3f}`."
        )
    return lines[:6]


def _creator_brief_priority_shortlist(payload: dict[str, Any]) -> list[dict[str, object]]:
    opportunities = payload.get("opportunities", [])
    opportunities = opportunities if isinstance(opportunities, list) else []
    rows: list[dict[str, object]] = []
    for row in opportunities[:5]:
        if not isinstance(row, dict):
            continue
        connected_seed_artists = row.get("connected_seed_artists", [])
        connected_seed_artists = (
            [str(item) for item in connected_seed_artists if str(item).strip()]
            if isinstance(connected_seed_artists, list)
            else []
        )
        dominant_release_labels = row.get("dominant_release_labels", [])
        dominant_release_labels = (
            [str(item) for item in dominant_release_labels if str(item).strip()]
            if isinstance(dominant_release_labels, list)
            else []
        )
        rows.append(
            {
                "opportunity_rank": int(row.get("opportunity_rank", len(rows) + 1) or len(rows) + 1),
                "artist_name": str(row.get("artist_name", "")),
                "scene_name": str(row.get("scene_name", "")),
                "opportunity_score": float(row.get("opportunity_score", 0.0) or 0.0),
                "opportunity_band": str(row.get("opportunity_band", "")),
                "primary_driver": str(row.get("primary_driver", "")),
                "connected_seed_artists": connected_seed_artists,
                "dominant_release_labels": dominant_release_labels,
                "why_now": str(row.get("why_now", "")),
            }
        )
    return rows


def _creator_brief_migration_watch(payload: dict[str, Any]) -> list[dict[str, object]]:
    migration_rows = payload.get("fan_migration", [])
    migration_rows = migration_rows if isinstance(migration_rows, list) else []
    meaningful_rows = [
        item
        for item in migration_rows
        if isinstance(item, dict) and str(item.get("source_artist", "")) != str(item.get("target_artist", ""))
    ]
    rows: list[dict[str, object]] = []
    for row in sorted(
        meaningful_rows or [item for item in migration_rows if isinstance(item, dict)],
        key=lambda item: (float(item.get("source_out_share", 0.0) or 0.0), int(item.get("transition_count", 0) or 0)),
        reverse=True,
    )[:5]:
        rows.append(
            {
                "source_artist": str(row.get("source_artist", "")),
                "target_artist": str(row.get("target_artist", "")),
                "source_scene_id": row.get("source_scene_id"),
                "target_scene_id": row.get("target_scene_id"),
                "transition_count": int(row.get("transition_count", 0) or 0),
                "source_out_share": float(row.get("source_out_share", 0.0) or 0.0),
                "target_in_share": float(row.get("target_in_share", 0.0) or 0.0),
            }
        )
    return rows


def _creator_brief_release_watch(payload: dict[str, Any]) -> list[dict[str, object]]:
    whitespace_rows = payload.get("release_whitespace", [])
    whitespace_rows = whitespace_rows if isinstance(whitespace_rows, list) else []
    rows: list[dict[str, object]] = []
    for row in whitespace_rows[:5]:
        if not isinstance(row, dict):
            continue
        dominant_release_labels = row.get("dominant_release_labels", [])
        dominant_release_labels = (
            [str(item) for item in dominant_release_labels if str(item).strip()]
            if isinstance(dominant_release_labels, list)
            else []
        )
        rows.append(
            {
                "artist_name": str(row.get("artist_name", "")),
                "scene_name": str(row.get("scene_name", "")),
                "release_whitespace_score": float(row.get("release_whitespace_score", 0.0) or 0.0),
                "days_since_latest_release": row.get("days_since_latest_release"),
                "dominant_release_labels": dominant_release_labels,
            }
        )
    return rows


@dataclass(frozen=True)
class CreatorBriefHandlerDeps:
    load_history_if_needed: Callable[[argparse.Namespace, logging.Logger], pd.DataFrame]
    recent_history: Callable[[pd.DataFrame, int], pd.DataFrame]
    resolve_artists: Callable[..., list[str]]


CreatorBriefHandler = Callable[[argparse.Namespace, Any, logging.Logger], int]


def build_creator_label_intelligence_handler(deps: CreatorBriefHandlerDeps) -> CreatorBriefHandler:
    def _handle_creator_label_intelligence(
        args: argparse.Namespace,
        client: Any,
        logger: logging.Logger,
    ) -> int:
        output_dir = Path(args.output_dir).expanduser().resolve()
        recent_history = deps.recent_history(deps.load_history_if_needed(args, logger), int(args.lookback_days))
        artists = deps.resolve_artists(
            args,
            logger,
            history_top_n=args.top_n,
            history_lookback_days=args.lookback_days,
        )
        if not artists:
            raise RuntimeError("No seed artists were found for creator-label-intelligence.")

        artifact_path = _resolve_multimodal_artifact(
            output_dir=output_dir,
            run_dir=getattr(args, "run_dir", None),
            multimodal_artifact=getattr(args, "multimodal_artifact", None),
        )
        multimodal_space = joblib.load(artifact_path) if artifact_path is not None else None
        engineered_history, resolved_space, space_info = prepare_creator_intelligence_inputs(
            history_df=recent_history,
            logger=logger,
            multimodal_space=multimodal_space,
            max_artists=max(8, int(args.max_artists)),
        )
        if artifact_path is not None:
            space_info["path"] = str(artifact_path)

        intelligence_payload = build_creator_label_intelligence(
            history_df=engineered_history,
            space=resolved_space,
            seed_artists=artists,
            client=client,
            market=str(args.spotify_market).upper(),
            related_limit=max(1, int(args.related_limit)),
            neighbor_k=max(1, int(args.neighbor_k)),
            release_limit=max(1, int(args.release_limit)),
            scene_count=int(args.scene_count) if args.scene_count else None,
        )

        payload = {
            "command": "creator-label-intelligence",
            "market": str(args.spotify_market).upper(),
            "catalog_mode": str(getattr(client, "mode", "spotify_public_api")),
            "lookback_days": int(args.lookback_days),
            "related_limit": int(args.related_limit),
            "neighbor_k": int(args.neighbor_k),
            "release_limit": int(args.release_limit),
            "scene_count": int(args.scene_count) if args.scene_count else None,
            "multimodal_source": space_info,
            **intelligence_payload,
        }
        payload["executive_summary"] = _creator_brief_executive_summary(payload)
        payload["comparison_views"] = {
            "ranking_comparison": _creator_brief_ranking_comparison(payload),
            "scene_comparison": _creator_brief_scene_comparison(payload),
            "seed_comparison": _creator_brief_seed_comparison(payload),
            "scene_seed_comparison": _creator_brief_scene_seed_comparison(payload),
            "opportunity_lane_comparison": _creator_brief_opportunity_lane_comparison(payload),
        }
        payload["brief_views"] = {
            "priority_shortlist": _creator_brief_priority_shortlist(payload),
            "migration_watch": _creator_brief_migration_watch(payload),
            "release_watch": _creator_brief_release_watch(payload),
            "scene_strategy_watch": _creator_brief_scene_strategy_watch(payload),
        }
        payload["report_family"] = {
            "mode": "standalone_report_family_nested_under_public_insights",
            "primary_report": f"{payload['command']}_brief",
            "comparison_views": [
                "ranking_comparison",
                "scene_comparison",
                "seed_comparison",
                "scene_seed_comparison",
                "opportunity_lane_comparison",
            ],
            "brief_views": [
                "priority_shortlist",
                "migration_watch",
                "release_watch",
                "scene_strategy_watch",
            ],
            "reading_order": list(_CREATOR_REPORT_FAMILY_READING_ORDER),
        }

        markdown_lines = [
            "# Creator And Label Intelligence Brief",
            "",
            f"- Market: `{payload['market']}`",
            f"- Catalog mode: `{payload['catalog_mode']}`",
            f"- Packaging: `{payload['report_family']['mode']}`",
            f"- Seed artists: `{len(artists)}`",
            f"- Multimodal source: `{space_info['mode']}`",
            f"- Nodes: `{payload['graph_summary']['node_count']}`",
            f"- Scenes: `{payload['graph_summary']['scene_count']}`",
            f"- Opportunities: `{payload['graph_summary']['opportunity_count']}`",
            "",
            "## Executive Summary",
            "",
        ]
        for line in payload["executive_summary"]:
            markdown_lines.append(f"- {line}")
        markdown_lines.extend(["", "## Ranking Rubric", ""])
        for label, weight in payload["ranking_rubric"]["weights"].items():
            markdown_lines.append(f"- {label}: weight `{float(weight):.2f}`")
        markdown_lines.extend(["", "## Immediate Opportunity Shortlist", ""])
        for row in payload["brief_views"]["priority_shortlist"]:
            seed_text = ", ".join(row["connected_seed_artists"][:2]) if row["connected_seed_artists"] else "current seeds"
            label_text = ", ".join(row["dominant_release_labels"][:2]) if row["dominant_release_labels"] else "n/a"
            markdown_lines.append(
                f"- #{row['opportunity_rank']} {row['artist_name']} ({row['opportunity_band']}): score `{row['opportunity_score']:.3f}`, "
                f"scene `{row['scene_name']}`, driver `{row['primary_driver']}`, seed bridges `{seed_text}`, labels `{label_text}`. {row['why_now']}"
            )
        markdown_lines.extend(["", "## Scene Comparison", ""])
        for row in payload["comparison_views"]["scene_comparison"][:10]:
            seed_text = ", ".join(row["top_seed_artists"][:3]) if row["top_seed_artists"] else "n/a"
            markdown_lines.append(
                f"- {row['scene_name']}: play_share `{row['scene_local_play_share']:.3f}`, artists `{row['artist_count']}`, "
                f"seeds `{row['seed_count']}`, opportunities `{row['opportunity_count']}`, priority `{row['priority_now_count']}`, "
                f"watchlist `{row['watchlist_count']}`, avg score `{row['avg_opportunity_score']:.3f}`, "
                f"release pressure `{row['scene_release_pressure']:.3f}`, label concentration `{row['scene_label_concentration']:.3f}`, "
                f"seed coverage `{seed_text}`, top opportunity `{row['top_opportunity_artist'] or 'n/a'}`"
            )
        markdown_lines.extend(["", "## Opportunity Lanes", ""])
        for row in payload["comparison_views"]["opportunity_lane_comparison"][:12]:
            markdown_lines.append(
                f"- {row['scene_name']} / {row['primary_driver']}: posture `{row['lane_posture']}`, "
                f"avg score `{row['avg_opportunity_score']:.3f}`, opportunities `{row['opportunity_count']}`, "
                f"priority `{row['priority_now_count']}`, seed bridges `{row['seed_bridge_count']}`, "
                f"inbound migration `{row['incoming_migration_share']:.3f}`, whitespace `{row['release_whitespace_score']:.3f}`"
            )
        markdown_lines.extend(["", "## Seed Comparison", ""])
        for row in payload["comparison_views"]["seed_comparison"][:10]:
            markdown_lines.append(
                f"- {row['seed_artist']}: best bridge `{row['top_adjacent_artist']}` hybrid `{row['top_hybrid_score']:.3f}`, "
                f"transition `{row['top_transition_share']:.3f}`, target opportunity `{row['top_target_opportunity_score']:.3f}`, "
                f"top scene `{row['top_scene_name'] or 'n/a'}` avg `{row['top_scene_avg_opportunity_score']:.3f}`, "
                f"scene coverage `{row['scene_coverage_count']}`"
            )
        markdown_lines.extend(["", "## Scene Vs Seed Comparison", ""])
        for row in payload["comparison_views"]["scene_seed_comparison"][:12]:
            markdown_lines.append(
                f"- {row['scene_name']} x {row['seed_artist']}: opportunities `{row['opportunity_count']}`, "
                f"avg score `{row['avg_opportunity_score']:.3f}`, bridges `{row['bridge_artist_count']}`, "
                f"top opportunity `{row['top_opportunity_artist']}` via `{row['top_driver']}`"
            )
        markdown_lines.extend(["", "## Audience Migration", ""])
        for row in payload["brief_views"]["migration_watch"]:
            markdown_lines.append(
                f"- {row['source_artist']} -> {row['target_artist']}: transition share `{row['source_out_share']:.3f}`, "
                f"count `{row['transition_count']}`, target intake `{row['target_in_share']:.3f}`"
            )
        markdown_lines.extend(["", "## Release Whitespace", ""])
        if payload["brief_views"]["release_watch"]:
            for row in payload["brief_views"]["release_watch"]:
                markdown_lines.append(
                    f"- {row['artist_name']}: whitespace `{row['release_whitespace_score']:.3f}`, scene `{row['scene_name']}`, "
                    f"days since latest `{row['days_since_latest_release']}`, labels `{', '.join(row['dominant_release_labels'][:3]) or 'n/a'}`"
                )
        else:
            markdown_lines.append("- Public-catalog release metadata was unavailable for this run, so whitespace stayed empty.")
        markdown_lines.extend(["", "## Scene Strategy Watch", ""])
        for row in payload["brief_views"]["scene_strategy_watch"][:10]:
            markdown_lines.append(
                f"- {row['scene_name']}: posture `{row['strategy_posture']}`, release pressure `{row['scene_release_pressure']:.3f}`, "
                f"label concentration `{row['scene_label_concentration']:.3f}`, inbound migration `{row['incoming_migration_share']:.3f}`, "
                f"outbound migration `{row['outgoing_migration_share']:.3f}`, whitespace anchor `{row['release_whitespace_anchor_artist'] or 'n/a'}`. "
                f"{row['action_note']}"
            )
        markdown_lines.extend(["", "## Opportunity Scoreboard", ""])
        for row in payload["opportunities"][:10]:
            markdown_lines.append(
                f"- #{row['opportunity_rank']} {row['artist_name']}: score `{row['opportunity_score']}`, "
                f"band `{row['opportunity_band']}`, reasons `{'; '.join(row['rationale'])}`, "
                f"components `adj={row['adjacency_component']:.3f}, mig={row['migration_component']:.3f}, "
                f"rel={(float(row['freshness_component']) + float(row['whitespace_component'])):.3f}, "
                f"scene={(float(row['scene_momentum_component']) + float(row['label_concentration_component'])):.3f}, "
                f"gap={(float(row['local_gap_component']) + float(row['popularity_tail_component'])):.3f}`, "
                f"why now `{row['why_now']}`"
            )
        markdown_lines.extend(["", "## Supporting Graph", "", "### Artist Adjacency", ""])
        for row in payload["artist_adjacency"][:10]:
            markdown_lines.append(
                f"- {row['source_artist']} -> {row['target_artist']}: hybrid `{row['hybrid_score']}`, "
                f"similarity `{row['embedding_similarity']}`, transition `{row['transition_share']}`"
            )
        markdown_lines.extend(["", "### Scene Map", ""])
        for row in payload["scenes"][:10]:
            markdown_lines.append(
                f"- {row['scene_name']}: artists `{row['artist_count']}`, seeds `{row['seed_count']}`, "
                f"genres `{', '.join(row['dominant_genres'][:3]) or 'n/a'}`, "
                f"labels `{', '.join(row['dominant_labels'][:3]) or 'n/a'}`"
            )

        stem = f"creator_label_intelligence_{_slugify('-'.join(artists[:3]) or 'history')}"
        json_path, md_path = _write_report(
            output_dir=output_dir,
            category="creator_label_intelligence",
            stem=stem,
            payload=payload,
            markdown_lines=markdown_lines,
        )
        report_dir = json_path.parent
        view_markdown_paths = {
            "ranking_comparison": _write_markdown_lines(
                report_dir / f"{stem}_ranking_view.md",
                [
                    "# Creator Opportunity Ranking View",
                    "",
                    f"- Seeds: `{', '.join(artists)}`",
                    f"- Market: `{payload['market']}`",
                    "",
                    "## Ranking Table",
                    "",
                    *[
                        (
                            f"- #{row['opportunity_rank']} {row['artist_name']} ({row['opportunity_band']}): "
                            f"score `{row['opportunity_score']:.3f}`, scene `{row['scene_name']}`, "
                            f"seeds `{', '.join(row['seed_bridges']) or 'n/a'}`, driver `{row['primary_driver']}`, "
                            f"adj `{row['adjacency_component']:.3f}`, mig `{row['migration_component']:.3f}`, "
                            f"release `{row['release_component']:.3f}`, scene `{row['scene_component']:.3f}`, "
                            f"gap `{row['gap_component']:.3f}`"
                        )
                        for row in payload["comparison_views"]["ranking_comparison"]
                    ],
                ],
            ),
            "scene_comparison": _write_markdown_lines(
                report_dir / f"{stem}_scene_view.md",
                [
                    "# Creator Scene Comparison View",
                    "",
                    f"- Seeds: `{', '.join(artists)}`",
                    "",
                    "## Scene Table",
                    "",
                    *[
                        (
                            f"- {row['scene_name']}: play_share `{row['scene_local_play_share']:.3f}`, avg opportunity `{row['avg_opportunity_score']:.3f}`, "
                            f"priority `{row['priority_now_count']}`, watchlist `{row['watchlist_count']}`, "
                            f"release pressure `{row['scene_release_pressure']:.3f}`, label concentration `{row['scene_label_concentration']:.3f}`, "
                            f"top migration `{row['top_migration_route'] or 'n/a'}`"
                        )
                        for row in payload["comparison_views"]["scene_comparison"]
                    ],
                ],
            ),
            "seed_comparison": _write_markdown_lines(
                report_dir / f"{stem}_seed_view.md",
                [
                    "# Creator Seed Comparison View",
                    "",
                    f"- Seeds: `{', '.join(artists)}`",
                    "",
                    "## Seed Table",
                    "",
                    *[
                        (
                            f"- {row['seed_artist']}: bridge `{row['top_adjacent_artist']}` hybrid `{row['top_hybrid_score']:.3f}`, "
                            f"top scene `{row['top_scene_name'] or 'n/a'}` avg `{row['top_scene_avg_opportunity_score']:.3f}`, "
                            f"top opportunity `{row['top_opportunity_artist'] or 'n/a'}`, scene coverage `{row['scene_coverage_count']}`"
                        )
                        for row in payload["comparison_views"]["seed_comparison"]
                    ],
                ],
            ),
            "scene_seed_comparison": _write_markdown_lines(
                report_dir / f"{stem}_scene_seed_view.md",
                [
                    "# Creator Scene Vs Seed View",
                    "",
                    f"- Seeds: `{', '.join(artists)}`",
                    "",
                    "## Cross View",
                    "",
                    *[
                        (
                            f"- {row['scene_name']} x {row['seed_artist']}: avg `{row['avg_opportunity_score']:.3f}`, "
                            f"opportunities `{row['opportunity_count']}`, bridges `{row['bridge_artist_count']}`, "
                            f"top `{row['top_opportunity_artist']}` via `{row['top_driver']}`"
                        )
                        for row in payload["comparison_views"]["scene_seed_comparison"]
                    ],
                ],
            ),
            "opportunity_lane_comparison": _write_markdown_lines(
                report_dir / f"{stem}_opportunity_lane_view.md",
                [
                    "# Creator Opportunity Lane View",
                    "",
                    f"- Seeds: `{', '.join(artists)}`",
                    "",
                    "## Lane Table",
                    "",
                    *[
                        (
                            f"- {row['scene_name']} / {row['primary_driver']}: posture `{row['lane_posture']}`, "
                            f"avg `{row['avg_opportunity_score']:.3f}`, priority `{row['priority_now_count']}`, "
                            f"bridges `{row['seed_bridge_count']}`, migration `{row['incoming_migration_share']:.3f}`, "
                            f"whitespace `{row['release_whitespace_score']:.3f}`"
                        )
                        for row in payload["comparison_views"]["opportunity_lane_comparison"]
                    ],
                ],
            ),
        }
        family_manifest = {
            "primary_report": str(md_path),
            "primary_report_json": str(json_path),
            "comparison_view_markdown": {label: str(path) for label, path in view_markdown_paths.items()},
            "comparison_view_csv": {},
            "brief_view_markdown": {},
            "brief_view_csv": {},
            "reading_order": list(payload["report_family"]["reading_order"]),
        }
        brief_markdown_paths = {
            "scene_strategy_watch": _write_markdown_lines(
                report_dir / f"{stem}_scene_strategy_watch.md",
                [
                    "# Creator Scene Strategy Watch",
                    "",
                    f"- Seeds: `{', '.join(artists)}`",
                    "",
                    "## Strategy Table",
                    "",
                    *[
                        (
                            f"- {row['scene_name']}: posture `{row['strategy_posture']}`, release `{row['scene_release_pressure']:.3f}`, "
                            f"label concentration `{row['scene_label_concentration']:.3f}`, incoming `{row['incoming_migration_share']:.3f}`, "
                            f"outgoing `{row['outgoing_migration_share']:.3f}`. {row['action_note']}"
                        )
                        for row in payload["brief_views"]["scene_strategy_watch"]
                    ],
                ],
            ),
        }
        csv_paths = {
            "artist_adjacency": _write_csv_rows(report_dir / f"{stem}_artist_adjacency.csv", payload["artist_adjacency"]),
            "nodes": _write_csv_rows(report_dir / f"{stem}_nodes.csv", payload["nodes"]),
            "edges": _write_csv_rows(report_dir / f"{stem}_edges.csv", payload["edges"]),
            "scenes": _write_csv_rows(report_dir / f"{stem}_scenes.csv", payload["scenes"]),
            "release_whitespace": _write_csv_rows(
                report_dir / f"{stem}_release_whitespace.csv",
                payload["release_whitespace"],
            ),
            "fan_migration": _write_csv_rows(report_dir / f"{stem}_fan_migration.csv", payload["fan_migration"]),
            "opportunities": _write_csv_rows(report_dir / f"{stem}_opportunities.csv", payload["opportunities"]),
            "ranking_comparison": _write_csv_rows(
                report_dir / f"{stem}_ranking_comparison.csv",
                payload["comparison_views"]["ranking_comparison"],
            ),
            "scene_comparison": _write_csv_rows(
                report_dir / f"{stem}_scene_comparison.csv",
                payload["comparison_views"]["scene_comparison"],
            ),
            "seed_comparison": _write_csv_rows(
                report_dir / f"{stem}_seed_comparison.csv",
                payload["comparison_views"]["seed_comparison"],
            ),
            "scene_seed_comparison": _write_csv_rows(
                report_dir / f"{stem}_scene_seed_comparison.csv",
                payload["comparison_views"]["scene_seed_comparison"],
            ),
            "opportunity_lane_comparison": _write_csv_rows(
                report_dir / f"{stem}_opportunity_lane_comparison.csv",
                payload["comparison_views"]["opportunity_lane_comparison"],
            ),
            "priority_shortlist": _write_csv_rows(
                report_dir / f"{stem}_priority_shortlist.csv",
                payload["brief_views"]["priority_shortlist"],
            ),
            "migration_watch": _write_csv_rows(
                report_dir / f"{stem}_migration_watch.csv",
                payload["brief_views"]["migration_watch"],
            ),
            "release_watch": _write_csv_rows(
                report_dir / f"{stem}_release_watch.csv",
                payload["brief_views"]["release_watch"],
            ),
            "scene_strategy_watch": _write_csv_rows(
                report_dir / f"{stem}_scene_strategy_watch.csv",
                payload["brief_views"]["scene_strategy_watch"],
            ),
        }
        family_manifest["comparison_view_csv"] = {
            label: str(path)
            for label, path in csv_paths.items()
            if label.endswith("comparison") and path is not None
        }
        family_manifest["brief_view_markdown"] = {label: str(path) for label, path in brief_markdown_paths.items()}
        family_manifest["brief_view_csv"] = {
            label: str(path)
            for label, path in csv_paths.items()
            if label in {"priority_shortlist", "migration_watch", "release_watch", "scene_strategy_watch"} and path is not None
        }
        packaged_at = datetime.now(timezone.utc).isoformat()
        family_manifest = normalize_creator_report_family_manifest(
            family_manifest,
            report_dir=report_dir,
            stem=stem,
            refreshed_at=packaged_at,
            refresh_source="creator_label_intelligence_write",
        )
        packaging_metadata = (
            family_manifest.get("packaging_metadata", {})
            if isinstance(family_manifest.get("packaging_metadata"), dict)
            else {}
        )
        anchor_views = packaging_metadata.get("anchor_views", {}) if isinstance(packaging_metadata, dict) else {}
        family_manifest_md_path = _write_markdown_lines(
            report_dir / f"{stem}_report_family.md",
            [
                "# Creator Report Family",
                "",
                f"- Primary report: `{md_path.name}`",
                f"- Primary report JSON: `{json_path.name}`",
                f"- Packaging mode: `{payload['report_family']['mode']}`",
                f"- Packaging refreshed: `{packaging_metadata.get('normalized_at', '')}`",
                "",
                "## Reading Order",
                "",
                *[f"- `{item}`" for item in family_manifest["reading_order"]],
                "",
                "## Comparison Views",
                "",
                *[
                    f"- `{label}`: markdown `{Path(path).name}`"
                    for label, path in family_manifest["comparison_view_markdown"].items()
                ],
                "",
                "## Brief Views",
                "",
                *[
                    f"- `{label}`: markdown `{Path(path).name}`"
                    for label, path in family_manifest["brief_view_markdown"].items()
                ],
                "",
                "## Data Tables",
                "",
                *[
                    f"- `{label}`: `{Path(str(path)).name}`"
                    for label, path in family_manifest["comparison_view_csv"].items()
                ],
                *[
                    f"- `{label}`: `{Path(str(path)).name}`"
                    for label, path in family_manifest["brief_view_csv"].items()
                ],
                "",
                "## Refresh Anchors",
                "",
                *[
                    (
                        f"- `{anchor_name}`: key `{row.get('view_key', '')}`, ready `{row.get('ready', False)}`, "
                        f"markdown `{Path(str(row.get('markdown_path', ''))).name if str(row.get('markdown_path', '')).strip() else 'missing'}`, "
                        f"csv `{Path(str(row.get('csv_path', ''))).name if str(row.get('csv_path', '')).strip() else 'missing'}`"
                    )
                    for anchor_name, row in anchor_views.items()
                    if isinstance(row, dict)
                ],
            ],
        )
        family_manifest["artifact_index_markdown"] = str(family_manifest_md_path)
        family_manifest_path = write_json(report_dir / f"{stem}_report_family.json", family_manifest)
        print(f"creator_label_intelligence_json={json_path}")
        print(f"creator_label_intelligence_md={md_path}")
        for label, path in csv_paths.items():
            if path is not None:
                print(f"creator_label_intelligence_{label}_csv={path}")
        for label, path in view_markdown_paths.items():
            print(f"creator_label_intelligence_{label}_md={path}")
        for label, path in brief_markdown_paths.items():
            print(f"creator_label_intelligence_{label}_md={path}")
        print(f"creator_label_intelligence_report_family_json={family_manifest_path}")
        print(f"creator_label_intelligence_report_family_md={family_manifest_md_path}")
        print(f"creator_label_intelligence_nodes={payload['graph_summary']['node_count']}")
        print(f"creator_label_intelligence_scenes={payload['graph_summary']['scene_count']}")
        print(f"creator_label_intelligence_opportunities={payload['graph_summary']['opportunity_count']}")
        return 0

    return _handle_creator_label_intelligence


__all__ = [
    "CreatorBriefHandlerDeps",
    "_creator_brief_executive_summary",
    "_creator_brief_migration_watch",
    "_creator_brief_opportunity_lane_comparison",
    "_creator_brief_priority_shortlist",
    "_creator_brief_ranking_comparison",
    "_creator_brief_release_watch",
    "_creator_brief_scene_comparison",
    "_creator_brief_scene_strategy_watch",
    "_creator_brief_scene_seed_comparison",
    "_creator_brief_seed_comparison",
    "build_creator_label_intelligence_handler",
    "normalize_creator_report_family_manifest",
]
