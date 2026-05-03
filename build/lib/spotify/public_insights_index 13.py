from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_artifacts import safe_read_json, write_json, write_markdown


_SKIP_CATEGORIES = {"playlist_state", "release_state", "summary"}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_markdown_path(json_path: Path) -> Path | None:
    md_path = json_path.with_suffix(".md")
    return md_path if md_path.exists() else None


def _report_mtime_iso(path: Path) -> str:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        timestamp = 0.0
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _recommendation_count(payload: dict[str, Any]) -> int:
    recommendations = payload.get("recommendations", [])
    return len(recommendations) if isinstance(recommendations, list) else 0


def _warning_count(payload: dict[str, Any]) -> int:
    warnings = payload.get("warnings", [])
    return len(warnings) if isinstance(warnings, list) else 0


def _brief_summary(payload: dict[str, Any]) -> dict[str, Any]:
    command = str(payload.get("command", "")).strip()
    if command == "personal-release-radar":
        releases = payload.get("priority_releases", [])
        releases = releases if isinstance(releases, list) else []
        top_release = releases[0] if releases and isinstance(releases[0], dict) else {}
        return {
            "priority_releases": len(releases),
            "new_releases": len(payload.get("new_releases_since_last_run", []) or []),
            "top_release": str(top_release.get("album_name", "")).strip(),
            "top_artist": str(top_release.get("artist_name", "")).strip(),
        }
    if command == "artist-catalog-completeness":
        artists = [row for row in payload.get("artists", []) if isinstance(row, dict)]
        coverages = [_float_or_none(row.get("coverage_ratio")) for row in artists]
        coverages = [value for value in coverages if value is not None]
        return {
            "artists_audited": len(artists),
            "avg_coverage_ratio": sum(coverages) / len(coverages) if coverages else None,
            "lowest_coverage_artist": min(
                artists,
                key=lambda row: _float_or_none(row.get("coverage_ratio")) if _float_or_none(row.get("coverage_ratio")) is not None else 1.0,
                default={},
            ).get("artist_name", ""),
        }
    if command == "playlist-intelligence":
        playlist = payload.get("playlist", {}) if isinstance(payload.get("playlist"), dict) else {}
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        return {
            "playlist": str(playlist.get("name", "")).strip(),
            "tracks_loaded": int(summary.get("tracks_loaded", 0) or 0),
            "local_overlap_ratio": _float_or_none(summary.get("local_overlap_ratio")),
            "duplicate_track_groups": int(summary.get("duplicate_track_groups", 0) or 0),
        }
    if command == "album-profile":
        album = payload.get("album", {}) if isinstance(payload.get("album"), dict) else {}
        summary = payload.get("album_summary", {}) if isinstance(payload.get("album_summary"), dict) else {}
        return {
            "album": str(album.get("name", "")).strip(),
            "artist_names": album.get("artists", []),
            "coverage_ratio": _float_or_none(summary.get("coverage_ratio")),
        }
    return {}


def collect_public_insights_reports(
    output_dir: Path,
    *,
    category: str | None = None,
    max_reports: int = 25,
) -> list[dict[str, Any]]:
    root = output_dir.expanduser().resolve() / "analysis" / "public_spotify"
    if not root.exists():
        return []

    category_filter = str(category or "").strip()
    rows: list[dict[str, Any]] = []
    for json_path in sorted(root.glob("*/*.json")):
        category_name = json_path.parent.name
        if category_name in _SKIP_CATEGORIES:
            continue
        if category_filter and category_name != category_filter:
            continue
        payload = safe_read_json(json_path, default=None)
        if not isinstance(payload, dict):
            continue
        md_path = _report_markdown_path(json_path)
        rows.append(
            {
                "category": category_name,
                "command": str(payload.get("command", category_name)).strip() or category_name,
                "json_path": str(json_path.resolve()),
                "markdown_path": str(md_path.resolve()) if md_path is not None else "",
                "updated_at": _report_mtime_iso(json_path),
                "warning_count": _warning_count(payload),
                "recommendation_count": _recommendation_count(payload),
                "summary": _brief_summary(payload),
            }
        )

    rows.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
    return rows[: max(1, int(max_reports))]


def build_public_insights_index(
    output_dir: Path,
    *,
    category: str | None = None,
    max_reports: int = 25,
) -> tuple[dict[str, Any], list[str]]:
    reports = collect_public_insights_reports(output_dir, category=category, max_reports=max_reports)
    open_next = [
        {
            "category": row["category"],
            "command": row["command"],
            "markdown_path": row["markdown_path"],
            "reason": _open_reason(row),
        }
        for row in reports[:5]
    ]
    payload = {
        "command": "summary",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "output_dir": str(output_dir.expanduser().resolve()),
        "category_filter": str(category or "").strip() or None,
        "reports_indexed": len(reports),
        "open_next": open_next,
        "reports": reports,
        "policy_note": "Public Spotify metadata reports are display/link-out artifacts only and are not training features.",
    }
    lines = [
        "# Spotify Public Insights Summary",
        "",
        f"- Reports indexed: `{len(reports)}`",
        f"- Category filter: `{payload['category_filter'] or 'all'}`",
        "",
        "## Open Next",
        "",
    ]
    if not open_next:
        lines.append("- No public-insights reports found yet.")
    for row in open_next:
        target = row.get("markdown_path") or row.get("category")
        lines.append(f"- `{row['command']}`: {row['reason']} ({target})")
    lines.extend(["", "## Recent Reports", ""])
    for row in reports:
        summary = row.get("summary", {})
        summary_text = ", ".join(f"{key}={value}" for key, value in summary.items() if value not in (None, "", []))
        suffix = f" | {summary_text}" if summary_text else ""
        lines.append(
            f"- `{row['updated_at']}` | `{row['category']}` | warnings=`{row['warning_count']}` "
            f"recommendations=`{row['recommendation_count']}`{suffix}"
        )
    return payload, lines


def _open_reason(row: dict[str, Any]) -> str:
    warnings = int(row.get("warning_count", 0) or 0)
    recommendations = int(row.get("recommendation_count", 0) or 0)
    summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
    if warnings:
        return f"has {warnings} API warning(s) to review"
    if recommendations:
        return f"has {recommendations} recommendation(s)"
    if row.get("command") == "personal-release-radar":
        return f"found {int(summary.get('priority_releases', 0) or 0)} priority release(s)"
    return "latest public metadata report"


def write_public_insights_index(
    *,
    output_dir: Path,
    destination_dir: Path,
    category: str | None = None,
    max_reports: int = 25,
) -> tuple[Path, Path]:
    payload, lines = build_public_insights_index(output_dir, category=category, max_reports=max_reports)
    destination_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(destination_dir / "public_insights_index.json", payload)
    md_path = write_markdown(destination_dir / "public_insights_index.md", lines)
    return json_path, md_path
