from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import re
from typing import Any

import joblib
import pandas as pd

from .creator_label_intelligence import build_creator_label_intelligence, prepare_creator_intelligence_inputs
from .data import load_streaming_history
from .env import load_local_env
from .public_catalog import SpotifyPublicCatalogClient, SpotifyPublicCatalogError, parse_spotify_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.public_insights",
        description="Policy-safe Spotify public metadata tools for explanation, discovery, and catalog exploration.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated reports.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--include-video", action="store_true", help="Include video history files when needed.")
    parser.add_argument("--spotify-market", type=str, default="US", help="Two-letter market code for Spotify requests.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    explain = subparsers.add_parser("explain-artists", help="Explain your top artists with Spotify public metadata.")
    explain.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to explain.")
    explain.add_argument("--top-n", type=int, default=5, help="Number of history-derived artists to explain.")
    explain.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top artists.")
    explain.add_argument("--related-limit", type=int, default=5, help="Number of related artists to include.")

    releases = subparsers.add_parser("release-tracker", help="Track new releases from your favorite artists.")
    releases.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to track.")
    releases.add_argument("--top-n", type=int, default=10, help="Number of history-derived artists to track.")
    releases.add_argument("--lookback-days", type=int, default=365, help="History window for deriving top artists.")
    releases.add_argument("--since-days", type=int, default=120, help="Only include releases from the last N days.")
    releases.add_argument(
        "--include-groups",
        type=str,
        default="album,single",
        help="Comma-separated Spotify album groups to request.",
    )
    releases.add_argument("--per-artist-limit", type=int, default=10, help="Maximum releases per artist in the report.")

    market = subparsers.add_parser("market-check", help="Check market availability for your top tracks.")
    market.add_argument("--tracks", type=str, default=None, help="Pipe-separated track URLs, URIs, or IDs.")
    market.add_argument("--top-n", type=int, default=20, help="Number of history-derived tracks to inspect.")
    market.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top tracks.")
    market.add_argument(
        "--markets",
        type=str,
        default="US",
        help="Comma-separated market codes to check, for example US,GB,IN.",
    )

    discography = subparsers.add_parser("discography", help="Build a discography timeline for favorite artists.")
    discography.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to inspect.")
    discography.add_argument("--top-n", type=int, default=5, help="Number of history-derived artists to inspect.")
    discography.add_argument("--lookback-days", type=int, default=365, help="History window for deriving top artists.")
    discography.add_argument(
        "--include-groups",
        type=str,
        default="album,single",
        help="Comma-separated Spotify album groups to request.",
    )
    discography.add_argument("--album-limit", type=int, default=20, help="Maximum album rows per artist.")

    playlist = subparsers.add_parser("playlist-view", help="Inspect a public Spotify playlist.")
    playlist.add_argument("--playlist", type=str, required=True, help="Spotify playlist URL, URI, or ID.")
    playlist.add_argument("--item-limit", type=int, default=50, help="Maximum playlist items to load.")

    discovery = subparsers.add_parser("discovery-search", help="Run a Spotify search query for discovery.")
    discovery.add_argument("--query", type=str, required=True, help="Spotify search query.")
    discovery.add_argument(
        "--types",
        type=str,
        default="artist,album,track,playlist",
        help="Comma-separated Spotify item types to search.",
    )
    discovery.add_argument("--limit", type=int, default=10, help="Maximum results per type.")

    linkouts = subparsers.add_parser("catalog-linkouts", help="Build link-out bundles for your top artists and tracks.")
    linkouts.add_argument("--top-artists", type=int, default=10, help="Number of recent artists to include.")
    linkouts.add_argument("--top-tracks", type=int, default=20, help="Number of recent tracks to include.")
    linkouts.add_argument("--lookback-days", type=int, default=180, help="History window for deriving items.")

    graph = subparsers.add_parser("artist-graph", help="Build a related-artist graph from seed artists.")
    graph.add_argument("--artists", type=str, default=None, help="Pipe-separated seed artists.")
    graph.add_argument("--top-n", type=int, default=5, help="Number of history-derived seed artists.")
    graph.add_argument("--lookback-days", type=int, default=180, help="History window for deriving seed artists.")
    graph.add_argument("--related-limit", type=int, default=10, help="Number of related artists per seed.")

    cross_media = subparsers.add_parser(
        "cross-media-taste-graph",
        help="Build a cross-media session-intelligence graph across music, podcasts, shows, and audiobooks.",
    )
    cross_media.add_argument("--lookback-days", type=int, default=180, help="History window for recent listening context.")
    cross_media.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="Gap in minutes that starts a new inferred listening session.",
    )
    cross_media.add_argument("--node-limit", type=int, default=40, help="Maximum graph nodes to emit in the report.")
    cross_media.add_argument("--edge-limit", type=int, default=80, help="Maximum graph edges to emit in the report.")
    cross_media.add_argument(
        "--session-limit",
        type=int,
        default=8,
        help="Maximum mixed-session summaries to include in the report.",
    )
    cross_media.add_argument(
        "--bridge-limit",
        type=int,
        default=6,
        help="Maximum history-derived seed nodes to expand into cross-media catalog bridges.",
    )
    cross_media.add_argument(
        "--recommendation-limit",
        type=int,
        default=3,
        help="Maximum catalog results per seed/type bridge.",
    )

    inbox = subparsers.add_parser("release-inbox", help="Track only newly seen releases since the last run.")
    inbox.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to track.")
    inbox.add_argument("--top-n", type=int, default=10, help="Number of history-derived artists to track.")
    inbox.add_argument("--lookback-days", type=int, default=365, help="History window for deriving seed artists.")
    inbox.add_argument("--since-days", type=int, default=120, help="Only include releases from the last N days.")
    inbox.add_argument("--include-groups", type=str, default="album,single", help="Album groups to include.")
    inbox.add_argument("--per-artist-limit", type=int, default=10, help="Maximum releases per artist in the inbox.")

    diff = subparsers.add_parser("playlist-diff", help="Track changes in a public playlist over time.")
    diff.add_argument("--playlist", type=str, required=True, help="Spotify playlist URL, URI, or ID.")
    diff.add_argument("--item-limit", type=int, default=100, help="Maximum playlist items to load.")

    gap = subparsers.add_parser("market-gap", help="Find market coverage gaps for tracks against all Spotify markets.")
    gap.add_argument("--tracks", type=str, default=None, help="Pipe-separated track URLs, URIs, or IDs.")
    gap.add_argument("--top-n", type=int, default=20, help="Number of history-derived tracks to inspect.")
    gap.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top tracks.")

    archive = subparsers.add_parser("playlist-archive", help="Archive playlist metadata, items, and image URLs.")
    archive.add_argument("--playlist", type=str, required=True, help="Spotify playlist URL, URI, or ID.")
    archive.add_argument("--item-limit", type=int, default=100, help="Maximum playlist items to load.")

    crosswalk = subparsers.add_parser("catalog-crosswalk", help="Build ISRC/UPC/EAN crosswalks for top tracks.")
    crosswalk.add_argument("--tracks", type=str, default=None, help="Pipe-separated track URLs, URIs, or IDs.")
    crosswalk.add_argument("--top-n", type=int, default=20, help="Number of history-derived tracks to include.")
    crosswalk.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top tracks.")

    media = subparsers.add_parser("media-explorer", help="Explore public shows, episodes, and audiobooks.")
    media.add_argument("--query", type=str, default=None, help="Spotify search query for media exploration.")
    media.add_argument("--media-type", choices=("show", "episode", "audiobook"), default="show")
    media.add_argument("--item-id", type=str, default=None, help="Optional direct Spotify URL, URI, or ID.")
    media.add_argument("--limit", type=int, default=10, help="Maximum search results or child items to include.")

    intelligence = subparsers.add_parser(
        "creator-label-intelligence",
        help="Build an A&R intelligence graph using local multimodal artist space plus Spotify public metadata.",
    )
    intelligence.add_argument("--artists", type=str, default=None, help="Pipe-separated seed artists.")
    intelligence.add_argument("--top-n", type=int, default=8, help="Number of history-derived seed artists.")
    intelligence.add_argument("--lookback-days", type=int, default=365, help="History window for deriving the graph.")
    intelligence.add_argument("--related-limit", type=int, default=8, help="Related artists to inspect per seed.")
    intelligence.add_argument("--neighbor-k", type=int, default=5, help="Local multimodal neighbors to add per seed.")
    intelligence.add_argument("--release-limit", type=int, default=8, help="Maximum releases to inspect per artist.")
    intelligence.add_argument("--scene-count", type=int, default=None, help="Optional fixed number of scene clusters.")
    intelligence.add_argument(
        "--max-artists",
        type=int,
        default=250,
        help="Maximum local artists to keep when deriving a multimodal space from recent history.",
    )
    intelligence.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run directory containing analysis/multimodal/multimodal_artist_space.joblib.",
    )
    intelligence.add_argument(
        "--multimodal-artifact",
        type=str,
        default=None,
        help="Optional direct path to a multimodal_artist_space.joblib artifact.",
    )

    return parser


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "report"


def _split_pipe_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _split_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _history_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["ts"] = pd.to_datetime(frame.get("ts"), errors="coerce", utc=True)
    frame = frame[frame["ts"].notna()].copy()
    return frame.sort_values("ts").reset_index(drop=True)


def _recent_history(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    frame = _history_frame(df)
    if frame.empty:
        return frame
    end_ts = frame["ts"].max()
    cutoff_ts = end_ts - pd.Timedelta(days=max(1, int(lookback_days)))
    return frame[frame["ts"] >= cutoff_ts].copy()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def _first_nonempty_value(record: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        if column not in record:
            continue
        text = _clean_text(record.get(column))
        if text:
            return text
    return ""


def _entity_key(prefix: str, *, uri: str, name: str) -> str:
    stable = _clean_text(uri) or _clean_text(name).casefold()
    return f"{prefix}:{stable}"


def _first_image_url(images: Any) -> str | None:
    if not isinstance(images, list):
        return None
    for image in images:
        if not isinstance(image, dict):
            continue
        url = _clean_text(image.get("url"))
        if url:
            return url
    return None


def _cross_media_history_frame(
    df: pd.DataFrame,
    *,
    lookback_days: int,
    session_gap_minutes: int,
) -> pd.DataFrame:
    recent = _recent_history(df, lookback_days)
    if recent.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "media_family",
                "node_type",
                "node_key",
                "node_name",
                "node_uri",
                "item_type",
                "item_key",
                "item_name",
                "item_uri",
                "ms_played",
                "session_id",
                "session_position",
                "time_diff_seconds",
            ]
        )

    rows: list[dict[str, Any]] = []
    for record in recent.to_dict(orient="records"):
        artist_name = _first_nonempty_value(record, ("master_metadata_album_artist_name",))
        track_name = _first_nonempty_value(record, ("master_metadata_track_name",))
        track_uri = _first_nonempty_value(record, ("spotify_track_uri",))

        show_name = _first_nonempty_value(record, ("episode_show_name", "show_name"))
        show_uri = _first_nonempty_value(record, ("spotify_show_uri", "show_uri"))
        episode_name = _first_nonempty_value(record, ("episode_name", "episode_title"))
        episode_uri = _first_nonempty_value(record, ("spotify_episode_uri", "episode_uri"))

        audiobook_name = _first_nonempty_value(record, ("audiobook_title", "audiobook_name"))
        audiobook_uri = _first_nonempty_value(record, ("audiobook_uri", "spotify_audiobook_uri"))
        chapter_name = _first_nonempty_value(
            record,
            ("audiobook_chapter_title", "chapter_name", "chapter_title"),
        )
        chapter_uri = _first_nonempty_value(record, ("audiobook_chapter_uri", "chapter_uri"))

        event: dict[str, Any] | None = None
        if audiobook_name:
            item_name = chapter_name or episode_name or audiobook_name
            item_uri = chapter_uri or episode_uri or audiobook_uri
            item_type = "chapter" if item_name and item_name.casefold() != audiobook_name.casefold() else "audiobook"
            event = {
                "ts": record.get("ts"),
                "media_family": "audiobook",
                "node_type": "audiobook",
                "node_key": _entity_key("audiobook", uri=audiobook_uri, name=audiobook_name),
                "node_name": audiobook_name,
                "node_uri": audiobook_uri,
                "item_type": item_type,
                "item_key": _entity_key(item_type, uri=item_uri, name=item_name or audiobook_name),
                "item_name": item_name or audiobook_name,
                "item_uri": item_uri,
            }
        elif show_name or episode_name or episode_uri:
            canonical_show_name = show_name or episode_name or episode_uri
            item_name = episode_name or canonical_show_name
            item_uri = episode_uri or show_uri
            item_type = "episode" if episode_name else "show"
            event = {
                "ts": record.get("ts"),
                "media_family": "podcast",
                "node_type": "show",
                "node_key": _entity_key("show", uri=show_uri, name=canonical_show_name),
                "node_name": canonical_show_name,
                "node_uri": show_uri,
                "item_type": item_type,
                "item_key": _entity_key(item_type, uri=item_uri, name=item_name or canonical_show_name),
                "item_name": item_name or canonical_show_name,
                "item_uri": item_uri,
            }
        elif artist_name or track_name or track_uri:
            canonical_artist_name = artist_name or track_name or track_uri
            item_name = track_name or canonical_artist_name
            item_uri = track_uri
            item_type = "track" if track_name else "artist"
            event = {
                "ts": record.get("ts"),
                "media_family": "music",
                "node_type": "artist",
                "node_key": _entity_key("artist", uri="", name=canonical_artist_name),
                "node_name": canonical_artist_name,
                "node_uri": "",
                "item_type": item_type,
                "item_key": _entity_key(item_type, uri=item_uri, name=item_name or canonical_artist_name),
                "item_name": item_name or canonical_artist_name,
                "item_uri": item_uri,
            }

        if event is None:
            continue
        ms_played = pd.to_numeric(record.get("ms_played"), errors="coerce")
        event["ms_played"] = float(ms_played) if pd.notna(ms_played) else 0.0
        rows.append(event)

    if not rows:
        return pd.DataFrame(
            columns=[
                "ts",
                "media_family",
                "node_type",
                "node_key",
                "node_name",
                "node_uri",
                "item_type",
                "item_key",
                "item_name",
                "item_uri",
                "ms_played",
                "session_id",
                "session_position",
                "time_diff_seconds",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    frame = frame[frame["ts"].notna()].sort_values("ts").reset_index(drop=True)
    if frame.empty:
        return frame

    threshold_seconds = max(1, int(session_gap_minutes)) * 60
    frame["time_diff_seconds"] = frame["ts"].diff().dt.total_seconds().fillna(0.0).astype("float32")
    frame["session_id"] = (frame["time_diff_seconds"] > threshold_seconds).cumsum().astype("int32")
    frame["session_position"] = frame.groupby("session_id", sort=False).cumcount().astype("int32")
    return frame


def _node_top_items(group: pd.DataFrame, *, limit: int = 3) -> list[dict[str, Any]]:
    if group.empty:
        return []
    item_rows = (
        group.groupby(["item_key", "item_name", "item_type"], dropna=False)
        .agg(
            play_count=("item_key", "size"),
            total_ms_played=("ms_played", "sum"),
            last_seen=("ts", "max"),
        )
        .reset_index()
        .sort_values(["play_count", "total_ms_played", "item_name"], ascending=[False, False, True])
    )
    rows: list[dict[str, Any]] = []
    for _, row in item_rows.head(max(1, int(limit))).iterrows():
        rows.append(
            {
                "item_key": str(row.get("item_key", "")).strip(),
                "item_name": str(row.get("item_name", "")).strip(),
                "item_type": str(row.get("item_type", "")).strip(),
                "play_count": int(row.get("play_count", 0) or 0),
                "total_ms_played": int(float(row.get("total_ms_played", 0.0) or 0.0)),
                "last_seen": pd.Timestamp(row.get("last_seen")).isoformat() if pd.notna(row.get("last_seen")) else None,
            }
        )
    return rows


def _top_nodes_by_media(nodes: list[dict[str, Any]], *, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for media_family in ("music", "podcast", "audiobook"):
        rows = [row for row in nodes if str(row.get("media_family", "")) == media_family]
        rows.sort(
            key=lambda row: (
                int(row.get("play_count", 0) or 0),
                int(row.get("session_count", 0) or 0),
                str(row.get("name", "")),
            ),
            reverse=True,
        )
        grouped[media_family] = rows[: max(1, int(limit))]
    return grouped


def _seed_nodes_for_bridges(nodes: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if not nodes:
        return []
    grouped = _top_nodes_by_media(nodes, limit=max(1, min(2, int(limit))))
    seeds: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for media_family in ("music", "podcast", "audiobook"):
        for row in grouped.get(media_family, []):
            key = str(row.get("node_key", "")).strip()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            seeds.append(row)
            if len(seeds) >= max(1, int(limit)):
                return seeds
    remaining = sorted(
        nodes,
        key=lambda row: (
            int(row.get("play_count", 0) or 0),
            int(row.get("session_count", 0) or 0),
            str(row.get("name", "")),
        ),
        reverse=True,
    )
    for row in remaining:
        key = str(row.get("node_key", "")).strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        seeds.append(row)
        if len(seeds) >= max(1, int(limit)):
            break
    return seeds


def _cross_media_bridge_types(node_type: str) -> tuple[str, ...]:
    if node_type == "artist":
        return ("show", "episode", "audiobook")
    if node_type == "show":
        return ("artist", "audiobook")
    if node_type == "audiobook":
        return ("artist", "show", "episode")
    return ("artist", "show", "episode", "audiobook")


def _format_catalog_result(item_type: str, row: dict[str, Any]) -> dict[str, Any]:
    if item_type == "artist":
        return {
            "item_type": item_type,
            "name": _clean_text(row.get("name")),
            "spotify_id": _clean_text(row.get("id")),
            "popularity": row.get("popularity"),
            "genres": [
                _clean_text(value)
                for value in row.get("genres", [])
                if _clean_text(value)
            ],
            "spotify_url": _clean_text(row.get("external_urls", {}).get("spotify")),
            "image_url": _first_image_url(row.get("images")),
        }
    if item_type == "show":
        return {
            "item_type": item_type,
            "name": _clean_text(row.get("name")),
            "spotify_id": _clean_text(row.get("id")),
            "publisher": _clean_text(row.get("publisher")),
            "total_episodes": int(row.get("total_episodes", 0) or 0),
            "spotify_url": _clean_text(row.get("external_urls", {}).get("spotify")),
            "image_url": _first_image_url(row.get("images")),
        }
    if item_type == "episode":
        return {
            "item_type": item_type,
            "name": _clean_text(row.get("name")),
            "spotify_id": _clean_text(row.get("id")),
            "show_name": _clean_text(row.get("show", {}).get("name")),
            "release_date": _clean_text(row.get("release_date")),
            "duration_ms": int(row.get("duration_ms", 0) or 0),
            "spotify_url": _clean_text(row.get("external_urls", {}).get("spotify")),
        }
    return {
        "item_type": item_type,
        "name": _clean_text(row.get("name")),
        "spotify_id": _clean_text(row.get("id")),
        "author_names": [
            _clean_text(author.get("name"))
            for author in row.get("authors", [])
            if isinstance(author, dict) and _clean_text(author.get("name"))
        ],
        "narrator_names": [
            _clean_text(narrator.get("name"))
            for narrator in row.get("narrators", [])
            if isinstance(narrator, dict) and _clean_text(narrator.get("name"))
        ],
        "total_chapters": int(row.get("total_chapters", 0) or 0),
        "spotify_url": _clean_text(row.get("external_urls", {}).get("spotify")),
        "image_url": _first_image_url(row.get("images")),
    }


def _cross_media_graph_payload(
    frame: pd.DataFrame,
    *,
    node_limit: int,
    edge_limit: int,
    session_limit: int,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "summary": {
                "events_analyzed": 0,
                "sessions_analyzed": 0,
                "unique_nodes": 0,
                "visible_nodes": 0,
                "visible_edges": 0,
                "media_families_seen": [],
                "cross_media_transition_ratio": 0.0,
                "mixed_session_ratio": 0.0,
                "media_mix": {},
            },
            "session_intelligence": {
                "top_nodes_by_media": {"music": [], "podcast": [], "audiobook": []},
                "dominant_transitions": [],
                "mixed_sessions": [],
                "seed_nodes": [],
            },
            "nodes": [],
            "edges": [],
        }

    total_sessions = int(frame["session_id"].nunique())
    total_events = int(len(frame))

    node_rows: list[dict[str, Any]] = []
    for node_key, group in frame.groupby("node_key", sort=False):
        first = group.iloc[0]
        node_rows.append(
            {
                "node_key": str(node_key),
                "name": str(first.get("node_name", "")).strip(),
                "node_type": str(first.get("node_type", "")).strip(),
                "media_family": str(first.get("media_family", "")).strip(),
                "spotify_uri": _clean_text(first.get("node_uri")) or _clean_text(first.get("item_uri")),
                "play_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
                "session_share": float(group["session_id"].nunique() / max(1, total_sessions)),
                "total_ms_played": int(float(group["ms_played"].sum())),
                "first_seen": group["ts"].min().isoformat(),
                "last_seen": group["ts"].max().isoformat(),
                "top_items": _node_top_items(group),
            }
        )
    node_rows.sort(key=lambda row: (int(row["play_count"]), int(row["session_count"]), row["name"]), reverse=True)
    visible_nodes = node_rows[: max(1, int(node_limit))]

    shifted = frame.shift(1)
    transition_mask = frame["session_id"].eq(shifted["session_id"]) & frame["node_key"].ne(shifted["node_key"])
    transitions = frame.loc[transition_mask].copy()
    if not transitions.empty:
        transitions["source_key"] = shifted.loc[transition_mask, "node_key"].to_numpy()
        transitions["source_name"] = shifted.loc[transition_mask, "node_name"].to_numpy()
        transitions["source_type"] = shifted.loc[transition_mask, "node_type"].to_numpy()
        transitions["source_media_family"] = shifted.loc[transition_mask, "media_family"].to_numpy()
        transitions["target_key"] = transitions["node_key"].to_numpy()
        transitions["target_name"] = transitions["node_name"].to_numpy()
        transitions["target_type"] = transitions["node_type"].to_numpy()
        transitions["target_media_family"] = transitions["media_family"].to_numpy()
        transitions["cross_media"] = transitions["source_media_family"] != transitions["target_media_family"]
        edge_frame = (
            transitions.groupby(
                [
                    "source_key",
                    "source_name",
                    "source_type",
                    "source_media_family",
                    "target_key",
                    "target_name",
                    "target_type",
                    "target_media_family",
                    "cross_media",
                ],
                dropna=False,
            )
            .agg(
                transition_count=("target_key", "size"),
                target_ms_played=("ms_played", "sum"),
            )
            .reset_index()
            .sort_values(
                ["transition_count", "cross_media", "source_name", "target_name"],
                ascending=[False, False, True, True],
            )
        )
        edge_rows = [
            {
                "source_key": str(row.get("source_key", "")).strip(),
                "source_name": str(row.get("source_name", "")).strip(),
                "source_type": str(row.get("source_type", "")).strip(),
                "source_media_family": str(row.get("source_media_family", "")).strip(),
                "target_key": str(row.get("target_key", "")).strip(),
                "target_name": str(row.get("target_name", "")).strip(),
                "target_type": str(row.get("target_type", "")).strip(),
                "target_media_family": str(row.get("target_media_family", "")).strip(),
                "cross_media": bool(row.get("cross_media", False)),
                "transition_count": int(row.get("transition_count", 0) or 0),
                "target_ms_played": int(float(row.get("target_ms_played", 0.0) or 0.0)),
            }
            for _, row in edge_frame.iterrows()
        ]
    else:
        edge_rows = []
    visible_edges = edge_rows[: max(1, int(edge_limit))]

    session_rows: list[dict[str, Any]] = []
    for session_id, group in frame.groupby("session_id", sort=False):
        names = [str(value).strip() for value in group["node_name"].tolist() if str(value).strip()]
        media_sequence = [str(value).strip() for value in group["media_family"].tolist() if str(value).strip()]
        media_distinct = sorted(set(media_sequence))
        cross_media_switches = sum(
            1
            for idx in range(1, len(media_sequence))
            if media_sequence[idx] != media_sequence[idx - 1]
        )
        session_rows.append(
            {
                "session_id": int(session_id),
                "start_ts": group["ts"].min().isoformat(),
                "end_ts": group["ts"].max().isoformat(),
                "play_count": int(len(group)),
                "total_ms_played": int(float(group["ms_played"].sum())),
                "media_families": media_distinct,
                "media_family_count": int(len(media_distinct)),
                "cross_media_switches": int(cross_media_switches),
                "sequence_preview": names[:6],
            }
        )
    session_rows.sort(
        key=lambda row: (
            int(row["media_family_count"]),
            int(row["cross_media_switches"]),
            int(row["play_count"]),
            str(row["end_ts"]),
        ),
        reverse=True,
    )
    mixed_sessions = [row for row in session_rows if int(row["media_family_count"]) > 1][: max(1, int(session_limit))]

    media_mix: dict[str, dict[str, Any]] = {}
    for media_family, group in frame.groupby("media_family", sort=False):
        media_mix[str(media_family)] = {
            "play_count": int(len(group)),
            "total_ms_played": int(float(group["ms_played"].sum())),
            "session_count": int(group["session_id"].nunique()),
        }

    total_transitions = sum(int(row.get("transition_count", 0) or 0) for row in edge_rows)
    cross_media_transitions = sum(
        int(row.get("transition_count", 0) or 0)
        for row in edge_rows
        if bool(row.get("cross_media"))
    )
    mixed_session_count = sum(1 for row in session_rows if int(row["media_family_count"]) > 1)

    return {
        "summary": {
            "events_analyzed": total_events,
            "sessions_analyzed": total_sessions,
            "unique_nodes": len(node_rows),
            "visible_nodes": len(visible_nodes),
            "visible_edges": len(visible_edges),
            "media_families_seen": sorted(media_mix.keys()),
            "cross_media_transition_ratio": float(cross_media_transitions / max(1, total_transitions)),
            "mixed_session_ratio": float(mixed_session_count / max(1, total_sessions)),
            "media_mix": media_mix,
        },
        "session_intelligence": {
            "top_nodes_by_media": _top_nodes_by_media(node_rows),
            "dominant_transitions": edge_rows[:10],
            "mixed_sessions": mixed_sessions,
            "seed_nodes": _seed_nodes_for_bridges(node_rows, limit=max(1, min(6, len(node_rows)))),
        },
        "nodes": visible_nodes,
        "edges": visible_edges,
    }


def _cross_media_catalog_bridges(
    client: SpotifyPublicCatalogClient,
    *,
    seed_nodes: list[dict[str, Any]],
    market: str,
    recommendation_limit: int,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for seed in seed_nodes:
        seed_name = _clean_text(seed.get("name"))
        if not seed_name:
            continue
        results_by_type: dict[str, list[dict[str, Any]]] = {}
        for item_type in _cross_media_bridge_types(str(seed.get("node_type", "")).strip()):
            try:
                rows = client.search(
                    query=seed_name,
                    item_types=[item_type],
                    limit=max(1, int(recommendation_limit)),
                    market=market,
                ).get(item_type, [])
            except (SpotifyPublicCatalogError, ValueError) as exc:
                logger.warning("Cross-media bridge search failed for %s -> %s: %s", seed_name, item_type, exc)
                rows = []
            formatted = [
                _format_catalog_result(item_type, row)
                for row in rows[: max(1, int(recommendation_limit))]
                if isinstance(row, dict)
            ]
            if formatted:
                results_by_type[item_type] = formatted
        bundles.append(
            {
                "seed_name": seed_name,
                "seed_type": str(seed.get("node_type", "")).strip(),
                "media_family": str(seed.get("media_family", "")).strip(),
                "play_count": int(seed.get("play_count", 0) or 0),
                "query": seed_name,
                "results": results_by_type,
            }
        )
    return bundles


def _top_artists_from_history(df: pd.DataFrame, *, lookback_days: int, limit: int) -> list[str]:
    recent = _recent_history(df, lookback_days)
    if "master_metadata_album_artist_name" not in recent.columns:
        return []
    names = recent["master_metadata_album_artist_name"].fillna("").astype(str).str.strip()
    counts = names[names != ""].value_counts()
    return [str(name) for name in counts.head(max(1, int(limit))).index.tolist()]


def _top_tracks_from_history(df: pd.DataFrame, *, lookback_days: int, limit: int) -> list[dict[str, Any]]:
    recent = _recent_history(df, lookback_days)
    required = {"spotify_track_uri", "master_metadata_track_name", "master_metadata_album_artist_name"}
    if not required.issubset(recent.columns):
        return []

    frame = recent[list(required)].copy()
    frame["spotify_track_uri"] = frame["spotify_track_uri"].fillna("").astype(str).str.strip()
    frame["master_metadata_track_name"] = frame["master_metadata_track_name"].fillna("").astype(str).str.strip()
    frame["master_metadata_album_artist_name"] = (
        frame["master_metadata_album_artist_name"].fillna("").astype(str).str.strip()
    )
    frame = frame[(frame["spotify_track_uri"] != "") & (frame["master_metadata_track_name"] != "")]
    if frame.empty:
        return []

    grouped = (
        frame.groupby("spotify_track_uri", dropna=False)
        .agg(
            plays=("spotify_track_uri", "size"),
            track_name=("master_metadata_track_name", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
            artist_name=(
                "master_metadata_album_artist_name",
                lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0],
            ),
        )
        .sort_values(["plays", "track_name"], ascending=[False, True])
        .reset_index()
    )
    return grouped.head(max(1, int(limit))).to_dict(orient="records")


def _parse_release_date(value: str, precision: str) -> pd.Timestamp | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if precision == "year":
        raw = f"{raw}-01-01"
    elif precision == "month":
        raw = f"{raw}-01"
    timestamp = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp


def _dedupe_album_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("name", "")).casefold(),
            str(row.get("release_date", "")),
            str(row.get("album_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


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
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(markdown_lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    return path


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _playlist_state_path(output_dir: Path, playlist_slug: str) -> Path:
    return output_dir / "analysis" / "public_spotify" / "playlist_state" / f"{playlist_slug}.json"


def _release_state_path(output_dir: Path, artist_slug: str) -> Path:
    return output_dir / "analysis" / "public_spotify" / "release_state" / f"{artist_slug}.json"


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


def _playlist_snapshot(playlist_payload: dict[str, Any], item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_urls = [
        str(image.get("url", "")).strip()
        for image in playlist_payload.get("images", [])
        if isinstance(image, dict) and str(image.get("url", "")).strip()
    ]
    return {
        "snapshot_id": str(playlist_payload.get("snapshot_id", "")).strip(),
        "name": str(playlist_payload.get("name", "")).strip(),
        "description": str(playlist_payload.get("description", "")).strip(),
        "owner_name": str(playlist_payload.get("owner", {}).get("display_name", "")).strip(),
        "followers_total": int(playlist_payload.get("followers", {}).get("total", 0) or 0),
        "public": playlist_payload.get("public"),
        "collaborative": playlist_payload.get("collaborative"),
        "image_urls": image_urls,
        "items": item_rows,
    }


def _playlist_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return {
            "is_first_snapshot": True,
            "added_tracks": current.get("items", []),
            "removed_tracks": [],
            "metadata_changes": {},
        }

    prev_items = {
        str(row.get("spotify_url", "")).strip() or str(row.get("track_name", "")).strip(): row
        for row in previous.get("items", [])
        if isinstance(row, dict)
    }
    curr_items = {
        str(row.get("spotify_url", "")).strip() or str(row.get("track_name", "")).strip(): row
        for row in current.get("items", [])
        if isinstance(row, dict)
    }
    added = [row for key, row in curr_items.items() if key not in prev_items]
    removed = [row for key, row in prev_items.items() if key not in curr_items]
    metadata_changes: dict[str, dict[str, Any]] = {}
    for key in ("snapshot_id", "name", "description", "owner_name", "followers_total", "public", "collaborative", "image_urls"):
        before = previous.get(key)
        after = current.get(key)
        if before != after:
            metadata_changes[key] = {"before": before, "after": after}
    return {
        "is_first_snapshot": False,
        "added_tracks": added,
        "removed_tracks": removed,
        "metadata_changes": metadata_changes,
    }


def _release_state_rows(previous: dict[str, Any] | None) -> set[str]:
    if previous is None:
        return set()
    rows = previous.get("release_ids", [])
    if not isinstance(rows, list):
        return set()
    return {str(item).strip() for item in rows if str(item).strip()}


def _build_client(args: argparse.Namespace) -> SpotifyPublicCatalogClient:
    client = SpotifyPublicCatalogClient.from_env(market=str(args.spotify_market or "US"))
    if client is None:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required for Spotify public metadata tools.")
    return client


def _load_history_if_needed(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    return load_streaming_history(
        Path(args.data_dir).expanduser().resolve(),
        include_video=bool(args.include_video),
        logger=logger,
    )


def _resolve_artists(
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    history_top_n: int,
    history_lookback_days: int,
) -> list[str]:
    explicit = _split_pipe_list(getattr(args, "artists", None))
    if explicit:
        return explicit
    history_df = _load_history_if_needed(args, logger)
    return _top_artists_from_history(history_df, lookback_days=history_lookback_days, limit=history_top_n)


def _handle_explain_artists(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    artists = _resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    rows: list[dict[str, Any]] = []
    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            rows.append({"queried_name": artist_name, "matched": False})
            continue
        related = client.get_related_artists(metadata.spotify_id, limit=max(1, int(args.related_limit)))
        rows.append(
            {
                "queried_name": artist_name,
                "matched": True,
                "spotify_id": metadata.spotify_id,
                "matched_name": metadata.name,
                "popularity": metadata.popularity,
                "followers_total": metadata.followers_total,
                "genres": metadata.genres,
                "spotify_url": metadata.spotify_url,
                "image_url": metadata.image_url,
                "related_artists": [item.name for item in related],
            }
        )

    payload = {
        "command": "explain-artists",
        "market": str(args.spotify_market).upper(),
        "lookback_days": int(args.lookback_days),
        "artists": rows,
    }
    markdown_lines = [
        "# Spotify Artist Explainer",
        "",
        f"- Market: `{str(args.spotify_market).upper()}`",
        f"- Artists reviewed: `{len(rows)}`",
        "",
    ]
    for row in rows:
        if not row.get("matched"):
            markdown_lines.append(f"- {row['queried_name']}: no Spotify public match found")
            continue
        genres = ", ".join(row["genres"][:3]) if row.get("genres") else "n/a"
        related = ", ".join(row["related_artists"][:5]) if row.get("related_artists") else "n/a"
        markdown_lines.append(
            f"- {row['matched_name']}: popularity `{row['popularity']}`, followers `{row['followers_total']}`, "
            f"genres `{genres}`, related `{related}`"
        )

    stem = f"artist_explainer_{_slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="artist_explainer",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"artist_explainer_json={json_path}")
    print(f"artist_explainer_md={md_path}")
    print(f"artists_explained={len(rows)}")
    return 0


def _handle_release_tracker(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    artists = _resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    include_groups = ",".join(_split_csv_list(args.include_groups))
    cutoff_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(args.since_days)))

    artist_rows: list[dict[str, Any]] = []
    total_releases = 0
    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            artist_rows.append({"artist_name": artist_name, "matched": False, "releases": []})
            continue
        albums = client.get_artist_albums(
            metadata.spotify_id,
            include_groups=include_groups,
            limit=max(10, int(args.per_artist_limit) * 3),
            market=str(args.spotify_market).upper(),
        )
        deduped = _dedupe_album_rows(albums)
        recent_releases: list[dict[str, Any]] = []
        for album in deduped:
            release_date = str(album.get("release_date", "")).strip()
            precision = str(album.get("release_date_precision", "day")).strip()
            release_ts = _parse_release_date(release_date, precision)
            if release_ts is None or release_ts < cutoff_ts:
                continue
            recent_releases.append(
                {
                    "album_id": str(album.get("id", "")).strip(),
                    "album_name": str(album.get("name", "")).strip(),
                    "album_type": str(album.get("album_type", "")).strip(),
                    "release_date": release_date,
                    "release_date_precision": precision,
                    "total_tracks": int(album.get("total_tracks", 0) or 0),
                    "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                    "image_url": next(
                        (
                            str(image.get("url", "")).strip()
                            for image in album.get("images", [])
                            if isinstance(image, dict) and str(image.get("url", "")).strip()
                        ),
                        None,
                    ),
                }
            )
        recent_releases.sort(key=lambda row: (_parse_release_date(row["release_date"], row["release_date_precision"]) or cutoff_ts), reverse=True)
        limited = recent_releases[: max(1, int(args.per_artist_limit))]
        total_releases += len(limited)
        artist_rows.append(
            {
                "artist_name": metadata.name,
                "spotify_url": metadata.spotify_url,
                "matched": True,
                "releases": limited,
            }
        )

    payload = {
        "command": "release-tracker",
        "market": str(args.spotify_market).upper(),
        "since_days": int(args.since_days),
        "include_groups": include_groups,
        "artists": artist_rows,
    }
    markdown_lines = [
        "# Spotify Release Tracker",
        "",
        f"- Market: `{str(args.spotify_market).upper()}`",
        f"- Since: last `{int(args.since_days)}` days",
        f"- Artists tracked: `{len(artist_rows)}`",
        f"- Releases found: `{total_releases}`",
        "",
    ]
    for row in artist_rows:
        releases = row.get("releases", [])
        if not row.get("matched"):
            markdown_lines.append(f"- {row['artist_name']}: no Spotify public match found")
            continue
        if not releases:
            markdown_lines.append(f"- {row['artist_name']}: no recent releases in the selected window")
            continue
        markdown_lines.append(f"- {row['artist_name']}:")
        for release in releases[:5]:
            markdown_lines.append(
                f"  - {release['release_date']} | {release['album_type']} | {release['album_name']} "
                f"({release['total_tracks']} tracks)"
            )

    stem = f"release_tracker_{int(args.since_days)}d_{_slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="release_tracker",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"release_tracker_json={json_path}")
    print(f"release_tracker_md={md_path}")
    print(f"releases_found={total_releases}")
    return 0


def _handle_market_check(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    explicit_tracks = _split_pipe_list(args.tracks)
    if explicit_tracks:
        track_rows = [{"spotify_track_uri": value, "track_name": value, "artist_name": ""} for value in explicit_tracks]
    else:
        history_df = _load_history_if_needed(args, logger)
        track_rows = _top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_n)
    markets = [market.strip().upper() for market in _split_csv_list(args.markets)]
    results: list[dict[str, Any]] = []
    for track in track_rows:
        raw_identifier = str(track.get("spotify_track_uri", "")).strip()
        if not raw_identifier:
            continue
        track_payload = client.get_track(raw_identifier, market=str(args.spotify_market).upper())
        available_markets = [
            str(value).strip().upper()
            for value in track_payload.get("available_markets", [])
            if str(value).strip()
        ]
        artists = [
            str(item.get("name", "")).strip()
            for item in track_payload.get("artists", [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]
        market_map = {market: market in set(available_markets) for market in markets}
        results.append(
            {
                "track_name": str(track_payload.get("name", track.get("track_name", raw_identifier))).strip(),
                "artist_names": artists,
                "album_name": str(track_payload.get("album", {}).get("name", "")).strip(),
                "spotify_track_id": str(track_payload.get("id", "")).strip(),
                "spotify_url": str(track_payload.get("external_urls", {}).get("spotify", "")).strip(),
                "history_plays": int(track.get("plays", 0) or 0),
                "available_markets_count": len(available_markets),
                "market_availability": market_map,
            }
        )

    total_checks = len(results) * max(1, len(markets))
    available_checks = sum(
        1 for row in results for value in row.get("market_availability", {}).values() if bool(value)
    )
    payload = {
        "command": "market-check",
        "markets": markets,
        "tracks": results,
        "available_ratio": float(available_checks / max(1, total_checks)),
    }
    markdown_lines = [
        "# Spotify Market Availability",
        "",
        f"- Markets: `{', '.join(markets)}`",
        f"- Tracks checked: `{len(results)}`",
        f"- Availability ratio: `{float(payload['available_ratio']):.1%}`",
        "",
    ]
    for row in results[:20]:
        flags = ", ".join(f"{market}={'yes' if available else 'no'}" for market, available in row["market_availability"].items())
        markdown_lines.append(
            f"- {row['track_name']} by {', '.join(row['artist_names'])}: {flags} "
            f"(history plays `{row['history_plays']}`)"
        )

    stem = f"market_check_{_slugify('-'.join(markets))}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="market_check",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"market_check_json={json_path}")
    print(f"market_check_md={md_path}")
    print(f"tracks_checked={len(results)}")
    return 0


def _handle_discography(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    artists = _resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    include_groups = ",".join(_split_csv_list(args.include_groups))
    rows: list[dict[str, Any]] = []
    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            rows.append({"artist_name": artist_name, "matched": False, "albums": []})
            continue
        albums = client.get_artist_albums(
            metadata.spotify_id,
            include_groups=include_groups,
            limit=max(10, int(args.album_limit) * 3),
            market=str(args.spotify_market).upper(),
        )
        deduped = _dedupe_album_rows(albums)
        timeline_rows: list[dict[str, Any]] = []
        for album in deduped:
            timeline_rows.append(
                {
                    "album_name": str(album.get("name", "")).strip(),
                    "album_type": str(album.get("album_type", "")).strip(),
                    "release_date": str(album.get("release_date", "")).strip(),
                    "release_date_precision": str(album.get("release_date_precision", "day")).strip(),
                    "total_tracks": int(album.get("total_tracks", 0) or 0),
                    "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                }
            )
        timeline_rows.sort(
            key=lambda row: _parse_release_date(row["release_date"], row["release_date_precision"]) or pd.Timestamp(0, tz="UTC")
        )
        rows.append(
            {
                "artist_name": metadata.name,
                "spotify_url": metadata.spotify_url,
                "matched": True,
                "albums": timeline_rows[: max(1, int(args.album_limit))],
            }
        )

    payload = {
        "command": "discography",
        "market": str(args.spotify_market).upper(),
        "include_groups": include_groups,
        "artists": rows,
    }
    markdown_lines = [
        "# Spotify Discography Timeline",
        "",
        f"- Market: `{str(args.spotify_market).upper()}`",
        f"- Artists covered: `{len(rows)}`",
        "",
    ]
    for row in rows:
        if not row.get("matched"):
            markdown_lines.append(f"- {row['artist_name']}: no Spotify public match found")
            continue
        markdown_lines.append(f"- {row['artist_name']}:")
        for album in row["albums"][:10]:
            markdown_lines.append(
                f"  - {album['release_date']} | {album['album_type']} | {album['album_name']} "
                f"({album['total_tracks']} tracks)"
            )

    stem = f"discography_{_slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="discography",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"discography_json={json_path}")
    print(f"discography_md={md_path}")
    print(f"artists_covered={len(rows)}")
    return 0


def _playlist_item_rows(
    client: SpotifyPublicCatalogClient,
    *,
    playlist_id: str,
    market: str,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    playlist_payload = client.get_playlist(playlist_id, market=market)
    items = client.get_playlist_tracks(playlist_id, limit=limit, market=market)

    track_rows: list[dict[str, Any]] = []
    artist_counter: Counter[str] = Counter()
    for item in items:
        track = item.get("track", {})
        if not isinstance(track, dict):
            continue
        artists = [
            str(artist.get("name", "")).strip()
            for artist in track.get("artists", [])
            if isinstance(artist, dict) and str(artist.get("name", "")).strip()
        ]
        for artist_name in artists:
            artist_counter[artist_name] += 1
        track_rows.append(
            {
                "added_at": str(item.get("added_at", "")).strip(),
                "track_name": str(track.get("name", "")).strip(),
                "artist_names": artists,
                "album_name": str(track.get("album", {}).get("name", "")).strip(),
                "duration_ms": int(track.get("duration_ms", 0) or 0),
                "explicit": bool(track.get("explicit", False)),
                "spotify_url": str(track.get("external_urls", {}).get("spotify", "")).strip(),
            }
        )
    return playlist_payload, track_rows, artist_counter


def _handle_playlist_view(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    playlist_id = parse_spotify_id(args.playlist, expected_kind="playlist")
    playlist_payload, track_rows, artist_counter = _playlist_item_rows(
        client,
        playlist_id=playlist_id,
        market=str(args.spotify_market).upper(),
        limit=max(1, int(args.item_limit)),
    )

    payload = {
        "command": "playlist-view",
        "market": str(args.spotify_market).upper(),
        "playlist": {
            "playlist_id": playlist_id,
            "name": str(playlist_payload.get("name", "")).strip(),
            "description": str(playlist_payload.get("description", "")).strip(),
            "owner_name": str(playlist_payload.get("owner", {}).get("display_name", "")).strip(),
            "public": playlist_payload.get("public"),
            "collaborative": playlist_payload.get("collaborative"),
            "followers_total": int(playlist_payload.get("followers", {}).get("total", 0) or 0),
            "spotify_url": str(playlist_payload.get("external_urls", {}).get("spotify", "")).strip(),
            "tracks_total": int(playlist_payload.get("tracks", {}).get("total", 0) or 0),
        },
        "top_artists": [{"artist_name": name, "track_count": count} for name, count in artist_counter.most_common(10)],
        "items": track_rows,
    }
    markdown_lines = [
        "# Spotify Playlist View",
        "",
        f"- Playlist: `{payload['playlist']['name']}`",
        f"- Owner: `{payload['playlist']['owner_name']}`",
        f"- Followers: `{payload['playlist']['followers_total']}`",
        f"- Tracks loaded: `{len(track_rows)}`",
        "",
        "## Top Artists In Playlist",
        "",
    ]
    for row in payload["top_artists"]:
        markdown_lines.append(f"- {row['artist_name']}: `{row['track_count']}` tracks")
    markdown_lines.extend(["", "## Tracks", ""])
    for row in track_rows[:20]:
        markdown_lines.append(f"- {row['track_name']} by {', '.join(row['artist_names'])} | {row['album_name']}")

    stem = f"playlist_{_slugify(payload['playlist']['name'] or playlist_id)}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="playlist_view",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"playlist_view_json={json_path}")
    print(f"playlist_view_md={md_path}")
    print(f"playlist_tracks_loaded={len(track_rows)}")
    return 0


def _handle_discovery_search(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    item_types = _split_csv_list(args.types)
    results = client.search(
        query=str(args.query),
        item_types=item_types,
        limit=max(1, int(args.limit)),
        market=str(args.spotify_market).upper(),
    )

    payload_results: dict[str, list[dict[str, Any]]] = {}
    markdown_lines = [
        "# Spotify Discovery Search",
        "",
        f"- Query: `{args.query}`",
        f"- Types: `{', '.join(item_types)}`",
        f"- Market: `{str(args.spotify_market).upper()}`",
        "",
    ]

    for item_type in item_types:
        rows = results.get(item_type, [])
        formatted_rows: list[dict[str, Any]] = []
        markdown_lines.append(f"## {item_type.title()} Results")
        markdown_lines.append("")
        for row in rows:
            if item_type == "artist":
                formatted = {
                    "name": str(row.get("name", "")).strip(),
                    "spotify_id": str(row.get("id", "")).strip(),
                    "popularity": row.get("popularity"),
                    "genres": row.get("genres", []),
                    "followers_total": row.get("followers", {}).get("total"),
                    "spotify_url": str(row.get("external_urls", {}).get("spotify", "")).strip(),
                }
                markdown_lines.append(
                    f"- {formatted['name']}: popularity `{formatted['popularity']}`, "
                    f"followers `{formatted['followers_total']}`"
                )
            elif item_type == "album":
                formatted = {
                    "name": str(row.get("name", "")).strip(),
                    "spotify_id": str(row.get("id", "")).strip(),
                    "artist_names": [
                        str(item.get("name", "")).strip()
                        for item in row.get("artists", [])
                        if isinstance(item, dict) and str(item.get("name", "")).strip()
                    ],
                    "release_date": str(row.get("release_date", "")).strip(),
                    "album_type": str(row.get("album_type", "")).strip(),
                    "spotify_url": str(row.get("external_urls", {}).get("spotify", "")).strip(),
                }
                markdown_lines.append(
                    f"- {formatted['name']} by {', '.join(formatted['artist_names'])}: `{formatted['release_date']}`"
                )
            elif item_type == "track":
                formatted = {
                    "name": str(row.get("name", "")).strip(),
                    "spotify_id": str(row.get("id", "")).strip(),
                    "artist_names": [
                        str(item.get("name", "")).strip()
                        for item in row.get("artists", [])
                        if isinstance(item, dict) and str(item.get("name", "")).strip()
                    ],
                    "album_name": str(row.get("album", {}).get("name", "")).strip(),
                    "explicit": bool(row.get("explicit", False)),
                    "spotify_url": str(row.get("external_urls", {}).get("spotify", "")).strip(),
                }
                markdown_lines.append(
                    f"- {formatted['name']} by {', '.join(formatted['artist_names'])} | {formatted['album_name']}"
                )
            else:
                formatted = {
                    "name": str(row.get("name", "")).strip(),
                    "spotify_id": str(row.get("id", "")).strip(),
                    "owner_name": str(row.get("owner", {}).get("display_name", "")).strip(),
                    "tracks_total": int(row.get("tracks", {}).get("total", 0) or 0),
                    "spotify_url": str(row.get("external_urls", {}).get("spotify", "")).strip(),
                }
                markdown_lines.append(
                    f"- {formatted['name']} by {formatted['owner_name']} | tracks `{formatted['tracks_total']}`"
                )
            formatted_rows.append(formatted)
        if not rows:
            markdown_lines.append("- No results")
        markdown_lines.append("")
        payload_results[item_type] = formatted_rows

    payload = {
        "command": "discovery-search",
        "query": str(args.query),
        "market": str(args.spotify_market).upper(),
        "results": payload_results,
    }
    stem = f"discovery_{_slugify(str(args.query))}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="discovery_search",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"discovery_search_json={json_path}")
    print(f"discovery_search_md={md_path}")
    print(f"result_types={len(payload_results)}")
    return 0


def _handle_catalog_linkouts(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    history_df = _load_history_if_needed(args, logger)
    artist_names = _top_artists_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_artists)
    track_rows = _top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_tracks)

    artist_rows: list[dict[str, Any]] = []
    for artist_name in artist_names:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            continue
        artist_rows.append(
            {
                "artist_name": metadata.name,
                "spotify_id": metadata.spotify_id,
                "spotify_url": metadata.spotify_url,
                "image_url": metadata.image_url,
                "popularity": metadata.popularity,
                "genres": metadata.genres,
            }
        )

    track_linkouts: list[dict[str, Any]] = []
    for row in track_rows:
        raw_identifier = str(row.get("spotify_track_uri", "")).strip()
        if not raw_identifier:
            continue
        track_payload = client.get_track(raw_identifier, market=str(args.spotify_market).upper())
        track_linkouts.append(
            {
                "track_name": str(track_payload.get("name", row.get("track_name", raw_identifier))).strip(),
                "spotify_id": str(track_payload.get("id", "")).strip(),
                "spotify_url": str(track_payload.get("external_urls", {}).get("spotify", "")).strip(),
                "album_name": str(track_payload.get("album", {}).get("name", "")).strip(),
                "album_id": str(track_payload.get("album", {}).get("id", "")).strip(),
                "album_url": str(track_payload.get("album", {}).get("external_urls", {}).get("spotify", "")).strip(),
                "image_url": next(
                    (
                        str(image.get("url", "")).strip()
                        for image in track_payload.get("album", {}).get("images", [])
                        if isinstance(image, dict) and str(image.get("url", "")).strip()
                    ),
                    None,
                ),
                "artist_names": [
                    str(item.get("name", "")).strip()
                    for item in track_payload.get("artists", [])
                    if isinstance(item, dict) and str(item.get("name", "")).strip()
                ],
                "history_plays": int(row.get("plays", 0) or 0),
            }
        )

    payload = {
        "command": "catalog-linkouts",
        "market": str(args.spotify_market).upper(),
        "artists": artist_rows,
        "tracks": track_linkouts,
    }
    markdown_lines = [
        "# Spotify Catalog Linkouts",
        "",
        f"- Artists bundled: `{len(artist_rows)}`",
        f"- Tracks bundled: `{len(track_linkouts)}`",
        "",
        "## Artists",
        "",
    ]
    for row in artist_rows[:10]:
        markdown_lines.append(f"- {row['artist_name']}: {row['spotify_url']}")
    markdown_lines.extend(["", "## Tracks", ""])
    for row in track_linkouts[:20]:
        markdown_lines.append(f"- {row['track_name']} by {', '.join(row['artist_names'])}: {row['spotify_url']}")

    stem = f"catalog_linkouts_{int(args.lookback_days)}d"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="catalog_linkouts",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"catalog_linkouts_json={json_path}")
    print(f"catalog_linkouts_md={md_path}")
    print(f"catalog_items={len(artist_rows) + len(track_linkouts)}")
    return 0


def _handle_artist_graph(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    artists = _resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            continue
        nodes[metadata.spotify_id] = {
            "spotify_id": metadata.spotify_id,
            "name": metadata.name,
            "spotify_url": metadata.spotify_url,
            "genres": metadata.genres,
            "popularity": metadata.popularity,
            "followers_total": metadata.followers_total,
            "image_url": metadata.image_url,
            "seed": True,
        }
        related = client.get_related_artists(metadata.spotify_id, limit=max(1, int(args.related_limit)))
        for target in related:
            nodes[target.spotify_id] = {
                "spotify_id": target.spotify_id,
                "name": target.name,
                "spotify_url": target.spotify_url,
                "genres": target.genres,
                "popularity": target.popularity,
                "followers_total": target.followers_total,
                "image_url": target.image_url,
                "seed": False,
            }
            edges.append({"source": metadata.spotify_id, "target": target.spotify_id})

    payload = {
        "command": "artist-graph",
        "seed_artists": artists,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    markdown_lines = [
        "# Spotify Artist Graph",
        "",
        f"- Seed artists: `{len(artists)}`",
        f"- Nodes: `{len(nodes)}`",
        f"- Edges: `{len(edges)}`",
        "",
    ]
    for seed_name in artists:
        markdown_lines.append(f"- Seed: {seed_name}")

    stem = f"artist_graph_{_slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="artist_graph",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"artist_graph_json={json_path}")
    print(f"artist_graph_md={md_path}")
    print(f"artist_graph_nodes={len(nodes)}")
    return 0


def _handle_cross_media_taste_graph(
    args: argparse.Namespace,
    client: SpotifyPublicCatalogClient,
    logger: logging.Logger,
) -> int:
    history_df = _load_history_if_needed(args, logger)
    frame = _cross_media_history_frame(
        history_df,
        lookback_days=int(args.lookback_days),
        session_gap_minutes=int(args.session_gap_minutes),
    )
    graph_payload = _cross_media_graph_payload(
        frame,
        node_limit=int(args.node_limit),
        edge_limit=int(args.edge_limit),
        session_limit=int(args.session_limit),
    )
    seed_nodes = _seed_nodes_for_bridges(
        graph_payload.get("session_intelligence", {}).get("seed_nodes", []),
        limit=max(1, int(args.bridge_limit)),
    )
    bridges = _cross_media_catalog_bridges(
        client,
        seed_nodes=seed_nodes,
        market=str(args.spotify_market).upper(),
        recommendation_limit=int(args.recommendation_limit),
        logger=logger,
    )

    payload = {
        "command": "cross-media-taste-graph",
        "market": str(args.spotify_market).upper(),
        "lookback_days": int(args.lookback_days),
        "session_gap_minutes": int(args.session_gap_minutes),
        "summary": graph_payload["summary"],
        "session_intelligence": {
            **graph_payload["session_intelligence"],
            "seed_nodes": seed_nodes,
        },
        "nodes": graph_payload["nodes"],
        "edges": graph_payload["edges"],
        "recommendation_bridges": bridges,
    }
    markdown_lines = [
        "# Spotify Cross-Media Taste Graph",
        "",
        f"- Market: `{payload['market']}`",
        f"- Lookback: last `{payload['lookback_days']}` days",
        f"- Events analyzed: `{payload['summary']['events_analyzed']}`",
        f"- Sessions analyzed: `{payload['summary']['sessions_analyzed']}`",
        f"- Cross-media transition ratio: `{payload['summary']['cross_media_transition_ratio']:.1%}`",
        f"- Mixed-session ratio: `{payload['summary']['mixed_session_ratio']:.1%}`",
        "",
        "## Media Mix",
        "",
    ]
    for media_family, stats in payload["summary"]["media_mix"].items():
        markdown_lines.append(
            f"- {media_family}: plays `{stats['play_count']}`, sessions `{stats['session_count']}`, "
            f"ms played `{stats['total_ms_played']}`"
        )

    markdown_lines.extend(["", "## Top Nodes By Medium", ""])
    top_nodes_by_media = payload["session_intelligence"]["top_nodes_by_media"]
    for media_family in ("music", "podcast", "audiobook"):
        rows = top_nodes_by_media.get(media_family, [])
        if not rows:
            markdown_lines.append(f"- {media_family}: none")
            continue
        names = ", ".join(f"{row['name']} ({row['play_count']})" for row in rows[:3])
        markdown_lines.append(f"- {media_family}: {names}")

    markdown_lines.extend(["", "## Mixed Sessions", ""])
    for row in payload["session_intelligence"]["mixed_sessions"][: max(1, int(args.session_limit))]:
        markdown_lines.append(
            f"- session `{row['session_id']}` | media `{', '.join(row['media_families'])}` | "
            f"switches `{row['cross_media_switches']}` | sequence `{', '.join(row['sequence_preview'])}`"
        )
    if not payload["session_intelligence"]["mixed_sessions"]:
        markdown_lines.append("- No mixed-media sessions found in the selected window.")

    markdown_lines.extend(["", "## Bridge Queries", ""])
    for bridge in bridges:
        result_types = ", ".join(sorted(bridge.get("results", {}).keys())) or "none"
        markdown_lines.append(
            f"- {bridge['seed_name']} ({bridge['media_family']}): query `{bridge['query']}` -> {result_types}"
        )

    stem = f"cross_media_taste_graph_{int(args.lookback_days)}d"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="cross_media_taste_graph",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"cross_media_taste_graph_json={json_path}")
    print(f"cross_media_taste_graph_md={md_path}")
    print(f"cross_media_nodes={len(payload['nodes'])}")
    return 0


def _handle_release_inbox(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    artists = _resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    include_groups = ",".join(_split_csv_list(args.include_groups))
    cutoff_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(args.since_days)))
    state_key = (
        _slugify("explicit-" + "-".join(artists))
        if getattr(args, "artists", None)
        else f"history-{int(args.lookback_days)}d-top{int(args.top_n)}-since{int(args.since_days)}d"
    )
    state_path = _release_state_path(Path(args.output_dir).expanduser().resolve(), state_key)
    previous_state = _read_json_if_exists(state_path)
    seen_release_ids = _release_state_rows(previous_state)

    inbox_rows: list[dict[str, Any]] = []
    current_release_ids: list[str] = []
    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            continue
        albums = client.get_artist_albums(
            metadata.spotify_id,
            include_groups=include_groups,
            limit=max(10, int(args.per_artist_limit) * 3),
            market=str(args.spotify_market).upper(),
        )
        deduped = _dedupe_album_rows(albums)
        for album in deduped:
            album_id = str(album.get("id", "")).strip()
            release_date = str(album.get("release_date", "")).strip()
            precision = str(album.get("release_date_precision", "day")).strip()
            release_ts = _parse_release_date(release_date, precision)
            if not album_id or release_ts is None or release_ts < cutoff_ts:
                continue
            current_release_ids.append(album_id)
            inbox_rows.append(
                {
                    "artist_name": metadata.name,
                    "album_id": album_id,
                    "album_name": str(album.get("name", "")).strip(),
                    "album_type": str(album.get("album_type", "")).strip(),
                    "release_date": release_date,
                    "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                    "is_new_since_last_run": album_id not in seen_release_ids,
                }
            )

    inbox_rows.sort(key=lambda row: row["release_date"], reverse=True)
    payload = {
        "command": "release-inbox",
        "since_days": int(args.since_days),
        "artists": artists,
        "new_releases": [row for row in inbox_rows if row["is_new_since_last_run"]],
        "all_recent_releases": inbox_rows,
    }
    _write_state(state_path, {"release_ids": sorted(set(current_release_ids))})

    markdown_lines = [
        "# Spotify Release Inbox",
        "",
        f"- Artists tracked: `{len(artists)}`",
        f"- New releases since last run: `{len(payload['new_releases'])}`",
        "",
    ]
    for row in payload["new_releases"][:20]:
        markdown_lines.append(f"- {row['release_date']} | {row['artist_name']} | {row['album_name']}")

    stem = f"release_inbox_{state_key}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="release_inbox",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"release_inbox_json={json_path}")
    print(f"release_inbox_md={md_path}")
    print(f"new_releases={len(payload['new_releases'])}")
    return 0


def _handle_playlist_diff(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    playlist_id = parse_spotify_id(args.playlist, expected_kind="playlist")
    output_dir = Path(args.output_dir).expanduser().resolve()
    playlist_payload, track_rows, _artist_counter = _playlist_item_rows(
        client,
        playlist_id=playlist_id,
        market=str(args.spotify_market).upper(),
        limit=max(1, int(args.item_limit)),
    )
    snapshot = _playlist_snapshot(playlist_payload, track_rows)
    playlist_slug = _slugify(snapshot.get("name", playlist_id) or playlist_id)
    state_path = _playlist_state_path(output_dir, playlist_id)
    previous_state = _read_json_if_exists(state_path)
    diff = _playlist_diff(previous_state, snapshot)
    payload = {
        "command": "playlist-diff",
        "playlist_id": playlist_id,
        "playlist_name": snapshot.get("name"),
        "diff": diff,
        "current_snapshot": snapshot,
    }
    _write_state(state_path, snapshot)

    markdown_lines = [
        "# Spotify Playlist Diff",
        "",
        f"- Playlist: `{snapshot.get('name')}`",
        f"- Added tracks: `{len(diff['added_tracks'])}`",
        f"- Removed tracks: `{len(diff['removed_tracks'])}`",
        f"- Metadata changes: `{len(diff['metadata_changes'])}`",
        "",
    ]
    for row in diff["added_tracks"][:20]:
        markdown_lines.append(f"- Added: {row.get('track_name')} by {', '.join(row.get('artist_names', []))}")
    for row in diff["removed_tracks"][:20]:
        markdown_lines.append(f"- Removed: {row.get('track_name')} by {', '.join(row.get('artist_names', []))}")

    stem = f"playlist_diff_{playlist_slug}"
    json_path, md_path = _write_report(
        output_dir=output_dir,
        category="playlist_diff",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"playlist_diff_json={json_path}")
    print(f"playlist_diff_md={md_path}")
    print(f"playlist_added_tracks={len(diff['added_tracks'])}")
    return 0


def _handle_market_gap(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    explicit_tracks = _split_pipe_list(args.tracks)
    if explicit_tracks:
        track_rows = [{"spotify_track_uri": value, "track_name": value, "artist_name": ""} for value in explicit_tracks]
    else:
        history_df = _load_history_if_needed(args, logger)
        track_rows = _top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_n)
    all_markets = client.get_available_markets()
    all_market_set = set(all_markets)

    rows: list[dict[str, Any]] = []
    for track in track_rows:
        track_payload = client.get_track(str(track.get("spotify_track_uri", "")).strip(), market=str(args.spotify_market).upper())
        available_markets = {
            str(value).strip().upper()
            for value in track_payload.get("available_markets", [])
            if str(value).strip()
        }
        missing_markets = sorted(all_market_set.difference(available_markets))
        rows.append(
            {
                "track_name": str(track_payload.get("name", track.get("track_name", ""))).strip(),
                "artist_names": [
                    str(item.get("name", "")).strip()
                    for item in track_payload.get("artists", [])
                    if isinstance(item, dict) and str(item.get("name", "")).strip()
                ],
                "spotify_url": str(track_payload.get("external_urls", {}).get("spotify", "")).strip(),
                "available_markets_count": len(available_markets),
                "total_markets": len(all_markets),
                "availability_ratio": float(len(available_markets) / max(1, len(all_markets))),
                "missing_markets_sample": missing_markets[:20],
            }
        )

    payload = {
        "command": "market-gap",
        "tracks": rows,
        "all_markets_count": len(all_markets),
    }
    markdown_lines = [
        "# Spotify Market Gap Finder",
        "",
        f"- Global market count: `{len(all_markets)}`",
        f"- Tracks checked: `{len(rows)}`",
        "",
    ]
    for row in rows[:20]:
        markdown_lines.append(
            f"- {row['track_name']} by {', '.join(row['artist_names'])}: "
            f"`{row['availability_ratio']:.1%}` market coverage"
        )

    stem = "market_gap_global"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="market_gap",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"market_gap_json={json_path}")
    print(f"market_gap_md={md_path}")
    print(f"market_gap_tracks={len(rows)}")
    return 0


def _handle_playlist_archive(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    playlist_id = parse_spotify_id(args.playlist, expected_kind="playlist")
    playlist_payload, track_rows, artist_counter = _playlist_item_rows(
        client,
        playlist_id=playlist_id,
        market=str(args.spotify_market).upper(),
        limit=max(1, int(args.item_limit)),
    )
    snapshot = _playlist_snapshot(playlist_payload, track_rows)
    payload = {
        "command": "playlist-archive",
        "playlist_id": playlist_id,
        "snapshot": snapshot,
        "top_artists": [{"artist_name": name, "track_count": count} for name, count in artist_counter.most_common(20)],
    }
    markdown_lines = [
        "# Spotify Playlist Archive",
        "",
        f"- Playlist: `{snapshot.get('name')}`",
        f"- Snapshot ID: `{snapshot.get('snapshot_id')}`",
        f"- Followers: `{snapshot.get('followers_total')}`",
        f"- Image URLs captured: `{len(snapshot.get('image_urls', []))}`",
        f"- Tracks archived: `{len(snapshot.get('items', []))}`",
        "",
    ]
    for url in snapshot.get("image_urls", [])[:5]:
        markdown_lines.append(f"- Image URL: {url}")

    stem = f"playlist_archive_{_slugify(snapshot.get('name', playlist_id) or playlist_id)}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="playlist_archive",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"playlist_archive_json={json_path}")
    print(f"playlist_archive_md={md_path}")
    print(f"playlist_archive_tracks={len(snapshot.get('items', []))}")
    return 0


def _handle_catalog_crosswalk(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    explicit_tracks = _split_pipe_list(args.tracks)
    if explicit_tracks:
        track_rows = [{"spotify_track_uri": value, "track_name": value, "artist_name": ""} for value in explicit_tracks]
    else:
        history_df = _load_history_if_needed(args, logger)
        track_rows = _top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_n)

    rows: list[dict[str, Any]] = []
    for row in track_rows:
        track_payload = client.get_track(str(row.get("spotify_track_uri", "")).strip(), market=str(args.spotify_market).upper())
        album_id = str(track_payload.get("album", {}).get("id", "")).strip()
        album_payload = client.get_album(album_id, market=str(args.spotify_market).upper()) if album_id else {}
        track_ids = track_payload.get("external_ids", {}) if isinstance(track_payload.get("external_ids"), dict) else {}
        album_ids = album_payload.get("external_ids", {}) if isinstance(album_payload.get("external_ids"), dict) else {}
        rows.append(
            {
                "track_name": str(track_payload.get("name", row.get("track_name", ""))).strip(),
                "artist_names": [
                    str(item.get("name", "")).strip()
                    for item in track_payload.get("artists", [])
                    if isinstance(item, dict) and str(item.get("name", "")).strip()
                ],
                "spotify_track_id": str(track_payload.get("id", "")).strip(),
                "spotify_track_url": str(track_payload.get("external_urls", {}).get("spotify", "")).strip(),
                "isrc": str(track_ids.get("isrc", "")).strip() or None,
                "ean": str(album_ids.get("ean", "")).strip() or None,
                "upc": str(album_ids.get("upc", "")).strip() or None,
                "album_name": str(track_payload.get("album", {}).get("name", "")).strip(),
                "album_id": album_id,
                "album_url": str(track_payload.get("album", {}).get("external_urls", {}).get("spotify", "")).strip(),
            }
        )

    payload = {"command": "catalog-crosswalk", "tracks": rows}
    markdown_lines = [
        "# Spotify Catalog Crosswalk",
        "",
        f"- Tracks mapped: `{len(rows)}`",
        "",
    ]
    for row in rows[:20]:
        markdown_lines.append(
            f"- {row['track_name']} by {', '.join(row['artist_names'])}: "
            f"ISRC `{row['isrc']}`, UPC `{row['upc']}`, EAN `{row['ean']}`"
        )

    stem = "catalog_crosswalk_tracks"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="catalog_crosswalk",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"catalog_crosswalk_json={json_path}")
    print(f"catalog_crosswalk_md={md_path}")
    print(f"catalog_crosswalk_tracks={len(rows)}")
    return 0


def _handle_media_explorer(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    media_type = str(args.media_type)
    payload: dict[str, Any]
    if args.item_id:
        raw_id = str(args.item_id).strip()
        if media_type == "show":
            show_payload = client.get_show(raw_id, market=str(args.spotify_market).upper())
            episodes = client.get_show_episodes(raw_id, limit=max(1, int(args.limit)), market=str(args.spotify_market).upper())
            payload = {
                "command": "media-explorer",
                "media_type": media_type,
                "item": show_payload,
                "episodes": episodes,
            }
        elif media_type == "episode":
            payload = {
                "command": "media-explorer",
                "media_type": media_type,
                "item": client.get_episode(raw_id, market=str(args.spotify_market).upper()),
            }
        else:
            payload = {
                "command": "media-explorer",
                "media_type": media_type,
                "item": client.get_audiobook(raw_id, market=str(args.spotify_market).upper()),
            }
    else:
        query = str(args.query or "").strip()
        if not query:
            raise RuntimeError("media-explorer requires either --query or --item-id.")
        payload = {
            "command": "media-explorer",
            "media_type": media_type,
            "query": query,
            "results": client.search(
                query=query,
                item_types=[media_type],
                limit=max(1, int(args.limit)),
                market=str(args.spotify_market).upper(),
            ).get(media_type, []),
        }

    markdown_lines = [
        "# Spotify Media Explorer",
        "",
        f"- Media type: `{media_type}`",
    ]
    if "query" in payload:
        markdown_lines.append(f"- Query: `{payload['query']}`")
        markdown_lines.append(f"- Results: `{len(payload.get('results', []))}`")
    else:
        item_name = str(payload.get("item", {}).get("name", "")).strip()
        markdown_lines.append(f"- Item: `{item_name}`")
    markdown_lines.append("")

    stem = f"media_{media_type}_{_slugify(str(payload.get('query', payload.get('item', {}).get('name', 'item'))))}"
    json_path, md_path = _write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="media_explorer",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"media_explorer_json={json_path}")
    print(f"media_explorer_md={md_path}")
    print(f"media_type={media_type}")
    return 0


def _handle_creator_label_intelligence(
    args: argparse.Namespace,
    client: SpotifyPublicCatalogClient,
    logger: logging.Logger,
) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    recent_history = _recent_history(_load_history_if_needed(args, logger), int(args.lookback_days))
    artists = _resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
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
        "lookback_days": int(args.lookback_days),
        "related_limit": int(args.related_limit),
        "neighbor_k": int(args.neighbor_k),
        "release_limit": int(args.release_limit),
        "scene_count": int(args.scene_count) if args.scene_count else None,
        "multimodal_source": space_info,
        **intelligence_payload,
    }

    markdown_lines = [
        "# Creator And Label Intelligence Graph",
        "",
        f"- Market: `{payload['market']}`",
        f"- Seed artists: `{len(artists)}`",
        f"- Multimodal source: `{space_info['mode']}`",
        f"- Nodes: `{payload['graph_summary']['node_count']}`",
        f"- Scenes: `{payload['graph_summary']['scene_count']}`",
        f"- Opportunities: `{payload['graph_summary']['opportunity_count']}`",
        "",
        "## Artist Adjacency",
        "",
    ]
    for row in payload["artist_adjacency"][:10]:
        markdown_lines.append(
            f"- {row['source_artist']} -> {row['target_artist']}: hybrid `{row['hybrid_score']}`, "
            f"similarity `{row['embedding_similarity']}`, transition `{row['transition_share']}`"
        )
    markdown_lines.extend(["", "## Scene Map", ""])
    for row in payload["scenes"][:10]:
        markdown_lines.append(
            f"- {row['scene_name']}: artists `{row['artist_count']}`, seeds `{row['seed_count']}`, "
            f"genres `{', '.join(row['dominant_genres'][:3]) or 'n/a'}`, "
            f"labels `{', '.join(row['dominant_labels'][:3]) or 'n/a'}`"
        )
    markdown_lines.extend(["", "## Release Whitespace", ""])
    for row in payload["release_whitespace"][:10]:
        markdown_lines.append(
            f"- {row['artist_name']}: whitespace `{row['release_whitespace_score']}`, "
            f"latest `{row['latest_release_date']}`, labels `{', '.join(row['dominant_release_labels'][:3]) or 'n/a'}`"
        )
    markdown_lines.extend(["", "## Fan Migration", ""])
    for row in payload["fan_migration"][:10]:
        markdown_lines.append(
            f"- {row['source_artist']} -> {row['target_artist']}: count `{row['transition_count']}`, "
            f"share `{row['source_out_share']}`"
        )
    markdown_lines.extend(["", "## Long-Tail Opportunities", ""])
    for row in payload["opportunities"][:10]:
        markdown_lines.append(
            f"- {row['artist_name']}: score `{row['opportunity_score']}`, "
            f"reasons `{'; '.join(row['rationale'])}`"
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
    }
    print(f"creator_label_intelligence_json={json_path}")
    print(f"creator_label_intelligence_md={md_path}")
    for label, path in csv_paths.items():
        if path is not None:
            print(f"creator_label_intelligence_{label}_csv={path}")
    print(f"creator_label_intelligence_nodes={payload['graph_summary']['node_count']}")
    print(f"creator_label_intelligence_scenes={payload['graph_summary']['scene_count']}")
    print(f"creator_label_intelligence_opportunities={payload['graph_summary']['opportunity_count']}")
    return 0


def main() -> int:
    load_local_env()
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.public_insights")

    try:
        client = _build_client(args)
        if args.command == "explain-artists":
            return _handle_explain_artists(args, client, logger)
        if args.command == "release-tracker":
            return _handle_release_tracker(args, client, logger)
        if args.command == "market-check":
            return _handle_market_check(args, client, logger)
        if args.command == "discography":
            return _handle_discography(args, client, logger)
        if args.command == "playlist-view":
            return _handle_playlist_view(args, client, logger)
        if args.command == "discovery-search":
            return _handle_discovery_search(args, client, logger)
        if args.command == "catalog-linkouts":
            return _handle_catalog_linkouts(args, client, logger)
        if args.command == "artist-graph":
            return _handle_artist_graph(args, client, logger)
        if args.command == "cross-media-taste-graph":
            return _handle_cross_media_taste_graph(args, client, logger)
        if args.command == "release-inbox":
            return _handle_release_inbox(args, client, logger)
        if args.command == "playlist-diff":
            return _handle_playlist_diff(args, client, logger)
        if args.command == "market-gap":
            return _handle_market_gap(args, client, logger)
        if args.command == "playlist-archive":
            return _handle_playlist_archive(args, client, logger)
        if args.command == "catalog-crosswalk":
            return _handle_catalog_crosswalk(args, client, logger)
        if args.command == "media-explorer":
            return _handle_media_explorer(args, client, logger)
        if args.command == "creator-label-intelligence":
            return _handle_creator_label_intelligence(args, client, logger)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except (RuntimeError, SpotifyPublicCatalogError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
