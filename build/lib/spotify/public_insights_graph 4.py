from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .public_catalog import SpotifyPublicCatalogClient, SpotifyPublicCatalogError


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


def _first_nonempty_tuple_value(row: tuple[object, ...], index_by_column: dict[str, int], columns: tuple[str, ...]) -> str:
    for column in columns:
        column_idx = index_by_column.get(column)
        if column_idx is None:
            continue
        text = _clean_text(row[column_idx])
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
    index_by_column = {column: idx for idx, column in enumerate(recent.columns)}
    ts_idx = index_by_column.get("ts")
    ms_played_idx = index_by_column.get("ms_played")

    for row in recent.itertuples(index=False, name=None):
        artist_name = _first_nonempty_tuple_value(row, index_by_column, ("master_metadata_album_artist_name",))
        track_name = _first_nonempty_tuple_value(row, index_by_column, ("master_metadata_track_name",))
        track_uri = _first_nonempty_tuple_value(row, index_by_column, ("spotify_track_uri",))

        show_name = _first_nonempty_tuple_value(row, index_by_column, ("episode_show_name", "show_name"))
        show_uri = _first_nonempty_tuple_value(row, index_by_column, ("spotify_show_uri", "show_uri"))
        episode_name = _first_nonempty_tuple_value(row, index_by_column, ("episode_name", "episode_title"))
        episode_uri = _first_nonempty_tuple_value(row, index_by_column, ("spotify_episode_uri", "episode_uri"))

        audiobook_name = _first_nonempty_tuple_value(row, index_by_column, ("audiobook_title", "audiobook_name"))
        audiobook_uri = _first_nonempty_tuple_value(row, index_by_column, ("audiobook_uri", "spotify_audiobook_uri"))
        chapter_name = _first_nonempty_tuple_value(
            row,
            index_by_column,
            ("audiobook_chapter_title", "chapter_name", "chapter_title"),
        )
        chapter_uri = _first_nonempty_tuple_value(row, index_by_column, ("audiobook_chapter_uri", "chapter_uri"))

        event: dict[str, Any] | None = None
        if audiobook_name:
            item_name = chapter_name or episode_name or audiobook_name
            item_uri = chapter_uri or episode_uri or audiobook_uri
            item_type = "chapter" if item_name and item_name.casefold() != audiobook_name.casefold() else "audiobook"
            event = {
                "ts": row[ts_idx] if ts_idx is not None else None,
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
                "ts": row[ts_idx] if ts_idx is not None else None,
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
                "ts": row[ts_idx] if ts_idx is not None else None,
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
        ms_played = pd.to_numeric(row[ms_played_idx], errors="coerce") if ms_played_idx is not None else float("nan")
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
    for row in item_rows.head(max(1, int(limit))).itertuples(index=False):
        rows.append(
            {
                "item_key": str(row.item_key).strip(),
                "item_name": str(row.item_name).strip(),
                "item_type": str(row.item_type).strip(),
                "play_count": int(row.play_count or 0),
                "total_ms_played": int(float(row.total_ms_played or 0.0)),
                "last_seen": pd.Timestamp(row.last_seen).isoformat() if pd.notna(row.last_seen) else None,
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
                "source_key": str(row.source_key).strip(),
                "source_name": str(row.source_name).strip(),
                "source_type": str(row.source_type).strip(),
                "source_media_family": str(row.source_media_family).strip(),
                "target_key": str(row.target_key).strip(),
                "target_name": str(row.target_name).strip(),
                "target_type": str(row.target_type).strip(),
                "target_media_family": str(row.target_media_family).strip(),
                "cross_media": bool(row.cross_media),
                "transition_count": int(row.transition_count or 0),
                "target_ms_played": int(float(row.target_ms_played or 0.0)),
            }
            for row in edge_frame.itertuples(index=False)
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

    name_counts = (
        frame.groupby(
            ["spotify_track_uri", "master_metadata_track_name", "master_metadata_album_artist_name"],
            dropna=False,
            sort=False,
        )
        .size()
        .rename("name_plays")
        .reset_index()
    )
    grouped = (
        name_counts.groupby("spotify_track_uri", dropna=False, sort=False)["name_plays"]
        .sum()
        .rename("plays")
        .reset_index()
        .merge(
            name_counts.sort_values(
                ["spotify_track_uri", "name_plays", "master_metadata_track_name", "master_metadata_album_artist_name"],
                ascending=[True, False, True, True],
                kind="stable",
            )
            .drop_duplicates(subset=["spotify_track_uri"], keep="first")
            .rename(
                columns={
                    "master_metadata_track_name": "track_name",
                    "master_metadata_album_artist_name": "artist_name",
                }
            )[["spotify_track_uri", "track_name", "artist_name"]],
            on="spotify_track_uri",
            how="left",
        )
        .sort_values(["plays", "track_name", "artist_name"], ascending=[False, True, True], kind="stable")
        .reset_index(drop=True)
    )
    return grouped.head(max(1, int(limit))).to_dict(orient="records")


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


__all__ = [
    "_cross_media_catalog_bridges",
    "_cross_media_graph_payload",
    "_cross_media_history_frame",
    "_history_frame",
    "_playlist_diff",
    "_playlist_snapshot",
    "_recent_history",
    "_release_state_rows",
    "_seed_nodes_for_bridges",
    "_top_artists_from_history",
    "_top_tracks_from_history",
]
