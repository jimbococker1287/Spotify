from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .public_catalog import SpotifyPublicCatalogError, parse_spotify_id
from .public_insights_index import build_public_insights_index

PublicInsightsHandler = Callable[[argparse.Namespace, Any, logging.Logger], int]


@dataclass(frozen=True)
class PublicInsightsHandlerDeps:
    split_csv_list: Callable[[str], list[str]]
    split_pipe_list: Callable[[str | None], list[str]]
    resolve_artists: Callable[..., list[str]]
    load_history_if_needed: Callable[[argparse.Namespace, logging.Logger], pd.DataFrame]
    top_artists_from_history: Callable[..., list[str]]
    top_tracks_from_history: Callable[..., list[dict[str, Any]]]
    parse_release_date: Callable[[str, str], pd.Timestamp | None]
    dedupe_album_rows: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    write_report: Callable[..., tuple[Path, Path]]
    read_json_if_exists: Callable[[Path], dict[str, Any] | None]
    write_state: Callable[[Path, dict[str, Any]], None]
    playlist_state_path: Callable[[Path, str], Path]
    release_state_path: Callable[[Path, str], Path]
    playlist_snapshot: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]
    playlist_diff: Callable[[dict[str, Any] | None, dict[str, Any]], dict[str, Any]]
    cross_media_history_frame: Callable[..., pd.DataFrame]
    cross_media_graph_payload: Callable[..., dict[str, Any]]
    seed_nodes_for_bridges: Callable[..., list[dict[str, Any]]]
    cross_media_catalog_bridges: Callable[..., list[dict[str, Any]]]
    slugify: Callable[[str], str]


def _as_of_timestamp(args: argparse.Namespace) -> pd.Timestamp:
    raw_value = str(getattr(args, "as_of_date", "") or "").strip()
    if not raw_value:
        return pd.Timestamp.now(tz="UTC")
    timestamp = pd.Timestamp(raw_value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _playlist_item_rows(
    client: Any,
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
                "track_id": str(track.get("id", "")).strip(),
                "track_uri": str(track.get("uri", "")).strip(),
                "track_name": str(track.get("name", "")).strip(),
                "artist_names": artists,
                "album_name": str(track.get("album", {}).get("name", "")).strip(),
                "album_id": str(track.get("album", {}).get("id", "")).strip(),
                "duration_ms": int(track.get("duration_ms", 0) or 0),
                "explicit": bool(track.get("explicit", False)),
                "popularity": track.get("popularity"),
                "spotify_url": str(track.get("external_urls", {}).get("spotify", "")).strip(),
                "available_markets_count": (
                    len(track.get("available_markets", [])) if isinstance(track.get("available_markets"), list) else None
                ),
            }
        )
    return playlist_payload, track_rows, artist_counter


def _first_public_image_url(images: Any) -> str | None:
    if not isinstance(images, list):
        return None
    for image in images:
        if not isinstance(image, dict):
            continue
        url = str(image.get("url", "")).strip()
        if url:
            return url
    return None


def _artist_names_from_items(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [
        str(item.get("name", "")).strip()
        for item in items
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    ]


def _normalized_catalog_text(value: Any) -> str:
    return "".join(char.lower() for char in str(value or "") if char.isalnum())


def _spotify_track_id_from_history_uri(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return parse_spotify_id(raw, expected_kind="track")
    except ValueError:
        return raw


def _history_catalog_coverage_index(history_df: pd.DataFrame, *, lookback_days: int) -> dict[str, Any]:
    if history_df.empty:
        return {
            "events_analyzed": 0,
            "track_ids": Counter(),
            "artist_track_names": Counter(),
            "artist_album_names": Counter(),
        }

    frame = history_df.copy()
    if "ts" in frame.columns:
        timestamps = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        if timestamps.notna().any():
            cutoff = timestamps.max() - pd.Timedelta(days=max(1, int(lookback_days)))
            frame = frame.loc[timestamps >= cutoff].copy()

    track_ids: Counter[str] = Counter()
    artist_track_names: Counter[tuple[str, str]] = Counter()
    artist_album_names: Counter[tuple[str, str]] = Counter()
    artist_col = "master_metadata_album_artist_name"
    track_col = "master_metadata_track_name"
    album_col = "master_metadata_album_album_name"

    for row in frame.to_dict("records"):
        artist_name = _normalized_catalog_text(row.get(artist_col))
        track_name = _normalized_catalog_text(row.get(track_col))
        album_name = _normalized_catalog_text(row.get(album_col))
        track_id = _spotify_track_id_from_history_uri(row.get("spotify_track_uri"))
        if track_id:
            track_ids[track_id] += 1
        if artist_name and track_name:
            artist_track_names[(artist_name, track_name)] += 1
        if artist_name and album_name:
            artist_album_names[(artist_name, album_name)] += 1

    return {
        "events_analyzed": int(len(frame)),
        "track_ids": track_ids,
        "artist_track_names": artist_track_names,
        "artist_album_names": artist_album_names,
    }


def _catalog_track_history_match(
    track: dict[str, Any],
    *,
    artist_names: list[str],
    coverage_index: dict[str, Any],
) -> tuple[bool, int, str]:
    track_id = str(track.get("id", "")).strip()
    track_ids = coverage_index["track_ids"]
    if track_id and track_id in track_ids:
        return True, int(track_ids[track_id]), "spotify_track_id"

    track_name = _normalized_catalog_text(track.get("name"))
    artist_track_names = coverage_index["artist_track_names"]
    for artist_name in artist_names:
        key = (_normalized_catalog_text(artist_name), track_name)
        if key in artist_track_names:
            return True, int(artist_track_names[key]), "artist_track_name"
    return False, 0, "none"


def _safe_public_catalog_call(
    warnings: list[dict[str, Any]],
    *,
    operation: str,
    default: Any,
    func: Callable[..., Any],
    call_args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> Any:
    try:
        return func(*call_args, **kwargs)
    except (SpotifyPublicCatalogError, RuntimeError, ValueError) as exc:
        warnings.append(
            {
                "operation": operation,
                "status_code": getattr(exc, "status_code", None),
                "message": str(exc),
                "degraded": True,
            }
        )
        return default


def _load_history_for_optional_coverage(
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    try:
        return deps.load_history_if_needed(args, logger), []
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return pd.DataFrame(), [{"operation": "load-history", "message": str(exc), "degraded": True}]


def _release_sort_timestamp(
    release_date: str,
    release_date_precision: str,
    *,
    deps: PublicInsightsHandlerDeps,
) -> pd.Timestamp:
    return deps.parse_release_date(release_date, release_date_precision) or pd.Timestamp(0, tz="UTC")


def _report_recommendation(action: str, reason: str, *, priority: str = "medium") -> dict[str, str]:
    return {"action": action, "reason": reason, "priority": priority}


def _has_rate_limit_warning(warnings: list[dict[str, Any]]) -> bool:
    return any(warning.get("status_code") == 429 for warning in warnings)


def _rate_limit_recommendation(command_hint: str) -> dict[str, str]:
    return _report_recommendation(
        f"Wait 1-2 minutes, then rerun `{command_hint}`.",
        "Spotify returned HTTP 429 after retries, so some public catalog pages were skipped.",
        priority="high",
    )


def _handle_explain_artists(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    artists = deps.resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
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

    stem = f"artist_explainer_{deps.slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = deps.write_report(
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


def _handle_release_tracker(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    artists = deps.resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    include_groups = ",".join(deps.split_csv_list(args.include_groups))
    as_of_ts = _as_of_timestamp(args)
    cutoff_ts = as_of_ts - pd.Timedelta(days=max(1, int(args.since_days)))

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
        deduped = deps.dedupe_album_rows(albums)
        recent_releases: list[dict[str, Any]] = []
        for album in deduped:
            release_date = str(album.get("release_date", "")).strip()
            precision = str(album.get("release_date_precision", "day")).strip()
            release_ts = deps.parse_release_date(release_date, precision)
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
        recent_releases.sort(
            key=lambda row: (deps.parse_release_date(row["release_date"], row["release_date_precision"]) or cutoff_ts),
            reverse=True,
        )
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
        "as_of_date": as_of_ts.date().isoformat(),
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

    stem = f"release_tracker_{int(args.since_days)}d_{deps.slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = deps.write_report(
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


def _handle_artist_top_tracks(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    artists = deps.resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    market = str(args.spotify_market).upper()
    artist_rows: list[dict[str, Any]] = []
    total_tracks = 0
    warnings: list[dict[str, Any]] = []

    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            artist_rows.append({"queried_name": artist_name, "matched": False, "tracks": []})
            continue

        tracks = _safe_public_catalog_call(
            warnings,
            operation=f"artist-top-tracks:{metadata.spotify_id}",
            default=[],
            func=client.get_artist_top_tracks,
            call_args=(metadata.spotify_id,),
            market=market,
            limit=max(1, int(args.track_limit)),
        )
        formatted_tracks: list[dict[str, Any]] = []
        for rank, track in enumerate(tracks, start=1):
            album = track.get("album", {}) if isinstance(track.get("album"), dict) else {}
            formatted_tracks.append(
                {
                    "rank": rank,
                    "track_id": str(track.get("id", "")).strip(),
                    "track_name": str(track.get("name", "")).strip(),
                    "spotify_url": str(track.get("external_urls", {}).get("spotify", "")).strip(),
                    "album_id": str(album.get("id", "")).strip(),
                    "album_name": str(album.get("name", "")).strip(),
                    "album_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                    "album_image_url": _first_public_image_url(album.get("images")),
                    "artist_names": _artist_names_from_items(track.get("artists")),
                    "duration_ms": track.get("duration_ms"),
                    "explicit": bool(track.get("explicit", False)),
                    "popularity": track.get("popularity"),
                }
            )

        total_tracks += len(formatted_tracks)
        artist_rows.append(
            {
                "queried_name": artist_name,
                "matched": True,
                "artist_id": metadata.spotify_id,
                "artist_name": metadata.name,
                "spotify_url": metadata.spotify_url,
                "genres": metadata.genres,
                "popularity": metadata.popularity,
                "followers_total": metadata.followers_total,
                "tracks": formatted_tracks,
            }
        )

    payload = {
        "command": "artist-top-tracks",
        "market": market,
        "lookback_days": int(args.lookback_days),
        "artists": artist_rows,
        "warnings": warnings,
        "policy_note": "Public Spotify track metadata is exported for display/link-out only.",
    }
    markdown_lines = [
        "# Spotify Artist Top Tracks",
        "",
        f"- Market: `{market}`",
        f"- Artists reviewed: `{len(artist_rows)}`",
        f"- Tracks exported: `{total_tracks}`",
        f"- API warnings: `{len(warnings)}`",
        "",
    ]
    for artist in artist_rows:
        if not artist.get("matched"):
            markdown_lines.append(f"- {artist['queried_name']}: no Spotify public match found")
            continue
        markdown_lines.append(f"## {artist['artist_name']}")
        markdown_lines.append("")
        for track in artist["tracks"][: max(1, int(args.track_limit))]:
            markdown_lines.append(
                f"- {track['rank']}. {track['track_name']} | {track['album_name']} | "
                f"popularity `{track['popularity']}`"
            )
        markdown_lines.append("")

    stem = f"artist_top_tracks_{deps.slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = deps.write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="artist_top_tracks",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"artist_top_tracks_json={json_path}")
    print(f"artist_top_tracks_md={md_path}")
    print(f"artist_top_tracks={total_tracks}")
    return 0


def _handle_new_releases(
    args: argparse.Namespace,
    client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    market = str(args.spotify_market).upper()
    warnings: list[dict[str, Any]] = []
    albums = _safe_public_catalog_call(
        warnings,
        operation="new-releases",
        default=[],
        func=client.get_new_releases,
        market=market,
        limit=max(1, int(args.limit)),
    )
    rows: list[dict[str, Any]] = []
    for album in albums:
        rows.append(
            {
                "album_id": str(album.get("id", "")).strip(),
                "album_name": str(album.get("name", "")).strip(),
                "album_type": str(album.get("album_type", "")).strip(),
                "release_date": str(album.get("release_date", "")).strip(),
                "release_date_precision": str(album.get("release_date_precision", "")).strip(),
                "total_tracks": album.get("total_tracks"),
                "artist_names": _artist_names_from_items(album.get("artists")),
                "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                "image_url": _first_public_image_url(album.get("images")),
            }
        )

    payload = {
        "command": "new-releases",
        "market": market,
        "albums": rows,
        "warnings": warnings,
        "policy_note": "Public Spotify album metadata is exported for display/link-out only.",
    }
    markdown_lines = [
        "# Spotify New Releases",
        "",
        f"- Market: `{market}`",
        f"- Albums exported: `{len(rows)}`",
        f"- API warnings: `{len(warnings)}`",
        "",
    ]
    for row in rows:
        markdown_lines.append(
            f"- {row['release_date']} | {row['album_name']} by {', '.join(row['artist_names']) or 'n/a'} "
            f"({row['album_type'] or 'album'})"
        )

    stem = f"new_releases_{market.lower()}"
    json_path, md_path = deps.write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="new_releases",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"new_releases_json={json_path}")
    print(f"new_releases_md={md_path}")
    print(f"new_releases={len(rows)}")
    return 0


def _handle_market_check(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    explicit_tracks = deps.split_pipe_list(args.tracks)
    if explicit_tracks:
        track_rows = [{"spotify_track_uri": value, "track_name": value, "artist_name": ""} for value in explicit_tracks]
    else:
        history_df = deps.load_history_if_needed(args, logger)
        track_rows = deps.top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_n)
    markets = [market.strip().upper() for market in deps.split_csv_list(args.markets)]
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

    stem = f"market_check_{deps.slugify('-'.join(markets))}"
    json_path, md_path = deps.write_report(
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


def _handle_discography(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    artists = deps.resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    include_groups = ",".join(deps.split_csv_list(args.include_groups))
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
        deduped = deps.dedupe_album_rows(albums)
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
            key=lambda row: deps.parse_release_date(row["release_date"], row["release_date_precision"]) or pd.Timestamp(0, tz="UTC")
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

    stem = f"discography_{deps.slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = deps.write_report(
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


def _handle_artist_catalog_completeness(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    history_df, history_warnings = _load_history_for_optional_coverage(args, logger, deps=deps)
    explicit_artists = deps.split_pipe_list(getattr(args, "artists", None))
    artists = (
        explicit_artists
        if explicit_artists
        else deps.top_artists_from_history(
            history_df,
            lookback_days=int(args.lookback_days),
            limit=max(1, int(args.top_n)),
        )
    )
    coverage_index = _history_catalog_coverage_index(history_df, lookback_days=int(args.lookback_days))
    include_groups = ",".join(deps.split_csv_list(args.include_groups))
    market = str(args.spotify_market).upper()
    album_limit = max(1, int(args.album_limit))
    track_limit = max(1, int(args.track_limit))
    gap_limit = max(1, int(args.gap_limit))

    artist_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = list(history_warnings)
    for artist_name in artists:
        metadata = client.search_artist(artist_name)
        if metadata is None:
            artist_rows.append({"queried_name": artist_name, "matched": False, "albums": []})
            continue

        raw_albums = _safe_public_catalog_call(
            warnings,
            operation=f"artist-albums:{metadata.spotify_id}",
            default=[],
            func=client.get_artist_albums,
            call_args=(metadata.spotify_id,),
            include_groups=include_groups,
            limit=max(album_limit, album_limit * 2),
            market=market,
        )
        albums = sorted(
            deps.dedupe_album_rows(raw_albums),
            key=lambda album: _release_sort_timestamp(
                str(album.get("release_date", "")).strip(),
                str(album.get("release_date_precision", "day")).strip(),
                deps=deps,
            ),
            reverse=True,
        )[:album_limit]
        album_rows: list[dict[str, Any]] = []
        loaded_track_count = 0
        listened_track_count = 0
        play_count = 0

        for album in albums:
            album_id = str(album.get("id", "")).strip()
            album_artist_names = _artist_names_from_items(album.get("artists")) or [metadata.name]
            album_name = str(album.get("name", "")).strip()
            tracks = (
                _safe_public_catalog_call(
                    warnings,
                    operation=f"album-tracks:{album_id}",
                    default=[],
                    func=client.get_album_tracks,
                    call_args=(album_id,),
                    limit=track_limit,
                    market=market,
                )
                if album_id
                else []
            )
            track_rows: list[dict[str, Any]] = []
            album_listened_tracks = 0
            album_play_count = 0

            for track in tracks:
                track_artist_names = _artist_names_from_items(track.get("artists")) or album_artist_names
                listened, track_plays, match_source = _catalog_track_history_match(
                    track,
                    artist_names=track_artist_names,
                    coverage_index=coverage_index,
                )
                album_listened_tracks += int(listened)
                album_play_count += int(track_plays)
                track_rows.append(
                    {
                        "track_id": str(track.get("id", "")).strip(),
                        "track_number": track.get("track_number"),
                        "track_name": str(track.get("name", "")).strip(),
                        "artist_names": track_artist_names,
                        "spotify_url": str(track.get("external_urls", {}).get("spotify", "")).strip(),
                        "duration_ms": track.get("duration_ms"),
                        "explicit": bool(track.get("explicit", False)),
                        "listened": listened,
                        "history_plays": track_plays,
                        "match_source": match_source,
                    }
                )

            album_name_key_hits = 0
            artist_album_names = coverage_index["artist_album_names"]
            for album_artist_name in album_artist_names:
                album_key = (_normalized_catalog_text(album_artist_name), _normalized_catalog_text(album_name))
                album_name_key_hits += int(artist_album_names.get(album_key, 0))

            loaded_track_count += len(track_rows)
            listened_track_count += album_listened_tracks
            play_count += album_play_count
            album_track_count = len(track_rows)
            album_rows.append(
                {
                    "album_id": album_id,
                    "album_name": album_name,
                    "album_type": str(album.get("album_type", "")).strip(),
                    "release_date": str(album.get("release_date", "")).strip(),
                    "release_date_precision": str(album.get("release_date_precision", "")).strip(),
                    "total_tracks": album.get("total_tracks"),
                    "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                    "image_url": _first_public_image_url(album.get("images")),
                    "artist_names": album_artist_names,
                    "tracks_loaded": album_track_count,
                    "tracks_listened": album_listened_tracks,
                    "coverage_ratio": float(album_listened_tracks / album_track_count) if album_track_count else 0.0,
                    "history_plays": album_play_count,
                    "album_name_history_hits": album_name_key_hits,
                    "heard": bool(album_listened_tracks or album_name_key_hits),
                    "priority_gap_score": float(1.0 - (album_listened_tracks / album_track_count)) if album_track_count else 0.0,
                    "missing_tracks": [row for row in track_rows if not row["listened"]][:gap_limit],
                    "tracks": track_rows,
                }
            )

        missing_albums = sorted(
            [row for row in album_rows if not row["heard"]],
            key=lambda row: (
                float(row.get("priority_gap_score", 0.0)),
                _release_sort_timestamp(
                    str(row.get("release_date", "")).strip(),
                    str(row.get("release_date_precision", "day")).strip(),
                    deps=deps,
                ),
            ),
            reverse=True,
        )[:gap_limit]
        recommendations: list[dict[str, str]] = []
        coverage_ratio = float(listened_track_count / loaded_track_count) if loaded_track_count else 0.0
        if missing_albums:
            recommendations.append(
                _report_recommendation(
                    f"Start with `{missing_albums[0]['album_name']}`.",
                    "It is a fully unheard public-catalog release for a selected artist.",
                    priority="high",
                )
            )
        if coverage_ratio < 0.5 and loaded_track_count:
            recommendations.append(
                _report_recommendation(
                    "Use this artist for a focused catalog catch-up session.",
                    f"Local history only covers {coverage_ratio:.1%} of loaded catalog tracks.",
                    priority="medium",
                )
            )
        artist_rows.append(
            {
                "queried_name": artist_name,
                "matched": True,
                "artist_id": metadata.spotify_id,
                "artist_name": metadata.name,
                "spotify_url": metadata.spotify_url,
                "genres": metadata.genres,
                "popularity": metadata.popularity,
                "followers_total": metadata.followers_total,
                "albums_loaded": len(album_rows),
                "catalog_tracks_loaded": loaded_track_count,
                "catalog_tracks_listened": listened_track_count,
                "coverage_ratio": coverage_ratio,
                "history_plays": play_count,
                "recommendations": recommendations,
                "missing_albums": missing_albums,
                "missing_tracks": [
                    track
                    for album_row in album_rows
                    for track in album_row.get("missing_tracks", [])
                ][:gap_limit],
                "albums": album_rows,
            }
        )

    run_recommendations: list[dict[str, str]] = []
    if _has_rate_limit_warning(warnings):
        run_recommendations.append(_rate_limit_recommendation("artist-catalog-completeness --top-n 3"))

    payload = {
        "command": "artist-catalog-completeness",
        "market": market,
        "lookback_days": int(args.lookback_days),
        "include_groups": include_groups,
        "history_events_analyzed": coverage_index["events_analyzed"],
        "artists": artist_rows,
        "recommendations": run_recommendations,
        "warnings": warnings,
        "policy_note": (
            "Spotify public catalog metadata is compared against local export history for reporting only. "
            "Do not train recommendation models on public Spotify content."
        ),
    }
    markdown_lines = [
        "# Spotify Artist Catalog Completeness",
        "",
        f"- Market: `{market}`",
        f"- History window: `{int(args.lookback_days)}` days",
        f"- History events analyzed: `{coverage_index['events_analyzed']}`",
        f"- Artists audited: `{len(artist_rows)}`",
        f"- API warnings: `{len(warnings)}`",
        "",
    ]
    if run_recommendations:
        markdown_lines.extend(["## Run Recommendations", ""])
        for recommendation in run_recommendations:
            markdown_lines.append(
                f"- [{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}"
            )
        markdown_lines.append("")
    for artist in artist_rows:
        if not artist.get("matched"):
            markdown_lines.append(f"- {artist['queried_name']}: no Spotify public match found")
            continue
        markdown_lines.append(f"## {artist['artist_name']}")
        markdown_lines.append("")
        markdown_lines.append(
            f"- Catalog coverage: `{float(artist['coverage_ratio']):.1%}` "
            f"({artist['catalog_tracks_listened']}/{artist['catalog_tracks_loaded']} loaded tracks)"
        )
        markdown_lines.append(f"- Albums loaded: `{artist['albums_loaded']}`")
        markdown_lines.append(f"- Local history plays matched: `{artist['history_plays']}`")
        if artist.get("recommendations"):
            markdown_lines.append("- Recommendations:")
            for recommendation in artist["recommendations"]:
                markdown_lines.append(
                    f"  - [{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}"
                )
        if artist["missing_albums"]:
            markdown_lines.append("- Highest-priority unheard albums:")
            for album in artist["missing_albums"][:gap_limit]:
                markdown_lines.append(
                    f"  - {album['release_date']} | {album['album_type']} | {album['album_name']} "
                    f"({album['tracks_loaded']} tracks loaded)"
                )
        if artist["missing_tracks"]:
            markdown_lines.append("- Example missing tracks:")
            for track in artist["missing_tracks"][:gap_limit]:
                markdown_lines.append(f"  - {track['track_name']} by {', '.join(track['artist_names']) or 'n/a'}")
        markdown_lines.append("")

    stem = f"artist_catalog_completeness_{deps.slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = deps.write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="artist_catalog_completeness",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"artist_catalog_completeness_json={json_path}")
    print(f"artist_catalog_completeness_md={md_path}")
    print(f"artists_audited={len(artist_rows)}")
    return 0


def _handle_playlist_view(
    args: argparse.Namespace,
    client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
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

    stem = f"playlist_{deps.slugify(payload['playlist']['name'] or playlist_id)}"
    json_path, md_path = deps.write_report(
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


def _handle_public_insights_summary(
    args: argparse.Namespace,
    _client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload, markdown_lines = build_public_insights_index(
        output_dir,
        category=str(getattr(args, "category", "") or "").strip() or None,
        max_reports=max(1, int(getattr(args, "max_reports", 25))),
    )
    json_path, md_path = deps.write_report(
        output_dir=output_dir,
        category="summary",
        stem="public_insights_summary",
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"public_insights_summary_json={json_path}")
    print(f"public_insights_summary_md={md_path}")
    print(f"public_insights_reports_indexed={payload['reports_indexed']}")
    return 0


def _handle_playlist_intelligence(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    playlist_id = parse_spotify_id(args.playlist, expected_kind="playlist")
    market = str(args.spotify_market).upper()
    output_dir = Path(args.output_dir).expanduser().resolve()
    warnings: list[dict[str, Any]] = []
    playlist_payload, track_rows, artist_counter = _playlist_item_rows(
        client,
        playlist_id=playlist_id,
        market=market,
        limit=max(1, int(args.item_limit)),
    )
    history_df, history_warnings = _load_history_for_optional_coverage(args, logger, deps=deps)
    warnings.extend(history_warnings)
    coverage_index = _history_catalog_coverage_index(history_df, lookback_days=int(args.lookback_days))
    favorite_artists = deps.top_artists_from_history(
        history_df,
        lookback_days=int(args.lookback_days),
        limit=max(1, int(args.top_n)),
    )
    favorite_artist_set = {_normalized_catalog_text(name) for name in favorite_artists}
    playlist_artist_set = {_normalized_catalog_text(name) for name in artist_counter}

    duplicate_counter: Counter[str] = Counter()
    duplicate_labels: dict[str, str] = {}
    enriched_tracks: list[dict[str, Any]] = []
    matched_tracks = 0
    history_plays = 0
    explicit_count = 0
    total_duration_ms = 0
    favorite_artist_track_count = 0

    for row in track_rows:
        artist_names = list(row.get("artist_names", []))
        synthetic_key = str(row.get("track_id", "")).strip() or "|".join(
            [_normalized_catalog_text(row.get("track_name")), *[_normalized_catalog_text(name) for name in artist_names]]
        )
        duplicate_counter[synthetic_key] += 1
        duplicate_labels[synthetic_key] = f"{row.get('track_name')} by {', '.join(artist_names) or 'n/a'}"
        listened, plays, match_source = _catalog_track_history_match(
            {"id": row.get("track_id"), "name": row.get("track_name")},
            artist_names=artist_names,
            coverage_index=coverage_index,
        )
        matched_tracks += int(listened)
        history_plays += int(plays)
        explicit_count += int(bool(row.get("explicit", False)))
        total_duration_ms += int(row.get("duration_ms", 0) or 0)
        is_favorite_artist = any(_normalized_catalog_text(name) in favorite_artist_set for name in artist_names)
        favorite_artist_track_count += int(is_favorite_artist)
        enriched_tracks.append(
            {
                **row,
                "listened": listened,
                "history_plays": plays,
                "match_source": match_source,
                "is_favorite_artist": is_favorite_artist,
            }
        )

    duplicate_tracks = [
        {"track_key": key, "track_label": duplicate_labels.get(key, key), "count": count}
        for key, count in duplicate_counter.items()
        if count > 1
    ]
    missing_favorite_artists = [
        artist_name
        for artist_name in favorite_artists
        if _normalized_catalog_text(artist_name) not in playlist_artist_set
    ]
    track_count = len(enriched_tracks)
    playlist_info = {
        "playlist_id": playlist_id,
        "name": str(playlist_payload.get("name", "")).strip(),
        "description": str(playlist_payload.get("description", "")).strip(),
        "owner_name": str(playlist_payload.get("owner", {}).get("display_name", "")).strip(),
        "followers_total": int(playlist_payload.get("followers", {}).get("total", 0) or 0),
        "spotify_url": str(playlist_payload.get("external_urls", {}).get("spotify", "")).strip(),
        "tracks_total": int(playlist_payload.get("tracks", {}).get("total", 0) or 0),
    }
    summary = {
        "tracks_loaded": track_count,
        "local_overlap_tracks": matched_tracks,
        "local_overlap_ratio": float(matched_tracks / track_count) if track_count else 0.0,
        "history_plays": history_plays,
        "favorite_artist_tracks": favorite_artist_track_count,
        "favorite_artist_ratio": float(favorite_artist_track_count / track_count) if track_count else 0.0,
        "unique_artists": len(artist_counter),
        "duplicate_track_groups": len(duplicate_tracks),
        "explicit_tracks": explicit_count,
        "duration_minutes": round(total_duration_ms / 60000.0, 2) if total_duration_ms else 0.0,
    }
    recommendations: list[dict[str, str]] = []
    if missing_favorite_artists:
        recommendations.append(
            _report_recommendation(
                f"Add a track from `{missing_favorite_artists[0]}`.",
                "This favorite artist is absent from the playlist.",
                priority="high",
            )
        )
    if duplicate_tracks:
        recommendations.append(
            _report_recommendation(
                f"Remove duplicate entry `{duplicate_tracks[0]['track_label']}`.",
                "Duplicate tracks make the playlist less diverse.",
                priority="medium",
            )
        )
    if summary["local_overlap_ratio"] < 0.25 and track_count:
        recommendations.append(
            _report_recommendation(
                "Position this playlist as discovery-heavy.",
                "Most tracks are not present in local listening history.",
                priority="low",
            )
        )
    elif summary["local_overlap_ratio"] > 0.75 and track_count:
        recommendations.append(
            _report_recommendation(
                "Use this playlist as a comfort/high-confidence set.",
                "Most tracks already overlap with local listening history.",
                priority="low",
            )
        )
    payload = {
        "command": "playlist-intelligence",
        "market": market,
        "lookback_days": int(args.lookback_days),
        "playlist": playlist_info,
        "summary": summary,
        "top_playlist_artists": [{"artist_name": name, "track_count": count} for name, count in artist_counter.most_common(20)],
        "missing_favorite_artists": missing_favorite_artists,
        "duplicate_tracks": duplicate_tracks,
        "recommendations": recommendations,
        "items": enriched_tracks,
        "warnings": warnings,
        "policy_note": "Playlist public metadata is compared against local export history for reporting only.",
    }
    markdown_lines = [
        "# Spotify Playlist Intelligence",
        "",
        f"- Playlist: `{playlist_info['name']}`",
        f"- Owner: `{playlist_info['owner_name']}`",
        f"- Tracks loaded: `{summary['tracks_loaded']}`",
        f"- Local overlap: `{summary['local_overlap_ratio']:.1%}`",
        f"- Favorite-artist ratio: `{summary['favorite_artist_ratio']:.1%}`",
        f"- Duplicate track groups: `{summary['duplicate_track_groups']}`",
        f"- Missing favorite artists: `{len(missing_favorite_artists)}`",
        f"- API warnings: `{len(warnings)}`",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in recommendations:
        markdown_lines.append(f"- [{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}")
    if not recommendations:
        markdown_lines.append("- No immediate cleanup recommendations.")
    markdown_lines.extend([
        "",
        "## Missing Favorite Artists",
        "",
    ])
    for artist_name in missing_favorite_artists[:20]:
        markdown_lines.append(f"- {artist_name}")
    if not missing_favorite_artists:
        markdown_lines.append("- None")
    markdown_lines.extend(["", "## Duplicate Tracks", ""])
    for row in duplicate_tracks[:20]:
        markdown_lines.append(f"- {row['track_label']}: `{row['count']}` occurrences")
    if not duplicate_tracks:
        markdown_lines.append("- None")

    stem = f"playlist_intelligence_{deps.slugify(playlist_info['name'] or playlist_id)}"
    json_path, md_path = deps.write_report(
        output_dir=output_dir,
        category="playlist_intelligence",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"playlist_intelligence_json={json_path}")
    print(f"playlist_intelligence_md={md_path}")
    print(f"playlist_intelligence_tracks={track_count}")
    return 0


def _handle_discovery_search(
    args: argparse.Namespace,
    client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    item_types = deps.split_csv_list(args.types)
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
    stem = f"discovery_{deps.slugify(str(args.query))}"
    json_path, md_path = deps.write_report(
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


def _handle_catalog_linkouts(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    history_df = deps.load_history_if_needed(args, logger)
    artist_names = deps.top_artists_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_artists)
    track_rows = deps.top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_tracks)

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
    json_path, md_path = deps.write_report(
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


def _handle_artist_graph(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    artists = deps.resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
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

    stem = f"artist_graph_{deps.slugify('-'.join(artists[:3]) or 'history')}"
    json_path, md_path = deps.write_report(
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
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    history_df = deps.load_history_if_needed(args, logger)
    frame = deps.cross_media_history_frame(
        history_df,
        lookback_days=int(args.lookback_days),
        session_gap_minutes=int(args.session_gap_minutes),
    )
    graph_payload = deps.cross_media_graph_payload(
        frame,
        node_limit=int(args.node_limit),
        edge_limit=int(args.edge_limit),
        session_limit=int(args.session_limit),
    )
    seed_nodes = deps.seed_nodes_for_bridges(
        graph_payload.get("session_intelligence", {}).get("seed_nodes", []),
        limit=max(1, int(args.bridge_limit)),
    )
    bridges = deps.cross_media_catalog_bridges(
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
    json_path, md_path = deps.write_report(
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


def _handle_release_inbox(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    artists = deps.resolve_artists(args, logger, history_top_n=args.top_n, history_lookback_days=args.lookback_days)
    include_groups = ",".join(deps.split_csv_list(args.include_groups))
    as_of_ts = _as_of_timestamp(args)
    cutoff_ts = as_of_ts - pd.Timedelta(days=max(1, int(args.since_days)))
    state_key = (
        deps.slugify("explicit-" + "-".join(artists))
        if getattr(args, "artists", None)
        else f"history-{int(args.lookback_days)}d-top{int(args.top_n)}-since{int(args.since_days)}d"
    )
    state_path = deps.release_state_path(Path(args.output_dir).expanduser().resolve(), state_key)
    previous_state = deps.read_json_if_exists(state_path)
    seen_release_ids = {
        str(item).strip()
        for item in (previous_state or {}).get("release_ids", [])
        if str(item).strip()
    }

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
        deduped = deps.dedupe_album_rows(albums)
        for album in deduped:
            album_id = str(album.get("id", "")).strip()
            release_date = str(album.get("release_date", "")).strip()
            precision = str(album.get("release_date_precision", "day")).strip()
            release_ts = deps.parse_release_date(release_date, precision)
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
        "as_of_date": as_of_ts.date().isoformat(),
        "artists": artists,
        "new_releases": [row for row in inbox_rows if row["is_new_since_last_run"]],
        "all_recent_releases": inbox_rows,
    }
    deps.write_state(state_path, {"release_ids": sorted(set(current_release_ids))})

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
    json_path, md_path = deps.write_report(
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


def _handle_personal_release_radar(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    history_df, history_warnings = _load_history_for_optional_coverage(args, logger, deps=deps)
    explicit_seed_artists = deps.split_pipe_list(getattr(args, "artists", None))
    seed_artists = (
        explicit_seed_artists
        if explicit_seed_artists
        else deps.top_artists_from_history(
            history_df,
            lookback_days=int(args.lookback_days),
            limit=max(1, int(args.top_n)),
        )
    )
    coverage_index = _history_catalog_coverage_index(history_df, lookback_days=int(args.lookback_days))
    include_groups = ",".join(deps.split_csv_list(args.include_groups))
    market = str(args.spotify_market).upper()
    as_of_ts = _as_of_timestamp(args)
    cutoff_ts = as_of_ts - pd.Timedelta(days=max(1, int(args.since_days)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    state_key = (
        deps.slugify("explicit-" + "-".join(seed_artists))
        if getattr(args, "artists", None)
        else f"radar-history-{int(args.lookback_days)}d-top{int(args.top_n)}-since{int(args.since_days)}d"
    )
    state_path = deps.release_state_path(output_dir, state_key)
    previous_state = deps.read_json_if_exists(state_path)
    seen_release_ids = {
        str(item).strip()
        for item in (previous_state or {}).get("release_ids", [])
        if str(item).strip()
    }
    warnings: list[dict[str, Any]] = list(history_warnings)
    related_lookup_disabled = False

    candidates: dict[str, dict[str, Any]] = {}
    for seed_name in seed_artists:
        metadata = _safe_public_catalog_call(
            warnings,
            operation=f"search-artist:{seed_name}",
            default=None,
            func=client.search_artist,
            artist_name=seed_name,
        )
        if metadata is None:
            continue
        candidates[metadata.spotify_id] = {
            "artist_id": metadata.spotify_id,
            "artist_name": metadata.name,
            "spotify_url": metadata.spotify_url,
            "source": "seed",
            "seed_artist": metadata.name,
            "genres": metadata.genres,
            "popularity": metadata.popularity,
        }
        if bool(args.include_related) and not related_lookup_disabled:
            related = _safe_public_catalog_call(
                warnings,
                operation=f"related-artists:{metadata.spotify_id}",
                default=[],
                func=client.get_related_artists,
                call_args=(metadata.spotify_id,),
                limit=max(1, int(args.related_limit)),
            )
            if warnings and warnings[-1].get("operation") == f"related-artists:{metadata.spotify_id}" and warnings[-1].get("status_code") == 403:
                related_lookup_disabled = True
            for related_artist in related:
                candidates.setdefault(
                    related_artist.spotify_id,
                    {
                        "artist_id": related_artist.spotify_id,
                        "artist_name": related_artist.name,
                        "spotify_url": related_artist.spotify_url,
                        "source": "related",
                        "seed_artist": metadata.name,
                        "genres": related_artist.genres,
                        "popularity": related_artist.popularity,
                    },
                )

    release_rows: list[dict[str, Any]] = []
    current_release_ids: list[str] = []
    for artist in candidates.values():
        albums = _safe_public_catalog_call(
            warnings,
            operation=f"artist-albums:{artist['artist_id']}",
            default=[],
            func=client.get_artist_albums,
            call_args=(artist["artist_id"],),
            include_groups=include_groups,
            limit=max(10, int(args.per_artist_limit) * 3),
            market=market,
        )
        for album in deps.dedupe_album_rows(albums):
            album_id = str(album.get("id", "")).strip()
            release_date = str(album.get("release_date", "")).strip()
            precision = str(album.get("release_date_precision", "day")).strip()
            release_ts = deps.parse_release_date(release_date, precision)
            if not album_id or release_ts is None or release_ts < cutoff_ts:
                continue
            album_artist_names = _artist_names_from_items(album.get("artists")) or [str(artist["artist_name"])]
            artist_album_names = coverage_index["artist_album_names"]
            local_album_hits = sum(
                int(artist_album_names.get((_normalized_catalog_text(name), _normalized_catalog_text(album.get("name"))), 0))
                for name in album_artist_names
            )
            days_old = max(0, int((as_of_ts - release_ts).days))
            is_new_since_last_run = album_id not in seen_release_ids
            already_heard = local_album_hits > 0
            source_boost = 0.35 if artist["source"] == "seed" else 0.18
            novelty_boost = 0.25 if is_new_since_last_run else 0.0
            unheard_boost = 0.2 if not already_heard else -0.05
            recency_boost = max(0.0, 1.0 - days_old / max(1, int(args.since_days))) * 0.2
            priority_score = round(source_boost + novelty_boost + unheard_boost + recency_boost, 4)
            current_release_ids.append(album_id)
            release_rows.append(
                {
                    "artist_id": artist["artist_id"],
                    "artist_name": artist["artist_name"],
                    "artist_source": artist["source"],
                    "seed_artist": artist["seed_artist"],
                    "album_id": album_id,
                    "album_name": str(album.get("name", "")).strip(),
                    "album_type": str(album.get("album_type", "")).strip(),
                    "release_date": release_date,
                    "release_date_precision": precision,
                    "days_old": days_old,
                    "total_tracks": int(album.get("total_tracks", 0) or 0),
                    "spotify_url": str(album.get("external_urls", {}).get("spotify", "")).strip(),
                    "image_url": _first_public_image_url(album.get("images")),
                    "is_new_since_last_run": is_new_since_last_run,
                    "already_heard_album": already_heard,
                    "local_album_history_hits": local_album_hits,
                    "priority_score": priority_score,
                }
            )

    release_rows.sort(
        key=lambda row: (
            float(row["priority_score"]),
            bool(row["is_new_since_last_run"]),
            str(row["release_date"]),
            str(row["artist_name"]),
        ),
        reverse=True,
    )
    recommendations: list[dict[str, str]] = []
    if release_rows:
        top_release = release_rows[0]
        recommendations.append(
            _report_recommendation(
                f"Queue `{top_release['album_name']}` by `{top_release['artist_name']}` first.",
                f"It has the highest radar score ({float(top_release['priority_score']):.2f}).",
                priority="high",
            )
        )
    seed_unheard = [
        row
        for row in release_rows
        if row["artist_source"] == "seed" and not row["already_heard_album"]
    ]
    if seed_unheard:
        recommendations.append(
            _report_recommendation(
                f"Prioritize seed-artist gap `{seed_unheard[0]['album_name']}`.",
                "It is from a seed artist and does not appear in local album history.",
                priority="high",
            )
        )
    related_finds = [row for row in release_rows if row["artist_source"] == "related"]
    if related_finds:
        recommendations.append(
            _report_recommendation(
                f"Sample related-artist release `{related_finds[0]['album_name']}`.",
                f"It expands from seed `{related_finds[0]['seed_artist']}`.",
                priority="medium",
            )
        )
    if _has_rate_limit_warning(warnings):
        recommendations.append(_rate_limit_recommendation("personal-release-radar --top-n 3"))
    deps.write_state(state_path, {"release_ids": sorted(set(current_release_ids))})
    payload = {
        "command": "personal-release-radar",
        "market": market,
        "since_days": int(args.since_days),
        "as_of_date": as_of_ts.date().isoformat(),
        "include_groups": include_groups,
        "seed_artists": seed_artists,
        "candidate_artists": list(candidates.values()),
        "priority_releases": release_rows,
        "new_releases_since_last_run": [row for row in release_rows if row["is_new_since_last_run"]],
        "recommendations": recommendations,
        "warnings": warnings,
        "policy_note": "Public release metadata is used for display/link-out reporting only.",
    }
    markdown_lines = [
        "# Spotify Personal Release Radar",
        "",
        f"- Market: `{market}`",
        f"- Seed artists: `{len(seed_artists)}`",
        f"- Candidate artists: `{len(candidates)}`",
        f"- Priority releases: `{len(release_rows)}`",
        f"- New since last run: `{len(payload['new_releases_since_last_run'])}`",
        f"- API warnings: `{len(warnings)}`",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in recommendations:
        markdown_lines.append(f"- [{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}")
    if not recommendations:
        markdown_lines.append("- No current release recommendations.")
    markdown_lines.extend([
        "",
        "## Priority Releases",
        "",
    ])
    for row in release_rows[:20]:
        heard = "heard" if row["already_heard_album"] else "unheard"
        new = "new" if row["is_new_since_last_run"] else "seen"
        markdown_lines.append(
            f"- `{row['priority_score']:.2f}` | {row['release_date']} | {row['artist_name']} | "
            f"{row['album_name']} ({row['album_type']}, {new}, {heard})"
        )

    stem = f"personal_release_radar_{state_key}"
    json_path, md_path = deps.write_report(
        output_dir=output_dir,
        category="personal_release_radar",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"personal_release_radar_json={json_path}")
    print(f"personal_release_radar_md={md_path}")
    print(f"personal_release_radar_releases={len(release_rows)}")
    return 0


def _handle_playlist_diff(
    args: argparse.Namespace,
    client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    playlist_id = parse_spotify_id(args.playlist, expected_kind="playlist")
    output_dir = Path(args.output_dir).expanduser().resolve()
    playlist_payload, track_rows, _artist_counter = _playlist_item_rows(
        client,
        playlist_id=playlist_id,
        market=str(args.spotify_market).upper(),
        limit=max(1, int(args.item_limit)),
    )
    snapshot = deps.playlist_snapshot(playlist_payload, track_rows)
    playlist_slug = deps.slugify(snapshot.get("name", playlist_id) or playlist_id)
    state_path = deps.playlist_state_path(output_dir, playlist_id)
    previous_state = deps.read_json_if_exists(state_path)
    diff = deps.playlist_diff(previous_state, snapshot)
    payload = {
        "command": "playlist-diff",
        "playlist_id": playlist_id,
        "playlist_name": snapshot.get("name"),
        "diff": diff,
        "current_snapshot": snapshot,
    }
    deps.write_state(state_path, snapshot)

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
    json_path, md_path = deps.write_report(
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


def _handle_market_gap(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    explicit_tracks = deps.split_pipe_list(args.tracks)
    if explicit_tracks:
        track_rows = [{"spotify_track_uri": value, "track_name": value, "artist_name": ""} for value in explicit_tracks]
    else:
        history_df = deps.load_history_if_needed(args, logger)
        track_rows = deps.top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_n)
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
    json_path, md_path = deps.write_report(
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


def _handle_playlist_archive(
    args: argparse.Namespace,
    client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    playlist_id = parse_spotify_id(args.playlist, expected_kind="playlist")
    playlist_payload, track_rows, artist_counter = _playlist_item_rows(
        client,
        playlist_id=playlist_id,
        market=str(args.spotify_market).upper(),
        limit=max(1, int(args.item_limit)),
    )
    snapshot = deps.playlist_snapshot(playlist_payload, track_rows)
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

    stem = f"playlist_archive_{deps.slugify(snapshot.get('name', playlist_id) or playlist_id)}"
    json_path, md_path = deps.write_report(
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


def _handle_catalog_crosswalk(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    explicit_tracks = deps.split_pipe_list(args.tracks)
    if explicit_tracks:
        track_rows = [{"spotify_track_uri": value, "track_name": value, "artist_name": ""} for value in explicit_tracks]
    else:
        history_df = deps.load_history_if_needed(args, logger)
        track_rows = deps.top_tracks_from_history(history_df, lookback_days=args.lookback_days, limit=args.top_n)

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
    json_path, md_path = deps.write_report(
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


def _handle_album_profile(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
    album_id = parse_spotify_id(args.album, expected_kind="album")
    market = str(args.spotify_market).upper()
    warnings: list[dict[str, Any]] = []
    album_payload = _safe_public_catalog_call(
        warnings,
        operation=f"album:{album_id}",
        default={},
        func=client.get_album,
        call_args=(album_id,),
        market=market,
    )
    album_metadata_obj = (
        _safe_public_catalog_call(
            warnings,
            operation=f"album-metadata:{album_id}",
            default=None,
            func=client.get_album_metadata,
            call_args=(album_id,),
            market=market,
        )
        if hasattr(client, "get_album_metadata")
        else None
    )
    album_metadata = (
        asdict(album_metadata_obj)
        if album_metadata_obj is not None
        else {
            "query": album_id,
            "spotify_id": str(album_payload.get("id", album_id)).strip(),
            "name": str(album_payload.get("name", album_id)).strip(),
            "spotify_url": str(album_payload.get("external_urls", {}).get("spotify", "")).strip(),
            "album_type": str(album_payload.get("album_type", "")).strip(),
            "release_date": str(album_payload.get("release_date", "")).strip(),
            "release_date_precision": str(album_payload.get("release_date_precision", "")).strip(),
            "total_tracks": album_payload.get("total_tracks"),
            "label": str(album_payload.get("label", "")).strip() or None,
            "upc": None,
            "ean": None,
            "artists": _artist_names_from_items(album_payload.get("artists")),
            "image_url": _first_public_image_url(album_payload.get("images")),
            "available_markets_count": (
                len(album_payload.get("available_markets", []))
                if isinstance(album_payload.get("available_markets"), list)
                else None
            ),
            "restriction_reasons": [],
        }
    )
    track_source_payload = album_payload.get("tracks", {}) if isinstance(album_payload, dict) else {}
    raw_tracks: list[dict[str, Any]] = []
    if hasattr(client, "get_album_tracks"):
        raw_tracks = _safe_public_catalog_call(
            warnings,
            operation=f"album-tracks:{album_id}",
            default=[],
            func=client.get_album_tracks,
            call_args=(album_id,),
            limit=max(1, int(args.track_limit)),
            market=market,
        )
    if not raw_tracks and isinstance(track_source_payload, dict) and isinstance(track_source_payload.get("items"), list):
        raw_tracks = [track for track in track_source_payload["items"] if isinstance(track, dict)]

    history_df, history_warnings = _load_history_for_optional_coverage(args, logger, deps=deps)
    warnings.extend(history_warnings)
    coverage_index = _history_catalog_coverage_index(history_df, lookback_days=int(args.lookback_days))
    album_artist_names = list(album_metadata.get("artists", []) or _artist_names_from_items(album_payload.get("artists")))
    track_items: list[dict[str, Any]] = []
    collaborator_counter: Counter[str] = Counter()
    total_duration_ms = 0
    explicit_count = 0
    listened_count = 0
    history_plays = 0

    for track in raw_tracks[: max(1, int(args.track_limit))]:
        track_artist_names = _artist_names_from_items(track.get("artists")) or album_artist_names
        for artist_name in track_artist_names:
            if artist_name not in set(album_artist_names):
                collaborator_counter[artist_name] += 1
        listened, track_plays, match_source = _catalog_track_history_match(
            track,
            artist_names=track_artist_names,
            coverage_index=coverage_index,
        )
        duration_ms = int(track.get("duration_ms", 0) or 0)
        total_duration_ms += duration_ms
        explicit_count += int(bool(track.get("explicit", False)))
        listened_count += int(listened)
        history_plays += int(track_plays)
        track_items.append(
            {
                "track_id": str(track.get("id", "")).strip(),
                "track_number": track.get("track_number"),
                "name": str(track.get("name", "")).strip(),
                "duration_ms": duration_ms,
                "explicit": bool(track.get("explicit", False)),
                "spotify_url": str(track.get("external_urls", {}).get("spotify", "")).strip(),
                "artist_names": track_artist_names,
                "listened": listened,
                "history_plays": track_plays,
                "match_source": match_source,
            }
        )

    track_count = len(track_items)
    album_summary = {
        "tracks_loaded": track_count,
        "tracks_listened": listened_count,
        "coverage_ratio": float(listened_count / track_count) if track_count else 0.0,
        "history_plays": history_plays,
        "explicit_tracks": explicit_count,
        "duration_ms": total_duration_ms,
        "duration_minutes": round(total_duration_ms / 60000.0, 2) if total_duration_ms else 0.0,
        "collaborators": [
            {"artist_name": artist_name, "track_count": count}
            for artist_name, count in collaborator_counter.most_common()
        ],
    }
    recommendations: list[dict[str, str]] = []
    missing_tracks = [track for track in track_items if not track["listened"]]
    if missing_tracks:
        recommendations.append(
            _report_recommendation(
                f"Start with `{missing_tracks[0]['name']}`.",
                "It is the first loaded album track not found in local history.",
                priority="high",
            )
        )
    if album_summary["coverage_ratio"] == 1.0 and track_count:
        recommendations.append(
            _report_recommendation(
                "Treat this album as a familiar/comfort reference.",
                "Every loaded track appears in local history.",
                priority="low",
            )
        )

    payload = {
        "command": "album-profile",
        "album_id": album_id,
        "market": market,
        "album": album_metadata,
        "album_summary": album_summary,
        "tracks": track_items,
        "recommendations": recommendations,
        "warnings": warnings,
        "policy_note": (
            "Spotify public catalog metadata is exported for display/link-out only. "
            "Do not download Spotify content, alter cover art, or train models on public Spotify content."
        ),
    }
    markdown_lines = [
        "# Spotify Album Profile",
        "",
        f"- Album: `{album_metadata.get('name', '')}`",
        f"- Artists: `{', '.join(album_metadata.get('artists', []) or []) or 'n/a'}`",
        f"- Spotify URL: {album_metadata.get('spotify_url') or 'n/a'}",
        f"- Type: `{album_metadata.get('album_type') or 'n/a'}`",
        f"- Release date: `{album_metadata.get('release_date') or 'n/a'}` precision=`{album_metadata.get('release_date_precision') or 'n/a'}`",
        f"- Total tracks: `{album_metadata.get('total_tracks')}`",
        f"- Tracks loaded: `{album_summary['tracks_loaded']}`",
        f"- Local coverage: `{album_summary['coverage_ratio']:.1%}` "
        f"({album_summary['tracks_listened']}/{album_summary['tracks_loaded']})",
        f"- Duration: `{album_summary['duration_minutes']}` minutes",
        f"- Explicit tracks: `{album_summary['explicit_tracks']}`",
        f"- Label: `{album_metadata.get('label') or 'n/a'}`",
        f"- UPC: `{album_metadata.get('upc') or 'n/a'}` EAN: `{album_metadata.get('ean') or 'n/a'}`",
        f"- Available markets: `{album_metadata.get('available_markets_count')}`",
        f"- API warnings: `{len(warnings)}`",
        f"- Cover art URL: {album_metadata.get('image_url') or 'n/a'}",
        "",
        "Policy note: keep Spotify artwork unmodified and include the Spotify link when displaying this metadata.",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in recommendations:
        markdown_lines.append(f"- [{recommendation['priority']}] {recommendation['action']} {recommendation['reason']}")
    if not recommendations:
        markdown_lines.append("- No immediate album actions.")
    markdown_lines.extend([
        "",
        "## Tracks",
        "",
    ])
    for track in track_items:
        heard = "heard" if track.get("listened") else "not in local history"
        markdown_lines.append(
            f"- {track.get('track_number')}. {track.get('name')} "
            f"by {', '.join(track.get('artist_names', []) or []) or 'n/a'} "
            f"({heard}, plays `{track.get('history_plays')}`)"
        )

    if album_summary["collaborators"]:
        markdown_lines.extend(["", "## Collaborators", ""])
        for row in album_summary["collaborators"][:10]:
            markdown_lines.append(f"- {row['artist_name']}: `{row['track_count']}` tracks")

    stem = f"album_profile_{deps.slugify(str(album_metadata.get('name') or album_id))}"
    json_path, md_path = deps.write_report(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        category="album_profile",
        stem=stem,
        payload=payload,
        markdown_lines=markdown_lines,
    )
    print(f"album_profile_json={json_path}")
    print(f"album_profile_md={md_path}")
    print(f"album_profile_tracks={len(track_items)}")
    return 0


def _handle_media_explorer(
    args: argparse.Namespace,
    client: Any,
    _logger: logging.Logger,
    *,
    deps: PublicInsightsHandlerDeps,
) -> int:
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

    stem = f"media_{media_type}_{deps.slugify(str(payload.get('query', payload.get('item', {}).get('name', 'item'))))}"
    json_path, md_path = deps.write_report(
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


def build_standard_handler_registry(deps: PublicInsightsHandlerDeps) -> dict[str, PublicInsightsHandler]:
    return {
        "explain-artists": partial(_handle_explain_artists, deps=deps),
        "release-tracker": partial(_handle_release_tracker, deps=deps),
        "artist-top-tracks": partial(_handle_artist_top_tracks, deps=deps),
        "new-releases": partial(_handle_new_releases, deps=deps),
        "market-check": partial(_handle_market_check, deps=deps),
        "discography": partial(_handle_discography, deps=deps),
        "artist-catalog-completeness": partial(_handle_artist_catalog_completeness, deps=deps),
        "playlist-view": partial(_handle_playlist_view, deps=deps),
        "playlist-intelligence": partial(_handle_playlist_intelligence, deps=deps),
        "discovery-search": partial(_handle_discovery_search, deps=deps),
        "catalog-linkouts": partial(_handle_catalog_linkouts, deps=deps),
        "artist-graph": partial(_handle_artist_graph, deps=deps),
        "cross-media-taste-graph": partial(_handle_cross_media_taste_graph, deps=deps),
        "release-inbox": partial(_handle_release_inbox, deps=deps),
        "personal-release-radar": partial(_handle_personal_release_radar, deps=deps),
        "playlist-diff": partial(_handle_playlist_diff, deps=deps),
        "market-gap": partial(_handle_market_gap, deps=deps),
        "playlist-archive": partial(_handle_playlist_archive, deps=deps),
        "catalog-crosswalk": partial(_handle_catalog_crosswalk, deps=deps),
        "album-profile": partial(_handle_album_profile, deps=deps),
        "media-explorer": partial(_handle_media_explorer, deps=deps),
        "summary": partial(_handle_public_insights_summary, deps=deps),
    }
