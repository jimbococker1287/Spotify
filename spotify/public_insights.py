from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Any

import pandas as pd

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
        parser.error(f"Unknown command: {args.command}")
        return 2
    except (RuntimeError, SpotifyPublicCatalogError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
