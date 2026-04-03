from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from .catalog_utils import dedupe_album_rows, parse_release_date
from .data import load_streaming_history
from .env import load_local_env
from .public_catalog import SpotifyArtistMetadata, SpotifyPublicCatalogClient, SpotifyPublicCatalogError
from .public_insights_cli import build_public_insights_parser
from .public_insights_creator_brief import (
    CreatorBriefHandlerDeps,
    _creator_brief_executive_summary,
    _creator_brief_migration_watch,
    _creator_brief_priority_shortlist,
    _creator_brief_ranking_comparison,
    _creator_brief_release_watch,
    _creator_brief_scene_comparison,
    _creator_brief_scene_seed_comparison,
    _creator_brief_seed_comparison,
    build_creator_label_intelligence_handler,
)
from .public_insights_graph import (
    _cross_media_catalog_bridges,
    _cross_media_graph_payload,
    _cross_media_history_frame,
    _playlist_diff,
    _playlist_snapshot,
    _recent_history,
    _release_state_rows,
    _seed_nodes_for_bridges,
    _top_artists_from_history,
    _top_tracks_from_history,
)
from .public_insights_handlers import PublicInsightsHandlerDeps, build_standard_handler_registry
from .run_artifacts import safe_read_json as _safe_read_json
from .run_artifacts import write_json, write_markdown


class _OfflineSpotifyPublicCatalogClient:
    mode = "offline_local_only"

    def search_artist(self, artist_name: str) -> SpotifyArtistMetadata | None:
        return None

    def get_related_artists(self, artist_id_or_uri: str, *, limit: int = 10) -> list[SpotifyArtistMetadata]:
        return []

    def get_artist_albums(
        self,
        artist_id_or_uri: str,
        *,
        include_groups: str = "album,single",
        limit: int = 50,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def get_album(self, album_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        return {"id": album_id_or_uri, "label": ""}


def _build_parser() -> argparse.ArgumentParser:
    return build_public_insights_parser()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "report"


def _split_pipe_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _split_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_release_date(value: str, precision: str) -> pd.Timestamp | None:
    return parse_release_date(value, precision)


def _dedupe_album_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return dedupe_album_rows(rows)


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


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    payload = _safe_read_json(path, default=None)
    return payload if isinstance(payload, dict) else None


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _playlist_state_path(output_dir: Path, playlist_slug: str) -> Path:
    return output_dir / "analysis" / "public_spotify" / "playlist_state" / f"{playlist_slug}.json"


def _release_state_path(output_dir: Path, artist_slug: str) -> Path:
    return output_dir / "analysis" / "public_spotify" / "release_state" / f"{artist_slug}.json"




def _build_client(args: argparse.Namespace) -> SpotifyPublicCatalogClient | _OfflineSpotifyPublicCatalogClient:
    client = SpotifyPublicCatalogClient.from_env(market=str(args.spotify_market or "US"))
    if client is None:
        if str(getattr(args, "command", "")).strip() == "creator-label-intelligence":
            return _OfflineSpotifyPublicCatalogClient()  # type: ignore[return-value]
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


_STANDARD_HANDLER_DEPS = PublicInsightsHandlerDeps(
    split_csv_list=_split_csv_list,
    split_pipe_list=_split_pipe_list,
    resolve_artists=_resolve_artists,
    load_history_if_needed=_load_history_if_needed,
    top_artists_from_history=_top_artists_from_history,
    top_tracks_from_history=_top_tracks_from_history,
    parse_release_date=_parse_release_date,
    dedupe_album_rows=_dedupe_album_rows,
    write_report=_write_report,
    read_json_if_exists=_read_json_if_exists,
    write_state=_write_state,
    playlist_state_path=_playlist_state_path,
    release_state_path=_release_state_path,
    playlist_snapshot=_playlist_snapshot,
    playlist_diff=_playlist_diff,
    cross_media_history_frame=_cross_media_history_frame,
    cross_media_graph_payload=_cross_media_graph_payload,
    seed_nodes_for_bridges=_seed_nodes_for_bridges,
    cross_media_catalog_bridges=_cross_media_catalog_bridges,
    slugify=_slugify,
)

_STANDARD_HANDLER_REGISTRY = build_standard_handler_registry(_STANDARD_HANDLER_DEPS)

_CREATOR_BRIEF_HANDLER = build_creator_label_intelligence_handler(
    CreatorBriefHandlerDeps(
        load_history_if_needed=_load_history_if_needed,
        recent_history=_recent_history,
        resolve_artists=_resolve_artists,
    )
)


def _handle_explain_artists(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["explain-artists"](args, client, logger)


def _handle_release_tracker(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["release-tracker"](args, client, logger)


def _handle_market_check(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["market-check"](args, client, logger)


def _handle_discography(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["discography"](args, client, logger)


def _handle_playlist_view(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["playlist-view"](args, client, _logger)


def _handle_discovery_search(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["discovery-search"](args, client, _logger)


def _handle_catalog_linkouts(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["catalog-linkouts"](args, client, logger)


def _handle_artist_graph(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["artist-graph"](args, client, logger)


def _handle_cross_media_taste_graph(
    args: argparse.Namespace,
    client: SpotifyPublicCatalogClient,
    logger: logging.Logger,
) -> int:
    return _STANDARD_HANDLER_REGISTRY["cross-media-taste-graph"](args, client, logger)


def _handle_release_inbox(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["release-inbox"](args, client, logger)


def _handle_playlist_diff(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["playlist-diff"](args, client, _logger)


def _handle_market_gap(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["market-gap"](args, client, logger)


def _handle_playlist_archive(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["playlist-archive"](args, client, _logger)


def _handle_catalog_crosswalk(args: argparse.Namespace, client: SpotifyPublicCatalogClient, logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["catalog-crosswalk"](args, client, logger)


def _handle_media_explorer(args: argparse.Namespace, client: SpotifyPublicCatalogClient, _logger: logging.Logger) -> int:
    return _STANDARD_HANDLER_REGISTRY["media-explorer"](args, client, _logger)


_handle_creator_label_intelligence = _CREATOR_BRIEF_HANDLER


PublicInsightsHandler = Callable[[argparse.Namespace, Any, logging.Logger], int]

_COMMAND_HANDLERS: dict[str, PublicInsightsHandler] = {
    **_STANDARD_HANDLER_REGISTRY,
    "creator-label-intelligence": _handle_creator_label_intelligence,
}


def _dispatch_command(
    args: argparse.Namespace,
    client: Any,
    logger: logging.Logger,
    parser: argparse.ArgumentParser,
) -> int:
    handler = _COMMAND_HANDLERS.get(str(args.command))
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args, client, logger)


def main() -> int:
    load_local_env()
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.public_insights")

    try:
        client = _build_client(args)
        return _dispatch_command(args, client, logger, parser)
    except (RuntimeError, SpotifyPublicCatalogError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
