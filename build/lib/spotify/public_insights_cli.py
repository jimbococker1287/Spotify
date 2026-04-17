from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PublicInsightsCommandSpec:
    name: str
    help_text: str
    configure: Callable[[argparse.ArgumentParser], None]


def _configure_explain_artists(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to explain.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of history-derived artists to explain.")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top artists.")
    parser.add_argument("--related-limit", type=int, default=5, help="Number of related artists to include.")


def _configure_release_tracker(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to track.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of history-derived artists to track.")
    parser.add_argument("--lookback-days", type=int, default=365, help="History window for deriving top artists.")
    parser.add_argument("--since-days", type=int, default=120, help="Only include releases from the last N days.")
    parser.add_argument(
        "--include-groups",
        type=str,
        default="album,single",
        help="Comma-separated Spotify album groups to request.",
    )
    parser.add_argument("--per-artist-limit", type=int, default=10, help="Maximum releases per artist in the report.")


def _configure_market_check(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracks", type=str, default=None, help="Pipe-separated track URLs, URIs, or IDs.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of history-derived tracks to inspect.")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top tracks.")
    parser.add_argument(
        "--markets",
        type=str,
        default="US",
        help="Comma-separated market codes to check, for example US,GB,IN.",
    )


def _configure_discography(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to inspect.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of history-derived artists to inspect.")
    parser.add_argument("--lookback-days", type=int, default=365, help="History window for deriving top artists.")
    parser.add_argument(
        "--include-groups",
        type=str,
        default="album,single",
        help="Comma-separated Spotify album groups to request.",
    )
    parser.add_argument("--album-limit", type=int, default=20, help="Maximum album rows per artist.")


def _configure_playlist_view(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--playlist", type=str, required=True, help="Spotify playlist URL, URI, or ID.")
    parser.add_argument("--item-limit", type=int, default=50, help="Maximum playlist items to load.")


def _configure_discovery_search(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", type=str, required=True, help="Spotify search query.")
    parser.add_argument(
        "--types",
        type=str,
        default="artist,album,track,playlist",
        help="Comma-separated Spotify item types to search.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum results per type.")


def _configure_catalog_linkouts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-artists", type=int, default=10, help="Number of recent artists to include.")
    parser.add_argument("--top-tracks", type=int, default=20, help="Number of recent tracks to include.")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for deriving items.")


def _configure_artist_graph(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artists", type=str, default=None, help="Pipe-separated seed artists.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of history-derived seed artists.")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for deriving seed artists.")
    parser.add_argument("--related-limit", type=int, default=10, help="Number of related artists per seed.")


def _configure_cross_media_taste_graph(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for recent listening context.")
    parser.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="Gap in minutes that starts a new inferred listening session.",
    )
    parser.add_argument("--node-limit", type=int, default=40, help="Maximum graph nodes to emit in the report.")
    parser.add_argument("--edge-limit", type=int, default=80, help="Maximum graph edges to emit in the report.")
    parser.add_argument(
        "--session-limit",
        type=int,
        default=8,
        help="Maximum mixed-session summaries to include in the report.",
    )
    parser.add_argument(
        "--bridge-limit",
        type=int,
        default=6,
        help="Maximum history-derived seed nodes to expand into cross-media catalog bridges.",
    )
    parser.add_argument(
        "--recommendation-limit",
        type=int,
        default=3,
        help="Maximum catalog results per seed/type bridge.",
    )


def _configure_release_inbox(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artists", type=str, default=None, help="Pipe-separated artist names to track.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of history-derived artists to track.")
    parser.add_argument("--lookback-days", type=int, default=365, help="History window for deriving seed artists.")
    parser.add_argument("--since-days", type=int, default=120, help="Only include releases from the last N days.")
    parser.add_argument("--include-groups", type=str, default="album,single", help="Album groups to include.")
    parser.add_argument("--per-artist-limit", type=int, default=10, help="Maximum releases per artist in the inbox.")


def _configure_playlist_diff(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--playlist", type=str, required=True, help="Spotify playlist URL, URI, or ID.")
    parser.add_argument("--item-limit", type=int, default=100, help="Maximum playlist items to load.")


def _configure_market_gap(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracks", type=str, default=None, help="Pipe-separated track URLs, URIs, or IDs.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of history-derived tracks to inspect.")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top tracks.")


def _configure_playlist_archive(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--playlist", type=str, required=True, help="Spotify playlist URL, URI, or ID.")
    parser.add_argument("--item-limit", type=int, default=100, help="Maximum playlist items to load.")


def _configure_catalog_crosswalk(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracks", type=str, default=None, help="Pipe-separated track URLs, URIs, or IDs.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of history-derived tracks to include.")
    parser.add_argument("--lookback-days", type=int, default=180, help="History window for deriving top tracks.")


def _configure_media_explorer(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", type=str, default=None, help="Spotify search query for media exploration.")
    parser.add_argument("--media-type", choices=("show", "episode", "audiobook"), default="show")
    parser.add_argument("--item-id", type=str, default=None, help="Optional direct Spotify URL, URI, or ID.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum search results or child items to include.")


def _configure_creator_label_intelligence(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artists", type=str, default=None, help="Pipe-separated seed artists.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of history-derived seed artists.")
    parser.add_argument("--lookback-days", type=int, default=365, help="History window for deriving the graph.")
    parser.add_argument("--related-limit", type=int, default=8, help="Related artists to inspect per seed.")
    parser.add_argument("--neighbor-k", type=int, default=5, help="Local multimodal neighbors to add per seed.")
    parser.add_argument("--release-limit", type=int, default=8, help="Maximum releases to inspect per artist.")
    parser.add_argument("--scene-count", type=int, default=None, help="Optional fixed number of scene clusters.")
    parser.add_argument(
        "--max-artists",
        type=int,
        default=250,
        help="Maximum local artists to keep when deriving a multimodal space from recent history.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run directory containing analysis/multimodal/multimodal_artist_space.joblib.",
    )
    parser.add_argument(
        "--multimodal-artifact",
        type=str,
        default=None,
        help="Optional direct path to a multimodal_artist_space.joblib artifact.",
    )


_COMMAND_SPECS = (
    PublicInsightsCommandSpec(
        "explain-artists",
        "Explain your top artists with Spotify public metadata.",
        _configure_explain_artists,
    ),
    PublicInsightsCommandSpec(
        "release-tracker",
        "Track new releases from your favorite artists.",
        _configure_release_tracker,
    ),
    PublicInsightsCommandSpec("market-check", "Check market availability for your top tracks.", _configure_market_check),
    PublicInsightsCommandSpec(
        "discography",
        "Build a discography timeline for favorite artists.",
        _configure_discography,
    ),
    PublicInsightsCommandSpec("playlist-view", "Inspect a public Spotify playlist.", _configure_playlist_view),
    PublicInsightsCommandSpec(
        "discovery-search",
        "Run a Spotify search query for discovery.",
        _configure_discovery_search,
    ),
    PublicInsightsCommandSpec(
        "catalog-linkouts",
        "Build link-out bundles for your top artists and tracks.",
        _configure_catalog_linkouts,
    ),
    PublicInsightsCommandSpec("artist-graph", "Build a related-artist graph from seed artists.", _configure_artist_graph),
    PublicInsightsCommandSpec(
        "cross-media-taste-graph",
        "Build a cross-media session-intelligence graph across music, podcasts, shows, and audiobooks.",
        _configure_cross_media_taste_graph,
    ),
    PublicInsightsCommandSpec("release-inbox", "Track only newly seen releases since the last run.", _configure_release_inbox),
    PublicInsightsCommandSpec("playlist-diff", "Track changes in a public playlist over time.", _configure_playlist_diff),
    PublicInsightsCommandSpec(
        "market-gap",
        "Find market coverage gaps for tracks against all Spotify markets.",
        _configure_market_gap,
    ),
    PublicInsightsCommandSpec(
        "playlist-archive",
        "Archive playlist metadata, items, and image URLs.",
        _configure_playlist_archive,
    ),
    PublicInsightsCommandSpec(
        "catalog-crosswalk",
        "Build ISRC/UPC/EAN crosswalks for top tracks.",
        _configure_catalog_crosswalk,
    ),
    PublicInsightsCommandSpec(
        "media-explorer",
        "Explore public shows, episodes, and audiobooks.",
        _configure_media_explorer,
    ),
    PublicInsightsCommandSpec(
        "creator-label-intelligence",
        "Build an A&R intelligence graph using local multimodal artist space plus Spotify public metadata.",
        _configure_creator_label_intelligence,
    ),
)


def build_public_insights_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.public_insights",
        description="Policy-safe Spotify public metadata tools for explanation, discovery, and catalog exploration.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated reports.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--include-video", action="store_true", help="Include video history files when needed.")
    parser.add_argument("--spotify-market", type=str, default="US", help="Two-letter market code for Spotify requests.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for spec in _COMMAND_SPECS:
        subparser = subparsers.add_parser(spec.name, help=spec.help_text)
        spec.configure(subparser)
    return parser
