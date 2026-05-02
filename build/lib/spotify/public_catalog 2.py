from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1"
SEARCH_URL = f"{API_BASE_URL}/search"
DEFAULT_BACKEND = "urllib"
ARTIST_ALBUM_PAGE_SIZE = 5
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_DEFAULT_SLEEP_SECONDS = 1.0
RATE_LIMIT_MAX_SLEEP_SECONDS = 8.0
DEFAULT_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.2


class SpotifyPublicCatalogError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SpotifyArtistMetadata:
    query: str
    spotify_id: str
    name: str
    spotify_url: str
    genres: list[str]
    popularity: int | None
    followers_total: int | None
    image_url: str | None


@dataclass(frozen=True)
class SpotifyAlbumMetadata:
    query: str
    spotify_id: str
    name: str
    spotify_url: str
    album_type: str
    release_date: str
    release_date_precision: str
    total_tracks: int | None
    label: str | None
    upc: str | None
    ean: str | None
    artists: list[str]
    image_url: str | None
    available_markets_count: int | None
    restriction_reasons: list[str]


def _normalize_artist_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _normalize_backend(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "default", "auto"}:
        return DEFAULT_BACKEND
    if normalized not in {"urllib", "spotipy"}:
        raise ValueError("Spotify public catalog backend must be `urllib` or `spotipy`.")
    return normalized


def _float_setting(value: str | float | int | None, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_spotify_id(value: str, *, expected_kind: str | None = None) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Spotify identifier is required.")
    placeholder = raw.casefold().strip()
    placeholder_tokens = {"real_id_here", "<playlist-url>", "<actual_playlist_id>", "<id>", "playlist-url"}
    if "<" in raw or ">" in raw or placeholder in placeholder_tokens:
        noun = f" {expected_kind}" if expected_kind else ""
        raise ValueError(
            f"Replace the Spotify{noun} placeholder with a real Spotify URL, URI, or base62 ID. "
            "Example playlist URL: https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        )

    if raw.startswith("spotify:"):
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"Unsupported Spotify URI: {value}")
        kind = parts[1].strip()
        identifier = parts[2].strip()
        if expected_kind and kind != expected_kind:
            raise ValueError(f"Expected Spotify {expected_kind} URI, received {kind}.")
        if not identifier:
            raise ValueError(f"Unsupported Spotify URI: {value}")
        return identifier

    if raw.startswith("https://") or raw.startswith("http://"):
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0] == "embed":
            parts = parts[1:]
        if parts and parts[0].startswith("intl-"):
            parts = parts[1:]
        if len(parts) >= 4 and parts[0] == "user":
            parts = parts[2:4]
        if len(parts) < 2:
            raise ValueError(f"Unsupported Spotify URL: {value}")
        kind = parts[0].strip()
        identifier = parts[1].strip()
        if expected_kind and kind != expected_kind:
            raise ValueError(f"Expected Spotify {expected_kind} URL, received {kind}.")
        if not identifier:
            raise ValueError(f"Unsupported Spotify URL: {value}")
        return identifier

    return raw


class SpotifyPublicCatalogClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        market: str = "US",
        timeout_seconds: float = 10.0,
        backend: str | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: float | None = None,
        min_request_interval_seconds: float | None = None,
    ) -> None:
        self._client_id = client_id.strip()
        self._client_secret = client_secret.strip()
        self._market = (market or "US").strip().upper()
        self._timeout_seconds = float(timeout_seconds)
        self._backend = _normalize_backend(backend or os.getenv("SPOTIFY_PUBLIC_CATALOG_BACKEND", DEFAULT_BACKEND))
        self._spotipy_client: Any | None = None
        cache_dir_raw = cache_dir if cache_dir is not None else os.getenv("SPOTIFY_PUBLIC_CACHE_DIR", "").strip()
        self._cache_dir = Path(cache_dir_raw).expanduser().resolve() if cache_dir_raw else None
        self._cache_ttl_seconds = max(
            0.0,
            _float_setting(
                cache_ttl_seconds,
                default=_float_setting(os.getenv("SPOTIFY_PUBLIC_CACHE_TTL_SECONDS"), default=DEFAULT_CACHE_TTL_SECONDS),
            ),
        )
        self._min_request_interval_seconds = max(
            0.0,
            _float_setting(
                min_request_interval_seconds,
                default=_float_setting(
                    os.getenv("SPOTIFY_PUBLIC_REQUEST_INTERVAL_SECONDS"),
                    default=DEFAULT_REQUEST_INTERVAL_SECONDS,
                ),
            ),
        )
        self._last_request_at = 0.0
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0
        self._artist_cache: dict[str, SpotifyArtistMetadata | None] = {}
        self._json_cache: dict[str, dict[str, Any]] = {}

        if not self._client_id or not self._client_secret:
            raise ValueError("Spotify client credentials are required.")
        if self._backend == "spotipy":
            self._spotipy_client = self._build_spotipy_client()

    @classmethod
    def from_env(
        cls,
        *,
        market: str = "US",
        timeout_seconds: float = 10.0,
        backend: str | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: float | None = None,
        min_request_interval_seconds: float | None = None,
    ) -> SpotifyPublicCatalogClient | None:
        client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            return None
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            market=market,
            timeout_seconds=timeout_seconds,
            backend=backend,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            min_request_interval_seconds=min_request_interval_seconds,
        )

    def search_artist(self, artist_name: str) -> SpotifyArtistMetadata | None:
        query = artist_name.strip()
        if not query:
            return None

        cache_key = query.casefold()
        if cache_key in self._artist_cache:
            return self._artist_cache[cache_key]

        items = self._search_items(query=query, item_type="artist", limit=5)
        if not items:
            self._artist_cache[cache_key] = None
            return None

        best_item = self._select_best_artist_match(query, items)
        if best_item is None:
            self._artist_cache[cache_key] = None
            return None

        metadata = self._artist_metadata_from_item(best_item, query=query)
        self._artist_cache[cache_key] = metadata
        return metadata

    def get_artist(self, artist_id_or_uri: str) -> SpotifyArtistMetadata:
        artist_id = parse_spotify_id(artist_id_or_uri, expected_kind="artist")
        if self._spotipy_client is not None:
            payload = self._spotipy_client.artist(artist_id)
            return self._artist_metadata_from_item(payload if isinstance(payload, dict) else {}, query=artist_id)
        payload = self._api_get_json(f"{API_BASE_URL}/artists/{artist_id}")
        return self._artist_metadata_from_item(payload, query=str(payload.get("name", artist_id)))

    def get_related_artists(self, artist_id_or_uri: str, *, limit: int = 10) -> list[SpotifyArtistMetadata]:
        artist_id = parse_spotify_id(artist_id_or_uri, expected_kind="artist")
        if self._spotipy_client is not None:
            payload = self._spotipy_client.artist_related_artists(artist_id)
        else:
            payload = self._api_get_json(f"{API_BASE_URL}/artists/{artist_id}/related-artists")
        rows = payload.get("artists", [])
        if not isinstance(rows, list):
            raise SpotifyPublicCatalogError("Spotify related-artists response was missing an artist list.")
        return [self._artist_metadata_from_item(item, query=str(item.get("name", ""))) for item in rows[: max(1, int(limit))]]

    def get_artist_albums(
        self,
        artist_id_or_uri: str,
        *,
        include_groups: str = "album,single",
        limit: int = 50,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        artist_id = parse_spotify_id(artist_id_or_uri, expected_kind="artist")
        max_items = max(1, int(limit))
        if self._spotipy_client is not None:
            collected: list[dict[str, Any]] = []
            offset = 0
            page_size = min(ARTIST_ALBUM_PAGE_SIZE, max_items)
            while len(collected) < max_items:
                payload = self._spotipy_client.artist_albums(
                    artist_id,
                    include_groups=include_groups,
                    country=(market or self._market).strip().upper(),
                    limit=page_size,
                    offset=offset,
                )
                rows = payload.get("items", []) if isinstance(payload, dict) else []
                if not isinstance(rows, list) or not rows:
                    break
                collected.extend(row for row in rows if isinstance(row, dict))
                offset += len(rows)
                if not payload.get("next"):
                    break
            return collected[:max_items]
        return self._paginate_items(
            f"{API_BASE_URL}/artists/{artist_id}/albums",
            item_key="items",
            params={
                "include_groups": include_groups,
                "limit": min(ARTIST_ALBUM_PAGE_SIZE, max_items),
                "market": (market or self._market).strip().upper(),
            },
            max_items=max_items,
        )

    def get_album(self, album_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        album_id = parse_spotify_id(album_id_or_uri, expected_kind="album")
        if self._spotipy_client is not None:
            payload = self._spotipy_client.album(album_id, market=(market or self._market).strip().upper())
            return payload if isinstance(payload, dict) else {}
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/albums/{album_id}", params))

    def get_album_tracks(
        self,
        album_id_or_uri: str,
        *,
        limit: int = 50,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        album_id = parse_spotify_id(album_id_or_uri, expected_kind="album")
        max_items = max(1, int(limit))
        if self._spotipy_client is not None:
            collected: list[dict[str, Any]] = []
            offset = 0
            page_size = min(50, max_items)
            while len(collected) < max_items:
                payload = self._spotipy_client.album_tracks(
                    album_id,
                    limit=page_size,
                    offset=offset,
                    market=(market or self._market).strip().upper(),
                )
                rows = payload.get("items", []) if isinstance(payload, dict) else []
                if not isinstance(rows, list) or not rows:
                    break
                collected.extend(row for row in rows if isinstance(row, dict))
                offset += len(rows)
                if not payload.get("next"):
                    break
            return collected[:max_items]
        return self._paginate_items(
            f"{API_BASE_URL}/albums/{album_id}/tracks",
            item_key="items",
            params={
                "market": (market or self._market).strip().upper(),
                "limit": min(50, max_items),
            },
            max_items=max_items,
        )

    def get_album_metadata(self, album_id_or_uri: str, *, market: str | None = None) -> SpotifyAlbumMetadata:
        payload = self.get_album(album_id_or_uri, market=market)
        return self._album_metadata_from_item(payload, query=album_id_or_uri)

    def get_track(self, track_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        track_id = parse_spotify_id(track_id_or_uri, expected_kind="track")
        if self._spotipy_client is not None:
            payload = self._spotipy_client.track(track_id, market=(market or self._market).strip().upper())
            return payload if isinstance(payload, dict) else {}
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/tracks/{track_id}", params))

    def get_artist_top_tracks(
        self,
        artist_id_or_uri: str,
        *,
        market: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        artist_id = parse_spotify_id(artist_id_or_uri, expected_kind="artist")
        market_code = (market or self._market).strip().upper()
        if self._spotipy_client is not None:
            payload = self._spotipy_client.artist_top_tracks(artist_id, country=market_code)
        else:
            payload = self._api_get_json(
                self._build_url(f"{API_BASE_URL}/artists/{artist_id}/top-tracks", {"market": market_code})
            )
        rows = payload.get("tracks", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise SpotifyPublicCatalogError("Spotify artist top-tracks response did not include a track list.")
        return [row for row in rows[: max(1, int(limit))] if isinstance(row, dict)]

    def get_new_releases(
        self,
        *,
        market: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        market_code = (market or self._market).strip().upper()
        max_items = max(1, int(limit))
        if self._spotipy_client is not None:
            collected: list[dict[str, Any]] = []
            offset = 0
            page_size = min(50, max_items)
            while len(collected) < max_items:
                payload = self._spotipy_client.new_releases(country=market_code, limit=page_size, offset=offset)
                root = payload.get("albums", {}) if isinstance(payload, dict) else {}
                rows = root.get("items", []) if isinstance(root, dict) else []
                if not isinstance(rows, list) or not rows:
                    break
                collected.extend(row for row in rows if isinstance(row, dict))
                offset += len(rows)
                if not root.get("next"):
                    break
            return collected[:max_items]

        collected: list[dict[str, Any]] = []
        next_url = self._build_url(
            f"{API_BASE_URL}/browse/new-releases",
            {
                "country": market_code,
                "limit": min(50, max_items),
            },
        )
        while next_url and len(collected) < max_items:
            payload = self._api_get_json(next_url)
            root = payload.get("albums", {}) if isinstance(payload, dict) else {}
            rows = root.get("items", []) if isinstance(root, dict) else []
            if not isinstance(rows, list):
                raise SpotifyPublicCatalogError("Spotify new-releases response did not include an album list.")
            for row in rows:
                if isinstance(row, dict):
                    collected.append(row)
                if len(collected) >= max_items:
                    break
            next_raw = root.get("next") if isinstance(root, dict) else ""
            next_url = str(next_raw).strip() if next_raw else ""
        return collected

    def get_playlist(self, playlist_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        playlist_id = parse_spotify_id(playlist_id_or_uri, expected_kind="playlist")
        if self._spotipy_client is not None:
            payload = self._spotipy_client.playlist(playlist_id, market=(market or self._market).strip().upper())
            return payload if isinstance(payload, dict) else {}
        params = {"market": (market or self._market).strip().upper()}
        try:
            return self._api_get_json(self._build_url(f"{API_BASE_URL}/playlists/{playlist_id}", params))
        except SpotifyPublicCatalogError as exc:
            raise self._playlist_error(playlist_id, exc) from exc

    def get_playlist_tracks(
        self,
        playlist_id_or_uri: str,
        *,
        limit: int = 100,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        playlist_id = parse_spotify_id(playlist_id_or_uri, expected_kind="playlist")
        max_items = max(1, int(limit))
        if self._spotipy_client is not None:
            collected: list[dict[str, Any]] = []
            offset = 0
            page_size = min(100, max_items)
            while len(collected) < max_items:
                payload = self._spotipy_client.playlist_items(
                    playlist_id,
                    limit=page_size,
                    offset=offset,
                    market=(market or self._market).strip().upper(),
                    additional_types=("track",),
                )
                rows = payload.get("items", []) if isinstance(payload, dict) else []
                if not isinstance(rows, list) or not rows:
                    break
                collected.extend(row for row in rows if isinstance(row, dict))
                offset += len(rows)
                if not payload.get("next"):
                    break
            return collected[:max_items]
        try:
            return self._paginate_items(
                f"{API_BASE_URL}/playlists/{playlist_id}/items",
                item_key="items",
                params={
                    "market": (market or self._market).strip().upper(),
                    "limit": min(100, max_items),
                },
                max_items=max_items,
            )
        except SpotifyPublicCatalogError as exc:
            raise self._playlist_error(playlist_id, exc) from exc

    def get_available_markets(self) -> list[str]:
        payload = self._api_get_json(f"{API_BASE_URL}/markets")
        rows = payload.get("markets", [])
        if not isinstance(rows, list):
            raise SpotifyPublicCatalogError("Spotify available-markets response did not include a market list.")
        return [str(item).strip().upper() for item in rows if str(item).strip()]

    def get_show(self, show_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        show_id = parse_spotify_id(show_id_or_uri, expected_kind="show")
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/shows/{show_id}", params))

    def get_show_episodes(
        self,
        show_id_or_uri: str,
        *,
        limit: int = 20,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        show_id = parse_spotify_id(show_id_or_uri, expected_kind="show")
        return self._paginate_items(
            f"{API_BASE_URL}/shows/{show_id}/episodes",
            item_key="items",
            params={
                "market": (market or self._market).strip().upper(),
                "limit": min(50, max(1, int(limit))),
            },
            max_items=max(1, int(limit)),
        )

    def get_episode(self, episode_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        episode_id = parse_spotify_id(episode_id_or_uri, expected_kind="episode")
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/episodes/{episode_id}", params))

    def get_audiobook(self, audiobook_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        audiobook_id = parse_spotify_id(audiobook_id_or_uri, expected_kind="audiobook")
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/audiobooks/{audiobook_id}", params))

    def search(
        self,
        *,
        query: str,
        item_types: list[str],
        limit: int = 10,
        market: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        cleaned_types = [item.strip() for item in item_types if item.strip()]
        if not cleaned_types:
            raise ValueError("At least one Spotify search item type is required.")

        if self._spotipy_client is not None:
            payload = self._spotipy_client.search(
                q=query.strip(),
                type=",".join(cleaned_types),
                limit=max(1, int(limit)),
                market=(market or self._market).strip().upper(),
            )
            payload = payload if isinstance(payload, dict) else {}
        else:
            payload = self._api_get_json(
                self._build_url(
                    SEARCH_URL,
                    {
                        "q": query.strip(),
                        "type": ",".join(cleaned_types),
                        "limit": max(1, int(limit)),
                        "market": (market or self._market).strip().upper(),
                    },
                )
            )
        results: dict[str, list[dict[str, Any]]] = {}
        for item_type in cleaned_types:
            root = payload.get(f"{item_type}s", {})
            rows = root.get("items", []) if isinstance(root, dict) else []
            results[item_type] = rows if isinstance(rows, list) else []
        return results

    def _build_spotipy_client(self) -> Any:
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
        except ImportError as exc:
            raise SpotifyPublicCatalogError("Spotipy backend requested, but `spotipy` is not installed.") from exc

        auth_manager = SpotifyClientCredentials(
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        return spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=self._timeout_seconds,
        )

    def _search_items(self, *, query: str, item_type: str, limit: int) -> list[dict[str, Any]]:
        if self._spotipy_client is not None:
            payload = self._spotipy_client.search(
                q=query,
                type=item_type,
                limit=max(1, int(limit)),
                market=self._market,
            )
            payload = payload if isinstance(payload, dict) else {}
        else:
            payload = self._api_get_json(
                self._build_url(
                    SEARCH_URL,
                    {
                        "q": query,
                        "type": item_type,
                        "limit": max(1, int(limit)),
                        "market": self._market,
                    },
                )
            )
        root = payload.get(f"{item_type}s", {})
        rows = root.get("items", []) if isinstance(root, dict) else []
        if not isinstance(rows, list):
            raise SpotifyPublicCatalogError(f"Spotify search response for {item_type} did not include an item list.")
        return rows

    def _select_best_artist_match(self, query: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        normalized_query = _normalize_artist_name(query)
        if not normalized_query:
            return items[0] if items else None

        def sort_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
            name = str(item.get("name", ""))
            normalized_name = _normalize_artist_name(name)
            exact_match = int(normalized_name == normalized_query)
            contains_match = int(normalized_query in normalized_name or normalized_name in normalized_query)
            popularity = self._optional_int(item.get("popularity")) or 0
            followers = self._optional_int(item.get("followers", {}).get("total")) or 0
            return (exact_match, contains_match, popularity, followers)

        return max(items, key=sort_key, default=None)

    def _artist_metadata_from_item(self, item: dict[str, Any], *, query: str) -> SpotifyArtistMetadata:
        return SpotifyArtistMetadata(
            query=query,
            spotify_id=str(item.get("id", "")).strip(),
            name=str(item.get("name", query)).strip() or query,
            spotify_url=str(item.get("external_urls", {}).get("spotify", "")).strip(),
            genres=[str(value).strip() for value in item.get("genres", []) if str(value).strip()],
            popularity=self._optional_int(item.get("popularity")),
            followers_total=self._optional_int(item.get("followers", {}).get("total")),
            image_url=self._first_image_url(item.get("images")),
        )

    def _album_metadata_from_item(self, item: dict[str, Any], *, query: str) -> SpotifyAlbumMetadata:
        external_ids = item.get("external_ids", {}) if isinstance(item.get("external_ids"), dict) else {}
        restrictions = item.get("restrictions", {}) if isinstance(item.get("restrictions"), dict) else {}
        restriction_reason = str(restrictions.get("reason", "")).strip()
        available_markets = item.get("available_markets", [])
        return SpotifyAlbumMetadata(
            query=query,
            spotify_id=str(item.get("id", "")).strip(),
            name=str(item.get("name", query)).strip() or query,
            spotify_url=str(item.get("external_urls", {}).get("spotify", "")).strip(),
            album_type=str(item.get("album_type", "")).strip(),
            release_date=str(item.get("release_date", "")).strip(),
            release_date_precision=str(item.get("release_date_precision", "")).strip(),
            total_tracks=self._optional_int(item.get("total_tracks")),
            label=str(item.get("label", "")).strip() or None,
            upc=str(external_ids.get("upc", "")).strip() or None,
            ean=str(external_ids.get("ean", "")).strip() or None,
            artists=[
                str(artist.get("name", "")).strip()
                for artist in item.get("artists", [])
                if isinstance(artist, dict) and str(artist.get("name", "")).strip()
            ],
            image_url=self._first_image_url(item.get("images")),
            available_markets_count=len(available_markets) if isinstance(available_markets, list) else None,
            restriction_reasons=[restriction_reason] if restriction_reason else [],
        )

    def _paginate_items(
        self,
        base_url: str,
        *,
        item_key: str,
        params: dict[str, Any],
        max_items: int,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        next_url = self._build_url(base_url, params)
        while next_url and len(collected) < max_items:
            payload = self._api_get_json(next_url)
            rows = payload.get(item_key, [])
            if not isinstance(rows, list):
                raise SpotifyPublicCatalogError("Spotify paginated response did not include an item list.")
            for row in rows:
                if isinstance(row, dict):
                    collected.append(row)
                if len(collected) >= max_items:
                    break
            next_raw = payload.get("next")
            next_url = str(next_raw).strip() if next_raw else ""
        return collected

    def _build_url(self, base_url: str, params: dict[str, Any]) -> str:
        clean_params = {key: value for key, value in params.items() if value not in (None, "")}
        if not clean_params:
            return base_url
        return f"{base_url}?{urlencode(clean_params)}"

    def _api_get_json(self, url: str) -> dict[str, Any]:
        if url in self._json_cache:
            return self._json_cache[url]
        cached_payload = self._read_cached_json(url)
        if cached_payload is not None:
            self._json_cache[url] = cached_payload
            return cached_payload

        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self._get_access_token()}",
                "Accept": "application/json",
            },
            method="GET",
        )
        payload = self._request_json(request)
        self._json_cache[url] = payload
        self._write_cached_json(url, payload)
        return payload

    def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._access_token_expires_at:
            return self._access_token

        credentials = f"{self._client_id}:{self._client_secret}".encode("utf-8")
        encoded_credentials = base64.b64encode(credentials).decode("ascii")
        request = Request(
            TOKEN_URL,
            data=urlencode({"grant_type": "client_credentials"}).encode("utf-8"),
            headers={
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        payload = self._request_json(request)
        token = str(payload.get("access_token", "")).strip()
        expires_in = float(payload.get("expires_in", 0) or 0)
        if not token or expires_in <= 0:
            raise SpotifyPublicCatalogError("Spotify token response did not include a usable access token.")

        self._access_token = token
        self._access_token_expires_at = now + max(1.0, expires_in - 30.0)
        return token

    def _request_json(self, request: Request) -> dict[str, Any]:
        raw_body = ""
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                self._wait_for_request_slot()
                with urlopen(request, timeout=self._timeout_seconds) as response:
                    raw_body = response.read().decode("utf-8")
                self._last_request_at = time.time()
                break
            except HTTPError as exc:
                self._last_request_at = time.time()
                if int(exc.code) == 429 and attempt < RATE_LIMIT_MAX_RETRIES:
                    time.sleep(self._retry_after_seconds(exc))
                    continue
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else exc.reason
                raise SpotifyPublicCatalogError(
                    f"Spotify Web API request failed with HTTP {exc.code}: {detail}",
                    status_code=int(exc.code),
                ) from exc
            except URLError as exc:
                self._last_request_at = time.time()
                raise SpotifyPublicCatalogError(f"Spotify Web API request failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise SpotifyPublicCatalogError("Spotify Web API returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise SpotifyPublicCatalogError("Spotify Web API returned an unexpected response shape.")
        return payload

    def _wait_for_request_slot(self) -> None:
        if self._min_request_interval_seconds <= 0 or self._last_request_at <= 0:
            return
        elapsed = time.time() - self._last_request_at
        sleep_seconds = self._min_request_interval_seconds - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _retry_after_seconds(self, exc: HTTPError) -> float:
        raw_retry_after = ""
        headers = getattr(exc, "headers", None)
        if headers is not None:
            raw_retry_after = str(headers.get("Retry-After", "")).strip()
        try:
            seconds = float(raw_retry_after)
        except ValueError:
            seconds = RATE_LIMIT_DEFAULT_SLEEP_SECONDS
        return min(RATE_LIMIT_MAX_SLEEP_SECONDS, max(RATE_LIMIT_DEFAULT_SLEEP_SECONDS, seconds))

    def _cache_file_for_url(self, url: str) -> Path | None:
        if self._cache_dir is None or self._cache_ttl_seconds <= 0:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _read_cached_json(self, url: str) -> dict[str, Any] | None:
        cache_path = self._cache_file_for_url(url)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("url") != url:
            return None
        created_at = _float_setting(payload.get("created_at"), default=0.0)
        if created_at <= 0 or time.time() - created_at > self._cache_ttl_seconds:
            return None
        cached_payload = payload.get("payload")
        return cached_payload if isinstance(cached_payload, dict) else None

    def _write_cached_json(self, url: str, payload: dict[str, Any]) -> None:
        cache_path = self._cache_file_for_url(url)
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps({"url": url, "created_at": time.time(), "payload": payload}, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(cache_path)
        except Exception:
            return

    def _playlist_error(self, playlist_id: str, exc: SpotifyPublicCatalogError) -> SpotifyPublicCatalogError:
        if exc.status_code not in {400, 404}:
            return exc
        return SpotifyPublicCatalogError(
            "Spotify playlist lookup failed. Use a real public or app-accessible playlist URL, URI, or base62 ID; "
            f"`{playlist_id}` was not found or is not accessible to these client credentials. "
            "If you copied an example command, replace placeholders such as REAL_ID_HERE or <playlist-url>. "
            f"Original error: {exc}",
            status_code=exc.status_code,
        )

    def _first_image_url(self, images: Any) -> str | None:
        if not isinstance(images, list):
            return None
        for item in images:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if url:
                return url
        return None

    def _optional_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
