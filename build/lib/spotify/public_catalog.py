from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1"
SEARCH_URL = f"{API_BASE_URL}/search"


class SpotifyPublicCatalogError(RuntimeError):
    pass


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


def _normalize_artist_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def parse_spotify_id(value: str, *, expected_kind: str | None = None) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Spotify identifier is required.")

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
    ) -> None:
        self._client_id = client_id.strip()
        self._client_secret = client_secret.strip()
        self._market = (market or "US").strip().upper()
        self._timeout_seconds = float(timeout_seconds)
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0
        self._artist_cache: dict[str, SpotifyArtistMetadata | None] = {}
        self._json_cache: dict[str, dict[str, Any]] = {}

        if not self._client_id or not self._client_secret:
            raise ValueError("Spotify client credentials are required.")

    @classmethod
    def from_env(
        cls,
        *,
        market: str = "US",
        timeout_seconds: float = 10.0,
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
        payload = self._api_get_json(f"{API_BASE_URL}/artists/{artist_id}")
        return self._artist_metadata_from_item(payload, query=str(payload.get("name", artist_id)))

    def get_related_artists(self, artist_id_or_uri: str, *, limit: int = 10) -> list[SpotifyArtistMetadata]:
        artist_id = parse_spotify_id(artist_id_or_uri, expected_kind="artist")
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
        return self._paginate_items(
            f"{API_BASE_URL}/artists/{artist_id}/albums",
            item_key="items",
            params={
                "include_groups": include_groups,
                "limit": min(50, max(1, int(limit))),
                "market": (market or self._market).strip().upper(),
            },
            max_items=max(1, int(limit)),
        )

    def get_album(self, album_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        album_id = parse_spotify_id(album_id_or_uri, expected_kind="album")
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/albums/{album_id}", params))

    def get_track(self, track_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        track_id = parse_spotify_id(track_id_or_uri, expected_kind="track")
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/tracks/{track_id}", params))

    def get_playlist(self, playlist_id_or_uri: str, *, market: str | None = None) -> dict[str, Any]:
        playlist_id = parse_spotify_id(playlist_id_or_uri, expected_kind="playlist")
        params = {"market": (market or self._market).strip().upper()}
        return self._api_get_json(self._build_url(f"{API_BASE_URL}/playlists/{playlist_id}", params))

    def get_playlist_tracks(
        self,
        playlist_id_or_uri: str,
        *,
        limit: int = 100,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        playlist_id = parse_spotify_id(playlist_id_or_uri, expected_kind="playlist")
        return self._paginate_items(
            f"{API_BASE_URL}/playlists/{playlist_id}/items",
            item_key="items",
            params={
                "market": (market or self._market).strip().upper(),
                "limit": min(100, max(1, int(limit))),
            },
            max_items=max(1, int(limit)),
        )

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

    def _search_items(self, *, query: str, item_type: str, limit: int) -> list[dict[str, Any]]:
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
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else exc.reason
            raise SpotifyPublicCatalogError(f"Spotify Web API request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise SpotifyPublicCatalogError(f"Spotify Web API request failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise SpotifyPublicCatalogError("Spotify Web API returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise SpotifyPublicCatalogError("Spotify Web API returned an unexpected response shape.")
        return payload

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
