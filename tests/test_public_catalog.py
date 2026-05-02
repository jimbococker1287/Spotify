from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from spotify.public_catalog import SpotifyAlbumMetadata, SpotifyArtistMetadata, SpotifyPublicCatalogClient, parse_spotify_id


def test_from_env_returns_none_when_credentials_missing(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

    assert SpotifyPublicCatalogClient.from_env() is None


def test_search_artist_prefers_exact_name_match(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient("client-id", "client-secret")

    def fake_get_json(_url: str) -> dict:
        return {
            "artists": {
                "items": [
                    {
                        "id": "tribute",
                        "name": "Taylor Swift Tribute Band",
                        "genres": ["tribute"],
                        "popularity": 40,
                        "followers": {"total": 10_000},
                        "images": [],
                        "external_urls": {"spotify": "https://open.spotify.com/artist/tribute"},
                    },
                    {
                        "id": "exact",
                        "name": "Taylor Swift",
                        "genres": ["pop"],
                        "popularity": 95,
                        "followers": {"total": 100_000_000},
                        "images": [{"url": "https://i.scdn.co/image/exact"}],
                        "external_urls": {"spotify": "https://open.spotify.com/artist/exact"},
                    },
                ]
            }
        }

    monkeypatch.setattr(client, "_api_get_json", fake_get_json)

    result = client.search_artist("Taylor Swift")

    assert result == SpotifyArtistMetadata(
        query="Taylor Swift",
        spotify_id="exact",
        name="Taylor Swift",
        spotify_url="https://open.spotify.com/artist/exact",
        genres=["pop"],
        popularity=95,
        followers_total=100_000_000,
        image_url="https://i.scdn.co/image/exact",
    )


def test_search_artist_caches_results(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient("client-id", "client-secret")
    calls = {"count": 0}

    def fake_get_json(_url: str) -> dict:
        calls["count"] += 1
        return {
            "artists": {
                "items": [
                    {
                        "id": "exact",
                        "name": "Daft Punk",
                        "genres": ["electronic"],
                        "popularity": 88,
                        "followers": {"total": 5_000_000},
                        "images": [],
                        "external_urls": {"spotify": "https://open.spotify.com/artist/exact"},
                    }
                ]
            }
        }

    monkeypatch.setattr(client, "_api_get_json", fake_get_json)

    first = client.search_artist("Daft Punk")
    second = client.search_artist("daft punk")

    assert first is second
    assert calls["count"] == 1


def test_parse_spotify_id_supports_uri_url_and_raw_id() -> None:
    assert parse_spotify_id("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M", expected_kind="playlist") == "37i9dQZF1DXcBWIGoYBM5M"
    assert (
        parse_spotify_id(
            "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh?si=abc",
            expected_kind="track",
        )
        == "4iV5W9uYEdYUVa79Axb7Rh"
    )
    assert (
        parse_spotify_id(
            "https://open.spotify.com/embed/playlist/37i9dQZF1DXcBWIGoYBM5M?theme=0",
            expected_kind="playlist",
        )
        == "37i9dQZF1DXcBWIGoYBM5M"
    )
    assert parse_spotify_id("2CIMQHirSU0MQqyYHq0eOx") == "2CIMQHirSU0MQqyYHq0eOx"


def test_get_album_metadata_normalizes_get_album_response(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient("client-id", "client-secret")

    def fake_get_json(url: str) -> dict:
        assert url == "https://api.spotify.com/v1/albums/album123?market=GB"
        return {
            "id": "album123",
            "name": "Night Rooms",
            "album_type": "album",
            "release_date": "2026-04-01",
            "release_date_precision": "day",
            "total_tracks": 11,
            "label": "Indie Arc",
            "external_ids": {"upc": "123456789012", "ean": "9876543210987"},
            "external_urls": {"spotify": "https://open.spotify.com/album/album123"},
            "available_markets": ["GB", "US"],
            "restrictions": {"reason": "market"},
            "images": [{"url": "https://i.scdn.co/image/album"}],
            "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
        }

    monkeypatch.setattr(client, "_api_get_json", fake_get_json)

    assert client.get_album_metadata("spotify:album:album123", market="GB") == SpotifyAlbumMetadata(
        query="spotify:album:album123",
        spotify_id="album123",
        name="Night Rooms",
        spotify_url="https://open.spotify.com/album/album123",
        album_type="album",
        release_date="2026-04-01",
        release_date_precision="day",
        total_tracks=11,
        label="Indie Arc",
        upc="123456789012",
        ean="9876543210987",
        artists=["Artist A", "Artist B"],
        image_url="https://i.scdn.co/image/album",
        available_markets_count=2,
        restriction_reasons=["market"],
    )


def test_get_artist_top_tracks_uses_raw_web_api(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient("client-id", "client-secret")

    def fake_get_json(url: str) -> dict:
        assert url == "https://api.spotify.com/v1/artists/artist123/top-tracks?market=GB"
        return {
            "tracks": [
                {"id": "track1", "name": "First"},
                {"id": "track2", "name": "Second"},
            ]
        }

    monkeypatch.setattr(client, "_api_get_json", fake_get_json)

    result = client.get_artist_top_tracks("spotify:artist:artist123", market="GB", limit=1)

    assert result == [{"id": "track1", "name": "First"}]


def test_get_new_releases_reads_nested_album_page(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient("client-id", "client-secret")

    def fake_get_json(url: str) -> dict:
        assert url == "https://api.spotify.com/v1/browse/new-releases?country=US&limit=2"
        return {
            "albums": {
                "items": [
                    {"id": "album1", "name": "First"},
                    {"id": "album2", "name": "Second"},
                ],
                "next": None,
            }
        }

    monkeypatch.setattr(client, "_api_get_json", fake_get_json)

    result = client.get_new_releases(market="US", limit=2)

    assert [row["id"] for row in result] == ["album1", "album2"]


def test_get_artist_albums_caps_page_size_for_api_compatibility(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient("client-id", "client-secret")

    def fake_get_json(url: str) -> dict:
        assert url == (
            "https://api.spotify.com/v1/artists/artist123/albums?"
            "include_groups=album%2Csingle&limit=5&market=US"
        )
        return {"items": [{"id": "album1"}], "next": None}

    monkeypatch.setattr(client, "_api_get_json", fake_get_json)

    assert client.get_artist_albums("artist123", limit=30, market="US") == [{"id": "album1"}]


def test_request_json_retries_spotify_rate_limit(monkeypatch) -> None:
    client = SpotifyPublicCatalogClient(
        "client-id",
        "client-secret",
        timeout_seconds=0.1,
        min_request_interval_seconds=0.0,
    )
    calls = {"count": 0, "slept": []}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(_request, *, timeout):
        assert timeout == 0.1
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                "https://api.spotify.com/v1/test",
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                BytesIO(b"too many"),
            )
        return FakeResponse()

    monkeypatch.setattr("spotify.public_catalog.urlopen", fake_urlopen)
    monkeypatch.setattr("spotify.public_catalog.time.sleep", lambda seconds: calls["slept"].append(seconds))

    assert client._request_json("request") == {"ok": True}
    assert calls == {"count": 2, "slept": [1.0]}


def test_api_get_json_uses_persistent_cache(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return None

        def read(self) -> bytes:
            return b'{"albums": {"items": [{"id": "album1"}]}}'

    def fake_urlopen(_request, *, timeout):
        calls["count"] += 1
        return FakeResponse()

    monkeypatch.setattr("spotify.public_catalog.urlopen", fake_urlopen)

    first = SpotifyPublicCatalogClient(
        "client-id",
        "client-secret",
        cache_dir=cache_dir,
        min_request_interval_seconds=0.0,
    )
    second = SpotifyPublicCatalogClient(
        "client-id",
        "client-secret",
        cache_dir=cache_dir,
        min_request_interval_seconds=0.0,
    )
    monkeypatch.setattr(first, "_get_access_token", lambda: "token")
    monkeypatch.setattr(second, "_get_access_token", lambda: "token")

    url = "https://api.spotify.com/v1/browse/new-releases?country=US&limit=1"

    assert first._api_get_json(url)["albums"]["items"][0]["id"] == "album1"
    assert second._api_get_json(url)["albums"]["items"][0]["id"] == "album1"
    assert calls["count"] == 1
    assert list(cache_dir.glob("*.json"))


def test_parse_spotify_id_rejects_playlist_placeholders() -> None:
    try:
        parse_spotify_id("REAL_ID_HERE", expected_kind="playlist")
    except ValueError as exc:
        assert "placeholder" in str(exc)
        assert "open.spotify.com/playlist" in str(exc)
    else:
        raise AssertionError("Expected placeholder playlist ID to fail")


def test_spotipy_backend_uses_spotipy_artist_top_tracks(monkeypatch) -> None:
    class FakeSpotipyClient:
        def artist_top_tracks(self, artist_id: str, *, country: str) -> dict:
            assert artist_id == "artist123"
            assert country == "IN"
            return {"tracks": [{"id": "track1", "name": "First"}]}

    monkeypatch.setattr(
        SpotifyPublicCatalogClient,
        "_build_spotipy_client",
        lambda self: FakeSpotipyClient(),
    )

    client = SpotifyPublicCatalogClient("client-id", "client-secret", backend="spotipy")

    assert client.get_artist_top_tracks("artist123", market="IN") == [{"id": "track1", "name": "First"}]


def test_spotipy_backend_uses_artist_albums_and_playlist_items(monkeypatch) -> None:
    class FakeSpotipyClient:
        def artist_albums(self, artist_id, *, include_groups=None, country=None, limit=5, offset=0):
            assert artist_id == "artist123"
            assert include_groups == "album,single"
            assert country == "GB"
            assert limit == 5
            assert offset == 0
            return {"items": [{"id": "album1"}], "next": None}

        def playlist(self, playlist_id, *, market=None):
            assert playlist_id == "playlist123"
            assert market == "GB"
            return {"id": "playlist123", "name": "Mix"}

        def playlist_items(self, playlist_id, *, limit=50, offset=0, market=None, additional_types=("track", "episode")):
            assert playlist_id == "playlist123"
            assert market == "GB"
            assert additional_types == ("track",)
            return {"items": [{"track": {"id": "track1"}}], "next": None}

    monkeypatch.setattr(
        SpotifyPublicCatalogClient,
        "_build_spotipy_client",
        lambda self: FakeSpotipyClient(),
    )

    client = SpotifyPublicCatalogClient("client-id", "client-secret", backend="spotipy")

    assert client.get_artist_albums("artist123", market="GB") == [{"id": "album1"}]
    assert client.get_playlist("playlist123", market="GB")["name"] == "Mix"
    assert client.get_playlist_tracks("playlist123", market="GB") == [{"track": {"id": "track1"}}]
