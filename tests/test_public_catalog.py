from __future__ import annotations

from spotify.public_catalog import SpotifyArtistMetadata, SpotifyPublicCatalogClient, parse_spotify_id


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
