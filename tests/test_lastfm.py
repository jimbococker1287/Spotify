from __future__ import annotations

from spotify.lastfm import LastFmArtistChartRow, LastFmClient, LastFmTagRow


def test_from_env_returns_none_when_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)

    assert LastFmClient.from_env() is None


def test_get_top_artists_parses_geo_payload(monkeypatch) -> None:
    client = LastFmClient("demo-key")

    def fake_call(method: str, params: dict[str, object]) -> dict:
        assert method == "geo.gettopartists"
        assert params["country"] == "United States"
        return {
            "topartists": {
                "artist": [
                    {
                        "name": "Taylor Swift",
                        "playcount": "12345",
                        "listeners": "6789",
                        "url": "https://www.last.fm/music/Taylor+Swift",
                        "@attr": {"rank": "1"},
                    }
                ]
            }
        }

    monkeypatch.setattr(client, "_call", fake_call)

    result = client.get_top_artists(limit=50, country="United States")

    assert result == [
        LastFmArtistChartRow(
            rank=1,
            name="Taylor Swift",
            playcount=12345,
            listeners=6789,
            url="https://www.last.fm/music/Taylor+Swift",
        )
    ]


def test_get_tag_top_artists_and_artist_tags(monkeypatch) -> None:
    client = LastFmClient("demo-key")

    def fake_call(method: str, params: dict[str, object]) -> dict:
        if method == "tag.gettopartists":
            assert params["tag"] == "hip-hop"
            return {
                "topartists": {
                    "artist": [
                        {
                            "name": "Kendrick Lamar",
                            "url": "https://www.last.fm/music/Kendrick+Lamar",
                            "@attr": {"rank": "1"},
                        }
                    ]
                }
            }
        assert method == "artist.gettoptags"
        assert params["artist"] == "Kendrick Lamar"
        return {
            "toptags": {
                "tag": [
                    {
                        "name": "Hip-Hop",
                        "count": "100",
                        "url": "https://www.last.fm/tag/hip-hop",
                    }
                ]
            }
        }

    monkeypatch.setattr(client, "_call", fake_call)

    assert client.get_tag_top_artists("hip-hop") == [
        LastFmArtistChartRow(
            rank=1,
            name="Kendrick Lamar",
            playcount=None,
            listeners=None,
            url="https://www.last.fm/music/Kendrick+Lamar",
        )
    ]
    assert client.get_artist_top_tags("Kendrick Lamar") == [
        LastFmTagRow(
            name="Hip-Hop",
            count=100,
            url="https://www.last.fm/tag/hip-hop",
        )
    ]
