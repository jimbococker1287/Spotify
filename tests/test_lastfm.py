from __future__ import annotations

from spotify.lastfm import LastFmArtistChartRow, LastFmClient


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
