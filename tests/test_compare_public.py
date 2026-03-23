from __future__ import annotations

import pandas as pd

from spotify.compare_public import build_public_comparison
from spotify.lastfm import LastFmArtistChartRow


def test_build_public_comparison_reports_overlap_and_distinctive_artists() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T00:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-03T00:00:00Z",
                "2026-03-04T00:00:00Z",
                "2026-03-05T00:00:00Z",
            ],
            "master_metadata_album_artist_name": [
                "Taylor Swift",
                "Taylor Swift",
                "Phoebe Bridgers",
                "Bon Iver",
                "Bon Iver",
            ],
        }
    )
    public_rows = [
        LastFmArtistChartRow(rank=1, name="Taylor Swift", playcount=1000, listeners=900, url="https://a"),
        LastFmArtistChartRow(rank=2, name="Kendrick Lamar", playcount=900, listeners=800, url="https://b"),
        LastFmArtistChartRow(rank=3, name="Bon Iver", playcount=800, listeners=700, url="https://c"),
    ]

    result = build_public_comparison(
        history_df=history,
        public_rows=public_rows,
        lookback_days=30,
        top_n=3,
        scope="country",
        country="United States",
    )

    assert result["baseline"]["provider"] == "Last.fm"
    assert result["overlap"]["shared_artist_count"] == 2
    assert result["overlap"]["recent_play_share_on_public_artists"] == 0.8
    assert [row["artist_name"] for row in result["shared_artists"]] == ["Taylor Swift", "Bon Iver"]
    assert result["your_distinctive_artists"] == [{"artist_name": "Phoebe Bridgers", "your_plays": 1}]
    assert [row["artist_name"] for row in result["public_artists_new_to_you"]] == ["Kendrick Lamar"]
