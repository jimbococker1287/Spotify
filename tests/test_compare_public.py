from __future__ import annotations

import pandas as pd

from datetime import date

from spotify.compare_public import (
    build_daily_public_similarity_mart,
    build_daily_spotify_wrapped_comparison,
    build_genre_proxy_comparison,
    build_public_comparison,
    build_spotify_wrapped_comparison,
)
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


def test_spotify_wrapped_comparison_aligns_dates_media_and_scopes() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2024-12-31T23:59:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-02-03T00:00:00Z",
                "2025-03-03T00:00:00Z",
                "2025-11-16T00:00:00Z",
            ],
            "ms_played": [60_000, 120_000, 180_000, 240_000, 300_000, 60_000],
            "master_metadata_album_artist_name": [
                "Taylor Swift",
                "Taylor Swift",
                "Kendrick Lamar",
                "Phoebe Bridgers",
                None,
                "Taylor Swift",
            ],
            "master_metadata_track_name": [
                "Old",
                "Custom Song",
                "luther (with sza)",
                "Distinctive Song",
                None,
                "Late",
            ],
            "episode_show_name": [None, None, None, None, "The Daily", None],
            "episode_name": [None, None, None, None, "A Show", None],
        }
    )

    result = build_spotify_wrapped_comparison(history, edition=2025, top_n=10)

    assert result["history_window"]["selected_events"] == 4
    assert result["history_window"]["selected_start_date"] == "2025-01-01"
    assert result["history_window"]["selected_end_date"] == "2025-11-15"
    assert result["personal_media_mix"]["podcast"]["duration_minutes"] == 5.0

    global_artists = result["scope_comparisons"]["global"]["dimensions"]["artists"]
    us_artists = result["scope_comparisons"]["united_states"]["dimensions"]["artists"]
    us_tracks = result["scope_comparisons"]["united_states"]["dimensions"]["tracks"]
    us_podcasts = result["scope_comparisons"]["united_states"]["dimensions"]["podcasts"]

    assert global_artists["metrics"]["shared_top_count"] == 2
    assert us_artists["metrics"]["shared_top_count"] == 2
    assert us_tracks["metrics"]["shared_top_count"] == 1
    assert us_tracks["shared_top"][0]["name"] == "luther (with sza)"
    assert us_podcasts["metrics"]["duration_share_on_public_top"] == 1.0
    assert us_artists["monthly_alignment"][0]["month"] == "2025-01"
    assert result["relative_scope_summary"]["closest_scope"] in {"global", "united_states"}


def test_genre_proxy_reports_personal_share_and_tag_chart_overlap() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-04T00:00:00Z",
            ],
            "ms_played": [120_000, 180_000, 300_000],
            "master_metadata_album_artist_name": ["Kendrick Lamar", "Kendrick Lamar", "Bon Iver"],
            "master_metadata_track_name": ["A", "B", "C"],
        }
    )
    public_rows = [
        LastFmArtistChartRow(1, "Kendrick Lamar", 100, 90, "https://example.test/kendrick"),
        LastFmArtistChartRow(2, "Drake", 80, 70, "https://example.test/drake"),
    ]

    result = build_genre_proxy_comparison(
        history,
        genre="hip-hop",
        public_rows=public_rows,
        artist_tags={"Kendrick Lamar": ["Hip-Hop", "Rap"], "Bon Iver": ["indie"]},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 11, 15),
        top_n=10,
    )

    assert result["status"] == "ok"
    assert result["personal_genre_share"]["music_event_share"] == 2 / 3
    assert result["personal_genre_share"]["music_duration_share"] == 0.5
    assert result["shared_public_tag_artist_count"] == 1
    assert result["personal_top_genre_artists"][0]["artist_name"] == "Kendrick Lamar"


def test_daily_spotify_wrapped_comparison_covers_every_active_date() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2024-12-31T23:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T01:00:00Z",
                "2025-11-16T00:00:00Z",
            ],
            "ms_played": [60_000, 120_000, 180_000, 240_000],
            "master_metadata_album_artist_name": [
                "Drake",
                "Drake",
                None,
                "Phoebe Bridgers",
            ],
            "master_metadata_track_name": ["Old", "Custom", None, "Late"],
            "episode_show_name": [None, None, "The Daily", None],
            "episode_name": [None, None, "Episode", None],
        }
    )

    daily = build_daily_spotify_wrapped_comparison(history)

    assert len(daily) == 18
    assert set(daily["listening_date"]) == {"2024-12-31", "2025-01-02", "2025-11-16"}
    assert set(daily["reference_scope"]) == {"global", "united_states"}
    assert set(daily["dimension"]) == {"artists", "tracks", "podcasts"}
    assert set(daily.loc[daily["listening_date"] == "2024-12-31", "reference_alignment"]) == {"historical_projection"}
    assert set(daily.loc[daily["listening_date"] == "2025-01-02", "reference_alignment"]) == {"date_aligned"}
    assert set(daily.loc[daily["listening_date"] == "2025-11-16", "reference_alignment"]) == {"post_window_projection"}

    us_artist = daily[
        (daily["listening_date"] == "2025-01-02")
        & (daily["reference_scope"] == "united_states")
        & (daily["dimension"] == "artists")
    ].iloc[0]
    assert us_artist["event_count"] == 1
    assert us_artist["event_share_on_public_top"] == 1.0
    assert us_artist["personal_top_entity"] == "Drake"
    assert us_artist["personal_top_entity_public_rank"] == 2

    global_podcast = daily[
        (daily["listening_date"] == "2025-01-02")
        & (daily["reference_scope"] == "global")
        & (daily["dimension"] == "podcasts")
    ].iloc[0]
    assert global_podcast["event_count"] == 1
    assert global_podcast["duration_minutes"] == 3.0

    mart = build_daily_public_similarity_mart(daily)
    assert len(mart) == 9
    assert set(mart["closer_scope"]).issubset({"global", "united_states", "tie"})
