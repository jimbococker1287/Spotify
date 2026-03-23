from __future__ import annotations

import pandas as pd

from spotify.public_insights import _playlist_diff, _release_state_rows, _top_artists_from_history, _top_tracks_from_history


def test_top_artists_from_history_uses_recent_window() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2025-01-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-03T00:00:00Z",
            ],
            "master_metadata_album_artist_name": [
                "Old Artist",
                "Taylor Swift",
                "Taylor Swift",
                "Phoebe Bridgers",
            ],
        }
    )

    result = _top_artists_from_history(history, lookback_days=30, limit=2)

    assert result == ["Taylor Swift", "Phoebe Bridgers"]


def test_top_tracks_from_history_groups_by_track_uri() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T00:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-03T00:00:00Z",
            ],
            "spotify_track_uri": [
                "spotify:track:aaa",
                "spotify:track:aaa",
                "spotify:track:bbb",
            ],
            "master_metadata_track_name": [
                "Track A",
                "Track A",
                "Track B",
            ],
            "master_metadata_album_artist_name": [
                "Artist A",
                "Artist A",
                "Artist B",
            ],
        }
    )

    result = _top_tracks_from_history(history, lookback_days=30, limit=2)

    assert result == [
        {
            "spotify_track_uri": "spotify:track:aaa",
            "plays": 2,
            "track_name": "Track A",
            "artist_name": "Artist A",
        },
        {
            "spotify_track_uri": "spotify:track:bbb",
            "plays": 1,
            "track_name": "Track B",
            "artist_name": "Artist B",
        },
    ]


def test_playlist_diff_detects_additions_and_metadata_changes() -> None:
    previous = {
        "snapshot_id": "old",
        "name": "Playlist",
        "description": "before",
        "items": [{"track_name": "Track A", "spotify_url": "https://open.spotify.com/track/a"}],
        "image_urls": ["https://image/old"],
    }
    current = {
        "snapshot_id": "new",
        "name": "Playlist",
        "description": "after",
        "items": [
            {"track_name": "Track A", "spotify_url": "https://open.spotify.com/track/a"},
            {"track_name": "Track B", "spotify_url": "https://open.spotify.com/track/b"},
        ],
        "image_urls": ["https://image/new"],
    }

    diff = _playlist_diff(previous, current)

    assert diff["is_first_snapshot"] is False
    assert [row["track_name"] for row in diff["added_tracks"]] == ["Track B"]
    assert diff["removed_tracks"] == []
    assert "description" in diff["metadata_changes"]
    assert "snapshot_id" in diff["metadata_changes"]


def test_release_state_rows_handles_missing_state() -> None:
    assert _release_state_rows(None) == set()
    assert _release_state_rows({"release_ids": ["a", "b"]}) == {"a", "b"}
