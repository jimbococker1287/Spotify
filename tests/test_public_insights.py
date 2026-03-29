from __future__ import annotations

import argparse
import pandas as pd

from spotify.public_insights import (
    _build_client,
    _cross_media_graph_payload,
    _cross_media_history_frame,
    _playlist_diff,
    _release_state_rows,
    _top_artists_from_history,
    _top_tracks_from_history,
)


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


def test_cross_media_history_frame_classifies_music_podcast_and_audiobook_rows() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T10:00:00Z",
                "2026-03-01T10:05:00Z",
                "2026-03-01T10:10:00Z",
                "2026-03-02T12:00:00Z",
            ],
            "master_metadata_album_artist_name": ["Artist A", None, None, "Artist A"],
            "master_metadata_track_name": ["Song A", None, None, "Song B"],
            "spotify_track_uri": ["spotify:track:aaa", None, None, "spotify:track:bbb"],
            "episode_show_name": [None, "Deep Cuts", None, None],
            "episode_name": [None, "Episode 7", None, None],
            "spotify_episode_uri": [None, "spotify:episode:ep7", None, None],
            "audiobook_title": [None, None, "Book Club", None],
            "audiobook_chapter_title": [None, None, "Chapter 1", None],
            "audiobook_uri": [None, None, "spotify:audiobook:book1", None],
            "audiobook_chapter_uri": [None, None, "spotify:chapter:ch1", None],
            "ms_played": [180000, 240000, 600000, 200000],
        }
    )

    frame = _cross_media_history_frame(history, lookback_days=30, session_gap_minutes=30)

    assert frame["media_family"].tolist() == ["music", "podcast", "audiobook", "music"]
    assert frame["node_type"].tolist() == ["artist", "show", "audiobook", "artist"]
    assert frame["node_name"].tolist() == ["Artist A", "Deep Cuts", "Book Club", "Artist A"]
    assert frame["item_name"].tolist() == ["Song A", "Episode 7", "Chapter 1", "Song B"]
    assert frame["session_id"].tolist() == [0, 0, 0, 1]


def test_cross_media_history_frame_supports_alternate_podcast_columns() -> None:
    history = pd.DataFrame(
        {
            "ts": ["2026-03-01T10:00:00Z"],
            "show_name": ["Deep Cuts"],
            "episode_title": ["Episode 7"],
            "episode_uri": ["spotify:episode:ep7"],
            "ms_played": [240000],
        }
    )

    frame = _cross_media_history_frame(history, lookback_days=30, session_gap_minutes=30)

    assert frame["media_family"].tolist() == ["podcast"]
    assert frame["node_name"].tolist() == ["Deep Cuts"]
    assert frame["item_name"].tolist() == ["Episode 7"]
    assert frame["item_uri"].tolist() == ["spotify:episode:ep7"]


def test_cross_media_graph_payload_summarizes_mixed_sessions_and_transitions() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T10:00:00Z",
                "2026-03-01T10:05:00Z",
                "2026-03-01T10:10:00Z",
                "2026-03-02T12:00:00Z",
            ],
            "master_metadata_album_artist_name": ["Artist A", None, None, "Artist A"],
            "master_metadata_track_name": ["Song A", None, None, "Song B"],
            "spotify_track_uri": ["spotify:track:aaa", None, None, "spotify:track:bbb"],
            "episode_show_name": [None, "Deep Cuts", None, None],
            "episode_name": [None, "Episode 7", None, None],
            "spotify_episode_uri": [None, "spotify:episode:ep7", None, None],
            "audiobook_title": [None, None, "Book Club", None],
            "audiobook_chapter_title": [None, None, "Chapter 1", None],
            "audiobook_uri": [None, None, "spotify:audiobook:book1", None],
            "audiobook_chapter_uri": [None, None, "spotify:chapter:ch1", None],
            "ms_played": [180000, 240000, 600000, 200000],
        }
    )
    frame = _cross_media_history_frame(history, lookback_days=30, session_gap_minutes=30)

    payload = _cross_media_graph_payload(frame, node_limit=10, edge_limit=10, session_limit=5)

    assert payload["summary"]["events_analyzed"] == 4
    assert payload["summary"]["sessions_analyzed"] == 2
    assert payload["summary"]["media_families_seen"] == ["audiobook", "music", "podcast"]
    assert payload["summary"]["cross_media_transition_ratio"] == 1.0
    assert payload["summary"]["mixed_session_ratio"] == 0.5
    assert payload["session_intelligence"]["top_nodes_by_media"]["music"][0]["name"] == "Artist A"
    assert payload["session_intelligence"]["mixed_sessions"][0]["media_families"] == ["audiobook", "music", "podcast"]
    assert payload["edges"][0]["source_name"] == "Artist A"
    assert payload["edges"][0]["target_name"] == "Deep Cuts"
    assert payload["edges"][0]["cross_media"] is True


def test_build_client_allows_offline_creator_label_intelligence_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

    client = _build_client(argparse.Namespace(spotify_market="US", command="creator-label-intelligence"))

    assert client.mode == "offline_local_only"
    assert client.search_artist("Artist A") is None
