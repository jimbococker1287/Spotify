from __future__ import annotations

import argparse
import json
import logging
import pandas as pd

from spotify.public_catalog import SpotifyAlbumMetadata, SpotifyArtistMetadata, SpotifyPublicCatalogError
from spotify.public_insights import (
    _build_parser,
    _build_client,
    _cross_media_graph_payload,
    _cross_media_history_frame,
    _dispatch_command,
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


def test_top_tracks_from_history_uses_most_frequent_metadata_per_uri() -> None:
    history = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T00:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-03T00:00:00Z",
                "2026-03-04T00:00:00Z",
            ],
            "spotify_track_uri": [
                "spotify:track:aaa",
                "spotify:track:aaa",
                "spotify:track:aaa",
                "spotify:track:bbb",
            ],
            "master_metadata_track_name": [
                "Track A",
                "Track A",
                "Track Alt",
                "Track B",
            ],
            "master_metadata_album_artist_name": [
                "Artist A",
                "Artist A",
                "Artist Z",
                "Artist B",
            ],
        }
    )

    result = _top_tracks_from_history(history, lookback_days=30, limit=2)

    assert result[0] == {
        "spotify_track_uri": "spotify:track:aaa",
        "plays": 3,
        "track_name": "Track A",
        "artist_name": "Artist A",
    }


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


def test_build_client_allows_summary_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

    client = _build_client(argparse.Namespace(spotify_market="US", command="summary"))

    assert client.mode == "offline_local_only"


def test_album_profile_exports_get_album_metadata(tmp_path) -> None:
    class FakeClient:
        def get_album(self, album_id, *, market=None):
            assert album_id == "album123"
            assert market == "US"
            return {
                "id": "album123",
                "name": "Night Rooms",
                "tracks": {
                    "items": [
                        {
                            "track_number": 1,
                            "name": "Intro",
                            "duration_ms": 123000,
                            "explicit": False,
                            "external_urls": {"spotify": "https://open.spotify.com/track/t1"},
                            "artists": [{"name": "Artist A"}],
                        }
                    ]
                },
            }

        def get_album_metadata(self, album_id, *, market=None):
            assert album_id == "album123"
            assert market == "US"
            return SpotifyAlbumMetadata(
                query=album_id,
                spotify_id="album123",
                name="Night Rooms",
                spotify_url="https://open.spotify.com/album/album123",
                album_type="album",
                release_date="2026-04-01",
                release_date_precision="day",
                total_tracks=1,
                label="Indie Arc",
                upc="123",
                ean=None,
                artists=["Artist A"],
                image_url="https://i.scdn.co/image/album",
                available_markets_count=2,
                restriction_reasons=[],
            )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "missing-raw"),
            "album-profile",
            "--album",
            "spotify:album:album123",
            "--track-limit",
            "3",
        ]
    )

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = tmp_path / "analysis" / "public_spotify" / "album_profile" / "album_profile_night-rooms.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["album"]["spotify_id"] == "album123"
    assert payload["album"]["label"] == "Indie Arc"
    assert payload["tracks"][0]["name"] == "Intro"
    assert payload["album_summary"]["tracks_loaded"] == 1
    assert payload["album_summary"]["coverage_ratio"] == 0.0
    assert payload["recommendations"][0]["action"] == "Start with `Intro`."
    assert payload["warnings"][0]["operation"] == "load-history"
    assert "display/link-out only" in payload["policy_note"]


def test_artist_top_tracks_exports_report(tmp_path) -> None:
    class FakeClient:
        def search_artist(self, artist_name):
            assert artist_name == "Artist A"
            return SpotifyArtistMetadata(
                query=artist_name,
                spotify_id="artist123",
                name="Artist A",
                spotify_url="https://open.spotify.com/artist/artist123",
                genres=["indie"],
                popularity=72,
                followers_total=12345,
                image_url=None,
            )

        def get_artist_top_tracks(self, artist_id, *, market=None, limit=10):
            assert artist_id == "artist123"
            assert market == "US"
            assert limit == 2
            return [
                {
                    "id": "track1",
                    "name": "First Song",
                    "external_urls": {"spotify": "https://open.spotify.com/track/track1"},
                    "album": {
                        "id": "album1",
                        "name": "First Album",
                        "external_urls": {"spotify": "https://open.spotify.com/album/album1"},
                        "images": [{"url": "https://i.scdn.co/image/album"}],
                    },
                    "artists": [{"name": "Artist A"}],
                    "duration_ms": 180000,
                    "explicit": False,
                    "popularity": 80,
                }
            ]

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "artist-top-tracks",
            "--artists",
            "Artist A",
            "--track-limit",
            "2",
        ]
    )

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = tmp_path / "analysis" / "public_spotify" / "artist_top_tracks" / "artist_top_tracks_artist-a.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["artists"][0]["tracks"][0]["track_name"] == "First Song"
    assert payload["artists"][0]["tracks"][0]["album_image_url"] == "https://i.scdn.co/image/album"
    assert "display/link-out only" in payload["policy_note"]


def test_new_releases_exports_report(tmp_path) -> None:
    class FakeClient:
        def get_new_releases(self, *, market=None, limit=20):
            assert market == "GB"
            assert limit == 3
            return [
                {
                    "id": "album1",
                    "name": "Night Rooms",
                    "album_type": "album",
                    "release_date": "2026-04-24",
                    "release_date_precision": "day",
                    "total_tracks": 11,
                    "artists": [{"name": "Artist A"}],
                    "external_urls": {"spotify": "https://open.spotify.com/album/album1"},
                    "images": [{"url": "https://i.scdn.co/image/album"}],
                }
            ]

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--spotify-market",
            "GB",
            "new-releases",
            "--limit",
            "3",
        ]
    )

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = tmp_path / "analysis" / "public_spotify" / "new_releases" / "new_releases_gb.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["albums"][0]["album_name"] == "Night Rooms"
    assert payload["albums"][0]["artist_names"] == ["Artist A"]
    assert "display/link-out only" in payload["policy_note"]


def test_public_insights_summary_indexes_recent_reports(tmp_path) -> None:
    radar_dir = tmp_path / "analysis" / "public_spotify" / "personal_release_radar"
    radar_dir.mkdir(parents=True)
    (radar_dir / "radar.json").write_text(
        json.dumps(
            {
                "command": "personal-release-radar",
                "priority_releases": [{"album_name": "Fresh One", "artist_name": "Artist A"}],
                "new_releases_since_last_run": [{"album_name": "Fresh One"}],
                "recommendations": [{"priority": "high"}],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (radar_dir / "radar.md").write_text("# Radar\n", encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path), "summary", "--max-reports", "5"])

    result = _dispatch_command(args, object(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = tmp_path / "analysis" / "public_spotify" / "summary" / "public_insights_summary.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["reports_indexed"] == 1
    assert payload["reports"][0]["summary"]["top_release"] == "Fresh One"
    assert payload["open_next"][0]["reason"] == "has 1 recommendation(s)"
    assert "not training features" in payload["policy_note"]


def test_artist_catalog_completeness_exports_history_coverage_report(tmp_path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "Streaming_History_Audio_2026_0.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2026-04-01T00:00:00Z",
                    "spotify_track_uri": "spotify:track:track1",
                    "master_metadata_track_name": "Known Song",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Known Album",
                },
                {
                    "ts": "2026-04-02T00:00:00Z",
                    "spotify_track_uri": "",
                    "master_metadata_track_name": "Name Matched Song",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Known Album",
                },
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def search_artist(self, artist_name):
            assert artist_name == "Artist A"
            return SpotifyArtistMetadata(
                query=artist_name,
                spotify_id="artist123",
                name="Artist A",
                spotify_url="https://open.spotify.com/artist/artist123",
                genres=["indie"],
                popularity=72,
                followers_total=12345,
                image_url=None,
            )

        def get_artist_albums(self, artist_id, *, include_groups=None, limit=50, market=None):
            assert artist_id == "artist123"
            assert include_groups == "album,single"
            assert market == "US"
            return [
                {
                    "id": "album1",
                    "name": "Known Album",
                    "album_type": "album",
                    "release_date": "2026-01-01",
                    "release_date_precision": "day",
                    "total_tracks": 2,
                    "artists": [{"name": "Artist A"}],
                    "external_urls": {"spotify": "https://open.spotify.com/album/album1"},
                    "images": [{"url": "https://i.scdn.co/image/album1"}],
                },
                {
                    "id": "album2",
                    "name": "Unheard Album",
                    "album_type": "single",
                    "release_date": "2026-02-01",
                    "release_date_precision": "day",
                    "total_tracks": 1,
                    "artists": [{"name": "Artist A"}],
                    "external_urls": {"spotify": "https://open.spotify.com/album/album2"},
                    "images": [],
                },
            ]

        def get_album_tracks(self, album_id, *, limit=50, market=None):
            assert market == "US"
            if album_id == "album1":
                return [
                    {
                        "id": "track1",
                        "track_number": 1,
                        "name": "Known Song",
                        "artists": [{"name": "Artist A"}],
                        "external_urls": {"spotify": "https://open.spotify.com/track/track1"},
                    },
                    {
                        "id": "track2",
                        "track_number": 2,
                        "name": "Name Matched Song",
                        "artists": [{"name": "Artist A"}],
                        "external_urls": {"spotify": "https://open.spotify.com/track/track2"},
                    },
                ]
            return [
                {
                    "id": "track3",
                    "track_number": 1,
                    "name": "Unheard Song",
                    "artists": [{"name": "Artist A"}],
                    "external_urls": {"spotify": "https://open.spotify.com/track/track3"},
                }
            ]

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "artist-catalog-completeness",
            "--artists",
            "Artist A",
            "--album-limit",
            "2",
            "--track-limit",
            "10",
        ]
    )

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = (
        tmp_path
        / "analysis"
        / "public_spotify"
        / "artist_catalog_completeness"
        / "artist_catalog_completeness_artist-a.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    artist = payload["artists"][0]
    assert artist["catalog_tracks_loaded"] == 3
    assert artist["catalog_tracks_listened"] == 2
    assert artist["coverage_ratio"] == 2 / 3
    known_album = next(album for album in artist["albums"] if album["album_name"] == "Known Album")
    assert known_album["tracks"][1]["match_source"] == "artist_track_name"
    assert artist["missing_albums"][0]["album_name"] == "Unheard Album"
    assert artist["missing_tracks"][0]["track_name"] == "Unheard Song"
    assert artist["recommendations"][0]["priority"] == "high"
    assert payload["warnings"] == []
    assert "reporting only" in payload["policy_note"]


def test_playlist_intelligence_exports_overlap_and_duplicate_report(tmp_path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "Streaming_History_Audio_2026_0.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2026-04-01T00:00:00Z",
                    "spotify_track_uri": "spotify:track:track1",
                    "master_metadata_track_name": "Known Song",
                    "master_metadata_album_artist_name": "Artist A",
                },
                {
                    "ts": "2026-04-02T00:00:00Z",
                    "spotify_track_uri": "spotify:track:fav",
                    "master_metadata_track_name": "Favorite Missing",
                    "master_metadata_album_artist_name": "Favorite Artist",
                },
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def get_playlist(self, playlist_id, *, market=None):
            assert playlist_id == "playlist123"
            return {
                "name": "Drive Mix",
                "description": "test",
                "owner": {"display_name": "Owner"},
                "followers": {"total": 12},
                "external_urls": {"spotify": "https://open.spotify.com/playlist/playlist123"},
                "tracks": {"total": 3},
            }

        def get_playlist_tracks(self, playlist_id, *, limit=100, market=None):
            assert playlist_id == "playlist123"
            return [
                {
                    "added_at": "2026-04-10T00:00:00Z",
                    "track": {
                        "id": "track1",
                        "name": "Known Song",
                        "artists": [{"name": "Artist A"}],
                        "album": {"id": "album1", "name": "Album A"},
                        "duration_ms": 180000,
                        "explicit": False,
                        "external_urls": {"spotify": "https://open.spotify.com/track/track1"},
                    },
                },
                {
                    "added_at": "2026-04-10T00:00:00Z",
                    "track": {
                        "id": "track1",
                        "name": "Known Song",
                        "artists": [{"name": "Artist A"}],
                        "album": {"id": "album1", "name": "Album A"},
                        "duration_ms": 180000,
                        "explicit": False,
                        "external_urls": {"spotify": "https://open.spotify.com/track/track1"},
                    },
                },
                {
                    "added_at": "2026-04-10T00:00:00Z",
                    "track": {
                        "id": "track2",
                        "name": "New Song",
                        "artists": [{"name": "Artist B"}],
                        "album": {"id": "album2", "name": "Album B"},
                        "duration_ms": 200000,
                        "explicit": True,
                        "external_urls": {"spotify": "https://open.spotify.com/track/track2"},
                    },
                },
            ]

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "playlist-intelligence",
            "--playlist",
            "spotify:playlist:playlist123",
            "--top-n",
            "2",
        ]
    )

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = tmp_path / "analysis" / "public_spotify" / "playlist_intelligence" / "playlist_intelligence_drive-mix.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["tracks_loaded"] == 3
    assert payload["summary"]["local_overlap_tracks"] == 2
    assert payload["summary"]["duplicate_track_groups"] == 1
    assert payload["summary"]["explicit_tracks"] == 1
    assert payload["missing_favorite_artists"] == ["Favorite Artist"]
    assert payload["recommendations"][0]["action"] == "Add a track from `Favorite Artist`."


def test_personal_release_radar_ranks_new_unheard_releases(tmp_path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "Streaming_History_Audio_2026_0.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2026-04-01T00:00:00Z",
                    "spotify_track_uri": "spotify:track:old",
                    "master_metadata_track_name": "Old Song",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Already Heard",
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def search_artist(self, artist_name):
            return SpotifyArtistMetadata(
                query=artist_name,
                spotify_id="artist123",
                name="Artist A",
                spotify_url="https://open.spotify.com/artist/artist123",
                genres=[],
                popularity=70,
                followers_total=1000,
                image_url=None,
            )

        def get_related_artists(self, artist_id, *, limit=10):
            return [
                SpotifyArtistMetadata(
                    query="Related A",
                    spotify_id="related123",
                    name="Related A",
                    spotify_url="https://open.spotify.com/artist/related123",
                    genres=[],
                    popularity=40,
                    followers_total=500,
                    image_url=None,
                )
            ]

        def get_artist_albums(self, artist_id, *, include_groups=None, limit=50, market=None):
            if artist_id == "artist123":
                return [
                    {
                        "id": "album-new",
                        "name": "Fresh Unheard",
                        "album_type": "album",
                        "release_date": "2026-04-20",
                        "release_date_precision": "day",
                        "total_tracks": 10,
                        "artists": [{"name": "Artist A"}],
                        "external_urls": {"spotify": "https://open.spotify.com/album/album-new"},
                    },
                    {
                        "id": "album-heard",
                        "name": "Already Heard",
                        "album_type": "single",
                        "release_date": "2026-04-18",
                        "release_date_precision": "day",
                        "total_tracks": 1,
                        "artists": [{"name": "Artist A"}],
                        "external_urls": {"spotify": "https://open.spotify.com/album/album-heard"},
                    },
                ]
            return [
                {
                    "id": "album-related",
                    "name": "Related Fresh",
                    "album_type": "single",
                    "release_date": "2026-04-19",
                    "release_date_precision": "day",
                    "total_tracks": 1,
                    "artists": [{"name": "Related A"}],
                    "external_urls": {"spotify": "https://open.spotify.com/album/album-related"},
                }
            ]

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "personal-release-radar",
            "--artists",
            "Artist A",
            "--since-days",
            "30",
            "--as-of-date",
            "2026-05-01",
            "--include-related",
            "--related-limit",
            "1",
        ]
    )

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = (
        tmp_path
        / "analysis"
        / "public_spotify"
        / "personal_release_radar"
        / "personal_release_radar_explicit-artist-a.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["priority_releases"][0]["album_name"] == "Fresh Unheard"
    assert payload["priority_releases"][0]["artist_source"] == "seed"
    assert any(row["already_heard_album"] for row in payload["priority_releases"])
    assert any(row["artist_source"] == "related" for row in payload["priority_releases"])
    assert payload["new_releases_since_last_run"]
    assert payload["recommendations"][0]["priority"] == "high"


def test_new_releases_degrades_to_warning_when_endpoint_unavailable(tmp_path) -> None:
    class FakeClient:
        def get_new_releases(self, *, market=None, limit=20):
            raise SpotifyPublicCatalogError("Spotify Web API request failed with HTTP 403: forbidden", status_code=403)

    parser = _build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path), "new-releases", "--limit", "3"])

    result = _dispatch_command(args, FakeClient(), logging.getLogger("spotify.test.public_insights"), parser)

    assert result == 0
    report_path = tmp_path / "analysis" / "public_spotify" / "new_releases" / "new_releases_us.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["albums"] == []
    assert payload["warnings"][0]["status_code"] == 403
