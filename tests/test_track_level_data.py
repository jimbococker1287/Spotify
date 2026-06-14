from __future__ import annotations

import pandas as pd
import pytest

from spotify.track_level_data import (
    build_track_level_dataset,
    build_track_level_examples,
    split_track_level_examples,
)


def _uri(name: str) -> str:
    return f"spotify:track:{name}"


def test_builds_chronological_session_histories_and_multitask_labels() -> None:
    rows = pd.DataFrame(
        {
            "ts": [
                "2026-01-01T00:20:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T00:10:00Z",
            ],
            "spotify_track_uri": [_uri("a"), _uri("a"), _uri("d"), _uri("b")],
            "skipped": [False, False, None, "true"],
            "reason_end": ["trackdone", "trackdone", "fwdbtn", "fwdbtn"],
            "ms_played": [190_000, 180_000, 1_500, 8_000],
        }
    )

    dataset = build_track_level_dataset(rows)

    assert dataset.source_row_count == 4
    assert dataset.valid_track_row_count == 4
    assert dataset.unique_track_count == 3
    assert dataset.session_count == 2
    assert [example.target_track_uri for example in dataset.examples] == [_uri("b"), _uri("a")]

    first, second = dataset.examples
    assert first.history_track_uris == (_uri("a"),)
    assert first.history_time_gaps_seconds == (0.0,)
    assert first.target_time_gap_seconds == 600.0
    assert first.labels.skipped is True
    assert first.labels.listen_duration_ms == 8_000
    assert first.labels.session_end is False
    assert first.labels.repeat is False

    assert second.history_track_uris == (_uri("a"), _uri("b"))
    assert second.history_time_gaps_seconds == (0.0, 600.0)
    assert second.target_time_gap_seconds == 600.0
    assert second.labels.next_track_uri == _uri("a")
    assert second.labels.session_end is True
    assert second.labels.repeat is True
    assert second.labels.dwell_ms == 190_000


def test_preserves_long_tail_tracks_and_filters_only_invalid_track_rows() -> None:
    rows = pd.DataFrame(
        {
            "ts": pd.date_range("2026-02-01", periods=6, freq="5min", tz="UTC"),
            "spotify_track_uri": [
                _uri("popular"),
                _uri("popular"),
                _uri("tail-one"),
                "",
                None,
                _uri("tail-two"),
            ],
        }
    )

    dataset = build_track_level_dataset(rows)

    assert dataset.valid_track_row_count == 4
    assert dataset.unique_track_count == 3
    assert [example.target_track_uri for example in dataset.examples] == [
        _uri("popular"),
        _uri("tail-one"),
        _uri("tail-two"),
    ]


def test_history_truncation_keeps_aligned_gaps_without_future_information() -> None:
    rows = pd.DataFrame(
        {
            "ts": [
                "2026-03-01T00:00:00Z",
                "2026-03-01T00:01:00Z",
                "2026-03-01T00:03:00Z",
                "2026-03-01T00:06:00Z",
            ],
            "spotify_track_uri": [_uri("a"), _uri("b"), _uri("c"), _uri("d")],
        }
    )

    examples = build_track_level_examples(rows, max_history=2)
    final = examples[-1]

    assert final.history_track_uris == (_uri("b"), _uri("c"))
    assert final.history_time_gaps_seconds == (0.0, 120.0)
    assert final.target_time_gap_seconds == 180.0
    assert final.target_track_uri not in final.history_track_uris


def test_missing_optional_fields_stay_nullable_and_reason_end_can_derive_skip() -> None:
    rows = pd.DataFrame(
        {
            "ts": [
                "2026-04-01T00:00:00Z",
                "2026-04-01T00:01:00Z",
                "2026-04-01T00:02:00Z",
                "2026-04-01T00:03:00Z",
            ],
            "spotify_track_uri": [_uri("a"), _uri("b"), _uri("c"), _uri("d")],
            "reason_end": ["unknown", pd.NA, "trackdone", "fwdbtn"],
            "ms_played": [10_000, -1, None, None],
        }
    )
    original = rows.copy(deep=True)

    examples = build_track_level_examples(rows)

    assert examples[0].labels.skipped is None
    assert examples[0].labels.listen_duration_ms is None
    assert examples[1].labels.skipped is False
    assert examples[1].labels.listen_duration_ms is None
    assert examples[2].labels.skipped is True
    pd.testing.assert_frame_equal(rows, original)


def test_temporal_split_is_deterministic_chronological_and_session_isolated() -> None:
    timestamps: list[pd.Timestamp] = []
    tracks: list[str] = []
    base = pd.Timestamp("2026-05-01T00:00:00Z")
    for session_number in range(5):
        session_start = base + pd.Timedelta(hours=session_number * 2)
        timestamps.extend(session_start + pd.Timedelta(minutes=offset) for offset in (0, 5, 10))
        tracks.extend([_uri(f"{session_number}-a"), _uri(f"{session_number}-b"), _uri(f"{session_number}-c")])
    dataset = build_track_level_dataset(pd.DataFrame({"ts": timestamps, "spotify_track_uri": tracks}))

    first = split_track_level_examples(dataset)
    second = split_track_level_examples(dataset)

    assert first == second
    assert first.train and first.validation and first.test
    train_sessions = {example.session_id for example in first.train}
    validation_sessions = {example.session_id for example in first.validation}
    test_sessions = {example.session_id for example in first.test}
    assert train_sessions.isdisjoint(validation_sessions | test_sessions)
    assert validation_sessions.isdisjoint(test_sessions)
    assert max(example.target_timestamp for example in first.train) < min(
        example.target_timestamp for example in first.validation
    )
    assert max(example.target_timestamp for example in first.validation) < min(
        example.target_timestamp for example in first.test
    )
    assert first.all_examples == dataset.examples


def test_validates_required_columns_and_history_configuration() -> None:
    with pytest.raises(ValueError, match="spotify_track_uri"):
        build_track_level_dataset(pd.DataFrame({"ts": ["2026-01-01"]}))

    rows = pd.DataFrame(
        {
            "ts": ["2026-01-01T00:00:00Z"],
            "spotify_track_uri": [_uri("a")],
        }
    )
    with pytest.raises(ValueError, match="session_gap_minutes"):
        build_track_level_dataset(rows, session_gap_minutes=0)
    with pytest.raises(ValueError, match="min_history"):
        build_track_level_dataset(rows, min_history=2, max_history=1)
