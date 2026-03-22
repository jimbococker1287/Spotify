from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from spotify.data import (
    CONTEXT_FEATURES,
    _rolling_artist_counts,
    _rolling_artist_counts_multi,
    append_technical_log_features,
    engineer_features,
    prepare_training_data,
)


def test_engineer_features_adds_extended_session_and_transition_columns() -> None:
    logger = logging.getLogger("spotify.test.features")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=12, freq="15min"),
            "master_metadata_album_artist_name": ["A", "B", "A", "C", "A", "B", "A", "C", "B", "A", "C", "A"],
            "platform": ["ios"] * 12,
            "reason_start": ["trackdone"] * 12,
            "reason_end": ["trackdone"] * 12,
            "shuffle": [False, True] * 6,
            "skipped": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "offline": [False] * 12,
            "incognito_mode": [False] * 12,
        }
    )

    engineered = engineer_features(df, max_artists=3, logger=logger)

    expected = {
        "session_repeat_ratio_so_far",
        "recent_skip_rate_5",
        "recent_skip_rate_20",
        "recent_artist_unique_ratio_5",
        "recent_artist_unique_ratio_20",
        "artist_hour_rate_smooth",
        "artist_dow_rate_smooth",
        "prev_artist_transition_rate_smooth",
    }

    assert expected.issubset(set(engineered.columns))
    for column in expected:
        values = engineered[column].to_numpy(dtype="float32", copy=False)
        assert np.isfinite(values).all()
        assert np.min(values) >= 0.0


def test_rolling_artist_counts_supports_integer_artist_labels() -> None:
    ts_seconds = np.array([0, 10, 20, 30, 40, 50], dtype="int64")
    artists = np.array([1, 2, 1, 1, 2, 1], dtype="int32")

    counts = _rolling_artist_counts(ts_seconds, artists, window_seconds=25)

    assert np.allclose(counts, np.array([0, 0, 1, 1, 0, 1], dtype="float32"))


def test_rolling_artist_counts_multi_matches_individual_windows() -> None:
    ts_seconds = np.array([0, 10, 20, 30, 40, 50], dtype="int64")
    artists = np.array([1, 2, 1, 1, 2, 1], dtype="int32")

    counts_25, counts_45 = _rolling_artist_counts_multi(ts_seconds, artists, window_seconds=(25, 45))

    assert np.allclose(counts_25, _rolling_artist_counts(ts_seconds, artists, window_seconds=25))
    assert np.allclose(counts_45, _rolling_artist_counts(ts_seconds, artists, window_seconds=45))


def test_prepare_training_data_builds_expected_sequences(tmp_path) -> None:
    logger = logging.getLogger("spotify.test.prepare-data")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    rows = 10
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "master_metadata_album_artist_name": [f"artist_{idx}" for idx in range(rows)],
            "artist_label": np.arange(rows, dtype="int32"),
            "skipped": np.array([0, 1] * 5, dtype="float32"),
        }
    )
    for idx, column in enumerate(CONTEXT_FEATURES):
        if column in df.columns:
            continue
        df[column] = np.arange(rows, dtype="float32") + float(idx)

    prepared = prepare_training_data(
        df=df,
        sequence_length=3,
        scaler_path=tmp_path / "context_scaler.joblib",
        logger=logger,
    )

    total_sequences = len(prepared.X_seq_train) + len(prepared.X_seq_val) + len(prepared.X_seq_test)
    assert total_sequences == 6
    assert np.array_equal(prepared.X_seq_train[0], np.array([0, 1, 2], dtype="int32"))
    assert int(prepared.y_train[0]) == 3
    assert float(prepared.y_skip_train[0]) == 1.0


def test_append_technical_log_features_zero_fills_when_export_missing(tmp_path) -> None:
    logger = logging.getLogger("spotify.test.tech-features.empty")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
            "platform": ["ios", "osx"],
        }
    )

    result = append_technical_log_features(df.copy(), data_dir=tmp_path, logger=logger)

    technical_columns = [column for column in CONTEXT_FEATURES if column.startswith("tech_")]
    assert technical_columns
    for column in technical_columns:
        assert result[column].tolist() == [0.0, 0.0]


def test_append_technical_log_features_adds_recent_device_context(tmp_path) -> None:
    logger = logging.getLogger("spotify.test.tech-features.join")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    tech_dir = tmp_path / "Spotify Technical Log Information"
    tech_dir.mkdir(parents=True)
    base_ms = 1_767_000_000_000

    payloads = {
        "ConnectionInfo.json": [
            {
                "context_time": base_ms + 3_600_000,
                "context_device_type": "mobile",
                "message_reachability_type": "wlan",
                "message_ipv6_failed": False,
            },
            {
                "context_time": base_ms + 7_200_000,
                "context_device_type": "mobile",
                "message_reachability_type": "none",
                "message_ipv6_failed": True,
            },
            {
                "context_time": base_ms + 7_200_000,
                "context_device_type": "desktop",
                "message_reachability_type": "5g",
                "message_ipv6_failed": False,
            },
        ],
        "AudioStreamingSettingsReport.json": [
            {
                "context_time": base_ms + 3_500_000,
                "context_device_type": "mobile",
                "message_allow_downgrade": True,
                "message_user_selected_play_bitrate_wifi": 320_000,
                "message_user_selected_play_bitrate_cellular": 128_000,
            },
            {
                "context_time": base_ms + 3_500_000,
                "context_device_type": "desktop",
                "message_allow_downgrade": False,
                "message_user_selected_play_bitrate_wifi": 700_000,
                "message_user_selected_play_bitrate_cellular": 700_000,
            },
        ],
        "PlaybackError.json": [
            {
                "context_time": base_ms + 7_100_000,
                "context_device_type": "mobile",
                "message_fatal": True,
            },
            {
                "context_time": base_ms + 7_100_000,
                "context_device_type": "desktop",
                "message_fatal": False,
            },
        ],
        "Stutter.json": [
            {
                "context_time": base_ms + 7_150_000,
                "context_device_type": "mobile",
            }
        ],
        "TrackNotPlayed.json": [
            {
                "context_time": base_ms + 7_050_000,
                "context_device_type": "mobile",
            }
        ],
        "CloudPlaybackPlaybackStats.json": [
            {
                "context_time": base_ms + 7_000_000,
                "message_num_stalls": 2,
            }
        ],
    }
    for name, payload in payloads.items():
        (tech_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    base_ms + 7_300_000,
                    base_ms + 7_300_000,
                ]
                ,
                unit="ms",
                utc=True,
            ),
            "platform": ["ios", "osx"],
        }
    )

    result = append_technical_log_features(df.copy(), data_dir=tmp_path, logger=logger)

    mobile = result.iloc[0]
    desktop = result.iloc[1]

    assert float(mobile["tech_connection_events_1h"]) == 1.0
    assert float(mobile["tech_connection_none_24h"]) == 1.0
    assert float(mobile["tech_ipv6_failures_24h"]) == 1.0
    assert float(mobile["tech_playback_errors_24h"]) == 1.0
    assert float(mobile["tech_playback_fatal_errors_24h"]) == 1.0
    assert float(mobile["tech_stutter_events_24h"]) == 1.0
    assert float(mobile["tech_track_not_played_24h"]) == 1.0
    assert float(mobile["tech_last_reachability_offline"]) == 1.0
    assert float(mobile["tech_last_ipv6_failed"]) == 1.0
    assert float(mobile["tech_allow_downgrade"]) == 1.0
    assert float(mobile["tech_bitrate_wifi_kbps"]) == 320.0
    assert float(mobile["tech_bitrate_cellular_kbps"]) == 128.0

    assert float(desktop["tech_connection_events_1h"]) == 1.0
    assert float(desktop["tech_connection_none_24h"]) == 0.0
    assert float(desktop["tech_ipv6_failures_24h"]) == 0.0
    assert float(desktop["tech_playback_errors_24h"]) == 1.0
    assert float(desktop["tech_playback_fatal_errors_24h"]) == 0.0
    assert float(desktop["tech_stutter_events_24h"]) == 0.0
    assert float(desktop["tech_track_not_played_24h"]) == 0.0
    assert float(desktop["tech_last_reachability_cellular"]) == 1.0
    assert float(desktop["tech_last_reachability_offline"]) == 0.0
    assert float(desktop["tech_last_ipv6_failed"]) == 0.0
    assert float(desktop["tech_allow_downgrade"]) == 0.0
    assert float(desktop["tech_bitrate_wifi_kbps"]) == 700.0
    assert float(desktop["tech_bitrate_cellular_kbps"]) == 700.0

    assert float(mobile["tech_cloud_stats_events_24h"]) == 1.0
    assert float(mobile["tech_cloud_stalls_24h"]) == 2.0
    assert float(desktop["tech_cloud_stats_events_24h"]) == 1.0
    assert float(desktop["tech_cloud_stalls_24h"]) == 2.0
