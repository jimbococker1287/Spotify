from __future__ import annotations

import pandas as pd

from spotify.data import append_audio_features


class _DummyLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str, *args, **_kwargs) -> None:
        self.info_messages.append(message % args if args else message)

    def warning(self, message: str, *args, **_kwargs) -> None:
        self.warning_messages.append(message % args if args else message)


def test_append_audio_features_zero_fills_when_disabled() -> None:
    logger = _DummyLogger()
    df = pd.DataFrame({"spotify_track_uri": ["spotify:track:123"]})

    result = append_audio_features(df, enable_spotify_features=False, logger=logger)

    assert result["danceability"].tolist() == [0.0]
    assert result["energy"].tolist() == [0.0]
    assert result["tempo"].tolist() == [0.0]
    assert logger.info_messages
    assert not logger.warning_messages


def test_append_audio_features_zero_fills_and_warns_when_enabled() -> None:
    logger = _DummyLogger()
    df = pd.DataFrame({"spotify_track_uri": ["spotify:track:123"]})

    result = append_audio_features(df, enable_spotify_features=True, logger=logger)

    assert result["danceability"].tolist() == [0.0]
    assert result["energy"].tolist() == [0.0]
    assert result["tempo"].tolist() == [0.0]
    assert logger.warning_messages
    assert "deprecated" in logger.warning_messages[0].lower()
