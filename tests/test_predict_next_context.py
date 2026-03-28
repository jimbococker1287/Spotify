from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import spotify.data as data_module
from spotify.predict_next import load_prediction_input_context


class _StubScaler:
    def __init__(self, mean: list[float], scale: list[float]) -> None:
        self.mean_ = np.asarray(mean, dtype="float32")
        self.scale_ = np.asarray(scale, dtype="float32")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype="float32") - self.mean_.reshape(1, -1)) / self.scale_.reshape(1, -1)


def test_load_prediction_input_context_uses_persistent_cache(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    history_file = data_dir / "Streaming_History_Audio_2024_0.json"
    history_file.write_text("[]", encoding="utf-8")

    (run_dir / "feature_metadata.json").write_text(
        json.dumps(
            {
                "artist_labels": ["Artist A", "Artist B", "Artist C"],
                "context_features": ["hour", "offline", "tech_playback_errors_24h"],
                "skew_context_features": ["tech_playback_errors_24h"],
                "sequence_length": 2,
            }
        ),
        encoding="utf-8",
    )
    joblib.dump(_StubScaler(mean=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]), run_dir / "context_scaler.joblib")

    engineered = pd.DataFrame(
        {
            "ts": [1, 2, 3],
            "artist_label": [0, 1, 2],
            "hour": [9.0, 10.0, 11.0],
            "offline": [0.0, 0.0, 0.0],
            "tech_playback_errors_24h": [0.0, 1.0, 3.0],
        }
    )
    load_calls = {"count": 0}

    def _fake_load_streaming_history(data_dir: Path, include_video: bool, logger: logging.Logger) -> pd.DataFrame:
        _ = (data_dir, include_video, logger)
        load_calls["count"] += 1
        return pd.DataFrame({"ts": [1, 2, 3]})

    monkeypatch.setattr(data_module, "load_streaming_history", _fake_load_streaming_history)
    monkeypatch.setattr(data_module, "engineer_features", lambda df, max_artists, logger, artist_classes: engineered.copy())
    monkeypatch.setattr(data_module, "append_technical_log_features", lambda df, data_dir, logger: df.copy())
    monkeypatch.setattr(data_module, "append_audio_features", lambda df, enable_spotify_features, logger: df.copy())

    logger = logging.getLogger("spotify.test.predict_next_context")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    first = load_prediction_input_context(run_dir=run_dir, data_dir=data_dir, include_video=False, logger=logger)
    second = load_prediction_input_context(run_dir=run_dir, data_dir=data_dir, include_video=False, logger=logger)

    assert load_calls["count"] == 1
    assert first.latest_sequence_names == ["Artist B", "Artist C"]
    assert first.context_raw is not None
    assert first.context_features == ["hour", "offline", "tech_playback_errors_24h"]
    assert float(first.context_raw[0, 2]) == pytest.approx(float(np.log1p(3.0)))
    assert first.friction_reference is not None
    assert float(first.friction_reference["aggregate_threshold"]) > 0.0
    assert second.latest_sequence_names == ["Artist B", "Artist C"]
    assert (run_dir / ".cache" / "prediction_input_context_audio.joblib").exists()

    history_file.write_text("[{}]", encoding="utf-8")
    third = load_prediction_input_context(run_dir=run_dir, data_dir=data_dir, include_video=False, logger=logger)

    assert load_calls["count"] == 2
    assert third.latest_sequence_names == ["Artist B", "Artist C"]
