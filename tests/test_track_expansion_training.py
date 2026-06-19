from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spotify.track_expansion_training as training
from spotify.track_expansion_training import (
    OOV_ITEM_ID,
    TrackTrainingConfig,
    encode_track_examples,
    fit_track_vocabulary,
    prepare_track_model_data,
    reconstruct_session_interactions,
    run_bounded_retrieval_benchmarks,
)
from spotify.track_level_data import build_track_level_dataset, split_track_level_examples


def _dataset(session_count: int = 8):
    timestamps: list[pd.Timestamp] = []
    tracks: list[str] = []
    skipped: list[object] = []
    dwell: list[object] = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for session in range(session_count):
        start = base + pd.Timedelta(hours=session * 2)
        session_tracks = ("a", "b", f"tail-{session}", "a")
        for position, track in enumerate(session_tracks):
            timestamps.append(start + pd.Timedelta(minutes=position * 5))
            tracks.append(f"track:{track}")
            skipped.append(None if position == 1 else position == 2)
            dwell.append(None if position == 1 else 30_000 + position * 1_000)
    return build_track_level_dataset(
        pd.DataFrame(
            {
                "ts": timestamps,
                "spotify_track_uri": tracks,
                "skipped": skipped,
                "ms_played": dwell,
            }
        ),
        max_history=16,
    )


def test_reconstruct_session_interactions_does_not_duplicate_histories() -> None:
    dataset = _dataset(3)

    interactions = reconstruct_session_interactions(dataset.examples)

    assert len(interactions) == 12
    assert interactions.groupby("session_id").size().tolist() == [4, 4, 4]
    assert interactions.loc[interactions["session_id"].eq(0), "track_id"].tolist() == [
        "track:a",
        "track:b",
        "track:tail-0",
        "track:a",
    ]


def test_vocabulary_is_fit_on_training_examples_only() -> None:
    dataset = _dataset()
    splits = split_track_level_examples(dataset)

    vocabulary = fit_track_vocabulary(splits.train, max_items=3)
    validation_only = next(
        example.target_track_uri
        for example in splits.validation
        if example.target_track_uri.startswith("track:tail-")
    )

    assert vocabulary.encode("track:a") >= 2
    assert vocabulary.encode(validation_only) == OOV_ITEM_ID


def test_encoding_right_aligns_histories_and_masks_missing_labels() -> None:
    dataset = _dataset()
    splits = split_track_level_examples(dataset)
    data = prepare_track_model_data(
        splits.train,
        splits.validation,
        splits.test,
        max_items=5,
        sequence_length=6,
        max_train_examples=100,
        max_validation_examples=100,
        max_test_examples=100,
    )
    first = splits.train[0]
    encoded = encode_track_examples(
        [first],
        vocabulary=data.vocabulary,
        sequence_length=6,
        context_scaler=data.context_scaler,
        dwell_log_scale=data.dwell_log_scale,
    )

    assert encoded.sequence_ids.shape == (1, 6)
    assert np.count_nonzero(encoded.sequence_ids[0]) == len(first.history_track_uris)
    assert encoded.sequence_ids[0, -1] == data.vocabulary.encode(first.history_track_uris[-1])
    if first.labels.skipped is None:
        assert encoded.sample_weights["skip_output"][0] == 0.0
    assert encoded.sample_weights["explicit_positive_output"][0] == 0.0
    assert encoded.context.shape[1] == 11


def test_bounded_retrieval_benchmarks_report_overall_and_catalog_recall() -> None:
    dataset = _dataset(12)
    splits = split_track_level_examples(dataset)

    results = run_bounded_retrieval_benchmarks(
        splits.train,
        splits.validation,
        k=3,
        evaluation_limit=100,
        cooccurrence_max_items=6,
        cooccurrence_shrinkage=1.0,
        ease_max_items=5,
        ease_l2=2.0,
    )

    assert [row["model_name"] for row in results] == ["session_cooccurrence", "ease"]
    assert all(row["status"] == "complete" for row in results)
    assert all(0.0 <= float(row["recall_at_k"]) <= 1.0 for row in results)
    assert all(0.0 <= float(row["target_catalog_coverage"]) <= 1.0 for row in results)
    assert all(float(row["in_catalog_recall_at_k"]) >= float(row["recall_at_k"]) for row in results)


def test_prepare_track_model_data_validates_item_budget() -> None:
    dataset = _dataset()
    splits = split_track_level_examples(dataset)

    with pytest.raises(ValueError, match="max_items"):
        prepare_track_model_data(
            splits.train,
            splits.validation,
            splits.test,
            max_items=1,
            sequence_length=6,
            max_train_examples=10,
            max_validation_examples=10,
            max_test_examples=10,
        )


def test_stream_retrieval_metrics_applies_history_window() -> None:
    example = _dataset(3).examples[2]
    observed: list[tuple[str, ...]] = []

    class Retriever:
        catalog = ("spotify:track:3",)

        def recommend(self, history, *, k, exclude_seen):
            observed.append(tuple(history))
            return []

    training._stream_retrieval_metrics(
        Retriever(),
        [example],
        k=10,
        limit=1,
        history_window=2,
    )

    assert observed == [example.history_track_uris[-2:]]


class _FakeMeantimeModel:
    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size
        self.batch_sizes: list[int] = []
        self.optimizer = None

    def compile(self, *, optimizer, **_kwargs) -> None:
        self.optimizer = optimizer

    def train_on_batch(self, inputs, _targets, **_kwargs) -> float:
        self.batch_sizes.append(len(inputs[0]))
        return 0.25

    def save(self, _path: Path) -> None:
        return None

    def __call__(self, inputs, *, training: bool):
        assert training is False
        return np.ones((len(inputs[0]), self.vocabulary_size), dtype="float32")


class _FakeMultitaskModel:
    output_names = [
        "next_item_output",
        "skip_output",
        "dwell_output",
        "session_end_output",
        "explicit_positive_output",
        "repeat_output",
    ]

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size
        self.batch_sizes: list[int] = []

    def train_on_batch(self, inputs, _targets, **_kwargs) -> dict[str, float]:
        self.batch_sizes.append(len(inputs[0]))
        return {"loss": 0.5}

    def save(self, _path: Path) -> None:
        return None

    def __call__(self, inputs, *, training: bool):
        assert training is False
        row_count = len(inputs[0])
        return [
            np.ones((row_count, self.vocabulary_size), dtype="float32"),
            *[
                np.full((row_count, 1), 0.5, dtype="float32")
                for _name in self.output_names[1:]
            ],
        ]


def _training_data_and_config(tmp_path: Path):
    dataset = _dataset(12)
    splits = split_track_level_examples(dataset)
    config = TrackTrainingConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        sequence_length=6,
        evaluation_k=3,
        neural_max_items=8,
        max_train_examples=100,
        max_validation_examples=100,
        max_test_examples=100,
        epochs=1,
        batch_size=7,
    )
    data = prepare_track_model_data(
        splits.train,
        splits.validation,
        splits.test,
        max_items=config.neural_max_items,
        sequence_length=config.sequence_length,
        max_train_examples=config.max_train_examples,
        max_validation_examples=config.max_validation_examples,
        max_test_examples=config.max_test_examples,
    )
    return data, config


def test_train_meantime_merges_optuna_params_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data, config = _training_data_and_config(tmp_path)
    builder_calls: list[dict[str, object]] = []
    models: list[_FakeMeantimeModel] = []

    def fake_builder(**kwargs):
        builder_calls.append(dict(kwargs["params"]))
        model = _FakeMeantimeModel(int(kwargs["vocabulary_size"]))
        models.append(model)
        return model

    monkeypatch.setattr(
        "spotify.meantime_model.build_meantime_model",
        fake_builder,
    )

    training._train_meantime(
        data,
        config=config,
        checkpoint_dir=tmp_path,
        model_params={
            "embedding_dim": 64,
            "learning_rate": 2e-4,
            "batch_size": 2,
        },
    )
    training._train_meantime(
        data,
        config=replace(config, batch_size=5),
        checkpoint_dir=tmp_path,
    )

    assert builder_calls[0]["embedding_dim"] == 64
    assert builder_calls[0]["num_heads"] == 2
    assert "learning_rate" not in builder_calls[0]
    assert "batch_size" not in builder_calls[0]
    assert max(models[0].batch_sizes) <= 2
    assert float(models[0].optimizer.learning_rate.numpy()) == pytest.approx(2e-4)
    assert builder_calls[1]["embedding_dim"] == 32
    assert max(models[1].batch_sizes) <= 5
    assert float(models[1].optimizer.learning_rate.numpy()) == pytest.approx(1e-3)


def test_train_multitask_merges_optuna_params_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data, config = _training_data_and_config(tmp_path)
    builder_calls: list[dict[str, object]] = []
    models: list[_FakeMultitaskModel] = []

    def fake_builder(**kwargs):
        builder_calls.append(dict(kwargs["params"]))
        model = _FakeMultitaskModel(int(kwargs["num_items"]))
        models.append(model)
        return model

    monkeypatch.setattr(
        "spotify.multitask_model.build_multitask_recommender",
        fake_builder,
    )

    training._train_multitask(
        data,
        architecture="ple",
        config=config,
        checkpoint_dir=tmp_path,
        model_params={
            "architecture": "mmoe",
            "num_experts": 5,
            "learning_rate": 3e-4,
            "batch_size": 2,
        },
    )
    training._train_multitask(
        data,
        architecture="mmoe",
        config=replace(config, batch_size=5),
        checkpoint_dir=tmp_path,
    )

    assert builder_calls[0]["architecture"] == "ple"
    assert builder_calls[0]["num_experts"] == 5
    assert builder_calls[0]["learning_rate"] == pytest.approx(3e-4)
    assert builder_calls[0]["sequence_encoder"] == "average"
    assert "batch_size" not in builder_calls[0]
    assert max(models[0].batch_sizes) <= 2
    assert builder_calls[1]["architecture"] == "mmoe"
    assert builder_calls[1]["learning_rate"] == pytest.approx(1e-3)
    assert max(models[1].batch_sizes) <= 5
