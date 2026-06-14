from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotify.track_expansion_training import (
    OOV_ITEM_ID,
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
