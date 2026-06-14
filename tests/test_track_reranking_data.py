from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from spotify.track_level_data import (
    TrackLevelTemporalSplits,
    build_track_level_dataset,
    split_track_level_examples,
)
from spotify.track_reranking_data import (
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    TrackRerankingConfig,
    build_track_reranking_data,
    save_track_reranking_data,
)


def _splits(session_count: int = 14) -> TrackLevelTemporalSplits:
    timestamps: list[pd.Timestamp] = []
    tracks: list[str] = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    patterns = (
        ("track:a", "track:b", "track:c", "track:a"),
        ("track:b", "track:d", "track:a", "track:b"),
        ("track:c", "track:a", "track:e", "track:c"),
    )
    for session_index in range(session_count):
        start = base + pd.Timedelta(hours=session_index * 2)
        for position, track in enumerate(patterns[session_index % len(patterns)]):
            timestamps.append(start + pd.Timedelta(minutes=position * 4))
            tracks.append(track)
    dataset = build_track_level_dataset(
        pd.DataFrame({"ts": timestamps, "spotify_track_uri": tracks}),
        max_history=16,
    )
    return split_track_level_examples(dataset)


def _config(**overrides) -> TrackRerankingConfig:
    values = {
        "max_items": 5,
        "candidate_count": 5,
        "retrieval_pool_size": 5,
        "retriever_fit_fraction": 0.6,
        "cooccurrence_shrinkage": 1.0,
        "max_train_queries": 100,
        "max_validation_queries": 100,
        "max_test_queries": 100,
        "random_seed": 17,
    }
    values.update(overrides)
    return TrackRerankingConfig(**values)


def _candidate_feature_map(split, group_index: int) -> dict[str, np.ndarray]:
    selector = split.group_slice(group_index)
    return {
        uri: split.candidate_features[row_index]
        for row_index, uri in zip(
            range(selector.start, selector.stop),
            split.candidate_track_uris[selector],
        )
    }


def test_every_candidate_group_contains_exactly_one_target() -> None:
    splits = _splits()
    data = build_track_reranking_data(splits, config=_config())
    examples_by_id = {example.example_id: example for example in splits.all_examples}

    for split in (data.train, data.validation, data.test):
        for group_index, example_id in enumerate(split.example_ids):
            selector = split.group_slice(group_index)
            labels = split.labels[selector, 0]
            candidates = split.candidate_track_uris[selector]
            target = examples_by_id[int(example_id)].target_track_uri

            assert labels.sum() == 1.0
            assert candidates[int(np.argmax(labels))] == target
            assert len(candidates) == len(set(candidates))


def test_candidate_features_do_not_change_when_only_target_label_changes() -> None:
    base_splits = _splits()
    source = base_splits.validation[0]
    alternative_target = next(
        item
        for item in ("track:a", "track:b", "track:c", "track:d", "track:e")
        if item != source.target_track_uri
    )
    alternative = replace(
        source,
        labels=replace(source.labels, next_track_uri=alternative_target),
    )
    first = TrackLevelTemporalSplits(
        train=base_splits.train,
        validation=(source,),
        test=(),
    )
    second = TrackLevelTemporalSplits(
        train=base_splits.train,
        validation=(alternative,),
        test=(),
    )

    first_data = build_track_reranking_data(first, config=_config())
    second_data = build_track_reranking_data(second, config=_config())
    first_features = _candidate_feature_map(first_data.validation, 0)
    second_features = _candidate_feature_map(second_data.validation, 0)

    assert set(first_features) == set(second_features)
    for candidate in first_features:
        np.testing.assert_allclose(first_features[candidate], second_features[candidate])
    assert first_data.validation.candidate_track_uris == second_data.validation.candidate_track_uris
    assert not np.array_equal(first_data.validation.labels, second_data.validation.labels)


def test_candidates_and_temporal_sampling_are_deterministic() -> None:
    splits = _splits(20)
    config = _config(
        candidate_count=4,
        retrieval_pool_size=5,
        max_train_queries=5,
        max_validation_queries=3,
        max_test_queries=3,
    )

    first = build_track_reranking_data(splits, config=config)
    second = build_track_reranking_data(splits, config=config)

    assert first.train.query_ids == second.train.query_ids
    assert first.train.candidate_track_uris == second.train.candidate_track_uris
    np.testing.assert_array_equal(first.train.group_offsets, second.train.group_offsets)
    np.testing.assert_array_equal(first.train.labels, second.train.labels)
    np.testing.assert_allclose(first.train.context_features, second.train.context_features)
    np.testing.assert_allclose(first.train.candidate_features, second.train.candidate_features)
    assert list(first.train.example_ids) == sorted(first.train.example_ids)


def test_dcn_arrays_have_expected_group_and_feature_shapes(tmp_path) -> None:
    data = build_track_reranking_data(_splits(), config=_config(candidate_count=4))

    for split in (data.train, data.validation, data.test):
        assert split.context_features.shape == (len(split), len(CONTEXT_FEATURE_NAMES))
        assert split.candidate_features.shape == (len(split), len(CANDIDATE_FEATURE_NAMES))
        assert split.labels.shape == (len(split), 1)
        assert split.group_ids.shape == (len(split),)
        assert split.candidate_item_ids.shape == (len(split),)
        assert split.group_offsets.shape == (split.query_count + 1,)
        assert split.group_offsets[-1] == len(split)
        assert np.isfinite(split.context_features).all()
        assert np.isfinite(split.candidate_features).all()

    manifest_path = save_track_reranking_data(data, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    payload = np.load(tmp_path / "train_reranking.npz")

    assert manifest["status"] == "complete"
    assert manifest["leakage_controls"]["retriever_fit_scope"] == "earlier_training_sessions_only"
    assert payload["context_features"].shape == data.train.context_features.shape
    assert payload["candidate_features"].shape == data.train.candidate_features.shape
