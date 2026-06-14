from __future__ import annotations

import numpy as np
import pytest

from spotify.sequence_pretraining import (
    augment_sequences,
    build_attribute_prediction_batch,
    build_augmented_positive_pairs,
    build_masked_item_batch,
    build_same_target_positive_pairs,
    build_sequence_attribute_targets,
    crop_sequences,
    numpy_info_nce_loss,
    subsequence_sequences,
)


def test_masked_item_batch_is_deterministic_and_respects_padding_and_intent() -> None:
    sequences = np.asarray(
        [
            [-1, 1, 2, 3, 4],
            [-1, -1, 5, 6, 7],
            [-1, -1, -1, -1, 8],
        ],
        dtype="int32",
    )

    first = build_masked_item_batch(
        sequences,
        mask_token_id=99,
        mask_probability=0.45,
        padding_id=-1,
        protect_last_n=1,
        seed=17,
    )
    second = build_masked_item_batch(
        sequences,
        mask_token_id=99,
        mask_probability=0.45,
        padding_id=-1,
        protect_last_n=1,
        seed=17,
    )

    for first_value, second_value in zip(first.__dict__.values(), second.__dict__.values()):
        np.testing.assert_array_equal(first_value, second_value)

    np.testing.assert_array_equal(first.input_sequences[:, -1], sequences[:, -1])
    assert not np.any(first.prediction_mask[sequences == -1])
    assert not first.prediction_mask[2].any()
    np.testing.assert_array_equal(
        first.target_items,
        sequences[first.source_rows, first.target_positions],
    )
    assert np.all(first.input_sequences[first.source_rows, first.target_positions] == 99)


def test_crop_and_subsequence_preserve_order_and_recent_intent() -> None:
    sequences = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [-1, -1, 7, 8, 9, 10],
        ],
        dtype="int32",
    )

    crop = crop_sequences(
        sequences,
        crop_ratio=0.5,
        padding_id=-1,
        protect_last_n=2,
        seed=3,
    )
    subsequence = subsequence_sequences(
        sequences,
        drop_probability=0.75,
        padding_id=-1,
        protect_last_n=2,
        min_items=3,
        seed=3,
    )

    np.testing.assert_array_equal(crop.sequences[0], [-1, -1, -1, 4, 5, 6])
    np.testing.assert_array_equal(crop.sequences[1], [-1, -1, -1, -1, 9, 10])
    for row_index, source in enumerate(sequences):
        source_valid = source[source != -1]
        view_valid = subsequence.sequences[row_index, subsequence.valid_mask[row_index]]
        indices = [int(np.flatnonzero(source_valid == item)[0]) for item in view_valid]
        assert indices == sorted(indices)
        np.testing.assert_array_equal(view_valid[-2:], source_valid[-2:])
        assert len(view_valid) >= 3


def test_named_augmentations_and_positive_views_are_reproducible() -> None:
    sequences = np.arange(1, 25, dtype="int32").reshape(4, 6)

    first = build_augmented_positive_pairs(
        sequences,
        left_augmentation="crop",
        right_augmentation="mask",
        mask_token_id=99,
        protect_last_n=1,
        seed=21,
    )
    second = build_augmented_positive_pairs(
        sequences,
        left_augmentation="crop",
        right_augmentation="mask",
        mask_token_id=99,
        protect_last_n=1,
        seed=21,
    )

    np.testing.assert_array_equal(first.left.sequences, second.left.sequences)
    np.testing.assert_array_equal(first.right.sequences, second.right.sequences)
    np.testing.assert_array_equal(first.left_rows, np.arange(4))
    np.testing.assert_array_equal(first.left_rows, first.right_rows)
    np.testing.assert_array_equal(first.left.sequences[:, -1], sequences[:, -1])
    np.testing.assert_array_equal(first.right.sequences[:, -1], sequences[:, -1])

    with pytest.raises(ValueError, match="mask_token_id"):
        augment_sequences(sequences, "mask")


def test_same_target_pairs_exclude_singletons_and_self_pairs() -> None:
    sequences = np.arange(30, dtype="int32").reshape(6, 5)
    targets = np.asarray([4, 4, 9, 9, 9, 12], dtype="int32")

    first = build_same_target_positive_pairs(sequences, targets, seed=11)
    second = build_same_target_positive_pairs(sequences, targets, seed=11)

    np.testing.assert_array_equal(first.left_rows, second.left_rows)
    np.testing.assert_array_equal(first.right_rows, second.right_rows)
    assert len(first.left_rows) == 5
    assert np.all(first.left_rows != first.right_rows)
    np.testing.assert_array_equal(targets[first.left_rows], targets[first.right_rows])
    assert 5 not in first.left_rows
    np.testing.assert_array_equal(first.left.sequences, sequences[first.left_rows])
    np.testing.assert_array_equal(first.right.sequences, sequences[first.right_rows])


def test_info_nce_rewards_aligned_pairs_and_can_mask_false_negatives() -> None:
    aligned = np.eye(4, dtype="float32")
    permuted = aligned[[1, 0, 3, 2]]

    aligned_loss = numpy_info_nce_loss(aligned, aligned, temperature=0.1)
    permuted_loss = numpy_info_nce_loss(aligned, permuted, temperature=0.1)
    grouped_loss = numpy_info_nce_loss(
        aligned,
        aligned,
        temperature=0.1,
        group_ids=np.zeros(4, dtype="int32"),
    )

    assert np.isfinite(aligned_loss)
    assert aligned_loss < permuted_loss
    assert grouped_loss == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="same rank-2 shape"):
        numpy_info_nce_loss(aligned, aligned[:2])
    with pytest.raises(ValueError, match="temperature"):
        numpy_info_nce_loss(aligned, aligned, temperature=0.0)


def test_attribute_helpers_select_valid_items_and_aggregate_metadata() -> None:
    sequences = np.asarray([[-1, 0, 1, 2], [-1, -1, 2, 3]], dtype="int32")
    attributes = {
        "genre": np.asarray(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype="int8",
        ),
        "energy": np.asarray([0.1, 0.3, 0.7, 0.9], dtype="float32"),
    }

    batch = build_attribute_prediction_batch(
        sequences,
        attributes,
        selection_probability=1.0,
        padding_id=-1,
        protect_last_n=1,
        seed=4,
    )

    np.testing.assert_array_equal(batch.item_ids, [0, 1, 2])
    np.testing.assert_array_equal(batch.targets["genre"], attributes["genre"][batch.item_ids])
    np.testing.assert_array_equal(batch.targets["energy"], attributes["energy"][batch.item_ids])

    maximum = build_sequence_attribute_targets(
        sequences,
        attributes,
        padding_id=-1,
        aggregation="max",
    )
    mean = build_sequence_attribute_targets(
        sequences,
        attributes["energy"],
        padding_id=-1,
        aggregation="mean",
    )
    last = build_sequence_attribute_targets(
        sequences,
        attributes,
        padding_id=-1,
        aggregation="last",
    )

    np.testing.assert_array_equal(maximum["genre"], [[1, 1, 0], [0, 1, 1]])
    np.testing.assert_allclose(mean["attributes"], [np.mean([0.1, 0.3, 0.7]), np.mean([0.7, 0.9])])
    np.testing.assert_array_equal(last["genre"], [attributes["genre"][2], attributes["genre"][3]])


def test_invalid_inputs_fail_with_actionable_errors() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        build_masked_item_batch([1, 2, 3], mask_token_id=9)
    with pytest.raises(ValueError, match="crop_ratio"):
        crop_sequences([[1, 2]], crop_ratio=0.0)
    with pytest.raises(ValueError, match="matching the sequence batch"):
        build_same_target_positive_pairs([[1, 2], [2, 3]], [1])
    with pytest.raises(ValueError, match="does not cover"):
        build_attribute_prediction_batch(
            [[0, 4]],
            np.zeros((4, 2)),
            selection_probability=1.0,
        )
