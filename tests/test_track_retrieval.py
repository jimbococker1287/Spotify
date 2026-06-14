from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotify.track_retrieval import (
    EASERetriever,
    PopularityRetriever,
    SessionCooccurrenceRetriever,
    candidate_diagnostics,
    evaluate_retriever,
    prepare_implicit_feedback,
    recall_at_k,
    sample_bpr_triplets,
)


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s2", "s2", "s3", "s3", "s4"],
            "track_id": ["a", "b", "c", "a", "b", "a", "d", "e"],
            "plays": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_popularity_retriever_is_weighted_deterministic_and_excludes_seen() -> None:
    model = PopularityRetriever().fit(_interactions(), weight_col="plays")

    first = model.recommend(["a"], k=4)
    second = model.recommend(["a"], k=4)

    assert [row.item_id for row in first] == ["b", "c", "d", "e"]
    assert first == second
    assert model.catalog == ("a", "b", "c", "d", "e")


def test_session_cooccurrence_recommends_tracks_from_related_sessions() -> None:
    model = SessionCooccurrenceRetriever().fit(_interactions(), shrinkage=1.0)

    recommendations = model.recommend(["c"], k=3)
    cold_start = model.recommend(["unknown"], k=2)

    assert [row.item_id for row in recommendations[:2]] == ["b", "a"]
    assert [row.item_id for row in cold_start] == ["a", "b"]
    assert all(row.item_id != "c" for row in recommendations)


def test_ease_retriever_produces_finite_deterministic_candidates() -> None:
    model = EASERetriever().fit(_interactions(), l2=2.0)

    recommendations = model.recommend(["c"], k=4)

    assert recommendations == model.recommend(["c"], k=4)
    assert len(recommendations) == 4
    assert all(np.isfinite(row.score) for row in recommendations)
    assert "c" not in {row.item_id for row in recommendations}
    assert model.item_weights_.shape == (5, 5)
    assert np.allclose(np.diag(model.item_weights_), 0.0)


def test_linear_retrievers_score_only_unique_known_history_rows() -> None:
    model = SessionCooccurrenceRetriever().fit(_interactions())

    repeated = model.recommend(["c", "c", "unknown"], k=3)
    unique = model.recommend(["c"], k=3)

    assert repeated == unique


def test_prepare_implicit_feedback_aggregates_and_indexes_stably() -> None:
    interactions = pd.concat(
        [
            _interactions(),
            pd.DataFrame([{"session_id": "s1", "track_id": "a", "plays": 3.0}]),
        ],
        ignore_index=True,
    )

    feedback = prepare_implicit_feedback(
        interactions,
        weight_col="plays",
        binary=False,
    )
    dense = feedback.to_dense()

    assert feedback.user_ids == ("s1", "s2", "s3", "s4")
    assert feedback.item_ids == ("a", "b", "c", "d", "e")
    assert feedback.shape == (4, 5)
    assert dense[0, 0] == pytest.approx(4.0)
    assert np.count_nonzero(dense) == 8


def test_prepare_implicit_feedback_applies_support_filters() -> None:
    feedback = prepare_implicit_feedback(
        _interactions(),
        min_user_interactions=2,
        min_item_interactions=2,
    )

    assert feedback.user_ids == ("s1", "s2")
    assert feedback.item_ids == ("a", "b")
    assert np.array_equal(feedback.to_dense(), np.ones((2, 2), dtype="float32"))


def test_bpr_sampling_is_deterministic_and_never_samples_a_positive() -> None:
    feedback = prepare_implicit_feedback(_interactions())

    first = sample_bpr_triplets(feedback, negatives_per_positive=2, random_seed=17)
    second = sample_bpr_triplets(feedback, negatives_per_positive=2, random_seed=17)
    positives_by_user = {
        user: set(feedback.item_indices[feedback.user_indices == user].tolist()) for user in range(feedback.shape[0])
    }

    assert len(first) == 16
    assert np.array_equal(first.user_indices, second.user_indices)
    assert np.array_equal(first.negative_item_indices, second.negative_item_indices)
    assert all(
        negative not in positives_by_user[int(user)]
        for user, negative in zip(first.user_indices, first.negative_item_indices)
    )


def test_recall_supports_numpy_ground_truth_arrays() -> None:
    predictions = {"query": ["a", "b", "c"]}
    truths = {"query": np.asarray(["b", "missing"], dtype=object)}

    assert recall_at_k(predictions, truths, k=2) == pytest.approx(0.5)


def test_recall_and_candidate_diagnostics_support_scored_candidates() -> None:
    model = SessionCooccurrenceRetriever().fit(_interactions())
    histories = {"q1": ["c"], "q2": ["d"]}
    truths = {"q1": "a", "q2": ["a", "b"]}
    predictions = model.batch_recommend(histories, k=2)

    recall = recall_at_k(predictions, truths, k=2)
    diagnostics = candidate_diagnostics(
        predictions,
        truths=truths,
        catalog=model.catalog,
        k=2,
    )

    assert recall == pytest.approx(1.0)
    assert diagnostics.recall_at_k == pytest.approx(recall)
    assert diagnostics.query_count == 2
    assert diagnostics.mean_candidate_count == 2.0
    assert diagnostics.duplicate_rate == 0.0
    assert diagnostics.catalog_coverage == pytest.approx(0.4)
    assert diagnostics.as_dict()["k"] == 2


def test_evaluate_retriever_treats_missing_hits_as_zero_recall() -> None:
    model = PopularityRetriever().fit(_interactions())

    diagnostics = evaluate_retriever(
        model,
        histories={"known": ["a"], "miss": ["a"]},
        truths={"known": "b", "miss": "not-in-catalog"},
        k=1,
    )

    assert diagnostics.recall_at_k == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("factory", "fit_kwargs"),
    [
        (PopularityRetriever, {}),
        (SessionCooccurrenceRetriever, {}),
        (EASERetriever, {}),
    ],
)
def test_retrievers_validate_empty_and_null_interactions(factory, fit_kwargs) -> None:
    with pytest.raises(ValueError, match="At least one interaction"):
        factory().fit(pd.DataFrame(columns=["session_id", "track_id"]), **fit_kwargs)

    invalid = pd.DataFrame({"session_id": ["s1"], "track_id": [None]})
    with pytest.raises(ValueError, match="cannot be null"):
        factory().fit(invalid, **fit_kwargs)
