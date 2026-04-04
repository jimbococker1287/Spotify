from __future__ import annotations

import numpy as np

from spotify.retrieval_common import RetrievalServingArtifact
from spotify.retrieval_reranking import _predict_reranked_probabilities


class _ConstantReranker:
    classes_ = np.array([0, 1], dtype="int32")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.42, 0.58]], dtype="float32"), (len(features), 1))


def test_predict_reranked_probabilities_penalizes_ambiguous_immediate_repeats() -> None:
    artifact = RetrievalServingArtifact(
        model_name="retrieval_reranker",
        candidate_k=4,
        artist_embeddings=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.9],
            ],
            dtype="float32",
        ),
        sequence_projection=np.eye(2, dtype="float32"),
        context_projection=np.zeros((1, 2), dtype="float32"),
        item_bias=np.zeros(4, dtype="float32"),
        popularity=np.array([0.5, 0.4, 0.3, 0.2], dtype="float32"),
        reranker=_ConstantReranker(),
    )

    seq_batch = np.array([[0, 1, 2, 2]], dtype="int32")
    ctx_batch = np.zeros((1, 1), dtype="float32")
    session_vec = np.array([[0.7, 0.3]], dtype="float32")
    retrieval_scores = np.array([[1.2, 1.1, 1.31, 1.29]], dtype="float32")

    proba = _predict_reranked_probabilities(
        seq_batch=seq_batch,
        ctx_batch=ctx_batch,
        session_vec=session_vec,
        retrieval_scores=retrieval_scores,
        artifact=artifact,
    )

    assert proba.shape == (1, 4)
    assert float(proba[0, 3]) > float(proba[0, 2])
    assert np.isclose(float(proba.sum()), 1.0)
