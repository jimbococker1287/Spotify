from __future__ import annotations

import numpy as np

from .retrieval_common import _env_int


def _repeat_candidate_features(
    *,
    seq_batch: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    candidates = np.asarray(candidate_ids, dtype="int32")
    if seq_arr.ndim != 2 or candidates.ndim != 2 or len(seq_arr) != len(candidates):
        empty = np.zeros((0, 0), dtype="float32")
        return empty, empty, empty, empty, empty

    matches = seq_arr[:, :, None] == candidates[:, None, :]
    occurrence_fraction = matches.mean(axis=1, dtype="float32")
    in_session = matches.any(axis=1).astype("float32")
    last_artist = seq_arr[:, -1].astype("int32")
    is_last_artist = (candidates == last_artist.reshape(-1, 1)).astype("float32")

    recent3 = matches[:, -min(3, seq_arr.shape[1]) :, :].any(axis=1).astype("float32")
    recent5 = matches[:, -min(5, seq_arr.shape[1]) :, :].any(axis=1).astype("float32")
    return occurrence_fraction, in_session, is_last_artist, recent3, recent5


def _candidate_feature_matrix(
    *,
    seq_batch: np.ndarray,
    ctx_batch: np.ndarray,
    session_vec: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    artist_embeddings: np.ndarray,
    popularity: np.ndarray,
) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    ctx_arr = np.asarray(ctx_batch, dtype="float32")
    candidates = np.asarray(candidate_ids, dtype="int32")
    scores = np.asarray(candidate_scores, dtype="float32")

    cand_emb = np.asarray(artist_embeddings, dtype="float32")[candidates]
    last_artist = seq_arr[:, -1].astype("int32")
    last_emb = np.asarray(artist_embeddings, dtype="float32")[last_artist]
    occurrence_fraction, in_session, is_last_artist, recent_match_3, recent_match_5 = _repeat_candidate_features(
        seq_batch=seq_arr,
        candidate_ids=candidates,
    )
    rank_positions = np.broadcast_to(np.arange(candidates.shape[1], dtype="float32"), candidates.shape)
    rank_normalized = rank_positions / float(max(1, candidates.shape[1] - 1))
    top_score = scores[:, :1]
    score_margin = scores - top_score
    popularity_values = np.asarray(popularity, dtype="float32")[candidates]
    similarity_last = np.sum(cand_emb * last_emb[:, None, :], axis=2, dtype="float32")
    similarity_session = np.sum(cand_emb * session_vec[:, None, :], axis=2, dtype="float32")
    repeat_pressure = np.clip((0.55 * is_last_artist) + (0.25 * recent_match_3) + (0.20 * occurrence_fraction), 0.0, 1.5)
    novelty_support = np.clip(1.0 - np.minimum(1.0, occurrence_fraction + (0.35 * recent_match_3)), 0.0, 1.0)
    never_seen = 1.0 - in_session

    scalar_features = np.stack(
        [
            scores,
            score_margin,
            rank_normalized,
            popularity_values,
            in_session,
            occurrence_fraction,
            is_last_artist,
            similarity_last,
            similarity_session,
            recent_match_3,
            recent_match_5,
            never_seen,
            repeat_pressure,
            novelty_support,
        ],
        axis=2,
    ).reshape(-1, 14)

    repeated_ctx = np.repeat(ctx_arr.astype("float32", copy=False), candidates.shape[1], axis=0)
    return np.concatenate([scalar_features.astype("float32", copy=False), repeated_ctx], axis=1)


def _reranker_sample_weights(
    *,
    seq_batch: np.ndarray,
    candidate_ids: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    candidates = np.asarray(candidate_ids, dtype="int32")
    y_arr = np.asarray(y_true, dtype="int32").reshape(-1)
    if seq_arr.ndim != 2 or candidates.ndim != 2 or len(seq_arr) != len(candidates) or len(y_arr) != len(candidates):
        return np.ones(int(candidates.size), dtype="float32")

    _, in_session, is_last_artist, _, _ = _repeat_candidate_features(
        seq_batch=seq_arr,
        candidate_ids=candidates,
    )
    last_artist = seq_arr[:, -1].astype("int32")
    target_is_new_from_prev = (y_arr != last_artist).astype("float32").reshape(-1, 1)
    labels = (candidates == y_arr.reshape(-1, 1)).astype("float32")

    weights = np.ones(candidates.shape, dtype="float32")
    weights += 1.75 * labels * target_is_new_from_prev
    weights += 1.10 * (1.0 - labels) * is_last_artist * target_is_new_from_prev
    weights += 0.35 * (1.0 - labels) * (in_session - is_last_artist).clip(min=0.0) * target_is_new_from_prev
    return weights.reshape(-1)


def _apply_repeat_mitigation(
    *,
    seq_batch: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    rerank_scores: np.ndarray,
) -> np.ndarray:
    occurrence_fraction, _, is_last_artist, recent_match_3, _ = _repeat_candidate_features(
        seq_batch=seq_batch,
        candidate_ids=candidate_ids,
    )
    if rerank_scores.ndim != 2 or rerank_scores.size == 0:
        return rerank_scores.astype("float32", copy=False)

    top_scores = np.asarray(candidate_scores, dtype="float32")
    if top_scores.shape[1] >= 2:
        top_gap = top_scores[:, :1] - top_scores[:, 1:2]
    else:
        top_gap = np.zeros((len(top_scores), 1), dtype="float32")

    ambiguity_logits = np.clip(10.0 * (top_gap - 0.08), -18.0, 18.0)
    ambiguity = 1.0 / (1.0 + np.exp(ambiguity_logits))
    repeat_penalty = _env_int("SPOTIFY_RERANKER_REPEAT_PENALTY_BPS", 220) / 1000.0
    immediate_repeat_penalty = _env_int("SPOTIFY_RERANKER_IMMEDIATE_REPEAT_PENALTY_BPS", 360) / 1000.0
    novelty_boost = _env_int("SPOTIFY_RERANKER_NOVELTY_BOOST_BPS", 120) / 1000.0

    repeat_pressure = np.clip((0.50 * occurrence_fraction) + (0.25 * recent_match_3) + (0.50 * is_last_artist), 0.0, 1.5)
    novelty_support = np.clip(1.0 - np.minimum(1.0, occurrence_fraction + (0.35 * recent_match_3)), 0.0, 1.0)

    adjusted = np.asarray(rerank_scores, dtype="float32").copy()
    adjusted *= np.exp(-repeat_penalty * ambiguity * repeat_pressure)
    adjusted *= np.exp(-immediate_repeat_penalty * ambiguity * is_last_artist)
    adjusted *= 1.0 + (novelty_boost * ambiguity * novelty_support)
    adjusted = np.clip(adjusted, 1e-6, None)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted.astype("float32", copy=False)


def _inject_true_labels(
    *,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    y_true: np.ndarray,
    full_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids, dtype="int32").copy()
    scores = np.asarray(candidate_scores, dtype="float32").copy()
    y_arr = np.asarray(y_true, dtype="int32").reshape(-1)
    for row_idx in range(len(ids)):
        label = int(y_arr[row_idx])
        if np.any(ids[row_idx] == label):
            continue
        ids[row_idx, -1] = label
        scores[row_idx, -1] = float(full_scores[row_idx, label])
        order = np.argsort(-scores[row_idx])
        ids[row_idx] = ids[row_idx, order]
        scores[row_idx] = scores[row_idx, order]
    return ids, scores


__all__ = [
    "_apply_repeat_mitigation",
    "_candidate_feature_matrix",
    "_inject_true_labels",
    "_repeat_candidate_features",
    "_reranker_sample_weights",
]
