from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Collection
from typing import Hashable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


ItemId = Hashable
QueryId = Hashable


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate returned by a track retriever."""

    item_id: ItemId
    score: float


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Aggregate diagnostics for a set of candidate lists."""

    query_count: int
    mean_candidate_count: float
    min_candidate_count: int
    max_candidate_count: int
    duplicate_rate: float
    catalog_coverage: float
    recall_at_k: float | None
    k: int | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "query_count": self.query_count,
            "mean_candidate_count": self.mean_candidate_count,
            "min_candidate_count": self.min_candidate_count,
            "max_candidate_count": self.max_candidate_count,
            "duplicate_rate": self.duplicate_rate,
            "catalog_coverage": self.catalog_coverage,
            "recall_at_k": self.recall_at_k,
            "k": self.k,
        }


@dataclass(frozen=True)
class ImplicitFeedbackData:
    """Indexed positive feedback suitable for implicit-ALS or BPR libraries."""

    user_ids: tuple[Hashable, ...]
    item_ids: tuple[ItemId, ...]
    user_indices: np.ndarray
    item_indices: np.ndarray
    weights: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.user_ids), len(self.item_ids)

    def to_dense(self, *, dtype: str | np.dtype = "float32") -> np.ndarray:
        matrix = np.zeros(self.shape, dtype=dtype)
        np.add.at(matrix, (self.user_indices, self.item_indices), self.weights)
        return matrix


@dataclass(frozen=True)
class BPRTriplets:
    """Positive/negative index triplets for pairwise ranking training."""

    user_indices: np.ndarray
    positive_item_indices: np.ndarray
    negative_item_indices: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.user_indices)


@runtime_checkable
class TrackCandidateRetriever(Protocol):
    """Minimal interface shared by dependency-light candidate retrievers."""

    @property
    def catalog(self) -> tuple[ItemId, ...]: ...

    def recommend(
        self,
        history: Sequence[ItemId] = (),
        *,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> list[ScoredCandidate]: ...

    def batch_recommend(
        self,
        histories: Mapping[QueryId, Sequence[ItemId]],
        *,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> dict[QueryId, list[ScoredCandidate]]: ...


def _stable_id_key(value: Hashable) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _stable_unique(values: Sequence[Hashable]) -> tuple[Hashable, ...]:
    return tuple(sorted(set(values), key=_stable_id_key))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Interactions are missing required columns: {missing}.")


def _validated_interactions(
    interactions: pd.DataFrame,
    *,
    group_col: str | None,
    item_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame.")
    columns = [item_col]
    if group_col is not None:
        columns.append(group_col)
    if weight_col is not None:
        columns.append(weight_col)
    _require_columns(interactions, columns)
    if interactions.empty:
        raise ValueError("At least one interaction is required.")

    frame = interactions.loc[:, columns].copy()
    null_columns = [column for column in columns if frame[column].isna().any()]
    if null_columns:
        raise ValueError(f"Interaction identifiers and weights cannot be null: {null_columns}.")
    if weight_col is None:
        frame["_weight"] = 1.0
    else:
        try:
            frame["_weight"] = pd.to_numeric(frame[weight_col], errors="raise").astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{weight_col!r} must contain numeric values.") from exc
        weights = frame["_weight"].to_numpy(dtype="float64")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Interaction weights must be finite and strictly positive.")
    return frame


def _check_k(k: int) -> int:
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1.")
    return k


class _BaseRetriever:
    def __init__(self) -> None:
        self._catalog: tuple[ItemId, ...] = ()
        self._item_to_index: dict[ItemId, int] = {}
        self._popularity = np.empty(0, dtype="float64")

    @property
    def catalog(self) -> tuple[ItemId, ...]:
        if not self._catalog:
            raise RuntimeError("Retriever must be fitted before accessing its catalog.")
        return self._catalog

    def _set_catalog(self, items: Sequence[ItemId], popularity: np.ndarray) -> None:
        self._catalog = _stable_unique(items)
        self._item_to_index = {item: index for index, item in enumerate(self._catalog)}
        popularity_by_item = {
            item: float(score) for item, score in zip(items, np.asarray(popularity, dtype="float64").tolist())
        }
        self._popularity = np.asarray(
            [popularity_by_item.get(item, 0.0) for item in self._catalog],
            dtype="float64",
        )

    def _ensure_fitted(self) -> None:
        if not self._catalog:
            raise RuntimeError("Retriever must be fitted before recommendation.")

    def _rank_scores(
        self,
        scores: np.ndarray,
        *,
        history: Sequence[ItemId],
        k: int,
        exclude_seen: bool,
    ) -> list[ScoredCandidate]:
        self._ensure_fitted()
        k = _check_k(k)
        score_array = np.asarray(scores, dtype="float64").reshape(-1)
        if len(score_array) != len(self._catalog):
            raise ValueError("Score vector does not match the fitted catalog.")

        seen = set(history) if exclude_seen else set()
        ranked = [
            ScoredCandidate(item_id=item, score=float(score_array[index]))
            for index, item in enumerate(self._catalog)
            if item not in seen and np.isfinite(score_array[index])
        ]
        ranked.sort(key=lambda row: (-row.score, _stable_id_key(row.item_id)))
        return ranked[:k]

    def batch_recommend(
        self,
        histories: Mapping[QueryId, Sequence[ItemId]],
        *,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> dict[QueryId, list[ScoredCandidate]]:
        return {
            query_id: self.recommend(history, k=k, exclude_seen=exclude_seen) for query_id, history in histories.items()
        }


class PopularityRetriever(_BaseRetriever):
    """Weighted global popularity with deterministic tie-breaking."""

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        item_col: str = "track_id",
        weight_col: str | None = None,
    ) -> PopularityRetriever:
        frame = _validated_interactions(
            interactions,
            group_col=None,
            item_col=item_col,
            weight_col=weight_col,
        )
        totals = frame.groupby(item_col, sort=False, dropna=False)["_weight"].sum()
        items = totals.index.tolist()
        self._set_catalog(items, totals.to_numpy(dtype="float64"))
        return self

    def recommend(
        self,
        history: Sequence[ItemId] = (),
        *,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> list[ScoredCandidate]:
        return self._rank_scores(
            self._popularity,
            history=history,
            k=k,
            exclude_seen=exclude_seen,
        )


class _LinearItemRetriever(_BaseRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.item_weights_ = np.empty((0, 0), dtype="float64")

    def _history_vector(self, history: Sequence[ItemId]) -> np.ndarray:
        vector = np.zeros(len(self._catalog), dtype="float64")
        for item in history:
            index = self._item_to_index.get(item)
            if index is not None:
                vector[index] = 1.0
        return vector

    def recommend(
        self,
        history: Sequence[ItemId] = (),
        *,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> list[ScoredCandidate]:
        self._ensure_fitted()
        history_indices = sorted(
            {
                self._item_to_index[item]
                for item in history
                if item in self._item_to_index
            }
        )
        if history_indices:
            scores = np.sum(
                self.item_weights_[np.asarray(history_indices, dtype="int64")],
                axis=0,
            )
        else:
            scores = np.zeros(len(self._catalog), dtype="float64")
        if not history_indices or not np.any(np.isfinite(scores) & (scores != 0.0)):
            scores = self._popularity.copy()
        return self._rank_scores(scores, history=history, k=k, exclude_seen=exclude_seen)

    def _session_item_matrix(
        self,
        interactions: pd.DataFrame,
        *,
        session_col: str,
        item_col: str,
        weight_col: str | None,
        binary: bool,
    ) -> np.ndarray:
        frame = _validated_interactions(
            interactions,
            group_col=session_col,
            item_col=item_col,
            weight_col=weight_col,
        )
        grouped = frame.groupby([session_col, item_col], sort=False, dropna=False)["_weight"].sum()
        session_ids = _stable_unique(frame[session_col].tolist())
        items = _stable_unique(frame[item_col].tolist())
        session_to_index = {session_id: index for index, session_id in enumerate(session_ids)}
        item_to_index = {item: index for index, item in enumerate(items)}
        matrix = np.zeros((len(session_ids), len(items)), dtype="float64")
        for (session_id, item), weight in grouped.items():
            matrix[session_to_index[session_id], item_to_index[item]] = 1.0 if binary else float(weight)
        popularity = matrix.sum(axis=0)
        self._set_catalog(items, popularity)
        return matrix


class SessionCooccurrenceRetriever(_LinearItemRetriever):
    """Cosine-normalized item co-occurrence learned from session baskets."""

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        session_col: str = "session_id",
        item_col: str = "track_id",
        weight_col: str | None = None,
        binary: bool = True,
        shrinkage: float = 0.0,
    ) -> SessionCooccurrenceRetriever:
        if shrinkage < 0:
            raise ValueError("shrinkage cannot be negative.")
        matrix = self._session_item_matrix(
            interactions,
            session_col=session_col,
            item_col=item_col,
            weight_col=weight_col,
            binary=binary,
        )
        cooccurrence = matrix.T @ matrix
        frequency = np.diag(cooccurrence).copy()
        denominator = np.sqrt(np.outer(frequency, frequency)) + float(shrinkage)
        weights = np.divide(
            cooccurrence,
            denominator,
            out=np.zeros_like(cooccurrence),
            where=denominator > 0,
        )
        np.fill_diagonal(weights, 0.0)
        self.item_weights_ = weights
        return self


class EASERetriever(_LinearItemRetriever):
    """EASE-style closed-form linear item recommender over session baskets."""

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        session_col: str = "session_id",
        item_col: str = "track_id",
        weight_col: str | None = None,
        binary: bool = True,
        l2: float = 100.0,
    ) -> EASERetriever:
        if not np.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be finite and strictly positive.")
        matrix = self._session_item_matrix(
            interactions,
            session_col=session_col,
            item_col=item_col,
            weight_col=weight_col,
            binary=binary,
        )
        gram = matrix.T @ matrix
        diagonal = np.diag_indices_from(gram)
        gram[diagonal] += float(l2)
        precision = np.linalg.pinv(gram, hermitian=True)
        precision_diagonal = np.diag(precision)
        weights = np.divide(
            -precision,
            precision_diagonal.reshape(1, -1),
            out=np.zeros_like(precision),
            where=np.abs(precision_diagonal.reshape(1, -1)) > np.finfo("float64").eps,
        )
        np.fill_diagonal(weights, 0.0)
        self.item_weights_ = weights
        return self


def prepare_implicit_feedback(
    interactions: pd.DataFrame,
    *,
    user_col: str = "session_id",
    item_col: str = "track_id",
    weight_col: str | None = None,
    binary: bool = True,
    min_user_interactions: int = 1,
    min_item_interactions: int = 1,
) -> ImplicitFeedbackData:
    """Index and aggregate interactions for implicit-feedback training.

    Sessions, playlists, or actual users can all serve as ``user_col``. The
    returned arrays use deterministic indexes so artifacts are reproducible.
    """

    if min_user_interactions < 1 or min_item_interactions < 1:
        raise ValueError("Minimum interaction thresholds must be at least 1.")
    frame = _validated_interactions(
        interactions,
        group_col=user_col,
        item_col=item_col,
        weight_col=weight_col,
    )

    # Alternate filters until both sides satisfy their support threshold.
    while not frame.empty:
        previous_size = len(frame)
        user_counts = frame.groupby(user_col, dropna=False)[item_col].transform("nunique")
        item_counts = frame.groupby(item_col, dropna=False)[user_col].transform("nunique")
        frame = frame.loc[
            (user_counts >= int(min_user_interactions)) & (item_counts >= int(min_item_interactions))
        ].copy()
        if len(frame) == previous_size:
            break
    if frame.empty:
        raise ValueError("No interactions remain after applying support thresholds.")

    grouped = frame.groupby([user_col, item_col], sort=False, dropna=False)["_weight"].sum().reset_index()
    users = _stable_unique(grouped[user_col].tolist())
    items = _stable_unique(grouped[item_col].tolist())
    user_to_index = {user: index for index, user in enumerate(users)}
    item_to_index = {item: index for index, item in enumerate(items)}
    user_indices = grouped[user_col].map(user_to_index).to_numpy(dtype="int64")
    item_indices = grouped[item_col].map(item_to_index).to_numpy(dtype="int64")
    weights = grouped["_weight"].to_numpy(dtype="float32")
    if binary:
        weights = np.ones_like(weights, dtype="float32")

    order = np.lexsort((item_indices, user_indices))
    return ImplicitFeedbackData(
        user_ids=users,
        item_ids=items,
        user_indices=user_indices[order],
        item_indices=item_indices[order],
        weights=weights[order],
    )


def sample_bpr_triplets(
    feedback: ImplicitFeedbackData,
    *,
    negatives_per_positive: int = 1,
    random_seed: int = 42,
) -> BPRTriplets:
    """Sample deterministic uniform negatives for each positive interaction."""

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1.")
    user_count, item_count = feedback.shape
    if user_count == 0 or item_count < 2:
        return _empty_bpr_triplets()

    positives_by_user: list[set[int]] = [set() for _ in range(user_count)]
    for user_index, item_index in zip(
        feedback.user_indices.tolist(),
        feedback.item_indices.tolist(),
    ):
        positives_by_user[int(user_index)].add(int(item_index))

    rng = np.random.default_rng(int(random_seed))
    users: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    weights: list[float] = []
    for user_index, positive_index, weight in zip(
        feedback.user_indices.tolist(),
        feedback.item_indices.tolist(),
        feedback.weights.tolist(),
    ):
        user_positives = positives_by_user[int(user_index)]
        available_count = item_count - len(user_positives)
        if available_count == 0:
            continue

        if len(user_positives) * 2 >= item_count:
            available = np.asarray(
                [item for item in range(item_count) if item not in user_positives],
                dtype="int64",
            )
            sampled = rng.choice(
                available,
                size=int(negatives_per_positive),
                replace=available_count < int(negatives_per_positive),
            )
        else:
            sampled_values: list[int] = []
            while len(sampled_values) < int(negatives_per_positive):
                candidate = int(rng.integers(0, item_count))
                if candidate not in user_positives:
                    sampled_values.append(candidate)
            sampled = np.asarray(sampled_values, dtype="int64")

        users.extend([int(user_index)] * len(sampled))
        positives.extend([int(positive_index)] * len(sampled))
        negatives.extend(int(item) for item in sampled.tolist())
        weights.extend([float(weight)] * len(sampled))

    return BPRTriplets(
        user_indices=np.asarray(users, dtype="int64"),
        positive_item_indices=np.asarray(positives, dtype="int64"),
        negative_item_indices=np.asarray(negatives, dtype="int64"),
        weights=np.asarray(weights, dtype="float32"),
    )


def _empty_bpr_triplets() -> BPRTriplets:
    return BPRTriplets(
        user_indices=np.empty(0, dtype="int64"),
        positive_item_indices=np.empty(0, dtype="int64"),
        negative_item_indices=np.empty(0, dtype="int64"),
        weights=np.empty(0, dtype="float32"),
    )


def _candidate_ids(candidates: Sequence[ItemId | ScoredCandidate], *, k: int) -> list[ItemId]:
    return [candidate.item_id if isinstance(candidate, ScoredCandidate) else candidate for candidate in candidates[:k]]


def _truth_set(value: ItemId | Collection[ItemId] | np.ndarray) -> set[ItemId]:
    if isinstance(value, np.ndarray):
        return set(value.reshape(-1).tolist())
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        return {value}
    return set(value)


def recall_at_k(
    predictions: Mapping[QueryId, Sequence[ItemId | ScoredCandidate]],
    truths: Mapping[QueryId, ItemId | Collection[ItemId] | np.ndarray],
    *,
    k: int,
) -> float:
    """Compute macro Recall@K, treating missing predictions as empty lists."""

    k = _check_k(k)
    if not truths:
        raise ValueError("At least one ground-truth query is required.")
    recalls: list[float] = []
    for query_id, truth_value in truths.items():
        relevant = _truth_set(truth_value)
        if not relevant:
            continue
        predicted = set(_candidate_ids(predictions.get(query_id, ()), k=k))
        recalls.append(len(relevant.intersection(predicted)) / len(relevant))
    if not recalls:
        raise ValueError("Ground-truth values must contain at least one relevant item.")
    return float(np.mean(recalls))


def candidate_diagnostics(
    predictions: Mapping[QueryId, Sequence[ItemId | ScoredCandidate]],
    *,
    truths: Mapping[QueryId, ItemId | Collection[ItemId] | np.ndarray] | None = None,
    catalog: Sequence[ItemId] | None = None,
    k: int | None = None,
) -> CandidateDiagnostics:
    """Summarize candidate depth, duplication, coverage, and optional recall."""

    if k is not None:
        k = _check_k(k)
    query_ids = list(truths) if truths is not None else list(predictions)
    effective_k = (
        k
        if k is not None
        else max(
            (len(predictions.get(query_id, ())) for query_id in query_ids),
            default=0,
        )
    )
    candidate_lists = [_candidate_ids(predictions.get(query_id, ()), k=effective_k) for query_id in query_ids]
    counts = [len(candidates) for candidates in candidate_lists]
    total_count = sum(counts)
    unique_per_query = sum(len(set(candidates)) for candidates in candidate_lists)
    unique_candidates = set().union(*(set(candidates) for candidates in candidate_lists))

    if catalog is None:
        catalog_set = unique_candidates
    else:
        catalog_set = set(catalog)
    coverage = len(unique_candidates.intersection(catalog_set)) / len(catalog_set) if catalog_set else 0.0
    recall = recall_at_k(predictions, truths, k=effective_k) if truths is not None and effective_k > 0 else None
    return CandidateDiagnostics(
        query_count=len(query_ids),
        mean_candidate_count=float(np.mean(counts)) if counts else 0.0,
        min_candidate_count=min(counts, default=0),
        max_candidate_count=max(counts, default=0),
        duplicate_rate=(total_count - unique_per_query) / total_count if total_count else 0.0,
        catalog_coverage=float(coverage),
        recall_at_k=recall,
        k=(effective_k if effective_k > 0 else k),
    )


def evaluate_retriever(
    retriever: TrackCandidateRetriever,
    histories: Mapping[QueryId, Sequence[ItemId]],
    truths: Mapping[QueryId, ItemId | Collection[ItemId] | np.ndarray],
    *,
    k: int,
    exclude_seen: bool = True,
) -> CandidateDiagnostics:
    """Generate candidates and evaluate them with a shared diagnostic contract."""

    predictions = retriever.batch_recommend(
        histories,
        k=k,
        exclude_seen=exclude_seen,
    )
    return candidate_diagnostics(
        predictions,
        truths=truths,
        catalog=retriever.catalog,
        k=k,
    )


__all__ = [
    "BPRTriplets",
    "CandidateDiagnostics",
    "EASERetriever",
    "ImplicitFeedbackData",
    "PopularityRetriever",
    "ScoredCandidate",
    "SessionCooccurrenceRetriever",
    "TrackCandidateRetriever",
    "candidate_diagnostics",
    "evaluate_retriever",
    "prepare_implicit_feedback",
    "recall_at_k",
    "sample_bpr_triplets",
]
