from __future__ import annotations

from collections import defaultdict

import numpy as np


class SessionKNNClassifier:
    def __init__(
        self,
        *,
        n_neighbors: int = 64,
        candidate_cap: int = 512,
        smoothing: float = 1.0,
    ) -> None:
        self.n_neighbors = max(1, int(n_neighbors))
        self.candidate_cap = max(self.n_neighbors, int(candidate_cap))
        self.smoothing = max(0.0, float(smoothing))
        self.classes_: np.ndarray | None = None
        self._class_to_index: dict[int, int] = {}
        self._X_train = np.empty((0, 0), dtype="int32")
        self._y_train = np.empty((0,), dtype="int32")
        self._last_index: dict[int, list[int]] = {}
        self._pair_index: dict[tuple[int, int], list[int]] = {}
        self._fallback_last: dict[int, np.ndarray] = {}
        self._class_prior = np.empty((0,), dtype="float32")
        self._position_weights = np.empty((0,), dtype="float32")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SessionKNNClassifier":
        X_arr = np.asarray(X, dtype="int32")
        y_arr = np.asarray(y, dtype="int32").reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("SessionKNNClassifier expects a 2D sequence array.")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have the same number of rows.")
        if len(X_arr) == 0:
            raise ValueError("SessionKNNClassifier requires at least one training row.")

        self.classes_ = np.unique(y_arr).astype("int32")
        self._class_to_index = {int(label): idx for idx, label in enumerate(self.classes_.tolist())}
        self._X_train = X_arr
        self._y_train = y_arr
        self._position_weights = np.linspace(1.0, 2.5, num=X_arr.shape[1], dtype="float32")

        last_index: dict[int, list[int]] = defaultdict(list)
        pair_index: dict[tuple[int, int], list[int]] = defaultdict(list)
        fallback_votes: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(self.classes_), dtype="float32"))
        class_counts = np.zeros(len(self.classes_), dtype="float32")

        for idx, (row, target) in enumerate(zip(X_arr, y_arr)):
            last_artist = int(row[-1])
            prev_artist = int(row[-2]) if row.shape[0] > 1 else -1
            class_idx = self._class_to_index[int(target)]
            last_index[last_artist].append(idx)
            pair_index[(prev_artist, last_artist)].append(idx)
            fallback_votes[last_artist][class_idx] += 1.0
            class_counts[class_idx] += 1.0

        self._last_index = dict(last_index)
        self._pair_index = dict(pair_index)
        total = float(np.sum(class_counts))
        self._class_prior = class_counts / total if total > 0 else np.full(len(self.classes_), 1.0 / len(self.classes_))

        self._fallback_last = {}
        for last_artist, votes in fallback_votes.items():
            total_votes = float(np.sum(votes))
            if total_votes > 0:
                self._fallback_last[int(last_artist)] = votes / total_votes

        return self

    def _candidate_indices(self, query_row: np.ndarray) -> np.ndarray:
        last_artist = int(query_row[-1])
        prev_artist = int(query_row[-2]) if query_row.shape[0] > 1 else -1
        candidates = self._pair_index.get((prev_artist, last_artist))
        if not candidates:
            candidates = self._last_index.get(last_artist)
        if not candidates:
            start = max(0, len(self._X_train) - self.candidate_cap)
            return np.arange(start, len(self._X_train), dtype="int32")
        if len(candidates) > self.candidate_cap:
            candidates = candidates[-self.candidate_cap :]
        return np.asarray(candidates, dtype="int32")

    def _fallback_distribution(self, query_row: np.ndarray) -> np.ndarray:
        last_artist = int(query_row[-1])
        fallback = self._fallback_last.get(last_artist)
        if fallback is not None:
            return fallback.astype("float32", copy=True)
        return self._class_prior.astype("float32", copy=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("SessionKNNClassifier must be fitted before prediction.")

        X_arr = np.asarray(X, dtype="int32")
        if X_arr.ndim != 2:
            raise ValueError("SessionKNNClassifier expects a 2D sequence array.")

        out = np.zeros((len(X_arr), len(self.classes_)), dtype="float32")
        for row_idx, query_row in enumerate(X_arr):
            candidate_idx = self._candidate_indices(query_row)
            if candidate_idx.size == 0:
                out[row_idx] = self._fallback_distribution(query_row)
                continue

            candidate_rows = self._X_train[candidate_idx]
            matches = (candidate_rows == query_row.reshape(1, -1)).astype("float32")
            similarity = np.sum(matches * self._position_weights.reshape(1, -1), axis=1)
            nonzero = similarity > 0
            if not np.any(nonzero):
                out[row_idx] = self._fallback_distribution(query_row)
                continue

            candidate_idx = candidate_idx[nonzero]
            similarity = similarity[nonzero]
            if len(candidate_idx) > self.n_neighbors:
                topk = np.argpartition(similarity, -self.n_neighbors)[-self.n_neighbors :]
                candidate_idx = candidate_idx[topk]
                similarity = similarity[topk]

            votes = np.full(len(self.classes_), self.smoothing, dtype="float32") * self._class_prior
            for train_idx, score in zip(candidate_idx.tolist(), similarity.tolist()):
                class_idx = self._class_to_index[int(self._y_train[int(train_idx)])]
                votes[class_idx] += float(score)

            total_votes = float(np.sum(votes))
            out[row_idx] = votes / total_votes if total_votes > 0 else self._fallback_distribution(query_row)

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        assert self.classes_ is not None
        return self.classes_[indices]
