from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .dcn_v2_model import build_dcn_v2_model
from .run_artifacts import write_json


@dataclass(frozen=True)
class DCNCandidateSplit:
    """Row-wise candidate features and labels for one temporal split."""

    context_features: np.ndarray
    item_features: np.ndarray
    labels: np.ndarray
    query_ids: np.ndarray
    candidate_ids: np.ndarray | None = None
    event_times: np.ndarray | None = None
    sample_weights: np.ndarray | None = None

    def __len__(self) -> int:
        return int(np.asarray(self.labels).reshape(-1).shape[0])

    def validate(self, *, name: str, allow_empty: bool = True) -> None:
        context = np.asarray(self.context_features)
        items = np.asarray(self.item_features)
        labels = np.asarray(self.labels).reshape(-1)
        query_ids = np.asarray(self.query_ids).reshape(-1)
        row_count = len(labels)

        if context.ndim != 2 or context.shape[1] < 1:
            raise ValueError(f"{name}.context_features must be a rank-2 array with at least one column")
        if items.ndim != 2 or items.shape[1] < 1:
            raise ValueError(f"{name}.item_features must be a rank-2 array with at least one column")
        if not allow_empty and row_count == 0:
            raise ValueError(f"{name} must contain at least one candidate row")
        for field_name, values in (
            ("context_features", context),
            ("item_features", items),
            ("query_ids", query_ids),
        ):
            if len(values) != row_count:
                raise ValueError(
                    f"{name}.{field_name} has {len(values)} rows; expected {row_count}"
                )
        if not np.isfinite(context.astype("float64", copy=False)).all():
            raise ValueError(f"{name}.context_features must contain only finite numeric values")
        if not np.isfinite(items.astype("float64", copy=False)).all():
            raise ValueError(f"{name}.item_features must contain only finite numeric values")
        if not np.isfinite(labels.astype("float64", copy=False)).all():
            raise ValueError(f"{name}.labels must contain only finite values")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{name}.labels must be binary values (0 or 1)")
        _validate_ids(query_ids, name=f"{name}.query_ids")

        if self.candidate_ids is not None:
            candidate_ids = np.asarray(self.candidate_ids).reshape(-1)
            if len(candidate_ids) != row_count:
                raise ValueError(
                    f"{name}.candidate_ids has {len(candidate_ids)} rows; expected {row_count}"
                )
            _validate_ids(candidate_ids, name=f"{name}.candidate_ids")
            seen: set[tuple[object, object]] = set()
            for query_id, candidate_id in zip(query_ids.tolist(), candidate_ids.tolist()):
                key = (query_id, candidate_id)
                if key in seen:
                    raise ValueError(
                        f"{name}.candidate_ids must be unique within each query"
                    )
                seen.add(key)

        if self.event_times is not None:
            event_times = np.asarray(self.event_times).reshape(-1)
            if len(event_times) != row_count:
                raise ValueError(
                    f"{name}.event_times has {len(event_times)} rows; expected {row_count}"
                )
            _coerce_event_times(event_times, name=f"{name}.event_times")

        if self.sample_weights is not None:
            weights = np.asarray(self.sample_weights, dtype="float64").reshape(-1)
            if len(weights) != row_count:
                raise ValueError(
                    f"{name}.sample_weights has {len(weights)} rows; expected {row_count}"
                )
            if not np.isfinite(weights).all() or (weights < 0.0).any():
                raise ValueError(
                    f"{name}.sample_weights must contain finite, non-negative values"
                )


@dataclass(frozen=True)
class DCNTemporalDataset:
    """Pre-split candidate rows; callers retain ownership of candidate generation."""

    train: DCNCandidateSplit
    validation: DCNCandidateSplit
    test: DCNCandidateSplit

    def validate(self) -> None:
        self.train.validate(name="train", allow_empty=False)
        self.validation.validate(name="validation")
        self.test.validate(name="test")

        context_width = np.asarray(self.train.context_features).shape[1]
        item_width = np.asarray(self.train.item_features).shape[1]
        for name, split in (
            ("validation", self.validation),
            ("test", self.test),
        ):
            if np.asarray(split.context_features).shape[1] != context_width:
                raise ValueError(
                    f"{name}.context_features must have {context_width} columns"
                )
            if np.asarray(split.item_features).shape[1] != item_width:
                raise ValueError(f"{name}.item_features must have {item_width} columns")

        nonempty = [
            (name, split)
            for name, split in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
            if len(split)
        ]
        supplied_times = [split.event_times is not None for _name, split in nonempty]
        if any(supplied_times) and not all(supplied_times):
            raise ValueError(
                "event_times must be supplied for every non-empty temporal split or none"
            )
        if all(supplied_times):
            previous_name = ""
            previous_max: int | float | None = None
            for name, split in nonempty:
                minimum, maximum = _event_time_bounds(
                    np.asarray(split.event_times).reshape(-1),
                    name=f"{name}.event_times",
                )
                if previous_max is not None and minimum < previous_max:
                    raise ValueError(
                        f"temporal split order is invalid: {name} begins before "
                        f"{previous_name} ends"
                    )
                previous_name = name
                previous_max = maximum


@dataclass(frozen=True)
class DCNTrainingConfig:
    output_dir: Path
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_seed: int = 42
    max_train_rows: int | None = 100_000
    max_validation_rows: int | None = 25_000
    max_test_rows: int | None = 25_000
    k_values: tuple[int, ...] = (10, 50, 100)
    class_weighting: str = "balanced"
    positive_class_weight: float | None = None
    standardize_features: bool = True
    early_stopping_patience: int | None = 2
    model_params: Mapping[str, object] = field(default_factory=dict)
    checkpoint_filename: str = "dcn_v2.keras"
    result_filename: str = "dcn_v2_results.json"

    def validate(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be positive")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be positive")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a positive finite number")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        for name, value in (
            ("max_train_rows", self.max_train_rows),
            ("max_validation_rows", self.max_validation_rows),
            ("max_test_rows", self.max_test_rows),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be positive or None")
        if not self.k_values or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 1
            for k in self.k_values
        ):
            raise ValueError("k_values must contain positive integers")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values must not contain duplicates")
        if self.class_weighting not in {"none", "balanced"}:
            raise ValueError("class_weighting must be 'none' or 'balanced'")
        if self.positive_class_weight is not None and (
            isinstance(self.positive_class_weight, bool)
            or not isinstance(self.positive_class_weight, (int, float))
            or not math.isfinite(self.positive_class_weight)
            or self.positive_class_weight <= 0.0
        ):
            raise ValueError("positive_class_weight must be positive and finite")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 0
        ):
            raise ValueError(
                "early_stopping_patience must be a non-negative integer or None"
            )
        for name, value in (
            ("checkpoint_filename", self.checkpoint_filename),
            ("result_filename", self.result_filename),
        ):
            if not value or Path(value).name != value:
                raise ValueError(f"{name} must be a non-empty filename")


def _validate_ids(values: np.ndarray, *, name: str) -> None:
    for value in values.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable scalar values") from exc
        if value is None:
            raise ValueError(f"{name} must not contain missing values")
        try:
            if bool(np.asarray(value != value).item()):
                raise ValueError(f"{name} must not contain missing values")
        except (TypeError, ValueError):
            pass


def _coerce_event_times(values: np.ndarray, *, name: str) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype("float64", copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} must contain finite values")
        return numeric
    try:
        timestamps = values.astype("datetime64[ns]").astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric or datetime-like values") from exc
    if (timestamps == np.iinfo("int64").min).any():
        raise ValueError(f"{name} must not contain missing timestamps")
    return timestamps


def _event_time_bounds(
    values: np.ndarray,
    *,
    name: str,
) -> tuple[int | float, int | float]:
    coerced = _coerce_event_times(values, name=name)
    return coerced.min().item(), coerced.max().item()


def _take_split(split: DCNCandidateSplit, indices: np.ndarray) -> DCNCandidateSplit:
    def optional(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else np.asarray(values)[indices]

    return DCNCandidateSplit(
        context_features=np.asarray(split.context_features)[indices].astype(
            "float32", copy=False
        ),
        item_features=np.asarray(split.item_features)[indices].astype(
            "float32", copy=False
        ),
        labels=np.asarray(split.labels).reshape(-1)[indices].astype(
            "float32", copy=False
        ),
        query_ids=np.asarray(split.query_ids).reshape(-1)[indices],
        candidate_ids=optional(split.candidate_ids),
        event_times=optional(split.event_times),
        sample_weights=optional(split.sample_weights),
    )


def _bounded_split(
    split: DCNCandidateSplit,
    *,
    max_rows: int | None,
    seed: int,
) -> DCNCandidateSplit:
    row_count = len(split)
    if max_rows is None or row_count <= max_rows:
        return _take_split(split, np.arange(row_count, dtype="int64"))
    query_ids = np.asarray(split.query_ids).reshape(-1)
    groups: dict[object, list[int]] = {}
    for row_index, query_id in enumerate(query_ids.tolist()):
        groups.setdefault(query_id, []).append(row_index)

    group_values = list(groups.values())
    order = np.random.default_rng(seed).permutation(len(group_values))
    selected: list[int] = []
    for group_index in order.tolist():
        rows = group_values[group_index]
        if len(selected) + len(rows) <= max_rows:
            selected.extend(rows)
    if not selected:
        selected = group_values[int(order[0])][:max_rows]
    return _take_split(split, np.asarray(sorted(selected), dtype="int64"))


def compute_dcn_training_weights(
    split: DCNCandidateSplit,
    *,
    class_weighting: str = "balanced",
    positive_class_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine caller sample weights with optional binary class weighting."""
    split.validate(name="training_split", allow_empty=False)
    if class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be 'none' or 'balanced'")
    if positive_class_weight is not None and (
        not math.isfinite(positive_class_weight) or positive_class_weight <= 0.0
    ):
        raise ValueError("positive_class_weight must be positive and finite")

    labels = np.asarray(split.labels, dtype="int8").reshape(-1)
    weights = (
        np.ones(len(split), dtype="float32")
        if split.sample_weights is None
        else np.asarray(split.sample_weights, dtype="float32").reshape(-1).copy()
    )
    negative_factor = 1.0
    positive_factor = 1.0
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    if class_weighting == "balanced" and negative_count and positive_count:
        total = negative_count + positive_count
        negative_factor = total / (2.0 * negative_count)
        positive_factor = total / (2.0 * positive_count)
    if positive_class_weight is not None:
        positive_factor *= float(positive_class_weight)
    weights[labels == 0] *= negative_factor
    weights[labels == 1] *= positive_factor
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("combined training sample weights must have positive mass")
    return weights, {
        "negative": float(negative_factor),
        "positive": float(positive_factor),
    }


def evaluate_dcn_scores(
    split: DCNCandidateSplit,
    scores: np.ndarray,
    *,
    k_values: Sequence[int] = (10, 50, 100),
) -> dict[str, object]:
    """Compute pointwise and query-grouped ranking metrics."""
    split.validate(name="evaluation_split")
    normalized_k = tuple(int(k) for k in k_values)
    if not normalized_k or any(k < 1 for k in normalized_k):
        raise ValueError("k_values must contain positive integers")
    predictions = np.asarray(scores, dtype="float64").reshape(-1)
    if len(predictions) != len(split):
        raise ValueError(f"scores has {len(predictions)} rows; expected {len(split)}")
    if not np.isfinite(predictions).all():
        raise ValueError("scores must contain only finite values")
    if ((predictions < 0.0) | (predictions > 1.0)).any():
        raise ValueError("scores must be probabilities in [0, 1]")
    if not len(split):
        return {
            "status": "unavailable",
            "reason": "empty_split",
            "row_count": 0,
            "positive_count": 0,
            "pointwise": {"roc_auc": None, "log_loss": None},
            "ranking": {
                "query_count": 0,
                "evaluated_query_count": 0,
                "queries_without_positive": 0,
                "recall_at_k": {str(k): None for k in normalized_k},
                "ndcg_at_k": {str(k): None for k in normalized_k},
                "mrr_at_k": {str(k): None for k in normalized_k},
            },
        }

    labels = np.asarray(split.labels, dtype="int8").reshape(-1)
    weights = (
        np.ones(len(split), dtype="float64")
        if split.sample_weights is None
        else np.asarray(split.sample_weights, dtype="float64").reshape(-1)
    )
    weight_sum = float(np.sum(weights))
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    losses = -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))
    log_loss = (
        float(np.sum(losses * weights) / weight_sum) if weight_sum > 0.0 else None
    )
    auc = None
    if np.unique(labels[weights > 0.0]).size == 2:
        auc = float(roc_auc_score(labels, predictions, sample_weight=weights))

    query_ids = np.asarray(split.query_ids).reshape(-1)
    candidate_ids = (
        None
        if split.candidate_ids is None
        else np.asarray(split.candidate_ids).reshape(-1)
    )
    groups: dict[object, list[int]] = {}
    for row_index, query_id in enumerate(query_ids.tolist()):
        groups.setdefault(query_id, []).append(row_index)

    recalls = {k: [] for k in normalized_k}
    ndcgs = {k: [] for k in normalized_k}
    reciprocal_ranks = {k: [] for k in normalized_k}
    queries_without_positive = 0
    for row_indices in groups.values():
        indices = np.asarray(row_indices, dtype="int64")
        group_labels = labels[indices]
        positive_count = int(np.sum(group_labels))
        if positive_count == 0:
            queries_without_positive += 1
            continue
        tie_ids = (
            np.asarray([str(value) for value in candidate_ids[indices]], dtype=str)
            if candidate_ids is not None
            else indices.astype(str)
        )
        order = np.lexsort((indices, tie_ids, -predictions[indices]))
        ranked_labels = group_labels[order]
        for k in normalized_k:
            top_labels = ranked_labels[:k]
            hit_positions = np.flatnonzero(top_labels > 0)
            recalls[k].append(float(np.sum(top_labels) / positive_count))
            discounts = 1.0 / np.log2(np.arange(2, len(top_labels) + 2))
            dcg = float(np.sum(top_labels * discounts))
            ideal_count = min(positive_count, k)
            ideal_dcg = float(
                np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2)))
            )
            ndcgs[k].append(dcg / ideal_dcg if ideal_dcg else 0.0)
            reciprocal_ranks[k].append(
                0.0 if not len(hit_positions) else 1.0 / (int(hit_positions[0]) + 1)
            )

    def means(values: Mapping[int, list[float]]) -> dict[str, float | None]:
        return {
            str(k): (float(np.mean(rows)) if rows else None)
            for k, rows in values.items()
        }

    return {
        "status": "ok",
        "row_count": len(split),
        "positive_count": int(np.sum(labels)),
        "positive_rate": float(np.mean(labels)),
        "pointwise": {
            "roc_auc": auc,
            "log_loss": log_loss,
        },
        "ranking": {
            "query_count": len(groups),
            "evaluated_query_count": len(groups) - queries_without_positive,
            "queries_without_positive": queries_without_positive,
            "recall_at_k": means(recalls),
            "ndcg_at_k": means(ndcgs),
            "mrr_at_k": means(reciprocal_ranks),
        },
    }


def _fit_standardizer(
    train: DCNCandidateSplit,
) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}
    for name, values in (
        ("context", np.asarray(train.context_features, dtype="float64")),
        ("item", np.asarray(train.item_features, dtype="float64")),
    ):
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        transforms[f"{name}_mean"] = mean.astype("float32")
        transforms[f"{name}_scale"] = np.where(scale < 1e-6, 1.0, scale).astype(
            "float32"
        )
    return transforms


def _transform_split(
    split: DCNCandidateSplit,
    transforms: Mapping[str, np.ndarray],
) -> DCNCandidateSplit:
    indices = np.arange(len(split), dtype="int64")
    copied = _take_split(split, indices)
    return DCNCandidateSplit(
        context_features=(
            np.asarray(copied.context_features) - transforms["context_mean"]
        )
        / transforms["context_scale"],
        item_features=(
            np.asarray(copied.item_features) - transforms["item_mean"]
        )
        / transforms["item_scale"],
        labels=copied.labels,
        query_ids=copied.query_ids,
        candidate_ids=copied.candidate_ids,
        event_times=copied.event_times,
        sample_weights=copied.sample_weights,
    )


def _predict_scores(model, split: DCNCandidateSplit, *, batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(split), batch_size):
        stop = min(len(split), start + batch_size)
        batch = {
            "context_input": np.asarray(split.context_features)[start:stop],
            "item_input": np.asarray(split.item_features)[start:stop],
        }
        predictions.append(
            np.asarray(model(batch, training=False)).reshape(-1).astype("float64")
        )
    return (
        np.concatenate(predictions)
        if predictions
        else np.empty(0, dtype="float64")
    )


def _split_summary(split: DCNCandidateSplit) -> dict[str, object]:
    labels = np.asarray(split.labels).reshape(-1)
    summary: dict[str, object] = {
        "row_count": len(split),
        "query_count": len(set(np.asarray(split.query_ids).reshape(-1).tolist())),
        "positive_count": int(np.sum(labels)) if len(labels) else 0,
    }
    if split.event_times is not None and len(split):
        minimum, maximum = _event_time_bounds(
            np.asarray(split.event_times).reshape(-1),
            name="event_times",
        )
        summary["event_time_min"] = minimum
        summary["event_time_max"] = maximum
    return summary


def train_dcn_v2_reranker(
    dataset: DCNTemporalDataset,
    config: DCNTrainingConfig,
    *,
    trained_model_callback: Callable[[object, DCNTemporalDataset], None] | None = None,
) -> dict[str, object]:
    """Train, evaluate, and persist a bounded pointwise DCN-V2 reranker."""
    dataset.validate()
    config.validate()

    bounded = DCNTemporalDataset(
        train=_bounded_split(
            dataset.train,
            max_rows=config.max_train_rows,
            seed=config.random_seed,
        ),
        validation=_bounded_split(
            dataset.validation,
            max_rows=config.max_validation_rows,
            seed=config.random_seed + 1,
        ),
        test=_bounded_split(
            dataset.test,
            max_rows=config.max_test_rows,
            seed=config.random_seed + 2,
        ),
    )
    bounded.validate()
    transforms = _fit_standardizer(bounded.train)
    if config.standardize_features:
        bounded = DCNTemporalDataset(
            train=_transform_split(bounded.train, transforms),
            validation=_transform_split(bounded.validation, transforms),
            test=_transform_split(bounded.test, transforms),
        )
    else:
        transforms = {
            "context_mean": np.zeros(
                np.asarray(bounded.train.context_features).shape[1], dtype="float32"
            ),
            "context_scale": np.ones(
                np.asarray(bounded.train.context_features).shape[1], dtype="float32"
            ),
            "item_mean": np.zeros(
                np.asarray(bounded.train.item_features).shape[1], dtype="float32"
            ),
            "item_scale": np.ones(
                np.asarray(bounded.train.item_features).shape[1], dtype="float32"
            ),
        }

    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.random_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    model = build_dcn_v2_model(
        num_context_features=np.asarray(bounded.train.context_features).shape[1],
        num_item_features=np.asarray(bounded.train.item_features).shape[1],
        params=config.model_params,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        run_eagerly=True,
    )
    train_weights, class_factors = compute_dcn_training_weights(
        bounded.train,
        class_weighting=config.class_weighting,
        positive_class_weight=config.positive_class_weight,
    )

    rng = np.random.default_rng(config.random_seed)
    history: list[dict[str, object]] = []
    best_validation_loss = math.inf
    best_weights: list[np.ndarray] | None = None
    stale_epochs = 0
    for epoch_index in range(config.epochs):
        order = rng.permutation(len(bounded.train))
        batch_losses: list[float] = []
        batch_sizes: list[int] = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            result = model.train_on_batch(
                {
                    "context_input": np.asarray(bounded.train.context_features)[indices],
                    "item_input": np.asarray(bounded.train.item_features)[indices],
                },
                np.asarray(bounded.train.labels).reshape(-1)[indices],
                sample_weight=train_weights[indices],
                return_dict=True,
            )
            batch_losses.append(float(result["loss"]))
            batch_sizes.append(len(indices))
        epoch_row: dict[str, object] = {
            "epoch": epoch_index + 1,
            "train_loss": float(np.average(batch_losses, weights=batch_sizes)),
        }
        if len(bounded.validation):
            validation_scores = _predict_scores(
                model,
                bounded.validation,
                batch_size=config.batch_size,
            )
            validation_metrics = evaluate_dcn_scores(
                bounded.validation,
                validation_scores,
                k_values=config.k_values,
            )
            validation_loss = validation_metrics["pointwise"]["log_loss"]
            epoch_row["validation_log_loss"] = validation_loss
            if validation_loss is not None and validation_loss < best_validation_loss:
                best_validation_loss = float(validation_loss)
                best_weights = model.get_weights()
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(epoch_row)
        if (
            len(bounded.validation)
            and config.early_stopping_patience is not None
            and stale_epochs > config.early_stopping_patience
        ):
            break
    if best_weights is not None:
        model.set_weights(best_weights)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.checkpoint_filename
    model.save(checkpoint_path)

    if trained_model_callback is not None:
        trained_model_callback(model, bounded)

    metrics = {
        name: evaluate_dcn_scores(
            split,
            _predict_scores(model, split, batch_size=config.batch_size),
            k_values=config.k_values,
        )
        for name, split in (
            ("train", bounded.train),
            ("validation", bounded.validation),
            ("test", bounded.test),
        )
    }
    result: dict[str, object] = {
        "status": "completed",
        "model_name": "dcn_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "feature_contract": {
            "context_feature_count": int(
                np.asarray(bounded.train.context_features).shape[1]
            ),
            "item_feature_count": int(
                np.asarray(bounded.train.item_features).shape[1]
            ),
            "candidate_ids_used_for_tie_breaking": any(
                split.candidate_ids is not None
                for split in (bounded.train, bounded.validation, bounded.test)
            ),
        },
        "config": {
            "epochs_requested": config.epochs,
            "epochs_completed": len(history),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "random_seed": config.random_seed,
            "k_values": list(config.k_values),
            "class_weighting": config.class_weighting,
            "positive_class_weight": config.positive_class_weight,
            "standardize_features": config.standardize_features,
            "early_stopping_patience": config.early_stopping_patience,
            "model_params": dict(config.model_params),
        },
        "class_weight_factors": class_factors,
        "input_split_summaries": {
            name: _split_summary(split)
            for name, split in (
                ("train", dataset.train),
                ("validation", dataset.validation),
                ("test", dataset.test),
            )
        },
        "split_summaries": {
            name: _split_summary(split)
            for name, split in (
                ("train", bounded.train),
                ("validation", bounded.validation),
                ("test", bounded.test),
            )
        },
        "preprocessing": {
            name: values.tolist() for name, values in transforms.items()
        },
        "history": history,
        "metrics": metrics,
    }
    write_json(output_dir / config.result_filename, result)
    return result


__all__ = [
    "DCNCandidateSplit",
    "DCNTemporalDataset",
    "DCNTrainingConfig",
    "compute_dcn_training_weights",
    "evaluate_dcn_scores",
    "train_dcn_v2_reranker",
]
