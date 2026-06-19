from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import logging
import math
from pathlib import Path
import pickle
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import load_streaming_history
from .recommender_next_pass import (
    NextPassAdapters,
    NextPassConfig,
    StageRequest,
    run_recommender_next_pass,
)
from .run_artifacts import safe_read_json, write_json, write_markdown
from .track_dcn_training import (
    DCNCandidateSplit,
    DCNTemporalDataset,
    DCNTrainingConfig,
    evaluate_dcn_scores,
    train_dcn_v2_reranker,
)
from .track_expansion_gates import PromotionPolicy, evaluate_track_expansion_gates
from .track_expansion_training import (
    TrackModelData,
    TrackTrainingConfig,
    _stream_retrieval_metrics,
    _top_catalog_interactions,
    _train_meantime,
    _train_multitask,
    prepare_track_model_data,
    reconstruct_session_interactions,
)
from .track_expansion_tuning import (
    ObjectiveResult,
    TrackExpansionTuningConfig,
    TrackObjectiveContext,
    run_track_expansion_tuning,
)
from .track_level_data import (
    TrackLevelDataset,
    TrackLevelTemporalSplits,
    build_track_level_dataset,
    split_track_level_examples,
)
from .track_public_pretraining import PublicPretrainingConfig, run_public_pretraining
from .track_reranking_data import (
    CANDIDATE_FEATURE_NAMES,
    TrackRerankingConfig,
    TrackRerankingData,
    TrackRerankingSplit,
    build_track_reranking_data,
    save_track_reranking_data,
)
from .track_retrieval import EASERetriever, SessionCooccurrenceRetriever


SUPPORTED_TUNING_MODELS = (
    "session_cooccurrence",
    "ease",
    "meantime",
    "mmoe",
    "ple",
    "dcn_v2",
)


@dataclass(frozen=True)
class TrackNextPassConfig:
    raw_data_dir: Path
    output_dir: Path
    include_video: bool = True
    max_history: int = 256
    sequence_length: int = 64
    candidate_count: int = 30
    retrieval_pool_size: int = 100
    reranking_max_items: int = 1_500
    max_train_queries: int = 3_000
    max_validation_queries: int = 750
    max_test_queries: int = 750
    dcn_epochs: int = 2
    dcn_batch_size: int = 256
    tuning_models: tuple[str, ...] = SUPPORTED_TUNING_MODELS
    tuning_trials: int = 1
    tuning_max_train_examples: int = 2_000
    tuning_max_validation_examples: int = 500
    tuning_max_test_examples: int = 500
    public_manifest: Path | None = None
    public_records: Path | None = None
    random_seed: int = 42

    def validate(self) -> None:
        positive = (
            "max_history",
            "sequence_length",
            "candidate_count",
            "retrieval_pool_size",
            "reranking_max_items",
            "max_train_queries",
            "max_validation_queries",
            "max_test_queries",
            "dcn_epochs",
            "dcn_batch_size",
            "tuning_max_train_examples",
            "tuning_max_validation_examples",
            "tuning_max_test_examples",
        )
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.retrieval_pool_size < self.candidate_count:
            raise ValueError("retrieval_pool_size must be at least candidate_count")
        if self.tuning_trials < 0:
            raise ValueError("tuning_trials cannot be negative")
        unknown = sorted(set(self.tuning_models) - set(SUPPORTED_TUNING_MODELS))
        if unknown:
            raise ValueError(f"Unknown tuning models: {', '.join(unknown)}")
        if (self.public_manifest is None) != (self.public_records is None):
            raise ValueError(
                "public_manifest and public_records must be supplied together"
            )


@dataclass
class _NextPassState:
    dataset: TrackLevelDataset | None = None
    splits: TrackLevelTemporalSplits | None = None
    reranking: TrackRerankingData | None = None
    dcn_dataset: DCNTemporalDataset | None = None
    dcn_result: dict[str, object] | None = None
    tuning_result: dict[str, object] | None = None


def _dataset_fingerprint(splits: TrackLevelTemporalSplits) -> str:
    digest = hashlib.sha256()
    for split_name in ("train", "validation", "test"):
        digest.update(split_name.encode("ascii"))
        for example in getattr(splits, split_name):
            digest.update(
                (
                    f"{example.example_id}|{example.session_id}|"
                    f"{example.target_timestamp.isoformat()}|"
                    f"{example.target_track_uri}\n"
                ).encode("utf-8")
            )
    return digest.hexdigest()


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _expanded_query_ids(split: TrackRerankingSplit) -> np.ndarray:
    return np.asarray(
        [split.query_ids[int(group_id)] for group_id in split.group_ids],
        dtype=np.str_,
    )


def reranking_split_to_dcn(split: TrackRerankingSplit) -> DCNCandidateSplit:
    return DCNCandidateSplit(
        context_features=np.asarray(split.context_features, dtype="float32"),
        item_features=np.asarray(split.candidate_features, dtype="float32"),
        labels=np.asarray(split.labels, dtype="float32"),
        query_ids=_expanded_query_ids(split),
        candidate_ids=np.asarray(split.candidate_track_uris, dtype=np.str_),
    )


def reranking_data_to_dcn(data: TrackRerankingData) -> DCNTemporalDataset:
    return DCNTemporalDataset(
        train=reranking_split_to_dcn(data.train),
        validation=reranking_split_to_dcn(data.validation),
        test=reranking_split_to_dcn(data.test),
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    truth = np.asarray(labels, dtype="float64").reshape(-1)
    scores = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(truth) != len(scores) or not len(truth):
        raise ValueError("labels and probabilities must have the same non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    scores = np.clip(scores, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    bucket = np.maximum(bucket, 0)
    total = float(len(truth))
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if not np.any(mask):
            continue
        error += (
            float(np.sum(mask))
            / total
            * abs(float(np.mean(truth[mask])) - float(np.mean(scores[mask])))
        )
    return float(error)


def _predict_dcn(model, split: DCNCandidateSplit, *, batch_size: int) -> np.ndarray:
    if not len(split):
        return np.empty(0, dtype="float64")
    context = np.asarray(split.context_features, dtype="float32")
    items = np.asarray(split.item_features, dtype="float32")
    predictions: list[np.ndarray] = []
    for start in range(0, len(split), batch_size):
        stop = min(len(split), start + batch_size)
        batch = model(
            {
                "context_input": context[start:stop],
                "item_input": items[start:stop],
            },
            training=False,
        )
        predictions.append(np.asarray(batch, dtype="float64").reshape(-1))
    return np.concatenate(predictions)


def _standardized_mean_drift(
    train: np.ndarray,
    evaluation: np.ndarray,
) -> float:
    train_values = np.asarray(train, dtype="float64")
    evaluation_values = np.asarray(evaluation, dtype="float64")
    if not len(train_values) or not len(evaluation_values):
        return math.inf
    train_mean = np.mean(train_values, axis=0)
    train_scale = np.std(train_values, axis=0)
    train_scale = np.where(train_scale < 1e-6, 1.0, train_scale)
    return float(np.max(np.abs(np.mean(evaluation_values, axis=0) - train_mean) / train_scale))


def _ranking_value(
    result: Mapping[str, object],
    split_name: str,
    metric_name: str,
    k: int,
) -> float | None:
    metrics = result.get("metrics", {})
    split = metrics.get(split_name, {}) if isinstance(metrics, Mapping) else {}
    ranking = split.get("ranking", {}) if isinstance(split, Mapping) else {}
    values = ranking.get(metric_name, {}) if isinstance(ranking, Mapping) else {}
    value = values.get(str(k)) if isinstance(values, Mapping) else None
    return float(value) if isinstance(value, (int, float)) else None


def _popularity_candidate_baseline(
    data: TrackRerankingData,
    dataset: DCNTemporalDataset,
) -> dict[str, object]:
    feature_index = CANDIDATE_FEATURE_NAMES.index("popularity_reciprocal_rank")
    scale = data.candidate_standardizer.scale[feature_index]
    mean = data.candidate_standardizer.mean[feature_index]

    def evaluate(
        source: TrackRerankingSplit,
        target: DCNCandidateSplit,
    ) -> dict[str, object]:
        raw_scores = source.candidate_features[:, feature_index] * scale + mean
        scores = np.clip(np.asarray(raw_scores, dtype="float64"), 0.0, 1.0)
        return evaluate_dcn_scores(target, scores, k_values=(10, 50, 100))

    validation = evaluate(data.validation, dataset.validation)
    test = evaluate(data.test, dataset.test)
    validation_ranking = validation.get("ranking", {})
    test_ranking = test.get("ranking", {})
    if not isinstance(validation_ranking, Mapping) or not isinstance(
        test_ranking, Mapping
    ):
        raise RuntimeError("DCN evaluation did not return ranking metrics")
    validation_ndcg = validation_ranking.get("ndcg_at_k", {})
    validation_recall = validation_ranking.get("recall_at_k", {})
    test_ndcg = test_ranking.get("ndcg_at_k", {})
    test_recall = test_ranking.get("recall_at_k", {})
    if not all(
        isinstance(values, Mapping)
        for values in (validation_ndcg, validation_recall, test_ndcg, test_recall)
    ):
        raise RuntimeError("DCN evaluation ranking metrics are malformed")
    return {
        "model_name": "candidate_popularity",
        "ndcg_at_10": validation_ndcg.get("10"),
        "validation": {
            "ndcg_at_10": validation_ndcg.get("10"),
            "recall_at_100": validation_recall.get("100"),
        },
        "test": {
            "ndcg_at_10": test_ndcg.get("10"),
            "recall_at_100": test_recall.get("100"),
        },
    }


def _write_dcn_shap(
    *,
    model,
    dataset: DCNTemporalDataset,
    output_dir: Path,
    random_seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "dcn_v2_shap_values.pkl"
    summary_path = output_dir / "dcn_v2_shap_summary.json"
    rng = np.random.default_rng(random_seed)
    background_count = min(64, len(dataset.train))
    explain_source = dataset.validation if len(dataset.validation) else dataset.test
    explain_count = min(64, len(explain_source))
    if not background_count or not explain_count:
        payload: dict[str, object] = {
            "status": "unavailable",
            "method": "gradient_shap",
            "reason": "Non-empty training and evaluation candidate rows are required.",
        }
        write_json(summary_path, payload)
        return payload

    background_rows = np.sort(
        rng.choice(len(dataset.train), size=background_count, replace=False)
    )
    explain_rows = np.sort(
        rng.choice(len(explain_source), size=explain_count, replace=False)
    )
    background = [
        np.asarray(dataset.train.context_features)[background_rows],
        np.asarray(dataset.train.item_features)[background_rows],
    ]
    explained = [
        np.asarray(explain_source.context_features)[explain_rows],
        np.asarray(explain_source.item_features)[explain_rows],
    ]
    try:
        import tensorflow as tf

        context_baseline = np.mean(background[0], axis=0, keepdims=True)
        item_baseline = np.mean(background[1], axis=0, keepdims=True)
        context_delta = explained[0] - context_baseline
        item_delta = explained[1] - item_baseline
        context_gradients = np.zeros_like(explained[0], dtype="float64")
        item_gradients = np.zeros_like(explained[1], dtype="float64")
        alpha_count = 8
        for alpha in np.linspace(0.0625, 0.9375, alpha_count):
            context_batch = tf.convert_to_tensor(
                context_baseline + alpha * context_delta,
                dtype=tf.float32,
            )
            item_batch = tf.convert_to_tensor(
                item_baseline + alpha * item_delta,
                dtype=tf.float32,
            )
            with tf.GradientTape() as tape:
                tape.watch((context_batch, item_batch))
                predictions = model(
                    {
                        "context_input": context_batch,
                        "item_input": item_batch,
                    },
                    training=False,
                )
                score = tf.reduce_sum(predictions)
            context_gradient, item_gradient = tape.gradient(
                score,
                (context_batch, item_batch),
            )
            context_gradients += np.asarray(context_gradient, dtype="float64")
            item_gradients += np.asarray(item_gradient, dtype="float64")
        values = {
            "context": context_delta * (context_gradients / alpha_count),
            "item": item_delta * (item_gradients / alpha_count),
        }
        with artifact.open("wb") as outfile:
            pickle.dump(
                {
                    "model_name": "dcn_v2",
                    "method": "gradient_shap",
                    "values": values,
                    "background_rows": background_rows,
                    "explain_rows": explain_rows,
                },
                outfile,
            )
        payload = {
            "status": "complete",
            "method": "gradient_shap",
            "artifact": str(artifact.resolve()),
            "artifact_present": True,
            "background_rows": background_count,
            "explained_rows": explain_count,
        }
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "method": "gradient_shap",
            "reason": f"{type(exc).__name__}: {exc}",
            "artifact_present": False,
        }
    write_json(summary_path, payload)
    return payload


def _collect_dcn_evidence(
    *,
    model,
    reranking: TrackRerankingData,
    dataset: DCNTemporalDataset,
    output_dir: Path,
    random_seed: int,
) -> dict[str, object]:
    validation_scores = _predict_dcn(model, dataset.validation, batch_size=512)
    test_scores = _predict_dcn(model, dataset.test, batch_size=512)
    validation_ece = (
        expected_calibration_error(dataset.validation.labels, validation_scores)
        if len(validation_scores)
        else None
    )
    test_ece = (
        expected_calibration_error(dataset.test.labels, test_scores)
        if len(test_scores)
        else None
    )
    explanation = _write_dcn_shap(
        model=model,
        dataset=dataset,
        output_dir=output_dir / "explainability",
        random_seed=random_seed,
    )
    popularity = _popularity_candidate_baseline(reranking, dataset)
    return {
        "popularity_baseline": popularity,
        "calibration": {
            "status": "complete" if test_ece is not None else "unavailable",
            "validation_ece": validation_ece,
            "test_ece": test_ece,
            "ece": test_ece,
        },
        "explainability": explanation,
        "drift": {
            "status": "complete",
            "validation_drift_score": max(
                _standardized_mean_drift(
                    dataset.train.context_features,
                    dataset.validation.context_features,
                ),
                _standardized_mean_drift(
                    dataset.train.item_features,
                    dataset.validation.item_features,
                ),
            ),
            "test_drift_score": max(
                _standardized_mean_drift(
                    dataset.train.context_features,
                    dataset.test.context_features,
                ),
                _standardized_mean_drift(
                    dataset.train.item_features,
                    dataset.test.item_features,
                ),
            ),
        },
    }


def _enrich_dcn_result(
    *,
    result: dict[str, object],
    evidence: Mapping[str, object],
    reranking: TrackRerankingData,
    dataset: DCNTemporalDataset,
    output_dir: Path,
    random_seed: int,
    dataset_fingerprint: str,
) -> dict[str, object]:
    validation_ndcg = _ranking_value(result, "validation", "ndcg_at_k", 10)
    test_ndcg = _ranking_value(result, "test", "ndcg_at_k", 10)
    validation_recall = _ranking_value(result, "validation", "recall_at_k", 100)
    test_recall = _ranking_value(result, "test", "recall_at_k", 100)
    enriched = {
        **result,
        "status": "complete",
        "model_family": "dcn",
        "checkpoint": result["checkpoint_path"],
        "artifacts": [result["checkpoint_path"]],
        "validation": {
            "ndcg_at_10": validation_ndcg,
            "recall_at_100": validation_recall,
            "target_catalog_coverage": (
                (
                    reranking.validation.source_example_count
                    - reranking.validation.skipped_oov_target_count
                )
                / max(1, reranking.validation.source_example_count)
            ),
        },
        "test": {
            "ndcg_at_10": test_ndcg,
            "recall_at_100": test_recall,
            "target_catalog_coverage": (
                (
                    reranking.test.source_example_count
                    - reranking.test.skipped_oov_target_count
                )
                / max(1, reranking.test.source_example_count)
            ),
        },
        "coverage": {
            "train_rows": len(dataset.train),
            "validation_rows": len(dataset.validation),
            "test_rows": len(dataset.test),
        },
        **dict(evidence),
        "reproducibility": {
            "random_seed": random_seed,
            "dataset_fingerprint": dataset_fingerprint,
            "temporal_split": "chronological_session_boundaries_64_16_20",
            "code_version": _code_version(),
        },
    }
    write_json(output_dir / "dcn_v2_evidence.json", enriched)
    return enriched


def _filter_item_support(interactions: pd.DataFrame, minimum: int) -> pd.DataFrame:
    if minimum <= 1:
        return interactions
    counts = interactions["track_id"].value_counts()
    retained = set(counts[counts >= minimum].index)
    return interactions.loc[interactions["track_id"].isin(retained)].copy()


def _multitask_score(result: Mapping[str, object]) -> float:
    validation = result.get("validation", {})
    if not isinstance(validation, Mapping):
        return 0.0
    ranking = validation.get("ndcg_at_k")
    weighted: list[tuple[float, float]] = []
    if isinstance(ranking, (int, float)) and math.isfinite(float(ranking)):
        weighted.append((0.7, float(ranking)))
    auxiliary = validation.get("auxiliary", {})
    if isinstance(auxiliary, Mapping):
        for name in ("skip_output", "session_end_output", "repeat_output"):
            metrics = auxiliary.get(name, {})
            if not isinstance(metrics, Mapping):
                continue
            value = metrics.get("auc")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                value = metrics.get("accuracy")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                weighted.append((0.1, float(value)))
    total_weight = sum(weight for weight, _value in weighted)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(weight * value for weight, value in weighted) / total_weight)


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Expected finite metric {key!r}")
    return float(value)


def _build_tuning_objectives(
    *,
    config: TrackNextPassConfig,
    splits: TrackLevelTemporalSplits,
    dcn_dataset: DCNTemporalDataset,
    artifact_dir: Path,
) -> Mapping[str, object]:
    interactions = reconstruct_session_interactions(splits.train)
    model_data: TrackModelData | None = None
    neural_models = {"meantime", "mmoe", "ple"}.intersection(config.tuning_models)
    if neural_models:
        model_data = prepare_track_model_data(
            splits.train,
            splits.validation,
            splits.test,
            max_items=config.reranking_max_items,
            sequence_length=config.sequence_length,
            max_train_examples=config.tuning_max_train_examples,
            max_validation_examples=config.tuning_max_validation_examples,
            max_test_examples=config.tuning_max_test_examples,
        )

    def session_cooccurrence(context: TrackObjectiveContext) -> ObjectiveResult:
        params = context.params
        bounded = _top_catalog_interactions(interactions, int(params["max_items"]))
        retriever = SessionCooccurrenceRetriever().fit(
            bounded,
            shrinkage=float(params["shrinkage"]),
        )
        metrics = _stream_retrieval_metrics(
            retriever,
            splits.validation,
            k=100,
            limit=config.tuning_max_validation_examples,
            history_window=int(params["history_window"]),
        )
        return ObjectiveResult(
            value=_required_float(metrics, "recall_at_k"),
            metadata={"validation": metrics},
        )

    def ease(context: TrackObjectiveContext) -> ObjectiveResult:
        params = context.params
        filtered = _filter_item_support(interactions, int(params["min_item_support"]))
        bounded = _top_catalog_interactions(filtered, int(params["max_items"]))
        retriever = EASERetriever().fit(
            bounded,
            l2=float(params["l2"]),
            binary=bool(params["binary_interactions"]),
        )
        metrics = _stream_retrieval_metrics(
            retriever,
            splits.validation,
            k=100,
            limit=config.tuning_max_validation_examples,
        )
        return ObjectiveResult(
            value=_required_float(metrics, "recall_at_k"),
            metadata={"validation": metrics},
        )

    def neural(context: TrackObjectiveContext) -> ObjectiveResult:
        assert model_data is not None
        params = dict(context.params)
        batch_size = int(params.pop("batch_size"))
        learning_rate = float(params.pop("learning_rate"))
        trial_dir = artifact_dir / context.model_name / f"trial_{context.trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        training_config = TrackTrainingConfig(
            raw_data_dir=config.raw_data_dir,
            output_dir=config.output_dir,
            include_video=config.include_video,
            max_history=config.max_history,
            sequence_length=config.sequence_length,
            evaluation_k=10,
            neural_models=(context.model_name,),
            neural_max_items=config.reranking_max_items,
            max_train_examples=config.tuning_max_train_examples,
            max_validation_examples=config.tuning_max_validation_examples,
            max_test_examples=config.tuning_max_test_examples,
            epochs=1,
            batch_size=batch_size,
            random_seed=config.random_seed + int(context.trial.number),
        )
        params["learning_rate"] = learning_rate
        if context.model_name == "meantime":
            result = _train_meantime(
                model_data,
                config=training_config,
                checkpoint_dir=trial_dir,
                model_params=params,
            )
            validation = result.get("validation", {})
            if not isinstance(validation, Mapping):
                raise RuntimeError("MEANTIME validation metrics are missing")
            value = _required_float(validation, "ndcg_at_k")
        else:
            result = _train_multitask(
                model_data,
                architecture=context.model_name,
                config=training_config,
                checkpoint_dir=trial_dir,
                model_params=params,
            )
            value = _multitask_score(result)
        return ObjectiveResult(
            value=value,
            metadata={
                "checkpoint": result.get("checkpoint"),
                "validation": result.get("validation"),
            },
        )

    def dcn(context: TrackObjectiveContext) -> ObjectiveResult:
        params = dict(context.params)
        batch_size = int(params.pop("batch_size"))
        learning_rate = float(params.pop("learning_rate"))
        trial_dir = artifact_dir / "dcn_v2" / f"trial_{context.trial.number}"
        result = train_dcn_v2_reranker(
            dcn_dataset,
            DCNTrainingConfig(
                output_dir=trial_dir,
                epochs=1,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=config.random_seed + int(context.trial.number),
                max_train_rows=None,
                max_validation_rows=None,
                max_test_rows=None,
                k_values=(10,),
                standardize_features=False,
                early_stopping_patience=None,
                model_params=params,
            ),
        )
        value = _ranking_value(result, "validation", "ndcg_at_k", 10)
        return ObjectiveResult(
            value=float(value or 0.0),
            metadata={
                "checkpoint": result.get("checkpoint_path"),
                "validation": result.get("metrics", {}).get("validation"),
            },
        )

    callbacks: dict[str, object] = {
        "session_cooccurrence": session_cooccurrence,
        "ease": ease,
        "meantime": neural,
        "mmoe": neural,
        "ple": neural,
        "dcn_v2": dcn,
    }
    return {
        name: callbacks[name]
        for name in config.tuning_models
    }


def _tuning_gate_manifest(summary: Mapping[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    studies = summary.get("studies", ())
    if isinstance(studies, Sequence) and not isinstance(studies, (str, bytes)):
        for study in studies:
            if not isinstance(study, Mapping):
                continue
            best = study.get("best_trial")
            if not isinstance(best, Mapping):
                continue
            rows.append(
                {
                    "model_name": study.get("model_name"),
                    "completed_trials": study.get("completed_trials"),
                    "trial_count": study.get("total_trials"),
                    "best_params": best.get("params"),
                    "parameters": best.get("params"),
                    "tuning_metric": best.get("metric_name"),
                    "tuning_value": best.get("value"),
                }
            )
    return {
        "status": summary.get("status"),
        "tuning_results": rows,
        "studies": studies,
    }


def _read_canonical_records(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def run_track_next_pass(
    *,
    config: TrackNextPassConfig,
    logger: logging.Logger,
) -> dict[str, object]:
    config.validate()
    state = _NextPassState()
    dataset_fingerprint = ""

    def ensure_data() -> tuple[TrackLevelDataset, TrackLevelTemporalSplits]:
        nonlocal dataset_fingerprint
        if state.dataset is None or state.splits is None:
            raw = load_streaming_history(
                config.raw_data_dir,
                include_video=config.include_video,
                logger=logger,
            )
            state.dataset = build_track_level_dataset(
                raw,
                max_history=max(config.max_history, config.sequence_length),
            )
            state.splits = split_track_level_examples(state.dataset)
            dataset_fingerprint = _dataset_fingerprint(state.splits)
        return state.dataset, state.splits

    def candidate_stage(request: StageRequest) -> dict[str, object]:
        dataset, splits = ensure_data()
        state.reranking = build_track_reranking_data(
            splits,
            config=TrackRerankingConfig(
                max_items=config.reranking_max_items,
                candidate_count=config.candidate_count,
                retrieval_pool_size=config.retrieval_pool_size,
                max_train_queries=config.max_train_queries,
                max_validation_queries=config.max_validation_queries,
                max_test_queries=config.max_test_queries,
                random_seed=config.random_seed,
            ),
        )
        manifest_path = save_track_reranking_data(state.reranking, request.artifact_dir)
        return {
            **state.reranking.to_manifest_dict(),
            "manifest_path": str(manifest_path.resolve()),
            "dataset_fingerprint": dataset_fingerprint,
            "source_dataset": {
                "examples": len(dataset.examples),
                "tracks": dataset.unique_track_count,
                "sessions": dataset.session_count,
            },
        }

    def dcn_stage(request: StageRequest) -> dict[str, object]:
        _dataset, _splits = ensure_data()
        if state.reranking is None:
            raise RuntimeError("Candidate data must be built before DCN training")
        state.dcn_dataset = reranking_data_to_dcn(state.reranking)
        evidence_rows: list[tuple[dict[str, object], DCNTemporalDataset]] = []

        def collect_evidence(model: object, bounded: DCNTemporalDataset) -> None:
            evidence_rows.append(
                (
                    _collect_dcn_evidence(
                        model=model,
                        reranking=state.reranking,
                        dataset=bounded,
                        output_dir=request.artifact_dir,
                        random_seed=config.random_seed,
                    ),
                    bounded,
                )
            )

        result = train_dcn_v2_reranker(
            state.dcn_dataset,
            DCNTrainingConfig(
                output_dir=request.artifact_dir,
                epochs=config.dcn_epochs,
                batch_size=config.dcn_batch_size,
                random_seed=config.random_seed,
                max_train_rows=None,
                max_validation_rows=None,
                max_test_rows=None,
                standardize_features=False,
                model_params={
                    "cross_layers": 3,
                    "cross_parameterization": "matrix",
                    "deep_units": (128, 64),
                    "dropout_rate": 0.1,
                    "architecture": "parallel",
                },
            ),
            trained_model_callback=collect_evidence,
        )
        if not evidence_rows:
            raise RuntimeError("DCN training did not produce evidence")
        evidence, evidence_dataset = evidence_rows[0]
        state.dcn_result = _enrich_dcn_result(
            result=result,
            evidence=evidence,
            reranking=state.reranking,
            dataset=evidence_dataset,
            output_dir=request.artifact_dir,
            random_seed=config.random_seed,
            dataset_fingerprint=dataset_fingerprint,
        )
        return state.dcn_result

    def tuning_stage(request: StageRequest) -> dict[str, object]:
        _dataset, splits = ensure_data()
        if state.dcn_dataset is None:
            raise RuntimeError("DCN candidate tensors must exist before tuning")
        objectives = _build_tuning_objectives(
            config=config,
            splits=splits,
            dcn_dataset=state.dcn_dataset,
            artifact_dir=request.artifact_dir / "trials",
        )
        summary = run_track_expansion_tuning(
            objectives=objectives,
            config=TrackExpansionTuningConfig(
                storage_path=request.artifact_dir / "track_expansion_optuna.db",
                output_dir=request.artifact_dir,
                selected_models=config.tuning_models,
                trial_budgets=config.tuning_trials,
                sampler_seed=config.random_seed,
                pruner="median",
                n_jobs=1,
                study_prefix="track_next_pass",
            ),
        )
        state.tuning_result = summary.to_dict()
        return state.tuning_result

    def public_stage(request: StageRequest) -> dict[str, object]:
        if config.public_manifest is None or config.public_records is None:
            payload: dict[str, object] = {
                "status": "blocked",
                "stage": "governance",
                "reason": (
                    "No approved public manifest and canonical local records were supplied."
                ),
            }
        else:
            result = run_public_pretraining(
                config.public_manifest,
                record_loader=lambda _manifest: _read_canonical_records(
                    config.public_records
                ),
                config=PublicPretrainingConfig(seed=config.random_seed),
            )
            payload = result.to_dict()
        write_json(request.artifact_dir / "public_pretraining_result.json", payload)
        return payload

    def gates_stage(request: StageRequest) -> dict[str, object]:
        root = config.output_dir / "analysis" / "recommender_expansion"
        baseline = safe_read_json(
            root / "track_popularity_baseline.json",
            default={},
        )
        training = safe_read_json(
            root / "training" / "training_manifest.json",
            default={},
        )
        tuning = _tuning_gate_manifest(state.tuning_result or {})
        public_output = request.upstream.get("public_pretraining", {})
        public_manifest = (
            public_output.get("output")
            if isinstance(public_output, Mapping)
            else None
        )
        report = evaluate_track_expansion_gates(
            baseline_manifest=baseline if isinstance(baseline, Mapping) else {},
            training_manifest=training if isinstance(training, Mapping) else None,
            tuning_manifest=tuning,
            dcn_manifest=state.dcn_result,
            public_pretraining_manifest=(
                public_manifest if isinstance(public_manifest, Mapping) else None
            ),
            policy=PromotionPolicy(),
            artifact_base_dir=config.output_dir,
        )
        payload = report.to_dict()
        write_json(request.artifact_dir / "promotion_gate_report.json", payload)
        write_markdown(
            request.artifact_dir / "promotion_gate_report.md",
            report.to_markdown(),
        )
        return {
            "status": "complete",
            "gate_status": report.status,
            "report_path": str(
                (request.artifact_dir / "promotion_gate_report.json").resolve()
            ),
            **payload,
        }

    adapters = NextPassAdapters(
        candidate_dataset_builder=candidate_stage,
        dcn_trainer=dcn_stage,
        optuna_tuners={"track_expansion": tuning_stage},
        public_pretrainer=public_stage,
        promotion_gates=gates_stage,
    )
    manifest = run_recommender_next_pass(
        config=NextPassConfig(
            output_dir=config.output_dir,
            enable_public_pretraining=config.public_manifest is not None,
            stage_options={"track_next_pass": asdict(config)},
        ),
        adapters=adapters,
        logger=logger,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run candidate generation, DCN-V2, Optuna, governed public-data "
            "checks, and promotion gates for the track recommender expansion."
        )
    )
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--retrieval-pool-size", type=int, default=100)
    parser.add_argument("--reranking-max-items", type=int, default=1_500)
    parser.add_argument("--max-train-queries", type=int, default=3_000)
    parser.add_argument("--max-validation-queries", type=int, default=750)
    parser.add_argument("--max-test-queries", type=int, default=750)
    parser.add_argument("--dcn-epochs", type=int, default=2)
    parser.add_argument("--dcn-batch-size", type=int, default=256)
    parser.add_argument("--tuning-models", default=",".join(SUPPORTED_TUNING_MODELS))
    parser.add_argument("--tuning-trials", type=int, default=1)
    parser.add_argument("--tuning-max-train-examples", type=int, default=2_000)
    parser.add_argument("--tuning-max-validation-examples", type=int, default=500)
    parser.add_argument("--tuning-max-test-examples", type=int, default=500)
    parser.add_argument("--public-manifest")
    parser.add_argument("--public-records")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("spotify.track_next_pass")
    manifest = run_track_next_pass(
        config=TrackNextPassConfig(
            raw_data_dir=Path(args.raw_data_dir),
            output_dir=Path(args.output_dir),
            include_video=not args.no_video,
            max_history=args.max_history,
            sequence_length=args.sequence_length,
            candidate_count=args.candidate_count,
            retrieval_pool_size=args.retrieval_pool_size,
            reranking_max_items=args.reranking_max_items,
            max_train_queries=args.max_train_queries,
            max_validation_queries=args.max_validation_queries,
            max_test_queries=args.max_test_queries,
            dcn_epochs=args.dcn_epochs,
            dcn_batch_size=args.dcn_batch_size,
            tuning_models=tuple(
                value.strip()
                for value in str(args.tuning_models).split(",")
                if value.strip()
            ),
            tuning_trials=args.tuning_trials,
            tuning_max_train_examples=args.tuning_max_train_examples,
            tuning_max_validation_examples=args.tuning_max_validation_examples,
            tuning_max_test_examples=args.tuning_max_test_examples,
            public_manifest=Path(args.public_manifest) if args.public_manifest else None,
            public_records=Path(args.public_records) if args.public_records else None,
            random_seed=args.random_seed,
        ),
        logger=logger,
    )
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        print(artifacts.get("manifest", ""))
    return 0 if manifest["status"] in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
