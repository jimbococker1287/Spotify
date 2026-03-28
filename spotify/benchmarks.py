from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json
import os
import time

import numpy as np

from .data import PreparedData
from .probability_bundles import align_proba_to_num_classes, save_prediction_bundle
from .ranking import ranking_metrics_from_proba


@dataclass
class ClassicalBenchmarkResult:
    model_name: str
    model_family: str
    fit_seconds: float
    val_top1: float
    val_top5: float
    val_ndcg_at5: float
    val_mrr_at5: float
    val_coverage_at5: float
    val_diversity_at5: float
    test_top1: float
    test_top5: float
    test_ndcg_at5: float
    test_mrr_at5: float
    test_coverage_at5: float
    test_diversity_at5: float
    prediction_bundle_path: str = ""
    estimator_artifact_path: str = ""


@dataclass(frozen=True)
class ClassicalFeatureBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def _sequence_feature_block(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] <= 0:
        raise ValueError("Expected a 2D non-empty sequence matrix.")

    row_count, seq_len = seq.shape
    seq_float = seq.astype("float32", copy=False)
    features = np.empty((row_count, 9), dtype="float32")

    features[:, 0] = seq_float[:, -1]
    if seq_len > 1:
        features[:, 1] = seq_float[:, -2]
        features[:, 7] = (seq[:, -1] == seq[:, -2]).astype("float32")
    else:
        features[:, 1] = seq_float[:, -1]
        features[:, 7].fill(1.0)
    features[:, 2] = seq_float[:, 0]
    features[:, 3] = seq_float.mean(axis=1, dtype="float32")
    features[:, 4] = seq_float.std(axis=1, dtype="float32")

    # Vectorize the previously row-wise uniqueness work so every downstream
    # classical stage can reuse a much faster feature path.
    sorted_seq = np.sort(seq, axis=1)
    unique_counts = np.ones(row_count, dtype="float32")
    if seq_len > 1:
        unique_counts += np.count_nonzero(sorted_seq[:, 1:] != sorted_seq[:, :-1], axis=1).astype("float32")
    features[:, 5] = unique_counts / float(seq_len)

    reversed_matches = seq[:, ::-1] == seq[:, -1:]
    all_same = reversed_matches.all(axis=1)
    first_diff = np.argmax(~reversed_matches, axis=1) + 1
    features[:, 6] = np.where(all_same, seq_len, first_diff).astype("float32")

    recent_width = min(5, seq_len)
    recent_sorted = np.sort(seq[:, -recent_width:], axis=1)
    recent_unique_counts = np.ones(row_count, dtype="float32")
    if recent_width > 1:
        recent_unique_counts += np.count_nonzero(
            recent_sorted[:, 1:] != recent_sorted[:, :-1],
            axis=1,
        ).astype("float32")
    features[:, 8] = recent_unique_counts / float(recent_width)

    return features


def build_tabular_features(data: PreparedData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.concatenate([_sequence_feature_block(data.X_seq_train), data.X_ctx_train.astype("float32", copy=False)], axis=1)
    X_val = np.concatenate([_sequence_feature_block(data.X_seq_val), data.X_ctx_val.astype("float32", copy=False)], axis=1)
    X_test = np.concatenate([_sequence_feature_block(data.X_seq_test), data.X_ctx_test.astype("float32", copy=False)], axis=1)
    return X_train, X_val, X_test


def build_classical_feature_bundle(data: PreparedData) -> ClassicalFeatureBundle:
    X_train, X_val, X_test = build_tabular_features(data)
    return ClassicalFeatureBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_train_seq=data.X_seq_train.astype("int32", copy=False),
        X_val_seq=data.X_seq_val.astype("int32", copy=False),
        X_test_seq=data.X_seq_test.astype("int32", copy=False),
        y_train=np.asarray(data.y_train),
        y_val=np.asarray(data.y_val),
        y_test=np.asarray(data.y_test),
    )


def build_serving_tabular_features(X_seq: np.ndarray, X_ctx: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _sequence_feature_block(np.asarray(X_seq)),
            np.asarray(X_ctx).astype("float32", copy=False),
        ],
        axis=1,
    )


def build_full_tabular_dataset(
    data: PreparedData,
    *,
    feature_bundle: ClassicalFeatureBundle | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bundle = feature_bundle if feature_bundle is not None else build_classical_feature_bundle(data)
    X = np.concatenate([bundle.X_train, bundle.X_val, bundle.X_test], axis=0)
    y = np.concatenate([bundle.y_train, bundle.y_val, bundle.y_test], axis=0)
    return X, y


def _encode_labels_to_local_indices(y_true: np.ndarray, class_labels: np.ndarray | None) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    if class_labels is None:
        return y_true.astype("int64", copy=False)
    labels = np.asarray(class_labels).reshape(-1)
    if labels.size == 0:
        return np.full(y_true.shape[0], -1, dtype="int64")
    lookup = {label.item() if isinstance(label, np.generic) else label: idx for idx, label in enumerate(labels)}
    mapped = np.fromiter((lookup.get(item.item() if isinstance(item, np.generic) else item, -1) for item in y_true), dtype="int64")
    if mapped.size != y_true.size:
        return np.full(y_true.shape[0], -1, dtype="int64")
    return mapped


def _topk_from_proba(
    proba: np.ndarray,
    y_true: np.ndarray,
    k: int,
    *,
    class_labels: np.ndarray | None = None,
) -> float:
    if proba.ndim != 2:
        return float("nan")
    kk = max(1, min(k, proba.shape[1]))
    topk_idx = np.argpartition(proba, -kk, axis=1)[:, -kk:]
    if class_labels is not None:
        labels = np.asarray(class_labels).reshape(-1)
        if labels.size == proba.shape[1]:
            topk_values = labels[topk_idx]
            y_values = np.asarray(y_true).reshape(-1, 1)
            return float(np.mean(np.any(topk_values == y_values, axis=1)))
    y_idx = np.asarray(y_true).reshape(-1, 1)
    return float(np.mean(np.any(topk_idx == y_idx, axis=1)))


def sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx], y[idx]


def sample_indices(n_rows: int, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype="int64")
    return np.asarray(rng.choice(n_rows, size=max_rows, replace=False), dtype="int64")


def resolve_classical_parallelism() -> tuple[int, int]:
    cpu_count = os.cpu_count() or 1
    workers_raw = os.getenv("SPOTIFY_CLASSICAL_MODEL_WORKERS", "1").strip()
    estimator_jobs_raw = os.getenv("SPOTIFY_SKLEARN_NJOBS", "").strip()
    max_workers_raw = os.getenv("SPOTIFY_MAX_CLASSICAL_WORKERS", "auto").strip().lower()

    workers = 1
    try:
        workers = max(1, min(cpu_count, int(workers_raw)))
    except Exception:
        workers = 1

    if max_workers_raw == "auto":
        try:
            import psutil  # type: ignore

            total_ram_gb = int(psutil.virtual_memory().total // (1024**3))
            if total_ram_gb < 12:
                workers = min(workers, 1)
            elif total_ram_gb < 18:
                workers = min(workers, 2)
            elif total_ram_gb < 26:
                workers = min(workers, 3)
            else:
                workers = min(workers, 4)
        except Exception:
            pass
    elif max_workers_raw:
        try:
            workers = min(workers, max(1, int(max_workers_raw)))
        except Exception:
            pass

    if estimator_jobs_raw:
        try:
            estimator_jobs = int(estimator_jobs_raw)
        except Exception:
            estimator_jobs = -1
    else:
        if workers > 1:
            # When we already parallelize across classical models, default to a
            # single internal estimator worker to avoid multiplying memory
            # pressure across tree/instance-based models.
            estimator_jobs = 1
        else:
            estimator_jobs = -1

    if estimator_jobs == 0:
        estimator_jobs = 1

    return workers, estimator_jobs


def get_classical_model_registry(
    random_seed: int,
    estimator_n_jobs: int = -1,
) -> dict[str, tuple[str, Callable[[dict[str, object] | None], object]]]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .session_knn import SessionKNNClassifier

    def build_logreg(params: dict[str, object] | None):
        params = params or {}
        c_value = float(params.get("C", 1.0))
        max_iter = int(params.get("max_iter", 500))
        solver = str(params.get("solver", "lbfgs"))
        if solver not in ("lbfgs", "saga", "newton-cg", "sag", "liblinear"):
            solver = "lbfgs"
        use_n_jobs = estimator_n_jobs if solver in ("saga", "sag", "liblinear") else None
        kwargs = {
            "C": c_value,
            "max_iter": max_iter,
            "solver": solver,
        }
        if use_n_jobs is not None:
            kwargs["n_jobs"] = use_n_jobs
        return make_pipeline(StandardScaler(), LogisticRegression(**kwargs))

    def build_random_forest(params: dict[str, object] | None):
        params = params or {}
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=(int(params["max_depth"]) if "max_depth" in params and params["max_depth"] is not None else None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_seed,
            n_jobs=estimator_n_jobs,
            class_weight="balanced_subsample",
        )

    def build_extra_trees(params: dict[str, object] | None):
        params = params or {}
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 350)),
            max_depth=(int(params["max_depth"]) if "max_depth" in params and params["max_depth"] is not None else None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_seed,
            n_jobs=estimator_n_jobs,
            class_weight="balanced_subsample",
        )

    def build_hist_gbm(params: dict[str, object] | None):
        params = params or {}
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 250)),
            learning_rate=float(params.get("learning_rate", 0.06)),
            max_depth=int(params.get("max_depth", 12)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=random_seed,
            # Avoid internal stratified validation split that can fail on rare classes
            # in sampled training subsets used by tuning/backtesting.
            early_stopping=False,
        )

    def build_knn(params: dict[str, object] | None):
        params = params or {}
        n_neighbors = int(params.get("n_neighbors", 35))
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=estimator_n_jobs),
        )

    def build_nb(_params: dict[str, object] | None):
        return make_pipeline(StandardScaler(), GaussianNB())

    def build_mlp(params: dict[str, object] | None):
        params = params or {}
        h1 = int(params.get("hidden_1", 256))
        h2 = int(params.get("hidden_2", 128))
        alpha = float(params.get("alpha", 1e-4))
        learning_rate_init = float(params.get("learning_rate_init", 1e-3))
        batch_size = int(params.get("batch_size", 256))
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(h1, h2),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                max_iter=int(params.get("max_iter", 80)),
                early_stopping=True,
                random_state=random_seed,
            ),
        )

    def build_session_knn(params: dict[str, object] | None):
        params = params or {}
        return SessionKNNClassifier(
            n_neighbors=int(params.get("n_neighbors", 64)),
            candidate_cap=int(params.get("candidate_cap", 512)),
            smoothing=float(params.get("smoothing", 1.0)),
        )

    def build_catboost(params: dict[str, object] | None):
        params = params or {}
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise ImportError(
                "catboost is not installed. Install the project dependencies again or run "
                "'.venv/bin/pip install catboost'."
            ) from exc

        thread_count = estimator_n_jobs if estimator_n_jobs > 0 else -1
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=int(params.get("iterations", 180)),
            depth=int(params.get("depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.10)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 5.0)),
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
            thread_count=thread_count,
        )

    return {
        "logreg": ("linear", build_logreg),
        "random_forest": ("tree_ensemble", build_random_forest),
        "extra_trees": ("tree_ensemble", build_extra_trees),
        "hist_gbm": ("tree_ensemble", build_hist_gbm),
        "knn": ("instance_based", build_knn),
        "gaussian_nb": ("probabilistic", build_nb),
        "mlp": ("shallow_neural", build_mlp),
        "session_knn": ("session_memory", build_session_knn),
        "catboost": ("boosting", build_catboost),
    }


def validate_classical_models(selected_models: tuple[str, ...], random_seed: int) -> None:
    registry = get_classical_model_registry(random_seed)
    unknown = [name for name in selected_models if name not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown classical model names: {', '.join(unknown)}. Known models: {known}")


def build_classical_estimator(
    model_name: str,
    random_seed: int,
    params: dict[str, object] | None = None,
    estimator_n_jobs: int = -1,
) -> tuple[str, object]:
    registry = get_classical_model_registry(random_seed, estimator_n_jobs=estimator_n_jobs)
    if model_name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown classical model name: {model_name}. Known models: {known}")
    family, builder = registry[model_name]
    return family, builder(params)


def evaluate_classical_estimator(
    estimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float, dict[str, float], dict[str, float]]:
    val_top1 = float("nan")
    test_top1 = float("nan")
    val_top5 = float("nan")
    test_top5 = float("nan")
    val_ranking = {
        "ndcg_at5": float("nan"),
        "mrr_at5": float("nan"),
        "coverage_at5": float("nan"),
        "diversity_at5": float("nan"),
    }
    test_ranking = {
        "ndcg_at5": float("nan"),
        "mrr_at5": float("nan"),
        "coverage_at5": float("nan"),
        "diversity_at5": float("nan"),
    }
    if hasattr(estimator, "predict_proba"):
        try:
            val_proba = np.asarray(estimator.predict_proba(X_val))
            test_proba = np.asarray(estimator.predict_proba(X_test))
            classes = np.asarray(getattr(estimator, "classes_", []))
            if classes.size == val_proba.shape[1]:
                val_pred = classes[np.argmax(val_proba, axis=1)]
            else:
                val_pred = np.argmax(val_proba, axis=1)
            if classes.size == test_proba.shape[1]:
                test_pred = classes[np.argmax(test_proba, axis=1)]
            else:
                test_pred = np.argmax(test_proba, axis=1)
            val_top1 = float(np.mean(val_pred == y_val))
            test_top1 = float(np.mean(test_pred == y_test))
            val_top5 = _topk_from_proba(val_proba, y_val, k=5, class_labels=classes)
            test_top5 = _topk_from_proba(test_proba, y_test, k=5, class_labels=classes)
            val_class_count = int(val_proba.shape[1])
            test_class_count = int(test_proba.shape[1])

            val_y_rank = _encode_labels_to_local_indices(np.asarray(y_val), classes if classes.size else None)
            test_y_rank = _encode_labels_to_local_indices(np.asarray(y_test), classes if classes.size else None)

            val_extra = ranking_metrics_from_proba(val_proba, val_y_rank, num_items=val_class_count, k=5)
            test_extra = ranking_metrics_from_proba(
                test_proba,
                test_y_rank,
                num_items=test_class_count,
                k=5,
            )
            val_ranking = {
                "ndcg_at5": float(val_extra["ndcg_at_k"]),
                "mrr_at5": float(val_extra["mrr_at_k"]),
                "coverage_at5": float(val_extra["coverage_at_k"]),
                "diversity_at5": float(val_extra["diversity_at_k"]),
            }
            test_ranking = {
                "ndcg_at5": float(test_extra["ndcg_at_k"]),
                "mrr_at5": float(test_extra["mrr_at_k"]),
                "coverage_at5": float(test_extra["coverage_at_k"]),
                "diversity_at5": float(test_extra["diversity_at_k"]),
            }
        except Exception:
            pass
    if np.isnan(val_top1) or np.isnan(test_top1):
        val_pred = estimator.predict(X_val)
        test_pred = estimator.predict(X_test)
        val_top1 = float(np.mean(val_pred == y_val))
        test_top1 = float(np.mean(test_pred == y_test))
    return val_top1, val_top5, test_top1, test_top5, val_ranking, test_ranking


def collect_aligned_probabilities(
    estimator,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        val_proba = np.asarray(estimator.predict_proba(X_val))
        test_proba = np.asarray(estimator.predict_proba(X_test))
    except Exception:
        return None
    classes = np.asarray(getattr(estimator, "classes_", []))
    aligned_val = align_proba_to_num_classes(val_proba, classes if classes.size else None, num_classes)
    aligned_test = align_proba_to_num_classes(test_proba, classes if classes.size else None, num_classes)
    return aligned_val, aligned_test


def _fit_single_classical_model(
    model_name: str,
    random_seed: int,
    estimator_n_jobs: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val_full: np.ndarray,
    X_test_full: np.ndarray,
    num_classes: int,
    prediction_output_dir: Path,
    estimator_output_dir: Path,
) -> ClassicalBenchmarkResult:
    family, estimator = build_classical_estimator(
        model_name,
        random_seed,
        estimator_n_jobs=estimator_n_jobs,
    )
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    estimator_artifact_path = ""
    try:
        import joblib

        estimator_path = estimator_output_dir / f"classical_{model_name}.joblib"
        estimator_output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, estimator_path, compress=3)
        estimator_artifact_path = str(estimator_path)
    except Exception:
        estimator_artifact_path = ""
    val_top1, val_top5, test_top1, test_top5, val_ranking, test_ranking = evaluate_classical_estimator(
        estimator,
        X_val,
        y_val,
        X_test,
        y_test,
    )
    prediction_bundle_path = ""
    aligned_probs = collect_aligned_probabilities(estimator, X_val_full, X_test_full, num_classes=num_classes)
    if aligned_probs is not None:
        val_proba, test_proba = aligned_probs
        bundle_path = save_prediction_bundle(
            prediction_output_dir / f"classical_{model_name}.npz",
            val_proba=val_proba,
            test_proba=test_proba,
        )
        prediction_bundle_path = str(bundle_path)
    return ClassicalBenchmarkResult(
        model_name=model_name,
        model_family=family,
        fit_seconds=float(fit_seconds),
        val_top1=val_top1,
        val_top5=val_top5,
        val_ndcg_at5=float(val_ranking["ndcg_at5"]),
        val_mrr_at5=float(val_ranking["mrr_at5"]),
        val_coverage_at5=float(val_ranking["coverage_at5"]),
        val_diversity_at5=float(val_ranking["diversity_at5"]),
        test_top1=test_top1,
        test_top5=test_top5,
        test_ndcg_at5=float(test_ranking["ndcg_at5"]),
        test_mrr_at5=float(test_ranking["mrr_at5"]),
        test_coverage_at5=float(test_ranking["coverage_at5"]),
        test_diversity_at5=float(test_ranking["diversity_at5"]),
        prediction_bundle_path=prediction_bundle_path,
        estimator_artifact_path=estimator_artifact_path,
    )


def run_classical_benchmarks(
    data: PreparedData,
    output_dir: Path,
    selected_models: tuple[str, ...],
    random_seed: int,
    max_train_samples: int,
    max_eval_samples: int,
    logger,
    feature_bundle: ClassicalFeatureBundle | None = None,
) -> list[ClassicalBenchmarkResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_classical_models(selected_models, random_seed)

    bundle = feature_bundle if feature_bundle is not None else build_classical_feature_bundle(data)
    X_train, X_val, X_test = bundle.X_train, bundle.X_val, bundle.X_test
    y_train, y_val, y_test = bundle.y_train, bundle.y_val, bundle.y_test
    X_val_full = X_val
    X_test_full = X_test
    X_train_seq = bundle.X_train_seq
    X_val_seq = bundle.X_val_seq
    X_test_seq = bundle.X_test_seq
    X_val_seq_full = X_val_seq
    X_test_seq_full = X_test_seq

    rng = np.random.default_rng(random_seed)
    train_idx = sample_indices(len(X_train), max_train_samples, rng)
    val_idx = sample_indices(len(X_val), max_eval_samples, rng)
    test_idx = sample_indices(len(X_test), max_eval_samples, rng)

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_val = X_val[val_idx]
    y_val = y_val[val_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]
    X_train_seq = X_train_seq[train_idx]
    X_val_seq = X_val_seq[val_idx]
    X_test_seq = X_test_seq[test_idx]

    logger.info(
        "Classical benchmark dataset sizes: train=%d, val=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    prediction_output_dir = output_dir / "prediction_bundles"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    estimator_output_dir = output_dir / "estimators"
    estimator_output_dir.mkdir(parents=True, exist_ok=True)

    workers, estimator_n_jobs = resolve_classical_parallelism()
    if workers > 1 and len(selected_models) > 1:
        logger.info(
            "Classical model parallelism enabled: workers=%d estimator_n_jobs=%d",
            workers,
            estimator_n_jobs,
        )
    else:
        logger.info("Classical model parallelism: workers=1 estimator_n_jobs=%d", estimator_n_jobs)

    results: list[ClassicalBenchmarkResult] = []
    if workers > 1 and len(selected_models) > 1:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for name in selected_models:
                logger.info("Training classical model %s", name)
                future = executor.submit(
                    _fit_single_classical_model,
                    name,
                    random_seed,
                    estimator_n_jobs,
                    X_train_seq if name == "session_knn" else X_train,
                    y_train,
                    X_val_seq if name == "session_knn" else X_val,
                    y_val,
                    X_test_seq if name == "session_knn" else X_test,
                    y_test,
                    X_val_seq_full if name == "session_knn" else X_val_full,
                    X_test_seq_full if name == "session_knn" else X_test_full,
                    data.num_artists,
                    prediction_output_dir,
                    estimator_output_dir,
                )
                futures[future] = name

            ordered: dict[str, ClassicalBenchmarkResult] = {}
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                ordered[name] = result
                logger.info(
                    "[CLASSICAL][TEST] %s: Top-1=%.4f | Top-5=%s",
                    name,
                    result.test_top1,
                    f"{result.test_top5:.4f}" if not np.isnan(result.test_top5) else "n/a",
                )
            results = [ordered[name] for name in selected_models if name in ordered]
    else:
        for name in selected_models:
            logger.info("Training classical model %s", name)
            result = _fit_single_classical_model(
                model_name=name,
                random_seed=random_seed,
                estimator_n_jobs=estimator_n_jobs,
                X_train=(X_train_seq if name == "session_knn" else X_train),
                y_train=y_train,
                X_val=(X_val_seq if name == "session_knn" else X_val),
                y_val=y_val,
                X_test=(X_test_seq if name == "session_knn" else X_test),
                y_test=y_test,
                X_val_full=(X_val_seq_full if name == "session_knn" else X_val_full),
                X_test_full=(X_test_seq_full if name == "session_knn" else X_test_full),
                num_classes=data.num_artists,
                prediction_output_dir=prediction_output_dir,
                estimator_output_dir=estimator_output_dir,
            )
            logger.info(
                "[CLASSICAL][TEST] %s: Top-1=%.4f | Top-5=%s",
                name,
                result.test_top1,
                f"{result.test_top5:.4f}" if not np.isnan(result.test_top5) else "n/a",
            )
            results.append(result)

    payload = [asdict(r) for r in results]
    with (output_dir / "classical_results.json").open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)

    return results
