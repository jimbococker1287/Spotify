from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json
import time

import numpy as np

from .data import PreparedData


@dataclass
class ClassicalBenchmarkResult:
    model_name: str
    model_family: str
    fit_seconds: float
    val_top1: float
    val_top5: float
    test_top1: float
    test_top5: float


def _sequence_feature_block(seq: np.ndarray) -> np.ndarray:
    seq = seq.astype("float32")
    last_artist = seq[:, -1:]
    first_artist = seq[:, :1]
    seq_mean = seq.mean(axis=1, keepdims=True)
    seq_std = seq.std(axis=1, keepdims=True)
    unique_ratio = np.array([len(np.unique(row)) / float(len(row)) for row in seq], dtype="float32").reshape(-1, 1)
    return np.concatenate([last_artist, first_artist, seq_mean, seq_std, unique_ratio], axis=1)


def build_tabular_features(data: PreparedData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.concatenate([_sequence_feature_block(data.X_seq_train), data.X_ctx_train.astype("float32")], axis=1)
    X_val = np.concatenate([_sequence_feature_block(data.X_seq_val), data.X_ctx_val.astype("float32")], axis=1)
    X_test = np.concatenate([_sequence_feature_block(data.X_seq_test), data.X_ctx_test.astype("float32")], axis=1)
    return X_train, X_val, X_test


def build_full_tabular_dataset(data: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    X_train, X_val, X_test = build_tabular_features(data)
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate(
        [
            data.y_train.astype(int),
            data.y_val.astype(int),
            data.y_test.astype(int),
        ],
        axis=0,
    )
    return X, y


def _topk_from_proba(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if proba.ndim != 2:
        return float("nan")
    kk = max(1, min(k, proba.shape[1]))
    topk_idx = np.argpartition(proba, -kk, axis=1)[:, -kk:]
    return float(np.mean(np.any(topk_idx == y_true.reshape(-1, 1), axis=1)))


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


def get_classical_model_registry(
    random_seed: int,
) -> dict[str, tuple[str, Callable[[dict[str, object] | None], object]]]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def build_logreg(params: dict[str, object] | None):
        params = params or {}
        c_value = float(params.get("C", 1.0))
        max_iter = int(params.get("max_iter", 500))
        return make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=max_iter, solver="lbfgs"))

    def build_random_forest(params: dict[str, object] | None):
        params = params or {}
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=(int(params["max_depth"]) if "max_depth" in params and params["max_depth"] is not None else None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    def build_extra_trees(params: dict[str, object] | None):
        params = params or {}
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 350)),
            max_depth=(int(params["max_depth"]) if "max_depth" in params and params["max_depth"] is not None else None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_seed,
            n_jobs=-1,
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
        )

    def build_knn(params: dict[str, object] | None):
        params = params or {}
        n_neighbors = int(params.get("n_neighbors", 35))
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1))

    def build_nb(_params: dict[str, object] | None):
        return make_pipeline(StandardScaler(), GaussianNB())

    def build_mlp(params: dict[str, object] | None):
        params = params or {}
        h1 = int(params.get("hidden_1", 256))
        h2 = int(params.get("hidden_2", 128))
        alpha = float(params.get("alpha", 1e-4))
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(h1, h2),
                alpha=alpha,
                max_iter=int(params.get("max_iter", 60)),
                early_stopping=True,
                random_state=random_seed,
            ),
        )

    return {
        "logreg": ("linear", build_logreg),
        "random_forest": ("tree_ensemble", build_random_forest),
        "extra_trees": ("tree_ensemble", build_extra_trees),
        "hist_gbm": ("tree_ensemble", build_hist_gbm),
        "knn": ("instance_based", build_knn),
        "gaussian_nb": ("probabilistic", build_nb),
        "mlp": ("shallow_neural", build_mlp),
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
) -> tuple[str, object]:
    registry = get_classical_model_registry(random_seed)
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
) -> tuple[float, float, float, float]:
    val_pred = estimator.predict(X_val)
    test_pred = estimator.predict(X_test)
    val_top1 = float(np.mean(val_pred == y_val))
    test_top1 = float(np.mean(test_pred == y_test))

    val_top5 = float("nan")
    test_top5 = float("nan")
    if hasattr(estimator, "predict_proba"):
        try:
            val_proba = estimator.predict_proba(X_val)
            test_proba = estimator.predict_proba(X_test)
            val_top5 = _topk_from_proba(np.asarray(val_proba), y_val, k=5)
            test_top5 = _topk_from_proba(np.asarray(test_proba), y_test, k=5)
        except Exception:
            pass
    return val_top1, val_top5, test_top1, test_top5


def run_classical_benchmarks(
    data: PreparedData,
    output_dir: Path,
    selected_models: tuple[str, ...],
    random_seed: int,
    max_train_samples: int,
    max_eval_samples: int,
    logger,
) -> list[ClassicalBenchmarkResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_classical_models(selected_models, random_seed)

    X_train, X_val, X_test = build_tabular_features(data)
    y_train = data.y_train.astype(int)
    y_val = data.y_val.astype(int)
    y_test = data.y_test.astype(int)

    rng = np.random.default_rng(random_seed)
    X_train, y_train = sample_rows(X_train, y_train, max_train_samples, rng)
    X_val, y_val = sample_rows(X_val, y_val, max_eval_samples, rng)
    X_test, y_test = sample_rows(X_test, y_test, max_eval_samples, rng)

    logger.info(
        "Classical benchmark dataset sizes: train=%d, val=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    results: list[ClassicalBenchmarkResult] = []
    for name in selected_models:
        family, estimator = build_classical_estimator(name, random_seed)
        logger.info("Training classical model %s", name)

        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        val_top1, val_top5, test_top1, test_top5 = evaluate_classical_estimator(estimator, X_val, y_val, X_test, y_test)

        result = ClassicalBenchmarkResult(
            model_name=name,
            model_family=family,
            fit_seconds=float(fit_seconds),
            val_top1=val_top1,
            val_top5=val_top5,
            test_top1=test_top1,
            test_top5=test_top5,
        )
        logger.info(
            "[CLASSICAL][TEST] %s: Top-1=%.4f | Top-5=%s",
            name,
            test_top1,
            f"{test_top5:.4f}" if not np.isnan(test_top5) else "n/a",
        )
        results.append(result)

    payload = [asdict(r) for r in results]
    with (output_dir / "classical_results.json").open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)

    return results
