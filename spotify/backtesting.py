from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import os
import csv
import time

import numpy as np

from .benchmarks import (
    ClassicalFeatureBundle,
    build_classical_estimator,
    build_full_tabular_dataset,
    get_classical_model_registry,
    resolve_classical_parallelism,
)
from .data import PreparedData
from .recommender_safety import build_temporal_backtest_windows, write_temporal_backtest_artifacts


@dataclass
class BacktestFoldResult:
    model_name: str
    model_type: str
    model_family: str
    fold: int
    train_rows: int
    test_rows: int
    fit_seconds: float
    top1: float
    top5: float
    adaptation_mode: str = "cold"


@dataclass(frozen=True)
class _BacktestWindowSlices:
    fold: int
    train_slice: slice
    test_slice: slice
    raw_test_start: int
    raw_test_end: int


def _build_expanding_windows(n_rows: int, folds: int) -> list[tuple[int, int]]:
    return [(window.test_start, window.test_end) for window in build_temporal_backtest_windows(n_rows, folds)]


def _run_backtest_job(
    model_name: str,
    fold_idx: int,
    random_seed: int,
    estimator_n_jobs: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> BacktestFoldResult:
    family, estimator = build_classical_estimator(
        model_name,
        random_seed + fold_idx,
        estimator_n_jobs=estimator_n_jobs,
    )
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = float(time.perf_counter() - started)
    top1, top5 = _score_backtest_estimator(estimator, X_test, y_test)
    return BacktestFoldResult(
        model_name=model_name,
        model_type="classical",
        model_family=family,
        adaptation_mode="cold",
        fold=fold_idx,
        train_rows=len(X_train),
        test_rows=len(X_test),
        fit_seconds=fit_seconds,
        top1=top1,
        top5=top5,
    )


def _extract_artist_predictions(prediction) -> np.ndarray:
    if isinstance(prediction, dict):
        if "artist_output" in prediction:
            return np.asarray(prediction["artist_output"])
        first_value = next(iter(prediction.values()))
        return np.asarray(first_value)
    if isinstance(prediction, (list, tuple)):
        return np.asarray(prediction[0])
    return np.asarray(prediction)


def _topk_accuracy_from_proba(proba: np.ndarray, y_true: np.ndarray, *, k: int) -> float:
    proba_arr = np.asarray(proba)
    y_arr = np.asarray(y_true).reshape(-1)
    if proba_arr.ndim != 2 or len(proba_arr) != len(y_arr):
        return float("nan")
    kk = max(1, min(int(k), int(proba_arr.shape[1])))
    topk_idx = np.argpartition(proba_arr, -kk, axis=1)[:, -kk:]
    hits = np.any(topk_idx == y_arr.reshape(-1, 1), axis=1)
    return float(np.mean(hits))


def _topk_accuracy_from_labeled_proba(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    k: int,
    class_labels: np.ndarray | None = None,
) -> float:
    proba_arr = np.asarray(proba)
    y_arr = np.asarray(y_true).reshape(-1)
    if proba_arr.ndim != 2 or len(proba_arr) != len(y_arr):
        return float("nan")
    labels = None if class_labels is None else np.asarray(class_labels).reshape(-1)
    if labels is not None and labels.size == proba_arr.shape[1]:
        kk = max(1, min(int(k), int(proba_arr.shape[1])))
        topk_idx = np.argpartition(proba_arr, -kk, axis=1)[:, -kk:]
        topk_labels = labels[topk_idx]
        return float(np.mean(np.any(topk_labels == y_arr.reshape(-1, 1), axis=1)))
    return _topk_accuracy_from_proba(proba_arr, y_arr, k=k)


def _score_backtest_estimator(estimator, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    y_arr = np.asarray(y_test).reshape(-1)
    if len(y_arr) == 0:
        return float("nan"), float("nan")
    if hasattr(estimator, "predict_proba"):
        try:
            proba = np.asarray(estimator.predict_proba(X_test))
        except Exception:
            proba = np.empty((0, 0), dtype="float32")
        if proba.ndim == 2 and len(proba) == len(y_arr):
            classes = np.asarray(getattr(estimator, "classes_", []))
            if classes.size == proba.shape[1]:
                pred = classes[np.argmax(proba, axis=1)]
                top5 = _topk_accuracy_from_labeled_proba(proba, y_arr, k=5, class_labels=classes)
            else:
                pred = np.argmax(proba, axis=1)
                top5 = _topk_accuracy_from_proba(proba, y_arr, k=5)
            return float(np.mean(pred == y_arr)), top5
    pred = estimator.predict(X_test)
    return float(np.mean(np.asarray(pred).reshape(-1) == y_arr)), float("nan")


def _deep_train_validation_split(n_rows: int, reference_rows: int) -> tuple[slice, slice] | None:
    if n_rows <= 1:
        return None
    val_rows = min(n_rows - 1, max(1, min(reference_rows, max(1, n_rows // 5))))
    split_idx = n_rows - val_rows
    if split_idx <= 0:
        return None
    return slice(0, split_idx), slice(split_idx, n_rows)


def _resolve_backtest_model_groups(selected_models: tuple[str, ...], random_seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from .config import DEFAULT_MODEL_NAMES

    classical_registry = get_classical_model_registry(random_seed)
    deep_registry = set(DEFAULT_MODEL_NAMES)
    classical: list[str] = []
    deep: list[str] = []
    unknown: list[str] = []

    for model_name in selected_models:
        if model_name in classical_registry:
            classical.append(model_name)
        elif model_name in deep_registry:
            deep.append(model_name)
        else:
            unknown.append(model_name)

    if unknown:
        known = ", ".join(sorted(set(classical_registry) | deep_registry))
        raise ValueError(f"Unknown temporal backtest model names: {', '.join(unknown)}. Known models: {known}")

    return tuple(classical), tuple(deep)


def _resolve_deep_builders(
    *,
    data: PreparedData,
    selected_models: tuple[str, ...],
    deep_model_builders,
) -> dict[str, object]:
    if not selected_models:
        return {}

    if deep_model_builders is None:
        from .modeling import build_model_builders

        built = build_model_builders(
            sequence_length=int(data.X_seq_train.shape[1]),
            num_artists=int(data.num_artists),
            num_ctx=int(data.num_ctx),
            selected_names=selected_models,
        )
        return {name: builder for name, builder in built}

    if isinstance(deep_model_builders, dict):
        resolved = dict(deep_model_builders)
    else:
        resolved = {name: builder for name, builder in deep_model_builders}

    missing = [name for name in selected_models if name not in resolved]
    if missing:
        raise ValueError(f"Missing deep temporal backtest builders for: {', '.join(missing)}")
    return resolved


def _resolve_deep_strategy(strategy, logger):
    if strategy is not None:
        return strategy

    from .runtime import configure_process_env, load_tensorflow_runtime, select_distribution_strategy

    configure_process_env()
    tf = load_tensorflow_runtime(logger)
    return select_distribution_strategy(tf, logger=logger)


def _resolve_backtest_adaptation_mode(raw: str | None) -> str:
    value = str(raw or os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold")).strip().lower()
    if value not in ("cold", "warm", "continual"):
        return "cold"
    return value


def _resolve_backtest_sample_cap(env_name: str, fallback: int) -> int:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return max(0, int(fallback))
    try:
        return max(0, int(raw_value))
    except Exception:
        return max(0, int(fallback))


def _tail_slice_between(start: int, end: int, max_rows: int) -> slice:
    if max_rows <= 0 or end - start <= max_rows:
        return slice(start, end)
    return slice(end - max_rows, end)


def _resolve_backtest_windows(
    windows: list[tuple[int, int]],
    *,
    max_train_samples: int,
    max_eval_samples: int,
) -> list[_BacktestWindowSlices]:
    resolved: list[_BacktestWindowSlices] = []
    for fold_idx, (test_start, test_end) in enumerate(windows, start=1):
        train_slice = _tail_slice_between(0, test_start, max_train_samples)
        test_slice = _tail_slice_between(test_start, test_end, max_eval_samples)
        if train_slice.stop <= train_slice.start or test_slice.stop <= test_slice.start:
            continue
        resolved.append(
            _BacktestWindowSlices(
                fold=fold_idx,
                train_slice=train_slice,
                test_slice=test_slice,
                raw_test_start=test_start,
                raw_test_end=test_end,
            )
        )
    return resolved


def _resolve_backtest_workers(raw_value: str | None, *, job_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    capped_jobs = max(1, int(job_count))
    raw = str(raw_value or "").strip().lower()
    if raw and raw != "auto":
        try:
            return min(cpu_count, capped_jobs, max(1, int(raw)))
        except Exception:
            return 1
    if capped_jobs <= 1:
        return 1
    return min(cpu_count, capped_jobs, 2)


def _run_deep_backtest_job(
    *,
    model_name: str,
    fold_idx: int,
    random_seed: int,
    model_builder,
    strategy,
    batch_size: int,
    epochs: int,
    X_seq_fit: np.ndarray,
    X_ctx_fit: np.ndarray,
    y_fit: np.ndarray,
    y_skip_fit: np.ndarray,
    X_seq_val: np.ndarray,
    X_ctx_val: np.ndarray,
    y_val: np.ndarray,
    y_skip_val: np.ndarray,
    X_seq_test: np.ndarray,
    X_ctx_test: np.ndarray,
    y_test: np.ndarray,
    initial_weights=None,
    weight_sink: dict[str, object] | None = None,
    adaptation_mode: str = "cold",
) -> BacktestFoldResult:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    try:
        try:
            tf.keras.utils.set_random_seed(int(random_seed))
        except Exception:
            tf.random.set_seed(int(random_seed))
        started = time.perf_counter()
        with strategy.scope():
            model = model_builder()
            single_head = len(model.outputs) == 1
            if single_head:
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=[
                        "sparse_categorical_accuracy",
                        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5"),
                    ],
                )
                fit_targets = y_fit
                val_targets = y_val
                monitor_metric = "val_sparse_categorical_accuracy"
            else:
                model.compile(
                    optimizer="adam",
                    loss={
                        "artist_output": "sparse_categorical_crossentropy",
                        "skip_output": "binary_crossentropy",
                    },
                    metrics={
                        "artist_output": [
                            "sparse_categorical_accuracy",
                            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5"),
                        ],
                        "skip_output": ["accuracy"],
                    },
                    loss_weights={"artist_output": 1.0, "skip_output": 0.2},
                )
                fit_targets = {
                    "artist_output": y_fit,
                    "skip_output": y_skip_fit,
                }
                val_targets = {
                    "artist_output": y_val,
                    "skip_output": y_skip_val,
                }
                monitor_metric = "val_artist_output_sparse_categorical_accuracy"
            if initial_weights:
                try:
                    model.set_weights(initial_weights)
                except Exception:
                    pass

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=1,
                mode="max",
                restore_best_weights=True,
            )
        ]
        effective_batch = max(1, min(int(batch_size), len(X_seq_fit)))
        model.fit(
            (X_seq_fit, X_ctx_fit),
            fit_targets,
            validation_data=((X_seq_val, X_ctx_val), val_targets),
            epochs=max(1, int(epochs)),
            batch_size=effective_batch,
            verbose=0,
            callbacks=callbacks,
        )
        fit_seconds = float(time.perf_counter() - started)
        prediction = model.predict((X_seq_test, X_ctx_test), verbose=0)
        artist_proba = _extract_artist_predictions(prediction)
        if weight_sink is not None:
            try:
                weight_sink["weights"] = model.get_weights()
            except Exception:
                pass
        top1 = float(np.mean(np.argmax(artist_proba, axis=1) == np.asarray(y_test).reshape(-1)))
        top5 = _topk_accuracy_from_proba(artist_proba, y_test, k=5)
        return BacktestFoldResult(
            model_name=model_name,
            model_type="deep",
            model_family="neural",
            adaptation_mode=adaptation_mode,
            fold=fold_idx,
            train_rows=len(X_seq_fit),
            test_rows=len(X_seq_test),
            fit_seconds=fit_seconds,
            top1=top1,
            top5=top5,
        )
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _write_backtest_csv(results: list[BacktestFoldResult], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "model_name",
                "model_type",
                "model_family",
                "adaptation_mode",
                "fold",
                "train_rows",
                "test_rows",
                "fit_seconds",
                "top1",
                "top5",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def _plot_backtest(results: list[BacktestFoldResult], output_path: Path) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    by_model: dict[str, list[BacktestFoldResult]] = {}
    for row in results:
        by_model.setdefault(row.model_name, []).append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, model_rows in sorted(by_model.items()):
        ordered = sorted(model_rows, key=lambda item: item.fold)
        x = [row.fold for row in ordered]
        y = [row.top1 for row in ordered]
        ax.plot(x, y, marker="o", label=model_name)

    ax.set_title("Temporal Backtest Top-1 Accuracy")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_xticks(sorted({row.fold for row in results}))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_temporal_backtest(
    data: PreparedData,
    output_dir: Path,
    selected_models: tuple[str, ...],
    random_seed: int,
    folds: int,
    max_train_samples: int,
    max_eval_samples: int,
    logger,
    feature_bundle: ClassicalFeatureBundle | None = None,
    deep_model_builders=None,
    strategy=None,
    adaptation_mode: str | None = None,
) -> list[BacktestFoldResult]:
    if folds <= 0:
        logger.info("Skipping temporal backtesting because folds <= 0.")
        return []

    classical_models, deep_models = _resolve_backtest_model_groups(selected_models, random_seed)
    resolved_adaptation_mode = _resolve_backtest_adaptation_mode(adaptation_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all = build_full_tabular_dataset(data, feature_bundle=feature_bundle)
    windows = _build_expanding_windows(len(X_all), folds)
    if not windows:
        logger.warning("Unable to build temporal backtest windows from %d rows.", len(X_all))
        return []

    logger.info(
        "Temporal backtesting windows=%d across %d rows",
        len(windows),
        len(X_all),
    )

    resolved_max_train_samples = _resolve_backtest_sample_cap("SPOTIFY_BACKTEST_MAX_TRAIN_SAMPLES", max_train_samples)
    resolved_max_eval_samples = _resolve_backtest_sample_cap("SPOTIFY_BACKTEST_MAX_EVAL_SAMPLES", max_eval_samples)
    logger.info(
        "Temporal backtesting caps: train_rows=%d eval_rows=%d",
        resolved_max_train_samples,
        resolved_max_eval_samples,
    )
    resolved_windows = _resolve_backtest_windows(
        windows,
        max_train_samples=resolved_max_train_samples,
        max_eval_samples=resolved_max_eval_samples,
    )
    if not resolved_windows:
        logger.warning("Temporal backtesting resolved to zero runnable folds after sampling caps.")
        return []

    needs_session_sequences = "session_knn" in classical_models
    needs_deep_inputs = bool(deep_models)
    X_all_seq = None
    if needs_session_sequences or needs_deep_inputs:
        if feature_bundle is None:
            X_all_seq = np.concatenate(
                [
                    data.X_seq_train.astype("int32", copy=False),
                    data.X_seq_val.astype("int32", copy=False),
                    data.X_seq_test.astype("int32", copy=False),
                ],
                axis=0,
            )
        else:
            X_all_seq = np.concatenate(
                [
                    np.asarray(feature_bundle.X_train_seq, dtype="int32"),
                    np.asarray(feature_bundle.X_val_seq, dtype="int32"),
                    np.asarray(feature_bundle.X_test_seq, dtype="int32"),
                ],
                axis=0,
            )
    X_all_ctx = None
    y_skip_all = None
    if needs_deep_inputs:
        X_all_ctx = np.concatenate(
            [
                data.X_ctx_train.astype("float32", copy=False),
                data.X_ctx_val.astype("float32", copy=False),
                data.X_ctx_test.astype("float32", copy=False),
            ],
            axis=0,
        )
        y_skip_all = np.concatenate(
            [
                np.asarray(data.y_skip_train),
                np.asarray(data.y_skip_val),
                np.asarray(data.y_skip_test),
            ],
            axis=0,
        )

    results: list[BacktestFoldResult] = []
    jobs: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for window in resolved_windows:
        X_train_tab = X_all[window.train_slice]
        y_train = y_all[window.train_slice]
        X_test_tab = X_all[window.test_slice]
        y_test = y_all[window.test_slice]
        for model_name in classical_models:
            if model_name == "session_knn":
                assert X_all_seq is not None
                jobs.append(
                    (
                        window.fold,
                        model_name,
                        X_all_seq[window.train_slice],
                        y_train,
                        X_all_seq[window.test_slice],
                        y_test,
                    )
                )
            else:
                jobs.append((window.fold, model_name, X_train_tab, y_train, X_test_tab, y_test))

    workers_raw = os.getenv("SPOTIFY_BACKTEST_WORKERS", "")
    _, estimator_n_jobs = resolve_classical_parallelism()
    backtest_workers = _resolve_backtest_workers(workers_raw, job_count=len(jobs))
    if backtest_workers > 1:
        estimator_n_jobs = 1
    logger.info(
        "Backtest parallelism: workers=%d estimator_n_jobs=%d jobs=%d",
        backtest_workers,
        estimator_n_jobs,
        len(jobs),
    )

    if backtest_workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=backtest_workers) as executor:
            futures = {}
            for fold_idx, model_name, X_train, y_train, X_test, y_test in jobs:
                future = executor.submit(
                    _run_backtest_job,
                    model_name,
                    fold_idx,
                    random_seed,
                    estimator_n_jobs,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
                futures[future] = (fold_idx, model_name)

            collected: dict[tuple[int, str], BacktestFoldResult] = {}
            for future in as_completed(futures):
                fold_idx, model_name = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    logger.warning("Backtest fit failed for %s fold=%d: %s", model_name, fold_idx, exc)
                    continue
                collected[(fold_idx, model_name)] = row

        for fold_idx, model_name, *_ in jobs:
            key = (fold_idx, model_name)
            if key not in collected:
                continue
            row = collected[key]
            results.append(row)
            logger.info(
                "[BACKTEST] fold=%d model=%s top1=%.4f top5=%s",
                row.fold,
                row.model_name,
                row.top1,
                f"{row.top5:.4f}" if not np.isnan(row.top5) else "n/a",
            )
    else:
        for fold_idx, model_name, X_train, y_train, X_test, y_test in jobs:
            try:
                row = _run_backtest_job(
                    model_name=model_name,
                    fold_idx=fold_idx,
                    random_seed=random_seed,
                    estimator_n_jobs=estimator_n_jobs,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
            except Exception as exc:
                logger.warning("Backtest fit failed for %s fold=%d: %s", model_name, fold_idx, exc)
                continue
            results.append(row)
            logger.info(
                "[BACKTEST] fold=%d model=%s top1=%.4f top5=%s",
                row.fold,
                row.model_name,
                row.top1,
                f"{row.top5:.4f}" if not np.isnan(row.top5) else "n/a",
            )

    if deep_models:
        deep_epochs_raw = os.getenv("SPOTIFY_DEEP_BACKTEST_EPOCHS", "3").strip()
        deep_batch_raw = os.getenv("SPOTIFY_DEEP_BACKTEST_BATCH_SIZE", "256").strip()
        try:
            deep_epochs = max(1, int(deep_epochs_raw))
        except Exception:
            deep_epochs = 3
        try:
            deep_batch_size = max(1, int(deep_batch_raw))
        except Exception:
            deep_batch_size = 256

        resolved_strategy = _resolve_deep_strategy(strategy, logger)
        deep_builders = _resolve_deep_builders(
            data=data,
            selected_models=deep_models,
            deep_model_builders=deep_model_builders,
        )
        logger.info(
            "Deep temporal backtesting: models=%s epochs=%d batch_size=%d adaptation=%s",
            ",".join(deep_models),
            deep_epochs,
            deep_batch_size,
            resolved_adaptation_mode,
        )
        prior_weights: dict[str, object] = {}
        previous_test_end = 0

        assert X_all_seq is not None
        assert X_all_ctx is not None
        assert y_skip_all is not None
        for window in resolved_windows:
            X_train_seq = X_all_seq[window.train_slice]
            X_train_ctx = X_all_ctx[window.train_slice]
            y_train = y_all[window.train_slice]
            y_skip_train = y_skip_all[window.train_slice]

            X_test_seq = X_all_seq[window.test_slice]
            X_test_ctx = X_all_ctx[window.test_slice]
            y_test = y_all[window.test_slice]

            if len(X_train_seq) <= 1 or len(X_test_seq) == 0:
                continue

            split_slices = _deep_train_validation_split(len(X_train_seq), len(X_test_seq))
            if split_slices is None:
                continue
            fit_slice, val_slice = split_slices

            for model_name in deep_models:
                incremental_fit_seq = X_train_seq[fit_slice]
                incremental_fit_ctx = X_train_ctx[fit_slice]
                incremental_y = y_train[fit_slice]
                incremental_skip = y_skip_train[fit_slice]
                if (
                    resolved_adaptation_mode == "continual"
                    and model_name in prior_weights
                    and previous_test_end < window.raw_test_start
                ):
                    inc_seq_all = X_all_seq[previous_test_end : window.raw_test_start]
                    inc_ctx_all = X_all_ctx[previous_test_end : window.raw_test_start]
                    inc_y_all = y_all[previous_test_end : window.raw_test_start]
                    inc_skip_all = y_skip_all[previous_test_end : window.raw_test_start]
                    if len(inc_seq_all) > 1:
                        incremental_fit_seq = inc_seq_all
                        incremental_fit_ctx = inc_ctx_all
                        incremental_y = inc_y_all
                        incremental_skip = inc_skip_all
                weight_sink: dict[str, object] = {}
                try:
                    row = _run_deep_backtest_job(
                        model_name=model_name,
                        fold_idx=window.fold,
                        random_seed=random_seed + window.fold,
                        model_builder=deep_builders[model_name],
                        strategy=resolved_strategy,
                        batch_size=deep_batch_size,
                        epochs=deep_epochs,
                        X_seq_fit=incremental_fit_seq,
                        X_ctx_fit=incremental_fit_ctx,
                        y_fit=incremental_y,
                        y_skip_fit=incremental_skip,
                        X_seq_val=X_train_seq[val_slice],
                        X_ctx_val=X_train_ctx[val_slice],
                        y_val=y_train[val_slice],
                        y_skip_val=y_skip_train[val_slice],
                        X_seq_test=X_test_seq,
                        X_ctx_test=X_test_ctx,
                        y_test=y_test,
                        initial_weights=(prior_weights.get(model_name) if resolved_adaptation_mode in ("warm", "continual") else None),
                        weight_sink=weight_sink,
                        adaptation_mode=resolved_adaptation_mode,
                    )
                except Exception as exc:
                    logger.warning("Deep backtest fit failed for %s fold=%d: %s", model_name, fold_idx, exc)
                    continue
                if resolved_adaptation_mode in ("warm", "continual") and "weights" in weight_sink:
                    prior_weights[model_name] = weight_sink["weights"]
                results.append(row)
                logger.info(
                    "[BACKTEST] fold=%d model=%s type=%s adaptation=%s top1=%.4f top5=%s",
                    row.fold,
                    row.model_name,
                    row.model_type,
                    row.adaptation_mode,
                    row.top1,
                    f"{row.top5:.4f}" if not np.isnan(row.top5) else "n/a",
                )
            previous_test_end = window.raw_test_end

    if results:
        runtime_by_model: dict[str, list[float]] = {}
        for row in results:
            runtime_by_model.setdefault(row.model_name, []).append(float(row.fit_seconds))
        summary_parts = []
        for model_name, timings in sorted(
            runtime_by_model.items(),
            key=lambda item: (float(np.mean(item[1])), item[0]),
            reverse=True,
        ):
            summary_parts.append(f"{model_name}={float(np.mean(timings)):.2f}s")
        logger.info("Backtest runtime summary: %s", ", ".join(summary_parts))

    write_temporal_backtest_artifacts([asdict(row) for row in results], output_dir=output_dir, metric_name="top1")

    return results
