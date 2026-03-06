from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import csv
import json
import time

import numpy as np

from .benchmarks import (
    build_classical_estimator,
    build_full_tabular_dataset,
    evaluate_classical_estimator,
    resolve_classical_parallelism,
    validate_classical_models,
)
from .data import PreparedData


@dataclass
class BacktestFoldResult:
    model_name: str
    model_family: str
    fold: int
    train_rows: int
    test_rows: int
    fit_seconds: float
    top1: float
    top5: float


def _build_expanding_windows(n_rows: int, folds: int) -> list[tuple[int, int]]:
    if folds <= 0 or n_rows <= 0:
        return []

    base_train = max(100, n_rows // (folds + 1))
    if base_train >= n_rows:
        return []
    test_size = max(1, (n_rows - base_train) // folds)
    windows: list[tuple[int, int]] = []

    train_end = base_train
    for _ in range(folds):
        test_start = train_end
        test_end = min(n_rows, test_start + test_size)
        if test_end <= test_start:
            break
        windows.append((test_start, test_end))
        train_end = test_end
        if train_end >= n_rows:
            break

    return windows


def _tail_cap(X: np.ndarray, y: np.ndarray, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    return X[-max_rows:], y[-max_rows:]


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
    _, _, top1, top5, _, _ = evaluate_classical_estimator(estimator, X_test, y_test, X_test, y_test)
    return BacktestFoldResult(
        model_name=model_name,
        model_family=family,
        fold=fold_idx,
        train_rows=len(X_train),
        test_rows=len(X_test),
        fit_seconds=fit_seconds,
        top1=top1,
        top5=top5,
    )


def _write_backtest_csv(results: list[BacktestFoldResult], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "model_name",
                "model_family",
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
) -> list[BacktestFoldResult]:
    if folds <= 0:
        logger.info("Skipping temporal backtesting because folds <= 0.")
        return []

    validate_classical_models(selected_models, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all = build_full_tabular_dataset(data)
    windows = _build_expanding_windows(len(X_all), folds)
    if not windows:
        logger.warning("Unable to build temporal backtest windows from %d rows.", len(X_all))
        return []

    logger.info(
        "Temporal backtesting windows=%d across %d rows",
        len(windows),
        len(X_all),
    )

    workers_raw = os.getenv("SPOTIFY_BACKTEST_WORKERS", "1").strip()
    try:
        backtest_workers = max(1, int(workers_raw))
    except Exception:
        backtest_workers = 1
    cpu_count = os.cpu_count() or 1
    backtest_workers = min(backtest_workers, cpu_count)
    _, estimator_n_jobs = resolve_classical_parallelism()
    if backtest_workers > 1:
        estimator_n_jobs = 1
    logger.info(
        "Backtest parallelism: workers=%d estimator_n_jobs=%d",
        backtest_workers,
        estimator_n_jobs,
    )

    results: list[BacktestFoldResult] = []
    jobs: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, (test_start, test_end) in enumerate(windows, start=1):
        X_train, y_train = X_all[:test_start], y_all[:test_start]
        X_test, y_test = X_all[test_start:test_end], y_all[test_start:test_end]
        X_train, y_train = _tail_cap(X_train, y_train, max_train_samples)
        X_test, y_test = _tail_cap(X_test, y_test, max_eval_samples)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        for model_name in selected_models:
            jobs.append((fold_idx, model_name, X_train, y_train, X_test, y_test))

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

    _write_backtest_csv(results, output_dir / "temporal_backtest.csv")
    _plot_backtest(results, output_dir / "temporal_backtest_top1.png")
    with (output_dir / "temporal_backtest.json").open("w", encoding="utf-8") as out:
        json.dump([asdict(row) for row in results], out, indent=2)

    return results
