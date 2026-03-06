from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import time

import numpy as np

from .benchmarks import (
    build_classical_estimator,
    build_full_tabular_dataset,
    evaluate_classical_estimator,
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

    results: list[BacktestFoldResult] = []
    for fold_idx, (test_start, test_end) in enumerate(windows, start=1):
        X_train, y_train = X_all[:test_start], y_all[:test_start]
        X_test, y_test = X_all[test_start:test_end], y_all[test_start:test_end]
        X_train, y_train = _tail_cap(X_train, y_train, max_train_samples)
        X_test, y_test = _tail_cap(X_test, y_test, max_eval_samples)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        for model_name in selected_models:
            family, estimator = build_classical_estimator(model_name, random_seed + fold_idx)
            started = time.perf_counter()
            try:
                estimator.fit(X_train, y_train)
            except Exception as exc:
                logger.warning("Backtest fit failed for %s fold=%d: %s", model_name, fold_idx, exc)
                continue
            fit_seconds = float(time.perf_counter() - started)

            _, _, top1, top5 = evaluate_classical_estimator(estimator, X_test, y_test, X_test, y_test)
            row = BacktestFoldResult(
                model_name=model_name,
                model_family=family,
                fold=fold_idx,
                train_rows=len(X_train),
                test_rows=len(X_test),
                fit_seconds=fit_seconds,
                top1=top1,
                top5=top5,
            )
            results.append(row)
            logger.info(
                "[BACKTEST] fold=%d model=%s top1=%.4f top5=%s",
                fold_idx,
                model_name,
                top1,
                f"{top5:.4f}" if not np.isnan(top5) else "n/a",
            )

    _write_backtest_csv(results, output_dir / "temporal_backtest.csv")
    _plot_backtest(results, output_dir / "temporal_backtest_top1.png")
    with (output_dir / "temporal_backtest.json").open("w", encoding="utf-8") as out:
        json.dump([asdict(row) for row in results], out, indent=2)

    return results
