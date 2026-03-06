from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json
import sqlite3

import numpy as np

VAL_KEY = "val_artist_output_sparse_categorical_accuracy"
TRN_KEY = "artist_output_sparse_categorical_accuracy"


def histories_to_dict(histories: dict[str, object]) -> dict[str, dict[str, list[float]]]:
    return {name: history.history for name, history in histories.items()}


def save_histories_json(histories: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "histories.json"
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(histories_to_dict(histories), outfile, indent=2)
    return path


def plot_model_comparison(histories: dict[str, object], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model_comparison.png"

    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
        plt.plot(history.history[val_key], label=name)

    for name, history in histories.items():
        if "val_artist_output_top_5" in history.history:
            plt.plot(history.history["val_artist_output_top_5"], linestyle="--", label=f"{name} (Top-5)")
        elif "val_top_5" in history.history:
            plt.plot(history.history["val_top_5"], linestyle="--", label=f"{name} (Top-5)")

    plt.title("Validation Artist Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_learning_curves(histories: dict[str, object], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for model_name, history in histories.items():
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        trn_key = TRN_KEY if TRN_KEY in history.history else "sparse_categorical_accuracy"
        val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"

        axs[0].plot(history.history[trn_key], label="Train Artist Accuracy")
        axs[0].plot(history.history[val_key], label="Val Artist Accuracy")

        if "val_artist_output_top_5" in history.history:
            axs[0].plot(history.history["val_artist_output_top_5"], label="Val Artist Top-5", linestyle="--")
        elif "val_top_5" in history.history:
            axs[0].plot(history.history["val_top_5"], label="Val Artist Top-5", linestyle="--")

        axs[0].set_title(f"{model_name} Artist Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()

        axs[1].plot(history.history["loss"], label="Train Loss")
        axs[1].plot(history.history["val_loss"], label="Val Loss")
        axs[1].set_title(f"{model_name} Artist Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()

        fig.tight_layout()

        path = output_dir / f"{model_name}_learning_curve.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


def save_utilization_plot(cpu_usage: list[float], gpu_usage: list[float], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "utilization.png"

    plt.figure(figsize=(10, 4))
    plt.plot(cpu_usage, label="CPU utilization (%)")
    if gpu_usage:
        plt.plot(gpu_usage, label="GPU utilization (%)")
    plt.title("CPU / GPU utilization during end-to-end training")
    plt.xlabel("Seconds")
    plt.ylabel("Utilization %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def persist_to_sqlite(df, histories: dict[str, object], cpu_usage: list[float], gpu_usage: list[float], db_path: Path) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        try:
            df.to_sql("spotify_history", conn, if_exists="replace", index=False)
        except Exception:
            pass

        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS final_accuracy")
        cur.execute("DROP TABLE IF EXISTS learning_curves")
        cur.execute("DROP TABLE IF EXISTS utilization")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS final_accuracy(
                model TEXT PRIMARY KEY,
                val_top1 REAL,
                val_top5 REAL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_curves(
                model TEXT,
                epoch INTEGER,
                train_artist_acc REAL,
                val_artist_acc REAL,
                val_artist_top5 REAL,
                train_loss REAL,
                val_loss REAL,
                PRIMARY KEY (model, epoch)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS utilization(
                timestamp INTEGER,
                cpu_usage REAL,
                gpu_usage REAL
            )
            """
        )

        for model_name, history in histories.items():
            val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
            final_top1 = history.history[val_key][-1]
            final_top5 = history.history.get("val_artist_output_top_5", history.history.get("val_top_5", [np.nan]))[-1]
            cur.execute(
                "INSERT OR REPLACE INTO final_accuracy(model, val_top1, val_top5) VALUES (?, ?, ?)",
                (model_name, float(final_top1), float(final_top5) if not np.isnan(final_top5) else None),
            )

        for model_name, history in histories.items():
            trn_key = TRN_KEY if TRN_KEY in history.history else "sparse_categorical_accuracy"
            val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
            val_top5_series = history.history.get(
                "val_artist_output_top_5",
                history.history.get("val_top_5", [np.nan] * len(history.history[val_key])),
            )
            for epoch_idx in range(len(history.history[trn_key])):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO learning_curves(
                        model, epoch, train_artist_acc, val_artist_acc, val_artist_top5, train_loss, val_loss
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_name,
                        epoch_idx + 1,
                        float(history.history[trn_key][epoch_idx]),
                        float(history.history[val_key][epoch_idx]),
                        float(val_top5_series[epoch_idx]) if not np.isnan(val_top5_series[epoch_idx]) else None,
                        float(history.history["loss"][epoch_idx]),
                        float(history.history["val_loss"][epoch_idx]),
                    ),
                )

        for idx, cpu_value in enumerate(cpu_usage):
            gpu_value = gpu_usage[idx] if idx < len(gpu_usage) else None
            cur.execute(
                "INSERT INTO utilization(timestamp, cpu_usage, gpu_usage) VALUES (?, ?, ?)",
                (idx, float(cpu_value), float(gpu_value) if gpu_value is not None else None),
            )

        conn.commit()
    finally:
        conn.close()

    return db_path


def plot_run_leaderboard(results: list[dict[str, object]], output_dir: Path) -> Path | None:
    if not results:
        return None

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_leaderboard.png"

    labels = [str(row["model_name"]) for row in results]
    val_top1 = [float(row.get("val_top1", np.nan)) for row in results]
    test_top1 = [float(row.get("test_top1", np.nan)) for row in results]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 5))
    ax.bar(x - width / 2, val_top1, width=width, label="Val Top-1")
    ax.bar(x + width / 2, test_top1, width=width, label="Test Top-1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Run Leaderboard")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def append_experiment_history(
    history_csv: Path,
    run_id: str,
    profile: str,
    run_name: str | None,
    results: list[dict[str, object]],
    data_records: int,
) -> Path:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_csv.exists()

    fieldnames = [
        "timestamp",
        "run_id",
        "run_name",
        "profile",
        "model_name",
        "model_type",
        "model_family",
        "val_top1",
        "val_top5",
        "test_top1",
        "test_top5",
        "fit_seconds",
        "epochs",
        "data_records",
    ]

    with history_csv.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().isoformat(timespec="seconds")
        for row in results:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "run_name": run_name or "",
                    "profile": profile,
                    "model_name": row.get("model_name", ""),
                    "model_type": row.get("model_type", ""),
                    "model_family": row.get("model_family", ""),
                    "val_top1": row.get("val_top1", ""),
                    "val_top5": row.get("val_top5", ""),
                    "test_top1": row.get("test_top1", ""),
                    "test_top5": row.get("test_top5", ""),
                    "fit_seconds": row.get("fit_seconds", ""),
                    "epochs": row.get("epochs", ""),
                    "data_records": data_records,
                }
            )
    return history_csv


def plot_history_best_runs(history_csv: Path, output_dir: Path) -> Path | None:
    if not history_csv.exists():
        return None

    rows: list[dict[str, str]] = []
    with history_csv.open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows.extend(reader)

    if not rows:
        return None

    best_by_run: dict[str, tuple[str, float]] = {}
    ordered_runs: list[str] = []
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        if run_id not in best_by_run:
            ordered_runs.append(run_id)
            best_by_run[run_id] = ("", float("-inf"))
        try:
            score = float(row.get("val_top1", "nan"))
        except ValueError:
            score = float("nan")
        if np.isnan(score):
            continue
        model_name = row.get("model_name", "")
        if score > best_by_run[run_id][1]:
            best_by_run[run_id] = (model_name, score)

    plot_runs = [rid for rid in ordered_runs if best_by_run.get(rid, ("", float("-inf")))[1] != float("-inf")]
    if not plot_runs:
        return None

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "history_best_runs.png"

    y = [best_by_run[rid][1] for rid in plot_runs]
    labels = [f"{rid}\n{best_by_run[rid][0]}" for rid in plot_runs]
    x = np.arange(len(plot_runs))

    fig, ax = plt.subplots(figsize=(max(10, len(plot_runs) * 0.7), 5))
    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best Val Top-1 Accuracy")
    ax.set_title("Best Model Per Run")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def append_optuna_history(
    history_csv: Path,
    run_id: str,
    profile: str,
    run_name: str | None,
    results: list[dict[str, object]],
) -> Path:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_csv.exists()

    fieldnames = [
        "timestamp",
        "run_id",
        "run_name",
        "profile",
        "model_name",
        "base_model_name",
        "n_trials",
        "val_top1",
        "test_top1",
        "fit_seconds",
        "best_params_json",
    ]

    with history_csv.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        timestamp = datetime.now().isoformat(timespec="seconds")
        for row in results:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "run_name": run_name or "",
                    "profile": profile,
                    "model_name": row.get("model_name", ""),
                    "base_model_name": row.get("base_model_name", ""),
                    "n_trials": row.get("n_trials", ""),
                    "val_top1": row.get("val_top1", ""),
                    "test_top1": row.get("test_top1", ""),
                    "fit_seconds": row.get("fit_seconds", ""),
                    "best_params_json": json.dumps(row.get("best_params", {}), sort_keys=True),
                }
            )
    return history_csv


def plot_optuna_best_runs(history_csv: Path, output_dir: Path) -> Path | None:
    if not history_csv.exists():
        return None

    rows: list[dict[str, str]] = []
    with history_csv.open("r", encoding="utf-8") as infile:
        rows.extend(csv.DictReader(infile))
    if not rows:
        return None

    best_by_run: dict[str, tuple[str, float]] = {}
    run_order: list[str] = []
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        if run_id not in best_by_run:
            run_order.append(run_id)
            best_by_run[run_id] = ("", float("-inf"))
        try:
            val_top1 = float(row.get("val_top1", "nan"))
        except ValueError:
            continue
        if np.isnan(val_top1):
            continue
        model_name = row.get("model_name", "")
        if val_top1 > best_by_run[run_id][1]:
            best_by_run[run_id] = (model_name, val_top1)

    plot_runs = [rid for rid in run_order if best_by_run[rid][1] != float("-inf")]
    if not plot_runs:
        return None

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "history_optuna_best_runs.png"
    x = np.arange(len(plot_runs))
    y = [best_by_run[rid][1] for rid in plot_runs]
    labels = [f"{rid}\n{best_by_run[rid][0]}" for rid in plot_runs]

    fig, ax = plt.subplots(figsize=(max(10, len(plot_runs) * 0.7), 5))
    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best Tuned Val Top-1")
    ax.set_title("Best Optuna-Tuned Model Per Run")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def append_backtest_history(
    history_csv: Path,
    run_id: str,
    profile: str,
    run_name: str | None,
    rows: list[dict[str, object]],
) -> Path:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_csv.exists()

    fieldnames = [
        "timestamp",
        "run_id",
        "run_name",
        "profile",
        "model_name",
        "model_family",
        "fold",
        "train_rows",
        "test_rows",
        "fit_seconds",
        "top1",
        "top5",
    ]

    with history_csv.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        timestamp = datetime.now().isoformat(timespec="seconds")
        for row in rows:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "run_name": run_name or "",
                    "profile": profile,
                    "model_name": row.get("model_name", ""),
                    "model_family": row.get("model_family", ""),
                    "fold": row.get("fold", ""),
                    "train_rows": row.get("train_rows", ""),
                    "test_rows": row.get("test_rows", ""),
                    "fit_seconds": row.get("fit_seconds", ""),
                    "top1": row.get("top1", ""),
                    "top5": row.get("top5", ""),
                }
            )
    return history_csv


def plot_backtest_history(history_csv: Path, output_dir: Path) -> Path | None:
    if not history_csv.exists():
        return None

    rows: list[dict[str, str]] = []
    with history_csv.open("r", encoding="utf-8") as infile:
        rows.extend(csv.DictReader(infile))
    if not rows:
        return None

    run_order: list[str] = []
    model_scores: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        model_name = row.get("model_name", "")
        if not run_id or not model_name:
            continue
        if run_id not in run_order:
            run_order.append(run_id)
        try:
            score = float(row.get("top1", "nan"))
        except ValueError:
            continue
        if np.isnan(score):
            continue
        model_scores.setdefault((run_id, model_name), []).append(score)

    if not model_scores:
        return None

    # Keep chart readable by plotting top models from the latest run only.
    latest_run = run_order[-1]
    model_names = sorted({model for run, model in model_scores if run == latest_run})
    if not model_names:
        return None

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "history_backtest_mean_top1.png"
    x = np.arange(len(run_order))

    fig, ax = plt.subplots(figsize=(max(10, len(run_order) * 0.7), 5))
    for model_name in model_names:
        y: list[float] = []
        for run_id in run_order:
            scores = model_scores.get((run_id, model_name), [])
            y.append(float(np.mean(scores)) if scores else np.nan)
        ax.plot(x, y, marker="o", label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(run_order, rotation=45, ha="right")
    ax.set_ylabel("Mean Backtest Top-1")
    ax.set_title("Temporal Backtest Trends")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
