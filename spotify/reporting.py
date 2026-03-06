from __future__ import annotations

from pathlib import Path
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
