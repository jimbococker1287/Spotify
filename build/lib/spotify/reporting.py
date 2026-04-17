from __future__ import annotations

from itertools import zip_longest
from datetime import datetime
import hashlib
from pathlib import Path
import csv
import json
import math
import sqlite3

import numpy as np

from .run_artifacts import copy_file_if_changed

VAL_KEY = "val_artist_output_sparse_categorical_accuracy"
TRN_KEY = "artist_output_sparse_categorical_accuracy"


def histories_to_dict(histories: dict[str, object]) -> dict[str, dict[str, list[float]]]:
    return {name: history.history for name, history in histories.items()}


def _metric_keys(history_values: dict[str, list[float]]) -> tuple[str, str]:
    trn_key = TRN_KEY if TRN_KEY in history_values else "sparse_categorical_accuracy"
    val_key = VAL_KEY if VAL_KEY in history_values else "val_sparse_categorical_accuracy"
    return trn_key, val_key


def _iter_final_accuracy_rows(histories: dict[str, object]):
    for model_name, history in histories.items():
        history_values = history.history
        _trn_key, val_key = _metric_keys(history_values)
        final_top1 = history_values[val_key][-1]
        final_top5 = history_values.get("val_artist_output_top_5", history_values.get("val_top_5", [np.nan]))[-1]
        yield (
            model_name,
            float(final_top1),
            float(final_top5) if not np.isnan(final_top5) else None,
        )


def _iter_learning_curve_rows(histories: dict[str, object]):
    for model_name, history in histories.items():
        history_values = history.history
        trn_key, val_key = _metric_keys(history_values)
        train_artist_acc = history_values[trn_key]
        val_artist_acc = history_values[val_key]
        val_top5_series = history_values.get(
            "val_artist_output_top_5",
            history_values.get("val_top_5", [np.nan] * len(val_artist_acc)),
        )
        train_loss = history_values["loss"]
        val_loss = history_values["val_loss"]
        for epoch_idx, (train_acc, val_acc, val_top5, train_loss_value, val_loss_value) in enumerate(
            zip(train_artist_acc, val_artist_acc, val_top5_series, train_loss, val_loss),
            start=1,
        ):
            yield (
                model_name,
                epoch_idx,
                float(train_acc),
                float(val_acc),
                float(val_top5) if not np.isnan(val_top5) else None,
                float(train_loss_value),
                float(val_loss_value),
            )


def _iter_utilization_rows(cpu_usage: list[float], gpu_usage: list[float]):
    for idx, (cpu_value, gpu_value) in enumerate(zip_longest(cpu_usage, gpu_usage, fillvalue=None)):
        if cpu_value is None:
            continue
        yield (
            idx,
            float(cpu_value),
            float(gpu_value) if gpu_value is not None else None,
        )


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
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=MEMORY")
            cur.execute("PRAGMA synchronous=OFF")
            cur.execute("PRAGMA temp_store=MEMORY")
            cur.execute("PRAGMA cache_size=-65536")
        except Exception:
            pass

        try:
            df.to_sql("spotify_history", conn, if_exists="replace", index=False)
        except Exception as exc:
            raise RuntimeError(f"Unable to persist spotify_history to SQLite at {db_path}") from exc

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

        cur.executemany(
            "INSERT OR REPLACE INTO final_accuracy(model, val_top1, val_top5) VALUES (?, ?, ?)",
            _iter_final_accuracy_rows(histories),
        )
        cur.executemany(
            """
            INSERT OR REPLACE INTO learning_curves(
                model, epoch, train_artist_acc, val_artist_acc, val_artist_top5, train_loss, val_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            _iter_learning_curve_rows(histories),
        )
        cur.executemany(
            "INSERT INTO utilization(timestamp, cpu_usage, gpu_usage) VALUES (?, ?, ?)",
            _iter_utilization_rows(cpu_usage, gpu_usage),
        )

        conn.commit()
    finally:
        conn.close()

    return db_path


def _deep_reporting_cache_enabled_from_env() -> bool:
    import os

    raw = os.getenv("SPOTIFY_CACHE_DEEP_REPORTING", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _reporting_source_digest() -> str:
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:24]


def _build_deep_reporting_cache_key(
    *,
    cache_fingerprint: str,
    histories: dict[str, object],
    cpu_usage: list[float],
    gpu_usage: list[float],
) -> str:
    payload = {
        "prepared_fingerprint": str(cache_fingerprint).strip(),
        "histories": histories_to_dict(histories),
        "cpu_usage": [float(value) for value in cpu_usage],
        "gpu_usage": [float(value) for value in gpu_usage],
        "source_digest": _reporting_source_digest(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _deep_reporting_cache_artifact_map(
    *,
    root_dir: Path,
    histories: dict[str, object],
    db_name: str,
) -> dict[str, Path]:
    paths = {
        "model_comparison": root_dir / "model_comparison.png",
        "histories_json": root_dir / "histories.json",
        "utilization": root_dir / "utilization.png",
        "sqlite": root_dir / db_name,
    }
    for model_name in histories:
        paths[f"learning_curve:{model_name}"] = root_dir / f"{model_name}_learning_curve.png"
    return paths


def restore_deep_reporting_artifacts(
    *,
    histories: dict[str, object],
    cpu_usage: list[float],
    gpu_usage: list[float],
    output_dir: Path,
    db_path: Path,
    cache_root: Path | None,
    cache_fingerprint: str,
) -> tuple[Path, list[Path], Path, Path, Path] | None:
    if not histories:
        return None
    if cache_root is None or not _deep_reporting_cache_enabled_from_env() or not str(cache_fingerprint).strip():
        return None

    cache_key = _build_deep_reporting_cache_key(
        cache_fingerprint=cache_fingerprint,
        histories=histories,
        cpu_usage=cpu_usage,
        gpu_usage=gpu_usage,
    )
    cache_dir = (cache_root / str(cache_fingerprint).strip() / cache_key).resolve()
    cached_paths = _deep_reporting_cache_artifact_map(root_dir=cache_dir, histories=histories, db_name=db_path.name)
    output_paths = _deep_reporting_cache_artifact_map(root_dir=output_dir, histories=histories, db_name=db_path.name)

    if any(not path.exists() for path in cached_paths.values()):
        return None

    for key, source_path in cached_paths.items():
        copy_file_if_changed(source_path, output_paths[key])

    learning_paths = [output_paths[f"learning_curve:{model_name}"] for model_name in histories]
    return (
        output_paths["model_comparison"],
        learning_paths,
        output_paths["histories_json"],
        output_paths["utilization"],
        output_paths["sqlite"],
    )


def save_deep_reporting_artifacts(
    *,
    histories: dict[str, object],
    cpu_usage: list[float],
    gpu_usage: list[float],
    output_dir: Path,
    db_path: Path,
    cache_root: Path | None,
    cache_fingerprint: str,
) -> None:
    if not histories:
        return
    if cache_root is None or not _deep_reporting_cache_enabled_from_env() or not str(cache_fingerprint).strip():
        return

    cache_key = _build_deep_reporting_cache_key(
        cache_fingerprint=cache_fingerprint,
        histories=histories,
        cpu_usage=cpu_usage,
        gpu_usage=gpu_usage,
    )
    cache_dir = (cache_root / str(cache_fingerprint).strip() / cache_key).resolve()
    cached_paths = _deep_reporting_cache_artifact_map(root_dir=cache_dir, histories=histories, db_name=db_path.name)
    output_paths = _deep_reporting_cache_artifact_map(root_dir=output_dir, histories=histories, db_name=db_path.name)

    for key, destination_path in cached_paths.items():
        source_path = output_paths[key]
        if source_path.exists():
            copy_file_if_changed(source_path, destination_path)


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
        "val_ndcg_at5",
        "val_mrr_at5",
        "val_coverage_at5",
        "val_diversity_at5",
        "test_top1",
        "test_top5",
        "test_ndcg_at5",
        "test_mrr_at5",
        "test_coverage_at5",
        "test_diversity_at5",
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
                    "val_ndcg_at5": row.get("val_ndcg_at5", ""),
                    "val_mrr_at5": row.get("val_mrr_at5", ""),
                    "val_coverage_at5": row.get("val_coverage_at5", ""),
                    "val_diversity_at5": row.get("val_diversity_at5", ""),
                    "test_top1": row.get("test_top1", ""),
                    "test_top5": row.get("test_top5", ""),
                    "test_ndcg_at5": row.get("test_ndcg_at5", ""),
                    "test_mrr_at5": row.get("test_mrr_at5", ""),
                    "test_coverage_at5": row.get("test_coverage_at5", ""),
                    "test_diversity_at5": row.get("test_diversity_at5", ""),
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
        "model_type",
        "model_family",
        "adaptation_mode",
        "fold",
        "train_rows",
        "test_rows",
        "fit_seconds",
        "top1",
        "top5",
    ]
    active_fieldnames = list(fieldnames)
    if file_exists:
        try:
            with history_csv.open("r", encoding="utf-8", newline="") as infile:
                reader = csv.reader(infile)
                existing_header = next(reader, [])
            if existing_header and "model_type" not in existing_header:
                active_fieldnames = [name for name in fieldnames if name != "model_type"]
            if existing_header and "adaptation_mode" not in existing_header:
                active_fieldnames = [name for name in active_fieldnames if name != "adaptation_mode"]
        except Exception:
            active_fieldnames = list(fieldnames)

    with history_csv.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=active_fieldnames)
        if not file_exists:
            writer.writeheader()
        timestamp = datetime.now().isoformat(timespec="seconds")
        for row in rows:
            payload = {
                "timestamp": timestamp,
                "run_id": run_id,
                "run_name": run_name or "",
                "profile": profile,
                "model_name": row.get("model_name", ""),
                "model_type": row.get("model_type", ""),
                "model_family": row.get("model_family", ""),
                "adaptation_mode": row.get("adaptation_mode", ""),
                "fold": row.get("fold", ""),
                "train_rows": row.get("train_rows", ""),
                "test_rows": row.get("test_rows", ""),
                "fit_seconds": row.get("fit_seconds", ""),
                "top1": row.get("top1", ""),
                "top5": row.get("top5", ""),
            }
            writer.writerow({key: payload.get(key, "") for key in active_fieldnames})
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


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def write_run_report(
    *,
    run_dir: Path,
    history_dir: Path,
    manifest: dict[str, object],
    results: list[dict[str, object]],
    champion_gate: dict[str, object],
    history_csv: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "run_report.md"

    sorted_results = sorted(
        results,
        key=lambda row: _safe_float(row.get("val_top1")),
        reverse=True,
    )
    best = sorted_results[0] if sorted_results else {}
    fit_values = [_safe_float(row.get("fit_seconds")) for row in results]
    fit_values = [value for value in fit_values if not math.isnan(value)]
    total_fit_seconds = float(np.sum(fit_values)) if fit_values else float("nan")

    recent_best_rows: list[tuple[str, str, float]] = []
    if history_csv.exists():
        seen_runs: dict[str, tuple[str, float]] = {}
        run_order: list[str] = []
        with history_csv.open("r", encoding="utf-8") as infile:
            for row in csv.DictReader(infile):
                run_id = str(row.get("run_id", "")).strip()
                if not run_id:
                    continue
                if run_id not in seen_runs:
                    run_order.append(run_id)
                    seen_runs[run_id] = ("", float("-inf"))
                score = _safe_float(row.get("val_top1"))
                if math.isnan(score):
                    continue
                model_name = str(row.get("model_name", "")).strip()
                if score > seen_runs[run_id][1]:
                    seen_runs[run_id] = (model_name, score)
        for run_id in run_order[-10:]:
            model_name, score = seen_runs[run_id]
            if score == float("-inf"):
                continue
            recent_best_rows.append((run_id, model_name, score))

    lines: list[str] = []
    lines.append("# Spotify Run Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Run ID: `{manifest.get('run_id', '')}`")
    lines.append(f"- Run Name: `{manifest.get('run_name', '')}`")
    lines.append(f"- Profile: `{manifest.get('profile', '')}`")
    lines.append(f"- Data records: `{manifest.get('data_records', '')}`")
    lines.append(f"- Best model: `{best.get('model_name', '')}` ({best.get('model_type', '')})")
    lines.append(f"- Best val Top-1: `{_safe_float(best.get('val_top1')):.4f}`")
    if fit_values:
        lines.append(f"- Aggregate fit time (sum): `{total_fit_seconds:.2f}s`")
    lines.append("")
    lines.append("## Champion Gate")
    metric_source = str(champion_gate.get("metric_source", "val_top1"))
    champion_score = _safe_float(champion_gate.get("champion_score"))
    challenger_score = _safe_float(champion_gate.get("challenger_score"))
    if math.isnan(champion_score):
        if metric_source == "backtest_top1":
            champion_score = _safe_float(champion_gate.get("champion_backtest_top1"))
        else:
            champion_score = _safe_float(champion_gate.get("champion_val_top1"))
    if math.isnan(challenger_score):
        if metric_source == "backtest_top1":
            challenger_score = _safe_float(champion_gate.get("challenger_backtest_top1"))
        else:
            challenger_score = _safe_float(champion_gate.get("challenger_val_top1"))

    lines.append(f"- Status: `{champion_gate.get('status', '')}`")
    lines.append(f"- Promoted: `{champion_gate.get('promoted', False)}`")
    lines.append(f"- Metric source: `{metric_source}`")
    lines.append(f"- Threshold: `{_safe_float(champion_gate.get('threshold')):.6f}`")
    lines.append(f"- Regression: `{_safe_float(champion_gate.get('regression')):.6f}`")
    lines.append(
        f"- Previous champion: `{champion_gate.get('champion_run_id', '')}` / "
        f"`{champion_gate.get('champion_model_name', '')}`"
    )
    lines.append(f"- Champion score: `{champion_score:.4f}`")
    lines.append(
        f"- Current challenger: `{champion_gate.get('challenger_model_name', '')}` "
        f"(`{challenger_score:.4f}`)"
    )
    lines.append("")
    lines.append("## Model Results")
    lines.append(
        "| model | type | val_top1 | val_top5 | val_ndcg@5 | val_mrr@5 | test_top1 | test_top5 | test_ndcg@5 | test_mrr@5 | fit_s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted_results:
        lines.append(
            "| "
            + f"{row.get('model_name', '')} | {row.get('model_type', '')}"
            + f" | {_safe_float(row.get('val_top1')):.4f}"
            + f" | {_safe_float(row.get('val_top5')):.4f}"
            + f" | {_safe_float(row.get('val_ndcg_at5')):.4f}"
            + f" | {_safe_float(row.get('val_mrr_at5')):.4f}"
            + f" | {_safe_float(row.get('test_top1')):.4f}"
            + f" | {_safe_float(row.get('test_top5')):.4f}"
            + f" | {_safe_float(row.get('test_ndcg_at5')):.4f}"
            + f" | {_safe_float(row.get('test_mrr_at5')):.4f}"
            + f" | {_safe_float(row.get('fit_seconds')):.2f} |"
        )
    lines.append("")

    if recent_best_rows:
        lines.append("## Historical Trend (Recent Best Runs)")
        lines.append("| run_id | best_model | best_val_top1 |")
        lines.append("|---|---|---:|")
        for run_id, model_name, score in recent_best_rows:
            lines.append(f"| {run_id} | {model_name} | {score:.4f} |")
        lines.append("")

    lines.append("## Key Artifacts")
    for rel in (
        "run_leaderboard.png",
        "model_comparison.png",
        "utilization.png",
        "benchmark_protocol.json",
        "benchmark_protocol.md",
        "experiment_registry.json",
        "optuna/optuna_results.json",
        "backtest/temporal_backtest.csv",
        "../history/history_best_runs.png",
        "../history/history_optuna_best_runs.png",
        "../history/history_backtest_mean_top1.png",
    ):
        path = (run_dir / rel).resolve()
        if path.exists():
            lines.append(f"- [{rel}]({rel})")
    analysis_dir = run_dir / "analysis"
    if analysis_dir.exists():
        seen_analysis_paths: set[Path] = set()
        for pattern in (
            "ensemble_*_summary.json",
            "*_confidence_summary.json",
            "*_conformal_summary.json",
            "*drift*.json",
            "*drift*.csv",
            "*drift*.png",
            "*robustness*.json",
            "*robustness*.csv",
            "*robustness*.png",
            "*policy_simulation*.json",
            "*policy_simulation*.csv",
            "*ablation*.json",
            "*ablation*.csv",
            "*significance*.json",
            "*significance*.csv",
            "moonshot_summary.json",
            "*multimodal*.json",
            "*multimodal*.csv",
            "*causal_skip*.json",
            "*causal_skip*.csv",
            "*digital_twin*.json",
            "*digital_twin*.csv",
            "*journey*.json",
            "*journey*.csv",
            "*safe_bandit*.json",
            "*safe_bandit*.csv",
            "*group_auto_dj*.json",
            "*group_auto_dj*.csv",
            "*stress_test*.json",
            "*stress_test*.csv",
            "*_reliability.png",
            "*_segment_metrics.csv",
            "*_top_errors.csv",
        ):
            for path in sorted(analysis_dir.rglob(pattern)):
                resolved = path.resolve()
                if not path.is_file() or resolved in seen_analysis_paths:
                    continue
                seen_analysis_paths.add(resolved)
                rel = path.relative_to(run_dir).as_posix()
                lines.append(f"- [{rel}]({rel})")
    lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path
