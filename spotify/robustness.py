from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd

from .data import PreparedData
from .evaluation import _build_split_frames, _segment_bucket_frames
from .probability_bundles import load_prediction_bundle


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _topk_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if proba.ndim != 2 or len(proba) == 0:
        return float("nan")
    kk = max(1, min(int(k), int(proba.shape[1])))
    topk = np.argpartition(proba, -kk, axis=1)[:, -kk:]
    return float(np.mean(np.any(topk == np.asarray(y_true).reshape(-1, 1), axis=1)))


def _friction_bucket(frame: pd.DataFrame) -> np.ndarray:
    friction_cols = [col for col in frame.columns if str(col).startswith("tech_") or str(col) == "offline"]
    if not friction_cols:
        return np.full(len(frame), "unknown", dtype=object)
    friction_values = frame[friction_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32", copy=False)
    if friction_values.size == 0:
        return np.full(len(frame), "unknown", dtype=object)
    centered = friction_values - np.nanmedian(friction_values, axis=0, keepdims=True)
    score = np.sum(np.maximum(centered, 0.0), axis=1)
    threshold = float(np.quantile(score, 0.75)) if len(score) else 0.0
    return np.where(score >= threshold, "high_friction", "normal_friction")


def _repeat_heavy_bucket(frame: pd.DataFrame) -> np.ndarray:
    if "session_repeat_ratio_so_far" not in frame.columns:
        return np.full(len(frame), "unknown", dtype=object)
    values = pd.to_numeric(frame["session_repeat_ratio_so_far"], errors="coerce").fillna(0.0).to_numpy(dtype="float32", copy=False)
    threshold = float(np.quantile(values, 0.5)) if len(values) else 0.0
    return np.where(values >= threshold, "repeat_heavy", "repeat_light")


def _platform_bucket(frame: pd.DataFrame) -> np.ndarray:
    if "platform_code" not in frame.columns:
        return np.full(len(frame), "unknown", dtype=object)
    values = pd.to_numeric(frame["platform_code"], errors="coerce").fillna(-1.0).to_numpy(dtype="int32", copy=False)
    return np.asarray([f"platform_{value}" for value in values], dtype=object)


def _bucket_map(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    base = dict(_segment_bucket_frames(frame))
    base["friction_regime"] = _friction_bucket(frame)
    base["repeat_regime"] = _repeat_heavy_bucket(frame)
    base["platform_bucket"] = _platform_bucket(frame)
    return base


def _segment_metrics(
    *,
    model_name: str,
    split: str,
    frame: pd.DataFrame,
    proba: np.ndarray,
    y_true: np.ndarray,
) -> list[dict[str, object]]:
    frame = frame.reset_index(drop=True)
    y_arr = np.asarray(y_true).reshape(-1)
    proba_arr = np.asarray(proba, dtype="float32")
    n = min(len(frame), len(y_arr), len(proba_arr))
    if n <= 0:
        return []
    frame = frame.iloc[:n].reset_index(drop=True)
    y_arr = y_arr[:n]
    proba_arr = proba_arr[:n]
    pred = np.argmax(proba_arr, axis=1)
    conf = np.max(proba_arr, axis=1)

    rows: list[dict[str, object]] = []
    for segment, bucket_values in _bucket_map(frame).items():
        for bucket in sorted({str(item) for item in bucket_values.tolist()}):
            mask = bucket_values == bucket
            count = int(np.sum(mask))
            if count <= 0:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "segment": segment,
                    "bucket": bucket,
                    "count": count,
                    "top1": float(np.mean(pred[mask] == y_arr[mask])),
                    "top5": _topk_accuracy(proba_arr[mask], y_arr[mask], 5),
                    "mean_confidence": float(np.mean(conf[mask])),
                }
            )
    return rows


def _plot_max_gaps(summary_rows: list[dict[str, object]], output_path: Path) -> Path | None:
    if not summary_rows:
        return None
    import matplotlib.pyplot as plt

    labels = [str(row["model_name"]) for row in summary_rows]
    values = [float(row["max_top1_gap"]) for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.5))
    ax.bar(x, values, color="#b45309")
    ax.set_ylabel("Worst Segment Top-1 Gap")
    ax.set_title("Robustness Gap by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_robustness_slice_evaluation(
    *,
    data: PreparedData,
    results: list[dict[str, object]],
    sequence_length: int,
    run_dir: Path,
    logger,
) -> list[Path]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    val_frame, test_frame = _build_split_frames(data, sequence_length=sequence_length)
    slice_rows: list[dict[str, object]] = []

    for row in results:
        model_name = str(row.get("model_name", "")).strip()
        bundle_raw = str(row.get("prediction_bundle_path", "")).strip()
        if not model_name or not bundle_raw:
            continue
        bundle_path = Path(bundle_raw)
        if not bundle_path.exists():
            continue
        try:
            val_proba, test_proba = load_prediction_bundle(bundle_path)
        except Exception as exc:
            logger.warning("Robustness slices skipped for %s: %s", model_name, exc)
            continue
        slice_rows.extend(
            _segment_metrics(
                model_name=model_name,
                split="val",
                frame=val_frame,
                proba=val_proba,
                y_true=data.y_val,
            )
        )
        slice_rows.extend(
            _segment_metrics(
                model_name=model_name,
                split="test",
                frame=test_frame,
                proba=test_proba,
                y_true=data.y_test,
            )
        )

    if not slice_rows:
        return []

    summary_rows: list[dict[str, object]] = []
    for model_name in sorted({str(row["model_name"]) for row in slice_rows}):
        model_rows = [row for row in slice_rows if str(row["model_name"]) == model_name and str(row["split"]) == "test"]
        if not model_rows:
            continue
        top1_values = [float(row["top1"]) for row in model_rows]
        max_top1 = max(top1_values)
        min_top1 = min(top1_values)
        worst = min(model_rows, key=lambda row: float(row["top1"]))
        summary_rows.append(
            {
                "model_name": model_name,
                "max_top1_gap": float(max_top1 - min_top1),
                "worst_segment": str(worst["segment"]),
                "worst_bucket": str(worst["bucket"]),
                "worst_top1": float(worst["top1"]),
            }
        )
    summary_rows.sort(key=lambda row: float(row["max_top1_gap"]), reverse=True)

    csv_path = _write_csv(
        analysis_dir / "robustness_slices.csv",
        slice_rows,
        ["model_name", "split", "segment", "bucket", "count", "top1", "top5", "mean_confidence"],
    )
    summary_csv = _write_csv(
        analysis_dir / "robustness_summary.csv",
        summary_rows,
        ["model_name", "max_top1_gap", "worst_segment", "worst_bucket", "worst_top1"],
    )
    summary_json = analysis_dir / "robustness_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    artifacts: list[Path] = [csv_path, summary_csv, summary_json]
    plot_path = _plot_max_gaps(summary_rows, analysis_dir / "robustness_gap.png")
    if plot_path is not None:
        artifacts.append(plot_path)
    return artifacts
