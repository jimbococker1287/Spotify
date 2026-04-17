from __future__ import annotations

from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd

from .data import PreparedData
from .evaluation import _build_split_frames, _segment_bucket_frames
from .probability_bundles import load_prediction_bundle
from .run_artifacts import write_csv_rows

DEFAULT_GUARDRAIL_SEGMENT = "repeat_from_prev"
DEFAULT_GUARDRAIL_BUCKET = "new"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


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


def _support_floor(n_rows: int) -> int:
    floor_raw = os.getenv("SPOTIFY_ROBUSTNESS_MIN_BUCKET_COUNT", "").strip()
    if floor_raw:
        try:
            return max(1, int(floor_raw))
        except ValueError:
            pass
    return max(25, int(math.ceil(max(1, n_rows) * 0.005)))


def _safe_metric(value) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(metric):
        return float("nan")
    return metric


def _resolve_guardrail_target() -> tuple[str, str]:
    segment = os.getenv("SPOTIFY_ROBUSTNESS_GUARDRAIL_SEGMENT", DEFAULT_GUARDRAIL_SEGMENT).strip()
    bucket = os.getenv("SPOTIFY_ROBUSTNESS_GUARDRAIL_BUCKET", DEFAULT_GUARDRAIL_BUCKET).strip()
    return (
        segment or DEFAULT_GUARDRAIL_SEGMENT,
        bucket or DEFAULT_GUARDRAIL_BUCKET,
    )


def _find_slice_row(
    rows: list[dict[str, object]],
    *,
    segment: str,
    bucket: str,
) -> dict[str, object]:
    return next(
        (
            row
            for row in rows
            if str(row.get("segment", "")).strip() == segment
            and str(row.get("bucket", "")).strip() == bucket
        ),
        {},
    )


def _build_guardrail_payload(
    summary_rows: list[dict[str, object]],
    *,
    segment: str,
    bucket: str,
) -> dict[str, object]:
    per_model: list[dict[str, object]] = []
    for row in summary_rows:
        gap = _safe_metric(row.get("guardrail_gap"))
        top1 = _safe_metric(row.get("guardrail_top1"))
        global_top1 = _safe_metric(row.get("global_top1"))
        model_type = str(row.get("model_type", "")).strip().lower()
        per_model.append(
            {
                "model_name": str(row.get("model_name", "")),
                "model_type": model_type,
                "operational_model": bool(row.get("operational_model", False)),
                "segment": segment,
                "bucket": bucket,
                "slice_top1": top1,
                "slice_gap": gap,
                "slice_count": int(row.get("guardrail_bucket_count", 0) or 0),
                "global_top1": global_top1,
            }
        )
    per_model.sort(
        key=lambda row: (
            _safe_metric(row.get("slice_gap")),
            -_safe_metric(row.get("slice_top1")),
        ),
        reverse=True,
    )
    available_rows = [row for row in per_model if math.isfinite(_safe_metric(row.get("slice_gap")))]
    worst_row = available_rows[0] if available_rows else {}
    operational_rows = [
        row
        for row in available_rows
        if bool(row.get("operational_model", False))
    ]
    operational_worst_row = operational_rows[0] if operational_rows else {}
    return {
        "segment": segment,
        "bucket": bucket,
        "model_count": int(len(per_model)),
        "available_model_count": int(len(available_rows)),
        "worst_model_name": str(worst_row.get("model_name", "")),
        "worst_gap": _safe_metric(worst_row.get("slice_gap")),
        "worst_top1": _safe_metric(worst_row.get("slice_top1")),
        "worst_bucket_count": int(worst_row.get("slice_count", 0) or 0),
        "operational_model_count": int(len(operational_rows)),
        "operational_worst_model_name": str(operational_worst_row.get("model_name", "")),
        "operational_worst_model_type": str(operational_worst_row.get("model_type", "")),
        "operational_worst_gap": _safe_metric(operational_worst_row.get("slice_gap")),
        "operational_worst_top1": _safe_metric(operational_worst_row.get("slice_top1")),
        "operational_worst_bucket_count": int(operational_worst_row.get("slice_count", 0) or 0),
        "models": per_model,
    }


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
    test_global_top1: dict[str, float] = {}
    test_row_counts: dict[str, int] = {}
    model_type_by_name = {
        str(row.get("model_name", "")).strip(): str(row.get("model_type", "")).strip().lower()
        for row in results
        if str(row.get("model_name", "")).strip()
    }
    guardrail_segment, guardrail_bucket = _resolve_guardrail_target()

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
        test_pred = np.argmax(np.asarray(test_proba, dtype="float32"), axis=1)
        test_truth = np.asarray(data.y_test).reshape(-1)
        aligned_n = min(len(test_pred), len(test_truth))
        if aligned_n > 0:
            test_global_top1[model_name] = float(np.mean(test_pred[:aligned_n] == test_truth[:aligned_n]))
            test_row_counts[model_name] = int(aligned_n)
        val_metrics = _segment_metrics(
            model_name=model_name,
            split="val",
            frame=val_frame,
            proba=val_proba,
            y_true=data.y_val,
        )
        test_metrics = _segment_metrics(
            model_name=model_name,
            split="test",
            frame=test_frame,
            proba=test_proba,
            y_true=data.y_test,
        )
        slice_rows.extend(val_metrics)
        slice_rows.extend(test_metrics)

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
        total_rows = int(test_row_counts.get(model_name, 0))
        support_floor = _support_floor(total_rows)
        supported_rows = [row for row in model_rows if int(row.get("count", 0) or 0) >= support_floor]
        actionable_rows = supported_rows or model_rows
        worst = min(actionable_rows, key=lambda row: float(row["top1"]))
        global_top1 = float(test_global_top1.get(model_name, float("nan")))
        actionable_gap = (
            float(global_top1 - float(worst["top1"]))
            if math.isfinite(global_top1)
            else float(max_top1 - float(worst["top1"]))
        )
        guardrail_row = _find_slice_row(model_rows, segment=guardrail_segment, bucket=guardrail_bucket)
        guardrail_top1 = _safe_metric(guardrail_row.get("top1"))
        guardrail_gap = (
            float(max(global_top1 - guardrail_top1, 0.0))
            if math.isfinite(global_top1) and math.isfinite(guardrail_top1)
            else float("nan")
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "model_type": model_type_by_name.get(model_name, ""),
                "operational_model": model_type_by_name.get(model_name, "") in {
                    "classical",
                    "classical_tuned",
                    "retrieval",
                    "retrieval_reranker",
                    "ensemble",
                },
                "max_top1_gap": float(max(actionable_gap, 0.0)),
                "raw_max_top1_gap": float(max_top1 - min_top1),
                "global_top1": global_top1,
                "support_floor_count": support_floor,
                "worst_segment": str(worst["segment"]),
                "worst_bucket": str(worst["bucket"]),
                "worst_top1": float(worst["top1"]),
                "worst_bucket_count": int(worst.get("count", 0) or 0),
                "worst_bucket_share": (
                    float((int(worst.get("count", 0) or 0)) / total_rows)
                    if total_rows > 0
                    else float("nan")
                ),
                "guardrail_segment": guardrail_segment,
                "guardrail_bucket": guardrail_bucket,
                "guardrail_top1": guardrail_top1,
                "guardrail_gap": guardrail_gap,
                "guardrail_bucket_count": int(guardrail_row.get("count", 0) or 0),
            }
        )
    summary_rows.sort(key=lambda row: float(row["max_top1_gap"]), reverse=True)
    guardrail_payload = _build_guardrail_payload(
        summary_rows,
        segment=guardrail_segment,
        bucket=guardrail_bucket,
    )

    csv_path = _write_csv(
        analysis_dir / "robustness_slices.csv",
        slice_rows,
        ["model_name", "split", "segment", "bucket", "count", "top1", "top5", "mean_confidence"],
    )
    summary_csv = _write_csv(
        analysis_dir / "robustness_summary.csv",
        summary_rows,
        [
            "model_name",
            "model_type",
            "operational_model",
            "max_top1_gap",
            "raw_max_top1_gap",
            "global_top1",
            "support_floor_count",
            "worst_segment",
            "worst_bucket",
            "worst_top1",
            "worst_bucket_count",
            "worst_bucket_share",
            "guardrail_segment",
            "guardrail_bucket",
            "guardrail_top1",
            "guardrail_gap",
            "guardrail_bucket_count",
        ],
    )
    summary_json = analysis_dir / "robustness_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    guardrail_csv = _write_csv(
        analysis_dir / "robustness_guardrails.csv",
        list(guardrail_payload.get("models", [])),
        [
            "model_name",
            "model_type",
            "operational_model",
            "segment",
            "bucket",
            "slice_top1",
            "slice_gap",
            "slice_count",
            "global_top1",
        ],
    )
    guardrail_json = analysis_dir / "robustness_guardrails.json"
    guardrail_json.write_text(json.dumps(guardrail_payload, indent=2), encoding="utf-8")
    artifacts: list[Path] = [csv_path, summary_csv, summary_json, guardrail_csv, guardrail_json]
    plot_path = _plot_max_gaps(summary_rows, analysis_dir / "robustness_gap.png")
    if plot_path is not None:
        artifacts.append(plot_path)
    return artifacts
