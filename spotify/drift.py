from __future__ import annotations

from pathlib import Path
import csv
import json
import math

import numpy as np
import pandas as pd

from .data import PreparedData
from .evaluation import _segment_bucket_frames


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _build_split_frames(data: PreparedData, sequence_length: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_train = len(data.X_seq_train)
    n_val = len(data.X_seq_val)
    n_test = len(data.X_seq_test)
    n_total = n_train + n_val + n_test

    ordered = data.df.sort_values("ts").reset_index(drop=True)
    aligned = ordered.iloc[sequence_length : sequence_length + n_total].reset_index(drop=True)
    if len(aligned) < n_total:
        pad = n_total - len(aligned)
        aligned = pd.concat([aligned, pd.DataFrame(index=np.arange(pad))], ignore_index=True)

    train_end = n_train
    val_end = n_train + n_val
    train_frame = aligned.iloc[:train_end].reset_index(drop=True)
    val_frame = aligned.iloc[train_end:val_end].reset_index(drop=True)
    test_frame = aligned.iloc[val_end : val_end + n_test].reset_index(drop=True)
    return train_frame, val_frame, test_frame


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype="float64").reshape(-1)
    b_arr = np.asarray(b, dtype="float64").reshape(-1)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    var_a = float(np.var(a_arr))
    var_b = float(np.var(b_arr))
    pooled = 0.5 * (var_a + var_b)
    if pooled <= 0.0:
        return 0.0
    return float(np.sqrt(pooled))


def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype="float64").reshape(-1)
    b_arr = np.asarray(b, dtype="float64").reshape(-1)
    width = max(a_arr.size, b_arr.size)
    if width <= 0:
        return float("nan")

    pa = np.zeros(width, dtype="float64")
    pb = np.zeros(width, dtype="float64")
    if a_arr.size:
        pa[: a_arr.size] = a_arr
    if b_arr.size:
        pb[: b_arr.size] = b_arr

    sum_a = float(np.sum(pa))
    sum_b = float(np.sum(pb))
    if sum_a <= 0.0 or sum_b <= 0.0:
        return float("nan")

    pa /= sum_a
    pb /= sum_b
    m = 0.5 * (pa + pb)
    valid_a = pa > 0
    valid_b = pb > 0
    kl_am = float(np.sum(pa[valid_a] * np.log2(pa[valid_a] / m[valid_a])))
    kl_bm = float(np.sum(pb[valid_b] * np.log2(pb[valid_b] / m[valid_b])))
    return 0.5 * (kl_am + kl_bm)


def _context_feature_rows(
    *,
    feature_names: list[str],
    train_ctx: np.ndarray,
    val_ctx: np.ndarray,
    test_ctx: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, feature_name in enumerate(feature_names):
        train_values = np.asarray(train_ctx[:, idx], dtype="float64")
        val_values = np.asarray(val_ctx[:, idx], dtype="float64")
        test_values = np.asarray(test_ctx[:, idx], dtype="float64")

        pooled_val = _pooled_std(train_values, val_values)
        pooled_test = _pooled_std(train_values, test_values)
        val_mean = float(np.mean(val_values)) if val_values.size else float("nan")
        test_mean = float(np.mean(test_values)) if test_values.size else float("nan")
        train_mean = float(np.mean(train_values)) if train_values.size else float("nan")
        val_smd = (val_mean - train_mean) / pooled_val if pooled_val and not math.isnan(pooled_val) else float("nan")
        test_smd = (test_mean - train_mean) / pooled_test if pooled_test and not math.isnan(pooled_test) else float("nan")

        rows.append(
            {
                "feature": feature_name,
                "train_mean": train_mean,
                "val_mean": val_mean,
                "test_mean": test_mean,
                "val_std_mean_diff": val_smd,
                "test_std_mean_diff": test_smd,
                "max_abs_std_mean_diff": max(abs(_safe_float(val_smd)), abs(_safe_float(test_smd))),
            }
        )
    rows.sort(key=lambda row: _safe_float(row.get("max_abs_std_mean_diff")), reverse=True)
    return rows


def _segment_share_rows(
    *,
    train_frame: pd.DataFrame,
    compare_frame: pd.DataFrame,
    split: str,
) -> list[dict[str, object]]:
    train_buckets = _segment_bucket_frames(train_frame)
    compare_buckets = _segment_bucket_frames(compare_frame)
    rows: list[dict[str, object]] = []
    for segment_name, train_values in train_buckets.items():
        compare_values = compare_buckets.get(segment_name)
        if compare_values is None:
            continue

        train_total = max(1, len(train_values))
        compare_total = max(1, len(compare_values))
        train_counts = {str(value): int(count) for value, count in zip(*np.unique(train_values, return_counts=True))}
        compare_counts = {str(value): int(count) for value, count in zip(*np.unique(compare_values, return_counts=True))}
        all_buckets = sorted(set(train_counts) | set(compare_counts))

        for bucket in all_buckets:
            train_share = float(train_counts.get(bucket, 0)) / float(train_total)
            compare_share = float(compare_counts.get(bucket, 0)) / float(compare_total)
            rows.append(
                {
                    "split": split,
                    "segment": segment_name,
                    "bucket": bucket,
                    "train_share": train_share,
                    "compare_share": compare_share,
                    "abs_share_shift": abs(compare_share - train_share),
                }
            )
    rows.sort(key=lambda row: _safe_float(row.get("abs_share_shift")), reverse=True)
    return rows


def _plot_top_feature_drift(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    top_rows = rows[: min(15, len(rows))]
    labels = [str(row["feature"]) for row in top_rows]
    val_shift = [_safe_float(row.get("val_std_mean_diff")) for row in top_rows]
    test_shift = [_safe_float(row.get("test_std_mean_diff")) for row in top_rows]
    x = np.arange(len(top_rows))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(top_rows) * 0.7), 5))
    ax.bar(x - width / 2, val_shift, width=width, label="train->val")
    ax.bar(x + width / 2, test_shift, width=width, label="train->test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Standardized mean difference")
    ax.set_title("Top Context Feature Drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_top_segment_drift(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    top_rows = rows[: min(15, len(rows))]
    labels = [f"{row['split']}:{row['segment']}={row['bucket']}" for row in top_rows]
    values = [_safe_float(row.get("abs_share_shift")) for row in top_rows]
    x = np.arange(len(top_rows))

    fig, ax = plt.subplots(figsize=(max(10, len(top_rows) * 0.7), 5))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Absolute share shift")
    ax.set_title("Top Segment Drift")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_drift_diagnostics(
    *,
    data: PreparedData,
    sequence_length: int,
    output_dir: Path,
    logger,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, val_frame, test_frame = _build_split_frames(data, sequence_length)
    context_rows = _context_feature_rows(
        feature_names=list(data.context_features),
        train_ctx=np.asarray(data.X_ctx_train, dtype="float32"),
        val_ctx=np.asarray(data.X_ctx_val, dtype="float32"),
        test_ctx=np.asarray(data.X_ctx_test, dtype="float32"),
    )
    segment_rows = _segment_share_rows(train_frame=train_frame, compare_frame=val_frame, split="val")
    segment_rows.extend(_segment_share_rows(train_frame=train_frame, compare_frame=test_frame, split="test"))

    train_artist_counts = np.bincount(np.asarray(data.y_train, dtype="int64"), minlength=max(1, int(data.num_artists)))
    val_artist_counts = np.bincount(np.asarray(data.y_val, dtype="int64"), minlength=max(1, int(data.num_artists)))
    test_artist_counts = np.bincount(np.asarray(data.y_test, dtype="int64"), minlength=max(1, int(data.num_artists)))

    target_drift = {
        "train_vs_val_jsd": _js_divergence(train_artist_counts, val_artist_counts),
        "train_vs_test_jsd": _js_divergence(train_artist_counts, test_artist_counts),
    }
    top_context = context_rows[0] if context_rows else {}
    top_segment = segment_rows[0] if segment_rows else {}

    summary_payload = {
        "train_rows": int(len(data.X_seq_train)),
        "val_rows": int(len(data.X_seq_val)),
        "test_rows": int(len(data.X_seq_test)),
        "context_feature_count": int(len(data.context_features)),
        "target_drift": target_drift,
        "largest_context_shift": top_context,
        "largest_segment_shift": top_segment,
    }

    summary_path = output_dir / "data_drift_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    context_csv = output_dir / "context_feature_drift.csv"
    _write_csv(
        context_csv,
        context_rows,
        fieldnames=[
            "feature",
            "train_mean",
            "val_mean",
            "test_mean",
            "val_std_mean_diff",
            "test_std_mean_diff",
            "max_abs_std_mean_diff",
        ],
    )

    segment_csv = output_dir / "segment_drift.csv"
    _write_csv(
        segment_csv,
        segment_rows,
        fieldnames=["split", "segment", "bucket", "train_share", "compare_share", "abs_share_shift"],
    )

    context_plot = output_dir / "context_feature_drift.png"
    _plot_top_feature_drift(context_rows, context_plot)

    segment_plot = output_dir / "segment_drift.png"
    _plot_top_segment_drift(segment_rows, segment_plot)

    logger.info(
        "Saved drift diagnostics: target_jsd(train->test)=%.4f top_context=%s",
        _safe_float(target_drift.get("train_vs_test_jsd")),
        str(top_context.get("feature", "")),
    )

    return [summary_path, context_csv, segment_csv, context_plot, segment_plot]
