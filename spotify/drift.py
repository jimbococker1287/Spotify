from __future__ import annotations

from pathlib import Path
import csv
import json
import math

import numpy as np
import pandas as pd

from .data import PreparedData
from .evaluation import _segment_bucket_frames
from .recommender_safety import (
    SequenceSplitSnapshot,
    compute_context_feature_drift_rows,
    compute_segment_share_shift_rows,
    compute_target_distribution_drift,
)


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


def _plot_top_feature_drift(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    top_rows = rows[: min(15, len(rows))]
    labels = [f"{row['compare_split']}:{row['feature']}" for row in top_rows]
    values = [_safe_float(row.get("std_mean_diff")) for row in top_rows]
    x = np.arange(len(top_rows))

    fig, ax = plt.subplots(figsize=(max(10, len(top_rows) * 0.7), 5))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Standardized mean difference")
    ax.set_title("Top Context Feature Drift")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_top_segment_drift(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    top_rows = rows[: min(15, len(rows))]
    labels = [f"{row['compare_split']}:{row['segment']}={row['bucket']}" for row in top_rows]
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
    train_split = SequenceSplitSnapshot(
        name="train",
        context=np.asarray(data.X_ctx_train, dtype="float32"),
        targets=np.asarray(data.y_train, dtype="int64"),
        frame=train_frame,
    )
    comparison_splits = [
        SequenceSplitSnapshot(
            name="val",
            context=np.asarray(data.X_ctx_val, dtype="float32"),
            targets=np.asarray(data.y_val, dtype="int64"),
            frame=val_frame,
        ),
        SequenceSplitSnapshot(
            name="test",
            context=np.asarray(data.X_ctx_test, dtype="float32"),
            targets=np.asarray(data.y_test, dtype="int64"),
            frame=test_frame,
        ),
    ]

    context_rows = compute_context_feature_drift_rows(
        feature_names=list(data.context_features),
        reference_split=train_split,
        comparison_splits=comparison_splits,
    )
    segment_rows = compute_segment_share_shift_rows(
        reference_split=train_split,
        comparison_splits=comparison_splits,
        segment_extractors=_segment_bucket_frames,
    )
    target_drift = compute_target_distribution_drift(
        reference_split=train_split,
        comparison_splits=comparison_splits,
    )

    top_context = context_rows[0] if context_rows else {}
    top_segment = segment_rows[0] if segment_rows else {}
    summary_payload = {
        "train_rows": int(len(data.X_seq_train)),
        "val_rows": int(len(data.X_seq_val)),
        "test_rows": int(len(data.X_seq_test)),
        "split_rows": {
            "train": int(len(data.X_seq_train)),
            "val": int(len(data.X_seq_val)),
            "test": int(len(data.X_seq_test)),
        },
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
            "reference_split",
            "compare_split",
            "feature",
            "reference_mean",
            "compare_mean",
            "std_mean_diff",
            "abs_std_mean_diff",
        ],
    )

    segment_csv = output_dir / "segment_drift.csv"
    _write_csv(
        segment_csv,
        segment_rows,
        fieldnames=[
            "reference_split",
            "compare_split",
            "segment",
            "bucket",
            "reference_share",
            "compare_share",
            "abs_share_shift",
        ],
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
