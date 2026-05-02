from __future__ import annotations

from pathlib import Path
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
from .run_artifacts import write_csv_rows


_TECHNICAL_FEATURES = {"offline"}
_TEMPORAL_FEATURES = {
    "hour",
    "dayofweek",
    "month",
    "time_diff",
    "session_position",
    "session_elapsed_seconds",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
}


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
    write_csv_rows(path, rows, fieldnames=fieldnames)


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


def _feature_group(feature_name: str) -> str:
    name = str(feature_name).strip()
    if not name:
        return "other"
    if name.startswith("tech_") or name in _TECHNICAL_FEATURES:
        return "technical"
    if name in _TEMPORAL_FEATURES:
        return "temporal"
    if any(
        token in name
        for token in (
            "skip",
            "repeat",
            "artist_",
            "transition",
            "session_",
            "listen_",
            "days_since",
            "hours_since",
            "plays_",
        )
    ):
        return "behavioral"
    if name in {"danceability", "energy", "tempo"}:
        return "content"
    return "other"


def _summarize_context_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        compare_split = str(row.get("compare_split", "")).strip()
        feature = str(row.get("feature", "")).strip()
        grouped.setdefault((compare_split, _feature_group(feature)), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (compare_split, feature_group), group_rows in grouped.items():
        abs_diffs = [_safe_float(item.get("abs_std_mean_diff")) for item in group_rows]
        abs_diffs = [value for value in abs_diffs if math.isfinite(value)]
        if not abs_diffs:
            continue
        top_row = max(group_rows, key=lambda item: _safe_float(item.get("abs_std_mean_diff")))
        summary_rows.append(
            {
                "compare_split": compare_split,
                "feature_group": feature_group,
                "feature_count": int(len(group_rows)),
                "mean_abs_std_mean_diff": float(np.mean(abs_diffs)),
                "max_abs_std_mean_diff": float(max(abs_diffs)),
                "top_feature": str(top_row.get("feature", "")),
                "top_feature_abs_std_mean_diff": _safe_float(top_row.get("abs_std_mean_diff")),
            }
        )
    summary_rows.sort(
        key=lambda row: (_safe_float(row.get("max_abs_std_mean_diff")), _safe_float(row.get("mean_abs_std_mean_diff"))),
        reverse=True,
    )
    return summary_rows


def _largest_group_shift(group_rows: list[dict[str, object]], *, compare_split: str, feature_group: str) -> dict[str, object]:
    candidates = [
        row
        for row in group_rows
        if str(row.get("compare_split", "")).strip() == compare_split
        and str(row.get("feature_group", "")).strip() == feature_group
    ]
    return max(candidates, key=lambda row: _safe_float(row.get("max_abs_std_mean_diff")), default={})


def _interpret_target_drift(
    *,
    target_drift: dict[str, float],
    largest_segment_shift: dict[str, object],
    group_rows: list[dict[str, object]],
) -> dict[str, object]:
    test_rows = [row for row in group_rows if str(row.get("compare_split", "")).strip() == "test"]
    dominant = max(test_rows, key=lambda row: _safe_float(row.get("max_abs_std_mean_diff")), default={})
    dominant_group = str(dominant.get("feature_group", "")).strip() or "other"
    target_jsd = _safe_float(target_drift.get("train_vs_test_jsd"))
    segment_name = str(largest_segment_shift.get("segment", "")).strip()
    if dominant_group == "technical":
        summary = "Technical playback conditions are the primary context shift driver, so regressions may reflect environment changes as much as preference drift."
    elif dominant_group == "behavioral":
        summary = "Behavioral context is shifting faster than technical conditions, which suggests the current drift is mostly user-behavior or product-mix change."
    elif dominant_group == "temporal":
        summary = "Temporal usage patterns are the main drift driver, so hour/day mix should be normalized before blaming the model."
    else:
        summary = "No single drift family dominates, so treat regressions as mixed-context movement until more runs are collected."
    if math.isfinite(target_jsd) and target_jsd >= 0.15 and segment_name:
        summary += f" The largest segment-share change is `{segment_name}`."
    return {
        "dominant_context_driver": dominant_group,
        "summary": summary,
        "target_jsd": target_jsd,
    }


def _build_drift_brief(summary_payload: dict[str, object]) -> dict[str, object]:
    target_drift = summary_payload.get("target_drift", {})
    target_drift = target_drift if isinstance(target_drift, dict) else {}
    largest_context_shift = summary_payload.get("largest_context_shift", {})
    largest_context_shift = largest_context_shift if isinstance(largest_context_shift, dict) else {}
    largest_segment_shift = summary_payload.get("largest_segment_shift", {})
    largest_segment_shift = largest_segment_shift if isinstance(largest_segment_shift, dict) else {}
    interpretation = summary_payload.get("drift_interpretation", {})
    interpretation = interpretation if isinstance(interpretation, dict) else {}

    target_jsd = _safe_float(target_drift.get("train_vs_test_jsd"))
    segment_shift = _safe_float(largest_segment_shift.get("abs_share_shift"))
    dominant_group = str(interpretation.get("dominant_context_driver", "")).strip() or "other"
    status = "attention" if (math.isfinite(target_jsd) and target_jsd >= 0.15) else "stable"
    if math.isfinite(segment_shift) and segment_shift >= 0.10:
        status = "attention"

    findings: list[str] = []
    if math.isfinite(target_jsd):
        findings.append(f"Target drift JSD is `{target_jsd:.3f}` from train to test.")
    top_feature = str(largest_context_shift.get("feature", "")).strip()
    top_context_gap = _safe_float(largest_context_shift.get("abs_std_mean_diff"))
    if top_feature and math.isfinite(top_context_gap):
        findings.append(f"Largest context shift is `{top_feature}` at `{top_context_gap:.3f}` standardized mean difference.")
    segment_name = str(largest_segment_shift.get("segment", "")).strip()
    bucket_name = str(largest_segment_shift.get("bucket", "")).strip()
    if segment_name and bucket_name and math.isfinite(segment_shift):
        findings.append(f"Largest segment-share shift is `{segment_name}={bucket_name}` at `{segment_shift:.3f}`.")
    if interpretation.get("summary"):
        findings.append(str(interpretation.get("summary")))

    actions = [
        "Inspect the top shifted context feature before treating ranking changes as pure model regressions.",
        "Review the largest segment-share change and compare it against the latest promoted baseline.",
        "Use the dominant drift family to decide whether the next fix belongs in calibration, routing, or data normalization.",
    ]
    inspect_paths = [
        "analysis/data_drift_summary.json",
        "analysis/context_feature_drift.csv",
        "analysis/segment_drift.csv",
        "analysis/context_feature_drift_by_group.csv",
    ]
    return {
        "status": status,
        "target_jsd": target_jsd,
        "dominant_context_driver": dominant_group,
        "largest_context_feature": top_feature,
        "largest_context_shift": top_context_gap,
        "largest_segment_label": f"{segment_name}={bucket_name}" if segment_name and bucket_name else "",
        "largest_segment_shift": segment_shift,
        "summary": findings[:4],
        "recommended_actions": actions,
        "inspect_paths": inspect_paths,
    }


def _build_drift_brief_markdown(brief_payload: dict[str, object]) -> list[str]:
    lines = [
        "# Data Drift Brief",
        "",
        f"- Status: `{brief_payload.get('status', 'unknown')}`",
        f"- Target drift JSD: `{_safe_float(brief_payload.get('target_jsd')):.3f}`",
        f"- Dominant context driver: `{brief_payload.get('dominant_context_driver', 'other')}`",
        f"- Largest context feature: `{brief_payload.get('largest_context_feature', '')}`",
        f"- Largest segment shift: `{brief_payload.get('largest_segment_label', '')}` at `{_safe_float(brief_payload.get('largest_segment_shift')):.3f}`",
        "",
        "## Summary",
        "",
    ]
    for item in brief_payload.get("summary", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Actions", ""])
    for item in brief_payload.get("recommended_actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Inspect Paths", ""])
    for item in brief_payload.get("inspect_paths", []):
        lines.append(f"- `{item}`")
    return lines


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
    context_group_rows = _summarize_context_groups(context_rows)

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
        "largest_behavioral_context_shift": _largest_group_shift(context_group_rows, compare_split="test", feature_group="behavioral"),
        "largest_technical_context_shift": _largest_group_shift(context_group_rows, compare_split="test", feature_group="technical"),
        "largest_temporal_context_shift": _largest_group_shift(context_group_rows, compare_split="test", feature_group="temporal"),
        "context_drift_by_group": context_group_rows,
        "drift_interpretation": _interpret_target_drift(
            target_drift=target_drift,
            largest_segment_shift=top_segment,
            group_rows=context_group_rows,
        ),
    }

    summary_path = output_dir / "data_drift_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    brief_payload = _build_drift_brief(summary_payload)
    brief_json_path = output_dir / "data_drift_brief.json"
    brief_json_path.write_text(json.dumps(brief_payload, indent=2), encoding="utf-8")
    brief_md_path = output_dir / "data_drift_brief.md"
    brief_md_path.write_text("\n".join(_build_drift_brief_markdown(brief_payload)).rstrip() + "\n", encoding="utf-8")

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

    group_csv = output_dir / "context_feature_drift_by_group.csv"
    _write_csv(
        group_csv,
        context_group_rows,
        fieldnames=[
            "compare_split",
            "feature_group",
            "feature_count",
            "mean_abs_std_mean_diff",
            "max_abs_std_mean_diff",
            "top_feature",
            "top_feature_abs_std_mean_diff",
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

    return [summary_path, brief_json_path, brief_md_path, context_csv, segment_csv, group_csv, context_plot, segment_plot]
