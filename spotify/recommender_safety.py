from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import csv
import json
import math
import os
import re

import numpy as np
import pandas as pd

from .run_artifacts import write_csv_rows
from .uncertainty import (
    SplitConformalCalibration,
    apply_temperature_scaling,
    fit_operating_abstention_threshold,
    fit_split_conformal_classifier,
    fit_temperature_scaling,
    summarize_prediction_sets,
)


@dataclass(frozen=True)
class SequenceSplitSnapshot:
    name: str
    context: np.ndarray
    targets: np.ndarray | None = None
    frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class TemporalBacktestWindow:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    train_rows: int
    test_rows: int


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _max_finite(*values: object) -> float:
    finite_values: list[float] = []
    for value in values:
        resolved = _safe_float(value)
        if not math.isnan(resolved):
            finite_values.append(resolved)
    if not finite_values:
        return float("nan")
    return float(max(finite_values))


def _risk_metric_max(risk_payload: dict[str, object], *, base_name: str) -> float:
    return _max_finite(
        risk_payload.get(f"val_{base_name}"),
        risk_payload.get(f"test_{base_name}"),
    )


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(value):
        return float(default)
    return float(min(maximum, max(minimum, value)))


def _env_float_with_prefix(
    suffix: str,
    default: float,
    *,
    env_prefix: str | None = None,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    if env_prefix:
        scoped_name = f"SPOTIFY_{str(env_prefix).strip().upper()}_{suffix}"
        scoped_raw = os.getenv(scoped_name, "").strip()
        if scoped_raw:
            return _env_float(scoped_name, default, minimum=minimum, maximum=maximum)
    return _env_float(f"SPOTIFY_{suffix}", default, minimum=minimum, maximum=maximum)


def _slugify(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw).strip())
    value = re.sub(r"_{2,}", "_", value).strip("_")
    return value or "metric"


def _coerce_row(row: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    raise TypeError(f"Unsupported row type for safety artifact writing: {type(row)!r}")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _sample_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": float("nan"), "std": float("nan"), "count": 0.0, "stderr": float("nan")}
    count = float(len(scores))
    mean = float(sum(scores) / count)
    if len(scores) < 2:
        std = 0.0
    else:
        variance = sum((score - mean) ** 2 for score in scores) / float(len(scores) - 1)
        std = math.sqrt(max(0.0, variance))
    stderr = float(std / math.sqrt(count)) if count > 0 else float("nan")
    return {"mean": mean, "std": float(std), "count": count, "stderr": stderr}


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


def _js_divergence_from_counts(a_counts: np.ndarray, b_counts: np.ndarray) -> float:
    a_arr = np.asarray(a_counts, dtype="float64").reshape(-1)
    b_arr = np.asarray(b_counts, dtype="float64").reshape(-1)
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


def _labels_and_counts(values: np.ndarray) -> tuple[list[object], dict[object, int]]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return [], {}
    labels, counts = np.unique(arr, return_counts=True)
    label_list = [label.item() if isinstance(label, np.generic) else label for label in labels.tolist()]
    return label_list, {label: int(count) for label, count in zip(label_list, counts.tolist())}


def _distribution_counts(reference: np.ndarray, compare: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_labels, reference_counts = _labels_and_counts(reference)
    compare_labels, compare_counts = _labels_and_counts(compare)
    all_labels = list(dict.fromkeys(reference_labels + compare_labels))
    if not all_labels:
        return np.zeros(0, dtype="float64"), np.zeros(0, dtype="float64")
    ref = np.asarray([reference_counts.get(label, 0) for label in all_labels], dtype="float64")
    comp = np.asarray([compare_counts.get(label, 0) for label in all_labels], dtype="float64")
    return ref, comp


def _string_bucket_counts(values: np.ndarray) -> tuple[dict[str, int], int]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {}, 0
    labels, counts = _labels_and_counts(arr.astype(object, copy=False))
    return {str(label): int(counts.get(label, 0)) for label in labels}, int(arr.size)


def build_temporal_backtest_windows(
    n_rows: int,
    folds: int,
    *,
    min_train_rows: int = 100,
) -> list[TemporalBacktestWindow]:
    if folds <= 0 or n_rows <= 0:
        return []

    base_train = max(int(min_train_rows), n_rows // (folds + 1))
    if base_train >= n_rows:
        return []
    test_size = max(1, (n_rows - base_train) // folds)
    windows: list[TemporalBacktestWindow] = []

    train_end = base_train
    for fold_idx in range(1, folds + 1):
        test_start = train_end
        test_end = min(n_rows, test_start + test_size)
        if test_end <= test_start:
            break
        windows.append(
            TemporalBacktestWindow(
                fold=fold_idx,
                train_end=int(train_end),
                test_start=int(test_start),
                test_end=int(test_end),
                train_rows=int(test_start),
                test_rows=int(test_end - test_start),
            )
        )
        train_end = test_end
        if train_end >= n_rows:
            break

    return windows


def summarize_backtest_rows(
    rows: list[Mapping[str, object] | object],
    *,
    metric_name: str = "top1",
    model_key: str = "model_name",
) -> list[dict[str, object]]:
    by_model: dict[str, list[float]] = {}
    for raw_row in rows:
        row = _coerce_row(raw_row)
        model_name = str(row.get(model_key, "")).strip()
        score = _safe_float(row.get(metric_name))
        if not model_name or math.isnan(score):
            continue
        by_model.setdefault(model_name, []).append(score)

    summary_rows: list[dict[str, object]] = []
    for model_name, scores in by_model.items():
        stats = _sample_stats(scores)
        summary_rows.append(
            {
                "model_name": model_name,
                "metric_name": metric_name,
                "mean": stats["mean"],
                "std": stats["std"],
                "count": int(stats["count"]),
            }
        )
    summary_rows.sort(key=lambda row: _safe_float(row.get("mean")), reverse=True)
    return summary_rows


def write_temporal_backtest_artifacts(
    rows: list[Mapping[str, object] | object],
    *,
    output_dir: Path,
    metric_name: str = "top1",
    model_key: str = "model_name",
    fold_key: str = "fold",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = [_coerce_row(row) for row in rows]

    artifact_paths: list[Path] = []
    csv_path = _write_csv(output_dir / "temporal_backtest.csv", payload_rows)
    artifact_paths.append(csv_path)

    json_path = output_dir / "temporal_backtest.json"
    json_path.write_text(json.dumps(payload_rows, indent=2), encoding="utf-8")
    artifact_paths.append(json_path)

    summary_rows = summarize_backtest_rows(payload_rows, metric_name=metric_name, model_key=model_key)
    summary_csv = _write_csv(
        output_dir / "temporal_backtest_summary.csv",
        summary_rows,
        fieldnames=["model_name", "metric_name", "mean", "std", "count"],
    )
    artifact_paths.append(summary_csv)

    summary_json = output_dir / "temporal_backtest_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    artifact_paths.append(summary_json)

    valid_rows = []
    for row in payload_rows:
        model_name = str(row.get(model_key, "")).strip()
        fold = row.get(fold_key)
        score = _safe_float(row.get(metric_name))
        if not model_name or fold is None or math.isnan(score):
            continue
        try:
            fold_value = int(fold)
        except (TypeError, ValueError):
            continue
        valid_rows.append({"model_name": model_name, "fold": fold_value, metric_name: score})

    if valid_rows:
        import matplotlib.pyplot as plt

        by_model: dict[str, list[dict[str, object]]] = {}
        for row in valid_rows:
            by_model.setdefault(str(row["model_name"]), []).append(row)

        fig, ax = plt.subplots(figsize=(10, 5))
        for model_name, model_rows in sorted(by_model.items()):
            ordered = sorted(model_rows, key=lambda item: int(item["fold"]))
            x = [int(row["fold"]) for row in ordered]
            y = [_safe_float(row.get(metric_name)) for row in ordered]
            ax.plot(x, y, marker="o", label=model_name)

        ax.set_title(f"Temporal Backtest {metric_name}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric_name)
        ax.set_xticks(sorted({int(row["fold"]) for row in valid_rows}))
        ax.legend()
        fig.tight_layout()
        plot_path = output_dir / f"temporal_backtest_{_slugify(metric_name)}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        artifact_paths.append(plot_path)

    return artifact_paths


def run_temporal_backtest_benchmark(
    *,
    n_rows: int,
    folds: int,
    evaluators: Mapping[str, Callable[[TemporalBacktestWindow], Mapping[str, object] | None]],
    metric_name: str = "top1",
    output_dir: Path | None = None,
    logger=None,
    min_train_rows: int = 100,
) -> list[dict[str, object]]:
    windows = build_temporal_backtest_windows(n_rows, folds, min_train_rows=min_train_rows)
    results: list[dict[str, object]] = []
    for window in windows:
        for model_name, evaluator in evaluators.items():
            try:
                payload = evaluator(window)
            except Exception as exc:
                if logger is not None:
                    logger.warning("Generic backtest evaluator failed for %s fold=%d: %s", model_name, window.fold, exc)
                continue
            if payload is None:
                continue
            row = dict(payload)
            row.setdefault("model_name", model_name)
            row.setdefault("fold", int(window.fold))
            row.setdefault("train_rows", int(window.train_rows))
            row.setdefault("test_rows", int(window.test_rows))
            results.append(row)

    if output_dir is not None:
        write_temporal_backtest_artifacts(results, output_dir=output_dir, metric_name=metric_name)
    return results


def compute_context_feature_drift_rows(
    *,
    feature_names: list[str],
    reference_split: SequenceSplitSnapshot,
    comparison_splits: list[SequenceSplitSnapshot],
) -> list[dict[str, object]]:
    reference_ctx = np.asarray(reference_split.context, dtype="float64")
    if reference_ctx.ndim != 2:
        return []

    rows: list[dict[str, object]] = []
    for idx, feature_name in enumerate(feature_names):
        if idx >= reference_ctx.shape[1]:
            continue
        reference_values = reference_ctx[:, idx]
        reference_mean = float(np.mean(reference_values)) if reference_values.size else float("nan")
        for compare_split in comparison_splits:
            compare_ctx = np.asarray(compare_split.context, dtype="float64")
            if compare_ctx.ndim != 2 or idx >= compare_ctx.shape[1]:
                continue
            compare_values = compare_ctx[:, idx]
            compare_mean = float(np.mean(compare_values)) if compare_values.size else float("nan")
            pooled = _pooled_std(reference_values, compare_values)
            std_mean_diff = (
                (compare_mean - reference_mean) / pooled if pooled and not math.isnan(pooled) else float("nan")
            )
            rows.append(
                {
                    "reference_split": reference_split.name,
                    "compare_split": compare_split.name,
                    "feature": feature_name,
                    "reference_mean": reference_mean,
                    "compare_mean": compare_mean,
                    "std_mean_diff": std_mean_diff,
                    "abs_std_mean_diff": abs(_safe_float(std_mean_diff)),
                }
            )
    rows.sort(key=lambda row: _safe_float(row.get("abs_std_mean_diff")), reverse=True)
    return rows


def compute_segment_share_shift_rows(
    *,
    reference_split: SequenceSplitSnapshot,
    comparison_splits: list[SequenceSplitSnapshot],
    segment_extractors: (
        Mapping[str, Callable[[pd.DataFrame], np.ndarray]]
        | Callable[[pd.DataFrame], Mapping[str, np.ndarray]]
    ),
) -> list[dict[str, object]]:
    if reference_split.frame is None or not segment_extractors:
        return []

    def _extract(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if callable(segment_extractors):
            resolved = segment_extractors(frame)
            return {str(name): np.asarray(values).reshape(-1) for name, values in dict(resolved).items()}
        return {
            str(name): np.asarray(extractor(frame)).reshape(-1)
            for name, extractor in segment_extractors.items()
        }

    reference_frame = reference_split.frame.reset_index(drop=True)
    reference_segments = _extract(reference_frame)
    reference_counts_by_segment = {
        segment_name: _string_bucket_counts(reference_values)
        for segment_name, reference_values in reference_segments.items()
    }
    rows: list[dict[str, object]] = []
    for compare_split in comparison_splits:
        if compare_split.frame is None:
            continue
        compare_frame = compare_split.frame.reset_index(drop=True)
        compare_segments = _extract(compare_frame)
        compare_counts_by_segment = {
            segment_name: _string_bucket_counts(compare_values)
            for segment_name, compare_values in compare_segments.items()
        }
        for segment_name, (reference_counts, reference_total_raw) in reference_counts_by_segment.items():
            compare_counts_payload = compare_counts_by_segment.get(segment_name)
            if compare_counts_payload is None:
                continue
            compare_counts, compare_total_raw = compare_counts_payload
            reference_total = max(1, reference_total_raw)
            compare_total = max(1, compare_total_raw)
            all_buckets = sorted(set(reference_counts) | set(compare_counts))
            for bucket in all_buckets:
                reference_share = float(reference_counts.get(bucket, 0)) / float(reference_total)
                compare_share = float(compare_counts.get(bucket, 0)) / float(compare_total)
                rows.append(
                    {
                        "reference_split": reference_split.name,
                        "compare_split": compare_split.name,
                        "segment": segment_name,
                        "bucket": bucket,
                        "reference_share": reference_share,
                        "compare_share": compare_share,
                        "abs_share_shift": abs(compare_share - reference_share),
                    }
                )
    rows.sort(key=lambda row: _safe_float(row.get("abs_share_shift")), reverse=True)
    return rows


def compute_target_distribution_drift(
    *,
    reference_split: SequenceSplitSnapshot,
    comparison_splits: list[SequenceSplitSnapshot],
) -> dict[str, float]:
    if reference_split.targets is None:
        return {}
    reference_targets = np.asarray(reference_split.targets).reshape(-1)
    target_drift: dict[str, float] = {}
    for compare_split in comparison_splits:
        if compare_split.targets is None:
            continue
        compare_targets = np.asarray(compare_split.targets).reshape(-1)
        reference_counts, compare_counts = _distribution_counts(reference_targets, compare_targets)
        target_drift[f"{reference_split.name}_vs_{compare_split.name}_jsd"] = _js_divergence_from_counts(
            reference_counts,
            compare_counts,
        )
    return target_drift


def build_conformal_abstention_summary(
    *,
    tag: str,
    val_proba: np.ndarray,
    val_y: np.ndarray,
    test_proba: np.ndarray | None = None,
    test_y: np.ndarray | None = None,
    alpha: float = 0.10,
    target_selective_risk: float = 0.50,
    min_accepted_rate: float = 0.10,
    min_risk_drop: float = 0.02,
    env_prefix: str | None = None,
    enable_temperature_scaling: bool = False,
) -> dict[str, object] | None:
    target_selective_risk = _env_float_with_prefix(
        "CONFORMAL_TARGET_SELECTIVE_RISK",
        target_selective_risk,
        env_prefix=env_prefix,
        minimum=0.0,
        maximum=0.99,
    )
    min_accepted_rate = _env_float_with_prefix(
        "CONFORMAL_MIN_ACCEPTED_RATE",
        min_accepted_rate,
        env_prefix=env_prefix,
        minimum=0.01,
        maximum=0.99,
    )
    min_risk_drop = _env_float_with_prefix(
        "CONFORMAL_MIN_RISK_DROP",
        min_risk_drop,
        env_prefix=env_prefix,
        minimum=0.0,
        maximum=0.99,
    )
    val_proba_resolved = np.asarray(val_proba, dtype="float32")
    test_proba_resolved = None if test_proba is None else np.asarray(test_proba, dtype="float32")
    temperature = 1.0
    calibration_method = "raw"
    if enable_temperature_scaling:
        temperature = fit_temperature_scaling(val_proba_resolved, val_y)
        val_proba_resolved = apply_temperature_scaling(val_proba_resolved, temperature)
        if test_proba_resolved is not None:
            test_proba_resolved = apply_temperature_scaling(test_proba_resolved, temperature)
        calibration_method = "temperature_scaling"

    calibration = fit_split_conformal_classifier(val_proba_resolved, val_y, alpha=alpha)
    if calibration is None:
        return None
    operating_threshold = fit_operating_abstention_threshold(
        val_proba_resolved,
        val_y,
        base_threshold=calibration.threshold,
        target_selective_risk=target_selective_risk,
        min_accepted_rate=min_accepted_rate,
        min_risk_drop=min_risk_drop,
    )
    calibration = SplitConformalCalibration(
        method=calibration.method,
        alpha=calibration.alpha,
        qhat=calibration.qhat,
        threshold=calibration.threshold,
        sample_count=calibration.sample_count,
        empirical_coverage=calibration.empirical_coverage,
        mean_set_size=calibration.mean_set_size,
        operating_threshold=operating_threshold,
    )

    payload: dict[str, object] = {
        "tag": str(tag).strip(),
        "calibration": calibration.to_dict(),
        "operating_point": {
            "target_selective_risk": float(target_selective_risk),
            "min_accepted_rate": float(min_accepted_rate),
            "min_risk_drop": float(min_risk_drop),
            "abstention_threshold": float(calibration.abstention_threshold),
        },
        "probability_calibration": {
            "method": calibration_method,
            "temperature": float(temperature),
        },
        "val": summarize_prediction_sets(val_proba_resolved, val_y, calibration=calibration),
    }
    if test_proba_resolved is not None and test_y is not None:
        payload["test"] = summarize_prediction_sets(test_proba_resolved, test_y, calibration=calibration)
    return payload


def _group_passes_risk_caps(
    group_name: str,
    current_risk_metrics: dict[str, dict[str, float]] | None,
    *,
    max_selective_risk: float | None,
    max_abstention_rate: float | None,
) -> bool:
    if max_selective_risk is None and max_abstention_rate is None:
        return True
    risk_payload = (current_risk_metrics or {}).get(group_name, {})
    if not isinstance(risk_payload, dict) or not risk_payload:
        return False
    selective_risk = _risk_metric_max(risk_payload, base_name="selective_risk")
    abstention_rate = _risk_metric_max(risk_payload, base_name="abstention_rate")
    if max_selective_risk is not None:
        if math.isnan(selective_risk) or selective_risk > float(max_selective_risk):
            return False
    if max_abstention_rate is not None:
        if math.isnan(abstention_rate) or abstention_rate > float(max_abstention_rate):
            return False
    return True


def _group_score_lists(
    rows: list[dict[str, object]],
    *,
    group_key: str,
    metric_name: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        group_name = str(row.get(group_key, "")).strip()
        score = _safe_float(row.get(metric_name))
        if not group_name or math.isnan(score):
            continue
        grouped.setdefault(group_name, []).append(score)
    return grouped


def _aggregate_scores(scores: list[float], aggregate: str) -> float:
    if not scores:
        return float("nan")
    if aggregate == "max":
        return float(max(scores))
    return float(sum(scores) / float(len(scores)))


def _best_current_group(
    rows: list[dict[str, object]],
    *,
    group_key: str,
    metric_name: str,
    aggregate: str,
    higher_is_better: bool,
    eligible_groups: set[str] | None = None,
) -> tuple[str, float, list[float]]:
    score_map = _group_score_lists(rows, group_key=group_key, metric_name=metric_name)
    best_group = ""
    best_score = float("-inf") if higher_is_better else float("inf")
    best_scores: list[float] = []
    for group_name, scores in score_map.items():
        if eligible_groups is not None and group_name not in eligible_groups:
            continue
        aggregated = _aggregate_scores(scores, aggregate)
        if math.isnan(aggregated):
            continue
        better = aggregated > best_score if higher_is_better else aggregated < best_score
        if better:
            best_group = group_name
            best_score = aggregated
            best_scores = list(scores)
    return best_group, best_score, best_scores


def _rank_current_groups(
    rows: list[dict[str, object]],
    *,
    group_key: str,
    metric_name: str,
    aggregate: str,
    higher_is_better: bool,
    current_risk_metrics: dict[str, dict[str, float]] | None,
    max_selective_risk: float | None,
    max_abstention_rate: float | None,
) -> list[dict[str, object]]:
    score_map = _group_score_lists(rows, group_key=group_key, metric_name=metric_name)
    ranked: list[dict[str, object]] = []
    for group_name, scores in score_map.items():
        aggregated = _aggregate_scores(scores, aggregate)
        if math.isnan(aggregated):
            continue
        risk_payload = (current_risk_metrics or {}).get(group_name, {})
        risk_payload = risk_payload if isinstance(risk_payload, dict) else {}
        selective_risk = _risk_metric_max(risk_payload, base_name="selective_risk")
        abstention_rate = _risk_metric_max(risk_payload, base_name="abstention_rate")
        blockers: list[str] = []
        if max_selective_risk is not None:
            if math.isnan(selective_risk):
                blockers.append("missing_selective_risk")
            elif selective_risk > float(max_selective_risk):
                blockers.append("selective_risk")
        if max_abstention_rate is not None:
            if math.isnan(abstention_rate):
                blockers.append("missing_abstention_rate")
            elif abstention_rate > float(max_abstention_rate):
                blockers.append("abstention_rate")
        ranked.append(
            {
                "group_name": group_name,
                "score": aggregated,
                "scores": list(scores),
                "risk_eligible": not blockers,
                "risk_blockers": blockers,
                "selective_risk": selective_risk,
                "abstention_rate": abstention_rate,
            }
        )

    ranked.sort(
        key=lambda row: _safe_float(row.get("score")),
        reverse=bool(higher_is_better),
    )
    return ranked


def _best_history_group(
    history_csv: Path | None,
    *,
    current_run_id: str,
    group_key: str,
    metric_name: str,
    aggregate: str,
    higher_is_better: bool,
    current_profile: str | None,
    require_profile_match: bool,
    run_id_key: str,
    profile_key: str,
) -> tuple[str, str, float, list[float]]:
    if history_csv is None or not history_csv.exists():
        return "", "", (float("-inf") if higher_is_better else float("inf")), []

    by_run_group: dict[tuple[str, str], list[float]] = {}
    with history_csv.open("r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            run_id = str(row.get(run_id_key, "")).strip()
            if not run_id or run_id == current_run_id:
                continue
            if require_profile_match and current_profile:
                profile = str(row.get(profile_key, "")).strip().lower()
                if profile and profile != str(current_profile).strip().lower():
                    continue
            group_name = str(row.get(group_key, "")).strip()
            score = _safe_float(row.get(metric_name))
            if not group_name or math.isnan(score):
                continue
            by_run_group.setdefault((run_id, group_name), []).append(score)

    best_run_id = ""
    best_group = ""
    best_score = float("-inf") if higher_is_better else float("inf")
    best_scores: list[float] = []
    for (run_id, group_name), scores in by_run_group.items():
        aggregated = _aggregate_scores(scores, aggregate)
        if math.isnan(aggregated):
            continue
        better = aggregated > best_score if higher_is_better else aggregated < best_score
        if better:
            best_run_id = run_id
            best_group = group_name
            best_score = aggregated
            best_scores = list(scores)

    return best_run_id, best_group, best_score, best_scores


def _preferred_history_group(
    history_csv: Path | None,
    *,
    current_run_id: str,
    target_run_id: str | None,
    target_group_name: str | None,
    metric_name: str,
    aggregate: str,
    current_profile: str | None,
    require_profile_match: bool,
    run_id_key: str,
    profile_key: str,
    group_key: str,
) -> tuple[str, str, float, list[float]]:
    resolved_run_id = str(target_run_id or "").strip()
    resolved_group_name = str(target_group_name or "").strip()
    if (
        history_csv is None
        or not history_csv.exists()
        or not resolved_run_id
        or not resolved_group_name
        or resolved_run_id == current_run_id
    ):
        return "", "", float("nan"), []

    scores: list[float] = []
    with history_csv.open("r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            run_id = str(row.get(run_id_key, "")).strip()
            if run_id != resolved_run_id:
                continue
            if require_profile_match and current_profile:
                profile = str(row.get(profile_key, "")).strip().lower()
                if profile and profile != str(current_profile).strip().lower():
                    continue
            group_name = str(row.get(group_key, "")).strip()
            if group_name != resolved_group_name:
                continue
            score = _safe_float(row.get(metric_name))
            if math.isnan(score):
                continue
            scores.append(score)

    if not scores:
        return "", "", float("nan"), []
    return (
        resolved_run_id,
        resolved_group_name,
        _aggregate_scores(scores, aggregate),
        scores,
    )


def _no_current_result_payload(
    *,
    metric_name: str,
    regression_threshold: float,
    aggregate: str,
    score_direction: str,
    require_profile_match: bool,
    require_significant_lift: bool,
    significance_z: float,
    max_selective_risk: float | None,
    max_abstention_rate: float | None,
) -> dict[str, object]:
    return {
        "status": "no_current_results",
        "promoted": False,
        "metric_name": metric_name,
        "aggregate": aggregate,
        "score_direction": score_direction,
        "profile_match": bool(require_profile_match),
        "require_significant_lift": bool(require_significant_lift),
        "significance_z": float(significance_z),
        "significance_margin": float("nan"),
        "significant_lift": None,
        "threshold": float(regression_threshold),
        "regression": float("nan"),
        "champion_run_id": "",
        "champion_model_name": "",
        "champion_score": float("nan"),
        "challenger_model_name": "",
        "challenger_score": float("nan"),
        "champion_score_std": float("nan"),
        "challenger_score_std": float("nan"),
        "champion_score_count": 0.0,
        "challenger_score_count": 0.0,
        "max_selective_risk": (float(max_selective_risk) if max_selective_risk is not None else float("nan")),
        "max_abstention_rate": (float(max_abstention_rate) if max_abstention_rate is not None else float("nan")),
        "challenger_selective_risk": float("nan"),
        "challenger_abstention_rate": float("nan"),
        "selected_candidate_rank": 0,
        "eligible_candidate_count": 0,
        "challenger_selection_reason": "no_current_results",
        "top_candidate_model_name": "",
        "top_candidate_score": float("nan"),
        "top_candidate_risk_blockers": [],
    }


def evaluate_promotion_gate(
    *,
    history_csv: Path | None,
    current_run_id: str,
    current_rows: list[dict[str, object]],
    metric_name: str,
    regression_threshold: float,
    group_key: str = "model_name",
    aggregate: str = "mean",
    score_direction: str = "maximize",
    current_profile: str | None = None,
    require_profile_match: bool = True,
    require_significant_lift: bool = False,
    significance_z: float = 1.96,
    current_risk_metrics: dict[str, dict[str, float]] | None = None,
    max_selective_risk: float | None = None,
    max_abstention_rate: float | None = None,
    preferred_champion_run_id: str | None = None,
    preferred_champion_model_name: str | None = None,
    run_id_key: str = "run_id",
    profile_key: str = "profile",
) -> dict[str, object]:
    threshold = max(0.0, float(regression_threshold))
    resolved_aggregate = str(aggregate).strip().lower()
    if resolved_aggregate not in ("mean", "max"):
        resolved_aggregate = "mean"

    resolved_direction = str(score_direction).strip().lower()
    if resolved_direction not in ("maximize", "minimize"):
        resolved_direction = "maximize"
    higher_is_better = resolved_direction == "maximize"

    ranked_candidates = _rank_current_groups(
        current_rows,
        group_key=group_key,
        metric_name=metric_name,
        aggregate=resolved_aggregate,
        higher_is_better=higher_is_better,
        current_risk_metrics=current_risk_metrics,
        max_selective_risk=max_selective_risk,
        max_abstention_rate=max_abstention_rate,
    )
    score_map = _group_score_lists(current_rows, group_key=group_key, metric_name=metric_name)
    eligible_groups = {
        group_name
        for group_name in score_map
        if _group_passes_risk_caps(
            group_name,
            current_risk_metrics,
            max_selective_risk=max_selective_risk,
            max_abstention_rate=max_abstention_rate,
        )
    }
    top_candidate = ranked_candidates[0] if ranked_candidates else {}
    challenger_model_name, challenger_score, challenger_scores = _best_current_group(
        current_rows,
        group_key=group_key,
        metric_name=metric_name,
        aggregate=resolved_aggregate,
        higher_is_better=higher_is_better,
        eligible_groups=(eligible_groups if eligible_groups else None),
    )
    if not challenger_model_name:
        return _no_current_result_payload(
            metric_name=metric_name,
            regression_threshold=threshold,
            aggregate=resolved_aggregate,
            score_direction=resolved_direction,
            require_profile_match=require_profile_match,
            require_significant_lift=require_significant_lift,
            significance_z=significance_z,
            max_selective_risk=max_selective_risk,
            max_abstention_rate=max_abstention_rate,
        )
    selected_candidate_rank = next(
        (
            idx
            for idx, row in enumerate(ranked_candidates, start=1)
            if str(row.get("group_name", "")).strip() == challenger_model_name
        ),
        0,
    )
    challenger_selection_reason = (
        "highest_scoring_candidate"
        if selected_candidate_rank <= 1
        else "highest_scoring_risk_eligible_candidate"
    )

    champion_run_id, champion_model_name, champion_score, champion_scores = _preferred_history_group(
        history_csv,
        current_run_id=current_run_id,
        target_run_id=preferred_champion_run_id,
        target_group_name=preferred_champion_model_name,
        metric_name=metric_name,
        aggregate=resolved_aggregate,
        current_profile=current_profile,
        require_profile_match=require_profile_match,
        run_id_key=run_id_key,
        profile_key=profile_key,
        group_key=group_key,
    )
    if not champion_run_id or not champion_model_name or math.isnan(champion_score):
        champion_run_id, champion_model_name, champion_score, champion_scores = _best_history_group(
            history_csv,
            current_run_id=current_run_id,
            group_key=group_key,
            metric_name=metric_name,
            aggregate=resolved_aggregate,
            higher_is_better=higher_is_better,
            current_profile=current_profile,
            require_profile_match=require_profile_match,
            run_id_key=run_id_key,
            profile_key=profile_key,
        )

    challenger_risk = (current_risk_metrics or {}).get(challenger_model_name, {})
    challenger_selective_risk = _risk_metric_max(challenger_risk, base_name="selective_risk")
    challenger_abstention_rate = _risk_metric_max(challenger_risk, base_name="abstention_rate")

    if (higher_is_better and champion_score == float("-inf")) or (not higher_is_better and champion_score == float("inf")):
        status = "no_prior_champion"
        promoted = True
        if max_selective_risk is not None and not math.isnan(challenger_selective_risk):
            if challenger_selective_risk > float(max_selective_risk):
                promoted = False
                status = "fail_selective_risk"
        if promoted and max_abstention_rate is not None and not math.isnan(challenger_abstention_rate):
            if challenger_abstention_rate > float(max_abstention_rate):
                promoted = False
                status = "fail_abstention_rate"
        return {
            "status": status,
            "promoted": promoted,
            "metric_name": metric_name,
            "aggregate": resolved_aggregate,
            "score_direction": resolved_direction,
            "profile_match": bool(require_profile_match),
            "require_significant_lift": bool(require_significant_lift),
            "significance_z": float(significance_z),
            "significance_margin": float("nan"),
            "significant_lift": None,
            "threshold": threshold,
            "regression": 0.0,
            "champion_run_id": "",
            "champion_model_name": "",
            "champion_score": float("nan"),
            "challenger_model_name": challenger_model_name,
            "challenger_score": challenger_score,
            "champion_score_std": float("nan"),
            "challenger_score_std": float("nan"),
            "champion_score_count": 0.0,
            "challenger_score_count": float(len(challenger_scores)),
            "max_selective_risk": (float(max_selective_risk) if max_selective_risk is not None else float("nan")),
            "max_abstention_rate": (float(max_abstention_rate) if max_abstention_rate is not None else float("nan")),
            "challenger_selective_risk": challenger_selective_risk,
            "challenger_abstention_rate": challenger_abstention_rate,
            "selected_candidate_rank": int(selected_candidate_rank),
            "eligible_candidate_count": int(sum(1 for row in ranked_candidates if bool(row.get("risk_eligible", False)))),
            "challenger_selection_reason": challenger_selection_reason,
            "top_candidate_model_name": str(top_candidate.get("group_name", "")),
            "top_candidate_score": _safe_float(top_candidate.get("score")),
            "top_candidate_risk_blockers": list(top_candidate.get("risk_blockers", [])),
        }

    regression = (
        float(champion_score - challenger_score)
        if higher_is_better
        else float(challenger_score - champion_score)
    )
    promoted = regression <= threshold
    status = "pass" if promoted else "fail"
    challenger_stats = _sample_stats(challenger_scores)
    champion_stats = _sample_stats(champion_scores)
    significance_margin = float("nan")
    significant_lift = None

    improvement = (
        float(challenger_score - champion_score)
        if higher_is_better
        else float(champion_score - challenger_score)
    )
    if require_significant_lift and not math.isnan(challenger_stats["stderr"]) and not math.isnan(champion_stats["stderr"]):
        combined_stderr = math.sqrt(max(0.0, challenger_stats["stderr"] ** 2 + champion_stats["stderr"] ** 2))
        significance_margin = float(max(0.0, significance_z) * combined_stderr)
        significant_lift = bool(improvement > significance_margin)
        if promoted and improvement > 0.0 and not significant_lift:
            promoted = False
            status = "fail_not_significant"

    if promoted and max_selective_risk is not None and not math.isnan(challenger_selective_risk):
        if challenger_selective_risk > float(max_selective_risk):
            promoted = False
            status = "fail_selective_risk"
    if promoted and max_abstention_rate is not None and not math.isnan(challenger_abstention_rate):
        if challenger_abstention_rate > float(max_abstention_rate):
            promoted = False
            status = "fail_abstention_rate"

    return {
        "status": status,
        "promoted": promoted,
        "metric_name": metric_name,
        "aggregate": resolved_aggregate,
        "score_direction": resolved_direction,
        "profile_match": bool(require_profile_match),
        "require_significant_lift": bool(require_significant_lift),
        "significance_z": float(significance_z),
        "significance_margin": significance_margin,
        "significant_lift": significant_lift,
        "threshold": threshold,
        "regression": regression,
        "champion_run_id": champion_run_id,
        "champion_model_name": champion_model_name,
        "champion_score": champion_score,
        "challenger_model_name": challenger_model_name,
        "challenger_score": challenger_score,
        "champion_score_std": champion_stats["std"],
        "challenger_score_std": challenger_stats["std"],
        "champion_score_count": champion_stats["count"],
        "challenger_score_count": challenger_stats["count"],
        "max_selective_risk": (float(max_selective_risk) if max_selective_risk is not None else float("nan")),
        "max_abstention_rate": (float(max_abstention_rate) if max_abstention_rate is not None else float("nan")),
        "challenger_selective_risk": challenger_selective_risk,
        "challenger_abstention_rate": challenger_abstention_rate,
        "selected_candidate_rank": int(selected_candidate_rank),
        "eligible_candidate_count": int(sum(1 for row in ranked_candidates if bool(row.get("risk_eligible", False)))),
        "challenger_selection_reason": challenger_selection_reason,
        "top_candidate_model_name": str(top_candidate.get("group_name", "")),
        "top_candidate_score": _safe_float(top_candidate.get("score")),
        "top_candidate_risk_blockers": list(top_candidate.get("risk_blockers", [])),
    }
