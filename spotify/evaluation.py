from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import re

import numpy as np
import pandas as pd

from .benchmarks import ClassicalFeatureBundle, build_classical_estimator, build_classical_feature_bundle, sample_rows
from .data import PreparedData
from .probability_bundles import load_prediction_bundle
from .uncertainty import fit_split_conformal_classifier, summarize_prediction_sets


def _slugify(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw).strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "model"


def _to_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _build_label_lookup(df: pd.DataFrame) -> dict[int, str]:
    if "artist_label" not in df.columns or "master_metadata_album_artist_name" not in df.columns:
        return {}
    unique = (
        df[["artist_label", "master_metadata_album_artist_name"]]
        .drop_duplicates(subset=["artist_label"])
        .sort_values("artist_label")
    )
    return {
        int(row["artist_label"]): str(row["master_metadata_album_artist_name"])
        for _, row in unique.iterrows()
    }


def _encode_labels_to_local_indices(y_true: np.ndarray, class_labels: np.ndarray | None) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    if class_labels is None:
        return y_true.astype("int64", copy=False)
    labels = np.asarray(class_labels).reshape(-1)
    if labels.size == 0:
        return np.full(y_true.shape[0], -1, dtype="int64")
    lookup = {label.item() if isinstance(label, np.generic) else label: idx for idx, label in enumerate(labels)}
    mapped = np.fromiter((lookup.get(item.item() if isinstance(item, np.generic) else item, -1) for item in y_true), dtype="int64")
    if mapped.size != y_true.size:
        return np.full(y_true.shape[0], -1, dtype="int64")
    return mapped


def _prediction_stats(
    proba: np.ndarray,
    y_true: np.ndarray,
    class_labels: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict[str, object]:
    proba = np.asarray(proba)
    y_true = np.asarray(y_true).reshape(-1)
    if proba.ndim != 2 or len(proba) == 0 or len(y_true) != len(proba):
        return {
            "top1": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "mean_confidence": float("nan"),
            "pred_labels": np.array([], dtype="int64"),
            "confidences": np.array([], dtype="float32"),
            "correct": np.array([], dtype=bool),
            "reliability_rows": [],
        }

    pred_local = np.argmax(proba, axis=1).astype("int64")
    confidences = np.max(proba, axis=1).astype("float32")
    if class_labels is not None:
        labels = np.asarray(class_labels).reshape(-1)
        if labels.size == proba.shape[1]:
            pred_labels = labels[pred_local].astype("int64", copy=False)
        else:
            pred_labels = pred_local
    else:
        pred_labels = pred_local

    correct = pred_labels.reshape(-1) == y_true
    top1 = float(np.mean(correct))
    mean_conf = float(np.mean(confidences))

    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype="float64")
    ece = 0.0
    reliability_rows: list[dict[str, float]] = []
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        count = int(np.sum(mask))
        if count == 0:
            reliability_rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0,
                    "mean_confidence": float("nan"),
                    "accuracy": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        bucket_conf = float(np.mean(confidences[mask]))
        bucket_acc = float(np.mean(correct[mask]))
        gap = abs(bucket_acc - bucket_conf)
        ece += (count / len(confidences)) * gap
        reliability_rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_confidence": bucket_conf,
                "accuracy": bucket_acc,
                "gap": float(gap),
            }
        )

    y_local = _encode_labels_to_local_indices(y_true, class_labels)
    known = y_local >= 0
    if not np.any(known):
        brier = float("nan")
    else:
        proba_known = proba[known]
        y_known = y_local[known]
        one_hot = np.zeros_like(proba_known, dtype="float64")
        one_hot[np.arange(len(y_known)), y_known] = 1.0
        brier = float(np.mean(np.sum(np.square(proba_known - one_hot), axis=1)))

    return {
        "top1": top1,
        "ece": float(ece),
        "brier": brier,
        "mean_confidence": mean_conf,
        "pred_labels": pred_labels,
        "confidences": confidences,
        "correct": correct,
        "reliability_rows": reliability_rows,
    }


def _plot_reliability(val_rows: list[dict[str, float]], test_rows: list[dict[str, float]], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")

    def _extract(rows: list[dict[str, float]]) -> tuple[list[float], list[float]]:
        x: list[float] = []
        y: list[float] = []
        for row in rows:
            conf = _to_float(row.get("mean_confidence"))
            acc = _to_float(row.get("accuracy"))
            if math.isnan(conf) or math.isnan(acc):
                continue
            x.append(conf)
            y.append(acc)
        return x, y

    val_x, val_y = _extract(val_rows)
    test_x, test_y = _extract(test_rows)
    if val_x:
        ax.plot(val_x, val_y, marker="o", label="val")
    if test_x:
        ax.plot(test_x, test_y, marker="o", label="test")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_confidence_hist(correct: np.ndarray, confidences: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    correct_conf = confidences[correct]
    wrong_conf = confidences[~correct]
    if len(correct_conf):
        ax.hist(correct_conf, bins=20, alpha=0.6, label="correct")
    if len(wrong_conf):
        ax.hist(wrong_conf, bins=20, alpha=0.6, label="incorrect")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _segment_bucket_frames(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    hour_series = pd.to_numeric(
        frame.get("hour", pd.Series(np.zeros(len(frame), dtype="float32"))),
        errors="coerce",
    ).fillna(0.0)
    dow_series = pd.to_numeric(
        frame.get("dayofweek", pd.Series(np.zeros(len(frame), dtype="float32"))),
        errors="coerce",
    ).fillna(0.0)
    session_series = pd.to_numeric(
        frame.get("session_position", pd.Series(np.zeros(len(frame), dtype="float32"))),
        errors="coerce",
    ).fillna(0.0)
    repeat_series = pd.to_numeric(
        frame.get("is_artist_repeat_from_prev", pd.Series(np.zeros(len(frame), dtype="float32"))),
        errors="coerce",
    ).fillna(0.0)
    skipped_series = pd.to_numeric(
        frame.get("skipped", pd.Series(np.zeros(len(frame), dtype="float32"))),
        errors="coerce",
    ).fillna(0.0)

    hour = hour_series.to_numpy(dtype="int32", copy=False)
    dayofweek = dow_series.to_numpy(dtype="int32", copy=False)
    session_pos = session_series.to_numpy(dtype="float32", copy=False)
    repeat_prev = repeat_series.to_numpy(dtype="int8", copy=False)
    skipped = skipped_series.to_numpy(dtype="int8", copy=False)

    hour_bucket = np.full(len(frame), "night", dtype=object)
    hour_bucket[(hour >= 5) & (hour < 12)] = "morning"
    hour_bucket[(hour >= 12) & (hour < 17)] = "afternoon"
    hour_bucket[(hour >= 17) & (hour < 22)] = "evening"

    session_bucket = np.full(len(frame), "late", dtype=object)
    session_bucket[session_pos < 3] = "early"
    session_bucket[(session_pos >= 3) & (session_pos < 10)] = "mid"

    return {
        "hour_bucket": hour_bucket,
        "is_weekend": np.where(dayofweek >= 5, "weekend", "weekday"),
        "session_phase": session_bucket,
        "repeat_from_prev": np.where(repeat_prev > 0, "repeat", "new"),
        "skip_flag": np.where(skipped > 0, "skip", "no_skip"),
    }


def _segment_rows(
    *,
    split: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
) -> list[dict[str, object]]:
    n = min(len(frame), len(y_true), len(pred_labels), len(confidences))
    if n <= 0:
        return []
    frame = frame.iloc[:n].reset_index(drop=True)
    y_true = np.asarray(y_true).reshape(-1)[:n]
    pred_labels = np.asarray(pred_labels).reshape(-1)[:n]
    confidences = np.asarray(confidences).reshape(-1)[:n]

    buckets = _segment_bucket_frames(frame)
    rows: list[dict[str, object]] = []
    for segment_name, values in buckets.items():
        unique_values = sorted({str(item) for item in values.tolist()})
        for bucket in unique_values:
            mask = values == bucket
            count = int(np.sum(mask))
            if count <= 0:
                continue
            top1 = float(np.mean(pred_labels[mask] == y_true[mask]))
            mean_conf = float(np.mean(confidences[mask]))
            rows.append(
                {
                    "split": split,
                    "segment": segment_name,
                    "bucket": bucket,
                    "count": count,
                    "top1": top1,
                    "mean_confidence": mean_conf,
                }
            )
    return rows


def _top_error_rows(
    y_true: np.ndarray,
    pred_labels: np.ndarray,
    label_lookup: dict[int, str],
    top_n: int = 25,
) -> list[dict[str, object]]:
    n = min(len(y_true), len(pred_labels))
    if n <= 0:
        return []
    y_true = np.asarray(y_true).reshape(-1)[:n]
    pred_labels = np.asarray(pred_labels).reshape(-1)[:n]

    mismatched = pred_labels != y_true
    if not np.any(mismatched):
        return []

    keys, counts = np.unique(
        np.stack([y_true[mismatched], pred_labels[mismatched]], axis=1),
        axis=0,
        return_counts=True,
    )
    pairs = sorted(zip(keys.tolist(), counts.tolist()), key=lambda item: int(item[1]), reverse=True)
    rows: list[dict[str, object]] = []
    for pair, count in pairs[:top_n]:
        true_id = int(pair[0])
        pred_id = int(pair[1])
        rows.append(
            {
                "true_label": true_id,
                "true_artist": label_lookup.get(true_id, str(true_id)),
                "pred_label": pred_id,
                "pred_artist": label_lookup.get(pred_id, str(pred_id)),
                "count": int(count),
            }
        )
    return rows


def _build_split_frames(data: PreparedData, sequence_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = len(data.X_seq_train)
    n_val = len(data.X_seq_val)
    n_test = len(data.X_seq_test)
    n_total = n_train + n_val + n_test

    ordered = data.df.sort_values("ts").reset_index(drop=True)
    aligned = ordered.iloc[sequence_length : sequence_length + n_total].reset_index(drop=True)
    if len(aligned) < n_total:
        # Defensive fallback if upstream indexing assumptions change.
        pad = n_total - len(aligned)
        aligned = pd.concat([aligned, pd.DataFrame(index=np.arange(pad))], ignore_index=True)

    val_start = n_train
    test_start = n_train + n_val
    val_frame = aligned.iloc[val_start:test_start].reset_index(drop=True)
    test_frame = aligned.iloc[test_start : test_start + n_test].reset_index(drop=True)
    return val_frame, test_frame


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_prediction_diagnostics(
    *,
    tag: str,
    val_proba: np.ndarray,
    val_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    label_lookup: dict[int, str],
    class_labels: np.ndarray | None = None,
    enable_conformal: bool = True,
    conformal_alpha: float = 0.10,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = _slugify(tag)

    val_stats = _prediction_stats(val_proba, val_y, class_labels=class_labels)
    test_stats = _prediction_stats(test_proba, test_y, class_labels=class_labels)

    reliability_rows: list[dict[str, object]] = []
    for split, rows in (("val", val_stats["reliability_rows"]), ("test", test_stats["reliability_rows"])):
        for row in rows:
            payload = dict(row)
            payload["split"] = split
            reliability_rows.append(payload)

    segment_rows = _segment_rows(
        split="val",
        frame=val_frame,
        y_true=np.asarray(val_y).reshape(-1),
        pred_labels=np.asarray(val_stats["pred_labels"]).reshape(-1),
        confidences=np.asarray(val_stats["confidences"]).reshape(-1),
    )
    segment_rows.extend(
        _segment_rows(
            split="test",
            frame=test_frame,
            y_true=np.asarray(test_y).reshape(-1),
            pred_labels=np.asarray(test_stats["pred_labels"]).reshape(-1),
            confidences=np.asarray(test_stats["confidences"]).reshape(-1),
        )
    )

    top_errors = _top_error_rows(
        y_true=np.asarray(test_y).reshape(-1),
        pred_labels=np.asarray(test_stats["pred_labels"]).reshape(-1),
        label_lookup=label_lookup,
        top_n=25,
    )

    summary_payload = {
        "tag": safe_tag,
        "val_top1": _to_float(val_stats["top1"]),
        "test_top1": _to_float(test_stats["top1"]),
        "val_ece": _to_float(val_stats["ece"]),
        "test_ece": _to_float(test_stats["ece"]),
        "val_brier": _to_float(val_stats["brier"]),
        "test_brier": _to_float(test_stats["brier"]),
        "val_mean_confidence": _to_float(val_stats["mean_confidence"]),
        "test_mean_confidence": _to_float(test_stats["mean_confidence"]),
        "conformal_enabled": bool(enable_conformal),
    }

    summary_path = output_dir / f"{safe_tag}_confidence_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    artifacts = [summary_path]

    if enable_conformal:
        val_y_local = _encode_labels_to_local_indices(np.asarray(val_y), class_labels)
        test_y_local = _encode_labels_to_local_indices(np.asarray(test_y), class_labels)
        calibration = fit_split_conformal_classifier(val_proba, val_y_local, alpha=conformal_alpha)
        if calibration is not None:
            val_conformal = summarize_prediction_sets(val_proba, val_y_local, calibration=calibration)
            test_conformal = summarize_prediction_sets(test_proba, test_y_local, calibration=calibration)
            conformal_payload = {
                "tag": safe_tag,
                "calibration": calibration.to_dict(),
                "val": val_conformal,
                "test": test_conformal,
            }
            conformal_summary_path = output_dir / f"{safe_tag}_conformal_summary.json"
            conformal_summary_path.write_text(json.dumps(conformal_payload, indent=2), encoding="utf-8")
            artifacts.append(conformal_summary_path)
            summary_payload.update(
                {
                    "conformal_threshold": float(calibration.threshold),
                    "conformal_alpha": float(calibration.alpha),
                    "val_conformal_coverage": _to_float(val_conformal["coverage"]),
                    "test_conformal_coverage": _to_float(test_conformal["coverage"]),
                    "val_abstention_rate": _to_float(val_conformal["abstention_rate"]),
                    "test_abstention_rate": _to_float(test_conformal["abstention_rate"]),
                    "val_accepted_rate": _to_float(val_conformal.get("accepted_rate")),
                    "test_accepted_rate": _to_float(test_conformal.get("accepted_rate")),
                    "val_selective_accuracy": _to_float(val_conformal.get("selective_accuracy")),
                    "test_selective_accuracy": _to_float(test_conformal.get("selective_accuracy")),
                    "val_selective_risk": _to_float(val_conformal.get("selective_risk")),
                    "test_selective_risk": _to_float(test_conformal.get("selective_risk")),
                }
            )
            summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    reliability_csv = output_dir / f"{safe_tag}_reliability.csv"
    _write_csv(
        reliability_csv,
        reliability_rows,
        fieldnames=["split", "bin_left", "bin_right", "count", "mean_confidence", "accuracy", "gap"],
    )

    segment_csv = output_dir / f"{safe_tag}_segment_metrics.csv"
    _write_csv(
        segment_csv,
        segment_rows,
        fieldnames=["split", "segment", "bucket", "count", "top1", "mean_confidence"],
    )

    top_errors_csv = output_dir / f"{safe_tag}_top_errors.csv"
    _write_csv(
        top_errors_csv,
        top_errors,
        fieldnames=["true_label", "true_artist", "pred_label", "pred_artist", "count"],
    )

    reliability_plot = output_dir / f"{safe_tag}_reliability.png"
    _plot_reliability(
        val_rows=val_stats["reliability_rows"],
        test_rows=test_stats["reliability_rows"],
        output_path=reliability_plot,
        title=f"{tag} Reliability",
    )

    confidence_hist = output_dir / f"{safe_tag}_confidence_hist.png"
    _plot_confidence_hist(
        correct=np.asarray(test_stats["correct"]).reshape(-1),
        confidences=np.asarray(test_stats["confidences"]).reshape(-1),
        output_path=confidence_hist,
        title=f"{tag} Confidence Distribution (Test)",
    )

    artifacts.extend([reliability_csv, segment_csv, top_errors_csv, reliability_plot, confidence_hist])
    return artifacts


def _extract_artist_proba(prediction) -> np.ndarray:
    if isinstance(prediction, dict):
        if "artist_output" in prediction:
            return np.asarray(prediction["artist_output"])
        first_value = next(iter(prediction.values()))
        return np.asarray(first_value)
    if isinstance(prediction, (list, tuple)):
        return np.asarray(prediction[0])
    return np.asarray(prediction)


def run_extended_evaluation(
    *,
    data: PreparedData,
    results: list[dict[str, object]],
    sequence_length: int,
    run_dir: Path,
    random_seed: int,
    max_train_samples: int,
    enable_conformal: bool,
    conformal_alpha: float,
    logger,
    feature_bundle: ClassicalFeatureBundle | None = None,
) -> list[Path]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    label_lookup = _build_label_lookup(data.df)
    val_frame, test_frame = _build_split_frames(data, sequence_length=sequence_length)

    deep_rows = [row for row in results if str(row.get("model_type", "")).strip() == "deep"]
    if deep_rows:
        best_deep = max(deep_rows, key=lambda row: _to_float(row.get("val_top1")))
        deep_name = str(best_deep.get("model_name", "")).strip()
        if deep_name:
            prediction_bundle_raw = str(best_deep.get("prediction_bundle_path", "")).strip()
            prediction_bundle_path = Path(prediction_bundle_raw) if prediction_bundle_raw else None
            try:
                if prediction_bundle_path is not None and prediction_bundle_path.exists():
                    val_proba, test_proba = load_prediction_bundle(prediction_bundle_path)
                else:
                    model_path = run_dir / f"best_{deep_name}.keras"
                    if not model_path.exists():
                        raise FileNotFoundError(f"Deep model checkpoint not found: {model_path}")
                    import tensorflow as tf

                    model = tf.keras.models.load_model(model_path, compile=False)
                    val_pred = model.predict((data.X_seq_val, data.X_ctx_val), verbose=0)
                    test_pred = model.predict((data.X_seq_test, data.X_ctx_test), verbose=0)
                    val_proba = _extract_artist_proba(val_pred)
                    test_proba = _extract_artist_proba(test_pred)
                try:
                    artifacts.extend(
                        _save_prediction_diagnostics(
                            tag=f"deep_{deep_name}",
                            val_proba=val_proba,
                            val_y=data.y_val.astype("int64"),
                            test_proba=test_proba,
                            test_y=data.y_test.astype("int64"),
                            val_frame=val_frame,
                            test_frame=test_frame,
                            output_dir=analysis_dir,
                            label_lookup=label_lookup,
                            enable_conformal=enable_conformal,
                            conformal_alpha=conformal_alpha,
                        )
                    )
                    logger.info("Saved extended diagnostics for deep model: %s", deep_name)
                except Exception as exc:
                    logger.warning("Deep diagnostics skipped for %s due to error: %s", deep_name, exc)
            except Exception as exc:
                logger.warning("Deep diagnostics skipped for %s due to error: %s", deep_name, exc)

    classical_rows = [
        row
        for row in results
        if str(row.get("model_type", "")).strip() in ("classical", "classical_tuned")
    ]
    if classical_rows:
        best_classical = max(classical_rows, key=lambda row: _to_float(row.get("val_top1")))
        row_type = str(best_classical.get("model_type", "")).strip()
        model_name = str(best_classical.get("model_name", "")).strip()
        base_name = (
            str(best_classical.get("base_model_name", "")).strip() if row_type == "classical_tuned" else model_name
        )
        best_params = best_classical.get("best_params", {})
        params = best_params if isinstance(best_params, dict) and row_type == "classical_tuned" else None

        if base_name:
            prediction_bundle_raw = str(best_classical.get("prediction_bundle_path", "")).strip()
            prediction_bundle_path = Path(prediction_bundle_raw) if prediction_bundle_raw else None
            try:
                if prediction_bundle_path is not None and prediction_bundle_path.exists():
                    val_proba, test_proba = load_prediction_bundle(prediction_bundle_path)
                    artifacts.extend(
                        _save_prediction_diagnostics(
                            tag=f"classical_{model_name}",
                            val_proba=val_proba,
                            val_y=data.y_val.astype("int64"),
                            test_proba=test_proba,
                            test_y=data.y_test.astype("int64"),
                            val_frame=val_frame,
                            test_frame=test_frame,
                            output_dir=analysis_dir,
                            label_lookup=label_lookup,
                            class_labels=None,
                            enable_conformal=enable_conformal,
                            conformal_alpha=conformal_alpha,
                        )
                    )
                    logger.info("Saved extended diagnostics for classical model: %s", model_name)
                else:
                    bundle = feature_bundle if feature_bundle is not None else build_classical_feature_bundle(data)
                    X_train, X_val, X_test = bundle.X_train, bundle.X_val, bundle.X_test
                    y_train, y_val, y_test = bundle.y_train, bundle.y_val, bundle.y_test
                    rng = np.random.default_rng(random_seed)
                    X_fit, y_fit = sample_rows(X_train, y_train, max_train_samples, rng)
                    _, estimator = build_classical_estimator(
                        base_name,
                        random_seed,
                        params=params,
                        estimator_n_jobs=-1,
                    )
                    estimator.fit(X_fit, y_fit)
                    if not hasattr(estimator, "predict_proba"):
                        logger.info("Skipping classical diagnostics for %s: estimator has no predict_proba.", model_name)
                    else:
                        val_proba = np.asarray(estimator.predict_proba(X_val))
                        test_proba = np.asarray(estimator.predict_proba(X_test))
                        classes = np.asarray(getattr(estimator, "classes_", []))
                        artifacts.extend(
                            _save_prediction_diagnostics(
                                tag=f"classical_{model_name}",
                                val_proba=val_proba,
                                val_y=y_val,
                                test_proba=test_proba,
                                test_y=y_test,
                                val_frame=val_frame,
                                test_frame=test_frame,
                                output_dir=analysis_dir,
                                label_lookup=label_lookup,
                                class_labels=(classes if classes.size else None),
                                enable_conformal=enable_conformal,
                                conformal_alpha=conformal_alpha,
                            )
                        )
                        logger.info("Saved extended diagnostics for classical model: %s", model_name)
            except Exception as exc:
                logger.warning("Classical diagnostics skipped for %s due to error: %s", model_name, exc)

    retrieval_rows = [
        row
        for row in results
        if str(row.get("model_type", "")).strip() in ("retrieval", "retrieval_reranker")
    ]
    for row in retrieval_rows:
        model_name = str(row.get("model_name", "")).strip()
        model_type = str(row.get("model_type", "")).strip()
        prediction_bundle_raw = str(row.get("prediction_bundle_path", "")).strip()
        prediction_bundle_path = Path(prediction_bundle_raw) if prediction_bundle_raw else None
        if not model_name or prediction_bundle_path is None or not prediction_bundle_path.exists():
            continue
        try:
            val_proba, test_proba = load_prediction_bundle(prediction_bundle_path)
            artifacts.extend(
                _save_prediction_diagnostics(
                    tag=f"{model_type}_{model_name}",
                    val_proba=val_proba,
                    val_y=data.y_val.astype("int64"),
                    test_proba=test_proba,
                    test_y=data.y_test.astype("int64"),
                    val_frame=val_frame,
                    test_frame=test_frame,
                    output_dir=analysis_dir,
                    label_lookup=label_lookup,
                    class_labels=None,
                    enable_conformal=enable_conformal,
                    conformal_alpha=conformal_alpha,
                )
            )
            logger.info("Saved extended diagnostics for retrieval model: %s", model_name)
        except Exception as exc:
            logger.warning("Retrieval diagnostics skipped for %s due to error: %s", model_name, exc)

    ensemble_rows = [row for row in results if str(row.get("model_type", "")).strip() == "ensemble"]
    if ensemble_rows:
        best_ensemble = max(ensemble_rows, key=lambda row: _to_float(row.get("val_top1")))
        ensemble_name = str(best_ensemble.get("model_name", "")).strip()
        prediction_bundle_raw = str(best_ensemble.get("prediction_bundle_path", "")).strip()
        prediction_bundle_path = Path(prediction_bundle_raw) if prediction_bundle_raw else None
        if ensemble_name and prediction_bundle_path is not None and prediction_bundle_path.exists():
            try:
                val_proba, test_proba = load_prediction_bundle(prediction_bundle_path)
                artifacts.extend(
                    _save_prediction_diagnostics(
                        tag=f"ensemble_{ensemble_name}",
                        val_proba=val_proba,
                        val_y=data.y_val.astype("int64"),
                        test_proba=test_proba,
                        test_y=data.y_test.astype("int64"),
                        val_frame=val_frame,
                        test_frame=test_frame,
                        output_dir=analysis_dir,
                        label_lookup=label_lookup,
                        class_labels=None,
                        enable_conformal=enable_conformal,
                        conformal_alpha=conformal_alpha,
                    )
                )
                logger.info("Saved extended diagnostics for ensemble model: %s", ensemble_name)
            except Exception as exc:
                logger.warning("Ensemble diagnostics skipped for %s due to error: %s", ensemble_name, exc)

    return artifacts
