from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import math
import time

import numpy as np

from .data import PreparedData
from .evaluation import _build_label_lookup, _build_split_frames, _save_prediction_diagnostics
from .probability_bundles import load_prediction_bundle, save_prediction_bundle
from .ranking import ranking_metrics_from_proba


@dataclass(frozen=True)
class EnsembleBuildResult:
    row: dict[str, object]
    artifact_paths: list[Path]


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _topk_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if proba.ndim != 2 or len(proba) == 0:
        return float("nan")
    kk = max(1, min(int(k), int(proba.shape[1])))
    topk = np.argpartition(proba, -kk, axis=1)[:, -kk:]
    matches = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(matches))


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-3)
    clipped = np.clip(np.asarray(proba, dtype="float64"), 1e-9, 1.0)
    logits = np.log(clipped) / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (exp / denom).astype("float32")


def _negative_log_likelihood(proba: np.ndarray, y_true: np.ndarray) -> float:
    if proba.ndim != 2 or len(proba) == 0:
        return float("inf")
    idx = np.arange(len(y_true))
    clipped = np.clip(proba[idx, y_true.astype("int64")], 1e-9, 1.0)
    return float(-np.mean(np.log(clipped)))


def _fit_temperature(val_proba: np.ndarray, y_val: np.ndarray) -> float:
    temperatures = np.concatenate(
        [
            np.array([0.65, 0.8, 1.0, 1.25, 1.6], dtype="float64"),
            np.geomspace(0.5, 3.0, num=15, dtype="float64"),
        ]
    )
    best_temp = 1.0
    best_nll = float("inf")
    for temp in sorted({round(float(item), 6) for item in temperatures.tolist()}):
        calibrated = _apply_temperature(val_proba, temp)
        nll = _negative_log_likelihood(calibrated, y_val)
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return best_temp


def _blend_probabilities(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not probabilities:
        return np.empty((0, 0), dtype="float32")
    blend = np.zeros_like(probabilities[0], dtype="float64")
    for weight, proba in zip(weights.tolist(), probabilities):
        blend += float(weight) * np.asarray(proba, dtype="float64")
    row_sums = blend.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return (blend / row_sums).astype("float32")


def _base_model_key(row: dict[str, object]) -> str:
    base_name = str(row.get("base_model_name", "")).strip()
    return base_name or str(row.get("model_name", "")).strip()


def _select_candidate_rows(results: list[dict[str, object]], max_candidates: int = 4) -> list[dict[str, object]]:
    rows = [
        row
        for row in results
        if str(row.get("model_type", "")).strip() in ("deep", "classical", "classical_tuned")
        and str(row.get("prediction_bundle_path", "")).strip()
        and Path(str(row.get("prediction_bundle_path", "")).strip()).exists()
    ]
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("val_top1")), reverse=True)

    selected: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for target_type in ("classical_tuned", "classical", "deep"):
        for row in ordered:
            row_type = str(row.get("model_type", "")).strip()
            key = _base_model_key(row)
            if row_type != target_type or not key or key in seen_keys:
                continue
            selected.append(row)
            seen_keys.add(key)
            break

    for row in ordered:
        if len(selected) >= max_candidates:
            break
        key = _base_model_key(row)
        if not key or key in seen_keys:
            continue
        selected.append(row)
        seen_keys.add(key)

    return selected[:max_candidates]


def build_probability_ensemble(
    *,
    data: PreparedData,
    results: list[dict[str, object]],
    sequence_length: int,
    run_dir: Path,
    logger,
) -> EnsembleBuildResult | None:
    candidates = _select_candidate_rows(results)
    if len(candidates) < 2:
        logger.info("Skipping ensemble build: need at least two bundle-backed models, found %d.", len(candidates))
        return None

    started = time.perf_counter()
    candidate_payloads: list[dict[str, object]] = []
    for row in candidates:
        bundle_path = Path(str(row.get("prediction_bundle_path", "")).strip())
        val_proba, test_proba = load_prediction_bundle(bundle_path)
        if len(val_proba) != len(data.y_val) or len(test_proba) != len(data.y_test):
            logger.info(
                "Skipping ensemble candidate %s due to incompatible bundle shapes: val=%d/%d test=%d/%d",
                row.get("model_name", ""),
                len(val_proba),
                len(data.y_val),
                len(test_proba),
                len(data.y_test),
            )
            continue
        candidate_payloads.append(
            {
                "row": row,
                "val_proba": val_proba,
                "test_proba": test_proba,
            }
        )

    if len(candidate_payloads) < 2:
        logger.info("Skipping ensemble build: only %d candidates have compatible full holdout bundles.", len(candidate_payloads))
        return None

    best_choice: dict[str, object] | None = None
    score_options = (0.0, 1.0, 2.5, 4.0)
    y_val = data.y_val.astype("int64")
    y_test = data.y_test.astype("int64")
    num_items = int(data.num_artists)

    for size in range(2, len(candidate_payloads) + 1):
        for combo in combinations(candidate_payloads, size):
            raw_scores = np.array(
                [max(1e-6, _safe_float(item["row"].get("val_top1"))) for item in combo],
                dtype="float64",
            )
            centered = raw_scores - float(np.mean(raw_scores))
            for alpha in score_options:
                if alpha <= 0.0:
                    weights = np.full(len(combo), 1.0 / len(combo), dtype="float64")
                else:
                    logits = alpha * centered
                    logits -= logits.max()
                    weights = np.exp(logits)
                    weights /= np.sum(weights)

                val_blend = _blend_probabilities([item["val_proba"] for item in combo], weights)
                temperature = _fit_temperature(val_blend, y_val)
                val_calibrated = _apply_temperature(val_blend, temperature)
                val_top1 = float(np.mean(np.argmax(val_calibrated, axis=1) == y_val))
                val_top5 = _topk_accuracy(val_calibrated, y_val, k=5)
                val_ranking = ranking_metrics_from_proba(val_calibrated, y_val, num_items=num_items, k=5)
                nll = _negative_log_likelihood(val_calibrated, y_val)

                choice = {
                    "combo": combo,
                    "weights": weights.astype("float32"),
                    "alpha": float(alpha),
                    "temperature": float(temperature),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_ndcg_at5": float(val_ranking["ndcg_at_k"]),
                    "val_mrr_at5": float(val_ranking["mrr_at_k"]),
                    "val_coverage_at5": float(val_ranking["coverage_at_k"]),
                    "val_diversity_at5": float(val_ranking["diversity_at_k"]),
                    "val_nll": nll,
                }
                if best_choice is None:
                    best_choice = choice
                    continue
                incumbent = (
                    _safe_float(best_choice["val_top1"]),
                    _safe_float(best_choice["val_ndcg_at5"]),
                    -_safe_float(best_choice["val_nll"]),
                )
                challenger = (
                    val_top1,
                    float(val_ranking["ndcg_at_k"]),
                    -nll,
                )
                if challenger > incumbent:
                    best_choice = choice

    if best_choice is None:
        logger.info("Skipping ensemble build: no viable blend combinations were produced.")
        return None

    selected_combo = list(best_choice["combo"])
    weights = np.asarray(best_choice["weights"], dtype="float32")
    temperature = float(best_choice["temperature"])
    selected_names = [str(item["row"].get("model_name", "")).strip() for item in selected_combo]

    test_blend = _blend_probabilities([item["test_proba"] for item in selected_combo], weights)
    test_calibrated = _apply_temperature(test_blend, temperature)
    test_top1 = float(np.mean(np.argmax(test_calibrated, axis=1) == y_test))
    test_top5 = _topk_accuracy(test_calibrated, y_test, k=5)
    test_ranking = ranking_metrics_from_proba(test_calibrated, y_test, num_items=num_items, k=5)

    prediction_dir = run_dir / "prediction_bundles"
    bundle_path = save_prediction_bundle(
        prediction_dir / "ensemble_blended_ensemble.npz",
        val_proba=_apply_temperature(_blend_probabilities([item["val_proba"] for item in selected_combo], weights), temperature),
        test_proba=test_calibrated,
    )

    analysis_dir = run_dir / "analysis"
    label_lookup = _build_label_lookup(data.df)
    val_frame, test_frame = _build_split_frames(data, sequence_length=sequence_length)
    diagnostics = _save_prediction_diagnostics(
        tag="ensemble_blended_ensemble",
        val_proba=_apply_temperature(_blend_probabilities([item["val_proba"] for item in selected_combo], weights), temperature),
        val_y=y_val,
        test_proba=test_calibrated,
        test_y=y_test,
        val_frame=val_frame,
        test_frame=test_frame,
        output_dir=analysis_dir,
        label_lookup=label_lookup,
        class_labels=None,
    )

    summary = {
        "selected_models": selected_names,
        "weights": {name: float(weight) for name, weight in zip(selected_names, weights.tolist())},
        "weight_search_alpha": float(best_choice["alpha"]),
        "calibration_temperature": temperature,
        "prediction_bundle_path": str(bundle_path),
    }
    summary.update(
        {
            "val_top1": float(best_choice["val_top1"]),
            "val_top5": float(best_choice["val_top5"]),
            "val_ndcg_at5": float(best_choice["val_ndcg_at5"]),
            "val_mrr_at5": float(best_choice["val_mrr_at5"]),
            "val_coverage_at5": float(best_choice["val_coverage_at5"]),
            "val_diversity_at5": float(best_choice["val_diversity_at5"]),
            "test_top1": test_top1,
            "test_top5": test_top5,
            "test_ndcg_at5": float(test_ranking["ndcg_at_k"]),
            "test_mrr_at5": float(test_ranking["mrr_at_k"]),
            "test_coverage_at5": float(test_ranking["coverage_at_k"]),
            "test_diversity_at5": float(test_ranking["diversity_at_k"]),
        }
    )
    summary_path = analysis_dir / "ensemble_blended_ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    elapsed = float(time.perf_counter() - started)
    row = {
        "model_name": "blended_ensemble",
        "model_type": "ensemble",
        "model_family": "blend",
        "val_top1": float(best_choice["val_top1"]),
        "val_top5": float(best_choice["val_top5"]),
        "val_ndcg_at5": float(best_choice["val_ndcg_at5"]),
        "val_mrr_at5": float(best_choice["val_mrr_at5"]),
        "val_coverage_at5": float(best_choice["val_coverage_at5"]),
        "val_diversity_at5": float(best_choice["val_diversity_at5"]),
        "test_top1": test_top1,
        "test_top5": test_top5,
        "test_ndcg_at5": float(test_ranking["ndcg_at_k"]),
        "test_mrr_at5": float(test_ranking["mrr_at_k"]),
        "test_coverage_at5": float(test_ranking["coverage_at_k"]),
        "test_diversity_at5": float(test_ranking["diversity_at_k"]),
        "fit_seconds": elapsed,
        "epochs": "",
        "prediction_bundle_path": str(bundle_path),
        "ensemble_members": selected_names,
        "ensemble_weights": summary["weights"],
        "calibration_temperature": temperature,
    }
    artifact_paths = [bundle_path, summary_path, *diagnostics]
    logger.info(
        "Built blended ensemble from %s | val_top1=%.4f test_top1=%.4f temp=%.3f",
        ",".join(selected_names),
        float(row["val_top1"]),
        float(row["test_top1"]),
        temperature,
    )
    return EnsembleBuildResult(row=row, artifact_paths=artifact_paths)
