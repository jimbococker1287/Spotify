from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from .data import PreparedData
from .probability_bundles import load_prediction_bundle
from .ranking import topk_indices_2d


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _train_transition_matrix(data: PreparedData) -> np.ndarray:
    last_artist = np.asarray(data.X_seq_train[:, -1], dtype="int32")
    target = np.asarray(data.y_train, dtype="int32")
    matrix = np.ones((data.num_artists, data.num_artists), dtype="float32")
    np.add.at(matrix, (last_artist, target), 1.0)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix.astype("float32", copy=False)


def _novelty_weights(data: PreparedData) -> np.ndarray:
    counts = np.bincount(np.asarray(data.y_train, dtype="int32"), minlength=data.num_artists).astype("float32")
    counts += 1.0
    popularity = counts / np.sum(counts)
    return (1.0 - popularity).astype("float32", copy=False)


def _synthetic_utility(
    *,
    seq_batch: np.ndarray,
    transition_matrix: np.ndarray,
    novelty_weights: np.ndarray,
) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    last_artist = seq_arr[:, -1]
    transition_score = transition_matrix[last_artist]
    session_repeat = np.zeros_like(transition_score, dtype="float32")
    if seq_arr.size:
        row_idx = np.arange(seq_arr.shape[0], dtype="int32").reshape(-1, 1)
        session_repeat[row_idx, seq_arr] = 1.0
    novelty = np.broadcast_to(novelty_weights.reshape(1, -1), transition_score.shape)
    utility = (0.65 * transition_score) + (0.20 * session_repeat) + (0.15 * novelty)
    utility /= utility.sum(axis=1, keepdims=True)
    return utility.astype("float32", copy=False)


def _discounted_reward(topk_idx: np.ndarray, utility: np.ndarray) -> float:
    topk_arr = np.asarray(topk_idx, dtype="int32")
    utility_arr = np.asarray(utility, dtype="float32")
    if topk_arr.ndim != 2 or topk_arr.size == 0 or utility_arr.ndim != 2 or utility_arr.shape[0] != topk_arr.shape[0]:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(topk_arr.shape[1], dtype="float64") + 2.0)
    gathered = np.take_along_axis(utility_arr, topk_arr, axis=1)
    rewards = np.sum(np.asarray(gathered, dtype="float64") * discounts.reshape(1, -1), axis=1)
    return float(np.mean(rewards)) if rewards.size else float("nan")


def _hit_at_k(topk_idx: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.any(topk_idx == np.asarray(y_true).reshape(-1, 1), axis=1))) if len(topk_idx) else float("nan")


def run_policy_simulation(
    *,
    data: PreparedData,
    results: list[dict[str, object]],
    run_dir: Path,
    logger,
    k: int = 5,
) -> list[Path]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    transition_matrix = _train_transition_matrix(data)
    novelty = _novelty_weights(data)
    val_utility = _synthetic_utility(seq_batch=data.X_seq_val, transition_matrix=transition_matrix, novelty_weights=novelty)
    test_utility = _synthetic_utility(seq_batch=data.X_seq_test, transition_matrix=transition_matrix, novelty_weights=novelty)

    rows: list[dict[str, object]] = []
    for row in results:
        model_name = str(row.get("model_name", "")).strip()
        model_type = str(row.get("model_type", "")).strip()
        bundle_raw = str(row.get("prediction_bundle_path", "")).strip()
        if not model_name or not bundle_raw:
            continue
        bundle_path = Path(bundle_raw)
        if not bundle_path.exists():
            continue
        try:
            val_proba, test_proba = load_prediction_bundle(bundle_path)
        except Exception as exc:
            logger.warning("Policy simulation skipped for %s: %s", model_name, exc)
            continue
        kk = max(1, min(int(k), int(val_proba.shape[1]) if val_proba.ndim == 2 else int(k)))
        val_topk = topk_indices_2d(val_proba, kk)
        test_topk = topk_indices_2d(test_proba, kk)
        rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "val_hit_at_k": _hit_at_k(val_topk, data.y_val),
                "test_hit_at_k": _hit_at_k(test_topk, data.y_test),
                "val_discounted_reward": _discounted_reward(val_topk, val_utility),
                "test_discounted_reward": _discounted_reward(test_topk, test_utility),
                "val_expected_utility_mass": float(np.mean(np.take_along_axis(val_utility, val_topk, axis=1).sum(axis=1))),
                "test_expected_utility_mass": float(np.mean(np.take_along_axis(test_utility, test_topk, axis=1).sum(axis=1))),
            }
        )

    if not rows:
        return []

    rows.sort(key=lambda item: float(item["test_discounted_reward"]), reverse=True)
    csv_path = _write_csv(
        analysis_dir / "policy_simulation_summary.csv",
        [
            "model_name",
            "model_type",
            "val_hit_at_k",
            "test_hit_at_k",
            "val_discounted_reward",
            "test_discounted_reward",
            "val_expected_utility_mass",
            "test_expected_utility_mass",
        ],
        rows,
    )
    json_path = analysis_dir / "policy_simulation_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return [csv_path, json_path]
