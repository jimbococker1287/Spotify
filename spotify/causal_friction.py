from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import build_serving_tabular_features
from .data import PreparedData


@dataclass(frozen=True)
class CausalSkipDecompositionArtifact:
    context_features: list[str]
    friction_feature_indices: list[int]
    preference_estimator: object
    friction_estimator: object
    meta_estimator: object


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_arr = np.asarray(y_true).reshape(-1)
    score_arr = np.asarray(scores).reshape(-1)
    if y_arr.size == 0 or y_arr.size != score_arr.size or np.unique(y_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, score_arr))


def _fit_estimator(X: np.ndarray, y: np.ndarray, random_seed: int) -> object:
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=random_seed,
        ),
    )
    estimator.fit(X, y)
    return estimator


def _decision_scores(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype="float32").reshape(-1)
    proba = np.asarray(estimator.predict_proba(X), dtype="float32")
    clipped = np.clip(proba[:, 1], 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype("float32", copy=False)


def _feature_indices(context_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    friction_idx = np.asarray(
        [idx for idx, name in enumerate(context_features) if str(name).startswith("tech_") or str(name) == "offline"],
        dtype="int64",
    )
    non_friction_idx = np.asarray([idx for idx in range(len(context_features)) if idx not in set(friction_idx.tolist())], dtype="int64")
    return non_friction_idx, friction_idx


def fit_causal_skip_decomposition(
    *,
    data: PreparedData,
    output_dir: Path,
    random_seed: int,
    logger,
) -> tuple[CausalSkipDecompositionArtifact, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train = build_serving_tabular_features(data.X_seq_train, data.X_ctx_train)
    full_val = build_serving_tabular_features(data.X_seq_val, data.X_ctx_val)
    full_test = build_serving_tabular_features(data.X_seq_test, data.X_ctx_test)

    non_friction_idx, friction_idx = _feature_indices(list(data.context_features))
    seq_feature_count = full_train.shape[1] - data.X_ctx_train.shape[1]
    preference_keep = np.concatenate([np.arange(seq_feature_count, dtype="int64"), seq_feature_count + non_friction_idx], axis=0)
    friction_keep = seq_feature_count + friction_idx

    if friction_keep.size == 0:
        friction_keep = np.asarray([seq_feature_count], dtype="int64")
        full_train = np.pad(full_train, ((0, 0), (0, 1)), constant_values=0.0)
        full_val = np.pad(full_val, ((0, 0), (0, 1)), constant_values=0.0)
        full_test = np.pad(full_test, ((0, 0), (0, 1)), constant_values=0.0)

    friction_score_train = np.sum(np.maximum(data.X_ctx_train[:, friction_idx], 0.0), axis=1) if friction_idx.size else np.zeros(len(data.X_ctx_train), dtype="float32")
    low_friction_threshold = float(np.quantile(friction_score_train, 0.50)) if len(friction_score_train) else 0.0
    low_friction_mask = friction_score_train <= low_friction_threshold
    if int(np.sum(low_friction_mask)) < 4:
        low_friction_mask = np.ones(len(friction_score_train), dtype=bool)

    preference_estimator = _fit_estimator(full_train[low_friction_mask][:, preference_keep], data.y_skip_train[low_friction_mask], random_seed)
    friction_estimator = _fit_estimator(full_train[:, friction_keep], data.y_skip_train, random_seed + 1)

    pref_logit_train = _decision_scores(preference_estimator, full_train[:, preference_keep])
    friction_logit_train = _decision_scores(friction_estimator, full_train[:, friction_keep])
    meta_features_train = np.column_stack([pref_logit_train, friction_logit_train]).astype("float32")
    meta_estimator = _fit_estimator(meta_features_train, data.y_skip_train, random_seed + 2)

    artifact = CausalSkipDecompositionArtifact(
        context_features=list(data.context_features),
        friction_feature_indices=friction_idx.astype("int32").tolist(),
        preference_estimator=preference_estimator,
        friction_estimator=friction_estimator,
        meta_estimator=meta_estimator,
    )

    def _decompose(split: str, X_full: np.ndarray, y_true: np.ndarray) -> tuple[list[dict[str, object]], dict[str, float]]:
        pref_logit = _decision_scores(preference_estimator, X_full[:, preference_keep])
        friction_logit = _decision_scores(friction_estimator, X_full[:, friction_keep])
        meta_features = np.column_stack([pref_logit, friction_logit]).astype("float32")
        total_risk = np.asarray(meta_estimator.predict_proba(meta_features), dtype="float32")[:, 1]
        meta_inner = meta_estimator.named_steps["logisticregression"]
        pref_only_features = np.column_stack([pref_logit, np.zeros_like(friction_logit)])
        pref_only_risk = np.asarray(meta_estimator.predict_proba(pref_only_features), dtype="float32")[:, 1]
        friction_only_features = np.column_stack([np.zeros_like(pref_logit), friction_logit])
        friction_only_risk = np.asarray(meta_estimator.predict_proba(friction_only_features), dtype="float32")[:, 1]
        rows = [
            {
                "split": split,
                "row_index": idx,
                "preference_skip_risk": float(pref_only_risk[idx]),
                "friction_skip_risk": float(friction_only_risk[idx]),
                "total_skip_risk": float(total_risk[idx]),
                "friction_uplift": float(total_risk[idx] - pref_only_risk[idx]),
                "observed_skip": int(y_true[idx]),
            }
            for idx in range(len(y_true))
        ]
        summary = {
            "auc_total": _safe_auc(y_true, total_risk),
            "auc_preference_only": _safe_auc(y_true, pref_only_risk),
            "mean_friction_uplift": float(np.mean(total_risk - pref_only_risk)),
        }
        return rows, summary

    val_rows, val_summary = _decompose("val", full_val, data.y_skip_val)
    test_rows, test_summary = _decompose("test", full_test, data.y_skip_test)
    decomposition_rows = val_rows + test_rows

    artifact_path = output_dir / "causal_skip_decomposition.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    summary_path = output_dir / "causal_skip_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "interpretation": "proxy_structural_decomposition_not_identified_causal_effect",
                "friction_feature_count": int(len(friction_idx)),
                "val": val_summary,
                "test": test_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    decomposition_path = _write_csv(
        output_dir / "causal_skip_decomposition.csv",
        decomposition_rows,
        ["split", "row_index", "preference_skip_risk", "friction_skip_risk", "total_skip_risk", "friction_uplift", "observed_skip"],
    )
    logger.info(
        "Built causal skip decomposition: val_auc=%.4f test_auc=%.4f uplift=%.4f",
        float(val_summary["auc_total"]),
        float(test_summary["auc_total"]),
        float(test_summary["mean_friction_uplift"]),
    )
    return artifact, [artifact_path, summary_path, decomposition_path]
