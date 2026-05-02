from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PreparedData
from .run_artifacts import write_csv_rows


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_arr = np.asarray(y_true).reshape(-1)
    score_arr = np.asarray(y_score).reshape(-1)
    if y_arr.size == 0 or y_arr.size != score_arr.size:
        return float("nan")
    if np.unique(y_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, score_arr))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    write_csv_rows(path, rows, fieldnames=fieldnames)


def _healthy_reference_value(feature_name: str, values: np.ndarray) -> float:
    name = str(feature_name).strip().lower()
    arr = np.asarray(values, dtype="float32")
    if arr.size == 0:
        return 0.0
    if "bitrate" in name:
        return float(np.percentile(arr, 75))
    if "reachability_wifi" in name:
        return float(max(0.0, np.percentile(arr, 90)))
    if "offline" in name or "failure" in name or "error" in name or "stutter" in name or "downgrade" in name:
        return 0.0
    return 0.0


def _plot_coefficients(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    ordered = sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)[:15]
    names = [str(row["feature"]) for row in ordered][::-1]
    values = [float(row["coefficient"]) for row in ordered][::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d97706" if value >= 0 else "#2563eb" for value in values]
    ax.barh(names, values, color=colors)
    ax.set_title("Friction Proxy Coefficients")
    ax.set_xlabel("Standardized logistic coefficient")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_friction_proxy_analysis(
    *,
    data: PreparedData,
    output_dir: Path,
    logger,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    feature_names = list(data.context_features)
    friction_idx = [
        idx
        for idx, name in enumerate(feature_names)
        if str(name).startswith("tech_") or str(name) == "offline"
    ]
    if not friction_idx:
        summary_path = output_dir / "friction_proxy_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "skipped_no_friction_features",
                    "interpretation": "proxy_counterfactual_not_causal",
                    "friction_feature_count": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return [summary_path]

    all_idx = np.arange(len(feature_names), dtype="int64")
    baseline_idx = np.asarray([idx for idx in all_idx.tolist() if int(idx) not in set(friction_idx)], dtype="int64")

    X_train = np.asarray(data.X_ctx_train, dtype="float32")
    X_val = np.asarray(data.X_ctx_val, dtype="float32")
    X_test = np.asarray(data.X_ctx_test, dtype="float32")
    y_train = np.asarray(data.y_skip_train, dtype="int32")
    y_val = np.asarray(data.y_skip_val, dtype="int32")
    y_test = np.asarray(data.y_skip_test, dtype="int32")

    baseline_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced"))
    full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced"))

    if baseline_idx.size > 0:
        baseline_model.fit(X_train[:, baseline_idx], y_train)
        baseline_val = np.asarray(baseline_model.predict_proba(X_val[:, baseline_idx]), dtype="float32")[:, 1]
        baseline_test = np.asarray(baseline_model.predict_proba(X_test[:, baseline_idx]), dtype="float32")[:, 1]
        baseline_val_auc = _safe_auc(y_val, baseline_val)
        baseline_test_auc = _safe_auc(y_test, baseline_test)
    else:
        baseline_val_auc = float("nan")
        baseline_test_auc = float("nan")

    full_model.fit(X_train, y_train)
    full_val = np.asarray(full_model.predict_proba(X_val), dtype="float32")[:, 1]
    full_test = np.asarray(full_model.predict_proba(X_test), dtype="float32")[:, 1]

    healthy_reference = X_train.copy()
    for idx in friction_idx:
        healthy_reference[:, idx] = _healthy_reference_value(feature_names[idx], X_train[:, idx])

    X_val_counterfactual = X_val.copy()
    X_test_counterfactual = X_test.copy()
    for idx in friction_idx:
        target_value = _healthy_reference_value(feature_names[idx], healthy_reference[:, idx])
        X_val_counterfactual[:, idx] = target_value
        X_test_counterfactual[:, idx] = target_value

    cf_val = np.asarray(full_model.predict_proba(X_val_counterfactual), dtype="float32")[:, 1]
    cf_test = np.asarray(full_model.predict_proba(X_test_counterfactual), dtype="float32")[:, 1]

    scaler = full_model.named_steps["standardscaler"]
    estimator = full_model.named_steps["logisticregression"]
    scale = np.asarray(getattr(scaler, "scale_", np.ones(len(feature_names))), dtype="float32")
    scale[scale == 0] = 1.0
    raw_coef = np.asarray(estimator.coef_, dtype="float32").reshape(-1)
    standardized_coef = raw_coef / scale
    coefficient_rows = [
        {
            "feature": feature_names[idx],
            "coefficient": float(standardized_coef[idx]),
            "is_friction_feature": int(idx in friction_idx),
        }
        for idx in range(len(feature_names))
    ]
    coefficient_rows.sort(key=lambda row: abs(float(row["coefficient"])), reverse=True)

    top_friction_rows: list[dict[str, object]] = []
    for idx in friction_idx:
        isolated = X_val.copy()
        isolated[:, idx] = _healthy_reference_value(feature_names[idx], X_train[:, idx])
        delta = full_val - np.asarray(full_model.predict_proba(isolated), dtype="float32")[:, 1]
        top_friction_rows.append(
            {
                "feature": feature_names[idx],
                "mean_risk_delta": float(np.mean(delta)),
                "median_risk_delta": float(np.median(delta)),
            }
        )
    top_friction_rows.sort(key=lambda row: abs(float(row["mean_risk_delta"])), reverse=True)

    summary_path = output_dir / "friction_proxy_summary.json"
    coefficient_path = output_dir / "friction_feature_coefficients.csv"
    delta_path = output_dir / "friction_counterfactual_delta.csv"
    plot_path = output_dir / "friction_feature_coefficients.png"

    summary_payload = {
        "status": "ok",
        "interpretation": "proxy_counterfactual_not_causal",
        "friction_feature_count": int(len(friction_idx)),
        "friction_features": [feature_names[idx] for idx in friction_idx],
        "baseline_model": {
            "val_auc": float(baseline_val_auc),
            "test_auc": float(baseline_test_auc),
        },
        "full_model": {
            "val_auc": float(_safe_auc(y_val, full_val)),
            "test_auc": float(_safe_auc(y_test, full_test)),
        },
        "auc_lift": {
            "val": float(_safe_auc(y_val, full_val) - baseline_val_auc) if not np.isnan(baseline_val_auc) else float("nan"),
            "test": float(_safe_auc(y_test, full_test) - baseline_test_auc) if not np.isnan(baseline_test_auc) else float("nan"),
        },
        "proxy_counterfactual": {
            "val_mean_skip_risk": float(np.mean(full_val)),
            "val_mean_skip_risk_without_friction": float(np.mean(cf_val)),
            "val_mean_delta": float(np.mean(full_val - cf_val)),
            "test_mean_skip_risk": float(np.mean(full_test)),
            "test_mean_skip_risk_without_friction": float(np.mean(cf_test)),
            "test_mean_delta": float(np.mean(full_test - cf_test)),
        },
        "top_friction_features": top_friction_rows[:10],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    artifacts.append(summary_path)

    _write_csv(coefficient_path, coefficient_rows, ["feature", "coefficient", "is_friction_feature"])
    _write_csv(delta_path, top_friction_rows, ["feature", "mean_risk_delta", "median_risk_delta"])
    _plot_coefficients(coefficient_rows, plot_path)
    artifacts.extend([coefficient_path, delta_path, plot_path])

    logger.info(
        "Friction proxy analysis: val_auc=%.4f test_auc=%.4f mean_test_delta=%.4f",
        float(summary_payload["full_model"]["val_auc"]),
        float(summary_payload["full_model"]["test_auc"]),
        float(summary_payload["proxy_counterfactual"]["test_mean_delta"]),
    )
    return artifacts
