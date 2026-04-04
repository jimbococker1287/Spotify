from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re

import numpy as np

from .config import PipelineConfig
from .model_types import analysis_prefix_for_model_type
from .probability_bundles import load_prediction_bundle
from .recommender_safety import build_conformal_abstention_summary
from .run_artifacts import write_json
from .tracking import MlflowTracker


def _slugify(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "run"


def _build_run_id(config: PipelineConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _slugify(config.run_name) if config.run_name else config.profile
    return f"{stamp}_{suffix}"


def _track_file(tracker: MlflowTracker | None, path: Path) -> None:
    if tracker is not None and path.exists():
        tracker.log_artifact(path)


def _existing_path(raw_path: object) -> Path | None:
    text = str(raw_path or "").strip()
    return Path(text) if text else None


def _append_existing_artifact_path(artifact_paths: list[Path], raw_path: object) -> None:
    path = _existing_path(raw_path)
    if path is not None and path.exists():
        artifact_paths.append(path)


def _write_json_artifact(path: Path, payload: object, artifact_paths: list[Path] | None = None) -> Path:
    write_json(path, payload)
    if artifact_paths is not None:
        artifact_paths.append(path)
    return path


def _analysis_prefix_for_model_type(model_type: str) -> str | None:
    return analysis_prefix_for_model_type(model_type)


def _safe_json_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _update_json_artifact(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _existing_summary_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _risk_metrics_from_summary(summary_payload: dict[str, object]) -> dict[str, float]:
    return {
        "val_selective_risk": _safe_json_float(summary_payload.get("val_selective_risk", float("nan"))),
        "val_abstention_rate": _safe_json_float(summary_payload.get("val_abstention_rate", float("nan"))),
        "test_selective_risk": _safe_json_float(summary_payload.get("test_selective_risk", float("nan"))),
        "test_abstention_rate": _safe_json_float(summary_payload.get("test_abstention_rate", float("nan"))),
    }


def _risk_metrics_complete(metrics: dict[str, float]) -> bool:
    required = (
        metrics.get("val_selective_risk", float("nan")),
        metrics.get("val_abstention_rate", float("nan")),
        metrics.get("test_selective_risk", float("nan")),
        metrics.get("test_abstention_rate", float("nan")),
    )
    return all(not np.isnan(float(value)) for value in required)


def _backfill_risk_summary(
    *,
    run_dir: Path,
    row: dict[str, object],
    val_y: np.ndarray,
    test_y: np.ndarray,
    conformal_alpha: float,
) -> dict[str, float]:
    model_name = str(row.get("model_name", "")).strip()
    model_type = str(row.get("model_type", "")).strip().lower()
    prefix = _analysis_prefix_for_model_type(model_type)
    prediction_bundle_raw = str(row.get("prediction_bundle_path", "")).strip()
    prediction_bundle_path = Path(prediction_bundle_raw) if prediction_bundle_raw else None
    if not model_name or prefix is None or prediction_bundle_path is None or not prediction_bundle_path.exists():
        return {}

    try:
        val_proba, test_proba = load_prediction_bundle(prediction_bundle_path)
    except Exception:
        return {}

    conformal_payload = build_conformal_abstention_summary(
        tag=f"{prefix}_{model_name}",
        val_proba=val_proba,
        val_y=np.asarray(val_y, dtype="int64"),
        test_proba=test_proba,
        test_y=np.asarray(test_y, dtype="int64"),
        alpha=float(conformal_alpha),
        target_selective_risk=(0.45 if model_type in ("classical", "classical_tuned") else 0.50),
        min_accepted_rate=(0.70 if model_type in ("classical", "classical_tuned") else 0.10),
        env_prefix=("CLASSICAL" if model_type in ("classical", "classical_tuned") else None),
        enable_temperature_scaling=bool(model_type in ("classical", "classical_tuned")),
    )
    if not isinstance(conformal_payload, dict):
        return {}

    analysis_dir = run_dir / "analysis"
    summary_path = analysis_dir / f"{prefix}_{model_name}_confidence_summary.json"
    conformal_path = analysis_dir / f"{prefix}_{model_name}_conformal_summary.json"

    calibration = dict(conformal_payload.get("calibration", {}))
    val_conformal = dict(conformal_payload.get("val", {}))
    test_conformal = dict(conformal_payload.get("test", {}))
    summary_payload = _existing_summary_payload(summary_path)
    summary_payload.update(
        {
            "tag": f"{prefix}_{model_name}",
            "conformal_enabled": True,
            "conformal_alpha": _safe_json_float(calibration.get("alpha")),
            "probability_calibration_method": str(
                dict(conformal_payload.get("probability_calibration", {})).get("method", "raw")
            ),
            "probability_calibration_temperature": _safe_json_float(
                dict(conformal_payload.get("probability_calibration", {})).get("temperature", 1.0)
            ),
            "conformal_threshold": _safe_json_float(calibration.get("threshold")),
            "conformal_operating_threshold": _safe_json_float(calibration.get("operating_threshold")),
            "val_abstention_rate": _safe_json_float(val_conformal.get("abstention_rate")),
            "test_abstention_rate": _safe_json_float(test_conformal.get("abstention_rate")),
            "val_accepted_rate": _safe_json_float(val_conformal.get("accepted_rate")),
            "test_accepted_rate": _safe_json_float(test_conformal.get("accepted_rate")),
            "val_selective_accuracy": _safe_json_float(val_conformal.get("selective_accuracy")),
            "test_selective_accuracy": _safe_json_float(test_conformal.get("selective_accuracy")),
            "val_selective_risk": _safe_json_float(val_conformal.get("selective_risk")),
            "test_selective_risk": _safe_json_float(test_conformal.get("selective_risk")),
            "val_conformal_coverage": _safe_json_float(val_conformal.get("coverage")),
            "test_conformal_coverage": _safe_json_float(test_conformal.get("coverage")),
        }
    )
    _update_json_artifact(summary_path, summary_payload)
    _update_json_artifact(conformal_path, conformal_payload)
    return _risk_metrics_from_summary(summary_payload)


def _load_current_risk_metrics(
    run_dir: Path,
    result_rows: list[dict[str, object]],
    *,
    val_y: np.ndarray | None = None,
    test_y: np.ndarray | None = None,
    conformal_alpha: float = 0.10,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    analysis_dir = run_dir / "analysis"
    if not analysis_dir.exists():
        return metrics
    for row in result_rows:
        model_name = str(row.get("model_name", "")).strip()
        model_type = str(row.get("model_type", "")).strip().lower()
        prefix = _analysis_prefix_for_model_type(model_type)
        if not model_name or prefix is None:
            continue
        path = analysis_dir / f"{prefix}_{model_name}_confidence_summary.json"
        payload = _existing_summary_payload(path)
        row_metrics = _risk_metrics_from_summary(payload)
        if (
            not _risk_metrics_complete(row_metrics)
            and val_y is not None
            and test_y is not None
        ):
            backfilled = _backfill_risk_summary(
                run_dir=run_dir,
                row=row,
                val_y=np.asarray(val_y),
                test_y=np.asarray(test_y),
                conformal_alpha=conformal_alpha,
            )
            if backfilled:
                row_metrics = backfilled
        if _risk_metrics_complete(row_metrics):
            metrics[model_name] = row_metrics
    return metrics


def _release_deep_runtime_resources(tf_module, logger) -> None:
    import gc

    gc.collect()
    if tf_module is None:
        logger.info("Released Python memory after deep-model stage.")
        return

    try:
        tf_module.keras.backend.clear_session()
    except Exception as exc:
        logger.warning("TensorFlow session cleanup encountered a non-fatal error: %s", exc)
    gc.collect()
    logger.info("Released TensorFlow and Python memory after deep-model stage.")


__all__ = [
    "_analysis_prefix_for_model_type",
    "_append_existing_artifact_path",
    "_build_run_id",
    "_load_current_risk_metrics",
    "_release_deep_runtime_resources",
    "_track_file",
    "_write_json_artifact",
]
