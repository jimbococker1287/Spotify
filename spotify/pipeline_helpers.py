from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re

from .config import PipelineConfig
from .model_types import analysis_prefix_for_model_type
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


def _load_current_risk_metrics(run_dir: Path, result_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
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
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics[model_name] = {
            "val_selective_risk": _safe_json_float(payload.get("val_selective_risk", float("nan"))),
            "val_abstention_rate": _safe_json_float(payload.get("val_abstention_rate", float("nan"))),
            "test_selective_risk": _safe_json_float(payload.get("test_selective_risk", float("nan"))),
            "test_abstention_rate": _safe_json_float(payload.get("test_abstention_rate", float("nan"))),
        }
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
