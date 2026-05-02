from __future__ import annotations

from pathlib import Path
import os
import re
import time

import numpy as np


MLFLOW_METADATA_ARTIFACT_SUFFIXES = frozenset(
    {
        ".csv",
        ".html",
        ".json",
        ".log",
        ".md",
        ".png",
        ".svg",
        ".txt",
        ".yaml",
        ".yml",
    }
)
MLFLOW_BINARY_ARTIFACT_SUFFIXES = frozenset(
    {
        ".bin",
        ".db",
        ".duckdb",
        ".h5",
        ".hdf5",
        ".joblib",
        ".keras",
        ".npy",
        ".npz",
        ".onnx",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
    }
)
DEFAULT_MLFLOW_ARTIFACT_MODE = "metadata"
DEFAULT_MLFLOW_ARTIFACT_MAX_MB = 25.0


def _sanitize_metric_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\-/]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "metric"


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def normalize_mlflow_artifact_mode(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value in ("", "1", "true", "yes", "on", "default", "light", "metadata", "safe"):
        return DEFAULT_MLFLOW_ARTIFACT_MODE
    if value in ("0", "false", "no", "off", "none", "disabled"):
        return "off"
    if value in ("all", "full"):
        return "all"
    return DEFAULT_MLFLOW_ARTIFACT_MODE


def resolve_mlflow_artifact_policy(
    *,
    mode_raw: object | None = None,
    max_artifact_mb_raw: object | None = None,
) -> tuple[str, float]:
    if mode_raw is None:
        mode_raw = os.getenv("SPOTIFY_MLFLOW_ARTIFACT_MODE", DEFAULT_MLFLOW_ARTIFACT_MODE)
    mode = normalize_mlflow_artifact_mode(mode_raw)

    if max_artifact_mb_raw is None:
        max_artifact_mb_raw = os.getenv("SPOTIFY_MLFLOW_ARTIFACT_MAX_MB", str(DEFAULT_MLFLOW_ARTIFACT_MAX_MB))
    try:
        max_artifact_mb = max(0.0, float(max_artifact_mb_raw))
    except (TypeError, ValueError):
        max_artifact_mb = DEFAULT_MLFLOW_ARTIFACT_MAX_MB
    return mode, max_artifact_mb


def should_log_mlflow_artifact(
    path: Path,
    *,
    mode: str | None = None,
    max_artifact_mb: float | None = None,
) -> bool:
    resolved_mode, resolved_max_artifact_mb = resolve_mlflow_artifact_policy(
        mode_raw=mode,
        max_artifact_mb_raw=max_artifact_mb,
    )
    if resolved_mode == "off":
        return False
    if resolved_mode == "all":
        return path.is_file()
    if not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in MLFLOW_BINARY_ARTIFACT_SUFFIXES:
        return False
    if suffix not in MLFLOW_METADATA_ARTIFACT_SUFFIXES:
        return False

    max_bytes = int(resolved_max_artifact_mb * 1024 * 1024)
    if max_bytes > 0 and _file_size(path) > max_bytes:
        return False
    return True


class MlflowTracker:
    def __init__(
        self,
        enabled: bool,
        run_id: str,
        run_name: str | None,
        tracking_uri: str | None,
        experiment_name: str,
        default_tracking_dir: Path,
        logger,
    ):
        self.enabled = False
        self._mlflow = None
        self._active_run = None
        self._logger = logger
        self._closed = False
        self._artifact_mode, self._artifact_max_artifact_mb = resolve_mlflow_artifact_policy()

        if not enabled:
            return

        try:
            import mlflow
        except Exception as exc:
            logger.warning("MLflow tracking requested but mlflow is unavailable: %s", exc)
            return

        self._mlflow = mlflow
        if tracking_uri:
            resolved_tracking_uri = tracking_uri
        else:
            default_tracking_dir.mkdir(parents=True, exist_ok=True)
            db_path = (default_tracking_dir / "mlflow.db").resolve()
            resolved_tracking_uri = f"sqlite:///{db_path}"
        try:
            mlflow.set_tracking_uri(resolved_tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._cleanup_stale_runs(mlflow=mlflow, experiment_name=experiment_name)
            self._active_run = mlflow.start_run(run_name=(run_name or run_id))
            mlflow.set_tags({"run_id": run_id, "pipeline": "spotify"})
        except Exception as exc:
            logger.warning("Failed to initialize MLflow tracking: %s", exc)
            self._active_run = None
            return

        self.enabled = True
        logger.info("MLflow tracking enabled: experiment=%s uri=%s", experiment_name, resolved_tracking_uri)
        logger.info(
            "MLflow artifact policy: mode=%s max_mb=%.1f",
            self._artifact_mode,
            self._artifact_max_artifact_mb,
        )

    def _cleanup_stale_runs(self, *, mlflow, experiment_name: str) -> None:
        stale_hours_raw = str(os.getenv("SPOTIFY_MLFLOW_STALE_RUN_HOURS", "12")).strip()
        try:
            stale_hours = max(1.0, float(stale_hours_raw))
        except ValueError:
            stale_hours = 12.0

        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return
            now_ms = int(time.time() * 1000)
            stale_ms = int(stale_hours * 3600 * 1000)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attributes.status = 'RUNNING'",
                max_results=200,
            )
            cleaned = 0
            for run in runs:
                if run.info.end_time is not None:
                    continue
                start_time = int(run.info.start_time or 0)
                if start_time <= 0 or (now_ms - start_time) < stale_ms:
                    continue
                client.set_terminated(run.info.run_id, status="KILLED")
                cleaned += 1
            if cleaned > 0:
                self._logger.info("MLflow stale-run cleanup marked %d run(s) as KILLED.", cleaned)
        except Exception as exc:
            self._logger.warning("MLflow stale-run cleanup failed: %s", exc)

    def log_params(self, params: dict[str, object]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        payload: dict[str, str | int | float | bool] = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (tuple, list, set)):
                payload[key] = ",".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                payload[key] = value
            else:
                payload[key] = str(value)
        if not payload:
            return
        try:
            self._mlflow.log_params(payload)
        except Exception as exc:
            self._logger.warning("MLflow log_params failed: %s", exc)

    def log_result_rows(self, rows: list[dict[str, object]]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        for idx, row in enumerate(rows):
            model_name = str(row.get("model_name", f"model_{idx}"))
            prefix = _sanitize_metric_name(f"models.{model_name}")
            for metric_name in (
                "val_top1",
                "val_top5",
                "val_ndcg_at5",
                "val_mrr_at5",
                "val_coverage_at5",
                "val_diversity_at5",
                "test_top1",
                "test_top5",
                "test_ndcg_at5",
                "test_mrr_at5",
                "test_coverage_at5",
                "test_diversity_at5",
                "fit_seconds",
            ):
                value = row.get(metric_name)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isnan(numeric):
                    continue
                key = _sanitize_metric_name(f"{prefix}.{metric_name}")
                try:
                    self._mlflow.log_metric(key, numeric, step=idx)
                except Exception as exc:
                    self._logger.warning("MLflow metric logging failed for %s: %s", key, exc)

    def log_backtest_rows(self, rows: list[dict[str, object]]) -> None:
        if not self.enabled or self._mlflow is None or not rows:
            return
        grouped: dict[str, list[float]] = {}
        for row in rows:
            model_name = str(row.get("model_name", "model"))
            value = row.get("top1")
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(score):
                continue
            grouped.setdefault(model_name, []).append(score)
        for model_name, scores in grouped.items():
            metric_name = _sanitize_metric_name(f"backtest.{model_name}.mean_top1")
            try:
                self._mlflow.log_metric(metric_name, float(np.mean(scores)))
            except Exception as exc:
                self._logger.warning("MLflow backtest metric logging failed for %s: %s", metric_name, exc)

    def log_artifact(self, path: Path) -> None:
        if not self.enabled or self._mlflow is None:
            return
        if not path.exists():
            return
        if not should_log_mlflow_artifact(
            path,
            mode=self._artifact_mode,
            max_artifact_mb=self._artifact_max_artifact_mb,
        ):
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            self._logger.warning("MLflow log_artifact failed for %s: %s", path, exc)

    def end(self, status: str | None = None) -> None:
        if not self.enabled or self._mlflow is None or self._closed:
            return
        try:
            if status:
                self._mlflow.end_run(status=str(status).upper())
            else:
                self._mlflow.end_run()
            self._closed = True
        except Exception as exc:
            self._logger.warning("MLflow end_run failed: %s", exc)
