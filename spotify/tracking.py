from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def _sanitize_metric_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\-/]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "metric"


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
            self._active_run = mlflow.start_run(run_name=(run_name or run_id))
            mlflow.set_tags({"run_id": run_id, "pipeline": "spotify"})
        except Exception as exc:
            logger.warning("Failed to initialize MLflow tracking: %s", exc)
            self._active_run = None
            return

        self.enabled = True
        logger.info("MLflow tracking enabled: experiment=%s uri=%s", experiment_name, resolved_tracking_uri)

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
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            self._logger.warning("MLflow log_artifact failed for %s: %s", path, exc)

    def end(self) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            self._mlflow.end_run()
        except Exception as exc:
            self._logger.warning("MLflow end_run failed: %s", exc)
