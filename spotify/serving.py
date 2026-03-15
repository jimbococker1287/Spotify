from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np

from .benchmarks import build_serving_tabular_features
from .probability_bundles import align_proba_to_num_classes


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _extract_artist_proba(prediction) -> np.ndarray:
    if isinstance(prediction, dict):
        if "artist_output" in prediction:
            return np.asarray(prediction["artist_output"])
        return np.asarray(next(iter(prediction.values())))
    if isinstance(prediction, (list, tuple)):
        return np.asarray(prediction[0])
    return np.asarray(prediction)


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-3)
    clipped = np.clip(np.asarray(proba, dtype="float64"), 1e-9, 1.0)
    logits = np.log(clipped) / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (exp / denom).astype("float32")


def load_run_results(run_dir: Path) -> list[dict[str, object]]:
    run_results_path = run_dir / "run_results.json"
    if not run_results_path.exists():
        raise FileNotFoundError(f"Missing run results file: {run_results_path}")
    return json.loads(run_results_path.read_text(encoding="utf-8"))


def load_artist_labels(run_dir: Path) -> list[str]:
    metadata_path = run_dir / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata.get("artist_labels", []))


def _row_by_name(results: list[dict[str, object]], model_name: str) -> dict[str, object] | None:
    for row in results:
        if str(row.get("model_name", "")).strip() == model_name:
            return row
    return None


def _row_is_serveable(
    row: dict[str, object],
    *,
    run_dir: Path,
    results_index: dict[str, dict[str, object]],
    stack: set[str] | None = None,
) -> bool:
    stack = stack or set()
    model_name = str(row.get("model_name", "")).strip()
    model_type = str(row.get("model_type", "")).strip().lower()
    if not model_name or model_name in stack:
        return False

    if model_type == "deep":
        return (run_dir / f"best_{model_name}.keras").exists()

    if model_type in ("classical", "classical_tuned"):
        estimator_artifact_path = str(row.get("estimator_artifact_path", "")).strip()
        return bool(estimator_artifact_path) and Path(estimator_artifact_path).exists()

    if model_type == "ensemble":
        members = row.get("ensemble_members", [])
        if not isinstance(members, list) or not members:
            return False
        next_stack = set(stack)
        next_stack.add(model_name)
        for member_name in members:
            member_row = results_index.get(str(member_name).strip())
            if member_row is None or not _row_is_serveable(member_row, run_dir=run_dir, results_index=results_index, stack=next_stack):
                return False
        return True

    return False


def resolve_model_row(
    run_dir: Path,
    *,
    explicit_model_name: str | None,
    alias_model_name: str | None,
) -> dict[str, object]:
    results = load_run_results(run_dir)
    results_index = {str(row.get("model_name", "")).strip(): row for row in results if str(row.get("model_name", "")).strip()}

    if explicit_model_name:
        row = results_index.get(str(explicit_model_name).strip())
        if row is None:
            raise FileNotFoundError(f"Model '{explicit_model_name}' not found in run results.")
        if not _row_is_serveable(row, run_dir=run_dir, results_index=results_index):
            raise FileNotFoundError(f"Model '{explicit_model_name}' is not serveable from run artifacts.")
        return row

    if alias_model_name:
        row = results_index.get(str(alias_model_name).strip())
        if row is not None and _row_is_serveable(row, run_dir=run_dir, results_index=results_index):
            return row

    serveable_rows = [
        row for row in results if _row_is_serveable(row, run_dir=run_dir, results_index=results_index)
    ]
    if not serveable_rows:
        raise RuntimeError("No serveable models found in run results.")
    return max(serveable_rows, key=lambda row: _safe_float(row.get("val_top1")))


@dataclass
class LoadedPredictor:
    model_name: str
    model_type: str
    artist_labels: list[str]
    _predict_impl: object

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        return self._predict_impl.predict_proba(seq_batch, ctx_batch)


class _DeepPredictorImpl:
    def __init__(self, run_dir: Path, model_name: str):
        import tensorflow as tf

        model_path = run_dir / f"best_{model_name}.keras"
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        pred = self.model.predict((seq_batch, ctx_batch), verbose=0)
        return _extract_artist_proba(pred).astype("float32")


class _ClassicalPredictorImpl:
    def __init__(self, row: dict[str, object], num_classes: int):
        estimator_path = Path(str(row.get("estimator_artifact_path", "")).strip())
        self.estimator = joblib.load(estimator_path)
        self.base_name = str(row.get("base_model_name", "")).strip() or str(row.get("model_name", "")).strip()
        self.num_classes = int(num_classes)

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        if self.base_name == "session_knn":
            features = np.asarray(seq_batch, dtype="int32")
        else:
            features = build_serving_tabular_features(np.asarray(seq_batch), np.asarray(ctx_batch))
        raw = np.asarray(self.estimator.predict_proba(features))
        classes = np.asarray(getattr(self.estimator, "classes_", []))
        return align_proba_to_num_classes(raw, classes if classes.size else None, self.num_classes)


class _EnsemblePredictorImpl:
    def __init__(
        self,
        *,
        row: dict[str, object],
        run_dir: Path,
        artist_labels: list[str],
        results_index: dict[str, dict[str, object]],
        cache: dict[str, LoadedPredictor],
    ):
        self.temperature = float(row.get("calibration_temperature", 1.0))
        member_names = row.get("ensemble_members", [])
        if not isinstance(member_names, list) or not member_names:
            raise RuntimeError("Ensemble row is missing ensemble_members.")
        raw_weights = row.get("ensemble_weights", {})
        if not isinstance(raw_weights, dict):
            raw_weights = {}
        self.members: list[tuple[LoadedPredictor, float]] = []
        for name in member_names:
            member_name = str(name).strip()
            member_row = results_index.get(member_name)
            if member_row is None:
                raise RuntimeError(f"Ensemble member '{member_name}' not found in run results.")
            predictor = load_predictor(
                run_dir=run_dir,
                row=member_row,
                artist_labels=artist_labels,
                results_index=results_index,
                cache=cache,
            )
            self.members.append((predictor, float(raw_weights.get(member_name, 0.0))))

        total = sum(weight for _, weight in self.members)
        if total <= 0:
            uniform = 1.0 / len(self.members)
            self.members = [(predictor, uniform) for predictor, _ in self.members]
        else:
            self.members = [(predictor, weight / total) for predictor, weight in self.members]

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        blended = None
        for predictor, weight in self.members:
            member_proba = predictor.predict_proba(seq_batch, ctx_batch)
            if blended is None:
                blended = np.asarray(member_proba, dtype="float64") * float(weight)
            else:
                blended += np.asarray(member_proba, dtype="float64") * float(weight)
        assert blended is not None
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        normalized = (blended / row_sums).astype("float32")
        return _apply_temperature(normalized, self.temperature)


def load_predictor(
    *,
    run_dir: Path,
    row: dict[str, object],
    artist_labels: list[str],
    results_index: dict[str, dict[str, object]] | None = None,
    cache: dict[str, LoadedPredictor] | None = None,
) -> LoadedPredictor:
    model_name = str(row.get("model_name", "")).strip()
    if not model_name:
        raise RuntimeError("Cannot load predictor for row with no model_name.")
    cache = cache if cache is not None else {}
    if model_name in cache:
        return cache[model_name]

    results_index = results_index or {str(item.get("model_name", "")).strip(): item for item in load_run_results(run_dir)}
    model_type = str(row.get("model_type", "")).strip().lower()

    if model_type == "deep":
        impl = _DeepPredictorImpl(run_dir=run_dir, model_name=model_name)
    elif model_type in ("classical", "classical_tuned"):
        impl = _ClassicalPredictorImpl(row=row, num_classes=len(artist_labels))
    elif model_type == "ensemble":
        impl = _EnsemblePredictorImpl(
            row=row,
            run_dir=run_dir,
            artist_labels=artist_labels,
            results_index=results_index,
            cache=cache,
        )
    else:
        raise RuntimeError(f"Unsupported serveable model type: {model_type}")

    loaded = LoadedPredictor(
        model_name=model_name,
        model_type=model_type,
        artist_labels=artist_labels,
        _predict_impl=impl,
    )
    cache[model_name] = loaded
    return loaded
