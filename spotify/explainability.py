from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import pickle
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .reporting import VAL_KEY
from .run_artifacts import copy_file_if_changed, write_json


SHAP_CACHE_SCHEMA_VERSION = "shap-cache-v2"
DEFAULT_CLASSICAL_SHAP_MAX_ESTIMATOR_MB = 256.0


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _classical_shap_max_estimator_mb() -> float:
    raw = os.getenv("SPOTIFY_CLASSICAL_SHAP_MAX_ESTIMATOR_MB", "").strip()
    if not raw:
        return DEFAULT_CLASSICAL_SHAP_MAX_ESTIMATOR_MB
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_CLASSICAL_SHAP_MAX_ESTIMATOR_MB


def _pack_kernel_explainer_inputs(seq: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    seq_array = np.asarray(seq)
    ctx_array = np.asarray(ctx)
    if seq_array.shape[0] != ctx_array.shape[0]:
        raise ValueError("Sequence and context batches must have matching row counts.")
    return np.concatenate([seq_array.reshape(seq_array.shape[0], -1), ctx_array.reshape(ctx_array.shape[0], -1)], axis=1)


def _unpack_kernel_explainer_inputs(
    packed: np.ndarray,
    *,
    seq_shape: tuple[int, ...],
    ctx_shape: tuple[int, ...],
    seq_dtype,
    ctx_dtype,
) -> tuple[np.ndarray, np.ndarray]:
    packed_array = np.asarray(packed)
    if packed_array.ndim == 1:
        packed_array = packed_array.reshape(1, -1)

    seq_width = int(np.prod(seq_shape))
    ctx_width = int(np.prod(ctx_shape))
    expected_width = seq_width + ctx_width
    if packed_array.shape[1] != expected_width:
        raise ValueError(f"Packed SHAP input width {packed_array.shape[1]} does not match expected {expected_width}.")

    seq = packed_array[:, :seq_width].reshape((-1, *seq_shape)).astype(seq_dtype, copy=False)
    ctx = packed_array[:, seq_width:].reshape((-1, *ctx_shape)).astype(ctx_dtype, copy=False)
    return seq, ctx


def _extract_artist_predictions(prediction) -> np.ndarray:
    if isinstance(prediction, dict):
        if "artist_output" in prediction:
            return np.asarray(prediction["artist_output"])
        first_value = next(iter(prediction.values()))
        return np.asarray(first_value)
    if isinstance(prediction, (list, tuple)):
        return np.asarray(prediction[0])
    return np.asarray(prediction)


def _artist_output_model(model):
    if len(model.outputs) == 1:
        return model

    output_names = [str(name) for name in getattr(model, "output_names", ())]
    artist_index = output_names.index("artist_output") if "artist_output" in output_names else 0

    import tensorflow as tf

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.outputs[artist_index],
        name=f"{model.name}_artist_explainer",
    )


def _shap_cache_enabled_from_env() -> bool:
    raw = os.getenv("SPOTIFY_CACHE_SHAP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()[:24]


def _shap_source_digest() -> str:
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:24]


def _build_shap_cache_key(
    *,
    cache_fingerprint: str,
    best_name: str,
    model_path: Path,
    data,
    background_rows: int,
    explain_rows: int,
    kernel_nsamples: int,
) -> str:
    payload = {
        "schema_version": SHAP_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": str(cache_fingerprint).strip(),
        "best_model_name": best_name,
        "model_digest": _file_digest(model_path),
        "source_digest": _shap_source_digest(),
        "bg_rows": min(background_rows, int(len(data.X_seq_train))),
        "expl_rows": min(explain_rows, int(len(data.X_seq_test))),
        "kernel_nsamples": kernel_nsamples,
        "seq_shape": tuple(int(value) for value in np.asarray(data.X_seq_train).shape[1:]),
        "ctx_shape": tuple(int(value) for value in np.asarray(data.X_ctx_train).shape[1:]),
    }
    import json

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _resolve_shap_cache_path(
    *,
    cache_root: Path,
    cache_fingerprint: str,
    best_name: str,
    cache_key: str,
) -> Path:
    return (cache_root / cache_fingerprint / best_name / cache_key / "shap_values.pkl").resolve()


def run_shap_analysis(
    histories,
    output_dir: Path,
    data,
    logger,
    *,
    cache_root: Path | None = None,
    cache_fingerprint: str = "",
) -> Path | None:
    legacy_out_path = output_dir / "shap_values.pkl"
    ranked_models: list[tuple[float, str, object]] = []
    for name, history in histories.items():
        val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
        ranked_models.append((float(history.history[val_key][-1]), name, history))
    ranked_models.sort(reverse=True)

    if not ranked_models:
        logger.info("Skipping SHAP analysis: no trained models available.")
        return None

    best_eligible_name = ranked_models[0][1]
    background_rows = _positive_int_env("SPOTIFY_DEEP_SHAP_BACKGROUND_ROWS", 128)
    explain_rows = _positive_int_env("SPOTIFY_DEEP_SHAP_EXPLAIN_ROWS", 64)
    kernel_nsamples = _positive_int_env("SPOTIFY_DEEP_SHAP_KERNEL_NSAMPLES", 200)
    generated_best = False
    pending_models: list[tuple[float, str, object, Path | None]] = []
    for score, model_name, history in ranked_models:
        model_path = output_dir / f"best_{model_name}.keras"
        if not model_path.exists():
            logger.info("Skipping SHAP analysis for %s: checkpoint not found.", model_name)
            continue
        cache_path = None
        cache_enabled = (
            _shap_cache_enabled_from_env()
            and cache_root is not None
            and bool(str(cache_fingerprint).strip())
        )
        if cache_enabled and cache_root is not None:
            cache_key = _build_shap_cache_key(
                cache_fingerprint=cache_fingerprint,
                best_name=model_name,
                model_path=model_path,
                data=data,
                background_rows=background_rows,
                explain_rows=explain_rows,
                kernel_nsamples=kernel_nsamples,
            )
            cache_path = _resolve_shap_cache_path(
                cache_root=cache_root,
                cache_fingerprint=cache_fingerprint,
                best_name=model_name,
                cache_key=cache_key,
            )
            if cache_path.exists():
                out_path = output_dir / f"shap_values_{model_name}.pkl"
                copy_file_if_changed(cache_path, out_path)
                if model_name == best_eligible_name:
                    copy_file_if_changed(cache_path, legacy_out_path)
                    generated_best = True
                logger.info("Reused SHAP cache for %s", model_name)
                continue
        pending_models.append((score, model_name, history, cache_path))

    if not pending_models:
        return legacy_out_path if generated_best else None

    try:
        import shap
    except ImportError:
        logger.info("Skipping SHAP analysis: shap library not available.")
        return None

    from .model_loading import load_trusted_keras_model

    for _score, model_name, _history, cache_path in pending_models:
        model_path = output_dir / f"best_{model_name}.keras"
        out_path = output_dir / f"shap_values_{model_name}.pkl"
        try:
            model = load_trusted_keras_model(
                model_path,
                model_name=model_name,
                compile=False,
            )
            model = _artist_output_model(model)

            bg_seq = data.X_seq_train[:background_rows]
            bg_ctx = data.X_ctx_train[:background_rows]
            expl_seq = data.X_seq_test[:explain_rows]
            expl_ctx = data.X_ctx_test[:explain_rows]

            try:
                explainer = shap.DeepExplainer(model, [bg_seq, bg_ctx])
                shap_values = explainer.shap_values([expl_seq, expl_ctx])
            except Exception:
                try:
                    explainer = shap.GradientExplainer(model, [bg_seq, bg_ctx])
                    shap_values = explainer.shap_values([expl_seq, expl_ctx])
                except Exception:
                    bg_kernel = _pack_kernel_explainer_inputs(bg_seq, bg_ctx)
                    expl_kernel = _pack_kernel_explainer_inputs(expl_seq, expl_ctx)

                    def predict_fn(inp):
                        seq_batch, ctx_batch = _unpack_kernel_explainer_inputs(
                            inp,
                            seq_shape=tuple(bg_seq.shape[1:]),
                            ctx_shape=tuple(bg_ctx.shape[1:]),
                            seq_dtype=bg_seq.dtype,
                            ctx_dtype=bg_ctx.dtype,
                        )
                        prediction = model(
                            [seq_batch, ctx_batch],
                            training=False,
                        )
                        return _extract_artist_predictions(prediction)

                    explainer = shap.KernelExplainer(predict_fn, bg_kernel)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        shap_values = explainer.shap_values(
                            expl_kernel,
                            nsamples=kernel_nsamples,
                        )

            with out_path.open("wb") as outfile:
                pickle.dump({"name": model_name, "values": shap_values}, outfile)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                copy_file_if_changed(out_path, cache_path)
            if model_name == best_eligible_name:
                copy_file_if_changed(out_path, legacy_out_path)
                generated_best = True
            logger.info("SHAP analysis complete for %s: %s", model_name, out_path)
        except Exception as exc:
            logger.warning("SHAP analysis failed for %s: %s", model_name, exc)

    return legacy_out_path if generated_best else None


def _mean_abs_feature_values(values, feature_count: int) -> np.ndarray:
    raw = getattr(values, "values", values)
    if isinstance(raw, list):
        arrays = [np.asarray(value) for value in raw]
        array = np.stack(arrays, axis=0)
    else:
        array = np.asarray(raw)
    matching_axes = [idx for idx, size in enumerate(array.shape) if size == feature_count]
    if not matching_axes:
        raise ValueError("Could not identify the feature axis in explainability values.")
    feature_axis = matching_axes[-1]
    array = np.moveaxis(array, feature_axis, -1)
    reduce_axes = tuple(range(array.ndim - 1))
    return np.mean(np.abs(array), axis=reduce_axes)


def run_classical_explainability(
    results,
    *,
    data,
    output_dir: Path,
    logger,
) -> list[Path]:
    """Create SHAP or deterministic fallback importance for every estimator."""
    from .benchmarks import CLASSICAL_SEQUENCE_FEATURE_NAMES, build_tabular_features

    try:
        import joblib
    except ImportError:
        logger.info("Skipping classical explainability: joblib is unavailable.")
        return []

    X_train, X_val, _X_test = build_tabular_features(data)
    feature_names = [*CLASSICAL_SEQUENCE_FEATURE_NAMES, *list(data.context_features)]
    background_rows = _positive_int_env("SPOTIFY_CLASSICAL_SHAP_BACKGROUND_ROWS", 64)
    sample_rows = _positive_int_env("SPOTIFY_CLASSICAL_SHAP_SAMPLE_ROWS", 128)
    permutation_repeats = _positive_int_env("SPOTIFY_CLASSICAL_PERMUTATION_REPEATS", 1)
    shap_max_estimator_mb = _classical_shap_max_estimator_mb()
    background = X_train[: min(background_rows, len(X_train))]
    sample = X_val[: min(sample_rows, len(X_val))]
    sample_y = np.asarray(data.y_val[: len(sample)])
    artifacts: list[Path] = []

    for row in results:
        model_name = str(getattr(row, "model_name", "")).strip()
        estimator_path = Path(str(getattr(row, "estimator_artifact_path", "")).strip())
        if not model_name or not estimator_path.exists():
            continue
        estimator_size_mb = estimator_path.stat().st_size / (1024 * 1024)
        if estimator_size_mb > shap_max_estimator_mb:
            configured_native_path = str(
                getattr(row, "native_importance_artifact_path", "")
            ).strip()
            native_path = (
                Path(configured_native_path)
                if configured_native_path
                else estimator_path.with_name(f"{estimator_path.stem}_native_importance.npy")
            )
            if not native_path.exists():
                logger.warning(
                    "Skipping explainability for %s because its %.1f MB estimator exceeds the %.1f MB "
                    "safety limit and no native-importance sidecar exists.",
                    model_name,
                    estimator_size_mb,
                    shap_max_estimator_mb,
                )
                continue
            scores = np.asarray(np.load(native_path), dtype="float64").reshape(-1)
            if len(scores) != len(feature_names):
                logger.warning(
                    "Skipping explainability for %s because native importance width %d does not match "
                    "feature count %d.",
                    model_name,
                    len(scores),
                    len(feature_names),
                )
                continue
            order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
            summary_path = output_dir / f"classical_explainability_{model_name}.json"
            write_json(
                summary_path,
                {
                    "model_name": model_name,
                    "method": "native_artifact_size_guard",
                    "estimator_size_mb": estimator_size_mb,
                    "features": [
                        {"feature": feature_names[idx], "importance": float(scores[idx])}
                        for idx in order
                    ],
                },
            )
            artifacts.append(summary_path)
            logger.info(
                "Used native-importance sidecar for %s because its %.1f MB estimator exceeds the "
                "%.1f MB SHAP safety limit.",
                model_name,
                estimator_size_mb,
                shap_max_estimator_mb,
            )
            continue
        estimator = None
        final_estimator = None
        transformed_background = None
        transformed_sample = None
        shap_values = None
        try:
            estimator = joblib.load(estimator_path)
            final_estimator = estimator
            transformed_background = background
            transformed_sample = sample
            if hasattr(estimator, "steps") and getattr(estimator, "steps", None):
                final_estimator = estimator.steps[-1][1]
                if len(estimator.steps) > 1:
                    transformer = estimator[:-1]
                    transformed_background = np.asarray(transformer.transform(background))
                    transformed_sample = np.asarray(transformer.transform(sample))
            if final_estimator.__class__.__name__ == "LabelEncodedXGBoostClassifier":
                final_estimator = final_estimator.estimator

            method = "native"
            scores = None
            shap_values = None
            try:
                estimator_module = final_estimator.__class__.__module__
                if estimator_module.startswith("xgboost"):
                    import xgboost

                    booster = final_estimator.get_booster()
                    contributions = np.asarray(
                        booster.predict(
                            xgboost.DMatrix(transformed_sample[:64]),
                            pred_contribs=True,
                        )
                    )
                    shap_values = contributions[..., :-1]
                    method = "shap_native_xgboost"
                else:
                    import shap

                if shap_values is None and (
                    hasattr(final_estimator, "feature_importances_")
                    or hasattr(final_estimator, "get_booster")
                ):
                    shap_values = shap.TreeExplainer(final_estimator).shap_values(transformed_sample[:64])
                elif shap_values is None and hasattr(final_estimator, "coef_"):
                    shap_values = shap.LinearExplainer(final_estimator, transformed_background).shap_values(
                        transformed_sample[:64]
                    )
                if shap_values is not None:
                    scores = _mean_abs_feature_values(shap_values, len(feature_names))
                    if method == "native":
                        method = "shap"
                    shap_path = output_dir / f"classical_shap_values_{model_name}.pkl"
                    with shap_path.open("wb") as outfile:
                        pickle.dump({"name": model_name, "values": shap_values, "features": feature_names}, outfile)
                    artifacts.append(shap_path)
            except Exception as exc:
                logger.info(
                    "SHAP unavailable for classical model %s (%s); using fallback importance.",
                    model_name,
                    exc,
                )

            if scores is None and hasattr(final_estimator, "feature_importances_"):
                scores = np.asarray(final_estimator.feature_importances_, dtype="float64")
            if scores is None and hasattr(final_estimator, "coef_"):
                coefficients = np.asarray(final_estimator.coef_, dtype="float64")
                scores = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
            if scores is None:
                from sklearn.inspection import permutation_importance

                permutation = permutation_importance(
                    estimator,
                    sample,
                    sample_y,
                    n_repeats=permutation_repeats,
                    random_state=42,
                    scoring="accuracy",
                    n_jobs=1,
                )
                scores = np.asarray(permutation.importances_mean, dtype="float64")
                method = "permutation"

            scores = np.asarray(scores, dtype="float64").reshape(-1)
            if len(scores) != len(feature_names):
                raise ValueError(
                    f"Importance width {len(scores)} does not match feature count {len(feature_names)}."
                )
            order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
            summary_path = output_dir / f"classical_explainability_{model_name}.json"
            write_json(
                summary_path,
                {
                    "model_name": model_name,
                    "method": method,
                    "features": [
                        {"feature": feature_names[idx], "importance": float(scores[idx])}
                        for idx in order
                    ],
                },
            )
            artifacts.append(summary_path)
        except Exception as exc:
            logger.warning("Classical explainability failed for %s: %s", model_name, exc)
        finally:
            del shap_values
            del transformed_sample
            del transformed_background
            del final_estimator
            del estimator
            gc.collect()
    return artifacts
