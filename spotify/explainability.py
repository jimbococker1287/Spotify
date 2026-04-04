from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle

import numpy as np

from .reporting import VAL_KEY
from .run_artifacts import copy_file_if_changed


SHAP_CACHE_SCHEMA_VERSION = "shap-cache-v1"


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
) -> str:
    payload = {
        "schema_version": SHAP_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": str(cache_fingerprint).strip(),
        "best_model_name": best_name,
        "model_digest": _file_digest(model_path),
        "source_digest": _shap_source_digest(),
        "bg_rows": min(128, int(len(data.X_seq_train))),
        "expl_rows": min(64, int(len(data.X_seq_test))),
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
    out_path = output_dir / "shap_values.pkl"

    best_name = None
    best_score = -np.inf
    for name, history in histories.items():
        val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
        score = history.history[val_key][-1]
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        logger.info("Skipping SHAP analysis: no trained models available.")
        return None

    model_path = output_dir / f"best_{best_name}.keras"
    if not model_path.exists():
        logger.info("Skipping SHAP analysis: best checkpoint not found at %s", model_path)
        return None

    cache_enabled = _shap_cache_enabled_from_env() and cache_root is not None and bool(str(cache_fingerprint).strip())
    cache_path = None
    if cache_enabled:
        cache_key = _build_shap_cache_key(
            cache_fingerprint=cache_fingerprint,
            best_name=best_name,
            model_path=model_path,
            data=data,
        )
        cache_path = _resolve_shap_cache_path(
            cache_root=cache_root,
            cache_fingerprint=cache_fingerprint,
            best_name=best_name,
            cache_key=cache_key,
        )
        if cache_path.exists():
            copy_file_if_changed(cache_path, out_path)
            logger.info("Reused SHAP cache for %s", best_name)
            return out_path

    try:
        import shap
    except ImportError:
        logger.info("Skipping SHAP analysis: shap library not available.")
        return None

    try:
        import tensorflow as tf

        best_model = tf.keras.models.load_model(model_path, compile=False)
        if len(best_model.outputs) != 1:
            logger.info(
                "Skipping SHAP analysis: best model (%s) is multi-output.",
                best_name,
            )
            return None

        bg_seq = data.X_seq_train[:128]
        bg_ctx = data.X_ctx_train[:128]
        expl_seq = data.X_seq_test[:64]
        expl_ctx = data.X_ctx_test[:64]

        try:
            explainer = shap.DeepExplainer(best_model, [bg_seq, bg_ctx])
            shap_values = explainer.shap_values([expl_seq, expl_ctx])
        except Exception:
            try:
                explainer = shap.GradientExplainer(best_model, [bg_seq, bg_ctx])
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
                    prediction = best_model.predict([seq_batch, ctx_batch], verbose=0)
                    return _extract_artist_predictions(prediction)

                explainer = shap.KernelExplainer(predict_fn, bg_kernel)
                shap_values = explainer.shap_values(expl_kernel, nsamples=200)

        with out_path.open("wb") as outfile:
            pickle.dump({"name": best_name, "values": shap_values}, outfile)
        if cache_enabled and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            copy_file_if_changed(out_path, cache_path)

        logger.info("SHAP analysis complete and saved to %s", out_path)
        return out_path
    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)
        return None
