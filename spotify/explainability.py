from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from .reporting import VAL_KEY


def run_shap_analysis(histories, output_dir: Path, data, logger) -> Path | None:
    try:
        import shap
    except ImportError:
        logger.info("Skipping SHAP analysis: shap library not available.")
        return None

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

    try:
        import tensorflow as tf

        model_path = output_dir / f"best_{best_name}.keras"
        if not model_path.exists():
            logger.info("Skipping SHAP analysis: best checkpoint not found at %s", model_path)
            return None

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
                def predict_fn(inp):
                    return best_model.predict([inp[0], inp[1]])

                explainer = shap.KernelExplainer(predict_fn, [bg_seq, bg_ctx])
                shap_values = explainer.shap_values([expl_seq, expl_ctx], nsamples=200)

        out_path = output_dir / "shap_values.pkl"
        with out_path.open("wb") as outfile:
            pickle.dump({"name": best_name, "values": shap_values}, outfile)

        logger.info("SHAP analysis complete and saved to %s", out_path)
        return out_path
    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)
        return None
