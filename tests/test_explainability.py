from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np

from spotify.explainability import run_shap_analysis


class _FakeHistory:
    def __init__(self, score: float):
        self.history = {"val_sparse_categorical_accuracy": [score]}


def _logger() -> logging.Logger:
    logger = logging.getLogger("test-explainability")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    return logger


def test_kernel_explainer_fallback_packs_multi_input_data(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeDeepExplainer:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("deep explainer unavailable")

    class FakeGradientExplainer:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("gradient explainer unavailable")

    class FakeKernelExplainer:
        def __init__(self, predict_fn, data):
            captured["background_type"] = type(data)
            captured["background_shape"] = np.asarray(data).shape
            self.predict_fn = predict_fn

        def shap_values(self, data, nsamples=200):
            captured["explain_type"] = type(data)
            captured["explain_shape"] = np.asarray(data).shape
            captured["nsamples"] = nsamples
            preds = self.predict_fn(data)
            captured["prediction_shape"] = np.asarray(preds).shape
            return np.zeros_like(np.asarray(data), dtype="float32")

    fake_shap = types.SimpleNamespace(
        DeepExplainer=FakeDeepExplainer,
        GradientExplainer=FakeGradientExplainer,
        KernelExplainer=FakeKernelExplainer,
    )

    class FakeModel:
        outputs = [object()]

        def predict(self, inputs, verbose=0):
            seq_batch, ctx_batch = inputs
            captured["predict_verbose"] = verbose
            captured["seq_batch_shape"] = tuple(seq_batch.shape)
            captured["ctx_batch_shape"] = tuple(ctx_batch.shape)
            return np.ones((seq_batch.shape[0], 3), dtype="float32")

    fake_tf = SimpleNamespace(
        keras=SimpleNamespace(
            models=SimpleNamespace(load_model=lambda *_args, **_kwargs: FakeModel())
        )
    )

    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    model_path = tmp_path / "best_gru_artist.keras"
    model_path.write_text("placeholder", encoding="utf-8")

    data = SimpleNamespace(
        X_seq_train=np.arange(24, dtype="int32").reshape(8, 3),
        X_ctx_train=np.arange(16, dtype="float32").reshape(8, 2),
        X_seq_test=np.arange(18, dtype="int32").reshape(6, 3),
        X_ctx_test=np.arange(12, dtype="float32").reshape(6, 2),
    )

    out_path = run_shap_analysis(
        histories={"gru_artist": _FakeHistory(0.5)},
        output_dir=tmp_path,
        data=data,
        logger=_logger(),
    )

    assert out_path == tmp_path / "shap_values.pkl"
    assert out_path.exists()
    assert captured["background_type"] is np.ndarray
    assert captured["background_shape"] == (8, 5)
    assert captured["explain_type"] is np.ndarray
    assert captured["explain_shape"] == (6, 5)
    assert captured["seq_batch_shape"] == (6, 3)
    assert captured["ctx_batch_shape"] == (6, 2)
    assert captured["prediction_shape"] == (6, 3)
    assert captured["predict_verbose"] == 0
    assert captured["nsamples"] == 200
