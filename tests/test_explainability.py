from __future__ import annotations

import builtins
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


def test_run_shap_analysis_reuses_cache_without_importing_shap_or_tensorflow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_CACHE_SHAP", "1")

    class FakeDeepExplainer:
        def __init__(self, *_args, **_kwargs):
            pass

        def shap_values(self, inputs):
            seq_batch, _ctx_batch = inputs
            return np.zeros((seq_batch.shape[0], 3), dtype="float32")

    fake_shap = types.SimpleNamespace(
        DeepExplainer=FakeDeepExplainer,
        GradientExplainer=FakeDeepExplainer,
        KernelExplainer=FakeDeepExplainer,
    )

    class FakeModel:
        outputs = [object()]

    fake_tf = SimpleNamespace(
        keras=SimpleNamespace(
            models=SimpleNamespace(load_model=lambda *_args, **_kwargs: FakeModel())
        )
    )

    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    cache_root = tmp_path / "cache"
    output_dir_a = tmp_path / "run_a"
    output_dir_a.mkdir(parents=True, exist_ok=True)
    model_path_a = output_dir_a / "best_gru_artist.keras"
    model_path_a.write_text("same-model", encoding="utf-8")

    data = SimpleNamespace(
        X_seq_train=np.arange(24, dtype="int32").reshape(8, 3),
        X_ctx_train=np.arange(16, dtype="float32").reshape(8, 2),
        X_seq_test=np.arange(18, dtype="int32").reshape(6, 3),
        X_ctx_test=np.arange(12, dtype="float32").reshape(6, 2),
    )
    histories = {"gru_artist": _FakeHistory(0.5)}

    first_path = run_shap_analysis(
        histories=histories,
        output_dir=output_dir_a,
        data=data,
        logger=_logger(),
        cache_root=cache_root,
        cache_fingerprint="prepared123",
    )

    assert first_path == output_dir_a / "shap_values.pkl"
    assert first_path.exists()

    sys.modules.pop("shap", None)
    sys.modules.pop("tensorflow", None)
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "shap" or name.startswith("shap.") or name == "tensorflow" or name.startswith("tensorflow."):
            raise AssertionError("SHAP cache hit should not import shap or tensorflow")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    output_dir_b = tmp_path / "run_b"
    output_dir_b.mkdir(parents=True, exist_ok=True)
    model_path_b = output_dir_b / "best_gru_artist.keras"
    model_path_b.write_text("same-model", encoding="utf-8")

    second_path = run_shap_analysis(
        histories=histories,
        output_dir=output_dir_b,
        data=data,
        logger=_logger(),
        cache_root=cache_root,
        cache_fingerprint="prepared123",
    )

    assert second_path == output_dir_b / "shap_values.pkl"
    assert second_path.exists()
    assert second_path.read_bytes() == first_path.read_bytes()
