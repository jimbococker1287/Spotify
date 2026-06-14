from __future__ import annotations

import builtins
import json
import logging
import sys
import types
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import spotify.explainability as explainability
import spotify.model_loading as model_loading
from spotify.explainability import run_classical_explainability, run_shap_analysis


class _FakeHistory:
    def __init__(self, score: float):
        self.history = {"val_sparse_categorical_accuracy": [score]}


class _FakeMultiOutputHistory:
    def __init__(self, score: float):
        self.history = {
            "val_sparse_categorical_accuracy": [score],
            "artist_output_loss": [1.0],
            "skip_output_loss": [0.1],
        }


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

        def __call__(self, inputs, training=False):
            seq_batch, ctx_batch = inputs
            captured["prediction_training"] = training
            captured["seq_batch_shape"] = tuple(seq_batch.shape)
            captured["ctx_batch_shape"] = tuple(ctx_batch.shape)
            return np.ones((seq_batch.shape[0], 3), dtype="float32")

    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    monkeypatch.setattr(
        model_loading,
        "load_trusted_keras_model",
        lambda *_args, **_kwargs: FakeModel(),
    )

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
    assert captured["prediction_training"] is False
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

    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    monkeypatch.setattr(
        model_loading,
        "load_trusted_keras_model",
        lambda *_args, **_kwargs: FakeModel(),
    )

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

    monkeypatch.setitem(sys.modules, "shap", types.ModuleType("forbidden_shap"))
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


def test_run_shap_analysis_explains_artist_head_for_multi_output_history(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeDeepExplainer:
        def __init__(self, model, _background):
            captured["explainer_model"] = model

        def shap_values(self, inputs):
            return np.zeros((len(inputs[0]), 3), dtype="float32")

    class FakeLoadedModel:
        name = "transformer"
        inputs = [object(), object()]
        outputs = ["artist-tensor", "skip-tensor"]
        output_names = ["artist_output", "skip_output"]

    class FakeArtistModel:
        outputs = [object()]

    def build_artist_model(model):
        captured["wrapper_inputs"] = model.inputs
        captured["wrapper_output"] = model.outputs[0]
        captured["wrapper_name"] = f"{model.name}_artist_explainer"
        return FakeArtistModel()

    fake_shap = SimpleNamespace(
        DeepExplainer=FakeDeepExplainer,
        GradientExplainer=FakeDeepExplainer,
        KernelExplainer=FakeDeepExplainer,
    )
    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    monkeypatch.setattr(
        model_loading,
        "load_trusted_keras_model",
        lambda *_args, **_kwargs: FakeLoadedModel(),
    )
    monkeypatch.setattr(explainability, "_artist_output_model", build_artist_model)

    model_path = tmp_path / "best_transformer.keras"
    model_path.write_text("placeholder", encoding="utf-8")

    out_path = run_shap_analysis(
        histories={"transformer": _FakeMultiOutputHistory(0.5)},
        output_dir=tmp_path,
        data=SimpleNamespace(
            X_seq_train=np.arange(24, dtype="int32").reshape(8, 3),
            X_ctx_train=np.arange(16, dtype="float32").reshape(8, 2),
            X_seq_test=np.arange(18, dtype="int32").reshape(6, 3),
            X_ctx_test=np.arange(12, dtype="float32").reshape(6, 2),
        ),
        logger=_logger(),
    )

    assert out_path == tmp_path / "shap_values.pkl"
    assert out_path.exists()
    assert captured["wrapper_output"] == "artist-tensor"
    assert captured["wrapper_name"] == "transformer_artist_explainer"
    assert isinstance(captured["explainer_model"], FakeArtistModel)


def test_classical_explainability_uses_native_importance_for_oversized_forest(tmp_path, monkeypatch) -> None:
    tree_explainer_calls = 0

    class ForbiddenTreeExplainer:
        def __init__(self, *_args, **_kwargs):
            nonlocal tree_explainer_calls
            tree_explainer_calls += 1

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=ForbiddenTreeExplainer))
    monkeypatch.setenv("SPOTIFY_CLASSICAL_SHAP_MAX_ESTIMATOR_MB", "0")

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(30, 10)).astype("float32")
    y_train = np.tile(np.arange(3, dtype="int32"), 10)
    estimator = RandomForestClassifier(n_estimators=3, random_state=11).fit(X_train, y_train)
    estimator_path = tmp_path / "forest.joblib"
    joblib.dump(estimator, estimator_path)
    native_path = tmp_path / "forest_native_importance.npy"
    np.save(native_path, estimator.feature_importances_)

    data = SimpleNamespace(
        X_seq_train=np.arange(90, dtype="int32").reshape(30, 3),
        X_ctx_train=X_train[:, -1:].copy(),
        X_seq_val=np.arange(36, dtype="int32").reshape(12, 3),
        X_ctx_val=rng.normal(size=(12, 1)).astype("float32"),
        X_seq_test=np.arange(18, dtype="int32").reshape(6, 3),
        X_ctx_test=rng.normal(size=(6, 1)).astype("float32"),
        y_val=np.tile(np.arange(3, dtype="int32"), 4),
        context_features=("context_feature",),
    )
    result = SimpleNamespace(
        model_name="large_forest",
        estimator_artifact_path=str(estimator_path),
        native_importance_artifact_path=str(native_path),
    )

    artifacts = run_classical_explainability(
        [result],
        data=data,
        output_dir=tmp_path,
        logger=_logger(),
    )

    summary_path = tmp_path / "classical_explainability_large_forest.json"
    assert summary_path in artifacts
    assert tree_explainer_calls == 0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["method"] == "native_artifact_size_guard"
