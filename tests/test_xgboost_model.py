from __future__ import annotations

import sys
import types

import pytest

from spotify import xgboost_model


class FakeXGBClassifier:
    def __init__(self, **params: object) -> None:
        self.params = params


class FakeTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def suggest_int(self, name: str, *args: object, **kwargs: object) -> int:
        self.calls.append(("int", name, args, kwargs))
        return int(args[0])

    def suggest_float(self, name: str, *args: object, **kwargs: object) -> float:
        self.calls.append(("float", name, args, kwargs))
        return float(args[0])


def test_build_xgboost_classifier_uses_multiclass_histogram_defaults(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(XGBClassifier=FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)

    estimator = xgboost_model.build_xgboost_classifier(
        {"max_depth": 4, "objective": "binary:logistic", "eval_metric": "error"},
        random_seed=17,
        n_jobs=3,
        num_classes=8,
    )

    assert estimator.estimator.params["tree_method"] == "hist"
    assert estimator.estimator.params["max_depth"] == 4
    assert estimator.estimator.params["objective"] == "multi:softprob"
    assert estimator.estimator.params["eval_metric"] == "mlogloss"
    assert estimator.estimator.params["random_state"] == 17
    assert estimator.estimator.params["n_jobs"] == 3
    assert estimator.estimator.params["num_class"] == 8


def test_build_xgboost_classifier_allows_class_count_inference(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(XGBClassifier=FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)

    estimator = xgboost_model.build_xgboost_classifier({}, random_seed=1, n_jobs=1)

    assert "num_class" not in estimator.estimator.params


def test_build_xgboost_classifier_import_is_lazy_and_actionable(monkeypatch) -> None:
    def missing_xgboost(name: str):
        if name == "xgboost":
            raise ModuleNotFoundError("No module named 'xgboost'")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(xgboost_model, "import_module", missing_xgboost)

    with pytest.raises(ImportError, match="optional 'xgboost' package"):
        xgboost_model.build_xgboost_classifier({}, random_seed=1, n_jobs=1)


def test_build_xgboost_classifier_rejects_invalid_multiclass_count(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(XGBClassifier=FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)

    with pytest.raises(ValueError, match="at least 2"):
        xgboost_model.build_xgboost_classifier({}, random_seed=1, n_jobs=1, num_classes=1)


def test_suggest_xgboost_params_uses_trial_like_object() -> None:
    trial = FakeTrial()

    params = xgboost_model.suggest_xgboost_params(trial)

    assert set(params) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    assert ("int", "n_estimators", (150, 600), {"step": 50}) in trial.calls
    assert ("float", "learning_rate", (0.01, 0.2), {"log": True}) in trial.calls


def test_get_xgboost_native_importance_fills_and_normalizes_features() -> None:
    class Booster:
        def get_score(self, *, importance_type: str) -> dict[str, float]:
            assert importance_type == "gain"
            return {"tempo": 3.0, "energy": 1.0}

    estimator = types.SimpleNamespace(get_booster=lambda: Booster())

    importance = xgboost_model.get_xgboost_native_importance(
        estimator,
        feature_names=["tempo", "energy", "danceability"],
    )

    assert importance == {"tempo": 0.75, "energy": 0.25, "danceability": 0.0}


def test_label_encoded_wrapper_handles_missing_integer_classes(monkeypatch) -> None:
    class FittedFakeXGBClassifier(FakeXGBClassifier):
        def fit(self, _X, y):
            self.fitted_y = np.asarray(y)
            return self

        def predict(self, X):
            return np.asarray([0, 2], dtype="int64")[: len(X)]

        def predict_proba(self, X):
            return np.ones((len(X), 3), dtype="float32") / 3.0

    import numpy as np

    fake_module = types.SimpleNamespace(XGBClassifier=FittedFakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)
    estimator = xgboost_model.build_xgboost_classifier({}, random_seed=1, n_jobs=1)

    estimator.fit(np.zeros((3, 2)), np.asarray([0, 2, 5]))

    assert estimator.classes_.tolist() == [0, 2, 5]
    assert estimator.estimator.fitted_y.tolist() == [0, 1, 2]
    assert estimator.predict(np.zeros((2, 2))).tolist() == [0, 5]
