from __future__ import annotations

import builtins
import sys
import types

import pytest

from spotify.lightgbm_model import (
    build_lightgbm_classifier,
    get_lightgbm_booster,
    get_lightgbm_feature_importance,
    suggest_lightgbm_params,
    unwrap_lightgbm_classifier,
)


class _FakeLGBMClassifier:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("lightgbm")
    module.LGBMClassifier = _FakeLGBMClassifier
    monkeypatch.setitem(sys.modules, "lightgbm", module)


def test_build_lightgbm_classifier_uses_conservative_multiclass_defaults(monkeypatch) -> None:
    _install_fake_lightgbm(monkeypatch)

    classifier = build_lightgbm_classifier(None, random_seed=17, n_jobs=4)

    assert isinstance(classifier, _FakeLGBMClassifier)
    assert classifier.kwargs["objective"] == "multiclass"
    assert classifier.kwargs["random_state"] == 17
    assert classifier.kwargs["n_jobs"] == 4
    assert classifier.kwargs["n_estimators"] == 400
    assert classifier.kwargs["learning_rate"] == 0.05
    assert classifier.kwargs["num_leaves"] == 31
    assert classifier.kwargs["min_child_samples"] == 50
    assert classifier.kwargs["class_weight"] is None
    assert classifier.kwargs["subsample_freq"] == 1
    assert classifier.kwargs["deterministic"] is True


def test_build_lightgbm_classifier_applies_explicit_overrides_without_mutation(monkeypatch) -> None:
    _install_fake_lightgbm(monkeypatch)
    params = {
        "n_estimators": 650,
        "num_leaves": 63,
        "class_weight": "balanced",
        "extra_trees": True,
    }

    classifier = build_lightgbm_classifier(params, random_seed=23, n_jobs=0)

    assert params == {
        "n_estimators": 650,
        "num_leaves": 63,
        "class_weight": "balanced",
        "extra_trees": True,
    }
    assert classifier.kwargs["n_estimators"] == 650
    assert classifier.kwargs["num_leaves"] == 63
    assert classifier.kwargs["class_weight"] == "balanced"
    assert classifier.kwargs["extra_trees"] is True
    assert classifier.kwargs["n_jobs"] == 1


@pytest.mark.parametrize("controlled_name", ["objective", "random_state", "random_seed", "n_jobs", "num_threads"])
def test_build_lightgbm_classifier_rejects_controlled_params(monkeypatch, controlled_name: str) -> None:
    _install_fake_lightgbm(monkeypatch)

    with pytest.raises(ValueError, match=controlled_name):
        build_lightgbm_classifier({controlled_name: "override"}, random_seed=1, n_jobs=2)


def test_build_lightgbm_classifier_has_clear_lazy_import_error(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "lightgbm", raising=False)
    real_import = builtins.__import__

    def import_without_lightgbm(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("missing in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lightgbm)

    with pytest.raises(ImportError, match=r"optional.*python -m pip install lightgbm"):
        build_lightgbm_classifier({}, random_seed=1, n_jobs=1)


class _RecordingTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def suggest_int(self, name: str, *args: object, **kwargs: object) -> int:
        self.calls.append(("int", name, args, kwargs))
        return int(args[0])

    def suggest_float(self, name: str, *args: object, **kwargs: object) -> float:
        self.calls.append(("float", name, args, kwargs))
        return float(args[0])

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.calls.append(("categorical", name, (choices,), {}))
        return choices[0]


def test_suggest_lightgbm_params_declares_bounded_optuna_space() -> None:
    trial = _RecordingTrial()

    params = suggest_lightgbm_params(trial)

    assert set(params) == {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "min_split_gain",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    calls = {name: (kind, args, kwargs) for kind, name, args, kwargs in trial.calls}
    assert calls["n_estimators"] == ("int", (250, 700), {"step": 50})
    assert calls["learning_rate"] == ("float", (0.02, 0.12), {"log": True})
    assert calls["num_leaves"] == ("categorical", ([15, 31, 63],), {})
    assert calls["max_depth"] == ("categorical", ([-1, 8, 12, 16],), {})
    assert calls["reg_lambda"] == ("float", (0.1, 10.0), {"log": True})


def test_lightgbm_unwrap_and_native_importance_helpers() -> None:
    class Booster:
        def feature_importance(self, *, importance_type: str) -> tuple[str, list[int]]:
            return importance_type, [9, 4]

    classifier = types.SimpleNamespace(booster_=Booster())
    pipeline = types.SimpleNamespace(named_steps={"prep": object(), "classifier": classifier})

    assert unwrap_lightgbm_classifier(pipeline) is classifier
    assert get_lightgbm_booster(pipeline) is classifier.booster_
    assert get_lightgbm_feature_importance(pipeline) == ("gain", [9, 4])
    assert get_lightgbm_feature_importance(pipeline, importance_type="split") == ("split", [9, 4])


def test_native_importance_helpers_validate_state_and_type() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        get_lightgbm_booster(object())
    with pytest.raises(ValueError, match="gain.*split"):
        get_lightgbm_feature_importance(types.SimpleNamespace(booster_=object()), importance_type="weight")
