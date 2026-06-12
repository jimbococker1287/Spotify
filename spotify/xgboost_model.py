from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable, Mapping

import numpy as np


_FIXED_CLASSIFIER_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
}

_DEFAULT_CLASSIFIER_PARAMS: dict[str, object] = {
    "tree_method": "hist",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 2.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "verbosity": 0,
}


class LabelEncodedXGBoostClassifier:
    """Make XGBoost robust to temporally missing integer classes."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.classes_: np.ndarray = np.asarray([], dtype="int64")

    def fit(self, X, y, **kwargs):
        y_values = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_values)
        encoded = np.searchsorted(self.classes_, y_values)
        self.estimator.fit(X, encoded, **kwargs)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        encoded = np.asarray(self.estimator.predict(X), dtype="int64")
        return self.classes_[encoded]

    def get_booster(self):
        return self.estimator.get_booster()

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_


def build_xgboost_classifier(
    params: Mapping[str, object] | None,
    random_seed: int,
    n_jobs: int,
    num_classes: int | None = None,
) -> Any:
    """Build an optional XGBoost classifier with stable multiclass probabilities."""
    try:
        xgboost = import_module("xgboost")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "XGBoost is required for the xgboost estimator; install the optional 'xgboost' package."
        ) from exc

    if num_classes is not None and int(num_classes) < 2:
        raise ValueError("num_classes must be at least 2 for multiclass classification.")

    classifier_params = dict(_DEFAULT_CLASSIFIER_PARAMS)
    classifier_params.update(dict(params or {}))
    classifier_params.update(_FIXED_CLASSIFIER_PARAMS)
    classifier_params["random_state"] = int(random_seed)
    classifier_params["n_jobs"] = int(n_jobs)
    if num_classes is not None:
        classifier_params["num_class"] = int(num_classes)

    return LabelEncodedXGBoostClassifier(xgboost.XGBClassifier(**classifier_params))


def suggest_xgboost_params(trial: Any) -> dict[str, object]:
    """Return a conservative histogram-tree search space for an Optuna-like trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def get_xgboost_native_importance(
    estimator: Any,
    *,
    importance_type: str = "gain",
    feature_names: Iterable[str] | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Read booster-native feature importance without importing SHAP."""
    booster = estimator.get_booster()
    scores = {str(name): float(value) for name, value in booster.get_score(importance_type=importance_type).items()}

    if feature_names is not None:
        scores = {str(name): scores.get(str(name), 0.0) for name in feature_names}

    total = sum(scores.values())
    if normalize and total > 0.0:
        scores = {name: value / total for name, value in scores.items()}
    return scores


# Explicit alias for integrations that name search-space functions by model family.
xgboost_optuna_search_space = suggest_xgboost_params


__all__ = [
    "build_xgboost_classifier",
    "get_xgboost_native_importance",
    "LabelEncodedXGBoostClassifier",
    "suggest_xgboost_params",
    "xgboost_optuna_search_space",
]
