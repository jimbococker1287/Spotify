from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class TrialLike(Protocol):
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int: ...


_CONTROLLED_PARAMS = frozenset(
    {
        "objective",
        "random_seed",
        "random_state",
        "n_jobs",
        "num_threads",
    }
)


def build_lightgbm_classifier(
    params: Mapping[str, object] | None,
    random_seed: int,
    n_jobs: int,
) -> object:
    """Build a conservative LightGBM classifier for a large multiclass table.

    Class weights are deliberately disabled by default. Automatic inverse-frequency
    weights change the training prior and commonly degrade the calibration of
    ``predict_proba``. A caller that accepts that tradeoff may explicitly provide
    ``class_weight`` in ``params`` and should calibrate on an unweighted holdout.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM is optional and is not installed. Install it with "
            "'python -m pip install lightgbm' to use build_lightgbm_classifier()."
        ) from exc

    supplied = dict(params or {})
    controlled = sorted(_CONTROLLED_PARAMS.intersection(supplied))
    if controlled:
        names = ", ".join(controlled)
        raise ValueError(
            f"LightGBM parameters controlled by the adapter cannot be overridden: {names}. "
            "Use the random_seed and n_jobs arguments instead."
        )

    defaults: dict[str, object] = {
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 50,
        "min_split_gain": 0.0,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "max_bin": 255,
        "class_weight": None,
        "importance_type": "gain",
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    defaults.update(supplied)
    defaults.update(
        {
            "objective": "multiclass",
            "random_state": int(random_seed),
            "n_jobs": 1 if int(n_jobs) == 0 else int(n_jobs),
        }
    )
    return LGBMClassifier(**defaults)


def suggest_lightgbm_params(trial: TrialLike) -> dict[str, object]:
    """Return the declared Optuna search space for the multiclass adapter."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 700, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 8, 12, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 120, step=10),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2, step=0.02),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def unwrap_lightgbm_classifier(estimator: object) -> object:
    """Unwrap common sklearn wrappers and return the LightGBM classifier."""
    current = estimator
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        named_steps = getattr(current, "named_steps", None)
        if named_steps:
            current = next(reversed(named_steps.values()))
            continue
        for attribute in ("estimator_", "estimator", "classifier_", "classifier", "model_"):
            nested = getattr(current, attribute, None)
            if nested is not None and nested is not current:
                current = nested
                break
        else:
            return current
    return current


def get_lightgbm_booster(estimator: object) -> object:
    """Return a fitted native Booster suitable for SHAP/native importance."""
    classifier = unwrap_lightgbm_classifier(estimator)
    booster = getattr(classifier, "booster_", None)
    if booster is None:
        raise ValueError("The LightGBM classifier is not fitted and has no native booster_.")
    return booster


def get_lightgbm_feature_importance(
    estimator: object,
    *,
    importance_type: str = "gain",
) -> object:
    """Return native feature importance from a fitted LightGBM estimator."""
    if importance_type not in {"gain", "split"}:
        raise ValueError("importance_type must be either 'gain' or 'split'.")
    booster = get_lightgbm_booster(estimator)
    feature_importance = getattr(booster, "feature_importance", None)
    if not callable(feature_importance):
        raise ValueError("The native LightGBM booster does not expose feature_importance().")
    return feature_importance(importance_type=importance_type)
