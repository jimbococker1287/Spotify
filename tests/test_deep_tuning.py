from __future__ import annotations

import pytest

from spotify.deep_tuning import SUPPORTED_DEEP_OPTUNA_MODELS, suggest_deep_model_params


class _Trial:
    number = 0

    def suggest_categorical(self, _name, choices):
        return choices[0]

    def suggest_float(self, _name, low, _high, **_kwargs):
        return low

    def suggest_int(self, _name, low, _high, **_kwargs):
        return low


@pytest.mark.parametrize("model_name", SUPPORTED_DEEP_OPTUNA_MODELS)
def test_deep_models_have_optuna_search_spaces(model_name: str) -> None:
    params = suggest_deep_model_params(_Trial(), model_name)

    assert params


def test_unknown_deep_model_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unsupported deep Optuna model"):
        suggest_deep_model_params(_Trial(), "unknown")
