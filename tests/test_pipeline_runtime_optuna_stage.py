from __future__ import annotations

import logging
from types import SimpleNamespace

from spotify.pipeline_runtime_optuna_stage import run_optuna_tuning


class _PhaseRecorder:
    def __init__(self) -> None:
        self.skips: list[tuple[str, str]] = []

    def skip(self, phase_name: str, *, reason: str) -> None:
        self.skips.append((phase_name, reason))


def test_optuna_stage_skips_when_shortlist_is_empty() -> None:
    recorder = _PhaseRecorder()
    context = SimpleNamespace(
        run_classical_models=True,
        config=SimpleNamespace(
            enable_optuna=True,
            optuna_model_names=(),
            temporal_backtest_model_names=("logreg",),
        ),
        phase_recorder=recorder,
        logger=logging.getLogger("spotify.test.optuna.empty"),
    )
    deps = SimpleNamespace(
        run_optuna_tuning=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Tuning should not run without selected models")
        )
    )

    selected_backtest_models, tuned_specs = run_optuna_tuning(
        context=context,
        deps=deps,
        classical_feature_bundle=object(),
        classical_results=[],
        optuna_cache_stats={},
    )

    assert selected_backtest_models == ("logreg",)
    assert tuned_specs == {}
    assert recorder.skips == [("optuna_tuning", "no_models_selected")]
