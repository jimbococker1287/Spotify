from __future__ import annotations

import json
from pathlib import Path

import pytest

import spotify.track_expansion_tuning as tuning
from spotify.track_expansion_tuning import (
    ObjectiveResult,
    SUPPORTED_TRACK_EXPANSION_MODELS,
    TrackExpansionTuningConfig,
    get_track_expansion_search_space,
    run_track_expansion_tuning,
    suggest_track_expansion_params,
    validate_track_expansion_tuning,
)


class _FirstChoiceTrial:
    number = 0

    def suggest_categorical(self, _name, choices):
        return choices[0]

    def suggest_float(self, _name, low, _high, **_kwargs):
        return low

    def suggest_int(self, _name, low, _high, **_kwargs):
        return low


@pytest.mark.parametrize("model_name", SUPPORTED_TRACK_EXPANSION_MODELS)
def test_all_track_models_have_temporal_search_space_contracts(model_name: str) -> None:
    contract = get_track_expansion_search_space(model_name)
    params = suggest_track_expansion_params(_FirstChoiceTrial(), model_name)

    assert contract.model_name == model_name
    assert contract.metric.split == "temporal_validation"
    assert contract.metric.direction == "maximize"
    assert params
    assert set(params) == {parameter.name for parameter in contract.parameters}


def test_dcn_search_space_decodes_json_safe_deep_units() -> None:
    contract = get_track_expansion_search_space("dcn-v2")
    params = suggest_track_expansion_params(_FirstChoiceTrial(), "dcn_v2_reranker")
    raw_parameter = next(parameter for parameter in contract.parameters if parameter.name == "deep_units")

    assert raw_parameter.choices == ("64x32", "128x64", "256x128x64")
    assert params["deep_units"] == (64, 32)


def test_validation_is_independent_of_optuna_import(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tuning, "_load_optuna", lambda: None)
    config = TrackExpansionTuningConfig(
        storage_path=tmp_path / "studies.sqlite3",
        selected_models=("ease",),
        trial_budgets=2,
    )

    request = validate_track_expansion_tuning(
        objectives={"ease": lambda _context: 0.5},
        config=config,
    )
    result = run_track_expansion_tuning(
        objectives={"ease": lambda _context: 0.5},
        config=config,
    )

    assert request.trial_budgets["ease"] == 2
    assert result.status == "dependency_unavailable"
    assert result.studies[0].status == "dependency_unavailable"
    assert Path(result.summary_path).exists()
    assert json.loads(Path(result.summary_path).read_text())["dependency_error"]


def test_sqlite_study_resumes_to_total_trial_budget_and_writes_summaries(
    tmp_path: Path,
) -> None:
    pytest.importorskip("optuna")
    calls: list[int] = []

    def objective(context):
        calls.append(context.trial.number)
        return ObjectiveResult(
            value=float(context.params["shrinkage"]),
            metadata={"validation_window": "2025Q4"},
        )

    first = run_track_expansion_tuning(
        objectives={"session_cooccurrence": objective},
        config=TrackExpansionTuningConfig(
            storage_path=tmp_path / "studies.sqlite3",
            output_dir=tmp_path / "summaries",
            selected_models=("session_cooccurrence",),
            trial_budgets=2,
            sampler_seed=17,
            pruner="none",
        ),
    )
    second = run_track_expansion_tuning(
        objectives={"session_cooccurrence": objective},
        config=TrackExpansionTuningConfig(
            storage_path=tmp_path / "studies.sqlite3",
            output_dir=tmp_path / "summaries",
            selected_models=("session_cooccurrence",),
            trial_budgets=3,
            sampler_seed=17,
            pruner="none",
        ),
    )

    assert first.studies[0].total_trials == 2
    assert second.studies[0].existing_trials == 2
    assert second.studies[0].executed_trials == 1
    assert second.studies[0].total_trials == 3
    assert calls == [0, 1, 2]
    assert second.studies[0].best_trial is not None
    assert Path(second.studies[0].trial_table_path).exists()
    best_payload = json.loads(Path(second.studies[0].best_trial_path).read_text())
    assert best_payload["best_trial"]["params"]
    assert second.studies[0].trials[-1].user_attrs["validation_window"] == "2025Q4"


def test_sampler_seed_produces_same_parameter_sequence_across_new_studies(
    tmp_path: Path,
) -> None:
    pytest.importorskip("optuna")

    def objective(context):
        return float(context.params["l2"])

    results = []
    for index in range(2):
        results.append(
            run_track_expansion_tuning(
                objectives={"ease": objective},
                config=TrackExpansionTuningConfig(
                    storage_path=tmp_path / f"study-{index}.sqlite3",
                    output_dir=tmp_path / f"summary-{index}",
                    selected_models=("ease",),
                    trial_budgets=4,
                    sampler_seed=73,
                    pruner="none",
                ),
            )
        )

    first_params = [trial.params for trial in results[0].studies[0].trials]
    second_params = [trial.params for trial in results[1].studies[0].trials]
    assert first_params == second_params


def test_intermediate_metrics_enable_optuna_pruning(tmp_path: Path) -> None:
    pytest.importorskip("optuna")

    def objective(context):
        score = 0.9 if context.trial.number == 0 else 0.1
        return ObjectiveResult(
            value=score,
            intermediate_values=((0, score), (1, score)),
        )

    result = run_track_expansion_tuning(
        objectives={"ease": objective},
        config=TrackExpansionTuningConfig(
            storage_path=tmp_path / "pruning.sqlite3",
            output_dir=tmp_path / "pruning-summary",
            selected_models=("ease",),
            trial_budgets=2,
            sampler_seed=4,
            pruner="median",
            median_startup_trials=0,
            median_warmup_steps=0,
        ),
    )

    assert result.studies[0].completed_trials == 1
    assert result.studies[0].pruned_trials == 1
    assert [trial.state for trial in result.studies[0].trials] == ["complete", "pruned"]


def test_validation_rejects_missing_callbacks_and_bad_budgets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing objective callback for ple"):
        validate_track_expansion_tuning(
            objectives={"mmoe": lambda _context: 0.5},
            config=TrackExpansionTuningConfig(
                storage_path=tmp_path / "studies.db",
                selected_models=("mmoe", "ple"),
                trial_budgets=1,
            ),
        )

    with pytest.raises(ValueError, match="non-negative integer"):
        validate_track_expansion_tuning(
            objectives={"mmoe": lambda _context: 0.5},
            config=TrackExpansionTuningConfig(
                storage_path=tmp_path / "studies.db",
                selected_models=("mmoe",),
                trial_budgets={"mmoe": -1},
            ),
        )


def test_existing_study_direction_must_match_temporal_metric(tmp_path: Path) -> None:
    optuna = pytest.importorskip("optuna")
    storage_path = tmp_path / "wrong-direction.sqlite3"
    optuna.create_study(
        study_name="track_expansion_ease",
        storage=f"sqlite:///{storage_path.as_posix()}",
        direction="minimize",
    )

    with pytest.raises(RuntimeError, match="requires 'maximize'"):
        run_track_expansion_tuning(
            objectives={"ease": lambda _context: 0.5},
            config=TrackExpansionTuningConfig(
                storage_path=storage_path,
                output_dir=tmp_path / "summary",
                selected_models=("ease",),
                trial_budgets=1,
            ),
        )
