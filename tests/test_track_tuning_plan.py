from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotify.track_tuning_plan import (
    TRACK_TUNING_PLAN_SCHEMA_VERSION,
    TrackTuningPlanConfig,
    build_track_tuning_plan,
    write_track_tuning_plan,
)


def _study(
    model_name: str,
    *,
    completed: int,
    total: int,
    best_value: float | None,
    metric_name: str = "validation_ndcg_at_10",
    duration_seconds: float | None = 1.0,
    failed: int = 0,
    pruned: int = 0,
) -> dict[str, object]:
    trials = []
    if duration_seconds is not None:
        trials = [
            {
                "number": index,
                "state": "complete" if index < completed else "fail",
                "value": best_value,
                "duration_seconds": duration_seconds,
            }
            for index in range(total)
        ]
    return {
        "model_name": model_name,
        "status": "budget_complete",
        "metric": {"name": metric_name, "direction": "maximize", "split": "temporal_validation"},
        "completed_trials": completed,
        "total_trials": total,
        "failed_trials": failed,
        "pruned_trials": pruned,
        "best_trial": (
            {
                "number": 0,
                "value": best_value,
                "metric_name": metric_name,
                "direction": "maximize",
                "params": {"learning_rate": 0.001},
            }
            if best_value is not None
            else None
        ),
        "trials": trials,
    }


def _summary(studies: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "complete",
        "generated_at": "2026-06-18T12:00:00+00:00",
        "schema_version": "track-expansion-optuna-v1",
        "studies": studies,
        "summary_path": "track_expansion_optuna_summary.json",
    }


def test_build_track_tuning_plan_recommends_resumable_total_budgets(tmp_path: Path) -> None:
    summary_path = tmp_path / "track_expansion_optuna_summary.json"
    summary_path.write_text(
        json.dumps(
            _summary(
                [
                    _study(
                        "session_cooccurrence",
                        completed=2,
                        total=2,
                        best_value=0.04,
                        metric_name="validation_recall_at_100",
                        duration_seconds=0.2,
                    ),
                    _study(
                        "dcn_v2",
                        completed=6,
                        total=6,
                        best_value=0.7,
                        duration_seconds=12.0,
                    ),
                    _study(
                        "meantime",
                        completed=0,
                        total=1,
                        best_value=None,
                        duration_seconds=None,
                    ),
                ]
            )
        ),
        encoding="utf-8",
    )

    plan = build_track_tuning_plan(optuna_summary_path=summary_path)
    payload = plan.to_dict()
    by_model = {row["model_name"]: row for row in payload["model_plans"]}

    assert payload["schema_version"] == TRACK_TUNING_PLAN_SCHEMA_VERSION
    assert payload["status"] == "ready"
    assert by_model["session_cooccurrence"]["recommended_additional_trials"] == 16
    assert by_model["session_cooccurrence"]["recommended_total_trials"] == 18
    assert by_model["dcn_v2"]["recommended_total_trials"] == 8
    assert by_model["dcn_v2"]["priority"] == "high"
    assert by_model["meantime"]["recommended_total_trials"] == 9
    assert json.loads(json.dumps(payload)) == payload


def test_make_commands_are_shell_safe_and_grouped_by_target_total(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            _summary(
                [
                    _study("ease", completed=1, total=1, best_value=0.03, duration_seconds=0.1),
                    _study(
                        "session_cooccurrence",
                        completed=1,
                        total=1,
                        best_value=0.03,
                        duration_seconds=0.1,
                    ),
                    _study("dcn-v2", completed=4, total=4, best_value=0.5, duration_seconds=20.0),
                ]
            )
        ),
        encoding="utf-8",
    )

    plan = build_track_tuning_plan(optuna_summary_path=summary_path)
    commands = [command.to_dict() for command in plan.make_commands]

    assert commands
    assert plan.make_command == commands[0]["command"]
    assert all(command["command"].startswith("make train-recommender-next-pass EXTRA_ARGS=") for command in commands)
    assert all("\n" not in command["command"] and ";" not in command["command"] for command in commands)
    assert any(command["model_names"] == ["ease", "session_cooccurrence"] for command in commands)
    assert "dcn_v2" in {model for command in commands for model in command["model_names"]}


def test_manifest_discovers_optuna_summary_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    summary_path = (
        output_dir
        / "analysis"
        / "recommender_expansion"
        / "next_pass"
        / "stages"
        / "optuna_track_expansion"
        / "track_expansion_optuna_summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(_summary([_study("ple", completed=1, total=1, best_value=0.18, duration_seconds=5.0)])),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "next_pass_manifest.json"
    manifest_path.write_text(
        json.dumps({"config": {"output_dir": str(output_dir)}, "stages": {"optuna_tuning": {"status": "complete"}}}),
        encoding="utf-8",
    )

    plan = build_track_tuning_plan(next_pass_manifest_path=manifest_path)

    assert plan.optuna_summary_path == str(summary_path)
    assert plan.source_status == "complete"
    assert plan.model_plans[0].model_name == "ple"


def test_missing_summary_returns_blocked_plan(tmp_path: Path) -> None:
    plan = build_track_tuning_plan(optuna_summary_path=tmp_path / "missing.json")

    assert plan.status == "blocked"
    assert plan.make_command is None
    assert plan.model_plans == ()
    assert "unreadable" in plan.warnings[0]


def test_write_track_tuning_plan_persists_json(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary([_study("mmoe", completed=2, total=2, best_value=0.21, duration_seconds=4.0)])),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"

    plan = write_track_tuning_plan(plan_path, optuna_summary_path=summary_path)

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload == plan.to_dict()
    assert payload["model_plans"][0]["model_family"] == "multitask"


def test_config_validation_rejects_inverted_runtime_thresholds() -> None:
    with pytest.raises(ValueError, match="cheap_trial_seconds"):
        TrackTuningPlanConfig(cheap_trial_seconds=5.0, expensive_trial_seconds=1.0).validate()
