from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from spotify.recommender_next_pass import (
    LazyImportAdapter,
    NextPassAdapters,
    NextPassConfig,
    StageRequest,
    run_recommender_next_pass,
)


def _complete(name: str, calls: list[str]):
    def adapter(request: StageRequest) -> dict[str, object]:
        calls.append(name)
        return {
            "status": "complete",
            "artifact": request.artifact_dir / f"{name}.json",
            "upstream": sorted(request.upstream),
        }

    return adapter


def test_next_pass_runs_in_dependency_order_and_persists_outputs(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    observed_options: list[dict[str, object]] = []

    def dcn_adapter(request: StageRequest) -> dict[str, object]:
        calls.append("dcn")
        observed_options.append(dict(request.options))
        assert request.upstream["candidate_dataset"]["status"] == "complete"
        return {"status": "complete", "checkpoint": request.artifact_dir / "dcn.keras"}

    manifest = run_recommender_next_pass(
        config=NextPassConfig(
            output_dir=tmp_path,
            enable_public_pretraining=True,
            stage_options={"dcn_training": {"epochs": 2}},
        ),
        adapters=NextPassAdapters(
            candidate_dataset_builder=_complete("candidates", calls),
            dcn_trainer=dcn_adapter,
            optuna_tuners={
                "retrieval": _complete("tune_retrieval", calls),
                "dcn": _complete("tune_dcn", calls),
            },
            public_pretrainer=_complete("public", calls),
            promotion_gates=_complete("promotion", calls),
        ),
    )

    assert calls == [
        "candidates",
        "dcn",
        "tune_retrieval",
        "tune_dcn",
        "public",
        "promotion",
    ]
    assert observed_options == [{"epochs": 2}]
    assert manifest["status"] == "complete"
    stages = manifest["stages"]
    assert stages["optuna_tuning"]["status"] == "complete"
    assert set(stages["optuna_tuning"]["tuners"]) == {"retrieval", "dcn"}

    root = tmp_path / "analysis" / "recommender_expansion" / "next_pass"
    persisted = json.loads(
        (root / "next_pass_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "complete"
    checkpoint = persisted["stages"]["dcn_training"]["output"]["checkpoint"]
    assert checkpoint.endswith("/stages/dcn_training/dcn.keras")
    continuation = (root / "CONTINUE_NEXT_PASS.md").read_text(encoding="utf-8")
    assert "All configured stages completed" in continuation


def test_failed_dcn_preserves_candidate_result_and_blocks_dependents(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail_dcn(_request: StageRequest) -> object:
        calls.append("dcn")
        raise RuntimeError("training diverged")

    manifest = run_recommender_next_pass(
        config=NextPassConfig(output_dir=tmp_path),
        adapters=NextPassAdapters(
            candidate_dataset_builder=_complete("candidates", calls),
            dcn_trainer=fail_dcn,
            optuna_tuners={"all": _complete("tune", calls)},
            promotion_gates=_complete("promotion", calls),
        ),
    )

    assert calls == ["candidates", "dcn"]
    assert manifest["status"] == "partial"
    stages = manifest["stages"]
    assert stages["candidate_dataset"]["status"] == "complete"
    assert stages["candidate_dataset"]["output"]["artifact"].endswith(
        "/stages/candidate_dataset/candidates.json"
    )
    assert stages["dcn_training"]["status"] == "failed"
    assert stages["dcn_training"]["error"] == {
        "type": "RuntimeError",
        "message": "training diverged",
    }
    assert stages["optuna_tuning"]["status"] == "blocked"
    assert stages["public_pretraining"]["status"] == "skipped"
    assert stages["promotion_gates"]["status"] == "blocked"

    root = tmp_path / "analysis" / "recommender_expansion" / "next_pass"
    persisted = json.loads(
        (root / "next_pass_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["stages"]["candidate_dataset"]["status"] == "complete"
    continuation = (root / "CONTINUE_NEXT_PASS.md").read_text(encoding="utf-8")
    assert "Resume from the first unresolved stage: `dcn_training`" in continuation


def test_optuna_tuners_are_isolated_and_partial_stage_blocks_promotion(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail_tuner(_request: StageRequest) -> object:
        calls.append("tune_failed")
        raise ValueError("invalid search space")

    manifest = run_recommender_next_pass(
        config=NextPassConfig(
            output_dir=tmp_path,
            enable_public_pretraining=True,
        ),
        adapters=NextPassAdapters(
            candidate_dataset_builder=_complete("candidates", calls),
            dcn_trainer=_complete("dcn", calls),
            optuna_tuners={
                "failed": fail_tuner,
                "successful": _complete("tune_successful", calls),
            },
            public_pretrainer=_complete("public", calls),
            promotion_gates=_complete("promotion", calls),
        ),
    )

    assert calls == ["candidates", "dcn", "tune_failed", "tune_successful"]
    tuning = manifest["stages"]["optuna_tuning"]
    assert tuning["status"] == "partial"
    assert tuning["tuners"]["failed"]["status"] == "failed"
    assert tuning["tuners"]["successful"]["status"] == "complete"
    assert manifest["stages"]["public_pretraining"]["status"] == "blocked"
    assert manifest["stages"]["promotion_gates"]["status"] == "blocked"
    assert manifest["status"] == "partial"


def test_adapter_reported_blocked_status_stops_downstream_stages(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def blocked_candidates(_request: StageRequest) -> dict[str, str]:
        calls.append("candidates")
        return {
            "status": "blocked",
            "reason": "candidate source is not licensed",
        }

    manifest = run_recommender_next_pass(
        config=NextPassConfig(output_dir=tmp_path),
        adapters=NextPassAdapters(
            candidate_dataset_builder=blocked_candidates,
            dcn_trainer=_complete("dcn", calls),
            optuna_tuners={"all": _complete("tune", calls)},
            promotion_gates=_complete("promotion", calls),
        ),
    )

    assert calls == ["candidates"]
    assert manifest["status"] == "blocked"
    assert manifest["stages"]["candidate_dataset"]["status"] == "blocked"
    assert manifest["stages"]["dcn_training"]["status"] == "blocked"


def test_lazy_adapter_imports_on_invocation_and_adapts_named_arguments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    imported: list[str] = []

    def implementation(
        artifact_dir: Path,
        options: dict[str, object],
    ) -> dict[str, object]:
        return {
            "artifact_dir": artifact_dir,
            "epochs": options["epochs"],
        }

    def fake_import(module_name: str):
        imported.append(module_name)
        return SimpleNamespace(build=implementation)

    monkeypatch.setattr(
        "spotify.recommender_next_pass.import_module",
        fake_import,
    )
    adapter = LazyImportAdapter(
        module_names=("worker.candidates",),
        callable_names=("build",),
    )
    assert imported == []

    config = NextPassConfig(output_dir=tmp_path)
    result = adapter(
        StageRequest(
            stage_name="candidate_dataset",
            artifact_dir=tmp_path / "stage",
            manifest_path=tmp_path / "manifest.json",
            config=config,
            options={"epochs": 3},
            upstream={},
        )
    )

    assert imported == ["worker.candidates"]
    assert result == {
        "artifact_dir": tmp_path / "stage",
        "epochs": 3,
    }


def test_initial_checkpoint_lists_first_not_yet_run_stage(tmp_path: Path) -> None:
    calls: list[str] = []

    def inspect_initial_checkpoint(request: StageRequest) -> dict[str, object]:
        continuation = (
            request.config.artifact_root / "CONTINUE_NEXT_PASS.md"
        ).read_text(encoding="utf-8")
        assert "first unresolved stage: `candidate_dataset`" in continuation
        return _complete("candidates", calls)(request)

    run_recommender_next_pass(
        config=NextPassConfig(output_dir=tmp_path),
        adapters=NextPassAdapters(
            candidate_dataset_builder=inspect_initial_checkpoint,
            dcn_trainer=_complete("dcn", calls),
            optuna_tuners={"all": _complete("tune", calls)},
            promotion_gates=_complete("promotion", calls),
        ),
    )
