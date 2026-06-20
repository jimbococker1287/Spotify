from __future__ import annotations

import json
from pathlib import Path

from spotify.track_expansion_gates import evaluate_track_expansion_gates
from spotify.track_model_promotion_evidence import (
    PromotionEvidenceConfig,
    build_track_model_promotion_evidence,
    load_track_model_promotion_evidence,
    write_track_model_promotion_evidence,
)


def _training_manifest(tmp_path: Path) -> dict[str, object]:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("meantime", "mmoe", "ple"):
        (checkpoint_dir / f"{name}.keras").write_text("model", encoding="utf-8")
    return {
        "status": "complete",
        "config": {"random_seed": 42, "epochs": 1, "sequence_length": 64},
        "dataset": {
            "train_examples": 100,
            "validation_examples": 30,
            "test_examples": 25,
        },
        "tensor_summary": {
            "train_rows": 80,
            "validation_rows": 20,
            "test_rows": 15,
            "validation_target_vocabulary_coverage": 0.80,
            "test_target_vocabulary_coverage": 0.70,
        },
        "retrieval_results": [
            {
                "model_name": "session_cooccurrence",
                "status": "complete",
                "training_interactions": 120,
                "evaluated_examples": 30,
                "recall_at_k": 0.20,
                "ndcg_at_k": 0.10,
                "mrr_at_k": 0.06,
                "target_catalog_coverage": 0.75,
                "k": 100,
            }
        ],
        "neural_results": [
            {
                "model_name": "meantime",
                "status": "complete",
                "checkpoint": str(checkpoint_dir / "meantime.keras"),
                "validation": {
                    "recall_at_k": 0.18,
                    "ndcg_at_k": 0.09,
                    "target_vocabulary_coverage": 0.80,
                    "k": 100,
                },
                "test": {
                    "recall_at_k": 0.16,
                    "ndcg_at_k": 0.08,
                    "target_vocabulary_coverage": 0.70,
                    "k": 100,
                },
            },
            {
                "model_name": "mmoe",
                "status": "complete",
                "checkpoint": str(checkpoint_dir / "mmoe.keras"),
                "validation": {
                    "recall_at_k": 0.19,
                    "target_vocabulary_coverage": 0.80,
                    "k": 100,
                },
                "test": {
                    "recall_at_k": 0.17,
                    "target_vocabulary_coverage": 0.70,
                    "k": 100,
                },
            },
            {
                "model_name": "ple",
                "status": "complete",
                "checkpoint": str(checkpoint_dir / "ple.keras"),
                "validation": {
                    "recall_at_k": 0.17,
                    "target_vocabulary_coverage": 0.80,
                    "k": 100,
                },
                "test": {
                    "recall_at_k": 0.15,
                    "target_vocabulary_coverage": 0.70,
                    "k": 100,
                },
            },
        ],
    }


def _baseline() -> dict[str, object]:
    return {"status": "complete", "model": "track_popularity", "recall_at_k": 0.10, "k": 100}


def _tuning_manifest(tmp_path: Path) -> dict[str, object]:
    return {
        "status": "complete",
        "storage_path": str(tmp_path / "track_expansion_optuna.db"),
        "studies": [
            {
                "model_name": "meantime",
                "completed_trials": 3,
                "total_trials": 3,
                "best_trial": {
                    "metric_name": "validation_recall_at_100",
                    "params": {"learning_rate": 0.001},
                    "value": 0.18,
                },
            },
            {
                "model_name": "dcn_v2",
                "completed_trials": 3,
                "best_trial": {"params": {"learning_rate": 0.001}},
            },
        ],
    }


def _check(decision: object, name: str) -> object:
    return next(check for check in decision.checks if check.name == name)  # type: ignore[attr-defined]


def test_builds_gate_ready_evidence_with_aliases_and_truthful_placeholders(tmp_path: Path) -> None:
    evidence = build_track_model_promotion_evidence(
        training_manifest=_training_manifest(tmp_path),
        baseline_manifest=_baseline(),
        tuning_manifest=_tuning_manifest(tmp_path),
        config=PromotionEvidenceConfig(
            generated_at="2026-06-18T12:00:00+00:00",
            code_version="abc123",
            dataset_fingerprint="sha256:data",
        ),
    )

    assert evidence.status == "warn"
    assert tuple(model.model_name for model in evidence.models) == (
        "meantime",
        "mmoe",
        "ple",
        "session_cooccurrence",
    )
    meantime = evidence.gate_training_manifest["neural_results"][0]  # type: ignore[index]
    assert meantime["validation"]["recall_at_100"] == 0.18  # type: ignore[index]
    assert meantime["test"]["recall_at_100"] == 0.16  # type: ignore[index]
    assert meantime["popularity_baseline"]["recall_at_100"] == 0.10  # type: ignore[index]
    assert meantime["reproducibility"]["random_seed"] == 42  # type: ignore[index]
    assert meantime["calibration"]["status"] == "unavailable"  # type: ignore[index]
    assert meantime["explainability"]["method"] == "integrated_gradients"  # type: ignore[index]
    tuning_rows = evidence.gate_tuning_manifest["tuning_results"]  # type: ignore[index]
    assert tuning_rows == [
        {
            "model_name": "meantime",
            "completed_trials": 3,
            "trial_count": 3,
            "best_params": {"learning_rate": 0.001},
            "parameters": {"learning_rate": 0.001},
            "tuning_metric": "validation_recall_at_100",
            "tuning_value": 0.18,
            "storage_path": str(tmp_path / "track_expansion_optuna.db"),
        }
    ]

    report = evaluate_track_expansion_gates(
        baseline_manifest=evidence.baseline_manifest,
        training_manifest=evidence.gate_training_manifest,
        tuning_manifest=evidence.gate_tuning_manifest,
    )
    by_name = {decision.model_name: decision for decision in report.decisions}
    assert by_name["meantime"].status == "warn"
    assert _check(by_name["meantime"], "temporal_metrics").status == "pass"
    assert _check(by_name["meantime"], "model_artifact").status == "pass"
    assert _check(by_name["meantime"], "reproducibility").status == "pass"
    assert _check(by_name["meantime"], "calibration").status == "warn"
    assert by_name["session_cooccurrence"].status == "fail"
    assert _check(by_name["session_cooccurrence"], "temporal_metrics").status == "fail"


def test_preserves_complete_evidence_and_allows_gate_promotion(tmp_path: Path) -> None:
    manifest = _training_manifest(tmp_path)
    explanation = tmp_path / "meantime_ig.json"
    explanation.write_text("explanation", encoding="utf-8")
    manifest["neural_results"] = [
        {
            **manifest["neural_results"][0],  # type: ignore[index]
            "calibration": {"status": "complete", "ece": 0.04},
            "explainability": {
                "status": "complete",
                "method": "integrated_gradients",
                "artifact_path": str(explanation),
            },
            "drift": {"status": "complete", "target_drift_jsd": 0.05},
        }
    ]
    manifest["retrieval_results"] = []

    evidence = build_track_model_promotion_evidence(
        training_manifest=manifest,
        baseline_manifest=_baseline(),
        config=PromotionEvidenceConfig(
            generated_at="2026-06-18T12:00:00+00:00",
            code_version="abc123",
            dataset_fingerprint="sha256:data",
        ),
    )
    report = evaluate_track_expansion_gates(
        baseline_manifest=evidence.baseline_manifest,
        training_manifest=evidence.gate_training_manifest,
    )

    assert evidence.status == "ready"
    assert report.promoted_models == ("meantime",)
    assert report.status == "pass"


def test_retrieval_rows_do_not_invent_test_metrics_or_model_artifacts(tmp_path: Path) -> None:
    manifest = _training_manifest(tmp_path)
    manifest["neural_results"] = []

    evidence = build_track_model_promotion_evidence(
        training_manifest=manifest,
        baseline_manifest=_baseline(),
        config=PromotionEvidenceConfig(
            generated_at="2026-06-18T12:00:00+00:00",
            code_version="abc123",
            dataset_fingerprint="sha256:data",
        ),
    )

    retrieval = evidence.gate_training_manifest["retrieval_results"][0]  # type: ignore[index]
    assert retrieval["validation"]["recall_at_100"] == 0.20  # type: ignore[index]
    assert retrieval["test"] == {}
    assert retrieval["artifact_present"] is False
    assert "artifacts" not in retrieval
    assert "no temporal test metrics" in retrieval["evidence_notes"][0]  # type: ignore[index]


def test_load_and_write_round_trip_json_and_markdown(tmp_path: Path) -> None:
    training_path = tmp_path / "training_manifest.json"
    baseline_path = tmp_path / "baseline.json"
    tuning_path = tmp_path / "tuning.json"
    training_path.write_text(json.dumps(_training_manifest(tmp_path)), encoding="utf-8")
    baseline_path.write_text(json.dumps(_baseline()), encoding="utf-8")
    tuning_path.write_text(json.dumps(_tuning_manifest(tmp_path)), encoding="utf-8")

    evidence = load_track_model_promotion_evidence(
        training_manifest_path=training_path,
        baseline_manifest_path=baseline_path,
        tuning_manifest_path=tuning_path,
        config=PromotionEvidenceConfig(
            generated_at="2026-06-18T12:00:00+00:00",
            code_version="abc123",
            dataset_fingerprint="sha256:data",
        ),
    )
    json_path, markdown_path = write_track_model_promotion_evidence(evidence, tmp_path / "evidence")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model_count"] == 4
    assert payload["gate_training_manifest"]["neural_results"][0]["model_name"] == "meantime"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Track Model Promotion Evidence" in markdown
    assert "| `meantime` | `neural_sequence` |" in markdown
