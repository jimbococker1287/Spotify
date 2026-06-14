from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotify.track_expansion_gates import PromotionPolicy, evaluate_track_expansion_gates


def _baseline() -> dict[str, object]:
    return {"model": "track_popularity", "recall_at_k": 0.10}


def _complete_training(tmp_path: Path, *, model_name: str = "meantime") -> dict[str, object]:
    checkpoint = tmp_path / f"{model_name}.keras"
    explanation = tmp_path / f"{model_name}_explanations.json"
    checkpoint.write_text("model", encoding="utf-8")
    explanation.write_text("explanations", encoding="utf-8")
    return {
        "config": {"random_seed": 42, "epochs": 2},
        "dataset_fingerprint": "sha256:dataset",
        "temporal_split": "session_end_time_64_16_20",
        "code_version": "abc123",
        "tensor_summary": {
            "train_rows": 1000,
            "validation_rows": 200,
            "test_rows": 250,
        },
        "neural_results": [
            {
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "validation": {"recall_at_k": 0.12, "target_vocabulary_coverage": 0.80},
                "test": {"recall_at_k": 0.11, "target_vocabulary_coverage": 0.75},
                "calibration": {"status": "complete", "ece": 0.04},
                "explainability": {
                    "status": "complete",
                    "method": "integrated_gradients",
                    "artifact_path": str(explanation),
                },
                "drift": {"status": "complete", "target_drift_jsd": 0.08},
            }
        ],
    }


def _check(decision: object, name: str) -> object:
    return next(check for check in decision.checks if check.name == name)  # type: ignore[attr-defined]


def test_fully_evidenced_candidate_is_promoted_and_serializable(tmp_path: Path) -> None:
    report = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        training_manifest=_complete_training(tmp_path),
    )

    assert report.status == "pass"
    assert report.promoted_models == ("meantime",)
    decision = report.decisions[0]
    assert decision.status == "pass"
    assert decision.promoted is True
    assert decision.blockers == ()
    assert _check(decision, "explainability").evidence["shap_artifact_status"] == "not_applicable"
    json.dumps(report.to_dict(), sort_keys=True)
    assert "# Track Expansion Promotion Gates" in report.to_markdown()
    assert "| `temporal_metrics` | `pass` |" in report.to_markdown()


def test_missing_required_evidence_fails_without_promotion() -> None:
    report = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        training_manifest={
            "neural_results": [
                {
                    "model_name": "meantime",
                    "validation": {"recall_at_k": 0.2},
                }
            ]
        },
    )

    decision = report.decisions[0]
    assert report.status == "fail"
    assert decision.promoted is False
    assert {
        "model_artifact",
        "temporal_metrics",
        "split_coverage",
        "calibration",
        "explainability",
        "drift",
        "reproducibility",
    }.issubset(decision.blockers)


@pytest.mark.parametrize("evidence_name", ["calibration", "drift", "explainability"])
def test_explicit_unavailable_evidence_warns_and_blocks(tmp_path: Path, evidence_name: str) -> None:
    training = _complete_training(tmp_path)
    candidate = training["neural_results"][0]  # type: ignore[index]
    candidate[evidence_name] = {"status": "unavailable", "reason": "not computed in this run"}  # type: ignore[index]

    decision = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        training_manifest=training,
    ).decisions[0]

    assert decision.status == "warn"
    assert decision.promoted is False
    assert _check(decision, evidence_name).status == "warn"
    assert evidence_name in decision.blockers


def test_regression_beyond_popularity_tolerance_fails(tmp_path: Path) -> None:
    training = _complete_training(tmp_path)
    candidate = training["neural_results"][0]  # type: ignore[index]
    candidate["test"]["recall_at_k"] = 0.08  # type: ignore[index]

    decision = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        training_manifest=training,
        policy=PromotionPolicy(max_popularity_regression=0.01),
    ).decisions[0]

    regression = _check(decision, "popularity_regression")
    assert decision.promoted is False
    assert regression.status == "fail"
    assert regression.evidence["test_delta"] == pytest.approx(-0.02)


def test_dcn_requires_shap_compatible_artifact(tmp_path: Path) -> None:
    manifest = _complete_training(tmp_path, model_name="dcn_v2")
    candidate = manifest["neural_results"][0]  # type: ignore[index]
    candidate["model_family"] = "dcn"  # type: ignore[index]

    first = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        dcn_manifest={"results": [candidate], **{key: value for key, value in manifest.items() if key != "neural_results"}},
    ).decisions[0]
    assert _check(first, "explainability").status == "fail"
    assert _check(first, "explainability").evidence["shap_artifact_status"] == "wrong_method"

    candidate["explainability"]["method"] = "gradient_shap"  # type: ignore[index]
    second = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        dcn_manifest={"results": [candidate], **{key: value for key, value in manifest.items() if key != "neural_results"}},
    ).decisions[0]
    assert _check(second, "explainability").status == "pass"
    assert _check(second, "explainability").evidence["shap_artifact_status"] == "present"
    assert second.promoted is True


def test_public_transfer_requires_validated_license_provenance(tmp_path: Path) -> None:
    manifest = _complete_training(tmp_path)
    candidate = manifest["neural_results"][0]  # type: ignore[index]
    public_manifest = {
        "config": manifest["config"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "temporal_split": manifest["temporal_split"],
        "code_version": manifest["code_version"],
        "tensor_summary": manifest["tensor_summary"],
        "transferred_models": [candidate],
        "public_datasets": [
            {
                "dataset_name": "LFM-1b",
                "version": "1.0",
                "license_name": "research license",
                "source_url": "https://example.test/lfm",
                "allowed_use": "research",
                "license_validated": True,
            }
        ],
    }

    failed = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        public_pretraining_manifest=public_manifest,
    ).decisions[0]
    assert failed.promoted is False
    assert _check(failed, "public_license_provenance").status == "fail"
    assert "datasets[0].spotify_platform_content" in _check(
        failed,
        "public_license_provenance",
    ).evidence["missing_or_invalid"]

    public_manifest["public_datasets"][0]["spotify_platform_content"] = False  # type: ignore[index]
    passed = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        public_pretraining_manifest=public_manifest,
    ).decisions[0]
    assert _check(passed, "public_license_provenance").status == "pass"
    assert passed.promoted is True


def test_tuned_candidate_requires_tuning_trials_and_best_params(tmp_path: Path) -> None:
    training = _complete_training(tmp_path)
    tuning = {
        "results": [
            {
                "model_name": "meantime",
                "completed_trials": 0,
                "best_params": {},
            }
        ]
    }

    failed = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        training_manifest=training,
        tuning_manifest=tuning,
    ).decisions[0]
    assert _check(failed, "reproducibility").status == "fail"

    tuning["results"][0]["completed_trials"] = 12  # type: ignore[index]
    tuning["results"][0]["best_params"] = {"learning_rate": 0.001}  # type: ignore[index]
    passed = evaluate_track_expansion_gates(
        baseline_manifest=_baseline(),
        training_manifest=training,
        tuning_manifest=tuning,
    ).decisions[0]
    assert _check(passed, "reproducibility").status == "pass"
    assert passed.sources == ("training", "tuning")
    assert passed.promoted is True


def test_candidate_order_and_output_are_deterministic(tmp_path: Path) -> None:
    first_training = _complete_training(tmp_path, model_name="ple")
    second_training = _complete_training(tmp_path, model_name="meantime")
    manifest = {
        **{key: value for key, value in first_training.items() if key != "neural_results"},
        "neural_results": [
            first_training["neural_results"][0],
            second_training["neural_results"][0],
        ],
    }

    first = evaluate_track_expansion_gates(baseline_manifest=_baseline(), training_manifest=manifest)
    second = evaluate_track_expansion_gates(baseline_manifest=_baseline(), training_manifest=manifest)

    assert tuple(decision.model_name for decision in first.decisions) == ("meantime", "ple")
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_empty_manifests_produce_failed_report_without_candidates() -> None:
    report = evaluate_track_expansion_gates(baseline_manifest=_baseline())

    assert report.status == "fail"
    assert report.decisions == ()
    assert report.promoted_models == ()


def test_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        PromotionPolicy(min_target_coverage=1.1)
