from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.research_platform_lab import build_research_platform_lab


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_research_platform_lab_creates_registries_and_maturity_brief(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.research_platform_lab")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_a",
            "profile": "full",
            "timestamp": "2026-05-01T12:00:00",
            "champion_gate": {"status": "pass", "promoted": True},
        },
    )
    _write_json(run_dir / "run_results.json", [{"model_name": "blended_ensemble", "test_top1": 0.50}])
    _write_json(
        run_dir / "benchmark_protocol.json",
        {
            "benchmark_contract": {"contract_version": "v4", "comparison_mode": "repeated_runs"},
            "protocol": {"temporal_backtest": {"models": ["blended_ensemble", "retrieval_reranker"]}},
        },
    )
    _write_json(
        run_dir / "safety_platform_contract.json",
        {
            "reuse_summary": {"api_group_count": 3, "wrapper_count": 2},
        },
    )
    _write_json(analysis_dir / "data_drift_summary.json", {"target_drift": {"train_vs_test_jsd": 0.218}})
    _write_json(analysis_dir / "robustness_summary.json", [{"worst_segment": "repeat_from_prev"}])
    _write_json(analysis_dir / "moonshot_summary.json", {"stress_worst_skip_risk": 0.59})
    _write_json(
        analysis_dir / "ensemble_blended_ensemble_conformal_summary.json",
        {
            "test": {"selective_risk": 0.39, "abstention_rate": 0.19, "accepted_rate": 0.81},
        },
    )

    _write_csv(
        output_dir / "analytics" / "control_room_history.csv",
        [
            {
                "run_id": "run_a",
                "target_drift_jsd": 0.218,
                "test_selective_risk": 0.39,
                "test_abstention_rate": 0.19,
                "robustness_gap": 0.10,
                "stress_skip_risk": 0.59,
                "ops_coverage_ratio": 1.0,
            }
        ],
    )
    _write_json(
        output_dir / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": "run_a"},
            "primary_claim": {"key": "shift_robustness"},
            "claims": [
                {
                    "key": "shift_robustness",
                    "title": "Shift robustness",
                    "status": "analysis_ready",
                    "summary": "Repeated slice risk is measurable.",
                    "metrics": {
                        "target_drift_jsd": 0.218,
                        "selective_risk": 0.39,
                        "stress_skip_risk": 0.59,
                    },
                    "supporting_artifacts": ["a.json"],
                    "missing_checks": [],
                }
            ],
            "claim_support_matrix": [
                {
                    "claim_key": "shift_robustness",
                    "live_signal_status": "ready",
                    "benchmark_evidence_status": "ready",
                    "repeated_evidence_status": "ready",
                    "slice_evidence_status": "ready",
                    "risk_evidence_status": "ready",
                    "artifact_pack_status": "ready",
                    "next_gate": "ready_to_package",
                }
            ],
            "submission_readiness": {
                "status": "analysis_ready",
                "ready_for_external_review": False,
                "blockers": ["Add repeated deep comparator"],
            },
        },
    )
    _write_json(
        output_dir / "history" / "benchmark_lock_smokebench_manifest.json",
        {
            "benchmark_id": "smokebench",
            "canonical_profile": "small",
            "comparison_mode": "repeated_runs",
            "comparison_ready": True,
            "run_count": 3,
            "present_artifact_count": 7,
            "required_artifact_count": 7,
            "significant_pair_count": 1,
        },
    )
    _write_csv(
        output_dir / "history" / "benchmark_lock_smokebench_summary.csv",
        [
            {
                "benchmark_id": "smokebench",
                "model_name": "blended_ensemble",
                "model_type": "ensemble",
                "val_top1_mean": 0.40,
                "test_top1_mean": 0.51,
                "run_count": 3,
            },
            {
                "benchmark_id": "smokebench",
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "val_top1_mean": 0.37,
                "test_top1_mean": 0.49,
                "run_count": 3,
            },
        ],
    )
    _write_csv(
        output_dir / "history" / "benchmark_lock_smokebench_significance.csv",
        [
            {
                "left_model": "blended_ensemble",
                "right_model": "retrieval_reranker",
                "mean_diff_val_top1": 0.03,
                "z_score": 2.5,
                "significant_at_95": 1,
            }
        ],
    )

    paths = build_research_platform_lab(output_dir=output_dir, run_dir=None, logger=logger)

    assert paths
    result_root = output_dir / "analysis" / "research_platform_lab"
    run_registry = pd.read_csv(result_root / "run_research_registry.csv")
    benchmark_atlas = pd.read_csv(result_root / "benchmark_lock_atlas.csv")
    claim_registry = pd.read_csv(result_root / "research_claim_registry.csv")
    brief_text = (result_root / "research_platform_maturity.md").read_text(encoding="utf-8")

    assert run_registry.iloc[0]["run_id"] == "run_a"
    assert bool(run_registry.iloc[0]["benchmark_protocol_present"])
    assert benchmark_atlas.iloc[0]["benchmark_id"] == "smokebench"
    assert claim_registry.iloc[0]["claim_key"] == "shift_robustness"
    assert "Research Platform Maturity" in brief_text
