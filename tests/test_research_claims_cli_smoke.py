from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


def test_research_claims_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.research_claims", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "research-claim pack" in result.stdout.lower()
    assert "--benchmark-manifest" in result.stdout
    assert "--run-dir" in result.stdout


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_research_platform_lab_cli_smoke_runs_on_minimal_fixture(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_cli"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_cli",
            "profile": "small",
            "timestamp": "2026-05-03T12:00:00",
            "champion_gate": {"status": "pass", "promoted": False},
        },
    )
    _write_json(run_dir / "run_results.json", [{"model_name": "gru_artist", "test_top1": 0.14}])
    _write_json(run_dir / "benchmark_protocol.json", {"benchmark_contract": {"contract_version": "v5", "comparison_mode": "repeated_seed_lock"}})
    _write_json(run_dir / "safety_platform_contract.json", {"reuse_summary": {"api_group_count": 1, "wrapper_count": 1}})
    _write_json(analysis_dir / "data_drift_summary.json", {"target_drift": {"train_vs_test_jsd": 0.11}})
    _write_json(analysis_dir / "robustness_summary.json", [{"worst_segment": "repeat_from_prev"}])
    _write_json(analysis_dir / "moonshot_summary.json", {"stress_worst_skip_risk": 0.35})
    _write_json(
        output_dir / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": "run_cli"},
            "primary_claim": {"key": "shift_robustness"},
            "claims": [
                {
                    "key": "shift_robustness",
                    "title": "Shift robustness",
                    "status": "analysis_ready",
                    "summary": "Core evidence is present.",
                    "metrics": {"target_drift_jsd": 0.11},
                    "supporting_artifacts": [str((run_dir / "run_results.json").resolve())],
                    "missing_checks": [],
                }
            ],
            "claim_support_matrix": [
                {
                    "claim_key": "shift_robustness",
                    "live_signal_status": "ready",
                    "benchmark_evidence_status": "n/a",
                    "repeated_evidence_status": "ready",
                    "slice_evidence_status": "ready",
                    "risk_evidence_status": "ready",
                    "artifact_pack_status": "ready",
                    "next_gate": "ready_to_package",
                }
            ],
            "submission_readiness": {"status": "analysis_ready", "ready_for_external_review": True, "blockers": []},
        },
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "spotify.research_platform_lab", "--output-dir", str(output_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "research_platform_maturity.md" in result.stdout
    assert "run_research_registry.csv" in result.stdout


def test_research_claims_cli_smoke_writes_conservative_readiness_and_portable_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_cli_claims"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_cli_claims",
            "profile": "full",
            "timestamp": "2026-05-04T12:00:00+00:00",
        },
    )
    _write_json(
        run_dir / "run_results.json",
        [
            {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.24, "test_top1": 0.30},
            {"model_name": "gru_artist", "model_type": "deep", "val_top1": 0.11, "test_top1": 0.17},
        ],
    )
    _write_json(
        analysis_dir / "data_drift_summary.json",
        {
            "target_drift": {"train_vs_test_jsd": 0.21},
            "largest_segment_shift": {"segment": "repeat_from_prev", "bucket": "new", "abs_share_shift": 0.22},
            "drift_interpretation": {"dominant_context_driver": "behavioral"},
        },
    )
    _write_json(
        analysis_dir / "robustness_summary.json",
        [
            {
                "model_name": "retrieval_reranker",
                "max_top1_gap": 0.57,
                "worst_segment": "repeat_from_prev",
                "worst_bucket": "new",
                "worst_bucket_count": 1200,
                "worst_bucket_share": 0.44,
            }
        ],
    )
    _write_json(
        analysis_dir / "moonshot_summary.json",
        {"stress_worst_skip_scenario": "friction_spike", "stress_worst_skip_risk": 0.61},
    )
    _write_json(
        analysis_dir / "retrieval_reranker_retrieval_reranker_confidence_summary.json",
        {"test_selective_risk": 0.68, "test_abstention_rate": 0.0},
    )
    _write_json(
        analysis_dir / "retrieval_reranker_retrieval_reranker_conformal_summary.json",
        {"test": {"coverage": 0.86, "abstention_rate": 0.0, "selective_risk": 0.68}},
    )
    _write_json(
        analysis_dir / "friction_proxy_summary.json",
        {
            "auc_lift": {"test": 0.0},
            "baseline_model": {"test_auc": 1.0},
            "full_model": {"test_auc": 1.0},
            "proxy_counterfactual": {"test_mean_delta": 0.0},
        },
    )
    (analysis_dir / "ablation_summary.csv").write_text("group,model_name\n", encoding="utf-8")
    (analysis_dir / "backtest_significance.csv").write_text("left_model,right_model\n", encoding="utf-8")

    for run_id, gap, drift in (
        ("run_cli_claims_prev_1", 0.49, 0.20),
        ("run_cli_claims_prev_2", 0.41, 0.19),
    ):
        prior_run_dir = output_dir / "runs" / run_id
        prior_analysis = prior_run_dir / "analysis"
        prior_analysis.mkdir(parents=True, exist_ok=True)
        _write_json(
            prior_run_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "profile": "full",
                "timestamp": f"2026-05-03T12:00:0{1 if run_id.endswith('1') else 2}+00:00",
            },
        )
        _write_json(
            prior_analysis / "robustness_summary.json",
            [
                {
                    "model_name": "retrieval_reranker",
                    "max_top1_gap": gap,
                    "worst_segment": "repeat_from_prev",
                    "worst_bucket": "new",
                    "worst_bucket_count": 1100,
                    "worst_bucket_share": 0.42,
                }
            ],
        )
        _write_json(
            prior_analysis / "data_drift_summary.json",
            {
                "target_drift": {"train_vs_test_jsd": drift},
                "largest_segment_shift": {"segment": "repeat_from_prev", "bucket": "new", "abs_share_shift": 0.18},
                "drift_interpretation": {"dominant_context_driver": "behavioral"},
            },
        )

    history_dir = output_dir / "history"
    benchmark_manifest = history_dir / "benchmark_lock_demo_manifest.json"
    _write_csv(
        history_dir / "benchmark_lock_demo_summary.csv",
        [
            {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "run_count": 3, "val_top1_mean": 0.24, "test_top1_mean": 0.29, "val_top1_ci95": 0.01},
            {"model_name": "extra_trees", "model_type": "classical", "run_count": 3, "val_top1_mean": 0.19, "test_top1_mean": 0.17, "val_top1_ci95": 0.01},
        ],
    )
    _write_csv(
        history_dir / "benchmark_lock_demo_significance.csv",
        [
            {
                "left_model": "retrieval_reranker",
                "right_model": "extra_trees",
                "shared_runs": 3,
                "mean_diff_val_top1": 0.05,
                "ci95_diff_val_top1": 0.02,
                "z_score": 2.8,
                "significant_at_95": 1,
            }
        ],
    )
    _write_json(
        benchmark_manifest,
        {
            "benchmark_id": "demo",
            "comparison_ready": False,
            "model_class_mix": {
                "declared": {"expected_model_classes": ["candidate", "classical", "deep"], "research_grade": True},
                "observed": {"model_classes": ["candidate", "classical"]},
            },
            "comparator_guard": {
                "research_grade": True,
                "requires_deep_comparator": True,
                "deep_comparator_ready": False,
                "detail": "Research-grade comparator guard failed: no repeated deep comparator appears in the benchmark summary with at least `3` run(s).",
            },
            "summary": ["Benchmark lock is not comparison-ready because it lacks repeated deep comparator evidence."],
        },
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spotify.research_claims",
            "--output-dir",
            str(output_dir),
            "--run-dir",
            str(run_dir),
            "--benchmark-manifest",
            str(benchmark_manifest),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "research_claims_json=" in result.stdout
    claims_payload = json.loads((output_dir / "analysis" / "research_claims" / "research_claims.json").read_text(encoding="utf-8"))
    assert claims_payload["submission_readiness"]["status"] == "promising_but_unlocked"
    assert claims_payload["submission_readiness"]["ready_for_external_review"] is False
    candidate_claim = next(claim for claim in claims_payload["claims"] if claim["key"] == "candidate_ranking")
    assert candidate_claim["supporting_artifacts_portable"] == [
        "runs/run_cli_claims/run_results.json",
        "runs/run_cli_claims/analysis/ablation_summary.csv",
        "runs/run_cli_claims/analysis/backtest_significance.csv",
        "history/benchmark_lock_demo_manifest.json",
    ]
    primary_support = next(row for row in claims_payload["claim_support_matrix"] if row["role"] == "primary")
    assert primary_support["artifact_portability_status"] == "ready"
