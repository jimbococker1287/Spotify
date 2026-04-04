from __future__ import annotations

import csv
import json
from pathlib import Path

from spotify.research_claims import build_research_claims_report, write_research_claims_report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fixture_outputs(tmp_path: Path) -> tuple[Path, Path]:
    outputs = tmp_path / "outputs"
    run_dir = outputs / "runs" / "run_full"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_full",
            "profile": "full",
            "timestamp": "2026-03-29T12:00:00+00:00",
        },
    )
    _write_json(
        run_dir / "run_results.json",
        [
            {"model_name": "blended_ensemble", "model_type": "ensemble", "val_top1": 0.41, "test_top1": 0.31},
            {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.23, "test_top1": 0.30},
            {"model_name": "transformer_xl", "model_type": "deep", "val_top1": 0.08, "test_top1": 0.05},
            {"model_name": "gru_artist", "model_type": "deep", "val_top1": 0.03, "test_top1": 0.02},
        ],
    )
    _write_json(
        analysis_dir / "data_drift_summary.json",
        {
            "target_drift": {"train_vs_test_jsd": 0.21},
            "largest_segment_shift": {"segment": "repeat_from_prev", "bucket": "new", "abs_share_shift": 0.22},
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
            }
        ],
    )
    _write_json(
        analysis_dir / "moonshot_summary.json",
        {
            "stress_worst_skip_scenario": "friction_spike",
            "stress_worst_skip_risk": 0.61,
        },
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
    _write_json(
        analysis_dir / "ensemble_blended_ensemble_confidence_summary.json",
        {
            "test_selective_risk": 0.68,
            "test_abstention_rate": 0.0,
        },
    )
    _write_json(
        analysis_dir / "ensemble_blended_ensemble_conformal_summary.json",
        {
            "test": {"coverage": 0.86, "abstention_rate": 0.0, "selective_risk": 0.68},
        },
    )

    for run_id, gap, drift in (
        ("run_full_prev_1", 0.49, 0.20),
        ("run_full_prev_2", 0.41, 0.19),
    ):
        prior_run_dir = outputs / "runs" / run_id
        prior_analysis = prior_run_dir / "analysis"
        prior_analysis.mkdir(parents=True, exist_ok=True)
        _write_json(
            prior_run_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "profile": "full",
                "timestamp": f"2026-03-28T12:00:0{1 if run_id.endswith('1') else 2}+00:00",
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
                    "worst_bucket_count": 1500,
                    "worst_bucket_share": 0.44,
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

    (analysis_dir / "ablation_summary.csv").write_text("group,model_name\n", encoding="utf-8")
    (analysis_dir / "backtest_significance.csv").write_text("left_model,right_model\n", encoding="utf-8")

    history_dir = outputs / "history"
    benchmark_manifest = history_dir / "benchmark_lock_demo_manifest.json"
    _write_csv(
        history_dir / "benchmark_lock_demo_summary.csv",
        [
            {"model_name": "extra_trees", "model_type": "classical", "run_count": 2, "val_top1_mean": 0.23, "test_top1_mean": 0.15, "val_top1_ci95": 0.01},
            {"model_name": "gru_artist", "model_type": "deep", "run_count": 2, "val_top1_mean": 0.16, "test_top1_mean": 0.15, "val_top1_ci95": 0.01},
        ],
    )
    _write_csv(
        history_dir / "benchmark_lock_demo_significance.csv",
        [
            {
                "left_model": "extra_trees",
                "right_model": "gru_artist",
                "shared_runs": 2,
                "mean_diff_val_top1": 0.07,
                "ci95_diff_val_top1": 0.02,
                "z_score": 2.5,
                "significant_at_95": 1,
            }
        ],
    )
    _write_json(
        benchmark_manifest,
        {
            "benchmark_id": "demo",
            "comparison_ready": False,
            "summary": ["Benchmark lock is not ready yet."],
        },
    )

    return outputs, benchmark_manifest


def test_build_research_claims_report_ranks_primary_and_backup(tmp_path: Path) -> None:
    outputs, benchmark_manifest = _fixture_outputs(tmp_path)
    run_dir = outputs / "runs" / "run_full"

    report = build_research_claims_report(
        outputs,
        run_dir=run_dir,
        benchmark_manifest_path=benchmark_manifest,
    )

    assert report["primary_claim"]["key"] == "shift_robustness"
    assert report["primary_claim"]["status"] == "analysis_ready"
    assert report["backup_claim"]["key"] == "candidate_ranking"
    assert report["backup_claim"]["status"] == "promising_but_unlocked"
    assert report["believable_submission_path"] is True
    assert any("repeated-seed benchmark lock" in item for item in report["claim_gaps"])
    assert report["primary_claim"]["metrics"]["repeated_run_count"] == 3
    assert report["primary_claim"]["metrics"]["consistent_slice_run_count"] == 3
    assert report["primary_claim"]["metrics"]["dominant_context_driver"] == "behavioral"


def test_write_research_claims_report_creates_brief_and_outline(tmp_path: Path) -> None:
    outputs, benchmark_manifest = _fixture_outputs(tmp_path)
    run_dir = outputs / "runs" / "run_full"
    report = build_research_claims_report(
        outputs,
        run_dir=run_dir,
        benchmark_manifest_path=benchmark_manifest,
    )

    paths = write_research_claims_report(report, output_dir=outputs)

    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["outline_md"].exists()
    markdown = paths["md"].read_text(encoding="utf-8")
    outline = paths["outline_md"].read_text(encoding="utf-8")
    assert "Primary Claim" in markdown
    assert "Backup Claim" in markdown
    assert "Publication Outline" in outline


def test_candidate_ranking_claim_uses_retrieval_benchmark_lock_when_available(tmp_path: Path) -> None:
    outputs, _ = _fixture_outputs(tmp_path)
    run_dir = outputs / "runs" / "run_full"
    history_dir = outputs / "history"
    benchmark_manifest = history_dir / "benchmark_lock_demo_ready_manifest.json"
    _write_csv(
        history_dir / "benchmark_lock_demo_ready_summary.csv",
        [
            {"model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "run_count": 3, "val_top1_mean": 0.24, "test_top1_mean": 0.29, "val_top1_ci95": 0.01},
            {"model_name": "gru_artist", "model_type": "deep", "run_count": 3, "val_top1_mean": 0.16, "test_top1_mean": 0.15, "val_top1_ci95": 0.01},
        ],
    )
    _write_csv(
        history_dir / "benchmark_lock_demo_ready_significance.csv",
        [
            {
                "left_model": "retrieval_reranker",
                "right_model": "gru_artist",
                "shared_runs": 3,
                "mean_diff_val_top1": 0.08,
                "ci95_diff_val_top1": 0.02,
                "z_score": 3.1,
                "significant_at_95": 1,
            }
        ],
    )
    _write_json(
        benchmark_manifest,
        {
            "benchmark_id": "demo_ready",
            "comparison_ready": True,
            "summary": ["Benchmark lock is comparison ready."],
        },
    )

    report = build_research_claims_report(
        outputs,
        run_dir=run_dir,
        benchmark_manifest_path=benchmark_manifest,
    )

    candidate_claim = next(claim for claim in report["claims"] if claim["key"] == "candidate_ranking")
    assert candidate_claim["metrics"]["benchmark_retrieval_model_name"] == "retrieval_reranker"
    assert candidate_claim["metrics"]["benchmark_significant_lift"] is True
    assert not any("Add retrieval and reranker models" in item for item in candidate_claim["missing_checks"])
