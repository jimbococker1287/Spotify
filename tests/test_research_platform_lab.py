from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from spotify.research_platform_lab import build_research_platform_lab


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _set_mtime(path: Path, when: int) -> None:
    os.utime(path, (when, when))


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
                    "supporting_artifacts": [str((run_dir / "run_results.json").resolve())],
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
    assert run_registry.iloc[0]["portability_signal_status"] == "ready"
    assert benchmark_atlas.iloc[0]["benchmark_id"] == "smokebench"
    assert benchmark_atlas.iloc[0]["comparison_status"] == "ready"
    assert claim_registry.iloc[0]["claim_key"] == "shift_robustness"
    assert claim_registry.iloc[0]["claim_readiness_status"] == "ready"
    assert "Research Platform Maturity" in brief_text
    assert "blocked by `Add repeated deep comparator`" in brief_text


def test_research_platform_lab_surfaces_blocked_claims_incomplete_locks_and_stale_paths(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.research_platform_lab.truthfulness")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_b"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_b",
            "profile": "full",
            "timestamp": "2026-05-02T12:00:00",
            "champion_gate": {"status": "attention", "promoted": False},
        },
    )
    _write_json(run_dir / "run_results.json", [{"model_name": "retrieval_reranker", "test_top1": 0.41}])
    _write_json(
        run_dir / "benchmark_protocol.json",
        {
            "benchmark_contract": {"contract_version": "v5", "comparison_mode": "repeated_seed_lock"},
            "protocol": {"temporal_backtest": {"models": ["retrieval_reranker"]}},
        },
    )
    _write_json(
        run_dir / "safety_platform_contract.json",
        {
            "reuse_summary": {"api_group_count": 0, "wrapper_count": 0},
            "portability_notes": ["Portability evidence is still thin."],
        },
    )
    _write_json(analysis_dir / "data_drift_summary.json", {"target_drift": {"train_vs_test_jsd": 0.19}})
    _write_json(analysis_dir / "robustness_summary.json", [{"worst_segment": "repeat_from_prev"}])
    _write_json(analysis_dir / "moonshot_summary.json", {"stress_worst_skip_risk": 0.62})
    confidence_path = analysis_dir / "ensemble_blended_ensemble_confidence_summary.json"
    conformal_path = analysis_dir / "ensemble_blended_ensemble_conformal_summary.json"
    friction_summary_path = analysis_dir / "friction_proxy_summary.json"
    friction_delta_path = analysis_dir / "friction_counterfactual_delta.csv"
    _write_json(confidence_path, {"test_selective_risk": 0.51, "test_abstention_rate": 0.0})
    _write_json(conformal_path, {"test": {"coverage": 0.74}})
    _write_json(
        friction_summary_path,
        {
            "proxy_counterfactual": {"test_mean_delta": 0.0},
            "baseline_model": {"test_auc": 1.0},
            "full_model": {"test_auc": 1.0},
            "auc_lift": {"test": 0.0},
        },
    )
    _write_csv(friction_delta_path, [{"split": "test", "mean_delta": 0.0}])

    _write_csv(
        output_dir / "analytics" / "control_room_history.csv",
        [
            {
                "run_id": "run_b",
                "target_drift_jsd": 0.19,
                "test_selective_risk": 0.51,
                "test_abstention_rate": 0.02,
                "robustness_gap": 0.17,
                "stress_skip_risk": 0.62,
                "ops_coverage_ratio": 0.75,
            }
        ],
    )

    history_dir = output_dir / "history"
    benchmark_manifest = history_dir / "benchmark_lock_smokebench_manifest.json"
    benchmark_manifest_md = history_dir / "benchmark_lock_smokebench_manifest.md"
    benchmark_summary = history_dir / "benchmark_lock_smokebench_summary.csv"
    benchmark_significance = history_dir / "benchmark_lock_smokebench_significance.csv"
    benchmark_rows = history_dir / "benchmark_lock_smokebench_rows.csv"
    benchmark_summary_json = history_dir / "benchmark_lock_smokebench_summary.json"
    benchmark_plot = history_dir / "benchmark_lock_smokebench_ci95.png"

    _write_json(
        benchmark_manifest,
        {
            "benchmark_id": "smokebench",
            "canonical_profile": "small",
            "comparison_mode": "repeated_seed_lock",
            "comparison_ready": False,
            "run_count": 3,
            "present_artifact_count": 6,
            "required_artifact_count": 7,
            "required_artifacts": [
                str(benchmark_rows.resolve()),
                str(benchmark_summary.resolve()),
                str(benchmark_summary_json.resolve()),
                str(benchmark_plot.resolve()),
                str(benchmark_significance.resolve()),
                str(benchmark_manifest.resolve()),
                str(benchmark_manifest_md.resolve()),
            ],
            "comparison_blockers": ["Research-grade comparator guard failed: no repeated deep comparator appears in the benchmark summary with at least `3` run(s)."],
            "comparator_guard": {"status": "fail", "deep_comparator_ready": False},
            "model_class_mix": {"observed": {"model_classes": ["candidate", "classical"]}},
            "significant_pair_count": 0,
        },
    )
    benchmark_manifest_md.write_text("# manifest\n", encoding="utf-8")
    _write_csv(
        benchmark_summary,
        [
            {
                "benchmark_id": "smokebench",
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "val_top1_mean": 0.33,
                "test_top1_mean": 0.31,
                "run_count": 3,
            }
        ],
    )
    _write_csv(
        benchmark_significance,
        [
            {
                "left_model": "retrieval_reranker",
                "right_model": "extra_trees",
                "mean_diff_val_top1": 0.03,
                "z_score": 1.5,
                "significant_at_95": 0,
            }
        ],
    )
    _write_csv(benchmark_rows, [{"run_id": "run_b"}])
    benchmark_summary_json.write_text("[]", encoding="utf-8")
    benchmark_plot.write_bytes(b"png")

    claims_path = output_dir / "analysis" / "research_claims" / "research_claims.json"
    _write_json(
        claims_path,
        {
            "run": {"run_id": "run_b"},
            "primary_claim": {"key": "candidate_ranking"},
            "claims": [
                {
                    "key": "candidate_ranking",
                    "title": "Candidate ranking",
                    "status": "promising_but_unlocked",
                    "summary": "Live ranking signal exists but the lock is incomplete.",
                    "metrics": {
                        "live_test_top1_lift_vs_deep": 0.11,
                        "benchmark_comparison_ready": False,
                        "benchmark_significant_lift": False,
                    },
                    "supporting_artifacts": [
                        str(benchmark_manifest.resolve()),
                        str((run_dir / "run_results.json").resolve()),
                    ],
                    "missing_checks": ["Benchmark lock is not comparison-ready because the research-grade comparator guard did not observe a repeated deep comparator."],
                },
                {
                    "key": "risk_aware_abstention",
                    "title": "Risk-aware abstention",
                    "status": "not_supported",
                    "summary": "Risk is visible but coverage has not been traded off in a calibrated way.",
                    "metrics": {
                        "selective_risk": 0.51,
                        "abstention_rate": 0.0,
                        "conformal_coverage": 0.74,
                    },
                    "supporting_artifacts": [
                        str(confidence_path.resolve()),
                        str(conformal_path.resolve()),
                    ],
                    "missing_checks": [
                        "Tune abstention thresholds until coverage loss buys a meaningful selective-risk reduction.",
                        "Report accuracy-coverage tradeoffs rather than only full-coverage conformal metrics.",
                    ],
                },
                {
                    "key": "friction_counterfactual",
                    "title": "Friction counterfactual",
                    "status": "not_supported",
                    "summary": "Friction artifacts are degenerate and need a trustworthiness audit.",
                    "metrics": {
                        "test_mean_delta": 0.0,
                        "baseline_test_auc": 1.0,
                        "full_test_auc": 1.0,
                        "test_auc_lift": 0.0,
                    },
                    "supporting_artifacts": [
                        str(friction_summary_path.resolve()),
                        str(friction_delta_path.resolve()),
                    ],
                    "missing_checks": [
                        "Audit the friction label path because AUC is saturated while the counterfactual delta is effectively zero.",
                        "Add a non-degenerate intervention or synthetic perturbation check before using this as a headline claim.",
                    ],
                }
            ],
            "claim_support_matrix": [
                {
                    "claim_key": "candidate_ranking",
                    "live_signal_status": "ready",
                    "benchmark_evidence_status": "gap",
                    "repeated_evidence_status": "gap",
                    "slice_evidence_status": "n/a",
                    "risk_evidence_status": "n/a",
                    "artifact_pack_status": "ready",
                    "next_gate": "Finish the benchmark lock",
                },
                {
                    "claim_key": "risk_aware_abstention",
                    "live_signal_status": "gap",
                    "benchmark_evidence_status": "n/a",
                    "repeated_evidence_status": "n/a",
                    "slice_evidence_status": "n/a",
                    "risk_evidence_status": "gap",
                    "artifact_pack_status": "ready",
                    "next_gate": "Tune abstention thresholds until coverage loss buys a meaningful selective-risk reduction.",
                },
                {
                    "claim_key": "friction_counterfactual",
                    "live_signal_status": "gap",
                    "benchmark_evidence_status": "n/a",
                    "repeated_evidence_status": "n/a",
                    "slice_evidence_status": "n/a",
                    "risk_evidence_status": "ready",
                    "artifact_pack_status": "ready",
                    "next_gate": "Audit the friction label path because AUC is saturated while the counterfactual delta is effectively zero.",
                }
            ],
            "submission_readiness": {
                "status": "promising_but_unlocked",
                "ready_for_external_review": False,
                "blockers": ["Finish the benchmark lock"],
            },
        },
    )

    base_time = 1_725_000_000
    _set_mtime(benchmark_manifest, base_time + 4)
    _set_mtime(benchmark_manifest_md, base_time + 4)
    _set_mtime(claims_path, base_time + 2)
    _set_mtime(run_dir / "run_results.json", base_time + 5)
    _set_mtime(benchmark_summary, base_time + 7)
    _set_mtime(benchmark_significance, base_time + 7)
    for path in [confidence_path, conformal_path, friction_summary_path, friction_delta_path]:
        _set_mtime(path, base_time + 1)

    build_research_platform_lab(output_dir=output_dir, run_dir=None, logger=logger)

    result_root = output_dir / "analysis" / "research_platform_lab"
    run_registry = pd.read_csv(result_root / "run_research_registry.csv")
    benchmark_atlas = pd.read_csv(result_root / "benchmark_lock_atlas.csv")
    claim_registry = pd.read_csv(result_root / "research_claim_registry.csv")
    maturity_payload = json.loads((result_root / "research_platform_maturity.json").read_text(encoding="utf-8"))
    maturity_markdown = (result_root / "research_platform_maturity.md").read_text(encoding="utf-8")
    next_experiments = json.loads((result_root / "research_next_experiments.json").read_text(encoding="utf-8"))
    next_experiments_markdown = (result_root / "research_next_experiments.md").read_text(encoding="utf-8")

    assert run_registry.iloc[0]["portability_signal_status"] == "attention"
    assert run_registry.iloc[0]["claim_pack_freshness_status"] == "stale"
    assert str(run_registry.iloc[0]["claim_pack_stale_source_path"]).endswith("run_results.json")

    assert benchmark_atlas.iloc[0]["comparison_status"] == "incomplete"
    assert benchmark_atlas.iloc[0]["manifest_freshness_status"] == "stale"
    assert "deep comparator" in str(benchmark_atlas.iloc[0]["top_comparison_blocker"]).lower()

    candidate_claim = claim_registry.loc[claim_registry["claim_key"] == "candidate_ranking"].iloc[0]
    assert candidate_claim["claim_readiness_status"] == "blocked"
    assert bool(candidate_claim["blocked"]) is True
    assert candidate_claim["supporting_artifact_freshness_status"] == "stale"
    assert str(candidate_claim["stale_supporting_artifact_path"]).endswith("benchmark_lock_smokebench_manifest.json")

    assert maturity_payload["claim_blocked_count"] == 3
    assert maturity_payload["incomplete_benchmark_lock_count"] == 1
    assert "blocked by `Finish the benchmark lock`" in maturity_markdown
    assert "portability signals are `attention`" in maturity_markdown
    assert "manifest freshness looks stale" in maturity_markdown

    experiments_by_key = {
        (str(experiment["claim_key"]), str(experiment["experiment_type"])): experiment
        for experiment in next_experiments["experiments"]
    }
    deep_experiment = experiments_by_key[("candidate_ranking", "deep_comparator_benchmark_coverage")]
    risk_experiment = experiments_by_key[("risk_aware_abstention", "risk_coverage_tradeoff_evidence")]
    friction_experiment = experiments_by_key[("friction_counterfactual", "friction_counterfactual_trustworthiness")]

    assert next_experiments["blocked_claim_count"] == 3
    assert next_experiments["experiments"][0]["experiment_type"] == "deep_comparator_benchmark_coverage"
    assert "smokebench" in deep_experiment["recommended_experiment"]
    assert "repeated deep comparator" in deep_experiment["title"].lower().replace("-", " ")
    assert any("deep comparator" in criterion.lower() for criterion in deep_experiment["success_criteria"])
    assert "accuracy/coverage/selective-risk table" in risk_experiment["recommended_experiment"]
    assert "coverage" in risk_experiment["success_criteria"][0]
    assert "friction label path" in friction_experiment["recommended_experiment"]
    assert "non-degenerate" in friction_experiment["recommended_experiment"]
    assert "Research Next Experiments" in next_experiments_markdown
    assert "friction_counterfactual_trustworthiness" in next_experiments_markdown


def test_research_platform_lab_registers_creator_evidence_and_downgrades_stale_passports(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.research_platform_lab.creator_evidence")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_creator"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_creator",
            "profile": "small",
            "timestamp": "2026-06-01T12:00:00",
            "champion_gate": {"status": "attention", "promoted": False},
        },
    )

    evidence_root = output_dir / "analysis" / "creator_evidence_lab"
    evidence_root.mkdir(parents=True, exist_ok=True)
    ready_source = output_dir / "creator_ready_source.json"
    stale_source = output_dir / "creator_stale_source.json"
    missing_source = output_dir / "creator_missing_source.json"
    ready_source.write_text("{}", encoding="utf-8")
    stale_source.write_text("{}", encoding="utf-8")

    passports_path = evidence_root / "creator_opportunity_evidence_passports.json"
    csv_path = evidence_root / "creator_opportunity_evidence_passports.csv"
    markdown_path = evidence_root / "creator_opportunity_evidence_passports.md"
    manifest_path = evidence_root / "creator_evidence_manifest.json"
    csv_path.write_text("passport_id\n", encoding="utf-8")
    markdown_path.write_text("# Creator Evidence\n", encoding="utf-8")
    freshness_gate = {
        "key": "evidence_freshness",
        "status": "pass",
        "observed": {"timestamp_coverage": 1.0, "latest_source_age_days": 5},
        "threshold": {"timestamp_coverage": 1.0, "maximum_source_age_days": 90},
    }
    _write_json(
        passports_path,
        [
            {
                "passport_id": "ready",
                "artist_name": "Artist Ready",
                "market": "US",
                "evidence_grade": "publishable",
                "verified": True,
                "contract_version": "2026-06-v1",
                "occurrence_count": 2,
                "report_family_count": 2,
                "latest_source_age_days": 5,
                "source_artifact_paths": [str(ready_source)],
                "gates": [freshness_gate],
            },
            {
                "passport_id": "stale",
                "artist_name": "Artist Stale",
                "market": "US",
                "evidence_grade": "publishable",
                "verified": True,
                "contract_version": "2026-06-v1",
                "occurrence_count": 2,
                "report_family_count": 2,
                "latest_source_age_days": 5,
                "source_artifact_paths": [str(stale_source)],
                "gates": [freshness_gate],
            },
            {
                "passport_id": "watch",
                "artist_name": "Artist Missing Source",
                "market": "US",
                "evidence_grade": "watch_only",
                "verified": False,
                "contract_version": "2026-06-v1",
                "occurrence_count": 1,
                "report_family_count": 1,
                "latest_source_age_days": 5,
                "source_artifact_paths": [str(missing_source)],
                "gates": [freshness_gate],
            },
        ],
    )
    _write_json(
        manifest_path,
        {
            "contract": {
                "contract_version": "2026-06-v1",
                "verified_grade": "publishable",
            },
            "passport_count": 3,
            "grade_counts": {"publishable": 2, "watch_only": 1, "suppress": 0},
            "artifact_paths": {
                "json": str(passports_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "manifest": str(manifest_path),
            },
        },
    )
    base_time = 1_800_000_000
    _set_mtime(ready_source, base_time)
    _set_mtime(csv_path, base_time + 1)
    _set_mtime(markdown_path, base_time + 1)
    _set_mtime(passports_path, base_time + 2)
    _set_mtime(manifest_path, base_time + 3)
    _set_mtime(stale_source, base_time + 4)

    build_research_platform_lab(output_dir=output_dir, run_dir=run_dir, logger=logger)

    result_root = output_dir / "analysis" / "research_platform_lab"
    registry = pd.read_csv(result_root / "creator_evidence_registry.csv")
    maturity = json.loads((result_root / "research_platform_maturity.json").read_text(encoding="utf-8"))
    maturity_markdown = (result_root / "research_platform_maturity.md").read_text(encoding="utf-8")
    manifest = json.loads((result_root / "research_platform_manifest.json").read_text(encoding="utf-8"))

    ready = registry.loc[registry["artist_name"] == "Artist Ready"].iloc[0]
    stale = registry.loc[registry["artist_name"] == "Artist Stale"].iloc[0]
    missing = registry.loc[registry["artist_name"] == "Artist Missing Source"].iloc[0]
    assert bool(ready["effective_verified"]) is True
    assert bool(stale["effective_verified"]) is False
    assert stale["freshness_status"] == "stale"
    assert missing["source_artifact_path_status"] == "missing"

    evidence_summary = maturity["creator_evidence"]
    assert evidence_summary["status"] == "stale"
    assert evidence_summary["effective_verified_passport_count"] == 1
    assert evidence_summary["grade_counts"] == {"publishable": 2, "suppress": 0, "watch_only": 1}
    assert evidence_summary["contract_status"] == "ready"
    assert evidence_summary["artifact_path_status"] == "ready"
    assert evidence_summary["missing_source_artifact_count"] == 1
    assert "directional/watch signals" in maturity_markdown
    assert manifest["tables"]["creator_evidence_registry"]["row_count"] == 3
    assert manifest["creator_evidence"]["freshness_status"] == "stale"
