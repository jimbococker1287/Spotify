from __future__ import annotations

import json
from pathlib import Path

from spotify.outward_package import build_outward_package_report, write_outward_package_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_sample_outputs(output_root: Path) -> None:
    commute_demo_md = output_root / "analysis" / "taste_os_demo" / "showcase" / "examples" / "commute.md"
    _write_text(commute_demo_md, "# Commute\n")
    _write_json(
        output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.json",
        {
            "run_context": {
                "run_dir": str((output_root / "runs" / "run_full_anchor").resolve()),
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
            },
            "canonical_examples": [
                {
                    "label": "Commute / Friction Spike",
                    "mode": "commute",
                    "scenario": "friction_spike",
                    "top_artist": "Kid Cudi",
                    "fallback_policy_name": "safe_bucket_normal_friction",
                    "adaptive_replans": 1,
                    "adaptive_safe_route_steps": 4,
                    "demo_md_path": str(commute_demo_md),
                },
                {},
                {},
                {},
            ],
            "mode_comparison": {
                "rows": [
                    {"mode": "focus", "top_artist": "Tame Impala"},
                    {"mode": "commute", "top_artist": "Kid Cudi"},
                    {"mode": "discovery", "top_artist": "Arctic Monkeys"},
                ]
            },
        },
    )
    _write_text(output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.md", "# Taste OS\n")
    _write_json(
        output_root / "analytics" / "control_room.json",
        {
            "latest_run": {"run_id": "run_full_anchor"},
            "ops_health": {"status": "healthy", "headline": "Operational review is healthy."},
            "safety": {
                "robustness_max_top1_gap": 0.158,
                "test_jsd_target_drift": 0.218,
                "test_selective_risk": 0.657,
                "test_abstention_rate": 0.137,
            },
            "qoe": {"stress_worst_skip_risk": 0.613},
            "operating_rhythm": {"overall_status": "healthy", "recommended_review_command": "make control-room"},
        },
    )
    _write_text(output_root / "analytics" / "control_room.md", "# Control Room\n")
    _write_text(output_root / "analytics" / "control_room_weekly_summary.md", "# Weekly\n")
    _write_text(output_root / "analytics" / "control_room_triage.md", "# Triage\n")

    creator_dir = output_root / "analysis" / "public_spotify" / "creator_label_intelligence"
    creator_main = creator_dir / "creator.md"
    scene_seed = creator_dir / "scene_seed.md"
    ranking = creator_dir / "ranking.md"
    scene = creator_dir / "scene.md"
    seed = creator_dir / "seed.md"
    for path in (creator_main, scene_seed, ranking, scene, seed):
        _write_text(path, f"# {path.stem}\n")
    _write_json(
        creator_dir / "creator_report_family.json",
        {
            "primary_report": str(creator_main),
            "comparison_view_markdown": {
                "ranking_comparison": str(ranking),
                "scene_comparison": str(scene),
                "seed_comparison": str(seed),
                "scene_seed_comparison": str(scene_seed),
            },
        },
    )

    _write_json(
        output_root / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": "run_full_anchor"},
            "primary_claim": {
                "key": "shift_robustness",
                "status": "analysis_ready",
                "summary": "Shift is measurable.",
                "missing_checks": ["Repeat seeds."],
                "supporting_artifacts": ["/tmp/a.json"],
                "metrics": {
                    "worst_robustness_gap": 0.158,
                    "target_drift_jsd": 0.218,
                    "selective_risk": 0.500,
                    "abstention_rate": 0.391,
                    "stress_skip_risk": 0.613,
                },
            },
            "backup_claim": {
                "key": "candidate_ranking",
                "status": "promising_but_unlocked",
                "summary": "Ranking is promising.",
            },
            "benchmark_lock": {"benchmark_id": "smokebench", "comparison_ready": False},
            "believable_submission_path": True,
        },
    )
    _write_text(output_root / "analysis" / "research_claims" / "research_claims.md", "# Research Claims\n")
    _write_text(output_root / "history" / "benchmark_lock_smokebench_manifest.md", "# Benchmark\n")
    _write_json(
        output_root / "history" / "benchmark_lock_smokebench_manifest.json",
        {"benchmark_id": "smokebench", "comparison_ready": False},
    )


def test_outward_package_copies_four_branch_assets_and_generates_summary(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _build_sample_outputs(output_root)

    report = build_outward_package_report(output_root)
    paths = write_outward_package_artifacts(report, output_dir=output_root)

    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["four_branch_summary_md"].exists()
    assert paths["safety_research_showcase_md"].exists()

    package_root = output_root / "analysis" / "outward_package"
    assert (package_root / "flagship" / "claim_to_demo.md").exists()
    assert (package_root / "taste_os" / "taste_os_showcase.md").exists()
    assert (package_root / "control_room" / "control_room.md").exists()
    assert (package_root / "creator_intelligence" / "creator_label_intelligence.md").exists()
    assert (package_root / "safety_research" / "research_claims.md").exists()

    package_payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "copied_artifacts" in package_payload
    assert package_payload["copied_artifacts"]["claim_to_demo_md"]
    assert package_payload["copied_artifacts"]["taste_os_md"]
    safety_showcase = paths["safety_research_showcase_md"].read_text(encoding="utf-8")
    assert "Believable submission path: `True`" in safety_showcase
