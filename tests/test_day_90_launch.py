from __future__ import annotations

import json
from pathlib import Path

from spotify.day_90_launch import build_day_90_launch_report, write_day_90_launch_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sample_outputs(output_root: Path) -> None:
    showcase_dir = output_root / "analysis" / "taste_os_demo" / "showcase"
    examples_dir = showcase_dir / "examples"
    demo_md = examples_dir / "commute.md"
    _write_text(demo_md, "# Commute Demo\n")
    _write_json(
        showcase_dir / "taste_os_showcase.json",
        {
            "run_context": {"run_dir": str((output_root / "runs" / "run_anchor").resolve())},
            "showcase_summary": {"canonical_example_count": 4, "mode_comparison_count": 4},
            "review_order": ["a", "b", "c", "d"],
            "canonical_examples": [
                {
                    "label": "Commute / Friction Spike",
                    "mode": "commute",
                    "scenario": "friction_spike",
                    "story": "Friction rises and the route tightens.",
                    "story_outcome": "A safer path takes over.",
                    "top_artist": "Kid Cudi",
                    "fallback_policy_name": "safe_global",
                    "adaptive_replans": 1,
                    "adaptive_safe_route_steps": 4,
                    "demo_md_path": str(demo_md.resolve()),
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
                    {"mode": "workout", "top_artist": "Daft Punk"},
                ]
            },
        },
    )
    _write_text(showcase_dir / "taste_os_showcase.md", "# Showcase\n")

    _write_json(
        output_root / "analytics" / "control_room.json",
        {
            "latest_run": {"run_id": "run_anchor"},
            "ops_health": {"status": "healthy", "headline": "Operational review is healthy."},
            "operating_rhythm": {"overall_status": "healthy", "recommended_review_command": "make control-room"},
            "safety": {"robustness_max_top1_gap": 0.158, "test_jsd_target_drift": 0.218},
            "qoe": {"stress_worst_skip_risk": 0.439},
        },
    )
    _write_text(output_root / "analytics" / "control_room.md", "# Control Room\n")
    _write_text(output_root / "analytics" / "control_room_weekly_summary.md", "# Weekly\n")
    _write_text(output_root / "analytics" / "control_room_triage.md", "# Triage\n")

    creator_dir = output_root / "analysis" / "public_spotify" / "creator_label_intelligence"
    creator_md = creator_dir / "creator.md"
    strategy_md = creator_dir / "scene_strategy_watch.md"
    scene_seed_md = creator_dir / "scene_seed.md"
    report_family_md = creator_dir / "creator_report_family.md"
    ranking_md = creator_dir / "ranking.md"
    scene_md = creator_dir / "scene.md"
    seed_md = creator_dir / "seed.md"
    opportunity_md = creator_dir / "opportunity_lane.md"
    for path in (
        creator_md,
        strategy_md,
        scene_seed_md,
        report_family_md,
        ranking_md,
        scene_md,
        seed_md,
        opportunity_md,
    ):
        _write_text(path, f"# {path.stem}\n")
    _write_json(
        creator_dir / "creator_report_family.json",
        {
            "primary_report": str(creator_md.resolve()),
            "artifact_index_markdown": str(report_family_md.resolve()),
            "comparison_view_markdown": {
                "ranking_comparison": str(ranking_md.resolve()),
                "scene_comparison": str(scene_md.resolve()),
                "seed_comparison": str(seed_md.resolve()),
                "scene_seed_comparison": str(scene_seed_md.resolve()),
                "opportunity_lane_comparison": str(opportunity_md.resolve()),
            },
            "brief_view_markdown": {"scene_strategy_watch": str(strategy_md.resolve())},
        },
    )

    research_dir = output_root / "analysis" / "research_claims"
    _write_json(
        research_dir / "research_claims.json",
        {
            "run": {"run_id": "run_anchor"},
            "primary_claim": {
                "key": "shift_robustness",
                "title": "Failure concentration is measurable under drift",
                "status": "analysis_ready",
                "summary": "Repeated runs show the same supported slice breaking first.",
                "missing_checks": ["Add a mitigation pass for repeat_from_prev=new."],
            },
            "backup_claim": {
                "key": "candidate_ranking",
                "status": "promising_but_unlocked",
                "summary": "Ranking still matters.",
            },
            "benchmark_lock": {"benchmark_id": "smokebench-v3", "comparison_ready": True},
            "believable_submission_path": True,
            "submission_readiness": {
                "status": "analysis_ready",
                "ready_for_external_review": True,
                "summary": ["Primary claim is analysis ready.", "Benchmark lock is comparison ready."],
                "blockers": ["Add a mitigation pass for repeat_from_prev=new."],
            },
        },
    )
    _write_text(research_dir / "research_claims.md", "# Research Claims\n")
    _write_text(research_dir / "claim_support_matrix.md", "# Claim Support Matrix\n")
    _write_text(research_dir / "submission_readiness.md", "# Submission Readiness\n")
    _write_text(research_dir / "publication_outline.md", "# Publication Outline\n")

    run_dir = output_root / "runs" / "run_anchor"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / "safety_platform_contract.md", "# Safety Platform Contract\n")
    _write_json(run_dir / "safety_platform_contract.json", {"benchmark_contract_version": "2026-week10-v1"})

    history_dir = output_root / "history"
    _write_text(history_dir / "benchmark_lock_smokebench-v3_manifest.md", "# Benchmark\n")
    _write_json(history_dir / "benchmark_lock_smokebench-v3_manifest.json", {"comparison_ready": True})


def test_day_90_launch_builds_canonical_manifest_and_checklist(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _sample_outputs(output_root)

    report = build_day_90_launch_report(output_root)

    assert report["ready_to_show"] is True
    assert report["release_status"] in {"launch_ready", "show_ready_with_notes"}
    assert len(report["canonical_artifacts"]) == 6
    assert len(report["delivery_checklist"]) == 5
    assert any(row["key"] == "taste_os_demo" for row in report["canonical_artifacts"])


def test_day_90_launch_writes_markdown_and_json(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _sample_outputs(output_root)

    report = build_day_90_launch_report(output_root)
    paths = write_day_90_launch_artifacts(report, output_dir=output_root)

    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["canonical_manifest_md"].exists()
    assert paths["delivery_checklist_md"].exists()
    markdown = paths["md"].read_text(encoding="utf-8")
    assert "Day-90 Launch Package" in markdown
    assert "Canonical Artifacts" in markdown
    checklist = paths["delivery_checklist_md"].read_text(encoding="utf-8")
    assert "Delivery Checklist" in checklist
