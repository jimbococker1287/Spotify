from __future__ import annotations

import json
from pathlib import Path

from spotify.phase_readiness import (
    build_weeks_1_8_readiness_report,
    build_weeks_1_13_readiness_report,
    write_weeks_1_8_readiness_report,
    write_weeks_1_13_readiness_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _touch(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_creator_family(root: Path, stem: str, *, scene_seed_rows: int = 2) -> None:
    base_dir = root / "outputs/analysis/public_spotify/creator_label_intelligence"
    primary_report = base_dir / f"{stem}.md"
    _touch(primary_report, "# creator\n")
    _write_json(
        base_dir / f"{stem}.json",
        {
            "comparison_views": {
                "ranking_comparison": [{"artist_name": "A"}],
                "scene_comparison": [{"scene_name": "scene-1"}],
                "seed_comparison": [{"seed_artist": "Seed A"}],
                "scene_seed_comparison": [{"scene_name": "scene-1"}] * scene_seed_rows,
            }
        },
    )
    manifest = {
        "primary_report": str(primary_report.resolve()),
        "comparison_view_markdown": {},
        "comparison_view_csv": {},
    }
    for key in ("ranking_comparison", "scene_comparison", "seed_comparison", "scene_seed_comparison"):
        md_path = base_dir / f"{stem}_{key}.md"
        csv_path = base_dir / f"{stem}_{key}.csv"
        _touch(md_path, f"# {key}\n")
        _touch(csv_path, "col\n")
        manifest["comparison_view_markdown"][key] = str(md_path.resolve())
        manifest["comparison_view_csv"][key] = str(csv_path.resolve())
    _write_json(base_dir / f"{stem}_report_family.json", manifest)


def test_phase_readiness_reports_built_surfaces_even_with_ops_attention(tmp_path: Path) -> None:
    for relative in (
        "docs/personal_taste_os.md",
        "docs/taste_os_demo_contract.md",
        "docs/taste_os_demo_walkthrough.md",
        "docs/taste_os_product_story.md",
        "docs/control_room_operating_rhythm.md",
        "docs/creator_label_intelligence_brief.md",
    ):
        _touch(tmp_path / relative)

    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.json",
        {
            "showcase_summary": {"canonical_example_count": 4, "mode_comparison_count": 4},
            "review_order": ["Focus / Steady", "Discovery / Skip Recovery", "Commute / Friction Spike", "Workout / Repeat Request"],
        },
    )
    _touch(tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.md")
    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.json",
        {
            "rows": [
                {"top_artist": "Artist A"},
                {"top_artist": "Artist B"},
                {"top_artist": "Artist C"},
                {"top_artist": "Artist B"},
            ]
        },
    )
    _touch(tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.md")

    _write_json(
        tmp_path / "outputs/analytics/control_room.json",
        {
            "review_actions": [{"priority": "high"}],
            "operating_rhythm": {"overall_status": "attention", "recommended_review_command": "make control-room"},
            "ops_health": {
                "status": "attention",
                "operational_high_priority_count": 1,
                "strategic_high_priority_count": 0,
            },
            "latest_run": {"run_id": "run_001"},
            "async_handoff": {"share_artifacts": ["a.md", "b.md", "c.md"]},
        },
    )
    for relative in (
        "outputs/analytics/control_room.md",
        "outputs/analytics/control_room_weekly_summary.json",
        "outputs/analytics/control_room_weekly_summary.md",
        "outputs/analytics/control_room_triage.json",
        "outputs/analytics/control_room_triage.md",
    ):
        if relative.endswith(".json"):
            _write_json(tmp_path / relative, {})
        else:
            _touch(tmp_path / relative)

    for stem in (
        "creator_label_intelligence_indie",
        "creator_label_intelligence_rap",
        "creator_label_intelligence_mixed",
    ):
        _create_creator_family(tmp_path, stem)

    (tmp_path / "outputs/runs/run_demo").mkdir(parents=True)

    report = build_weeks_1_8_readiness_report(tmp_path)

    assert report["weeks_1_8_ready_for_week_9_10"] is True
    assert report["overall"]["completeness_status"] == "ready"
    assert report["overall"]["operational_status"] == "attention"
    assert report["sections"][0]["metrics"]["unique_opening_artists"] == 3
    assert report["sections"][1]["metrics"]["high_priority_review_actions"] == 1
    assert report["sections"][1]["metrics"]["ops_health_status"] == "attention"
    assert report["sections"][2]["metrics"]["report_family_count"] == 3

    artifacts = write_weeks_1_8_readiness_report(report, output_dir=tmp_path / "outputs/analytics")
    assert artifacts["json"].exists()
    assert artifacts["md"].exists()
    assert "Weeks 1-8 Readiness" in artifacts["md"].read_text(encoding="utf-8")


def test_phase_readiness_blocks_progress_when_creator_surface_is_incomplete(tmp_path: Path) -> None:
    for relative in (
        "docs/personal_taste_os.md",
        "docs/taste_os_demo_contract.md",
        "docs/taste_os_demo_walkthrough.md",
        "docs/taste_os_product_story.md",
        "docs/control_room_operating_rhythm.md",
        "docs/creator_label_intelligence_brief.md",
    ):
        _touch(tmp_path / relative)

    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.json",
        {
            "showcase_summary": {"canonical_example_count": 4, "mode_comparison_count": 4},
            "review_order": ["a", "b", "c", "d"],
        },
    )
    _touch(tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.md")
    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.json",
        {"rows": [{"top_artist": "A"}, {"top_artist": "B"}, {"top_artist": "C"}, {"top_artist": "D"}]},
    )
    _touch(tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.md")

    _write_json(
        tmp_path / "outputs/analytics/control_room.json",
        {
            "review_actions": [],
            "operating_rhythm": {"overall_status": "healthy", "recommended_review_command": "make control-room"},
            "ops_health": {
                "status": "healthy",
                "operational_high_priority_count": 0,
                "strategic_high_priority_count": 0,
            },
            "latest_run": {"run_id": "run_002"},
            "async_handoff": {"share_artifacts": ["a.md", "b.md", "c.md"]},
        },
    )
    for relative in (
        "outputs/analytics/control_room.md",
        "outputs/analytics/control_room_weekly_summary.json",
        "outputs/analytics/control_room_weekly_summary.md",
        "outputs/analytics/control_room_triage.json",
        "outputs/analytics/control_room_triage.md",
    ):
        if relative.endswith(".json"):
            _write_json(tmp_path / relative, {})
        else:
            _touch(tmp_path / relative)

    _create_creator_family(tmp_path, "creator_label_intelligence_only_one")

    report = build_weeks_1_8_readiness_report(tmp_path)

    assert report["weeks_1_8_ready_for_week_9_10"] is False
    assert report["sections"][2]["completeness_status"] == "missing"
    assert any("Generate at least three creator report families" in action for action in report["next_actions"])


def test_phase_readiness_reports_weeks_1_13_when_all_branches_are_packaged(tmp_path: Path) -> None:
    for relative in (
        "docs/personal_taste_os.md",
        "docs/taste_os_demo_contract.md",
        "docs/taste_os_demo_walkthrough.md",
        "docs/taste_os_product_story.md",
        "docs/control_room_operating_rhythm.md",
        "docs/creator_label_intelligence_brief.md",
        "docs/recommender_safety_platform.md",
        "docs/benchmark_contract.md",
        "docs/publication_outline.md",
        "docs/higher_level_branches.md",
        "docs/outward_package.md",
    ):
        _touch(tmp_path / relative)

    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.json",
        {
            "canonical_examples": [{}, {}, {}, {}],
            "mode_comparison": {
                "rows": [
                    {"top_artist": "Artist A"},
                    {"top_artist": "Artist B"},
                    {"top_artist": "Artist C"},
                    {"top_artist": "Artist B"},
                ]
            },
            "showcase_summary": {"canonical_example_count": 4, "mode_comparison_count": 4},
            "review_order": ["a", "b", "c", "d"],
        },
    )
    _touch(tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.md")
    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.json",
        {
            "rows": [
                {"top_artist": "Artist A"},
                {"top_artist": "Artist B"},
                {"top_artist": "Artist C"},
                {"top_artist": "Artist B"},
            ]
        },
    )
    _touch(tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.md")

    _write_json(
        tmp_path / "outputs/analytics/control_room.json",
        {
            "review_actions": [],
            "operating_rhythm": {"overall_status": "healthy", "recommended_review_command": "make control-room"},
            "ops_health": {
                "status": "healthy",
                "operational_high_priority_count": 0,
                "strategic_high_priority_count": 0,
                "headline": "Operational review is healthy.",
            },
            "latest_run": {"run_id": "run_003"},
            "async_handoff": {"share_artifacts": ["a.md", "b.md", "c.md"]},
        },
    )
    for relative in (
        "outputs/analytics/control_room.md",
        "outputs/analytics/control_room_weekly_summary.json",
        "outputs/analytics/control_room_weekly_summary.md",
        "outputs/analytics/control_room_triage.json",
        "outputs/analytics/control_room_triage.md",
    ):
        if relative.endswith(".json"):
            _write_json(tmp_path / relative, {})
        else:
            _touch(tmp_path / relative)

    for stem in (
        "creator_label_intelligence_indie",
        "creator_label_intelligence_rap",
        "creator_label_intelligence_mixed",
    ):
        _create_creator_family(tmp_path, stem)

    _write_json(
        tmp_path / "outputs/analysis/research_claims/research_claims.json",
        {
            "primary_claim": {"key": "shift_robustness", "status": "analysis_ready"},
            "backup_claim": {"key": "candidate_ranking", "status": "promising_but_unlocked"},
            "benchmark_lock": {"benchmark_id": "smokebench", "comparison_ready": True},
            "believable_submission_path": True,
        },
    )
    _touch(tmp_path / "outputs/analysis/research_claims/research_claims.md")
    _write_json(
        tmp_path / "outputs/history/benchmark_lock_smokebench_manifest.json",
        {"benchmark_id": "smokebench", "comparison_ready": True},
    )
    _touch(tmp_path / "outputs/history/benchmark_lock_smokebench_manifest.md")

    for relative in (
        "outputs/analysis/portfolio_branches/portfolio_branches.json",
        "outputs/analysis/outward_package/outward_package.json",
    ):
        _write_json(tmp_path / relative, {})
    for relative in (
        "outputs/analysis/portfolio_branches/portfolio_branches.md",
        "outputs/analysis/outward_package/outward_package.md",
        "outputs/analysis/outward_package/four_branch_summary.md",
        "outputs/analysis/outward_package/safety_research/safety_research_showcase.md",
        "outputs/analysis/outward_package/taste_os/taste_os_showcase.md",
        "outputs/analysis/outward_package/control_room/control_room.md",
        "outputs/analysis/outward_package/creator_intelligence/creator_label_intelligence.md",
        "outputs/analysis/outward_package/creator_intelligence/scene_seed_view.md",
        "outputs/analysis/outward_package/safety_research/research_claims.md",
        "outputs/analysis/outward_package/safety_research/benchmark_lock_manifest.md",
    ):
        _touch(tmp_path / relative)

    (tmp_path / "outputs/runs/run_demo").mkdir(parents=True)

    report = build_weeks_1_13_readiness_report(tmp_path)

    assert report["weeks_1_13_ready_for_day_90"] is True
    assert report["overall"]["completeness_status"] == "ready"
    assert report["overall"]["efficiency_status"] == "ready"
    assert len(report["sections"]) == 5
    assert report["sections"][3]["metrics"]["benchmark_comparison_ready"] is True
    assert report["sections"][4]["metrics"]["primary_branch_count"] == 4

    artifacts = write_weeks_1_13_readiness_report(report, output_dir=tmp_path / "outputs/analytics")
    assert artifacts["json"].exists()
    assert artifacts["md"].exists()
    assert "Weeks 1-13 Readiness" in artifacts["md"].read_text(encoding="utf-8")
