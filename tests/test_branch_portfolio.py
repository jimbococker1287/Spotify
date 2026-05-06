from __future__ import annotations

import json
from pathlib import Path

from spotify.branch_portfolio import build_branch_portfolio_report, write_branch_portfolio_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_sample_outputs(output_root: Path) -> None:
    _write_json(
        output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.json",
        {
            "canonical_examples": [{}, {}, {}, {}],
            "mode_comparison": {
                "rows": [
                    {"mode": "focus", "top_artist": "Tame Impala"},
                    {"mode": "workout", "top_artist": "Daft Punk"},
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
            "ops_health": {
                "status": "healthy",
                "headline": "Operational review is healthy.",
            },
            "operating_rhythm": {
                "overall_status": "healthy",
                "recommended_review_command": "make control-room",
            },
        },
    )
    _write_text(output_root / "analytics" / "control_room.md", "# Control Room\n")
    _write_text(output_root / "analytics" / "control_room_weekly_summary.md", "# Weekly\n")
    _write_text(output_root / "analytics" / "control_room_triage.md", "# Triage\n")

    creator_dir = output_root / "analysis" / "public_spotify" / "creator_label_intelligence"
    primary_report = creator_dir / "sample.md"
    scene_seed_view = creator_dir / "sample_scene_seed_view.md"
    opportunity_lane_view = creator_dir / "sample_opportunity_lane_view.md"
    scene_strategy_view = creator_dir / "sample_scene_strategy_watch.md"
    ranking_view = creator_dir / "sample_ranking_view.md"
    scene_view = creator_dir / "sample_scene_view.md"
    seed_view = creator_dir / "sample_seed_view.md"
    report_family_md = creator_dir / "sample_report_family.md"
    for path in (primary_report, scene_seed_view, opportunity_lane_view, scene_strategy_view, ranking_view, scene_view, seed_view, report_family_md):
        _write_text(path, f"# {path.stem}\n")
    _write_json(
        creator_dir / "sample_report_family.json",
        {
            "primary_report": str(primary_report),
            "artifact_index_markdown": str(report_family_md),
            "comparison_view_markdown": {
                "ranking_comparison": str(ranking_view),
                "scene_comparison": str(scene_view),
                "seed_comparison": str(seed_view),
                "scene_seed_comparison": str(scene_seed_view),
                "opportunity_lane_comparison": str(opportunity_lane_view),
            },
            "brief_view_markdown": {"scene_strategy_watch": str(scene_strategy_view)},
        },
    )
    creator_market_dir = output_root / "analysis" / "creator_market_intelligence"
    _write_text(creator_market_dir / "creator_market_brief.md", "# Creator Market Brief\n")
    _write_json(
        creator_market_dir / "creator_market_brief.json",
        {
            "report_family_count": 3,
            "top_scene": {"scene_name": "scene-2"},
            "top_lane": {"scene_name": "scene-1", "primary_driver": "seed_adjacency"},
        },
    )
    _write_text(
        creator_market_dir / "scene_market_pulse.csv",
        "scene_name,family_count\nscene-2,3\n",
    )
    _write_text(
        creator_market_dir / "opportunity_lane_atlas.csv",
        "scene_name,primary_driver\nscene-1,seed_adjacency\n",
    )
    _write_json(
        creator_market_dir / "creator_market_manifest.json",
        {"report_family_count": 3},
    )

    run_dir = output_root / "runs" / "run_full_anchor"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / "safety_platform_contract.md", "# Safety Contract\n")
    _write_json(run_dir / "safety_platform_contract.json", {"benchmark_contract_version": "2026-week10-v1"})
    _write_json(
        output_root / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": "run_full_anchor"},
            "primary_claim": {"key": "shift_robustness", "status": "analysis_ready"},
            "backup_claim": {"key": "candidate_ranking", "status": "promising_but_unlocked"},
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


def test_build_branch_portfolio_report_distinguishes_four_primary_branches(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _build_sample_outputs(output_root)

    report = build_branch_portfolio_report(output_root)

    branches = {branch["key"]: branch for branch in report["branches"]}
    assert len(branches) == 4
    assert branches["taste_os"]["status"] == "ready"
    assert branches["control_room"]["status"] == "ready"
    assert branches["creator_intelligence"]["status"] == "ready"
    assert branches["safety_research"]["status"] == "ready_with_gaps"
    assert "audience" in branches["taste_os"]
    assert "success_metric" in branches["safety_research"]
    assert any("report_family.md" in artifact for artifact in branches["creator_intelligence"]["artifacts"])
    assert any("creator_market_brief.md" in artifact for artifact in branches["creator_intelligence"]["artifacts"])
    assert any("scene_market_pulse.csv" in artifact for artifact in branches["creator_intelligence"]["artifacts"])
    assert branches["creator_intelligence"]["market_layer"]["status"] == "ready"
    assert "aggregates `3` creator families" in branches["creator_intelligence"]["market_layer"]["live_signal"]
    assert any("safety_platform_contract.md" in artifact for artifact in branches["safety_research"]["artifacts"])


def test_write_branch_portfolio_artifacts_creates_markdown_and_json(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _build_sample_outputs(output_root)

    report = build_branch_portfolio_report(output_root)
    paths = write_branch_portfolio_artifacts(report, output_dir=output_root)

    assert paths["json"].exists()
    assert paths["md"].exists()
    markdown = paths["md"].read_text(encoding="utf-8")
    assert "Higher-Level Branch Map" in markdown
    assert "Personal Taste OS" in markdown
    assert "Creator-market layer" in markdown
    assert "Priority Rules" in markdown
