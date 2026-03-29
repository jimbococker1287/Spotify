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
    ranking_view = creator_dir / "sample_ranking_view.md"
    scene_view = creator_dir / "sample_scene_view.md"
    seed_view = creator_dir / "sample_seed_view.md"
    for path in (primary_report, scene_seed_view, ranking_view, scene_view, seed_view):
        _write_text(path, f"# {path.stem}\n")
    _write_json(
        creator_dir / "sample_report_family.json",
        {
            "primary_report": str(primary_report),
            "comparison_view_markdown": {
                "ranking_comparison": str(ranking_view),
                "scene_comparison": str(scene_view),
                "seed_comparison": str(seed_view),
                "scene_seed_comparison": str(scene_seed_view),
            },
        },
    )

    _write_json(
        output_root / "analysis" / "research_claims" / "research_claims.json",
        {
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
    assert "Priority Rules" in markdown

