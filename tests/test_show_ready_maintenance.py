from __future__ import annotations

import json
import os
from pathlib import Path

from spotify.show_ready_maintenance import build_show_ready_maintenance_report, write_show_ready_maintenance_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _touch(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_creator_manifest(root: Path, stem: str, *, include_index: bool) -> None:
    base_dir = root / "outputs/analysis/public_spotify/creator_label_intelligence"
    primary_report = base_dir / f"{stem}.md"
    ranking_md = base_dir / f"{stem}_ranking_comparison.md"
    scene_md = base_dir / f"{stem}_scene_comparison.md"
    seed_md = base_dir / f"{stem}_seed_comparison.md"
    scene_seed_md = base_dir / f"{stem}_scene_seed_comparison.md"
    lane_md = base_dir / f"{stem}_opportunity_lane_comparison.md"
    strategy_md = base_dir / f"{stem}_scene_strategy_watch.md"
    for path in (primary_report, ranking_md, scene_md, seed_md, scene_seed_md, lane_md, strategy_md):
        _touch(path, f"# {path.stem}\n")
    manifest = {
        "primary_report": str(primary_report.resolve()),
        "comparison_view_markdown": {
            "ranking_comparison": str(ranking_md.resolve()),
            "scene_comparison": str(scene_md.resolve()),
            "seed_comparison": str(seed_md.resolve()),
            "scene_seed_comparison": str(scene_seed_md.resolve()),
            "opportunity_lane_comparison": str(lane_md.resolve()),
        },
        "comparison_view_csv": {},
        "brief_view_markdown": {"scene_strategy_watch": str(strategy_md.resolve())},
        "brief_view_csv": {},
    }
    if include_index:
        index_md = base_dir / f"{stem}_report_family.md"
        _touch(index_md, "# report family\n")
        manifest["artifact_index_markdown"] = str(index_md.resolve())
    _write_json(base_dir / f"{stem}_report_family.json", manifest)


def _write_day_90_launch(root: Path, *, canonical_paths: list[Path], release_status: str = "show_ready_with_notes") -> None:
    rows = [
        {"key": f"artifact_{idx}", "label": f"Artifact {idx}", "artifact": str(path.resolve())}
        for idx, path in enumerate(canonical_paths, start=1)
    ]
    _write_json(
        root / "outputs/analysis/day_90_launch/day_90_launch.json",
        {
            "release_status": release_status,
            "canonical_artifacts": rows,
            "delivery_checklist": [],
        },
    )
    _touch(root / "outputs/analysis/day_90_launch/day_90_launch.md")


def test_show_ready_maintenance_flags_anchor_and_backfill_gaps(tmp_path: Path) -> None:
    for stem in ("creator_a", "creator_b", "creator_c"):
        _create_creator_manifest(tmp_path, stem, include_index=False)

    _write_json(
        tmp_path / "outputs/analytics/control_room.json",
        {
            "latest_run": {"run_id": "run_gap"},
            "operating_rhythm": {"overall_status": "stale", "recommended_review_command": "make control-room"},
            "ops_health": {"status": "attention"},
        },
    )
    _write_json(
        tmp_path / "outputs/analysis/research_claims/research_claims.json",
        {"run": {"run_id": "run_gap", "profile": "full"}},
    )
    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.json",
        {"review_order": ["one", "two", "three", "four"]},
    )

    canonical_one = tmp_path / "outputs/analysis/day_90_launch/canonical/front_door.html"
    canonical_two = tmp_path / "outputs/analysis/day_90_launch/canonical/taste_os_demo.md"
    _touch(canonical_one)
    _touch(canonical_two)
    _write_day_90_launch(tmp_path, canonical_paths=[canonical_one, canonical_two])

    run_dir = tmp_path / "outputs/runs/run_gap"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_manifest.json", {"run_id": "run_gap"})
    old_ts = run_dir.stat().st_mtime - 3600
    os.utime(canonical_one, (old_ts, old_ts))
    os.utime(canonical_two, (old_ts, old_ts))

    report = build_show_ready_maintenance_report(tmp_path / "outputs")

    assert report["overall_status"] == "attention"
    assert report["anchor_alignment"]["status"] == "attention"
    assert report["creator_index_coverage"]["missing_count"] == 3
    assert report["safety_platform_contract"]["present"] is False
    assert report["canonical_artifact_freshness"]["stale_count"] == 2
    assert report["cadence"]["cadence_status"] == "stale"
    assert any("show-ready-backfill" in action for action in report["next_actions"])

    artifacts = write_show_ready_maintenance_report(report, output_dir=tmp_path / "outputs")
    assert artifacts["json"].exists()
    assert artifacts["md"].exists()
    assert "Show-Ready Maintenance" in artifacts["md"].read_text(encoding="utf-8")


def test_show_ready_maintenance_reports_ready_when_package_is_aligned(tmp_path: Path) -> None:
    for stem in ("creator_a", "creator_b", "creator_c"):
        _create_creator_manifest(tmp_path, stem, include_index=True)

    run_dir = tmp_path / "outputs/runs/run_ready"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_manifest.json", {"run_id": "run_ready"})
    _touch(run_dir / "safety_platform_contract.md")
    _write_json(run_dir / "safety_platform_contract.json", {"benchmark_contract_version": "v1"})

    _write_json(
        tmp_path / "outputs/analytics/control_room.json",
        {
            "latest_run": {"run_id": "run_ready"},
            "operating_rhythm": {"overall_status": "healthy", "recommended_review_command": "make control-room"},
            "ops_health": {"status": "healthy"},
        },
    )
    _write_json(
        tmp_path / "outputs/analysis/research_claims/research_claims.json",
        {"run": {"run_id": "run_ready", "profile": "full"}},
    )
    _write_json(
        tmp_path / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.json",
        {"run_id": "run_ready", "generated_at": "2026-05-02T00:00:00+00:00"},
    )

    canonical_one = tmp_path / "outputs/analysis/day_90_launch/canonical/front_door.html"
    canonical_two = tmp_path / "outputs/analysis/day_90_launch/canonical/taste_os_demo.md"
    _touch(canonical_one)
    _touch(canonical_two)
    _write_day_90_launch(tmp_path, canonical_paths=[canonical_one, canonical_two], release_status="launch_ready")

    report = build_show_ready_maintenance_report(tmp_path / "outputs")

    assert report["overall_status"] == "ready"
    assert report["anchor_alignment"]["aligned_branch_count"] == 3
    assert report["creator_index_coverage"]["status"] == "ready"
    assert report["safety_platform_contract"]["present"] is True
    assert report["canonical_artifact_freshness"]["status"] == "ready"
    assert report["cadence"]["status"] == "ready"
