from __future__ import annotations

import json
from pathlib import Path

from spotify.show_ready_backfill import backfill_show_ready_artifacts, write_show_ready_backfill_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _touch(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_legacy_creator_manifest(root: Path, stem: str) -> Path:
    base_dir = root / "outputs/analysis/public_spotify/creator_label_intelligence"
    primary_report = base_dir / f"{stem}.md"
    ranking_md = base_dir / f"{stem}_ranking_comparison.md"
    scene_md = base_dir / f"{stem}_scene_comparison.md"
    seed_md = base_dir / f"{stem}_seed_comparison.md"
    scene_seed_md = base_dir / f"{stem}_scene_seed_comparison.md"
    lane_md = base_dir / f"{stem}_opportunity_lane_comparison.md"
    strategy_md = base_dir / f"{stem}_scene_strategy_watch.md"
    lane_csv = base_dir / f"{stem}_opportunity_lane_comparison.csv"
    strategy_csv = base_dir / f"{stem}_scene_strategy_watch.csv"
    for path in (primary_report, ranking_md, scene_md, seed_md, scene_seed_md, lane_md, strategy_md):
        _touch(path, f"# {path.stem}\n")
    _touch(lane_csv, "scene_name,primary_driver\nscene-1,seed_adjacency\n")
    _touch(strategy_csv, "scene_name,strategy_posture\nscene-1,accelerate_capture\n")
    manifest_path = base_dir / f"{stem}_report_family.json"
    _write_json(
        manifest_path,
        {
            "primary_report": str(primary_report.resolve()),
            "comparison_view_markdown": {
                "ranking_comparison": str(ranking_md.resolve()),
                "scene_comparison": str(scene_md.resolve()),
                "seed_comparison": str(seed_md.resolve()),
                "scene_seed_comparison": str(scene_seed_md.resolve()),
            },
            "comparison_view_csv": {},
            "brief_view_markdown": {},
            "brief_view_csv": {},
        },
    )
    return manifest_path


def _create_stale_path_creator_manifest(root: Path, stem: str) -> Path:
    base_dir = root / "outputs/analysis/public_spotify/creator_label_intelligence"
    primary_report = base_dir / f"{stem}.md"
    report_family_md = base_dir / f"{stem}_report_family.md"
    lane_md = base_dir / f"{stem}_opportunity_lane_comparison.md"
    lane_csv = base_dir / f"{stem}_opportunity_lane_comparison.csv"
    strategy_md = base_dir / f"{stem}_scene_strategy_watch.md"
    strategy_csv = base_dir / f"{stem}_scene_strategy_watch.csv"
    for path in (primary_report, report_family_md, lane_md, strategy_md):
        _touch(path, f"# {path.stem}\n")
    _touch(lane_csv, "scene_name,primary_driver\nscene-1,seed_adjacency\n")
    _touch(strategy_csv, "scene_name,strategy_posture\nscene-1,accelerate_capture\n")
    manifest_path = base_dir / f"{stem}_report_family.json"
    _write_json(
        manifest_path,
        {
            "primary_report": f"/tmp/old-workspace/{primary_report.name}",
            "artifact_index_markdown": f"/tmp/old-workspace/{report_family_md.name}",
            "comparison_view_markdown": {
                "opportunity_lane_comparison": f"/tmp/old-workspace/{lane_md.name}",
            },
            "comparison_view_csv": {
                "opportunity_lane_comparison": f"/tmp/old-workspace/{lane_csv.name}",
            },
            "brief_view_markdown": {
                "scene_strategy_watch": f"/tmp/old-workspace/{strategy_md.name}",
            },
            "brief_view_csv": {
                "scene_strategy_watch": f"/tmp/old-workspace/{strategy_csv.name}",
            },
        },
    )
    return manifest_path


def test_show_ready_backfill_restores_legacy_indexes_and_safety_contract(tmp_path: Path) -> None:
    manifest_paths = [
        _create_legacy_creator_manifest(tmp_path, "creator_label_intelligence_indie"),
        _create_legacy_creator_manifest(tmp_path, "creator_label_intelligence_rap"),
    ]
    _write_json(
        tmp_path / "outputs/analysis/research_claims/research_claims.json",
        {"run": {"run_id": "run_demo", "profile": "full"}},
    )

    report = backfill_show_ready_artifacts(tmp_path / "outputs", refresh_artifacts=False)

    creator_report = report["creator_report_family_indexes"]
    assert creator_report["manifest_count"] == 2
    assert creator_report["backfilled_count"] == 2
    assert creator_report["normalized_count"] == 2
    assert creator_report["refresh_anchor_ready_count"] == 2
    assert report["safety_platform_contract"]["status"] == "backfilled"

    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = Path(str(payload["artifact_index_markdown"]))
        assert index_path.exists()
        assert payload["comparison_view_markdown"]["opportunity_lane_comparison"].endswith("_opportunity_lane_comparison.md")
        assert payload["comparison_view_csv"]["opportunity_lane_comparison"].endswith("_opportunity_lane_comparison.csv")
        assert payload["brief_view_markdown"]["scene_strategy_watch"].endswith("_scene_strategy_watch.md")
        assert payload["brief_view_csv"]["scene_strategy_watch"].endswith("_scene_strategy_watch.csv")
        assert payload["packaging_metadata"]["refresh_anchor_ready"] is True
        assert payload["packaging_metadata"]["anchor_views"]["opportunity_lane"]["ready"] is True
        assert payload["packaging_metadata"]["anchor_views"]["scene_strategy"]["ready"] is True

    assert (tmp_path / "outputs/runs/run_demo/safety_platform_contract.json").exists()
    assert (tmp_path / "outputs/runs/run_demo/safety_platform_contract.md").exists()

    artifacts = write_show_ready_backfill_report(report, output_dir=tmp_path / "outputs")
    assert artifacts["json"].exists()
    assert artifacts["md"].exists()
    assert "Show-Ready Backfill" in artifacts["md"].read_text(encoding="utf-8")


def test_show_ready_backfill_reanchors_stale_paths_without_regenerating_index(tmp_path: Path) -> None:
    manifest_path = _create_stale_path_creator_manifest(tmp_path, "creator_label_intelligence_stale")
    _write_json(
        tmp_path / "outputs/analysis/research_claims/research_claims.json",
        {"run": {"run_id": "run_demo", "profile": "full"}},
    )

    report = backfill_show_ready_artifacts(tmp_path / "outputs", refresh_artifacts=False)

    creator_report = report["creator_report_family_indexes"]
    assert creator_report["manifest_count"] == 1
    assert creator_report["backfilled_count"] == 0
    assert creator_report["already_present_count"] == 1
    assert creator_report["normalized_count"] == 1
    assert creator_report["reanchored_reference_count"] == 6
    assert creator_report["conventional_recovery_count"] == 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["primary_report"].endswith("creator_label_intelligence_stale.md")
    assert payload["artifact_index_markdown"].endswith("creator_label_intelligence_stale_report_family.md")
    assert payload["comparison_view_markdown"]["opportunity_lane_comparison"].endswith(
        "creator_label_intelligence_stale_opportunity_lane_comparison.md"
    )
    assert payload["brief_view_markdown"]["scene_strategy_watch"].endswith(
        "creator_label_intelligence_stale_scene_strategy_watch.md"
    )
    assert payload["packaging_metadata"]["repair_summary"]["reanchored_reference_count"] == 6
