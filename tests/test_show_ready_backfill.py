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
