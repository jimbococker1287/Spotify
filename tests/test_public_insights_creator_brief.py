from __future__ import annotations

import json
from pathlib import Path

from spotify.public_insights import (
    _creator_brief_executive_summary,
    _creator_brief_migration_watch,
    _creator_brief_opportunity_lane_comparison,
    _creator_brief_priority_shortlist,
    _creator_brief_ranking_comparison,
    _creator_brief_release_watch,
    _creator_brief_scene_comparison,
    _creator_brief_scene_strategy_watch,
    _creator_brief_scene_seed_comparison,
    _creator_brief_seed_comparison,
)
from spotify.public_insights_creator_brief import (
    _apply_creator_evidence_to_payload,
    _load_creator_evidence_status,
    normalize_creator_report_family_manifest,
)


def _payload() -> dict[str, object]:
    return {
        "graph_summary": {
            "node_count": 6,
            "scene_count": 2,
            "opportunity_count": 3,
        },
        "scenes": [
            {
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "artist_count": 4,
                "seed_count": 2,
                "scene_local_play_share": 0.62,
                "scene_release_pressure": 1.02,
                "scene_label_concentration": 0.66,
            },
            {
                "scene_id": 1,
                "scene_name": "rap / trap",
                "artist_count": 2,
                "seed_count": 0,
                "scene_local_play_share": 0.18,
                "scene_release_pressure": 0.31,
                "scene_label_concentration": 0.88,
            },
        ],
        "artist_adjacency": [
            {
                "source_artist": "Artist A",
                "target_artist": "Artist C",
                "hybrid_score": 0.72,
                "transition_share": 0.28,
            },
            {
                "source_artist": "Artist B",
                "target_artist": "Emerging E",
                "hybrid_score": 0.69,
                "transition_share": 0.19,
            },
        ],
        "release_whitespace": [
            {
                "artist_name": "Artist C",
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "release_whitespace_score": 1.31,
            }
        ],
        "fan_migration": [
            {
                "source_artist": "Artist A",
                "target_artist": "Artist B",
                "source_scene_id": 0,
                "target_scene_id": 0,
                "source_out_share": 0.41,
                "target_in_share": 0.33,
            }
        ],
        "opportunities": [
            {
                "artist_name": "Emerging E",
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "opportunity_score": 0.77,
                "opportunity_rank": 1,
                "opportunity_band": "priority_now",
                "primary_driver": "seed_adjacency",
                "connected_seed_artists": ["Artist B"],
                "dominant_release_labels": ["Indie Arc"],
                "adjacency_component": 0.25,
                "migration_component": 0.08,
                "freshness_component": 0.12,
                "whitespace_component": 0.05,
                "scene_momentum_component": 0.05,
                "label_concentration_component": 0.03,
                "local_gap_component": 0.12,
                "popularity_tail_component": 0.07,
                "why_now": "Emerging E is the cleanest adjacency bridge out of Artist B inside `indie pop / alt z`.",
            },
            {
                "artist_name": "Artist C",
                "scene_id": 0,
                "scene_name": "indie pop / alt z",
                "opportunity_score": 0.66,
                "opportunity_rank": 2,
                "opportunity_band": "priority_now",
                "primary_driver": "release_whitespace",
                "connected_seed_artists": ["Artist A"],
                "dominant_release_labels": ["Indie Arc"],
                "adjacency_component": 0.19,
                "migration_component": 0.06,
                "freshness_component": 0.10,
                "whitespace_component": 0.12,
                "scene_momentum_component": 0.05,
                "label_concentration_component": 0.03,
                "local_gap_component": 0.07,
                "popularity_tail_component": 0.04,
                "why_now": "Artist C shows whitespace in `indie pop / alt z`, which makes the lane feel under-served.",
            },
            {
                "artist_name": "Artist D",
                "scene_id": 1,
                "scene_name": "rap / trap",
                "opportunity_score": 0.34,
                "opportunity_rank": 3,
                "opportunity_band": "explore",
                "primary_driver": "fan_migration",
                "connected_seed_artists": ["Artist B"],
                "dominant_release_labels": ["Trapline"],
                "adjacency_component": 0.09,
                "migration_component": 0.11,
                "freshness_component": 0.02,
                "whitespace_component": 0.01,
                "scene_momentum_component": 0.02,
                "label_concentration_component": 0.05,
                "local_gap_component": 0.03,
                "popularity_tail_component": 0.01,
                "why_now": "Audience movement already points toward Artist D, making `rap / trap` the strongest migration lane.",
            },
        ],
    }


def test_creator_brief_scene_comparison_surfaces_top_scene_and_opportunity_counts() -> None:
    rows = _creator_brief_scene_comparison(_payload())

    assert rows[0]["scene_name"] == "indie pop / alt z"
    assert rows[0]["opportunity_count"] == 2
    assert rows[0]["top_opportunity_artist"] == "Emerging E"
    assert rows[0]["priority_now_count"] == 2
    assert rows[0]["scene_release_pressure"] == 1.02


def test_creator_brief_seed_comparison_surfaces_best_bridge_per_seed() -> None:
    rows = _creator_brief_seed_comparison(_payload())

    assert rows[0]["seed_artist"] == "Artist A"
    assert rows[0]["top_adjacent_artist"] == "Artist C"
    assert rows[0]["top_scene_name"] == "indie pop / alt z"
    assert rows[1]["seed_artist"] == "Artist B"
    assert rows[1]["scene_coverage_count"] >= 1


def test_creator_brief_ranking_comparison_surfaces_score_breakdown() -> None:
    rows = _creator_brief_ranking_comparison(_payload())

    assert rows[0]["artist_name"] == "Emerging E"
    assert rows[0]["release_component"] == 0.17
    assert rows[0]["scene_component"] == 0.08
    assert rows[0]["gap_component"] == 0.19


def test_creator_brief_scene_seed_comparison_surfaces_cross_view() -> None:
    rows = _creator_brief_scene_seed_comparison(_payload())

    assert rows[0]["scene_name"] == "indie pop / alt z"
    assert rows[0]["seed_artist"] in {"Artist A", "Artist B"}
    assert rows[0]["bridge_artist_count"] >= 1


def test_creator_brief_opportunity_lane_comparison_surfaces_driver_and_posture() -> None:
    rows = _creator_brief_opportunity_lane_comparison(_payload())

    assert rows[0]["scene_name"] == "indie pop / alt z"
    assert rows[0]["primary_driver"] in {"seed_adjacency", "release_whitespace"}
    assert rows[0]["priority_now_count"] >= 1
    assert rows[0]["lane_posture"] in {
        "adjacency_expansion",
        "cadence_capture",
        "competitive_scene",
        "watch",
    }


def test_creator_brief_scene_strategy_watch_combines_release_label_and_migration() -> None:
    rows = _creator_brief_scene_strategy_watch(_payload())

    assert rows[0]["scene_name"] == "indie pop / alt z"
    assert rows[0]["release_whitespace_anchor_artist"] == "Artist C"
    assert rows[0]["incoming_migration_share"] == 0.41
    assert rows[0]["strategy_posture"] in {"accelerate_capture", "protect_window", "steady_watch"}


def test_creator_brief_priority_shortlist_surfaces_rank_driver_and_seed_bridge() -> None:
    rows = _creator_brief_priority_shortlist(_payload())

    assert rows[0]["artist_name"] == "Emerging E"
    assert rows[0]["opportunity_rank"] == 1
    assert rows[0]["primary_driver"] == "seed_adjacency"
    assert rows[0]["connected_seed_artists"] == ["Artist B"]
    assert "evidence_status" not in rows[0]


def test_creator_brief_migration_watch_and_release_watch_surface_top_rows() -> None:
    migration_rows = _creator_brief_migration_watch(_payload())
    release_rows = _creator_brief_release_watch(_payload())

    assert migration_rows[0]["source_artist"] == "Artist A"
    assert migration_rows[0]["target_artist"] == "Artist B"
    assert release_rows[0]["artist_name"] == "Artist C"
    assert release_rows[0]["release_whitespace_score"] == 1.31


def test_creator_brief_executive_summary_calls_out_scene_opportunity_and_migration() -> None:
    lines = _creator_brief_executive_summary(_payload())

    assert any("6" in line and "2" in line for line in lines)
    assert any("Emerging E" in line for line in lines)
    assert any("Artist A -> Artist B" in line for line in lines)
    assert any("strongest opportunity lane" in line for line in lines)


def test_normalize_creator_report_family_manifest_recovers_refreshable_anchor_views(tmp_path) -> None:
    report_dir = tmp_path / "creator_label_intelligence"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = "creator_label_intelligence_demo"

    (report_dir / f"{stem}.md").write_text("# Brief\n", encoding="utf-8")
    (report_dir / f"{stem}.json").write_text("{}", encoding="utf-8")
    lane_md = report_dir / f"{stem}_opportunity_lane_comparison.md"
    lane_csv = report_dir / f"{stem}_opportunity_lane_comparison.csv"
    strategy_md = report_dir / f"{stem}_scene_strategy_watch.md"
    strategy_csv = report_dir / f"{stem}_scene_strategy_watch.csv"
    lane_md.write_text("# Lane\n", encoding="utf-8")
    lane_csv.write_text("scene_name,primary_driver\nscene-1,seed_adjacency\n", encoding="utf-8")
    strategy_md.write_text("# Strategy\n", encoding="utf-8")
    strategy_csv.write_text("scene_name,strategy_posture\nscene-1,accelerate_capture\n", encoding="utf-8")

    manifest = {
        "comparison_view_markdown": {},
        "comparison_view_csv": {},
        "brief_view_markdown": {},
        "brief_view_csv": {},
        "reading_order": ["primary_report"],
    }

    normalized = normalize_creator_report_family_manifest(
        manifest,
        report_dir=report_dir,
        stem=stem,
        refreshed_at="2026-05-05T00:00:00+00:00",
        refresh_source="unit_test",
    )

    assert normalized["primary_report"] == str((report_dir / f"{stem}.md").resolve())
    assert normalized["primary_report_json"] == str((report_dir / f"{stem}.json").resolve())
    assert normalized["comparison_view_markdown"]["opportunity_lane_comparison"] == str(lane_md.resolve())
    assert normalized["comparison_view_csv"]["opportunity_lane_comparison"] == str(lane_csv.resolve())
    assert normalized["brief_view_markdown"]["scene_strategy_watch"] == str(strategy_md.resolve())
    assert normalized["brief_view_csv"]["scene_strategy_watch"] == str(strategy_csv.resolve())
    assert normalized["reading_order"] == ["primary_report", "opportunity_lane_comparison", "scene_strategy_watch"]

    packaging = normalized["packaging_metadata"]
    assert packaging["normalized_at"] == "2026-05-05T00:00:00+00:00"
    assert packaging["refresh_source"] == "unit_test"
    assert packaging["refresh_anchor_ready"] is True
    assert packaging["anchor_views"]["opportunity_lane"]["ready"] is True
    assert packaging["anchor_views"]["scene_strategy"]["ready"] is True
    assert packaging["view_inventory"]["opportunity_lane_comparison"]["legacy_markdown_filenames"] == [
        f"{stem}_opportunity_lane_comparison.md"
    ]
    assert "creator_evidence_status" not in normalized


def test_normalize_creator_report_family_manifest_reanchors_stale_workspace_paths(tmp_path) -> None:
    report_dir = tmp_path / "creator_label_intelligence"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = "creator_label_intelligence_legacy"

    primary_md = report_dir / f"{stem}.md"
    primary_json = report_dir / f"{stem}.json"
    report_family_md = report_dir / f"{stem}_report_family.md"
    lane_md = report_dir / f"{stem}_opportunity_lane_comparison.md"
    lane_csv = report_dir / f"{stem}_opportunity_lane_comparison.csv"
    strategy_md = report_dir / f"{stem}_scene_strategy_watch.md"
    strategy_csv = report_dir / f"{stem}_scene_strategy_watch.csv"
    for path, content in (
        (primary_md, "# Brief\n"),
        (primary_json, "{}"),
        (report_family_md, "# Report Family\n"),
        (lane_md, "# Lane\n"),
        (lane_csv, "scene_name,primary_driver\nscene-1,seed_adjacency\n"),
        (strategy_md, "# Strategy\n"),
        (strategy_csv, "scene_name,strategy_posture\nscene-1,accelerate_capture\n"),
    ):
        path.write_text(content, encoding="utf-8")

    manifest = {
        "primary_report": f"/tmp/old-workspace/{primary_md.name}",
        "primary_report_json": f"/tmp/old-workspace/{primary_json.name}",
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
    }

    normalized = normalize_creator_report_family_manifest(
        manifest,
        report_dir=report_dir,
        stem=stem,
        refreshed_at="2026-05-05T12:00:00+00:00",
        refresh_source="unit_test",
    )

    assert normalized["primary_report"] == str(primary_md.resolve())
    assert normalized["primary_report_json"] == str(primary_json.resolve())
    assert normalized["artifact_index_markdown"] == str(report_family_md.resolve())
    assert normalized["comparison_view_markdown"]["opportunity_lane_comparison"] == str(lane_md.resolve())
    assert normalized["comparison_view_csv"]["opportunity_lane_comparison"] == str(lane_csv.resolve())
    assert normalized["brief_view_markdown"]["scene_strategy_watch"] == str(strategy_md.resolve())
    assert normalized["brief_view_csv"]["scene_strategy_watch"] == str(strategy_csv.resolve())

    repair_summary = normalized["packaging_metadata"]["repair_summary"]
    assert repair_summary["primary_report_resolution_source"] == "basename_reanchor"
    assert repair_summary["primary_report_json_resolution_source"] == "basename_reanchor"
    assert repair_summary["artifact_index_resolution_source"] == "basename_reanchor"
    assert repair_summary["reanchored_reference_count"] == 7
    assert repair_summary["conventional_recovery_count"] == 0


def test_creator_brief_ingests_evidence_and_downgrades_unverified_rows_to_watch(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    evidence_root = output_dir / "analysis" / "creator_evidence_lab"
    evidence_root.mkdir(parents=True, exist_ok=True)
    source_ready = output_dir / "source_ready.json"
    source_watch = output_dir / "source_watch.json"
    source_ready.write_text("{}", encoding="utf-8")
    source_watch.write_text("{}", encoding="utf-8")

    passports_path = evidence_root / "creator_opportunity_evidence_passports.json"
    csv_path = evidence_root / "creator_opportunity_evidence_passports.csv"
    markdown_path = evidence_root / "creator_opportunity_evidence_passports.md"
    manifest_path = evidence_root / "creator_evidence_manifest.json"
    csv_path.write_text("passport_id\n", encoding="utf-8")
    markdown_path.write_text("# Evidence\n", encoding="utf-8")
    passports = [
        {
            "passport_id": "ready",
            "artist_name": "Emerging E",
            "market": "US",
            "evidence_grade": "publishable",
            "verified": True,
            "contract_version": "2026-06-v1",
            "claim": "Saved evidence is directional and not a forecast.",
            "limitations": ["Observational only."],
            "source_artifact_paths": [str(source_ready)],
            "gates": [
                {
                    "key": "evidence_freshness",
                    "status": "pass",
                    "observed": {"latest_source_age_days": 2},
                    "threshold": {"maximum_source_age_days": 90},
                }
            ],
        },
        {
            "passport_id": "watch",
            "artist_name": "Artist C",
            "market": "US",
            "evidence_grade": "watch_only",
            "verified": False,
            "contract_version": "2026-06-v1",
            "claim": "Saved evidence is directional and not a forecast.",
            "limitations": ["Recurrence is incomplete."],
            "source_artifact_paths": [str(source_watch)],
            "gates": [
                {
                    "key": "evidence_freshness",
                    "status": "watch",
                    "observed": {"latest_source_age_days": 20},
                    "threshold": {"maximum_source_age_days": 90},
                }
            ],
        },
    ]
    passports_path.write_text(json.dumps(passports), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "contract": {
                    "contract_version": "2026-06-v1",
                    "verified_grade": "publishable",
                },
                "passport_count": 2,
                "grade_counts": {"publishable": 1, "watch_only": 1, "suppress": 0},
                "artifact_paths": {
                    "json": str(passports_path),
                    "csv": str(csv_path),
                    "markdown": str(markdown_path),
                    "manifest": str(manifest_path),
                },
            }
        ),
        encoding="utf-8",
    )

    evidence_status = _load_creator_evidence_status(output_dir)
    payload = _payload()
    payload["market"] = "US"
    _apply_creator_evidence_to_payload(payload, evidence_status)

    assert evidence_status is not None
    assert evidence_status["contract_status"] == "ready"
    assert evidence_status["artifact_path_status"] == "ready"
    opportunities = {row["artist_name"]: row for row in payload["opportunities"]}
    assert opportunities["Emerging E"]["evidence_verified"] is True
    assert opportunities["Emerging E"]["evidence_adjusted_band"] == "priority_now"
    assert opportunities["Artist C"]["evidence_status"] == "watch"
    assert opportunities["Artist C"]["evidence_adjusted_band"] == "watchlist"
    assert "Directional watch signal only" in opportunities["Artist C"]["why_now"]
    assert opportunities["Artist D"]["evidence_status"] == "missing"
    assert "not a confident immediate priority" in opportunities["Artist D"]["why_now"]

    shortlist = _creator_brief_priority_shortlist(payload)
    assert shortlist[0]["evidence_status"] == "verified"
    assert shortlist[1]["opportunity_band"] == "watchlist"
    directional_payload = dict(payload)
    directional_payload["opportunities"] = [opportunities["Artist C"]]
    executive_summary = _creator_brief_executive_summary(directional_payload)
    assert any("directional/watch" in line for line in executive_summary)

    report_dir = output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"
    report_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_creator_report_family_manifest(
        {},
        report_dir=report_dir,
        stem="creator_label_intelligence_evidence",
    )
    assert normalized["creator_evidence_status"]["contract_status"] == "ready"
    assert normalized["packaging_metadata"]["creator_evidence"]["passport_count"] == 2
