from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.creator_market_intelligence import build_creator_market_intelligence


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_creator_market_intelligence_rolls_up_report_families(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.creator_market_intelligence")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    root = tmp_path / "outputs" / "analysis" / "public_spotify" / "creator_label_intelligence"
    family_a = "creator_label_intelligence_seed-a-seed-b"
    family_b = "creator_label_intelligence_seed-c-seed-d"

    _write_json(root / f"{family_a}_report_family.json", {"primary_report": "a.md"})
    _write_json(root / f"{family_b}_report_family.json", {"primary_report": "b.md"})
    _write_csv(
        root / f"{family_a}_scene_comparison.csv",
        [
            {
                "scene_id": 1,
                "scene_name": "scene-1",
                "scene_local_play_share": 0.25,
                "avg_opportunity_score": 0.31,
                "priority_now_count": 2,
                "watchlist_count": 1,
                "scene_release_pressure": 0.10,
                "scene_label_concentration": 0.20,
                "top_opportunity_artist": "Artist 1",
                "top_opportunity_score": 0.35,
            },
            {
                "scene_id": 2,
                "scene_name": "scene-2",
                "scene_local_play_share": 0.55,
                "avg_opportunity_score": 0.42,
                "priority_now_count": 4,
                "watchlist_count": 1,
                "scene_release_pressure": 0.20,
                "scene_label_concentration": 0.10,
                "top_opportunity_artist": "Artist 2",
                "top_opportunity_score": 0.46,
            },
        ],
    )
    _write_csv(
        root / f"{family_b}_scene_comparison.csv",
        [
            {
                "scene_id": 1,
                "scene_name": "scene-1",
                "scene_local_play_share": 0.20,
                "avg_opportunity_score": 0.28,
                "priority_now_count": 1,
                "watchlist_count": 2,
                "scene_release_pressure": 0.15,
                "scene_label_concentration": 0.25,
                "top_opportunity_artist": "Artist 3",
                "top_opportunity_score": 0.33,
            },
            {
                "scene_id": 2,
                "scene_name": "scene-2",
                "scene_local_play_share": 0.60,
                "avg_opportunity_score": 0.45,
                "priority_now_count": 5,
                "watchlist_count": 1,
                "scene_release_pressure": 0.30,
                "scene_label_concentration": 0.05,
                "top_opportunity_artist": "Artist 4",
                "top_opportunity_score": 0.49,
            },
        ],
    )
    _write_csv(
        root / f"{family_a}_opportunities.csv",
        [
            {
                "artist_name": "Artist 2",
                "scene_name": "scene-2",
                "primary_driver": "seed_adjacency",
                "opportunity_band": "priority_now",
                "opportunity_score": 0.46,
                "scene_local_play_share": 0.55,
                "scene_release_pressure": 0.20,
                "scene_label_concentration": 0.10,
                "fan_migration_score": 0.62,
                "release_whitespace_score": 0.55,
                "local_gap_score": 0.41,
                "scene_momentum_score": 0.70,
                "connected_seed_artists": json.dumps(["Seed A", "Seed B"]),
                "dominant_release_labels": json.dumps(["Indie"]),
                "days_since_latest_release": 110,
            },
            {
                "artist_name": "Artist 1",
                "scene_name": "scene-1",
                "primary_driver": "migration_capture",
                "opportunity_band": "watchlist",
                "opportunity_score": 0.35,
                "scene_local_play_share": 0.25,
                "scene_release_pressure": 0.10,
                "scene_label_concentration": 0.20,
                "fan_migration_score": 0.51,
                "release_whitespace_score": 0.10,
                "local_gap_score": 0.28,
                "scene_momentum_score": 0.40,
                "connected_seed_artists": json.dumps(["Seed A"]),
                "dominant_release_labels": json.dumps([]),
                "days_since_latest_release": 10,
            },
        ],
    )
    _write_csv(
        root / f"{family_b}_opportunities.csv",
        [
            {
                "artist_name": "Artist 4",
                "scene_name": "scene-2",
                "primary_driver": "seed_adjacency",
                "opportunity_band": "priority_now",
                "opportunity_score": 0.49,
                "scene_local_play_share": 0.60,
                "scene_release_pressure": 0.30,
                "scene_label_concentration": 0.05,
                "fan_migration_score": 0.65,
                "release_whitespace_score": 0.70,
                "local_gap_score": 0.45,
                "scene_momentum_score": 0.75,
                "connected_seed_artists": json.dumps(["Seed C", "Seed D"]),
                "dominant_release_labels": json.dumps(["Alt"]),
                "days_since_latest_release": 180,
            }
        ],
    )
    _write_csv(
        root / f"{family_a}_migration_watch.csv",
        [
            {
                "source_artist": "Artist X",
                "target_artist": "Artist 2",
                "source_scene_id": 1,
                "target_scene_id": 2,
                "source_out_share": 0.31,
                "target_in_share": 0.28,
                "transition_count": 44,
            }
        ],
    )
    _write_csv(
        root / f"{family_b}_migration_watch.csv",
        [
            {
                "source_artist": "Artist Y",
                "target_artist": "Artist 4",
                "source_scene_id": 1,
                "target_scene_id": 2,
                "source_out_share": 0.40,
                "target_in_share": 0.36,
                "transition_count": 60,
            }
        ],
    )
    _write_csv(
        root / f"{family_a}_scene_seed_comparison.csv",
        [
            {
                "scene_name": "scene-2",
                "seed_artist": "Seed A",
                "avg_opportunity_score": 0.43,
                "bridge_artist_count": 3,
                "opportunity_count": 2,
                "scene_local_play_share": 0.55,
                "scene_release_pressure": 0.20,
                "scene_label_concentration": 0.10,
                "top_driver": "seed_adjacency",
                "top_opportunity_artist": "Artist 2",
            }
        ],
    )
    _write_csv(
        root / f"{family_b}_scene_seed_comparison.csv",
        [
            {
                "scene_name": "scene-2",
                "seed_artist": "Seed C",
                "avg_opportunity_score": 0.48,
                "bridge_artist_count": 4,
                "opportunity_count": 3,
                "scene_local_play_share": 0.60,
                "scene_release_pressure": 0.30,
                "scene_label_concentration": 0.05,
                "top_driver": "seed_adjacency",
                "top_opportunity_artist": "Artist 4",
            }
        ],
    )

    paths = build_creator_market_intelligence(output_dir=tmp_path / "outputs", logger=logger)

    assert paths
    output_root = tmp_path / "outputs" / "analysis" / "creator_market_intelligence"
    scene_pulse = pd.read_csv(output_root / "scene_market_pulse.csv")
    migration_network = pd.read_csv(output_root / "market_migration_network.csv")
    whitespace_atlas = pd.read_csv(output_root / "release_whitespace_atlas.csv")
    brief_text = (output_root / "creator_market_brief.md").read_text(encoding="utf-8")

    assert scene_pulse.iloc[0]["scene_name"] == "scene-2"
    assert migration_network.iloc[0]["target_artist"] in {"Artist 4", "Artist 2"}
    assert not whitespace_atlas.empty
    assert "Creator Market Brief" in brief_text
