from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.listener_archetypes import build_listener_archetypes


def test_build_listener_archetypes_from_warehouse_daily_activity(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.listener_archetypes")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    warehouse_dir = tmp_path / "outputs" / "analytics" / "warehouse" / "silver"
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(
        [
            {
                "played_date": "2026-01-01",
                "total_streams": 50,
                "total_ms_played": 10_800_000,
                "unique_artists": 6,
                "unique_tracks": 18,
                "skip_rate": 0.05,
                "shuffle_rate": 0.05,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-01-02",
                "total_streams": 12,
                "total_ms_played": 1_200_000,
                "unique_artists": 10,
                "unique_tracks": 11,
                "skip_rate": 0.55,
                "shuffle_rate": 0.8,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-01-03",
                "total_streams": 40,
                "total_ms_played": 9_000_000,
                "unique_artists": 5,
                "unique_tracks": 14,
                "skip_rate": 0.10,
                "shuffle_rate": 0.1,
                "offline_rate": 0.7,
                "primary_platform": "offline",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-01-04",
                "total_streams": 8,
                "total_ms_played": 700_000,
                "unique_artists": 3,
                "unique_tracks": 6,
                "skip_rate": 0.20,
                "shuffle_rate": 0.1,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
        ]
    )
    daily.to_parquet(warehouse_dir / "listener_daily_activity.parquet", index=False)

    paths = build_listener_archetypes(
        data_dir=tmp_path / "data" / "raw",
        output_dir=tmp_path / "outputs",
        include_video=False,
        logger=logger,
    )

    assert paths
    output_root = tmp_path / "outputs" / "analysis" / "listener_archetypes"
    assignments = pd.read_csv(output_root / "listener_archetype_assignments.csv")
    summary = json.loads((output_root / "listener_archetype_summary.json").read_text(encoding="utf-8"))
    brief_text = (output_root / "taste_state_brief.md").read_text(encoding="utf-8")

    assert len(assignments.index) == 4
    assert "archetype_label" in assignments.columns
    assert int(summary["cluster_count"]) >= 2
    assert "Taste State Brief" in brief_text
