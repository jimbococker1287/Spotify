from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.listener_archetypes import build_listener_archetypes


def test_build_listener_archetypes_from_warehouse_daily_activity(tmp_path: Path, monkeypatch) -> None:
    logger = logging.getLogger("spotify.test.listener_archetypes")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    monkeypatch.setenv("SPOTIFY_LISTENER_ARCHETYPE_CLUSTERS", "4")

    warehouse_dir = tmp_path / "outputs" / "analytics" / "warehouse" / "silver"
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(
        [
            {
                "played_date": "2026-01-01",
                "total_streams": 50,
                "total_ms_played": 10_800_000,
                "unique_artists": 5,
                "unique_tracks": 16,
                "skip_rate": 0.05,
                "shuffle_rate": 0.05,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-01-02",
                "total_streams": 48,
                "total_ms_played": 10_500_000,
                "unique_artists": 4,
                "unique_tracks": 14,
                "skip_rate": 0.04,
                "shuffle_rate": 0.04,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-01-03",
                "total_streams": 46,
                "total_ms_played": 9_900_000,
                "unique_artists": 4,
                "unique_tracks": 15,
                "skip_rate": 0.06,
                "shuffle_rate": 0.06,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-02-01",
                "total_streams": 14,
                "total_ms_played": 1_500_000,
                "unique_artists": 12,
                "unique_tracks": 13,
                "skip_rate": 0.18,
                "shuffle_rate": 0.88,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-02-02",
                "total_streams": 12,
                "total_ms_played": 1_300_000,
                "unique_artists": 11,
                "unique_tracks": 12,
                "skip_rate": 0.16,
                "shuffle_rate": 0.84,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-02-03",
                "total_streams": 13,
                "total_ms_played": 1_400_000,
                "unique_artists": 12,
                "unique_tracks": 13,
                "skip_rate": 0.20,
                "shuffle_rate": 0.82,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-03-01",
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
                "played_date": "2026-03-02",
                "total_streams": 38,
                "total_ms_played": 8_800_000,
                "unique_artists": 4,
                "unique_tracks": 13,
                "skip_rate": 0.08,
                "shuffle_rate": 0.1,
                "offline_rate": 0.72,
                "primary_platform": "offline",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-03-03",
                "total_streams": 39,
                "total_ms_played": 8_900_000,
                "unique_artists": 5,
                "unique_tracks": 15,
                "skip_rate": 0.09,
                "shuffle_rate": 0.12,
                "offline_rate": 0.68,
                "primary_platform": "offline",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-04-01",
                "total_streams": 18,
                "total_ms_played": 2_200_000,
                "unique_artists": 10,
                "unique_tracks": 15,
                "skip_rate": 0.62,
                "shuffle_rate": 0.92,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-04-02",
                "total_streams": 16,
                "total_ms_played": 2_000_000,
                "unique_artists": 9,
                "unique_tracks": 14,
                "skip_rate": 0.60,
                "shuffle_rate": 0.90,
                "offline_rate": 0.0,
                "primary_platform": "ios",
                "track_stream_share": 1.0,
            },
            {
                "played_date": "2026-04-03",
                "total_streams": 17,
                "total_ms_played": 2_100_000,
                "unique_artists": 10,
                "unique_tracks": 15,
                "skip_rate": 0.64,
                "shuffle_rate": 0.94,
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
    regime_shifts = pd.read_csv(output_root / "taste_evolution_regime_shifts.csv")
    seasonal = pd.read_csv(output_root / "listener_archetype_seasonal.csv")
    evolution_summary = json.loads((output_root / "taste_evolution_brief.json").read_text(encoding="utf-8"))
    evolution_text = (output_root / "taste_evolution_brief.md").read_text(encoding="utf-8")

    assert len(assignments.index) == 12
    assert "archetype_label" in assignments.columns
    assert int(summary["cluster_count"]) == 4
    assert "Taste State Brief" in brief_text
    assert list(regime_shifts["month"].astype(str)) == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert bool(regime_shifts["dominant_changed_from_prev_month"].fillna(False).astype(bool).iloc[1:].any())
    assert set(seasonal["season"].astype(str)) == {"spring", "winter"}
    assert "Taste Evolution Brief" in evolution_text
    assert evolution_summary["status"] == "ok"
    assert evolution_summary["biggest_regime_shift"]["month"] in {"2026-02", "2026-03", "2026-04"}
