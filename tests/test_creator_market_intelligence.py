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


def _logger() -> logging.Logger:
    logger = logging.getLogger("spotify.test.creator_market_intelligence")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_build_creator_market_intelligence_rolls_up_report_families(tmp_path: Path) -> None:
    logger = _logger()
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
    lane_atlas = pd.read_csv(output_root / "opportunity_lane_atlas.csv")
    migration_network = pd.read_csv(output_root / "market_migration_network.csv")
    whitespace_atlas = pd.read_csv(output_root / "release_whitespace_atlas.csv")
    manifest_payload = json.loads((output_root / "creator_market_manifest.json").read_text(encoding="utf-8"))
    brief_payload = json.loads((output_root / "creator_market_brief.json").read_text(encoding="utf-8"))
    brief_text = (output_root / "creator_market_brief.md").read_text(encoding="utf-8")

    assert scene_pulse.iloc[0]["scene_name"] == "scene-2"
    seed_lane = lane_atlas.loc[lane_atlas["primary_driver"].eq("seed_adjacency")].iloc[0]
    assert float(seed_lane["avg_opportunity_score"]) == 0.475
    assert migration_network.iloc[0]["target_artist"] in {"Artist 4", "Artist 2"}
    assert not whitespace_atlas.empty
    assert manifest_payload["report_family_count"] == 2
    assert manifest_payload["manifest_backed_report_family_count"] == 2
    assert manifest_payload["asset_backed_report_family_count"] == 2
    assert manifest_payload["complete_report_family_count"] == 0
    assert manifest_payload["partial_report_family_count"] == 2
    assert manifest_payload["partial_report_family_ids"] == [family_a, family_b]
    assert manifest_payload["raw_opportunity_count"] == 3
    assert manifest_payload["verified_opportunity_count"] == 0
    assert manifest_payload["evidence_passport_count"] == 3
    assert set(manifest_payload["evidence_artifact_paths"]) == {"csv", "json", "manifest", "markdown"}
    assert all(Path(path).exists() for path in manifest_payload["evidence_artifact_paths"].values())
    assert brief_payload["report_family_count"] == 2
    assert brief_payload["raw_opportunity_count"] == 3
    assert brief_payload["verified_opportunity_count"] == 0
    assert "Creator Market Brief" in brief_text
    assert "aggregating `2` creator report families" in brief_text
    assert "Evidence passports verify `0` of `3`" in brief_text


def test_build_creator_market_intelligence_counts_complete_and_partial_families(tmp_path: Path) -> None:
    logger = _logger()
    root = tmp_path / "outputs" / "analysis" / "public_spotify" / "creator_label_intelligence"
    complete_family = "creator_label_intelligence_complete-family"
    partial_family = "creator_label_intelligence_partial-family"

    _write_json(root / f"{complete_family}_report_family.json", {"primary_report": "complete.md"})
    _write_csv(
        root / f"{complete_family}_ranking_comparison.csv",
        [
            {
                "artist_name": "Artist Complete",
                "scene_name": "scene-complete",
                "primary_driver": "seed_adjacency",
                "opportunity_score": 0.51,
            }
        ],
    )
    _write_csv(
        root / f"{complete_family}_scene_comparison.csv",
        [
            {
                "scene_id": 1,
                "scene_name": "scene-complete",
                "scene_local_play_share": 0.61,
                "avg_opportunity_score": 0.52,
                "priority_now_count": 2,
                "watchlist_count": 0,
                "scene_release_pressure": 0.18,
                "scene_label_concentration": 0.09,
                "top_opportunity_artist": "Artist Complete",
                "top_opportunity_score": 0.53,
            }
        ],
    )
    _write_csv(
        root / f"{complete_family}_opportunities.csv",
        [
            {
                "artist_name": "Artist Complete",
                "scene_name": "scene-complete",
                "primary_driver": "seed_adjacency",
                "opportunity_band": "priority_now",
                "opportunity_score": 0.53,
                "scene_local_play_share": 0.61,
                "scene_release_pressure": 0.18,
                "scene_label_concentration": 0.09,
                "fan_migration_score": 0.48,
                "release_whitespace_score": 0.42,
                "local_gap_score": 0.37,
                "scene_momentum_score": 0.58,
                "connected_seed_artists": json.dumps(["Seed Complete"]),
                "dominant_release_labels": json.dumps(["Indie"]),
                "days_since_latest_release": 90,
            }
        ],
    )
    _write_csv(
        root / f"{complete_family}_migration_watch.csv",
        [
            {
                "source_artist": "Artist Source",
                "target_artist": "Artist Complete",
                "source_scene_id": 1,
                "target_scene_id": 1,
                "source_out_share": 0.26,
                "target_in_share": 0.31,
                "transition_count": 28,
            }
        ],
    )
    seed_bridge_rows = [
        {
            "scene_name": "scene-complete",
            "seed_artist": "Seed Complete",
            "avg_opportunity_score": 0.52,
            "bridge_artist_count": 2,
            "opportunity_count": 1,
            "scene_local_play_share": 0.61,
            "scene_release_pressure": 0.18,
            "scene_label_concentration": 0.09,
            "top_driver": "seed_adjacency",
            "top_opportunity_artist": "Artist Complete",
        }
    ]
    _write_csv(root / f"{complete_family}_seed_comparison.csv", seed_bridge_rows)
    _write_csv(root / f"{complete_family}_scene_seed_comparison.csv", seed_bridge_rows)
    _write_csv(
        root / f"{partial_family}_scene_seed_comparison.csv",
        [
            {
                "scene_name": "scene-partial",
                "seed_artist": "Seed Partial",
                "avg_opportunity_score": 0.27,
                "bridge_artist_count": 1,
                "opportunity_count": 1,
                "scene_local_play_share": 0.22,
                "scene_release_pressure": 0.12,
                "scene_label_concentration": 0.16,
                "top_driver": "migration_capture",
                "top_opportunity_artist": "Artist Partial",
            }
        ],
    )

    paths = build_creator_market_intelligence(output_dir=tmp_path / "outputs", logger=logger)

    assert paths
    output_root = tmp_path / "outputs" / "analysis" / "creator_market_intelligence"
    manifest_payload = json.loads((output_root / "creator_market_manifest.json").read_text(encoding="utf-8"))
    brief_payload = json.loads((output_root / "creator_market_brief.json").read_text(encoding="utf-8"))

    assert manifest_payload["report_family_count"] == 2
    assert manifest_payload["manifest_backed_report_family_count"] == 1
    assert manifest_payload["asset_backed_report_family_count"] == 2
    assert manifest_payload["complete_report_family_count"] == 1
    assert manifest_payload["partial_report_family_count"] == 1
    assert manifest_payload["partial_report_family_ids"] == [partial_family]
    assert brief_payload["report_family_count"] == 2


def test_build_creator_market_intelligence_writes_multi_family_trend_deltas(tmp_path: Path) -> None:
    logger = _logger()
    root = tmp_path / "outputs" / "analysis" / "public_spotify" / "creator_label_intelligence"
    early_family = "creator_label_intelligence_alpha-family"
    late_family = "creator_label_intelligence_beta-family"

    _write_json(
        root / f"{early_family}_report_family.json",
        {
            "primary_report": "alpha.md",
            "packaging_metadata": {"normalized_at": "2026-01-01T00:00:00+00:00"},
        },
    )
    _write_json(
        root / f"{late_family}_report_family.json",
        {
            "primary_report": "beta.md",
            "packaging_metadata": {"normalized_at": "2026-02-01T00:00:00+00:00"},
        },
    )
    _write_csv(
        root / f"{early_family}_scene_comparison.csv",
        [
            {
                "scene_id": 2,
                "scene_name": "scene-neon",
                "scene_local_play_share": 0.10,
                "avg_opportunity_score": 0.20,
                "priority_now_count": 1,
                "watchlist_count": 0,
                "scene_release_pressure": 0.05,
                "scene_label_concentration": 0.10,
                "top_opportunity_artist": "Artist Early",
                "top_opportunity_score": 0.20,
            }
        ],
    )
    _write_csv(
        root / f"{late_family}_scene_comparison.csv",
        [
            {
                "scene_id": 2,
                "scene_name": "scene-neon",
                "scene_local_play_share": 0.35,
                "avg_opportunity_score": 0.42,
                "priority_now_count": 4,
                "watchlist_count": 1,
                "scene_release_pressure": 0.20,
                "scene_label_concentration": 0.08,
                "top_opportunity_artist": "Artist Late",
                "top_opportunity_score": 0.55,
            }
        ],
    )
    _write_csv(
        root / f"{early_family}_opportunities.csv",
        [
            {
                "artist_name": "Lane Artist Early",
                "scene_name": "scene-neon",
                "primary_driver": "seed_adjacency",
                "opportunity_band": "watchlist",
                "opportunity_score": 0.30,
                "scene_local_play_share": 0.10,
                "scene_release_pressure": 0.05,
                "scene_label_concentration": 0.10,
                "fan_migration_score": 0.40,
                "release_whitespace_score": 0.20,
                "local_gap_score": 0.20,
                "scene_momentum_score": 0.30,
                "connected_seed_artists": json.dumps(["Seed Early"]),
                "dominant_release_labels": json.dumps(["Indie"]),
                "days_since_latest_release": 30,
            }
        ],
    )
    _write_csv(
        root / f"{late_family}_opportunities.csv",
        [
            {
                "artist_name": "Lane Artist Late",
                "scene_name": "scene-neon",
                "primary_driver": "seed_adjacency",
                "opportunity_band": "priority_now",
                "opportunity_score": 0.55,
                "scene_local_play_share": 0.35,
                "scene_release_pressure": 0.20,
                "scene_label_concentration": 0.08,
                "fan_migration_score": 0.65,
                "release_whitespace_score": 0.60,
                "local_gap_score": 0.45,
                "scene_momentum_score": 0.70,
                "connected_seed_artists": json.dumps(["Seed Late"]),
                "dominant_release_labels": json.dumps(["Indie"]),
                "days_since_latest_release": 240,
            }
        ],
    )
    _write_csv(
        root / f"{early_family}_migration_watch.csv",
        [
            {
                "source_artist": "Route Source",
                "target_artist": "Route Target",
                "source_scene_id": 2,
                "target_scene_id": 2,
                "source_out_share": 0.20,
                "target_in_share": 0.15,
                "transition_count": 25,
            }
        ],
    )
    _write_csv(
        root / f"{late_family}_migration_watch.csv",
        [
            {
                "source_artist": "Route Source",
                "target_artist": "Route Target",
                "source_scene_id": 2,
                "target_scene_id": 2,
                "source_out_share": 0.36,
                "target_in_share": 0.31,
                "transition_count": 80,
            }
        ],
    )

    paths = build_creator_market_intelligence(output_dir=tmp_path / "outputs", logger=logger)

    output_root = tmp_path / "outputs" / "analysis" / "creator_market_intelligence"
    trend_csv = output_root / "creator_market_trend_deltas.csv"
    trend_json = output_root / "creator_market_trend_deltas.json"
    trend_md = output_root / "creator_market_trend_deltas.md"
    strategy_csv = output_root / "creator_market_strategy_cards.csv"
    strategy_json = output_root / "creator_market_strategy_cards.json"
    strategy_md = output_root / "creator_market_strategy_cards.md"
    trend_deltas = pd.read_csv(trend_csv)
    strategy_cards = pd.read_csv(strategy_csv)
    strategy_payload = json.loads(strategy_json.read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_root / "creator_market_manifest.json").read_text(encoding="utf-8"))
    brief_payload = json.loads((output_root / "creator_market_brief.json").read_text(encoding="utf-8"))
    brief_markdown = (output_root / "creator_market_brief.md").read_text(encoding="utf-8")
    markdown_text = trend_md.read_text(encoding="utf-8")
    strategy_markdown = strategy_md.read_text(encoding="utf-8")

    assert trend_csv in paths
    assert trend_json in paths
    assert trend_md in paths
    assert strategy_csv in paths
    assert strategy_json in paths
    assert strategy_md in paths
    assert set(trend_deltas["signal_type"]) >= {
        "rising_scene",
        "repeated_opportunity_lane",
        "repeated_migration_route",
        "stale_release_whitespace",
    }
    rising_scene = trend_deltas.loc[trend_deltas["signal_type"].eq("rising_scene")].iloc[0]
    assert rising_scene["scene_name"] == "scene-neon"
    assert rising_scene["first_report_family_id"] == early_family
    assert rising_scene["latest_report_family_id"] == late_family
    assert float(rising_scene["delta_value"]) > 0.0

    repeated_lane = trend_deltas.loc[trend_deltas["signal_type"].eq("repeated_opportunity_lane")].iloc[0]
    assert repeated_lane["signal_key"] == "scene-neon / seed_adjacency"
    assert int(repeated_lane["family_count"]) == 2

    repeated_route = trend_deltas.loc[trend_deltas["signal_type"].eq("repeated_migration_route")].iloc[0]
    assert repeated_route["signal_key"] == "Route Source -> Route Target"
    assert int(repeated_route["family_count"]) == 2

    stale_release = trend_deltas.loc[trend_deltas["signal_type"].eq("stale_release_whitespace")].iloc[0]
    assert stale_release["signal_key"] == "Lane Artist Late"
    assert float(stale_release["latest_value"]) == 240.0

    assert len(json.loads(trend_json.read_text(encoding="utf-8"))) == len(trend_deltas.index)
    assert manifest_payload["tables"]["creator_market_trend_deltas"]["row_count"] == len(trend_deltas.index)
    assert manifest_payload["tables"]["creator_market_trend_deltas"]["markdown_path"] == str(trend_md)
    assert brief_payload["trend_delta_counts"]["rising_scene"] == 1
    assert "Rising Scenes" in markdown_text
    assert "Repeated Migration Routes" in markdown_text

    assert set(strategy_cards["card_type"]) == {
        "scene_momentum",
        "opportunity_lane",
        "migration_route",
        "release_whitespace_gap",
    }
    assert strategy_cards["rank"].tolist() == list(range(1, len(strategy_cards.index) + 1))
    assert strategy_cards["card_id"].is_unique
    assert strategy_cards.iloc[0]["card_type"] == "scene_momentum"
    repeated_cards = strategy_cards.loc[
        strategy_cards["card_type"].isin(["scene_momentum", "opportunity_lane", "migration_route"])
    ]
    assert set(repeated_cards["confidence"]) == {"cross_family_partial"}
    assert set(repeated_cards["priority"]) == {"medium"}
    whitespace_card = strategy_cards.loc[strategy_cards["card_type"].eq("release_whitespace_gap")].iloc[0]
    assert whitespace_card["confidence"] == "single_family_validation"
    assert whitespace_card["priority"] == "low"
    assert "Verify" in whitespace_card["card_name"]

    assert len(strategy_payload) == len(strategy_cards.index)
    assert all(isinstance(card["evidence_metrics"], dict) and card["evidence_metrics"] for card in strategy_payload)
    assert all(card["source_artifact_references"] for card in strategy_payload)
    assert all(
        reference["artifact"].endswith(".csv") and reference["selector"]
        for card in strategy_payload
        for reference in card["source_artifact_references"]
    )
    assert all(
        (output_root / reference["artifact"]).exists()
        for card in strategy_payload
        for reference in card["source_artifact_references"]
    )
    assert any(
        reference["artifact"] == "market_migration_network.csv"
        for card in strategy_payload
        for reference in card["source_artifact_references"]
    )
    assert manifest_payload["strategy_card_count"] == len(strategy_cards.index)
    assert manifest_payload["tables"]["creator_market_strategy_cards"]["row_count"] == len(strategy_cards.index)
    assert manifest_payload["strategy_card_artifact_paths"] == {
        "csv": str(strategy_csv),
        "json": str(strategy_json),
        "markdown": str(strategy_md),
    }
    assert brief_payload["strategy_card_count"] == len(strategy_cards.index)
    assert brief_payload["top_strategy_cards"][0]["card_id"] == strategy_payload[0]["card_id"]
    assert "Strategy cards rank" in " ".join(brief_payload["summary"])
    assert strategy_payload[0]["card_name"] in brief_markdown
    assert "Creator Market Strategy Cards" in strategy_markdown
    assert "Validation signal" in strategy_markdown

    first_ranking = strategy_cards[["rank", "card_id", "card_type"]].to_dict(orient="records")
    build_creator_market_intelligence(output_dir=tmp_path / "outputs", logger=logger)
    reranked_cards = pd.read_csv(strategy_csv)
    assert reranked_cards[["rank", "card_id", "card_type"]].to_dict(orient="records") == first_ranking


def test_build_creator_market_intelligence_flags_sparse_release_metadata(tmp_path: Path) -> None:
    logger = _logger()
    root = tmp_path / "outputs" / "analysis" / "public_spotify" / "creator_label_intelligence"
    family_a = "creator_label_intelligence_sparse-a"
    family_b = "creator_label_intelligence_sparse-b"

    _write_json(
        root / f"{family_a}_report_family.json",
        {
            "primary_report": "sparse-a.md",
            "packaging_metadata": {"normalized_at": "2026-01-01T00:00:00+00:00"},
        },
    )
    _write_json(
        root / f"{family_b}_report_family.json",
        {
            "primary_report": "sparse-b.md",
            "packaging_metadata": {"normalized_at": "2026-01-02T00:00:00+00:00"},
        },
    )
    for family_id, artist_name in [(family_a, "Sparse Artist A"), (family_b, "Sparse Artist B")]:
        _write_csv(
            root / f"{family_id}_opportunities.csv",
            [
                {
                    "artist_name": artist_name,
                    "scene_name": "scene-sparse",
                    "primary_driver": "release_whitespace",
                    "opportunity_band": "watchlist",
                    "opportunity_score": 0.22,
                    "scene_local_play_share": 0.10,
                    "scene_release_pressure": 0.05,
                    "scene_label_concentration": 0.10,
                    "fan_migration_score": 0.20,
                    "release_whitespace_score": 0.0,
                    "local_gap_score": 0.15,
                    "scene_momentum_score": 0.20,
                    "connected_seed_artists": json.dumps([]),
                    "dominant_release_labels": json.dumps([]),
                    "days_since_latest_release": "",
                }
            ],
        )

    build_creator_market_intelligence(output_dir=tmp_path / "outputs", logger=logger)

    output_root = tmp_path / "outputs" / "analysis" / "creator_market_intelligence"
    trend_deltas = pd.read_csv(output_root / "creator_market_trend_deltas.csv")
    whitespace_atlas = pd.read_csv(output_root / "release_whitespace_atlas.csv")
    strategy_cards = pd.read_csv(output_root / "creator_market_strategy_cards.csv")
    strategy_payload = json.loads(
        (output_root / "creator_market_strategy_cards.json").read_text(encoding="utf-8")
    )
    markdown_text = (output_root / "creator_market_trend_deltas.md").read_text(encoding="utf-8")

    sparse_rows = trend_deltas.loc[trend_deltas["signal_type"].eq("sparse_release_whitespace_coverage")]
    assert len(sparse_rows.index) == 1
    sparse_row = sparse_rows.iloc[0]
    assert sparse_row["signal_key"] == "release_metadata_coverage"
    assert float(sparse_row["coverage_ratio"]) == 0.0
    assert int(sparse_row["metadata_row_count"]) == 0
    assert int(sparse_row["opportunity_row_count"]) == 2
    assert sparse_row["severity"] == "high"
    assert whitespace_atlas.empty
    assert "coverage `0.000`" in markdown_text
    assert "release_whitespace_gap" not in set(strategy_cards["card_type"])
    evidence_card = strategy_cards.loc[
        strategy_cards["card_type"].eq("release_whitespace_evidence")
    ].iloc[0]
    assert evidence_card["confidence"] == "evidence_gap"
    assert evidence_card["priority"] == "medium"
    assert "Backfill release evidence" in evidence_card["card_name"]
    assert "0.750" in evidence_card["validation_signal"]
    evidence_payload = next(
        card for card in strategy_payload if card["card_type"] == "release_whitespace_evidence"
    )
    assert evidence_payload["evidence_metrics"]["coverage_ratio"] == 0.0
    assert evidence_payload["source_artifact_references"] == [
        {
            "artifact": "creator_market_trend_deltas.csv",
            "selector": (
                "signal_type=sparse_release_whitespace_coverage;"
                "signal_key=release_metadata_coverage"
            ),
        }
    ]
