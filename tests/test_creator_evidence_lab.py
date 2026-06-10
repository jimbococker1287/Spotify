from __future__ import annotations

import csv
from datetime import datetime
from datetime import timezone
import json
import logging
from pathlib import Path

from spotify.creator_evidence_lab import build_creator_evidence_passports


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _logger() -> logging.Logger:
    logger = logging.getLogger("spotify.test.creator_evidence_lab")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _write_family(
    root: Path,
    *,
    family_id: str,
    normalized_at: str,
    opportunities: list[dict[str, object]],
    nodes: list[dict[str, object]],
) -> None:
    _write_json(
        root / f"{family_id}_report_family.json",
        {
            "primary_report_json": str(root / f"{family_id}.json"),
            "packaging_metadata": {"normalized_at": normalized_at},
        },
    )
    _write_json(root / f"{family_id}.json", {"market": "US"})
    _write_csv(root / f"{family_id}_opportunities.csv", opportunities)
    _write_csv(root / f"{family_id}_nodes.csv", nodes)


def test_creator_evidence_lab_writes_deterministic_truthful_passports(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    root = output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"
    family_a = "creator_label_intelligence_family-a"
    family_b = "creator_label_intelligence_family-b"
    opportunity_columns = {
        "scene_name": "scene-a",
        "primary_driver": "seed_adjacency",
        "opportunity_band": "priority_now",
    }
    node_defaults = {
        "spotify_url": "https://open.spotify.com/artist/example",
        "public_popularity": 35,
        "followers_total": 1000,
        "latest_release_date": "2025-12-01",
        "days_since_latest_release": 45,
        "dominant_release_labels": '["Indie"]',
    }

    _write_family(
        root,
        family_id=family_a,
        normalized_at="2026-01-10T00:00:00+00:00",
        opportunities=[
            {"artist_name": "Artist Ready", "opportunity_score": 0.41, **opportunity_columns},
            {
                "artist_name": "Artist Watch",
                "opportunity_score": 0.38,
                **opportunity_columns,
            },
            {
                "artist_name": "Artist Suppress",
                "opportunity_score": 0.35,
                **opportunity_columns,
            },
        ],
        nodes=[
            {"artist_name": "Artist Ready", "local_play_count": 120, "spotify_id": "ready", **node_defaults},
            {"artist_name": "Artist Watch", "local_play_count": "", "spotify_id": "watch", **node_defaults},
            {
                "artist_name": "Artist Suppress",
                "local_play_count": 5,
                "spotify_id": "suppress",
                "spotify_url": "https://open.spotify.com/artist/suppress",
                "public_popularity": 20,
                "followers_total": 100,
                "latest_release_date": "",
                "days_since_latest_release": "",
                "dominant_release_labels": "[]",
            },
        ],
    )
    _write_family(
        root,
        family_id=family_b,
        normalized_at="2026-01-20T00:00:00+00:00",
        opportunities=[
            {"artist_name": "Artist Ready", "opportunity_score": 0.45, **opportunity_columns},
            {
                "artist_name": "Artist Suppress",
                "opportunity_score": 0.37,
                **opportunity_columns,
            },
        ],
        nodes=[
            {"artist_name": "Artist Ready", "local_play_count": 90, "spotify_id": "ready", **node_defaults},
            {
                "artist_name": "Artist Suppress",
                "local_play_count": 7,
                "spotify_id": "suppress",
                "spotify_url": "https://open.spotify.com/artist/suppress",
                "public_popularity": 20,
                "followers_total": 100,
                "latest_release_date": "",
                "days_since_latest_release": "",
                "dominant_release_labels": "[]",
            },
        ],
    )

    result = build_creator_evidence_passports(
        output_dir=output_dir,
        logger=_logger(),
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    first_bytes = {path.name: path.read_bytes() for path in result["paths"]}
    second_result = build_creator_evidence_passports(
        output_dir=output_dir,
        logger=_logger(),
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    assert first_bytes == {path.name: path.read_bytes() for path in second_result["paths"]}
    passports = {
        row["artist_name"]: row
        for row in json.loads(
            (
                output_dir
                / "analysis"
                / "creator_evidence_lab"
                / "creator_opportunity_evidence_passports.json"
            ).read_text(encoding="utf-8")
        )
    }
    manifest = json.loads(
        (
            output_dir / "analysis" / "creator_evidence_lab" / "creator_evidence_manifest.json"
        ).read_text(encoding="utf-8")
    )
    markdown = (
        output_dir
        / "analysis"
        / "creator_evidence_lab"
        / "creator_opportunity_evidence_passports.md"
    ).read_text(encoding="utf-8")

    assert passports["Artist Ready"]["evidence_grade"] == "publishable"
    assert passports["Artist Ready"]["verified"] is True
    assert passports["Artist Ready"]["average_opportunity_score"] == 0.43
    assert passports["Artist Watch"]["evidence_grade"] == "watch_only"
    assert passports["Artist Suppress"]["evidence_grade"] == "suppress"
    assert "not a forecast" in passports["Artist Ready"]["claim"]
    assert passports["Artist Suppress"]["release_metadata_coverage"] == 0.0
    assert all(passports["Artist Ready"]["source_artifact_paths"])

    assert manifest["raw_opportunity_count"] == 5
    assert manifest["verified_opportunity_count"] == 2
    assert manifest["passport_count"] == 3
    assert manifest["verified_passport_count"] == 1
    assert manifest["grade_counts"] == {"publishable": 1, "suppress": 1, "watch_only": 1}
    assert set(manifest["artifact_paths"]) == {"csv", "json", "manifest", "markdown"}
    assert "Raw opportunity scores remain unchanged" in markdown


def test_creator_evidence_lab_uses_current_day_for_default_freshness_anchor(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    root = output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"
    opportunity = {
        "artist_name": "Artist Old",
        "opportunity_score": 0.44,
        "scene_name": "scene-old",
        "primary_driver": "seed_adjacency",
        "opportunity_band": "priority_now",
    }
    node = {
        "artist_name": "Artist Old",
        "local_play_count": 100,
        "spotify_id": "old",
        "latest_release_date": "2019-01-01",
    }
    _write_family(
        root,
        family_id="creator_label_intelligence_old-a",
        normalized_at="2020-01-01T00:00:00+00:00",
        opportunities=[opportunity],
        nodes=[node],
    )
    _write_family(
        root,
        family_id="creator_label_intelligence_old-b",
        normalized_at="2020-01-02T00:00:00+00:00",
        opportunities=[opportunity],
        nodes=[node],
    )

    result = build_creator_evidence_passports(output_dir=output_dir, logger=_logger())

    assert result["manifest"]["evaluation_anchor_source"] == "current_utc_day"
    assert result["passports"][0]["evidence_grade"] == "watch_only"
    freshness_gate = next(
        gate for gate in result["passports"][0]["gates"] if gate["key"] == "evidence_freshness"
    )
    assert freshness_gate["status"] == "watch"
