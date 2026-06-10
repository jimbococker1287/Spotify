from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from datetime import timezone
import hashlib
import json
import logging
import math
from pathlib import Path

from .creator_benchmark_contract import describe_creator_evidence_contract
from .creator_benchmark_contract import evaluate_creator_opportunity_evidence
from .run_artifacts import safe_read_csv
from .run_artifacts import safe_read_json
from .run_artifacts import write_csv_rows
from .run_artifacts import write_json
from .run_artifacts import write_markdown


_PASSPORT_COLUMNS = [
    "passport_id",
    "artist_name",
    "market",
    "evidence_grade",
    "verified",
    "contract_version",
    "occurrence_count",
    "report_family_count",
    "report_family_ids",
    "dominant_scene",
    "dominant_primary_driver",
    "average_opportunity_score",
    "minimum_opportunity_score",
    "maximum_opportunity_score",
    "opportunity_score_range",
    "minimum_support_count",
    "catalog_metadata_coverage",
    "release_metadata_coverage",
    "latest_source_age_days",
    "claim",
    "limitations",
    "source_artifact_paths",
    "gate_statuses",
]


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: object) -> int | None:
    numeric = _safe_float(value)
    return int(numeric) if numeric is not None else None


def _parse_timestamp(value: object) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _family_timestamp(manifest: dict[str, object]) -> tuple[datetime | None, str]:
    packaging = manifest.get("packaging_metadata", {})
    packaging = packaging if isinstance(packaging, dict) else {}
    for key in ("normalized_at", "refreshed_at", "packaged_at", "generated_at", "created_at", "timestamp"):
        parsed = _parse_timestamp(packaging.get(key))
        if parsed is not None:
            return parsed, f"packaging_metadata.{key}"
    for key in (
        "normalized_at",
        "refreshed_at",
        "packaged_at",
        "generated_at",
        "created_at",
        "backfilled_artifact_index_at",
        "timestamp",
    ):
        parsed = _parse_timestamp(manifest.get(key))
        if parsed is not None:
            return parsed, key
    return None, ""


def _mode(values: list[str]) -> str:
    normalized = [_text(value) for value in values if _text(value)]
    if not normalized:
        return ""
    counts = Counter(normalized)
    return sorted(counts, key=lambda value: (-counts[value], value.casefold(), value))[0]


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _family_root(output_dir: Path) -> Path:
    return output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"


def _resolve_primary_json(root: Path, family_id: str, manifest: dict[str, object]) -> Path:
    conventional = root / f"{family_id}.json"
    if conventional.exists():
        return conventional
    referenced = Path(_text(manifest.get("primary_report_json"))).expanduser()
    return referenced if referenced.exists() else conventional


def _load_rows(path: Path) -> list[dict[str, object]]:
    frame = safe_read_csv(path)
    return frame.to_dict(orient="records") if not frame.empty else []


def _load_family_records(output_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    root = _family_root(output_dir)
    family_ids = {
        path.stem.removesuffix("_report_family")
        for path in root.glob("*_report_family.json")
        if path.is_file()
    }
    family_ids.update(
        path.stem.removesuffix("_opportunities")
        for path in root.glob("*_opportunities.csv")
        if path.is_file()
    )
    records: list[dict[str, object]] = []
    source_paths: set[str] = set()
    for family_id in sorted(family_ids):
        manifest_path = root / f"{family_id}_report_family.json"
        opportunities_path = root / f"{family_id}_opportunities.csv"
        nodes_path = root / f"{family_id}_nodes.csv"
        manifest = safe_read_json(manifest_path, default={})
        manifest = manifest if isinstance(manifest, dict) else {}
        primary_json_path = _resolve_primary_json(root, family_id, manifest)
        primary_payload = safe_read_json(primary_json_path, default={})
        primary_payload = primary_payload if isinstance(primary_payload, dict) else {}
        timestamp, timestamp_source = _family_timestamp(manifest)

        opportunities = _load_rows(opportunities_path)
        if not opportunities and isinstance(primary_payload.get("opportunities"), list):
            opportunities = [
                dict(row)
                for row in primary_payload["opportunities"]
                if isinstance(row, dict)
            ]
        nodes = _load_rows(nodes_path)
        if not nodes and isinstance(primary_payload.get("nodes"), list):
            nodes = [dict(row) for row in primary_payload["nodes"] if isinstance(row, dict)]
        node_lookup = {
            _text(row.get("artist_name")).casefold(): row
            for row in nodes
            if _text(row.get("artist_name"))
        }
        market = _text(primary_payload.get("market"))

        for path in (manifest_path, opportunities_path, nodes_path, primary_json_path):
            if path.exists():
                source_paths.add(str(path))
        for opportunity in opportunities:
            artist_name = _text(opportunity.get("artist_name"))
            node = node_lookup.get(artist_name.casefold(), {})
            node_support = _safe_int(node.get("local_play_count"))
            records.append(
                {
                    "report_family_id": family_id,
                    "market": market,
                    "family_timestamp": timestamp,
                    "family_timestamp_source": timestamp_source,
                    "artist_name": artist_name,
                    "scene_name": _text(opportunity.get("scene_name")),
                    "primary_driver": _text(opportunity.get("primary_driver")),
                    "opportunity_band": _text(opportunity.get("opportunity_band")),
                    "opportunity_score": _safe_float(opportunity.get("opportunity_score")),
                    "support_count": (
                        node_support
                        if node_support is not None
                        else _safe_int(
                            opportunity.get("local_play_count", opportunity.get("support_count"))
                        )
                    ),
                    "catalog_metadata_available": any(
                        _text(node.get(key)) or _text(opportunity.get(key))
                        for key in ("spotify_id", "spotify_url", "public_popularity", "followers_total")
                    ),
                    "release_metadata_available": any(
                        [
                            bool(
                                _text(node.get("latest_release_date"))
                                or _text(opportunity.get("latest_release_date"))
                            ),
                            _safe_float(node.get("days_since_latest_release")) is not None
                            or _safe_float(opportunity.get("days_since_latest_release")) is not None,
                            bool(
                                _text(node.get("dominant_release_labels")) not in {"", "[]"}
                                or _text(opportunity.get("dominant_release_labels")) not in {"", "[]"}
                            ),
                        ]
                    ),
                    "source_artifact_paths": sorted(
                        {
                            str(path)
                            for path in (manifest_path, opportunities_path, nodes_path, primary_json_path)
                            if path.exists()
                        }
                    ),
                }
            )
    return records, sorted(source_paths)


def _evaluation_anchor(
    records: list[dict[str, object]],
    as_of: datetime | None,
) -> tuple[datetime | None, str]:
    if as_of is not None:
        resolved = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc), "caller_supplied"
    current_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return current_day, "current_utc_day"


def _passport_id(artist_name: str, market: str) -> str:
    normalized = f"{market.strip().casefold()}|{artist_name.strip().casefold()}".encode("utf-8")
    return f"creator-opportunity-{hashlib.sha256(normalized).hexdigest()[:16]}"


def _build_claim(artist_name: str, scene: str, driver: str, family_count: int) -> str:
    scene_text = scene or "an unresolved scene"
    driver_text = driver or "an unresolved driver"
    return (
        f"Saved creator opportunity artifacts provide a directional signal for {artist_name} in {scene_text} "
        f"through {driver_text}, observed across {family_count} report family/families; this is not a forecast "
        "of audience, revenue, release, or commercial outcomes."
    )


def _build_limitations(
    *,
    family_count: int,
    support_values: list[int],
    occurrence_count: int,
    catalog_metadata_count: int,
    release_metadata_count: int,
    timestamp_count: int,
) -> list[str]:
    limitations = [
        "The passport summarizes saved observational artifacts and does not establish causality or future demand.",
        "Upstream opportunity scores are preserved as raw ranking signals and are not recalibrated by this evidence grade.",
    ]
    if family_count < 2:
        limitations.append("Cross-family recurrence is not established.")
    if not support_values:
        limitations.append("No direct support count is available; score magnitude is not used as a substitute.")
    if catalog_metadata_count < occurrence_count:
        limitations.append(
            f"Catalog metadata covers {catalog_metadata_count} of {occurrence_count} saved occurrences."
        )
    if release_metadata_count < occurrence_count:
        limitations.append(
            f"Release metadata covers {release_metadata_count} of {occurrence_count} saved occurrences."
        )
    if timestamp_count < occurrence_count:
        limitations.append(
            f"Saved source timestamps cover {timestamp_count} of {occurrence_count} occurrences."
        )
    return limitations


def build_creator_evidence_passports(
    *,
    output_dir: Path,
    logger,
    as_of: datetime | None = None,
) -> dict[str, object]:
    records, consumed_source_paths = _load_family_records(output_dir)
    evaluation_anchor, evaluation_anchor_source = _evaluation_anchor(records, as_of)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        artist_name = _text(record.get("artist_name"))
        market = _text(record.get("market"))
        grouped.setdefault((market.casefold(), artist_name.casefold()), []).append(record)

    passports: list[dict[str, object]] = []
    for (_market_key, _artist_key), occurrences in sorted(grouped.items()):
        occurrences = sorted(
            occurrences,
            key=lambda row: (
                _text(row.get("report_family_id")),
                _text(row.get("scene_name")),
                _text(row.get("primary_driver")),
            ),
        )
        artist_name = _mode([_text(row.get("artist_name")) for row in occurrences])
        market = _mode([_text(row.get("market")) for row in occurrences])
        family_ids = sorted({_text(row.get("report_family_id")) for row in occurrences if _text(row.get("report_family_id"))})
        scores = [
            float(value)
            for value in (row.get("opportunity_score") for row in occurrences)
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        scenes = [_text(row.get("scene_name")) for row in occurrences]
        drivers = [_text(row.get("primary_driver")) for row in occurrences]
        support_values = [
            int(value)
            for value in (row.get("support_count") for row in occurrences)
            if isinstance(value, int)
        ]
        catalog_metadata_count = sum(bool(row.get("catalog_metadata_available")) for row in occurrences)
        release_metadata_count = sum(bool(row.get("release_metadata_available")) for row in occurrences)
        timestamps = [
            value
            for value in (row.get("family_timestamp") for row in occurrences)
            if isinstance(value, datetime)
        ]
        latest_source_age_days = (
            (evaluation_anchor - max(timestamps)).total_seconds() / 86400.0
            if evaluation_anchor is not None and timestamps
            else None
        )
        dominant_scene = _mode(scenes)
        dominant_driver = _mode(drivers)
        claim = _build_claim(artist_name, dominant_scene, dominant_driver, len(family_ids))
        limitations = _build_limitations(
            family_count=len(family_ids),
            support_values=support_values,
            occurrence_count=len(occurrences),
            catalog_metadata_count=catalog_metadata_count,
            release_metadata_count=release_metadata_count,
            timestamp_count=len(timestamps),
        )
        evaluation = evaluate_creator_opportunity_evidence(
            artist_name=artist_name,
            occurrence_count=len(occurrences),
            family_count=len(family_ids),
            score_values=scores,
            scene_values=scenes,
            driver_values=drivers,
            support_values=support_values,
            catalog_metadata_count=catalog_metadata_count,
            release_metadata_count=release_metadata_count,
            timestamped_occurrence_count=len(timestamps),
            latest_source_age_days=latest_source_age_days,
            claim_text=claim,
            limitations=limitations,
        )
        source_artifact_paths = sorted(
            {
                path
                for row in occurrences
                for path in row.get("source_artifact_paths", [])
                if _text(path)
            }
        )
        score_range = max(scores) - min(scores) if len(scores) >= 2 else None
        passports.append(
            {
                "passport_id": _passport_id(artist_name, market),
                "artist_name": artist_name,
                "market": market,
                "evidence_grade": evaluation["evidence_grade"],
                "verified": bool(evaluation["verified"]),
                "contract_version": evaluation["contract_version"],
                "occurrence_count": len(occurrences),
                "report_family_count": len(family_ids),
                "report_family_ids": family_ids,
                "dominant_scene": dominant_scene,
                "dominant_primary_driver": dominant_driver,
                "opportunity_bands": sorted({_text(row.get("opportunity_band")) for row in occurrences if _text(row.get("opportunity_band"))}),
                "average_opportunity_score": sum(scores) / len(scores) if scores else None,
                "minimum_opportunity_score": min(scores) if scores else None,
                "maximum_opportunity_score": max(scores) if scores else None,
                "opportunity_score_range": score_range,
                "minimum_support_count": min(support_values) if support_values else None,
                "catalog_metadata_coverage": catalog_metadata_count / len(occurrences) if occurrences else 0.0,
                "release_metadata_coverage": release_metadata_count / len(occurrences) if occurrences else 0.0,
                "latest_source_age_days": latest_source_age_days,
                "claim": claim,
                "limitations": limitations,
                "gates": evaluation["gates"],
                "source_artifact_paths": source_artifact_paths,
            }
        )
    grade_order = {"publishable": 0, "watch_only": 1, "suppress": 2}
    passports.sort(
        key=lambda row: (
            grade_order.get(_text(row.get("evidence_grade")), 99),
            -int(row.get("report_family_count", 0) or 0),
            _text(row.get("artist_name")).casefold(),
            _text(row.get("passport_id")),
        )
    )

    output_root = output_dir / "analysis" / "creator_evidence_lab"
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "creator_opportunity_evidence_passports.json"
    csv_path = output_root / "creator_opportunity_evidence_passports.csv"
    markdown_path = output_root / "creator_opportunity_evidence_passports.md"
    manifest_path = output_root / "creator_evidence_manifest.json"
    grade_counts = {
        grade: sum(_text(row.get("evidence_grade")) == grade for row in passports)
        for grade in ("publishable", "watch_only", "suppress")
    }
    verified_opportunity_count = sum(
        int(row.get("occurrence_count", 0) or 0)
        for row in passports
        if bool(row.get("verified"))
    )

    write_json(json_path, passports, sort_keys=True)
    write_csv_rows(
        csv_path,
        [
            {
                **{column: row.get(column) for column in _PASSPORT_COLUMNS},
                "report_family_ids": _json_text(row.get("report_family_ids", [])),
                "limitations": _json_text(row.get("limitations", [])),
                "source_artifact_paths": _json_text(row.get("source_artifact_paths", [])),
                "gate_statuses": _json_text(
                    {
                        _text(gate.get("key")): _text(gate.get("status"))
                        for gate in row.get("gates", [])
                        if isinstance(gate, dict)
                    }
                ),
            }
            for row in passports
        ],
        fieldnames=_PASSPORT_COLUMNS,
    )
    markdown_lines = [
        "# Creator Opportunity Evidence Passports",
        "",
        f"- Raw saved opportunity rows: `{len(records)}`",
        f"- Opportunity passports: `{len(passports)}`",
        f"- Verified raw opportunity rows: `{verified_opportunity_count}`",
        f"- Publishable passports: `{grade_counts['publishable']}`",
        f"- Watch-only passports: `{grade_counts['watch_only']}`",
        f"- Suppressed passports: `{grade_counts['suppress']}`",
        f"- Evaluation anchor: `{evaluation_anchor.isoformat() if evaluation_anchor else 'unavailable'}`",
        f"- Evaluation anchor source: `{evaluation_anchor_source}`",
        "",
        "A publishable grade means every contract gate passed. Raw opportunity scores remain unchanged and are not outcome forecasts.",
        "",
    ]
    for passport in passports:
        markdown_lines.extend(
            [
                f"## {passport['artist_name']}",
                "",
                f"- Grade: `{passport['evidence_grade']}`",
                f"- Families: `{passport['report_family_count']}`",
                f"- Raw average opportunity score: `{passport['average_opportunity_score']}`",
                f"- Claim: {passport['claim']}",
                "- Limitations: " + " ".join(str(item) for item in passport["limitations"]),
                "- Gates: "
                + ", ".join(
                    f"`{gate['key']}={gate['status']}`"
                    for gate in passport["gates"]
                ),
                "",
            ]
        )
    write_markdown(markdown_path, markdown_lines)
    artifact_paths = {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "manifest": str(manifest_path),
    }
    manifest = {
        "contract": describe_creator_evidence_contract(),
        "evaluation_anchor": evaluation_anchor.isoformat() if evaluation_anchor else None,
        "evaluation_anchor_source": evaluation_anchor_source,
        "raw_opportunity_count": len(records),
        "passport_count": len(passports),
        "verified_opportunity_count": verified_opportunity_count,
        "verified_passport_count": grade_counts["publishable"],
        "grade_counts": grade_counts,
        "consumed_source_artifact_paths": consumed_source_paths,
        "artifact_paths": artifact_paths,
    }
    write_json(manifest_path, manifest, sort_keys=True)
    logger.info(
        "Built %d creator opportunity evidence passports from %d raw opportunity rows.",
        len(passports),
        len(records),
    )
    return {
        "passports": passports,
        "manifest": manifest,
        "paths": [json_path, csv_path, markdown_path, manifest_path],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic creator opportunity evidence passports.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory containing saved creator artifacts.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = build_creator_evidence_passports(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        logger=logging.getLogger("spotify.creator_evidence_lab"),
    )
    for path in result["paths"]:
        print(path)
    return 0 if result["paths"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
