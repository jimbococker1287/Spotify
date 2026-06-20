from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spotify.public_pretraining_preflight import (
    discover_public_pretraining_manifests,
    run_public_pretraining_preflight,
    validate_public_pretraining_manifest,
    validate_public_pretraining_records,
    write_public_pretraining_preflight_report,
)
from spotify.public_training_data import (
    CANONICAL_INTERACTION_COLUMNS,
    DatasetSourceManifest,
    SourceFileProvenance,
    write_source_manifest,
)


def _source_manifest(source: Path, *, approved: bool = True) -> DatasetSourceManifest:
    return DatasetSourceManifest(
        dataset_id="public-listens",
        display_name="Synthetic public listens",
        adapter="lfm_listening_log",
        version="test-v1",
        task_fit=("sequential recommendation",),
        required_columns=("user_id", "track_id", "timestamp"),
        license_name="CC BY 4.0",
        license_url="https://example.org/license",
        access_url="https://example.org/dataset",
        access_caveats=("Synthetic test data only.",),
        files=(
            SourceFileProvenance.from_local_file(
                source,
                source_url="https://example.org/source",
                acquired_at="2026-06-18T12:00:00+00:00",
            ),
        ),
        training_use_approved=approved,
        reviewed_by="test-reviewer" if approved else "",
        reviewed_at="2026-06-18T12:01:00+00:00" if approved else "",
    )


def _canonical_records(*, dataset_id: str = "public-listens") -> pd.DataFrame:
    rows = [
        ("u1", "s1", "i1", "2026-01-01T00:00:00Z", "train"),
        ("u1", "s1", "i2", "2026-01-01T00:01:00Z", "train"),
        ("u2", "s2", "i3", "2026-01-02T00:00:00Z", "pretrain"),
    ]
    frame = pd.DataFrame(pd.NA, index=range(len(rows)), columns=CANONICAL_INTERACTION_COLUMNS)
    frame["source_dataset"] = dataset_id
    frame["user_id"] = [row[0] for row in rows]
    frame["session_id"] = [row[1] for row in rows]
    frame["item_id"] = [row[2] for row in rows]
    frame["timestamp"] = [row[3] for row in rows]
    frame["event_type"] = "listen"
    frame["explicit_positive"] = True
    frame["split"] = [row[4] for row in rows]
    return frame


def test_discovers_only_local_dataset_manifest_candidates(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    source = root / "public.csv"
    source.write_text("user_id,track_id,timestamp\nu1,t1,2026-01-01T00:00:00Z\n", encoding="utf-8")
    manifest_path = write_source_manifest(
        root / "public_source_manifest.json",
        _source_manifest(source),
    )
    (root / "not_a_manifest.json").write_text('{"manifest": "nope"}', encoding="utf-8")

    discovered = discover_public_pretraining_manifests((root,))

    assert discovered == (manifest_path.resolve(),)


def test_approved_manifest_without_records_is_ready_and_json_friendly(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest_path = write_source_manifest(tmp_path / "manifest.json", _source_manifest(source))
    output_path = tmp_path / "preflight.json"

    report = run_public_pretraining_preflight(
        search_roots=(),
        manifest_paths=(manifest_path,),
        output_path=output_path,
    )
    payload = report.to_dict()

    assert report.status == "ready"
    assert payload["ready_manifest_count"] == 1
    assert payload["manifests"][0]["record_checks"] == []
    assert output_path.exists()
    json.dumps(payload)
    json.loads(output_path.read_text(encoding="utf-8"))


def test_unapproved_manifest_blocks_with_review_actions(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest_path = write_source_manifest(
        tmp_path / "manifest.json",
        _source_manifest(source, approved=False),
    )

    result = validate_public_pretraining_manifest(manifest_path)

    assert result.status == "blocked"
    assert "not approved" in result.reason
    assert any("training_use_approved" in action for action in result.next_actions)


def test_checksum_mismatch_blocks_with_reproducible_provenance_action(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("original", encoding="utf-8")
    manifest_path = write_source_manifest(tmp_path / "manifest.json", _source_manifest(source))
    source.write_text("modified", encoding="utf-8")

    result = validate_public_pretraining_manifest(manifest_path)

    assert result.status == "blocked"
    assert "SHA-256 mismatch" in result.reason
    assert any("Recompute files[].sha256" in action for action in result.next_actions)


def test_supplied_canonical_records_are_validated_against_manifest(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest = _source_manifest(source)
    manifest_path = write_source_manifest(tmp_path / "manifest.json", manifest)
    records_path = tmp_path / "canonical_records.csv"
    _canonical_records().to_csv(records_path, index=False)

    report = run_public_pretraining_preflight(
        search_roots=(),
        manifest_paths=(manifest_path,),
        records_by_manifest={"public-listens": (records_path,)},
    )
    record_check = report.manifests[0].record_checks[0]

    assert report.status == "ready"
    assert record_check.status == "ready"
    assert record_check.row_count == 3
    assert record_check.source_datasets == ("public-listens",)
    assert record_check.splits == ("pretrain", "train")
    assert record_check.min_timestamp == "2026-01-01T00:00:00+00:00"
    assert record_check.max_timestamp == "2026-01-02T00:00:00+00:00"


def test_wrong_record_dataset_blocks_with_source_dataset_action(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest = _source_manifest(source)
    result = validate_public_pretraining_records(_canonical_records(dataset_id="wrong"), manifest=manifest)

    assert result.status == "blocked"
    assert "source_dataset values must match" in result.reason
    assert any("source_dataset" in action for action in result.next_actions)


def test_missing_record_columns_blocks_with_canonical_schema_action(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest_path = write_source_manifest(tmp_path / "manifest.json", _source_manifest(source))
    records_path = tmp_path / "bad_records.csv"
    pd.DataFrame({"source_dataset": ["public-listens"], "item_id": ["i1"]}).to_csv(records_path, index=False)

    result = validate_public_pretraining_manifest(manifest_path, record_paths=(records_path,))

    assert result.status == "blocked"
    assert result.record_checks[0].status == "blocked"
    assert any("canonical interaction schema" in action for action in result.next_actions)


def test_network_record_paths_are_not_downloaded(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest = _source_manifest(source)

    result = validate_public_pretraining_manifest(
        manifest,
        record_paths=("https://example.org/canonical_records.csv",),
    )

    assert result.status == "blocked"
    assert result.record_checks[0].status == "blocked"
    assert "Network URLs are not accepted" in result.record_checks[0].reason
    assert any("will not download" in action for action in result.next_actions)


def test_write_report_round_trips_json(tmp_path: Path) -> None:
    report = run_public_pretraining_preflight(search_roots=(tmp_path,))

    output_path = write_public_pretraining_preflight_report(report, tmp_path / "report.json")

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["manifest_count"] == 0
