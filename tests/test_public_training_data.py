from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from spotify.public_training_data import (
    CANONICAL_INTERACTION_COLUMNS,
    CANONICAL_ITEM_COLUMNS,
    DATASET_MANIFEST_TEMPLATES,
    CanonicalSchemaError,
    DatasetSourceManifest,
    SourceFileProvenance,
    SourceManifestError,
    SpotifyTrainingContentError,
    iter_fma_metadata,
    iter_kuairand_policy_logs,
    iter_lfm_listening_logs,
    iter_million_song_taste_profile,
    iter_music4all_interactions,
    iter_music4all_metadata,
    iter_open_bandit_policy_logs,
    load_source_manifest,
    validate_no_spotify_training_content,
    validate_source_manifest,
    write_source_manifest,
)


def _manifest(
    path: Path,
    *,
    adapter: str,
    dataset_id: str | None = None,
    required_columns: tuple[str, ...] = ("user_id", "item_id"),
) -> DatasetSourceManifest:
    return DatasetSourceManifest(
        dataset_id=dataset_id or adapter,
        display_name=f"Test {adapter}",
        adapter=adapter,
        version="test-v1",
        task_fit=("unit testing",),
        required_columns=required_columns,
        license_name="CC BY 4.0",
        license_url="https://example.org/license",
        access_url="https://example.org/dataset",
        access_caveats=("Test fixture only.",),
        files=(
            SourceFileProvenance.from_local_file(
                path,
                source_url="https://example.org/dataset/file",
                acquired_at="2026-06-13T12:00:00+00:00",
            ),
        ),
        training_use_approved=True,
        reviewed_by="test-suite",
        reviewed_at="2026-06-13T12:01:00+00:00",
    )


def test_taste_profile_adapter_streams_canonical_chunks(tmp_path: Path) -> None:
    source = tmp_path / "triplets.tsv"
    source.write_text("u1\ts1\t4\nu1\ts2\t1\nu2\ts3\t7\n", encoding="utf-8")
    manifest = _manifest(
        source,
        adapter="million_song_taste_profile",
        required_columns=("user_id", "song_id", "play_count"),
    )

    chunks = list(iter_million_song_taste_profile(source, manifest=manifest, chunksize=2))

    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert tuple(chunks[0].columns) == CANONICAL_INTERACTION_COLUMNS
    assert chunks[0]["item_id"].tolist() == ["s1", "s2"]
    assert chunks[0]["interaction_value"].tolist() == [4.0, 1.0]
    assert chunks[0]["event_type"].tolist() == ["play_count", "play_count"]
    assert chunks[0]["explicit_positive"].tolist() == [True, True]


def test_lfm_adapter_supports_timestamped_headerless_logs(tmp_path: Path) -> None:
    source = tmp_path / "user_events.txt"
    source.write_text("u1\ta1\tal1\tt1\t1700000000\nu2\ta2\tal2\tt2\t1700000060\n", encoding="utf-8")
    manifest = _manifest(
        source,
        adapter="lfm_listening_log",
        required_columns=("user_id", "track_id", "timestamp"),
    )

    frame = next(iter_lfm_listening_logs(source, manifest=manifest, has_header=False))

    assert frame["user_id"].tolist() == ["u1", "u2"]
    assert frame["item_id"].tolist() == ["t1", "t2"]
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"
    context = json.loads(frame.loc[0, "context_json"])
    assert context == {"album_id": "al1", "artist_id": "a1"}


def test_music4all_interactions_and_metadata_are_separate_streams(tmp_path: Path) -> None:
    interactions = tmp_path / "interactions.tsv"
    interactions.write_text("user_id\tmusic_id\tplay_count\tsplit\nu1\tm1\t3\ttrain\n", encoding="utf-8")
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text(
        "music_id\ttitle\tartist\tgenre\tduration_ms\tlicense\n"
        "m1\tBlue Hour\tExample Artist\tambient\t123000\tCC BY\n",
        encoding="utf-8",
    )
    interaction_manifest = _manifest(interactions, adapter="music4all")
    metadata_manifest = _manifest(metadata, adapter="music4all", required_columns=("item_id",))

    interaction_frame = next(iter_music4all_interactions(interactions, manifest=interaction_manifest))
    item_frame = next(iter_music4all_metadata(metadata, manifest=metadata_manifest))

    assert tuple(item_frame.columns) == CANONICAL_ITEM_COLUMNS
    assert interaction_frame.loc[0, "interaction_value"] == 3
    assert interaction_frame.loc[0, "split"] == "train"
    assert item_frame.loc[0, "item_name"] == "Blue Hour"
    assert item_frame.loc[0, "artist_name"] == "Example Artist"
    assert item_frame.loc[0, "duration_ms"] == 123000
    assert item_frame.loc[0, "content_license"] == "CC BY"


def test_fma_adapter_flattens_native_multirow_header_and_converts_duration(tmp_path: Path) -> None:
    source = tmp_path / "tracks.csv"
    source.write_text(
        ",album,artist,track,track,track\n"
        ",title,name,title,duration,license\n"
        "track_id,,,,,\n"
        "10,Night Rooms,Ada,Still Water,61.5,CC BY-NC\n",
        encoding="utf-8",
    )
    manifest = _manifest(source, adapter="fma_metadata", required_columns=("track_id",))

    frame = next(iter_fma_metadata(source, manifest=manifest))

    assert frame.loc[0, "item_id"] == "10"
    assert frame.loc[0, "item_name"] == "Still Water"
    assert frame.loc[0, "artist_name"] == "Ada"
    assert frame.loc[0, "album_name"] == "Night Rooms"
    assert frame.loc[0, "duration_ms"] == 61_500
    assert frame.loc[0, "content_license"] == "CC BY-NC"


def test_kuairand_adapter_preserves_feedback_and_randomization_context(tmp_path: Path) -> None:
    source = tmp_path / "kuairand.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "video_id": [99],
            "timestamp": [1_700_000_000_000],
            "play_time_ms": [12_000],
            "is_click": [1],
            "is_rand": [1],
            "is_like": [0],
        }
    ).to_csv(source, index=False)
    manifest = _manifest(source, adapter="kuairand_policy_log")

    frame = next(iter_kuairand_policy_logs(source, manifest=manifest))

    assert frame.loc[0, "item_id"] == "99"
    assert frame.loc[0, "dwell_ms"] == 12_000
    assert frame.loc[0, "reward"] == 1
    assert frame.loc[0, "policy_id"] == "1"
    assert json.loads(frame.loc[0, "context_json"]) == {"is_like": 0}


def test_open_bandit_adapter_preserves_propensity_contract(tmp_path: Path) -> None:
    source = tmp_path / "open_bandit.csv"
    pd.DataFrame(
        {
            "item_id": ["item-1"],
            "position": [2],
            "click": [1],
            "propensity_score": [0.25],
            "policy": ["bts"],
            "campaign": ["all"],
            "user_feature_0": ["F"],
        }
    ).to_csv(source, index=False)
    manifest = _manifest(
        source,
        adapter="open_bandit_policy_log",
        required_columns=("item_id", "position", "click", "propensity_score"),
    )

    frame = next(iter_open_bandit_policy_logs(source, manifest=manifest))

    assert frame.loc[0, "user_id"] is pd.NA
    assert frame.loc[0, "event_type"] == "impression"
    assert frame.loc[0, "reward"] == 1
    assert frame.loc[0, "propensity"] == 0.25
    assert frame.loc[0, "position"] == 2
    assert json.loads(frame.loc[0, "context_json"]) == {"user_feature_0": "F"}


def test_manifest_round_trip_and_checksum_validation(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")
    manifest = _manifest(source, adapter="music4all")
    manifest_path = write_source_manifest(tmp_path / "manifest.json", manifest)

    loaded = load_source_manifest(manifest_path)
    validate_source_manifest(loaded, source_paths=(source,))

    assert loaded == manifest


def test_builtin_manifest_templates_document_every_supported_source_family() -> None:
    assert set(DATASET_MANIFEST_TEMPLATES) == {
        "million_song_taste_profile",
        "lfm",
        "music4all",
        "fma",
        "kuairand",
        "open_bandit",
    }
    for template in DATASET_MANIFEST_TEMPLATES.values():
        assert template["task_fit"]
        assert template["required_columns"]
        assert template["license_name"]
        assert template["license_url"]
        assert template["access_url"]
        assert template["access_caveats"]


def test_manifest_rejects_unapproved_training_use(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")
    manifest = replace(_manifest(source, adapter="music4all"), training_use_approved=False)

    with pytest.raises(SourceManifestError, match="not approved"):
        validate_source_manifest(manifest, source_paths=(source,))


def test_manifest_detects_file_changes(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")
    manifest = _manifest(source, adapter="music4all")
    source.write_text("user_id,item_id\nu1,changed\n", encoding="utf-8")

    with pytest.raises(SourceManifestError, match="Size mismatch|SHA-256 mismatch"):
        validate_source_manifest(manifest, source_paths=(source,))


def test_network_paths_are_never_downloaded() -> None:
    with pytest.raises(SourceManifestError, match="Network URLs are not accepted"):
        SourceFileProvenance.from_local_file(
            "https://example.org/data.csv",
            source_url="https://example.org/data.csv",
        )


def test_spotify_columns_and_values_are_blocked() -> None:
    with pytest.raises(SpotifyTrainingContentError, match="columns are blocked"):
        validate_no_spotify_training_content(pd.DataFrame({"spotify_track_uri": ["redacted"]}))

    with pytest.raises(SpotifyTrainingContentError, match="URI/API URL"):
        validate_no_spotify_training_content(pd.DataFrame({"item_id": ["spotify:track:abc"]}))


def test_spotify_manifest_is_blocked_even_when_columns_are_renamed(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")
    manifest = replace(
        _manifest(source, adapter="music4all"),
        access_url="https://api.spotify.com/v1/tracks",
    )

    with pytest.raises(SpotifyTrainingContentError, match="not permitted"):
        validate_source_manifest(manifest, source_paths=(source,))


def test_adapter_requires_matching_manifest_adapter(tmp_path: Path) -> None:
    source = tmp_path / "triplets.tsv"
    source.write_text("u1\ts1\t1\n", encoding="utf-8")
    manifest = _manifest(source, adapter="music4all")

    with pytest.raises(SourceManifestError, match="does not match"):
        next(iter_million_song_taste_profile(source, manifest=manifest))


def test_open_bandit_rejects_invalid_propensity(tmp_path: Path) -> None:
    source = tmp_path / "open_bandit.csv"
    pd.DataFrame(
        {"item_id": ["i1"], "position": [1], "click": [0], "propensity_score": [0.0]}
    ).to_csv(source, index=False)
    manifest = _manifest(source, adapter="open_bandit_policy_log")

    with pytest.raises(CanonicalSchemaError, match="interval"):
        next(iter_open_bandit_policy_logs(source, manifest=manifest))
