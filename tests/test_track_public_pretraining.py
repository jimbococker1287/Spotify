from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.public_training_data import (
    CANONICAL_INTERACTION_COLUMNS,
    DatasetSourceManifest,
    SourceFileProvenance,
    write_source_manifest,
)
from spotify.track_public_pretraining import (
    MASK_ITEM_ID,
    CheckpointTransferContract,
    PublicPretrainingCheckpoint,
    PublicPretrainingConfig,
    build_public_pretraining_batches,
    map_public_records_to_examples,
    run_public_pretraining,
)


def _manifest(source: Path, *, approved: bool = True) -> DatasetSourceManifest:
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
                acquired_at="2026-06-14T12:00:00+00:00",
            ),
        ),
        training_use_approved=approved,
        reviewed_by="test-reviewer" if approved else "",
        reviewed_at="2026-06-14T12:01:00+00:00" if approved else "",
    )


def _canonical_records() -> pd.DataFrame:
    rows = [
        ("u1", "s1", "i1", "2026-01-01T00:00:00Z", "train"),
        ("u1", "s1", "i2", "2026-01-01T00:01:00Z", "train"),
        ("u1", "s1", "i3", "2026-01-01T00:02:00Z", "train"),
        ("u2", "s2", "validation-only", "2026-01-02T00:00:00Z", "validation"),
        ("u2", "s2", "validation-only-2", "2026-01-02T00:01:00Z", "validation"),
        ("u3", "s3", "test-only", "2026-01-03T00:00:00Z", "test"),
        ("u3", "s3", "test-only-2", "2026-01-03T00:01:00Z", "test"),
        ("u4", "s4", "i2", "2026-01-04T00:00:00Z", "pretrain"),
        ("u4", "s4", "i4", "2026-01-04T00:01:00Z", "pretrain"),
        ("u4", "s4", "i5", "2026-01-04T00:02:00Z", "pretrain"),
    ]
    frame = pd.DataFrame(pd.NA, index=range(len(rows)), columns=CANONICAL_INTERACTION_COLUMNS)
    frame["source_dataset"] = "public-listens"
    frame["user_id"] = [row[0] for row in rows]
    frame["session_id"] = [row[1] for row in rows]
    frame["item_id"] = [row[2] for row in rows]
    frame["timestamp"] = pd.to_datetime([row[3] for row in rows], utc=True)
    frame["event_type"] = "listen"
    frame["explicit_positive"] = True
    frame["split"] = [row[4] for row in rows]
    return frame


def test_unapproved_source_blocks_before_source_loading_or_training(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("this file must never be opened by the loader\n", encoding="utf-8")
    approved_manifest = _manifest(source)
    missing_source_record = replace(
        approved_manifest.files[0],
        path=str(tmp_path / "source-that-does-not-exist.csv"),
    )
    unapproved = replace(
        approved_manifest,
        files=(missing_source_record,),
        training_use_approved=False,
        reviewed_by="",
        reviewed_at="",
    )
    manifest_path = write_source_manifest(tmp_path / "manifest.json", unapproved)
    calls = {"loader": 0, "trainer": 0}

    def loader(_: DatasetSourceManifest) -> pd.DataFrame:
        calls["loader"] += 1
        raise AssertionError("unapproved source loader must not run")

    def trainer(_request: object) -> PublicPretrainingCheckpoint:
        calls["trainer"] += 1
        raise AssertionError("unapproved source trainer must not run")

    result = run_public_pretraining(manifest_path, record_loader=loader, trainer=trainer)

    assert result.status == "blocked"
    assert result.stage == "governance"
    assert "not approved" in result.reason
    assert calls == {"loader": 0, "trainer": 0}


def test_digest_mismatch_blocks_before_loader(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("original", encoding="utf-8")
    manifest = _manifest(source)
    source.write_text("modified", encoding="utf-8")
    loader_called = False

    def loader(_: DatasetSourceManifest) -> pd.DataFrame:
        nonlocal loader_called
        loader_called = True
        return _canonical_records()

    result = run_public_pretraining(manifest, record_loader=loader)

    assert result.status == "blocked"
    assert result.stage == "governance"
    assert "SHA-256 mismatch" in result.reason
    assert loader_called is False


def test_public_mapping_excludes_validation_and_test_splits(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest = _manifest(source)
    config = PublicPretrainingConfig(max_sequence_length=4, seed=7)

    examples = map_public_records_to_examples(
        _canonical_records(),
        manifest=manifest,
        config=config,
    )

    assert examples.sequences.shape == (2, 4)
    assert examples.excluded_split_rows == 4
    assert "validation-only" not in examples.item_to_index
    assert "test-only" not in examples.item_to_index
    assert examples.summary()["private_validation_or_test_included"] is False
    expected_items = {
        int(examples.item_to_index[item])
        for item in ("i1", "i2", "i3", "i4", "i5")
    }
    assert set(examples.sequences.ravel()) - {0} == expected_items


def test_bounded_pretraining_batches_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest = _manifest(source)
    config = PublicPretrainingConfig(
        max_sequence_length=4,
        max_masked_sequences=1,
        max_masked_predictions=1,
        max_contrastive_pairs=1,
        mask_probability=1.0,
        seed=19,
    )
    examples = map_public_records_to_examples(
        _canonical_records(),
        manifest=manifest,
        config=config,
    )

    first = build_public_pretraining_batches(examples, config=config)
    second = build_public_pretraining_batches(examples, config=config)

    np.testing.assert_array_equal(first.masked_items.input_sequences, second.masked_items.input_sequences)
    np.testing.assert_array_equal(first.masked_items.target_items, second.masked_items.target_items)
    np.testing.assert_array_equal(
        first.contrastive_pairs.left.sequences,
        second.contrastive_pairs.left.sequences,
    )
    np.testing.assert_array_equal(
        first.contrastive_pairs.right.sequences,
        second.contrastive_pairs.right.sequences,
    )
    assert len(first.masked_items.target_items) == 1
    assert len(first.contrastive_pairs.left_rows) == 1
    assert np.count_nonzero(first.masked_items.input_sequences == MASK_ITEM_ID) == 1


def test_injected_trainer_emits_json_friendly_transfer_contract(tmp_path: Path) -> None:
    source = tmp_path / "public.csv"
    source.write_text("synthetic", encoding="utf-8")
    manifest = _manifest(source)
    checkpoint_path = tmp_path / "public_encoder.weights"
    trainer_calls = 0

    def trainer(request) -> PublicPretrainingCheckpoint:
        nonlocal trainer_calls
        trainer_calls += 1
        assert request.dataset_id == manifest.dataset_id
        assert request.examples.excluded_split_rows == 4
        checkpoint_path.write_bytes(b"synthetic encoder checkpoint")
        return PublicPretrainingCheckpoint(
            checkpoint_path=checkpoint_path,
            encoder_family="session-transformer",
            embedding_dim=32,
            transferable_components=("item_embedding", "sequence_encoder"),
            target_model_families=("MEANTIME", "SASRec"),
            metrics={"masked_item_loss": np.float32(0.25)},
            metadata={"epochs": np.int64(1)},
        )

    result = run_public_pretraining(
        manifest,
        record_loader=lambda _: _canonical_records(),
        config=PublicPretrainingConfig(max_sequence_length=4, seed=5),
        trainer=trainer,
    )
    payload = result.to_dict()

    assert result.status == "ready"
    assert result.stage == "complete"
    assert trainer_calls == 1
    assert isinstance(result.transfer_contract, CheckpointTransferContract)
    assert result.transfer_contract.private_data_included is False
    assert result.transfer_contract.checkpoint_sha256
    assert result.transfer_contract.vocabulary_digest == payload["data"]["vocabulary_digest"]
    assert payload["training"]["metrics"]["masked_item_loss"] == 0.25
    json.dumps(payload)
