from __future__ import annotations

import json
from pathlib import Path

from spotify.artifact_cleanup import (
    load_result_rows_for_cleanup,
    prune_existing_runs,
    prune_old_auxiliary_artifacts,
    prune_run_artifacts,
    retained_full_run_dirs,
    select_model_for_run_cleanup,
)
from spotify.champion_alias import write_champion_alias


class _StubLogger:
    def info(self, *args, **kwargs) -> None:
        return None


def test_prune_run_artifacts_deletes_large_non_selected_estimators(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    estimator_dir = run_dir / "estimators"
    estimator_dir.mkdir(parents=True)

    kept_path = estimator_dir / "classical_mlp.joblib"
    kept_path.write_bytes(b"keep-me")
    deleted_path = estimator_dir / "classical_extra_trees.joblib"
    deleted_path.write_bytes(b"delete-me")

    classical_results_path = run_dir / "classical_results.json"
    classical_results_path.write_text(
        json.dumps(
            [
                {"model_name": "mlp", "estimator_artifact_path": str(kept_path)},
                {"model_name": "extra_trees", "estimator_artifact_path": str(deleted_path)},
            ]
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "model_name": "mlp",
            "model_type": "classical",
            "val_top1": 0.31,
            "estimator_artifact_path": str(kept_path),
        },
        {
            "model_name": "extra_trees",
            "model_type": "classical",
            "val_top1": 0.29,
            "estimator_artifact_path": str(deleted_path),
        },
    ]

    summary = prune_run_artifacts(
        run_dir=run_dir,
        result_rows=rows,
        selected_model=("mlp", "classical"),
        logger=_StubLogger(),
        cleanup_mode="light",
        min_size_mb=0.0,
    )

    assert kept_path.exists()
    assert not deleted_path.exists()
    assert rows[0]["estimator_artifact_path"] == str(kept_path.resolve())
    assert rows[1]["estimator_artifact_path"] == ""
    assert summary["freed_bytes"] > 0
    classical_rows = json.loads(classical_results_path.read_text(encoding="utf-8"))
    assert classical_rows[1]["estimator_artifact_path"] == ""


def test_prune_run_artifacts_keeps_ensemble_dependencies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    estimator_dir = run_dir / "estimators"
    estimator_dir.mkdir(parents=True)

    extra_trees_path = estimator_dir / "classical_extra_trees.joblib"
    extra_trees_path.write_bytes(b"keep-extra-trees")
    random_forest_path = estimator_dir / "classical_random_forest.joblib"
    random_forest_path.write_bytes(b"delete-random-forest")

    rows = [
        {
            "model_name": "extra_trees",
            "model_type": "classical",
            "val_top1": 0.34,
            "estimator_artifact_path": str(extra_trees_path),
        },
        {
            "model_name": "random_forest",
            "model_type": "classical",
            "val_top1": 0.33,
            "estimator_artifact_path": str(random_forest_path),
        },
        {
            "model_name": "blended_ensemble",
            "model_type": "ensemble",
            "val_top1": 0.36,
            "ensemble_members": ["extra_trees"],
            "ensemble_weights": {"extra_trees": 1.0},
        },
    ]

    summary = prune_run_artifacts(
        run_dir=run_dir,
        result_rows=rows,
        selected_model=("blended_ensemble", "ensemble"),
        logger=_StubLogger(),
        cleanup_mode="light",
        min_size_mb=0.0,
    )

    assert extra_trees_path.exists()
    assert not random_forest_path.exists()
    assert "extra_trees" in summary["kept_models"]
    assert rows[0]["estimator_artifact_path"] == str(extra_trees_path.resolve())
    assert rows[1]["estimator_artifact_path"] == ""


def test_load_result_rows_for_cleanup_supports_legacy_classical_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "classical_results.json").write_text(
        json.dumps([{"model_name": "logreg", "val_top1": 0.2, "estimator_artifact_path": "x.joblib"}]),
        encoding="utf-8",
    )

    rows = load_result_rows_for_cleanup(run_dir)

    assert len(rows) == 1
    assert rows[0]["model_type"] == "classical"


def test_select_model_for_run_cleanup_prefers_champion_alias_for_matching_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    write_champion_alias(
        output_dir=output_dir,
        run_id="run_a",
        run_dir=run_dir,
        model_name="gru_artist",
        model_type="deep",
    )

    selected = select_model_for_run_cleanup(
        run_dir=run_dir,
        output_dir=output_dir,
        result_rows=[
            {"model_name": "extra_trees", "model_type": "classical", "val_top1": 0.3, "estimator_artifact_path": "x"}
        ],
    )

    assert selected == ("gru_artist", "deep")


def test_prune_existing_runs_updates_multiple_run_dirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    estimator_dir = run_dir / "estimators"
    estimator_dir.mkdir(parents=True)
    kept = estimator_dir / "classical_logreg.joblib"
    kept.write_bytes(b"keep")
    deleted = estimator_dir / "classical_extra_trees.joblib"
    deleted.write_bytes(b"delete")
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {"model_name": "logreg", "model_type": "classical", "val_top1": 0.4, "estimator_artifact_path": str(kept)},
                {"model_name": "extra_trees", "model_type": "classical", "val_top1": 0.3, "estimator_artifact_path": str(deleted)},
            ]
        ),
        encoding="utf-8",
    )

    summary = prune_existing_runs(
        output_dir=output_dir,
        run_dirs=None,
        logger=_StubLogger(),
        cleanup_mode="light",
        min_size_mb=0.0,
    )

    assert summary["run_count"] == 1
    assert summary["deleted_file_count"] == 1
    assert kept.exists()
    assert not deleted.exists()


def test_prune_run_artifacts_deletes_orphaned_optuna_estimators(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    optuna_dir = run_dir / "optuna" / "estimators"
    estimator_dir = run_dir / "estimators"
    optuna_dir.mkdir(parents=True)
    estimator_dir.mkdir(parents=True)

    kept = estimator_dir / "classical_logreg.joblib"
    kept.write_bytes(b"keep")
    orphan = optuna_dir / "classical_tuned_extra_trees.joblib"
    orphan.write_bytes(b"remove")

    rows = [
        {
            "model_name": "logreg",
            "model_type": "classical",
            "val_top1": 0.4,
            "estimator_artifact_path": str(kept),
        }
    ]

    summary = prune_run_artifacts(
        run_dir=run_dir,
        result_rows=rows,
        selected_model=("logreg", "classical"),
        logger=_StubLogger(),
        cleanup_mode="light",
        min_size_mb=0.0,
    )

    assert kept.exists()
    assert not orphan.exists()
    assert summary["deleted_files"]


def test_retained_full_run_dirs_keeps_recent_full_runs_and_current(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    current = output_dir / "runs" / "20260315_120000_fast-run"
    current.mkdir(parents=True)
    old_full = output_dir / "runs" / "20260314_120000_everything-20260314-120000"
    old_full.mkdir(parents=True)
    newer_full = output_dir / "runs" / "20260315_110000_full-run"
    newer_full.mkdir(parents=True)
    (newer_full / "run_manifest.json").write_text(
        json.dumps({"profile": "full", "timestamp": "2026-03-15T11:00:00"}),
        encoding="utf-8",
    )

    retained = retained_full_run_dirs(
        output_dir=output_dir,
        keep_last_n=1,
        current_run_dir=current,
    )

    assert current.resolve() in retained
    assert newer_full.resolve() in retained
    assert old_full.resolve() not in retained


def test_prune_old_auxiliary_artifacts_keeps_recent_full_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    current_run = output_dir / "runs" / "20260315_130000_fast-current"
    current_run.mkdir(parents=True)
    (current_run / "prediction_bundles").mkdir()
    (current_run / "prediction_bundles" / "current.npz").write_bytes(b"current")

    recent_full = output_dir / "runs" / "20260315_120000_everything-20260315-120000"
    recent_full.mkdir(parents=True)
    (recent_full / "prediction_bundles").mkdir()
    (recent_full / "prediction_bundles" / "recent.npz").write_bytes(b"recent")
    (recent_full / "spotify_training.db").write_bytes(b"db")

    old_full = output_dir / "runs" / "20260314_120000_everything-20260314-120000"
    old_full.mkdir(parents=True)
    (old_full / "prediction_bundles").mkdir()
    (old_full / "prediction_bundles" / "old.npz").write_bytes(b"old")
    (old_full / "spotify_training.db").write_bytes(b"db")
    (old_full / "run_results.json").write_text(
        json.dumps([{"model_name": "x", "prediction_bundle_path": str(old_full / "prediction_bundles" / "old.npz")}]),
        encoding="utf-8",
    )

    summary = prune_old_auxiliary_artifacts(
        output_dir=output_dir,
        current_run_dir=current_run,
        logger=_StubLogger(),
        keep_last_full_runs=1,
        prune_prediction_bundles=True,
        prune_run_databases=True,
    )

    assert (current_run / "prediction_bundles" / "current.npz").exists()
    assert (recent_full / "prediction_bundles" / "recent.npz").exists()
    assert (recent_full / "spotify_training.db").exists()
    assert not (old_full / "prediction_bundles").exists()
    assert not (old_full / "spotify_training.db").exists()
    updated = json.loads((old_full / "run_results.json").read_text(encoding="utf-8"))
    assert updated[0]["prediction_bundle_path"] == ""
    assert summary["deleted_file_count"] >= 2
