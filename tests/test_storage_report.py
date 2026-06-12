from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from spotify.storage_report import build_storage_report, write_storage_report


def test_build_storage_report_groups_sizes_by_category(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    estimator_dir = run_dir / "estimators"
    prediction_dir = run_dir / "prediction_bundles"
    analytics_dir = output_dir / "analytics"
    estimator_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    analytics_dir.mkdir(parents=True)

    (estimator_dir / "classical_mlp.joblib").write_bytes(b"a" * 10)
    (prediction_dir / "deep_gru.npz").write_bytes(b"b" * 5)
    (analytics_dir / "spotify_analytics.duckdb").write_bytes(b"c" * 7)

    report = build_storage_report(output_dir, top_n=5)

    assert report["total_bytes"] == 22
    categories = {row["category"]: row["bytes"] for row in report["category_totals"]}
    assert categories["classical_estimators"] == 10
    assert categories["prediction_bundles"] == 5
    assert categories["analytics"] == 7
    assert report["runs"][0]["run_id"] == "run_a"


def test_write_storage_report_creates_json_and_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("hello", encoding="utf-8")

    json_path, md_path = write_storage_report(output_dir, top_n=3)

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["output_dir"] == str(output_dir.resolve())
    assert "Storage Report" in md_path.read_text(encoding="utf-8")


def test_build_storage_report_includes_external_mlflow_artifact_roots(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)

    mlflow_db_dir = output_dir / "mlruns"
    mlflow_db_dir.mkdir(parents=True)
    external_root = tmp_path / "mlruns" / "1"
    external_root.mkdir(parents=True)
    artifact_path = external_root / "run_manifest.json"
    artifact_path.write_text("{}", encoding="utf-8")

    db_path = mlflow_db_dir / "mlflow.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE experiments (experiment_id INTEGER, artifact_location TEXT)")
        conn.execute(
            "INSERT INTO experiments (experiment_id, artifact_location) VALUES (?, ?)",
            (1, str(external_root)),
        )
        conn.commit()

    report = build_storage_report(output_dir, top_n=5)

    categories = {row["category"]: row["bytes"] for row in report["category_totals"]}
    scanned_roots = {row["path"] for row in report["scanned_roots"]}
    assert categories["mlflow_artifacts"] == 2
    assert str(external_root.resolve()) in scanned_roots
    assert report["total_bytes"] >= 2


def test_build_storage_report_groups_strict_top_level_transient_candidates(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    tmp_dir = output_dir / "tmp-debug"
    mplconfig_dir = output_dir / ".mplconfig"
    tmp_dir.mkdir(parents=True)
    mplconfig_dir.mkdir(parents=True)
    (tmp_dir / "trace.log").write_bytes(b"a" * 3)
    (tmp_dir / "snapshot.json").write_bytes(b"b" * 4)
    (output_dir / ".DS_Store").write_bytes(b"c" * 5)
    (mplconfig_dir / "fontlist.json").write_bytes(b"d" * 6)

    report = build_storage_report(output_dir)

    cleanup = report["cleanup_opportunities"]
    safe_to_review = cleanup["safe_to_review"]
    groups = {row["group"]: row for row in safe_to_review["groups"]}
    assert report["potential_reclaim_bytes"] == 18
    assert safe_to_review["candidate_count"] == 3
    assert safe_to_review["file_count"] == 4
    assert groups["tmp_prefix"]["bytes"] == 7
    assert groups["tmp_prefix"]["file_count"] == 2
    assert groups["ds_store"]["bytes"] == 5
    assert groups["mplconfig_cache"]["bytes"] == 6
    assert all(row["review_status"] == "safe_to_review" for row in safe_to_review["candidates"])
    assert cleanup["guidance"]["deletes_files"] is False


def test_build_storage_report_does_not_flag_nested_run_artifacts_as_transient(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    nested_tmp = output_dir / "runs" / "run_a" / "analysis" / "tmp_predictions.json"
    nested_tmp.parent.mkdir(parents=True)
    nested_tmp.write_bytes(b"a" * 9)
    normal_history = output_dir / "history" / "tmp_named_benchmark.json"
    normal_history.parent.mkdir(parents=True)
    normal_history.write_bytes(b"b" * 4)

    report = build_storage_report(output_dir)

    cleanup = report["cleanup_opportunities"]
    assert cleanup["safe_to_review"]["candidate_count"] == 0
    assert cleanup["potential_reclaim_bytes"] == 0
    assert cleanup["policy_managed"]["runs"]["bytes"] == 9
    assert cleanup["policy_managed"]["runs"]["included_in_potential_reclaim"] is False


def test_build_storage_report_exposes_high_storage_pressure(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True)
    large_path = models_dir / "large_model.bin"
    with large_path.open("wb") as handle:
        handle.truncate(21 * 1024**3)

    report = build_storage_report(output_dir)

    pressure = report["storage_pressure"]
    assert pressure["status"] == "high"
    assert pressure["total_bytes"] == 21 * 1024**3
    assert pressure["potential_reclaim_bytes"] == 0
    assert report["cleanup_opportunities"]["policy_managed"]["mlflow"]["bytes"] == 0


def test_write_storage_report_includes_non_executing_cleanup_guidance(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    transient_path = output_dir / "tmp-local-debug.log"
    transient_path.write_text("debug", encoding="utf-8")

    json_path, md_path = write_storage_report(output_dir)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert transient_path.exists()
    assert payload["cleanup_opportunities"]["safe_to_review"]["candidate_count"] == 1
    assert "## Storage Pressure" in markdown
    assert "## Cleanup Opportunities" in markdown
    assert "### Safe To Review" in markdown
    assert "### Policy-Managed Storage" in markdown
    assert "No files were deleted." in markdown
    assert "`scripts/prune_artifacts.py`" in markdown


def test_write_storage_report_reuses_prebuilt_report(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    report = build_storage_report(output_dir)

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("write_storage_report rebuilt an already available report")

    monkeypatch.setattr("spotify.storage_report.build_storage_report", fail_rebuild)
    json_path, _ = write_storage_report(output_dir, report=report)

    assert json.loads(json_path.read_text(encoding="utf-8"))["generated_at"] == report["generated_at"]
