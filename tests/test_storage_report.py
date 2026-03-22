from __future__ import annotations

import json
from pathlib import Path

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
