from __future__ import annotations

import json
from pathlib import Path

from spotify.benchmark_contract import (
    build_benchmark_lock_manifest,
    describe_canonical_benchmark_contract,
    write_benchmark_lock_manifest,
)


def test_describe_canonical_benchmark_contract_exposes_week10_defaults() -> None:
    contract = describe_canonical_benchmark_contract()

    assert contract["contract_version"] == "2026-week10-v1"
    assert contract["canonical_profile"] == "small"
    assert contract["minimum_repeated_runs"] == 3
    assert "benchmark_protocol.json" in contract["required_run_artifacts"]
    assert contract["significance_policy"]["z_threshold"] == 1.96


def test_write_benchmark_lock_manifest_marks_ready_contract(tmp_path: Path) -> None:
    benchmark_id = "20260329"
    output_dir = tmp_path / "history"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"benchmark-lock-{benchmark_id}"
    summary_rows = [
        {"model_name": "retrieval_reranker", "run_count": 3},
        {"model_name": "gru_artist", "run_count": 3},
    ]
    significance_rows = [
        {"left_model": "retrieval_reranker", "right_model": "gru_artist", "significant_at_95": 1},
    ]
    raw_rows = [
        {"run_id": "run_a", "profile": "small"},
        {"run_id": "run_b", "profile": "small"},
        {"run_id": "run_c", "profile": "small"},
    ]

    rows_path = output_dir / f"benchmark_lock_{benchmark_id}_rows.csv"
    summary_csv = output_dir / f"benchmark_lock_{benchmark_id}_summary.csv"
    rows_path.write_text("run_id\nrun_a\n", encoding="utf-8")
    summary_csv.write_text("model_name\nretrieval_reranker\n", encoding="utf-8")
    (output_dir / f"benchmark_lock_{benchmark_id}_summary.json").write_text("[]", encoding="utf-8")
    (output_dir / f"benchmark_lock_{benchmark_id}_ci95.png").write_bytes(b"png")
    (output_dir / f"benchmark_lock_{benchmark_id}_significance.csv").write_text("left_model,right_model\n", encoding="utf-8")

    json_path, md_path = write_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        run_name_prefix=prefix,
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        raw_rows=raw_rows,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["comparison_ready"] is True
    assert payload["present_artifact_count"] == payload["required_artifact_count"]
    assert md_path.exists()

    payload = build_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        run_name_prefix=prefix,
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        raw_rows=raw_rows,
    )

    assert payload["comparison_ready"] is True
    assert payload["present_artifact_count"] >= 7


def test_build_benchmark_lock_manifest_requires_repeated_runs_per_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "history"
    benchmark_id = "demo"

    for name in (
        f"benchmark_lock_{benchmark_id}_rows.csv",
        f"benchmark_lock_{benchmark_id}_summary.csv",
        f"benchmark_lock_{benchmark_id}_summary.json",
        f"benchmark_lock_{benchmark_id}_ci95.png",
        f"benchmark_lock_{benchmark_id}_significance.csv",
        f"benchmark_lock_{benchmark_id}_manifest.json",
        f"benchmark_lock_{benchmark_id}_manifest.md",
    ):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            path.write_bytes(b"png")
        else:
            path.write_text("ok\n", encoding="utf-8")

    payload = build_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        run_name_prefix="benchmark-lock-demo",
        summary_rows=[
            {"model_name": "logreg", "run_count": 3},
            {"model_name": "dense", "run_count": 1},
        ],
        significance_rows=[{"left_model": "logreg", "right_model": "dense", "significant_at_95": 0}],
        raw_rows=[
            {"run_id": "run_a", "profile": "small"},
            {"run_id": "run_b", "profile": "small"},
            {"run_id": "run_c", "profile": "small"},
        ],
    )

    assert payload["comparison_ready"] is False
    per_model_check = next(row for row in payload["stability_checks"] if row["key"] == "minimum_repeated_runs_per_model")
    assert per_model_check["status"] == "fail"
