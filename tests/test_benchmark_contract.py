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
    prefix = f"benchmark-lock-{benchmark_id}"
    summary_rows = [
        {"model_name": "retrieval_reranker"},
        {"model_name": "gru_artist"},
    ]
    significance_rows = [
        {"left_model": "retrieval_reranker", "right_model": "gru_artist", "significant_at_95": 1},
    ]
    raw_rows = [
        {"run_id": "run_a", "profile": "small"},
        {"run_id": "run_b", "profile": "small"},
        {"run_id": "run_c", "profile": "small"},
    ]

    json_path, md_path = write_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        run_name_prefix=prefix,
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        raw_rows=raw_rows,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["comparison_ready"] is False
    assert payload["present_artifact_count"] == 0
    assert md_path.exists()

    rows_path = output_dir / f"benchmark_lock_{benchmark_id}_rows.csv"
    summary_csv = output_dir / f"benchmark_lock_{benchmark_id}_summary.csv"
    rows_path.write_text("run_id\nrun_a\n", encoding="utf-8")
    summary_csv.write_text("model_name\nretrieval_reranker\n", encoding="utf-8")
    (output_dir / f"benchmark_lock_{benchmark_id}_summary.json").write_text("[]", encoding="utf-8")
    (output_dir / f"benchmark_lock_{benchmark_id}_ci95.png").write_bytes(b"png")

    payload = build_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        run_name_prefix=prefix,
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        raw_rows=raw_rows,
    )

    assert payload["comparison_ready"] is True
    assert payload["present_artifact_count"] >= 6
