from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from spotify.compare_run_timings import build_timing_comparison, write_timing_comparison


def _write_timing_run(
    root: Path,
    *,
    run_id: str,
    profile: str,
    total_seconds: float,
    measured_seconds: float,
    overhead_seconds: float,
    phase_rows: list[dict[str, object]],
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "run_phase_timings.json"
    manifest_path = run_dir / "run_manifest.json"

    timing_payload = {
        "run_id": run_id,
        "final_status": "FINISHED",
        "total_seconds": total_seconds,
        "measured_seconds": measured_seconds,
        "unmeasured_overhead_seconds": overhead_seconds,
        "phase_count": len(phase_rows),
        "completed_phase_count": len([row for row in phase_rows if row.get("status") == "ok"]),
        "non_skipped_phase_count": len([row for row in phase_rows if row.get("status") != "skipped"]),
        "slowest_phase": max(phase_rows, key=lambda row: float(row.get("duration_seconds", 0.0)), default={}),
        "slowest_phases": sorted(phase_rows, key=lambda row: float(row.get("duration_seconds", 0.0)), reverse=True)[:5],
        "phases": phase_rows,
    }
    timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_id,
                "profile": profile,
                "timestamp": "2026-03-29T16:00:00",
                "phase_timings": {
                    "json_path": str(timing_path),
                    "total_seconds": total_seconds,
                    "measured_seconds": measured_seconds,
                    "unmeasured_overhead_seconds": overhead_seconds,
                    "slowest_phase": timing_payload["slowest_phase"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_build_timing_comparison_reports_total_and_phase_deltas(tmp_path: Path) -> None:
    baseline_dir = _write_timing_run(
        tmp_path,
        run_id="baseline_run",
        profile="full",
        total_seconds=100.0,
        measured_seconds=95.0,
        overhead_seconds=5.0,
        phase_rows=[
            {"phase_name": "data_loading", "status": "ok", "duration_seconds": 10.0},
            {"phase_name": "moonshot_lab", "status": "ok", "duration_seconds": 40.0},
        ],
    )
    candidate_dir = _write_timing_run(
        tmp_path,
        run_id="candidate_run",
        profile="full",
        total_seconds=85.0,
        measured_seconds=82.0,
        overhead_seconds=3.0,
        phase_rows=[
            {"phase_name": "data_loading", "status": "ok", "duration_seconds": 8.0},
            {"phase_name": "moonshot_lab", "status": "ok", "duration_seconds": 30.0},
        ],
    )

    comparison = build_timing_comparison(
        baseline_bundle={
            "run_id": "baseline_run",
            "run_name": "baseline_run",
            "profile": "full",
            "timestamp": "2026-03-29T16:00:00",
            "timing_path": baseline_dir / "run_phase_timings.json",
            "run_dir": baseline_dir,
            "summary": json.loads((baseline_dir / "run_phase_timings.json").read_text(encoding="utf-8")),
        },
        candidate_bundle={
            "run_id": "candidate_run",
            "run_name": "candidate_run",
            "profile": "full",
            "timestamp": "2026-03-29T16:05:00",
            "timing_path": candidate_dir / "run_phase_timings.json",
            "run_dir": candidate_dir,
            "summary": json.loads((candidate_dir / "run_phase_timings.json").read_text(encoding="utf-8")),
        },
        top_n=5,
    )

    assert comparison["overall"]["delta_seconds"] == -15.0
    assert comparison["overall"]["speedup_ratio"] == 100.0 / 85.0
    assert comparison["top_phase_deltas"][0]["phase_name"] == "moonshot_lab"
    assert comparison["faster_phases"][0]["phase_name"] == "moonshot_lab"

    json_path, csv_path, md_path = write_timing_comparison(tmp_path / "artifacts", comparison)

    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    assert "Run Timing Comparison" in md_path.read_text(encoding="utf-8")


def test_compare_run_timings_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.compare_run_timings", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.compare_run_timings" in result.stdout
    assert "--baseline" in result.stdout
    assert "--candidate" in result.stdout
