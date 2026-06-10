from __future__ import annotations

import json
from pathlib import Path

from spotify.run_tradeoffs import build_run_tradeoff_dossier


def _manifest(run_dir: Path, *, run_id: str, profile: str = "full", promoted: bool = True) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "profile": profile,
        "timestamp": "2026-06-01T12:00:00",
        "data_records": 1000,
        "num_artists": 100,
        "num_context_features": 20,
        "champion_gate": {"promoted": promoted, "status": "pass" if promoted else "fail"},
        "artifact_cleanup": {"enabled": True, "mode": "light", "threshold_mb": 100.0},
    }


def _write_timings(
    run_dir: Path,
    *,
    run_id: str,
    total_seconds: float,
    measured_seconds: float,
    phases: list[dict[str, object]],
) -> None:
    (run_dir / "run_phase_timings.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "final_status": "FINISHED",
                "total_seconds": total_seconds,
                "measured_seconds": measured_seconds,
                "unmeasured_overhead_seconds": total_seconds - measured_seconds,
                "phases": phases,
            }
        ),
        encoding="utf-8",
    )


def _quality_rows(*, status: str = "better") -> list[dict[str, object]]:
    return [
        {
            "key": "best_model_test_top1",
            "label": "Best model test top1",
            "current": 0.61,
            "baseline": 0.60,
            "delta": 0.01,
            "status": status,
            "direction": "higher",
        },
        {
            "key": "robustness_gap",
            "label": "Worst robustness gap",
            "current": 0.10,
            "baseline": 0.10,
            "delta": 0.0,
            "status": "flat",
            "direction": "lower",
        },
    ]


def test_run_tradeoff_dossier_reports_runtime_storage_and_phase_regressions(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "runs" / "baseline"
    selected_dir = tmp_path / "runs" / "selected"
    baseline = _manifest(baseline_dir, run_id="baseline")
    selected = _manifest(selected_dir, run_id="selected")
    _write_timings(
        baseline_dir,
        run_id="baseline",
        total_seconds=100.0,
        measured_seconds=90.0,
        phases=[
            {"phase_name": "data_loading", "status": "ok", "duration_seconds": 20.0, "metadata": {}},
            {"phase_name": "training", "status": "ok", "duration_seconds": 60.0, "metadata": {}},
            {"phase_name": "reporting", "status": "ok", "duration_seconds": 10.0, "metadata": {}},
        ],
    )
    _write_timings(
        selected_dir,
        run_id="selected",
        total_seconds=125.0,
        measured_seconds=115.0,
        phases=[
            {"phase_name": "data_loading", "status": "ok", "duration_seconds": 25.0, "metadata": {}},
            {"phase_name": "training", "status": "ok", "duration_seconds": 80.0, "metadata": {}},
            {"phase_name": "reporting", "status": "ok", "duration_seconds": 10.0, "metadata": {}},
        ],
    )
    (baseline_dir / "retained.bin").write_bytes(b"x" * 1024)
    (selected_dir / "retained.bin").write_bytes(b"x" * 4096)

    dossier = build_run_tradeoff_dossier(
        selected_manifest=selected,
        baseline_manifest=baseline,
        quality_safety_deltas=_quality_rows(),
    )

    assert dossier["status"] == "complete"
    assert dossier["comparability"]["comparable"] is True
    assert dossier["runtime"]["delta_seconds"] == 25.0
    assert dossier["runtime"]["delta_percent"] == 25.0
    assert dossier["storage"]["delta_bytes"] > 0
    assert dossier["largest_phase_regressions"][0]["phase_name"] == "training"
    assert dossier["largest_phase_regressions"][0]["delta_seconds"] == 20.0
    assert dossier["verdict"] == "tradeoff_review"


def test_run_tradeoff_dossier_handles_legacy_and_incomparable_runs(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "runs" / "baseline"
    selected_dir = tmp_path / "runs" / "selected"
    baseline = _manifest(baseline_dir, run_id="baseline", profile="full")
    selected = _manifest(selected_dir, run_id="selected", profile="core")
    _write_timings(
        selected_dir,
        run_id="selected",
        total_seconds=20.0,
        measured_seconds=18.0,
        phases=[{"phase_name": "training", "status": "ok", "duration_seconds": 18.0, "metadata": {}}],
    )

    dossier = build_run_tradeoff_dossier(
        selected_manifest=selected,
        baseline_manifest=baseline,
        quality_safety_deltas=_quality_rows(status="flat"),
    )

    blocker_codes = {row["code"] for row in dossier["comparability"]["blockers"]}
    assert dossier["status"] == "partial"
    assert dossier["comparability"]["comparable"] is False
    assert "profile_mismatch" in blocker_codes
    assert "baseline_timing_missing" in blocker_codes
    assert "phase_details_missing" not in blocker_codes
    assert dossier["runtime"]["available"] is False
    assert dossier["verdict"] == "not_comparable"


def test_run_tradeoff_dossier_bounds_directory_scans(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "runs" / "baseline"
    selected_dir = tmp_path / "runs" / "selected"
    baseline = _manifest(baseline_dir, run_id="baseline")
    selected = _manifest(selected_dir, run_id="selected")
    phases = [{"phase_name": "training", "status": "ok", "duration_seconds": 10.0, "metadata": {}}]
    _write_timings(baseline_dir, run_id="baseline", total_seconds=10.0, measured_seconds=10.0, phases=phases)
    _write_timings(selected_dir, run_id="selected", total_seconds=10.0, measured_seconds=10.0, phases=phases)
    for index in range(5):
        (baseline_dir / f"baseline_{index}.bin").write_bytes(b"x")
        (selected_dir / f"selected_{index}.bin").write_bytes(b"x")

    dossier = build_run_tradeoff_dossier(
        selected_manifest=selected,
        baseline_manifest=baseline,
        quality_safety_deltas=_quality_rows(status="flat"),
        max_scan_entries=2,
    )

    blocker_codes = {row["code"] for row in dossier["comparability"]["blockers"]}
    assert "selected_storage_scan_incomplete" in blocker_codes
    assert "baseline_storage_scan_incomplete" in blocker_codes
    assert dossier["storage"]["available"] is False
    assert dossier["storage"]["selected"]["entries_scanned"] == 2
    assert dossier["storage"]["baseline"]["entries_scanned"] == 2
