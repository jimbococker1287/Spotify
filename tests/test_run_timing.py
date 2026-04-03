from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from spotify.run_timing import RunPhaseRecorder


def test_run_phase_recorder_writes_json_and_csv_artifacts(tmp_path: Path) -> None:
    recorder = RunPhaseRecorder(run_id="demo_run")

    with recorder.phase("data_loading", include_video=False) as phase:
        phase["raw_rows"] = 42
        phase["raw_columns"] = 9

    recorder.skip("moonshot_lab", reason="disabled_for_profile", profile="fast")

    json_path, csv_path, payload = recorder.write_artifacts(run_dir=tmp_path, final_status="FINISHED")

    assert json_path == tmp_path / "run_phase_timings.json"
    assert csv_path == tmp_path / "run_phase_timings.csv"
    assert payload["run_id"] == "demo_run"
    assert payload["final_status"] == "FINISHED"
    assert payload["phase_count"] == 2
    assert payload["completed_phase_count"] == 1
    assert payload["non_skipped_phase_count"] == 1
    assert payload["slowest_phase"]["phase_name"] == "data_loading"
    assert payload["measured_seconds"] >= 0.0
    assert payload["unmeasured_overhead_seconds"] >= 0.0

    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_payload["phases"][0]["metadata"]["raw_rows"] == 42
    assert saved_payload["phases"][1]["status"] == "skipped"
    assert saved_payload["phases"][1]["metadata"]["reason"] == "disabled_for_profile"

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["phase_name"] == "data_loading"
    assert '"raw_rows": 42' in rows[0]["metadata_json"]
    assert rows[1]["status"] == "skipped"


def test_run_phase_recorder_marks_failed_phases() -> None:
    recorder = RunPhaseRecorder(run_id="failed_run")

    with pytest.raises(RuntimeError, match="boom"):
        with recorder.phase("optuna_tuning", trials=2):
            raise RuntimeError("boom")

    payload = recorder.summary(final_status="FAILED")

    assert payload["phase_count"] == 1
    assert payload["slowest_phase"]["phase_name"] == "optuna_tuning"
    assert payload["phases"][0]["status"] == "failed"
    assert payload["phases"][0]["metadata"]["error"] == "RuntimeError: boom"
