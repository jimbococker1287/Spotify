from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from spotify.champion_gate_refresh import refresh_champion_gate
from spotify.probability_bundles import save_prediction_bundle
from spotify.run_artifacts import safe_read_json, write_json


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_completed_run(outputs_dir: Path) -> tuple[Path, str]:
    run_id = "20260402_181212_everything-20260402-181212"
    run_dir = outputs_dir / "runs" / run_id
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "prediction_bundles").mkdir(parents=True, exist_ok=True)
    cache_path = outputs_dir / "cache" / "prepared_data" / "fixture" / "prepared_bundle.joblib"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "prepared": SimpleNamespace(
                y_val=np.array([2, 1], dtype="int32"),
                y_test=np.array([2, 1], dtype="int32"),
            )
        },
        cache_path,
    )
    bundle_path = save_prediction_bundle(
        run_dir / "prediction_bundles" / "classical_extra_trees.npz",
        val_proba=np.array(
            [
                [0.02, 0.03, 0.95],
                [0.05, 0.92, 0.03],
            ],
            dtype="float32",
        ),
        test_proba=np.array(
            [
                [0.94, 0.04, 0.02],
                [0.53, 0.37, 0.10],
            ],
            dtype="float32",
        ),
    )
    run_results = [
        {
            "model_name": "extra_trees",
            "model_type": "classical",
            "val_top1": 0.42,
            "test_top1": 0.31,
            "prediction_bundle_path": str(bundle_path),
        }
    ]
    gate_payload = {
        "status": "fail",
        "promoted": False,
        "metric_source": "backtest_top1",
        "threshold": 0.02,
        "regression": 0.03,
        "profile_match": True,
        "require_significant_lift": False,
        "significance_z": 1.96,
        "max_selective_risk": 0.0,
        "max_abstention_rate": 0.50,
        "challenger_model_name": "extra_trees",
        "challenger_selective_risk": None,
        "challenger_abstention_rate": None,
        "conformal_alpha": 0.10,
    }
    manifest = {
        "run_id": run_id,
        "run_name": "everything-20260402-181212",
        "profile": "full",
        "timestamp": "2026-04-02T18:12:12+00:00",
        "champion_gate": gate_payload,
        "cache": {"cache_path": str(cache_path)},
    }
    write_json(run_dir / "run_results.json", run_results)
    write_json(run_dir / "champion_gate.json", gate_payload)
    write_json(run_dir / "run_manifest.json", manifest)

    _write_csv(
        outputs_dir / "history" / "experiment_history.csv",
        ["run_id", "model_name", "val_top1", "profile"],
        [
            {"run_id": "20260401_120000_everything", "model_name": "mlp", "val_top1": 0.50, "profile": "full"},
        ],
    )
    _write_csv(
        outputs_dir / "history" / "backtest_history.csv",
        ["run_id", "profile", "model_name", "top1"],
        [
            {"run_id": "20260401_120000_everything", "profile": "full", "model_name": "mlp", "top1": 0.31},
            {"run_id": "20260401_120000_everything", "profile": "full", "model_name": "mlp", "top1": 0.29},
            {"run_id": run_id, "profile": "full", "model_name": "extra_trees", "top1": 0.29},
            {"run_id": run_id, "profile": "full", "model_name": "extra_trees", "top1": 0.30},
        ],
    )
    return run_dir, run_id


def test_refresh_champion_gate_backfills_missing_challenger_risk_metrics(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir, run_id = _build_completed_run(outputs_dir)

    payload = refresh_champion_gate(
        outputs_dir=outputs_dir,
        run_dir=run_dir,
        refresh_control_room=False,
    )

    gate_payload = safe_read_json(run_dir / "champion_gate.json", default={})
    manifest_payload = safe_read_json(run_dir / "run_manifest.json", default={})

    assert payload["run_id"] == run_id
    assert payload["metric_source"] == "backtest_top1"
    assert payload["status"] == "pass"
    assert payload["promoted"] is True
    assert payload["challenger_model_name"] == "extra_trees"
    assert payload["challenger_selective_risk"] == 0.0
    assert payload["challenger_abstention_rate"] == 0.0
    assert payload["risk_metric_model_count"] == 1
    assert payload["control_room_refreshed"] is False
    assert gate_payload["challenger_selective_risk"] == 0.0
    assert gate_payload["challenger_abstention_rate"] == 0.0
    assert manifest_payload["champion_gate"]["challenger_selective_risk"] == 0.0
    assert manifest_payload["champion_gate"]["challenger_abstention_rate"] == 0.0
    assert (run_dir / "analysis" / "classical_extra_trees_confidence_summary.json").exists()
    assert (run_dir / "analysis" / "classical_extra_trees_conformal_summary.json").exists()


def test_champion_gate_refresh_module_updates_existing_run(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir, _ = _build_completed_run(outputs_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spotify.champion_gate_refresh",
            "--outputs-dir",
            str(outputs_dir),
            "--run-dir",
            str(run_dir),
            "--skip-control-room",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    gate_payload = safe_read_json(run_dir / "champion_gate.json", default={})
    assert result.returncode == 0
    assert "challenger=extra_trees" in result.stdout
    assert "challenger_selective_risk=0.0" in result.stdout
    assert gate_payload["challenger_selective_risk"] == 0.0
