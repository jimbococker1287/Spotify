from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from tests.test_regression_alert import _write_run


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "control_room_guard.py"


def test_control_room_guard_returns_four_on_threshold_violation(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_b = _write_run(
        output_dir,
        run_id="run_b",
        timestamp="2026-03-22T20:00:00",
        promoted=True,
        status="pass",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=-0.01,
        robustness_gap=0.27,
        stress_skip_risk=0.44,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--run-dir",
            str(run_b),
            "--max-robustness-gap",
            "0.20",
            "--max-stress-skip-risk",
            "0.40",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 4
    assert "violations=2" in result.stdout
    assert "violation[1]=robustness_gap" in result.stdout
    assert "violation[2]=stress_skip_risk" in result.stdout
    triage_json = output_dir / "analytics" / "control_room_triage.json"
    triage_md = output_dir / "analytics" / "control_room_triage.md"
    assert triage_json.exists()
    assert triage_md.exists()
    triage_payload = json.loads(triage_json.read_text(encoding="utf-8"))
    assert len(triage_payload["violations"]) == 2
    assert any(item["area"] == "robustness" for item in triage_payload["triage_items"])
    markdown = triage_md.read_text(encoding="utf-8")
    assert "Inspect: Open analysis/robustness_guardrails.json and analysis/robustness_summary.json" in markdown
    assert "Fix: Add slice-aware safeguards" in markdown
    assert "Rerun: Re-run the fast schedule" in markdown


def test_control_room_guard_returns_five_on_stale_run_request(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    _write_run(
        output_dir,
        run_id="run_a",
        timestamp="2026-03-20T20:00:00",
        promoted=True,
        status="pass",
        model_name="retrieval_reranker",
        model_type="retrieval_reranker",
        val_top1=0.59,
        test_top1=0.56,
        regression=-0.01,
        robustness_gap=0.09,
        stress_skip_risk=0.22,
    )
    _write_run(
        output_dir,
        run_id="run_b",
        timestamp="2026-03-22T20:00:00",
        promoted=True,
        status="pass",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=-0.01,
        robustness_gap=0.09,
        stress_skip_risk=0.24,
    )

    stale_run_dir = output_dir / "runs" / "run_a"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--run-dir",
            str(stale_run_dir),
            "--max-robustness-gap",
            "0.20",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 5
    assert "control_room_status=stale:run_b" in result.stdout


def test_control_room_guard_triage_includes_promotion_playbook(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    _write_run(
        output_dir,
        run_id="run_a",
        timestamp="2026-03-20T20:00:00",
        promoted=True,
        status="pass",
        model_name="retrieval_reranker",
        model_type="retrieval_reranker",
        val_top1=0.59,
        test_top1=0.56,
        regression=-0.01,
        robustness_gap=0.09,
        stress_skip_risk=0.22,
    )
    run_b = _write_run(
        output_dir,
        run_id="run_b",
        timestamp="2026-03-22T20:00:00",
        promoted=False,
        status="fail",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=0.02,
        robustness_gap=0.09,
        stress_skip_risk=0.24,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--run-dir",
            str(run_b),
            "--allow-fail",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    triage_payload = json.loads((output_dir / "analytics" / "control_room_triage.json").read_text(encoding="utf-8"))
    assert any(item["area"] == "promotion" for item in triage_payload["triage_items"])
    promotion_item = next(item for item in triage_payload["triage_items"] if item["area"] == "promotion")
    assert "run_manifest.json" in promotion_item["inspect_files"]
    assert any("Compare the challenger" in step for step in promotion_item["inspect_steps"])


def test_control_room_guard_supports_slice_and_standing_benchmark_thresholds(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_b = _write_run(
        output_dir,
        run_id="run_b",
        timestamp="2026-03-22T20:00:00",
        promoted=True,
        status="pass",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=-0.01,
        robustness_gap=0.27,
        stress_skip_risk=0.44,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--run-dir",
            str(run_b),
            "--max-repeat-from-prev-new-gap",
            "0.20",
            "--max-stress-benchmark-skip-risk",
            "0.40",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 4
    assert "repeat_from_prev_new_gap=0.270" in result.stdout
    assert "stress_benchmark_skip_risk=0.440" in result.stdout
    assert "violation[1]=repeat_from_prev_new_gap" in result.stdout
    assert "violation[2]=stress_benchmark_skip_risk" in result.stdout


def test_control_room_guard_triage_includes_instrumentation_playbook_when_analysis_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    run_dir.mkdir(parents=True)

    gate_payload = {
        "status": "fail",
        "promoted": False,
        "metric_source": "backtest_top1",
        "regression": 0.02,
        "threshold": 0.005,
        "challenger_model_name": "mlp",
    }
    (run_dir / "champion_gate.json").write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "run_name": "run_a",
                "profile": "full",
                "timestamp": "2026-03-22T20:00:00",
                "data_records": 1200,
                "num_artists": 80,
                "num_context_features": 12,
                "champion_gate": gate_payload,
                "champion_alias": {"model_name": "", "model_type": ""},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "mlp",
                    "model_type": "classical",
                    "val_top1": 0.31,
                    "test_top1": 0.28,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--run-dir",
            str(run_dir),
            "--allow-fail",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    triage_payload = json.loads((output_dir / "analytics" / "control_room_triage.json").read_text(encoding="utf-8"))
    instrumentation_item = next(item for item in triage_payload["triage_items"] if item["area"] == "instrumentation")
    assert any("analysis generation was skipped" in step for step in instrumentation_item["inspect_steps"])
    assert any("coverage section shows the expected artifacts" in step for step in instrumentation_item["rerun_steps"])


def test_control_room_guard_allows_newer_explicit_run_pending_control_room(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    _write_run(
        output_dir,
        run_id="run_full",
        timestamp="2026-03-20T20:00:00",
        promoted=True,
        status="pass",
        model_name="retrieval_reranker",
        model_type="retrieval_reranker",
        val_top1=0.59,
        test_top1=0.56,
        regression=-0.01,
        robustness_gap=0.09,
        stress_skip_risk=0.22,
    )
    run_pending = _write_run(
        output_dir,
        run_id="run_pending",
        timestamp="2026-03-22T20:00:00",
        promoted=True,
        status="pass",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=-0.01,
        robustness_gap=0.12,
        stress_skip_risk=0.24,
    )
    (run_pending / "run_results.json").unlink()
    (run_pending / "analysis" / "friction_proxy_summary.json").unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--run-dir",
            str(run_pending),
            "--max-robustness-gap",
            "0.20",
            "--max-stress-skip-risk",
            "0.40",
            "--max-target-drift-jsd",
            "0.20",
            "--max-selective-risk",
            "0.50",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert "run=run_pending" in result.stdout
    assert "control_room_status=pending_control_room:run_full" in result.stdout
    assert "violations=0" in result.stdout
