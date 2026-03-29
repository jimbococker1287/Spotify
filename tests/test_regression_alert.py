from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "regression_alert.py"


def _write_run(
    output_dir: Path,
    *,
    run_id: str,
    timestamp: str,
    promoted: bool,
    status: str,
    model_name: str,
    model_type: str,
    val_top1: float,
    test_top1: float,
    regression: float,
    robustness_gap: float,
    stress_skip_risk: float,
) -> Path:
    run_dir = output_dir / "runs" / run_id
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    champion_alias = {"model_name": model_name, "model_type": model_type} if promoted else {"model_name": "", "model_type": ""}
    gate_payload = {
        "status": status,
        "promoted": promoted,
        "metric_source": "backtest_top1",
        "regression": regression,
        "threshold": 0.005,
        "challenger_model_name": model_name,
    }
    (run_dir / "champion_gate.json").write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_id,
                "profile": "full",
                "timestamp": timestamp,
                "data_records": 1234,
                "num_artists": 80,
                "num_context_features": 12,
                "champion_gate": gate_payload,
                "champion_alias": champion_alias,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": model_name,
                    "model_type": model_type,
                    "val_top1": val_top1,
                    "test_top1": test_top1,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / f"{model_type}_{model_name}_confidence_summary.json").write_text(
        json.dumps(
            {
                "test_ece": 0.05,
                "test_selective_risk": 0.20,
                "test_abstention_rate": 0.08,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "data_drift_summary.json").write_text(
        json.dumps(
            {
                "target_drift": {"train_vs_test_jsd": 0.10},
                "largest_context_shift": {"feature": "offline", "max_abs_std_mean_diff": 0.6},
                "largest_segment_shift": {
                    "split": "test",
                    "segment": "repeat_from_prev",
                    "bucket": "new",
                    "abs_share_shift": 0.12,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "friction_proxy_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "friction_feature_count": 2,
                "proxy_counterfactual": {"test_mean_delta": 0.01},
                "top_friction_features": [{"feature": "offline", "mean_risk_delta": 0.01}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "robustness_summary.json").write_text(
        json.dumps(
            [
                {
                    "model_name": model_name,
                    "max_top1_gap": robustness_gap,
                    "worst_segment": "repeat_from_prev",
                    "worst_bucket": "new",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "moonshot_summary.json").write_text(
        json.dumps(
            {
                "digital_twin_test_auc": 0.71,
                "causal_test_auc_total": 0.69,
                "stress_worst_skip_scenario": "evening_drift",
                "stress_worst_skip_risk": stress_skip_risk,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_regression_alert_returns_three_on_high_review_action(tmp_path: Path) -> None:
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
        promoted=True,
        status="pass",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=-0.01,
        robustness_gap=0.27,
        stress_skip_risk=0.24,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--outputs-dir", str(output_dir), "--run-dir", str(run_b)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 3
    assert "review_status=ok" in result.stdout
    assert "review_priority=high" in result.stdout
    assert "handoff=" in result.stdout
    assert "[HIGH] Harden the worst slice before the next full run" in result.stdout


def test_regression_alert_review_threshold_off_returns_zero(tmp_path: Path) -> None:
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
        promoted=True,
        status="pass",
        model_name="blended_ensemble",
        model_type="ensemble",
        val_top1=0.61,
        test_top1=0.58,
        regression=-0.01,
        robustness_gap=0.27,
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
            "--review-threshold",
            "off",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert "review_priority=high" in result.stdout


def test_regression_alert_defaults_to_control_room_selected_run(tmp_path: Path) -> None:
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
    _write_run(
        output_dir,
        run_id="run_smoke_check",
        timestamp="2026-03-22T20:00:00",
        promoted=False,
        status="fail",
        model_name="dense",
        model_type="deep",
        val_top1=0.52,
        test_top1=0.48,
        regression=0.04,
        robustness_gap=0.08,
        stress_skip_risk=0.20,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--outputs-dir",
            str(output_dir),
            "--review-threshold",
            "off",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert "run=run_full" in result.stdout
    assert "promoted=True" in result.stdout
    assert "next_step=" in result.stdout
