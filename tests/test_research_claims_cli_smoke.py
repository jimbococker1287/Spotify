from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_research_claims_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.research_claims", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "research-claim pack" in result.stdout.lower()
    assert "--benchmark-manifest" in result.stdout
    assert "--run-dir" in result.stdout


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_research_platform_lab_cli_smoke_runs_on_minimal_fixture(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_cli"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_cli",
            "profile": "small",
            "timestamp": "2026-05-03T12:00:00",
            "champion_gate": {"status": "pass", "promoted": False},
        },
    )
    _write_json(run_dir / "run_results.json", [{"model_name": "gru_artist", "test_top1": 0.14}])
    _write_json(run_dir / "benchmark_protocol.json", {"benchmark_contract": {"contract_version": "v5", "comparison_mode": "repeated_seed_lock"}})
    _write_json(run_dir / "safety_platform_contract.json", {"reuse_summary": {"api_group_count": 1, "wrapper_count": 1}})
    _write_json(analysis_dir / "data_drift_summary.json", {"target_drift": {"train_vs_test_jsd": 0.11}})
    _write_json(analysis_dir / "robustness_summary.json", [{"worst_segment": "repeat_from_prev"}])
    _write_json(analysis_dir / "moonshot_summary.json", {"stress_worst_skip_risk": 0.35})
    _write_json(
        output_dir / "analysis" / "research_claims" / "research_claims.json",
        {
            "run": {"run_id": "run_cli"},
            "primary_claim": {"key": "shift_robustness"},
            "claims": [
                {
                    "key": "shift_robustness",
                    "title": "Shift robustness",
                    "status": "analysis_ready",
                    "summary": "Core evidence is present.",
                    "metrics": {"target_drift_jsd": 0.11},
                    "supporting_artifacts": [str((run_dir / "run_results.json").resolve())],
                    "missing_checks": [],
                }
            ],
            "claim_support_matrix": [
                {
                    "claim_key": "shift_robustness",
                    "live_signal_status": "ready",
                    "benchmark_evidence_status": "n/a",
                    "repeated_evidence_status": "ready",
                    "slice_evidence_status": "ready",
                    "risk_evidence_status": "ready",
                    "artifact_pack_status": "ready",
                    "next_gate": "ready_to_package",
                }
            ],
            "submission_readiness": {"status": "analysis_ready", "ready_for_external_review": True, "blockers": []},
        },
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "spotify.research_platform_lab", "--output-dir", str(output_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "research_platform_maturity.md" in result.stdout
    assert "run_research_registry.csv" in result.stdout
