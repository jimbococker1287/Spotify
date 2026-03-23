from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from .causal_friction import CausalSkipDecompositionArtifact
from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact, simulate_rollout
from .multimodal import MultimodalArtistSpace
from .safe_policy import SafeBanditPolicyArtifact


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": {},
    "high_friction_spike": {"friction_scale": 2.5},
    "session_restart": {"repeat_bias": 0.4},
    "evening_drift": {"hour_shift": 8.0},
    "listener_fatigue": {"fatigue_bias": 0.25},
}


def run_stress_test_lab(
    *,
    data: PreparedData,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    safe_policy: SafeBanditPolicyArtifact,
    causal_artifact: CausalSkipDecompositionArtifact | None,
    output_dir: Path,
    logger,
    random_seed: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    rows: list[dict[str, object]] = []
    baseline_policy = {"transition": 1.1, "continuity": 0.1, "novelty": 0.0, "repeat": 0.8}
    comparison_policies = {
        "baseline_exploit": baseline_policy,
        "safe_global": dict(safe_policy.global_policy),
    }

    for scenario_name, scenario in SCENARIOS.items():
        for policy_name, weights in comparison_policies.items():
            session_lengths: list[int] = []
            skip_risks: list[float] = []
            end_risks: list[float] = []
            for seq, ctx in zip(data.X_seq_test, data.X_ctx_test, strict=False):
                rollout = simulate_rollout(
                    twin=digital_twin,
                    multimodal_space=multimodal_space,
                    causal_artifact=causal_artifact,
                    start_sequence=seq,
                    start_context=ctx,
                    horizon=6,
                    policy_weights=weights,
                    scenario=scenario,
                    rng=rng,
                )
                session_lengths.append(int(rollout["session_length"]))
                skip_risks.append(float(rollout["mean_skip_risk"]))
                end_risks.append(float(rollout["mean_end_risk"]))
            rows.append(
                {
                    "scenario": scenario_name,
                    "policy_name": policy_name,
                    "mean_session_length": float(np.mean(session_lengths)) if session_lengths else float("nan"),
                    "mean_skip_risk": float(np.mean(skip_risks)) if skip_risks else float("nan"),
                    "mean_end_risk": float(np.mean(end_risks)) if end_risks else float("nan"),
                }
            )

    csv_path = _write_csv(
        output_dir / "stress_test_summary.csv",
        rows,
        ["scenario", "policy_name", "mean_session_length", "mean_skip_risk", "mean_end_risk"],
    )
    summary_path = output_dir / "stress_test_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Ran stress-test lab across %d scenarios", len(SCENARIOS))
    return [csv_path, summary_path]
