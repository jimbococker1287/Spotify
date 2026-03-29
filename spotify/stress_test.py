from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import time

import numpy as np

from .benchmarks import sample_indices
from .causal_friction import CausalSkipDecompositionArtifact
from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact, simulate_rollout
from .multimodal import MultimodalArtistSpace
from .safe_policy import SafeBanditPolicyArtifact


DEFAULT_MAX_STRESS_TEST_SESSIONS = 2500
DEFAULT_STRESS_TEST_PROGRESS_EVERY = 500


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


def _resolve_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except Exception:
        return default
    return max(minimum, value)


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
    sampling_rng = np.random.default_rng(random_seed)
    rollout_rng = np.random.default_rng(random_seed + 1)

    rows: list[dict[str, object]] = []
    baseline_policy = {"transition": 1.1, "continuity": 0.1, "novelty": 0.0, "repeat": 0.8}
    comparison_policies = {
        "baseline_exploit": baseline_policy,
        "safe_global": dict(safe_policy.global_policy),
    }
    seq_test = np.asarray(data.X_seq_test, dtype="int32")
    ctx_test = np.asarray(data.X_ctx_test, dtype="float32")
    total_sessions = int(len(seq_test))
    max_sessions = _resolve_env_int(
        "SPOTIFY_STRESS_TEST_MAX_SESSIONS",
        DEFAULT_MAX_STRESS_TEST_SESSIONS,
        minimum=0,
    )
    progress_every = _resolve_env_int(
        "SPOTIFY_STRESS_TEST_PROGRESS_EVERY",
        DEFAULT_STRESS_TEST_PROGRESS_EVERY,
        minimum=0,
    )
    sample_idx = sample_indices(total_sessions, max_sessions, sampling_rng)
    sampled_seq = seq_test[sample_idx]
    sampled_ctx = ctx_test[sample_idx]
    sampled_sessions = int(len(sampled_seq))
    sample_fraction = round(sampled_sessions / max(1, total_sessions), 4)

    logger.info(
        "Stress-test lab evaluating %d/%d held-out sessions across %d scenarios and %d policies.",
        sampled_sessions,
        total_sessions,
        len(SCENARIOS),
        len(comparison_policies),
    )

    for scenario_name, scenario in SCENARIOS.items():
        for policy_name, weights in comparison_policies.items():
            started_at = time.perf_counter()
            session_lengths: list[int] = []
            skip_risks: list[float] = []
            end_risks: list[float] = []
            logger.info(
                "Stress-test scenario=%s policy=%s sessions=%d",
                scenario_name,
                policy_name,
                sampled_sessions,
            )
            for row_idx, (seq, ctx) in enumerate(zip(sampled_seq, sampled_ctx, strict=False), start=1):
                rollout = simulate_rollout(
                    twin=digital_twin,
                    multimodal_space=multimodal_space,
                    causal_artifact=causal_artifact,
                    start_sequence=seq,
                    start_context=ctx,
                    horizon=6,
                    policy_weights=weights,
                    scenario=scenario,
                    rng=rollout_rng,
                )
                session_lengths.append(int(rollout["session_length"]))
                skip_risks.append(float(rollout["mean_skip_risk"]))
                end_risks.append(float(rollout["mean_end_risk"]))
                if progress_every > 0 and (row_idx % progress_every == 0 or row_idx == sampled_sessions):
                    logger.info(
                        "Stress-test progress scenario=%s policy=%s processed=%d/%d elapsed=%.1fs",
                        scenario_name,
                        policy_name,
                        row_idx,
                        sampled_sessions,
                        time.perf_counter() - started_at,
                    )
            elapsed_seconds = round(time.perf_counter() - started_at, 2)
            mean_session_length = float(np.mean(session_lengths)) if session_lengths else float("nan")
            mean_skip_risk = float(np.mean(skip_risks)) if skip_risks else float("nan")
            mean_end_risk = float(np.mean(end_risks)) if end_risks else float("nan")
            rows.append(
                {
                    "scenario": scenario_name,
                    "policy_name": policy_name,
                    "mean_session_length": mean_session_length,
                    "mean_skip_risk": mean_skip_risk,
                    "mean_end_risk": mean_end_risk,
                    "evaluated_sessions": sampled_sessions,
                    "total_test_sessions": total_sessions,
                    "sample_fraction": sample_fraction,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            logger.info(
                "Stress-test summary scenario=%s policy=%s skip_risk=%.4f end_risk=%.4f elapsed=%.1fs",
                scenario_name,
                policy_name,
                mean_skip_risk,
                mean_end_risk,
                elapsed_seconds,
            )

    csv_path = _write_csv(
        output_dir / "stress_test_summary.csv",
        rows,
        [
            "scenario",
            "policy_name",
            "mean_session_length",
            "mean_skip_risk",
            "mean_end_risk",
            "evaluated_sessions",
            "total_test_sessions",
            "sample_fraction",
            "elapsed_seconds",
        ],
    )
    summary_path = output_dir / "stress_test_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info(
        "Ran stress-test lab across %d scenarios using %d/%d sessions.",
        len(SCENARIOS),
        sampled_sessions,
        total_sessions,
    )
    return [csv_path, summary_path]
