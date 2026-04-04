from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np

from .benchmarks import sample_indices
from .causal_friction import CausalSkipDecompositionArtifact
from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact, simulate_rollout, simulate_rollout_batch_summary
from .multimodal import MultimodalArtistSpace
from .run_artifacts import write_csv_rows
from .safe_policy import SafeBanditPolicyArtifact


DEFAULT_MAX_STRESS_TEST_SESSIONS = 2500
DEFAULT_STRESS_TEST_PROGRESS_EVERY = 500
DEFAULT_STRESS_TEST_BATCH_SIZE = 256
DEFAULT_STRESS_BENCHMARK_SCENARIO = "evening_drift"
DEFAULT_STRESS_BENCHMARK_POLICY = "safe_routed"
DEFAULT_STRESS_BENCHMARK_REFERENCE_POLICY = "baseline_exploit"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


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


def _resolve_env_name(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _safe_float(value) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(metric):
        return float("nan")
    return metric


_DEFAULT_SIMULATE_ROLLOUT = simulate_rollout


def _normalize_policy(weights: dict[str, float]) -> dict[str, float]:
    return {
        key: float(max(0.0, value))
        for key, value in weights.items()
    }


def _scenario_safe_policy(
    *,
    safe_policy: SafeBanditPolicyArtifact,
    scenario_name: str,
) -> tuple[str, dict[str, float]]:
    policy_map = getattr(safe_policy, "policy_map", {}) or {}
    global_policy = dict(getattr(safe_policy, "global_policy", {}) or {})
    if scenario_name == "high_friction_spike":
        base = dict(policy_map.get("high_friction", global_policy))
        base["continuity"] = base.get("continuity", 0.0) + 0.15
        base["novelty"] = max(0.0, base.get("novelty", 0.0) - 0.10)
        return "safe_routed_high_friction", _normalize_policy(base)
    if scenario_name == "evening_drift":
        base = dict(policy_map.get("normal_friction", global_policy))
        base["continuity"] = base.get("continuity", 0.0) + 0.35
        base["repeat"] = base.get("repeat", 0.0) + 0.15
        base["transition"] = base.get("transition", 0.0) + 0.05
        base["novelty"] = max(0.0, base.get("novelty", 0.0) - 0.30)
        return "safe_routed_evening", _normalize_policy(base)
    if scenario_name == "listener_fatigue":
        base = dict(policy_map.get("high_friction", global_policy))
        base["continuity"] = base.get("continuity", 0.0) + 0.20
        base["repeat"] = base.get("repeat", 0.0) + 0.20
        base["novelty"] = max(0.0, base.get("novelty", 0.0) - 0.25)
        return "safe_routed_fatigue", _normalize_policy(base)
    if scenario_name == "session_restart":
        base = dict(policy_map.get("normal_friction", global_policy))
        base["repeat"] = base.get("repeat", 0.0) + 0.15
        base["continuity"] = base.get("continuity", 0.0) + 0.10
        return "safe_routed_restart", _normalize_policy(base)
    return "safe_routed_default", global_policy


def _build_stress_benchmark(rows: list[dict[str, object]]) -> dict[str, object]:
    benchmark_scenario = _resolve_env_name(
        "SPOTIFY_STRESS_BENCHMARK_SCENARIO",
        DEFAULT_STRESS_BENCHMARK_SCENARIO,
    )
    benchmark_policy = _resolve_env_name(
        "SPOTIFY_STRESS_BENCHMARK_POLICY",
        DEFAULT_STRESS_BENCHMARK_POLICY,
    )
    reference_policy = _resolve_env_name(
        "SPOTIFY_STRESS_BENCHMARK_REFERENCE_POLICY",
        DEFAULT_STRESS_BENCHMARK_REFERENCE_POLICY,
    )
    policy_selection_mode = "exact_match"
    benchmark_row: dict[str, object] = {}
    if benchmark_policy in {"safe_routed", "scenario_safe", "auto"}:
        routed_rows = [
            row
            for row in rows
            if str(row.get("scenario", "")).strip() == benchmark_scenario
            and str(row.get("policy_name", "")).strip().startswith("safe_routed")
        ]
        if routed_rows:
            benchmark_row = min(routed_rows, key=lambda row: _safe_float(row.get("mean_skip_risk")))
            benchmark_policy = str(benchmark_row.get("policy_name", "")).strip() or benchmark_policy
            policy_selection_mode = "scenario_routed_alias"
    if not benchmark_row:
        benchmark_row = next(
            (
                row
                for row in rows
                if str(row.get("scenario", "")).strip() == benchmark_scenario
                and str(row.get("policy_name", "")).strip() == benchmark_policy
            ),
            {},
        )
    reference_row = next(
        (
            row
            for row in rows
            if str(row.get("scenario", "")).strip() == benchmark_scenario
            and str(row.get("policy_name", "")).strip() == reference_policy
        ),
        {},
    )
    policy_rows = [
        row
        for row in rows
        if str(row.get("policy_name", "")).strip() == benchmark_policy
    ]
    ranked_policy_rows = sorted(policy_rows, key=lambda row: _safe_float(row.get("mean_skip_risk")))
    scenario_rank = next(
        (
            idx
            for idx, row in enumerate(ranked_policy_rows, start=1)
            if str(row.get("scenario", "")).strip() == benchmark_scenario
        ),
        0,
    )
    benchmark_skip_risk = _safe_float(benchmark_row.get("mean_skip_risk"))
    benchmark_end_risk = _safe_float(benchmark_row.get("mean_end_risk"))
    reference_skip_risk = _safe_float(reference_row.get("mean_skip_risk"))
    reference_end_risk = _safe_float(reference_row.get("mean_end_risk"))
    return {
        "benchmark_scenario": benchmark_scenario,
        "benchmark_policy_name": benchmark_policy,
        "benchmark_policy_selection_mode": policy_selection_mode,
        "reference_policy_name": reference_policy,
        "available": bool(benchmark_row),
        "reference_available": bool(reference_row),
        "skip_risk": benchmark_skip_risk,
        "end_risk": benchmark_end_risk,
        "reference_skip_risk": reference_skip_risk,
        "reference_end_risk": reference_end_risk,
        "skip_risk_delta_vs_reference": (
            float(benchmark_skip_risk - reference_skip_risk)
            if np.isfinite(benchmark_skip_risk) and np.isfinite(reference_skip_risk)
            else float("nan")
        ),
        "end_risk_delta_vs_reference": (
            float(benchmark_end_risk - reference_end_risk)
            if np.isfinite(benchmark_end_risk) and np.isfinite(reference_end_risk)
            else float("nan")
        ),
        "scenario_rank_by_skip_risk": int(scenario_rank),
        "scenario_count_for_policy": int(len(policy_rows)),
        "evaluated_sessions": int(benchmark_row.get("evaluated_sessions", 0) or 0),
        "total_test_sessions": int(benchmark_row.get("total_test_sessions", 0) or 0),
        "sample_fraction": _safe_float(benchmark_row.get("sample_fraction")),
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
    sampling_rng = np.random.default_rng(random_seed)
    rollout_rng = np.random.default_rng(random_seed + 1)

    rows: list[dict[str, object]] = []
    baseline_policy = {"transition": 1.1, "continuity": 0.1, "novelty": 0.0, "repeat": 0.8}
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
    batch_size = _resolve_env_int(
        "SPOTIFY_STRESS_TEST_BATCH_SIZE",
        DEFAULT_STRESS_TEST_BATCH_SIZE,
        minimum=1,
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
        3,
    )

    for scenario_name, scenario in SCENARIOS.items():
        routed_policy_name, routed_policy = _scenario_safe_policy(
            safe_policy=safe_policy,
            scenario_name=scenario_name,
        )
        comparison_policies = {
            "baseline_exploit": baseline_policy,
            "safe_global": dict(safe_policy.global_policy),
            routed_policy_name: routed_policy,
        }
        for policy_name, weights in comparison_policies.items():
            started_at = time.perf_counter()
            evaluated_count = 0
            session_length_sum = 0.0
            skip_risk_sum = 0.0
            end_risk_sum = 0.0
            logger.info(
                "Stress-test scenario=%s policy=%s sessions=%d",
                scenario_name,
                policy_name,
                sampled_sessions,
            )
            use_batch_rollout = simulate_rollout is _DEFAULT_SIMULATE_ROLLOUT
            if use_batch_rollout:
                processed = 0
                next_progress = progress_every if progress_every > 0 else 0
                for start_idx in range(0, sampled_sessions, batch_size):
                    stop_idx = min(start_idx + batch_size, sampled_sessions)
                    batch_summary = simulate_rollout_batch_summary(
                        twin=digital_twin,
                        multimodal_space=multimodal_space,
                        causal_artifact=causal_artifact,
                        start_sequences=sampled_seq[start_idx:stop_idx],
                        start_contexts=sampled_ctx[start_idx:stop_idx],
                        horizon=6,
                        policy_weights=weights,
                        scenario=scenario,
                        rng=rollout_rng,
                    )
                    batch_lengths = np.asarray(batch_summary["session_length"], dtype="float64")
                    batch_skip_risks = np.asarray(batch_summary["mean_skip_risk"], dtype="float64")
                    batch_end_risks = np.asarray(batch_summary["mean_end_risk"], dtype="float64")
                    evaluated_count += int(batch_lengths.size)
                    session_length_sum += float(batch_lengths.sum(dtype="float64"))
                    skip_risk_sum += float(batch_skip_risks.sum(dtype="float64"))
                    end_risk_sum += float(batch_end_risks.sum(dtype="float64"))
                    processed = stop_idx
                    if progress_every > 0:
                        while processed >= next_progress and next_progress > 0:
                            logger.info(
                                "Stress-test progress scenario=%s policy=%s processed=%d/%d elapsed=%.1fs",
                                scenario_name,
                                policy_name,
                                min(next_progress, sampled_sessions),
                                sampled_sessions,
                                time.perf_counter() - started_at,
                            )
                            next_progress += progress_every
                    if processed == sampled_sessions and (progress_every <= 0 or (processed % progress_every) != 0):
                        logger.info(
                            "Stress-test progress scenario=%s policy=%s processed=%d/%d elapsed=%.1fs",
                            scenario_name,
                            policy_name,
                            processed,
                            sampled_sessions,
                            time.perf_counter() - started_at,
                        )
            else:
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
                    evaluated_count += 1
                    session_length_sum += float(rollout["session_length"])
                    skip_risk_sum += float(rollout["mean_skip_risk"])
                    end_risk_sum += float(rollout["mean_end_risk"])
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
            if evaluated_count:
                mean_session_length = session_length_sum / float(evaluated_count)
                mean_skip_risk = skip_risk_sum / float(evaluated_count)
                mean_end_risk = end_risk_sum / float(evaluated_count)
            else:
                mean_session_length = float("nan")
                mean_skip_risk = float("nan")
                mean_end_risk = float("nan")
            rows.append(
                {
                    "scenario": scenario_name,
                    "policy_name": policy_name,
                    "policy_family": ("safe" if policy_name.startswith("safe_") else "baseline"),
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
            "policy_family",
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
    benchmark_payload = _build_stress_benchmark(rows)
    benchmark_csv = _write_csv(
        output_dir / "stress_test_benchmark.csv",
        [benchmark_payload],
        [
            "benchmark_scenario",
            "benchmark_policy_name",
            "benchmark_policy_selection_mode",
            "reference_policy_name",
            "available",
            "reference_available",
            "skip_risk",
            "end_risk",
            "reference_skip_risk",
            "reference_end_risk",
            "skip_risk_delta_vs_reference",
            "end_risk_delta_vs_reference",
            "scenario_rank_by_skip_risk",
            "scenario_count_for_policy",
            "evaluated_sessions",
            "total_test_sessions",
            "sample_fraction",
        ],
    )
    benchmark_path = output_dir / "stress_test_benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
    if bool(benchmark_payload.get("available")):
        logger.info(
            "Stress-test benchmark scenario=%s policy=%s skip_risk=%.4f delta_vs_%s=%.4f",
            benchmark_payload["benchmark_scenario"],
            benchmark_payload["benchmark_policy_name"],
            benchmark_payload["skip_risk"],
            benchmark_payload["reference_policy_name"],
            benchmark_payload["skip_risk_delta_vs_reference"],
        )
    logger.info(
        "Ran stress-test lab across %d scenarios using %d/%d sessions.",
        len(SCENARIOS),
        sampled_sessions,
        total_sessions,
    )
    return [csv_path, summary_path, benchmark_csv, benchmark_path]
