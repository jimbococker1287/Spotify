from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import os

import joblib
import numpy as np

from .causal_friction import CausalSkipDecompositionArtifact
from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact, simulate_rollout_batch_summary
from .multimodal import MultimodalArtistSpace
from .run_artifacts import write_csv_rows


@dataclass(frozen=True)
class SafeBanditPolicyArtifact:
    policy_map: dict[str, dict[str, float]]
    global_policy: dict[str, float]
    reward_metric: str
    global_policy_name: str = "safe_global"
    benchmark_scenario: str = ""
    benchmark_reference_policy_name: str = "exploit_preference"
    benchmark_metrics: dict[str, float] = field(default_factory=dict)


POLICY_TEMPLATES: dict[str, dict[str, float]] = {
    "exploit_preference": {"transition": 1.1, "continuity": 0.1, "novelty": 0.0, "repeat": 0.8},
    "novelty_boosted": {"transition": 0.8, "continuity": 0.2, "novelty": 0.5, "repeat": 0.6},
    "comfort_policy": {"transition": 1.0, "continuity": 0.4, "novelty": 0.1, "repeat": 0.9},
    "safe_balance": {"transition": 0.9, "continuity": 0.3, "novelty": 0.3, "repeat": 0.7},
}

DEFAULT_SAFE_POLICY_REFERENCE_POLICY_NAME = "exploit_preference"
DEFAULT_SAFE_POLICY_BENCHMARK_SCENARIO = "evening_drift"
_BENCHMARK_EPSILON = 1e-6


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _resolve_env_name(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _friction_feature_indices(data: PreparedData) -> np.ndarray:
    preferred_names = {
        "offline",
        "skipped",
        "session_skip_rate_so_far",
        "recent_skip_rate_5",
        "recent_skip_rate_20",
    }
    return np.asarray(
        [
            idx
            for idx, feature_name in enumerate(data.context_features)
            if str(feature_name).startswith("tech_") or str(feature_name) in preferred_names
        ],
        dtype="int64",
    )


def _friction_bucket(data: PreparedData) -> np.ndarray:
    if data.num_ctx == 0:
        return np.full(len(data.X_ctx_val), "default", dtype=object)
    friction_idx = _friction_feature_indices(data)
    if friction_idx.size == 0:
        return np.full(len(data.X_ctx_val), "default", dtype=object)
    ctx = np.asarray(data.X_ctx_val, dtype="float32")
    friction_values = ctx[:, friction_idx]
    if friction_values.size == 0:
        return np.full(len(data.X_ctx_val), "default", dtype=object)
    positive_pressure = np.maximum(friction_values, 0.0)
    score = np.sum(positive_pressure, axis=1)
    if len(score) <= 1:
        return np.full(len(score), "normal_friction", dtype=object)
    low_threshold = float(np.quantile(score, 0.25))
    high_threshold = float(np.quantile(score, 0.75))
    buckets = np.full(len(score), "normal_friction", dtype=object)
    buckets[score <= low_threshold] = "low_friction"
    buckets[score >= high_threshold] = "high_friction"
    return buckets


def _select_global_safe_policy(
    *,
    data: PreparedData,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    causal_artifact: CausalSkipDecompositionArtifact | None,
    policy_means: dict[str, list[float]],
    rng: np.random.Generator,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    from .stress_test import DEFAULT_STRESS_BENCHMARK_SCENARIO, SCENARIOS

    fallback_name = max(
        POLICY_TEMPLATES,
        key=lambda name: float(np.mean(policy_means[name])) if policy_means[name] else float("-inf"),
    )
    benchmark_name = _resolve_env_name(
        "SPOTIFY_STRESS_BENCHMARK_SCENARIO",
        DEFAULT_STRESS_BENCHMARK_SCENARIO,
    )
    benchmark_scenario = dict(
        SCENARIOS.get(
            benchmark_name,
            SCENARIOS.get(DEFAULT_STRESS_BENCHMARK_SCENARIO, {}),
        )
    )
    reference_name = DEFAULT_SAFE_POLICY_REFERENCE_POLICY_NAME
    seq_val = np.asarray(getattr(data, "X_seq_val", []), dtype="int32")
    ctx_val = np.asarray(getattr(data, "X_ctx_val", []), dtype="float32")
    if len(seq_val) == 0 or len(ctx_val) == 0:
        return fallback_name, [], {
            "benchmark_scenario": benchmark_name,
            "benchmark_reference_policy_name": reference_name,
            "global_selection_strategy": "fallback_mean_reward_no_validation_rows",
            "benchmark_available": False,
        }

    benchmark_rows: list[dict[str, object]] = []
    for policy_name, weights in POLICY_TEMPLATES.items():
        batch_summary = simulate_rollout_batch_summary(
            twin=digital_twin,
            multimodal_space=multimodal_space,
            causal_artifact=causal_artifact,
            start_sequences=seq_val,
            start_contexts=ctx_val,
            horizon=6,
            policy_weights=weights,
            scenario=benchmark_scenario,
            rng=rng,
        )
        skip_risk = _safe_float(np.nanmean(np.asarray(batch_summary["mean_skip_risk"], dtype="float64")))
        end_risk = _safe_float(np.nanmean(np.asarray(batch_summary["mean_end_risk"], dtype="float64")))
        mean_session_length = _safe_float(np.mean(np.asarray(batch_summary["session_length"], dtype="float64")))
        benchmark_rows.append(
            {
                "policy_name": policy_name,
                "mean_skip_risk": skip_risk,
                "mean_end_risk": end_risk,
                "mean_session_length": mean_session_length,
                "mean_reward": float(np.mean(policy_means.get(policy_name, [float("nan")]))),
            }
        )

    reference_row = next((row for row in benchmark_rows if row["policy_name"] == reference_name), {})
    reference_skip = _safe_float(reference_row.get("mean_skip_risk"))
    reference_end = _safe_float(reference_row.get("mean_end_risk"))
    for row in benchmark_rows:
        skip_risk = _safe_float(row.get("mean_skip_risk"))
        end_risk = _safe_float(row.get("mean_end_risk"))
        row["skip_risk_delta_vs_reference"] = (
            float(skip_risk - reference_skip)
            if math.isfinite(skip_risk) and math.isfinite(reference_skip)
            else float("nan")
        )
        row["end_risk_delta_vs_reference"] = (
            float(end_risk - reference_end)
            if math.isfinite(end_risk) and math.isfinite(reference_end)
            else float("nan")
        )
        row["beats_reference"] = bool(
            math.isfinite(skip_risk)
            and math.isfinite(reference_skip)
            and (
                skip_risk < (reference_skip - _BENCHMARK_EPSILON)
                or (
                    abs(skip_risk - reference_skip) <= _BENCHMARK_EPSILON
                    and math.isfinite(end_risk)
                    and math.isfinite(reference_end)
                    and end_risk < (reference_end - _BENCHMARK_EPSILON)
                )
            )
        )

    def _benchmark_sort_key(row: dict[str, object]) -> tuple[float, float, float, float, str]:
        return (
            _safe_float(row.get("mean_skip_risk")),
            _safe_float(row.get("mean_end_risk")),
            -_safe_float(row.get("mean_session_length")),
            -_safe_float(row.get("mean_reward")),
            str(row.get("policy_name", "")),
        )

    safe_candidates = [
        row
        for row in benchmark_rows
        if str(row.get("policy_name", "")) != reference_name
    ]
    beating_candidates = [row for row in safe_candidates if bool(row.get("beats_reference"))]
    if beating_candidates:
        selected_row = min(beating_candidates, key=_benchmark_sort_key)
        selection_strategy = "beats_reference_on_stress_benchmark"
    elif safe_candidates:
        selected_row = min(safe_candidates, key=_benchmark_sort_key)
        selection_strategy = "best_available_safe_policy_on_stress_benchmark"
    else:
        selected_row = min(benchmark_rows, key=_benchmark_sort_key, default={})
        selection_strategy = "fallback_reference_policy"

    selected_name = str(selected_row.get("policy_name", "")).strip() or fallback_name
    for rank, row in enumerate(sorted(benchmark_rows, key=_benchmark_sort_key), start=1):
        row["selection_rank"] = rank

    summary = {
        "benchmark_scenario": benchmark_name,
        "benchmark_reference_policy_name": reference_name,
        "benchmark_available": bool(benchmark_rows),
        "global_selection_strategy": selection_strategy,
        "selected_policy_name": selected_name,
        "selected_skip_risk": _safe_float(selected_row.get("mean_skip_risk")),
        "selected_end_risk": _safe_float(selected_row.get("mean_end_risk")),
        "reference_skip_risk": reference_skip,
        "reference_end_risk": reference_end,
        "benchmark_beats_reference": bool(selected_row.get("beats_reference")),
    }
    return selected_name, benchmark_rows, summary


def learn_safe_bandit_policy(
    *,
    data: PreparedData,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    causal_artifact: CausalSkipDecompositionArtifact | None,
    output_dir: Path,
    logger,
    random_seed: int,
) -> tuple[SafeBanditPolicyArtifact, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    buckets = _friction_bucket(data)
    rows: list[dict[str, object]] = []
    policy_map: dict[str, dict[str, float]] = {}
    novelty_values = 1.0 - np.asarray(multimodal_space.popularity, dtype="float32")

    for bucket in sorted({str(item) for item in buckets.tolist()}):
        mask = buckets == bucket
        bucket_seq = np.asarray(data.X_seq_val[mask], dtype="int32")
        bucket_ctx = np.asarray(data.X_ctx_val[mask], dtype="float32")
        bucket_targets = np.asarray(data.y_val[mask], dtype="int32")
        if bucket_targets.size == 0:
            continue
        best_name = "safe_balance"
        best_lcb = float("-inf")
        for policy_name, weights in POLICY_TEMPLATES.items():
            batch_summary = simulate_rollout_batch_summary(
                twin=digital_twin,
                multimodal_space=multimodal_space,
                causal_artifact=causal_artifact,
                start_sequences=bucket_seq,
                start_contexts=bucket_ctx,
                horizon=1,
                policy_weights=weights,
                scenario=None,
                rng=rng,
            )
            predictions = np.asarray(batch_summary["first_choice"], dtype="int32")
            if predictions.size == 0:
                continue
            predicted_novelty = novelty_values[np.clip(predictions, 0, len(novelty_values) - 1)]
            arr = (
                (predictions == bucket_targets).astype("float64")
                + (0.15 * predicted_novelty.astype("float64", copy=False))
                - (0.35 * np.asarray(batch_summary["mean_skip_risk"], dtype="float64"))
                - (0.40 * np.asarray(batch_summary["mean_end_risk"], dtype="float64"))
            )
            mean = float(np.mean(arr))
            stderr = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            lcb = mean - (1.96 * stderr)
            rows.append(
                {
                    "bucket": bucket,
                    "policy_name": policy_name,
                    "mean_reward": mean,
                    "stderr_reward": stderr,
                    "lcb_reward": lcb,
                    "n": int(len(arr)),
                }
            )
            if lcb > best_lcb:
                best_lcb = lcb
                best_name = policy_name
        policy_map[bucket] = dict(POLICY_TEMPLATES[best_name])

    policy_means = {
        name: [float(row["mean_reward"]) for row in rows if row["policy_name"] == name]
        for name in POLICY_TEMPLATES
    }
    benchmark_rng = np.random.default_rng(random_seed + 17)
    global_name, benchmark_rows, benchmark_summary = _select_global_safe_policy(
        data=data,
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        causal_artifact=causal_artifact,
        policy_means=policy_means,
        rng=benchmark_rng,
    )
    artifact = SafeBanditPolicyArtifact(
        policy_map=policy_map,
        global_policy=dict(POLICY_TEMPLATES[global_name]),
        reward_metric="hit_plus_novelty_minus_risk",
        global_policy_name=global_name,
        benchmark_scenario=str(benchmark_summary.get("benchmark_scenario", "")),
        benchmark_reference_policy_name=str(
            benchmark_summary.get(
                "benchmark_reference_policy_name",
                DEFAULT_SAFE_POLICY_REFERENCE_POLICY_NAME,
            )
        ),
        benchmark_metrics={
            "skip_risk": _safe_float(benchmark_summary.get("selected_skip_risk")),
            "end_risk": _safe_float(benchmark_summary.get("selected_end_risk")),
            "reference_skip_risk": _safe_float(benchmark_summary.get("reference_skip_risk")),
            "reference_end_risk": _safe_float(benchmark_summary.get("reference_end_risk")),
        },
    )

    artifact_path = output_dir / "safe_bandit_policy.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    csv_path = _write_csv(
        output_dir / "safe_bandit_policy_candidates.csv",
        rows,
        ["bucket", "policy_name", "mean_reward", "stderr_reward", "lcb_reward", "n"],
    )
    map_rows = [
        {"bucket": bucket, **{key: float(value) for key, value in weights.items()}}
        for bucket, weights in sorted(policy_map.items())
    ]
    map_path = _write_csv(
        output_dir / "safe_bandit_policy_map.csv",
        map_rows,
        ["bucket", "transition", "continuity", "novelty", "repeat"],
    )
    benchmark_path = _write_csv(
        output_dir / "safe_bandit_policy_benchmark.csv",
        benchmark_rows,
        [
            "policy_name",
            "mean_skip_risk",
            "mean_end_risk",
            "mean_session_length",
            "mean_reward",
            "skip_risk_delta_vs_reference",
            "end_risk_delta_vs_reference",
            "beats_reference",
            "selection_rank",
        ],
    )
    summary_path = output_dir / "safe_bandit_policy_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "global_policy_name": global_name,
                "global_policy": artifact.global_policy,
                "bucket_count": len(policy_map),
                "reward_metric": artifact.reward_metric,
                **benchmark_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Learned safe bandit policy across %d buckets | global=%s strategy=%s benchmark_beats_reference=%s",
        len(policy_map),
        global_name,
        benchmark_summary.get("global_selection_strategy", ""),
        benchmark_summary.get("benchmark_beats_reference", False),
    )
    return artifact, [artifact_path, csv_path, map_path, benchmark_path, summary_path]
