from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import joblib
import numpy as np

from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact, simulate_rollout_batch_summary
from .multimodal import MultimodalArtistSpace
from .run_artifacts import write_csv_rows


@dataclass(frozen=True)
class SafeBanditPolicyArtifact:
    policy_map: dict[str, dict[str, float]]
    global_policy: dict[str, float]
    reward_metric: str


POLICY_TEMPLATES: dict[str, dict[str, float]] = {
    "exploit_preference": {"transition": 1.1, "continuity": 0.1, "novelty": 0.0, "repeat": 0.8},
    "novelty_boosted": {"transition": 0.8, "continuity": 0.2, "novelty": 0.5, "repeat": 0.6},
    "comfort_policy": {"transition": 1.0, "continuity": 0.4, "novelty": 0.1, "repeat": 0.9},
    "safe_balance": {"transition": 0.9, "continuity": 0.3, "novelty": 0.3, "repeat": 0.7},
}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


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


def learn_safe_bandit_policy(
    *,
    data: PreparedData,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
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
                causal_artifact=None,
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
    global_name = max(
        POLICY_TEMPLATES,
        key=lambda name: float(np.mean(policy_means[name])) if policy_means[name] else float("-inf"),
    )
    artifact = SafeBanditPolicyArtifact(
        policy_map=policy_map,
        global_policy=dict(POLICY_TEMPLATES[global_name]),
        reward_metric="hit_plus_novelty_minus_risk",
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
    summary_path = output_dir / "safe_bandit_policy_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "global_policy_name": global_name,
                "global_policy": artifact.global_policy,
                "bucket_count": len(policy_map),
                "reward_metric": artifact.reward_metric,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Learned safe bandit policy across %d buckets", len(policy_map))
    return artifact, [artifact_path, csv_path, map_path, summary_path]
