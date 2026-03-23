from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import joblib
import numpy as np

from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact, simulate_rollout
from .multimodal import MultimodalArtistSpace


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _friction_bucket(data: PreparedData) -> np.ndarray:
    if data.num_ctx == 0:
        return np.full(len(data.X_ctx_val), "default", dtype=object)
    score = np.sum(np.maximum(np.asarray(data.X_ctx_val, dtype="float32"), 0.0), axis=1)
    threshold = float(np.quantile(score, 0.75)) if len(score) else 0.0
    return np.where(score >= threshold, "high_friction", "normal_friction")


def _reward(hit: float, novelty: float, skip_risk: float, end_risk: float) -> float:
    return float((1.0 * hit) + (0.15 * novelty) - (0.35 * skip_risk) - (0.40 * end_risk))


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

    for bucket in sorted({str(item) for item in buckets.tolist()}):
        mask = buckets == bucket
        best_name = "safe_balance"
        best_lcb = float("-inf")
        for policy_name, weights in POLICY_TEMPLATES.items():
            rewards: list[float] = []
            for seq, ctx, y_true in zip(data.X_seq_val[mask], data.X_ctx_val[mask], data.y_val[mask], strict=False):
                rollout = simulate_rollout(
                    twin=digital_twin,
                    multimodal_space=multimodal_space,
                    causal_artifact=None,
                    start_sequence=seq,
                    start_context=ctx,
                    horizon=1,
                    policy_weights=weights,
                    scenario=None,
                    rng=rng,
                )
                predicted = int(rollout["planned_sequence"][0]) if rollout["planned_sequence"] else int(seq[-1])
                novelty = float(1.0 - multimodal_space.popularity[predicted])
                rewards.append(
                    _reward(
                        hit=float(predicted == int(y_true)),
                        novelty=novelty,
                        skip_risk=float(rollout["mean_skip_risk"]),
                        end_risk=float(rollout["mean_end_risk"]),
                    )
                )
            if not rewards:
                continue
            arr = np.asarray(rewards, dtype="float64")
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

    global_name = max(
        POLICY_TEMPLATES,
        key=lambda name: float(np.mean([row["mean_reward"] for row in rows if row["policy_name"] == name])) if any(row["policy_name"] == name for row in rows) else float("-inf"),
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
