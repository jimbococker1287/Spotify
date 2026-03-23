from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from .benchmarks import build_serving_tabular_features
from .data import PreparedData
from .digital_twin import ListenerDigitalTwinArtifact
from .multimodal import MultimodalArtistSpace
from .safe_policy import POLICY_TEMPLATES, SafeBanditPolicyArtifact


GROUP_SCENARIOS: dict[str, dict[str, float | int | str]] = {
    "household": {
        "members": 3,
        "horizon": 5,
        "default_policy": "comfort_policy",
        "fairness_weight": 0.55,
        "energy_target": 0.56,
        "danceability_target": 0.58,
        "friction_safe_threshold": 1.15,
        "disagreement_safe_threshold": 0.11,
        "end_safe_threshold": 0.40,
        "friction_decay": 0.90,
    },
    "party": {
        "members": 4,
        "horizon": 6,
        "default_policy": "novelty_boosted",
        "fairness_weight": 0.35,
        "energy_target": 0.78,
        "danceability_target": 0.78,
        "friction_safe_threshold": 0.80,
        "disagreement_safe_threshold": 0.12,
        "end_safe_threshold": 0.48,
        "friction_decay": 0.88,
    },
    "car": {
        "members": 2,
        "horizon": 5,
        "default_policy": "safe_balance",
        "fairness_weight": 0.60,
        "energy_target": 0.50,
        "danceability_target": 0.52,
        "friction_safe_threshold": 0.55,
        "disagreement_safe_threshold": 0.10,
        "end_safe_threshold": 0.36,
        "friction_decay": 0.82,
    },
    "shared_space": {
        "members": 3,
        "horizon": 5,
        "default_policy": "comfort_policy",
        "fairness_weight": 0.65,
        "energy_target": 0.44,
        "danceability_target": 0.48,
        "friction_safe_threshold": 0.72,
        "disagreement_safe_threshold": 0.10,
        "end_safe_threshold": 0.38,
        "friction_decay": 0.86,
    },
}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _session_pool(data: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    seq_batches: list[np.ndarray] = []
    ctx_batches: list[np.ndarray] = []
    for seq, ctx in (
        (data.X_seq_test, data.X_ctx_test),
        (data.X_seq_val, data.X_ctx_val),
        (data.X_seq_train, data.X_ctx_train),
    ):
        seq_arr = np.asarray(seq, dtype="int32")
        ctx_arr = np.asarray(ctx, dtype="float32")
        if len(seq_arr) == 0:
            continue
        seq_batches.append(seq_arr)
        ctx_batches.append(ctx_arr)
    if not seq_batches:
        return (
            np.empty((0, 0), dtype="int32"),
            np.empty((0, len(getattr(data, "context_features", []))), dtype="float32"),
        )
    return (
        np.concatenate(seq_batches, axis=0).astype("int32", copy=False),
        np.concatenate(ctx_batches, axis=0).astype("float32", copy=False),
    )


def _friction_feature_indices(context_features: list[str]) -> np.ndarray:
    indices = [
        idx
        for idx, feature_name in enumerate(context_features)
        if str(feature_name).startswith("tech_")
        or "error" in str(feature_name)
        or str(feature_name) == "offline"
    ]
    return np.asarray(indices, dtype="int64")


def _friction_score(ctx_batch: np.ndarray, friction_indices: np.ndarray) -> float:
    if len(ctx_batch) == 0 or friction_indices.size == 0:
        return 0.0
    friction = np.maximum(np.asarray(ctx_batch, dtype="float32")[:, friction_indices], 0.0)
    return float(np.mean(np.sum(friction, axis=1)))


def _preference_disagreement(
    seq_batch: np.ndarray,
    *,
    twin: ListenerDigitalTwinArtifact,
) -> float:
    if len(seq_batch) <= 1:
        return 0.0
    last_artist = np.asarray(seq_batch[:, -1], dtype="int32")
    transition = np.asarray(twin.transition_matrix[last_artist], dtype="float32")
    centroid = np.mean(transition, axis=0, keepdims=True)
    return float(np.mean(np.abs(transition - centroid)))


def _mean_end_risk(
    *,
    seq_batch: np.ndarray,
    ctx_batch: np.ndarray,
    twin: ListenerDigitalTwinArtifact,
) -> float:
    if len(seq_batch) == 0:
        return float("nan")
    features = build_serving_tabular_features(seq_batch, ctx_batch)
    proba = np.asarray(twin.end_estimator.predict_proba(features), dtype="float32")[:, 1]
    return float(np.mean(proba)) if len(proba) else float("nan")


def _softmax(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype="float64")
    shifted = arr - np.max(arr)
    weights = np.exp(np.clip(shifted, -30.0, 30.0))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full(arr.shape, 1.0 / max(1, arr.size), dtype="float32")
    return (weights / total).astype("float32", copy=False)


def _member_distribution(
    *,
    sequence: np.ndarray,
    weights: dict[str, float],
    scenario: dict[str, float | int | str],
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
) -> np.ndarray:
    seq_arr = np.asarray(sequence, dtype="int32")
    artist_ids = np.arange(len(multimodal_space.artist_labels), dtype="int32")
    last_artist = int(seq_arr[-1])
    transition = float(weights["transition"]) * np.log(
        np.clip(digital_twin.transition_matrix[last_artist], 1e-6, 1.0)
    )
    continuity = float(weights.get("continuity", 0.0)) * (
        multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings.T
    )
    novelty = float(weights.get("novelty", 0.0)) * (1.0 - multimodal_space.popularity)
    repeat_penalty = float(weights.get("repeat", 0.0)) * np.isin(artist_ids, seq_arr).astype("float32")
    energy_penalty = 0.14 * np.abs(multimodal_space.energy - float(scenario["energy_target"]))
    danceability_penalty = 0.10 * np.abs(
        multimodal_space.danceability - float(scenario["danceability_target"])
    )
    raw = transition + continuity + novelty - repeat_penalty - energy_penalty - danceability_penalty
    return _softmax(raw)


def _bucket_policy(
    *,
    safe_policy: SafeBanditPolicyArtifact,
    bucket_name: str,
) -> tuple[str, dict[str, float]]:
    if bucket_name in safe_policy.policy_map:
        return f"safe_bucket_{bucket_name}", dict(safe_policy.policy_map[bucket_name])
    if safe_policy.policy_map:
        first_bucket = sorted(safe_policy.policy_map)[0]
        return f"safe_bucket_{first_bucket}", dict(safe_policy.policy_map[first_bucket])
    return "safe_global", dict(safe_policy.global_policy)


def _route_policy(
    *,
    safe_policy: SafeBanditPolicyArtifact,
    scenario_name: str,
    scenario: dict[str, float | int | str],
    disagreement: float,
    friction: float,
    mean_end_risk: float,
) -> tuple[str, dict[str, float], bool]:
    friction_threshold = float(scenario["friction_safe_threshold"])
    disagreement_threshold = float(scenario["disagreement_safe_threshold"])
    end_threshold = float(scenario["end_safe_threshold"])
    safe_by_friction = friction >= friction_threshold
    safe_by_disagreement = disagreement >= disagreement_threshold
    safe_by_end = mean_end_risk >= end_threshold

    if safe_by_friction:
        if friction >= (friction_threshold * 1.35):
            policy_name, policy = _bucket_policy(safe_policy=safe_policy, bucket_name="high_friction")
        else:
            policy_name, policy = _bucket_policy(safe_policy=safe_policy, bucket_name="normal_friction")
        return policy_name, policy, True
    if safe_by_disagreement or safe_by_end:
        return "safe_global", dict(safe_policy.global_policy), True

    default_policy = str(scenario.get("default_policy", "safe_balance"))
    if default_policy not in POLICY_TEMPLATES:
        default_policy = "safe_balance"
    return f"{scenario_name}_{default_policy}", dict(POLICY_TEMPLATES[default_policy]), False


def build_group_auto_dj_plans(
    *,
    data: PreparedData,
    artist_labels: list[str],
    multimodal_space: MultimodalArtistSpace,
    digital_twin: ListenerDigitalTwinArtifact,
    safe_policy: SafeBanditPolicyArtifact,
    output_dir: Path,
    logger,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_pool, ctx_pool = _session_pool(data)
    if len(seq_pool) == 0 or len(artist_labels) == 0:
        return []

    friction_indices = _friction_feature_indices(list(digital_twin.context_features))
    plan_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for scenario_offset, (scenario_name, scenario_raw) in enumerate(GROUP_SCENARIOS.items()):
        scenario = dict(scenario_raw)
        cohort_size = min(int(scenario["members"]), len(seq_pool))
        if cohort_size <= 0:
            continue
        indices = [
            int((scenario_offset + (member_idx * 2)) % len(seq_pool))
            for member_idx in range(cohort_size)
        ]
        seq_batch = np.asarray(seq_pool[indices], dtype="int32").copy()
        ctx_batch = np.asarray(ctx_pool[indices], dtype="float32").copy()
        chosen_artists: list[int] = []
        routed_safely = 0
        fairness_values: list[float] = []
        satisfaction_means: list[float] = []
        satisfaction_floors: list[float] = []
        disagreement_values: list[float] = []
        end_risk_values: list[float] = []

        for step in range(1, int(scenario["horizon"]) + 1):
            disagreement = _preference_disagreement(seq_batch, twin=digital_twin)
            friction = _friction_score(ctx_batch, friction_indices)
            mean_end_risk = _mean_end_risk(
                seq_batch=seq_batch,
                ctx_batch=ctx_batch,
                twin=digital_twin,
            )
            policy_name, policy_weights, safe_routed = _route_policy(
                safe_policy=safe_policy,
                scenario_name=scenario_name,
                scenario=scenario,
                disagreement=disagreement,
                friction=friction,
                mean_end_risk=mean_end_risk,
            )
            if safe_routed:
                routed_safely += 1

            member_probs = np.stack(
                [
                    _member_distribution(
                        sequence=sequence,
                        weights=policy_weights,
                        scenario=scenario,
                        digital_twin=digital_twin,
                        multimodal_space=multimodal_space,
                    )
                    for sequence in seq_batch
                ],
                axis=0,
            )
            mean_prob = np.mean(member_probs, axis=0)
            min_prob = np.min(member_probs, axis=0)
            recent_repeat = np.isin(
                np.arange(len(multimodal_space.artist_labels), dtype="int32"),
                np.asarray(chosen_artists[-2:], dtype="int32"),
            ).astype("float32")
            group_score = (
                (1.0 - float(scenario["fairness_weight"])) * mean_prob
                + float(scenario["fairness_weight"]) * min_prob
                + 0.06 * (1.0 - np.abs(multimodal_space.energy - float(scenario["energy_target"])))
                + 0.04 * (1.0 - np.abs(multimodal_space.danceability - float(scenario["danceability_target"])))
                - 0.05 * recent_repeat
            )
            chosen_artist = int(np.argmax(group_score))
            chosen_artists.append(chosen_artist)

            member_satisfaction = np.asarray(member_probs[:, chosen_artist], dtype="float32")
            mean_satisfaction = float(np.mean(member_satisfaction))
            min_satisfaction = float(np.min(member_satisfaction))
            fairness = min_satisfaction / max(mean_satisfaction, 1e-6)
            fairness_values.append(fairness)
            satisfaction_means.append(mean_satisfaction)
            satisfaction_floors.append(min_satisfaction)
            disagreement_values.append(disagreement)

            for member_idx in range(cohort_size):
                seq_batch[member_idx] = np.roll(seq_batch[member_idx], -1)
                seq_batch[member_idx, -1] = chosen_artist
            if ctx_batch.ndim == 2 and ctx_batch.shape[1] > 0:
                ctx_batch[:, 0] = np.mod(ctx_batch[:, 0] + 1.0, 24.0)
                if friction_indices.size:
                    ctx_batch[:, friction_indices] = np.maximum(
                        0.0,
                        ctx_batch[:, friction_indices] * float(scenario["friction_decay"]),
                    )

            projected_end_risk = _mean_end_risk(
                seq_batch=seq_batch,
                ctx_batch=ctx_batch,
                twin=digital_twin,
            )
            end_risk_values.append(projected_end_risk)
            plan_rows.append(
                {
                    "scenario": scenario_name,
                    "cohort_size": cohort_size,
                    "step": step,
                    "policy_name": policy_name,
                    "safe_routed": int(safe_routed),
                    "artist_label": chosen_artist,
                    "artist_name": artist_labels[chosen_artist],
                    "mean_member_satisfaction": mean_satisfaction,
                    "min_member_satisfaction": min_satisfaction,
                    "fairness": fairness,
                    "disagreement": disagreement,
                    "friction_score": friction,
                    "projected_end_risk": projected_end_risk,
                }
            )

        if not chosen_artists:
            continue
        summary_rows.append(
            {
                "scenario": scenario_name,
                "cohort_size": cohort_size,
                "planned_horizon": int(len(chosen_artists)),
                "default_policy": str(scenario["default_policy"]),
                "safe_route_rate": float(routed_safely / max(1, len(chosen_artists))),
                "mean_disagreement": float(np.mean(disagreement_values)) if disagreement_values else float("nan"),
                "mean_fairness": float(np.mean(fairness_values)) if fairness_values else float("nan"),
                "mean_member_satisfaction": float(np.mean(satisfaction_means)) if satisfaction_means else float("nan"),
                "min_member_satisfaction": float(np.min(satisfaction_floors)) if satisfaction_floors else float("nan"),
                "mean_projected_end_risk": float(np.mean(end_risk_values)) if end_risk_values else float("nan"),
                "first_artist": artist_labels[int(chosen_artists[0])],
                "last_artist": artist_labels[int(chosen_artists[-1])],
            }
        )

    if not plan_rows:
        return []

    plan_path = _write_csv(
        output_dir / "group_auto_dj_plans.csv",
        plan_rows,
        [
            "scenario",
            "cohort_size",
            "step",
            "policy_name",
            "safe_routed",
            "artist_label",
            "artist_name",
            "mean_member_satisfaction",
            "min_member_satisfaction",
            "fairness",
            "disagreement",
            "friction_score",
            "projected_end_risk",
        ],
    )
    summary_path = output_dir / "group_auto_dj_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    logger.info("Built group Auto-DJ plans across %d scenarios", len(summary_rows))
    return [plan_path, summary_path]
