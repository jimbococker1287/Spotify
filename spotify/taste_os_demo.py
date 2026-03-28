from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from .benchmarks import build_serving_tabular_features
from .champion_alias import resolve_prediction_run_dir
from .digital_twin import ListenerDigitalTwinArtifact
from .env import load_local_env
from .multimodal import MultimodalArtistSpace
from .predict_next import _prepare_inputs, load_prediction_input_context
from .safe_policy import POLICY_TEMPLATES, SafeBanditPolicyArtifact
from .serving import load_predictor, resolve_model_row


class _PredictorLike(Protocol):
    model_name: str
    model_type: str
    artist_labels: list[str]

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ModeConfig:
    name: str
    description: str
    horizon: int
    candidate_shortlist: int
    model_weight: float
    continuity_weight: float
    arc_weight: float
    novelty_weight: float
    freshness_weight: float
    repeat_penalty: float
    frequency_penalty: float
    hard_repeat_window: int
    energy_target: float
    energy_weight: float
    default_policy_name: str
    friction_guard_threshold: float
    friction_high_threshold: float
    end_guard_threshold: float
    surface_probability_weight: float
    surface_transition_weight: float
    surface_continuity_target: float
    surface_continuity_weight: float
    surface_arc_target: float
    surface_arc_weight: float
    surface_freshness_target: float
    surface_freshness_weight: float


@dataclass(frozen=True)
class AdaptiveEvent:
    after_step: int
    event_type: str
    description: str


@dataclass(frozen=True)
class AdaptiveScenario:
    name: str
    description: str
    events: tuple[AdaptiveEvent, ...]


MODE_CONFIGS: dict[str, ModeConfig] = {
    "focus": ModeConfig(
        name="focus",
        description="Low-friction, low-surprise arcs for concentrated listening sessions.",
        horizon=6,
        candidate_shortlist=14,
        model_weight=1.0,
        continuity_weight=0.42,
        arc_weight=0.18,
        novelty_weight=0.04,
        freshness_weight=0.02,
        repeat_penalty=1.00,
        frequency_penalty=0.55,
        hard_repeat_window=2,
        energy_target=0.48,
        energy_weight=0.14,
        default_policy_name="comfort_policy",
        friction_guard_threshold=1.70,
        friction_high_threshold=2.00,
        end_guard_threshold=0.42,
        surface_probability_weight=0.34,
        surface_transition_weight=0.08,
        surface_continuity_target=0.78,
        surface_continuity_weight=0.28,
        surface_arc_target=0.74,
        surface_arc_weight=0.18,
        surface_freshness_target=0.30,
        surface_freshness_weight=0.12,
    ),
    "workout": ModeConfig(
        name="workout",
        description="Rising-energy plans with momentum and controlled novelty.",
        horizon=6,
        candidate_shortlist=14,
        model_weight=0.90,
        continuity_weight=0.20,
        arc_weight=0.10,
        novelty_weight=0.28,
        freshness_weight=0.12,
        repeat_penalty=0.85,
        frequency_penalty=0.38,
        hard_repeat_window=2,
        energy_target=0.78,
        energy_weight=0.18,
        default_policy_name="novelty_boosted",
        friction_guard_threshold=1.35,
        friction_high_threshold=1.75,
        end_guard_threshold=0.46,
        surface_probability_weight=0.30,
        surface_transition_weight=0.12,
        surface_continuity_target=0.56,
        surface_continuity_weight=0.16,
        surface_arc_target=0.50,
        surface_arc_weight=0.12,
        surface_freshness_target=0.74,
        surface_freshness_weight=0.30,
    ),
    "commute": ModeConfig(
        name="commute",
        description="Shorter, resilient plans that recover quickly from disruption.",
        horizon=4,
        candidate_shortlist=12,
        model_weight=0.88,
        continuity_weight=0.34,
        arc_weight=0.16,
        novelty_weight=0.06,
        freshness_weight=0.04,
        repeat_penalty=1.05,
        frequency_penalty=0.62,
        hard_repeat_window=2,
        energy_target=0.56,
        energy_weight=0.11,
        default_policy_name="safe_balance",
        friction_guard_threshold=1.25,
        friction_high_threshold=1.65,
        end_guard_threshold=0.38,
        surface_probability_weight=0.40,
        surface_transition_weight=0.14,
        surface_continuity_target=0.58,
        surface_continuity_weight=0.20,
        surface_arc_target=0.58,
        surface_arc_weight=0.14,
        surface_freshness_target=0.48,
        surface_freshness_weight=0.12,
    ),
    "discovery": ModeConfig(
        name="discovery",
        description="Novelty-weighted plans that stay inside learned taste boundaries.",
        horizon=6,
        candidate_shortlist=16,
        model_weight=0.62,
        continuity_weight=0.10,
        arc_weight=0.04,
        novelty_weight=0.46,
        freshness_weight=0.24,
        repeat_penalty=0.82,
        frequency_penalty=0.48,
        hard_repeat_window=2,
        energy_target=0.62,
        energy_weight=0.09,
        default_policy_name="novelty_boosted",
        friction_guard_threshold=1.75,
        friction_high_threshold=2.10,
        end_guard_threshold=0.48,
        surface_probability_weight=0.22,
        surface_transition_weight=0.06,
        surface_continuity_target=0.20,
        surface_continuity_weight=0.16,
        surface_arc_target=0.24,
        surface_arc_weight=0.12,
        surface_freshness_target=0.88,
        surface_freshness_weight=0.44,
    ),
}


SCENARIOS: dict[str, AdaptiveScenario] = {
    "steady": AdaptiveScenario(
        name="steady",
        description="No disruption. The planner stays on its default mode behavior.",
        events=(),
    ),
    "skip_recovery": AdaptiveScenario(
        name="skip_recovery",
        description="The listener skips an early suggestion and the planner moves closer to the recent arc.",
        events=(
            AdaptiveEvent(
                after_step=1,
                event_type="skip",
                description="The listener skips the first suggested track.",
            ),
        ),
    ),
    "repeat_request": AdaptiveScenario(
        name="repeat_request",
        description="The listener repeats a track, so the planner leans into continuity and comfort.",
        events=(
            AdaptiveEvent(
                after_step=1,
                event_type="repeat_request",
                description="The listener repeats a track and signals they want a familiar lane.",
            ),
        ),
    ),
    "friction_spike": AdaptiveScenario(
        name="friction_spike",
        description="Playback friction spikes mid-session and the planner routes toward safer behavior.",
        events=(
            AdaptiveEvent(
                after_step=2,
                event_type="friction_spike",
                description="Playback errors or network friction spike after the second step.",
            ),
        ),
    ),
    "mixed_session": AdaptiveScenario(
        name="mixed_session",
        description="The session sees an early skip and then a friction spike, forcing two replans.",
        events=(
            AdaptiveEvent(
                after_step=1,
                event_type="skip",
                description="The listener skips an early suggestion.",
            ),
            AdaptiveEvent(
                after_step=3,
                event_type="friction_spike",
                description="Playback friction spikes later in the session.",
            ),
        ),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.taste_os_demo",
        description="Run the Personal Taste OS demo with adaptive steering and written artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument(
        "--mode",
        type=str,
        default="focus",
        choices=sorted(MODE_CONFIGS),
        help="Taste OS mode to simulate.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="steady",
        choices=sorted(SCENARIOS),
        help="Adaptive scenario to simulate after the first plan is built.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to surface.")
    parser.add_argument(
        "--recent-artists",
        type=str,
        default=None,
        help="Optional pipe-separated artist names to use as the recent sequence override.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files while rebuilding the latest context.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis/taste_os_demo",
        help="Directory to write the JSON and Markdown demo artifacts.",
    )
    parser.add_argument(
        "--stdout-format",
        type=str,
        default="summary",
        choices=("summary", "json"),
        help="Whether to print a short summary or the full JSON payload to stdout.",
    )
    return parser.parse_args()


def _load_artifact(path: Path, *, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} artifact: {path}")
    return joblib.load(path)


def _slugify(raw: str) -> str:
    cleaned = [
        char.lower() if char.isalnum() else "-"
        for char in str(raw).strip()
    ]
    value = "".join(cleaned)
    while "--" in value:
        value = value.replace("--", "-")
    return value.strip("-") or "demo"


def _energy_alignment(space: MultimodalArtistSpace, artist_id: int, *, target: float) -> float:
    if artist_id < 0 or artist_id >= len(space.artist_labels):
        return 0.0
    return float(1.0 - abs(float(space.energy[artist_id]) - float(target)))


def _normalized_artist_counts(artist_ids: np.ndarray, candidate_count: int) -> np.ndarray:
    if candidate_count <= 0 or len(artist_ids) == 0:
        return np.zeros(candidate_count, dtype="float64")
    counts = np.bincount(np.asarray(artist_ids, dtype="int32"), minlength=candidate_count).astype("float64")
    return counts / max(1.0, float(len(artist_ids)))


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64").reshape(-1)
    if arr.size == 0:
        return arr.astype("float32")
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.size, dtype="float64")
    if arr.size == 1:
        ranks[0] = 1.0
        return ranks.astype("float32")
    ranks[order] = np.linspace(0.0, 1.0, num=arr.size, dtype="float64")
    return ranks.astype("float32")


def _target_alignment(ranks: np.ndarray, *, target: float) -> np.ndarray:
    return np.clip(1.0 - np.abs(np.asarray(ranks, dtype="float64") - float(target)), 0.0, 1.0).astype("float32")


def _candidate_metric_arrays(
    *,
    sequence_labels: np.ndarray,
    multimodal_space: MultimodalArtistSpace,
    digital_twin: ListenerDigitalTwinArtifact,
) -> dict[str, np.ndarray]:
    seq_arr = np.asarray(sequence_labels, dtype="int32")
    last_artist = int(seq_arr[-1])
    recent_similarity = np.asarray(
        multimodal_space.embeddings[seq_arr] @ multimodal_space.embeddings.T,
        dtype="float64",
    )
    continuity = np.asarray(multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings.T, dtype="float64")
    arc_affinity = np.mean(recent_similarity, axis=0) if recent_similarity.size else np.zeros(len(multimodal_space.artist_labels), dtype="float64")
    freshness = np.clip(1.0 - np.max(recent_similarity, axis=0), 0.0, 1.0) if recent_similarity.size else np.ones(len(multimodal_space.artist_labels), dtype="float64")
    transition = np.asarray(digital_twin.transition_matrix[last_artist], dtype="float64")
    return {
        "continuity": continuity.astype("float32"),
        "arc_affinity": np.asarray(arc_affinity, dtype="float32"),
        "freshness": np.asarray(freshness, dtype="float32"),
        "transition_support": transition.astype("float32"),
    }


def _surface_reranked_indices(
    *,
    probs: np.ndarray,
    adjusted_scores: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
    mode: ModeConfig,
    top_k: int,
) -> tuple[list[int], dict[int, float]]:
    candidate_count = len(adjusted_scores)
    if candidate_count == 0:
        return [], {}

    shortlist_size = min(
        candidate_count,
        max(int(mode.candidate_shortlist), int(top_k) * 4, 8),
    )
    adjusted_top = np.argsort(adjusted_scores)[::-1][:shortlist_size].tolist()
    prob_top = np.argsort(probs)[::-1][: max(6, shortlist_size // 2)].tolist()
    union_shortlist = list(dict.fromkeys(adjusted_top + prob_top))
    if len(union_shortlist) < max(1, int(top_k)):
        union_shortlist = np.argsort(adjusted_scores)[::-1][: max(1, int(top_k))].tolist()
    shortlist = np.asarray(union_shortlist, dtype="int32")

    prob_ranks = _percentile_ranks(np.asarray(probs, dtype="float64")[shortlist])
    transition_ranks = _percentile_ranks(np.asarray(metric_arrays["transition_support"], dtype="float64")[shortlist])
    continuity_alignment = _target_alignment(
        _percentile_ranks(np.asarray(metric_arrays["continuity"], dtype="float64")[shortlist]),
        target=mode.surface_continuity_target,
    )
    arc_alignment = _target_alignment(
        _percentile_ranks(np.asarray(metric_arrays["arc_affinity"], dtype="float64")[shortlist]),
        target=mode.surface_arc_target,
    )
    freshness_alignment = _target_alignment(
        _percentile_ranks(np.asarray(metric_arrays["freshness"], dtype="float64")[shortlist]),
        target=mode.surface_freshness_target,
    )

    surface = (
        float(mode.surface_probability_weight) * np.asarray(prob_ranks, dtype="float64")
        + float(mode.surface_transition_weight) * np.asarray(transition_ranks, dtype="float64")
        + float(mode.surface_continuity_weight) * np.asarray(continuity_alignment, dtype="float64")
        + float(mode.surface_arc_weight) * np.asarray(arc_alignment, dtype="float64")
        + float(mode.surface_freshness_weight) * np.asarray(freshness_alignment, dtype="float64")
    )
    surface_map = {int(idx): float(surface[pos]) for pos, idx in enumerate(shortlist.tolist())}
    ranked = sorted(
        shortlist.tolist(),
        key=lambda idx: (
            surface_map.get(int(idx), float("-inf")),
            float(adjusted_scores[int(idx)]),
            float(probs[int(idx)]),
        ),
        reverse=True,
    )
    return [int(item) for item in ranked[: max(1, int(top_k))]], surface_map


def _mode_scores(
    *,
    base_scores: np.ndarray,
    sequence_labels: np.ndarray,
    mode: ModeConfig,
    multimodal_space: MultimodalArtistSpace,
    planned_history: list[int] | None = None,
) -> np.ndarray:
    scores = np.asarray(base_scores, dtype="float64").reshape(-1)
    if scores.size == 0:
        return scores.astype("float32")

    last_artist = int(sequence_labels[-1])
    artist_ids = np.arange(scores.size, dtype="int32")
    similarity = np.asarray(multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings.T, dtype="float64")
    recent_similarity = np.asarray(
        multimodal_space.embeddings[np.asarray(sequence_labels, dtype="int32")] @ multimodal_space.embeddings.T,
        dtype="float64",
    )
    arc_affinity = np.mean(recent_similarity, axis=0) if recent_similarity.size else np.zeros(scores.size, dtype="float64")
    freshness = np.clip(1.0 - np.max(recent_similarity, axis=0), 0.0, 1.0) if recent_similarity.size else np.ones(scores.size, dtype="float64")
    novelty = np.asarray(1.0 - multimodal_space.popularity, dtype="float64")
    recent_counts = _normalized_artist_counts(np.asarray(sequence_labels, dtype="int32"), scores.size)
    if planned_history:
        recent_counts = recent_counts + (1.2 * _normalized_artist_counts(np.asarray(planned_history, dtype="int32"), scores.size))
    repeats = np.isin(artist_ids, np.asarray(sequence_labels, dtype="int32")).astype("float64")
    repeats = np.maximum(repeats, np.asarray(recent_counts > 0.0, dtype="float64"))
    same_as_last = np.asarray(artist_ids == last_artist, dtype="float64")
    energy_delta = np.abs(np.asarray(multimodal_space.energy, dtype="float64") - float(mode.energy_target))

    adjusted = (
        float(mode.model_weight) * np.log(np.clip(scores, 1e-9, 1.0))
        + float(mode.continuity_weight) * similarity
        + float(mode.arc_weight) * arc_affinity
        + float(mode.novelty_weight) * novelty
        + float(mode.freshness_weight) * freshness
        - float(mode.repeat_penalty) * repeats
        - float(mode.frequency_penalty) * recent_counts
        - 0.65 * same_as_last
        - float(mode.energy_weight) * energy_delta
    )
    return adjusted.astype("float32")


def _select_next_artist(
    *,
    ranked_artist_ids: list[int],
    sequence_labels: np.ndarray,
    mode: ModeConfig,
    planned_history: list[int] | None = None,
) -> int:
    if not ranked_artist_ids:
        return int(sequence_labels[-1])

    planned = list(planned_history or [])
    recent_window = np.asarray(sequence_labels[-max(1, int(mode.hard_repeat_window)) :], dtype="int32")
    last_selected = int(planned[-1]) if planned else None

    for artist_id in ranked_artist_ids:
        if int(artist_id) == int(sequence_labels[-1]):
            continue
        if last_selected is not None and int(artist_id) == last_selected:
            continue
        if np.any(recent_window == int(artist_id)):
            continue
        return int(artist_id)

    for artist_id in ranked_artist_ids:
        if last_selected is not None and int(artist_id) == last_selected:
            continue
        return int(artist_id)

    return int(ranked_artist_ids[0])


def _candidate_rows(
    *,
    probs: np.ndarray,
    sequence_labels: np.ndarray,
    artist_labels: list[str],
    mode: ModeConfig,
    multimodal_space: MultimodalArtistSpace,
    digital_twin: ListenerDigitalTwinArtifact,
    top_k: int,
    planned_history: list[int] | None = None,
) -> list[dict[str, object]]:
    adjusted = _mode_scores(
        base_scores=probs,
        sequence_labels=sequence_labels,
        mode=mode,
        multimodal_space=multimodal_space,
        planned_history=planned_history,
    )
    metric_arrays = _candidate_metric_arrays(
        sequence_labels=sequence_labels,
        multimodal_space=multimodal_space,
        digital_twin=digital_twin,
    )
    top_indices, surface_map = _surface_reranked_indices(
        probs=probs,
        adjusted_scores=adjusted,
        metric_arrays=metric_arrays,
        mode=mode,
        top_k=top_k,
    )
    last_artist = int(sequence_labels[-1])

    rows: list[dict[str, object]] = []
    for rank, artist_id in enumerate(top_indices, start=1):
        continuity = float(metric_arrays["continuity"][int(artist_id)])
        novelty = float(1.0 - multimodal_space.popularity[int(artist_id)])
        rows.append(
            {
                "rank": rank,
                "artist_label": int(artist_id),
                "artist_name": artist_labels[int(artist_id)],
                "model_probability": round(float(probs[int(artist_id)]), 4),
                "mode_score": round(float(adjusted[int(artist_id)]), 4),
                "surface_score": round(float(surface_map.get(int(artist_id), float(adjusted[int(artist_id)]))), 4),
                "continuity": round(continuity, 4),
                "arc_affinity": round(float(metric_arrays["arc_affinity"][int(artist_id)]), 4),
                "freshness": round(float(metric_arrays["freshness"][int(artist_id)]), 4),
                "transition_support": round(float(metric_arrays["transition_support"][int(artist_id)]), 4),
                "novelty": round(novelty, 4),
                "energy_alignment": round(_energy_alignment(multimodal_space, int(artist_id), target=mode.energy_target), 4),
            }
        )
    return rows


def _journey_plan_rows(
    *,
    sequence_labels: np.ndarray,
    artist_labels: list[str],
    mode: ModeConfig,
    multimodal_space: MultimodalArtistSpace,
    digital_twin: ListenerDigitalTwinArtifact,
) -> list[dict[str, object]]:
    working = np.asarray(sequence_labels, dtype="int32").copy()
    rows: list[dict[str, object]] = []
    planned_history: list[int] = []

    for step in range(1, int(mode.horizon) + 1):
        last_artist = int(working[-1])
        transition = np.asarray(digital_twin.transition_matrix[last_artist], dtype="float32")
        adjusted = _mode_scores(
            base_scores=transition,
            sequence_labels=working,
            mode=mode,
            multimodal_space=multimodal_space,
            planned_history=planned_history,
        )
        ranked_artist_ids = [int(item) for item in np.argsort(adjusted)[::-1].tolist()]
        next_artist = _select_next_artist(
            ranked_artist_ids=ranked_artist_ids,
            sequence_labels=working,
            mode=mode,
            planned_history=planned_history,
        )
        rows.append(
            {
                "step": step,
                "artist_label": next_artist,
                "artist_name": artist_labels[next_artist],
                "transition_probability": round(float(transition[next_artist]), 4),
                "mode_score": round(float(adjusted[next_artist]), 4),
                "continuity": round(float(multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings[next_artist]), 4),
                "novelty": round(float(1.0 - multimodal_space.popularity[next_artist]), 4),
                "energy_alignment": round(_energy_alignment(multimodal_space, next_artist, target=mode.energy_target), 4),
            }
        )
        planned_history.append(next_artist)
        working = np.roll(working, -1)
        working[-1] = next_artist
    return rows


def _friction_feature_indices(context_features: list[str]) -> list[int]:
    return [
        idx
        for idx, feature_name in enumerate(context_features)
        if (
            str(feature_name).lower() == "offline"
            or (
                not any(
                    token in str(feature_name).lower()
                    for token in ("bitrate", "reachability", "allow_downgrade", "cloud_stats_events")
                )
                and any(
                    token in str(feature_name).lower()
                    for token in (
                        "error",
                        "fatal",
                        "stutter",
                        "stall",
                        "not_played",
                        "fail",
                        "connection_none",
                        "offline",
                    )
                )
            )
        )
    ]


def _friction_profile(
    *,
    context_raw_batch: np.ndarray | None,
    context_features: list[str],
    friction_reference: dict[str, object] | None,
) -> dict[str, object]:
    if context_raw_batch is None or len(context_features) == 0:
        return {
            "friction_score": 0.0,
            "friction_score_raw": 0.0,
            "friction_threshold": 0.0,
            "friction_bucket": "unknown",
        }

    friction_indices = _friction_feature_indices(context_features)
    if not friction_indices:
        return {
            "friction_score": 0.0,
            "friction_score_raw": 0.0,
            "friction_threshold": 0.0,
            "friction_bucket": "stable",
        }

    friction_values = np.asarray(context_raw_batch, dtype="float32").reshape(1, -1)[0, friction_indices]
    friction_medians = np.zeros(len(friction_indices), dtype="float32")
    friction_threshold = 0.0
    if isinstance(friction_reference, dict):
        ref_names = [str(item) for item in friction_reference.get("feature_names", [])]
        ref_medians = np.asarray(friction_reference.get("median_values", []), dtype="float32")
        ref_map = {
            ref_name: float(ref_medians[idx])
            for idx, ref_name in enumerate(ref_names)
            if idx < len(ref_medians)
        }
        friction_medians = np.asarray(
            [ref_map.get(str(context_features[idx]), 0.0) for idx in friction_indices],
            dtype="float32",
        )
        friction_threshold = float(friction_reference.get("aggregate_threshold", 0.0) or 0.0)

    centered = np.maximum(friction_values - friction_medians, 0.0)
    raw_score = float(np.sum(centered))
    normalized_score = raw_score / max(friction_threshold, 1e-6) if friction_threshold > 0 else raw_score

    if normalized_score >= 2.0:
        bucket = "high_friction"
    elif normalized_score >= 1.2:
        bucket = "normal_friction"
    else:
        bucket = "stable"

    return {
        "friction_score": round(normalized_score, 4),
        "friction_score_raw": round(raw_score, 4),
        "friction_threshold": round(friction_threshold, 4),
        "friction_bucket": bucket,
    }


def _risk_summary(
    *,
    sequence_labels: np.ndarray,
    context_batch: np.ndarray,
    context_raw_batch: np.ndarray | None,
    context_features: list[str],
    friction_reference: dict[str, object] | None,
    digital_twin: ListenerDigitalTwinArtifact,
) -> dict[str, object]:
    features = build_serving_tabular_features(
        np.asarray(sequence_labels, dtype="int32").reshape(1, -1),
        np.asarray(context_batch, dtype="float32"),
    )
    end_risk = float(np.asarray(digital_twin.end_estimator.predict_proba(features), dtype="float32")[:, 1][0])

    friction_profile = _friction_profile(
        context_raw_batch=context_raw_batch,
        context_features=context_features,
        friction_reference=friction_reference,
    )
    friction_score = float(friction_profile["friction_score"])

    if end_risk >= 0.45 or friction_score >= 2.0:
        risk_state = "guarded"
    elif end_risk >= 0.25 or friction_score >= 1.2:
        risk_state = "watch"
    else:
        risk_state = "normal"

    return {
        "current_end_risk": round(end_risk, 4),
        "friction_score": round(friction_score, 4),
        "friction_score_raw": float(friction_profile["friction_score_raw"]),
        "friction_threshold": float(friction_profile["friction_threshold"]),
        "friction_bucket": str(friction_profile["friction_bucket"]),
        "risk_state": risk_state,
    }


def _fallback_policy(
    *,
    mode: ModeConfig,
    risk_summary: dict[str, object],
    safe_policy: SafeBanditPolicyArtifact,
) -> dict[str, object]:
    end_risk = float(risk_summary.get("current_end_risk", 0.0))
    friction_score = float(risk_summary.get("friction_score", 0.0))
    friction_bucket = str(risk_summary.get("friction_bucket", ""))

    if end_risk >= float(mode.end_guard_threshold):
        return {
            "active_policy_name": "safe_global",
            "reason": "Session-end risk is elevated, so the demo routes to the global safe policy.",
            "policy_weights": dict(safe_policy.global_policy),
            "safe_routed": True,
        }

    if (
        friction_bucket == "high_friction"
        and friction_score >= float(mode.friction_high_threshold)
        and "high_friction" in safe_policy.policy_map
    ):
        return {
            "active_policy_name": "safe_bucket_high_friction",
            "reason": "Current context looks friction-heavy, so the demo routes to the high-friction safe bucket.",
            "policy_weights": dict(safe_policy.policy_map["high_friction"]),
            "safe_routed": True,
        }

    if (
        friction_bucket == "normal_friction"
        and friction_bucket in safe_policy.policy_map
        and friction_score >= float(mode.friction_guard_threshold)
    ):
        return {
            "active_policy_name": f"safe_bucket_{friction_bucket}",
            "reason": "The session is showing some technical friction, so the demo moves to a safer bucket before it escalates.",
            "policy_weights": dict(safe_policy.policy_map[friction_bucket]),
            "safe_routed": True,
        }

    default_policy = dict(POLICY_TEMPLATES.get(mode.default_policy_name, safe_policy.global_policy))
    return {
        "active_policy_name": mode.default_policy_name,
        "reason": "The current session is stable enough to stay on the mode's default policy.",
        "policy_weights": default_policy,
        "safe_routed": False,
    }


def _why_this_next(
    *,
    first_candidate: dict[str, object] | None,
    mode: ModeConfig,
    policy_name: str,
) -> list[str]:
    if not first_candidate:
        return []

    reasons = [
        f"This choice fits the {mode.name} profile: {mode.description}",
    ]
    probability = float(first_candidate.get("model_probability", first_candidate.get("transition_probability", 0.0)))
    continuity = float(first_candidate.get("continuity", 0.0))
    novelty = float(first_candidate.get("novelty", 0.0))
    freshness = float(first_candidate.get("freshness", 0.0))
    energy_alignment = float(first_candidate.get("energy_alignment", 0.0))

    if probability >= 0.20:
        reasons.append("The current model state already scores it strongly for this session tail.")
    if continuity >= 0.65:
        reasons.append("It stays close to the recent listening arc instead of making a hard jump.")
    if freshness >= 0.60 and mode.name in ("discovery", "workout"):
        reasons.append("It opens a fresher lane than the recent loop while staying inside the candidate shortlist.")
    if novelty >= 0.45 and mode.name in ("discovery", "workout"):
        reasons.append("It adds novelty without leaving the learned taste boundary.")
    if energy_alignment >= 0.70:
        reasons.append("Its energy profile stays near the target band for this mode.")
    if str(policy_name).startswith("safe"):
        reasons.append("The planner is guarding the session, so it is biasing toward a safer fallback path.")
    return reasons[:5]


def _rescale_context_from_raw(
    *,
    context_raw_batch: np.ndarray | None,
    context_batch: np.ndarray,
    scaler_mean: np.ndarray | None,
    scaler_scale: np.ndarray | None,
) -> np.ndarray:
    if context_raw_batch is None:
        return np.asarray(context_batch, dtype="float32").reshape(1, -1).copy()
    if scaler_mean is None or scaler_scale is None:
        return np.asarray(context_batch, dtype="float32").reshape(1, -1).copy()
    mean = np.asarray(scaler_mean, dtype="float32").reshape(-1)
    scale = np.asarray(scaler_scale, dtype="float32").reshape(-1)
    safe_scale = np.where(np.abs(scale) <= 1e-6, 1.0, scale)
    raw = np.asarray(context_raw_batch, dtype="float32").reshape(1, -1)
    if raw.shape[1] != mean.shape[0]:
        return np.asarray(context_batch, dtype="float32").reshape(1, -1).copy()
    return ((raw - mean.reshape(1, -1)) / safe_scale.reshape(1, -1)).astype("float32")


def _apply_event(
    *,
    mode: ModeConfig,
    event: AdaptiveEvent,
    context_batch: np.ndarray,
    context_raw_batch: np.ndarray | None,
    context_features: list[str],
    scaler_mean: np.ndarray | None,
    scaler_scale: np.ndarray | None,
) -> tuple[ModeConfig, np.ndarray, np.ndarray | None, dict[str, object]]:
    updated_mode = mode
    updated_context = np.asarray(context_batch, dtype="float32").reshape(1, -1).copy()
    updated_raw_context = None if context_raw_batch is None else np.asarray(context_raw_batch, dtype="float32").reshape(1, -1).copy()
    planner_change = ""

    if event.event_type == "skip":
        updated_mode = replace(
            mode,
            continuity_weight=min(0.65, float(mode.continuity_weight) + 0.12),
            novelty_weight=max(0.02, float(mode.novelty_weight) - 0.10),
            repeat_penalty=min(1.25, float(mode.repeat_penalty) + 0.10),
            default_policy_name="safe_balance",
        )
        planner_change = "The listener skipped a suggestion, so the planner reduced novelty and moved closer to the recent arc."
    elif event.event_type == "repeat_request":
        updated_mode = replace(
            mode,
            continuity_weight=min(0.65, float(mode.continuity_weight) + 0.08),
            novelty_weight=max(0.02, float(mode.novelty_weight) - 0.08),
            repeat_penalty=max(0.35, float(mode.repeat_penalty) - 0.20),
        )
        planner_change = "The listener repeated a track, so the planner leaned into comfort, continuity, and lower surprise."
    elif event.event_type == "friction_spike":
        friction_indices = _friction_feature_indices(list(context_features))
        for idx in friction_indices:
            if updated_raw_context is not None and idx < updated_raw_context.shape[1]:
                boost = max(0.75, abs(float(updated_raw_context[0, idx])) * 1.25)
                updated_raw_context[0, idx] = float(updated_raw_context[0, idx] + boost)
            else:
                updated_context[0, idx] = float(updated_context[0, idx] + 1.25)
        if updated_raw_context is not None:
            updated_context = _rescale_context_from_raw(
                context_raw_batch=updated_raw_context,
                context_batch=updated_context,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
            )
        updated_mode = replace(
            mode,
            continuity_weight=min(0.70, float(mode.continuity_weight) + 0.14),
            novelty_weight=max(0.02, float(mode.novelty_weight) - 0.14),
            default_policy_name="safe_balance",
        )
        planner_change = "Playback friction spiked, so the planner reduced novelty and prepared a safer route."
    else:
        planner_change = "The planner observed a session event and recalculated the next step."

    return updated_mode, updated_context, updated_raw_context, {
        "after_step": int(event.after_step),
        "event_type": event.event_type,
        "description": event.description,
        "planner_change": planner_change,
    }


def _roll_sequence(sequence_labels: np.ndarray, *, next_artist: int) -> np.ndarray:
    updated = np.asarray(sequence_labels, dtype="int32").copy()
    updated = np.roll(updated, -1)
    updated[-1] = int(next_artist)
    return updated


def _adaptive_session_payload(
    *,
    predictor: _PredictorLike,
    artist_labels: list[str],
    sequence_labels: np.ndarray,
    context_batch: np.ndarray,
    context_raw_batch: np.ndarray | None,
    context_features: list[str],
    friction_reference: dict[str, object] | None,
    scaler_mean: np.ndarray | None,
    scaler_scale: np.ndarray | None,
    mode: ModeConfig,
    scenario: AdaptiveScenario,
    top_k: int,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    safe_policy: SafeBanditPolicyArtifact,
) -> dict[str, object]:
    working_sequence = np.asarray(sequence_labels, dtype="int32").reshape(-1).copy()
    working_context = np.asarray(context_batch, dtype="float32").reshape(1, -1).copy()
    working_raw_context = None if context_raw_batch is None else np.asarray(context_raw_batch, dtype="float32").reshape(1, -1).copy()
    current_mode = mode
    transcript: list[dict[str, object]] = []
    applied_events: list[dict[str, object]] = []
    event_by_step = {int(event.after_step): event for event in scenario.events}
    pending_replan_reason = ""
    safe_route_steps = 0
    planned_history: list[int] = []

    for step in range(1, int(mode.horizon) + 1):
        probs = np.asarray(predictor.predict_proba(working_sequence.reshape(1, -1), working_context), dtype="float32")[0]
        candidates = _candidate_rows(
            probs=probs,
            sequence_labels=working_sequence,
            artist_labels=artist_labels,
            mode=current_mode,
            multimodal_space=multimodal_space,
            digital_twin=digital_twin,
            top_k=top_k,
            planned_history=planned_history,
        )
        candidate_ids = [int(row.get("artist_label", -1)) for row in candidates if isinstance(row, dict)]
        chosen_artist = _select_next_artist(
            ranked_artist_ids=[artist_id for artist_id in candidate_ids if artist_id >= 0],
            sequence_labels=working_sequence,
            mode=current_mode,
            planned_history=planned_history,
        ) if candidates else int(working_sequence[-1])
        chosen = next((row for row in candidates if int(row.get("artist_label", -1)) == chosen_artist), candidates[0] if candidates else {})
        risk_summary = _risk_summary(
            sequence_labels=working_sequence,
            context_batch=working_context,
            context_raw_batch=working_raw_context,
            context_features=context_features,
            friction_reference=friction_reference,
            digital_twin=digital_twin,
        )
        fallback_policy = _fallback_policy(
            mode=current_mode,
            risk_summary=risk_summary,
            safe_policy=safe_policy,
        )
        if bool(fallback_policy.get("safe_routed")):
            safe_route_steps += 1

        transcript_row = {
            "step": step,
            "plan_origin": "replanned" if pending_replan_reason else "initial",
            "why_changed": pending_replan_reason,
            "policy_name": str(fallback_policy.get("active_policy_name", "")),
            "selected_artist": str(chosen.get("artist_name", "")),
            "model_probability": float(chosen.get("model_probability", 0.0)),
            "continuity": float(chosen.get("continuity", 0.0)),
            "novelty": float(chosen.get("novelty", 0.0)),
            "energy_alignment": float(chosen.get("energy_alignment", 0.0)),
            "end_risk": float(risk_summary.get("current_end_risk", 0.0)),
            "friction_score": float(risk_summary.get("friction_score", 0.0)),
            "event_applied_after_step": "",
            "event_summary": "",
        }
        pending_replan_reason = ""

        if chosen:
            planned_history.append(int(chosen["artist_label"]))
            working_sequence = _roll_sequence(working_sequence, next_artist=int(chosen["artist_label"]))

        if step in event_by_step:
            current_mode, working_context, working_raw_context, event_row = _apply_event(
                mode=current_mode,
                event=event_by_step[step],
                context_batch=working_context,
                context_raw_batch=working_raw_context,
                context_features=context_features,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
            )
            transcript_row["event_applied_after_step"] = str(event_row["event_type"])
            transcript_row["event_summary"] = str(event_row["description"])
            pending_replan_reason = str(event_row["planner_change"])
            applied_events.append(event_row)

        transcript.append(transcript_row)

    final_tail = [artist_labels[int(item)] for item in working_sequence.tolist()]
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "events": applied_events,
        "replan_count": int(len(applied_events)),
        "safe_route_steps": int(safe_route_steps),
        "transcript": transcript,
        "final_sequence_tail": final_tail,
    }


def build_taste_os_demo_payload(
    *,
    predictor: _PredictorLike,
    artist_labels: list[str],
    sequence_labels: np.ndarray,
    sequence_names: list[str],
    context_batch: np.ndarray,
    context_raw_batch: np.ndarray | None = None,
    context_features: list[str] | None = None,
    friction_reference: dict[str, object] | None = None,
    scaler_mean: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    safe_policy: SafeBanditPolicyArtifact,
    mode_name: str,
    scenario_name: str = "steady",
    top_k: int,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    mode = MODE_CONFIGS[str(mode_name).strip().lower()]
    scenario = SCENARIOS[str(scenario_name).strip().lower()]
    seq_arr = np.asarray(sequence_labels, dtype="int32").reshape(-1)
    ctx_arr = np.asarray(context_batch, dtype="float32").reshape(1, -1)
    raw_ctx_source = context_batch if context_raw_batch is None else context_raw_batch
    raw_ctx_arr = np.asarray(raw_ctx_source, dtype="float32").reshape(1, -1)
    resolved_context_features = list(context_features or list(digital_twin.context_features))
    probs = np.asarray(predictor.predict_proba(seq_arr.reshape(1, -1), ctx_arr), dtype="float32")[0]

    top_candidates = _candidate_rows(
        probs=probs,
        sequence_labels=seq_arr,
        artist_labels=artist_labels,
        mode=mode,
        multimodal_space=multimodal_space,
        digital_twin=digital_twin,
        top_k=top_k,
    )
    journey_plan = _journey_plan_rows(
        sequence_labels=seq_arr,
        artist_labels=artist_labels,
        mode=mode,
        multimodal_space=multimodal_space,
        digital_twin=digital_twin,
    )
    risk_summary = _risk_summary(
        sequence_labels=seq_arr,
        context_batch=ctx_arr,
        context_raw_batch=raw_ctx_arr,
        context_features=resolved_context_features,
        friction_reference=friction_reference,
        digital_twin=digital_twin,
    )
    fallback_policy = _fallback_policy(
        mode=mode,
        risk_summary=risk_summary,
        safe_policy=safe_policy,
    )
    adaptive_session = _adaptive_session_payload(
        predictor=predictor,
        artist_labels=artist_labels,
        sequence_labels=seq_arr,
        context_batch=ctx_arr,
        context_raw_batch=raw_ctx_arr,
        context_features=resolved_context_features,
        friction_reference=friction_reference,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        mode=mode,
        scenario=scenario,
        top_k=top_k,
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        safe_policy=safe_policy,
    )

    top_choice = top_candidates[0] if top_candidates else {}
    return {
        "request": {
            "mode": mode.name,
            "scenario": scenario.name,
            "top_k": int(top_k),
        },
        "current_session": {
            "model_name": predictor.model_name,
            "model_type": predictor.model_type,
            "sequence_length": int(len(seq_arr)),
            "sequence_tail": list(sequence_names),
        },
        "mode": {
            "name": mode.name,
            "description": mode.description,
            "planned_horizon": int(mode.horizon),
            "default_policy_name": mode.default_policy_name,
        },
        "top_candidates": top_candidates,
        "journey_plan": journey_plan,
        "why_this_next": _why_this_next(
            first_candidate=top_choice,
            mode=mode,
            policy_name=str(fallback_policy.get("active_policy_name", "")),
        ),
        "risk_summary": risk_summary,
        "fallback_policy": fallback_policy,
        "adaptive_session": adaptive_session,
        "demo_summary": {
            "top_artist": str(top_choice.get("artist_name", "")),
            "adaptive_replans": int(adaptive_session["replan_count"]),
            "adaptive_safe_route_steps": int(adaptive_session["safe_route_steps"]),
            "final_sequence_tail": list(adaptive_session["final_sequence_tail"]),
        },
        "artifacts_used": artifact_paths or {},
    }


def write_taste_os_demo_artifacts(
    payload: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    request = payload.get("request", {}) if isinstance(payload, dict) else {}
    mode = _slugify(str(request.get("mode", "demo")))
    scenario = _slugify(str(request.get("scenario", "steady")))
    stem = f"taste_os_demo_{mode}_{scenario}"

    json_path = output_root / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    current_session = payload.get("current_session", {}) if isinstance(payload, dict) else {}
    mode_block = payload.get("mode", {}) if isinstance(payload, dict) else {}
    risk_summary = payload.get("risk_summary", {}) if isinstance(payload, dict) else {}
    fallback_policy = payload.get("fallback_policy", {}) if isinstance(payload, dict) else {}
    adaptive_session = payload.get("adaptive_session", {}) if isinstance(payload, dict) else {}

    lines = [
        "# Taste OS Demo",
        "",
        f"- Mode: `{request.get('mode', '')}`",
        f"- Scenario: `{request.get('scenario', '')}`",
        f"- Model: `{current_session.get('model_name', '')}` [{current_session.get('model_type', '')}]",
        f"- Sequence tail: `{ ' | '.join(current_session.get('sequence_tail', [])) if isinstance(current_session.get('sequence_tail', []), list) else '' }`",
        f"- Planned horizon: `{mode_block.get('planned_horizon', '')}`",
        "",
        "## Why This Next",
        "",
    ]
    for reason in payload.get("why_this_next", []) if isinstance(payload, dict) else []:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
        ]
    )
    for row in payload.get("top_candidates", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('rank', '')}. {row.get('artist_name', '')}` model_prob=`{row.get('model_probability', '')}` "
            f"surface_score=`{row.get('surface_score', '')}` mode_score=`{row.get('mode_score', '')}` "
            f"continuity=`{row.get('continuity', '')}` freshness=`{row.get('freshness', '')}` novelty=`{row.get('novelty', '')}`"
        )

    lines.extend(
        [
            "",
            "## Baseline Plan",
            "",
        ]
    )
    for row in payload.get("journey_plan", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- Step `{row.get('step', '')}` -> `{row.get('artist_name', '')}` transition=`{row.get('transition_probability', '')}` mode_score=`{row.get('mode_score', '')}`"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- End risk: `{risk_summary.get('current_end_risk', '')}`",
            f"- Friction score: `{risk_summary.get('friction_score', '')}`",
            f"- Friction raw / threshold: `{risk_summary.get('friction_score_raw', '')}` / `{risk_summary.get('friction_threshold', '')}`",
            f"- Friction bucket: `{risk_summary.get('friction_bucket', '')}`",
            f"- Risk state: `{risk_summary.get('risk_state', '')}`",
            f"- Fallback policy: `{fallback_policy.get('active_policy_name', '')}`",
            f"- Fallback reason: {fallback_policy.get('reason', '')}",
            "",
            "## Adaptive Session",
            "",
            f"- Scenario summary: {adaptive_session.get('description', '')}",
            f"- Replans: `{adaptive_session.get('replan_count', '')}`",
            f"- Safe-route steps: `{adaptive_session.get('safe_route_steps', '')}`",
            "",
            "### Transcript",
            "",
        ]
    )
    for row in adaptive_session.get("transcript", []) if isinstance(adaptive_session, dict) else []:
        if not isinstance(row, dict):
            continue
        line = (
            f"- Step `{row.get('step', '')}` [{row.get('plan_origin', '')}] -> `{row.get('selected_artist', '')}` "
            f"via `{row.get('policy_name', '')}` end_risk=`{row.get('end_risk', '')}`"
        )
        lines.append(line)
        why_changed = str(row.get("why_changed", "") or "").strip()
        if why_changed:
            lines.append(f"  Why changed: {why_changed}")
        event_applied = str(row.get("event_applied_after_step", "") or "").strip()
        if event_applied:
            lines.append(f"  Event after step: `{event_applied}` - {row.get('event_summary', '')}")

    md_path = output_root / f"{stem}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.taste_os_demo")

    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    prediction_context = load_prediction_input_context(
        run_dir=run_dir,
        data_dir=Path(args.data_dir),
        include_video=bool(args.include_video),
        logger=logger,
    )

    recent_artists = None
    if args.recent_artists:
        recent_artists = [part.strip() for part in args.recent_artists.split("|") if part.strip()]

    seq_batch, ctx_batch, sequence_names = _prepare_inputs(
        run_dir=run_dir,
        data_dir=Path(args.data_dir),
        recent_artists=recent_artists,
        include_video=bool(args.include_video),
        logger=logger,
        context=prediction_context,
    )

    predictor = load_predictor(
        run_dir=run_dir,
        row=model_row,
        artist_labels=list(prediction_context.artist_labels),
    )

    artifact_paths = {
        "multimodal_space": str((run_dir / "analysis" / "multimodal" / "multimodal_artist_space.joblib").resolve()),
        "digital_twin": str((run_dir / "analysis" / "digital_twin" / "listener_digital_twin.joblib").resolve()),
        "safe_policy": str((run_dir / "analysis" / "safe_policy" / "safe_bandit_policy.joblib").resolve()),
    }
    multimodal_space = _load_artifact(Path(artifact_paths["multimodal_space"]), label="multimodal artist space")
    digital_twin = _load_artifact(Path(artifact_paths["digital_twin"]), label="listener digital twin")
    safe_policy = _load_artifact(Path(artifact_paths["safe_policy"]), label="safe policy")

    payload = build_taste_os_demo_payload(
        predictor=predictor,
        artist_labels=list(prediction_context.artist_labels),
        sequence_labels=np.asarray(seq_batch[0], dtype="int32"),
        sequence_names=sequence_names,
        context_batch=np.asarray(ctx_batch, dtype="float32"),
        context_raw_batch=prediction_context.context_raw,
        context_features=list(prediction_context.context_features or []),
        friction_reference=prediction_context.friction_reference,
        scaler_mean=prediction_context.scaler_mean,
        scaler_scale=prediction_context.scaler_scale,
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        safe_policy=safe_policy,
        mode_name=str(args.mode),
        scenario_name=str(args.scenario),
        top_k=max(1, int(args.top_k)),
        artifact_paths=artifact_paths,
    )
    json_path, md_path = write_taste_os_demo_artifacts(
        payload,
        output_dir=Path(args.output_dir),
    )

    if args.stdout_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"taste_os_demo_json={json_path}")
        print(f"taste_os_demo_md={md_path}")
        print(f"mode={payload['mode']['name']}")
        print(f"scenario={payload['adaptive_session']['scenario']}")
        print(f"top_artist={payload['demo_summary']['top_artist']}")
        print(f"adaptive_replans={payload['demo_summary']['adaptive_replans']}")
        print(f"adaptive_safe_route_steps={payload['demo_summary']['adaptive_safe_route_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
