from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    continuity_weight: float
    novelty_weight: float
    repeat_penalty: float
    energy_target: float
    energy_weight: float
    default_policy_name: str


MODE_CONFIGS: dict[str, ModeConfig] = {
    "focus": ModeConfig(
        name="focus",
        description="Low-friction, low-surprise arcs for concentrated listening sessions.",
        horizon=6,
        continuity_weight=0.35,
        novelty_weight=0.08,
        repeat_penalty=0.95,
        energy_target=0.48,
        energy_weight=0.12,
        default_policy_name="comfort_policy",
    ),
    "workout": ModeConfig(
        name="workout",
        description="Rising-energy plans with momentum and controlled novelty.",
        horizon=6,
        continuity_weight=0.18,
        novelty_weight=0.22,
        repeat_penalty=0.70,
        energy_target=0.78,
        energy_weight=0.16,
        default_policy_name="novelty_boosted",
    ),
    "commute": ModeConfig(
        name="commute",
        description="Shorter, resilient plans that recover quickly from disruption.",
        horizon=4,
        continuity_weight=0.28,
        novelty_weight=0.10,
        repeat_penalty=0.80,
        energy_target=0.56,
        energy_weight=0.10,
        default_policy_name="safe_balance",
    ),
    "discovery": ModeConfig(
        name="discovery",
        description="Novelty-weighted plans that stay inside learned taste boundaries.",
        horizon=6,
        continuity_weight=0.16,
        novelty_weight=0.34,
        repeat_penalty=0.72,
        energy_target=0.62,
        energy_weight=0.08,
        default_policy_name="novelty_boosted",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.taste_os_demo",
        description="Run a thin Personal Taste OS demo using the current serving and moonshot artifacts.",
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
    return parser.parse_args()


def _load_artifact(path: Path, *, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} artifact: {path}")
    return joblib.load(path)


def _energy_alignment(space: MultimodalArtistSpace, artist_id: int, *, target: float) -> float:
    if artist_id < 0 or artist_id >= len(space.artist_labels):
        return 0.0
    return float(1.0 - abs(float(space.energy[artist_id]) - float(target)))


def _mode_scores(
    *,
    base_scores: np.ndarray,
    sequence_labels: np.ndarray,
    mode: ModeConfig,
    multimodal_space: MultimodalArtistSpace,
) -> np.ndarray:
    scores = np.asarray(base_scores, dtype="float64").reshape(-1)
    if scores.size == 0:
        return scores.astype("float32")

    last_artist = int(sequence_labels[-1])
    artist_ids = np.arange(scores.size, dtype="int32")
    similarity = np.asarray(multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings.T, dtype="float64")
    novelty = np.asarray(1.0 - multimodal_space.popularity, dtype="float64")
    repeats = np.isin(artist_ids, np.asarray(sequence_labels, dtype="int32")).astype("float64")
    energy_delta = np.abs(np.asarray(multimodal_space.energy, dtype="float64") - float(mode.energy_target))

    adjusted = (
        np.log(np.clip(scores, 1e-9, 1.0))
        + float(mode.continuity_weight) * similarity
        + float(mode.novelty_weight) * novelty
        - float(mode.repeat_penalty) * repeats
        - float(mode.energy_weight) * energy_delta
    )
    return adjusted.astype("float32")


def _candidate_rows(
    *,
    probs: np.ndarray,
    sequence_labels: np.ndarray,
    artist_labels: list[str],
    mode: ModeConfig,
    multimodal_space: MultimodalArtistSpace,
    top_k: int,
) -> list[dict[str, object]]:
    adjusted = _mode_scores(
        base_scores=probs,
        sequence_labels=sequence_labels,
        mode=mode,
        multimodal_space=multimodal_space,
    )
    top_indices = np.argsort(adjusted)[::-1][: max(1, int(top_k))]
    last_artist = int(sequence_labels[-1])

    rows: list[dict[str, object]] = []
    for rank, artist_id in enumerate(top_indices.tolist(), start=1):
        continuity = float(multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings[int(artist_id)])
        novelty = float(1.0 - multimodal_space.popularity[int(artist_id)])
        rows.append(
            {
                "rank": rank,
                "artist_label": int(artist_id),
                "artist_name": artist_labels[int(artist_id)],
                "model_probability": round(float(probs[int(artist_id)]), 4),
                "mode_score": round(float(adjusted[int(artist_id)]), 4),
                "continuity": round(continuity, 4),
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

    for step in range(1, int(mode.horizon) + 1):
        last_artist = int(working[-1])
        transition = np.asarray(digital_twin.transition_matrix[last_artist], dtype="float32")
        adjusted = _mode_scores(
            base_scores=transition,
            sequence_labels=working,
            mode=mode,
            multimodal_space=multimodal_space,
        )
        next_artist = int(np.argsort(adjusted)[::-1][0])
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
        working = np.roll(working, -1)
        working[-1] = next_artist
    return rows


def _why_this_next(
    *,
    first_candidate: dict[str, object] | None,
    mode: ModeConfig,
) -> list[str]:
    if not first_candidate:
        return []

    reasons = [
        f"This choice fits the {mode.name} profile: {mode.description}",
    ]
    probability = float(first_candidate.get("model_probability", 0.0))
    continuity = float(first_candidate.get("continuity", 0.0))
    novelty = float(first_candidate.get("novelty", 0.0))
    energy_alignment = float(first_candidate.get("energy_alignment", 0.0))

    if probability >= 0.20:
        reasons.append("The serving model already scores it strongly for the current session tail.")
    if continuity >= 0.65:
        reasons.append("It stays close to your recent listening arc instead of making a hard jump.")
    if novelty >= 0.45 and mode.name in ("discovery", "workout"):
        reasons.append("It adds novelty without leaving the learned taste boundary.")
    if energy_alignment >= 0.70:
        reasons.append("Its energy profile stays near the target band for this mode.")
    return reasons[:4]


def _friction_feature_indices(context_features: list[str]) -> list[int]:
    return [
        idx
        for idx, feature_name in enumerate(context_features)
        if str(feature_name).startswith("tech_")
        or "error" in str(feature_name)
        or str(feature_name) == "offline"
    ]


def _risk_summary(
    *,
    sequence_labels: np.ndarray,
    context_batch: np.ndarray,
    digital_twin: ListenerDigitalTwinArtifact,
) -> dict[str, object]:
    features = build_serving_tabular_features(
        np.asarray(sequence_labels, dtype="int32").reshape(1, -1),
        np.asarray(context_batch, dtype="float32"),
    )
    end_risk = float(np.asarray(digital_twin.end_estimator.predict_proba(features), dtype="float32")[:, 1][0])

    friction_indices = _friction_feature_indices(list(digital_twin.context_features))
    if friction_indices:
        friction_score = float(np.mean(np.maximum(np.asarray(context_batch, dtype="float32")[0, friction_indices], 0.0)))
    else:
        friction_score = 0.0

    if end_risk >= 0.45:
        risk_state = "guarded"
    elif end_risk >= 0.25:
        risk_state = "watch"
    else:
        risk_state = "normal"

    return {
        "current_end_risk": round(end_risk, 4),
        "friction_score": round(friction_score, 4),
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

    if end_risk >= 0.45:
        return {
            "active_policy_name": "safe_global",
            "reason": "Session-end risk is elevated, so the demo routes to the global safe policy.",
            "policy_weights": dict(safe_policy.global_policy),
        }

    if friction_score > 0.25 and "high_friction" in safe_policy.policy_map:
        return {
            "active_policy_name": "safe_bucket_high_friction",
            "reason": "Current context looks friction-heavy, so the demo routes to the high-friction safe bucket.",
            "policy_weights": dict(safe_policy.policy_map["high_friction"]),
        }

    default_policy = dict(POLICY_TEMPLATES.get(mode.default_policy_name, safe_policy.global_policy))
    return {
        "active_policy_name": mode.default_policy_name,
        "reason": "The current session is stable enough to stay on the mode's default policy.",
        "policy_weights": default_policy,
    }


def build_taste_os_demo_payload(
    *,
    predictor: _PredictorLike,
    artist_labels: list[str],
    sequence_labels: np.ndarray,
    sequence_names: list[str],
    context_batch: np.ndarray,
    digital_twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    safe_policy: SafeBanditPolicyArtifact,
    mode_name: str,
    top_k: int,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    mode = MODE_CONFIGS[str(mode_name).strip().lower()]
    seq_arr = np.asarray(sequence_labels, dtype="int32").reshape(-1)
    ctx_arr = np.asarray(context_batch, dtype="float32").reshape(1, -1)
    probs = np.asarray(predictor.predict_proba(seq_arr.reshape(1, -1), ctx_arr), dtype="float32")[0]

    top_candidates = _candidate_rows(
        probs=probs,
        sequence_labels=seq_arr,
        artist_labels=artist_labels,
        mode=mode,
        multimodal_space=multimodal_space,
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
        digital_twin=digital_twin,
    )
    fallback_policy = _fallback_policy(
        mode=mode,
        risk_summary=risk_summary,
        safe_policy=safe_policy,
    )

    return {
        "request": {
            "mode": mode.name,
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
            first_candidate=journey_plan[0] if journey_plan else (top_candidates[0] if top_candidates else None),
            mode=mode,
        ),
        "risk_summary": risk_summary,
        "fallback_policy": fallback_policy,
        "artifacts_used": artifact_paths or {},
    }


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
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        safe_policy=safe_policy,
        mode_name=str(args.mode),
        top_k=max(1, int(args.top_k)),
        artifact_paths=artifact_paths,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
