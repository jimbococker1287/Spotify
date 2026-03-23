from __future__ import annotations

from pathlib import Path
import json
import math

from .causal_friction import fit_causal_skip_decomposition
from .digital_twin import fit_listener_digital_twin
from .journey_planner import build_journey_plans
from .multimodal import build_multimodal_artist_space
from .safe_policy import learn_safe_bandit_policy
from .stress_test import run_stress_test_lab


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _safe_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def run_moonshot_lab(
    *,
    data,
    results: list[dict[str, object]],
    run_dir: Path,
    sequence_length: int,
    artist_labels: list[str],
    random_seed: int,
    logger,
) -> list[Path]:
    artifact_paths: list[Path] = []
    analysis_dir = run_dir / "analysis"
    multimodal_dir = analysis_dir / "multimodal"
    causal_dir = analysis_dir / "causal"
    digital_twin_dir = analysis_dir / "digital_twin"
    journey_dir = analysis_dir / "journey_planner"
    safe_policy_dir = analysis_dir / "safe_policy"
    stress_dir = analysis_dir / "stress_test"

    multimodal_space, multimodal_paths = build_multimodal_artist_space(
        df=data.df,
        artist_labels=artist_labels,
        results=results,
        output_dir=multimodal_dir,
        logger=logger,
    )
    artifact_paths.extend(multimodal_paths)

    causal_artifact, causal_paths = fit_causal_skip_decomposition(
        data=data,
        output_dir=causal_dir,
        random_seed=random_seed,
        logger=logger,
    )
    artifact_paths.extend(causal_paths)

    digital_twin, twin_paths = fit_listener_digital_twin(
        data=data,
        sequence_length=sequence_length,
        artist_labels=artist_labels,
        multimodal_space=multimodal_space,
        output_dir=digital_twin_dir,
        logger=logger,
    )
    artifact_paths.extend(twin_paths)

    journey_paths = build_journey_plans(
        data=data,
        artist_labels=artist_labels,
        multimodal_space=multimodal_space,
        digital_twin=digital_twin,
        output_dir=journey_dir,
        logger=logger,
    )
    artifact_paths.extend(journey_paths)

    safe_policy, safe_policy_paths = learn_safe_bandit_policy(
        data=data,
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        output_dir=safe_policy_dir,
        logger=logger,
        random_seed=random_seed,
    )
    artifact_paths.extend(safe_policy_paths)

    stress_paths = run_stress_test_lab(
        data=data,
        digital_twin=digital_twin,
        multimodal_space=multimodal_space,
        safe_policy=safe_policy,
        causal_artifact=causal_artifact,
        output_dir=stress_dir,
        logger=logger,
        random_seed=random_seed,
    )
    artifact_paths.extend(stress_paths)

    causal_summary = _read_json(causal_dir / "causal_skip_summary.json")
    twin_summary = _read_json(digital_twin_dir / "listener_digital_twin_summary.json")
    journey_summary = _read_json(journey_dir / "journey_plans_summary.json")
    stress_summary = _read_json(stress_dir / "stress_test_summary.json")

    journey_rows = journey_summary if isinstance(journey_summary, list) else []
    journey_horizons = [_safe_float(row.get("planned_horizon")) for row in journey_rows if isinstance(row, dict)]
    journey_scores = [_safe_float(row.get("plan_score")) for row in journey_rows if isinstance(row, dict)]

    stress_rows = stress_summary if isinstance(stress_summary, list) else []
    safe_stress_rows = [
        row for row in stress_rows if isinstance(row, dict) and str(row.get("policy_name", "")).strip() == "safe_global"
    ]
    worst_safe_row = max(
        safe_stress_rows,
        key=lambda row: _safe_float(row.get("mean_skip_risk")),
        default={},
    )

    summary_payload = {
        "multimodal_artist_count": int(len(multimodal_space.artist_labels)),
        "multimodal_feature_count": int(len(multimodal_space.feature_names)),
        "multimodal_embedding_dim": int(multimodal_space.embeddings.shape[1]),
        "multimodal_retrieval_fusion_enabled": any(
            str(feature_name).startswith("retrieval_embed_") for feature_name in multimodal_space.feature_names
        ),
        "causal_val_auc_total": _safe_float((causal_summary or {}).get("val", {}).get("auc_total")),
        "causal_test_auc_total": _safe_float((causal_summary or {}).get("test", {}).get("auc_total")),
        "causal_test_mean_friction_uplift": _safe_float(
            (causal_summary or {}).get("test", {}).get("mean_friction_uplift")
        ),
        "digital_twin_average_track_seconds": float(digital_twin.average_track_seconds),
        "digital_twin_val_auc": _safe_float((twin_summary or {}).get("val", {}).get("auc")),
        "digital_twin_test_auc": _safe_float((twin_summary or {}).get("test", {}).get("auc")),
        "journey_seed_count": int(len(journey_rows)),
        "journey_mean_horizon": _safe_mean(journey_horizons),
        "journey_mean_plan_score": _safe_mean(journey_scores),
        "safe_policy_bucket_count": int(len(safe_policy.policy_map)),
        "safe_policy_global_transition_weight": _safe_float(safe_policy.global_policy.get("transition")),
        "safe_policy_global_continuity_weight": _safe_float(safe_policy.global_policy.get("continuity")),
        "safe_policy_global_novelty_weight": _safe_float(safe_policy.global_policy.get("novelty")),
        "safe_policy_global_repeat_weight": _safe_float(safe_policy.global_policy.get("repeat")),
        "stress_scenario_count": int(
            len({str(row.get("scenario", "")).strip() for row in stress_rows if isinstance(row, dict)})
        ),
        "stress_worst_skip_scenario": str(worst_safe_row.get("scenario", "")),
        "stress_worst_skip_risk": _safe_float(worst_safe_row.get("mean_skip_risk")),
        "stress_worst_end_risk": _safe_float(worst_safe_row.get("mean_end_risk")),
        "artifact_count": int(len(artifact_paths) + 1),
    }
    summary_path = analysis_dir / "moonshot_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    artifact_paths.append(summary_path)
    logger.info(
        "Completed moonshot lab: embed_dim=%d journey_seeds=%d stress_scenarios=%d",
        int(multimodal_space.embeddings.shape[1]),
        int(len(journey_rows)),
        int(summary_payload["stress_scenario_count"]),
    )
    return artifact_paths
