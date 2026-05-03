from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .recommender_safety import (
    SequenceSplitSnapshot,
    TemporalBacktestWindow,
    build_conformal_abstention_summary,
    build_temporal_backtest_windows,
    compute_context_feature_drift_rows,
    compute_segment_share_shift_rows,
    compute_target_distribution_drift,
    evaluate_promotion_gate,
    run_temporal_backtest_benchmark,
    summarize_backtest_rows,
    write_temporal_backtest_artifacts,
)


@dataclass(frozen=True)
class SafetyPlatformGroup:
    key: str
    purpose: str
    entrypoints: tuple[str, ...]
    spotify_wrappers: tuple[str, ...]
    assumptions: tuple[str, ...]


_MINIMAL_PUBLIC_API: Final[tuple[SafetyPlatformGroup, ...]] = (
    SafetyPlatformGroup(
        key="temporal_backtest",
        purpose="Compare sequence models across rolling windows and write stable benchmark artifacts.",
        entrypoints=(
            "build_temporal_backtest_windows",
            "run_temporal_backtest_benchmark",
            "summarize_backtest_rows",
            "write_temporal_backtest_artifacts",
        ),
        spotify_wrappers=("spotify/backtesting.py:run_temporal_backtest",),
        assumptions=(
            "Callers provide evaluators that can score a TemporalBacktestWindow.",
            "Artifact writing is optional and only needs a filesystem path.",
        ),
    ),
    SafetyPlatformGroup(
        key="drift",
        purpose="Measure context, segment, and target drift between named evaluation splits.",
        entrypoints=(
            "SequenceSplitSnapshot",
            "compute_context_feature_drift_rows",
            "compute_segment_share_shift_rows",
            "compute_target_distribution_drift",
        ),
        spotify_wrappers=("spotify/drift.py:run_drift_diagnostics",),
        assumptions=(
            "Context arrays are two-dimensional numeric matrices.",
            "Segment-share drift only needs pandas frames when segment extractors are used.",
        ),
    ),
    SafetyPlatformGroup(
        key="promotion_gate",
        purpose="Promote or reject challengers using utility thresholds plus selective-risk and abstention caps.",
        entrypoints=("evaluate_promotion_gate",),
        spotify_wrappers=("spotify/governance.py:evaluate_champion_gate",),
        assumptions=(
            "History is stored as row-wise CSV with run, profile, and score columns.",
            "Current risk metrics are optional but should use val_selective_risk and val_abstention_rate keys.",
        ),
    ),
    SafetyPlatformGroup(
        key="abstention",
        purpose="Build conformal abstention summaries so a recommender can refuse unsafe predictions.",
        entrypoints=("build_conformal_abstention_summary",),
        spotify_wrappers=("spotify/evaluation.py:run_extended_evaluation",),
        assumptions=(
            "Inputs are class-probability matrices aligned to integer targets.",
            "Validation probabilities are required because calibration is split-conformal.",
        ),
    ),
)


def describe_minimal_safety_api() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in _MINIMAL_PUBLIC_API:
        rows.append(
            {
                "key": group.key,
                "purpose": group.purpose,
                "entrypoints": list(group.entrypoints),
                "spotify_wrappers": list(group.spotify_wrappers),
                "assumptions": list(group.assumptions),
            }
        )
    return rows


def describe_spotify_integration_map() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in _MINIMAL_PUBLIC_API:
        for wrapper in group.spotify_wrappers:
            module_path, _, wrapper_name = wrapper.partition(":")
            rows.append(
                {
                    "platform_group": group.key,
                    "spotify_module": module_path,
                    "wrapper_entrypoint": wrapper_name,
                    "purpose": group.purpose,
                }
            )
    return rows


__all__ = [
    "SafetyPlatformGroup",
    "SequenceSplitSnapshot",
    "TemporalBacktestWindow",
    "build_conformal_abstention_summary",
    "build_temporal_backtest_windows",
    "compute_context_feature_drift_rows",
    "compute_segment_share_shift_rows",
    "compute_target_distribution_drift",
    "describe_minimal_safety_api",
    "describe_spotify_integration_map",
    "evaluate_promotion_gate",
    "run_temporal_backtest_benchmark",
    "summarize_backtest_rows",
    "write_temporal_backtest_artifacts",
]
