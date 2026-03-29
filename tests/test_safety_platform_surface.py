from __future__ import annotations

from spotify import safety_platform
from spotify.recommender_safety import (
    build_conformal_abstention_summary,
    build_temporal_backtest_windows,
    evaluate_promotion_gate,
)


def test_minimal_safety_api_groups_cover_week9_surface() -> None:
    rows = safety_platform.describe_minimal_safety_api()

    assert [row["key"] for row in rows] == [
        "temporal_backtest",
        "drift",
        "promotion_gate",
        "abstention",
    ]
    assert "run_temporal_backtest_benchmark" in rows[0]["entrypoints"]
    assert "compute_target_distribution_drift" in rows[1]["entrypoints"]
    assert rows[2]["entrypoints"] == ["evaluate_promotion_gate"]
    assert rows[3]["entrypoints"] == ["build_conformal_abstention_summary"]


def test_spotify_integration_map_points_to_wrapper_modules() -> None:
    rows = safety_platform.describe_spotify_integration_map()

    assert any(
        row["platform_group"] == "temporal_backtest"
        and row["spotify_module"] == "spotify/backtesting.py"
        and row["wrapper_entrypoint"] == "run_temporal_backtest"
        for row in rows
    )
    assert any(
        row["platform_group"] == "promotion_gate"
        and row["spotify_module"] == "spotify/governance.py"
        and row["wrapper_entrypoint"] == "evaluate_champion_gate"
        for row in rows
    )


def test_safety_platform_reexports_core_generic_primitives() -> None:
    assert safety_platform.build_temporal_backtest_windows is build_temporal_backtest_windows
    assert safety_platform.evaluate_promotion_gate is evaluate_promotion_gate
    assert safety_platform.build_conformal_abstention_summary is build_conformal_abstention_summary
