from __future__ import annotations

import pandas as pd
import pytest

from spotify.public_listening_trends import (
    build_public_listening_trends,
    calculate_closer_scope_streaks,
    calculate_similarity_trends,
    flag_similarity_anomalies,
)


def _mart(dates: list[str], global_values: list[float], us_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listening_date": dates,
            "reference_edition": 2025,
            "reference_alignment": "date_aligned",
            "dimension": "artists",
            "event_count": 10,
            "duration_minutes": 30.0,
            "unique_entity_count": 5,
            "global_similarity": global_values,
            "united_states_similarity": us_values,
            "united_states_minus_global": [
                us_value - global_value for global_value, us_value in zip(global_values, us_values)
            ],
            "closer_scope": [
                "united_states" if us_value > global_value else "global" if global_value > us_value else "tie"
                for global_value, us_value in zip(global_values, us_values)
            ],
            "global_event_share_on_public_top": global_values,
            "united_states_event_share_on_public_top": us_values,
            "global_duration_share_on_public_top": global_values,
            "united_states_duration_share_on_public_top": us_values,
            "personal_top_entity": "Artist",
            "personal_top_entity_detail": None,
        }
    )


def test_trends_use_calendar_windows_and_exact_lags_for_sparse_dates() -> None:
    mart = _mart(
        ["2026-01-01", "2026-01-03", "2026-01-08"],
        [0.1, 0.3, 0.8],
        [0.2, 0.4, 0.9],
    )

    trends = calculate_similarity_trends(mart)
    last = trends.iloc[-1]

    assert last["global_similarity_7d_mean"] == pytest.approx(0.55)
    assert last["global_similarity_30d_mean"] == pytest.approx(0.4)
    assert last["global_similarity_7d_delta"] == pytest.approx(0.7)
    assert pd.isna(last["global_similarity_1d_delta"])

    streaks = calculate_closer_scope_streaks(mart)
    assert streaks["united_states_closer_streak"].tolist() == [1, 1, 1]


def test_zero_similarity_values_are_preserved_and_produce_finite_features() -> None:
    mart = _mart(
        pd.date_range("2026-01-01", periods=8, freq="D").strftime("%Y-%m-%d").tolist(),
        [0.0] * 8,
        [0.0] * 8,
    )

    result = build_public_listening_trends(mart, minimum_history=3)

    assert result["global_similarity_7d_mean"].eq(0.0).all()
    assert result["global_similarity_7d_volatility"].eq(0.0).all()
    assert result["global_similarity_anomaly_flag"].eq(False).all()
    assert result["similarity_anomaly_flag"].eq(False).all()
    assert result["closer_scope_streak"].eq(0).all()


def test_anomaly_flags_wait_for_history_and_detect_spike_after_constant_baseline() -> None:
    dates = pd.date_range("2026-02-01", periods=8, freq="D").strftime("%Y-%m-%d").tolist()
    mart = _mart(dates, [0.2] * 7 + [0.9], [0.3] * 8)

    result = flag_similarity_anomalies(mart, minimum_history=5, history_window_days=30)

    assert result.loc[:4, "similarity_anomaly_flag"].eq(False).all()
    assert result.iloc[-1]["global_similarity_anomaly_history_count"] == 7
    assert result.iloc[-1]["global_similarity_anomaly_baseline"] == pytest.approx(0.2)
    assert bool(result.iloc[-1]["global_similarity_anomaly_flag"]) is True
    assert bool(result.iloc[-1]["similarity_anomaly_flag"]) is True
    assert bool(result.iloc[-1]["united_states_similarity_anomaly_flag"]) is False


def test_anomaly_history_is_limited_to_calendar_window() -> None:
    mart = _mart(
        ["2026-01-01", "2026-01-02", "2026-02-15"],
        [0.1, 0.1, 0.9],
        [0.2, 0.2, 0.2],
    )

    result = flag_similarity_anomalies(mart, minimum_history=2, history_window_days=30)

    assert result.iloc[-1]["global_similarity_anomaly_history_count"] == 0
    assert pd.isna(result.iloc[-1]["global_similarity_anomaly_score"])
    assert bool(result.iloc[-1]["similarity_anomaly_flag"]) is False
