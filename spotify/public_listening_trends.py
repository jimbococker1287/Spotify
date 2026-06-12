from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SIMILARITY_COLUMNS = (
    "global_similarity",
    "united_states_similarity",
    "united_states_minus_global",
)
REQUIRED_COLUMNS = ("listening_date", "dimension", "closer_scope", *SIMILARITY_COLUMNS)
ROLLING_WINDOWS = (7, 30)


def calculate_similarity_trends(daily_mart: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-aware rolling means, deltas, and volatility by dimension."""
    frame = _prepare_daily_mart(daily_mart)
    feature_columns = _trend_columns()
    if frame.empty:
        return _empty_result(daily_mart, feature_columns)

    output = frame.copy()
    for _, positions in output.groupby("dimension", sort=False).indices.items():
        group_positions = np.asarray(positions)
        group = output.iloc[group_positions]
        dates = pd.DatetimeIndex(group["_listening_timestamp"])

        for metric in SIMILARITY_COLUMNS:
            values = pd.Series(group[metric].to_numpy(dtype=float), index=dates)
            for days in ROLLING_WINDOWS:
                rolling = values.rolling(f"{days}D", min_periods=1)
                output.loc[output.index[group_positions], f"{metric}_{days}d_mean"] = rolling.mean().to_numpy()
                output.loc[output.index[group_positions], f"{metric}_{days}d_volatility"] = (
                    rolling.std(ddof=0).to_numpy()
                )

                lagged = pd.Series(values.to_numpy(), index=dates + pd.Timedelta(days=days))
                output.loc[output.index[group_positions], f"{metric}_{days}d_delta"] = (
                    values.to_numpy() - lagged.reindex(dates).to_numpy()
                )

            one_day_lag = pd.Series(values.to_numpy(), index=dates + pd.Timedelta(days=1))
            output.loc[output.index[group_positions], f"{metric}_1d_delta"] = (
                values.to_numpy() - one_day_lag.reindex(dates).to_numpy()
            )

    return _finalize(output)


def calculate_closer_scope_streaks(daily_mart: pd.DataFrame) -> pd.DataFrame:
    """Add consecutive-calendar-day streaks for the closer public scope."""
    frame = _prepare_daily_mart(daily_mart)
    columns = ["closer_scope_streak", "global_closer_streak", "united_states_closer_streak"]
    if frame.empty:
        return _empty_result(daily_mart, columns)

    output = frame.copy()
    streaks = pd.Series(0, index=output.index, dtype="int64")
    for _, positions in output.groupby("dimension", sort=False).indices.items():
        previous_date: pd.Timestamp | None = None
        previous_scope: str | None = None
        streak = 0
        for position in positions:
            row_index = output.index[position]
            current_date = output.at[row_index, "_listening_timestamp"]
            current_scope = str(output.at[row_index, "closer_scope"])
            is_consecutive = previous_date is not None and current_date - previous_date == pd.Timedelta(days=1)
            if current_scope not in {"global", "united_states"}:
                streak = 0
            elif is_consecutive and current_scope == previous_scope:
                streak += 1
            else:
                streak = 1
            streaks.at[row_index] = streak
            previous_date = current_date
            previous_scope = current_scope

    output["closer_scope_streak"] = streaks
    output["global_closer_streak"] = streaks.where(output["closer_scope"].eq("global"), 0)
    output["united_states_closer_streak"] = streaks.where(
        output["closer_scope"].eq("united_states"), 0
    )
    return _finalize(output)


def flag_similarity_anomalies(
    daily_mart: pd.DataFrame,
    *,
    minimum_history: int = 14,
    history_window_days: int = 30,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """Flag similarity values far from their trailing median using a MAD score."""
    if minimum_history < 1:
        raise ValueError("minimum_history must be at least 1")
    if history_window_days < 1:
        raise ValueError("history_window_days must be at least 1")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")

    frame = _prepare_daily_mart(daily_mart)
    anomaly_columns = _anomaly_columns()
    if frame.empty:
        return _empty_result(daily_mart, anomaly_columns)

    output = frame.copy()
    overall_flags = pd.Series(False, index=output.index, dtype=bool)
    for metric in SIMILARITY_COLUMNS:
        history_counts = pd.Series(0, index=output.index, dtype="int64")
        baselines = pd.Series(np.nan, index=output.index, dtype=float)
        scores = pd.Series(np.nan, index=output.index, dtype=float)
        flags = pd.Series(False, index=output.index, dtype=bool)

        for _, positions in output.groupby("dimension", sort=False).indices.items():
            group = output.iloc[positions]
            dates = group["_listening_timestamp"].to_numpy()
            values = group[metric].to_numpy(dtype=float)
            for offset, position in enumerate(positions):
                current_date = dates[offset]
                lower_bound = current_date - np.timedelta64(history_window_days, "D")
                prior_values = values[:offset]
                history = prior_values[(dates[:offset] >= lower_bound) & (dates[:offset] < current_date)]
                history = history[np.isfinite(history)]
                row_index = output.index[position]
                history_counts.at[row_index] = len(history)
                if len(history) < minimum_history or not np.isfinite(values[offset]):
                    continue

                median = float(np.median(history))
                mad = float(np.median(np.abs(history - median)))
                robust_scale = max(1.4826 * mad, 1e-9)
                score = abs(float(values[offset]) - median) / robust_scale
                baselines.at[row_index] = median
                scores.at[row_index] = score
                flags.at[row_index] = score >= z_threshold

        prefix = f"{metric}_anomaly"
        output[f"{prefix}_history_count"] = history_counts
        output[f"{prefix}_baseline"] = baselines
        output[f"{prefix}_score"] = scores
        output[f"{prefix}_flag"] = flags
        overall_flags |= flags

    output["similarity_anomaly_flag"] = overall_flags
    return _finalize(output)


def build_public_listening_trends(
    daily_mart: pd.DataFrame,
    *,
    minimum_history: int = 14,
    history_window_days: int = 30,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """Build the complete daily listening trend and anomaly intelligence frame."""
    trends = calculate_similarity_trends(daily_mart)
    trends = calculate_closer_scope_streaks(trends)
    return flag_similarity_anomalies(
        trends,
        minimum_history=minimum_history,
        history_window_days=history_window_days,
        z_threshold=z_threshold,
    )


def _prepare_daily_mart(daily_mart: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in daily_mart.columns]
    if missing:
        raise ValueError(f"daily mart is missing required columns: {', '.join(missing)}")

    frame = daily_mart.copy()
    if frame.empty:
        return frame

    frame["_listening_timestamp"] = pd.to_datetime(frame["listening_date"], errors="coerce")
    if frame["_listening_timestamp"].isna().any():
        raise ValueError("listening_date must contain valid dates")
    if frame.duplicated(["listening_date", "dimension"]).any():
        raise ValueError("daily mart must contain at most one row per listening_date and dimension")

    for column in SIMILARITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["dimension", "_listening_timestamp"], kind="stable").reset_index(drop=True)


def _trend_columns() -> list[str]:
    columns: list[str] = []
    for metric in SIMILARITY_COLUMNS:
        columns.append(f"{metric}_1d_delta")
        for days in ROLLING_WINDOWS:
            columns.extend(
                (
                    f"{metric}_{days}d_mean",
                    f"{metric}_{days}d_delta",
                    f"{metric}_{days}d_volatility",
                )
            )
    return columns


def _anomaly_columns() -> list[str]:
    columns: list[str] = []
    for metric in SIMILARITY_COLUMNS:
        prefix = f"{metric}_anomaly"
        columns.extend(
            (
                f"{prefix}_history_count",
                f"{prefix}_baseline",
                f"{prefix}_score",
                f"{prefix}_flag",
            )
        )
    return [*columns, "similarity_anomaly_flag"]


def _empty_result(frame: pd.DataFrame, added_columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in added_columns:
        if column.endswith(("_flag",)):
            output[column] = pd.Series(dtype=bool)
        elif column.endswith(("_count", "_streak")):
            output[column] = pd.Series(dtype="int64")
        else:
            output[column] = pd.Series(dtype=float)
    return output


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=["_listening_timestamp"], errors="ignore")
    return output.sort_values(["listening_date", "dimension"], kind="stable").reset_index(drop=True)
