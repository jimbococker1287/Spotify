from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .analytics_warehouse import build_analytics_warehouse_bundle
from .run_artifacts import write_csv_rows
from .run_artifacts import write_json
from .run_artifacts import write_markdown


def _resolve_env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    import os

    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _rows_for_columns(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    trimmed = frame.copy()
    for column in columns:
        if column not in trimmed.columns:
            trimmed[column] = None
    return trimmed[columns].to_dict(orient="records")


def _json_ready_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    return normalized.to_dict(orient="records")


def _load_listener_daily_activity(
    *,
    data_dir: Path,
    output_dir: Path,
    include_video: bool,
    logger,
) -> pd.DataFrame:
    parquet_path = output_dir / "analytics" / "warehouse" / "silver" / "listener_daily_activity.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    bundle = build_analytics_warehouse_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=include_video,
        logger=logger,
    )
    return bundle.silver.get("listener_daily_activity", pd.DataFrame()).copy()


def _feature_frame(listener_daily: pd.DataFrame) -> pd.DataFrame:
    frame = listener_daily.copy()
    if frame.empty:
        return frame
    frame["played_date"] = pd.to_datetime(frame["played_date"], errors="coerce")
    frame = frame.loc[frame["played_date"].notna()].copy()
    if frame.empty:
        return frame
    numeric_columns = [
        "total_streams",
        "total_ms_played",
        "unique_artists",
        "unique_tracks",
        "skip_rate",
        "shuffle_rate",
        "offline_rate",
        "track_stream_share",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)
    frame["exploration_ratio"] = (
        frame["unique_artists"] / np.maximum(frame["total_streams"], 1.0)
    ).astype("float64", copy=False)
    frame["repeat_intensity"] = (1.0 - frame["exploration_ratio"]).astype("float64", copy=False)
    frame["engagement_hours"] = (frame["total_ms_played"] / 3_600_000.0).astype("float64", copy=False)
    frame["streams_log1p"] = np.log1p(frame["total_streams"]).astype("float64", copy=False)
    frame["ms_log1p"] = np.log1p(frame["total_ms_played"]).astype("float64", copy=False)
    frame["month"] = frame["played_date"].dt.to_period("M").astype(str)
    return frame


def _cluster_label_candidates(
    row: pd.Series,
    *,
    quantiles: dict[str, float],
) -> list[str]:
    labels: list[str] = []
    if row["track_stream_share"] <= 0.60:
        labels.append("mixed_media_reset")
    if row["offline_rate"] >= max(0.15, quantiles["offline_high"]):
        labels.append("offline_binge")
    if row["skip_rate"] >= quantiles["skip_high"] and row["shuffle_rate"] >= quantiles["shuffle_mid"]:
        labels.append("skip_surfing")
    if row["exploration_ratio"] >= quantiles["exploration_high"] and row["shuffle_rate"] >= quantiles["shuffle_mid"]:
        labels.append("exploratory_shuffle")
    if (
        row["repeat_intensity"] >= quantiles["repeat_mid"]
        and row["shuffle_rate"] < quantiles["shuffle_mid"]
        and row["skip_rate"] < quantiles["skip_high"]
    ):
        labels.append("steady_replay")
    if (
        row["engagement_hours"] >= quantiles["hours_high"]
        and row["skip_rate"] <= quantiles["skip_low"]
        and row["repeat_intensity"] >= quantiles["repeat_mid"]
    ):
        labels.append("deep_focus")
    if row["total_streams"] <= quantiles["streams_low"]:
        labels.append("quick_check_in")
    if row["skip_rate"] >= quantiles["skip_high"]:
        labels.append("restless_repeat")
    if not labels:
        labels.append("steady_listening")
    return labels


def _assign_cluster_labels(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    if cluster_summary.empty:
        return cluster_summary
    quantiles = {
        "skip_high": float(cluster_summary["skip_rate"].quantile(0.75)),
        "skip_low": float(cluster_summary["skip_rate"].quantile(0.25)),
        "offline_high": float(cluster_summary["offline_rate"].quantile(0.75)),
        "exploration_high": float(cluster_summary["exploration_ratio"].quantile(0.75)),
        "shuffle_mid": float(cluster_summary["shuffle_rate"].quantile(0.50)),
        "hours_high": float(cluster_summary["engagement_hours"].quantile(0.75)),
        "repeat_mid": float(cluster_summary["repeat_intensity"].quantile(0.50)),
        "streams_low": float(cluster_summary["total_streams"].quantile(0.25)),
    }
    used: dict[str, int] = {}
    labels: list[str] = []
    for _, row in cluster_summary.iterrows():
        selected = ""
        for candidate in _cluster_label_candidates(row, quantiles=quantiles):
            if used.get(candidate, 0) == 0:
                selected = candidate
                break
        if not selected:
            selected = _cluster_label_candidates(row, quantiles=quantiles)[0]
        used[selected] = used.get(selected, 0) + 1
        if used[selected] > 1:
            labels.append(f"{selected}_{used[selected]}")
        else:
            labels.append(selected)
    cluster_summary = cluster_summary.copy()
    cluster_summary["archetype_label"] = labels
    return cluster_summary


def _dominant_transitions(assignments: pd.DataFrame) -> pd.DataFrame:
    columns = ["from_archetype", "to_archetype", "transition_count", "transition_share"]
    if assignments.empty or "archetype_label" not in assignments.columns:
        return _empty_frame(columns)
    ordered = assignments.sort_values("played_date").reset_index(drop=True)
    next_labels = ordered["archetype_label"].shift(-1)
    transition_frame = ordered.iloc[:-1].copy()
    transition_frame["next_archetype"] = next_labels.iloc[:-1].to_numpy(copy=False)
    counts = (
        transition_frame.groupby(["archetype_label", "next_archetype"], dropna=False)
        .size()
        .reset_index(name="transition_count")
    )
    total = float(counts["transition_count"].sum()) if not counts.empty else 0.0
    counts["transition_share"] = (
        counts["transition_count"] / total if total > 0 else 0.0
    )
    counts = counts.rename(columns={"archetype_label": "from_archetype", "next_archetype": "to_archetype"})
    return counts[columns].sort_values(["transition_count", "from_archetype"], ascending=[False, True]).reset_index(drop=True)


def _season_parts(date_value: pd.Timestamp) -> tuple[int, int, str]:
    if pd.isna(date_value):
        return 0, 0, ""
    month = int(date_value.month)
    year = int(date_value.year)
    if month == 12:
        return year + 1, 0, "winter"
    if month in (1, 2):
        return year, 0, "winter"
    if month in (3, 4, 5):
        return year, 1, "spring"
    if month in (6, 7, 8):
        return year, 2, "summer"
    return year, 3, "fall"


def _seasonal_archetype_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season_label",
        "season_year",
        "season",
        "archetype_label",
        "day_count",
        "archetype_share",
        "mean_streams",
        "mean_skip_rate",
        "mean_exploration_ratio",
        "start_date",
        "end_date",
    ]
    if assignments.empty or "played_date" not in assignments.columns or "archetype_label" not in assignments.columns:
        return _empty_frame(columns)
    frame = assignments.copy()
    parts = frame["played_date"].apply(_season_parts)
    frame["season_year"] = parts.apply(lambda item: item[0]).astype("int64", copy=False)
    frame["season_order"] = parts.apply(lambda item: item[1]).astype("int64", copy=False)
    frame["season"] = parts.apply(lambda item: item[2]).astype(str)
    frame["season_label"] = frame["season_year"].astype(str) + "-" + frame["season"]
    seasonal = (
        frame.groupby(["season_year", "season_order", "season", "season_label", "archetype_label"], dropna=False)
        .agg(
            day_count=("played_date", "count"),
            mean_streams=("total_streams", "mean"),
            mean_skip_rate=("skip_rate", "mean"),
            mean_exploration_ratio=("exploration_ratio", "mean"),
            start_date=("played_date", "min"),
            end_date=("played_date", "max"),
        )
        .reset_index()
    )
    season_totals = seasonal.groupby(["season_year", "season_label"], dropna=False)["day_count"].transform("sum")
    seasonal["archetype_share"] = seasonal["day_count"] / np.maximum(season_totals, 1.0)
    seasonal = seasonal.sort_values(
        ["season_year", "season_order", "day_count", "archetype_label"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return seasonal[columns]


def _taste_regime_shifts(monthly_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "month",
        "previous_month",
        "dominant_archetype",
        "previous_dominant_archetype",
        "dominant_share",
        "runner_up_archetype",
        "runner_up_share",
        "dominance_gap",
        "active_archetype_count",
        "dominant_changed_from_prev_month",
        "regime_shift_score",
        "top_share_gain_archetype",
        "top_share_gain_delta",
        "top_share_loss_archetype",
        "top_share_loss_delta",
        "dominant_mean_skip_rate",
        "dominant_mean_exploration_ratio",
    ]
    if monthly_summary.empty:
        return _empty_frame(columns)

    share_pivot = monthly_summary.pivot_table(
        index="month",
        columns="archetype_label",
        values="archetype_share",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    rows: list[dict[str, object]] = []
    months = list(share_pivot.index)
    previous_dominant_label = ""
    for idx, month in enumerate(months):
        month_slice = (
            monthly_summary.loc[monthly_summary["month"].astype(str) == str(month)]
            .sort_values(["archetype_share", "day_count", "archetype_label"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        if month_slice.empty:
            continue
        dominant = month_slice.iloc[0]
        runner_up = month_slice.iloc[1] if len(month_slice.index) > 1 else None
        current_vector = share_pivot.loc[month]
        previous_month = str(months[idx - 1]) if idx > 0 else ""
        regime_shift_score = 0.0
        top_gain_label = ""
        top_gain_delta = 0.0
        top_loss_label = ""
        top_loss_delta = 0.0
        if idx > 0:
            previous_vector = share_pivot.loc[months[idx - 1]]
            deltas = (current_vector - previous_vector).sort_index()
            regime_shift_score = float(0.5 * deltas.abs().sum())
            if not deltas.empty:
                gain_idx = str(deltas.idxmax())
                gain_value = float(deltas.max())
                loss_idx = str(deltas.idxmin())
                loss_value = float(deltas.min())
                top_gain_label = gain_idx if gain_value > 0 else ""
                top_gain_delta = gain_value if gain_value > 0 else 0.0
                top_loss_label = loss_idx if loss_value < 0 else ""
                top_loss_delta = loss_value if loss_value < 0 else 0.0
        rows.append(
            {
                "month": str(month),
                "previous_month": previous_month,
                "dominant_archetype": str(dominant["archetype_label"]),
                "previous_dominant_archetype": previous_dominant_label,
                "dominant_share": float(dominant["archetype_share"]),
                "runner_up_archetype": str(runner_up["archetype_label"]) if runner_up is not None else "",
                "runner_up_share": float(runner_up["archetype_share"]) if runner_up is not None else 0.0,
                "dominance_gap": float(
                    float(dominant["archetype_share"]) - float(runner_up["archetype_share"]) if runner_up is not None else dominant["archetype_share"]
                ),
                "active_archetype_count": int(len(month_slice.index)),
                "dominant_changed_from_prev_month": bool(idx > 0 and str(dominant["archetype_label"]) != previous_dominant_label),
                "regime_shift_score": regime_shift_score,
                "top_share_gain_archetype": top_gain_label,
                "top_share_gain_delta": top_gain_delta,
                "top_share_loss_archetype": top_loss_label,
                "top_share_loss_delta": top_loss_delta,
                "dominant_mean_skip_rate": float(dominant["mean_skip_rate"]),
                "dominant_mean_exploration_ratio": float(dominant["mean_exploration_ratio"]),
            }
        )
        previous_dominant_label = str(dominant["archetype_label"])
    return pd.DataFrame(rows, columns=columns)


def _build_taste_state_brief(
    assignments: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    transitions: pd.DataFrame,
) -> dict[str, Any]:
    if assignments.empty or cluster_summary.empty:
        return {
            "status": "empty",
            "dominant_archetype": "",
            "summary": [],
            "actions": [],
        }
    dominant = (
        assignments.groupby("archetype_label", dropna=False)
        .size()
        .reset_index(name="days")
        .sort_values("days", ascending=False)
        .iloc[0]
    )
    most_variable_month = None
    if not monthly_summary.empty:
        grouped = (
            monthly_summary.groupby("month", dropna=False)["archetype_share"].max().reset_index(name="max_share")
        )
        if not grouped.empty:
            most_variable_month = grouped.sort_values("max_share").iloc[0]["month"]
    top_transition = transitions.iloc[0].to_dict() if not transitions.empty else {}
    highest_skip = cluster_summary.sort_values("skip_rate", ascending=False).iloc[0]
    highest_exploration = cluster_summary.sort_values("exploration_ratio", ascending=False).iloc[0]
    summary = [
        f"Dominant listener archetype is `{dominant['archetype_label']}` across `{int(dominant['days'])}` days.",
        f"Highest-skip archetype is `{highest_skip['archetype_label']}` at skip rate `{float(highest_skip['skip_rate']):.3f}`.",
        f"Most exploratory archetype is `{highest_exploration['archetype_label']}` with exploration ratio `{float(highest_exploration['exploration_ratio']):.3f}`.",
    ]
    if most_variable_month:
        summary.append(f"Most mixed taste-state month is `{most_variable_month}`.")
    if top_transition:
        summary.append(
            f"Most common day-to-day shift is `{top_transition.get('from_archetype', '')} -> {top_transition.get('to_archetype', '')}`."
        )
    actions = [
        "Use the dominant archetype when choosing the default Taste OS mode for a local demo.",
        "Use the highest-skip archetype as the first slice for skip-risk and recovery experiments.",
        "Use the most exploratory archetype when testing discovery or novelty-routing ideas.",
    ]
    return {
        "status": "ok",
        "dominant_archetype": str(dominant["archetype_label"]),
        "dominant_archetype_days": int(dominant["days"]),
        "highest_skip_archetype": str(highest_skip["archetype_label"]),
        "highest_skip_rate": float(highest_skip["skip_rate"]),
        "highest_exploration_archetype": str(highest_exploration["archetype_label"]),
        "highest_exploration_ratio": float(highest_exploration["exploration_ratio"]),
        "most_variable_month": str(most_variable_month or ""),
        "top_transition": top_transition,
        "summary": summary,
        "actions": actions,
    }


def _build_taste_evolution_brief(
    regime_shifts: pd.DataFrame,
    seasonal_summary: pd.DataFrame,
    transitions: pd.DataFrame,
) -> dict[str, Any]:
    if regime_shifts.empty and seasonal_summary.empty and transitions.empty:
        return {
            "status": "empty",
            "summary": [],
            "actions": [],
        }

    biggest_shift = {}
    if not regime_shifts.empty:
        candidates = regime_shifts.loc[regime_shifts["previous_month"].astype(str) != ""].copy()
        if not candidates.empty:
            biggest_shift = (
                candidates.sort_values(
                    ["regime_shift_score", "dominant_changed_from_prev_month", "month"],
                    ascending=[False, False, True],
                )
                .iloc[0]
                .to_dict()
            )

    top_transition = transitions.iloc[0].to_dict() if not transitions.empty else {}
    cross_state = {}
    if not transitions.empty:
        cross_state_rows = transitions.loc[
            transitions["from_archetype"].astype(str) != transitions["to_archetype"].astype(str)
        ].copy()
        if not cross_state_rows.empty:
            cross_state = (
                cross_state_rows.sort_values(
                    ["transition_count", "from_archetype", "to_archetype"],
                    ascending=[False, True, True],
                )
                .iloc[0]
                .to_dict()
            )

    most_seasonal = {}
    if not seasonal_summary.empty:
        seasonal_profile = (
            seasonal_summary.groupby(["season", "archetype_label"], dropna=False)["archetype_share"]
            .mean()
            .reset_index(name="mean_share")
        )
        if not seasonal_profile.empty:
            seasonality = (
                seasonal_profile.groupby("archetype_label", dropna=False)
                .agg(
                    peak_share=("mean_share", "max"),
                    trough_share=("mean_share", "min"),
                )
                .reset_index()
            )
            seasonality["seasonality_gap"] = seasonality["peak_share"] - seasonality["trough_share"]
            strongest = (
                seasonality.sort_values(["seasonality_gap", "peak_share", "archetype_label"], ascending=[False, False, True])
                .iloc[0]
            )
            profile_rows = seasonal_profile.loc[
                seasonal_profile["archetype_label"].astype(str) == str(strongest["archetype_label"])
            ].sort_values(["mean_share", "season"], ascending=[False, True])
            peak_row = profile_rows.iloc[0] if not profile_rows.empty else None
            most_seasonal = {
                "archetype_label": str(strongest["archetype_label"]),
                "seasonality_gap": float(strongest["seasonality_gap"]),
                "peak_season": str(peak_row["season"]) if peak_row is not None else "",
                "peak_season_share": float(peak_row["mean_share"]) if peak_row is not None else 0.0,
            }

    summary: list[str] = []
    if biggest_shift:
        if bool(biggest_shift.get("dominant_changed_from_prev_month")):
            summary.append(
                f"Largest month-over-month regime shift lands in `{biggest_shift.get('month', '')}`, where dominance moved from `{biggest_shift.get('previous_dominant_archetype', '')}` to `{biggest_shift.get('dominant_archetype', '')}` at shift score `{float(biggest_shift.get('regime_shift_score', 0.0)):.3f}`."
            )
        else:
            summary.append(
                f"Largest month-over-month regime shift lands in `{biggest_shift.get('month', '')}` with dominant archetype `{biggest_shift.get('dominant_archetype', '')}` at shift score `{float(biggest_shift.get('regime_shift_score', 0.0)):.3f}`."
            )
    if most_seasonal:
        summary.append(
            f"Most seasonal archetype is `{most_seasonal.get('archetype_label', '')}`, peaking in `{most_seasonal.get('peak_season', '')}` at share `{float(most_seasonal.get('peak_season_share', 0.0)):.3f}`."
        )
    if cross_state:
        summary.append(
            f"Largest cross-state transition is `{cross_state.get('from_archetype', '')} -> {cross_state.get('to_archetype', '')}` with share `{float(cross_state.get('transition_share', 0.0)):.3f}`."
        )
    elif top_transition:
        summary.append(
            f"Most common taste-state transition overall is `{top_transition.get('from_archetype', '')} -> {top_transition.get('to_archetype', '')}`."
        )

    actions = [
        "Use the highest regime-shift month as the first retrospective slice when explaining taste evolution changes over time.",
        "Use the most seasonal archetype as the calendar-aware slice for playlist, context, or discovery experiments.",
        "Use the biggest cross-state transition as the handoff sequence when designing taste-state recovery or escalation paths.",
    ]
    return {
        "status": "ok",
        "biggest_regime_shift": biggest_shift,
        "most_seasonal_archetype": most_seasonal,
        "top_transition": top_transition,
        "top_cross_state_transition": cross_state,
        "summary": summary,
        "actions": actions,
    }


def build_listener_archetypes(
    *,
    data_dir: Path,
    output_dir: Path,
    include_video: bool,
    logger,
) -> list[Path]:
    listener_daily = _load_listener_daily_activity(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=include_video,
        logger=logger,
    )
    feature_frame = _feature_frame(listener_daily)
    if feature_frame.empty:
        return []

    feature_columns = [
        "streams_log1p",
        "ms_log1p",
        "exploration_ratio",
        "skip_rate",
        "shuffle_rate",
        "offline_rate",
        "track_stream_share",
    ]
    model_frame = feature_frame[feature_columns].fillna(0.0)
    cluster_cap = min(5, max(3, int(len(model_frame) // 180) if len(model_frame) >= 540 else 3))
    n_clusters = _resolve_env_int(
        "SPOTIFY_LISTENER_ARCHETYPE_CLUSTERS",
        cluster_cap,
        minimum=2,
        maximum=max(2, min(8, len(model_frame))),
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_frame)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    assignments = feature_frame.copy()
    assignments["cluster_id"] = labels.astype("int32")
    cluster_summary = (
        assignments.groupby("cluster_id", dropna=False)
        .agg(
            day_count=("played_date", "count"),
            total_streams=("total_streams", "mean"),
            engagement_hours=("engagement_hours", "mean"),
            unique_artists=("unique_artists", "mean"),
            unique_tracks=("unique_tracks", "mean"),
            exploration_ratio=("exploration_ratio", "mean"),
            repeat_intensity=("repeat_intensity", "mean"),
            skip_rate=("skip_rate", "mean"),
            shuffle_rate=("shuffle_rate", "mean"),
            offline_rate=("offline_rate", "mean"),
            track_stream_share=("track_stream_share", "mean"),
            start_date=("played_date", "min"),
            end_date=("played_date", "max"),
        )
        .reset_index()
    )
    cluster_summary = _assign_cluster_labels(cluster_summary)
    assignments = assignments.merge(cluster_summary[["cluster_id", "archetype_label"]], on="cluster_id", how="left")

    monthly_summary = (
        assignments.groupby(["month", "archetype_label"], dropna=False)
        .agg(
            day_count=("played_date", "count"),
            mean_streams=("total_streams", "mean"),
            mean_skip_rate=("skip_rate", "mean"),
            mean_exploration_ratio=("exploration_ratio", "mean"),
        )
        .reset_index()
    )
    month_totals = monthly_summary.groupby("month", dropna=False)["day_count"].transform("sum")
    monthly_summary["archetype_share"] = monthly_summary["day_count"] / np.maximum(month_totals, 1.0)
    monthly_summary = monthly_summary.sort_values(["month", "day_count"], ascending=[True, False]).reset_index(drop=True)

    transitions = _dominant_transitions(assignments)
    regime_shifts = _taste_regime_shifts(monthly_summary)
    seasonal_summary = _seasonal_archetype_summary(assignments)
    brief_payload = _build_taste_state_brief(assignments, cluster_summary, monthly_summary, transitions)
    evolution_payload = _build_taste_evolution_brief(regime_shifts, seasonal_summary, transitions)

    output_root = output_dir / "analysis" / "listener_archetypes"
    output_root.mkdir(parents=True, exist_ok=True)
    assignments_csv = _write_csv(
        output_root / "listener_archetype_assignments.csv",
        _rows_for_columns(
            assignments,
            [
                "played_date",
                "month",
                "cluster_id",
                "archetype_label",
                "total_streams",
                "engagement_hours",
                "unique_artists",
                "unique_tracks",
                "exploration_ratio",
                "repeat_intensity",
                "skip_rate",
                "shuffle_rate",
                "offline_rate",
                "primary_platform",
            ],
        ),
        [
            "played_date",
            "month",
            "cluster_id",
            "archetype_label",
            "total_streams",
            "engagement_hours",
            "unique_artists",
            "unique_tracks",
            "exploration_ratio",
            "repeat_intensity",
            "skip_rate",
            "shuffle_rate",
            "offline_rate",
            "primary_platform",
        ],
    )
    summary_csv = _write_csv(
        output_root / "listener_archetype_summary.csv",
        _rows_for_columns(
            cluster_summary,
            [
                "cluster_id",
                "archetype_label",
                "day_count",
                "total_streams",
                "engagement_hours",
                "unique_artists",
                "unique_tracks",
                "exploration_ratio",
                "repeat_intensity",
                "skip_rate",
                "shuffle_rate",
                "offline_rate",
                "track_stream_share",
                "start_date",
                "end_date",
            ],
        ),
        [
            "cluster_id",
            "archetype_label",
            "day_count",
            "total_streams",
            "engagement_hours",
            "unique_artists",
            "unique_tracks",
            "exploration_ratio",
            "repeat_intensity",
            "skip_rate",
            "shuffle_rate",
            "offline_rate",
            "track_stream_share",
            "start_date",
            "end_date",
        ],
    )
    monthly_csv = _write_csv(
        output_root / "listener_archetype_monthly.csv",
        _rows_for_columns(
            monthly_summary,
            [
                "month",
                "archetype_label",
                "day_count",
                "archetype_share",
                "mean_streams",
                "mean_skip_rate",
                "mean_exploration_ratio",
            ],
        ),
        [
            "month",
            "archetype_label",
            "day_count",
            "archetype_share",
            "mean_streams",
            "mean_skip_rate",
            "mean_exploration_ratio",
        ],
    )
    transitions_csv = _write_csv(
        output_root / "listener_archetype_transitions.csv",
        transitions.to_dict(orient="records"),
        ["from_archetype", "to_archetype", "transition_count", "transition_share"],
    )
    regime_shift_columns = [
        "month",
        "previous_month",
        "dominant_archetype",
        "previous_dominant_archetype",
        "dominant_share",
        "runner_up_archetype",
        "runner_up_share",
        "dominance_gap",
        "active_archetype_count",
        "dominant_changed_from_prev_month",
        "regime_shift_score",
        "top_share_gain_archetype",
        "top_share_gain_delta",
        "top_share_loss_archetype",
        "top_share_loss_delta",
        "dominant_mean_skip_rate",
        "dominant_mean_exploration_ratio",
    ]
    regime_shifts_csv = _write_csv(
        output_root / "taste_evolution_regime_shifts.csv",
        _rows_for_columns(regime_shifts, regime_shift_columns),
        regime_shift_columns,
    )
    seasonal_columns = [
        "season_label",
        "season_year",
        "season",
        "archetype_label",
        "day_count",
        "archetype_share",
        "mean_streams",
        "mean_skip_rate",
        "mean_exploration_ratio",
        "start_date",
        "end_date",
    ]
    seasonal_csv = _write_csv(
        output_root / "listener_archetype_seasonal.csv",
        _rows_for_columns(seasonal_summary, seasonal_columns),
        seasonal_columns,
    )
    brief_json = write_json(output_root / "taste_state_brief.json", brief_payload)
    evolution_json = write_json(output_root / "taste_evolution_brief.json", evolution_payload)
    summary_json = write_json(
        output_root / "listener_archetype_summary.json",
        {
            "cluster_count": int(n_clusters),
            "day_count": int(len(assignments.index)),
            "archetypes": _json_ready_records(cluster_summary),
        },
    )
    regime_shifts_json = write_json(
        output_root / "taste_evolution_regime_shifts.json",
        _json_ready_records(regime_shifts),
    )
    seasonal_json = write_json(
        output_root / "listener_archetype_seasonal.json",
        _json_ready_records(seasonal_summary),
    )
    brief_md = write_markdown(
        output_root / "taste_state_brief.md",
        [
            "# Taste State Brief",
            "",
            *[f"- {line}" for line in brief_payload.get("summary", [])],
            "",
            "## Suggested Uses",
            "",
            *[f"- {line}" for line in brief_payload.get("actions", [])],
        ],
    )
    evolution_md = write_markdown(
        output_root / "taste_evolution_brief.md",
        [
            "# Taste Evolution Brief",
            "",
            *[f"- {line}" for line in evolution_payload.get("summary", [])],
            "",
            "## Suggested Uses",
            "",
            *[f"- {line}" for line in evolution_payload.get("actions", [])],
        ],
    )
    logger.info(
        "Built listener archetypes across %d days into %d clusters.",
        len(assignments.index),
        n_clusters,
    )
    return [
        assignments_csv,
        summary_csv,
        monthly_csv,
        transitions_csv,
        regime_shifts_csv,
        seasonal_csv,
        summary_json,
        brief_json,
        brief_md,
        regime_shifts_json,
        seasonal_json,
        evolution_json,
        evolution_md,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build listener archetype and taste-state artifacts.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw Spotify export directory.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing warehouse assets.")
    parser.add_argument("--no-video", action="store_true", help="Exclude video history files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.listener_archetypes")
    paths = build_listener_archetypes(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        include_video=not bool(args.no_video),
        logger=logger,
    )
    if not paths:
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
