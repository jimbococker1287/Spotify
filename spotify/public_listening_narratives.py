from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


DIMENSION_ORDER = ("artists", "tracks", "podcasts")
SCOPE_ORDER = ("global", "united_states")
PROJECTED_ALIGNMENTS = {"historical_projection", "post_window_projection"}
_TOLERANCE = 1e-12


def _number(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(result) if pd.notna(result) else 0.0


def _ordered(values: Iterable[str], preferred: tuple[str, ...]) -> list[str]:
    unique = {str(value) for value in values}
    return [value for value in preferred if value in unique] + sorted(unique - set(preferred))


def _label(values: list[str]) -> str:
    labels = {"united_states": "U.S."}
    rendered = [labels.get(value, value) for value in values]
    if len(rendered) == 1:
        return rendered[0]
    if len(rendered) == 2:
        return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _alignment_caveats(alignments: list[str]) -> list[str]:
    caveats: list[str] = []
    if "historical_projection" in alignments:
        caveats.append(
            "This date predates the public reference window, so the comparison is a historical projection."
        )
    if "post_window_projection" in alignments:
        caveats.append(
            "This date follows the public reference window, so the comparison is a post-window projection."
        )
    if "date_aligned" in alignments and any(value in PROJECTED_ALIGNMENTS for value in alignments):
        caveats.append("The period mixes date-aligned and projected daily comparisons.")
    return caveats


def _validate_frames(detailed: pd.DataFrame, mart: pd.DataFrame) -> None:
    detailed_required = {
        "listening_date",
        "reference_alignment",
        "reference_scope",
        "dimension",
        "shared_public_entity_count",
    }
    mart_required = {
        "listening_date",
        "reference_alignment",
        "dimension",
        "event_count",
        "duration_minutes",
        "global_similarity",
        "united_states_similarity",
        "global_event_share_on_public_top",
        "united_states_event_share_on_public_top",
        "global_duration_share_on_public_top",
        "united_states_duration_share_on_public_top",
    }
    missing_detailed = sorted(detailed_required - set(detailed.columns))
    missing_mart = sorted(mart_required - set(mart.columns))
    if missing_detailed or missing_mart:
        pieces = []
        if missing_detailed:
            pieces.append(f"detailed frame missing: {', '.join(missing_detailed)}")
        if missing_mart:
            pieces.append(f"mart frame missing: {', '.join(missing_mart)}")
        raise ValueError("; ".join(pieces))


def _strongest_alignment(rows: pd.DataFrame) -> dict[str, Any]:
    candidates: list[tuple[str, str, float]] = []
    for row in rows.itertuples(index=False):
        for scope in SCOPE_ORDER:
            candidates.append(
                (
                    str(row.dimension),
                    scope,
                    _number(getattr(row, f"{scope}_similarity")),
                )
            )
    target = max(score for _, _, score in candidates)
    tied = [(dimension, scope) for dimension, scope, score in candidates if abs(score - target) <= _TOLERANCE]
    dimensions = _ordered((dimension for dimension, _ in tied), DIMENSION_ORDER)
    scopes = _ordered((scope for _, scope in tied), SCOPE_ORDER)
    return {
        "dimensions": dimensions,
        "scopes": scopes,
        "similarity": target,
        "is_tie": len(tied) > 1,
        "explanation": (
            f"The strongest alignment was in {_label(dimensions)} versus the {_label(scopes)} reference "
            f"at {_percent(target)} rank-weighted similarity."
        ),
    }


def _most_distinctive_dimension(rows: pd.DataFrame) -> dict[str, Any]:
    candidates: list[tuple[str, float, list[str]]] = []
    for row in rows.itertuples(index=False):
        scores = {
            scope: _number(getattr(row, f"{scope}_similarity"))
            for scope in SCOPE_ORDER
        }
        best_score = max(scores.values())
        best_scopes = [scope for scope in SCOPE_ORDER if abs(scores[scope] - best_score) <= _TOLERANCE]
        candidates.append((str(row.dimension), best_score, best_scopes))
    target = min(score for _, score, _ in candidates)
    tied = [item for item in candidates if abs(item[1] - target) <= _TOLERANCE]
    dimensions = _ordered((dimension for dimension, _, _ in tied), DIMENSION_ORDER)
    scopes = _ordered((scope for _, _, item_scopes in tied for scope in item_scopes), SCOPE_ORDER)
    return {
        "dimensions": dimensions,
        "scopes": scopes,
        "similarity": target,
        "is_tie": len(tied) > 1,
        "explanation": (
            f"The most distinctive dimension was {_label(dimensions)}; even its closest public reference "
            f"({_label(scopes)}) reached {_percent(target)} rank-weighted similarity."
        ),
    }


def _concentration(rows: pd.DataFrame) -> dict[str, Any]:
    candidates: list[tuple[str, str, float, float]] = []
    for row in rows.itertuples(index=False):
        for scope in SCOPE_ORDER:
            candidates.append(
                (
                    str(row.dimension),
                    scope,
                    _number(getattr(row, f"{scope}_duration_share_on_public_top")),
                    _number(getattr(row, f"{scope}_event_share_on_public_top")),
                )
            )
    peak_duration = max(duration for _, _, duration, _ in candidates)
    duration_ties = [item for item in candidates if abs(item[2] - peak_duration) <= _TOLERANCE]
    peak_events = max(item[3] for item in duration_ties)
    tied = [item for item in duration_ties if abs(item[3] - peak_events) <= _TOLERANCE]
    dimensions = _ordered((dimension for dimension, _, _, _ in tied), DIMENSION_ORDER)
    scopes = _ordered((scope for _, scope, _, _ in tied), SCOPE_ORDER)
    if peak_duration == 0.0 and peak_events == 0.0:
        explanation = "None of the day's listening overlapped the compared public top lists."
    else:
        explanation = (
            f"Public-top concentration peaked for {_label(dimensions)} against the {_label(scopes)} reference: "
            f"{_percent(peak_duration)} of listening time and {_percent(peak_events)} of events."
        )
    return {
        "dimensions": dimensions,
        "scopes": scopes,
        "duration_share": peak_duration,
        "event_share": peak_events,
        "is_tie": len(tied) > 1,
        "explanation": explanation,
    }


def _headline(active_dimensions: list[str], strongest: dict[str, Any], concentration: dict[str, Any]) -> str:
    if concentration["duration_share"] == 0.0 and concentration["event_share"] == 0.0:
        if active_dimensions == ["podcasts"]:
            return "A podcast day outside the public top lists"
        return "A distinctly personal listening day"
    if active_dimensions == ["podcasts"]:
        return f"Podcasts aligned most with the {_label(strongest['scopes'])} public list"
    return f"{_label(strongest['dimensions']).capitalize()} led the day's public-list alignment"


def _activity_totals(active: pd.DataFrame) -> tuple[int, float]:
    by_dimension = active.set_index("dimension")
    music_rows = by_dimension.loc[by_dimension.index.intersection(["artists", "tracks"])]
    podcast_rows = by_dimension.loc[by_dimension.index.intersection(["podcasts"])]
    music_events = _number(pd.to_numeric(music_rows.get("event_count"), errors="coerce").fillna(0).max())
    music_duration = _number(pd.to_numeric(music_rows.get("duration_minutes"), errors="coerce").fillna(0).max())
    podcast_events = pd.to_numeric(podcast_rows.get("event_count"), errors="coerce").fillna(0).sum()
    podcast_duration = pd.to_numeric(podcast_rows.get("duration_minutes"), errors="coerce").fillna(0).sum()
    return int(music_events + podcast_events), float(music_duration + podcast_duration)


def _record_for_period(
    mart_rows: pd.DataFrame,
    detailed_rows: pd.DataFrame,
    *,
    period_type: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    active = mart_rows[pd.to_numeric(mart_rows["event_count"], errors="coerce").fillna(0) > 0].copy()
    active_dimensions = _ordered(active["dimension"].astype(str), DIMENSION_ORDER)
    alignments = _ordered(
        detailed_rows["reference_alignment"].dropna().astype(str),
        ("date_aligned", "historical_projection", "post_window_projection"),
    )

    if active.empty:
        return {
            "period_type": period_type,
            "period_start": period_start,
            "period_end": period_end,
            "headline": "No listening activity to compare",
            "concise_summary": "The comparison frames contain no listening events for this period.",
            "strongest_alignment": None,
            "most_distinctive_dimension": None,
            "notable_public_top_concentration": None,
            "reference_alignment": alignments,
            "caveats": _alignment_caveats(alignments) + ["No active listening dimensions were available."],
        }

    strongest = _strongest_alignment(active)
    distinctive = _most_distinctive_dimension(active)
    concentration = _concentration(active)
    event_count, duration = _activity_totals(active)
    overlap_count = int(
        pd.to_numeric(detailed_rows["shared_public_entity_count"], errors="coerce").fillna(0).sum()
    )
    caveats = _alignment_caveats(alignments)
    if overlap_count == 0:
        caveats.append("No entities overlapped the compared public top lists.")
    caveats.append("Public-top shares describe this listener's activity, not public audience share.")

    return {
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        "headline": _headline(active_dimensions, strongest, concentration),
        "concise_summary": (
            f"{event_count} events across {_label(active_dimensions)} totaled {duration:.1f} minutes. "
            f"{strongest['explanation']} {concentration['explanation']}"
        ),
        "strongest_alignment": strongest,
        "most_distinctive_dimension": distinctive,
        "notable_public_top_concentration": concentration,
        "reference_alignment": alignments,
        "caveats": caveats,
    }


def build_daily_listening_narratives(
    daily_detailed: pd.DataFrame,
    daily_mart: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build deterministic, evidence-carrying public-comparison narratives for each listening day."""
    _validate_frames(daily_detailed, daily_mart)
    records: list[dict[str, Any]] = []
    dates = sorted(set(daily_detailed["listening_date"].astype(str)) | set(daily_mart["listening_date"].astype(str)))
    for listening_date in dates:
        mart_rows = daily_mart[daily_mart["listening_date"].astype(str) == listening_date]
        detailed_rows = daily_detailed[daily_detailed["listening_date"].astype(str) == listening_date]
        record = _record_for_period(
            mart_rows,
            detailed_rows,
            period_type="day",
            period_start=listening_date,
            period_end=listening_date,
        )
        record["listening_date"] = listening_date
        records.append(record)
    return records


def _weekly_mart(daily_mart: pd.DataFrame) -> pd.DataFrame:
    frame = daily_mart.copy()
    frame["_date"] = pd.to_datetime(frame["listening_date"], errors="coerce")
    frame = frame[frame["_date"].notna()].copy()
    frame["week_start"] = (frame["_date"] - pd.to_timedelta(frame["_date"].dt.weekday, unit="D")).dt.date.astype(str)
    metric_columns = [
        "global_similarity",
        "united_states_similarity",
        "global_event_share_on_public_top",
        "united_states_event_share_on_public_top",
        "global_duration_share_on_public_top",
        "united_states_duration_share_on_public_top",
    ]
    rows: list[dict[str, Any]] = []
    for (week_start, dimension), group in frame.groupby(["week_start", "dimension"], sort=True):
        events = pd.to_numeric(group["event_count"], errors="coerce").fillna(0.0)
        durations = pd.to_numeric(group["duration_minutes"], errors="coerce").fillna(0.0)
        row: dict[str, Any] = {
            "week_start": str(week_start),
            "dimension": str(dimension),
            "event_count": int(events.sum()),
            "duration_minutes": float(durations.sum()),
        }
        for column in metric_columns:
            weights = durations if "duration_share" in column else events
            values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
            row[column] = float((values * weights).sum() / weights.sum()) if weights.sum() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_weekly_listening_narratives(
    daily_detailed: pd.DataFrame,
    daily_mart: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Roll daily comparison evidence into Monday-based weekly narrative records."""
    _validate_frames(daily_detailed, daily_mart)
    mart = _weekly_mart(daily_mart)
    if mart.empty:
        return []
    detailed = daily_detailed.copy()
    detailed["_date"] = pd.to_datetime(detailed["listening_date"], errors="coerce")
    detailed = detailed[detailed["_date"].notna()].copy()
    detailed["week_start"] = (
        detailed["_date"] - pd.to_timedelta(detailed["_date"].dt.weekday, unit="D")
    ).dt.date.astype(str)

    records: list[dict[str, Any]] = []
    for week_start in sorted(mart["week_start"].unique()):
        start = pd.Timestamp(week_start)
        week_end = (start + pd.Timedelta(days=6)).date().isoformat()
        record = _record_for_period(
            mart[mart["week_start"] == week_start],
            detailed[detailed["week_start"] == week_start],
            period_type="week",
            period_start=str(week_start),
            period_end=week_end,
        )
        record["week_start"] = str(week_start)
        record["week_end"] = week_end
        records.append(record)
    return records


def build_public_listening_narratives(
    daily_detailed: pd.DataFrame,
    daily_mart: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Return daily narratives and their weekly rollups in one serializable payload."""
    return {
        "daily": build_daily_listening_narratives(daily_detailed, daily_mart),
        "weekly": build_weekly_listening_narratives(daily_detailed, daily_mart),
    }
