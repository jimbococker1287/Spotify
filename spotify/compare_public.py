from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .data import load_streaming_history
from .env import load_local_env
from .lastfm import LastFmArtistChartRow, LastFmClient, LastFmError
from .public_listening_reference import (
    PublicListeningReference,
    PublicListeningRow,
    spotify_wrapped_reference,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.compare_public",
        description="Compare your listening history to official Spotify Wrapped lists or Last.fm charts.",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for comparison artifacts.")
    parser.add_argument(
        "--provider",
        choices=("spotify-wrapped", "lastfm"),
        default="spotify-wrapped",
        help="Public reference provider. Spotify Wrapped works offline; Last.fm requires LASTFM_API_KEY.",
    )
    parser.add_argument("--year", type=int, default=2025, help="Spotify Wrapped edition to compare.")
    parser.add_argument("--lookback-days", type=int, default=180, help="Recent listening window to compare.")
    parser.add_argument("--top-n", type=int, default=10, help="How many top entries to compare per dimension.")
    parser.add_argument(
        "--scope",
        choices=("country", "global", "both"),
        default="both",
        help="Compare against the U.S., global, or both public lists.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="United States",
        help="Country name for the public chart when --scope country is used.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files while rebuilding the listening summary.",
    )
    parser.add_argument(
        "--genre",
        action="append",
        default=[],
        help="Optional Last.fm tag proxy such as hip-hop. Repeat for multiple genres; requires LASTFM_API_KEY.",
    )
    parser.add_argument(
        "--genre-artist-limit",
        type=int,
        default=25,
        help="How many of your top artists to tag through Last.fm for each genre proxy.",
    )
    return parser.parse_args()


def _normalize_artist_name(value: str) -> str:
    return _normalize_name(value)


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_value = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def _history_artist_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ts", "master_metadata_album_artist_name"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise RuntimeError(f"Streaming history is missing required columns: {', '.join(sorted(missing))}")

    frame = df[["ts", "master_metadata_album_artist_name"]].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    frame["artist_name"] = frame["master_metadata_album_artist_name"].fillna("").astype(str).str.strip()
    frame = frame[frame["ts"].notna()]
    frame = frame[frame["artist_name"] != ""]
    frame["artist_key"] = frame["artist_name"].map(_normalize_artist_name)
    frame = frame[frame["artist_key"] != ""]
    return frame.sort_values("ts").reset_index(drop=True)


def build_public_comparison(
    history_df: pd.DataFrame,
    public_rows: list[LastFmArtistChartRow],
    *,
    lookback_days: int,
    top_n: int,
    scope: str,
    country: str | None,
) -> dict[str, object]:
    if not public_rows:
        raise RuntimeError("No public artist rows were provided for comparison.")

    history = _history_artist_frame(history_df)
    if history.empty:
        raise RuntimeError("No artist listening events were found in the streaming history.")

    history_end_ts = history["ts"].max()
    if pd.isna(history_end_ts):
        raise RuntimeError("Could not determine the latest timestamp in streaming history.")
    cutoff_ts = history_end_ts - pd.Timedelta(days=max(1, int(lookback_days)))
    recent = history[history["ts"] >= cutoff_ts].copy()
    if recent.empty:
        raise RuntimeError("No listening events fall inside the selected recent window.")

    public_rows = public_rows[: max(1, int(top_n))]
    public_by_key = {_normalize_artist_name(row.name): row for row in public_rows}
    public_keys = set(public_by_key)

    recent_artist_counts = recent["artist_name"].value_counts()
    recent_unique_artist_count = int(recent["artist_key"].nunique())
    recent_top_artists = recent_artist_counts.head(max(1, int(top_n)))
    recent_top_rows: list[dict[str, object]] = []
    for artist_name, plays in recent_top_artists.items():
        artist_key = _normalize_artist_name(str(artist_name))
        recent_top_rows.append(
            {
                "artist_name": str(artist_name),
                "artist_key": artist_key,
                "plays": int(plays),
            }
        )

    overlap_rows: list[dict[str, object]] = []
    distinctive_rows: list[dict[str, object]] = []
    for row in recent_top_rows:
        public_row = public_by_key.get(str(row["artist_key"]))
        if public_row is None:
            distinctive_rows.append(
                {
                    "artist_name": row["artist_name"],
                    "your_plays": row["plays"],
                }
            )
            continue
        overlap_rows.append(
            {
                "artist_name": row["artist_name"],
                "your_plays": row["plays"],
                "public_rank": int(public_row.rank),
                "public_listeners": public_row.listeners,
                "public_playcount": public_row.playcount,
                "public_url": public_row.url,
            }
        )

    recent_public_mask = recent["artist_key"].isin(public_keys)
    public_artists_new_to_you: list[dict[str, object]] = []
    heard_artist_keys = set(recent["artist_key"].unique().tolist())
    for public_row in public_rows:
        artist_key = _normalize_artist_name(public_row.name)
        if artist_key in heard_artist_keys:
            continue
        public_artists_new_to_you.append(
            {
                "artist_name": public_row.name,
                "public_rank": int(public_row.rank),
                "public_listeners": public_row.listeners,
                "public_playcount": public_row.playcount,
                "public_url": public_row.url,
            }
        )

    summary = {
        "baseline": {
            "provider": "Last.fm",
            "scope": scope,
            "country": country if scope == "country" else None,
            "top_n": len(public_rows),
        },
        "history_window": {
            "lookback_days": max(1, int(lookback_days)),
            "end_ts": history_end_ts.isoformat(),
            "cutoff_ts": cutoff_ts.isoformat(),
            "recent_streams": int(len(recent)),
            "recent_unique_artists": recent_unique_artist_count,
        },
        "overlap": {
            "shared_artist_count": int(len(overlap_rows)),
            "your_top_artist_overlap_ratio": float(len(overlap_rows) / max(1, len(recent_top_rows))),
            "public_top_artist_overlap_ratio": float(len(overlap_rows) / max(1, len(public_rows))),
            "recent_play_share_on_public_artists": float(recent_public_mask.mean()),
        },
        "shared_artists": overlap_rows,
        "your_distinctive_artists": distinctive_rows[:10],
        "public_artists_new_to_you": public_artists_new_to_you[:10],
    }
    return summary


def write_public_comparison(
    summary: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    baseline = summary["baseline"]
    window = summary["history_window"]
    overlap = summary["overlap"]
    scope = str(baseline["scope"])
    country = str(baseline.get("country") or "global").strip().lower().replace(" ", "-")
    lookback_days = int(window["lookback_days"])

    analysis_dir = output_dir / "analysis" / "public_compare"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lastfm_{scope}_{country}_{lookback_days}d"
    json_path = analysis_dir / f"{stem}.json"
    md_path = analysis_dir / f"{stem}.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Public Listening Comparison",
        "",
        f"- Provider: `{baseline['provider']}`",
        f"- Scope: `{scope}`",
        f"- Country: `{baseline.get('country') or 'global'}`",
        f"- Recent window: last `{lookback_days}` days ending `{window['end_ts']}`",
        f"- Recent streams: `{window['recent_streams']}`",
        f"- Recent unique artists: `{window['recent_unique_artists']}`",
        "",
        "## Similarity Snapshot",
        "",
        f"- Shared artists between your recent top list and the public top list: `{overlap['shared_artist_count']}`",
        f"- Overlap ratio across your recent top artists: `{float(overlap['your_top_artist_overlap_ratio']):.1%}`",
        f"- Overlap ratio across the public top artists: `{float(overlap['public_top_artist_overlap_ratio']):.1%}`",
        f"- Share of your recent plays spent on public-top artists: `{float(overlap['recent_play_share_on_public_artists']):.1%}`",
        "",
        "## Shared Artists",
        "",
    ]

    shared = summary.get("shared_artists", [])
    if isinstance(shared, list) and shared:
        for row in shared[:10]:
            if not isinstance(row, dict):
                continue
            listeners = row.get("public_listeners")
            lines.append(
                f"- {row.get('artist_name')}: your plays `{row.get('your_plays')}`, public rank "
                f"`#{row.get('public_rank')}`, listeners `{listeners if listeners is not None else 'n/a'}`"
            )
    else:
        lines.append("- No shared artists in the selected top lists.")

    lines.extend(
        [
            "",
            "## Your Distinctive Artists",
            "",
        ]
    )

    distinctive = summary.get("your_distinctive_artists", [])
    if isinstance(distinctive, list) and distinctive:
        for row in distinctive[:10]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('artist_name')}: your plays `{row.get('your_plays')}`")
    else:
        lines.append("- Your recent top artists were fully represented in the public chart.")

    lines.extend(
        [
            "",
            "## Public Artists New To You",
            "",
        ]
    )
    new_to_you = summary.get("public_artists_new_to_you", [])
    if isinstance(new_to_you, list) and new_to_you:
        for row in new_to_you[:10]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('artist_name')}: public rank `#{row.get('public_rank')}`, "
                f"listeners `{row.get('public_listeners') if row.get('public_listeners') is not None else 'n/a'}`"
            )
    else:
        lines.append("- You have already listened to every artist in the selected public chart slice.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _text_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].fillna("").astype(str).str.strip()


def _listening_events(
    history_df: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "ts" not in history_df.columns:
        raise RuntimeError("Streaming history is missing required column: ts")

    all_timestamps = pd.to_datetime(history_df["ts"], errors="coerce", utc=True)
    valid_timestamps = all_timestamps.dropna()
    if valid_timestamps.empty:
        raise RuntimeError("No valid timestamps were found in the streaming history.")

    frame = pd.DataFrame(index=history_df.index)
    frame["ts"] = all_timestamps
    if "ms_played" in history_df.columns:
        frame["ms_played"] = pd.to_numeric(history_df["ms_played"], errors="coerce").fillna(0).clip(lower=0)
    else:
        frame["ms_played"] = 0.0
    frame["artist_name"] = _text_column(history_df, "master_metadata_album_artist_name")
    frame["track_name"] = _text_column(history_df, "master_metadata_track_name")
    frame["podcast_show_name"] = _text_column(history_df, "episode_show_name")
    frame["podcast_episode_name"] = _text_column(history_df, "episode_name")
    frame["artist_key"] = frame["artist_name"].map(_normalize_name)
    frame["track_name_key"] = frame["track_name"].map(_normalize_name)
    frame["podcast_key"] = frame["podcast_show_name"].map(_normalize_name)
    frame["track_key"] = frame["track_name_key"] + "::" + frame["artist_key"]
    frame["media_type"] = "other"
    frame.loc[(frame["artist_key"] != "") & (frame["track_name_key"] != ""), "media_type"] = "music"
    frame.loc[frame["podcast_key"] != "", "media_type"] = "podcast"

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_exclusive_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    frame = frame[
        frame["ts"].notna()
        & (frame["ts"] >= start_ts)
        & (frame["ts"] < end_exclusive_ts)
        & (frame["media_type"] != "other")
    ].copy()
    frame["month"] = frame["ts"].dt.strftime("%Y-%m")

    coverage = {
        "available_start_ts": valid_timestamps.min().isoformat(),
        "available_end_ts": valid_timestamps.max().isoformat(),
        "selected_start_date": start_date.isoformat(),
        "selected_end_date": end_date.isoformat(),
        "selected_events": int(len(frame)),
        "selected_duration_minutes": float(frame["ms_played"].sum() / 60_000.0),
    }
    return frame.reset_index(drop=True), coverage


def _public_row_payload(row: PublicListeningRow) -> dict[str, object]:
    return {
        "rank": int(row.rank),
        "name": row.name,
        "artists": list(row.artists),
    }


def _reference_match(
    *,
    dimension: str,
    personal_row: pd.Series,
    public_row: PublicListeningRow,
) -> bool:
    if dimension == "artists":
        return str(personal_row["entity_key"]) == _normalize_name(public_row.name)
    if dimension == "podcasts":
        return str(personal_row["entity_key"]) == _normalize_name(public_row.name)
    if dimension == "tracks":
        if str(personal_row["track_name_key"]) != _normalize_name(public_row.name):
            return False
        personal_artist = str(personal_row["artist_key"])
        return any(personal_artist == _normalize_name(artist) for artist in public_row.artists)
    raise ValueError(f"Unsupported comparison dimension: {dimension}")


def _rank_weighted_jaccard(
    personal_top: pd.DataFrame,
    public_rows: tuple[PublicListeningRow, ...],
    matched_public_rank_by_key: dict[str, int],
) -> float:
    personal_weights: dict[str, float] = {}
    for personal_rank, (_, row) in enumerate(personal_top.iterrows(), start=1):
        entity_key = str(row["entity_key"])
        public_rank = matched_public_rank_by_key.get(entity_key)
        comparison_key = f"public:{public_rank}" if public_rank is not None else f"personal:{entity_key}"
        personal_weights[comparison_key] = 1.0 / personal_rank

    public_weights = {f"public:{int(row.rank)}": 1.0 / int(row.rank) for row in public_rows}
    union_keys = set(personal_weights) | set(public_weights)
    numerator = sum(min(personal_weights.get(key, 0.0), public_weights.get(key, 0.0)) for key in union_keys)
    denominator = sum(max(personal_weights.get(key, 0.0), public_weights.get(key, 0.0)) for key in union_keys)
    return float(numerator / denominator) if denominator else 0.0


def _similarity_label(score: float) -> str:
    if score >= 0.50:
        return "high alignment"
    if score >= 0.25:
        return "moderate alignment"
    if score >= 0.10:
        return "light alignment"
    return "distinct"


def _aggregate_dimension(events: pd.DataFrame, dimension: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if dimension == "artists":
        dimension_events = events[events["media_type"] == "music"].copy()
        dimension_events["entity_key"] = dimension_events["artist_key"]
        dimension_events["entity_name"] = dimension_events["artist_name"]
        dimension_events["entity_detail"] = ""
        ranking_basis = "stream count"
    elif dimension == "tracks":
        dimension_events = events[events["media_type"] == "music"].copy()
        dimension_events["entity_key"] = dimension_events["track_key"]
        dimension_events["entity_name"] = dimension_events["track_name"]
        dimension_events["entity_detail"] = dimension_events["artist_name"]
        ranking_basis = "stream count"
    elif dimension == "podcasts":
        dimension_events = events[events["media_type"] == "podcast"].copy()
        dimension_events["entity_key"] = dimension_events["podcast_key"]
        dimension_events["entity_name"] = dimension_events["podcast_show_name"]
        dimension_events["entity_detail"] = ""
        ranking_basis = "listening duration"
    else:
        raise ValueError(f"Unsupported comparison dimension: {dimension}")

    if dimension_events.empty:
        return dimension_events, pd.DataFrame(), ranking_basis

    aggregation = {
        "entity_name": ("entity_name", "first"),
        "entity_detail": ("entity_detail", "first"),
        "event_count": ("entity_key", "size"),
        "duration_ms": ("ms_played", "sum"),
    }
    if dimension == "tracks":
        aggregation["track_name_key"] = ("track_name_key", "first")
        aggregation["artist_key"] = ("artist_key", "first")
    grouped = dimension_events.groupby("entity_key", as_index=False).agg(**aggregation)
    sort_columns = ["duration_ms", "event_count"] if dimension == "podcasts" else ["event_count", "duration_ms"]
    grouped = grouped.sort_values(sort_columns, ascending=False, kind="stable").reset_index(drop=True)
    return dimension_events, grouped, ranking_basis


def _dimension_comparison(
    events: pd.DataFrame,
    *,
    dimension: str,
    public_rows: tuple[PublicListeningRow, ...],
    top_n: int,
) -> dict[str, object]:
    dimension_events, grouped, ranking_basis = _aggregate_dimension(events, dimension)
    selected_public_rows = public_rows[: max(1, int(top_n))]
    public_payload = [_public_row_payload(row) for row in selected_public_rows]
    if grouped.empty:
        return {
            "status": "no_personal_events",
            "ranking_basis": ranking_basis,
            "personal_event_count": 0,
            "personal_duration_minutes": 0.0,
            "metrics": {
                "shared_top_count": 0,
                "personal_top_overlap_ratio": 0.0,
                "public_top_overlap_ratio": 0.0,
                "event_share_on_public_top": 0.0,
                "duration_share_on_public_top": 0.0,
                "rank_weighted_jaccard_similarity": 0.0,
                "similarity_label": "no personal data",
            },
            "personal_top": [],
            "public_top": public_payload,
            "shared_top": [],
            "personal_distinctive": [],
            "public_new_to_you": public_payload,
            "monthly_alignment": [],
        }

    matched_public_rank_by_key: dict[str, int] = {}
    matched_public_by_key: dict[str, PublicListeningRow] = {}
    heard_public_ranks: set[int] = set()
    for _, personal_row in grouped.iterrows():
        for public_row in selected_public_rows:
            if _reference_match(dimension=dimension, personal_row=personal_row, public_row=public_row):
                entity_key = str(personal_row["entity_key"])
                matched_public_rank_by_key[entity_key] = int(public_row.rank)
                matched_public_by_key[entity_key] = public_row
                heard_public_ranks.add(int(public_row.rank))
                break

    personal_top = grouped.head(max(1, int(top_n))).copy()
    personal_top_payload: list[dict[str, object]] = []
    shared_payload: list[dict[str, object]] = []
    distinctive_payload: list[dict[str, object]] = []
    for personal_rank, (_, row) in enumerate(personal_top.iterrows(), start=1):
        entity_key = str(row["entity_key"])
        public_row = matched_public_by_key.get(entity_key)
        payload = {
            "personal_rank": personal_rank,
            "name": str(row["entity_name"]),
            "detail": str(row["entity_detail"]),
            "events": int(row["event_count"]),
            "duration_minutes": float(row["duration_ms"] / 60_000.0),
            "public_rank": int(public_row.rank) if public_row is not None else None,
        }
        personal_top_payload.append(payload)
        if public_row is None:
            distinctive_payload.append(payload)
        else:
            shared_payload.append(payload)

    matched_event_mask = dimension_events["entity_key"].isin(matched_public_rank_by_key)
    similarity = _rank_weighted_jaccard(
        personal_top,
        selected_public_rows,
        matched_public_rank_by_key,
    )
    total_duration = float(dimension_events["ms_played"].sum())
    monthly_payload: list[dict[str, object]] = []
    monthly_frame = dimension_events.assign(on_public_top=matched_event_mask.to_numpy())
    for month, month_rows in monthly_frame.groupby("month", sort=True):
        month_duration = float(month_rows["ms_played"].sum())
        matched_rows = month_rows[month_rows["on_public_top"]]
        monthly_payload.append(
            {
                "month": str(month),
                "events": int(len(month_rows)),
                "duration_minutes": float(month_duration / 60_000.0),
                "event_share_on_public_top": float(matched_rows.shape[0] / max(1, len(month_rows))),
                "duration_share_on_public_top": (
                    float(matched_rows["ms_played"].sum() / month_duration) if month_duration else 0.0
                ),
            }
        )

    public_new_to_you = [
        _public_row_payload(row) for row in selected_public_rows if int(row.rank) not in heard_public_ranks
    ]
    metrics = {
        "shared_top_count": int(len(shared_payload)),
        "personal_top_overlap_ratio": float(len(shared_payload) / max(1, len(personal_top))),
        "public_top_overlap_ratio": float(len(shared_payload) / max(1, len(selected_public_rows))),
        "event_share_on_public_top": float(matched_event_mask.mean()),
        "duration_share_on_public_top": (
            float(dimension_events.loc[matched_event_mask, "ms_played"].sum() / total_duration)
            if total_duration
            else 0.0
        ),
        "rank_weighted_jaccard_similarity": similarity,
        "similarity_label": _similarity_label(similarity),
    }
    return {
        "status": "ok",
        "ranking_basis": ranking_basis,
        "personal_event_count": int(len(dimension_events)),
        "personal_duration_minutes": float(total_duration / 60_000.0),
        "metrics": metrics,
        "personal_top": personal_top_payload,
        "public_top": public_payload,
        "shared_top": shared_payload,
        "personal_distinctive": distinctive_payload[:10],
        "public_new_to_you": public_new_to_you[:10],
        "monthly_alignment": monthly_payload,
    }


def _scope_comparison(
    events: pd.DataFrame,
    reference: PublicListeningReference,
    *,
    top_n: int,
) -> dict[str, object]:
    dimensions = {
        dimension: _dimension_comparison(
            events,
            dimension=dimension,
            public_rows=rows,
            top_n=top_n,
        )
        for dimension, rows in reference.dimensions.items()
    }
    scores = [
        float(result["metrics"]["rank_weighted_jaccard_similarity"])
        for result in dimensions.values()
        if result["status"] == "ok"
    ]
    overall_score = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "baseline": {
            "provider": reference.provider,
            "edition": reference.edition,
            "scope": reference.scope,
            "country": reference.country,
            "published_date": reference.published_date.isoformat(),
            "source_url": reference.source_url,
            "methodology_url": reference.methodology_url,
        },
        "overall_rank_similarity": overall_score,
        "overall_similarity_label": _similarity_label(overall_score),
        "dimensions": dimensions,
    }


def _personal_media_mix(events: pd.DataFrame) -> dict[str, object]:
    total_duration = float(events["ms_played"].sum())
    rows: dict[str, object] = {}
    for media_type in ("music", "podcast"):
        media_rows = events[events["media_type"] == media_type]
        duration = float(media_rows["ms_played"].sum())
        rows[media_type] = {
            "events": int(len(media_rows)),
            "duration_minutes": float(duration / 60_000.0),
            "event_share": float(len(media_rows) / max(1, len(events))),
            "duration_share": float(duration / total_duration) if total_duration else 0.0,
        }
    return rows


def build_spotify_wrapped_comparison(
    history_df: pd.DataFrame,
    *,
    edition: int = 2025,
    scopes: tuple[str, ...] = ("global", "country"),
    top_n: int = 10,
    genre_comparisons: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    references = [spotify_wrapped_reference(edition=edition, scope=scope) for scope in scopes]
    if not references:
        raise RuntimeError("At least one Spotify Wrapped comparison scope is required.")
    first_reference = references[0]
    events, coverage = _listening_events(
        history_df,
        start_date=first_reference.window_start,
        end_date=first_reference.window_end,
    )
    scope_results: dict[str, object] = {}
    for reference in references:
        scope_key = "global" if reference.scope == "global" else "united_states"
        scope_results[scope_key] = _scope_comparison(events, reference, top_n=top_n)

    scores = {
        scope: float(result["overall_rank_similarity"])
        for scope, result in scope_results.items()
        if isinstance(result, dict)
    }
    closest_scope = max(scores, key=scores.get) if scores else None
    dimension_scope_summary: dict[str, object] = {}
    if {"global", "united_states"}.issubset(scope_results):
        for dimension in ("artists", "tracks", "podcasts"):
            global_score = float(
                scope_results["global"]["dimensions"][dimension]["metrics"]["rank_weighted_jaccard_similarity"]
            )
            us_score = float(
                scope_results["united_states"]["dimensions"][dimension]["metrics"]["rank_weighted_jaccard_similarity"]
            )
            dimension_scope_summary[dimension] = {
                "closer_scope": "united_states" if us_score > global_score else "global",
                "global_score": global_score,
                "united_states_score": us_score,
                "united_states_minus_global": us_score - global_score,
            }

    return {
        "report_type": "public_listening_similarity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_window": {
            **coverage,
            "public_reference_window": "January through mid-November",
            "window_end_is_approximate": first_reference.window_end_is_approximate,
            "window_note": (
                "Spotify states that 2025 Wrapped covers January through mid-November but does not publish "
                "an exact cutoff date. This report uses November 15 as an explicit operational cutoff."
            ),
        },
        "personal_media_mix": _personal_media_mix(events),
        "scope_comparisons": scope_results,
        "relative_scope_summary": {
            "closest_scope": closest_scope,
            "scope_scores": scores,
            "dimensions": dimension_scope_summary,
        },
        "genre_comparisons": genre_comparisons or [],
        "methodology": {
            "duration_interpretation": (
                "Public duration distributions are not published. Duration metrics report the share of your "
                "own listening time spent on entities in each public top list."
            ),
            "rank_similarity": (
                "Rank-weighted Jaccard compares reciprocal-rank weights across your top list and the public list."
            ),
            "genre_interpretation": (
                "Genre comparisons, when requested, use Last.fm community tags as a proxy and are not Spotify "
                "genre or market-share estimates."
            ),
            "policy_boundary": (
                "The report uses Spotify's published aggregate Wrapped lists, not Spotify Platform content or "
                "API-derived listenership benchmarks."
            ),
        },
    }


def build_daily_spotify_wrapped_comparison(
    history_df: pd.DataFrame,
    *,
    edition: int = 2025,
    scopes: tuple[str, ...] = ("global", "country"),
    top_n: int = 10,
) -> pd.DataFrame:
    columns = [
        "listening_date",
        "reference_edition",
        "reference_scope",
        "reference_country",
        "reference_alignment",
        "dimension",
        "ranking_basis",
        "event_count",
        "duration_minutes",
        "unique_entity_count",
        "shared_public_entity_count",
        "event_share_on_public_top",
        "duration_share_on_public_top",
        "rank_weighted_jaccard_similarity",
        "similarity_label",
        "personal_top_entity",
        "personal_top_entity_detail",
        "personal_top_entity_public_rank",
        "public_source_url",
    ]
    if history_df.empty or "ts" not in history_df.columns:
        return pd.DataFrame(columns=columns)
    timestamps = pd.to_datetime(history_df["ts"], errors="coerce", utc=True).dropna()
    if timestamps.empty:
        return pd.DataFrame(columns=columns)

    start_date = timestamps.min().date()
    end_date = timestamps.max().date()
    events, _ = _listening_events(history_df, start_date=start_date, end_date=end_date)
    if events.empty:
        return pd.DataFrame(columns=columns)

    active_dates = pd.DataFrame({"listening_date": sorted(events["ts"].dt.date.astype(str).unique())})
    output_rows: list[dict[str, object]] = []
    references = [spotify_wrapped_reference(edition=edition, scope=scope) for scope in scopes]

    for reference in references:
        scope_name = "global" if reference.scope == "global" else "united_states"
        for dimension, reference_rows in reference.dimensions.items():
            dimension_events, grouped, ranking_basis = _aggregate_dimension(events, dimension)
            selected_public_rows = reference_rows[: max(1, int(top_n))]
            public_rank_by_artist = {_normalize_name(row.name): int(row.rank) for row in selected_public_rows}
            public_rank_by_track_artist = {
                (_normalize_name(row.name), _normalize_name(artist)): int(row.rank)
                for row in selected_public_rows
                for artist in row.artists
            }

            dimension_events = dimension_events.copy()
            dimension_events["listening_date"] = dimension_events["ts"].dt.date.astype(str)
            if dimension == "tracks":
                dimension_events["public_rank"] = [
                    public_rank_by_track_artist.get((track_key, artist_key))
                    for track_key, artist_key in zip(
                        dimension_events["track_name_key"],
                        dimension_events["artist_key"],
                    )
                ]
            else:
                dimension_events["public_rank"] = dimension_events["entity_key"].map(public_rank_by_artist)

            totals = dimension_events.groupby("listening_date", as_index=False).agg(
                event_count=("entity_key", "size"),
                duration_ms=("ms_played", "sum"),
                unique_entity_count=("entity_key", "nunique"),
            )
            matched = dimension_events[dimension_events["public_rank"].notna()]
            matched_totals = matched.groupby("listening_date", as_index=False).agg(
                public_event_count=("entity_key", "size"),
                public_duration_ms=("ms_played", "sum"),
                shared_public_entity_count=("entity_key", "nunique"),
            )

            if not grouped.empty:
                grouped = grouped.copy()
                entity_dates = dimension_events[["listening_date", "entity_key"]].drop_duplicates()
                daily_entities = entity_dates.merge(grouped, on="entity_key", how="left")
                daily_counts = dimension_events.groupby(["listening_date", "entity_key"], as_index=False).agg(
                    event_count=("entity_key", "size"), duration_ms=("ms_played", "sum")
                )
                daily_entities = daily_entities.drop(columns=["event_count", "duration_ms"]).merge(
                    daily_counts,
                    on=["listening_date", "entity_key"],
                    how="left",
                )
                if dimension == "tracks":
                    daily_entities["public_rank"] = [
                        public_rank_by_track_artist.get((track_key, artist_key))
                        for track_key, artist_key in zip(
                            daily_entities["track_name_key"],
                            daily_entities["artist_key"],
                        )
                    ]
                else:
                    daily_entities["public_rank"] = daily_entities["entity_key"].map(public_rank_by_artist)
                sort_columns = (
                    ["listening_date", "duration_ms", "event_count"]
                    if dimension == "podcasts"
                    else ["listening_date", "event_count", "duration_ms"]
                )
                daily_entities = daily_entities.sort_values(
                    sort_columns,
                    ascending=[True, False, False],
                    kind="stable",
                )
                daily_entities["personal_rank"] = daily_entities.groupby("listening_date").cumcount() + 1
                top_entities = daily_entities[daily_entities["personal_rank"] == 1][
                    [
                        "listening_date",
                        "entity_name",
                        "entity_detail",
                        "public_rank",
                    ]
                ].rename(
                    columns={
                        "entity_name": "personal_top_entity",
                        "entity_detail": "personal_top_entity_detail",
                        "public_rank": "personal_top_entity_public_rank",
                    }
                )

                similarity_rows: list[dict[str, object]] = []
                public_weights = {int(row.rank): 1.0 / int(row.rank) for row in selected_public_rows}
                for listening_date, day_entities in daily_entities.groupby("listening_date", sort=False):
                    personal_weights: dict[str, float] = {}
                    for row in day_entities.head(max(1, int(top_n))).itertuples(index=False):
                        public_rank = getattr(row, "public_rank")
                        key = (
                            f"public:{int(public_rank)}"
                            if pd.notna(public_rank)
                            else f"personal:{getattr(row, 'entity_key')}"
                        )
                        personal_weights[key] = 1.0 / int(getattr(row, "personal_rank"))
                    public_weight_map = {f"public:{rank}": weight for rank, weight in public_weights.items()}
                    union_keys = set(personal_weights) | set(public_weight_map)
                    numerator = sum(
                        min(personal_weights.get(key, 0.0), public_weight_map.get(key, 0.0)) for key in union_keys
                    )
                    denominator = sum(
                        max(personal_weights.get(key, 0.0), public_weight_map.get(key, 0.0)) for key in union_keys
                    )
                    similarity_rows.append(
                        {
                            "listening_date": str(listening_date),
                            "rank_weighted_jaccard_similarity": (
                                float(numerator / denominator) if denominator else 0.0
                            ),
                        }
                    )
                similarities = pd.DataFrame(similarity_rows)
            else:
                top_entities = pd.DataFrame(
                    columns=[
                        "listening_date",
                        "personal_top_entity",
                        "personal_top_entity_detail",
                        "personal_top_entity_public_rank",
                    ]
                )
                similarities = pd.DataFrame(columns=["listening_date", "rank_weighted_jaccard_similarity"])

            daily = (
                active_dates.merge(totals, on="listening_date", how="left")
                .merge(matched_totals, on="listening_date", how="left")
                .merge(top_entities, on="listening_date", how="left")
                .merge(similarities, on="listening_date", how="left")
            )
            numeric_zero_columns = [
                "event_count",
                "duration_ms",
                "unique_entity_count",
                "public_event_count",
                "public_duration_ms",
                "shared_public_entity_count",
                "rank_weighted_jaccard_similarity",
            ]
            for column in numeric_zero_columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
            daily["event_share_on_public_top"] = 0.0
            event_mask = daily["event_count"] > 0
            daily.loc[event_mask, "event_share_on_public_top"] = (
                daily.loc[event_mask, "public_event_count"] / daily.loc[event_mask, "event_count"]
            )
            daily["duration_share_on_public_top"] = 0.0
            duration_mask = daily["duration_ms"] > 0
            daily.loc[duration_mask, "duration_share_on_public_top"] = (
                daily.loc[duration_mask, "public_duration_ms"] / daily.loc[duration_mask, "duration_ms"]
            )

            for row in daily.to_dict(orient="records"):
                listening_date = date.fromisoformat(str(row["listening_date"]))
                if reference.window_start <= listening_date <= reference.window_end:
                    alignment = "date_aligned"
                elif listening_date < reference.window_start:
                    alignment = "historical_projection"
                else:
                    alignment = "post_window_projection"
                similarity = float(row["rank_weighted_jaccard_similarity"])
                output_rows.append(
                    {
                        "listening_date": str(row["listening_date"]),
                        "reference_edition": int(reference.edition),
                        "reference_scope": scope_name,
                        "reference_country": reference.country,
                        "reference_alignment": alignment,
                        "dimension": dimension,
                        "ranking_basis": ranking_basis,
                        "event_count": int(row["event_count"]),
                        "duration_minutes": float(row["duration_ms"] / 60_000.0),
                        "unique_entity_count": int(row["unique_entity_count"]),
                        "shared_public_entity_count": int(row["shared_public_entity_count"]),
                        "event_share_on_public_top": float(row["event_share_on_public_top"]),
                        "duration_share_on_public_top": float(row["duration_share_on_public_top"]),
                        "rank_weighted_jaccard_similarity": similarity,
                        "similarity_label": (
                            _similarity_label(similarity) if int(row["event_count"]) > 0 else "no personal data"
                        ),
                        "personal_top_entity": row.get("personal_top_entity"),
                        "personal_top_entity_detail": row.get("personal_top_entity_detail"),
                        "personal_top_entity_public_rank": (
                            int(row["personal_top_entity_public_rank"])
                            if pd.notna(row.get("personal_top_entity_public_rank"))
                            else None
                        ),
                        "public_source_url": reference.source_url,
                    }
                )

    return (
        pd.DataFrame(output_rows, columns=columns)
        .sort_values(["listening_date", "dimension", "reference_scope"])
        .reset_index(drop=True)
    )


def build_daily_public_similarity_mart(daily_comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "listening_date",
        "reference_edition",
        "reference_alignment",
        "dimension",
        "event_count",
        "duration_minutes",
        "unique_entity_count",
        "global_similarity",
        "united_states_similarity",
        "united_states_minus_global",
        "closer_scope",
        "global_event_share_on_public_top",
        "united_states_event_share_on_public_top",
        "global_duration_share_on_public_top",
        "united_states_duration_share_on_public_top",
        "personal_top_entity",
        "personal_top_entity_detail",
    ]
    if daily_comparison.empty:
        return pd.DataFrame(columns=columns)

    identity_columns = [
        "listening_date",
        "reference_edition",
        "reference_alignment",
        "dimension",
        "event_count",
        "duration_minutes",
        "unique_entity_count",
        "personal_top_entity",
        "personal_top_entity_detail",
    ]
    rename_map = {
        "rank_weighted_jaccard_similarity_global": "global_similarity",
        "rank_weighted_jaccard_similarity_united_states": "united_states_similarity",
        "event_share_on_public_top_global": "global_event_share_on_public_top",
        "event_share_on_public_top_united_states": "united_states_event_share_on_public_top",
        "duration_share_on_public_top_global": "global_duration_share_on_public_top",
        "duration_share_on_public_top_united_states": "united_states_duration_share_on_public_top",
    }
    global_rows = daily_comparison[daily_comparison["reference_scope"] == "global"][
        identity_columns
        + [
            "rank_weighted_jaccard_similarity",
            "event_share_on_public_top",
            "duration_share_on_public_top",
        ]
    ].rename(
        columns={
            "rank_weighted_jaccard_similarity": "rank_weighted_jaccard_similarity_global",
            "event_share_on_public_top": "event_share_on_public_top_global",
            "duration_share_on_public_top": "duration_share_on_public_top_global",
        }
    )
    us_rows = daily_comparison[daily_comparison["reference_scope"] == "united_states"][
        [
            "listening_date",
            "dimension",
            "rank_weighted_jaccard_similarity",
            "event_share_on_public_top",
            "duration_share_on_public_top",
        ]
    ].rename(
        columns={
            "rank_weighted_jaccard_similarity": "rank_weighted_jaccard_similarity_united_states",
            "event_share_on_public_top": "event_share_on_public_top_united_states",
            "duration_share_on_public_top": "duration_share_on_public_top_united_states",
        }
    )
    pivot = global_rows.merge(us_rows, on=["listening_date", "dimension"], how="outer").rename(columns=rename_map)
    for column in rename_map.values():
        if column not in pivot.columns:
            pivot[column] = 0.0
    pivot["united_states_minus_global"] = pivot["united_states_similarity"] - pivot["global_similarity"]
    pivot["closer_scope"] = "global"
    pivot.loc[pivot["united_states_similarity"] > pivot["global_similarity"], "closer_scope"] = "united_states"
    pivot.loc[
        pivot["united_states_similarity"].eq(pivot["global_similarity"]),
        "closer_scope",
    ] = "tie"
    return pivot[columns].sort_values(["listening_date", "dimension"]).reset_index(drop=True)


def _genre_aliases(genre: str) -> set[str]:
    normalized = _normalize_name(genre)
    if normalized in {"hiphop", "rap"}:
        return {_normalize_name(value) for value in ("hip-hop", "hip hop", "rap")}
    return {normalized}


def build_genre_proxy_comparison(
    history_df: pd.DataFrame,
    *,
    genre: str,
    public_rows: list[LastFmArtistChartRow],
    artist_tags: dict[str, list[str]],
    start_date: date,
    end_date: date,
    top_n: int = 25,
) -> dict[str, object]:
    events, _ = _listening_events(history_df, start_date=start_date, end_date=end_date)
    music = events[events["media_type"] == "music"].copy()
    aliases = _genre_aliases(genre)
    tags_by_artist_key = {
        _normalize_name(artist): {_normalize_name(tag) for tag in tags} for artist, tags in artist_tags.items()
    }
    tagged_mask = music["artist_key"].isin(tags_by_artist_key)
    genre_mask = music["artist_key"].map(
        lambda artist_key: bool(tags_by_artist_key.get(str(artist_key), set()).intersection(aliases))
    )
    genre_events = music[genre_mask].copy()
    public_by_key = {_normalize_name(row.name): row for row in public_rows[: max(1, int(top_n))]}

    if genre_events.empty:
        personal_top: list[dict[str, object]] = []
        shared: list[dict[str, object]] = []
    else:
        grouped = (
            genre_events.groupby(["artist_key", "artist_name"], as_index=False)
            .agg(events=("artist_key", "size"), duration_ms=("ms_played", "sum"))
            .sort_values(["events", "duration_ms"], ascending=False, kind="stable")
            .head(max(1, int(top_n)))
        )
        personal_top = []
        shared = []
        for rank, (_, row) in enumerate(grouped.iterrows(), start=1):
            public_row = public_by_key.get(str(row["artist_key"]))
            payload = {
                "personal_rank": rank,
                "artist_name": str(row["artist_name"]),
                "events": int(row["events"]),
                "duration_minutes": float(row["duration_ms"] / 60_000.0),
                "public_tag_rank": int(public_row.rank) if public_row is not None else None,
            }
            personal_top.append(payload)
            if public_row is not None:
                shared.append(payload)

    total_music_duration = float(music["ms_played"].sum())
    tagged_duration = float(music.loc[tagged_mask, "ms_played"].sum())
    genre_duration = float(genre_events["ms_played"].sum())
    return {
        "status": "ok",
        "provider": "Last.fm",
        "genre": genre,
        "proxy_tag": genre,
        "matching_tag_aliases": sorted(aliases),
        "artist_tag_coverage": {
            "music_event_share": float(tagged_mask.mean()) if len(music) else 0.0,
            "music_duration_share": float(tagged_duration / total_music_duration) if total_music_duration else 0.0,
        },
        "personal_genre_share": {
            "music_event_share": float(len(genre_events) / max(1, len(music))),
            "music_duration_share": float(genre_duration / total_music_duration) if total_music_duration else 0.0,
        },
        "shared_public_tag_artist_count": len(shared),
        "personal_top_genre_artists": personal_top,
        "shared_public_tag_artists": shared,
        "caveat": (
            "Last.fm tags are community-generated. Shares are calculated only from your history and are not "
            "Spotify public genre listening shares."
        ),
    }


def _top_artist_names(
    history_df: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    limit: int,
) -> list[str]:
    events, _ = _listening_events(history_df, start_date=start_date, end_date=end_date)
    music = events[events["media_type"] == "music"]
    if music.empty:
        return []
    return (
        music.groupby(["artist_key", "artist_name"], as_index=False)
        .agg(events=("artist_key", "size"), duration_ms=("ms_played", "sum"))
        .sort_values(["events", "duration_ms"], ascending=False, kind="stable")
        .head(max(1, int(limit)))["artist_name"]
        .astype(str)
        .tolist()
    )


def write_spotify_wrapped_comparison(
    summary: dict[str, object],
    *,
    output_dir: Path,
    edition: int,
) -> tuple[Path, Path]:
    scopes = list(summary["scope_comparisons"])
    scope_stem = "-".join(scopes).replace("united_states", "us")
    analysis_dir = output_dir / "analysis" / "public_compare"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stem = f"spotify_wrapped_{edition}_{scope_stem}"
    json_path = analysis_dir / f"{stem}.json"
    md_path = analysis_dir / f"{stem}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    history_window = summary["history_window"]
    media_mix = summary["personal_media_mix"]
    relative = summary["relative_scope_summary"]
    lines = [
        f"# Spotify Wrapped {edition} Listening Comparison",
        "",
        f"- Aligned personal window: `{history_window['selected_start_date']}` through "
        f"`{history_window['selected_end_date']}`",
        f"- Selected events: `{history_window['selected_events']}`",
        f"- Selected listening time: `{float(history_window['selected_duration_minutes']):,.1f}` minutes",
        f"- Closest public scope: `{relative.get('closest_scope') or 'n/a'}`",
        f"- Music time share: `{float(media_mix['music']['duration_share']):.1%}`",
        f"- Podcast time share: `{float(media_mix['podcast']['duration_share']):.1%}`",
        "",
        "> Spotify publishes rankings, not public listening-duration distributions. Time-share metrics below "
        "describe your listening concentration on public-top entities.",
        "",
    ]

    for scope_key, scope_result in summary["scope_comparisons"].items():
        baseline = scope_result["baseline"]
        lines.extend(
            [
                f"## {scope_key.replace('_', ' ').title()}",
                "",
                f"- Overall rank similarity: `{float(scope_result['overall_rank_similarity']):.1%}` "
                f"({scope_result['overall_similarity_label']})",
                f"- Official source: {baseline['source_url']}",
                "",
            ]
        )
        for dimension, result in scope_result["dimensions"].items():
            metrics = result["metrics"]
            lines.extend(
                [
                    f"### {dimension.title()}",
                    "",
                    f"- Shared top entries: `{metrics['shared_top_count']}`",
                    f"- Rank-weighted similarity: "
                    f"`{float(metrics['rank_weighted_jaccard_similarity']):.1%}` "
                    f"({metrics['similarity_label']})",
                    f"- Your event share on the public top list: `{float(metrics['event_share_on_public_top']):.1%}`",
                    f"- Your time share on the public top list: `{float(metrics['duration_share_on_public_top']):.1%}`",
                    "",
                ]
            )
            monthly = result["monthly_alignment"]
            if monthly:
                monthly_summary = ", ".join(
                    f"{row['month']} {float(row['duration_share_on_public_top']):.1%}" for row in monthly
                )
                lines.extend([f"Monthly public-top time share: {monthly_summary}", ""])
            shared = result["shared_top"]
            if shared:
                lines.append("Shared entries:")
                for row in shared[:10]:
                    detail = f" - {row['detail']}" if row.get("detail") else ""
                    lines.append(
                        f"- {row['name']}{detail}: your rank `#{row['personal_rank']}`, "
                        f"public rank `#{row['public_rank']}`"
                    )
                lines.append("")
            distinctive = result["personal_distinctive"]
            if distinctive:
                lines.append("Most distinctive entries:")
                for row in distinctive[:5]:
                    detail = f" - {row['detail']}" if row.get("detail") else ""
                    lines.append(f"- {row['name']}{detail}")
                lines.append("")

    genre_comparisons = summary.get("genre_comparisons", [])
    if genre_comparisons:
        lines.extend(["## Genre Proxies", ""])
        for genre_result in genre_comparisons:
            lines.append(f"### {str(genre_result.get('genre', 'Genre')).title()}")
            lines.append("")
            if genre_result.get("status") != "ok":
                lines.append(f"- Status: `{genre_result.get('status')}`")
                lines.append(f"- Detail: {genre_result.get('detail', 'Unavailable')}")
            else:
                share = genre_result["personal_genre_share"]
                coverage = genre_result["artist_tag_coverage"]
                lines.append(f"- Your music-event share: `{float(share['music_event_share']):.1%}`")
                lines.append(f"- Your music-time share: `{float(share['music_duration_share']):.1%}`")
                lines.append(f"- Tagged time coverage: `{float(coverage['music_duration_share']):.1%}`")
                lines.append(
                    f"- Shared artists with the Last.fm public tag chart: "
                    f"`{genre_result['shared_public_tag_artist_count']}`"
                )
            lines.append("")

    lines.extend(
        [
            "## Methodology Notes",
            "",
            f"- {history_window['window_note']}",
            f"- {summary['methodology']['duration_interpretation']}",
            f"- {summary['methodology']['genre_interpretation']}",
            f"- {summary['methodology']['policy_boundary']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.compare_public")

    scope = str(args.scope)
    history_df = load_streaming_history(
        Path(args.data_dir).expanduser().resolve(),
        include_video=bool(args.include_video),
        logger=logger,
    )

    if str(args.provider) == "spotify-wrapped":
        if scope in {"country", "both"} and str(args.country).strip().casefold() not in {
            "united states",
            "united states of america",
            "us",
            "u.s.",
            "usa",
        }:
            logger.error("The bundled Spotify Wrapped country reference currently supports the United States only.")
            return 1

        scopes = ("global", "country") if scope == "both" else (scope,)
        genre_results: list[dict[str, object]] = []
        requested_genres = [str(value).strip() for value in args.genre if str(value).strip()]
        if requested_genres:
            client = LastFmClient.from_env()
            if client is None:
                genre_results = [
                    {
                        "status": "unavailable_missing_lastfm_api_key",
                        "provider": "Last.fm",
                        "genre": genre,
                        "detail": "Set LASTFM_API_KEY to enable community-tag genre proxies.",
                    }
                    for genre in requested_genres
                ]
            else:
                reference = spotify_wrapped_reference(edition=int(args.year), scope=scopes[0])
                artist_names = _top_artist_names(
                    history_df,
                    start_date=reference.window_start,
                    end_date=reference.window_end,
                    limit=max(1, int(args.genre_artist_limit)),
                )
                artist_tags: dict[str, list[str]] = {}
                for artist_name in artist_names:
                    try:
                        artist_tags[artist_name] = [row.name for row in client.get_artist_top_tags(artist_name)]
                    except LastFmError as exc:
                        logger.warning("Could not load Last.fm tags for %s: %s", artist_name, exc)
                for genre in requested_genres:
                    try:
                        public_rows = client.get_tag_top_artists(
                            genre,
                            limit=max(1, int(args.genre_artist_limit)),
                        )
                        genre_results.append(
                            build_genre_proxy_comparison(
                                history_df,
                                genre=genre,
                                public_rows=public_rows,
                                artist_tags=artist_tags,
                                start_date=reference.window_start,
                                end_date=reference.window_end,
                                top_n=max(1, int(args.genre_artist_limit)),
                            )
                        )
                    except LastFmError as exc:
                        genre_results.append(
                            {
                                "status": "unavailable_lastfm_error",
                                "provider": "Last.fm",
                                "genre": genre,
                                "detail": str(exc),
                            }
                        )

        try:
            summary = build_spotify_wrapped_comparison(
                history_df,
                edition=int(args.year),
                scopes=scopes,
                top_n=max(1, int(args.top_n)),
                genre_comparisons=genre_results,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("Could not build Spotify Wrapped comparison: %s", exc)
            return 1
        json_path, md_path = write_spotify_wrapped_comparison(
            summary,
            output_dir=Path(args.output_dir).expanduser().resolve(),
            edition=int(args.year),
        )
        relative = summary["relative_scope_summary"]
        print(f"comparison_json={json_path}")
        print(f"comparison_md={md_path}")
        print(f"closest_scope={relative.get('closest_scope') or 'n/a'}")
        for compared_scope, score in relative["scope_scores"].items():
            print(f"{compared_scope}_rank_similarity={float(score):.4f}")
        return 0

    client = LastFmClient.from_env()
    if client is None:
        logger.error("LASTFM_API_KEY is not set. Add it to .env or .env.local to compare against Last.fm charts.")
        return 1
    if scope == "both":
        logger.error("The Last.fm provider accepts --scope country or --scope global, not both.")
        return 1
    country = str(args.country).strip() if scope == "country" else None
    try:
        public_rows = client.get_top_artists(
            limit=max(1, int(args.top_n)),
            country=country,
        )
    except LastFmError as exc:
        logger.error("Could not load public chart data: %s", exc)
        return 1
    summary = build_public_comparison(
        history_df=history_df,
        public_rows=public_rows,
        lookback_days=max(1, int(args.lookback_days)),
        top_n=max(1, int(args.top_n)),
        scope=scope,
        country=country,
    )
    json_path, md_path = write_public_comparison(
        summary,
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )

    overlap = summary["overlap"]
    print(f"comparison_json={json_path}")
    print(f"comparison_md={md_path}")
    print(f"shared_artists={overlap['shared_artist_count']}")
    print(f"recent_play_share_on_public_artists={float(overlap['recent_play_share_on_public_artists']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
