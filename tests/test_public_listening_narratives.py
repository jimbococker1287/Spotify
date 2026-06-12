from __future__ import annotations

import pandas as pd

from spotify.public_listening_narratives import (
    build_daily_listening_narratives,
    build_public_listening_narratives,
)


def _frames(days: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mart_rows = []
    detailed_rows = []
    for day in days:
        listening_date = str(day["date"])
        alignment = str(day.get("alignment", "date_aligned"))
        dimensions = day.get("dimensions", {})
        for dimension in ("artists", "tracks", "podcasts"):
            values = dimensions.get(dimension, {})  # type: ignore[union-attr]
            events = int(values.get("events", 0))
            duration = float(values.get("duration", 0.0))
            global_similarity = float(values.get("global_similarity", 0.0))
            us_similarity = float(values.get("us_similarity", 0.0))
            global_share = float(values.get("global_share", 0.0))
            us_share = float(values.get("us_share", 0.0))
            mart_rows.append(
                {
                    "listening_date": listening_date,
                    "reference_edition": 2025,
                    "reference_alignment": alignment,
                    "dimension": dimension,
                    "event_count": events,
                    "duration_minutes": duration,
                    "unique_entity_count": events,
                    "global_similarity": global_similarity,
                    "united_states_similarity": us_similarity,
                    "united_states_minus_global": us_similarity - global_similarity,
                    "closer_scope": (
                        "tie"
                        if global_similarity == us_similarity
                        else "united_states"
                        if us_similarity > global_similarity
                        else "global"
                    ),
                    "global_event_share_on_public_top": global_share,
                    "united_states_event_share_on_public_top": us_share,
                    "global_duration_share_on_public_top": global_share,
                    "united_states_duration_share_on_public_top": us_share,
                    "personal_top_entity": values.get("top"),
                    "personal_top_entity_detail": None,
                }
            )
            for scope, shared in (
                ("global", int(global_share > 0)),
                ("united_states", int(us_share > 0)),
            ):
                detailed_rows.append(
                    {
                        "listening_date": listening_date,
                        "reference_alignment": alignment,
                        "reference_scope": scope,
                        "dimension": dimension,
                        "shared_public_entity_count": shared,
                    }
                )
    return pd.DataFrame(detailed_rows), pd.DataFrame(mart_rows)


def test_daily_narrative_preserves_scope_and_dimension_ties() -> None:
    detailed, mart = _frames(
        [
            {
                "date": "2025-03-03",
                "dimensions": {
                    "artists": {"events": 2, "duration": 6, "global_similarity": 0.4, "us_similarity": 0.4},
                    "tracks": {"events": 2, "duration": 6, "global_similarity": 0.4, "us_similarity": 0.4},
                },
            }
        ]
    )

    record = build_daily_listening_narratives(detailed, mart)[0]

    assert record["strongest_alignment"]["is_tie"] is True
    assert record["strongest_alignment"]["dimensions"] == ["artists", "tracks"]
    assert record["strongest_alignment"]["scopes"] == ["global", "united_states"]
    assert record["most_distinctive_dimension"]["dimensions"] == ["artists", "tracks"]


def test_no_overlap_day_is_described_without_claiming_alignment() -> None:
    detailed, mart = _frames(
        [{"date": "2025-04-01", "dimensions": {"artists": {"events": 3, "duration": 10, "top": "Niche Artist"}}}]
    )

    record = build_daily_listening_narratives(detailed, mart)[0]

    assert record["headline"] == "A distinctly personal listening day"
    assert record["notable_public_top_concentration"]["duration_share"] == 0.0
    assert "None of the day's listening overlapped" in record["concise_summary"]
    assert "No entities overlapped" in record["caveats"][0]


def test_aligned_and_projected_dates_receive_different_caveats() -> None:
    detailed, mart = _frames(
        [
            {"date": "2025-05-01", "alignment": "date_aligned", "dimensions": {"artists": {"events": 1}}},
            {
                "date": "2025-11-20",
                "alignment": "post_window_projection",
                "dimensions": {"artists": {"events": 1}},
            },
        ]
    )

    records = build_daily_listening_narratives(detailed, mart)

    assert not any("projection" in caveat for caveat in records[0]["caveats"])
    assert any("post-window projection" in caveat for caveat in records[1]["caveats"])


def test_podcast_only_day_ignores_inactive_music_dimensions() -> None:
    detailed, mart = _frames(
        [
            {
                "date": "2025-06-08",
                "dimensions": {
                    "podcasts": {
                        "events": 2,
                        "duration": 50,
                        "global_similarity": 0.1,
                        "us_similarity": 0.3,
                        "us_share": 1.0,
                    }
                },
            }
        ]
    )

    record = build_daily_listening_narratives(detailed, mart)[0]

    assert record["headline"] == "Podcasts aligned most with the U.S. public list"
    assert record["strongest_alignment"]["dimensions"] == ["podcasts"]
    assert record["most_distinctive_dimension"]["dimensions"] == ["podcasts"]


def test_weekly_rollup_is_monday_based_and_carries_projection_caveat() -> None:
    detailed, mart = _frames(
        [
            {
                "date": "2025-11-17",
                "alignment": "post_window_projection",
                "dimensions": {"artists": {"events": 2, "duration": 4, "global_similarity": 0.2}},
            },
            {
                "date": "2025-11-19",
                "alignment": "post_window_projection",
                "dimensions": {"tracks": {"events": 1, "duration": 3, "us_similarity": 0.5, "us_share": 1.0}},
            },
        ]
    )

    payload = build_public_listening_narratives(detailed, mart)
    weekly = payload["weekly"][0]

    assert len(payload["daily"]) == 2
    assert weekly["week_start"] == "2025-11-17"
    assert weekly["week_end"] == "2025-11-23"
    assert weekly["strongest_alignment"]["dimensions"] == ["tracks"]
    assert weekly["strongest_alignment"]["scopes"] == ["united_states"]
    assert any("post-window projection" in caveat for caveat in weekly["caveats"])
