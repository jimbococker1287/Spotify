from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from spotify.public_listening_service import (
    ListeningQuery,
    PublicListeningService,
    query_daily_trend,
    query_summary_aggregates,
)


def _mart() -> pd.DataFrame:
    rows = [
        {
            "listening_date": "2024-12-31",
            "reference_edition": 2025,
            "reference_alignment": "historical_projection",
            "dimension": "artists",
            "event_count": 2,
            "duration_minutes": 5.0,
            "unique_entity_count": 2,
            "global_similarity": 0.10,
            "united_states_similarity": 0.20,
            "united_states_minus_global": 0.10,
            "closer_scope": "united_states",
            "global_event_share_on_public_top": 0.25,
            "united_states_event_share_on_public_top": 0.50,
            "global_duration_share_on_public_top": 0.20,
            "united_states_duration_share_on_public_top": 0.40,
            "personal_top_entity": "Artist A",
            "personal_top_entity_detail": None,
        },
        {
            "listening_date": "2025-01-02",
            "reference_edition": 2025,
            "reference_alignment": "date_aligned",
            "dimension": "artists",
            "event_count": 4,
            "duration_minutes": 12.0,
            "unique_entity_count": 3,
            "global_similarity": 0.70,
            "united_states_similarity": 0.90,
            "united_states_minus_global": 0.20,
            "closer_scope": "united_states",
            "global_event_share_on_public_top": 0.50,
            "united_states_event_share_on_public_top": 0.75,
            "global_duration_share_on_public_top": 0.45,
            "united_states_duration_share_on_public_top": 0.80,
            "personal_top_entity": "Artist B",
            "personal_top_entity_detail": None,
        },
        {
            "listening_date": "2025-01-02",
            "reference_edition": 2025,
            "reference_alignment": "date_aligned",
            "dimension": "tracks",
            "event_count": 4,
            "duration_minutes": 12.0,
            "unique_entity_count": 4,
            "global_similarity": 0.30,
            "united_states_similarity": 0.20,
            "united_states_minus_global": -0.10,
            "closer_scope": "global",
            "global_event_share_on_public_top": 0.50,
            "united_states_event_share_on_public_top": 0.25,
            "global_duration_share_on_public_top": 0.55,
            "united_states_duration_share_on_public_top": 0.20,
            "personal_top_entity": "Track A",
            "personal_top_entity_detail": "Artist B",
        },
        {
            "listening_date": "2025-01-03",
            "reference_edition": 2025,
            "reference_alignment": "date_aligned",
            "dimension": "artists",
            "event_count": 3,
            "duration_minutes": 8.0,
            "unique_entity_count": 2,
            "global_similarity": 0.40,
            "united_states_similarity": 0.40,
            "united_states_minus_global": 0.0,
            "closer_scope": "tie",
            "global_event_share_on_public_top": 0.33,
            "united_states_event_share_on_public_top": 0.33,
            "global_duration_share_on_public_top": 0.30,
            "united_states_duration_share_on_public_top": 0.30,
            "personal_top_entity": "Artist C",
            "personal_top_entity_detail": None,
        },
        {
            "listening_date": "2025-11-16",
            "reference_edition": 2025,
            "reference_alignment": "post_window_projection",
            "dimension": "podcasts",
            "event_count": 1,
            "duration_minutes": 25.0,
            "unique_entity_count": 1,
            "global_similarity": 0.0,
            "united_states_similarity": 0.0,
            "united_states_minus_global": 0.0,
            "closer_scope": "tie",
            "global_event_share_on_public_top": 0.0,
            "united_states_event_share_on_public_top": 0.0,
            "global_duration_share_on_public_top": 0.0,
            "united_states_duration_share_on_public_top": 0.0,
            "personal_top_entity": "Podcast A",
            "personal_top_entity_detail": None,
        },
    ]
    return pd.DataFrame(rows)


def test_dataframe_service_filters_daily_trend_and_summarizes() -> None:
    service = PublicListeningService(mart_df=_mart())
    query = ListeningQuery(
        start_date="2025-01-01",
        end_date=date(2025, 1, 3),
        dimension="artists",
        reference_alignment="date_aligned",
    )

    assert service.date_range() is not None
    assert service.date_range().start_date == date(2024, 12, 31)
    assert service.date_range().end_date == date(2025, 11, 16)
    assert service.dimensions() == ("artists", "podcasts", "tracks")
    assert service.reference_alignments() == (
        "date_aligned",
        "historical_projection",
        "post_window_projection",
    )
    assert service.closest_scopes() == ("global", "tie", "united_states")

    trend = service.daily_trend(query)
    assert [(row.listening_date, row.personal_top_entity) for row in trend] == [
        (date(2025, 1, 2), "Artist B"),
        (date(2025, 1, 3), "Artist C"),
    ]
    assert trend[0].best_scope_similarity == pytest.approx(0.9)

    summary = service.summary_aggregates(query)
    assert summary.row_count == 2
    assert summary.active_day_count == 2
    assert summary.total_event_count == 7
    assert summary.total_duration_minutes == pytest.approx(20.0)
    assert summary.average_global_similarity == pytest.approx(0.55)
    assert summary.average_united_states_similarity == pytest.approx(0.65)
    assert summary.average_best_scope_similarity == pytest.approx(0.65)
    assert (summary.global_closer_count, summary.united_states_closer_count, summary.tie_count) == (0, 1, 1)


def test_notable_days_order_by_best_scope_similarity() -> None:
    service = PublicListeningService(mart_df=_mart())

    aligned = service.top_aligned_days(limit=2)
    distinctive = service.top_distinctive_days(
        ListeningQuery(reference_alignment="date_aligned"),
        limit=2,
    )

    assert [(row.listening_date, row.dimension) for row in aligned] == [
        (date(2025, 1, 2), "artists"),
        (date(2025, 1, 3), "artists"),
    ]
    assert [(row.listening_date, row.dimension) for row in distinctive] == [
        (date(2025, 1, 2), "tracks"),
        (date(2025, 1, 3), "artists"),
    ]


def test_query_validation_rejects_bad_ranges_values_and_limits() -> None:
    service = PublicListeningService(mart_df=_mart())

    with pytest.raises(ValueError, match="start_date"):
        service.daily_trend(ListeningQuery(start_date="2025-02-01", end_date="2025-01-01"))
    with pytest.raises(ValueError, match="dimension"):
        service.daily_trend(ListeningQuery(dimension="artists' OR 1=1 --"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference_alignment"):
        service.daily_trend(ListeningQuery(reference_alignment="unknown"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        service.top_aligned_days(limit=0)
    with pytest.raises(ValueError, match="exactly one"):
        PublicListeningService(mart_df=_mart(), duckdb_path="analytics.duckdb")
    with pytest.raises(ValueError, match="missing required columns"):
        PublicListeningService(mart_df=pd.DataFrame({"listening_date": ["2025-01-01"]}))


def test_parquet_and_duckdb_sources_match_dataframe_queries(tmp_path: Path) -> None:
    mart = _mart()
    parquet_path = tmp_path / "mart.parquet"
    db_path = tmp_path / "analytics.duckdb"
    mart.to_parquet(parquet_path, index=False)

    with duckdb.connect(str(db_path)) as con:
        con.register("mart_df", mart)
        con.execute(
            """
            CREATE TABLE mart_public_listening_daily_similarity AS
            SELECT * FROM mart_df
            """
        )
        con.execute(
            """
            CREATE VIEW public_listening_daily_trend AS
            SELECT * FROM mart_public_listening_daily_similarity
            """
        )

    query = ListeningQuery(dimension="artists", closest_scope="united_states")
    parquet_rows = query_daily_trend(parquet_path=parquet_path, query=query)
    duckdb_rows = query_daily_trend(duckdb_path=db_path, query=query)
    dataframe_summary = query_summary_aggregates(mart_df=mart, query=query)

    assert parquet_rows == duckdb_rows
    assert [row.listening_date for row in duckdb_rows] == [date(2024, 12, 31), date(2025, 1, 2)]
    assert dataframe_summary.row_count == 2
