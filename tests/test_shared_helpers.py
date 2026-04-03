from __future__ import annotations

import pandas as pd

from spotify.catalog_utils import dedupe_album_rows, parse_release_date
from spotify.model_types import analysis_prefix_for_model_type


def test_analysis_prefix_for_model_type_normalizes_supported_variants() -> None:
    assert analysis_prefix_for_model_type("deep") == "deep"
    assert analysis_prefix_for_model_type(" classical_tuned ") == "classical"
    assert analysis_prefix_for_model_type("retrieval_reranker") == "retrieval_reranker"
    assert analysis_prefix_for_model_type("unknown") is None


def test_parse_release_date_supports_year_month_and_day_precision() -> None:
    assert parse_release_date("2024", "year") == pd.Timestamp("2024-01-01T00:00:00Z")
    assert parse_release_date("2024-05", "month") == pd.Timestamp("2024-05-01T00:00:00Z")
    assert parse_release_date("2024-05-17", "day") == pd.Timestamp("2024-05-17T00:00:00Z")
    assert parse_release_date("", "day") is None


def test_dedupe_album_rows_keeps_first_row_per_name_date_and_type() -> None:
    rows = [
        {"name": "Album A", "release_date": "2024-01-01", "album_type": "album", "label": "first"},
        {"name": "album a", "release_date": "2024-01-01", "album_type": "album", "label": "second"},
        {"name": "Album A", "release_date": "2024-01-01", "album_type": "single", "label": "third"},
    ]

    assert dedupe_album_rows(rows) == [
        {"name": "Album A", "release_date": "2024-01-01", "album_type": "album", "label": "first"},
        {"name": "Album A", "release_date": "2024-01-01", "album_type": "single", "label": "third"},
    ]
