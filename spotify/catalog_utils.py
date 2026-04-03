from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd


@lru_cache(maxsize=4096)
def parse_release_date(raw_value: str, precision: str) -> pd.Timestamp | None:
    value = str(raw_value or "").strip()
    if not value:
        return None

    precision_key = str(precision or "").strip()
    try:
        if precision_key == "year":
            return pd.Timestamp(year=int(value), month=1, day=1, tz="UTC")
        if precision_key == "month":
            year_text, month_text = value.split("-", 1)
            return pd.Timestamp(year=int(year_text), month=int(month_text), day=1, tz="UTC")
        if precision_key in {"", "day"}:
            year_text, month_text, day_text = value.split("-", 2)
            return pd.Timestamp(year=int(year_text), month=int(month_text), day=int(day_text), tz="UTC")
    except (TypeError, ValueError):
        pass

    if precision_key == "year":
        value = f"{value}-01-01"
    elif precision_key == "month":
        value = f"{value}-01"
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp


def dedupe_album_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("name", "")).casefold(),
            str(row.get("release_date", "")),
            str(row.get("album_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
