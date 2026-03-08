from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts",
    "master_metadata_album_artist_name",
)

BOOLEAN_COLUMNS: tuple[str, ...] = (
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
)

MAX_ALLOWED_MS_PLAYED = 86_400_000
MIN_VALID_TS_RATIO = 0.50
MIN_VALID_ARTIST_RATIO = 0.50


def _bool_like_invalid_count(series: pd.Series) -> int:
    valid_values = {0, 1, True, False, "0", "1", "true", "false", "True", "False"}
    cleaned = series.dropna()
    if cleaned.empty:
        return 0
    invalid = cleaned.apply(lambda value: value not in valid_values)
    return int(invalid.sum())


def evaluate_data_quality(df: pd.DataFrame) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    failures = 0
    row_count = int(len(df))

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    schema_ok = len(missing_columns) == 0
    checks.append(
        {
            "name": "required_columns_present",
            "status": "pass" if schema_ok else "fail",
            "details": {
                "missing_columns": missing_columns,
            },
        }
    )
    if not schema_ok:
        failures += 1

    ts_valid_count = 0
    ts_valid_ratio = 0.0
    if "ts" in df.columns and row_count > 0:
        parsed_ts = pd.to_datetime(df["ts"], errors="coerce")
        ts_valid_count = int(parsed_ts.notna().sum())
        ts_valid_ratio = float(ts_valid_count / row_count) if row_count else 0.0
    ts_ok = (row_count == 0) or (ts_valid_ratio >= MIN_VALID_TS_RATIO and ts_valid_count > 0)
    checks.append(
        {
            "name": "timestamp_quality",
            "status": "pass" if ts_ok else "fail",
            "details": {
                "valid_count": ts_valid_count,
                "total_count": row_count,
                "valid_ratio": ts_valid_ratio,
                "min_ratio": MIN_VALID_TS_RATIO,
            },
        }
    )
    if not ts_ok:
        failures += 1

    artist_valid_count = 0
    artist_valid_ratio = 0.0
    if "master_metadata_album_artist_name" in df.columns and row_count > 0:
        artist_series = df["master_metadata_album_artist_name"].astype("string")
        artist_valid = artist_series.notna() & (artist_series.str.strip() != "")
        artist_valid_count = int(artist_valid.sum())
        artist_valid_ratio = float(artist_valid_count / row_count) if row_count else 0.0
    artist_ok = (row_count == 0) or (artist_valid_ratio >= MIN_VALID_ARTIST_RATIO and artist_valid_count > 0)
    checks.append(
        {
            "name": "artist_name_quality",
            "status": "pass" if artist_ok else "fail",
            "details": {
                "valid_count": artist_valid_count,
                "total_count": row_count,
                "valid_ratio": artist_valid_ratio,
                "min_ratio": MIN_VALID_ARTIST_RATIO,
            },
        }
    )
    if not artist_ok:
        failures += 1

    if "ms_played" in df.columns:
        ms_played = pd.to_numeric(df["ms_played"], errors="coerce")
        negative_count = int((ms_played < 0).sum())
        too_large_count = int((ms_played > MAX_ALLOWED_MS_PLAYED).sum())
        ms_ok = (negative_count == 0) and (too_large_count == 0)
        checks.append(
            {
                "name": "ms_played_range",
                "status": "pass" if ms_ok else "fail",
                "details": {
                    "negative_count": negative_count,
                    "too_large_count": too_large_count,
                    "max_allowed": MAX_ALLOWED_MS_PLAYED,
                },
            }
        )
        if not ms_ok:
            failures += 1

    invalid_boolean_counts: dict[str, int] = {}
    for column in BOOLEAN_COLUMNS:
        if column not in df.columns:
            continue
        invalid_boolean_counts[column] = _bool_like_invalid_count(df[column])
    boolean_ok = all(count == 0 for count in invalid_boolean_counts.values())
    checks.append(
        {
            "name": "boolean_column_values",
            "status": "pass" if boolean_ok else "fail",
            "details": invalid_boolean_counts,
        }
    )
    if not boolean_ok:
        failures += 1

    return {
        "status": "pass" if failures == 0 else "fail",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": row_count,
        "failures": failures,
        "checks": checks,
    }


def run_data_quality_gate(df: pd.DataFrame, *, report_path: Path, logger) -> dict[str, object]:
    report = evaluate_data_quality(df)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["status"] == "fail":
        logger.error("Data quality gate failed. Report: %s", report_path)
        raise RuntimeError(f"Data quality gate failed. See report: {report_path}")

    logger.info("Data quality gate passed. Report: %s", report_path)
    return report
