from __future__ import annotations

import pandas as pd
import pytest

from spotify.data_quality import evaluate_data_quality, run_data_quality_gate


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "master_metadata_album_artist_name": ["Artist A", "Artist B"],
            "ms_played": [120_000, 200_000],
            "shuffle": [0, 1],
            "skipped": [False, True],
            "offline": [0, 1],
            "incognito_mode": [0, 0],
        }
    )


def test_evaluate_data_quality_passes_on_valid_frame() -> None:
    report = evaluate_data_quality(_valid_df())

    assert report["status"] == "pass"
    assert report["failures"] == 0


def test_evaluate_data_quality_fails_missing_required_columns() -> None:
    df = pd.DataFrame({"ts": ["2024-01-01T00:00:00Z"]})
    report = evaluate_data_quality(df)

    assert report["status"] == "fail"
    assert report["failures"] > 0


def test_run_data_quality_gate_raises_and_writes_report(tmp_path) -> None:
    bad_df = _valid_df()
    bad_df.loc[0, "ms_played"] = -10
    report_path = tmp_path / "data_quality_report.json"

    with pytest.raises(RuntimeError):
        run_data_quality_gate(bad_df, report_path=report_path, logger=_DummyLogger())

    assert report_path.exists()


class _DummyLogger:
    def error(self, *_args, **_kwargs) -> None:
        return None

    def info(self, *_args, **_kwargs) -> None:
        return None
