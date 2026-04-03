from __future__ import annotations

import json
import math
from pathlib import Path

from spotify.run_artifacts import (
    collect_run_manifests,
    safe_read_json,
    write_csv_rows,
    write_json,
    write_markdown,
)


def test_safe_read_json_falls_back_when_orjson_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"run_id":"run_nan","value": NaN}', encoding="utf-8")

    payload = safe_read_json(path, default={})

    assert payload["run_id"] == "run_nan"
    assert math.isnan(payload["value"])


def test_collect_run_manifests_keeps_manifest_with_nan_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_nan"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        '{"run_id":"run_nan","profile":"full","timestamp":"2026-03-29T12:43:13","champion_gate":{"regression": NaN}}',
        encoding="utf-8",
    )

    manifests = collect_run_manifests(tmp_path)

    assert len(manifests) == 1
    assert manifests[0]["run_id"] == "run_nan"
    assert math.isnan(manifests[0]["champion_gate"]["regression"])


def test_write_json_and_markdown_create_parent_directories(tmp_path: Path) -> None:
    json_path = write_json(tmp_path / "nested" / "report.json", {"ok": True})
    markdown_path = write_markdown(tmp_path / "nested" / "report.md", ["# Report", "", "- ok"])

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert markdown_path.read_text(encoding="utf-8") == "# Report\n\n- ok\n"


def test_write_csv_rows_infers_fieldnames_in_first_seen_order(tmp_path: Path) -> None:
    csv_path = write_csv_rows(
        tmp_path / "rows.csv",
        [
            {"alpha": 1, "beta": 2},
            {"beta": 3, "gamma": 4},
        ],
    )

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "alpha,beta,gamma",
        "1,2,",
        ",3,4",
    ]
