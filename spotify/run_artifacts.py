from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import orjson
except Exception:  # pragma: no cover - exercised through the stdlib fallback.
    orjson = None


def _clone_default(default: Any) -> Any:
    return deepcopy(default)


def safe_read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return _clone_default(default)

    try:
        if orjson is not None:
            return orjson.loads(path.read_bytes())
    except Exception:
        pass

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _clone_default(default)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def collect_run_manifests(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_manifest.json")):
        payload = safe_read_json(path, default=None)
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row["run_dir"] = str(path.parent.resolve())
        rows.append(row)
    return rows


def collect_run_results(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_results.json")):
        payload = safe_read_json(path, default=None)
        if not isinstance(payload, list):
            continue
        run_id = path.parent.name
        run_dir = str(path.parent.resolve())
        for row in payload:
            if not isinstance(row, dict):
                continue
            flat = dict(row)
            flat["run_id"] = run_id
            flat["run_dir"] = run_dir
            rows.append(flat)
    return rows


def collect_run_analysis_rows(output_dir: Path, filename: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob(f"*/analysis/{filename}")):
        payload = safe_read_json(path, default=None)
        if payload is None:
            continue
        run_id = path.parent.parent.name
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                flat = dict(row)
                flat["run_id"] = run_id
                rows.append(flat)
        elif isinstance(payload, dict):
            flat = dict(payload)
            flat["run_id"] = run_id
            rows.append(flat)
    return rows


def rows_to_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.json_normalize(rows) if rows else pd.DataFrame()
