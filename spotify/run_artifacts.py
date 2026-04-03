from __future__ import annotations

from copy import deepcopy
import csv
import filecmp
import importlib
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

try:
    import orjson
except Exception:  # pragma: no cover - exercised through the stdlib fallback.
    orjson = None


def _clone_default(default: Any) -> Any:
    return deepcopy(default)


def _pandas():
    return importlib.import_module("pandas")


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


def safe_read_csv(path: Path) -> "pd.DataFrame":
    pd = _pandas()
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = str(text)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == normalized:
                return path
        except Exception:
            pass
    path.write_text(normalized, encoding="utf-8")
    return path


def _serialize_json(payload: Any, *, indent: int = 2, sort_keys: bool = False) -> str:
    if orjson is not None and indent == 2:
        option = orjson.OPT_INDENT_2
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS
        try:
            return orjson.dumps(payload, option=option).decode("utf-8")
        except Exception:
            pass
    return json.dumps(payload, indent=indent, sort_keys=sort_keys)


def write_json(path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    return write_text(path, _serialize_json(payload, indent=indent, sort_keys=sort_keys))


def write_markdown(path: Path, lines: list[str] | tuple[str, ...] | str) -> Path:
    text = lines if isinstance(lines, str) else "\n".join(lines)
    return write_text(path, text.rstrip() + "\n")


def copy_file_if_changed(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if filecmp.cmp(source, destination, shallow=False):
                return destination
        except Exception:
            pass
    shutil.copy2(source, destination)
    return destination


def write_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | tuple[str, ...] | None = None,
    value_serializer: Any | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = list(fieldnames or [])
    if not resolved_fieldnames:
        seen: set[str] = set()
        for row in rows:
            for key in row:
                key_text = str(key)
                if key_text in seen:
                    continue
                resolved_fieldnames.append(key_text)
                seen.add(key_text)

    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=resolved_fieldnames)
        if resolved_fieldnames:
            writer.writeheader()
        for row in rows:
            payload = dict(row)
            if value_serializer is not None:
                payload = {key: value_serializer(value) for key, value in payload.items()}
            writer.writerow(payload)
    return path


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


def rows_to_frame(rows: list[dict[str, object]]) -> "pd.DataFrame":
    pd = _pandas()
    return pd.json_normalize(rows) if rows else pd.DataFrame()
