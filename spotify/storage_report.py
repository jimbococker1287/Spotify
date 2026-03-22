from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(num_bytes)} B"


def _categorize_output_path(output_dir: Path, path: Path) -> str:
    relative = path.resolve().relative_to(output_dir.resolve())
    parts = relative.parts
    if not parts:
        return "other"
    if parts[0] == "runs":
        if len(parts) >= 3 and parts[2] == "estimators":
            return "classical_estimators"
        if len(parts) >= 4 and parts[2] == "optuna" and parts[3] == "estimators":
            return "tuned_estimators"
        name = relative.name
        if name.startswith("best_") and name.endswith(".keras"):
            return "deep_checkpoints"
        if len(parts) >= 3 and parts[2] == "prediction_bundles":
            return "prediction_bundles"
        if len(parts) >= 3 and parts[2] == "analysis":
            return "analysis"
        if len(parts) >= 3 and parts[2] == "backtest":
            return "backtest_artifacts"
        if len(parts) >= 3 and parts[2] == "optuna":
            return "optuna_artifacts"
        if name.endswith(".png"):
            return "plots"
        if name.endswith(".log"):
            return "logs"
        if name.endswith(".db") or name.endswith(".duckdb"):
            return "databases"
        if name.endswith(".json") or name.endswith(".csv"):
            return "metadata"
        return "run_other"
    if parts[0] == "analytics":
        return "analytics"
    if parts[0] == "history":
        return "history"
    if parts[0] == "cache":
        return "cache"
    if parts[0] == "models":
        return "models"
    return "other"


def build_storage_report(output_dir: Path, *, top_n: int = 15) -> dict[str, object]:
    output_root = output_dir.expanduser().resolve()
    files = [path for path in output_root.rglob("*") if path.is_file()]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    section_totals: dict[str, int] = {}
    category_totals: dict[str, dict[str, int]] = {}
    run_totals: dict[str, dict[str, object]] = {}
    top_files: list[dict[str, object]] = []

    for path in files:
        size_bytes = _file_size(path)
        try:
            relative = path.resolve().relative_to(output_root)
        except ValueError:
            continue
        category = _categorize_output_path(output_root, path)
        section = relative.parts[0] if relative.parts else "other"

        section_totals[section] = section_totals.get(section, 0) + size_bytes
        bucket = category_totals.setdefault(category, {"bytes": 0, "file_count": 0})
        bucket["bytes"] += size_bytes
        bucket["file_count"] += 1

        if section == "runs" and len(relative.parts) >= 2:
            run_id = relative.parts[1]
            run_entry = run_totals.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "run_dir": str((output_root / "runs" / run_id).resolve()),
                    "total_bytes": 0,
                    "file_count": 0,
                    "categories": {},
                    "top_files": [],
                },
            )
            run_entry["total_bytes"] += size_bytes
            run_entry["file_count"] += 1
            categories = run_entry["categories"]
            categories[category] = int(categories.get(category, 0)) + size_bytes
            top_entries = run_entry["top_files"]
            top_entries.append({"path": str(path.resolve()), "bytes": size_bytes})
            top_entries.sort(key=lambda item: int(item["bytes"]), reverse=True)
            del top_entries[top_n:]

        top_files.append({"path": str(path.resolve()), "bytes": size_bytes, "category": category})

    top_files.sort(key=lambda item: int(item["bytes"]), reverse=True)
    top_files = top_files[:top_n]

    section_rows = [
        {"section": name, "bytes": total, "human_size": _format_bytes(total)}
        for name, total in sorted(section_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    category_rows = [
        {
            "category": name,
            "bytes": payload["bytes"],
            "human_size": _format_bytes(payload["bytes"]),
            "file_count": payload["file_count"],
        }
        for name, payload in sorted(category_totals.items(), key=lambda item: item[1]["bytes"], reverse=True)
    ]

    run_rows: list[dict[str, object]] = []
    for payload in sorted(run_totals.values(), key=lambda item: int(item["total_bytes"]), reverse=True):
        run_rows.append(
            {
                "run_id": payload["run_id"],
                "run_dir": payload["run_dir"],
                "total_bytes": payload["total_bytes"],
                "human_size": _format_bytes(int(payload["total_bytes"])),
                "file_count": payload["file_count"],
                "categories": [
                    {"category": name, "bytes": total, "human_size": _format_bytes(total)}
                    for name, total in sorted(payload["categories"].items(), key=lambda item: item[1], reverse=True)
                ],
                "top_files": [
                    {**item, "human_size": _format_bytes(int(item["bytes"]))}
                    for item in payload["top_files"]
                ],
            }
        )

    report = {
        "generated_at": generated_at,
        "output_dir": str(output_root),
        "total_bytes": sum(section_totals.values()),
        "human_size": _format_bytes(sum(section_totals.values())),
        "section_totals": section_rows,
        "category_totals": category_rows,
        "runs": run_rows,
        "top_files": [{**item, "human_size": _format_bytes(int(item["bytes"]))} for item in top_files],
    }
    return report


def write_storage_report(output_dir: Path, *, top_n: int = 15) -> tuple[Path, Path]:
    output_root = output_dir.expanduser().resolve()
    analytics_dir = output_root / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    report = build_storage_report(output_root, top_n=top_n)

    json_path = analytics_dir / "storage_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Storage Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Output root: `{report['output_dir']}`",
        f"- Total size: `{report['human_size']}`",
        "",
        "## Sections",
        "",
    ]
    for row in report["section_totals"]:
        lines.append(f"- `{row['section']}`: `{row['human_size']}`")

    lines.extend(["", "## Categories", ""])
    for row in report["category_totals"][:10]:
        lines.append(f"- `{row['category']}`: `{row['human_size']}` across `{row['file_count']}` files")

    lines.extend(["", "## Largest Runs", ""])
    for row in report["runs"][:10]:
        lines.append(f"- `{row['run_id']}`: `{row['human_size']}` across `{row['file_count']}` files")

    lines.extend(["", "## Largest Files", ""])
    for row in report["top_files"]:
        lines.append(f"- `{row['human_size']}` `{row['category']}` `{row['path']}`")

    md_path = analytics_dir / "storage_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
