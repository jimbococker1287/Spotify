from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.compare_run_timings",
        description="Compare pipeline phase timings between two instrumented Spotify runs.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline run dir, run_manifest.json, or run_phase_timings.json.")
    parser.add_argument("--candidate", required=True, help="Candidate run dir, run_manifest.json, or run_phase_timings.json.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for comparison artifacts. Defaults to <candidate-run>/analysis/timing_compare.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="How many phase deltas to include in the summary.")
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _json_path_from_manifest(manifest_path: Path) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_timings = payload.get("phase_timings", {}) if isinstance(payload, dict) else {}
    candidate_path = Path(str(phase_timings.get("json_path", "")).strip()).expanduser() if isinstance(phase_timings, dict) else Path()
    if candidate_path and candidate_path.exists():
        return candidate_path.resolve()
    sibling = manifest_path.parent / "run_phase_timings.json"
    if sibling.exists():
        return sibling.resolve()
    raise FileNotFoundError(f"Could not resolve run_phase_timings.json from {manifest_path}")


def _resolve_timing_json(path_like: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if path.is_dir():
        candidate = path / "run_phase_timings.json"
        if candidate.exists():
            return candidate
        manifest_path = path / "run_manifest.json"
        if manifest_path.exists():
            return _json_path_from_manifest(manifest_path)
    if path.is_file():
        if path.name == "run_phase_timings.json":
            return path
        if path.name == "run_manifest.json":
            return _json_path_from_manifest(path)
    raise FileNotFoundError(f"Expected a run dir, run_manifest.json, or run_phase_timings.json: {path_like}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _load_bundle(path_like: str) -> dict[str, Any]:
    timing_path = _resolve_timing_json(path_like)
    payload = _load_json(timing_path)
    run_dir = timing_path.parent
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    return {
        "timing_path": timing_path,
        "run_dir": run_dir,
        "manifest_path": manifest_path if manifest_path.exists() else None,
        "run_id": str(payload.get("run_id") or manifest.get("run_id", "")),
        "run_name": str(manifest.get("run_name", "")),
        "profile": str(manifest.get("profile", "")),
        "timestamp": str(manifest.get("timestamp", "")),
        "summary": payload,
    }


def _phase_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    phases = summary.get("phases", [])
    if not isinstance(phases, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in phases:
        if not isinstance(row, dict):
            continue
        phase_name = str(row.get("phase_name", "")).strip()
        if not phase_name:
            continue
        out[phase_name] = row
    return out


def _delta_pct(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return ((candidate - baseline) / baseline) * 100.0


def _speedup_ratio(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(candidate) or candidate <= 0:
        return float("nan")
    return baseline / candidate


def build_timing_comparison(
    *,
    baseline_bundle: dict[str, Any],
    candidate_bundle: dict[str, Any],
    top_n: int = 10,
) -> dict[str, Any]:
    baseline_summary = baseline_bundle["summary"]
    candidate_summary = candidate_bundle["summary"]
    baseline_phases = _phase_index(baseline_summary)
    candidate_phases = _phase_index(candidate_summary)
    phase_names = sorted(set(baseline_phases).union(candidate_phases))

    phase_rows: list[dict[str, Any]] = []
    for phase_name in phase_names:
        baseline_row = baseline_phases.get(phase_name, {})
        candidate_row = candidate_phases.get(phase_name, {})
        baseline_duration = _safe_float(baseline_row.get("duration_seconds", 0.0))
        candidate_duration = _safe_float(candidate_row.get("duration_seconds", 0.0))
        delta_seconds = candidate_duration - baseline_duration
        phase_rows.append(
            {
                "phase_name": phase_name,
                "baseline_status": str(baseline_row.get("status", "missing")),
                "candidate_status": str(candidate_row.get("status", "missing")),
                "baseline_duration_seconds": baseline_duration,
                "candidate_duration_seconds": candidate_duration,
                "delta_seconds": delta_seconds,
                "delta_pct": _delta_pct(baseline_duration, candidate_duration),
                "speedup_ratio": _speedup_ratio(baseline_duration, candidate_duration),
            }
        )

    phase_rows.sort(key=lambda row: abs(_safe_float(row.get("delta_seconds"))), reverse=True)
    top_limit = max(1, int(top_n))
    faster = [row for row in phase_rows if _safe_float(row.get("delta_seconds")) < 0][:top_limit]
    slower = [row for row in phase_rows if _safe_float(row.get("delta_seconds")) > 0][:top_limit]

    baseline_total = _safe_float(baseline_summary.get("total_seconds"))
    candidate_total = _safe_float(candidate_summary.get("total_seconds"))
    baseline_measured = _safe_float(baseline_summary.get("measured_seconds"))
    candidate_measured = _safe_float(candidate_summary.get("measured_seconds"))
    baseline_overhead = _safe_float(baseline_summary.get("unmeasured_overhead_seconds"))
    candidate_overhead = _safe_float(candidate_summary.get("unmeasured_overhead_seconds"))

    return {
        "baseline": {
            "run_id": baseline_bundle.get("run_id", ""),
            "run_name": baseline_bundle.get("run_name", ""),
            "profile": baseline_bundle.get("profile", ""),
            "timestamp": baseline_bundle.get("timestamp", ""),
            "timing_path": str(baseline_bundle["timing_path"]),
            "total_seconds": baseline_total,
            "measured_seconds": baseline_measured,
            "unmeasured_overhead_seconds": baseline_overhead,
        },
        "candidate": {
            "run_id": candidate_bundle.get("run_id", ""),
            "run_name": candidate_bundle.get("run_name", ""),
            "profile": candidate_bundle.get("profile", ""),
            "timestamp": candidate_bundle.get("timestamp", ""),
            "timing_path": str(candidate_bundle["timing_path"]),
            "total_seconds": candidate_total,
            "measured_seconds": candidate_measured,
            "unmeasured_overhead_seconds": candidate_overhead,
        },
        "overall": {
            "delta_seconds": candidate_total - baseline_total,
            "delta_pct": _delta_pct(baseline_total, candidate_total),
            "speedup_ratio": _speedup_ratio(baseline_total, candidate_total),
            "measured_delta_seconds": candidate_measured - baseline_measured,
            "measured_delta_pct": _delta_pct(baseline_measured, candidate_measured),
            "overhead_delta_seconds": candidate_overhead - baseline_overhead,
            "overhead_delta_pct": _delta_pct(baseline_overhead, candidate_overhead),
        },
        "top_phase_deltas": phase_rows[:top_limit],
        "faster_phases": faster,
        "slower_phases": slower,
        "phase_rows": phase_rows,
    }


def _format_metric(value: Any, *, suffix: str = "") -> str:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2f}{suffix}"


def write_timing_comparison(output_dir: Path, comparison: dict[str, Any]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run_timing_comparison.json"
    csv_path = output_dir / "run_timing_comparison.csv"
    md_path = output_dir / "run_timing_comparison.md"

    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "phase_name",
                "baseline_status",
                "candidate_status",
                "baseline_duration_seconds",
                "candidate_duration_seconds",
                "delta_seconds",
                "delta_pct",
                "speedup_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison.get("phase_rows", []))

    baseline = comparison.get("baseline", {})
    candidate = comparison.get("candidate", {})
    overall = comparison.get("overall", {})
    lines = [
        "# Run Timing Comparison",
        "",
        f"- Baseline: `{baseline.get('run_id', '')}` (`{baseline.get('profile', '')}`) total=`{_format_metric(baseline.get('total_seconds'), suffix='s')}`",
        f"- Candidate: `{candidate.get('run_id', '')}` (`{candidate.get('profile', '')}`) total=`{_format_metric(candidate.get('total_seconds'), suffix='s')}`",
        f"- Total delta: `{_format_metric(overall.get('delta_seconds'), suffix='s')}` (`{_format_metric(overall.get('delta_pct'), suffix='%')}`)",
        f"- Total speedup: `{_format_metric(overall.get('speedup_ratio'))}x`",
        f"- Measured delta: `{_format_metric(overall.get('measured_delta_seconds'), suffix='s')}` (`{_format_metric(overall.get('measured_delta_pct'), suffix='%')}`)",
        f"- Overhead delta: `{_format_metric(overall.get('overhead_delta_seconds'), suffix='s')}` (`{_format_metric(overall.get('overhead_delta_pct'), suffix='%')}`)",
        "",
        "## Largest Phase Deltas",
        "",
    ]

    rows = comparison.get("top_phase_deltas", [])
    if isinstance(rows, list) and rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('phase_name', '')}` baseline=`{_format_metric(row.get('baseline_duration_seconds'), suffix='s')}` "
                f"candidate=`{_format_metric(row.get('candidate_duration_seconds'), suffix='s')}` "
                f"delta=`{_format_metric(row.get('delta_seconds'), suffix='s')}` "
                f"speedup=`{_format_metric(row.get('speedup_ratio'))}x`"
            )
    else:
        lines.append("- No phase rows were available.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def main() -> int:
    args = _parse_args()
    baseline_bundle = _load_bundle(args.baseline)
    candidate_bundle = _load_bundle(args.candidate)
    comparison = build_timing_comparison(
        baseline_bundle=baseline_bundle,
        candidate_bundle=candidate_bundle,
        top_n=args.top_n,
    )
    default_output_dir = Path(str(candidate_bundle["run_dir"])) / "analysis" / "timing_compare"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir.resolve()
    json_path, csv_path, md_path = write_timing_comparison(output_dir, comparison)
    print(f"comparison_json={json_path}")
    print(f"comparison_csv={csv_path}")
    print(f"comparison_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
