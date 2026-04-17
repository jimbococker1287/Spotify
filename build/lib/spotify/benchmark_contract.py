from __future__ import annotations

from pathlib import Path
import json

_CONTRACT_VERSION = "2026-week10-v1"
_CANONICAL_PROFILE = "small"
_COMPARISON_MODE = "repeated_seed_lock"
_MIN_REPEATED_RUNS = 3
_PRIMARY_METRICS = (
    "val_top1",
    "test_top1",
    "val_top5",
    "test_top5",
)
_RISK_METRICS = (
    "ece",
    "brier",
    "abstention_rate",
    "selective_risk",
)
_REQUIRED_RUN_ARTIFACTS = (
    "run_manifest.json",
    "run_results.json",
    "benchmark_protocol.json",
    "benchmark_protocol.md",
    "experiment_registry.json",
)
_REQUIRED_LOCK_ARTIFACT_PATTERNS = (
    "benchmark_lock_<id>_rows.csv",
    "benchmark_lock_<id>_summary.csv",
    "benchmark_lock_<id>_summary.json",
    "benchmark_lock_<id>_ci95.png",
    "benchmark_lock_<id>_significance.csv",
    "benchmark_lock_<id>_manifest.json",
    "benchmark_lock_<id>_manifest.md",
)
_STABILITY_RULES = (
    "Keep the repeated-seed benchmark profile fixed at `small` until a new contract version is published.",
    "Do not change the run-name prefix, metric columns, or seed-count rule for in-contract comparisons.",
    "Treat protocol changes to sequence length, max artists, or data fingerprint as a new benchmark version.",
    "Require the same artifact pack for every benchmark-lock run before comparing winners.",
)


def describe_canonical_benchmark_contract() -> dict[str, object]:
    return {
        "contract_version": _CONTRACT_VERSION,
        "canonical_profile": _CANONICAL_PROFILE,
        "comparison_mode": _COMPARISON_MODE,
        "minimum_repeated_runs": _MIN_REPEATED_RUNS,
        "primary_metrics": list(_PRIMARY_METRICS),
        "risk_metrics": list(_RISK_METRICS),
        "required_run_artifacts": list(_REQUIRED_RUN_ARTIFACTS),
        "required_benchmark_lock_artifacts": list(_REQUIRED_LOCK_ARTIFACT_PATTERNS),
        "significance_policy": {
            "paired_axis": "run_id",
            "metric": "val_top1",
            "confidence_level": 0.95,
            "z_threshold": 1.96,
            "minimum_shared_runs": _MIN_REPEATED_RUNS,
        },
        "stability_rules": list(_STABILITY_RULES),
    }


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _materialize_required_paths(*, output_dir: Path, benchmark_id: str) -> list[str]:
    resolved: list[str] = []
    for pattern in _REQUIRED_LOCK_ARTIFACT_PATTERNS:
        resolved.append(str((output_dir / pattern.replace("<id>", benchmark_id)).resolve()))
    return resolved


def build_benchmark_lock_manifest(
    *,
    output_dir: Path,
    benchmark_id: str,
    run_name_prefix: str,
    summary_rows: list[dict[str, object]],
    significance_rows: list[dict[str, object]],
    raw_rows: list[dict[str, object]],
    assume_present_artifacts: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    contract = describe_canonical_benchmark_contract()
    unique_run_ids = sorted(
        {
            str(row.get("run_id", "")).strip()
            for row in raw_rows
            if str(row.get("run_id", "")).strip()
        }
    )
    observed_profiles = sorted(
        {
            str(row.get("profile", "")).strip()
            for row in raw_rows
            if str(row.get("profile", "")).strip()
        }
    )
    run_count = len(unique_run_ids)
    model_names = [
        str(row.get("model_name", "")).strip()
        for row in summary_rows
        if str(row.get("model_name", "")).strip()
    ]
    significant_pairs = sum(
        1
        for row in significance_rows
        if _safe_int(row.get("significant_at_95"), default=0) == 1
    )
    required_artifacts = _materialize_required_paths(output_dir=output_dir, benchmark_id=benchmark_id)
    assumed_present = {str(Path(path).resolve()) for path in (assume_present_artifacts or ()) if str(path).strip()}
    present_artifacts = [path for path in required_artifacts if Path(path).exists() or path in assumed_present]
    minimum_repeated_runs = int(contract["minimum_repeated_runs"])
    summary_run_counts = sorted(
        {
            _safe_int(row.get("run_count"), default=0)
            for row in summary_rows
            if str(row.get("model_name", "")).strip()
        }
    )
    per_model_runs_ok = bool(summary_rows) and all(run_count >= minimum_repeated_runs for run_count in summary_run_counts)

    stability_checks = [
        {
            "key": "minimum_repeated_runs",
            "status": "pass" if run_count >= minimum_repeated_runs else "fail",
            "detail": f"Observed `{run_count}` repeated run(s); contract expects at least `{minimum_repeated_runs}`.",
        },
        {
            "key": "minimum_repeated_runs_per_model",
            "status": "pass" if per_model_runs_ok else "fail",
            "detail": (
                f"Per-model repeated runs: `{', '.join(str(value) for value in summary_run_counts) if summary_run_counts else 'none'}`. "
                f"Every published model should appear in at least `{minimum_repeated_runs}` repeated run(s)."
            ),
        },
        {
            "key": "canonical_profile_only",
            "status": "pass" if observed_profiles == [_CANONICAL_PROFILE] else "fail",
            "detail": (
                f"Observed profiles: `{', '.join(observed_profiles) if observed_profiles else 'none'}`. "
                f"Canonical profile is `{_CANONICAL_PROFILE}`."
            ),
        },
        {
            "key": "summary_rows_present",
            "status": "pass" if bool(summary_rows) else "fail",
            "detail": f"Summary rows present: `{len(summary_rows)}`.",
        },
        {
            "key": "significance_rows_present",
            "status": "pass" if bool(significance_rows) else "fail",
            "detail": f"Significance pairs present: `{len(significance_rows)}`.",
        },
        {
            "key": "artifact_pack_complete",
            "status": "pass" if len(present_artifacts) == len(required_artifacts) else "fail",
            "detail": (
                f"Present benchmark-lock artifacts: `{len(present_artifacts)}` / `{len(required_artifacts)}`."
            ),
        },
    ]
    comparison_ready = all(row["status"] == "pass" for row in stability_checks)

    summary = [
        f"Benchmark lock `{benchmark_id}` tracked `{run_count}` repeated run(s) across `{len(model_names)}` model summary rows.",
        f"Observed profiles: `{', '.join(observed_profiles) if observed_profiles else 'none'}`.",
        f"Significance pairs at 95%: `{significant_pairs}` / `{len(significance_rows)}`.",
        (
            "The benchmark contract is comparison-ready."
            if comparison_ready
            else "The benchmark contract is not comparison-ready yet."
        ),
    ]

    return {
        "benchmark_id": benchmark_id,
        "run_name_prefix": run_name_prefix,
        "contract_version": contract["contract_version"],
        "comparison_mode": contract["comparison_mode"],
        "canonical_profile": contract["canonical_profile"],
        "minimum_repeated_runs": minimum_repeated_runs,
        "run_count": run_count,
        "observed_run_ids": unique_run_ids,
        "observed_profiles": observed_profiles,
        "model_names": model_names,
        "summary_run_counts": summary_run_counts,
        "required_artifacts": required_artifacts,
        "present_artifact_count": len(present_artifacts),
        "required_artifact_count": len(required_artifacts),
        "significance_policy": contract["significance_policy"],
        "significance_pair_count": len(significance_rows),
        "significant_pair_count": significant_pairs,
        "stability_rules": contract["stability_rules"],
        "stability_checks": stability_checks,
        "comparison_ready": comparison_ready,
        "summary": summary,
    }


def write_benchmark_lock_manifest(
    *,
    output_dir: Path,
    benchmark_id: str,
    run_name_prefix: str,
    summary_rows: list[dict[str, object]],
    significance_rows: list[dict[str, object]],
    raw_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"benchmark_lock_{benchmark_id}_manifest.json"
    md_path = output_dir / f"benchmark_lock_{benchmark_id}_manifest.md"
    payload = build_benchmark_lock_manifest(
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        run_name_prefix=run_name_prefix,
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        raw_rows=raw_rows,
        assume_present_artifacts=[str(json_path.resolve()), str(md_path.resolve())],
    )

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Lock Manifest",
        "",
        f"- Benchmark ID: `{payload['benchmark_id']}`",
        f"- Contract version: `{payload['contract_version']}`",
        f"- Comparison mode: `{payload['comparison_mode']}`",
        f"- Canonical profile: `{payload['canonical_profile']}`",
        f"- Repeated runs observed: `{payload['run_count']}`",
        f"- Comparison ready: `{payload['comparison_ready']}`",
        "",
        "## Summary",
        "",
    ]
    for item in payload["summary"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Stability Checks", ""])
    for row in payload["stability_checks"]:
        lines.append(f"- `{row['key']}`: `{row['status']}`. {row['detail']}")

    lines.extend(["", "## Significance Policy", ""])
    significance_policy = payload["significance_policy"]
    lines.extend(
        [
            f"- Paired axis: `{significance_policy['paired_axis']}`",
            f"- Metric: `{significance_policy['metric']}`",
            f"- Confidence level: `{significance_policy['confidence_level']}`",
            f"- Z threshold: `{significance_policy['z_threshold']}`",
            f"- Minimum shared runs: `{significance_policy['minimum_shared_runs']}`",
        ]
    )

    lines.extend(["", "## Required Artifacts", ""])
    for path in payload["required_artifacts"]:
        lines.append(f"- `{path}`")

    lines.extend(["", "## Stability Rules", ""])
    for item in payload["stability_rules"]:
        lines.append(f"- {item}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


__all__ = [
    "build_benchmark_lock_manifest",
    "describe_canonical_benchmark_contract",
    "write_benchmark_lock_manifest",
]
