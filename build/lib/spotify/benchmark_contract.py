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
    "Research-grade locks need a repeated deep comparator before candidate-ranking claims can be called comparison-ready.",
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
        "comparator_policy": {
            "research_grade_subject_classes": ["candidate"],
            "required_comparator_class": "deep",
            "minimum_repeated_runs": _MIN_REPEATED_RUNS,
        },
        "stability_rules": list(_STABILITY_RULES),
    }


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _normalize_declared_model_names(value: object) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    out: list[str] = []
    for item in raw_items:
        normalized = str(item).strip()
        if normalized:
            out.append(normalized)
    return out


def _normalize_model_type(
    value: object,
    *,
    model_name: str = "",
    declared_deep_models: set[str] | None = None,
    declared_classical_models: set[str] | None = None,
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "classical_tuned":
        return "classical"
    if normalized:
        return normalized
    if model_name and declared_deep_models and model_name in declared_deep_models:
        return "deep"
    if model_name and declared_classical_models and model_name in declared_classical_models:
        return "classical"
    return ""


def _model_class_for_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    if normalized == "deep":
        return "deep"
    if normalized == "classical":
        return "classical"
    if normalized in {"retrieval", "retrieval_reranker", "ensemble"}:
        return "candidate"
    return normalized


def _build_model_class_mix(
    *,
    declared_model_class_mix: dict[str, object] | None,
    summary_rows: list[dict[str, object]],
    raw_rows: list[dict[str, object]],
) -> tuple[dict[str, object], bool]:
    declared_payload = declared_model_class_mix if isinstance(declared_model_class_mix, dict) else {}
    deep_models = _normalize_declared_model_names(declared_payload.get("deep_models", []))
    classical_models = _normalize_declared_model_names(declared_payload.get("classical_models", []))
    deep_model_set = set(deep_models)
    classical_model_set = set(classical_models)
    retrieval_enabled = _coerce_bool(declared_payload.get("retrieval_enabled"))
    research_grade_flag = _coerce_bool(declared_payload.get("research_grade"))
    classical_only = bool(_coerce_bool(declared_payload.get("classical_only")) or False)

    expected_classes: list[str] = []
    if classical_models:
        expected_classes.append("classical")
    if deep_models and not classical_only:
        expected_classes.append("deep")
    if retrieval_enabled:
        expected_classes.append("candidate")

    observed_names_by_type: dict[str, list[str]] = {}
    observed_classes: set[str] = set()
    observed_types: set[str] = set()
    observed_seen: set[tuple[str, str]] = set()
    source_rows = list(summary_rows) if summary_rows else list(raw_rows)
    if summary_rows and raw_rows:
        source_rows.extend(raw_rows)
    for row in source_rows:
        model_name = str(row.get("model_name", "")).strip()
        model_type = _normalize_model_type(
            row.get("model_type", ""),
            model_name=model_name,
            declared_deep_models=deep_model_set,
            declared_classical_models=classical_model_set,
        )
        if not model_name or not model_type:
            continue
        identifier = (model_name, model_type)
        if identifier in observed_seen:
            continue
        observed_seen.add(identifier)
        observed_types.add(model_type)
        observed_classes.add(_model_class_for_type(model_type))
        observed_names_by_type.setdefault(model_type, []).append(model_name)

    for names in observed_names_by_type.values():
        names.sort()

    inferred_research_grade = "candidate" in observed_classes or "candidate" in expected_classes
    research_grade = research_grade_flag if research_grade_flag is not None else inferred_research_grade
    model_class_mix = {
        "declared": {
            "research_grade": research_grade,
            "classical_only": classical_only,
            "retrieval_enabled": retrieval_enabled,
            "deep_models": deep_models,
            "classical_models": classical_models,
            "expected_model_classes": sorted(set(expected_classes)),
        },
        "observed": {
            "model_classes": sorted(observed_classes),
            "model_types": sorted(observed_types),
            "model_names_by_type": observed_names_by_type,
        },
    }
    return model_class_mix, research_grade


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
    declared_model_class_mix: dict[str, object] | None = None,
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
    model_class_mix, research_grade = _build_model_class_mix(
        declared_model_class_mix=declared_model_class_mix,
        summary_rows=summary_rows,
        raw_rows=raw_rows,
    )
    deep_rows = [
        row
        for row in summary_rows
        if _model_class_for_type(
            _normalize_model_type(
                row.get("model_type", ""),
                model_name=str(row.get("model_name", "")).strip(),
                declared_deep_models=set(model_class_mix["declared"]["deep_models"]),
                declared_classical_models=set(model_class_mix["declared"]["classical_models"]),
            )
        )
        == "deep"
    ]
    deep_comparator_models = sorted(
        {
            str(row.get("model_name", "")).strip()
            for row in deep_rows
            if str(row.get("model_name", "")).strip()
        }
    )
    deep_comparator_ready = any(
        _safe_int(row.get("run_count"), default=0) >= minimum_repeated_runs
        for row in deep_rows
    )
    comparator_required = bool(research_grade)
    comparator_status = "pass" if (not comparator_required or deep_comparator_ready) else "fail"
    comparator_detail = (
        (
            f"Research-grade comparator guard observed repeated deep comparator(s): "
            f"`{', '.join(deep_comparator_models) if deep_comparator_models else 'none'}`."
        )
        if comparator_status == "pass" and comparator_required
        else (
            "This lock is not research-grade, so the repeated deep comparator guard is not required."
            if not comparator_required
            else (
                "Research-grade comparator guard failed: no repeated deep comparator appears in the benchmark summary "
                f"with at least `{minimum_repeated_runs}` run(s)."
            )
        )
    )
    comparator_guard = {
        "research_grade": research_grade,
        "requires_deep_comparator": comparator_required,
        "required_comparator_class": contract["comparator_policy"]["required_comparator_class"],
        "subject_model_classes": list(contract["comparator_policy"]["research_grade_subject_classes"]),
        "deep_comparator_ready": deep_comparator_ready,
        "deep_comparator_models": deep_comparator_models,
        "status": comparator_status,
        "detail": comparator_detail,
    }

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
        {
            "key": "repeated_deep_comparator",
            "status": comparator_status,
            "detail": comparator_detail,
        },
    ]
    comparison_ready = all(row["status"] == "pass" for row in stability_checks)
    comparison_blockers = [
        str(row["detail"])
        for row in stability_checks
        if str(row.get("status", "")) == "fail"
    ]

    summary = [
        f"Benchmark lock `{benchmark_id}` tracked `{run_count}` repeated run(s) across `{len(model_names)}` model summary rows.",
        f"Observed profiles: `{', '.join(observed_profiles) if observed_profiles else 'none'}`.",
        (
            "Observed model-class mix: "
            f"`{', '.join(model_class_mix['observed']['model_classes']) if model_class_mix['observed']['model_classes'] else 'none'}`."
        ),
        f"Significance pairs at 95%: `{significant_pairs}` / `{len(significance_rows)}`.",
        comparator_detail,
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
        "model_class_mix": model_class_mix,
        "summary_run_counts": summary_run_counts,
        "required_artifacts": required_artifacts,
        "present_artifact_count": len(present_artifacts),
        "required_artifact_count": len(required_artifacts),
        "significance_policy": contract["significance_policy"],
        "comparator_policy": contract["comparator_policy"],
        "comparator_guard": comparator_guard,
        "significance_pair_count": len(significance_rows),
        "significant_pair_count": significant_pairs,
        "stability_rules": contract["stability_rules"],
        "stability_checks": stability_checks,
        "comparison_ready": comparison_ready,
        "comparison_blockers": comparison_blockers,
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
    declared_model_class_mix: dict[str, object] | None = None,
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
        declared_model_class_mix=declared_model_class_mix,
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
        f"- Research grade: `{payload['comparator_guard']['research_grade']}`",
        "",
        "## Summary",
        "",
    ]
    for item in payload["summary"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Stability Checks", ""])
    for row in payload["stability_checks"]:
        lines.append(f"- `{row['key']}`: `{row['status']}`. {row['detail']}")

    lines.extend(["", "## Model-Class Mix", ""])
    lines.append(
        f"- Declared classes: `{', '.join(payload['model_class_mix']['declared']['expected_model_classes']) if payload['model_class_mix']['declared']['expected_model_classes'] else 'none'}`"
    )
    lines.append(
        f"- Observed classes: `{', '.join(payload['model_class_mix']['observed']['model_classes']) if payload['model_class_mix']['observed']['model_classes'] else 'none'}`"
    )
    for model_type, names in payload["model_class_mix"]["observed"]["model_names_by_type"].items():
        lines.append(f"- `{model_type}` models: `{', '.join(names) if names else 'none'}`")

    lines.extend(["", "## Comparator Guard", ""])
    lines.extend(
        [
            f"- Research grade: `{payload['comparator_guard']['research_grade']}`",
            f"- Requires repeated deep comparator: `{payload['comparator_guard']['requires_deep_comparator']}`",
            f"- Deep comparator ready: `{payload['comparator_guard']['deep_comparator_ready']}`",
            f"- Deep comparator models: `{', '.join(payload['comparator_guard']['deep_comparator_models']) if payload['comparator_guard']['deep_comparator_models'] else 'none'}`",
            f"- Detail: {payload['comparator_guard']['detail']}",
        ]
    )

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

    lines.extend(["", "## Comparator Policy", ""])
    comparator_policy = payload["comparator_policy"]
    lines.extend(
        [
            f"- Research-grade subject classes: `{', '.join(comparator_policy['research_grade_subject_classes'])}`",
            f"- Required comparator class: `{comparator_policy['required_comparator_class']}`",
            f"- Minimum repeated runs: `{comparator_policy['minimum_repeated_runs']}`",
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
