from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import median
from pathlib import Path

from .control_room import build_control_room_report
from .run_artifacts import collect_run_manifests, safe_read_json


_STATUS_RANK = {
    "not_supported": 0,
    "early_signal": 1,
    "promising_but_unlocked": 2,
    "analysis_ready": 3,
    "submission_candidate": 4,
}


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _format_metric(value: object) -> str:
    metric = _safe_float(value)
    if not math.isfinite(metric):
        return "n/a"
    return f"{metric:.3f}"


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as infile:
        return [dict(row) for row in csv.DictReader(infile)]


def _write_csv_rows(path: Path, rows: list[dict[str, object]]) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _analysis_prefix_for_model_type(model_type: str) -> str | None:
    normalized = str(model_type).strip().lower()
    if normalized == "deep":
        return "deep"
    if normalized in {"classical", "classical_tuned"}:
        return "classical"
    if normalized in {"retrieval", "retrieval_reranker", "ensemble"}:
        return normalized
    return None


def _read_model_summary(run_dir: Path, model_row: dict[str, object], suffix: str) -> dict[str, object]:
    model_name = str(model_row.get("model_name", "")).strip()
    model_type = str(model_row.get("model_type", "")).strip().lower()
    prefix = _analysis_prefix_for_model_type(model_type)
    if not model_name or prefix is None:
        return {}
    payload = safe_read_json(run_dir / "analysis" / f"{prefix}_{model_name}_{suffix}.json", default={})
    return payload if isinstance(payload, dict) else {}


def _choose_default_run_dir(output_root: Path) -> Path:
    control_room_path = output_root / "analytics" / "control_room.json"
    payload = safe_read_json(control_room_path, default={})
    if not payload:
        payload = build_control_room_report(output_root)
    latest_run = payload.get("latest_run", {})
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    run_id = str(latest_run.get("run_id", "")).strip()
    if run_id:
        run_dir = output_root / "runs" / run_id
        if run_dir.exists():
            return run_dir

    runs_dir = output_root / "runs"
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()] if runs_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_benchmark_manifest(output_root: Path) -> Path | None:
    manifests = sorted((output_root / "history").glob("benchmark_lock_*_manifest.json"))
    if not manifests:
        return None
    return max(manifests, key=lambda path: path.stat().st_mtime)


def _load_benchmark_bundle(manifest_path: Path | None) -> dict[str, object]:
    if manifest_path is None or not manifest_path.exists():
        return {"manifest_path": "", "manifest": {}, "summary_rows": [], "significance_rows": []}
    manifest = safe_read_json(manifest_path, default={})
    manifest = manifest if isinstance(manifest, dict) else {}
    benchmark_id = str(manifest.get("benchmark_id", "")).strip()
    summary_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    if benchmark_id:
        history_dir = manifest_path.parent
        summary_rows = _read_csv_rows(history_dir / f"benchmark_lock_{benchmark_id}_summary.csv")
        significance_rows = _read_csv_rows(history_dir / f"benchmark_lock_{benchmark_id}_significance.csv")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest": manifest,
        "summary_rows": summary_rows,
        "significance_rows": significance_rows,
    }


def _portable_path(path: Path, *, output_root: Path) -> tuple[str, bool, str]:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(output_root)), True, "output_relative"
    except ValueError:
        return str(resolved), False, "absolute_external"


def _artifact_records(paths: list[object], *, output_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in paths:
        raw_text = str(raw).strip()
        if not raw_text:
            continue
        candidate = Path(raw_text).expanduser()
        resolved = candidate.resolve()
        portable_path, portable, scope = _portable_path(candidate, output_root=output_root)
        records.append(
            {
                "absolute_path": str(resolved),
                "portable_path": portable_path,
                "portable": portable,
                "path_scope": scope,
                "exists": resolved.exists(),
            }
        )
    return records


def _artifact_portability_summary(claims: list[dict[str, object]]) -> dict[str, object]:
    records = [
        record
        for claim in claims
        if isinstance(claim, dict)
        for record in claim.get("supporting_artifact_records", [])
        if isinstance(record, dict)
    ]
    total = len(records)
    portable = sum(1 for record in records if bool(record.get("portable")))
    existing = sum(1 for record in records if bool(record.get("exists")))
    non_portable_examples = [
        str(record.get("absolute_path", "")).strip()
        for record in records
        if not bool(record.get("portable")) and str(record.get("absolute_path", "")).strip()
    ]
    return {
        "path_mode": "relative_to_output_dir_when_possible",
        "total_supporting_artifact_count": int(total),
        "portable_supporting_artifact_count": int(portable),
        "existing_supporting_artifact_count": int(existing),
        "non_portable_supporting_artifact_count": int(total - portable),
        "missing_supporting_artifact_count": int(total - existing),
        "all_supporting_artifacts_portable": bool(total > 0 and portable == total),
        "all_supporting_artifacts_present": bool(total > 0 and existing == total),
        "non_portable_examples": non_portable_examples[:5],
    }


def _enrich_claim_artifacts(
    claim: dict[str, object],
    *,
    output_root: Path,
) -> dict[str, object]:
    enriched = dict(claim)
    raw_paths = [str(item).strip() for item in claim.get("supporting_artifacts", []) if str(item).strip()]
    records = _artifact_records(raw_paths, output_root=output_root)
    portability_ready = bool(records) and all(bool(record.get("portable")) for record in records)
    artifact_pack_ready = bool(records) and all(bool(record.get("exists")) for record in records)
    enriched["supporting_artifacts"] = [str(record.get("absolute_path", "")).strip() for record in records]
    enriched["supporting_artifacts_portable"] = [str(record.get("portable_path", "")).strip() for record in records]
    enriched["supporting_artifact_records"] = records
    enriched["supporting_artifact_summary"] = {
        "path_mode": "relative_to_output_dir_when_possible",
        "artifact_count": int(len(records)),
        "portable_artifact_count": int(sum(1 for record in records if bool(record.get("portable")))),
        "existing_artifact_count": int(sum(1 for record in records if bool(record.get("exists")))),
        "portability_status": "ready" if portability_ready else "attention" if records else "missing",
        "artifact_pack_status": "ready" if artifact_pack_ready else "attention" if records else "missing",
    }
    return enriched


def _median_metric(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(median(finite))


def _shift_robustness_signature(run_dir: Path) -> dict[str, object]:
    manifest = safe_read_json(run_dir / "run_manifest.json", default={})
    manifest = manifest if isinstance(manifest, dict) else {}
    if not manifest:
        return {}
    robustness_rows = safe_read_json(run_dir / "analysis" / "robustness_summary.json", default=[])
    robustness_rows = robustness_rows if isinstance(robustness_rows, list) else []
    if not robustness_rows or not isinstance(robustness_rows[0], dict):
        return {}
    drift_summary = safe_read_json(run_dir / "analysis" / "data_drift_summary.json", default={})
    drift_summary = drift_summary if isinstance(drift_summary, dict) else {}
    if not drift_summary:
        return {}

    worst_row = dict(robustness_rows[0])
    target_drift = _safe_float(
        (drift_summary.get("target_drift", {}) if isinstance(drift_summary.get("target_drift", {}), dict) else {}).get(
            "train_vs_test_jsd"
        )
    )
    largest_segment_shift = _safe_float(
        (
            drift_summary.get("largest_segment_shift", {})
            if isinstance(drift_summary.get("largest_segment_shift", {}), dict)
            else {}
        ).get("abs_share_shift")
    )
    drift_interpretation = (
        drift_summary.get("drift_interpretation", {})
        if isinstance(drift_summary.get("drift_interpretation", {}), dict)
        else {}
    )
    return {
        "run_id": str(manifest.get("run_id", "")).strip() or run_dir.name,
        "profile": str(manifest.get("profile", "")).strip(),
        "timestamp": str(manifest.get("timestamp", "")).strip(),
        "worst_segment": str(worst_row.get("worst_segment", "")).strip(),
        "worst_bucket": str(worst_row.get("worst_bucket", "")).strip(),
        "supported_gap": _safe_float(worst_row.get("max_top1_gap")),
        "raw_gap": _safe_float(worst_row.get("raw_max_top1_gap")),
        "bucket_count": int(worst_row.get("worst_bucket_count", 0) or 0),
        "bucket_share": _safe_float(worst_row.get("worst_bucket_share")),
        "target_drift_jsd": target_drift,
        "largest_segment_shift": largest_segment_shift,
        "dominant_context_driver": str(drift_interpretation.get("dominant_context_driver", "")).strip(),
    }


def _shift_robustness_history(output_root: Path, *, run_dir: Path, limit: int = 6) -> dict[str, object]:
    current_signature = _shift_robustness_signature(run_dir)
    if not current_signature:
        return {
            "current_signature": {},
            "history": [],
            "run_count": 0,
            "consistent_slice_run_count": 0,
            "consistency_rate": float("nan"),
            "median_supported_gap": float("nan"),
            "median_target_drift_jsd": float("nan"),
            "dominant_context_driver": "",
        }

    current_profile = str(current_signature.get("profile", "")).strip()
    signatures: list[dict[str, object]] = []
    for manifest in collect_run_manifests(output_root):
        manifest_profile = str(manifest.get("profile", "")).strip()
        if current_profile and manifest_profile and manifest_profile != current_profile:
            continue
        run_dir_candidate = Path(str(manifest.get("run_dir", "")).strip()).expanduser()
        if not run_dir_candidate.exists():
            continue
        signature = _shift_robustness_signature(run_dir_candidate)
        if signature:
            signatures.append(signature)

    signatures.sort(
        key=lambda row: (
            str(row.get("timestamp", "")),
            str(row.get("run_id", "")),
        )
    )
    history = signatures[-limit:]
    target_segment = str(current_signature.get("worst_segment", "")).strip()
    target_bucket = str(current_signature.get("worst_bucket", "")).strip()
    consistent = [
        row
        for row in history
        if str(row.get("worst_segment", "")).strip() == target_segment
        and str(row.get("worst_bucket", "")).strip() == target_bucket
    ]
    driver_counts: dict[str, int] = {}
    for row in history:
        driver = str(row.get("dominant_context_driver", "")).strip()
        if driver:
            driver_counts[driver] = driver_counts.get(driver, 0) + 1
    dominant_driver = max(driver_counts.items(), key=lambda item: (item[1], item[0]))[0] if driver_counts else ""

    return {
        "current_signature": current_signature,
        "history": history,
        "run_count": int(len(history)),
        "consistent_slice_run_count": int(len(consistent)),
        "consistency_rate": (
            float(len(consistent) / len(history))
            if history
            else float("nan")
        ),
        "median_supported_gap": _median_metric([_safe_float(row.get("supported_gap")) for row in history]),
        "median_target_drift_jsd": _median_metric([_safe_float(row.get("target_drift_jsd")) for row in history]),
        "dominant_context_driver": dominant_driver,
    }


def _best_row(rows: list[dict[str, object]], *, predicate) -> dict[str, object]:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            _safe_float(row.get("test_top1")),
            _safe_float(row.get("val_top1")),
        ),
    )


def _best_benchmark_row(rows: list[dict[str, object]], *, predicate) -> dict[str, object]:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            _safe_float(row.get("val_top1_mean")),
            _safe_float(row.get("test_top1_mean")),
        ),
    )


def _benchmark_comparator_guard(benchmark_manifest: dict[str, object]) -> dict[str, object]:
    payload = benchmark_manifest.get("comparator_guard", {})
    return payload if isinstance(payload, dict) else {}


def _claim_candidate_ranking(
    *,
    run_dir: Path,
    run_results: list[dict[str, object]],
    benchmark_bundle: dict[str, object],
) -> dict[str, object]:
    retrieval_row = next(
        (row for row in run_results if str(row.get("model_name", "")).strip() == "retrieval_reranker"),
        {},
    )
    if not retrieval_row:
        retrieval_row = _best_row(
            run_results,
            predicate=lambda row: str(row.get("model_type", "")).strip().lower() != "deep",
        )
    best_non_deep = _best_row(
        run_results,
        predicate=lambda row: str(row.get("model_type", "")).strip().lower() != "deep",
    )
    best_deep = _best_row(
        run_results,
        predicate=lambda row: str(row.get("model_type", "")).strip().lower() == "deep",
    )
    benchmark_rows = benchmark_bundle.get("summary_rows", [])
    benchmark_rows = benchmark_rows if isinstance(benchmark_rows, list) else []
    significance_rows = benchmark_bundle.get("significance_rows", [])
    significance_rows = significance_rows if isinstance(significance_rows, list) else []
    benchmark_manifest = benchmark_bundle.get("manifest", {})
    benchmark_manifest = benchmark_manifest if isinstance(benchmark_manifest, dict) else {}
    comparator_guard = _benchmark_comparator_guard(benchmark_manifest)
    lock_retrieval = next(
        (
            row
            for row in benchmark_rows
            if str(row.get("model_name", "")).strip() == "retrieval_reranker"
        ),
        {},
    )
    lock_non_deep = _best_benchmark_row(
        benchmark_rows,
        predicate=lambda row: str(row.get("model_type", "")).strip().lower() != "deep",
    )
    lock_deep = _best_benchmark_row(
        benchmark_rows,
        predicate=lambda row: str(row.get("model_type", "")).strip().lower() == "deep",
    )

    retrieval_test = _safe_float(retrieval_row.get("test_top1"))
    best_deep_test = _safe_float(best_deep.get("test_top1"))
    live_delta = retrieval_test - best_deep_test if math.isfinite(retrieval_test) and math.isfinite(best_deep_test) else float("nan")
    benchmark_delta = (
        _safe_float(lock_retrieval.get("val_top1_mean")) - _safe_float(lock_deep.get("val_top1_mean"))
        if lock_retrieval and lock_deep
        else float("nan")
    )
    benchmark_ready = bool(benchmark_manifest.get("comparison_ready"))
    requires_deep_comparator = bool(comparator_guard.get("requires_deep_comparator"))
    deep_comparator_ready = bool(comparator_guard.get("deep_comparator_ready"))
    deep_comparator_detail = str(comparator_guard.get("detail", "")).strip()
    retrieval_benchmark_ready = (
        benchmark_ready
        and int(float(lock_retrieval.get("run_count", 0) or 0)) >= 3
        and (not requires_deep_comparator or deep_comparator_ready)
    )
    retrieval_vs_deep_significance = next(
        (
            row
            for row in significance_rows
            if {
                str(row.get("left_model", "")).strip(),
                str(row.get("right_model", "")).strip(),
            }
            == {"retrieval_reranker", str(lock_deep.get("model_name", "")).strip()}
        ),
        {},
    )
    significant_retrieval_lift = bool(int(float(retrieval_vs_deep_significance.get("significant_at_95", 0) or 0)))
    if (
        math.isfinite(live_delta)
        and live_delta >= 0.10
        and retrieval_benchmark_ready
        and math.isfinite(benchmark_delta)
        and benchmark_delta >= 0.02
        and significant_retrieval_lift
    ):
        status = "submission_candidate"
    elif math.isfinite(live_delta) and live_delta >= 0.10 and retrieval_benchmark_ready:
        status = "promising_but_unlocked"
    elif math.isfinite(live_delta) and live_delta >= 0.10:
        status = "promising_but_unlocked"
    elif math.isfinite(live_delta) and live_delta >= 0.03:
        status = "early_signal"
    else:
        status = "not_supported"

    evidence = []
    if retrieval_row and best_deep:
        evidence.append(
            f"Live run: `{retrieval_row.get('model_name', '')}` test_top1 `{_format_metric(retrieval_row.get('test_top1'))}` "
            f"vs best deep `{best_deep.get('model_name', '')}` at `{_format_metric(best_deep.get('test_top1'))}`."
        )
    if best_non_deep:
        evidence.append(
            f"Best overall non-deep surface in the same run is `{best_non_deep.get('model_name', '')}` with val_top1 "
            f"`{_format_metric(best_non_deep.get('val_top1'))}` and test_top1 `{_format_metric(best_non_deep.get('test_top1'))}`."
        )
    if lock_retrieval and lock_deep:
        evidence.append(
            f"Benchmark lock: `retrieval_reranker` mean val_top1 `{_format_metric(lock_retrieval.get('val_top1_mean'))}` "
            f"vs best deep `{lock_deep.get('model_name', '')}` at `{_format_metric(lock_deep.get('val_top1_mean'))}`."
        )
    elif lock_retrieval:
        evidence.append(
            f"Benchmark lock: `retrieval_reranker` is repeated-seed stable with mean val_top1 `{_format_metric(lock_retrieval.get('val_top1_mean'))}` "
            f"across `{int(float(lock_retrieval.get('run_count', 0) or 0))}` run(s)."
        )
    if retrieval_vs_deep_significance:
        evidence.append(
            f"Repeated-seed significance vs `{lock_deep.get('model_name', '')}` is `{int(float(retrieval_vs_deep_significance.get('significant_at_95', 0) or 0))}` "
            f"with mean val_top1 diff `{_format_metric(retrieval_vs_deep_significance.get('mean_diff_val_top1'))}`."
        )
    if requires_deep_comparator and not deep_comparator_ready and deep_comparator_detail:
        evidence.append(f"Benchmark comparator guard: {deep_comparator_detail}")

    missing_checks = []
    if requires_deep_comparator and not deep_comparator_ready:
        missing_checks.append(
            "Benchmark lock is not comparison-ready because the research-grade comparator guard did not observe a repeated deep comparator."
        )
    elif not benchmark_ready:
        missing_checks.append("Finish the repeated-seed benchmark lock with at least 3 runs and a complete manifest artifact pack.")
    if not lock_retrieval:
        missing_checks.append("Add retrieval and reranker models to the benchmark-lock script so the main claim is repeated-seed, not single-run only.")
    elif int(float(lock_retrieval.get("run_count", 0) or 0)) < 3:
        missing_checks.append("Repeat the benchmark lock until retrieval_reranker appears in at least 3 manifest-backed runs.")
    elif not lock_deep:
        missing_checks.append("Add a repeated deep comparator to the benchmark lock before calling this a submission-grade ranking claim.")
    elif not significant_retrieval_lift:
        missing_checks.append("Increase repeated-seed confidence until retrieval_reranker clears a 95% significance check against the best deep baseline.")
    if math.isfinite(live_delta) and live_delta < 0.10:
        missing_checks.append("Increase the live-run gap over deep baselines or narrow the claim to the surfaces that actually hold.")

    return {
        "key": "candidate_ranking",
        "title": "Candidate-ranking surfaces outperform current direct-softmax deep baselines",
        "status": status,
        "summary": (
            f"Retrieval/reranking or ensemble-style candidate surfaces show a live test-top1 lift of `{_format_metric(live_delta)}` "
            f"over the best deep baseline, but the benchmark-lock evidence is "
            f"`{('missing repeated deep comparator evidence' if requires_deep_comparator and not deep_comparator_ready else ('ready' if benchmark_ready else 'not ready'))}`."
        ),
        "evidence": evidence,
        "metrics": {
            "retrieval_model_name": str(retrieval_row.get("model_name", "")),
            "retrieval_test_top1": retrieval_test,
            "best_deep_model_name": str(best_deep.get("model_name", "")),
            "best_deep_test_top1": best_deep_test,
            "live_test_top1_lift_vs_deep": live_delta,
            "benchmark_best_non_deep": str(lock_non_deep.get("model_name", "")),
            "benchmark_retrieval_model_name": str(lock_retrieval.get("model_name", "")),
            "benchmark_best_deep": str(lock_deep.get("model_name", "")),
            "benchmark_val_top1_lift_vs_deep": benchmark_delta,
            "benchmark_comparison_ready": benchmark_ready,
            "benchmark_retrieval_ready": retrieval_benchmark_ready,
            "benchmark_significant_lift": significant_retrieval_lift,
            "benchmark_requires_deep_comparator": requires_deep_comparator,
            "benchmark_deep_comparator_ready": deep_comparator_ready,
            "benchmark_deep_comparator_detail": deep_comparator_detail,
            "benchmark_model_class_mix": benchmark_manifest.get("model_class_mix", {}),
        },
        "missing_checks": missing_checks,
        "supporting_artifacts": [
            str((run_dir / "run_results.json").resolve()),
            str((run_dir / "analysis" / "ablation_summary.csv").resolve()),
            str((run_dir / "analysis" / "backtest_significance.csv").resolve()),
            str(benchmark_bundle.get("manifest_path", "")),
        ],
    }


def _claim_shift_robustness(
    *,
    run_dir: Path,
    run_results: list[dict[str, object]],
) -> dict[str, object]:
    drift_summary = safe_read_json(run_dir / "analysis" / "data_drift_summary.json", default={})
    drift_summary = drift_summary if isinstance(drift_summary, dict) else {}
    robustness_rows = safe_read_json(run_dir / "analysis" / "robustness_summary.json", default=[])
    robustness_rows = robustness_rows if isinstance(robustness_rows, list) else []
    moonshot_summary = safe_read_json(run_dir / "analysis" / "moonshot_summary.json", default={})
    moonshot_summary = moonshot_summary if isinstance(moonshot_summary, dict) else {}

    best_serving_row = _best_row(run_results, predicate=lambda row: True)
    confidence_summary = _read_model_summary(run_dir, best_serving_row, "confidence_summary")
    conformal_summary = _read_model_summary(run_dir, best_serving_row, "conformal_summary")

    worst_robustness = robustness_rows[0] if robustness_rows else {}
    target_drift = _safe_float((drift_summary.get("target_drift", {}) if isinstance(drift_summary.get("target_drift", {}), dict) else {}).get("train_vs_test_jsd"))
    largest_segment_shift = drift_summary.get("largest_segment_shift", {})
    largest_segment_shift = largest_segment_shift if isinstance(largest_segment_shift, dict) else {}
    worst_gap = _safe_float(worst_robustness.get("max_top1_gap"))
    stress_skip_risk = _safe_float(moonshot_summary.get("stress_worst_skip_risk"))
    selective_risk = _safe_float(confidence_summary.get("test_selective_risk"))
    abstention_rate = _safe_float(confidence_summary.get("test_abstention_rate"))
    bucket_count = int(worst_robustness.get("worst_bucket_count", 0) or 0)
    bucket_share = _safe_float(worst_robustness.get("worst_bucket_share"))
    history_summary = _shift_robustness_history(run_dir.parent.parent, run_dir=run_dir)
    repeated_run_count = int(history_summary.get("run_count", 0) or 0)
    consistent_slice_run_count = int(history_summary.get("consistent_slice_run_count", 0) or 0)
    consistency_rate = _safe_float(history_summary.get("consistency_rate"))
    repeated_support = repeated_run_count >= 2 and consistent_slice_run_count >= 2 and (
        not math.isfinite(consistency_rate) or consistency_rate >= 0.66
    )

    support_count = sum(
        1
        for value, threshold in (
            (worst_gap, 0.15),
            (target_drift, 0.15),
            (stress_skip_risk, 0.35),
            (selective_risk, 0.50),
        )
        if math.isfinite(value) and value >= threshold
    )
    if repeated_support:
        support_count += 1

    if support_count >= 4:
        status = "analysis_ready"
    elif support_count >= 2:
        status = "promising_but_unlocked"
    elif support_count >= 1:
        status = "early_signal"
    else:
        status = "not_supported"

    evidence = [
        f"Worst supported robustness gap is `{_format_metric(worst_gap)}` on `{worst_robustness.get('worst_segment', 'segment')}={worst_robustness.get('worst_bucket', 'bucket')}` across `{bucket_count}` rows (`{_format_metric(bucket_share)}` share).",
        f"Target drift JSD is `{_format_metric(target_drift)}` and the largest segment shift is `{_format_metric(largest_segment_shift.get('abs_share_shift'))}`.",
        f"Best served model `{best_serving_row.get('model_name', '')}` has selective risk `{_format_metric(selective_risk)}` with abstention `{_format_metric(abstention_rate)}`.",
    ]
    if math.isfinite(stress_skip_risk):
        evidence.append(
            f"Worst stress scenario `{moonshot_summary.get('stress_worst_skip_scenario', 'unknown')}` reaches skip risk `{_format_metric(stress_skip_risk)}`."
        )
    if repeated_run_count >= 2:
        evidence.append(
            f"Across `{repeated_run_count}` recent `{history_summary.get('current_signature', {}).get('profile', '') or 'matching'}` runs, "
            f"`{worst_robustness.get('worst_segment', 'segment')}={worst_robustness.get('worst_bucket', 'bucket')}` stays the worst supported slice in "
            f"`{consistent_slice_run_count}` runs, with median supported gap `{_format_metric(history_summary.get('median_supported_gap'))}` and median target drift `{_format_metric(history_summary.get('median_target_drift_jsd'))}`."
        )
    dominant_driver = str(history_summary.get("dominant_context_driver", "")).strip()
    if dominant_driver:
        evidence.append(f"Repeated-run drift is primarily `{dominant_driver}` rather than technical or temporal movement.")

    missing_checks = []
    if math.isfinite(abstention_rate) and abstention_rate <= 0.01 and math.isfinite(selective_risk) and selective_risk >= 0.50:
        missing_checks.append("Retune abstention so the uncertainty story shows non-zero refusal rather than full-coverage risk.")
    if math.isfinite(worst_gap) and worst_gap >= 0.15:
        missing_checks.append(
            f"Add a slice-targeted ablation or mitigation for `{worst_robustness.get('worst_segment', 'segment')}={worst_robustness.get('worst_bucket', 'bucket')}` before turning this into a causal or mitigation claim."
        )
    if math.isfinite(target_drift) and target_drift >= 0.15 and not repeated_support:
        missing_checks.append("Repeat the drift and robustness slices across multiple completed runs so the shift story is not single-run only.")

    strong_repeated_support = (
        repeated_support
        and repeated_run_count >= 3
        and consistent_slice_run_count >= 3
        and (not math.isfinite(consistency_rate) or consistency_rate >= 0.80)
    )

    if status != "analysis_ready" and support_count >= 3 and strong_repeated_support and not missing_checks:
        status = "analysis_ready"

    return {
        "key": "shift_robustness",
        "title": "Failure concentration is measurable under drift, slice shift, and repeated-session regimes",
        "status": status,
        "summary": (
            f"The current full run shows drift `{_format_metric(target_drift)}`, worst-slice gap `{_format_metric(worst_gap)}`, "
            f"and selective risk `{_format_metric(selective_risk)}`; across `{repeated_run_count}` matching runs, the same failure slice recurs `{consistent_slice_run_count}` times."
        ),
        "evidence": evidence,
        "metrics": {
            "worst_robustness_model": str(worst_robustness.get("model_name", "")),
            "worst_robustness_gap": worst_gap,
            "worst_robustness_segment": str(worst_robustness.get("worst_segment", "")),
            "worst_robustness_bucket": str(worst_robustness.get("worst_bucket", "")),
            "worst_robustness_bucket_count": bucket_count,
            "worst_robustness_bucket_share": bucket_share,
            "target_drift_jsd": target_drift,
            "largest_segment_shift": _safe_float(largest_segment_shift.get("abs_share_shift")),
            "selective_risk": selective_risk,
            "abstention_rate": abstention_rate,
            "stress_skip_risk": stress_skip_risk,
            "repeated_run_count": repeated_run_count,
            "consistent_slice_run_count": consistent_slice_run_count,
            "consistent_slice_rate": consistency_rate,
            "repeated_median_supported_gap": _safe_float(history_summary.get("median_supported_gap")),
            "repeated_median_target_drift_jsd": _safe_float(history_summary.get("median_target_drift_jsd")),
            "dominant_context_driver": dominant_driver,
            "conformal_coverage": _safe_float((conformal_summary.get("test", {}) if isinstance(conformal_summary.get("test", {}), dict) else {}).get("coverage")),
        },
        "missing_checks": missing_checks,
        "supporting_artifacts": [
            str((run_dir / "analysis" / "data_drift_summary.json").resolve()),
            str((run_dir / "analysis" / "robustness_summary.json").resolve()),
            str((run_dir / "analysis" / "moonshot_summary.json").resolve()),
            str((run_dir / "analysis" / "ensemble_blended_ensemble_confidence_summary.json").resolve()),
            str((run_dir / "analysis" / "ensemble_blended_ensemble_conformal_summary.json").resolve()),
        ],
    }


def _claim_friction_counterfactual(*, run_dir: Path) -> dict[str, object]:
    friction_summary = safe_read_json(run_dir / "analysis" / "friction_proxy_summary.json", default={})
    friction_summary = friction_summary if isinstance(friction_summary, dict) else {}
    auc_lift = friction_summary.get("auc_lift", {})
    auc_lift = auc_lift if isinstance(auc_lift, dict) else {}
    proxy = friction_summary.get("proxy_counterfactual", {})
    proxy = proxy if isinstance(proxy, dict) else {}
    test_delta = _safe_float(proxy.get("test_mean_delta"))
    baseline_auc = _safe_float((friction_summary.get("baseline_model", {}) if isinstance(friction_summary.get("baseline_model", {}), dict) else {}).get("test_auc"))
    full_auc = _safe_float((friction_summary.get("full_model", {}) if isinstance(friction_summary.get("full_model", {}), dict) else {}).get("test_auc"))

    if math.isfinite(test_delta) and abs(test_delta) >= 0.01 and math.isfinite(_safe_float(auc_lift.get("test"))) and abs(_safe_float(auc_lift.get("test"))) >= 0.01:
        status = "early_signal"
    else:
        status = "not_supported"

    return {
        "key": "friction_counterfactual",
        "title": "Technical-friction counterfactuals shift skip risk in a measurable way",
        "status": status,
        "summary": (
            f"Current friction artifacts show test delta `{_format_metric(test_delta)}` with baseline/full AUCs "
            f"`{_format_metric(baseline_auc)}` / `{_format_metric(full_auc)}`, so this is not yet a trustworthy paper claim."
        ),
        "evidence": [
            f"Proxy counterfactual test mean delta is `{_format_metric(test_delta)}`.",
            f"Baseline vs full test AUC is `{_format_metric(baseline_auc)}` vs `{_format_metric(full_auc)}`.",
        ],
        "metrics": {
            "test_mean_delta": test_delta,
            "baseline_test_auc": baseline_auc,
            "full_test_auc": full_auc,
            "test_auc_lift": _safe_float(auc_lift.get("test")),
        },
        "missing_checks": [
            "Audit the friction label path because AUC is saturated while the counterfactual delta is effectively zero.",
            "Add a non-degenerate intervention or synthetic perturbation check before using this as a headline claim.",
        ],
        "supporting_artifacts": [
            str((run_dir / "analysis" / "friction_proxy_summary.json").resolve()),
            str((run_dir / "analysis" / "friction_counterfactual_delta.csv").resolve()),
        ],
    }


def _claim_risk_aware_abstention(*, run_dir: Path, run_results: list[dict[str, object]]) -> dict[str, object]:
    best_serving_row = _best_row(run_results, predicate=lambda row: True)
    confidence_summary = _read_model_summary(run_dir, best_serving_row, "confidence_summary")
    conformal_summary = _read_model_summary(run_dir, best_serving_row, "conformal_summary")
    selective_risk = _safe_float(confidence_summary.get("test_selective_risk"))
    abstention_rate = _safe_float(confidence_summary.get("test_abstention_rate"))
    coverage = _safe_float((conformal_summary.get("test", {}) if isinstance(conformal_summary.get("test", {}), dict) else {}).get("coverage"))

    if math.isfinite(abstention_rate) and abstention_rate > 0.01 and math.isfinite(selective_risk) and selective_risk < 0.50:
        status = "early_signal"
    else:
        status = "not_supported"

    return {
        "key": "risk_aware_abstention",
        "title": "Risk-aware abstention reduces high-confidence failures without collapsing coverage",
        "status": status,
        "summary": (
            f"Current conformal outputs show test coverage `{_format_metric(coverage)}`, selective risk `{_format_metric(selective_risk)}`, "
            f"and abstention `{_format_metric(abstention_rate)}`, which is not yet enough for a positive abstention claim."
        ),
        "evidence": [
            f"Selective risk is `{_format_metric(selective_risk)}`.",
            f"Abstention rate is `{_format_metric(abstention_rate)}`.",
            f"Conformal test coverage is `{_format_metric(coverage)}`.",
        ],
        "metrics": {
            "selective_risk": selective_risk,
            "abstention_rate": abstention_rate,
            "conformal_coverage": coverage,
        },
        "missing_checks": [
            "Tune abstention thresholds until coverage loss buys a meaningful selective-risk reduction.",
            "Report accuracy-coverage tradeoffs rather than only full-coverage conformal metrics.",
        ],
        "supporting_artifacts": [
            str((run_dir / "analysis" / "ensemble_blended_ensemble_confidence_summary.json").resolve()),
            str((run_dir / "analysis" / "ensemble_blended_ensemble_conformal_summary.json").resolve()),
        ],
    }


def _claim_sort_key(claim: dict[str, object]) -> tuple[int, float]:
    status = str(claim.get("status", "not_supported"))
    rank = _STATUS_RANK.get(status, 0)
    metrics = claim.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    lift = _safe_float(metrics.get("live_test_top1_lift_vs_deep"))
    robustness = _safe_float(metrics.get("worst_robustness_gap"))
    magnitude = max(
        [value for value in (lift, robustness) if math.isfinite(value)],
        default=0.0,
    )
    return (rank, magnitude)


def _publication_outline(primary_claim: dict[str, object], backup_claim: dict[str, object]) -> list[dict[str, object]]:
    primary_checks = primary_claim.get("missing_checks", [])
    primary_checks = primary_checks if isinstance(primary_checks, list) else []
    backup_checks = backup_claim.get("missing_checks", [])
    backup_checks = backup_checks if isinstance(backup_checks, list) else []
    return [
        {
            "section": "Positioning",
            "focus": primary_claim.get("title", ""),
            "notes": [
                "Frame the paper around robust recommendation under drift and deployment risk, not generic model-count benchmarking.",
                f"Lead with the strongest current claim: {primary_claim.get('summary', '')}",
            ],
        },
        {
            "section": "Main Evidence",
            "focus": primary_claim.get("title", ""),
            "notes": [
                "Use the live run leaderboard, benchmark lock summary, and significance outputs as the main result table.",
                *[str(item) for item in primary_checks[:2]],
            ],
        },
        {
            "section": "Backup Section",
            "focus": backup_claim.get("title", ""),
            "notes": [
                f"Use the backup claim as the second empirical section: {backup_claim.get('summary', '')}",
                *[str(item) for item in backup_checks[:2]],
            ],
        },
        {
            "section": "Limitations and Next Ablations",
            "focus": "Threats to validity",
            "notes": [
                "Call out any benchmark-contract failures directly instead of burying them.",
                "Explain which claims are single-run, which are repeated-seed, and which still depend on operator-facing diagnostics.",
                "Keep supporting artifact references portable by preferring paths relative to the output bundle over workspace-absolute paths.",
            ],
        },
    ]


def _evidence_status(ready: bool | None) -> str:
    if ready is None:
        return "n/a"
    return "ready" if ready else "gap"


def _existing_artifact_count(paths: list[str]) -> int:
    return sum(1 for path in paths if str(path).strip() and Path(str(path).strip()).exists())


def _claim_support_row(
    claim: dict[str, object],
    *,
    role: str,
    benchmark_ready: bool,
) -> dict[str, object]:
    metrics = claim.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    supporting_artifacts = [str(item).strip() for item in claim.get("supporting_artifacts", []) if str(item).strip()]
    supporting_artifact_records = [
        dict(item)
        for item in claim.get("supporting_artifact_records", [])
        if isinstance(item, dict)
    ]
    missing_checks = [str(item).strip() for item in claim.get("missing_checks", []) if str(item).strip()]
    claim_key = str(claim.get("key", "")).strip()

    if claim_key == "shift_robustness":
        repeated_ready = (
            int(metrics.get("repeated_run_count", 0) or 0) >= 3
            and int(metrics.get("consistent_slice_run_count", 0) or 0) >= 2
        )
        slice_ready = math.isfinite(_safe_float(metrics.get("worst_robustness_gap")))
        risk_ready = all(
            math.isfinite(_safe_float(metrics.get(key)))
            for key in ("selective_risk", "abstention_rate", "stress_skip_risk")
        )
        benchmark_status = _evidence_status(None)
    elif claim_key == "candidate_ranking":
        repeated_ready = bool(metrics.get("benchmark_retrieval_ready"))
        slice_ready = None
        risk_ready = None
        benchmark_status = _evidence_status(bool(metrics.get("benchmark_comparison_ready")) and benchmark_ready)
    elif claim_key == "risk_aware_abstention":
        repeated_ready = None
        slice_ready = None
        risk_ready = all(
            math.isfinite(_safe_float(metrics.get(key)))
            for key in ("selective_risk", "abstention_rate", "conformal_coverage")
        ) and _safe_float(metrics.get("abstention_rate")) > 0.01
        benchmark_status = _evidence_status(None)
    else:
        repeated_ready = None
        slice_ready = None
        risk_ready = math.isfinite(_safe_float(metrics.get("test_mean_delta")))
        benchmark_status = _evidence_status(None)

    artifact_count = len(supporting_artifact_records) if supporting_artifact_records else len(supporting_artifacts)
    existing_artifacts = (
        sum(1 for item in supporting_artifact_records if bool(item.get("exists")))
        if supporting_artifact_records
        else _existing_artifact_count(supporting_artifacts)
    )
    portable_artifacts = (
        sum(1 for item in supporting_artifact_records if bool(item.get("portable")))
        if supporting_artifact_records
        else 0
    )
    artifact_pack_ready = artifact_count > 0 and existing_artifacts == artifact_count
    artifact_portability_ready = artifact_count > 0 and portable_artifacts == artifact_count
    live_signal_ready = _STATUS_RANK.get(str(claim.get("status", "")), 0) >= 1

    return {
        "claim_key": claim_key,
        "role": role,
        "status": str(claim.get("status", "")),
        "live_signal_status": _evidence_status(live_signal_ready),
        "benchmark_evidence_status": benchmark_status,
        "repeated_evidence_status": _evidence_status(repeated_ready),
        "slice_evidence_status": _evidence_status(slice_ready),
        "risk_evidence_status": _evidence_status(risk_ready),
        "artifact_pack_status": _evidence_status(artifact_pack_ready),
        "artifact_portability_status": _evidence_status(artifact_portability_ready),
        "supporting_artifact_count": artifact_count,
        "existing_artifact_count": existing_artifacts,
        "portable_artifact_count": portable_artifacts,
        "missing_check_count": len(missing_checks),
        "next_gate": missing_checks[0] if missing_checks else "ready_to_package",
    }


def _build_submission_readiness(
    *,
    primary_claim: dict[str, object],
    backup_claim: dict[str, object],
    benchmark_lock: dict[str, object],
    evaluation_tables: dict[str, object],
    claim_support_matrix: list[dict[str, object]],
    claim_gaps: list[str],
) -> dict[str, object]:
    primary_rank = _STATUS_RANK.get(str(primary_claim.get("status", "")), 0)
    backup_rank = _STATUS_RANK.get(str(backup_claim.get("status", "")), 0)
    benchmark_ready = bool(benchmark_lock.get("comparison_ready"))
    primary_row = next((row for row in claim_support_matrix if str(row.get("role", "")) == "primary"), {})
    backup_row = next((row for row in claim_support_matrix if str(row.get("role", "")) == "backup"), {})
    run_leaderboard = evaluation_tables.get("run_leaderboard", [])
    benchmark_table = evaluation_tables.get("benchmark_lock", [])
    run_leaderboard = run_leaderboard if isinstance(run_leaderboard, list) else []
    benchmark_table = benchmark_table if isinstance(benchmark_table, list) else []
    primary_missing_checks = [
        str(item).strip()
        for item in primary_claim.get("missing_checks", [])
        if str(item).strip()
    ]
    backup_missing_checks = [
        str(item).strip()
        for item in backup_claim.get("missing_checks", [])
        if str(item).strip()
    ]

    checks = [
        {
            "key": "primary_claim_strength",
            "status": "pass" if primary_rank >= 3 else "attention" if primary_rank >= 2 else "fail",
            "detail": (
                f"Primary claim `{primary_claim.get('key', '')}` is `{primary_claim.get('status', '')}`."
            ),
        },
        {
            "key": "backup_claim_strength",
            "status": "pass" if backup_rank >= 1 else "fail",
            "detail": (
                f"Backup claim `{backup_claim.get('key', '')}` is `{backup_claim.get('status', '')}`."
            ),
        },
        {
            "key": "benchmark_lock",
            "status": "pass" if benchmark_ready else "attention",
            "detail": (
                f"Benchmark lock `{benchmark_lock.get('benchmark_id', '')}` comparison_ready=`{benchmark_ready}`."
            ),
        },
        {
            "key": "primary_claim_blockers",
            "status": "pass" if not primary_missing_checks else "attention",
            "detail": (
                "Primary claim blockers: "
                f"`{primary_missing_checks[0] if primary_missing_checks else 'none'}`."
            ),
        },
        {
            "key": "backup_claim_blockers",
            "status": "pass" if not backup_missing_checks else "attention",
            "detail": (
                "Backup claim blockers: "
                f"`{backup_missing_checks[0] if backup_missing_checks else 'none'}`."
            ),
        },
        {
            "key": "evaluation_tables",
            "status": "pass" if bool(run_leaderboard) and bool(benchmark_table) and bool(claim_support_matrix) else "fail",
            "detail": (
                f"Tables present: run_leaderboard=`{bool(run_leaderboard)}`, benchmark_lock=`{bool(benchmark_table)}`, "
                f"claim_support_matrix=`{bool(claim_support_matrix)}`."
            ),
        },
        {
            "key": "artifact_pack",
            "status": (
                "pass"
                if str(primary_row.get("artifact_pack_status", "")) == "ready"
                and str(backup_row.get("artifact_pack_status", "")) in {"ready", "n/a", ""}
                else "attention"
            ),
            "detail": (
                f"Primary artifact pack is `{primary_row.get('artifact_pack_status', 'gap')}` and backup artifact pack is "
                f"`{backup_row.get('artifact_pack_status', 'gap')}`."
            ),
        },
        {
            "key": "artifact_portability",
            "status": (
                "pass"
                if str(primary_row.get("artifact_portability_status", "")) == "ready"
                and str(backup_row.get("artifact_portability_status", "")) in {"ready", "n/a", ""}
                else "attention"
            ),
            "detail": (
                f"Primary artifact portability is `{primary_row.get('artifact_portability_status', 'gap')}` and backup portability is "
                f"`{backup_row.get('artifact_portability_status', 'gap')}`."
            ),
        },
        {
            "key": "open_gaps",
            "status": "pass" if len(claim_gaps) == 0 else "attention" if len(claim_gaps) <= 2 else "fail",
            "detail": f"Open claim gaps: `{len(claim_gaps)}`.",
        },
    ]

    has_open_checks = any(str(check.get("status", "")) != "pass" for check in checks)
    if primary_rank >= 4 and benchmark_ready and not has_open_checks:
        status = "submission_candidate"
    elif primary_rank >= 3 and backup_rank >= 1 and benchmark_ready and not has_open_checks:
        status = "analysis_ready"
    elif primary_rank >= 2:
        status = "promising_but_unlocked"
    else:
        status = "not_ready"

    blockers = [
        str(check["detail"])
        for check in checks
        if str(check.get("status", "")) in {"attention", "fail"}
    ]
    for item in claim_gaps[:3]:
        if item not in blockers:
            blockers.append(item)

    summary = [
        f"Primary claim is `{primary_claim.get('status', 'unknown')}` and backup claim is `{backup_claim.get('status', 'unknown')}`.",
        f"Benchmark lock is `{('ready' if benchmark_ready else 'not ready')}` with `{len(claim_gaps)}` open claim gaps.",
        (
            "Supporting artifacts are portable enough for packaging."
            if str(primary_row.get("artifact_portability_status", "")) == "ready"
            else "Supporting artifact paths still depend on workspace-specific references or missing files."
        ),
        f"Submission-readiness state is `{status}`.",
    ]

    return {
        "status": status,
        "ready_for_external_review": status in {"analysis_ready", "submission_candidate"} and not has_open_checks and benchmark_ready,
        "checks": checks,
        "blockers": blockers[:6],
        "summary": summary,
    }


def build_research_claims_report(
    output_dir: Path,
    *,
    run_dir: Path | None = None,
    benchmark_manifest_path: Path | None = None,
) -> dict[str, object]:
    output_root = output_dir.expanduser().resolve()
    resolved_run_dir = run_dir.expanduser().resolve() if run_dir is not None else _choose_default_run_dir(output_root)
    if not resolved_run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {resolved_run_dir}")
    resolved_benchmark_manifest = (
        benchmark_manifest_path.expanduser().resolve()
        if benchmark_manifest_path is not None
        else _latest_benchmark_manifest(output_root)
    )
    benchmark_bundle = _load_benchmark_bundle(resolved_benchmark_manifest)

    run_manifest = safe_read_json(resolved_run_dir / "run_manifest.json", default={})
    run_manifest = run_manifest if isinstance(run_manifest, dict) else {}
    run_results = safe_read_json(resolved_run_dir / "run_results.json", default=[])
    run_results = [dict(row) for row in run_results if isinstance(row, dict)] if isinstance(run_results, list) else []
    run_results.sort(
        key=lambda row: (
            _safe_float(row.get("test_top1")),
            _safe_float(row.get("val_top1")),
        ),
        reverse=True,
    )

    claims = [
        _claim_shift_robustness(run_dir=resolved_run_dir, run_results=run_results),
        _claim_candidate_ranking(run_dir=resolved_run_dir, run_results=run_results, benchmark_bundle=benchmark_bundle),
        _claim_friction_counterfactual(run_dir=resolved_run_dir),
        _claim_risk_aware_abstention(run_dir=resolved_run_dir, run_results=run_results),
    ]
    claims = [_enrich_claim_artifacts(claim, output_root=output_root) for claim in claims if isinstance(claim, dict)]
    claims.sort(key=_claim_sort_key, reverse=True)
    primary_claim = claims[0] if claims else {}
    backup_claim = claims[1] if len(claims) > 1 else {}

    top_model_rows = [
        {
            "model_name": str(row.get("model_name", "")),
            "model_type": str(row.get("model_type", "")),
            "val_top1": _safe_float(row.get("val_top1")),
            "test_top1": _safe_float(row.get("test_top1")),
        }
        for row in run_results[:8]
    ]
    benchmark_summary_rows = benchmark_bundle.get("summary_rows", [])
    benchmark_summary_rows = benchmark_summary_rows if isinstance(benchmark_summary_rows, list) else []
    benchmark_table = [
        {
            "model_name": str(row.get("model_name", "")),
            "model_type": str(row.get("model_type", "")),
            "run_count": int(float(row.get("run_count", 0) or 0)),
            "val_top1_mean": _safe_float(row.get("val_top1_mean")),
            "test_top1_mean": _safe_float(row.get("test_top1_mean")),
            "val_top1_ci95": _safe_float(row.get("val_top1_ci95")),
        }
        for row in sorted(
            benchmark_summary_rows,
            key=lambda row: _safe_float(row.get("val_top1_mean")),
            reverse=True,
        )[:8]
    ]

    claim_gaps = []
    for claim in claims:
        for item in claim.get("missing_checks", []):
            text = str(item).strip()
            if text and text not in claim_gaps:
                claim_gaps.append(text)

    report = {
        "run": {
            "run_id": str(run_manifest.get("run_id", resolved_run_dir.name)),
            "profile": str(run_manifest.get("profile", "")),
            "timestamp": str(run_manifest.get("timestamp", "")),
            "run_dir": str(resolved_run_dir),
            "run_dir_portable": _portable_path(resolved_run_dir, output_root=output_root)[0],
        },
        "benchmark_lock": {
            "manifest_path": str(benchmark_bundle.get("manifest_path", "")),
            "manifest_path_portable": (
                _portable_path(Path(str(benchmark_bundle.get("manifest_path", ""))), output_root=output_root)[0]
                if str(benchmark_bundle.get("manifest_path", "")).strip()
                else ""
            ),
            "benchmark_id": str((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("benchmark_id", "")),
            "comparison_ready": bool((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("comparison_ready")),
            "model_class_mix": dict((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("model_class_mix", {}))
            if isinstance((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("model_class_mix", {}), dict)
            else {},
            "comparator_guard": dict((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("comparator_guard", {}))
            if isinstance((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("comparator_guard", {}), dict)
            else {},
            "summary": list((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("summary", []))
            if isinstance((benchmark_bundle.get("manifest", {}) if isinstance(benchmark_bundle.get("manifest", {}), dict) else {}).get("summary", []), list)
            else [],
        },
        "claims": claims,
        "primary_claim": primary_claim,
        "backup_claim": backup_claim,
        "believable_submission_path": _STATUS_RANK.get(str(primary_claim.get("status", "")), 0) >= 2
        and _STATUS_RANK.get(str(backup_claim.get("status", "")), 0) >= 1,
        "artifact_portability": _artifact_portability_summary(claims),
        "evaluation_tables": {
            "run_leaderboard": top_model_rows,
            "benchmark_lock": benchmark_table,
        },
        "claim_gaps": claim_gaps[:8],
        "publication_outline": _publication_outline(primary_claim, backup_claim),
    }
    role_map = {
        str(primary_claim.get("key", "")): "primary",
        str(backup_claim.get("key", "")): "backup",
    }
    claim_support_matrix = [
        _claim_support_row(
            claim,
            role=role_map.get(str(claim.get("key", "")), "supporting"),
            benchmark_ready=bool(report["benchmark_lock"]["comparison_ready"]),
        )
        for claim in claims
        if isinstance(claim, dict)
    ]
    report["claim_support_matrix"] = claim_support_matrix
    report["submission_readiness"] = _build_submission_readiness(
        primary_claim=primary_claim,
        backup_claim=backup_claim,
        benchmark_lock=report["benchmark_lock"],
        evaluation_tables=report["evaluation_tables"],
        claim_support_matrix=claim_support_matrix,
        claim_gaps=report["claim_gaps"],
    )
    return report


def write_research_claims_report(
    report: dict[str, object],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    artifact_dir = output_dir.expanduser().resolve() / "analysis" / "research_claims"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    json_path = artifact_dir / "research_claims.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    primary_claim = report.get("primary_claim", {})
    primary_claim = primary_claim if isinstance(primary_claim, dict) else {}
    backup_claim = report.get("backup_claim", {})
    backup_claim = backup_claim if isinstance(backup_claim, dict) else {}
    benchmark_lock = report.get("benchmark_lock", {})
    benchmark_lock = benchmark_lock if isinstance(benchmark_lock, dict) else {}
    claim_support_matrix = report.get("claim_support_matrix", [])
    claim_support_matrix = claim_support_matrix if isinstance(claim_support_matrix, list) else []
    submission_readiness = report.get("submission_readiness", {})
    submission_readiness = submission_readiness if isinstance(submission_readiness, dict) else {}
    evaluation_tables = report.get("evaluation_tables", {})
    evaluation_tables = evaluation_tables if isinstance(evaluation_tables, dict) else {}

    run_leaderboard_csv = _write_csv_rows(
        artifact_dir / "run_leaderboard.csv",
        [row for row in evaluation_tables.get("run_leaderboard", []) if isinstance(row, dict)],
    )
    benchmark_lock_csv = _write_csv_rows(
        artifact_dir / "benchmark_lock_table.csv",
        [row for row in evaluation_tables.get("benchmark_lock", []) if isinstance(row, dict)],
    )
    claim_support_csv = _write_csv_rows(
        artifact_dir / "claim_support_matrix.csv",
        [row for row in claim_support_matrix if isinstance(row, dict)],
    )
    artifact_portability = report.get("artifact_portability", {})
    artifact_portability = artifact_portability if isinstance(artifact_portability, dict) else {}

    lines = [
        "# Research Claims",
        "",
        f"- Run: `{(report.get('run', {}) if isinstance(report.get('run', {}), dict) else {}).get('run_id', '')}`",
        f"- Profile: `{(report.get('run', {}) if isinstance(report.get('run', {}), dict) else {}).get('profile', '')}`",
        f"- Benchmark lock ready: `{benchmark_lock.get('comparison_ready', False)}`",
        f"- Believable submission path: `{report.get('believable_submission_path', False)}`",
        f"- Artifact path mode: `{artifact_portability.get('path_mode', '')}`",
        "",
        "## Primary Claim",
        "",
        f"- Title: {primary_claim.get('title', 'n/a')}",
        f"- Status: `{primary_claim.get('status', '')}`",
        f"- Summary: {primary_claim.get('summary', 'n/a')}",
        "",
    ]
    for item in primary_claim.get("evidence", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Backup Claim",
            "",
            f"- Title: {backup_claim.get('title', 'n/a')}",
            f"- Status: `{backup_claim.get('status', '')}`",
            f"- Summary: {backup_claim.get('summary', 'n/a')}",
            "",
        ]
    )
    for item in backup_claim.get("evidence", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Artifact Portability", ""])
    lines.append(
        f"- Portable supporting artifacts: `{artifact_portability.get('portable_supporting_artifact_count', 0)}` / `{artifact_portability.get('total_supporting_artifact_count', 0)}`"
    )
    lines.append(
        f"- Existing supporting artifacts: `{artifact_portability.get('existing_supporting_artifact_count', 0)}` / `{artifact_portability.get('total_supporting_artifact_count', 0)}`"
    )
    if artifact_portability.get("non_portable_examples"):
        lines.append(
            f"- Non-portable examples: `{', '.join(str(item) for item in artifact_portability.get('non_portable_examples', []))}`"
        )

    lines.extend(["", "## Claim Matrix", ""])
    for claim in report.get("claims", []):
        if not isinstance(claim, dict):
            continue
        lines.append(
            f"- `{claim.get('status', '')}` {claim.get('title', '')} "
            f"(artifacts `{(claim.get('supporting_artifact_summary', {}) if isinstance(claim.get('supporting_artifact_summary', {}), dict) else {}).get('portability_status', 'missing')}`)"
        )

    lines.extend(["", "## Evaluation Tables", "", "### Run Leaderboard", ""])
    for row in (report.get("evaluation_tables", {}) if isinstance(report.get("evaluation_tables", {}), dict) else {}).get("run_leaderboard", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('model_name', '')}` [{row.get('model_type', '')}] "
            f"val_top1=`{_format_metric(row.get('val_top1'))}` test_top1=`{_format_metric(row.get('test_top1'))}`"
        )

    lines.extend(["", "### Benchmark Lock", ""])
    for row in (report.get("evaluation_tables", {}) if isinstance(report.get("evaluation_tables", {}), dict) else {}).get("benchmark_lock", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('model_name', '')}` [{row.get('model_type', '')}] runs=`{row.get('run_count', 0)}` "
            f"mean_val_top1=`{_format_metric(row.get('val_top1_mean'))}` ci95=`{_format_metric(row.get('val_top1_ci95'))}`"
        )

    lines.extend(["", "## Claim Support Matrix", ""])
    for row in claim_support_matrix:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('role', '')}` `{row.get('claim_key', '')}` [{row.get('status', '')}] "
            f"live=`{row.get('live_signal_status', '')}` benchmark=`{row.get('benchmark_evidence_status', '')}` "
            f"repeated=`{row.get('repeated_evidence_status', '')}` slice=`{row.get('slice_evidence_status', '')}` "
            f"risk=`{row.get('risk_evidence_status', '')}` artifacts=`{row.get('artifact_pack_status', '')}` "
            f"portability=`{row.get('artifact_portability_status', '')}`"
        )

    lines.extend(["", "## Submission Readiness", ""])
    lines.append(f"- Status: `{submission_readiness.get('status', '')}`")
    lines.append(f"- Ready for external review: `{submission_readiness.get('ready_for_external_review', False)}`")
    for item in submission_readiness.get("summary", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Missing Checks", ""])
    for item in report.get("claim_gaps", []):
        lines.append(f"- {item}")

    md_path = artifact_dir / "research_claims.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outline_lines = [
        "# Publication Outline",
        "",
        f"- Primary claim: {primary_claim.get('title', 'n/a')}",
        f"- Backup claim: {backup_claim.get('title', 'n/a')}",
        "",
    ]
    for section in report.get("publication_outline", []):
        if not isinstance(section, dict):
            continue
        outline_lines.extend(
            [
                f"## {section.get('section', 'Section')}",
                "",
                f"- Focus: {section.get('focus', '')}",
            ]
        )
        for note in section.get("notes", []):
            outline_lines.append(f"- {note}")
        outline_lines.append("")

    outline_path = artifact_dir / "publication_outline.md"
    outline_path.write_text("\n".join(outline_lines).rstrip() + "\n", encoding="utf-8")

    claim_support_md_lines = [
        "# Claim Support Matrix",
        "",
        f"- Run: `{(report.get('run', {}) if isinstance(report.get('run', {}), dict) else {}).get('run_id', '')}`",
        "",
    ]
    for row in claim_support_matrix:
        if not isinstance(row, dict):
            continue
        claim_support_md_lines.append(
            f"- `{row.get('role', '')}` `{row.get('claim_key', '')}`: benchmark `{row.get('benchmark_evidence_status', '')}`, "
            f"repeated `{row.get('repeated_evidence_status', '')}`, slice `{row.get('slice_evidence_status', '')}`, "
            f"risk `{row.get('risk_evidence_status', '')}`, artifacts `{row.get('artifact_pack_status', '')}`, "
            f"portability `{row.get('artifact_portability_status', '')}`."
        )
        claim_support_md_lines.append(f"Next gate: {row.get('next_gate', '')}")
    claim_support_md = artifact_dir / "claim_support_matrix.md"
    claim_support_md.write_text("\n".join(claim_support_md_lines).rstrip() + "\n", encoding="utf-8")

    submission_readiness_json = artifact_dir / "submission_readiness.json"
    submission_readiness_json.write_text(json.dumps(submission_readiness, indent=2), encoding="utf-8")
    submission_readiness_md_lines = [
        "# Submission Readiness",
        "",
        f"- Status: `{submission_readiness.get('status', '')}`",
        f"- Ready for external review: `{submission_readiness.get('ready_for_external_review', False)}`",
        "",
        "## Summary",
        "",
    ]
    for item in submission_readiness.get("summary", []):
        submission_readiness_md_lines.append(f"- {item}")
    submission_readiness_md_lines.extend(["", "## Checks", ""])
    for item in submission_readiness.get("checks", []):
        if not isinstance(item, dict):
            continue
        submission_readiness_md_lines.append(
            f"- `{item.get('key', '')}` [{item.get('status', '')}]: {item.get('detail', '')}"
        )
    submission_readiness_md_lines.extend(["", "## Blockers", ""])
    for item in submission_readiness.get("blockers", []):
        submission_readiness_md_lines.append(f"- {item}")
    submission_readiness_md = artifact_dir / "submission_readiness.md"
    submission_readiness_md.write_text("\n".join(submission_readiness_md_lines).rstrip() + "\n", encoding="utf-8")
    return {
        "json": json_path,
        "md": md_path,
        "outline_md": outline_path,
        "run_leaderboard_csv": run_leaderboard_csv or artifact_dir / "run_leaderboard.csv",
        "benchmark_lock_csv": benchmark_lock_csv or artifact_dir / "benchmark_lock_table.csv",
        "claim_support_csv": claim_support_csv or artifact_dir / "claim_support_matrix.csv",
        "claim_support_md": claim_support_md,
        "submission_readiness_json": submission_readiness_json,
        "submission_readiness_md": submission_readiness_md,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a research-claim pack from the current run and benchmark artifacts.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Outputs root containing runs and history.")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit outputs/runs/<run_id> directory.")
    parser.add_argument(
        "--benchmark-manifest",
        type=str,
        default=None,
        help="Explicit outputs/history/benchmark_lock_<id>_manifest.json file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    benchmark_manifest = Path(args.benchmark_manifest).expanduser().resolve() if args.benchmark_manifest else None

    report = build_research_claims_report(
        output_dir,
        run_dir=run_dir,
        benchmark_manifest_path=benchmark_manifest,
    )
    paths = write_research_claims_report(report, output_dir=output_dir)
    primary_claim = report.get("primary_claim", {})
    primary_claim = primary_claim if isinstance(primary_claim, dict) else {}
    backup_claim = report.get("backup_claim", {})
    backup_claim = backup_claim if isinstance(backup_claim, dict) else {}
    print(f"research_claims_json={paths['json']}")
    print(f"research_claims_md={paths['md']}")
    print(f"publication_outline_md={paths['outline_md']}")
    print(f"claim_support_matrix_md={paths['claim_support_md']}")
    print(f"submission_readiness_md={paths['submission_readiness_md']}")
    print(f"primary_claim={primary_claim.get('key', '')}")
    print(f"primary_status={primary_claim.get('status', '')}")
    print(f"backup_claim={backup_claim.get('key', '')}")
    print(f"backup_status={backup_claim.get('status', '')}")
    print(f"believable_submission_path={report.get('believable_submission_path', False)}")
    return 0


__all__ = [
    "build_research_claims_report",
    "write_research_claims_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
