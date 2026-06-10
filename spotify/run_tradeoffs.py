from __future__ import annotations

import math
import os
from pathlib import Path

from .run_artifacts import safe_read_json

_DEFAULT_MAX_SCAN_ENTRIES = 20_000
_MAX_TIMING_ARTIFACT_BYTES = 10 * 1024 * 1024
_RUNTIME_REGRESSION_FRACTION = 0.05
_RUNTIME_REGRESSION_SECONDS = 1.0
_STORAGE_REGRESSION_FRACTION = 0.05
_STORAGE_REGRESSION_BYTES = 1024 * 1024
_WORKLOAD_FIELDS = (
    "deep_models",
    "classical_models",
    "enable_retrieval_stack",
    "enable_self_supervised_pretraining",
    "enable_friction_analysis",
    "enable_moonshot_lab",
    "enable_optuna",
    "optuna_models",
    "optuna_trials",
    "enable_temporal_backtest",
    "temporal_backtest_models",
    "temporal_backtest_folds",
    "temporal_backtest_adaptation_mode",
)
_CACHE_METADATA_KEYS = (
    "cache_enabled",
    "cache_hit",
    "cache_hit_count",
    "cache_miss_count",
    "backtest_cache_enabled",
    "backtest_cache_hit",
    "reporting_cache_reused",
)


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _delta(current: object, baseline: object) -> float:
    current_value = _safe_float(current)
    baseline_value = _safe_float(baseline)
    if not math.isfinite(current_value) or not math.isfinite(baseline_value):
        return float("nan")
    return current_value - baseline_value


def _delta_percent(current: object, baseline: object) -> float:
    baseline_value = _safe_float(baseline)
    difference = _delta(current, baseline)
    if not math.isfinite(difference) or not math.isfinite(baseline_value) or baseline_value <= 0.0:
        return float("nan")
    return (difference / baseline_value) * 100.0


def _run_dir(manifest: dict[str, object]) -> Path | None:
    raw_path = str(manifest.get("run_dir", "")).strip()
    if not raw_path:
        return None
    return Path(raw_path).expanduser()


def _timing_snapshot(manifest: dict[str, object]) -> dict[str, object]:
    run_dir = _run_dir(manifest)
    timing_path = run_dir / "run_phase_timings.json" if run_dir is not None else None
    source = "artifact"
    payload = None
    if timing_path is not None:
        try:
            if timing_path.is_symlink():
                source = "artifact_symlink_rejected"
            elif timing_path.is_file() and timing_path.stat().st_size <= _MAX_TIMING_ARTIFACT_BYTES:
                payload = safe_read_json(timing_path, default=None)
            elif timing_path.is_file():
                source = "artifact_too_large"
        except OSError:
            source = "artifact_unreadable"
    if not isinstance(payload, dict):
        summary = manifest.get("phase_timings", {})
        payload = dict(summary) if isinstance(summary, dict) else {}
        if payload:
            source = "manifest_summary"
        elif source == "artifact":
            source = "missing"

    phases = payload.get("phases", []) if isinstance(payload, dict) else []
    phase_rows = [dict(row) for row in phases if isinstance(row, dict)] if isinstance(phases, list) else []
    total_seconds = _safe_float(payload.get("total_seconds"))
    measured_seconds = _safe_float(payload.get("measured_seconds"))
    overhead_seconds = _safe_float(payload.get("unmeasured_overhead_seconds"))
    return {
        "available": math.isfinite(total_seconds),
        "phase_details_available": bool(phase_rows),
        "source": source,
        "timing_path": str(timing_path) if timing_path is not None else "",
        "run_id": str(payload.get("run_id", "")),
        "final_status": str(payload.get("final_status", "")),
        "total_seconds": total_seconds,
        "measured_seconds": measured_seconds,
        "unmeasured_overhead_seconds": overhead_seconds,
        "phases": phase_rows,
    }


def _storage_snapshot(run_dir: Path | None, *, max_entries: int) -> dict[str, object]:
    if run_dir is None or not run_dir.is_dir():
        return {
            "available": False,
            "complete": False,
            "run_dir": str(run_dir or ""),
            "retained_bytes": float("nan"),
            "file_count": 0,
            "directory_count": 0,
            "symlink_count": 0,
            "entries_scanned": 0,
            "max_entries": int(max_entries),
            "errors": ["run_directory_missing"],
        }

    retained_bytes = 0
    file_count = 0
    directory_count = 0
    symlink_count = 0
    entries_scanned = 0
    errors: list[str] = []
    truncated = False
    pending = [run_dir]

    while pending and not truncated:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_scanned += 1
                    if entries_scanned > max_entries:
                        truncated = True
                        break
                    try:
                        if entry.is_symlink():
                            symlink_count += 1
                            retained_bytes += int(entry.stat(follow_symlinks=False).st_size)
                        elif entry.is_dir(follow_symlinks=False):
                            directory_count += 1
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            file_count += 1
                            retained_bytes += int(entry.stat(follow_symlinks=False).st_size)
                    except OSError as exc:
                        errors.append(f"{entry.path}: {type(exc).__name__}")
        except OSError as exc:
            errors.append(f"{directory}: {type(exc).__name__}")

    if truncated:
        errors.append("scan_entry_limit_reached")
    complete = not errors
    return {
        "available": True,
        "complete": complete,
        "run_dir": str(run_dir),
        "retained_bytes": int(retained_bytes),
        "file_count": int(file_count),
        "directory_count": int(directory_count),
        "symlink_count": int(symlink_count),
        "entries_scanned": int(min(entries_scanned, max_entries)),
        "max_entries": int(max_entries),
        "errors": errors[:10],
    }


def _phase_index(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        phase_name = str(row.get("phase_name", "")).strip()
        if not phase_name:
            continue
        phase = grouped.setdefault(
            phase_name,
            {
                "duration_seconds": 0.0,
                "statuses": [],
                "cache_signatures": [],
            },
        )
        duration = _safe_float(row.get("duration_seconds"))
        if math.isfinite(duration):
            phase["duration_seconds"] = float(phase["duration_seconds"]) + duration
        status = str(row.get("status", "")).strip() or "unknown"
        statuses = phase["statuses"]
        if isinstance(statuses, list):
            statuses.append(status)
        metadata = row.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        signature = tuple(
            (key, repr(metadata.get(key)))
            for key in _CACHE_METADATA_KEYS
            if key in metadata
        )
        signatures = phase["cache_signatures"]
        if signature and isinstance(signatures, list):
            signatures.append(signature)

    for phase in grouped.values():
        statuses = phase.get("statuses", [])
        phase["status"] = "|".join(sorted(set(str(item) for item in statuses))) if isinstance(statuses, list) else ""
        signatures = phase.get("cache_signatures", [])
        phase["cache_signature"] = tuple(signatures) if isinstance(signatures, list) else ()
    return grouped


def _blocker(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _manifest_comparability(
    selected: dict[str, object],
    baseline: dict[str, object],
) -> tuple[list[dict[str, str]], list[str]]:
    blockers: list[dict[str, str]] = []
    notes: list[str] = []

    selected_profile = str(selected.get("profile", "")).strip()
    baseline_profile = str(baseline.get("profile", "")).strip()
    if selected_profile and baseline_profile and selected_profile != baseline_profile:
        blockers.append(
            _blocker(
                "profile_mismatch",
                f"Selected profile `{selected_profile}` differs from baseline profile `{baseline_profile}`.",
            )
        )
    elif not selected_profile or not baseline_profile:
        notes.append("At least one run is missing its profile, so workload equivalence is not fully documented.")

    shape_mismatches: list[str] = []
    for key in ("data_records", "num_artists", "num_context_features"):
        selected_value = selected.get(key)
        baseline_value = baseline.get(key)
        if selected_value is None or baseline_value is None:
            continue
        if str(selected_value) != str(baseline_value):
            shape_mismatches.append(f"{key}={baseline_value}->{selected_value}")
    if shape_mismatches:
        blockers.append(
            _blocker(
                "data_shape_mismatch",
                f"Run data shape changed: {', '.join(shape_mismatches)}.",
            )
        )

    workload_mismatches = [
        key
        for key in _WORKLOAD_FIELDS
        if key in selected and key in baseline and selected.get(key) != baseline.get(key)
    ]
    if workload_mismatches:
        blockers.append(
            _blocker(
                "workload_configuration_mismatch",
                f"Workload configuration differs for: {', '.join(workload_mismatches)}.",
            )
        )

    selected_cleanup = selected.get("artifact_cleanup", {})
    baseline_cleanup = baseline.get("artifact_cleanup", {})
    selected_cleanup = selected_cleanup if isinstance(selected_cleanup, dict) else {}
    baseline_cleanup = baseline_cleanup if isinstance(baseline_cleanup, dict) else {}
    cleanup_fields = ("enabled", "mode", "threshold_mb")
    cleanup_mismatches = [
        key
        for key in cleanup_fields
        if key in selected_cleanup
        and key in baseline_cleanup
        and selected_cleanup.get(key) != baseline_cleanup.get(key)
    ]
    if cleanup_mismatches:
        blockers.append(
            _blocker(
                "artifact_cleanup_policy_mismatch",
                f"Artifact cleanup policy differs for: {', '.join(cleanup_mismatches)}.",
            )
        )
    if not selected_cleanup or not baseline_cleanup:
        notes.append("Cleanup policy metadata is missing for at least one run; storage reflects only bytes retained now.")
    return blockers, notes


def _timing_comparison(
    selected: dict[str, object],
    baseline: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, str]], list[str]]:
    blockers: list[dict[str, str]] = []
    notes: list[str] = []
    selected_available = bool(selected.get("available"))
    baseline_available = bool(baseline.get("available"))
    if not selected_available:
        blockers.append(_blocker("selected_timing_missing", "Selected run has no usable timing artifact or summary."))
    if not baseline_available:
        blockers.append(_blocker("baseline_timing_missing", "Baseline run has no usable timing artifact or summary."))

    selected_run_id = str(selected.get("run_id", "")).strip()
    baseline_run_id = str(baseline.get("run_id", "")).strip()
    if selected_run_id and baseline_run_id and selected_run_id == baseline_run_id:
        blockers.append(_blocker("timing_run_identity_collision", "Timing artifacts identify the same run on both sides."))

    if selected.get("source") == "manifest_summary" or baseline.get("source") == "manifest_summary":
        notes.append("At least one timing comparison uses a manifest summary because detailed timing JSON is unavailable.")

    phase_regressions: list[dict[str, object]] = []
    if selected_available and baseline_available:
        selected_status = str(selected.get("final_status", "")).strip()
        baseline_status = str(baseline.get("final_status", "")).strip()
        if selected_status and baseline_status and selected_status != baseline_status:
            blockers.append(
                _blocker(
                    "pipeline_status_mismatch",
                    f"Timing final status differs: baseline `{baseline_status}`, selected `{selected_status}`.",
                )
            )

        selected_phases = _phase_index(selected.get("phases", []) if isinstance(selected.get("phases"), list) else [])
        baseline_phases = _phase_index(baseline.get("phases", []) if isinstance(baseline.get("phases"), list) else [])
        if not selected_phases or not baseline_phases:
            blockers.append(
                _blocker(
                    "phase_details_missing",
                    "Detailed phase rows are unavailable for at least one run.",
                )
            )
        else:
            selected_names = set(selected_phases)
            baseline_names = set(baseline_phases)
            if selected_names != baseline_names:
                missing_selected = sorted(baseline_names - selected_names)
                missing_baseline = sorted(selected_names - baseline_names)
                detail_parts = []
                if missing_selected:
                    detail_parts.append(f"missing from selected: {', '.join(missing_selected[:5])}")
                if missing_baseline:
                    detail_parts.append(f"missing from baseline: {', '.join(missing_baseline[:5])}")
                blockers.append(_blocker("phase_set_mismatch", f"Phase sets differ ({'; '.join(detail_parts)})."))

            status_mismatches: list[str] = []
            cache_mismatches: list[str] = []
            for phase_name in sorted(selected_names.intersection(baseline_names)):
                selected_phase = selected_phases[phase_name]
                baseline_phase = baseline_phases[phase_name]
                selected_phase_status = str(selected_phase.get("status", ""))
                baseline_phase_status = str(baseline_phase.get("status", ""))
                if selected_phase_status != baseline_phase_status:
                    status_mismatches.append(phase_name)
                    continue
                selected_cache = selected_phase.get("cache_signature", ())
                baseline_cache = baseline_phase.get("cache_signature", ())
                if selected_cache and baseline_cache and selected_cache != baseline_cache:
                    cache_mismatches.append(phase_name)

                selected_duration = _safe_float(selected_phase.get("duration_seconds"))
                baseline_duration = _safe_float(baseline_phase.get("duration_seconds"))
                difference = _delta(selected_duration, baseline_duration)
                if not math.isfinite(difference) or difference <= 0.0:
                    continue
                phase_regressions.append(
                    {
                        "phase_name": phase_name,
                        "baseline_status": baseline_phase_status,
                        "selected_status": selected_phase_status,
                        "baseline_duration_seconds": baseline_duration,
                        "selected_duration_seconds": selected_duration,
                        "delta_seconds": difference,
                        "delta_percent": _delta_percent(selected_duration, baseline_duration),
                    }
                )

            if status_mismatches:
                blockers.append(
                    _blocker(
                        "phase_status_mismatch",
                        f"Phase status differs for: {', '.join(status_mismatches[:8])}.",
                    )
                )
            if cache_mismatches:
                blockers.append(
                    _blocker(
                        "cache_state_mismatch",
                        f"Cache state differs for: {', '.join(cache_mismatches[:8])}.",
                    )
                )

    phase_regressions.sort(
        key=lambda row: _safe_float(row.get("delta_seconds")),
        reverse=True,
    )
    runtime = {
        "available": selected_available and baseline_available,
        "selected": {key: selected.get(key) for key in selected if key != "phases"},
        "baseline": {key: baseline.get(key) for key in baseline if key != "phases"},
        "delta_seconds": _delta(selected.get("total_seconds"), baseline.get("total_seconds")),
        "delta_percent": _delta_percent(selected.get("total_seconds"), baseline.get("total_seconds")),
        "measured_delta_seconds": _delta(selected.get("measured_seconds"), baseline.get("measured_seconds")),
        "measured_delta_percent": _delta_percent(selected.get("measured_seconds"), baseline.get("measured_seconds")),
        "overhead_delta_seconds": _delta(
            selected.get("unmeasured_overhead_seconds"),
            baseline.get("unmeasured_overhead_seconds"),
        ),
        "overhead_delta_percent": _delta_percent(
            selected.get("unmeasured_overhead_seconds"),
            baseline.get("unmeasured_overhead_seconds"),
        ),
    }
    return runtime, phase_regressions, blockers, notes


def _quality_rows(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        return []
    return [
        {
            "key": str(row.get("key", "")),
            "label": str(row.get("label", "")),
            "current": row.get("current"),
            "baseline": row.get("baseline"),
            "delta": row.get("delta"),
            "status": str(row.get("status", "unknown")),
            "direction": str(row.get("direction", "")),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _is_material_regression(delta_value: object, baseline_value: object, *, fraction: float, floor: float) -> bool:
    difference = _safe_float(delta_value)
    baseline = _safe_float(baseline_value)
    if not math.isfinite(difference) or difference <= 0.0:
        return False
    threshold = floor
    if math.isfinite(baseline) and baseline > 0.0:
        threshold = max(threshold, baseline * fraction)
    return difference > threshold


def _build_verdict(
    *,
    baseline_available: bool,
    selected_manifest: dict[str, object],
    blockers: list[dict[str, str]],
    runtime: dict[str, object],
    storage: dict[str, object],
    quality_rows: list[dict[str, object]],
) -> tuple[str, str, list[str]]:
    worse_rows = [row for row in quality_rows if str(row.get("status")) == "worse"]
    better_rows = [row for row in quality_rows if str(row.get("status")) == "better"]
    gate = selected_manifest.get("champion_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    promotion_known = "promoted" in gate
    selected_promoted = bool(gate.get("promoted"))

    if not baseline_available:
        return (
            "insufficient_evidence",
            "No promoted baseline is available, so the selected run cannot receive a tradeoff verdict.",
            ["Establish a promoted baseline before interpreting runtime, storage, or quality tradeoffs."],
        )
    if worse_rows:
        labels = ", ".join(str(row.get("label", "")) for row in worse_rows[:4])
        reasons = [f"Tracked quality or safety regressed for: {labels}."]
        if blockers:
            reasons.append(f"`{len(blockers)}` comparability blocker(s) also limit the resource comparison.")
        return (
            "baseline_preferred",
            "Keep the promoted baseline: at least one available quality or safety signal regressed.",
            reasons,
        )
    if promotion_known and not selected_promoted:
        return (
            "baseline_preferred",
            "Keep the promoted baseline because the selected run did not pass promotion.",
            [f"Selected promotion status is `{gate.get('status', 'unknown')}`."],
        )
    if blockers:
        return (
            "not_comparable",
            "Do not choose between runs from this dossier until the comparability blockers are cleared.",
            [str(item.get("detail", "")) for item in blockers[:4]],
        )

    runtime_baseline = runtime.get("baseline", {})
    runtime_baseline = runtime_baseline if isinstance(runtime_baseline, dict) else {}
    storage_baseline = storage.get("baseline", {})
    storage_baseline = storage_baseline if isinstance(storage_baseline, dict) else {}
    runtime_regression = _is_material_regression(
        runtime.get("delta_seconds"),
        runtime_baseline.get("total_seconds"),
        fraction=_RUNTIME_REGRESSION_FRACTION,
        floor=_RUNTIME_REGRESSION_SECONDS,
    )
    storage_regression = _is_material_regression(
        storage.get("delta_bytes"),
        storage_baseline.get("retained_bytes"),
        fraction=_STORAGE_REGRESSION_FRACTION,
        floor=_STORAGE_REGRESSION_BYTES,
    )
    if runtime_regression or storage_regression:
        resources = []
        if runtime_regression:
            resources.append("runtime")
        if storage_regression:
            resources.append("retained storage")
        return (
            "tradeoff_review",
            "Quality and safety do not regress, but the selected run has a material resource cost that needs review.",
            [f"Material increase detected in {' and '.join(resources)}."],
        )

    runtime_improved = _safe_float(runtime.get("delta_seconds")) < -_RUNTIME_REGRESSION_SECONDS
    storage_improved = _safe_float(storage.get("delta_bytes")) < -_STORAGE_REGRESSION_BYTES
    if better_rows or runtime_improved or storage_improved:
        reasons = []
        if better_rows:
            reasons.append(f"`{len(better_rows)}` tracked quality or safety metric(s) improved.")
        if runtime_improved:
            reasons.append("Total runtime decreased materially.")
        if storage_improved:
            reasons.append("Currently retained run-directory bytes decreased materially.")
        return (
            "selected_preferred",
            "The selected run is the better documented tradeoff under the available evidence.",
            reasons,
        )
    return (
        "no_clear_winner",
        "The selected run is broadly flat against the promoted baseline with no material resource advantage.",
        ["Keep the promoted baseline unless there is an untracked product reason to switch."],
    )


def build_run_tradeoff_dossier(
    *,
    selected_manifest: dict[str, object],
    baseline_manifest: dict[str, object] | None,
    quality_safety_deltas: object = None,
    top_phase_regressions: int = 5,
    max_scan_entries: int = _DEFAULT_MAX_SCAN_ENTRIES,
) -> dict[str, object]:
    selected = dict(selected_manifest) if isinstance(selected_manifest, dict) else {}
    baseline = dict(baseline_manifest) if isinstance(baseline_manifest, dict) else {}
    scan_limit = max(1, int(max_scan_entries))
    selected_run_id = str(selected.get("run_id", "")).strip()
    baseline_run_id = str(baseline.get("run_id", "")).strip()
    baseline_available = bool(baseline)

    blockers: list[dict[str, str]] = []
    notes = [
        "Storage is a read-only snapshot of bytes currently retained in each run directory, not original bytes produced.",
    ]
    if not selected:
        blockers.append(_blocker("selected_manifest_missing", "Selected run manifest is unavailable."))
    if not baseline_available:
        blockers.append(_blocker("promoted_baseline_missing", "No promoted baseline manifest is available."))
    elif selected_run_id and baseline_run_id and selected_run_id == baseline_run_id:
        blockers.append(_blocker("same_run_selected_and_baseline", "Selected run and promoted baseline are the same run."))

    if selected and baseline:
        manifest_blockers, manifest_notes = _manifest_comparability(selected, baseline)
        blockers.extend(manifest_blockers)
        notes.extend(manifest_notes)

    selected_timing = _timing_snapshot(selected)
    baseline_timing = _timing_snapshot(baseline)
    selected_timing_run_id = str(selected_timing.get("run_id", "")).strip()
    baseline_timing_run_id = str(baseline_timing.get("run_id", "")).strip()
    if selected_run_id and selected_timing_run_id and selected_run_id != selected_timing_run_id:
        blockers.append(
            _blocker(
                "selected_timing_run_id_mismatch",
                f"Selected timing artifact identifies `{selected_timing_run_id}`, not `{selected_run_id}`.",
            )
        )
    if baseline_run_id and baseline_timing_run_id and baseline_run_id != baseline_timing_run_id:
        blockers.append(
            _blocker(
                "baseline_timing_run_id_mismatch",
                f"Baseline timing artifact identifies `{baseline_timing_run_id}`, not `{baseline_run_id}`.",
            )
        )
    runtime, phase_regressions, timing_blockers, timing_notes = _timing_comparison(
        selected_timing,
        baseline_timing,
    )
    if baseline_available:
        blockers.extend(timing_blockers)
        notes.extend(timing_notes)

    selected_storage = _storage_snapshot(_run_dir(selected), max_entries=scan_limit)
    baseline_storage = _storage_snapshot(_run_dir(baseline), max_entries=scan_limit)
    storage_available = bool(selected_storage.get("complete")) and bool(baseline_storage.get("complete"))
    if baseline_available:
        if not bool(selected_storage.get("complete")):
            blockers.append(
                _blocker(
                    "selected_storage_scan_incomplete",
                    "Selected run-directory byte scan was missing, truncated, or encountered an error.",
                )
            )
        if not bool(baseline_storage.get("complete")):
            blockers.append(
                _blocker(
                    "baseline_storage_scan_incomplete",
                    "Baseline run-directory byte scan was missing, truncated, or encountered an error.",
                )
            )
    storage = {
        "available": storage_available,
        "selected": selected_storage,
        "baseline": baseline_storage,
        "delta_bytes": (
            _delta(selected_storage.get("retained_bytes"), baseline_storage.get("retained_bytes"))
            if storage_available
            else float("nan")
        ),
        "delta_percent": (
            _delta_percent(selected_storage.get("retained_bytes"), baseline_storage.get("retained_bytes"))
            if storage_available
            else float("nan")
        ),
        "file_count_delta": (
            int(selected_storage.get("file_count", 0)) - int(baseline_storage.get("file_count", 0))
            if storage_available
            else 0
        ),
    }

    quality_rows = _quality_rows(quality_safety_deltas)
    known_quality_rows = [row for row in quality_rows if str(row.get("status")) != "unknown"]
    unknown_quality_rows = [row for row in quality_rows if str(row.get("status")) == "unknown"]
    if baseline_available and not known_quality_rows:
        blockers.append(
            _blocker(
                "quality_safety_deltas_missing",
                "No comparable quality or safety deltas are available.",
            )
        )
    elif unknown_quality_rows:
        labels = ", ".join(str(row.get("label", "")) for row in unknown_quality_rows[:5])
        notes.append(f"Some quality or safety deltas are unavailable: {labels}.")
        blockers.append(
            _blocker(
                "quality_safety_metrics_incomplete",
                f"Some quality or safety deltas are unavailable: {labels}.",
            )
        )

    deduplicated_blockers: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for item in blockers:
        code = str(item.get("code", "")).strip()
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        deduplicated_blockers.append(item)
    blockers = deduplicated_blockers

    verdict, verdict_summary, verdict_reasons = _build_verdict(
        baseline_available=baseline_available,
        selected_manifest=selected,
        blockers=blockers,
        runtime=runtime,
        storage=storage,
        quality_rows=quality_rows,
    )
    if not baseline_available:
        status = "unavailable"
    elif blockers:
        status = "partial"
    else:
        status = "complete"

    summary = [verdict_summary]
    if runtime.get("available"):
        summary.append(
            f"Runtime delta is `{_safe_float(runtime.get('delta_seconds')):.3f}` seconds "
            f"(`{_safe_float(runtime.get('delta_percent')):.1f}%`)."
        )
    if storage.get("available"):
        summary.append(
            f"Retained storage delta is `{int(_safe_float(storage.get('delta_bytes')))}` bytes "
            f"(`{_safe_float(storage.get('delta_percent')):.1f}%`)."
        )
    if blockers:
        summary.append(f"Comparability is limited by `{len(blockers)}` blocker(s).")

    return {
        "status": status,
        "comparison_mode": "selected_vs_promoted_baseline",
        "selected_run": {
            "run_id": selected_run_id,
            "profile": str(selected.get("profile", "")),
            "timestamp": str(selected.get("timestamp", "")),
        },
        "baseline_run": {
            "run_id": baseline_run_id,
            "profile": str(baseline.get("profile", "")),
            "timestamp": str(baseline.get("timestamp", "")),
        },
        "comparability": {
            "comparable": baseline_available and not blockers,
            "blocker_count": int(len(blockers)),
            "blockers": blockers,
            "notes": notes[:10],
        },
        "runtime": runtime,
        "storage": storage,
        "largest_phase_regressions": phase_regressions[: max(1, int(top_phase_regressions))],
        "quality_safety_deltas": quality_rows,
        "quality_safety_better_count": sum(1 for row in quality_rows if str(row.get("status")) == "better"),
        "quality_safety_worse_count": sum(1 for row in quality_rows if str(row.get("status")) == "worse"),
        "verdict": verdict,
        "verdict_summary": verdict_summary,
        "verdict_reasons": verdict_reasons,
        "summary": summary,
    }


__all__ = ["build_run_tradeoff_dossier"]
