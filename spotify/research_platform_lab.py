from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .run_artifacts import collect_run_manifests
from .run_artifacts import latest_manifest_run_dir
from .run_artifacts import safe_read_csv
from .run_artifacts import safe_read_json
from .run_artifacts import write_csv_rows
from .run_artifacts import write_json
from .run_artifacts import write_markdown


_READY_STATUSES = {"analysis_ready", "submission_candidate"}
_REVIEW_READY_STATUSES = {"ready", "n/a"}
_GAP_STATUSES = {"gap", "missing", "attention", "blocked", "incomplete", "stale"}
_CREATOR_EVIDENCE_GRADES = {"publishable", "watch_only", "suppress"}


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _finite_or_none(value: object) -> float | None:
    metric = _safe_float(value)
    return metric if math.isfinite(metric) else None


def _safe_int(value: object) -> int:
    metric = _safe_float(value)
    return int(metric) if math.isfinite(metric) else 0


def _format_metric(value: object) -> str:
    metric = _safe_float(value)
    return f"{metric:.3f}" if math.isfinite(metric) else "n/a"


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if not str(value).strip():
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _parse_json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if not str(value).strip():
        return {}
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "experiment"


def _status_is_gap(value: object) -> bool:
    normalized = str(value).strip().lower()
    return bool(normalized and normalized in _GAP_STATUSES)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _rows_for_columns(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    trimmed = frame.copy()
    for column in columns:
        if column not in trimmed.columns:
            trimmed[column] = None
    return trimmed[columns].to_dict(orient="records")


def _normalize_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.where(numeric.notna(), 0.5, np.nan), index=series.index, dtype="float64")
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.where(numeric.notna(), 0.5, np.nan), index=series.index, dtype="float64")
    scaled = (numeric - min_value) / (max_value - min_value)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.astype("float64", copy=False)


def _resolve_anchor_run_dir(output_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    claims_payload = safe_read_json(output_dir / "analysis" / "research_claims" / "research_claims.json", default={})
    if isinstance(claims_payload, dict):
        run_payload = claims_payload.get("run", {})
        if isinstance(run_payload, dict):
            run_id = str(run_payload.get("run_id", "")).strip()
            if run_id:
                candidate = (output_dir / "runs" / run_id).resolve()
                if candidate.exists():
                    return candidate
    latest_dir = latest_manifest_run_dir(output_dir)
    if latest_dir is not None:
        return latest_dir
    raise FileNotFoundError("No completed run is available for research platform analysis.")


def _load_control_room_history(output_dir: Path) -> pd.DataFrame:
    history = safe_read_csv(output_dir / "analytics" / "control_room_history.csv")
    if history.empty:
        return history
    for column in [
        "target_drift_jsd",
        "test_selective_risk",
        "test_abstention_rate",
        "robustness_gap",
        "stress_skip_risk",
        "ops_coverage_ratio",
    ]:
        history[column] = pd.to_numeric(history.get(column), errors="coerce")
    return history


def _find_matching_history_row(history: pd.DataFrame, run_id: str) -> dict[str, object]:
    if history.empty or "run_id" not in history.columns:
        return {}
    matches = history.loc[history["run_id"].astype(str) == run_id]
    if matches.empty:
        return {}
    return matches.iloc[-1].to_dict()


def _artifact_ratio(flags: dict[str, bool]) -> float:
    if not flags:
        return float("nan")
    return float(sum(1 for value in flags.values() if value) / len(flags))


def _timestamp(path: Path | None) -> datetime | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _isoformat(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _age_hours(value: datetime | None, *, now: datetime) -> float:
    if value is None:
        return float("nan")
    return round((now - value).total_seconds() / 3600.0, 2)


def _coerce_paths(values: list[object]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        candidate = str(value).strip()
        if not candidate:
            continue
        out.append(Path(candidate).expanduser().resolve())
    return out


def _latest_existing_path(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _artifact_path_health(
    *,
    artifact_paths: list[Path],
    reference_paths: list[Path] | None = None,
    now: datetime,
) -> dict[str, object]:
    existing_paths = [path for path in artifact_paths if path.exists()]
    missing_paths = [path for path in artifact_paths if not path.exists()]
    reference_path = _latest_existing_path(reference_paths or [])
    reference_ts = _timestamp(reference_path)
    stale_paths = [
        path
        for path in existing_paths
        if reference_ts is not None
        and (_timestamp(path) or reference_ts) > reference_ts
    ]
    newest_existing = _latest_existing_path(existing_paths)
    path_status = (
        "missing"
        if not artifact_paths or not existing_paths
        else "attention"
        if missing_paths
        else "ready"
    )
    freshness_status = (
        "stale"
        if stale_paths
        else "ready"
        if reference_ts is not None and existing_paths
        else "n/a"
    )
    return {
        "path_status": path_status,
        "freshness_status": freshness_status,
        "existing_count": int(len(existing_paths)),
        "missing_count": int(len(missing_paths)),
        "stale_count": int(len(stale_paths)),
        "missing_path": str(missing_paths[0]) if missing_paths else "",
        "stale_path": str(stale_paths[0]) if stale_paths else "",
        "newest_path": str(newest_existing) if newest_existing is not None else "",
        "newest_path_timestamp": _isoformat(_timestamp(newest_existing)),
        "newest_path_age_hours": _age_hours(_timestamp(newest_existing), now=now),
        "reference_path": str(reference_path) if reference_path is not None else "",
        "reference_timestamp": _isoformat(reference_ts),
    }


def _resolve_creator_evidence_path(output_dir: Path, evidence_root: Path, raw_path: object) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [output_dir / candidate, evidence_root / candidate]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _creator_evidence_gate(passport: dict[str, object], key: str) -> dict[str, object]:
    gates = passport.get("gates", [])
    if not isinstance(gates, list):
        return {}
    for gate in gates:
        if isinstance(gate, dict) and str(gate.get("key", "")).strip() == key:
            return gate
    return {}


def _creator_evidence_freshness_status(
    passport: dict[str, object],
    *,
    source_health: dict[str, object],
) -> str:
    if str(source_health.get("freshness_status", "")) == "stale":
        return "stale"
    freshness_gate = _creator_evidence_gate(passport, "evidence_freshness")
    gate_status = str(freshness_gate.get("status", "")).strip().lower()
    if gate_status == "pass":
        return "ready"
    if gate_status == "fail":
        return "stale"
    if gate_status == "watch":
        observed = freshness_gate.get("observed", {})
        threshold = freshness_gate.get("threshold", {})
        observed = observed if isinstance(observed, dict) else {}
        threshold = threshold if isinstance(threshold, dict) else {}
        age = observed.get("latest_source_age_days")
        maximum_age = threshold.get("maximum_source_age_days")
        try:
            if age is not None and maximum_age is not None and float(age) > float(maximum_age):
                return "stale"
        except (TypeError, ValueError):
            pass
        return "attention"
    return "missing"


def _build_creator_evidence_registry(
    output_dir: Path,
    *,
    now: datetime,
) -> tuple[pd.DataFrame, dict[str, object]]:
    evidence_root = output_dir / "analysis" / "creator_evidence_lab"
    manifest_path = evidence_root / "creator_evidence_manifest.json"
    passports_path = evidence_root / "creator_opportunity_evidence_passports.json"
    manifest_payload = safe_read_json(manifest_path, default={})
    manifest = manifest_payload if isinstance(manifest_payload, dict) else {}
    passports_payload = safe_read_json(passports_path, default=[])
    passports = [row for row in passports_payload if isinstance(row, dict)] if isinstance(passports_payload, list) else []

    artifact_paths_payload = manifest.get("artifact_paths", {})
    expected_artifact_keys = {"json", "csv", "markdown", "manifest"}
    artifact_paths = [
        path
        for path in (
            _resolve_creator_evidence_path(output_dir, evidence_root, raw_path)
            for raw_path in artifact_paths_payload.values()
        )
        if path is not None
    ] if isinstance(artifact_paths_payload, dict) else []
    required_paths = [manifest_path, passports_path]
    declared_artifact_paths = list(dict.fromkeys([*required_paths, *artifact_paths]))
    artifact_health = _artifact_path_health(
        artifact_paths=declared_artifact_paths,
        now=now,
    )
    if any(not path.exists() for path in required_paths):
        artifact_path_status = "missing"
    elif (
        not isinstance(artifact_paths_payload, dict)
        or not expected_artifact_keys <= {str(key).strip() for key in artifact_paths_payload}
        or any(not str(artifact_paths_payload.get(key, "")).strip() for key in expected_artifact_keys)
    ):
        artifact_path_status = "attention"
    else:
        artifact_path_status = str(artifact_health.get("path_status", "missing"))

    contract = manifest.get("contract", {})
    contract = contract if isinstance(contract, dict) else {}
    contract_version = str(contract.get("contract_version", "")).strip()
    verified_grade = str(contract.get("verified_grade", "")).strip()
    passport_contract_versions = {
        str(passport.get("contract_version", "")).strip()
        for passport in passports
    }
    observed_grades = {str(passport.get("evidence_grade", "")).strip() for passport in passports}
    if not contract_version or not manifest_path.exists():
        contract_status = "missing"
    elif (
        verified_grade == "publishable"
        and all(version == contract_version for version in passport_contract_versions)
        and all(grade in _CREATOR_EVIDENCE_GRADES for grade in observed_grades)
    ):
        contract_status = "ready"
    else:
        contract_status = "attention"

    rows: list[dict[str, object]] = []
    freshness_counts: dict[str, int] = {}
    effective_verified_count = 0
    for passport in passports:
        source_paths = [
            path
            for path in (
                _resolve_creator_evidence_path(output_dir, evidence_root, raw_path)
                for raw_path in passport.get("source_artifact_paths", [])
            )
            if path is not None
        ] if isinstance(passport.get("source_artifact_paths"), list) else []
        source_health = _artifact_path_health(
            artifact_paths=source_paths,
            reference_paths=[manifest_path, passports_path],
            now=now,
        )
        freshness_status = _creator_evidence_freshness_status(
            passport,
            source_health=source_health,
        )
        freshness_counts[freshness_status] = freshness_counts.get(freshness_status, 0) + 1
        grade = str(passport.get("evidence_grade", "")).strip()
        effective_verified = bool(
            passport.get("verified")
            and grade == "publishable"
            and contract_status == "ready"
            and artifact_path_status == "ready"
            and freshness_status == "ready"
            and str(source_health.get("path_status", "")) == "ready"
        )
        effective_verified_count += int(effective_verified)
        rows.append(
            {
                "passport_id": str(passport.get("passport_id", "")).strip(),
                "artist_name": str(passport.get("artist_name", "")).strip(),
                "market": str(passport.get("market", "")).strip(),
                "evidence_grade": grade,
                "raw_verified": bool(passport.get("verified", False)),
                "effective_verified": effective_verified,
                "evidence_status": (
                    "verified"
                    if effective_verified
                    else "suppressed"
                    if grade == "suppress"
                    else "stale"
                    if freshness_status == "stale"
                    else "watch"
                ),
                "contract_version": str(passport.get("contract_version", "")).strip(),
                "contract_status": contract_status,
                "freshness_status": freshness_status,
                "source_artifact_path_status": str(source_health.get("path_status", "")),
                "source_artifact_count": int(len(source_paths)),
                "existing_source_artifact_count": _safe_int(source_health.get("existing_count")),
                "missing_source_artifact_count": _safe_int(source_health.get("missing_count")),
                "stale_source_artifact_count": _safe_int(source_health.get("stale_count")),
                "missing_source_artifact_path": str(source_health.get("missing_path", "")),
                "stale_source_artifact_path": str(source_health.get("stale_path", "")),
                "occurrence_count": _safe_int(passport.get("occurrence_count")),
                "report_family_count": _safe_int(passport.get("report_family_count")),
                "latest_source_age_days": _safe_float(passport.get("latest_source_age_days")),
                "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
                "passport_json_path": str(passports_path.resolve()) if passports_path.exists() else "",
            }
        )

    registry = pd.DataFrame(rows)
    if not registry.empty:
        registry = registry.sort_values(
            ["effective_verified", "evidence_grade", "artist_name", "passport_id"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

    if freshness_counts.get("stale"):
        freshness_status = "stale"
    elif freshness_counts.get("attention") or freshness_counts.get("missing"):
        freshness_status = "attention"
    elif passports:
        freshness_status = "ready"
    else:
        freshness_status = "missing"
    observed_grade_counts = {
        grade: sum(str(passport.get("evidence_grade", "")).strip() == grade for passport in passports)
        for grade in sorted(_CREATOR_EVIDENCE_GRADES)
    }
    manifest_grade_counts = manifest.get("grade_counts", {})
    manifest_grade_counts = manifest_grade_counts if isinstance(manifest_grade_counts, dict) else {}
    grade_count_status = (
        "ready"
        if all(int(manifest_grade_counts.get(grade, 0) or 0) == count for grade, count in observed_grade_counts.items())
        and int(manifest.get("passport_count", len(passports)) or 0) == len(passports)
        else "attention"
        if manifest_path.exists()
        else "missing"
    )
    source_artifact_path_status = (
        "missing"
        if not registry.empty
        and (registry["source_artifact_path_status"].astype(str) == "missing").any()
        else "ready"
        if not registry.empty
        else "missing"
    )
    missing_source_artifact_count = (
        sum(
            max(
                _safe_int(row.get("missing_source_artifact_count")),
                int(str(row.get("source_artifact_path_status", "")) == "missing"),
            )
            for row in rows
        )
        if rows
        else 0
    )
    overall_status = (
        "ready"
        if (
            artifact_path_status
            == contract_status
            == freshness_status
            == grade_count_status
            == source_artifact_path_status
            == "ready"
        )
        else "stale"
        if freshness_status == "stale"
        else "missing"
        if artifact_path_status == "missing"
        else "attention"
    )
    summary = {
        "status": overall_status,
        "passport_count": int(len(passports)),
        "effective_verified_passport_count": int(effective_verified_count),
        "grade_counts": observed_grade_counts,
        "grade_count_status": grade_count_status,
        "contract_status": contract_status,
        "contract_version": contract_version,
        "freshness_status": freshness_status,
        "freshness_counts": freshness_counts,
        "artifact_path_status": artifact_path_status,
        "artifact_existing_count": _safe_int(artifact_health.get("existing_count")),
        "artifact_missing_count": _safe_int(artifact_health.get("missing_count")),
        "missing_artifact_path": str(artifact_health.get("missing_path", "")),
        "source_artifact_path_status": source_artifact_path_status,
        "missing_source_artifact_count": int(missing_source_artifact_count),
        "stale_source_artifact_count": int(
            registry.get("stale_source_artifact_count", pd.Series(dtype="int64")).fillna(0).sum()
        ) if not registry.empty else 0,
        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
        "passport_json_path": str(passports_path.resolve()) if passports_path.exists() else "",
    }
    return registry, summary


def _research_stage(*, protocol_present: bool, contract_present: bool, artifact_ratio: float, promoted: bool) -> str:
    if protocol_present and contract_present and promoted and artifact_ratio >= 0.99:
        return "show_ready"
    if protocol_present and contract_present and artifact_ratio >= 0.75:
        return "benchmarked"
    if protocol_present and contract_present:
        return "contracted"
    if protocol_present or contract_present:
        return "instrumented"
    return "pre_contract"


def _build_run_research_registry(
    *,
    output_dir: Path,
    control_history: pd.DataFrame,
    claim_run_id: str,
    claim_pack_path: Path,
    now: datetime,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for manifest in collect_run_manifests(output_dir):
        run_id = str(manifest.get("run_id", "")).strip()
        if not run_id:
            continue
        run_dir = Path(str(manifest.get("run_dir", "")).strip()).expanduser()
        if not run_dir.exists():
            continue
        analysis_dir = run_dir / "analysis"
        protocol_path = run_dir / "benchmark_protocol.json"
        contract_path = run_dir / "safety_platform_contract.json"
        protocol = safe_read_json(protocol_path, default={})
        protocol = protocol if isinstance(protocol, dict) else {}
        contract = safe_read_json(contract_path, default={})
        contract = contract if isinstance(contract, dict) else {}
        history_row = _find_matching_history_row(control_history, run_id)
        portability_notes = contract.get("portability_notes", [])
        portability_notes = portability_notes if isinstance(portability_notes, list) else []
        artifact_flags = {
            "run_results": (run_dir / "run_results.json").exists(),
            "benchmark_protocol": protocol_path.exists(),
            "safety_platform_contract": contract_path.exists(),
            "data_drift_summary": (analysis_dir / "data_drift_summary.json").exists(),
            "robustness_summary": (analysis_dir / "robustness_summary.json").exists(),
            "moonshot_summary": (analysis_dir / "moonshot_summary.json").exists(),
        }
        ratio = _artifact_ratio(artifact_flags)
        protocol_benchmark = protocol.get("benchmark_contract", {}) if isinstance(protocol.get("benchmark_contract"), dict) else {}
        protocol_temporal = protocol.get("protocol", {}) if isinstance(protocol.get("protocol"), dict) else {}
        temporal_backtest = protocol_temporal.get("temporal_backtest", {}) if isinstance(protocol_temporal.get("temporal_backtest"), dict) else {}
        reuse_summary = contract.get("reuse_summary", {}) if isinstance(contract.get("reuse_summary"), dict) else {}
        conformal_count = len(list(analysis_dir.glob("*_conformal_summary.json")))
        champion_gate = manifest.get("champion_gate", {}) if isinstance(manifest.get("champion_gate"), dict) else {}
        portability_status = (
            "ready"
            if protocol_path.exists()
            and contract_path.exists()
            and int(reuse_summary.get("api_group_count", 0) or 0) > 0
            and int(reuse_summary.get("wrapper_count", 0) or 0) > 0
            else "attention"
            if protocol_path.exists() or contract_path.exists()
            else "missing"
        )
        run_manifest_path = run_dir / "run_manifest.json"
        claim_pack_attached = run_id == claim_run_id
        claim_pack_health = (
            _artifact_path_health(
                artifact_paths=[
                    run_dir / "run_results.json",
                    protocol_path,
                    contract_path,
                    analysis_dir / "data_drift_summary.json",
                    analysis_dir / "robustness_summary.json",
                    analysis_dir / "moonshot_summary.json",
                ],
                reference_paths=[claim_pack_path],
                now=now,
            )
            if claim_pack_attached and claim_pack_path.exists()
            else {
                "path_status": "n/a" if claim_pack_attached else "n/a",
                "freshness_status": "missing" if claim_pack_attached and not claim_pack_path.exists() else "n/a",
                "existing_count": 0,
                "missing_count": 0,
                "stale_count": 0,
                "missing_path": "",
                "stale_path": "",
                "newest_path": "",
                "newest_path_timestamp": "",
                "newest_path_age_hours": float("nan"),
                "reference_path": str(claim_pack_path.resolve()) if claim_pack_attached else "",
                "reference_timestamp": _isoformat(_timestamp(claim_pack_path)) if claim_pack_attached else "",
            }
        )
        rows.append(
            {
                "run_id": run_id,
                "profile": str(manifest.get("profile", "")).strip(),
                "timestamp": str(manifest.get("timestamp", "")).strip(),
                "promoted": bool(champion_gate.get("promoted", False)),
                "champion_gate_status": str(champion_gate.get("status", "")).strip() or ("pass" if champion_gate.get("promoted") else "unknown"),
                "benchmark_protocol_present": protocol_path.exists(),
                "safety_platform_contract_present": contract_path.exists(),
                "conformal_summary_count": int(conformal_count),
                "backtest_model_count": int(len(temporal_backtest.get("models", []))) if isinstance(temporal_backtest.get("models", []), list) else 0,
                "benchmark_contract_version": str(protocol_benchmark.get("contract_version", "")).strip(),
                "benchmark_comparison_mode": str(protocol_benchmark.get("comparison_mode", "")).strip(),
                "safety_api_group_count": int(reuse_summary.get("api_group_count", 0) or 0),
                "spotify_wrapper_count": int(reuse_summary.get("wrapper_count", 0) or 0),
                "portability_note_count": int(len(portability_notes)),
                "portability_signal_status": portability_status,
                "research_artifact_ratio": ratio,
                "research_stage": _research_stage(
                    protocol_present=protocol_path.exists(),
                    contract_present=contract_path.exists(),
                    artifact_ratio=ratio,
                    promoted=bool(champion_gate.get("promoted", False)),
                ),
                "claim_pack_attached": claim_pack_attached,
                "claim_pack_path": str(claim_pack_path.resolve()) if claim_pack_attached and claim_pack_path.exists() else "",
                "claim_pack_freshness_status": str(claim_pack_health.get("freshness_status", "")),
                "claim_pack_stale_source_path": str(claim_pack_health.get("stale_path", "")),
                "claim_pack_stale_source_count": int(claim_pack_health.get("stale_count", 0) or 0),
                "run_manifest_path": str(run_manifest_path.resolve()),
                "run_manifest_timestamp": _isoformat(_timestamp(run_manifest_path)),
                "run_manifest_age_hours": _age_hours(_timestamp(run_manifest_path), now=now),
                "benchmark_protocol_path": str(protocol_path.resolve()) if protocol_path.exists() else "",
                "safety_platform_contract_path": str(contract_path.resolve()) if contract_path.exists() else "",
                "target_drift_jsd": _safe_float(history_row.get("target_drift_jsd")),
                "test_selective_risk": _safe_float(history_row.get("test_selective_risk")),
                "test_abstention_rate": _safe_float(history_row.get("test_abstention_rate")),
                "robustness_gap": _safe_float(history_row.get("robustness_gap")),
                "stress_skip_risk": _safe_float(history_row.get("stress_skip_risk")),
                "ops_coverage_ratio": _safe_float(history_row.get("ops_coverage_ratio")),
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry
    return registry.sort_values(["timestamp", "run_id"], ascending=[False, False]).reset_index(drop=True)


def _load_benchmark_lock_atlas(output_dir: Path, *, now: datetime) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    history_dir = output_dir / "history"
    for path in sorted(history_dir.glob("benchmark_lock_*_manifest.json")):
        payload = safe_read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        benchmark_id = str(payload.get("benchmark_id", "")).strip()
        if not benchmark_id:
            continue
        summary_df = safe_read_csv(history_dir / f"benchmark_lock_{benchmark_id}_summary.csv")
        significance_df = safe_read_csv(history_dir / f"benchmark_lock_{benchmark_id}_significance.csv")
        for column in ["val_top1_mean", "test_top1_mean", "run_count"]:
            summary_df[column] = pd.to_numeric(summary_df.get(column), errors="coerce")
        for column in ["mean_diff_val_top1", "z_score", "significant_at_95"]:
            significance_df[column] = pd.to_numeric(significance_df.get(column), errors="coerce")
        best_row = (
            summary_df.sort_values(["val_top1_mean", "test_top1_mean"], ascending=[False, False]).iloc[0].to_dict()
            if not summary_df.empty
            else {}
        )
        significant_pairs = significance_df.loc[significance_df["significant_at_95"].fillna(0.0) >= 1.0].copy()
        lead_pair = (
            significant_pairs.assign(abs_diff=significant_pairs["mean_diff_val_top1"].abs())
            .sort_values(["abs_diff", "z_score"], ascending=[False, False])
            .iloc[0]
            .to_dict()
            if not significant_pairs.empty
            else {}
        )
        present_artifacts = int(payload.get("present_artifact_count", 0) or 0)
        required_artifacts = int(payload.get("required_artifact_count", 0) or 0)
        required_ratio = float(present_artifacts / required_artifacts) if required_artifacts > 0 else float("nan")
        comparison_blockers = payload.get("comparison_blockers", [])
        comparison_blockers = comparison_blockers if isinstance(comparison_blockers, list) else []
        comparator_guard = payload.get("comparator_guard", {})
        comparator_guard = comparator_guard if isinstance(comparator_guard, dict) else {}
        model_class_mix = payload.get("model_class_mix", {})
        model_class_mix = model_class_mix if isinstance(model_class_mix, dict) else {}
        manifest_md_path = history_dir / f"benchmark_lock_{benchmark_id}_manifest.md"
        non_manifest_required_paths = [
            artifact_path
            for artifact_path in _coerce_paths(payload.get("required_artifacts", []))
            if artifact_path.name not in {
                f"benchmark_lock_{benchmark_id}_manifest.json",
                f"benchmark_lock_{benchmark_id}_manifest.md",
            }
        ]
        manifest_health = _artifact_path_health(
            artifact_paths=non_manifest_required_paths,
            reference_paths=[path, manifest_md_path],
            now=now,
        )
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "canonical_profile": str(payload.get("canonical_profile", "")).strip(),
                "comparison_mode": str(payload.get("comparison_mode", "")).strip(),
                "comparison_ready": bool(payload.get("comparison_ready", False)),
                "comparison_status": "ready" if bool(payload.get("comparison_ready", False)) else "incomplete",
                "run_count": int(payload.get("run_count", 0) or 0),
                "model_count": int(len(summary_df.index)),
                "present_artifact_count": present_artifacts,
                "required_artifact_count": required_artifacts,
                "required_artifact_ratio": required_ratio,
                "significant_pair_count": int(payload.get("significant_pair_count", 0) or 0),
                "comparison_blocker_count": int(len(comparison_blockers)),
                "top_comparison_blocker": str(comparison_blockers[0]).strip() if comparison_blockers else "",
                "comparison_blockers_json": json.dumps(comparison_blockers),
                "comparator_guard_status": str(comparator_guard.get("status", "")).strip(),
                "deep_comparator_ready": bool(comparator_guard.get("deep_comparator_ready", False)),
                "observed_model_classes_json": json.dumps(
                    (
                        model_class_mix.get("observed", {})
                        if isinstance(model_class_mix.get("observed", {}), dict)
                        else {}
                    ).get("model_classes", []),
                    sort_keys=True,
                ),
                "best_model_name": str(best_row.get("model_name", "")).strip(),
                "best_model_type": str(best_row.get("model_type", "")).strip(),
                "best_val_top1_mean": _safe_float(best_row.get("val_top1_mean")),
                "best_test_top1_mean": _safe_float(best_row.get("test_top1_mean")),
                "top_significant_pair": (
                    f"{lead_pair.get('left_model', '')} vs {lead_pair.get('right_model', '')}"
                    if lead_pair
                    else ""
                ),
                "top_significant_margin": _safe_float(lead_pair.get("mean_diff_val_top1")),
                "manifest_freshness_status": str(manifest_health.get("freshness_status", "")),
                "manifest_stale_source_path": str(manifest_health.get("stale_path", "")),
                "manifest_stale_source_count": int(manifest_health.get("stale_count", 0) or 0),
                "manifest_age_hours": _age_hours(_timestamp(path), now=now),
                "summary_path": str((history_dir / f"benchmark_lock_{benchmark_id}_summary.csv").resolve()),
                "significance_path": str((history_dir / f"benchmark_lock_{benchmark_id}_significance.csv").resolve()),
                "benchmark_strength_score": 0.0,
                "manifest_path": str(path.resolve()),
            }
        )
    atlas = pd.DataFrame(rows)
    if atlas.empty:
        return atlas
    atlas["benchmark_strength_score"] = (
        0.35 * _normalize_series(atlas["run_count"], higher_is_better=True).fillna(0.0)
        + 0.25 * atlas["comparison_ready"].astype(int)
        + 0.20 * _normalize_series(atlas["required_artifact_ratio"], higher_is_better=True).fillna(0.0)
        + 0.20 * _normalize_series(atlas["significant_pair_count"], higher_is_better=True).fillna(0.0)
    )
    return atlas.sort_values(
        ["benchmark_strength_score", "run_count", "benchmark_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _build_claim_registry(output_dir: Path, *, now: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    claims_path = output_dir / "analysis" / "research_claims" / "research_claims.json"
    payload = safe_read_json(claims_path, default={})
    payload = payload if isinstance(payload, dict) else {}
    claims = payload.get("claims", [])
    claims = claims if isinstance(claims, list) else []
    support_rows = payload.get("claim_support_matrix", [])
    support_rows = support_rows if isinstance(support_rows, list) else []
    support_lookup = {
        str(row.get("claim_key", "")).strip(): row for row in support_rows if isinstance(row, dict)
    }
    primary_key = ""
    primary_claim = payload.get("primary_claim", {})
    if isinstance(primary_claim, dict):
        primary_key = str(primary_claim.get("key", "")).strip()

    rows: list[dict[str, object]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        key = str(claim.get("key", "")).strip()
        support = support_lookup.get(key, {})
        metrics = claim.get("metrics", {}) if isinstance(claim.get("metrics"), dict) else {}
        missing_checks = claim.get("missing_checks", [])
        missing_checks = missing_checks if isinstance(missing_checks, list) else []
        supporting_artifacts = claim.get("supporting_artifacts", [])
        supporting_artifacts = supporting_artifacts if isinstance(supporting_artifacts, list) else []
        artifact_health = _artifact_path_health(
            artifact_paths=_coerce_paths(supporting_artifacts),
            reference_paths=[claims_path],
            now=now,
        )
        claim_readiness_status = (
            "blocked"
            if missing_checks
            or str(artifact_health.get("path_status", "")) not in _REVIEW_READY_STATUSES
            or str(artifact_health.get("freshness_status", "")) == "stale"
            else "ready"
            if str(claim.get("status", "")).strip() in _READY_STATUSES
            else "attention"
            if str(claim.get("status", "")).strip()
            else "missing"
        )
        rows.append(
            {
                "claim_key": key,
                "title": str(claim.get("title", "")).strip(),
                "role": "primary" if key == primary_key else "backup",
                "status": str(claim.get("status", "")).strip(),
                "claim_readiness_status": claim_readiness_status,
                "summary": str(claim.get("summary", "")).strip(),
                "live_signal_status": str(support.get("live_signal_status", "")).strip(),
                "benchmark_evidence_status": str(support.get("benchmark_evidence_status", "")).strip(),
                "repeated_evidence_status": str(support.get("repeated_evidence_status", "")).strip(),
                "slice_evidence_status": str(support.get("slice_evidence_status", "")).strip(),
                "risk_evidence_status": str(support.get("risk_evidence_status", "")).strip(),
                "artifact_pack_status": str(support.get("artifact_pack_status", "")).strip(),
                "supporting_artifact_count": int(len(supporting_artifacts)),
                "existing_supporting_artifact_count": int(artifact_health.get("existing_count", 0) or 0),
                "missing_supporting_artifact_count": int(artifact_health.get("missing_count", 0) or 0),
                "stale_supporting_artifact_count": int(artifact_health.get("stale_count", 0) or 0),
                "supporting_artifact_path_status": str(artifact_health.get("path_status", "")),
                "supporting_artifact_freshness_status": str(artifact_health.get("freshness_status", "")),
                "missing_supporting_artifact_path": str(artifact_health.get("missing_path", "")),
                "stale_supporting_artifact_path": str(artifact_health.get("stale_path", "")),
                "missing_check_count": int(len(missing_checks)),
                "blocked": bool(claim_readiness_status == "blocked"),
                "next_gate": str(support.get("next_gate", "")).strip(),
                "target_drift_jsd": _safe_float(metrics.get("target_drift_jsd")),
                "selective_risk": _safe_float(metrics.get("selective_risk")),
                "stress_skip_risk": _safe_float(metrics.get("stress_skip_risk")),
                "live_test_top1_lift_vs_deep": _safe_float(metrics.get("live_test_top1_lift_vs_deep")),
                "benchmark_comparison_ready": bool(metrics.get("benchmark_comparison_ready", False)),
                "benchmark_significant_lift": bool(metrics.get("benchmark_significant_lift", False)),
                "claims_path": str(claims_path.resolve()) if claims_path.exists() else "",
                "metrics_json": json.dumps(metrics, sort_keys=True),
                "missing_checks_json": json.dumps(missing_checks),
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry, payload
    return registry.sort_values(["role", "status", "claim_key"], ascending=[True, False, True]).reset_index(drop=True), payload


def _build_maturity_brief(
    *,
    anchor_run_dir: Path,
    run_registry: pd.DataFrame,
    benchmark_atlas: pd.DataFrame,
    claim_registry: pd.DataFrame,
    research_payload: dict[str, Any],
    creator_evidence_summary: dict[str, object],
) -> tuple[dict[str, Any], list[str]]:
    anchor_run_id = anchor_run_dir.name
    anchor_row = (
        run_registry.loc[run_registry["run_id"].astype(str) == anchor_run_id].iloc[0].to_dict()
        if not run_registry.empty and (run_registry["run_id"].astype(str) == anchor_run_id).any()
        else {}
    )
    benchmark_row = benchmark_atlas.iloc[0].to_dict() if not benchmark_atlas.empty else {}
    ready_claim_count = int(
        claim_registry["status"].isin(_READY_STATUSES).sum()
    ) if not claim_registry.empty and "status" in claim_registry.columns else 0
    blocked_claim_count = int(
        (claim_registry["claim_readiness_status"].astype(str) == "blocked").sum()
    ) if not claim_registry.empty and "claim_readiness_status" in claim_registry.columns else 0
    total_claim_count = int(len(claim_registry.index))
    submission_readiness = research_payload.get("submission_readiness", {})
    submission_readiness = submission_readiness if isinstance(submission_readiness, dict) else {}
    blockers = submission_readiness.get("blockers", [])
    blockers = blockers if isinstance(blockers, list) else []
    incomplete_benchmark_count = int(
        (benchmark_atlas["comparison_status"].astype(str) != "ready").sum()
    ) if not benchmark_atlas.empty and "comparison_status" in benchmark_atlas.columns else 0
    stale_benchmark_count = int(
        (benchmark_atlas["manifest_freshness_status"].astype(str) == "stale").sum()
    ) if not benchmark_atlas.empty and "manifest_freshness_status" in benchmark_atlas.columns else 0
    stale_claim_artifact_count = int(
        claim_registry.get("stale_supporting_artifact_count", pd.Series(dtype="int64")).fillna(0).astype(int).sum()
    ) if not claim_registry.empty else 0
    top_next_gate = ""
    if not claim_registry.empty and "next_gate" in claim_registry.columns:
        next_gates = [
            str(value).strip()
            for value in claim_registry["next_gate"]
            if str(value).strip() and str(value).strip() != "ready_to_package"
        ]
        if next_gates:
            top_next_gate = next_gates[0]

    summary = [
        f"Research platform anchor is `{anchor_run_id}`.",
        f"Run registry covers `{len(run_registry.index)}` completed runs, with `{int((run_registry['benchmark_protocol_present'] & run_registry['safety_platform_contract_present']).sum()) if not run_registry.empty else 0}` runs carrying both protocol and contract artifacts.",
    ]
    creator_grade_counts = creator_evidence_summary.get("grade_counts", {})
    creator_grade_counts = creator_grade_counts if isinstance(creator_grade_counts, dict) else {}
    summary.append(
        "Creator evidence registry is "
        f"`{creator_evidence_summary.get('status', 'missing')}` with "
        f"`{int(creator_evidence_summary.get('effective_verified_passport_count', 0) or 0)}` of "
        f"`{int(creator_evidence_summary.get('passport_count', 0) or 0)}` passports effectively verified; "
        f"grades are publishable `{int(creator_grade_counts.get('publishable', 0) or 0)}`, "
        f"watch-only `{int(creator_grade_counts.get('watch_only', 0) or 0)}`, and "
        f"suppress `{int(creator_grade_counts.get('suppress', 0) or 0)}`. "
        f"Contract `{creator_evidence_summary.get('contract_status', 'missing')}`, "
        f"freshness `{creator_evidence_summary.get('freshness_status', 'missing')}`, and "
        f"artifact paths `{creator_evidence_summary.get('artifact_path_status', 'missing')}` with "
        f"source paths `{creator_evidence_summary.get('source_artifact_path_status', 'missing')}`."
    )
    if str(creator_evidence_summary.get("status", "")) != "ready":
        summary.append(
            "Creator opportunities without an effectively verified passport remain directional/watch signals, not confident immediate priorities."
        )
    if anchor_row:
        summary.append(
            f"Anchor run stage is `{anchor_row.get('research_stage', '')}` with artifact ratio `{_safe_float(anchor_row.get('research_artifact_ratio')):.3f}` and promoted `{bool(anchor_row.get('promoted', False))}`."
        )
        summary.append(
            f"Anchor portability signals are `{anchor_row.get('portability_signal_status', 'missing')}` with `{int(anchor_row.get('safety_api_group_count', 0) or 0)}` reusable API groups and `{int(anchor_row.get('spotify_wrapper_count', 0) or 0)}` Spotify wrappers."
        )
        if str(anchor_row.get("claim_pack_freshness_status", "")) == "stale":
            summary.append(
                f"Attached claim pack looks stale because `{anchor_row.get('claim_pack_stale_source_path', '')}` is newer than the current claims snapshot."
            )
    if benchmark_row:
        if bool(benchmark_row.get("comparison_ready", False)):
            summary.append(
                f"Strongest benchmark lock is `{benchmark_row.get('benchmark_id', '')}` with comparison ready `True` and leading model `{benchmark_row.get('best_model_name', '')}`."
            )
        else:
            summary.append(
                f"Strongest benchmark lock is `{benchmark_row.get('benchmark_id', '')}` but it is incomplete; top blocker is `{benchmark_row.get('top_comparison_blocker', '')}`."
            )
        if str(benchmark_row.get("manifest_freshness_status", "")) == "stale":
            summary.append(
                f"Benchmark manifest freshness looks stale because `{benchmark_row.get('manifest_stale_source_path', '')}` is newer than the saved manifest."
            )
    if total_claim_count:
        summary.append(
            f"Claim registry tracks `{ready_claim_count}/{total_claim_count}` claims at `analysis_ready` or better, with `{blocked_claim_count}` claims currently blocked by missing checks or stale artifact evidence."
        )
    if blockers:
        summary.append(
            f"Submission readiness is `{str(submission_readiness.get('status', '')).strip()}` and blocked by `{str(blockers[0]).strip()}`."
        )
    elif top_next_gate:
        summary.append(f"Highest-leverage next gate is `{top_next_gate}`.")
    else:
        summary.append("Current claim pack is structurally complete enough to keep iterating locally without reopening the benchmark contract.")
    if incomplete_benchmark_count or stale_benchmark_count or stale_claim_artifact_count:
        summary.append(
            f"Truthfulness audit found `{incomplete_benchmark_count}` incomplete benchmark lock(s), `{stale_benchmark_count}` stale benchmark manifest(s), and `{stale_claim_artifact_count}` stale supporting artifact reference(s)."
        )

    actions = []
    if blockers:
        actions.append(f"Clear the lead submission blocker: `{str(blockers[0]).strip()}`.")
    if benchmark_row and not bool(benchmark_row.get("comparison_ready", False)):
        actions.append(
            f"Finish benchmark lock `{benchmark_row.get('benchmark_id', '')}` before treating candidate-ranking evidence as portable research proof."
        )
    if anchor_row and str(anchor_row.get("portability_signal_status", "")) != "ready":
        actions.append("Refresh `benchmark_protocol.json` and `safety_platform_contract.json` so portability signals are attached to the anchor run.")
    if anchor_row and str(anchor_row.get("claim_pack_freshness_status", "")) == "stale":
        actions.append("Regenerate `research_claims.json` after the newer anchor artifacts so the readiness story stops pointing at stale evidence.")
    if str(creator_evidence_summary.get("status", "")) != "ready":
        actions.append(
            "Refresh the creator evidence manifest and passports, repair missing source paths, and clear freshness or contract gaps before publishing creator priorities."
        )
    if not actions:
        actions = [
            "Use the run registry to choose which completed runs are strong enough to reopen as research anchors.",
            "Use the benchmark atlas when deciding whether a new repeated-run benchmark is actually stronger than the current lock.",
            "Use the claim registry as the single place to see what is submission-ready versus merely promising.",
        ]
    payload = {
        "anchor_run_id": anchor_run_id,
        "anchor_run": anchor_row,
        "strongest_benchmark_lock": benchmark_row,
        "claim_ready_count": ready_claim_count,
        "claim_blocked_count": blocked_claim_count,
        "claim_total_count": total_claim_count,
        "incomplete_benchmark_lock_count": incomplete_benchmark_count,
        "stale_benchmark_manifest_count": stale_benchmark_count,
        "stale_claim_artifact_count": stale_claim_artifact_count,
        "submission_status": str(submission_readiness.get("status", "")).strip(),
        "ready_for_external_review": bool(submission_readiness.get("ready_for_external_review", False)),
        "blockers": blockers,
        "top_next_gate": top_next_gate,
        "creator_evidence": creator_evidence_summary,
        "summary": summary,
        "actions": actions,
    }
    markdown_lines = [
        "# Research Platform Maturity",
        "",
        *[f"- {line}" for line in summary],
        "",
        "## Suggested Uses",
        "",
        *[f"- {line}" for line in actions],
    ]
    return payload, markdown_lines


def _select_anchor_registry_row(run_registry: pd.DataFrame) -> dict[str, object]:
    if run_registry.empty:
        return {}
    if "claim_pack_attached" in run_registry.columns:
        attached = run_registry.loc[run_registry["claim_pack_attached"].map(_truthy)]
        if not attached.empty:
            return attached.iloc[0].to_dict()
    return run_registry.iloc[0].to_dict()


def _select_benchmark_planner_row(benchmark_atlas: pd.DataFrame) -> dict[str, object]:
    if benchmark_atlas.empty:
        return {}
    rows = [row.to_dict() for _, row in benchmark_atlas.iterrows()]
    incomplete = [
        row
        for row in rows
        if str(row.get("comparison_status", "")).strip().lower() != "ready"
        or not _truthy(row.get("comparison_ready", False))
    ]
    deep_incomplete = [
        row
        for row in incomplete
        if not _truthy(row.get("deep_comparator_ready", False))
        or "deep comparator" in str(row.get("top_comparison_blocker", "")).lower()
    ]
    return (deep_incomplete or incomplete or rows)[0]


def _claim_planner_text(row: dict[str, object], missing_checks: list[object]) -> str:
    parts = [
        row.get("claim_key", ""),
        row.get("title", ""),
        row.get("summary", ""),
        row.get("next_gate", ""),
        *missing_checks,
    ]
    return " ".join(str(item).strip().lower() for item in parts if str(item).strip())


def _claim_gap_fields(row: dict[str, object]) -> list[str]:
    fields = [
        "live_signal_status",
        "benchmark_evidence_status",
        "repeated_evidence_status",
        "slice_evidence_status",
        "risk_evidence_status",
        "artifact_pack_status",
        "supporting_artifact_path_status",
        "supporting_artifact_freshness_status",
    ]
    return [field for field in fields if _status_is_gap(row.get(field, ""))]


def _experiment_priority(score: float) -> str:
    if score >= 115.0:
        return "critical"
    if score >= 95.0:
        return "high"
    if score >= 75.0:
        return "medium"
    return "watch"


def _experiment_score(base: float, row: dict[str, object], gap_fields: list[str]) -> float:
    role_bonus = 12.0 if str(row.get("role", "")).strip().lower() == "primary" else 0.0
    blocked_bonus = 10.0 if str(row.get("claim_readiness_status", "")).strip().lower() == "blocked" else 0.0
    missing_bonus = min(8.0, 2.0 * _safe_int(row.get("missing_check_count")))
    gap_bonus = min(8.0, 1.5 * len(gap_fields))
    artifact_bonus = 3.0 if _safe_int(row.get("stale_supporting_artifact_count")) else 0.0
    return round(base + role_bonus + blocked_bonus + missing_bonus + gap_bonus + artifact_bonus, 2)


def _experiment_inputs(
    *,
    row: dict[str, object],
    benchmark_row: dict[str, object],
    anchor_row: dict[str, object],
    missing_checks: list[object],
) -> dict[str, object]:
    return _json_ready(
        {
            "research_claim_registry": {
                "claim_readiness_status": str(row.get("claim_readiness_status", "")).strip(),
                "benchmark_evidence_status": str(row.get("benchmark_evidence_status", "")).strip(),
                "repeated_evidence_status": str(row.get("repeated_evidence_status", "")).strip(),
                "risk_evidence_status": str(row.get("risk_evidence_status", "")).strip(),
                "artifact_pack_status": str(row.get("artifact_pack_status", "")).strip(),
                "next_gate": str(row.get("next_gate", "")).strip(),
                "missing_checks": [str(item).strip() for item in missing_checks if str(item).strip()][:5],
            },
            "benchmark_lock_atlas": {
                "benchmark_id": str(benchmark_row.get("benchmark_id", "")).strip(),
                "comparison_status": str(benchmark_row.get("comparison_status", "")).strip(),
                "deep_comparator_ready": _truthy(benchmark_row.get("deep_comparator_ready", False)),
                "run_count": _safe_int(benchmark_row.get("run_count")),
                "top_comparison_blocker": str(benchmark_row.get("top_comparison_blocker", "")).strip(),
                "manifest_freshness_status": str(benchmark_row.get("manifest_freshness_status", "")).strip(),
            }
            if benchmark_row
            else {},
            "run_research_registry": {
                "run_id": str(anchor_row.get("run_id", "")).strip(),
                "research_stage": str(anchor_row.get("research_stage", "")).strip(),
                "portability_signal_status": str(anchor_row.get("portability_signal_status", "")).strip(),
                "test_selective_risk": _finite_or_none(anchor_row.get("test_selective_risk")),
                "test_abstention_rate": _finite_or_none(anchor_row.get("test_abstention_rate")),
                "ops_coverage_ratio": _finite_or_none(anchor_row.get("ops_coverage_ratio")),
            }
            if anchor_row
            else {},
        }
    )


def _experiment_record(
    *,
    row: dict[str, object],
    experiment_type: str,
    title: str,
    base_score: float,
    gap_fields: list[str],
    missing_checks: list[object],
    benchmark_row: dict[str, object],
    anchor_row: dict[str, object],
    why: str,
    recommended_experiment: str,
    success_criteria: list[str],
    expected_artifacts: list[object],
    triggered_by: list[str],
) -> dict[str, object]:
    score = _experiment_score(base_score, row, gap_fields)
    claim_key = str(row.get("claim_key", "")).strip() or "unknown_claim"
    artifacts = [str(item).strip() for item in expected_artifacts if str(item).strip()]
    return _json_ready(
        {
            "rank": 0,
            "experiment_id": _slug(f"{claim_key}_{experiment_type}"),
            "experiment_type": experiment_type,
            "priority": _experiment_priority(score),
            "rank_score": score,
            "claim_key": claim_key,
            "claim_title": str(row.get("title", "")).strip(),
            "claim_role": str(row.get("role", "")).strip(),
            "current_status": str(row.get("status", "")).strip(),
            "current_readiness": str(row.get("claim_readiness_status", "")).strip(),
            "target_status": "analysis_ready",
            "next_gate": str(row.get("next_gate", "")).strip(),
            "gaps_addressed": gap_fields,
            "triggered_by": triggered_by,
            "title": title,
            "why": why,
            "recommended_experiment": recommended_experiment,
            "success_criteria": success_criteria,
            "expected_artifacts": artifacts,
            "inputs_consumed": _experiment_inputs(
                row=row,
                benchmark_row=benchmark_row,
                anchor_row=anchor_row,
                missing_checks=missing_checks,
            ),
        }
    )


def _build_next_experiment_plan(
    *,
    run_registry: pd.DataFrame,
    benchmark_atlas: pd.DataFrame,
    claim_registry: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    anchor_row = _select_anchor_registry_row(run_registry)
    benchmark_row = _select_benchmark_planner_row(benchmark_atlas)
    claim_rows = [row.to_dict() for _, row in claim_registry.iterrows()] if not claim_registry.empty else []
    blocked_claims = [
        row
        for row in claim_rows
        if _truthy(row.get("blocked", False))
        or str(row.get("claim_readiness_status", "")).strip().lower() == "blocked"
    ]

    experiments: list[dict[str, object]] = []
    for row in blocked_claims:
        missing_checks = _parse_json_list(row.get("missing_checks_json", "[]"))
        metrics = _parse_json_dict(row.get("metrics_json", "{}"))
        claim_key = str(row.get("claim_key", "")).strip()
        claim_text = _claim_planner_text(row, missing_checks)
        gap_fields = _claim_gap_fields(row)
        triggered_base = [
            f"{field}={str(row.get(field, '')).strip()}"
            for field in gap_fields
            if str(row.get(field, "")).strip()
        ]
        if _safe_int(row.get("missing_check_count")):
            triggered_base.append(f"missing_check_count={_safe_int(row.get('missing_check_count'))}")

        benchmark_status = str(benchmark_row.get("comparison_status", "")).strip().lower()
        benchmark_needs_deep = bool(
            benchmark_row
            and (
                benchmark_status != "ready"
                or not _truthy(benchmark_row.get("deep_comparator_ready", False))
                or "deep comparator" in str(benchmark_row.get("top_comparison_blocker", "")).lower()
            )
        )
        benchmark_claim_gap = (
            claim_key == "candidate_ranking"
            or _status_is_gap(row.get("benchmark_evidence_status", ""))
            or _status_is_gap(row.get("repeated_evidence_status", ""))
            or ("benchmark_comparison_ready" in metrics and not _truthy(metrics.get("benchmark_comparison_ready", False)))
        )
        if benchmark_claim_gap and (
            benchmark_needs_deep
            or "deep comparator" in claim_text
            or "benchmark lock" in claim_text
            or "repeated-seed" in claim_text
            or "repeated seed" in claim_text
        ):
            benchmark_id = str(benchmark_row.get("benchmark_id", "")).strip() or "current_lock"
            target_runs = max(3, _safe_int(benchmark_row.get("run_count")))
            experiments.append(
                _experiment_record(
                    row=row,
                    experiment_type="deep_comparator_benchmark_coverage",
                    title="Add repeated deep-comparator benchmark coverage",
                    base_score=100.0,
                    gap_fields=gap_fields,
                    missing_checks=missing_checks,
                    benchmark_row=benchmark_row,
                    anchor_row=anchor_row,
                    why=(
                        f"`{claim_key}` is blocked by benchmark/comparator evidence while lock `{benchmark_id}` "
                        f"is `{str(benchmark_row.get('comparison_status', 'missing')).strip() or 'missing'}` "
                        f"with deep_comparator_ready `{_truthy(benchmark_row.get('deep_comparator_ready', False))}`."
                    ),
                    recommended_experiment=(
                        f"Rerun benchmark lock `{benchmark_id}` with a repeated direct-softmax/deep comparator and the candidate "
                        f"surface for at least `{target_runs}` manifest-backed runs, then regenerate the lock summary, "
                        "significance table, manifest, and claim pack."
                    ),
                    success_criteria=[
                        "Benchmark atlas reports `comparison_status=ready` for the selected lock.",
                        f"Summary includes a deep comparator and candidate surface with at least `{target_runs}` manifest-backed runs each.",
                        f"`{claim_key}` has `benchmark_evidence_status=ready`, `repeated_evidence_status=ready`, and no deep-comparator missing check.",
                    ],
                    expected_artifacts=[
                        benchmark_row.get("summary_path", ""),
                        benchmark_row.get("significance_path", ""),
                        benchmark_row.get("manifest_path", ""),
                        "outputs/analysis/research_claims/research_claims.json",
                    ],
                    triggered_by=[
                        *triggered_base,
                        str(benchmark_row.get("top_comparison_blocker", "")).strip(),
                    ],
                )
            )

        risk_terms = ("risk", "coverage", "abstention", "conformal", "selective")
        risk_claim_gap = claim_key == "risk_aware_abstention" or (
            _status_is_gap(row.get("risk_evidence_status", "")) and any(term in claim_text for term in risk_terms)
        )
        if risk_claim_gap or any(term in claim_text for term in ("accuracy-coverage", "coverage loss", "selective-risk")):
            anchor_run_id = str(anchor_row.get("run_id", "")).strip() or "anchor run"
            selective_risk = metrics.get("selective_risk", row.get("selective_risk"))
            abstention_rate = metrics.get("abstention_rate", anchor_row.get("test_abstention_rate"))
            coverage = metrics.get("conformal_coverage", anchor_row.get("ops_coverage_ratio"))
            experiments.append(
                _experiment_record(
                    row=row,
                    experiment_type="risk_coverage_tradeoff_evidence",
                    title="Measure risk/coverage tradeoff before promoting the safety claim",
                    base_score=92.0,
                    gap_fields=gap_fields,
                    missing_checks=missing_checks,
                    benchmark_row=benchmark_row,
                    anchor_row=anchor_row,
                    why=(
                        f"`{claim_key}` needs calibrated risk evidence; current selective risk is `{_format_metric(selective_risk)}`, "
                        f"abstention is `{_format_metric(abstention_rate)}`, and coverage is `{_format_metric(coverage)}`."
                    ),
                    recommended_experiment=(
                        f"On `{anchor_run_id}`, sweep abstention thresholds or conformal operating points and export an "
                        "accuracy/coverage/selective-risk table that shows whether reduced coverage buys a meaningful risk drop."
                    ),
                    success_criteria=[
                        f"`{claim_key}` has `risk_evidence_status=ready` with selective risk, abstention, and coverage all populated.",
                        "The selected operating point documents the accepted-rate or coverage cost next to the risk reduction.",
                        "The refreshed claim pack removes the risk/coverage missing check instead of relying on full-coverage metrics alone.",
                    ],
                    expected_artifacts=[
                        "outputs/analysis/risk_coverage_tradeoff.csv",
                        "outputs/analysis/research_claims/research_claims.json",
                        row.get("claims_path", ""),
                    ],
                    triggered_by=triggered_base + [str(item).strip() for item in missing_checks[:2] if str(item).strip()],
                )
            )

        friction_claim_gap = claim_key == "friction_counterfactual" or any(
            term in claim_text
            for term in (
                "friction",
                "counterfactual",
                "intervention",
                "synthetic perturbation",
                "label path",
                "degenerate",
            )
        )
        if friction_claim_gap:
            test_delta = metrics.get("test_mean_delta")
            auc_lift = metrics.get("test_auc_lift")
            experiments.append(
                _experiment_record(
                    row=row,
                    experiment_type="friction_counterfactual_trustworthiness",
                    title="Audit friction-counterfactual trustworthiness",
                    base_score=88.0,
                    gap_fields=gap_fields,
                    missing_checks=missing_checks,
                    benchmark_row=benchmark_row,
                    anchor_row=anchor_row,
                    why=(
                        f"`{claim_key}` still needs a non-degenerate trust check; current test delta is "
                        f"`{_format_metric(test_delta)}` and test AUC lift is `{_format_metric(auc_lift)}`."
                    ),
                    recommended_experiment=(
                        "Audit the friction label path, then run a non-degenerate intervention or synthetic perturbation "
                        "check that separates preference-driven skip risk from friction-driven skip risk across stable slices."
                    ),
                    success_criteria=[
                        "Counterfactual delta is non-zero in the expected direction and stable across at least one repeated slice or seed check.",
                        "Baseline/full AUC and label-path diagnostics rule out saturated or degenerate friction labels.",
                        f"`{claim_key}` has no friction-label or intervention missing check after the claim pack refresh.",
                    ],
                    expected_artifacts=[
                        "outputs/runs/<run_id>/analysis/friction_proxy_summary.json",
                        "outputs/runs/<run_id>/analysis/friction_counterfactual_delta.csv",
                        "outputs/analysis/research_claims/research_claims.json",
                    ],
                    triggered_by=triggered_base + [str(item).strip() for item in missing_checks[:2] if str(item).strip()],
                )
            )

        missing_artifacts = _safe_int(row.get("missing_supporting_artifact_count"))
        stale_artifacts = _safe_int(row.get("stale_supporting_artifact_count"))
        if missing_artifacts or stale_artifacts:
            experiments.append(
                _experiment_record(
                    row=row,
                    experiment_type="supporting_artifact_refresh_repair",
                    title="Repair stale or missing claim evidence paths",
                    base_score=68.0,
                    gap_fields=gap_fields,
                    missing_checks=missing_checks,
                    benchmark_row=benchmark_row,
                    anchor_row=anchor_row,
                    why=(
                        f"`{claim_key}` references `{missing_artifacts}` missing and `{stale_artifacts}` stale supporting artifact(s), "
                        "so the claim cannot be analysis-ready until the evidence pack is refreshed."
                    ),
                    recommended_experiment=(
                        "Regenerate the claim pack after the newest supporting artifacts and repair any non-portable or missing paths "
                        "before re-running the research platform lab."
                    ),
                    success_criteria=[
                        f"`{claim_key}` has zero missing and zero stale supporting artifacts in `research_claim_registry`.",
                        "Supporting artifact freshness is `ready` and the refreshed claim pack points at the newest benchmark/run files.",
                    ],
                    expected_artifacts=[
                        str(row.get("missing_supporting_artifact_path", "")).strip(),
                        str(row.get("stale_supporting_artifact_path", "")).strip(),
                        "outputs/analysis/research_claims/research_claims.json",
                    ],
                    triggered_by=[
                        *triggered_base,
                        f"missing_supporting_artifact_count={missing_artifacts}",
                        f"stale_supporting_artifact_count={stale_artifacts}",
                    ],
                )
            )

        if not any(str(item.get("claim_key", "")) == claim_key for item in experiments):
            experiments.append(
                _experiment_record(
                    row=row,
                    experiment_type="claim_evidence_gap_closure",
                    title="Close the lead claim evidence gate",
                    base_score=58.0,
                    gap_fields=gap_fields,
                    missing_checks=missing_checks,
                    benchmark_row=benchmark_row,
                    anchor_row=anchor_row,
                    why=f"`{claim_key}` is blocked but did not match a specialized planner template.",
                    recommended_experiment=(
                        f"Run the next gate for `{claim_key}`: `{str(row.get('next_gate', '')).strip() or 'refresh the missing evidence'}`."
                    ),
                    success_criteria=[
                        f"`{claim_key}` no longer has missing checks.",
                        f"`{claim_key}` moves from `{str(row.get('claim_readiness_status', '')).strip()}` to `ready` in the claim registry.",
                    ],
                    expected_artifacts=["outputs/analysis/research_claims/research_claims.json"],
                    triggered_by=triggered_base + [str(item).strip() for item in missing_checks[:2] if str(item).strip()],
                )
            )

    experiments = sorted(
        experiments,
        key=lambda item: (
            -_safe_float(item.get("rank_score")),
            str(item.get("claim_role", "")) != "primary",
            str(item.get("claim_key", "")),
            str(item.get("experiment_type", "")),
        ),
    )
    for index, experiment in enumerate(experiments, start=1):
        experiment["rank"] = index

    type_counts: dict[str, int] = {}
    for experiment in experiments:
        experiment_type = str(experiment.get("experiment_type", "")).strip()
        type_counts[experiment_type] = type_counts.get(experiment_type, 0) + 1

    payload: dict[str, Any] = {
        "planner_version": "v1",
        "source_tables": {
            "run_research_registry_rows": int(len(run_registry.index)),
            "benchmark_lock_atlas_rows": int(len(benchmark_atlas.index)),
            "research_claim_registry_rows": int(len(claim_registry.index)),
        },
        "blocked_claim_count": int(len(blocked_claims)),
        "blocked_claim_keys": [str(row.get("claim_key", "")).strip() for row in blocked_claims],
        "experiment_count": int(len(experiments)),
        "experiment_type_counts": type_counts,
        "top_experiment": experiments[0] if experiments else {},
        "experiments": experiments,
    }

    markdown_lines = [
        "# Research Next Experiments",
        "",
        (
            f"Planner consumed `{len(run_registry.index)}` run registry row(s), `{len(benchmark_atlas.index)}` benchmark lock row(s), "
            f"and `{len(claim_registry.index)}` claim registry row(s)."
        ),
        f"Blocked claims: `{len(blocked_claims)}`. Ranked experiments: `{len(experiments)}`.",
        "",
    ]
    if not experiments:
        markdown_lines.extend(
            [
                "No blocked claims currently require a planner-generated experiment.",
                "Keep using the run registry, benchmark atlas, and claim registry to choose the next research anchor.",
            ]
        )
    else:
        markdown_lines.extend(["## Ranked Experiments", ""])
        for experiment in experiments:
            criteria = experiment.get("success_criteria", [])
            criteria = criteria if isinstance(criteria, list) else []
            markdown_lines.extend(
                [
                    f"{int(experiment.get('rank', 0))}. `{experiment.get('experiment_id', '')}` - {experiment.get('title', '')}",
                    f"   - Claim: `{experiment.get('claim_key', '')}` ({experiment.get('current_readiness', '')} -> `{experiment.get('target_status', '')}`)",
                    f"   - Priority: `{experiment.get('priority', '')}` with score `{_format_metric(experiment.get('rank_score'))}`",
                    f"   - Why: {experiment.get('why', '')}",
                    f"   - Experiment: {experiment.get('recommended_experiment', '')}",
                ]
            )
            if criteria:
                markdown_lines.append(f"   - First success criterion: {criteria[0]}")
    return payload, markdown_lines


def build_research_platform_lab(*, output_dir: Path, run_dir: Path | None, logger) -> list[Path]:
    now = datetime.now(timezone.utc)
    anchor_run_dir = _resolve_anchor_run_dir(output_dir, run_dir)
    control_history = _load_control_room_history(output_dir)
    claim_pack_path = output_dir / "analysis" / "research_claims" / "research_claims.json"
    research_payload = safe_read_json(claim_pack_path, default={})
    research_payload = research_payload if isinstance(research_payload, dict) else {}
    claim_run_id = ""
    run_payload = research_payload.get("run", {})
    if isinstance(run_payload, dict):
        claim_run_id = str(run_payload.get("run_id", "")).strip()

    run_registry = _build_run_research_registry(
        output_dir=output_dir,
        control_history=control_history,
        claim_run_id=claim_run_id,
        claim_pack_path=claim_pack_path,
        now=now,
    )
    benchmark_atlas = _load_benchmark_lock_atlas(output_dir, now=now)
    claim_registry, research_payload = _build_claim_registry(output_dir, now=now)
    creator_evidence_registry, creator_evidence_summary = _build_creator_evidence_registry(
        output_dir,
        now=now,
    )
    if run_registry.empty and benchmark_atlas.empty and claim_registry.empty:
        return []

    maturity_payload, maturity_markdown = _build_maturity_brief(
        anchor_run_dir=anchor_run_dir,
        run_registry=run_registry,
        benchmark_atlas=benchmark_atlas,
        claim_registry=claim_registry,
        research_payload=research_payload,
        creator_evidence_summary=creator_evidence_summary,
    )
    next_experiment_payload, next_experiment_markdown = _build_next_experiment_plan(
        run_registry=run_registry,
        benchmark_atlas=benchmark_atlas,
        claim_registry=claim_registry,
    )

    output_root = output_dir / "analysis" / "research_platform_lab"
    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    tables = {
        "run_research_registry": (
            run_registry,
            [
                "run_id",
                "profile",
                "timestamp",
                "promoted",
                "champion_gate_status",
                "benchmark_protocol_present",
                "safety_platform_contract_present",
                "conformal_summary_count",
                "backtest_model_count",
                "benchmark_contract_version",
                "benchmark_comparison_mode",
                "safety_api_group_count",
                "spotify_wrapper_count",
                "portability_note_count",
                "portability_signal_status",
                "research_artifact_ratio",
                "research_stage",
                "claim_pack_attached",
                "claim_pack_path",
                "claim_pack_freshness_status",
                "claim_pack_stale_source_path",
                "claim_pack_stale_source_count",
                "run_manifest_path",
                "run_manifest_timestamp",
                "run_manifest_age_hours",
                "benchmark_protocol_path",
                "safety_platform_contract_path",
                "target_drift_jsd",
                "test_selective_risk",
                "test_abstention_rate",
                "robustness_gap",
                "stress_skip_risk",
                "ops_coverage_ratio",
            ],
        ),
        "benchmark_lock_atlas": (
            benchmark_atlas,
            [
                "benchmark_id",
                "canonical_profile",
                "comparison_mode",
                "comparison_ready",
                "comparison_status",
                "run_count",
                "model_count",
                "present_artifact_count",
                "required_artifact_count",
                "required_artifact_ratio",
                "significant_pair_count",
                "comparison_blocker_count",
                "top_comparison_blocker",
                "comparison_blockers_json",
                "comparator_guard_status",
                "deep_comparator_ready",
                "observed_model_classes_json",
                "best_model_name",
                "best_model_type",
                "best_val_top1_mean",
                "best_test_top1_mean",
                "top_significant_pair",
                "top_significant_margin",
                "manifest_freshness_status",
                "manifest_stale_source_path",
                "manifest_stale_source_count",
                "manifest_age_hours",
                "summary_path",
                "significance_path",
                "benchmark_strength_score",
                "manifest_path",
            ],
        ),
        "research_claim_registry": (
            claim_registry,
            [
                "claim_key",
                "title",
                "role",
                "status",
                "claim_readiness_status",
                "summary",
                "live_signal_status",
                "benchmark_evidence_status",
                "repeated_evidence_status",
                "slice_evidence_status",
                "risk_evidence_status",
                "artifact_pack_status",
                "supporting_artifact_count",
                "existing_supporting_artifact_count",
                "missing_supporting_artifact_count",
                "stale_supporting_artifact_count",
                "supporting_artifact_path_status",
                "supporting_artifact_freshness_status",
                "missing_supporting_artifact_path",
                "stale_supporting_artifact_path",
                "missing_check_count",
                "blocked",
                "next_gate",
                "target_drift_jsd",
                "selective_risk",
                "stress_skip_risk",
                "live_test_top1_lift_vs_deep",
                "benchmark_comparison_ready",
                "benchmark_significant_lift",
                "claims_path",
                "metrics_json",
                "missing_checks_json",
            ],
        ),
        "creator_evidence_registry": (
            creator_evidence_registry,
            [
                "passport_id",
                "artist_name",
                "market",
                "evidence_grade",
                "raw_verified",
                "effective_verified",
                "evidence_status",
                "contract_version",
                "contract_status",
                "freshness_status",
                "source_artifact_path_status",
                "source_artifact_count",
                "existing_source_artifact_count",
                "missing_source_artifact_count",
                "stale_source_artifact_count",
                "missing_source_artifact_path",
                "stale_source_artifact_path",
                "occurrence_count",
                "report_family_count",
                "latest_source_age_days",
                "manifest_path",
                "passport_json_path",
            ],
        ),
    }
    manifest_payload = {
        "anchor_run_id": maturity_payload["anchor_run_id"],
        "artifact_root": str(output_root),
        "creator_evidence": creator_evidence_summary,
        "tables": {},
        "reports": {},
    }
    for stem, (frame, columns) in tables.items():
        csv_path = _write_csv(output_root / f"{stem}.csv", _rows_for_columns(frame, columns), columns)
        json_path = write_json(output_root / f"{stem}.json", frame.to_dict(orient="records"))
        manifest_payload["tables"][stem] = {
            "row_count": int(len(frame.index)),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }
        paths.extend([csv_path, json_path])

    maturity_json = write_json(output_root / "research_platform_maturity.json", maturity_payload)
    maturity_md = write_markdown(output_root / "research_platform_maturity.md", maturity_markdown)
    next_experiment_json = write_json(output_root / "research_next_experiments.json", next_experiment_payload)
    next_experiment_md = write_markdown(output_root / "research_next_experiments.md", next_experiment_markdown)
    manifest_payload["reports"]["research_platform_maturity"] = {
        "json_path": str(maturity_json),
        "markdown_path": str(maturity_md),
    }
    manifest_payload["reports"]["research_next_experiments"] = {
        "experiment_count": int(next_experiment_payload.get("experiment_count", 0) or 0),
        "json_path": str(next_experiment_json),
        "markdown_path": str(next_experiment_md),
    }
    manifest_json = write_json(output_root / "research_platform_manifest.json", manifest_payload)
    paths.extend([maturity_json, maturity_md, next_experiment_json, next_experiment_md, manifest_json])
    logger.info(
        "Built research platform lab with %d runs, %d benchmark locks, %d claims, %d creator evidence passports, and %d next experiments.",
        len(run_registry.index),
        len(benchmark_atlas.index),
        len(claim_registry.index),
        len(creator_evidence_registry.index),
        int(next_experiment_payload.get("experiment_count", 0) or 0),
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build research-platform branch artifacts from completed runs, benchmark locks, and claim packs.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory containing run, history, and claim artifacts.")
    parser.add_argument("--run-dir", type=str, default="", help="Optional explicit run directory for the maturity anchor.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.research_platform_lab")
    run_dir = Path(args.run_dir).expanduser().resolve() if str(args.run_dir).strip() else None
    paths = build_research_platform_lab(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        run_dir=run_dir,
        logger=logger,
    )
    if not paths:
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
