from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
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


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


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
                "research_artifact_ratio": ratio,
                "research_stage": _research_stage(
                    protocol_present=protocol_path.exists(),
                    contract_present=contract_path.exists(),
                    artifact_ratio=ratio,
                    promoted=bool(champion_gate.get("promoted", False)),
                ),
                "claim_pack_attached": run_id == claim_run_id,
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


def _load_benchmark_lock_atlas(output_dir: Path) -> pd.DataFrame:
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
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "canonical_profile": str(payload.get("canonical_profile", "")).strip(),
                "comparison_mode": str(payload.get("comparison_mode", "")).strip(),
                "comparison_ready": bool(payload.get("comparison_ready", False)),
                "run_count": int(payload.get("run_count", 0) or 0),
                "model_count": int(len(summary_df.index)),
                "required_artifact_ratio": required_ratio,
                "significant_pair_count": int(payload.get("significant_pair_count", 0) or 0),
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


def _build_claim_registry(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = safe_read_json(output_dir / "analysis" / "research_claims" / "research_claims.json", default={})
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
        rows.append(
            {
                "claim_key": key,
                "title": str(claim.get("title", "")).strip(),
                "role": "primary" if key == primary_key else "backup",
                "status": str(claim.get("status", "")).strip(),
                "summary": str(claim.get("summary", "")).strip(),
                "live_signal_status": str(support.get("live_signal_status", "")).strip(),
                "benchmark_evidence_status": str(support.get("benchmark_evidence_status", "")).strip(),
                "repeated_evidence_status": str(support.get("repeated_evidence_status", "")).strip(),
                "slice_evidence_status": str(support.get("slice_evidence_status", "")).strip(),
                "risk_evidence_status": str(support.get("risk_evidence_status", "")).strip(),
                "artifact_pack_status": str(support.get("artifact_pack_status", "")).strip(),
                "supporting_artifact_count": int(len(claim.get("supporting_artifacts", [])) if isinstance(claim.get("supporting_artifacts", []), list) else 0),
                "missing_check_count": int(len(missing_checks)),
                "next_gate": str(support.get("next_gate", "")).strip(),
                "target_drift_jsd": _safe_float(metrics.get("target_drift_jsd")),
                "selective_risk": _safe_float(metrics.get("selective_risk")),
                "stress_skip_risk": _safe_float(metrics.get("stress_skip_risk")),
                "live_test_top1_lift_vs_deep": _safe_float(metrics.get("live_test_top1_lift_vs_deep")),
                "benchmark_comparison_ready": bool(metrics.get("benchmark_comparison_ready", False)),
                "benchmark_significant_lift": bool(metrics.get("benchmark_significant_lift", False)),
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
    total_claim_count = int(len(claim_registry.index))
    submission_readiness = research_payload.get("submission_readiness", {})
    submission_readiness = submission_readiness if isinstance(submission_readiness, dict) else {}
    blockers = submission_readiness.get("blockers", [])
    blockers = blockers if isinstance(blockers, list) else []
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
    if anchor_row:
        summary.append(
            f"Anchor run stage is `{anchor_row.get('research_stage', '')}` with artifact ratio `{_safe_float(anchor_row.get('research_artifact_ratio')):.3f}` and promoted `{bool(anchor_row.get('promoted', False))}`."
        )
    if benchmark_row:
        summary.append(
            f"Strongest benchmark lock is `{benchmark_row.get('benchmark_id', '')}` with comparison ready `{bool(benchmark_row.get('comparison_ready', False))}` and leading model `{benchmark_row.get('best_model_name', '')}`."
        )
    if total_claim_count:
        summary.append(
            f"Claim registry currently tracks `{ready_claim_count}/{total_claim_count}` claims at `analysis_ready` or better."
        )
    if blockers:
        summary.append(f"Top submission blocker remains `{str(blockers[0]).strip()}`.")
    elif top_next_gate:
        summary.append(f"Highest-leverage next gate is `{top_next_gate}`.")
    else:
        summary.append("Current claim pack is structurally complete enough to keep iterating locally without reopening the benchmark contract.")

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
        "claim_total_count": total_claim_count,
        "submission_status": str(submission_readiness.get("status", "")).strip(),
        "ready_for_external_review": bool(submission_readiness.get("ready_for_external_review", False)),
        "blockers": blockers,
        "top_next_gate": top_next_gate,
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


def build_research_platform_lab(*, output_dir: Path, run_dir: Path | None, logger) -> list[Path]:
    anchor_run_dir = _resolve_anchor_run_dir(output_dir, run_dir)
    control_history = _load_control_room_history(output_dir)
    research_payload = safe_read_json(output_dir / "analysis" / "research_claims" / "research_claims.json", default={})
    research_payload = research_payload if isinstance(research_payload, dict) else {}
    claim_run_id = ""
    run_payload = research_payload.get("run", {})
    if isinstance(run_payload, dict):
        claim_run_id = str(run_payload.get("run_id", "")).strip()

    run_registry = _build_run_research_registry(
        output_dir=output_dir,
        control_history=control_history,
        claim_run_id=claim_run_id,
    )
    benchmark_atlas = _load_benchmark_lock_atlas(output_dir)
    claim_registry, research_payload = _build_claim_registry(output_dir)
    if run_registry.empty and benchmark_atlas.empty and claim_registry.empty:
        return []

    maturity_payload, maturity_markdown = _build_maturity_brief(
        anchor_run_dir=anchor_run_dir,
        run_registry=run_registry,
        benchmark_atlas=benchmark_atlas,
        claim_registry=claim_registry,
        research_payload=research_payload,
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
                "research_artifact_ratio",
                "research_stage",
                "claim_pack_attached",
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
                "run_count",
                "model_count",
                "required_artifact_ratio",
                "significant_pair_count",
                "best_model_name",
                "best_model_type",
                "best_val_top1_mean",
                "best_test_top1_mean",
                "top_significant_pair",
                "top_significant_margin",
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
                "summary",
                "live_signal_status",
                "benchmark_evidence_status",
                "repeated_evidence_status",
                "slice_evidence_status",
                "risk_evidence_status",
                "artifact_pack_status",
                "supporting_artifact_count",
                "missing_check_count",
                "next_gate",
                "target_drift_jsd",
                "selective_risk",
                "stress_skip_risk",
                "live_test_top1_lift_vs_deep",
                "benchmark_comparison_ready",
                "benchmark_significant_lift",
                "metrics_json",
                "missing_checks_json",
            ],
        ),
    }
    manifest_payload = {
        "anchor_run_id": maturity_payload["anchor_run_id"],
        "artifact_root": str(output_root),
        "tables": {},
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
    manifest_json = write_json(output_root / "research_platform_manifest.json", manifest_payload)
    paths.extend([maturity_json, maturity_md, manifest_json])
    logger.info(
        "Built research platform lab with %d runs, %d benchmark locks, and %d claims.",
        len(run_registry.index),
        len(benchmark_atlas.index),
        len(claim_registry.index),
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
