from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
from pathlib import Path

import pandas as pd

from .run_artifacts import safe_read_csv, safe_read_json, write_csv_rows, write_json, write_markdown


SCORECARD_COLUMNS = [
    "branch_key",
    "branch_name",
    "scope_lane",
    "audience",
    "status",
    "readiness_score",
    "evidence_score",
    "freshness_score",
    "risk_score",
    "primary_metric_name",
    "primary_metric_value",
    "artifact_count",
    "artifact_root",
    "top_signal",
    "top_gap",
    "recommended_next_step",
    "proof_artifacts",
]

QUEUE_COLUMNS = [
    "rank",
    "branch_key",
    "branch_name",
    "initiative",
    "why_now",
    "success_metric",
    "required_artifacts",
    "command",
    "effort",
    "impact_score",
    "risk_reduction_score",
    "dependencies",
]

STRATEGY_CARD_COLUMNS = [
    "branch_key",
    "branch_name",
    "development_mode",
    "status",
    "readiness_score",
    "risk_score",
    "sprint_objective",
    "next_initiative",
    "why_now",
    "success_metric",
    "primary_command",
    "validation_command",
    "required_artifacts",
    "proof_artifacts",
    "decision_rule",
    "handoff_summary",
]


def _warehouse_asset_registered(manifest: dict[str, object], asset_name: str) -> bool:
    layers = manifest.get("layers", {})
    if not isinstance(layers, dict):
        return False
    for assets in layers.values():
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if isinstance(asset, dict) and str(asset.get("name", "") or "") == asset_name:
                return True
    return False


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: object, default: int = 0) -> int:
    metric = _safe_float(value)
    return int(metric) if math.isfinite(metric) else default


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(min(1.0, max(0.0, value)), 4)


def _artifact_count(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.exists())


def _proof(paths: list[Path]) -> str:
    return " | ".join(str(path) for path in paths if path.exists())


def _status_from_score(score: float, *, has_artifacts: bool, risk_score: float) -> str:
    if not has_artifacts:
        return "missing"
    if score >= 0.75 and risk_score <= 0.30:
        return "ready"
    if score >= 0.45:
        return "attention"
    return "blocked"


def _top_csv_row(frame: pd.DataFrame, sort_column: str) -> dict[str, object]:
    if frame.empty or sort_column not in frame.columns:
        return {}
    working = frame.copy()
    working[sort_column] = pd.to_numeric(working[sort_column], errors="coerce")
    working = working.sort_values(sort_column, ascending=False, na_position="last")
    if working.empty:
        return {}
    return working.iloc[0].to_dict()


def _branch_row(
    *,
    branch_key: str,
    branch_name: str,
    scope_lane: str,
    audience: str,
    status: str,
    readiness_score: float,
    evidence_score: float,
    freshness_score: float,
    risk_score: float,
    primary_metric_name: str,
    primary_metric_value: object,
    artifact_count: int,
    artifact_root: Path,
    top_signal: str,
    top_gap: str,
    recommended_next_step: str,
    proof_artifacts: str,
) -> dict[str, object]:
    return {
        "branch_key": branch_key,
        "branch_name": branch_name,
        "scope_lane": scope_lane,
        "audience": audience,
        "status": status,
        "readiness_score": _bounded(readiness_score),
        "evidence_score": _bounded(evidence_score),
        "freshness_score": _bounded(freshness_score),
        "risk_score": _bounded(risk_score),
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "artifact_count": int(artifact_count),
        "artifact_root": str(artifact_root),
        "top_signal": top_signal,
        "top_gap": top_gap,
        "recommended_next_step": recommended_next_step,
        "proof_artifacts": proof_artifacts,
    }


def _queue_row(
    *,
    branch: dict[str, object],
    initiative: str,
    why_now: str,
    success_metric: str,
    required_artifacts: list[str],
    command: str,
    effort: str,
    impact_score: float,
    risk_reduction_score: float,
    dependencies: list[str],
) -> dict[str, object]:
    return {
        "rank": 0,
        "branch_key": branch["branch_key"],
        "branch_name": branch["branch_name"],
        "initiative": initiative,
        "why_now": why_now,
        "success_metric": success_metric,
        "required_artifacts": " | ".join(required_artifacts),
        "command": command,
        "effort": effort,
        "impact_score": _bounded(impact_score),
        "risk_reduction_score": _bounded(risk_reduction_score),
        "dependencies": " | ".join(dependencies),
    }


def _analytics_engineering_branch(output_dir: Path) -> dict[str, object]:
    root = output_dir / "analytics" / "warehouse"
    manifest_path = root / "warehouse_manifest.json"
    lineage_path = root / "warehouse_lineage.json"
    verification_path = root / "warehouse_verification.json"
    scope_health_mart_path = root / "gold" / "mart_scope_expansion_health.parquet"
    duckdb_path = output_dir / "analytics" / "spotify_analytics.duckdb"
    manifest = safe_read_json(manifest_path, default={})
    manifest = manifest if isinstance(manifest, dict) else {}
    quality = manifest.get("quality", {}) if isinstance(manifest.get("quality"), dict) else {}
    quality_summary = quality.get("summary", {}) if isinstance(quality.get("summary"), dict) else {}
    lineage_graph = manifest.get("lineage_graph", {}) if isinstance(manifest.get("lineage_graph"), dict) else {}
    freshness_counts = quality_summary.get("branch_backed_freshness_status_counts", {})
    freshness_counts = freshness_counts if isinstance(freshness_counts, dict) else {}

    asset_count = _safe_int(quality_summary.get("asset_count"))
    row_anomalies = _safe_int(quality_summary.get("row_count_anomaly_count"))
    empty_assets = _safe_int(quality_summary.get("empty_asset_count"))
    lineage_edges = len(lineage_graph.get("edges", [])) if isinstance(lineage_graph.get("edges"), list) else 0
    attention_assets = sum(
        _safe_int(count)
        for status, count in freshness_counts.items()
        if str(status).strip().lower() not in {"fresh", "source_fresh"}
    )

    has_artifacts = manifest_path.exists()
    scope_health_queryable = scope_health_mart_path.exists() or _warehouse_asset_registered(
        manifest,
        "mart_scope_expansion_health",
    )
    duckdb_queryable = duckdb_path.exists()
    evidence_score = _bounded(
        (min(asset_count, 30) / 30.0) * 0.55
        + (min(lineage_edges, 24) / 24.0) * 0.30
        + (0.15 if scope_health_queryable else 0.0)
    )
    freshness_score = _bounded(1.0 - min(attention_assets / 8.0, 1.0))
    risk_score = _bounded(min(row_anomalies / 5.0, 1.0) * 0.65 + min(empty_assets / 20.0, 1.0) * 0.35)
    readiness = _bounded((0.50 * evidence_score) + (0.35 * freshness_score) + (0.15 * (1.0 - risk_score)))
    status = _status_from_score(readiness, has_artifacts=has_artifacts, risk_score=risk_score)
    if row_anomalies or empty_assets:
        top_gap = f"{row_anomalies} row-count anomaly(s) and {empty_assets} empty asset(s)."
        recommended_next_step = "Clear warehouse quality gaps before using the branch-health mart as the default cockpit."
    elif not scope_health_queryable:
        top_gap = "Warehouse contract is present; next gap is making branch-level health queryable from one mart."
        recommended_next_step = (
            "Promote this four-branch scorecard into the warehouse/DuckDB layer as the default branch-health mart."
        )
    elif not duckdb_queryable:
        top_gap = "Scope-health mart is present; next gap is refreshing the local DuckDB priority view."
        recommended_next_step = "Run `make analytics-db` so DuckDB exposes `scope_expansion_priority_queue`."
    else:
        top_gap = "Branch-health mart and DuckDB surface are present; next gap is a notebook or dashboard consumer."
        recommended_next_step = "Build a lightweight branch-cockpit notebook or dashboard from `scope_expansion_priority_queue`."

    return _branch_row(
        branch_key="analytics_engineering",
        branch_name="Data Engineering + Analytics Engineering",
        scope_lane="local warehouse, lineage, data contracts",
        audience="you, future notebook/dashboard work, and any infra review",
        status=status,
        readiness_score=readiness,
        evidence_score=evidence_score,
        freshness_score=freshness_score,
        risk_score=risk_score,
        primary_metric_name="warehouse_asset_count",
        primary_metric_value=asset_count,
        artifact_count=_artifact_count([manifest_path, lineage_path, verification_path]),
        artifact_root=root,
        top_signal=(
            f"{asset_count} warehouse assets with {lineage_edges} lineage edge(s); "
            f"scope-health mart present=`{scope_health_queryable}`."
        ),
        top_gap=top_gap,
        recommended_next_step=recommended_next_step,
        proof_artifacts=_proof([manifest_path, lineage_path, verification_path, scope_health_mart_path, duckdb_path]),
    )


def _quant_branch(output_dir: Path) -> dict[str, object]:
    root = output_dir / "analysis" / "quant_decision_lab"
    frontier_path = root / "model_decision_frontier.csv"
    policy_path = root / "policy_decision_frontier.csv"
    simulation_path = root / "scenario_utility_simulation.csv"
    bridge_path = root / "archetype_decision_bridge.json"
    brief_path = root / "quant_decision_brief.json"
    frontier = safe_read_csv(frontier_path)
    policies = safe_read_csv(policy_path)
    simulation = safe_read_csv(simulation_path)
    bridge = safe_read_json(bridge_path, default={})
    bridge = bridge if isinstance(bridge, dict) else {}

    top_model = _top_csv_row(frontier, "utility_score")
    top_combo = _top_csv_row(simulation, "utility_score")
    frontier_count = len(frontier.index)
    policy_count = len(policies.index)
    simulation_count = len(simulation.index)
    bridge_ok = str(bridge.get("status", "")).strip().lower() == "ok"
    top_utility = _safe_float(top_combo.get("utility_score", top_model.get("utility_score")), default=0.0)
    high_risk_rows = 0
    if not simulation.empty and "high_drift_context" in simulation.columns:
        high_risk_rows += int(simulation["high_drift_context"].astype(str).str.lower().isin({"true", "1"}).sum())
    if not simulation.empty and "high_skip_context" in simulation.columns:
        high_risk_rows += int(simulation["high_skip_context"].astype(str).str.lower().isin({"true", "1"}).sum())

    evidence_score = _bounded((min(frontier_count, 6) / 6.0) * 0.35 + (min(policy_count, 4) / 4.0) * 0.25 + (min(simulation_count, 8) / 8.0) * 0.40)
    freshness_score = 1.0 if bridge_ok else 0.55 if bridge_path.exists() else 0.25
    risk_score = _bounded(min(high_risk_rows / max(simulation_count, 1), 1.0) if simulation_count else 0.65)
    readiness = _bounded((0.55 * evidence_score) + (0.25 * freshness_score) + (0.20 * (1.0 - risk_score)))
    status = _status_from_score(readiness, has_artifacts=frontier_path.exists() or brief_path.exists(), risk_score=risk_score)
    model_name = str(top_combo.get("model_name") or top_model.get("model_name") or "")
    policy_name = str(top_combo.get("policy_name") or "")
    scenario = str(top_combo.get("scenario") or "")
    top_signal = (
        f"Top simulated combo is {model_name} / {policy_name} / {scenario} at utility {top_utility:.3f}."
        if model_name or policy_name or scenario
        else "Quant frontier artifacts are not populated yet."
    )

    return _branch_row(
        branch_key="data_science_quant",
        branch_name="Data Science + Quant",
        scope_lane="frontier analysis, policy utility, scenario simulation",
        audience="modeling review, experiment design, and local quant research",
        status=status,
        readiness_score=readiness,
        evidence_score=evidence_score,
        freshness_score=freshness_score,
        risk_score=risk_score,
        primary_metric_name="top_scenario_utility",
        primary_metric_value=round(top_utility, 4),
        artifact_count=_artifact_count([frontier_path, policy_path, simulation_path, bridge_path, brief_path]),
        artifact_root=root,
        top_signal=top_signal,
        top_gap=(
            "Scenario utility simulation is missing, so quant choices cannot yet be reviewed as concrete model/policy/scenario portfolios."
            if simulation.empty
            else "Utility frontier exists; next gap is repeated scenario sweeps across fresh runs."
        ),
        recommended_next_step="Run scenario utility sweeps after the next full run and compare whether the same model/policy choices stay efficient.",
        proof_artifacts=_proof([frontier_path, policy_path, simulation_path, bridge_path, brief_path]),
    )


def _creator_branch(output_dir: Path) -> dict[str, object]:
    root = output_dir / "analysis" / "creator_market_intelligence"
    manifest_path = root / "creator_market_manifest.json"
    brief_path = root / "creator_market_brief.json"
    trends_path = root / "creator_market_trend_deltas.csv"
    scene_path = root / "scene_market_pulse.csv"
    lane_path = root / "opportunity_lane_atlas.csv"
    manifest = safe_read_json(manifest_path, default={})
    manifest = manifest if isinstance(manifest, dict) else {}
    brief = safe_read_json(brief_path, default={})
    brief = brief if isinstance(brief, dict) else {}
    trends = safe_read_csv(trends_path)
    scenes = safe_read_csv(scene_path)

    family_count = _safe_int(manifest.get("report_family_count"))
    complete_count = _safe_int(manifest.get("complete_report_family_count"))
    partial_count = _safe_int(manifest.get("partial_report_family_count"))
    trend_count = len(trends.index)
    high_trends = int(trends["severity"].astype(str).str.lower().eq("high").sum()) if "severity" in trends.columns else 0
    top_scene = brief.get("top_scene", {}) if isinstance(brief.get("top_scene"), dict) else {}
    top_scene_name = str(top_scene.get("scene_name", "") or (_top_csv_row(scenes, "momentum_score").get("scene_name", "")))
    completion_ratio = complete_count / family_count if family_count else 0.0

    evidence_score = _bounded((min(family_count, 4) / 4.0) * 0.35 + (min(trend_count, 8) / 8.0) * 0.30 + (min(len(scenes.index), 6) / 6.0) * 0.20 + completion_ratio * 0.15)
    freshness_score = _bounded(1.0 - min(partial_count / max(family_count, 1), 1.0) * 0.55)
    risk_score = _bounded(min(partial_count / max(family_count, 1), 1.0) * 0.55 + min(high_trends / 6.0, 1.0) * 0.45)
    readiness = _bounded((0.55 * evidence_score) + (0.30 * freshness_score) + (0.15 * (1.0 - risk_score)))
    status = _status_from_score(readiness, has_artifacts=manifest_path.exists() or brief_path.exists(), risk_score=risk_score)

    return _branch_row(
        branch_key="creator_market_intelligence",
        branch_name="Creator / Market Intelligence",
        scope_lane="market pulse, opportunity lanes, release whitespace",
        audience="creator strategy, A&R-style analysis, and external market storytelling",
        status=status,
        readiness_score=readiness,
        evidence_score=evidence_score,
        freshness_score=freshness_score,
        risk_score=risk_score,
        primary_metric_name="report_family_count",
        primary_metric_value=family_count,
        artifact_count=_artifact_count([manifest_path, brief_path, trends_path, scene_path, lane_path]),
        artifact_root=root,
        top_signal=(
            f"Top market scene is {top_scene_name}; {trend_count} trend highlight(s) across {family_count} report family/families."
            if family_count
            else "Creator market outputs are missing or have no report-family coverage yet."
        ),
        top_gap=(
            f"{partial_count} partial report family/families still limit creator-market confidence."
            if partial_count
            else "Creator market outputs are coherent; next gap is converting trends into a repeatable strategy queue."
        ),
        recommended_next_step="Backfill creator report families and prioritize the strongest scenes/lanes into creator-market strategy cards.",
        proof_artifacts=_proof([manifest_path, brief_path, trends_path, scene_path, lane_path]),
    )


def _research_branch(output_dir: Path) -> dict[str, object]:
    root = output_dir / "analysis" / "research_platform_lab"
    maturity_path = root / "research_platform_maturity.json"
    next_path = root / "research_next_experiments.json"
    claim_path = root / "research_claim_registry.csv"
    benchmark_path = root / "benchmark_lock_atlas.csv"
    run_registry_path = root / "run_research_registry.csv"
    maturity = safe_read_json(maturity_path, default={})
    maturity = maturity if isinstance(maturity, dict) else {}
    next_payload = safe_read_json(next_path, default={})
    next_payload = next_payload if isinstance(next_payload, dict) else {}
    claims = safe_read_csv(claim_path)
    benchmarks = safe_read_csv(benchmark_path)

    blocked_claims = _safe_int(maturity.get("claim_blocked_count"))
    incomplete_locks = _safe_int(maturity.get("incomplete_benchmark_lock_count"))
    stale_refs = _safe_int(maturity.get("stale_claim_artifact_count")) + _safe_int(maturity.get("stale_benchmark_count"))
    next_experiments = next_payload.get("experiments", []) if isinstance(next_payload.get("experiments"), list) else []
    ready_claims = int(claims["claim_readiness_status"].astype(str).str.lower().eq("ready").sum()) if "claim_readiness_status" in claims.columns else 0
    claim_count = len(claims.index)
    ready_locks = int(benchmarks["comparison_status"].astype(str).str.lower().eq("ready").sum()) if "comparison_status" in benchmarks.columns else 0
    benchmark_count = len(benchmarks.index)
    top_blocker = str(maturity.get("top_blocker") or maturity.get("submission_blocker") or "")

    evidence_score = _bounded((ready_claims / max(claim_count, 1)) * 0.45 + (ready_locks / max(benchmark_count, 1)) * 0.35 + (min(len(next_experiments), 4) / 4.0) * 0.20)
    freshness_score = _bounded(1.0 - min(stale_refs / 5.0, 1.0))
    risk_score = _bounded(min(blocked_claims / max(claim_count, 1), 1.0) * 0.55 + min(incomplete_locks / max(benchmark_count, 1), 1.0) * 0.45)
    readiness = _bounded((0.55 * evidence_score) + (0.25 * freshness_score) + (0.20 * (1.0 - risk_score)))
    status = _status_from_score(readiness, has_artifacts=maturity_path.exists(), risk_score=risk_score)

    return _branch_row(
        branch_key="research_platform",
        branch_name="Research Platform",
        scope_lane="claim registry, benchmark locks, evidence roadmap",
        audience="publication-style review, evidence audits, and roadmap governance",
        status=status,
        readiness_score=readiness,
        evidence_score=evidence_score,
        freshness_score=freshness_score,
        risk_score=risk_score,
        primary_metric_name="blocked_claim_count",
        primary_metric_value=blocked_claims,
        artifact_count=_artifact_count([maturity_path, next_path, claim_path, benchmark_path, run_registry_path]),
        artifact_root=root,
        top_signal=f"{ready_claims}/{claim_count} claim(s) ready, {ready_locks}/{benchmark_count} benchmark lock(s) ready, {len(next_experiments)} next experiment card(s).",
        top_gap=top_blocker or (
            f"{blocked_claims} blocked claim(s), {incomplete_locks} incomplete benchmark lock(s), and {stale_refs} stale reference(s)."
            if blocked_claims or incomplete_locks or stale_refs
            else "Research evidence is organized; next gap is repeated-confirmation depth."
        ),
        recommended_next_step="Execute the highest-ranked research next-experiment card and regenerate claims after the benchmark evidence changes.",
        proof_artifacts=_proof([maturity_path, next_path, claim_path, benchmark_path, run_registry_path]),
    )


def build_scope_expansion_scorecard(output_dir: Path) -> list[dict[str, object]]:
    return [
        _analytics_engineering_branch(output_dir),
        _quant_branch(output_dir),
        _creator_branch(output_dir),
        _research_branch(output_dir),
    ]


def build_implementation_queue(scorecard: list[dict[str, object]]) -> list[dict[str, object]]:
    branch_lookup = {str(row["branch_key"]): row for row in scorecard}
    queue = [
        _queue_row(
            branch=branch_lookup["analytics_engineering"],
            initiative="Make the four-branch health scorecard queryable from the analytics layer",
            why_now=str(branch_lookup["analytics_engineering"]["top_gap"]),
            success_metric="DuckDB or warehouse consumers can query one branch-health table with status, score, proof artifacts, and next step.",
            required_artifacts=["branch_expansion_scorecard.csv", "warehouse_manifest.json", "warehouse_lineage.json"],
            command="make analytics-db && make scope-expansion-lab",
            effort="medium",
            impact_score=0.78,
            risk_reduction_score=0.66,
            dependencies=["fresh warehouse build"],
        ),
        _queue_row(
            branch=branch_lookup["data_science_quant"],
            initiative="Repeat scenario utility sweeps for the current frontier models",
            why_now=str(branch_lookup["data_science_quant"]["top_gap"]),
            success_metric="Top model/policy/scenario choice remains efficient across at least two fresh runs or the queue records why it changed.",
            required_artifacts=["scenario_utility_simulation.csv", "model_decision_frontier.csv", "archetype_decision_bridge.json"],
            command="make listener-archetypes && make quant-decision-lab",
            effort="medium",
            impact_score=0.84,
            risk_reduction_score=0.72,
            dependencies=["fresh completed run", "listener archetypes"],
        ),
        _queue_row(
            branch=branch_lookup["creator_market_intelligence"],
            initiative="Turn creator market trend deltas into strategy cards",
            why_now=str(branch_lookup["creator_market_intelligence"]["top_gap"]),
            success_metric="Top scenes, lanes, migration routes, and whitespace gaps each produce a named strategy card with proof artifacts.",
            required_artifacts=["creator_market_trend_deltas.csv", "scene_market_pulse.csv", "opportunity_lane_atlas.csv"],
            command="make public-insights && make creator-market-intelligence",
            effort="medium",
            impact_score=0.76,
            risk_reduction_score=0.55,
            dependencies=["creator report families", "public catalog metadata where available"],
        ),
        _queue_row(
            branch=branch_lookup["research_platform"],
            initiative="Close the highest-ranked evidence blocker from the research platform",
            why_now=str(branch_lookup["research_platform"]["top_gap"]),
            success_metric="Blocked claim count, incomplete benchmark-lock count, or stale evidence count drops after the next research-platform refresh.",
            required_artifacts=["research_next_experiments.json", "research_claim_registry.csv", "benchmark_lock_atlas.csv"],
            command="make benchmark-lock && make research-claims && make research-platform-lab",
            effort="high",
            impact_score=0.91,
            risk_reduction_score=0.88,
            dependencies=["benchmark lock", "research claims"],
        ),
    ]
    queue = sorted(
        queue,
        key=lambda row: (
            -_safe_float(row["risk_reduction_score"], default=0.0),
            -_safe_float(row["impact_score"], default=0.0),
            str(row["branch_key"]),
        ),
    )
    for rank, row in enumerate(queue, start=1):
        row["rank"] = rank
    return queue


def _scorecard_markdown(scorecard: list[dict[str, object]], queue: list[dict[str, object]]) -> list[str]:
    lines = [
        "# Scope Expansion Lab",
        "",
        "This artifact keeps the four local-first expansion lanes legible as the project grows.",
        "",
        "## Branch Scorecard",
        "",
        "| Branch | Status | Readiness | Evidence | Risk | Top Signal | Top Gap |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in scorecard:
        lines.append(
            "| {branch} | `{status}` | {readiness:.3f} | {evidence:.3f} | {risk:.3f} | {signal} | {gap} |".format(
                branch=row["branch_name"],
                status=row["status"],
                readiness=_safe_float(row["readiness_score"], default=0.0),
                evidence=_safe_float(row["evidence_score"], default=0.0),
                risk=_safe_float(row["risk_score"], default=0.0),
                signal=str(row["top_signal"]).replace("|", "/"),
                gap=str(row["top_gap"]).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Implementation Queue",
            "",
            "| Rank | Branch | Initiative | Impact | Risk Reduction | Command |",
            "| ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in queue:
        lines.append(
            "| {rank} | {branch} | {initiative} | {impact:.3f} | {risk:.3f} | `{command}` |".format(
                rank=row["rank"],
                branch=row["branch_name"],
                initiative=str(row["initiative"]).replace("|", "/"),
                impact=_safe_float(row["impact_score"], default=0.0),
                risk=_safe_float(row["risk_reduction_score"], default=0.0),
                command=row["command"],
            )
        )
    return lines


def _queue_markdown(queue: list[dict[str, object]]) -> list[str]:
    lines = ["# Scope Expansion Implementation Queue", ""]
    for row in queue:
        lines.extend(
            [
                f"## {row['rank']}. {row['initiative']}",
                "",
                f"- Branch: `{row['branch_name']}`",
                f"- Why now: {row['why_now']}",
                f"- Success metric: {row['success_metric']}",
                f"- Required artifacts: `{row['required_artifacts']}`",
                f"- Command: `{row['command']}`",
                f"- Effort: `{row['effort']}`",
                "",
            ]
        )
    return lines


def _development_mode(branch: dict[str, object]) -> str:
    status = str(branch.get("status", "") or "").strip().lower()
    readiness = _safe_float(branch.get("readiness_score"), default=0.0)
    risk = _safe_float(branch.get("risk_score"), default=1.0)
    if status == "missing":
        return "bootstrap"
    if status == "blocked" or risk >= 0.70:
        return "stabilize"
    if status == "ready" and risk <= 0.30:
        return "scale"
    if readiness >= 0.75:
        return "extend"
    return "investigate"


def _validation_command(branch_key: str) -> str:
    commands = {
        "analytics_engineering": (
            ".venv/bin/python -m pytest tests/test_analytics_warehouse.py tests/test_scope_expansion_lab.py"
        ),
        "data_science_quant": ".venv/bin/python -m pytest tests/test_quant_decision_lab.py tests/test_scope_expansion_lab.py",
        "creator_market_intelligence": (
            ".venv/bin/python -m pytest tests/test_creator_market_intelligence.py tests/test_scope_expansion_lab.py"
        ),
        "research_platform": ".venv/bin/python -m pytest tests/test_research_platform_lab.py tests/test_scope_expansion_lab.py",
    }
    return commands.get(branch_key, ".venv/bin/python -m pytest tests/test_scope_expansion_lab.py")


def _sprint_objective(mode: str, branch: dict[str, object], queue_item: dict[str, object]) -> str:
    branch_name = str(branch.get("branch_name", "") or "").strip()
    initiative = str(queue_item.get("initiative", "") or branch.get("recommended_next_step", "") or "").strip()
    if mode == "stabilize":
        return f"Reduce the highest {branch_name} risk by landing `{initiative}` with fresh proof artifacts."
    if mode == "scale":
        return f"Turn the current {branch_name} evidence into a repeatable consumer surface or dashboard handoff."
    if mode == "extend":
        return f"Deepen {branch_name} by converting the current signal into a reusable next-step workflow."
    if mode == "bootstrap":
        return f"Create the first credible {branch_name} artifact set and rerun the scope scorecard."
    return f"Clarify the main {branch_name} gap, then choose whether to stabilize, extend, or pause the branch."


def _decision_rule(mode: str, branch: dict[str, object], queue_item: dict[str, object]) -> str:
    success_metric = str(queue_item.get("success_metric", "") or "").strip()
    if mode == "stabilize":
        return f"Do not expand this branch until risk drops below `0.700` or `{success_metric}` moves in the right direction."
    if mode == "scale":
        return "Scale only if the validation command passes and the generated artifact is useful without opening raw logs."
    if mode == "extend":
        return f"Extend if `{success_metric}` can be checked from generated artifacts after one local refresh."
    if mode == "bootstrap":
        return "Bootstrap is complete when the scorecard status moves from `missing` to `attention` or better."
    return "Keep this branch in discovery until the next artifact clearly changes readiness, risk, or recommended command."


def _handoff_summary(branch: dict[str, object], queue_item: dict[str, object], mode: str) -> str:
    branch_name = str(branch.get("branch_name", "") or "").strip()
    initiative = str(queue_item.get("initiative", "") or branch.get("recommended_next_step", "") or "").strip()
    command = str(queue_item.get("command", "") or "").strip()
    return f"{branch_name} is in `{mode}` mode. Next land `{initiative}` with `{command}`."


def _build_strategy_cards(
    scorecard: list[dict[str, object]],
    queue: list[dict[str, object]],
) -> list[dict[str, object]]:
    queue_by_branch = {str(row.get("branch_key", "") or ""): row for row in queue}
    cards: list[dict[str, object]] = []
    for branch in scorecard:
        branch_key = str(branch.get("branch_key", "") or "")
        queue_item = queue_by_branch.get(branch_key, {})
        mode = _development_mode(branch)
        cards.append(
            {
                "branch_key": branch_key,
                "branch_name": branch.get("branch_name"),
                "development_mode": mode,
                "status": branch.get("status"),
                "readiness_score": branch.get("readiness_score"),
                "risk_score": branch.get("risk_score"),
                "sprint_objective": _sprint_objective(mode, branch, queue_item),
                "next_initiative": queue_item.get("initiative") or branch.get("recommended_next_step"),
                "why_now": queue_item.get("why_now") or branch.get("top_gap"),
                "success_metric": queue_item.get("success_metric"),
                "primary_command": queue_item.get("command"),
                "validation_command": _validation_command(branch_key),
                "required_artifacts": queue_item.get("required_artifacts"),
                "proof_artifacts": branch.get("proof_artifacts"),
                "decision_rule": _decision_rule(mode, branch, queue_item),
                "handoff_summary": _handoff_summary(branch, queue_item, mode),
            }
        )
    return sorted(
        cards,
        key=lambda row: (
            -_safe_float(row.get("risk_score"), default=0.0),
            str(row.get("branch_key", "") or ""),
        ),
    )


def _branch_development_cockpit(
    scorecard: list[dict[str, object]],
    queue: list[dict[str, object]],
    strategy_cards: list[dict[str, object]],
    *,
    generated_at: str,
) -> dict[str, object]:
    queue_by_branch = {str(row.get("branch_key", "") or ""): row for row in queue}
    cards_by_branch = {str(row.get("branch_key", "") or ""): row for row in strategy_cards}
    lanes: list[dict[str, object]] = []
    for branch in sorted(
        scorecard,
        key=lambda row: (
            -_safe_float(row.get("risk_score"), default=0.0),
            _safe_float(row.get("readiness_score"), default=0.0),
            str(row.get("branch_key", "") or ""),
        ),
    ):
        branch_key = str(branch.get("branch_key", "") or "")
        queue_item = queue_by_branch.get(branch_key, {})
        card = cards_by_branch.get(branch_key, {})
        lanes.append(
            {
                "branch_key": branch.get("branch_key"),
                "branch_name": branch.get("branch_name"),
                "status": branch.get("status"),
                "development_mode": card.get("development_mode") or _development_mode(branch),
                "readiness_score": branch.get("readiness_score"),
                "risk_score": branch.get("risk_score"),
                "next_initiative": card.get("next_initiative") or queue_item.get("initiative"),
                "next_command": card.get("primary_command") or queue_item.get("command"),
                "validation_command": card.get("validation_command"),
                "sprint_objective": card.get("sprint_objective"),
                "decision_rule": card.get("decision_rule"),
                "top_signal": branch.get("top_signal"),
                "top_gap": branch.get("top_gap"),
                "proof_artifacts": branch.get("proof_artifacts"),
            }
        )

    command_sequence: list[str] = []
    for row in queue:
        command = str(row.get("command", "") or "").strip()
        if command and command not in command_sequence:
            command_sequence.append(command)
    validation_sequence: list[str] = []
    for row in queue:
        card = cards_by_branch.get(str(row.get("branch_key", "") or ""), {})
        command = str(card.get("validation_command", "") or "").strip()
        if command and command not in validation_sequence:
            validation_sequence.append(command)

    riskiest_branch = lanes[0] if lanes else {}
    top_queue = queue[0] if queue else {}
    return {
        "generated_at": generated_at,
        "summary": {
            "branch_count": len(scorecard),
            "ready_branch_count": sum(1 for row in scorecard if row.get("status") == "ready"),
            "attention_branch_count": sum(1 for row in scorecard if row.get("status") == "attention"),
            "blocked_branch_count": sum(1 for row in scorecard if row.get("status") == "blocked"),
            "missing_branch_count": sum(1 for row in scorecard if row.get("status") == "missing"),
            "riskiest_branch_key": riskiest_branch.get("branch_key", ""),
            "top_queue_branch_key": top_queue.get("branch_key", ""),
            "top_queue_command": top_queue.get("command", ""),
        },
        "recommended_command_sequence": command_sequence,
        "recommended_validation_sequence": validation_sequence,
        "lanes": lanes,
    }


def _cockpit_markdown(cockpit: dict[str, object]) -> list[str]:
    summary = cockpit.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    command_sequence = cockpit.get("recommended_command_sequence", [])
    command_sequence = command_sequence if isinstance(command_sequence, list) else []
    validation_sequence = cockpit.get("recommended_validation_sequence", [])
    validation_sequence = validation_sequence if isinstance(validation_sequence, list) else []
    lanes = cockpit.get("lanes", [])
    lanes = lanes if isinstance(lanes, list) else []
    lines = [
        "# Branch Development Cockpit",
        "",
        f"- Generated at: `{cockpit.get('generated_at', '')}`",
        f"- Branches tracked: `{summary.get('branch_count', 0)}`",
        f"- Ready branches: `{summary.get('ready_branch_count', 0)}`",
        f"- Attention branches: `{summary.get('attention_branch_count', 0)}`",
        f"- Blocked branches: `{summary.get('blocked_branch_count', 0)}`",
        f"- Missing branches: `{summary.get('missing_branch_count', 0)}`",
        f"- Riskiest branch: `{summary.get('riskiest_branch_key', '')}`",
        f"- Top queue command: `{summary.get('top_queue_command', '')}`",
        "",
        "## Recommended Command Sequence",
        "",
    ]
    if command_sequence:
        for command in command_sequence:
            lines.append(f"- `{command}`")
    else:
        lines.append("- No branch commands are currently available.")
    lines.extend(
        [
            "",
            "## Recommended Validation Sequence",
            "",
        ]
    )
    if validation_sequence:
        for command in validation_sequence:
            lines.append(f"- `{command}`")
    else:
        lines.append("- No validation commands are currently available.")
    lines.extend(
        [
            "",
            "## Branch Lanes",
            "",
            "| Branch | Status | Mode | Readiness | Risk | Next Command |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lines.append(
            "| {branch} | `{status}` | `{mode}` | {readiness:.3f} | {risk:.3f} | `{command}` |".format(
                branch=lane.get("branch_name", ""),
                status=lane.get("status", ""),
                mode=lane.get("development_mode", ""),
                readiness=_safe_float(lane.get("readiness_score"), default=0.0),
                risk=_safe_float(lane.get("risk_score"), default=0.0),
                command=lane.get("next_command", ""),
            )
        )
    return lines


def _strategy_card_lines(card: dict[str, object]) -> list[str]:
    return [
        f"# {card.get('branch_name', '')} Strategy Card",
        "",
        f"- Status: `{card.get('status', '')}`",
        f"- Development mode: `{card.get('development_mode', '')}`",
        f"- Readiness score: `{_safe_float(card.get('readiness_score'), default=0.0):.3f}`",
        f"- Risk score: `{_safe_float(card.get('risk_score'), default=0.0):.3f}`",
        f"- Sprint objective: {card.get('sprint_objective', '')}",
        f"- Next initiative: {card.get('next_initiative', '')}",
        f"- Why now: {card.get('why_now', '')}",
        f"- Success metric: {card.get('success_metric', '')}",
        f"- Primary command: `{card.get('primary_command', '')}`",
        f"- Validation command: `{card.get('validation_command', '')}`",
        f"- Required artifacts: `{card.get('required_artifacts', '')}`",
        f"- Decision rule: {card.get('decision_rule', '')}",
        f"- Handoff summary: {card.get('handoff_summary', '')}",
        "",
        "Proof artifacts:",
        "",
        f"- `{card.get('proof_artifacts', '') or 'none yet'}`",
    ]


def _strategy_cards_markdown(cards: list[dict[str, object]]) -> list[str]:
    lines = [
        "# Branch Strategy Cards",
        "",
        "These cards turn the scope scorecard into branch-by-branch sprint instructions.",
        "",
        "| Branch | Mode | Risk | Next Initiative | Validation |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for card in cards:
        lines.append(
            "| {branch} | `{mode}` | {risk:.3f} | {initiative} | `{validation}` |".format(
                branch=card.get("branch_name", ""),
                mode=card.get("development_mode", ""),
                risk=_safe_float(card.get("risk_score"), default=0.0),
                initiative=str(card.get("next_initiative", "") or "").replace("|", "/"),
                validation=card.get("validation_command", ""),
            )
        )
    lines.append("")
    for card in cards:
        lines.extend(["---", "", *_strategy_card_lines(card), ""])
    return lines


def _write_individual_strategy_cards(root: Path, cards: list[dict[str, object]]) -> dict[str, str]:
    card_dir = root / "strategy_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for card in cards:
        branch_key = str(card.get("branch_key", "") or "branch").strip() or "branch"
        path = write_markdown(card_dir / f"{branch_key}.md", _strategy_card_lines(card))
        paths[branch_key] = str(path)
    return paths


def build_scope_expansion_lab(*, output_dir: Path, logger: logging.Logger) -> list[Path]:
    output_dir = output_dir.expanduser().resolve()
    root = output_dir / "analysis" / "scope_expansion"
    root.mkdir(parents=True, exist_ok=True)

    scorecard = build_scope_expansion_scorecard(output_dir)
    queue = build_implementation_queue(scorecard)
    strategy_cards = _build_strategy_cards(scorecard, queue)
    generated_at = datetime.now(timezone.utc).isoformat()
    cockpit = _branch_development_cockpit(scorecard, queue, strategy_cards, generated_at=generated_at)

    scorecard_csv = write_csv_rows(root / "branch_expansion_scorecard.csv", scorecard, fieldnames=SCORECARD_COLUMNS)
    scorecard_json = write_json(root / "branch_expansion_scorecard.json", scorecard)
    queue_csv = write_csv_rows(root / "branch_expansion_implementation_queue.csv", queue, fieldnames=QUEUE_COLUMNS)
    queue_json = write_json(root / "branch_expansion_implementation_queue.json", queue)
    strategy_cards_csv = write_csv_rows(root / "branch_strategy_cards.csv", strategy_cards, fieldnames=STRATEGY_CARD_COLUMNS)
    strategy_cards_json = write_json(root / "branch_strategy_cards.json", strategy_cards)
    scorecard_md = write_markdown(root / "branch_expansion_scorecard.md", _scorecard_markdown(scorecard, queue))
    queue_md = write_markdown(root / "branch_expansion_implementation_queue.md", _queue_markdown(queue))
    strategy_cards_md = write_markdown(root / "branch_strategy_cards.md", _strategy_cards_markdown(strategy_cards))
    individual_strategy_card_paths = _write_individual_strategy_cards(root, strategy_cards)
    cockpit_json = write_json(root / "branch_development_cockpit.json", cockpit)
    cockpit_md = write_markdown(root / "branch_development_cockpit.md", _cockpit_markdown(cockpit))
    manifest = {
        "generated_at": generated_at,
        "artifact_root": str(root),
        "branch_count": len(scorecard),
        "ready_branch_count": sum(1 for row in scorecard if row["status"] == "ready"),
        "attention_branch_count": sum(1 for row in scorecard if row["status"] == "attention"),
        "blocked_branch_count": sum(1 for row in scorecard if row["status"] == "blocked"),
        "missing_branch_count": sum(1 for row in scorecard if row["status"] == "missing"),
        "queue_count": len(queue),
        "strategy_card_count": len(strategy_cards),
        "top_queue_item": queue[0] if queue else {},
        "tables": {
            "branch_expansion_scorecard": {
                "row_count": len(scorecard),
                "csv_path": str(scorecard_csv),
                "json_path": str(scorecard_json),
                "markdown_path": str(scorecard_md),
            },
            "branch_expansion_implementation_queue": {
                "row_count": len(queue),
                "csv_path": str(queue_csv),
                "json_path": str(queue_json),
                "markdown_path": str(queue_md),
            },
            "branch_strategy_cards": {
                "row_count": len(strategy_cards),
                "csv_path": str(strategy_cards_csv),
                "json_path": str(strategy_cards_json),
                "markdown_path": str(strategy_cards_md),
                "individual_markdown_paths": individual_strategy_card_paths,
            },
        },
        "reports": {
            "branch_development_cockpit": {
                "json_path": str(cockpit_json),
                "markdown_path": str(cockpit_md),
            },
        },
    }
    manifest_json = write_json(root / "scope_expansion_manifest.json", manifest)
    paths = [
        scorecard_csv,
        scorecard_json,
        scorecard_md,
        queue_csv,
        queue_json,
        queue_md,
        strategy_cards_csv,
        strategy_cards_json,
        strategy_cards_md,
        *(Path(path) for path in individual_strategy_card_paths.values()),
        cockpit_json,
        cockpit_md,
        manifest_json,
    ]
    logger.info("Built scope expansion lab with %d branches and %d queue items.", len(scorecard), len(queue))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the four-branch scope expansion scorecard and implementation queue.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Project output directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.scope_expansion_lab")
    paths = build_scope_expansion_lab(output_dir=Path(args.output_dir), logger=logger)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
