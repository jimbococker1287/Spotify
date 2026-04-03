from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .branch_portfolio import build_branch_portfolio_report
from .portfolio_artifacts import PortfolioArtifactBundle, load_portfolio_artifact_bundle
from .run_artifacts import write_json, write_markdown


_STATUS_RANK = {"missing": 0, "attention": 1, "ready": 2}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _worst_status(*statuses: str) -> str:
    resolved = [status for status in statuses if status in _STATUS_RANK]
    if not resolved:
        return "missing"
    return min(resolved, key=lambda status: _STATUS_RANK[status])


def _missing_paths(paths: list[Path], root: Path) -> list[str]:
    return [_relative(path, root) for path in paths if not path.exists()]


def _latest_run_dir(outputs_root: Path) -> Path | None:
    runs_dir = outputs_root / "runs"
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    manifest_backed = [path for path in candidates if (path / "run_manifest.json").exists()]
    ranked_candidates = manifest_backed or candidates
    return max(ranked_candidates, key=lambda path: path.stat().st_mtime)


def _artifact_summary(paths: list[Path]) -> dict[str, Any]:
    existing = [path for path in paths if path.exists()]
    latest_artifact = max(existing, key=lambda path: path.stat().st_mtime) if existing else None
    return {
        "expected_count": len(paths),
        "present_count": len(existing),
        "latest_artifact": str(latest_artifact.resolve()) if latest_artifact is not None else "",
        "latest_artifact_timestamp": _isoformat(_timestamp(latest_artifact)) if latest_artifact is not None else None,
    }


def _creator_manifest_status(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    primary_report = Path(str(payload.get("primary_report", "")))
    comparison_md = payload.get("comparison_view_markdown", {})
    comparison_csv = payload.get("comparison_view_csv", {})
    comparison_md = comparison_md if isinstance(comparison_md, dict) else {}
    comparison_csv = comparison_csv if isinstance(comparison_csv, dict) else {}
    expected_view_keys = {
        "ranking_comparison",
        "scene_comparison",
        "seed_comparison",
        "scene_seed_comparison",
    }
    md_paths = {key: Path(str(value)) for key, value in comparison_md.items()}
    csv_paths = {key: Path(str(value)) for key, value in comparison_csv.items()}
    all_view_keys = expected_view_keys & set(md_paths) & set(csv_paths)
    json_path = primary_report.with_suffix(".json")
    comparison_rows = {}
    if json_path.exists():
        payload_json = _read_json(json_path)
        comparison_views = payload_json.get("comparison_views", {})
        comparison_views = comparison_views if isinstance(comparison_views, dict) else {}
        for key in expected_view_keys:
            rows = comparison_views.get(key, [])
            comparison_rows[key] = len(rows) if isinstance(rows, list) else 0
    missing_paths = []
    for view_key in expected_view_keys:
        if view_key not in md_paths or not md_paths[view_key].exists():
            missing_paths.append(str(md_paths.get(view_key, primary_report.parent / f"{view_key}.md")))
        if view_key not in csv_paths or not csv_paths[view_key].exists():
            missing_paths.append(str(csv_paths.get(view_key, primary_report.parent / f"{view_key}.csv")))
    if not primary_report.exists():
        missing_paths.append(str(primary_report))
    if not json_path.exists():
        missing_paths.append(str(json_path))

    completeness = "ready" if not missing_paths and len(all_view_keys) == 4 else "missing"
    operational = "ready" if all(comparison_rows.get(key, 0) > 0 for key in expected_view_keys) else "attention"
    return {
        "stem": primary_report.stem,
        "primary_report": str(primary_report),
        "json_report": str(json_path),
        "completeness_status": completeness,
        "operational_status": operational,
        "comparison_rows": comparison_rows,
        "missing_paths": missing_paths,
    }


def _build_taste_os_status(root: Path, *, artifact_bundle: PortfolioArtifactBundle | None = None) -> dict[str, Any]:
    bundle = artifact_bundle or load_portfolio_artifact_bundle(root / "outputs")
    doc_paths = [
        root / "docs/personal_taste_os.md",
        root / "docs/taste_os_demo_contract.md",
        root / "docs/taste_os_demo_walkthrough.md",
        root / "docs/taste_os_product_story.md",
    ]
    artifact_paths = [
        bundle.taste_os_showcase_json,
        bundle.taste_os_showcase_md,
        root / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.json",
        root / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.md",
    ]
    showcase_json = artifact_paths[0]
    comparison_json = artifact_paths[2]
    canonical_example_count = 0
    mode_comparison_count = 0
    unique_opening_artists = 0
    review_order: list[str] = []
    if showcase_json.exists():
        payload = bundle.taste_os_showcase_payload or _read_json(showcase_json)
        summary = payload.get("showcase_summary", {})
        summary = summary if isinstance(summary, dict) else {}
        canonical_example_count = int(summary.get("canonical_example_count", 0) or 0)
        mode_comparison_count = int(summary.get("mode_comparison_count", 0) or 0)
        review_order = [str(item) for item in payload.get("review_order", []) if str(item).strip()]
    if comparison_json.exists():
        payload = _read_json(comparison_json)
        rows = payload.get("rows", [])
        rows = rows if isinstance(rows, list) else []
        unique_opening_artists = len(
            {
                str(row.get("top_artist", "")).strip()
                for row in rows
                if isinstance(row, dict) and str(row.get("top_artist", "")).strip()
            }
        )

    completeness = (
        "ready"
        if not _missing_paths(doc_paths + artifact_paths, root)
        and canonical_example_count >= 4
        and mode_comparison_count >= 4
        else "missing"
    )
    operational = "ready" if unique_opening_artists >= 3 and len(review_order) >= 4 else "attention"
    efficiency = "ready" if artifact_paths[0].exists() and artifact_paths[2].exists() else "missing"
    recommended_actions: list[str] = []
    if canonical_example_count < 4 or mode_comparison_count < 4:
        recommended_actions.append("Regenerate the Taste OS showcase pack so the canonical examples and mode comparison are complete.")
    if unique_opening_artists < 3:
        recommended_actions.append("Retune the opening reranker until at least three steady modes open on distinct artists.")

    return {
        "label": "Weeks 1-4",
        "surface": "Taste OS",
        "completeness_status": completeness,
        "operational_status": operational,
        "efficiency_status": efficiency,
        "docs": [_relative(path, root) for path in doc_paths],
        "artifacts": [_relative(path, root) for path in artifact_paths],
        "missing_paths": _missing_paths(doc_paths + artifact_paths, root),
        "artifact_summary": _artifact_summary(artifact_paths),
        "metrics": {
            "canonical_example_count": canonical_example_count,
            "mode_comparison_count": mode_comparison_count,
            "unique_opening_artists": unique_opening_artists,
            "review_order_count": len(review_order),
        },
        "summary": (
            f"Taste OS has `{canonical_example_count}` canonical examples, `{mode_comparison_count}` mode rows, "
            f"and `{unique_opening_artists}` unique steady-mode openings."
        ),
        "recommended_actions": recommended_actions,
        "share_artifacts": [
            _relative(root / "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.md", root),
            _relative(root / "outputs/analysis/taste_os_demo/showcase/taste_os_mode_comparison.md", root),
        ],
    }


def _build_control_room_status(root: Path, *, artifact_bundle: PortfolioArtifactBundle | None = None) -> dict[str, Any]:
    bundle = artifact_bundle or load_portfolio_artifact_bundle(root / "outputs")
    doc_paths = [root / "docs/control_room_operating_rhythm.md"]
    artifact_paths = [
        bundle.control_room_json,
        bundle.control_room_md,
        root / "outputs/analytics/control_room_weekly_summary.json",
        root / "outputs/analytics/control_room_weekly_summary.md",
        root / "outputs/analytics/control_room_triage.json",
        root / "outputs/analytics/control_room_triage.md",
    ]
    high_priority_actions = 0
    operational_high_priority_actions = 0
    strategic_high_priority_actions = 0
    cadence_status = ""
    ops_health_status = ""
    recommended_review_command = ""
    latest_run_id = ""
    async_share_count = 0
    if artifact_paths[0].exists():
        payload = bundle.control_room_payload or _read_json(artifact_paths[0])
        review_actions = payload.get("review_actions", [])
        review_actions = review_actions if isinstance(review_actions, list) else []
        high_priority_actions = sum(
            1
            for row in review_actions
            if isinstance(row, dict) and str(row.get("priority", "")).strip().lower() == "high"
        )
        operating_rhythm = payload.get("operating_rhythm", {})
        operating_rhythm = operating_rhythm if isinstance(operating_rhythm, dict) else {}
        cadence_status = str(operating_rhythm.get("overall_status", "")).strip()
        ops_health = payload.get("ops_health", {})
        ops_health = ops_health if isinstance(ops_health, dict) else {}
        ops_health_status = str(ops_health.get("status", "")).strip()
        operational_high_priority_actions = int(ops_health.get("operational_high_priority_count", 0) or 0)
        strategic_high_priority_actions = int(ops_health.get("strategic_high_priority_count", 0) or 0)
        recommended_review_command = str(operating_rhythm.get("recommended_review_command", "")).strip()
        latest_run = payload.get("latest_run", {})
        latest_run = latest_run if isinstance(latest_run, dict) else {}
        latest_run_id = str(latest_run.get("run_id", "")).strip()
        async_handoff = payload.get("async_handoff", {})
        async_handoff = async_handoff if isinstance(async_handoff, dict) else {}
        share_artifacts = async_handoff.get("share_artifacts", [])
        async_share_count = len(share_artifacts) if isinstance(share_artifacts, list) else 0

    completeness = "ready" if not _missing_paths(doc_paths + artifact_paths, root) else "missing"
    if ops_health_status:
        operational = "ready" if ops_health_status == "healthy" else "attention"
    else:
        operational = "ready" if high_priority_actions == 0 and cadence_status == "healthy" else "attention"
    efficiency = "ready" if async_share_count >= 3 else "attention"
    recommended_actions: list[str] = []
    if operational_high_priority_actions > 0:
        recommended_actions.append("Clear the cadence or instrumentation blockers before treating the ops lane as healthy.")
    if cadence_status and cadence_status != "healthy":
        recommended_actions.append("Restore the recurring fast/full cadence so the control room reflects a current operating rhythm.")
    if strategic_high_priority_actions > 0:
        recommended_actions.append("Strategic safety findings are still open, but they should not be confused with core ops-health blockers.")
    if async_share_count < 3:
        recommended_actions.append("Keep the weekly summary and triage artifacts fresh so async review stays lightweight.")

    return {
        "label": "Weeks 5-6",
        "surface": "Control Room",
        "completeness_status": completeness,
        "operational_status": operational,
        "efficiency_status": efficiency,
        "docs": [_relative(path, root) for path in doc_paths],
        "artifacts": [_relative(path, root) for path in artifact_paths],
        "missing_paths": _missing_paths(doc_paths + artifact_paths, root),
        "artifact_summary": _artifact_summary(artifact_paths),
        "metrics": {
            "high_priority_review_actions": high_priority_actions,
            "operational_high_priority_review_actions": operational_high_priority_actions,
            "strategic_high_priority_review_actions": strategic_high_priority_actions,
            "cadence_status": cadence_status,
            "ops_health_status": ops_health_status,
            "async_share_artifact_count": async_share_count,
            "latest_run_id": latest_run_id,
        },
        "summary": (
            f"Control room covers run `{latest_run_id or 'n/a'}` with cadence status `{cadence_status or 'unknown'}`, "
            f"ops-health `{ops_health_status or 'unknown'}`, and `{strategic_high_priority_actions}` strategic high-priority finding(s)."
        ),
        "recommended_actions": recommended_actions,
        "share_artifacts": [
            _relative(root / "outputs/analytics/control_room.md", root),
            _relative(root / "outputs/analytics/control_room_weekly_summary.md", root),
            _relative(root / "outputs/analytics/control_room_triage.md", root),
        ],
        "recommended_review_command": recommended_review_command,
    }


def _build_creator_status(root: Path, *, artifact_bundle: PortfolioArtifactBundle | None = None) -> dict[str, Any]:
    bundle = artifact_bundle or load_portfolio_artifact_bundle(root / "outputs")
    doc_paths = [root / "docs/creator_label_intelligence_brief.md"]
    manifest_paths = list(bundle.creator_manifest_paths)
    manifest_statuses = [_creator_manifest_status(path) for path in manifest_paths]
    complete_families = sum(1 for row in manifest_statuses if row["completeness_status"] == "ready")
    operational_families = sum(1 for row in manifest_statuses if row["operational_status"] == "ready")
    completeness = (
        "ready"
        if doc_paths[0].exists() and len(manifest_paths) >= 3 and complete_families == len(manifest_paths)
        else "missing"
    )
    operational = "ready" if len(manifest_paths) >= 3 and operational_families == len(manifest_paths) else "attention"
    efficiency = "ready" if len(manifest_paths) >= 3 else "attention"
    recommended_actions: list[str] = []
    if len(manifest_paths) < 3:
        recommended_actions.append("Generate at least three creator report families across different seed styles.")
    if operational_families < len(manifest_paths):
        recommended_actions.append("Fill any empty comparison views so ranking, scene, seed, and scene-vs-seed all stay legible.")

    share_artifacts = [row["primary_report"] for row in manifest_statuses[:3]]
    return {
        "label": "Weeks 7-8",
        "surface": "Creator Intelligence",
        "completeness_status": completeness,
        "operational_status": operational,
        "efficiency_status": efficiency,
        "docs": [_relative(path, root) for path in doc_paths],
        "artifacts": [_relative(path, root) for path in manifest_paths],
        "missing_paths": _missing_paths(doc_paths, root),
        "artifact_summary": _artifact_summary(manifest_paths),
        "metrics": {
            "report_family_count": len(manifest_paths),
            "complete_report_family_count": complete_families,
            "operational_report_family_count": operational_families,
        },
        "summary": (
            f"Creator intelligence has `{len(manifest_paths)}` report families, with `{complete_families}` complete "
            f"and `{operational_families}` carrying non-empty comparison views."
        ),
        "recommended_actions": recommended_actions,
        "share_artifacts": [str(Path(path).resolve()) for path in share_artifacts],
        "report_families": manifest_statuses,
    }


def _build_platform_research_status(root: Path, *, artifact_bundle: PortfolioArtifactBundle) -> dict[str, Any]:
    doc_paths = [
        root / "docs/recommender_safety_platform.md",
        root / "docs/benchmark_contract.md",
        root / "docs/publication_outline.md",
    ]
    artifact_paths = [
        artifact_bundle.research_claims_json,
        artifact_bundle.research_claims_md,
    ]
    if artifact_bundle.benchmark_manifest_json is not None:
        artifact_paths.append(artifact_bundle.benchmark_manifest_json)
    if artifact_bundle.benchmark_manifest_md is not None:
        artifact_paths.append(artifact_bundle.benchmark_manifest_md)

    claims_payload = artifact_bundle.research_claims_payload
    primary_claim = claims_payload.get("primary_claim", {})
    primary_claim = primary_claim if isinstance(primary_claim, dict) else {}
    backup_claim = claims_payload.get("backup_claim", {})
    backup_claim = backup_claim if isinstance(backup_claim, dict) else {}
    primary_status = str(primary_claim.get("status", "")).strip()
    backup_status = str(backup_claim.get("status", "")).strip()
    benchmark_ready = bool(artifact_bundle.benchmark_manifest_payload.get("comparison_ready"))
    believable = bool(claims_payload.get("believable_submission_path"))

    completeness = "ready" if not _missing_paths(doc_paths + artifact_paths, root) else "missing"
    operational = (
        "ready"
        if believable and primary_status in {"analysis_ready", "submission_candidate"}
        else "attention"
    )
    efficiency = "ready" if benchmark_ready else "attention"
    recommended_actions: list[str] = []
    if not benchmark_ready:
        recommended_actions.append("Finish the repeated-seed benchmark lock so the safety and research branch is comparison-ready.")
    if not believable:
        recommended_actions.append("Strengthen the claim pack until the primary and backup claims form a believable submission path.")

    return {
        "label": "Weeks 9-11",
        "surface": "Safety and Research Platform",
        "completeness_status": completeness,
        "operational_status": operational,
        "efficiency_status": efficiency,
        "docs": [_relative(path, root) for path in doc_paths],
        "artifacts": [_relative(path, root) for path in artifact_paths],
        "missing_paths": _missing_paths(doc_paths + artifact_paths, root),
        "artifact_summary": _artifact_summary(artifact_paths),
        "metrics": {
            "primary_claim_status": primary_status,
            "backup_claim_status": backup_status,
            "benchmark_comparison_ready": benchmark_ready,
            "believable_submission_path": believable,
        },
        "summary": (
            f"Safety and research currently carry primary claim status `{primary_status or 'unknown'}`, "
            f"backup status `{backup_status or 'unknown'}`, and benchmark comparison-ready=`{benchmark_ready}`."
        ),
        "recommended_actions": recommended_actions,
        "share_artifacts": [
            _relative(artifact_bundle.research_claims_md, root),
            _relative(artifact_bundle.benchmark_manifest_md, root) if artifact_bundle.benchmark_manifest_md else "",
        ],
    }


def _build_integration_status(root: Path, *, artifact_bundle: PortfolioArtifactBundle) -> dict[str, Any]:
    doc_paths = [
        root / "docs/claim_to_demo.md",
        root / "docs/higher_level_branches.md",
        root / "docs/outward_package.md",
    ]
    artifact_paths = [
        root / "outputs/analysis/claim_to_demo/claim_to_demo.json",
        root / "outputs/analysis/claim_to_demo/claim_to_demo.md",
        root / "outputs/analysis/claim_to_demo/claim_to_demo_talk_track.md",
        root / "outputs/analysis/portfolio_branches/portfolio_branches.json",
        root / "outputs/analysis/portfolio_branches/portfolio_branches.md",
        root / "outputs/analysis/outward_package/outward_package.json",
        root / "outputs/analysis/outward_package/outward_package.md",
        root / "outputs/analysis/outward_package/four_branch_summary.md",
        root / "outputs/analysis/outward_package/safety_research/safety_research_showcase.md",
    ]
    branch_report = build_branch_portfolio_report(root / "outputs", artifact_bundle=artifact_bundle)
    branches = branch_report.get("branches", [])
    branches = branches if isinstance(branches, list) else []
    ready_branch_count = sum(
        1 for branch in branches if isinstance(branch, dict) and str(branch.get("status", "")).startswith("ready")
    )
    completeness = "ready" if not _missing_paths(doc_paths + artifact_paths, root) else "missing"
    operational = "ready" if ready_branch_count == len(branches) and len(branches) == 4 else "attention"
    copied_assets = [
        root / "outputs/analysis/outward_package/flagship/claim_to_demo.md",
        root / "outputs/analysis/outward_package/flagship/claim_to_demo_talk_track.md",
        root / "outputs/analysis/outward_package/taste_os/taste_os_showcase.md",
        root / "outputs/analysis/outward_package/control_room/control_room.md",
        root / "outputs/analysis/outward_package/creator_intelligence/creator_label_intelligence.md",
        root / "outputs/analysis/outward_package/creator_intelligence/scene_seed_view.md",
        root / "outputs/analysis/outward_package/safety_research/research_claims.md",
        root / "outputs/analysis/outward_package/safety_research/benchmark_lock_manifest.md",
    ]
    packaged_asset_count = sum(1 for path in copied_assets if path.exists())
    efficiency = "ready" if packaged_asset_count >= 7 else "attention"
    recommended_actions: list[str] = []
    if ready_branch_count < len(branches):
        recommended_actions.append("Keep tightening the four-branch hierarchy until every branch is at least ready-with-gaps.")
    if packaged_asset_count < len(copied_assets):
        recommended_actions.append("Regenerate the outward-facing package so all copied showcase assets are present.")

    return {
        "label": "Weeks 12-13",
        "surface": "Portfolio Integration",
        "completeness_status": completeness,
        "operational_status": operational,
        "efficiency_status": efficiency,
        "docs": [_relative(path, root) for path in doc_paths],
        "artifacts": [_relative(path, root) for path in artifact_paths],
        "missing_paths": _missing_paths(doc_paths + artifact_paths, root),
        "artifact_summary": _artifact_summary(artifact_paths),
        "metrics": {
            "primary_branch_count": len(branches),
            "ready_branch_count": ready_branch_count,
            "packaged_asset_count": packaged_asset_count,
        },
        "summary": (
            f"Portfolio integration has `{ready_branch_count}` ready branch(es) across `{len(branches)}` primary branches "
            f"and `{packaged_asset_count}` packaged outward-facing assets."
        ),
        "recommended_actions": recommended_actions,
        "share_artifacts": [
            _relative(root / "outputs/analysis/claim_to_demo/claim_to_demo.md", root),
            _relative(root / "outputs/analysis/portfolio_branches/portfolio_branches.md", root),
            _relative(root / "outputs/analysis/outward_package/outward_package.md", root),
            _relative(root / "outputs/analysis/outward_package/four_branch_summary.md", root),
        ],
    }


def build_weeks_1_8_readiness_report(root: Path) -> dict[str, Any]:
    workspace_root = root.resolve()
    outputs_root = workspace_root / "outputs"
    latest_run = _latest_run_dir(outputs_root)
    bundle = load_portfolio_artifact_bundle(outputs_root)
    taste_status = _build_taste_os_status(workspace_root, artifact_bundle=bundle)
    control_status = _build_control_room_status(workspace_root, artifact_bundle=bundle)
    creator_status = _build_creator_status(workspace_root, artifact_bundle=bundle)
    sections = [taste_status, control_status, creator_status]

    completeness_status = _worst_status(*(row["completeness_status"] for row in sections))
    operational_status = _worst_status(*(row["operational_status"] for row in sections))
    efficiency_status = _worst_status(*(row["efficiency_status"] for row in sections))
    move_to_next_phase = completeness_status == "ready" and efficiency_status == "ready"

    next_actions: list[str] = []
    for row in sections:
        for action in row.get("recommended_actions", []):
            if action not in next_actions:
                next_actions.append(action)

    if move_to_next_phase and operational_status != "ready":
        next_actions.insert(0, "Weeks 1-8 are built and reviewable, but resolve the live operational attention items before treating the phase as fully healthy.")

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "outputs_root": str(outputs_root),
        "latest_run_dir": str(latest_run.resolve()) if latest_run is not None else "",
        "latest_run_timestamp": _isoformat(_timestamp(latest_run)) if latest_run is not None else None,
        "weeks_1_8_ready_for_week_9_10": move_to_next_phase,
        "overall": {
            "completeness_status": completeness_status,
            "operational_status": operational_status,
            "efficiency_status": efficiency_status,
        },
        "sections": sections,
        "next_actions": next_actions,
        "fast_review_path": [
            "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.md",
            "outputs/analytics/control_room.md",
            "outputs/analysis/public_spotify/creator_label_intelligence/creator_label_intelligence_tame-impala-arctic-monkeys-phoebe-bridgers.md",
        ],
    }


def build_weeks_1_13_readiness_report(root: Path) -> dict[str, Any]:
    workspace_root = root.resolve()
    outputs_root = workspace_root / "outputs"
    latest_run = _latest_run_dir(outputs_root)
    bundle = load_portfolio_artifact_bundle(outputs_root)

    sections = [
        _build_taste_os_status(workspace_root, artifact_bundle=bundle),
        _build_control_room_status(workspace_root, artifact_bundle=bundle),
        _build_creator_status(workspace_root, artifact_bundle=bundle),
        _build_platform_research_status(workspace_root, artifact_bundle=bundle),
        _build_integration_status(workspace_root, artifact_bundle=bundle),
    ]

    completeness_status = _worst_status(*(row["completeness_status"] for row in sections))
    operational_status = _worst_status(*(row["operational_status"] for row in sections))
    efficiency_status = _worst_status(*(row["efficiency_status"] for row in sections))
    ready_for_day_90 = completeness_status == "ready" and efficiency_status == "ready"

    next_actions: list[str] = []
    for row in sections:
        for action in row.get("recommended_actions", []):
            if action not in next_actions:
                next_actions.append(action)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "outputs_root": str(outputs_root),
        "latest_run_dir": str(latest_run.resolve()) if latest_run is not None else "",
        "latest_run_timestamp": _isoformat(_timestamp(latest_run)) if latest_run is not None else None,
        "weeks_1_13_ready_for_day_90": ready_for_day_90,
        "overall": {
            "completeness_status": completeness_status,
            "operational_status": operational_status,
            "efficiency_status": efficiency_status,
        },
        "sections": sections,
        "next_actions": next_actions,
        "fast_review_path": [
            "outputs/analysis/claim_to_demo/claim_to_demo.md",
            "outputs/analysis/taste_os_demo/showcase/taste_os_showcase.md",
            "outputs/analytics/control_room.md",
            "outputs/analysis/public_spotify/creator_label_intelligence/creator_label_intelligence_tame-impala-arctic-monkeys-phoebe-bridgers.md",
            "outputs/analysis/research_claims/research_claims.md",
            "outputs/analysis/outward_package/outward_package.md",
        ],
    }


def write_weeks_1_8_readiness_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Path]:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(resolved_output_dir / "weeks_1_8_readiness.json", report)
    md_path = resolved_output_dir / "weeks_1_8_readiness.md"

    overall = report.get("overall", {})
    overall = overall if isinstance(overall, dict) else {}
    sections = report.get("sections", [])
    sections = sections if isinstance(sections, list) else []
    markdown_lines = [
        "# Weeks 1-8 Readiness",
        "",
        f"- Ready for Weeks 9-10: `{bool(report.get('weeks_1_8_ready_for_week_9_10', False))}`",
        f"- Completeness: `{overall.get('completeness_status', '')}`",
        f"- Operational health: `{overall.get('operational_status', '')}`",
        f"- Efficiency: `{overall.get('efficiency_status', '')}`",
        f"- Latest run dir: `{report.get('latest_run_dir', '')}`",
        "",
        "## Fast Review Path",
        "",
    ]
    for item in report.get("fast_review_path", []):
        markdown_lines.append(f"- `{item}`")
    markdown_lines.extend(["", "## Section Status", ""])
    for section in sections:
        if not isinstance(section, dict):
            continue
        markdown_lines.extend(
            [
                f"### {section.get('label', '')}: {section.get('surface', '')}",
                "",
                f"- Completeness: `{section.get('completeness_status', '')}`",
                f"- Operational health: `{section.get('operational_status', '')}`",
                f"- Efficiency: `{section.get('efficiency_status', '')}`",
                f"- Summary: {section.get('summary', '')}",
            ]
        )
        metrics = section.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        for key, value in metrics.items():
            markdown_lines.append(f"- {key}: `{value}`")
        missing_paths = section.get("missing_paths", [])
        missing_paths = missing_paths if isinstance(missing_paths, list) else []
        if missing_paths:
            markdown_lines.append(f"- Missing: `{', '.join(missing_paths)}`")
        recommended_actions = section.get("recommended_actions", [])
        recommended_actions = recommended_actions if isinstance(recommended_actions, list) else []
        if recommended_actions:
            markdown_lines.append("- Recommended actions:")
            for action in recommended_actions:
                markdown_lines.append(f"  - {action}")
        markdown_lines.append("")
    markdown_lines.extend(["## Next Actions", ""])
    for action in report.get("next_actions", []):
        markdown_lines.append(f"- {action}")

    md_path = write_markdown(md_path, markdown_lines)
    return {"json": json_path, "md": md_path}


def write_weeks_1_13_readiness_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Path]:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(resolved_output_dir / "weeks_1_13_readiness.json", report)
    md_path = resolved_output_dir / "weeks_1_13_readiness.md"

    overall = report.get("overall", {})
    overall = overall if isinstance(overall, dict) else {}
    sections = report.get("sections", [])
    sections = sections if isinstance(sections, list) else []
    markdown_lines = [
        "# Weeks 1-13 Readiness",
        "",
        f"- Ready for Day 90: `{bool(report.get('weeks_1_13_ready_for_day_90', False))}`",
        f"- Completeness: `{overall.get('completeness_status', '')}`",
        f"- Operational health: `{overall.get('operational_status', '')}`",
        f"- Efficiency: `{overall.get('efficiency_status', '')}`",
        f"- Latest run dir: `{report.get('latest_run_dir', '')}`",
        "",
        "## Fast Review Path",
        "",
    ]
    for item in report.get("fast_review_path", []):
        markdown_lines.append(f"- `{item}`")
    markdown_lines.extend(["", "## Section Status", ""])
    for section in sections:
        if not isinstance(section, dict):
            continue
        markdown_lines.extend(
            [
                f"### {section.get('label', '')}: {section.get('surface', '')}",
                "",
                f"- Completeness: `{section.get('completeness_status', '')}`",
                f"- Operational health: `{section.get('operational_status', '')}`",
                f"- Efficiency: `{section.get('efficiency_status', '')}`",
                f"- Summary: {section.get('summary', '')}",
            ]
        )
        metrics = section.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        for key, value in metrics.items():
            markdown_lines.append(f"- {key}: `{value}`")
        recommended_actions = section.get("recommended_actions", [])
        recommended_actions = recommended_actions if isinstance(recommended_actions, list) else []
        if recommended_actions:
            markdown_lines.append("- Recommended actions:")
            for action in recommended_actions:
                markdown_lines.append(f"  - {action}")
        markdown_lines.append("")
    markdown_lines.extend(["## Next Actions", ""])
    for action in report.get("next_actions", []):
        markdown_lines.append(f"- {action}")

    md_path = write_markdown(md_path, markdown_lines)
    return {"json": json_path, "md": md_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.phase_readiness",
        description="Audit roadmap readiness and write a fast report across Taste OS, control room, creator, research, and package surfaces.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analytics",
        help="Directory where the readiness JSON and Markdown reports should be written.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="weeks-1-13",
        choices=("weeks-1-8", "weeks-1-13"),
        help="Whether to audit the first three roadmap phases or the full Week 1-13 stack.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace_root = Path.cwd().resolve()
    if args.scope == "weeks-1-8":
        report = build_weeks_1_8_readiness_report(workspace_root)
        artifacts = write_weeks_1_8_readiness_report(report, output_dir=Path(args.output_dir))
        print(f"weeks_1_8_readiness_json={artifacts['json']}")
        print(f"weeks_1_8_readiness_md={artifacts['md']}")
        print(f"weeks_1_8_ready_for_week_9_10={report['weeks_1_8_ready_for_week_9_10']}")
    else:
        report = build_weeks_1_13_readiness_report(workspace_root)
        artifacts = write_weeks_1_13_readiness_report(report, output_dir=Path(args.output_dir))
        print(f"weeks_1_13_readiness_json={artifacts['json']}")
        print(f"weeks_1_13_readiness_md={artifacts['md']}")
        print(f"weeks_1_13_ready_for_day_90={report['weeks_1_13_ready_for_day_90']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
