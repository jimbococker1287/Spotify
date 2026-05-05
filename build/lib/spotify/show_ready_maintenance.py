from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .portfolio_artifacts import load_portfolio_artifact_bundle
from .run_artifacts import safe_read_json, write_json, write_markdown


_STATUS_RANK = {"missing": 0, "attention": 1, "ready": 2}


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _timestamp(path: Path | None) -> datetime | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _age_hours(value: datetime | None, *, now: datetime) -> float | None:
    if value is None:
        return None
    return round((now - value).total_seconds() / 3600.0, 2)


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


def _latest_run_dir(outputs_root: Path) -> Path | None:
    runs_dir = outputs_root / "runs"
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    manifest_backed = [path for path in candidates if (path / "run_manifest.json").exists()]
    ranked = manifest_backed or candidates
    return max(ranked, key=lambda path: path.stat().st_mtime)


def _run_anchor_timestamp(run_dir: Path | None) -> datetime | None:
    if run_dir is None:
        return None
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        return _timestamp(manifest_path)
    return _timestamp(run_dir)


def _anchor_alignment_status(
    *,
    outputs_root: Path,
    control_room_payload: dict[str, Any],
    research_claims_payload: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    showcase_json = outputs_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.json"
    showcase_payload = _coerce_dict(safe_read_json(showcase_json, default={}))
    control_room_run_id = str(_coerce_dict(control_room_payload.get("latest_run")).get("run_id", "")).strip()
    research_run_id = str(_coerce_dict(research_claims_payload.get("run")).get("run_id", "")).strip()
    showcase_run_id = str(showcase_payload.get("run_id", "")).strip()

    rows = [
        {
            "branch": "control_room",
            "label": "Control Room",
            "run_id": control_room_run_id,
            "aligned_to_review_anchor": bool(control_room_run_id),
            "status": "ready" if control_room_run_id else "missing",
        },
        {
            "branch": "research_claims",
            "label": "Research Claims",
            "run_id": research_run_id,
            "aligned_to_review_anchor": bool(control_room_run_id and research_run_id and research_run_id == control_room_run_id),
            "status": (
                "ready"
                if control_room_run_id and research_run_id and research_run_id == control_room_run_id
                else "attention"
                if research_run_id
                else "missing"
            ),
        },
        {
            "branch": "taste_os_showcase",
            "label": "Taste OS Showcase",
            "run_id": showcase_run_id,
            "aligned_to_review_anchor": bool(control_room_run_id and showcase_run_id and showcase_run_id == control_room_run_id),
            "status": (
                "ready"
                if control_room_run_id and showcase_run_id and showcase_run_id == control_room_run_id
                else "attention"
                if showcase_json.exists()
                else "missing"
            ),
            "generated_at": showcase_payload.get("generated_at"),
            "path": str(showcase_json.resolve()),
            "modified_at": _isoformat(_timestamp(showcase_json)),
            "age_hours": _age_hours(_timestamp(showcase_json), now=now),
        },
    ]
    aligned_count = sum(1 for row in rows if bool(row.get("aligned_to_review_anchor")))
    status = _worst_status(*(str(row.get("status", "")) for row in rows))
    summary = (
        f"Review anchor is `{control_room_run_id or 'missing'}`; research is "
        f"`{('aligned' if research_run_id == control_room_run_id and control_room_run_id else 'not aligned')}`; "
        f"Taste OS showcase anchor is `{showcase_run_id or 'missing'}`."
    )
    return {
        "status": status,
        "review_anchor_run_id": control_room_run_id,
        "research_run_id": research_run_id,
        "showcase_run_id": showcase_run_id,
        "aligned_branch_count": aligned_count,
        "tracked_branch_count": len(rows),
        "summary": summary,
        "rows": rows,
    }


def _creator_index_coverage_status(*, output_root: Path) -> dict[str, Any]:
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
    manifests = []
    indexed_count = 0
    for manifest_path in bundle.creator_manifest_paths:
        payload = _coerce_dict(safe_read_json(manifest_path, default={}))
        index_path_text = str(payload.get("artifact_index_markdown", "")).strip()
        index_path = Path(index_path_text).expanduser() if index_path_text else None
        index_present = bool(index_path and index_path.exists())
        if index_present:
            indexed_count += 1
        manifests.append(
            {
                "manifest": str(manifest_path.resolve()),
                "primary_report": str(payload.get("primary_report", "")).strip(),
                "artifact_index_markdown": str(index_path.resolve()) if index_path and index_path.exists() else index_path_text,
                "index_present": index_present,
            }
        )
    manifest_count = len(manifests)
    missing_count = manifest_count - indexed_count
    status = "missing" if manifest_count == 0 else "ready" if missing_count == 0 else "attention"
    return {
        "status": status,
        "manifest_count": manifest_count,
        "indexed_count": indexed_count,
        "missing_count": missing_count,
        "summary": (
            f"Creator report families have `{indexed_count}` shareable indexes across `{manifest_count}` manifests."
        ),
        "rows": manifests,
    }


def _safety_platform_contract_status(*, output_root: Path) -> dict[str, Any]:
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
    research_payload = bundle.research_claims_payload
    run_meta = _coerce_dict(research_payload.get("run"))
    run_id = str(run_meta.get("run_id", "")).strip()
    json_path = bundle.safety_platform_contract_json
    md_path = bundle.safety_platform_contract_md
    present = bool(json_path and json_path.exists() and md_path and md_path.exists())
    return {
        "status": "ready" if present else "attention" if run_id else "missing",
        "run_id": run_id,
        "present": present,
        "json_path": str(json_path.resolve()) if json_path and json_path.exists() else "",
        "md_path": str(md_path.resolve()) if md_path and md_path.exists() else "",
        "summary": (
            f"Safety-platform contract is `{('published' if present else 'missing')}` for research run `{run_id or 'unknown'}`."
        ),
    }


def _canonical_artifact_freshness_status(
    *,
    outputs_root: Path,
    latest_run: Path | None,
    now: datetime,
) -> dict[str, Any]:
    day_90_launch_json = outputs_root / "analysis" / "day_90_launch" / "day_90_launch.json"
    launch_payload = _coerce_dict(safe_read_json(day_90_launch_json, default={}))
    canonical_rows = [row for row in _coerce_list(launch_payload.get("canonical_artifacts")) if isinstance(row, dict)]
    latest_run_ts = _run_anchor_timestamp(latest_run)
    package_refresh_ts = _timestamp(day_90_launch_json)
    rows = []
    ready_count = 0
    stale_count = 0
    missing_count = 0
    for row in canonical_rows:
        source_path_text = str(row.get("copied_artifact", "")).strip() or str(row.get("artifact", "")).strip()
        path = Path(source_path_text).expanduser() if source_path_text else None
        exists = bool(path and path.exists())
        modified_at = _timestamp(path)
        stale = bool(exists and latest_run_ts and package_refresh_ts and package_refresh_ts < latest_run_ts)
        status = "ready" if exists and not stale else "attention" if exists else "missing"
        if status == "ready":
            ready_count += 1
        elif status == "attention":
            stale_count += 1
        else:
            missing_count += 1
        rows.append(
            {
                "key": str(row.get("key", "")).strip(),
                "label": str(row.get("label", "")).strip(),
                "path": str(path.resolve()) if path and path.exists() else source_path_text,
                "exists": exists,
                "status": status,
                "modified_at": _isoformat(modified_at),
                "age_hours": _age_hours(modified_at, now=now),
                "older_than_latest_run": stale,
            }
        )
    status = "missing" if not canonical_rows or missing_count > 0 else "attention" if stale_count > 0 else "ready"
    return {
        "status": status,
        "release_status": str(launch_payload.get("release_status", "")).strip(),
        "latest_run_timestamp": _isoformat(latest_run_ts),
        "package_refresh_timestamp": _isoformat(package_refresh_ts),
        "canonical_artifact_count": len(canonical_rows),
        "ready_count": ready_count,
        "stale_count": stale_count,
        "missing_count": missing_count,
        "summary": (
            f"Canonical launch artifacts have `{ready_count}` ready, `{stale_count}` stale, and `{missing_count}` missing copies."
        ),
        "rows": rows,
    }


def _cadence_status(*, control_room_payload: dict[str, Any]) -> dict[str, Any]:
    rhythm = _coerce_dict(control_room_payload.get("operating_rhythm"))
    ops_health = _coerce_dict(control_room_payload.get("ops_health"))
    cadence = str(rhythm.get("overall_status", "")).strip()
    ops_status = str(ops_health.get("status", "")).strip()
    if cadence == "healthy" and ops_status == "healthy":
        status = "ready"
    elif cadence or ops_status:
        status = "attention"
    else:
        status = "missing"
    return {
        "status": status,
        "cadence_status": cadence,
        "ops_health_status": ops_status,
        "recommended_review_command": str(rhythm.get("recommended_review_command", "")).strip(),
        "summary": (
            f"Cadence is `{cadence or 'unknown'}` and ops health is `{ops_status or 'unknown'}`."
        ),
    }


def build_show_ready_maintenance_report(output_dir: Path | str = "outputs") -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    workspace_root = output_root.parent
    now = datetime.now(timezone.utc)
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
    latest_run = _latest_run_dir(output_root)

    anchor_alignment = _anchor_alignment_status(
        outputs_root=output_root,
        control_room_payload=bundle.control_room_payload,
        research_claims_payload=bundle.research_claims_payload,
        now=now,
    )
    creator_index_coverage = _creator_index_coverage_status(output_root=output_root)
    safety_platform_contract = _safety_platform_contract_status(output_root=output_root)
    canonical_artifact_freshness = _canonical_artifact_freshness_status(
        outputs_root=output_root,
        latest_run=latest_run,
        now=now,
    )
    cadence = _cadence_status(control_room_payload=bundle.control_room_payload)

    overall_status = _worst_status(
        str(anchor_alignment.get("status", "")),
        str(creator_index_coverage.get("status", "")),
        str(safety_platform_contract.get("status", "")),
        str(canonical_artifact_freshness.get("status", "")),
        str(cadence.get("status", "")),
    )

    next_actions: list[str] = []
    if int(creator_index_coverage.get("missing_count", 0) or 0) > 0:
        next_actions.append("Run `make show-ready-backfill` so every creator brief ships with a shareable report-family index.")
    if not bool(safety_platform_contract.get("present")):
        next_actions.append("Run `make show-ready-backfill` so the research anchor publishes the reusable safety-platform contract.")
    if str(anchor_alignment.get("status", "")) != "ready":
        next_actions.append("Regenerate the Taste OS showcase on the current review anchor so product, ops, and research artifacts point at one run.")
    if int(canonical_artifact_freshness.get("stale_count", 0) or 0) > 0:
        next_actions.append("Refresh the outward package and Day-90 launch after the latest run so the canonical copies stay current.")
    if str(cadence.get("cadence_status", "")) == "stale":
        next_actions.append("Restore the recurring fast/full cadence so the control room reflects a current operating rhythm.")
    elif str(cadence.get("status", "")) == "attention":
        next_actions.append("Keep the control-room review rhythm healthy so the show-ready package does not drift behind live ops.")

    maintenance_commands = [
        "make show-ready-backfill",
        "make show-ready-maintenance",
        "python -m spotify.phase_readiness --scope weeks-1-16",
        "make day-90-launch",
    ]
    if str(anchor_alignment.get("status", "")) != "ready":
        maintenance_commands.insert(0, "make taste-os-showcase")
    if str(cadence.get("cadence_status", "")) == "stale":
        maintenance_commands.append("make schedule-run MODE=fast")

    summary = (
        f"Show-ready maintenance is `{overall_status}` with anchor alignment `{anchor_alignment.get('status', '')}`, "
        f"creator index coverage `{creator_index_coverage.get('status', '')}`, safety contract `{safety_platform_contract.get('status', '')}`, "
        f"canonical freshness `{canonical_artifact_freshness.get('status', '')}`, and cadence `{cadence.get('status', '')}`."
    )
    return {
        "generated_at": now.isoformat(),
        "output_dir": str(output_root),
        "workspace_root": str(workspace_root),
        "latest_run_dir": str(latest_run.resolve()) if latest_run is not None else "",
        "latest_run_timestamp": _isoformat(_run_anchor_timestamp(latest_run)),
        "overall_status": overall_status,
        "summary": summary,
        "anchor_alignment": anchor_alignment,
        "creator_index_coverage": creator_index_coverage,
        "safety_platform_contract": safety_platform_contract,
        "canonical_artifact_freshness": canonical_artifact_freshness,
        "cadence": cadence,
        "maintenance_commands": maintenance_commands,
        "next_actions": next_actions,
    }


def write_show_ready_maintenance_report(
    report: dict[str, Any],
    *,
    output_dir: Path | str = "outputs",
) -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    artifact_dir = output_root / "analytics" / "show_ready_maintenance"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    json_path = write_json(artifact_dir / "show_ready_maintenance.json", report)

    anchor = _coerce_dict(report.get("anchor_alignment"))
    creator = _coerce_dict(report.get("creator_index_coverage"))
    safety = _coerce_dict(report.get("safety_platform_contract"))
    freshness = _coerce_dict(report.get("canonical_artifact_freshness"))
    cadence = _coerce_dict(report.get("cadence"))

    lines = [
        "# Show-Ready Maintenance",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Latest run dir: `{report.get('latest_run_dir', '')}`",
        f"- Latest run timestamp: `{report.get('latest_run_timestamp', '')}`",
        f"- Overall status: `{report.get('overall_status', '')}`",
        "",
        "## Summary",
        "",
        f"- {report.get('summary', '')}",
        "",
        "## Anchor Alignment",
        "",
        f"- Status: `{anchor.get('status', '')}`",
        f"- Review anchor run: `{anchor.get('review_anchor_run_id', '')}`",
        f"- Research run: `{anchor.get('research_run_id', '')}`",
        f"- Taste OS showcase run: `{anchor.get('showcase_run_id', '')}`",
        f"- Aligned branches: `{anchor.get('aligned_branch_count', 0)}` / `{anchor.get('tracked_branch_count', 0)}`",
        f"- Summary: {anchor.get('summary', '')}",
        "",
        "## Creator Index Coverage",
        "",
        f"- Status: `{creator.get('status', '')}`",
        f"- Indexed manifests: `{creator.get('indexed_count', 0)}` / `{creator.get('manifest_count', 0)}`",
        "",
        "## Safety Platform Contract",
        "",
        f"- Status: `{safety.get('status', '')}`",
        f"- Research run: `{safety.get('run_id', '')}`",
        f"- Present: `{safety.get('present', False)}`",
        "",
        "## Canonical Artifact Freshness",
        "",
        f"- Status: `{freshness.get('status', '')}`",
        f"- Release status: `{freshness.get('release_status', '')}`",
        f"- Package refreshed at: `{freshness.get('package_refresh_timestamp', '')}`",
        f"- Ready copies: `{freshness.get('ready_count', 0)}`",
        f"- Stale copies: `{freshness.get('stale_count', 0)}`",
        f"- Missing copies: `{freshness.get('missing_count', 0)}`",
        "",
        "## Cadence",
        "",
        f"- Status: `{cadence.get('status', '')}`",
        f"- Cadence status: `{cadence.get('cadence_status', '')}`",
        f"- Ops-health status: `{cadence.get('ops_health_status', '')}`",
        f"- Recommended review command: `{cadence.get('recommended_review_command', '')}`",
        "",
        "## Recommended Commands",
        "",
    ]
    for command in _coerce_list(report.get("maintenance_commands")):
        if isinstance(command, str):
            lines.append(f"- `{command}`")
    lines.extend(["", "## Next Actions", ""])
    for action in _coerce_list(report.get("next_actions")):
        if isinstance(action, str):
            lines.append(f"- {action}")
    lines.extend(["", "## Canonical Artifact Rows", ""])
    for row in _coerce_list(freshness.get("rows")):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('label', '')}` [{row.get('status', '')}] `{row.get('path', '')}`"
        )
    md_path = write_markdown(artifact_dir / "show_ready_maintenance.md", lines)
    return {"json": json_path, "md": md_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.show_ready_maintenance",
        description="Audit anchor alignment, canonical artifact freshness, and maintenance coverage for the post-launch package.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root outputs directory containing analysis, analytics, history, and runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_show_ready_maintenance_report(args.output_dir)
    paths = write_show_ready_maintenance_report(report, output_dir=args.output_dir)
    print(f"show_ready_maintenance_json={paths['json']}")
    print(f"show_ready_maintenance_md={paths['md']}")
    print(f"show_ready_overall_status={report['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
