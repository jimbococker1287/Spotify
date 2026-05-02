from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .branch_portfolio import build_branch_portfolio_report, write_branch_portfolio_artifacts
from .claim_to_demo import build_claim_to_demo_report, write_claim_to_demo_artifacts
from .day_90_launch import build_day_90_launch_report, write_day_90_launch_artifacts
from .front_door import build_front_door_report, write_front_door_artifacts
from .outward_package import build_outward_package_report, write_outward_package_artifacts
from .phase_readiness import (
    build_weeks_1_14_readiness_report,
    build_weeks_1_16_readiness_report,
    write_weeks_1_14_readiness_report,
    write_weeks_1_16_readiness_report,
)
from .portfolio_artifacts import load_portfolio_artifact_bundle
from .research_artifacts import _write_safety_platform_contract
from .benchmark_contract import describe_canonical_benchmark_contract
from .run_artifacts import write_json, write_markdown
from .show_ready_maintenance import build_show_ready_maintenance_report, write_show_ready_maintenance_report


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _read_manifest(path: Path) -> dict[str, Any]:
    return _coerce_dict(json.loads(path.read_text(encoding="utf-8")))


def _creator_reading_order(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    label_map = {
        "ranking_comparison": "Ranking Comparison",
        "scene_comparison": "Scene Comparison",
        "seed_comparison": "Seed Comparison",
        "scene_seed_comparison": "Scene-Seed Comparison",
        "opportunity_lane_comparison": "Opportunity-Lane Comparison",
        "scene_strategy_watch": "Scene Strategy Watch",
    }
    order = [
        "ranking_comparison",
        "scene_comparison",
        "seed_comparison",
        "scene_seed_comparison",
        "opportunity_lane_comparison",
        "scene_strategy_watch",
    ]
    available = []
    comparison_md = _coerce_dict(manifest.get("comparison_view_markdown"))
    brief_md = _coerce_dict(manifest.get("brief_view_markdown"))
    merged = {**comparison_md, **brief_md}
    for key in order:
        raw_path = str(merged.get(key, "")).strip()
        if raw_path:
            available.append((key, label_map.get(key, key.replace("_", " ").title())))
    return available


def _backfill_creator_report_family_index(manifest_path: Path) -> dict[str, object]:
    manifest = _read_manifest(manifest_path)
    current_index = str(manifest.get("artifact_index_markdown", "")).strip()
    if current_index:
        current_path = Path(current_index).expanduser()
        if current_path.exists():
            return {
                "manifest": str(manifest_path.resolve()),
                "status": "already_present",
                "artifact_index_markdown": str(current_path.resolve()),
            }

    stem = manifest_path.name.removesuffix("_report_family.json")
    report_dir = manifest_path.parent
    primary_report = str(manifest.get("primary_report", "")).strip()
    reading_order = _creator_reading_order(manifest)
    lines = [
        "# Creator Report-Family Index",
        "",
        f"- Manifest: `{manifest_path.resolve()}`",
        f"- Primary report: `{primary_report}`",
        "",
        "## Reading Order",
        "",
    ]
    for key, label in reading_order:
        source_group = _coerce_dict(manifest.get("comparison_view_markdown"))
        if key not in source_group:
            source_group = _coerce_dict(manifest.get("brief_view_markdown"))
        lines.append(f"- `{label}`: `{source_group.get(key, '')}`")
    lines.extend(["", "## Packaging Mode", "", "- Backfilled index for legacy report-family manifests so the Day-90 package can point at a shareable reading order."])
    md_path = report_dir / f"{stem}_report_family.md"
    write_markdown(md_path, lines)
    manifest["artifact_index_markdown"] = str(md_path.resolve())
    manifest["backfilled_artifact_index_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": str(manifest_path.resolve()),
        "status": "backfilled",
        "artifact_index_markdown": str(md_path.resolve()),
    }


def _backfill_safety_platform_contract(output_root: Path) -> dict[str, object]:
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
    research_payload = bundle.research_claims_payload
    run_meta = _coerce_dict(research_payload.get("run"))
    run_id = str(run_meta.get("run_id", "")).strip()
    profile = str(run_meta.get("profile", "")).strip()
    if not run_id:
        return {
            "status": "skipped",
            "reason": "research_run_missing",
        }
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "safety_platform_contract.json"
    md_path = run_dir / "safety_platform_contract.md"
    if json_path.exists() and md_path.exists():
        return {
            "status": "already_present",
            "run_id": run_id,
            "json_path": str(json_path.resolve()),
            "md_path": str(md_path.resolve()),
        }
    contract = describe_canonical_benchmark_contract()
    paths = _write_safety_platform_contract(
        output_dir=run_dir,
        run_id=run_id,
        profile=profile,
        contract=contract,
    )
    return {
        "status": "backfilled",
        "run_id": run_id,
        "json_path": str(paths[0].resolve()),
        "md_path": str(paths[1].resolve()),
    }


def backfill_show_ready_artifacts(
    output_dir: Path | str = "outputs",
    *,
    refresh_artifacts: bool = True,
) -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)

    creator_results = []
    for manifest_path in bundle.creator_manifest_paths:
        creator_results.append(_backfill_creator_report_family_index(manifest_path))

    safety_result = _backfill_safety_platform_contract(output_root)
    refreshed = {}
    if refresh_artifacts:
        bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
        refreshed["branch_portfolio"] = {
            key: str(path)
            for key, path in write_branch_portfolio_artifacts(
                build_branch_portfolio_report(output_root, artifact_bundle=bundle),
                output_dir=output_root,
            ).items()
        }
        refreshed["claim_to_demo"] = {
            key: str(path)
            for key, path in write_claim_to_demo_artifacts(
                build_claim_to_demo_report(output_root),
                output_dir=output_root,
            ).items()
        }
        refreshed["front_door"] = {
            key: str(path)
            for key, path in write_front_door_artifacts(
                build_front_door_report(output_root),
                output_dir=output_root,
            ).items()
        }
        refreshed["outward_package"] = {
            key: str(path)
            for key, path in write_outward_package_artifacts(
                build_outward_package_report(output_root),
                output_dir=output_root,
            ).items()
        }
        refreshed["day_90_launch"] = {
            key: str(path)
            for key, path in write_day_90_launch_artifacts(
                build_day_90_launch_report(output_root),
                output_dir=output_root,
            ).items()
        }
        refreshed["weeks_1_14_readiness"] = {
            key: str(path)
            for key, path in write_weeks_1_14_readiness_report(
                build_weeks_1_14_readiness_report(output_root.parent),
                output_dir=output_root / "analytics",
            ).items()
        }
        refreshed["show_ready_maintenance"] = {
            key: str(path)
            for key, path in write_show_ready_maintenance_report(
                build_show_ready_maintenance_report(output_root),
                output_dir=output_root,
            ).items()
        }
        refreshed["weeks_1_16_readiness"] = {
            key: str(path)
            for key, path in write_weeks_1_16_readiness_report(
                build_weeks_1_16_readiness_report(output_root.parent),
                output_dir=output_root / "analytics",
            ).items()
        }

    creator_backfilled = sum(1 for row in creator_results if str(row.get("status", "")) == "backfilled")
    creator_present = sum(1 for row in creator_results if str(row.get("status", "")) == "already_present")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "creator_report_family_indexes": {
            "manifest_count": len(creator_results),
            "backfilled_count": creator_backfilled,
            "already_present_count": creator_present,
            "results": creator_results,
        },
        "safety_platform_contract": safety_result,
        "refresh_artifacts": refresh_artifacts,
        "refreshed_artifacts": refreshed,
    }


def write_show_ready_backfill_report(
    report: dict[str, object],
    *,
    output_dir: Path | str = "outputs",
) -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    artifact_dir = output_root / "analytics" / "show_ready_backfill"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    json_path = write_json(artifact_dir / "show_ready_backfill.json", report)
    creator = _coerce_dict(report.get("creator_report_family_indexes"))
    safety = _coerce_dict(report.get("safety_platform_contract"))
    lines = [
        "# Show-Ready Backfill",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Output root: `{report.get('output_dir', '')}`",
        f"- Creator manifests: `{creator.get('manifest_count', 0)}`",
        f"- Creator indexes backfilled: `{creator.get('backfilled_count', 0)}`",
        f"- Creator indexes already present: `{creator.get('already_present_count', 0)}`",
        f"- Safety-platform contract status: `{safety.get('status', '')}`",
        "",
        "## Creator Report-Family Results",
        "",
    ]
    for row in _coerce_list(creator.get("results")):
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('status', '')}` `{row.get('manifest', '')}`")
        lines.append(f"Artifact index: `{row.get('artifact_index_markdown', '')}`")
    lines.extend(["", "## Safety Contract", ""])
    lines.append(f"- Status: `{safety.get('status', '')}`")
    if str(safety.get("run_id", "")).strip():
        lines.append(f"- Run ID: `{safety.get('run_id', '')}`")
    if str(safety.get("md_path", "")).strip():
        lines.append(f"- Markdown: `{safety.get('md_path', '')}`")
    if str(safety.get("json_path", "")).strip():
        lines.append(f"- JSON: `{safety.get('json_path', '')}`")
    lines.extend(["", "## Refreshed Artifacts", ""])
    for group, payload in _coerce_dict(report.get("refreshed_artifacts")).items():
        lines.append(f"- `{group}`")
        if isinstance(payload, dict):
            for key, value in payload.items():
                lines.append(f"  - {key}: `{value}`")
    md_path = write_markdown(artifact_dir / "show_ready_backfill.md", lines)
    return {"json": json_path, "md": md_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.show_ready_backfill",
        description="Backfill legacy creator report-family indexes and safety-platform contracts, then refresh show-ready artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root outputs directory containing analysis, analytics, history, and runs.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Backfill artifacts without regenerating the dependent Day-90 package surfaces.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = backfill_show_ready_artifacts(args.output_dir, refresh_artifacts=not args.no_refresh)
    paths = write_show_ready_backfill_report(report, output_dir=args.output_dir)
    creator = _coerce_dict(report.get("creator_report_family_indexes"))
    safety = _coerce_dict(report.get("safety_platform_contract"))
    print(f"show_ready_backfill_json={paths['json']}")
    print(f"show_ready_backfill_md={paths['md']}")
    print(f"creator_indexes_backfilled={creator.get('backfilled_count', 0)}")
    print(f"safety_contract_status={safety.get('status', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
