from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .branch_portfolio import build_branch_portfolio_report, write_branch_portfolio_artifacts
from .run_artifacts import safe_read_json, write_json, write_markdown


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _branch_lookup(report: dict[str, object]) -> dict[str, dict[str, object]]:
    branches = _coerce_list(report.get("branches"))
    lookup: dict[str, dict[str, object]] = {}
    for branch in branches:
        if isinstance(branch, dict):
            lookup[str(branch.get("key", ""))] = branch
    return lookup


def _latest_creator_report_family_path(output_root: Path) -> Path | None:
    base_dir = output_root / "analysis" / "public_spotify" / "creator_label_intelligence"
    manifests = sorted(base_dir.glob("*_report_family.json"))
    if not manifests:
        return None
    return max(manifests, key=lambda path: path.stat().st_mtime)


def _latest_benchmark_manifest_paths(output_root: Path) -> tuple[Path | None, Path | None]:
    json_paths = sorted((output_root / "history").glob("benchmark_lock_*_manifest.json"))
    md_paths = sorted((output_root / "history").glob("benchmark_lock_*_manifest.md"))
    json_path = max(json_paths, key=lambda path: path.stat().st_mtime) if json_paths else None
    md_path = max(md_paths, key=lambda path: path.stat().st_mtime) if md_paths else None
    return json_path, md_path


def _copy_if_exists(source: Path | None, destination: Path) -> str:
    if source is None or not source.exists():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination.resolve())


def _build_safety_research_showcase(
    *,
    output_root: Path,
    package_root: Path,
    research_claims_payload: dict[str, object],
) -> Path:
    benchmark_lock = _coerce_dict(research_claims_payload.get("benchmark_lock"))
    primary_claim = _coerce_dict(research_claims_payload.get("primary_claim"))
    backup_claim = _coerce_dict(research_claims_payload.get("backup_claim"))
    missing_checks = _coerce_list(primary_claim.get("missing_checks"))[:3]
    lines = [
        "# Safety And Research Showcase",
        "",
        f"- Reusable API surface: `{(output_root.parent / 'spotify' / 'safety_platform.py').resolve()}`",
        f"- Benchmark contract: `{(output_root.parent / 'docs' / 'benchmark_contract.md').resolve()}`",
        f"- Research outline: `{(output_root.parent / 'docs' / 'publication_outline.md').resolve()}`",
        f"- Primary claim: `{primary_claim.get('key', '')}` [{primary_claim.get('status', '')}]",
        f"- Backup claim: `{backup_claim.get('key', '')}` [{backup_claim.get('status', '')}]",
        f"- Benchmark lock: `{benchmark_lock.get('benchmark_id', '')}` comparison_ready=`{benchmark_lock.get('comparison_ready', False)}`",
        f"- Believable submission path: `{research_claims_payload.get('believable_submission_path', False)}`",
        "",
        "## Why This Branch Matters",
        "",
        f"- {primary_claim.get('summary', '')}",
        f"- {backup_claim.get('summary', '')}",
        "",
        "## Current Gaps",
        "",
    ]
    for item in missing_checks:
        lines.append(f"- {item}")
    lines.extend(["", "## Source Artifacts", ""])
    for path_str in _coerce_list(primary_claim.get("supporting_artifacts"))[:5]:
        lines.append(f"- `{path_str}`")
    showcase_path = package_root / "safety_research" / "safety_research_showcase.md"
    write_markdown(showcase_path, lines)
    return showcase_path


def build_outward_package_report(output_dir: Path | str = "outputs") -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    branch_report = build_branch_portfolio_report(output_root)
    branch_lookup = _branch_lookup(branch_report)

    taste_os_md = output_root / "analysis" / "taste_os_demo" / "showcase" / "taste_os_showcase.md"
    control_room_md = output_root / "analytics" / "control_room.md"

    creator_manifest_path = _latest_creator_report_family_path(output_root)
    creator_manifest = _coerce_dict(safe_read_json(creator_manifest_path, default={})) if creator_manifest_path else {}
    creator_primary_md = (
        Path(str(creator_manifest.get("primary_report", "")).strip()) if creator_manifest.get("primary_report") else None
    )
    creator_views = _coerce_dict(creator_manifest.get("comparison_view_markdown"))
    creator_scene_seed_md = (
        Path(str(creator_views.get("scene_seed_comparison", "")).strip())
        if creator_views.get("scene_seed_comparison")
        else None
    )

    research_claims_json = output_root / "analysis" / "research_claims" / "research_claims.json"
    research_claims_md = output_root / "analysis" / "research_claims" / "research_claims.md"
    research_claims_payload = _coerce_dict(safe_read_json(research_claims_json, default={}))
    benchmark_json, benchmark_md = _latest_benchmark_manifest_paths(output_root)

    selected_artifacts = {
        "taste_os": {
            "label": "Taste OS showcase",
            "source_md": str(taste_os_md.resolve()) if taste_os_md.exists() else "",
            "status": str(branch_lookup.get("taste_os", {}).get("status", "")),
        },
        "control_room": {
            "label": "Control-room review sample",
            "source_md": str(control_room_md.resolve()) if control_room_md.exists() else "",
            "status": str(branch_lookup.get("control_room", {}).get("status", "")),
        },
        "creator_intelligence": {
            "label": "Creator-intelligence strategy brief",
            "source_md": str(creator_primary_md.resolve()) if creator_primary_md and creator_primary_md.exists() else "",
            "supporting_md": (
                str(creator_scene_seed_md.resolve()) if creator_scene_seed_md and creator_scene_seed_md.exists() else ""
            ),
            "status": str(branch_lookup.get("creator_intelligence", {}).get("status", "")),
        },
        "safety_research": {
            "label": "Safety and research showcase",
            "source_md": str(research_claims_md.resolve()) if research_claims_md.exists() else "",
            "benchmark_md": str(benchmark_md.resolve()) if benchmark_md and benchmark_md.exists() else "",
            "status": str(branch_lookup.get("safety_research", {}).get("status", "")),
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "four_branch_summary": [
            "Taste OS is the product demo branch.",
            "Control Room is the operating-review branch.",
            "Creator / Label Intelligence is the external strategy branch.",
            "Safety and Research Platform is the infrastructure and evidence branch.",
        ],
        "selected_artifacts": selected_artifacts,
        "branch_report": branch_report,
        "research_claims": research_claims_payload,
    }


def write_outward_package_artifacts(report: dict[str, object], *, output_dir: Path | str = "outputs") -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    package_root = output_root / "analysis" / "outward_package"
    package_root.mkdir(parents=True, exist_ok=True)

    branch_paths = write_branch_portfolio_artifacts(_coerce_dict(report.get("branch_report")), output_dir=output_root)

    selected_artifacts = _coerce_dict(report.get("selected_artifacts"))
    copied_artifacts = {
        "taste_os_md": _copy_if_exists(
            Path(str(_coerce_dict(selected_artifacts.get("taste_os")).get("source_md", "")))
            if _coerce_dict(selected_artifacts.get("taste_os")).get("source_md")
            else None,
            package_root / "taste_os" / "taste_os_showcase.md",
        ),
        "control_room_md": _copy_if_exists(
            Path(str(_coerce_dict(selected_artifacts.get("control_room")).get("source_md", "")))
            if _coerce_dict(selected_artifacts.get("control_room")).get("source_md")
            else None,
            package_root / "control_room" / "control_room.md",
        ),
        "creator_primary_md": _copy_if_exists(
            Path(str(_coerce_dict(selected_artifacts.get("creator_intelligence")).get("source_md", "")))
            if _coerce_dict(selected_artifacts.get("creator_intelligence")).get("source_md")
            else None,
            package_root / "creator_intelligence" / "creator_label_intelligence.md",
        ),
        "creator_scene_seed_md": _copy_if_exists(
            Path(str(_coerce_dict(selected_artifacts.get("creator_intelligence")).get("supporting_md", "")))
            if _coerce_dict(selected_artifacts.get("creator_intelligence")).get("supporting_md")
            else None,
            package_root / "creator_intelligence" / "scene_seed_view.md",
        ),
        "research_claims_md": _copy_if_exists(
            Path(str(_coerce_dict(selected_artifacts.get("safety_research")).get("source_md", "")))
            if _coerce_dict(selected_artifacts.get("safety_research")).get("source_md")
            else None,
            package_root / "safety_research" / "research_claims.md",
        ),
        "benchmark_manifest_md": _copy_if_exists(
            Path(str(_coerce_dict(selected_artifacts.get("safety_research")).get("benchmark_md", "")))
            if _coerce_dict(selected_artifacts.get("safety_research")).get("benchmark_md")
            else None,
            package_root / "safety_research" / "benchmark_lock_manifest.md",
        ),
    }

    safety_showcase_path = _build_safety_research_showcase(
        output_root=output_root,
        package_root=package_root,
        research_claims_payload=_coerce_dict(report.get("research_claims")),
    )

    four_branch_summary_md = package_root / "four_branch_summary.md"
    summary_lines = [
        "# Four-Branch Summary",
        "",
    ]
    branch_report = _coerce_dict(report.get("branch_report"))
    for branch in _coerce_list(branch_report.get("branches")):
        if not isinstance(branch, dict):
            continue
        summary_lines.extend(
            [
                f"## {branch.get('label', '')}",
                "",
                f"- Audience: {branch.get('audience', '')}",
                f"- Success metric: {branch.get('success_metric', '')}",
                f"- Status: `{branch.get('status', '')}`",
                f"- Live signal: {branch.get('live_signal', '')}",
                "",
            ]
        )
    write_markdown(four_branch_summary_md, summary_lines)

    outward_payload = {
        "generated_at": report.get("generated_at"),
        "output_dir": report.get("output_dir"),
        "four_branch_summary": report.get("four_branch_summary", []),
        "selected_artifacts": selected_artifacts,
        "copied_artifacts": copied_artifacts,
        "generated_docs": {
            "branch_portfolio_json": str(branch_paths["json"]),
            "branch_portfolio_md": str(branch_paths["md"]),
            "four_branch_summary_md": str(four_branch_summary_md.resolve()),
            "safety_research_showcase_md": str(safety_showcase_path.resolve()),
        },
    }
    json_path = write_json(package_root / "outward_package.json", outward_payload)

    lines = [
        "# Outward-Facing Package",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Output root: `{report.get('output_dir', '')}`",
        "",
        "## Show The Repo In This Order",
        "",
    ]
    for item in _coerce_list(report.get("four_branch_summary")):
        lines.append(f"- {item}")
    lines.extend(["", "## Finalized Package Assets", ""])
    lines.append(f"- Taste OS showcase: `{copied_artifacts['taste_os_md']}`")
    lines.append(f"- Control-room sample: `{copied_artifacts['control_room_md']}`")
    lines.append(f"- Creator brief: `{copied_artifacts['creator_primary_md']}`")
    lines.append(f"- Creator scene-vs-seed view: `{copied_artifacts['creator_scene_seed_md']}`")
    lines.append(f"- Research claims: `{copied_artifacts['research_claims_md']}`")
    lines.append(f"- Benchmark manifest: `{copied_artifacts['benchmark_manifest_md']}`")
    lines.append(f"- Safety showcase: `{safety_showcase_path.resolve()}`")
    lines.append("")
    lines.extend(["## Supporting Summaries", ""])
    lines.append(f"- Branch portfolio: `{branch_paths['md']}`")
    lines.append(f"- Four-branch summary: `{four_branch_summary_md.resolve()}`")
    md_path = write_markdown(package_root / "outward_package.md", lines)
    return {
        "json": json_path,
        "md": md_path,
        "four_branch_summary_md": four_branch_summary_md,
        "safety_research_showcase_md": safety_showcase_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Week 13 outward-facing package from existing repo artifacts.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root outputs directory that contains analytics, history, and analysis artifacts.",
    )
    parser.add_argument(
        "--stdout-format",
        type=str,
        default="summary",
        choices=("summary", "json"),
        help="Whether to print a short summary or the full JSON payload to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_outward_package_report(args.output_dir)
    paths = write_outward_package_artifacts(report, output_dir=args.output_dir)
    if args.stdout_format == "json":
        import json

        print(json.dumps(report, indent=2))
    else:
        print(f"outward_package_json={paths['json']}")
        print(f"outward_package_md={paths['md']}")
        print(f"four_branch_summary_md={paths['four_branch_summary_md']}")
        print(f"safety_research_showcase_md={paths['safety_research_showcase_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
