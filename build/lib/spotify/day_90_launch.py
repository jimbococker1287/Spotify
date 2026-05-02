from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .branch_portfolio import build_branch_portfolio_report
from .claim_to_demo import build_claim_to_demo_report
from .front_door import build_front_door_report, write_front_door_artifacts
from .outward_package import build_outward_package_report, write_outward_package_artifacts
from .portfolio_artifacts import load_portfolio_artifact_bundle
from .run_artifacts import copy_file_if_changed, write_json, write_markdown


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _path_text(path: Path | None) -> str:
    return str(path.resolve()) if path is not None and path.exists() else ""


def _ensure_path(raw_path: object) -> Path | None:
    path_text = str(raw_path).strip()
    if not path_text:
        return None
    return Path(path_text).expanduser().resolve()


def _existing_count(paths: list[Path | None]) -> int:
    return sum(1 for path in paths if path is not None and path.exists())


def _count_unique_openings(showcase_payload: dict[str, object]) -> int:
    rows = _coerce_list(_coerce_dict(showcase_payload.get("mode_comparison")).get("rows"))
    return len(
        {
            str(row.get("top_artist", "")).strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("top_artist", "")).strip()
        }
    )


def _status(passed: bool, *, attention: bool = False) -> str:
    if passed:
        return "pass"
    if attention:
        return "attention"
    return "fail"


def _canonical_artifact_entries(
    *,
    output_root: Path,
    bundle,
    branch_report: dict[str, object],
    claim_report: dict[str, object],
    front_door_report: dict[str, object],
) -> list[dict[str, object]]:
    branch_lookup = {
        str(branch.get("key", "")).strip(): branch
        for branch in _coerce_list(branch_report.get("branches"))
        if isinstance(branch, dict)
    }
    showcase_payload = bundle.taste_os_showcase_payload
    showcase_summary = _coerce_dict(showcase_payload.get("showcase_summary"))
    canonical_count = int(showcase_summary.get("canonical_example_count", 0) or 0)
    review_order_count = len([item for item in _coerce_list(showcase_payload.get("review_order")) if str(item).strip()])
    unique_openings = _count_unique_openings(showcase_payload)

    creator_primary = bundle.creator_primary_report_path.resolve() if bundle.creator_primary_report_path and bundle.creator_primary_report_path.exists() else None
    creator_strategy = None
    strategy_candidate = bundle.creator_brief_markdown_paths.get("scene_strategy_watch")
    if strategy_candidate and strategy_candidate.exists():
        creator_strategy = strategy_candidate.resolve()
    creator_supporting = None
    for key in ("opportunity_lane_comparison", "scene_seed_comparison", "scene_comparison"):
        candidate = bundle.creator_comparison_markdown_paths.get(key)
        if candidate and candidate.exists():
            creator_supporting = candidate.resolve()
            break
    creator_anchor = creator_strategy or creator_supporting or creator_primary or bundle.creator_report_family_md_path
    safety_anchor = output_root / "analysis" / "outward_package" / "safety_research" / "safety_research_showcase.md"
    benchmark_ready = bool(_coerce_dict(bundle.research_claims_payload.get("benchmark_lock")).get("comparison_ready"))
    submission_readiness = _coerce_dict(bundle.research_claims_payload.get("submission_readiness"))
    primary_claim = _coerce_dict(bundle.research_claims_payload.get("primary_claim"))
    control_room_ready = bool(
        bundle.control_room_md.exists()
        and (output_root / "analytics" / "control_room_weekly_summary.md").exists()
        and (output_root / "analytics" / "control_room_triage.md").exists()
    )

    front_door_path = output_root / "analysis" / "front_door" / "index.html"
    claim_to_demo_path = output_root / "analysis" / "claim_to_demo" / "claim_to_demo.md"

    return [
        {
            "key": "front_door",
            "label": "Front Door",
            "branch": "portfolio_integration",
            "audience": "External reviewers who need the repo to make sense in five minutes.",
            "artifact": str(front_door_path.resolve()),
            "status": "ready" if len(_coerce_list(front_door_report.get("review_sequence"))) >= 4 else "attention",
            "why_it_exists": "This is the open-first landing page that turns the repo into one clear story.",
        },
        {
            "key": "taste_os_demo",
            "label": "Canonical Taste OS Demo",
            "branch": "taste_os",
            "audience": str(_coerce_dict(branch_lookup.get("taste_os")).get("audience", "")),
            "artifact": str(bundle.taste_os_showcase_md.resolve()),
            "status": "ready" if canonical_count >= 4 and review_order_count >= 4 and unique_openings >= 3 else "attention",
            "why_it_exists": "This is the product demo that proves the modes and adaptive route actually feel different.",
        },
        {
            "key": "control_room_review",
            "label": "Canonical Control-Room Review",
            "branch": "control_room",
            "audience": str(_coerce_dict(branch_lookup.get("control_room")).get("audience", "")),
            "artifact": str(bundle.control_room_md.resolve()),
            "status": "ready" if control_room_ready else "attention",
            "why_it_exists": "This is the recurring operator sample that shows the system is reviewable after a run.",
        },
        {
            "key": "creator_intelligence_sample",
            "label": "Canonical Creator-Intelligence Sample",
            "branch": "creator_intelligence",
            "audience": str(_coerce_dict(branch_lookup.get("creator_intelligence")).get("audience", "")),
            "artifact": _path_text(creator_anchor),
            "status": "ready" if creator_anchor is not None and creator_anchor.exists() else "attention",
            "why_it_exists": "This is the strategy surface that proves the same taste graph can power a second audience-facing product line.",
        },
        {
            "key": "safety_research_showcase",
            "label": "Canonical Safety / Research Showcase",
            "branch": "safety_research",
            "audience": str(_coerce_dict(branch_lookup.get("safety_research")).get("audience", "")),
            "artifact": str(safety_anchor.resolve()),
            "status": (
                "ready"
                if benchmark_ready
                and bool(submission_readiness.get("status"))
                and primary_claim
                and bundle.safety_platform_contract_md is not None
                else "attention"
            ),
            "why_it_exists": "This is the infrastructure-and-evidence sample that makes the repo defensible as a platform and research system.",
        },
        {
            "key": "claim_to_demo_bridge",
            "label": "Canonical Claim-To-Demo Bridge",
            "branch": "portfolio_integration",
            "audience": "Reviewers who want one artifact that crosses product, creator, ops, and research.",
            "artifact": str(claim_to_demo_path.resolve()),
            "status": "ready" if claim_to_demo_path.exists() else "attention",
            "why_it_exists": "This is the cross-branch bridge artifact that turns the four big lanes into one explainable review flow.",
        },
    ]


def _delivery_checklist(
    *,
    output_root: Path,
    bundle,
    branch_report: dict[str, object],
    claim_report: dict[str, object],
    canonical_artifacts: list[dict[str, object]],
) -> list[dict[str, object]]:
    branch_lookup = {
        str(branch.get("key", "")).strip(): branch
        for branch in _coerce_list(branch_report.get("branches"))
        if isinstance(branch, dict)
    }
    showcase_payload = bundle.taste_os_showcase_payload
    showcase_summary = _coerce_dict(showcase_payload.get("showcase_summary"))
    unique_openings = _count_unique_openings(showcase_payload)
    taste_ready = int(showcase_summary.get("canonical_example_count", 0) or 0) >= 4 and unique_openings >= 3

    control_room_sample_paths = [
        bundle.control_room_md,
        output_root / "analytics" / "control_room_weekly_summary.md",
        output_root / "analytics" / "control_room_triage.md",
    ]
    control_room_sample_ready = _existing_count(control_room_sample_paths) == len(control_room_sample_paths)
    creator_ready = str(_coerce_dict(branch_lookup.get("creator_intelligence")).get("status", "")).startswith("ready")
    safety_paths = [
        bundle.research_claims_md,
        bundle.safety_platform_contract_md,
        bundle.research_claim_support_md,
        bundle.research_submission_readiness_md,
        bundle.benchmark_manifest_md,
    ]
    safety_ready = _existing_count(safety_paths) >= 4
    benchmark_ready = bool(_coerce_dict(bundle.research_claims_payload.get("benchmark_lock")).get("comparison_ready"))
    primary_claim = _coerce_dict(bundle.research_claims_payload.get("primary_claim"))
    primary_status = str(primary_claim.get("status", "")).strip()
    four_ways_ready = sum(
        1 for row in canonical_artifacts if str(row.get("key", "")).strip() in {
            "taste_os_demo",
            "control_room_review",
            "creator_intelligence_sample",
            "safety_research_showcase",
        } and str(row.get("status", "")) == "ready"
    ) >= 4
    claim_to_demo_ready = len(_coerce_list(claim_report.get("review_sequence"))) >= 4 and len(_coerce_list(claim_report.get("branch_alignment"))) == 4

    return [
        {
            "key": "canonical_taste_os_demo_flow",
            "label": "One canonical Taste OS demo flow",
            "status": _status(taste_ready, attention=bundle.taste_os_showcase_md.exists()),
            "detail": (
                f"Taste OS showcase has `{showcase_summary.get('canonical_example_count', 0)}` canonical examples and "
                f"`{unique_openings}` distinct steady-mode openings."
            ),
            "artifact": str(bundle.taste_os_showcase_md.resolve()),
        },
        {
            "key": "recurring_control_room_review_workflow",
            "label": "One recurring control-room review workflow",
            "status": _status(
                control_room_sample_ready,
                attention=bool(bundle.control_room_md.exists()),
            ),
            "detail": "Control-room sample is only complete when the main markdown, weekly summary, and triage pack all exist together.",
            "artifact": str(bundle.control_room_md.resolve()),
        },
        {
            "key": "creator_intelligence_artifact_family",
            "label": "One creator-intelligence artifact family with polished markdown output",
            "status": _status(
                creator_ready,
                attention=bool(bundle.creator_primary_report_path and bundle.creator_primary_report_path.exists()),
            ),
            "detail": "Creator branch is only done when a primary report, supporting comparisons, and a report-family index are all present.",
            "artifact": _path_text(bundle.creator_report_family_md_path or bundle.creator_primary_report_path),
        },
        {
            "key": "documented_reusable_safety_layer",
            "label": "One clearly documented reusable safety layer",
            "status": _status(
                safety_ready,
                attention=bool(bundle.safety_platform_contract_md and bundle.safety_platform_contract_md.exists()),
            ),
            "detail": "This closes only when the platform contract, claim-support pack, submission readiness, and benchmark manifest all ship together.",
            "artifact": _path_text(bundle.safety_platform_contract_md),
        },
        {
            "key": "benchmark_protocol_and_publishable_claims",
            "label": "One frozen benchmark protocol and a shortlist of publishable claims",
            "status": _status(
                benchmark_ready and primary_status in {"analysis_ready", "submission_candidate"} and claim_to_demo_ready and four_ways_ready,
                attention=bool(bundle.research_claims_md.exists()),
            ),
            "detail": (
                f"Benchmark comparison-ready=`{benchmark_ready}`, primary claim=`{primary_status or 'unknown'}`, "
                f"and four-way showability=`{four_ways_ready}`."
            ),
            "artifact": _path_text(bundle.research_submission_readiness_md or bundle.research_claims_md),
        },
    ]


def build_day_90_launch_report(output_dir: Path | str = "outputs") -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
    branch_report = build_branch_portfolio_report(output_root, artifact_bundle=bundle)
    claim_report = build_claim_to_demo_report(output_root)
    front_door_report = build_front_door_report(output_root)
    outward_package_report = build_outward_package_report(output_root)

    canonical_artifacts = _canonical_artifact_entries(
        output_root=output_root,
        bundle=bundle,
        branch_report=branch_report,
        claim_report=claim_report,
        front_door_report=front_door_report,
    )
    delivery_checklist = _delivery_checklist(
        output_root=output_root,
        bundle=bundle,
        branch_report=branch_report,
        claim_report=claim_report,
        canonical_artifacts=canonical_artifacts,
    )
    pass_count = sum(1 for row in delivery_checklist if str(row.get("status", "")) == "pass")
    attention_count = sum(1 for row in delivery_checklist if str(row.get("status", "")) == "attention")
    fail_count = sum(1 for row in delivery_checklist if str(row.get("status", "")) == "fail")

    if fail_count == 0 and attention_count == 0:
        release_status = "launch_ready"
    elif fail_count == 0:
        release_status = "show_ready_with_notes"
    else:
        release_status = "not_ready"

    honesty_notes: list[str] = []
    for item in _coerce_list(claim_report.get("next_actions"))[:6]:
        if isinstance(item, str) and item not in honesty_notes:
            honesty_notes.append(item)

    launch_story = [
        "Open the front door first so the repo reads as one coherent system instead of a folder tree.",
        "Use the claim-to-demo bridge to move from product proof to creator strategy, operator review, and research evidence.",
        "Then fan out into the four canonical branch artifacts only if the audience wants depth.",
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "title": "Day-90 Launch Package",
        "subtitle": "A canonical showable package for the product, creator, ops, and safety/research branches.",
        "release_status": release_status,
        "ready_to_show": release_status in {"launch_ready", "show_ready_with_notes"},
        "canonical_artifacts": canonical_artifacts,
        "delivery_checklist": delivery_checklist,
        "delivery_summary": {
            "pass_count": pass_count,
            "attention_count": attention_count,
            "fail_count": fail_count,
        },
        "launch_story": launch_story,
        "honesty_notes": honesty_notes,
        "branch_report": branch_report,
        "claim_to_demo_report": claim_report,
        "front_door_report": front_door_report,
        "outward_package_report": outward_package_report,
    }


def write_day_90_launch_artifacts(
    report: dict[str, object],
    *,
    output_dir: Path | str = "outputs",
) -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    artifact_root = output_root / "analysis" / "day_90_launch"
    artifact_root.mkdir(parents=True, exist_ok=True)

    front_door_paths = write_front_door_artifacts(_coerce_dict(report.get("front_door_report")), output_dir=output_root)
    outward_paths = write_outward_package_artifacts(_coerce_dict(report.get("outward_package_report")), output_dir=output_root)

    def _copy_if_exists(raw_path: object, destination: Path) -> str:
        source = _ensure_path(raw_path)
        if source is None or not source.exists():
            return ""
        return str(copy_file_if_changed(source, destination).resolve())

    canonical_artifacts = []
    copied_artifacts: dict[str, str] = {
        "front_door_html": _copy_if_exists(front_door_paths["html"], artifact_root / "flagship" / "front_door.html"),
        "front_door_md": _copy_if_exists(front_door_paths["md"], artifact_root / "flagship" / "front_door.md"),
        "outward_package_md": _copy_if_exists(outward_paths["md"], artifact_root / "package" / "outward_package.md"),
    }
    for row in _coerce_list(report.get("canonical_artifacts")):
        if not isinstance(row, dict):
            continue
        artifact_key = str(row.get("key", "")).strip() or "artifact"
        copied_path = _copy_if_exists(row.get("artifact"), artifact_root / "canonical" / f"{artifact_key}{Path(str(row.get('artifact', ''))).suffix or '.md'}")
        canonical_artifacts.append({**row, "copied_artifact": copied_path})

    delivery_checklist = [dict(row) for row in _coerce_list(report.get("delivery_checklist")) if isinstance(row, dict)]
    manifest_payload = {
        "generated_at": report.get("generated_at"),
        "release_status": report.get("release_status"),
        "canonical_artifacts": canonical_artifacts,
    }
    manifest_json = write_json(artifact_root / "canonical_artifact_manifest.json", manifest_payload)
    manifest_lines = [
        "# Canonical Artifact Manifest",
        "",
        f"- Release status: `{report.get('release_status', '')}`",
        "",
    ]
    for row in canonical_artifacts:
        manifest_lines.extend(
            [
                f"## {row.get('label', '')}",
                "",
                f"- Branch: `{row.get('branch', '')}`",
                f"- Status: `{row.get('status', '')}`",
                f"- Audience: {row.get('audience', '')}",
                f"- Canonical artifact: `{row.get('artifact', '')}`",
                f"- Packaged copy: `{row.get('copied_artifact', '')}`",
                f"- Why it exists: {row.get('why_it_exists', '')}",
                "",
            ]
        )
    manifest_md = write_markdown(artifact_root / "canonical_artifact_manifest.md", manifest_lines)

    checklist_payload = {
        "generated_at": report.get("generated_at"),
        "release_status": report.get("release_status"),
        "delivery_summary": report.get("delivery_summary", {}),
        "delivery_checklist": delivery_checklist,
    }
    checklist_json = write_json(artifact_root / "delivery_checklist.json", checklist_payload)
    checklist_lines = [
        "# Delivery Checklist",
        "",
        f"- Release status: `{report.get('release_status', '')}`",
        f"- Ready to show: `{bool(report.get('ready_to_show', False))}`",
        "",
    ]
    for row in delivery_checklist:
        checklist_lines.append(
            f"- `{row.get('key', '')}` [{row.get('status', '')}]: {row.get('label', '')}"
        )
        checklist_lines.append(f"Detail: {row.get('detail', '')}")
        checklist_lines.append(f"Artifact: `{row.get('artifact', '')}`")
    checklist_md = write_markdown(artifact_root / "delivery_checklist.md", checklist_lines)

    launch_payload = {
        **report,
        "canonical_artifacts": canonical_artifacts,
        "copied_artifacts": copied_artifacts,
    }
    json_path = write_json(artifact_root / "day_90_launch.json", launch_payload)
    md_lines = [
        "# Day-90 Launch Package",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Release status: `{report.get('release_status', '')}`",
        f"- Ready to show: `{bool(report.get('ready_to_show', False))}`",
        "",
        "## Launch Story",
        "",
    ]
    for item in _coerce_list(report.get("launch_story")):
        md_lines.append(f"- {item}")
    md_lines.extend(["", "## Delivery Summary", ""])
    summary = _coerce_dict(report.get("delivery_summary"))
    md_lines.append(f"- Passed checks: `{summary.get('pass_count', 0)}`")
    md_lines.append(f"- Attention checks: `{summary.get('attention_count', 0)}`")
    md_lines.append(f"- Failed checks: `{summary.get('fail_count', 0)}`")
    md_lines.extend(["", "## Canonical Artifacts", ""])
    for row in canonical_artifacts:
        md_lines.append(f"- `{row.get('label', '')}` [{row.get('status', '')}]: `{row.get('copied_artifact', '') or row.get('artifact', '')}`")
        md_lines.append(str(row.get("why_it_exists", "")))
    md_lines.extend(["", "## Honesty Notes", ""])
    for item in _coerce_list(report.get("honesty_notes")):
        md_lines.append(f"- {item}")
    md_lines.extend(["", "## Supporting Files", ""])
    md_lines.append(f"- Front door: `{copied_artifacts['front_door_html']}`")
    md_lines.append(f"- Outward package: `{copied_artifacts['outward_package_md']}`")
    md_lines.append(f"- Canonical manifest: `{manifest_md.resolve()}`")
    md_lines.append(f"- Delivery checklist: `{checklist_md.resolve()}`")
    md_path = write_markdown(artifact_root / "day_90_launch.md", md_lines)

    return {
        "json": json_path,
        "md": md_path,
        "canonical_manifest_json": manifest_json,
        "canonical_manifest_md": manifest_md,
        "delivery_checklist_json": checklist_json,
        "delivery_checklist_md": checklist_md,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.day_90_launch",
        description="Build the Day-90 launch package from the strongest live product, ops, creator, and research artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root outputs directory containing the live analysis, analytics, history, and run artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_day_90_launch_report(args.output_dir)
    paths = write_day_90_launch_artifacts(report, output_dir=args.output_dir)
    print(f"day_90_launch_json={paths['json']}")
    print(f"day_90_launch_md={paths['md']}")
    print(f"canonical_manifest_md={paths['canonical_manifest_md']}")
    print(f"delivery_checklist_md={paths['delivery_checklist_md']}")
    print(f"release_status={report.get('release_status', '')}")
    print(f"ready_to_show={bool(report.get('ready_to_show', False))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
