from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .portfolio_artifacts import PortfolioArtifactBundle, load_portfolio_artifact_bundle
from .run_artifacts import safe_read_json, write_json, write_markdown


BRANCH_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "key": "taste_os",
        "label": "Personal Taste OS",
        "audience": "Product review, consumer-demo, and personalization strategy audiences.",
        "success_metric": "Modes feel visibly different and adaptive session steering is easy to explain.",
        "entry_command": "make taste-os-showcase",
        "docs": [
            "docs/personal_taste_os.md",
            "docs/taste_os_demo_walkthrough.md",
            "docs/taste_os_product_story.md",
        ],
        "keep_rule": "Only expand this branch when the opening choice, recovery transcript, or explanation gets clearer.",
        "deprioritize_rule": "Do not add extra model families or UI scaffolding unless it sharpens the Taste OS story.",
        "depends_on": ["safety_research"],
        "overlaps": ["control_room"],
    },
    {
        "key": "control_room",
        "label": "Control Room",
        "audience": "Operators, ML leads, and teammates reviewing system health asynchronously.",
        "success_metric": "One artifact answers what improved, what regressed, and what the next operating move is.",
        "entry_command": "make control-room",
        "docs": [
            "docs/control_room_operating_rhythm.md",
            "docs/weeks_1_8_readiness.md",
        ],
        "keep_rule": "Only keep signals that help product, safety, or research decisions after a run.",
        "deprioritize_rule": "Do not turn this into a raw metric dump or a duplicate of the run logs.",
        "depends_on": ["safety_research"],
        "overlaps": ["taste_os", "creator_intelligence"],
    },
    {
        "key": "creator_intelligence",
        "label": "Creator / Label Intelligence",
        "audience": "Creators, managers, A&R, and label strategy readers.",
        "success_metric": "The report family reads like a strategy brief with clear ranking, scene, and seed comparisons.",
        "entry_command": "spotify-public-insights creator-label-intelligence",
        "docs": [
            "docs/creator_label_intelligence_brief.md",
            "docs/project_threads.md",
        ],
        "keep_rule": "Only expand this branch when the opportunity map or report family becomes more decision-ready.",
        "deprioritize_rule": "Keep one-off catalog explorations nested here unless they become repeatable report families.",
        "depends_on": ["taste_os"],
        "overlaps": ["control_room"],
    },
    {
        "key": "safety_research",
        "label": "Recommender Safety and Research Platform",
        "audience": "Platform builders, ML researchers, and publication-oriented readers.",
        "success_metric": "The safety API is reusable and the benchmark plus claim pack makes experiments comparable.",
        "entry_command": "make research-claims",
        "docs": [
            "docs/recommender_safety_platform.md",
            "docs/benchmark_contract.md",
            "docs/publication_outline.md",
        ],
        "keep_rule": "Only add work here when it improves reuse, benchmark stability, or claim credibility.",
        "deprioritize_rule": "Do not promote single-run wins into flagship narratives without benchmark or robustness support.",
        "depends_on": [],
        "overlaps": ["taste_os", "control_room"],
    },
)

HIERARCHY_OF_BETS: tuple[str, ...] = (
    "Taste OS is the product front door.",
    "Control Room is the operating layer that reviews product and model behavior after runs.",
    "Creator / Label Intelligence is the external strategy surface built from the same taste graph.",
    "Recommender Safety and Research Platform is the reusable infrastructure and evidence layer under the other branches.",
)

PRIORITY_RULES: tuple[str, ...] = (
    "Group Auto-DJ stays a moonshot extension nested under Taste OS and research until it becomes a primary review surface.",
    "New model families stay behind Taste OS or benchmark work unless they improve the opening choice, recovery path, or evidence quality.",
    "One-off public-insights experiments stay nested inside creator intelligence unless they become a repeatable report family.",
    "If a feature does not clearly strengthen one of the four branches, it should wait.",
)


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _status_bucket(*, ready: bool, attention: bool = False, gaps: bool = False) -> str:
    if ready and gaps:
        return "ready_with_gaps"
    if ready:
        return "ready"
    if attention:
        return "attention"
    return "missing"


def _existing_path(path: Path) -> str:
    return str(path.resolve()) if path.exists() else ""


def _creator_market_layer(output_root: Path) -> dict[str, object]:
    artifact_root = output_root / "analysis" / "creator_market_intelligence"
    brief_json = artifact_root / "creator_market_brief.json"
    brief_md = artifact_root / "creator_market_brief.md"
    scene_market_pulse_csv = artifact_root / "scene_market_pulse.csv"
    opportunity_lane_atlas_csv = artifact_root / "opportunity_lane_atlas.csv"
    manifest_json = artifact_root / "creator_market_manifest.json"
    brief_payload = _coerce_dict(safe_read_json(brief_json, default={}))
    manifest_payload = _coerce_dict(safe_read_json(manifest_json, default={}))

    report_family_count = int(
        manifest_payload.get("report_family_count")
        or brief_payload.get("report_family_count")
        or 0
    )
    top_scene = _coerce_dict(brief_payload.get("top_scene"))
    top_lane = _coerce_dict(brief_payload.get("top_lane"))
    scene_name = str(top_scene.get("scene_name", "")).strip()
    lane_scene = str(top_lane.get("scene_name", "")).strip()
    lane_driver = str(top_lane.get("primary_driver", "")).strip()

    artifacts = [
        path
        for path in (
            _existing_path(brief_md),
            _existing_path(brief_json),
            _existing_path(scene_market_pulse_csv),
            _existing_path(opportunity_lane_atlas_csv),
            _existing_path(manifest_json),
        )
        if path
    ]
    ready = bool(_existing_path(brief_md) and _existing_path(scene_market_pulse_csv) and _existing_path(opportunity_lane_atlas_csv))
    attention = bool(artifacts)
    status = _status_bucket(
        ready=ready,
        attention=attention,
        gaps=bool(_existing_path(brief_md) and not ready),
    )
    details: list[str] = []
    if report_family_count:
        details.append(f"aggregates `{report_family_count}` creator families")
    if scene_name:
        details.append(f"top scene `{scene_name}`")
    if lane_scene and lane_driver:
        details.append(f"top lane `{lane_scene} / {lane_driver}`")
    live_signal = (
        f"Creator-market layer {'; '.join(details)}."
        if details
        else "Creator-market rollup artifacts are missing."
    )
    return {
        "status": status,
        "live_signal": live_signal,
        "brief_md": _existing_path(brief_md),
        "brief_json": _existing_path(brief_json),
        "scene_market_pulse_csv": _existing_path(scene_market_pulse_csv),
        "opportunity_lane_atlas_csv": _existing_path(opportunity_lane_atlas_csv),
        "manifest_json": _existing_path(manifest_json),
        "artifacts": artifacts,
    }


def _build_taste_os_branch(bundle: PortfolioArtifactBundle, definition: dict[str, object]) -> dict[str, object]:
    payload = bundle.taste_os_showcase_payload
    canonical_examples = _coerce_list(payload.get("canonical_examples"))
    mode_rows = _coerce_list(_coerce_dict(payload.get("mode_comparison")).get("rows"))
    unique_openings = {
        str(row.get("top_artist", "")).strip()
        for row in mode_rows
        if isinstance(row, dict) and str(row.get("top_artist", "")).strip()
    }
    ready = bundle.taste_os_showcase_md.exists() and len(canonical_examples) >= 4 and len(mode_rows) >= 4
    distinct_modes = len(unique_openings) >= 3
    status = _status_bucket(
        ready=ready and distinct_modes,
        attention=bundle.taste_os_showcase_md.exists(),
        gaps=ready and not distinct_modes,
    )
    summary = (
        f"Showcase pack has `{len(canonical_examples)}` canonical examples, `{len(mode_rows)}` steady-mode comparisons, "
        f"and `{len(unique_openings)}` distinct opening artists."
        if payload
        else "Taste OS showcase artifacts are missing."
    )
    return {
        **definition,
        "status": status,
        "live_signal": summary,
        "artifacts": [str(bundle.taste_os_showcase_md), str(bundle.taste_os_showcase_json)],
    }


def _build_control_room_branch(bundle: PortfolioArtifactBundle, definition: dict[str, object]) -> dict[str, object]:
    payload = bundle.control_room_payload
    ops_health = _coerce_dict(payload.get("ops_health"))
    rhythm = _coerce_dict(payload.get("operating_rhythm"))
    ready = bundle.control_room_md.exists() and str(ops_health.get("status", "")).strip() == "healthy"
    gaps = ready and str(rhythm.get("overall_status", "")).strip() != "healthy"
    status = _status_bucket(ready=ready, attention=bundle.control_room_md.exists(), gaps=gaps)
    summary = str(ops_health.get("headline", "")).strip() or "Control-room artifacts are missing."
    if ready:
        recommended_review = str(rhythm.get("recommended_review_command", "")).strip() or "make control-room"
        summary = f"{summary} Recommended review command: `{recommended_review}`."
    return {
        **definition,
        "status": status,
        "live_signal": summary,
        "artifacts": [
            str(bundle.control_room_md),
            str(bundle.output_root / "analytics" / "control_room_weekly_summary.md"),
            str(bundle.output_root / "analytics" / "control_room_triage.md"),
        ],
    }


def _build_creator_branch(bundle: PortfolioArtifactBundle, definition: dict[str, object]) -> dict[str, object]:
    existing_views = [path for path in bundle.creator_comparison_markdown_paths.values() if path.exists()]
    existing_brief_views = [path for path in bundle.creator_brief_markdown_paths.values() if path.exists()]
    primary_report = bundle.creator_primary_report_path
    report_family_index = bundle.creator_report_family_md_path
    market_layer = _creator_market_layer(bundle.output_root)
    ready = primary_report is not None and primary_report.exists() and len(existing_views) >= 4
    status = _status_bucket(ready=ready, attention=bundle.creator_manifest_path is not None)
    report_family_summary = (
        f"Latest creator family exposes `{len(existing_views)}` comparison views, `{len(existing_brief_views)}` brief views, "
        f"and `{('has' if report_family_index and report_family_index.exists() else 'does not have')}` a report-family index."
        if bundle.creator_manifest
        else "Creator-intelligence report-family artifacts are missing."
    )
    market_summary = str(market_layer.get("live_signal", "")).strip()
    summary = f"{report_family_summary} {market_summary}".strip()
    artifacts = [str(primary_report)] if primary_report is not None else []
    artifacts.extend(str(path) for path in existing_views)
    artifacts.extend(str(path) for path in existing_brief_views)
    if report_family_index is not None and report_family_index.exists():
        artifacts.append(str(report_family_index))
    if bundle.creator_manifest_path is not None:
        artifacts.append(str(bundle.creator_manifest_path))
    artifacts.extend(str(path) for path in _coerce_list(market_layer.get("artifacts")))
    return {
        **definition,
        "status": status,
        "live_signal": summary,
        "market_layer": market_layer,
        "artifacts": artifacts,
    }


def _build_safety_research_branch(bundle: PortfolioArtifactBundle, definition: dict[str, object]) -> dict[str, object]:
    payload = bundle.research_claims_payload
    benchmark_lock = _coerce_dict(payload.get("benchmark_lock"))
    primary_claim = _coerce_dict(payload.get("primary_claim"))
    submission_readiness = _coerce_dict(payload.get("submission_readiness"))
    primary_status = str(primary_claim.get("status", "")).strip()
    submission_status = str(submission_readiness.get("status", "")).strip()
    believable = bool(payload.get("believable_submission_path"))
    benchmark_ready = bool(benchmark_lock.get("comparison_ready"))
    platform_contract_ready = bool(bundle.safety_platform_contract_md and bundle.safety_platform_contract_md.exists())
    ready = bundle.research_claims_md.exists() and believable and primary_status in {"analysis_ready", "submission_candidate"}
    status = _status_bucket(ready=ready, attention=bundle.research_claims_md.exists(), gaps=ready and not benchmark_ready)
    summary = "Research-claim artifacts are missing."
    if payload:
        benchmark_id = str(benchmark_lock.get("benchmark_id", "")).strip() or "n/a"
        summary = (
            f"Primary claim `{primary_claim.get('key', '')}` is `{primary_status or 'unknown'}` and benchmark "
            f"`{benchmark_id}` is `{('comparison-ready' if benchmark_ready else 'not comparison-ready')}`. "
            f"Submission readiness=`{submission_status or 'unknown'}`. Safety contract present=`{platform_contract_ready}`."
        )
    artifacts = [str(bundle.research_claims_md), str(bundle.research_claims_json)]
    if bundle.research_publication_outline_md is not None:
        artifacts.append(str(bundle.research_publication_outline_md))
    if bundle.research_claim_support_md is not None:
        artifacts.append(str(bundle.research_claim_support_md))
    if bundle.research_submission_readiness_md is not None:
        artifacts.append(str(bundle.research_submission_readiness_md))
    if bundle.safety_platform_contract_md is not None:
        artifacts.append(str(bundle.safety_platform_contract_md))
    if bundle.safety_platform_contract_json is not None:
        artifacts.append(str(bundle.safety_platform_contract_json))
    if bundle.benchmark_manifest_md is not None:
        artifacts.append(str(bundle.benchmark_manifest_md))
    if bundle.benchmark_manifest_json is not None:
        artifacts.append(str(bundle.benchmark_manifest_json))
    return {
        **definition,
        "status": status,
        "live_signal": summary,
        "artifacts": artifacts,
    }


def build_branch_portfolio_report(
    output_dir: Path | str = "outputs",
    *,
    artifact_bundle: PortfolioArtifactBundle | None = None,
) -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    bundle = artifact_bundle or load_portfolio_artifact_bundle(output_root)
    definitions = {str(item["key"]): item for item in BRANCH_DEFINITIONS}
    branches = [
        _build_taste_os_branch(bundle, definitions["taste_os"]),
        _build_control_room_branch(bundle, definitions["control_room"]),
        _build_creator_branch(bundle, definitions["creator_intelligence"]),
        _build_safety_research_branch(bundle, definitions["safety_research"]),
    ]
    ready_count = sum(1 for branch in branches if str(branch.get("status", "")).startswith("ready"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "summary": {
            "primary_branch_count": len(branches),
            "ready_or_ready_with_gaps": ready_count,
            "priority_rule": "Only ship work that clearly strengthens one of the four primary branches.",
        },
        "hierarchy_of_bets": list(HIERARCHY_OF_BETS),
        "branches": branches,
        "priority_rules": list(PRIORITY_RULES),
    }


def write_branch_portfolio_artifacts(report: dict[str, object], *, output_dir: Path | str = "outputs") -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    artifact_dir = output_root / "analysis" / "portfolio_branches"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    json_path = write_json(artifact_dir / "portfolio_branches.json", report)

    lines = [
        "# Higher-Level Branch Map",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Output root: `{report.get('output_dir', '')}`",
        f"- Primary branches: `{_coerce_dict(report.get('summary')).get('primary_branch_count', 0)}`",
        f"- Ready or ready-with-gaps: `{_coerce_dict(report.get('summary')).get('ready_or_ready_with_gaps', 0)}`",
        "",
        "## Clean Hierarchy Of Bets",
        "",
    ]
    for item in _coerce_list(report.get("hierarchy_of_bets")):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Branches",
            "",
        ]
    )
    for branch in _coerce_list(report.get("branches")):
        if not isinstance(branch, dict):
            continue
        lines.extend(
            [
                f"### {branch.get('label', '')}",
                "",
                f"- Audience: {branch.get('audience', '')}",
                f"- Success metric: {branch.get('success_metric', '')}",
                f"- Status: `{branch.get('status', '')}`",
                f"- Entry command: `{branch.get('entry_command', '')}`",
                f"- Live signal: {branch.get('live_signal', '')}",
                f"- Keep strengthening by: {branch.get('keep_rule', '')}",
                f"- Deprioritize rule: {branch.get('deprioritize_rule', '')}",
                f"- Depends on: `{', '.join(str(item) for item in _coerce_list(branch.get('depends_on'))) or 'none'}`",
                f"- Overlaps with: `{', '.join(str(item) for item in _coerce_list(branch.get('overlaps'))) or 'none'}`",
            ]
        )
        market_layer = _coerce_dict(branch.get("market_layer"))
        if market_layer:
            lines.extend(
                [
                    f"- Creator-market layer: `{market_layer.get('status', '')}`",
                    f"- Creator-market signal: {market_layer.get('live_signal', '')}",
                ]
            )
        lines.extend(["", "Docs:"])
        for doc_path in _coerce_list(branch.get("docs")):
            lines.append(f"- `{doc_path}`")
        lines.extend(["", "Artifacts:"])
        for artifact in _coerce_list(branch.get("artifacts")):
            lines.append(f"- `{artifact}`")
        lines.append("")
    lines.extend(["## Priority Rules", ""])
    for item in _coerce_list(report.get("priority_rules")):
        lines.append(f"- {item}")

    md_path = write_markdown(artifact_dir / "portfolio_branches.md", lines)
    return {"json": json_path, "md": md_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Week 12 higher-level branch map for the repo.")
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
    report = build_branch_portfolio_report(args.output_dir)
    paths = write_branch_portfolio_artifacts(report, output_dir=args.output_dir)
    if args.stdout_format == "json":
        import json

        print(json.dumps(report, indent=2))
    else:
        summary = _coerce_dict(report.get("summary"))
        print(f"portfolio_branches_json={paths['json']}")
        print(f"portfolio_branches_md={paths['md']}")
        print(f"primary_branches={summary.get('primary_branch_count', 0)}")
        print(f"ready_or_ready_with_gaps={summary.get('ready_or_ready_with_gaps', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
