from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .branch_portfolio import build_branch_portfolio_report
from .portfolio_artifacts import load_portfolio_artifact_bundle
from .run_artifacts import copy_file_if_changed, write_json, write_markdown


_CLAIM_SCENARIO_PREFERENCE: dict[str, dict[str, int]] = {
    "shift_robustness": {
        "friction_spike": 120,
        "skip_recovery": 80,
        "repeat_request": 45,
        "steady": 20,
    },
    "candidate_ranking": {
        "skip_recovery": 100,
        "steady": 55,
        "repeat_request": 25,
    },
    "risk_aware_abstention": {
        "friction_spike": 120,
        "skip_recovery": 70,
        "steady": 30,
    },
}

_CLAIM_MODE_PREFERENCE: dict[str, dict[str, int]] = {
    "shift_robustness": {
        "commute": 45,
        "discovery": 35,
        "workout": 20,
        "focus": 10,
    },
    "candidate_ranking": {
        "discovery": 50,
        "focus": 30,
        "commute": 15,
    },
    "risk_aware_abstention": {
        "commute": 50,
        "focus": 20,
    },
}


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _format_metric(value: object) -> str:
    metric = _safe_float(value)
    if metric != metric:
        return "n/a"
    return f"{metric:.3f}"


def _run_id_from_path_string(value: object) -> str:
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    path = Path(raw_value)
    return path.name if path.name else raw_value


def _existing_path(raw_path: object) -> Path | None:
    path_text = str(raw_path).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.exists():
        return None
    return path.resolve()


def _path_text(path: Path | None) -> str:
    return str(path.resolve()) if path is not None and path.exists() else ""


def _build_submission_readiness(
    research_claims_payload: dict[str, object],
    primary_claim: dict[str, object],
) -> dict[str, object]:
    payload = _coerce_dict(research_claims_payload.get("submission_readiness"))
    if payload:
        return {
            "status": str(payload.get("status", "")).strip(),
            "ready_for_external_review": bool(payload.get("ready_for_external_review")),
            "summary": [str(item) for item in _coerce_list(payload.get("summary")) if str(item).strip()],
            "blockers": [str(item) for item in _coerce_list(payload.get("blockers")) if str(item).strip()],
        }

    believable = bool(research_claims_payload.get("believable_submission_path"))
    primary_status = str(primary_claim.get("status", "")).strip() or "unknown"
    fallback_status = "analysis_ready" if believable else primary_status
    return {
        "status": fallback_status,
        "ready_for_external_review": believable,
        "summary": [
            f"Primary claim `{str(primary_claim.get('key', '')).strip() or 'n/a'}` is `{primary_status}`.",
            f"Believable submission path is `{believable}`.",
        ],
        "blockers": [str(item) for item in _coerce_list(primary_claim.get("missing_checks")) if str(item).strip()][:3],
    }


def _branch_lookup(report: dict[str, object]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for branch in _coerce_list(report.get("branches")):
        if isinstance(branch, dict):
            lookup[str(branch.get("key", "")).strip()] = branch
    return lookup


def _choose_demo_examples(
    *,
    showcase_payload: dict[str, object],
    primary_claim_key: str,
) -> tuple[dict[str, object], dict[str, object]]:
    examples = [row for row in _coerce_list(showcase_payload.get("canonical_examples")) if isinstance(row, dict)]
    if not examples:
        return {}, {}

    scenario_pref = _CLAIM_SCENARIO_PREFERENCE.get(primary_claim_key, {})
    mode_pref = _CLAIM_MODE_PREFERENCE.get(primary_claim_key, {})

    def example_score(example: dict[str, object]) -> tuple[int, int, int, int, str]:
        scenario = str(example.get("scenario", "")).strip().lower()
        mode = str(example.get("mode", "")).strip().lower()
        safe_steps = int(example.get("adaptive_safe_route_steps", 0) or 0)
        replans = int(example.get("adaptive_replans", 0) or 0)
        return (
            int(scenario_pref.get(scenario, 0)),
            int(mode_pref.get(mode, 0)),
            int(safe_steps),
            int(replans),
            str(example.get("label", "")),
        )

    ranked = sorted(examples, key=example_score, reverse=True)
    flagship = ranked[0]
    supporting = ranked[1] if len(ranked) > 1 else ranked[0]
    return flagship, supporting


def _coherence_summary(
    *,
    showcase_run_id: str,
    control_room_run_id: str,
    research_run_id: str,
) -> dict[str, object]:
    known_ids = [run_id for run_id in (showcase_run_id, control_room_run_id, research_run_id) if run_id]
    aligned = bool(known_ids) and len(set(known_ids)) == 1
    recommended_actions: list[str] = []
    if not aligned and showcase_run_id and control_room_run_id and showcase_run_id != control_room_run_id:
        recommended_actions.append("Regenerate the Taste OS showcase on the current control-room review anchor so the flagship story uses one run.")
    if not aligned and research_run_id and control_room_run_id and research_run_id != control_room_run_id:
        recommended_actions.append("Point the research claim pack at the same review anchor used by the control room before showing the repo externally.")

    if aligned and known_ids:
        summary = f"Product, ops, and research artifacts all point at `{known_ids[0]}`."
    else:
        summary = (
            f"Taste OS is on `{showcase_run_id or 'n/a'}`, control room is on `{control_room_run_id or 'n/a'}`, "
            f"and research is on `{research_run_id or 'n/a'}`."
        )
    return {
        "aligned": aligned,
        "showcase_run_id": showcase_run_id,
        "control_room_run_id": control_room_run_id,
        "research_run_id": research_run_id,
        "summary": summary,
        "recommended_actions": recommended_actions,
    }


def _flagship_bridge_points(
    *,
    flagship_demo: dict[str, object],
    primary_claim: dict[str, object],
    control_room_payload: dict[str, object],
) -> list[str]:
    top_artist = str(flagship_demo.get("top_artist", "")).strip() or "the opening artist"
    fallback_policy = str(flagship_demo.get("fallback_policy_name", "")).strip() or "the active fallback policy"
    claim_title = str(primary_claim.get("title", "")).strip() or "the primary research claim"
    ops_headline = str(_coerce_dict(control_room_payload.get("ops_health")).get("headline", "")).strip()

    points = [
        f"The demo starts with `{top_artist}` and shows `{fallback_policy}` taking over when the session needs adaptation.",
        f"That product behavior is the user-facing version of `{claim_title}`.",
    ]
    if ops_headline:
        points.append(f"The control room confirms that this is reviewable system behavior, not a one-off notebook result: {ops_headline}")
    return points


def _evidence_scoreboard(
    *,
    showcase_payload: dict[str, object],
    flagship_demo: dict[str, object],
    primary_claim: dict[str, object],
    control_room_payload: dict[str, object],
) -> list[dict[str, object]]:
    showcase_summary = _coerce_dict(showcase_payload.get("showcase_summary"))
    primary_metrics = _coerce_dict(primary_claim.get("metrics"))
    safety = _coerce_dict(control_room_payload.get("safety"))
    qoe = _coerce_dict(control_room_payload.get("qoe"))

    rows = [
        {
            "label": "Distinct steady-mode openings",
            "value": int(
                len(
                    {
                        str(row.get("top_artist", "")).strip()
                        for row in _coerce_list(_coerce_dict(showcase_payload.get("mode_comparison")).get("rows"))
                        if isinstance(row, dict) and str(row.get("top_artist", "")).strip()
                    }
                )
            ),
            "formatted_value": str(
                int(
                    len(
                        {
                            str(row.get("top_artist", "")).strip()
                            for row in _coerce_list(_coerce_dict(showcase_payload.get("mode_comparison")).get("rows"))
                            if isinstance(row, dict) and str(row.get("top_artist", "")).strip()
                        }
                    )
                )
            ),
            "why_it_matters": "Shows that the product modes behave differently at the opening decision, not just in labels.",
            "source": "taste_os_showcase",
        },
        {
            "label": "Flagship demo replan count",
            "value": int(flagship_demo.get("adaptive_replans", 0) or 0),
            "formatted_value": str(int(flagship_demo.get("adaptive_replans", 0) or 0)),
            "why_it_matters": "Shows that the flagship story is adaptive rather than a static playlist preview.",
            "source": "taste_os_showcase",
        },
        {
            "label": "Worst supported robustness gap",
            "value": _safe_float(primary_metrics.get("worst_robustness_gap", safety.get("robustness_max_top1_gap"))),
            "formatted_value": _format_metric(primary_metrics.get("worst_robustness_gap", safety.get("robustness_max_top1_gap"))),
            "why_it_matters": "This is the clearest proof that performance breaks are concentrated in a real session slice.",
            "source": "research_claims",
        },
        {
            "label": "Repeated-run slice support",
            "value": _safe_float(primary_metrics.get("consistent_slice_rate")),
            "formatted_value": (
                f"{int(primary_metrics.get('consistent_slice_run_count', 0) or 0)}/"
                f"{int(primary_metrics.get('repeated_run_count', 0) or 0)}"
                if int(primary_metrics.get("repeated_run_count", 0) or 0) > 0
                else "n/a"
            ),
            "why_it_matters": "Shows whether the same failure slice recurs across matching completed runs instead of only once.",
            "source": "research_claims",
        },
        {
            "label": "Target drift JSD",
            "value": _safe_float(primary_metrics.get("target_drift_jsd", safety.get("test_jsd_target_drift"))),
            "formatted_value": _format_metric(primary_metrics.get("target_drift_jsd", safety.get("test_jsd_target_drift"))),
            "why_it_matters": "Quantifies how far the evaluation regime drifts from training behavior.",
            "source": "research_claims",
        },
        {
            "label": "Selective risk",
            "value": _safe_float(primary_metrics.get("selective_risk", safety.get("test_selective_risk"))),
            "formatted_value": _format_metric(primary_metrics.get("selective_risk", safety.get("test_selective_risk"))),
            "why_it_matters": "Shows whether the uncertainty layer is buying meaningful risk reduction.",
            "source": "research_claims",
        },
        {
            "label": "Abstention rate",
            "value": _safe_float(primary_metrics.get("abstention_rate", safety.get("test_abstention_rate"))),
            "formatted_value": _format_metric(primary_metrics.get("abstention_rate", safety.get("test_abstention_rate"))),
            "why_it_matters": "Shows how much coverage the current operating threshold gives up to control failures.",
            "source": "research_claims",
        },
        {
            "label": "Worst stress skip risk",
            "value": _safe_float(primary_metrics.get("stress_skip_risk", qoe.get("stress_worst_skip_risk"))),
            "formatted_value": _format_metric(primary_metrics.get("stress_skip_risk", qoe.get("stress_worst_skip_risk"))),
            "why_it_matters": "Keeps the safety story honest by showing the hardest remaining failure regime.",
            "source": "research_claims",
        },
        {
            "label": "Canonical examples",
            "value": int(showcase_summary.get("canonical_example_count", 0) or 0),
            "formatted_value": str(int(showcase_summary.get("canonical_example_count", 0) or 0)),
            "why_it_matters": "Shows that the product surface is packaged into a reviewable artifact family instead of a single cherry-picked run.",
            "source": "taste_os_showcase",
        },
    ]
    return rows


def build_claim_to_demo_report(output_dir: Path | str = "outputs") -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    bundle = load_portfolio_artifact_bundle(output_root, refresh=True)
    branch_report = build_branch_portfolio_report(output_root, artifact_bundle=bundle)
    branch_lookup = _branch_lookup(branch_report)

    showcase_payload = bundle.taste_os_showcase_payload
    control_room_payload = bundle.control_room_payload
    research_claims_payload = bundle.research_claims_payload
    primary_claim = _coerce_dict(research_claims_payload.get("primary_claim"))
    backup_claim = _coerce_dict(research_claims_payload.get("backup_claim"))
    submission_readiness = _build_submission_readiness(research_claims_payload, primary_claim)

    flagship_demo, supporting_demo = _choose_demo_examples(
        showcase_payload=showcase_payload,
        primary_claim_key=str(primary_claim.get("key", "")).strip(),
    )

    showcase_run_context = _coerce_dict(showcase_payload.get("run_context"))
    control_room_latest_run = _coerce_dict(control_room_payload.get("latest_run"))
    research_run = _coerce_dict(research_claims_payload.get("run"))

    coherence = _coherence_summary(
        showcase_run_id=_run_id_from_path_string(showcase_run_context.get("run_dir")),
        control_room_run_id=str(control_room_latest_run.get("run_id", "")).strip(),
        research_run_id=str(research_run.get("run_id", "")).strip(),
    )

    flagship_demo_path = _existing_path(flagship_demo.get("demo_md_path"))
    creator_primary_path = bundle.creator_primary_report_path.resolve() if bundle.creator_primary_report_path and bundle.creator_primary_report_path.exists() else None
    creator_report_family_path = (
        bundle.creator_report_family_md_path.resolve()
        if bundle.creator_report_family_md_path and bundle.creator_report_family_md_path.exists()
        else None
    )
    creator_strategy_path = None
    if bundle.creator_brief_markdown_paths.get("scene_strategy_watch") and bundle.creator_brief_markdown_paths["scene_strategy_watch"].exists():
        creator_strategy_path = bundle.creator_brief_markdown_paths["scene_strategy_watch"].resolve()
    creator_supporting_path = None
    for key in ("opportunity_lane_comparison", "scene_seed_comparison", "scene_comparison"):
        candidate = bundle.creator_comparison_markdown_paths.get(key)
        if candidate and candidate.exists():
            creator_supporting_path = candidate.resolve()
            break
    creator_review_path = creator_strategy_path or creator_supporting_path or creator_report_family_path or creator_primary_path
    research_claim_support_path = bundle.research_claim_support_md.resolve() if bundle.research_claim_support_md else None
    research_submission_readiness_path = (
        bundle.research_submission_readiness_md.resolve() if bundle.research_submission_readiness_md else None
    )
    research_publication_outline_path = (
        bundle.research_publication_outline_md.resolve() if bundle.research_publication_outline_md else None
    )
    safety_platform_contract_path = (
        bundle.safety_platform_contract_md.resolve() if bundle.safety_platform_contract_md and bundle.safety_platform_contract_md.exists() else None
    )

    bridge_points = _flagship_bridge_points(
        flagship_demo=flagship_demo,
        primary_claim=primary_claim,
        control_room_payload=control_room_payload,
    )
    if creator_review_path is not None:
        bridge_points.append(
            "Creator intelligence proves the same taste graph can become a strategy brief, not just a listener-facing demo."
        )
    if safety_platform_contract_path is not None:
        bridge_points.append(
            "The safety platform contract shows that the research story is backed by a reusable API and benchmark surface, not only a single narrative artifact."
        )
    scoreboard = _evidence_scoreboard(
        showcase_payload=showcase_payload,
        flagship_demo=flagship_demo,
        primary_claim=primary_claim,
        control_room_payload=control_room_payload,
    )

    review_sequence: list[dict[str, object]] = [
        {
            "step": 1,
            "label": "Product proof",
            "artifact": _path_text(flagship_demo_path) or str(bundle.taste_os_showcase_md.resolve()),
            "why": "Start with the strongest adaptive demo because it makes the system feel tangible immediately.",
        }
    ]
    if creator_review_path is not None:
        review_sequence.append(
            {
                "step": len(review_sequence) + 1,
                "label": "Creator strategy proof",
                "artifact": str(creator_review_path),
                "why": "Show that the same system can also produce a decision-ready creator or label strategy surface.",
            }
        )
    review_sequence.extend(
        [
            {
                "step": len(review_sequence) + 1,
                "label": "Operator proof",
                "artifact": str(bundle.control_room_md.resolve()),
                "why": "Show that the same system is reviewable operationally and not just visually convincing.",
            },
            {
                "step": len(review_sequence) + 2,
                "label": "Research proof",
                "artifact": _path_text(research_submission_readiness_path)
                or _path_text(research_claim_support_path)
                or str(bundle.research_claims_md.resolve()),
                "why": "Close by showing the strongest claim, its support matrix, and the current submission-readiness state.",
            },
        ]
    )

    branch_alignment_specs = [
        (
            "taste_os",
            "Product front door for the adaptive listening story.",
            _path_text(flagship_demo_path) or str(bundle.taste_os_showcase_md.resolve()),
        ),
        (
            "creator_intelligence",
            "External strategy branch that turns the same taste graph into creator and label opportunity views.",
            _path_text(creator_review_path) or _path_text(creator_primary_path),
        ),
        (
            "control_room",
            "Operating layer that reviews the same system asynchronously after a run.",
            str(bundle.control_room_md.resolve()),
        ),
        (
            "safety_research",
            "Evidence and platform branch that makes the story benchmarked, reusable, and publication-shaped.",
            _path_text(research_submission_readiness_path)
            or _path_text(research_claim_support_path)
            or _path_text(safety_platform_contract_path)
            or str(bundle.research_claims_md.resolve()),
        ),
    ]
    branch_alignment = []
    for branch_key, role_in_story, artifact in branch_alignment_specs:
        branch = _coerce_dict(branch_lookup.get(branch_key))
        branch_alignment.append(
            {
                "key": branch_key,
                "label": str(branch.get("label", branch_key)).strip(),
                "status": str(branch.get("status", "missing")).strip(),
                "audience": str(branch.get("audience", "")).strip(),
                "success_metric": str(branch.get("success_metric", "")).strip(),
                "live_signal": str(branch.get("live_signal", "")).strip(),
                "role_in_story": role_in_story,
                "artifact": artifact,
            }
        )

    next_actions = list(coherence.get("recommended_actions", []))
    if creator_review_path is None:
        next_actions.append("Regenerate the creator-intelligence report family so the flagship review has a strategy branch, not just product and ops.")
    for item in _coerce_list(primary_claim.get("missing_checks"))[:2]:
        if isinstance(item, str) and item not in next_actions:
            next_actions.append(item)
    for item in submission_readiness.get("blockers", [])[:2]:
        if item not in next_actions:
            next_actions.append(item)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "headline": "One package that connects the strongest product demo to creator strategy, operator review, and the leading research claim.",
        "coherence": coherence,
        "primary_claim": {
            "key": str(primary_claim.get("key", "")),
            "title": str(primary_claim.get("title", "")),
            "status": str(primary_claim.get("status", "")),
            "summary": str(primary_claim.get("summary", "")),
        },
        "backup_claim": {
            "key": str(backup_claim.get("key", "")),
            "status": str(backup_claim.get("status", "")),
            "summary": str(backup_claim.get("summary", "")),
        },
        "flagship_demo": {
            "label": str(flagship_demo.get("label", "")),
            "mode": str(flagship_demo.get("mode", "")),
            "scenario": str(flagship_demo.get("scenario", "")),
            "story": str(flagship_demo.get("story", "")),
            "story_outcome": str(flagship_demo.get("story_outcome", "")),
            "top_artist": str(flagship_demo.get("top_artist", "")),
            "backup_artist": str(flagship_demo.get("backup_artist", "")),
            "fallback_policy_name": str(flagship_demo.get("fallback_policy_name", "")),
            "adaptive_replans": int(flagship_demo.get("adaptive_replans", 0) or 0),
            "adaptive_safe_route_steps": int(flagship_demo.get("adaptive_safe_route_steps", 0) or 0),
            "demo_json_path": str(flagship_demo.get("demo_json_path", "")),
            "demo_md_path": str(flagship_demo.get("demo_md_path", "")),
        },
        "supporting_demo": {
            "label": str(supporting_demo.get("label", "")),
            "mode": str(supporting_demo.get("mode", "")),
            "scenario": str(supporting_demo.get("scenario", "")),
            "top_artist": str(supporting_demo.get("top_artist", "")),
            "demo_md_path": str(supporting_demo.get("demo_md_path", "")),
        },
        "submission_readiness": submission_readiness,
        "branch_alignment": branch_alignment,
        "review_sequence": review_sequence,
        "bridge_points": bridge_points,
        "evidence_scoreboard": scoreboard,
        "talk_tracks": {
            "ninety_second": [
                f"Taste OS is the product surface, and `{str(flagship_demo.get('label', '')).strip() or 'the flagship demo'}` is the fastest proof that the modes really behave differently.",
                "Creator intelligence shows that the same taste system can be repackaged as a strategy brief for a different audience without inventing a new platform.",
                f"The strongest repo-level claim is `{str(primary_claim.get('key', '')).strip() or 'the current primary claim'}`, which frames the same system as a measurable robustness and drift problem.",
                "Control room evidence and submission-readiness artifacts keep the story grounded in operating and research reality rather than vibes.",
            ],
            "three_minute": [
                "Open with the flagship Taste OS example and explain what changed in the route after the session event.",
                "Show the creator strategy artifact next so the audience sees that the same taste graph powers a second product surface with a distinct reader and decision style.",
                "Move to the control room and show that the run is healthy operationally while the strategic safety findings are still explicit.",
                "Close on the claim-support matrix and submission-readiness pack so the external story ends on evidence rather than vibes.",
            ],
        },
        "source_artifacts": {
            "taste_os_showcase_md": str(bundle.taste_os_showcase_md.resolve()),
            "control_room_md": str(bundle.control_room_md.resolve()),
            "research_claims_md": str(bundle.research_claims_md.resolve()),
            "benchmark_manifest_md": str(bundle.benchmark_manifest_md.resolve()) if bundle.benchmark_manifest_md else "",
            "creator_primary_md": _path_text(creator_primary_path),
            "creator_report_family_md": _path_text(creator_report_family_path),
            "creator_strategy_md": _path_text(creator_strategy_path),
            "creator_supporting_md": _path_text(creator_supporting_path),
            "safety_platform_contract_md": _path_text(safety_platform_contract_path),
            "research_claim_support_md": _path_text(research_claim_support_path),
            "submission_readiness_md": _path_text(research_submission_readiness_path),
            "publication_outline_md": _path_text(research_publication_outline_path),
        },
        "next_actions": next_actions,
    }


def write_claim_to_demo_artifacts(
    report: dict[str, object],
    *,
    output_dir: Path | str = "outputs",
) -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    artifact_root = output_root / "analysis" / "claim_to_demo"
    artifact_root.mkdir(parents=True, exist_ok=True)

    def _copy_if_exists(raw_path: object, destination: Path) -> str:
        source = Path(str(raw_path).strip()) if str(raw_path).strip() else None
        if source is None or not source.exists():
            return ""
        return str(copy_file_if_changed(source, destination).resolve())

    flagship_demo = _coerce_dict(report.get("flagship_demo"))
    supporting_demo = _coerce_dict(report.get("supporting_demo"))
    source_artifacts = _coerce_dict(report.get("source_artifacts"))

    copied_artifacts = {
        "flagship_demo_md": _copy_if_exists(flagship_demo.get("demo_md_path"), artifact_root / "taste_os" / "flagship_demo.md"),
        "supporting_demo_md": _copy_if_exists(supporting_demo.get("demo_md_path"), artifact_root / "taste_os" / "supporting_demo.md"),
        "control_room_md": _copy_if_exists(source_artifacts.get("control_room_md"), artifact_root / "control_room" / "control_room.md"),
        "creator_primary_md": _copy_if_exists(source_artifacts.get("creator_primary_md"), artifact_root / "creator" / "creator_primary.md"),
        "creator_report_family_md": _copy_if_exists(source_artifacts.get("creator_report_family_md"), artifact_root / "creator" / "creator_report_family.md"),
        "creator_strategy_md": _copy_if_exists(source_artifacts.get("creator_strategy_md"), artifact_root / "creator" / "scene_strategy_watch.md"),
        "creator_supporting_md": _copy_if_exists(source_artifacts.get("creator_supporting_md"), artifact_root / "creator" / "supporting_view.md"),
        "research_claims_md": _copy_if_exists(source_artifacts.get("research_claims_md"), artifact_root / "research" / "research_claims.md"),
        "benchmark_manifest_md": _copy_if_exists(source_artifacts.get("benchmark_manifest_md"), artifact_root / "research" / "benchmark_lock_manifest.md"),
        "research_claim_support_md": _copy_if_exists(source_artifacts.get("research_claim_support_md"), artifact_root / "research" / "claim_support_matrix.md"),
        "submission_readiness_md": _copy_if_exists(source_artifacts.get("submission_readiness_md"), artifact_root / "research" / "submission_readiness.md"),
        "publication_outline_md": _copy_if_exists(source_artifacts.get("publication_outline_md"), artifact_root / "research" / "publication_outline.md"),
        "safety_platform_contract_md": _copy_if_exists(source_artifacts.get("safety_platform_contract_md"), artifact_root / "research" / "safety_platform_contract.md"),
    }

    json_payload = {**report, "copied_artifacts": copied_artifacts}
    json_path = write_json(artifact_root / "claim_to_demo.json", json_payload)

    md_lines = [
        "# Claim To Demo Review Pack",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Output root: `{report.get('output_dir', '')}`",
        f"- Headline: {report.get('headline', '')}",
        "",
        "## Coherence",
        "",
    ]
    coherence = _coerce_dict(report.get("coherence"))
    md_lines.append(f"- Summary: {coherence.get('summary', '')}")
    md_lines.append(f"- Aligned: `{coherence.get('aligned', False)}`")
    md_lines.append("")
    md_lines.append("## Flagship Demo")
    md_lines.append("")
    md_lines.append(f"- Label: `{flagship_demo.get('label', '')}`")
    md_lines.append(f"- Mode / scenario: `{flagship_demo.get('mode', '')}` / `{flagship_demo.get('scenario', '')}`")
    md_lines.append(f"- Story: {flagship_demo.get('story', '')}")
    md_lines.append(f"- Outcome: {flagship_demo.get('story_outcome', '')}")
    md_lines.append(f"- Demo markdown: `{copied_artifacts['flagship_demo_md']}`")
    md_lines.append("")
    md_lines.append("## Review Sequence")
    md_lines.append("")
    for row in _coerce_list(report.get("review_sequence")):
        if not isinstance(row, dict):
            continue
        md_lines.append(f"- `{row.get('step', '')}`. {row.get('label', '')}: `{row.get('artifact', '')}`")
        md_lines.append(str(row.get("why", "")))
    md_lines.append("")
    md_lines.append("## Branch Alignment")
    md_lines.append("")
    for row in _coerce_list(report.get("branch_alignment")):
        if not isinstance(row, dict):
            continue
        md_lines.append(
            f"- `{row.get('label', '')}` [{row.get('status', '')}]: {row.get('role_in_story', '')}"
        )
        md_lines.append(f"Audience: {row.get('audience', '')}")
        md_lines.append(f"Success metric: {row.get('success_metric', '')}")
        if str(row.get("artifact", "")).strip():
            md_lines.append(f"Anchor artifact: `{row.get('artifact', '')}`")
    md_lines.append("")
    md_lines.append("## Bridge Points")
    md_lines.append("")
    for item in _coerce_list(report.get("bridge_points")):
        md_lines.append(f"- {item}")
    md_lines.append("")
    md_lines.append("## Evidence Scoreboard")
    md_lines.append("")
    for row in _coerce_list(report.get("evidence_scoreboard")):
        if not isinstance(row, dict):
            continue
        md_lines.append(f"- {row.get('label', '')}: `{row.get('formatted_value', '')}`")
        md_lines.append(str(row.get("why_it_matters", "")))
    md_lines.append("")
    md_lines.append("## Submission Readiness")
    md_lines.append("")
    submission_readiness = _coerce_dict(report.get("submission_readiness"))
    md_lines.append(f"- Status: `{submission_readiness.get('status', '')}`")
    md_lines.append(f"- Ready for external review: `{submission_readiness.get('ready_for_external_review', False)}`")
    for item in _coerce_list(submission_readiness.get("summary")):
        md_lines.append(f"- {item}")
    for item in _coerce_list(submission_readiness.get("blockers"))[:3]:
        md_lines.append(f"- Blocker: {item}")
    md_lines.append("")
    md_lines.append("## Next Actions")
    md_lines.append("")
    for item in _coerce_list(report.get("next_actions")):
        md_lines.append(f"- {item}")
    md_lines.append("")
    md_lines.append("## Source Assets")
    md_lines.append("")
    for key, value in copied_artifacts.items():
        if value:
            md_lines.append(f"- {key}: `{value}`")
    md_path = write_markdown(artifact_root / "claim_to_demo.md", md_lines)

    talk_track_lines = [
        "# Claim To Demo Talk Track",
        "",
        "## Ninety Seconds",
        "",
    ]
    for item in _coerce_list(_coerce_dict(report.get("talk_tracks")).get("ninety_second")):
        talk_track_lines.append(f"- {item}")
    talk_track_lines.extend(["", "## Three Minutes", ""])
    for item in _coerce_list(_coerce_dict(report.get("talk_tracks")).get("three_minute")):
        talk_track_lines.append(f"- {item}")
    talk_track_md = write_markdown(artifact_root / "claim_to_demo_talk_track.md", talk_track_lines)

    return {
        "json": json_path,
        "md": md_path,
        "talk_track_md": talk_track_md,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.claim_to_demo",
        description="Build a flagship claim-to-demo review pack from the current Taste OS, control-room, and research artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output root containing analysis/, analytics/, history/, and runs/ subdirectories.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    report = build_claim_to_demo_report(output_root)
    paths = write_claim_to_demo_artifacts(report, output_dir=output_root)
    print(f"claim_to_demo_json={paths['json']}")
    print(f"claim_to_demo_md={paths['md']}")
    print(f"claim_to_demo_talk_track_md={paths['talk_track_md']}")
    print(f"primary_claim={_coerce_dict(report.get('primary_claim')).get('key', '')}")
    print(f"flagship_demo={_coerce_dict(report.get('flagship_demo')).get('label', '')}")
    print(f"coherence_aligned={_coerce_dict(report.get('coherence')).get('aligned', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
