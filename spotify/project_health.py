from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Sequence

from .run_artifacts import write_csv_rows, write_json, write_markdown

ArtifactRequirement = str | tuple[str, ...]


@dataclass(frozen=True)
class ProjectSurface:
    key: str
    name: str
    audience: str
    thesis: str
    modules: tuple[str, ...]
    tests: tuple[str, ...]
    docs: tuple[str, ...]
    commands: tuple[str, ...] = ()
    make_targets: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRequirement, ...] = ()
    expansion: str = ""


PROJECT_SURFACES: tuple[ProjectSurface, ...] = (
    ProjectSurface(
        key="taste_os",
        name="Personal Taste OS",
        audience="product demo, adaptive listening, and portfolio storytelling",
        thesis="Turn listening history into explainable session steering, not only next-artist scoring.",
        modules=(
            "spotify/taste_os_demo.py",
            "spotify/taste_os_demo_core.py",
            "spotify/taste_os_service.py",
            "spotify/taste_os_page.py",
            "spotify/digital_twin.py",
            "spotify/journey_planner.py",
            "spotify/safe_policy.py",
        ),
        tests=(
            "tests/test_taste_os_demo.py",
            "tests/test_taste_os_service.py",
            "tests/test_digital_twin.py",
            "tests/test_safe_policy.py",
        ),
        docs=(
            "docs/personal_taste_os.md",
            "docs/taste_os_demo_contract.md",
            "docs/taste_os_demo_walkthrough.md",
            "docs/taste_os_product_story.md",
        ),
        commands=("spotify-taste-os-demo", "spotify-taste-os-serve"),
        make_targets=("taste-os-demo", "taste-os-showcase", "serve-taste-os"),
        artifacts=(
            ("analysis/taste_os_demo/taste_os_demo.json", "analysis/taste_os_demo/taste_os_demo_focus_steady.json"),
            ("analysis/taste_os_showcase/taste_os_showcase.json", "analysis/taste_os_demo/showcase/taste_os_showcase.json"),
        ),
        expansion="Productize mode-specific steering loops and make why-this-next copy the default explanation layer.",
    ),
    ProjectSurface(
        key="recommender_safety",
        name="Recommender Safety Platform",
        audience="reusable safety SDK, promotion governance, and model-risk review",
        thesis="Make drift, backtesting, promotion gates, and abstention reusable across sequence recommenders.",
        modules=(
            "spotify/recommender_safety.py",
            "spotify/safety_platform.py",
            "spotify/backtesting.py",
            "spotify/drift.py",
            "spotify/governance.py",
            "spotify/evaluation.py",
        ),
        tests=(
            "tests/test_recommender_safety_platform.py",
            "tests/test_drift_and_backtesting.py",
            "tests/test_governance_gate.py",
            "tests/test_uncertainty.py",
        ),
        docs=("docs/recommender_safety_platform.md", "docs/doctorate_roadmap.md", "docs/benchmark_contract.md"),
        commands=("spotify-refresh-champion-gate",),
        make_targets=("refresh-champion-gate", "benchmark-lock"),
        artifacts=(
            "history/backtest_history.csv",
            "history/benchmark_history.csv",
            "analysis/research_platform_lab/benchmark_lock_atlas.csv",
        ),
        expansion="Expose the minimum reusable API contract and add repeated evidence packs for promotion decisions.",
    ),
    ProjectSurface(
        key="control_room",
        name="Control Room",
        audience="operator review, scheduled run triage, and roadmap prioritization",
        thesis="Compress run artifacts into one operating readout with risks, cadence, and next bets.",
        modules=(
            "spotify/control_room.py",
            "spotify/control_room_core.py",
            "spotify/control_room_business.py",
            "spotify/control_room_rendering.py",
            "spotify/control_room_triage.py",
            "spotify/control_room_history.py",
        ),
        tests=("tests/test_control_room.py", "tests/test_control_room_cli_smoke.py", "tests/test_control_room_guard.py"),
        docs=("docs/control_room_operating_rhythm.md", "docs/project_threads.md"),
        commands=("spotify-control-room",),
        make_targets=("control-room", "control-room-guard", "regression-alert"),
        artifacts=("analytics/control_room.json", "analytics/control_room.md", "analytics/control_room_history.csv"),
        expansion="Make control-room output the default post-run cockpit and connect next bets to concrete validation commands.",
    ),
    ProjectSurface(
        key="creator_intelligence",
        name="Creator and Label Intelligence",
        audience="A&R briefs, scene strategy, and public catalog opportunity maps",
        thesis="Turn public metadata and listening behavior into scenes, adjacency, fan migration, and whitespace.",
        modules=(
            "spotify/creator_label_intelligence.py",
            "spotify/creator_label_intelligence_core.py",
            "spotify/creator_market_intelligence.py",
            "spotify/public_insights.py",
            "spotify/public_insights_handlers.py",
            "spotify/public_catalog.py",
        ),
        tests=(
            "tests/test_creator_label_intelligence.py",
            "tests/test_creator_market_intelligence.py",
            "tests/test_public_insights.py",
            "tests/test_public_catalog.py",
        ),
        docs=("docs/creator_label_intelligence_brief.md", "docs/creator_market_intelligence.md"),
        commands=("spotify-public-insights", "spotify-creator-market-intelligence"),
        make_targets=("public-insights", "creator-market-intelligence"),
        artifacts=(
            (
                "analysis/public_spotify/public_insights_index.json",
                "analysis/public_spotify/summary/public_insights_summary.json",
            ),
            "analysis/creator_market_intelligence/creator_market_manifest.json",
        ),
        expansion="Promote repeated report families into strategy cards for scenes, release lanes, and label concentration.",
    ),
    ProjectSurface(
        key="research_platform",
        name="Research Platform",
        audience="publication-grade experiments, claim support, and dissertation roadmap governance",
        thesis="Tie claims to code, benchmark locks, reproducible artifacts, and next experiments.",
        modules=(
            "spotify/research_platform_lab.py",
            "spotify/research_claims.py",
            "spotify/benchmark_contract.py",
            "spotify/research_artifacts.py",
            "spotify/retrieval_stack.py",
            "spotify/uncertainty.py",
        ),
        tests=(
            "tests/test_research_platform_lab.py",
            "tests/test_research_claims.py",
            "tests/test_benchmark_contract.py",
            "tests/test_retrieval_and_friction.py",
        ),
        docs=("docs/research_platform_lab.md", "docs/doctorate_roadmap.md", "docs/publication_outline.md"),
        commands=("spotify-research-platform-lab", "spotify-research-claims"),
        make_targets=("research-platform-lab", "research-claims", "benchmark-lock"),
        artifacts=(
            "analysis/research_platform_lab/research_claim_registry.csv",
            "analysis/research_platform_lab/research_next_experiments.json",
            ("analysis/research_claims/research_claims_report.json", "analysis/research_claims/research_claims.json"),
        ),
        expansion="Run repeated benchmark-lock evidence for the claims that still block publication-grade confidence.",
    ),
    ProjectSurface(
        key="analytics_quant",
        name="Analytics and Quant Decision Lab",
        audience="portfolio analytics, decision frontiers, and branch-health governance",
        thesis="Make cross-branch health and model/policy tradeoffs queryable from durable analytics artifacts.",
        modules=(
            "spotify/analytics_db.py",
            "spotify/analytics_warehouse.py",
            "spotify/quant_decision_lab.py",
            "spotify/scope_expansion_lab.py",
            "spotify/listener_archetypes.py",
        ),
        tests=(
            "tests/test_analytics_warehouse.py",
            "tests/test_quant_decision_lab.py",
            "tests/test_scope_expansion_lab.py",
            "tests/test_listener_archetypes.py",
        ),
        docs=("docs/analytics_db.md", "docs/data_science_quant.md", "docs/scope_expansion_lab.md"),
        commands=("spotify-quant-decision-lab", "spotify-scope-expansion-lab", "spotify-listener-archetypes"),
        make_targets=("analytics-db", "analytics-warehouse", "quant-decision-lab", "scope-expansion-lab"),
        artifacts=(
            "analytics/warehouse/warehouse_manifest.json",
            "analysis/quant_decision_lab/quant_decision_brief.json",
            "analysis/scope_expansion/branch_expansion_scorecard.csv",
        ),
        expansion="Turn branch health into a lightweight dashboard or notebook backed by the warehouse mart.",
    ),
    ProjectSurface(
        key="serving_deployment",
        name="Serving and Deployment",
        audience="local demos, service smoke tests, and production-shaped deployment paths",
        thesis="Keep prediction and Taste OS services deployable through stable APIs, bundles, and manifests.",
        modules=(
            "spotify/service_api.py",
            "spotify/predict_service.py",
            "spotify/predict_next.py",
            "spotify/serving.py",
            "spotify/serving_bundle.py",
            "spotify/deployment_registry.py",
            "spotify/release_readiness.py",
            "spotify/production_smoke.py",
        ),
        tests=(
            "tests/test_service_api.py",
            "tests/test_predict_service_validation.py",
            "tests/test_predict_service_cli_smoke.py",
            "tests/test_deploy_manifests.py",
            "tests/test_deployment_registry.py",
            "tests/test_release_readiness.py",
            "tests/test_production_smoke.py",
        ),
        docs=("deploy/README.md", "deploy/local/README.md", "deploy/kubernetes/README.md", "deploy/ecs/README.md"),
        commands=(
            "spotify-serve-api",
            "spotify-predict-api",
            "spotify-build-serving-bundle",
            "spotify-deploy-release",
            "spotify-release-readiness",
            "spotify-production-smoke",
        ),
        make_targets=("serve-api", "serve-predict", "build-serving-bundle", "deploy-release", "release-readiness", "production-smoke"),
        artifacts=(
            "models/champion/alias.json",
            (
                "analysis/production_smoke/production_smoke.json",
                "analysis/release_readiness/release_readiness_smoke.json",
                "releases/deployment_registry.json",
                "analysis/day_90_launch/canonical_artifact_manifest.json",
            ),
        ),
        expansion="Trend production-smoke latency and readiness outcomes across releases so deploy handoffs have history.",
    ),
)

SCORECARD_COLUMNS = [
    "surface_key",
    "surface_name",
    "audience",
    "status",
    "health_score",
    "code_score",
    "test_score",
    "doc_score",
    "surface_score",
    "artifact_score",
    "risk_score",
    "module_count",
    "test_count",
    "doc_count",
    "command_count",
    "make_target_count",
    "artifact_count",
    "top_gap",
    "next_step",
    "expansion_opportunity",
    "proof_modules",
    "proof_tests",
    "proof_docs",
    "proof_artifacts",
]

QUEUE_COLUMNS = [
    "rank",
    "surface_key",
    "surface_name",
    "initiative",
    "why_now",
    "success_metric",
    "command",
    "validation_command",
    "effort",
    "impact_score",
    "risk_reduction_score",
    "dependencies",
]


def _bounded(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def _ratio(found: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return _bounded(found / total)


def _existing_relative_paths(root: Path, paths: Sequence[str]) -> list[str]:
    return [path for path in paths if (root / path).exists()]


def _existing_artifact_paths(root: Path, requirements: Sequence[ArtifactRequirement]) -> list[str]:
    found: list[str] = []
    for requirement in requirements:
        candidates = (requirement,) if isinstance(requirement, str) else requirement
        for candidate in candidates:
            if (root / candidate).exists():
                found.append(candidate)
                break
    return found


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _script_names(project_root: Path) -> set[str]:
    text = _read_text(project_root / "pyproject.toml")
    return set(re.findall(r"^([A-Za-z0-9_.-]+)\s*=", text, flags=re.MULTILINE))


def _make_targets(project_root: Path) -> set[str]:
    text = _read_text(project_root / "Makefile")
    return set(re.findall(r"^([A-Za-z0-9_.-]+):(?:\s|$)", text, flags=re.MULTILINE))


def _status(score: float, risk_score: float) -> str:
    if score >= 0.82 and risk_score <= 0.22:
        return "ready"
    if score >= 0.64:
        return "develop"
    if score >= 0.40:
        return "needs_anchor"
    return "blocked"


def _gap_label(row: dict[str, object]) -> str:
    dimensions = [
        ("code", float(row["code_score"]), "Add or restore concrete code anchors."),
        ("tests", float(row["test_score"]), "Add focused tests for the riskiest behavior."),
        ("docs", float(row["doc_score"]), "Write a top-level framing doc with commands and artifacts."),
        ("surface", float(row["surface_score"]), "Expose the workflow through a command or Make target."),
        ("artifacts", float(row["artifact_score"]), "Generate named artifacts so progress is inspectable."),
    ]
    label, score, message = min(dimensions, key=lambda item: item[1])
    if score >= 0.99:
        return "No anchor gap; next gap is depth, freshness, and repeated evidence."
    return f"{label}: {message}"


def _next_step(surface: ProjectSurface, row: dict[str, object]) -> str:
    top_gap = str(row["top_gap"])
    if top_gap.startswith("tests:"):
        return f"Broaden {surface.name} coverage around the highest-risk path, then rerun `make test`."
    if top_gap.startswith("docs:"):
        return f"Add a concise {surface.name} operating doc with input artifacts, commands, and review criteria."
    if top_gap.startswith("surface:"):
        return f"Add a script or Make target so {surface.name} can be run without reading internals."
    if top_gap.startswith("artifacts:"):
        command = next(iter(surface.make_targets), "")
        return f"Run `{f'make {command}' if command else 'the surface command'}` and publish the expected artifacts."
    if top_gap.startswith("code:"):
        return f"Create the smallest code anchor that makes {surface.name} independently inspectable."
    return surface.expansion or f"Deepen {surface.name} with repeated evidence and clearer handoff artifacts."


def _score_surface(
    surface: ProjectSurface,
    *,
    project_root: Path,
    output_dir: Path,
    scripts: set[str],
    make_targets: set[str],
) -> dict[str, object]:
    proof_modules = _existing_relative_paths(project_root, surface.modules)
    proof_tests = _existing_relative_paths(project_root, surface.tests)
    proof_docs = _existing_relative_paths(project_root, surface.docs)
    proof_artifacts = _existing_artifact_paths(output_dir, surface.artifacts)
    command_count = sum(1 for command in surface.commands if command in scripts)
    make_target_count = sum(1 for target in surface.make_targets if target in make_targets)

    code_score = _ratio(len(proof_modules), len(surface.modules))
    test_score = _ratio(len(proof_tests), len(surface.tests))
    doc_score = _ratio(len(proof_docs), len(surface.docs))
    surface_score = _ratio(command_count + make_target_count, len(surface.commands) + len(surface.make_targets))
    artifact_score = _ratio(len(proof_artifacts), len(surface.artifacts))
    risk_score = _bounded((1.0 - test_score) * 0.42 + (1.0 - artifact_score) * 0.28 + (1.0 - surface_score) * 0.18 + (1.0 - doc_score) * 0.12)
    health_score = _bounded(
        code_score * 0.26
        + test_score * 0.26
        + doc_score * 0.20
        + surface_score * 0.16
        + artifact_score * 0.12
    )
    row: dict[str, object] = {
        "surface_key": surface.key,
        "surface_name": surface.name,
        "audience": surface.audience,
        "status": _status(health_score, risk_score),
        "health_score": health_score,
        "code_score": code_score,
        "test_score": test_score,
        "doc_score": doc_score,
        "surface_score": surface_score,
        "artifact_score": artifact_score,
        "risk_score": risk_score,
        "module_count": len(proof_modules),
        "test_count": len(proof_tests),
        "doc_count": len(proof_docs),
        "command_count": command_count,
        "make_target_count": make_target_count,
        "artifact_count": len(proof_artifacts),
        "expansion_opportunity": surface.expansion,
        "proof_modules": " | ".join(proof_modules),
        "proof_tests": " | ".join(proof_tests),
        "proof_docs": " | ".join(proof_docs),
        "proof_artifacts": " | ".join(proof_artifacts),
    }
    row["top_gap"] = _gap_label(row)
    row["next_step"] = _next_step(surface, row)
    return row


def _tracked_files(project_root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _repo_hygiene(project_root: Path) -> dict[str, object]:
    tracked = _tracked_files(project_root)
    build_python_files = sorted(str(path.relative_to(project_root)) for path in (project_root / "build" / "lib").rglob("*.py")) if (project_root / "build" / "lib").exists() else []
    local_dirs = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", ".venv-metal", "__pycache__", "mlruns", "outputs"}
    ds_store_files = sorted(
        str(path.relative_to(project_root))
        for path in project_root.rglob(".DS_Store")
        if not any(part in local_dirs for part in path.relative_to(project_root).parts)
    )
    tracked_build_files = [path for path in tracked if path.startswith("build/lib/")]
    tracked_ds_store_files = [path for path in tracked if path.endswith(".DS_Store")]
    pyproject_text = _read_text(project_root / "pyproject.toml")
    ruff_excludes_build = "exclude" in pyproject_text and "build" in pyproject_text
    gitignore_text = _read_text(project_root / ".gitignore")
    gitignore_mentions_build = bool(re.search(r"(^|\n)build/?(\n|$)", gitignore_text))

    actions: list[str] = []
    if tracked_build_files:
        actions.append("Remove tracked `build/lib` generated source from version control; keep Ruff excluding it while it remains ignored.")
    if tracked_ds_store_files:
        actions.append("Remove tracked `.DS_Store` files from the repository index.")
    if not gitignore_mentions_build:
        actions.append("Add `build/` to `.gitignore` after deciding how to handle tracked generated files.")
    if not actions:
        actions.append("Repository hygiene is clear enough for the next development pass.")

    risk_score = _bounded(
        min(len(tracked_build_files), 50) / 50.0 * 0.55
        + min(len(tracked_ds_store_files), 5) / 5.0 * 0.25
        + (0.0 if ruff_excludes_build else 0.20)
    )
    return {
        "build_lib_python_count": len(build_python_files),
        "ds_store_count": len(ds_store_files),
        "tracked_build_file_count": len(tracked_build_files),
        "tracked_ds_store_count": len(tracked_ds_store_files),
        "ruff_excludes_build": ruff_excludes_build,
        "gitignore_mentions_build": gitignore_mentions_build,
        "risk_score": risk_score,
        "efficiency_score": _bounded(1.0 - risk_score),
        "recommended_actions": actions,
        "tracked_ds_store_files": tracked_ds_store_files[:20],
        "tracked_build_file_examples": tracked_build_files[:20],
    }


def _queue_rows(scorecard: list[dict[str, object]], hygiene: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in scorecard:
        health = float(row["health_score"])
        risk = float(row["risk_score"])
        surface_key = str(row["surface_key"])
        make_target = {
            surface.key: next(iter(surface.make_targets), "")
            for surface in PROJECT_SURFACES
        }.get(surface_key, "")
        validation = "make test"
        if surface_key == "recommender_safety":
            validation = "python -m pytest tests/test_recommender_safety_platform.py tests/test_drift_and_backtesting.py"
        elif surface_key == "control_room":
            validation = "python -m pytest tests/test_control_room.py tests/test_control_room_guard.py"
        elif surface_key == "analytics_quant":
            validation = "python -m pytest tests/test_analytics_warehouse.py tests/test_scope_expansion_lab.py"
        elif surface_key == "serving_deployment":
            validation = "python -m pytest tests/test_service_api.py tests/test_deploy_manifests.py tests/test_deployment_registry.py tests/test_release_readiness.py tests/test_production_smoke.py"

        rows.append(
            {
                "rank": 0,
                "surface_key": surface_key,
                "surface_name": row["surface_name"],
                "initiative": row["next_step"],
                "why_now": row["top_gap"],
                "success_metric": "Health score improves while the generated artifacts and focused tests stay green.",
                "command": f"make {make_target}" if make_target else "",
                "validation_command": validation,
                "effort": "S" if health >= 0.80 else "M" if health >= 0.55 else "L",
                "impact_score": _bounded(0.45 + (1.0 - health) * 0.35 + risk * 0.20),
                "risk_reduction_score": _bounded(risk + (1.0 - float(row["test_score"])) * 0.25),
                "dependencies": "existing artifacts" if float(row["artifact_score"]) < 1.0 else "none",
            }
        )

    if float(hygiene.get("risk_score", 0.0)) > 0:
        rows.append(
            {
                "rank": 0,
                "surface_key": "repo_hygiene",
                "surface_name": "Repository Hygiene",
                "initiative": "Clean generated and OS-specific files from the project boundary.",
                "why_now": "Generated build output and/or `.DS_Store` files are visible in repository scans.",
                "success_metric": "`make lint` stays green without relying on generated-source noise.",
                "command": "",
                "validation_command": "make lint",
                "effort": "S",
                "impact_score": _bounded(0.62 + float(hygiene.get("risk_score", 0.0)) * 0.25),
                "risk_reduction_score": _bounded(float(hygiene.get("risk_score", 0.0))),
                "dependencies": "decide whether to delete tracked generated files",
            }
        )

    rows.sort(
        key=lambda item: (
            float(item["risk_reduction_score"]) * 0.55 + float(item["impact_score"]) * 0.45,
            str(item["surface_name"]),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _table(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = [str(row.get(column, "")).replace("\n", " ").replace("|", "/") for column in columns]
        body.append("| " + " | ".join(cells) + " |")
    return [header, divider, *body]


def _scorecard_markdown(scorecard: list[dict[str, object]], queue: list[dict[str, object]], hygiene: dict[str, object]) -> list[str]:
    ready_count = sum(1 for row in scorecard if row["status"] == "ready")
    average_health = _bounded(sum(float(row["health_score"]) for row in scorecard) / max(len(scorecard), 1))
    lines = [
        "# Project Health Review",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Surfaces reviewed: {len(scorecard)}",
        f"- Ready surfaces: {ready_count}",
        f"- Average health score: {average_health}",
        f"- Repository efficiency score: {hygiene.get('efficiency_score', 0.0)}",
        f"- Repository hygiene risk: {hygiene.get('risk_score', 0.0)}",
        "",
        "## Surface Scorecard",
        "",
        *_table(
            scorecard,
            [
                "surface_name",
                "status",
                "health_score",
                "risk_score",
                "top_gap",
                "next_step",
            ],
        ),
        "",
        "## Development Queue",
        "",
        *_table(
            queue[:8],
            [
                "rank",
                "surface_name",
                "initiative",
                "validation_command",
                "impact_score",
                "risk_reduction_score",
            ],
        ),
        "",
        "## Repository Hygiene",
        "",
    ]
    for action in hygiene.get("recommended_actions", []):
        lines.append(f"- {action}")
    return lines


def build_project_health(
    *,
    project_root: Path,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    logger = logger or logging.getLogger("spotify.project_health")
    project_root = project_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    result_root = output_dir / "analysis" / "project_health"
    scripts = _script_names(project_root)
    make_targets = _make_targets(project_root)

    scorecard = [
        _score_surface(
            surface,
            project_root=project_root,
            output_dir=output_dir,
            scripts=scripts,
            make_targets=make_targets,
        )
        for surface in PROJECT_SURFACES
    ]
    hygiene = _repo_hygiene(project_root)
    queue = _queue_rows(scorecard, hygiene)
    average_health = _bounded(sum(float(row["health_score"]) for row in scorecard) / max(len(scorecard), 1))
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "surface_count": len(scorecard),
        "ready_surface_count": sum(1 for row in scorecard if row["status"] == "ready"),
        "average_health_score": average_health,
        "repository_efficiency_score": hygiene.get("efficiency_score", 0.0),
        "scorecard": scorecard,
        "development_queue": queue,
        "repository_hygiene": hygiene,
    }
    paths = {
        "scorecard_csv": str(write_csv_rows(result_root / "project_health_scorecard.csv", scorecard, fieldnames=SCORECARD_COLUMNS)),
        "scorecard_json": str(write_json(result_root / "project_health_scorecard.json", scorecard)),
        "queue_csv": str(write_csv_rows(result_root / "project_development_queue.csv", queue, fieldnames=QUEUE_COLUMNS)),
        "queue_json": str(write_json(result_root / "project_development_queue.json", queue)),
        "hygiene_json": str(write_json(result_root / "repository_hygiene.json", hygiene)),
        "review_md": str(write_markdown(result_root / "project_health_review.md", _scorecard_markdown(scorecard, queue, hygiene))),
    }
    manifest = {
        "generated_at": generated_at,
        "surface_count": len(scorecard),
        "average_health_score": average_health,
        "repository_efficiency_score": hygiene.get("efficiency_score", 0.0),
        "top_queue_item": queue[0] if queue else None,
        "paths": paths,
    }
    paths["manifest_json"] = str(write_json(result_root / "project_health_manifest.json", manifest))
    payload["paths"] = paths
    logger.info("Wrote project health review to %s", result_root)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.project_health",
        description="Evaluate the project as a whole and write a development scorecard.",
    )
    parser.add_argument("--project-root", type=str, default=".", help="Repository root to inspect.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated artifacts.")
    parser.add_argument("--stdout-format", choices=("summary", "json"), default="summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    payload = build_project_health(
        project_root=Path(args.project_root),
        output_dir=Path(args.output_dir),
        logger=logging.getLogger("spotify.project_health"),
    )
    if args.stdout_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"project_health_review={payload['paths']['review_md']}")
        print(f"average_health_score={payload['average_health_score']}")
        top_item = payload["development_queue"][0] if payload["development_queue"] else {}
        if top_item:
            print(f"top_queue_item={top_item['surface_name']}: {top_item['initiative']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
