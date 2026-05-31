from __future__ import annotations

import json
from pathlib import Path

from spotify.project_health import build_project_health, main


def _touch(root: Path, relative_path: str, text: str = "ok") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_project_health_writes_scorecard_queue_and_hygiene(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    output_dir = project_root / "outputs"
    project_root.mkdir()

    _touch(
        project_root,
        "pyproject.toml",
        """
[project.scripts]
spotify-taste-os-demo = "spotify.taste_os_demo:main"
spotify-taste-os-serve = "spotify.taste_os_service:main"

[tool.ruff]
exclude = ["build"]
""",
    )
    _touch(
        project_root,
        "Makefile",
        """
taste-os-demo:
taste-os-showcase:
serve-taste-os:
""",
    )
    _touch(project_root, ".gitignore", ".DS_Store\n")
    _touch(project_root, "build/lib/spotify/generated.py")
    _touch(project_root, ".DS_Store", "finder noise")

    for relative_path in (
        "spotify/taste_os_demo.py",
        "spotify/taste_os_demo_core.py",
        "spotify/taste_os_service.py",
        "spotify/taste_os_page.py",
        "spotify/digital_twin.py",
        "spotify/journey_planner.py",
        "spotify/safe_policy.py",
        "tests/test_taste_os_demo.py",
        "tests/test_taste_os_service.py",
        "tests/test_digital_twin.py",
        "tests/test_safe_policy.py",
        "docs/personal_taste_os.md",
        "docs/taste_os_demo_contract.md",
        "docs/taste_os_demo_walkthrough.md",
        "docs/taste_os_product_story.md",
    ):
        _touch(project_root, relative_path)

    _touch(output_dir, "analysis/taste_os_demo/taste_os_demo_focus_steady.json", "{}")
    _touch(output_dir, "analysis/taste_os_demo/showcase/taste_os_showcase.json", "{}")

    payload = build_project_health(project_root=project_root, output_dir=output_dir)
    result_root = output_dir / "analysis" / "project_health"
    scorecard = json.loads((result_root / "project_health_scorecard.json").read_text(encoding="utf-8"))
    queue = json.loads((result_root / "project_development_queue.json").read_text(encoding="utf-8"))
    hygiene = json.loads((result_root / "repository_hygiene.json").read_text(encoding="utf-8"))
    review = (result_root / "project_health_review.md").read_text(encoding="utf-8")

    taste_os = next(row for row in scorecard if row["surface_key"] == "taste_os")
    assert payload["surface_count"] == len(scorecard)
    assert taste_os["status"] == "ready"
    assert taste_os["artifact_score"] == 1.0
    assert "taste_os_demo_focus_steady.json" in taste_os["proof_artifacts"]
    assert taste_os["top_gap"] == "No anchor gap; next gap is depth, freshness, and repeated evidence."
    assert queue[0]["rank"] == 1
    assert not any(row["surface_key"] == "repo_hygiene" for row in queue)
    assert hygiene["build_lib_python_count"] == 1
    assert hygiene["ds_store_count"] == 1
    assert hygiene["efficiency_score"] == 1.0
    assert "Project Health Review" in review
    assert "Repository efficiency score: 1.0" in review


def test_project_health_main_prints_summary(tmp_path: Path, capsys) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _touch(project_root, "pyproject.toml", "[project.scripts]\n")
    _touch(project_root, "Makefile", "")

    result = main(
        [
            "--project-root",
            str(project_root),
            "--output-dir",
            str(project_root / "outputs"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "project_health_review=" in captured.out
    assert (project_root / "outputs" / "analysis" / "project_health" / "project_health_manifest.json").exists()
