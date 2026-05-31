from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
import subprocess
import sys
import zipfile


def test_console_scripts_are_exposed() -> None:
    scripts = {entry_point.name: entry_point.value for entry_point in entry_points(group="console_scripts")}

    assert scripts["spotify-lab"] == "spotify.cli:main"
    assert scripts["spotify-predict"] == "spotify.predict_next:main"
    assert scripts["spotify-serve"] == "spotify.predict_service:main"
    assert scripts["spotify-build-serving-bundle"] == "spotify.serving_bundle:main"
    assert scripts["spotify-serve-api"] == "spotify.service_api:main"
    assert scripts["spotify-predict-api"] == "spotify.service_api:main_predict"
    assert scripts["spotify-taste-os-api"] == "spotify.service_api:main_taste_os"
    assert scripts["spotify-deploy-release"] == "spotify.deployment_registry:main"
    assert scripts["spotify-release-readiness"] == "spotify.release_readiness:main"
    assert scripts["spotify-production-smoke"] == "spotify.production_smoke:main"
    assert scripts["spotify-public-insights"] == "spotify.public_insights:main"
    assert scripts["spotify-compare-public"] == "spotify.compare_public:main"
    assert scripts["spotify-control-room"] == "spotify.control_room:main"
    assert scripts["spotify-research-claims"] == "spotify.research_claims:main"
    assert scripts["spotify-claim-to-demo"] == "spotify.claim_to_demo:main"
    assert scripts["spotify-front-door"] == "spotify.front_door:main"
    assert scripts["spotify-branch-portfolio"] == "spotify.branch_portfolio:main"
    assert scripts["spotify-outward-package"] == "spotify.outward_package:main"
    assert scripts["spotify-day-90-launch"] == "spotify.day_90_launch:main"
    assert scripts["spotify-show-ready-backfill"] == "spotify.show_ready_backfill:main"
    assert scripts["spotify-show-ready-maintenance"] == "spotify.show_ready_maintenance:main"
    assert scripts["spotify-taste-os-demo"] == "spotify.taste_os_demo:main"
    assert scripts["spotify-taste-os-showcase"] == "spotify.taste_os_showcase:main"
    assert scripts["spotify-taste-os-serve"] == "spotify.taste_os_service:main"
    assert scripts["spotify-refresh-champion-gate"] == "spotify.champion_gate_refresh:main"
    assert scripts["spotify-listener-archetypes"] == "spotify.listener_archetypes:main"
    assert scripts["spotify-quant-decision-lab"] == "spotify.quant_decision_lab:main"
    assert scripts["spotify-creator-market-intelligence"] == "spotify.creator_market_intelligence:main"
    assert scripts["spotify-research-platform-lab"] == "spotify.research_platform_lab:main"
    assert scripts["spotify-project-health"] == "spotify.project_health:main"


def test_built_wheel_excludes_backup_modules_and_ds_store(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(dist_dir),
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel_file:
        wheel_names = set(wheel_file.namelist())

    forbidden = {
        "spotify/.DS_Store",
        "spotify/data 2.py",
        "spotify/taste_os_demo 2.py",
    }
    assert wheel_names.isdisjoint(forbidden)
