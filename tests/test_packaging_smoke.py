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
    assert scripts["spotify-public-insights"] == "spotify.public_insights:main"
    assert scripts["spotify-compare-public"] == "spotify.compare_public:main"
    assert scripts["spotify-control-room"] == "spotify.control_room:main"
    assert scripts["spotify-research-claims"] == "spotify.research_claims:main"
    assert scripts["spotify-claim-to-demo"] == "spotify.claim_to_demo:main"
    assert scripts["spotify-front-door"] == "spotify.front_door:main"
    assert scripts["spotify-branch-portfolio"] == "spotify.branch_portfolio:main"
    assert scripts["spotify-outward-package"] == "spotify.outward_package:main"
    assert scripts["spotify-taste-os-demo"] == "spotify.taste_os_demo:main"
    assert scripts["spotify-taste-os-showcase"] == "spotify.taste_os_showcase:main"
    assert scripts["spotify-taste-os-serve"] == "spotify.taste_os_service:main"
    assert scripts["spotify-refresh-champion-gate"] == "spotify.champion_gate_refresh:main"


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
