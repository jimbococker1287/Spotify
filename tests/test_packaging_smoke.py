from __future__ import annotations

from importlib.metadata import entry_points


def test_console_scripts_are_exposed() -> None:
    scripts = {entry_point.name: entry_point.value for entry_point in entry_points(group="console_scripts")}

    assert scripts["spotify-lab"] == "spotify.cli:main"
    assert scripts["spotify-predict"] == "spotify.predict_next:main"
    assert scripts["spotify-serve"] == "spotify.predict_service:main"
    assert scripts["spotify-public-insights"] == "spotify.public_insights:main"
    assert scripts["spotify-compare-public"] == "spotify.compare_public:main"
    assert scripts["spotify-control-room"] == "spotify.control_room:main"
    assert scripts["spotify-taste-os-demo"] == "spotify.taste_os_demo:main"
    assert scripts["spotify-taste-os-showcase"] == "spotify.taste_os_showcase:main"
