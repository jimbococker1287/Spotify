from __future__ import annotations

import subprocess
import sys


def test_public_insights_module_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spotify.public_insights", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "python -m spotify.public_insights" in result.stdout
    assert "explain-artists" in result.stdout
    assert "release-tracker" in result.stdout
    assert "playlist-view" in result.stdout
    assert "catalog-linkouts" in result.stdout
    assert "artist-graph" in result.stdout
    assert "release-inbox" in result.stdout
    assert "playlist-diff" in result.stdout
    assert "market-gap" in result.stdout
    assert "playlist-archive" in result.stdout
    assert "catalog-crosswalk" in result.stdout
    assert "media-explorer" in result.stdout
