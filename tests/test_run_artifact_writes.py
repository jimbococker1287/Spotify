from __future__ import annotations

from pathlib import Path

from spotify.run_artifacts import copy_file_if_changed, write_json, write_markdown


def test_write_json_skips_rewriting_identical_payload(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"a": 1, "b": [1, 2, 3]})
    first_mtime = path.stat().st_mtime_ns

    write_json(path, {"a": 1, "b": [1, 2, 3]})
    second_mtime = path.stat().st_mtime_ns

    assert first_mtime == second_mtime


def test_write_markdown_skips_rewriting_identical_text(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_markdown(path, ["# Report", "", "- ok"])
    first_mtime = path.stat().st_mtime_ns

    write_markdown(path, ["# Report", "", "- ok"])
    second_mtime = path.stat().st_mtime_ns

    assert first_mtime == second_mtime


def test_copy_file_if_changed_skips_identical_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("# Same\n", encoding="utf-8")
    destination = tmp_path / "dest.md"

    copy_file_if_changed(source, destination)
    first_mtime = destination.stat().st_mtime_ns
    copy_file_if_changed(source, destination)
    second_mtime = destination.stat().st_mtime_ns

    assert first_mtime == second_mtime

