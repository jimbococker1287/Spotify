from __future__ import annotations

from spotify.runtime import build_acceleration_hint


def test_build_acceleration_hint_forced_cpu() -> None:
    hint = build_acceleration_hint(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": True,
            "tensorflow_metal_version": None,
        },
        logical_gpu_count=0,
    )

    assert hint is not None
    assert "SPOTIFY_FORCE_CPU" in hint


def test_build_acceleration_hint_recommends_python_311_on_macos_without_gpu() -> None:
    hint = build_acceleration_hint(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "tensorflow_metal_version": None,
        },
        logical_gpu_count=0,
    )

    assert hint is not None
    assert "Python 3.11" in hint
    assert "tensorflow-metal" in hint


def test_build_acceleration_hint_none_when_gpu_is_visible() -> None:
    hint = build_acceleration_hint(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.11",
            "force_cpu": False,
            "tensorflow_metal_version": "1.2.0",
        },
        logical_gpu_count=1,
    )

    assert hint is None


def test_build_acceleration_hint_does_not_treat_python_39_as_newer_than_313() -> None:
    hint = build_acceleration_hint(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.9",
            "force_cpu": False,
            "tensorflow_metal_version": None,
        },
        logical_gpu_count=0,
    )

    assert hint is not None
    assert "Python 3.11" not in hint
