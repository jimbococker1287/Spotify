from __future__ import annotations

from spotify.runtime import (
    build_acceleration_hint,
    should_disable_deep_models_for_cpu_only_full_pass,
    should_fail_fast_for_deep_tensorflow_runtime,
    should_prefer_compatibility_python_for_deep_runtime,
)


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


def test_should_disable_deep_models_for_cpu_only_full_pass_on_python_313_without_metal(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_FULL_DEEP_MODE_POLICY", raising=False)

    disable, reason = should_disable_deep_models_for_cpu_only_full_pass(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "tensorflow_metal_version": None,
        }
    )

    assert disable is True
    assert reason == "apple_silicon_python_313_no_tensorflow_metal"


def test_should_disable_deep_models_for_cpu_only_full_pass_honors_policy_on(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FULL_DEEP_MODE_POLICY", "on")

    disable, reason = should_disable_deep_models_for_cpu_only_full_pass(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": True,
            "tensorflow_metal_version": None,
        }
    )

    assert disable is False
    assert reason is None


def test_should_prefer_compatibility_python_for_deep_runtime_on_python_313_without_metal() -> None:
    prefer_alt, reason = should_prefer_compatibility_python_for_deep_runtime(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "tensorflow_metal_version": None,
        }
    )

    assert prefer_alt is True
    assert reason == "apple_silicon_python_313_no_tensorflow_metal"


def test_should_fail_fast_for_deep_tensorflow_runtime_defaults_on_for_python_313(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_FAIL_FAST_PY313_DEEP", raising=False)

    fail_fast, reason = should_fail_fast_for_deep_tensorflow_runtime(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "tensorflow_metal_version": None,
        }
    )

    assert fail_fast is True
    assert reason == "apple_silicon_python_313_no_tensorflow_metal"


def test_should_fail_fast_for_deep_tensorflow_runtime_honors_policy_off(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FAIL_FAST_PY313_DEEP", "off")

    fail_fast, reason = should_fail_fast_for_deep_tensorflow_runtime(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "tensorflow_metal_version": None,
        }
    )

    assert fail_fast is False
    assert reason is None
