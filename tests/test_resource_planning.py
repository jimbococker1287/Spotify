from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import spotify.resource_planning as resource_planning
from spotify.resource_planning import (
    HostResources,
    PythonRuntime,
    build_resource_plan,
    render_shell,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _runtime(path: str, *, gpu_count: int = 0, metal: bool = False) -> PythonRuntime:
    return PythonRuntime(
        executable=path,
        available=True,
        version="3.11.13",
        has_tensorflow=True,
        has_tensorflow_metal=metal,
        gpu_count=gpu_count,
    )


def _cached_gpu_environment(tmp_path: Path) -> dict[str, str]:
    cache_path = tmp_path / "gpu_probe.json"
    runtime = resource_planning._probe_metadata(str(ROOT_DIR / ".venv-metal/bin/python"))
    resource_planning._save_gpu_probe_cache(
        PythonRuntime(
            executable=runtime.executable,
            available=runtime.available,
            version=runtime.version,
            has_tensorflow=runtime.has_tensorflow,
            has_tensorflow_metal=runtime.has_tensorflow_metal,
            tensorflow_version=runtime.tensorflow_version,
            tensorflow_metal_version=runtime.tensorflow_metal_version,
            gpu_count=1,
        ),
        cache_path=cache_path,
        environ={"PYTHONNOUSERSITE": "1"},
    )
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "SPOTIFY_RESOURCE_GPU_PROBE_CACHE_PATH": str(cache_path),
    }


def test_auto_profile_selects_metal_and_low_memory_limits() -> None:
    host = HostResources(system="Darwin", machine="arm64", logical_cpus=10, total_ram_gb=16)
    default_python = _runtime(".venv/bin/python")
    metal_python = _runtime(".venv-metal/bin/python", gpu_count=1, metal=True)

    plan = build_resource_plan(
        "auto",
        environ={},
        host=host,
        default_python=default_python,
        metal_python=metal_python,
    )

    assert plan.errors == ()
    assert plan.resolved_profile == "gpu"
    assert plan.python is metal_python
    assert plan.environment["SPOTIFY_CLASSICAL_MODEL_WORKERS"] == "2"
    assert plan.environment["SPOTIFY_BACKTEST_WORKERS"] == "1"
    assert plan.environment["SPOTIFY_TF_DATA_CACHE"] == "off"
    assert plan.environment["SPOTIFY_TF_DATA_CACHE_FRACTION"] == "0.25"
    assert plan.environment["TF_NUM_INTRAOP_THREADS"] == "4"
    assert plan.environment["SPOTIFY_TF_COMPAT_VENV_ROUTED"] == "1"


def test_cpu_profile_preserves_explicit_resource_overrides() -> None:
    host = HostResources(system="Darwin", machine="arm64", logical_cpus=10, total_ram_gb=16)
    environment = {
        "PYTHON_BIN": "/custom/python",
        "SPOTIFY_CLASSICAL_MODEL_WORKERS": "7",
        "SPOTIFY_TF_DATA_CACHE": "off",
        "TF_NUM_INTRAOP_THREADS": "3",
        "OMP_NUM_THREADS": "6",
    }

    plan = build_resource_plan(
        "cpu",
        environ=environment,
        host=host,
        default_python=_runtime("/custom/python"),
        metal_python=_runtime(".venv-metal/bin/python", gpu_count=1, metal=True),
    )

    assert plan.errors == ()
    assert plan.environment["PYTHON_BIN"] == "/custom/python"
    assert plan.environment["SPOTIFY_CLASSICAL_MODEL_WORKERS"] == "7"
    assert plan.environment["SPOTIFY_MAX_CLASSICAL_WORKERS"] == "7"
    assert plan.environment["SPOTIFY_BACKTEST_WORKERS"] == "4"
    assert plan.environment["SPOTIFY_OPTUNA_JOBS"] == "4"
    assert plan.environment["SPOTIFY_TF_DATA_CACHE"] == "off"
    assert plan.environment["TF_NUM_INTRAOP_THREADS"] == "3"
    assert plan.environment["OMP_NUM_THREADS"] == "6"
    assert {
        "PYTHON_BIN",
        "SPOTIFY_CLASSICAL_MODEL_WORKERS",
        "SPOTIFY_TF_DATA_CACHE",
        "TF_NUM_INTRAOP_THREADS",
        "OMP_NUM_THREADS",
    }.issubset(plan.overrides)


def test_auto_route_can_be_disabled() -> None:
    host = HostResources(system="Darwin", machine="arm64", logical_cpus=10, total_ram_gb=16)
    default_python = _runtime(".venv/bin/python")
    metal_python = _runtime(".venv-metal/bin/python", gpu_count=1, metal=True)

    plan = build_resource_plan(
        "auto",
        environ={"SPOTIFY_AUTO_ROUTE_TF_PYTHON": "off"},
        host=host,
        default_python=default_python,
        metal_python=metal_python,
    )

    assert plan.python is default_python
    assert plan.resolved_profile == "auto"
    assert plan.environment["SPOTIFY_TF_DEVICE_MODE"] == "auto"
    assert "SPOTIFY_AUTO_ROUTE_TF_PYTHON" in plan.overrides


def test_force_cpu_wins_over_conflicting_device_override() -> None:
    host = HostResources(system="Darwin", machine="arm64", logical_cpus=10, total_ram_gb=16)
    plan = build_resource_plan(
        "gpu",
        environ={"SPOTIFY_FORCE_CPU": "1", "SPOTIFY_TF_DEVICE_MODE": "gpu"},
        host=host,
        default_python=_runtime(".venv/bin/python"),
        metal_python=_runtime(".venv-metal/bin/python", gpu_count=1, metal=True),
    )

    assert plan.errors == ()
    assert plan.resolved_profile == "cpu"
    assert plan.device == "cpu"
    assert plan.environment["SPOTIFY_FORCE_CPU"] == "1"
    assert plan.environment["SPOTIFY_TF_DEVICE_MODE"] == "gpu"


def test_gpu_profile_reports_missing_visible_device() -> None:
    host = HostResources(system="Darwin", machine="arm64", logical_cpus=8, total_ram_gb=16)
    unavailable_metal = _runtime(".venv-metal/bin/python", gpu_count=0, metal=True)

    plan = build_resource_plan(
        "gpu",
        environ={},
        host=host,
        default_python=_runtime(".venv/bin/python"),
        metal_python=unavailable_metal,
    )

    assert plan.resolved_profile == "gpu"
    assert any("No TensorFlow GPU is visible" in error for error in plan.errors)


def test_shell_render_includes_resolved_plan_report() -> None:
    host = HostResources(system="Linux", machine="x86_64", logical_cpus=4, total_ram_gb=8)
    plan = build_resource_plan(
        "cpu",
        environ={},
        host=host,
        default_python=_runtime("/usr/bin/python3"),
    )

    shell = render_shell(plan)

    assert "SPOTIFY_RESOURCE_PLAN_REPORT" in shell
    assert "SPOTIFY_RESOURCE_PROFILE_RESOLVED=cpu" in shell
    assert "SPOTIFY_CLASSICAL_MODEL_WORKERS=1" in shell


def test_gpu_probe_reuses_fresh_package_aware_cache(tmp_path: Path, monkeypatch) -> None:
    runtime = PythonRuntime(
        executable="/test/python",
        available=True,
        version="3.11.13",
        has_tensorflow=True,
        has_tensorflow_metal=True,
        tensorflow_version="2.16.2",
        tensorflow_metal_version="1.2.0",
    )
    cache_path = tmp_path / "gpu_probe.json"
    calls = 0

    def _run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout='{"gpu_count": 1}\n', stderr="")

    monkeypatch.setattr(resource_planning.subprocess, "run", _run)

    first = resource_planning._with_gpu_probe(runtime, cache_path=cache_path, environ={})
    second = resource_planning._with_gpu_probe(runtime, cache_path=cache_path, environ={})

    assert first.gpu_count == 1
    assert second.gpu_count == 1
    assert calls == 1


def test_probe_metadata_reads_venv_distributions_without_starting_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    venv_root = tmp_path / ".venv-metal"
    python = venv_root / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    python.chmod(0o755)
    (venv_root / "pyvenv.cfg").write_text("version = 3.11.13\n", encoding="utf-8")
    site_packages = venv_root / "lib/python3.11/site-packages"
    (site_packages / "tensorflow-2.16.2.dist-info").mkdir(parents=True)
    (site_packages / "tensorflow_metal-1.2.0.dist-info").mkdir()

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("virtualenv metadata discovery should not start Python")

    monkeypatch.setattr(resource_planning.subprocess, "run", _unexpected_run)

    runtime = resource_planning._probe_metadata(str(python))

    assert runtime.available is True
    assert runtime.version == "3.11.13"
    assert runtime.has_tensorflow is True
    assert runtime.has_tensorflow_metal is True
    assert runtime.tensorflow_version == "2.16.2"
    assert runtime.tensorflow_metal_version == "1.2.0"


def test_gpu_probe_cache_invalidates_when_tensorflow_version_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "gpu_probe.json"
    calls = 0

    def _run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout='{"gpu_count": 1}\n', stderr="")

    monkeypatch.setattr(resource_planning.subprocess, "run", _run)
    base = PythonRuntime(
        executable="/test/python",
        available=True,
        version="3.11.13",
        has_tensorflow=True,
        has_tensorflow_metal=True,
        tensorflow_version="2.16.2",
        tensorflow_metal_version="1.2.0",
    )
    changed = PythonRuntime(
        executable=base.executable,
        available=True,
        version=base.version,
        has_tensorflow=True,
        has_tensorflow_metal=True,
        tensorflow_version="2.17.0",
        tensorflow_metal_version=base.tensorflow_metal_version,
    )

    resource_planning._with_gpu_probe(base, cache_path=cache_path, environ={})
    resource_planning._with_gpu_probe(changed, cache_path=cache_path, environ={})

    assert calls == 2


def test_cpu_launcher_dry_run_reports_overrides_and_command() -> None:
    python_bin = ROOT_DIR / ".venv/bin/python"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(python_bin),
        "SPOTIFY_CLASSICAL_MODEL_WORKERS": "3",
        "SPOTIFY_TF_DATA_CACHE": "off",
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/run_everything_cpu_boost.sh",
            "planner-launcher-test",
            "--dry-run",
            "--no-shap",
        ],
        cwd=ROOT_DIR,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "profile: requested=cpu resolved=cpu" in result.stdout
    assert "classical=3" in result.stdout
    assert "mode=off" in result.stdout
    assert "Command:" in result.stdout
    assert "--run-name planner-launcher-test" in result.stdout


def test_gpu_dry_run_does_not_duplicate_explicit_no_shap(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/run_everything.sh",
            "--resource-profile",
            "gpu",
            "--dry-run",
            "planner-gpu-test",
            "--no-shap",
        ],
        cwd=ROOT_DIR,
        env=_cached_gpu_environment(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    command_line = next(line for line in result.stdout.splitlines() if line.startswith("Command:"))
    assert command_line.count("--no-shap") == 1


def test_auto_profile_applies_gpu_shap_default_after_resolution(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/run_everything.sh",
            "--resource-profile",
            "auto",
            "--dry-run",
            "planner-auto-test",
        ],
        cwd=ROOT_DIR,
        env=_cached_gpu_environment(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "profile: requested=auto resolved=gpu" in result.stdout
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("Command:"))
    assert command_line.count("--no-shap") == 1


@pytest.mark.parametrize(
    "launcher",
    (
        "scripts/run_everything_balanced_full.sh",
        "scripts/run_everything_all_deep_low_ram.sh",
    ),
)
def test_specialized_full_launchers_stop_after_preflight(launcher: str) -> None:
    environment = {
        **os.environ,
        "PYTHON_BIN": str(ROOT_DIR / ".venv/bin/python"),
        "SPOTIFY_FORCE_CPU": "1",
    }

    result = subprocess.run(
        ["bash", launcher, "planner-preflight-test", "--preflight"],
        cwd=ROOT_DIR,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "Full-run resource preflight" in result.stdout
    assert "Run dir:" not in result.stdout
