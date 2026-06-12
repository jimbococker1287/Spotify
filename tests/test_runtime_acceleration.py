from __future__ import annotations

import json
import os
import sys

import pytest

from scripts import check_acceleration
from spotify.runtime import (
    build_acceleration_hint,
    configure_process_env,
    configure_tensorflow_devices,
    configure_tensorflow_threading,
    resolve_mixed_precision_policy,
    resolve_tensorflow_device_mode,
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


def test_should_prefer_compatibility_python_allows_explicit_cpu_mode() -> None:
    prefer_alt, reason = should_prefer_compatibility_python_for_deep_runtime(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "device_mode": "cpu",
            "tensorflow_metal_version": None,
        }
    )

    assert prefer_alt is False
    assert reason is None


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


class _DummyTensorFlowConfig:
    def __init__(self, gpu_count: int, *, fail_visibility: bool = False):
        self.physical_gpus = [object() for _ in range(gpu_count)]
        self.visible_gpus = list(self.physical_gpus)
        self.fail_visibility = fail_visibility
        self.experimental = self
        self.threading = _DummyTensorFlowThreading()

    def list_physical_devices(self, device_type: str):
        return list(self.physical_gpus) if device_type == "GPU" else []

    def list_logical_devices(self, device_type: str):
        return list(self.visible_gpus) if device_type == "GPU" else []

    def set_visible_devices(self, devices, device_type: str):
        if self.fail_visibility:
            raise RuntimeError("runtime already initialized")
        if device_type == "GPU":
            self.visible_gpus = list(devices)

    def set_memory_growth(self, device, enabled: bool):
        assert device in self.physical_gpus
        assert enabled is True


class _DummyTensorFlowThreading:
    def __init__(self):
        self.intra_threads = None
        self.inter_threads = None

    def set_intra_op_parallelism_threads(self, value: int):
        self.intra_threads = value

    def set_inter_op_parallelism_threads(self, value: int):
        self.inter_threads = value


class _DummyTensorFlow:
    __version__ = "test"

    def __init__(self, gpu_count: int, *, fail_visibility: bool = False):
        self.config = _DummyTensorFlowConfig(gpu_count, fail_visibility=fail_visibility)


def test_resolve_tensorflow_device_mode_force_cpu_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "1")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "gpu")

    assert resolve_tensorflow_device_mode() == "cpu"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", "auto"),
        ("CPU", "cpu"),
        ("cpu_only", "cpu"),
        ("gpu_only", "gpu"),
    ],
)
def test_resolve_tensorflow_device_mode_aliases(monkeypatch, raw: str, expected: str) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", raw)

    assert resolve_tensorflow_device_mode() == expected


def test_resolve_tensorflow_device_mode_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "sometimes")

    with pytest.raises(ValueError, match="SPOTIFY_TF_DEVICE_MODE"):
        resolve_tensorflow_device_mode()


def test_resolve_tensorflow_device_mode_rejects_invalid_force_cpu(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "sometimes")

    with pytest.raises(ValueError, match="SPOTIFY_FORCE_CPU"):
        resolve_tensorflow_device_mode()


def test_configure_process_env_cpu_mode_overrides_visible_cuda_devices(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TF_METAL_ENABLE_FUSED_OPERATIONS", "1")

    configure_process_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["TF_METAL_ENABLE_FUSED_OPERATIONS"] == "0"


def test_configure_tensorflow_devices_hides_metal_gpu_in_cpu_mode(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "1")
    tf = _DummyTensorFlow(gpu_count=1)

    physical, logical, mode = configure_tensorflow_devices(tf)

    assert len(physical) == 1
    assert logical == []
    assert mode == "cpu"


def test_configure_tensorflow_devices_cpu_mode_fails_after_initialization(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "cpu")

    with pytest.raises(RuntimeError, match="before CPU mode could hide GPU devices"):
        configure_tensorflow_devices(_DummyTensorFlow(gpu_count=1, fail_visibility=True))


def test_configure_tensorflow_devices_requires_gpu_in_gpu_mode(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_FORCE_CPU", raising=False)
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "gpu")

    with pytest.raises(RuntimeError, match="no logical GPU"):
        configure_tensorflow_devices(_DummyTensorFlow(gpu_count=0))


def test_configure_tensorflow_devices_auto_mode_allows_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "auto")

    physical, logical, mode = configure_tensorflow_devices(_DummyTensorFlow(gpu_count=0))

    assert physical == []
    assert logical == []
    assert mode == "auto"


def test_configure_tensorflow_threading_applies_only_explicit_values(monkeypatch) -> None:
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "3")
    monkeypatch.delenv("TF_NUM_INTEROP_THREADS", raising=False)
    tf = _DummyTensorFlow(gpu_count=0)

    configured = configure_tensorflow_threading(tf)

    assert configured == (3, None)
    assert tf.config.threading.intra_threads == 3
    assert tf.config.threading.inter_threads is None


def test_configure_tensorflow_threading_rejects_non_positive_values(monkeypatch) -> None:
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "0")

    with pytest.raises(ValueError, match="TF_NUM_INTRAOP_THREADS"):
        configure_tensorflow_threading(_DummyTensorFlow(gpu_count=0))


def test_mixed_precision_auto_stays_float32_on_metal(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_MIXED_PRECISION", "auto")

    policy, reason = resolve_mixed_precision_policy(
        {
            "platform": "darwin",
            "machine": "arm64",
            "tensorflow_metal_version": None,
        },
        logical_gpu_count=1,
    )

    assert policy == "float32"
    assert reason == "auto(metal_stability)"


def test_mixed_precision_on_requires_visible_gpu(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_MIXED_PRECISION", "on")

    with pytest.raises(ValueError, match="requires a visible TensorFlow GPU"):
        resolve_mixed_precision_policy({}, logical_gpu_count=0)


def test_should_fail_fast_allows_explicit_cpu_mode(monkeypatch) -> None:
    monkeypatch.delenv("SPOTIFY_FAIL_FAST_PY313_DEEP", raising=False)

    fail_fast, reason = should_fail_fast_for_deep_tensorflow_runtime(
        {
            "platform": "darwin",
            "machine": "arm64",
            "python_version": "3.13",
            "force_cpu": False,
            "device_mode": "cpu",
            "tensorflow_metal_version": None,
        }
    )

    assert fail_fast is False
    assert reason is None


def test_preflight_json_mode_reports_invalid_device_mode(monkeypatch, capsys) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "sometimes")

    exit_code = check_acceleration.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 3
    assert payload["status"] == "error"
    assert payload["exit_code"] == 3
    assert "SPOTIFY_TF_DEVICE_MODE" in payload["errors"][0]


def test_preflight_cpu_mode_reports_hidden_gpu_as_ok(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_FORCE_CPU", "0")
    monkeypatch.setenv("SPOTIFY_TF_DEVICE_MODE", "cpu")
    monkeypatch.setitem(sys.modules, "tensorflow", _DummyTensorFlow(gpu_count=1))

    summary, exit_code = check_acceleration.collect_diagnostics()

    assert exit_code == 0
    assert summary["status"] == "ok"
    assert summary["device_mode"] == "cpu"
    assert summary["logical_gpus"] == []
