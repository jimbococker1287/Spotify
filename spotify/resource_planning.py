from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PROFILE_NAMES = ("auto", "cpu", "gpu")
GPU_PROBE_CACHE_TTL_SECONDS = 300
GPU_PROBE_TIMEOUT_SECONDS = 15
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
OPTIONAL_RESOURCE_ENV_KEYS = (
    "SPOTIFY_SKLEARN_NJOBS",
    "TF_GPU_THREAD_COUNT",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class HostResources:
    system: str
    machine: str
    logical_cpus: int
    total_ram_gb: int

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"


@dataclass(frozen=True)
class PythonRuntime:
    executable: str
    available: bool
    version: str = "unknown"
    has_tensorflow: bool = False
    has_tensorflow_metal: bool = False
    tensorflow_version: str | None = None
    tensorflow_metal_version: str | None = None
    gpu_count: int | None = None
    probe_error: str | None = None


@dataclass(frozen=True)
class ResourcePlan:
    requested_profile: str
    resolved_profile: str
    device: str
    host: HostResources
    python: PythonRuntime
    environment: dict[str, str]
    overrides: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"Resource plan: requested={self.requested_profile} resolved={self.resolved_profile} "
            f"python={self.python.executable} device={self.device} "
            f"cpus={self.host.logical_cpus} ram_gb={self.host.total_ram_gb}"
        )

    def report(self) -> str:
        env = self.environment
        gpu_count = "unknown" if self.python.gpu_count is None else str(self.python.gpu_count)
        lines = [
            "Full-run resource preflight",
            f"  profile: requested={self.requested_profile} resolved={self.resolved_profile}",
            f"  python: {self.python.executable} (Python {self.python.version})",
            f"  device: {self.device} (visible_gpus={gpu_count})",
            f"  host: logical_cpus={self.host.logical_cpus} total_ram_gb={self.host.total_ram_gb}",
            (
                "  workers: "
                f"classical={env.get('SPOTIFY_CLASSICAL_MODEL_WORKERS', 'auto')} "
                f"max_classical={env.get('SPOTIFY_MAX_CLASSICAL_WORKERS', 'auto')} "
                f"backtest={env.get('SPOTIFY_BACKTEST_WORKERS', 'auto')} "
                f"optuna_jobs={env.get('SPOTIFY_OPTUNA_JOBS', 'auto')} "
                f"optuna_models={env.get('SPOTIFY_OPTUNA_MODEL_WORKERS', 'auto')} "
                f"sklearn_jobs={env.get('SPOTIFY_SKLEARN_NJOBS', 'auto')}"
            ),
            (
                "  threads: "
                f"tf_intra={env.get('TF_NUM_INTRAOP_THREADS', 'auto')} "
                f"tf_inter={env.get('TF_NUM_INTEROP_THREADS', 'auto')} "
                f"tf_data={env.get('SPOTIFY_TF_DATA_THREADPOOL', 'auto')} "
                f"tf_gpu={env.get('TF_GPU_THREAD_COUNT', 'auto')} "
                f"blas={env.get('OMP_NUM_THREADS', 'auto')}"
            ),
            (
                "  tf cache: "
                f"mode={env.get('SPOTIFY_TF_DATA_CACHE', 'auto')} "
                f"fraction={env.get('SPOTIFY_TF_DATA_CACHE_FRACTION', 'auto')} "
                f"prefetch={env.get('SPOTIFY_TF_PREFETCH', 'auto')} "
                f"steps_per_execution={env.get('SPOTIFY_STEPS_PER_EXECUTION', 'auto')}"
            ),
            (
                "  run caches: "
                f"prepared={env.get('SPOTIFY_CACHE_PREPARED', 'auto')} "
                f"classical={env.get('SPOTIFY_CACHE_CLASSICAL', 'auto')} "
                f"deep={env.get('SPOTIFY_CACHE_DEEP', 'auto')} "
                f"optuna={env.get('SPOTIFY_CACHE_OPTUNA', 'auto')} "
                f"backtest={env.get('SPOTIFY_CACHE_BACKTEST', 'auto')} "
                f"retrieval={env.get('SPOTIFY_CACHE_RETRIEVAL', 'auto')} "
                f"shap={env.get('SPOTIFY_CACHE_SHAP', 'auto')}"
            ),
        ]
        if self.overrides:
            lines.append(f"  overrides: {', '.join(self.overrides)}")
        if self.warnings:
            lines.extend(f"  warning: {warning}" for warning in self.warnings)
        if self.errors:
            lines.extend(f"  error: {error}" for error in self.errors)
        return "\n".join(lines)


def _nonempty_environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {key: str(value) for key, value in environ.items() if str(value) != ""}


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def _is_false(value: str | None) -> bool:
    return str(value or "").strip().lower() in FALSE_VALUES


def _normalized_device_mode(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"cpu", "cpu_only"}:
        return "cpu"
    if normalized in {"gpu", "gpu_only"}:
        return "gpu"
    if normalized == "auto":
        return "auto"
    return None


def _int_value(value: str | None, fallback: int) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return fallback


def _balanced_classical_workers(host: HostResources) -> int:
    if host.logical_cpus >= 12:
        cpu_cap = 8
    elif host.logical_cpus >= 8:
        cpu_cap = 6
    elif host.logical_cpus >= 4:
        cpu_cap = 2
    else:
        cpu_cap = 1

    if host.total_ram_gb <= 0:
        memory_cap = 4
    elif host.total_ram_gb < 12:
        memory_cap = 1
    elif host.total_ram_gb < 18:
        memory_cap = 2
    elif host.total_ram_gb < 26:
        memory_cap = 3
    else:
        memory_cap = 4
    return min(cpu_cap, memory_cap)


def _cpu_classical_workers(host: HostResources) -> int:
    cpus = host.logical_cpus
    ram = host.total_ram_gb
    if ram >= 24:
        if cpus >= 12:
            return 6
        if cpus >= 8:
            return 5
        if cpus >= 6:
            return 4
        return 2
    if ram >= 16:
        if cpus >= 10:
            return 4
        if cpus >= 8:
            return 3
        return 2
    if ram >= 12:
        return 3 if cpus >= 8 else 2
    return 1


def _profile_defaults(profile: str, host: HostResources) -> dict[str, str]:
    cpus = host.logical_cpus
    low_unified_memory = host.is_apple_silicon and 0 < host.total_ram_gb <= 18

    if profile == "cpu":
        classical_workers = _cpu_classical_workers(host)
        backtest_workers = 4 if classical_workers >= 5 else 3 if classical_workers >= 4 else classical_workers
        optuna_jobs = 4 if classical_workers >= 4 else classical_workers
        optuna_model_workers = 2 if host.total_ram_gb >= 16 and classical_workers >= 4 else 1
        tf_intra = min(cpus, 8)
        tf_inter = 2 if cpus >= 8 else 1
        tf_data_threads = 4 if cpus >= 10 else 3 if cpus >= 8 else 0
        cache_fraction = "0.50"
        cache_mode = "auto"
        prefetch = "auto"
        steps_per_execution = "64"
    elif profile == "gpu":
        classical_workers = _cpu_classical_workers(host)
        if low_unified_memory:
            classical_workers = min(classical_workers, 2)
            backtest_workers = 1
            optuna_jobs = 2
            optuna_model_workers = 1
            tf_intra = min(cpus, 4)
            tf_inter = 1
            tf_data_threads = 2
            cache_fraction = "0.25"
            cache_mode = "off"
            prefetch = "1"
            steps_per_execution = "16"
        else:
            backtest_workers = 4 if classical_workers >= 5 else 3 if classical_workers >= 4 else classical_workers
            optuna_jobs = 4 if classical_workers >= 4 else classical_workers
            optuna_model_workers = 2 if host.total_ram_gb >= 16 and classical_workers >= 4 else 1
            tf_intra = min(cpus, 8)
            tf_inter = 2 if cpus >= 8 else 1
            tf_data_threads = 4 if cpus >= 10 else 3 if cpus >= 8 else 0
            cache_fraction = "0.50"
            cache_mode = "auto"
            prefetch = "auto"
            steps_per_execution = "64"
    else:
        classical_workers = _balanced_classical_workers(host)
        if cpus >= 10 and host.total_ram_gb >= 24 and classical_workers > 2:
            backtest_workers = 3
        elif classical_workers > 2:
            backtest_workers = 2
        else:
            backtest_workers = classical_workers
        optuna_jobs = 2 if classical_workers > 2 else classical_workers
        optuna_model_workers = 2 if cpus >= 8 and host.total_ram_gb >= 24 else 1
        tf_intra = cpus
        tf_inter = 4 if cpus > 8 else 2 if cpus > 2 else 1
        tf_data_threads = 0
        cache_fraction = "0.40"
        cache_mode = "auto"
        prefetch = "auto"
        steps_per_execution = "64"

    defaults = {
        "SPOTIFY_CLASSICAL_MODEL_WORKERS": str(classical_workers),
        "SPOTIFY_MAX_CLASSICAL_WORKERS": str(classical_workers),
        "SPOTIFY_BACKTEST_WORKERS": str(backtest_workers),
        "SPOTIFY_OPTUNA_JOBS": str(optuna_jobs),
        "SPOTIFY_OPTUNA_MODEL_WORKERS": str(optuna_model_workers),
        "TF_NUM_INTRAOP_THREADS": str(tf_intra),
        "TF_NUM_INTEROP_THREADS": str(tf_inter),
        "SPOTIFY_TF_DATA_THREADPOOL": str(tf_data_threads),
        "SPOTIFY_TF_DATA_CACHE": cache_mode,
        "SPOTIFY_TF_DATA_CACHE_FRACTION": cache_fraction,
        "SPOTIFY_TF_PREFETCH": prefetch,
        "SPOTIFY_STEPS_PER_EXECUTION": steps_per_execution,
        "LOKY_MAX_CPU_COUNT": str(cpus),
        "SPOTIFY_CACHE_PREPARED": "1",
        "SPOTIFY_CACHE_BACKTEST": "1",
        "SPOTIFY_CACHE_CLASSICAL": "1",
        "SPOTIFY_CACHE_DEEP": "1",
        "SPOTIFY_CACHE_DEEP_REPORTING": "1",
        "SPOTIFY_CACHE_OPTUNA": "1",
        "SPOTIFY_CACHE_RETRIEVAL": "1",
        "SPOTIFY_CACHE_SHAP": "1",
        "SPOTIFY_ISOLATE_MPL_CACHE": "0",
    }
    if profile in {"cpu", "gpu"}:
        defaults["SPOTIFY_MIXED_PRECISION"] = "off"
        defaults["SPOTIFY_FULL_DEEP_MODE_POLICY"] = "on"
    else:
        defaults["SPOTIFY_MIXED_PRECISION"] = "auto"
        defaults["SPOTIFY_FULL_DEEP_MODE_POLICY"] = "auto"
    if profile == "gpu":
        defaults["TF_GPU_THREAD_COUNT"] = "2" if low_unified_memory else "4"
        if low_unified_memory:
            defaults["SPOTIFY_SKLEARN_NJOBS"] = "1"
    return defaults


def _runtime_supports_gpu(runtime: PythonRuntime) -> bool:
    return runtime.available and runtime.has_tensorflow and bool(runtime.gpu_count and runtime.gpu_count > 0)


def build_resource_plan(
    requested_profile: str,
    *,
    environ: Mapping[str, str],
    host: HostResources,
    default_python: PythonRuntime,
    metal_python: PythonRuntime | None = None,
) -> ResourcePlan:
    requested = requested_profile.strip().lower()
    if requested not in PROFILE_NAMES:
        raise ValueError(f"Unknown resource profile {requested_profile!r}; choose auto, cpu, or gpu.")

    source_environment = _nonempty_environment(environ)
    explicit_python = source_environment.get("PYTHON_BIN")
    auto_route_python = not _is_false(source_environment.get("SPOTIFY_AUTO_ROUTE_TF_PYTHON"))
    selected_python = default_python
    if explicit_python:
        selected_python = default_python
    elif requested == "gpu" and metal_python is not None:
        selected_python = metal_python
    elif (
        requested == "auto"
        and auto_route_python
        and metal_python is not None
        and _runtime_supports_gpu(metal_python)
    ):
        selected_python = metal_python

    forced_cpu = _is_true(source_environment.get("SPOTIFY_FORCE_CPU"))
    device_override = _normalized_device_mode(source_environment.get("SPOTIFY_TF_DEVICE_MODE"))
    if forced_cpu:
        device = "cpu"
    elif device_override in {"cpu", "gpu"}:
        device = device_override
    elif requested == "cpu":
        device = "cpu"
    elif requested == "gpu":
        device = "gpu"
    elif _runtime_supports_gpu(selected_python):
        device = "gpu"
    else:
        device = "auto"

    if device == "gpu":
        resolved_profile = "gpu"
    elif requested == "cpu" or forced_cpu or device_override == "cpu":
        resolved_profile = "cpu"
    else:
        resolved_profile = "auto"

    defaults = _profile_defaults(resolved_profile, host)
    if device == "cpu":
        defaults.update(
            {
                "SPOTIFY_FORCE_CPU": "1" if requested == "cpu" else "0",
                "SPOTIFY_TF_DEVICE_MODE": "cpu",
            }
        )
    elif device == "gpu":
        defaults.update({"SPOTIFY_FORCE_CPU": "0", "SPOTIFY_TF_DEVICE_MODE": "gpu"})
    else:
        defaults.update({"SPOTIFY_FORCE_CPU": "0", "SPOTIFY_TF_DEVICE_MODE": "auto"})

    environment = dict(defaults)
    overrides: list[str] = []
    for key in defaults:
        if key in source_environment:
            environment[key] = source_environment[key]
            overrides.append(key)
    for key in OPTIONAL_RESOURCE_ENV_KEYS:
        if key in source_environment:
            environment[key] = source_environment[key]
            overrides.append(key)
    environment["PYTHON_BIN"] = explicit_python or selected_python.executable
    if explicit_python:
        overrides.append("PYTHON_BIN")
    elif requested == "auto" and selected_python is metal_python and selected_python is not default_python:
        environment["SPOTIFY_TF_COMPAT_VENV_ROUTED"] = "1"
    if "SPOTIFY_AUTO_ROUTE_TF_PYTHON" in source_environment:
        overrides.append("SPOTIFY_AUTO_ROUTE_TF_PYTHON")

    classical_workers = _int_value(
        environment.get("SPOTIFY_CLASSICAL_MODEL_WORKERS"),
        _balanced_classical_workers(host),
    )
    low_unified_memory = host.is_apple_silicon and 0 < host.total_ram_gb <= 18
    if "SPOTIFY_MAX_CLASSICAL_WORKERS" not in source_environment:
        environment["SPOTIFY_MAX_CLASSICAL_WORKERS"] = str(classical_workers)
    if resolved_profile in {"cpu", "gpu"} and not (
        resolved_profile == "gpu" and low_unified_memory
    ):
        if "SPOTIFY_BACKTEST_WORKERS" not in source_environment:
            environment["SPOTIFY_BACKTEST_WORKERS"] = str(
                4 if classical_workers >= 5 else 3 if classical_workers >= 4 else classical_workers
            )
        if "SPOTIFY_OPTUNA_JOBS" not in source_environment:
            environment["SPOTIFY_OPTUNA_JOBS"] = str(4 if classical_workers >= 4 else classical_workers)
        if "SPOTIFY_OPTUNA_MODEL_WORKERS" not in source_environment:
            environment["SPOTIFY_OPTUNA_MODEL_WORKERS"] = str(
                2 if host.total_ram_gb >= 16 and classical_workers >= 4 else 1
            )
    elif resolved_profile == "auto":
        if "SPOTIFY_BACKTEST_WORKERS" not in source_environment:
            if host.logical_cpus >= 10 and host.total_ram_gb >= 24 and classical_workers > 2:
                backtest_workers = 3
            elif classical_workers > 2:
                backtest_workers = 2
            else:
                backtest_workers = classical_workers
            environment["SPOTIFY_BACKTEST_WORKERS"] = str(backtest_workers)
        if "SPOTIFY_OPTUNA_JOBS" not in source_environment:
            environment["SPOTIFY_OPTUNA_JOBS"] = str(2 if classical_workers > 2 else classical_workers)
    if "SPOTIFY_SKLEARN_NJOBS" not in source_environment:
        environment["SPOTIFY_SKLEARN_NJOBS"] = "1" if classical_workers > 1 else "-1"

    if classical_workers > 1:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if key in source_environment:
                environment[key] = source_environment[key]
                overrides.append(key)
            else:
                environment[key] = "1"

    warnings: list[str] = []
    errors: list[str] = []
    if not selected_python.available:
        errors.append(f"Python executable is not available: {selected_python.executable}")
    elif not selected_python.has_tensorflow:
        errors.append(f"TensorFlow is not installed for {selected_python.executable}.")

    if requested == "gpu" and not host.is_apple_silicon and not explicit_python:
        errors.append("The gpu resource profile expects Apple Silicon with a Metal-enabled Python environment.")
    if requested == "gpu" and host.is_apple_silicon and metal_python is None and not explicit_python:
        errors.append("Missing .venv-metal/bin/python. Run `bash scripts/setup_metal_venv.sh` first.")
    if device == "gpu" and not _runtime_supports_gpu(selected_python):
        detail = f": {selected_python.probe_error}" if selected_python.probe_error else ""
        errors.append(f"No TensorFlow GPU is visible from {selected_python.executable}{detail}")
    if requested == "gpu" and device != "gpu":
        warnings.append("The requested gpu profile was superseded by an explicit CPU environment override.")
    if requested == "cpu" and device == "gpu":
        warnings.append("The requested cpu profile was superseded by SPOTIFY_TF_DEVICE_MODE=gpu.")
    if requested == "auto" and resolved_profile == "auto":
        warnings.append("No usable GPU was detected; TensorFlow remains in auto placement with balanced CPU/RAM limits.")

    return ResourcePlan(
        requested_profile=requested,
        resolved_profile=resolved_profile,
        device=device,
        host=host,
        python=selected_python,
        environment=environment,
        overrides=tuple(sorted(set(overrides))),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def detect_host_resources() -> HostResources:
    logical_cpus = os.cpu_count() or 1
    total_ram_bytes = 0
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            total_ram_bytes = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            total_ram_bytes = 0
    if total_ram_bytes <= 0:
        try:
            total_ram_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            total_ram_bytes = 0
    return HostResources(
        system=platform.system(),
        machine=platform.machine().lower(),
        logical_cpus=logical_cpus,
        total_ram_gb=int(total_ram_bytes // (1024**3)),
    )


def _probe_metadata(executable: str) -> PythonRuntime:
    resolved = shutil.which(executable) if os.sep not in executable else executable
    if not resolved or not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        return PythonRuntime(executable=executable, available=False)
    executable_path = Path(resolved)
    venv_root = executable_path.parent.parent
    pyvenv_config = venv_root / "pyvenv.cfg"
    if pyvenv_config.is_file():
        try:
            config = {
                key.strip(): value.strip()
                for line in pyvenv_config.read_text(encoding="utf-8").splitlines()
                if "=" in line
                for key, value in (line.split("=", 1),)
            }
            site_packages = next((venv_root / "lib").glob("python*/site-packages"))
            dist_info_names = {path.name.lower() for path in site_packages.glob("*.dist-info")}
        except (OSError, StopIteration):
            pass
        else:
            def _dist_version(distribution: str) -> str | None:
                prefix = f"{distribution.replace('-', '_').lower()}-"
                for name in sorted(dist_info_names):
                    if name.startswith(prefix) and name.endswith(".dist-info"):
                        return name[len(prefix) : -len(".dist-info")]
                return None

            tensorflow_version = _dist_version("tensorflow") or _dist_version("tensorflow_macos")
            tensorflow_metal_version = _dist_version("tensorflow_metal")
            return PythonRuntime(
                executable=executable,
                available=True,
                version=config.get("version", "unknown"),
                has_tensorflow=tensorflow_version is not None,
                has_tensorflow_metal=tensorflow_metal_version is not None,
                tensorflow_version=tensorflow_version,
                tensorflow_metal_version=tensorflow_metal_version,
            )
    probe = """
import importlib.metadata
import importlib.util
import json
import sys

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "version": ".".join(str(part) for part in sys.version_info[:3]),
    "tensorflow": importlib.util.find_spec("tensorflow") is not None,
    "tensorflow_version": version("tensorflow") or version("tensorflow-macos"),
    "tensorflow_metal_version": version("tensorflow-metal"),
}))
"""
    try:
        result = subprocess.run(
            [resolved, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", "1")},
        )
        payload = json.loads(result.stdout.strip())
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        return PythonRuntime(executable=executable, available=True, probe_error=str(exc))
    return PythonRuntime(
        executable=executable,
        available=True,
        version=str(payload.get("version", "unknown")),
        has_tensorflow=bool(payload.get("tensorflow")),
        has_tensorflow_metal=bool(payload.get("tensorflow_metal_version")),
        tensorflow_version=(
            str(payload["tensorflow_version"]) if payload.get("tensorflow_version") else None
        ),
        tensorflow_metal_version=(
            str(payload["tensorflow_metal_version"])
            if payload.get("tensorflow_metal_version")
            else None
        ),
    )


def _gpu_probe_cache_identity(
    runtime: PythonRuntime,
    *,
    environ: Mapping[str, str],
) -> dict[str, object]:
    try:
        executable = str(Path(runtime.executable).resolve())
    except OSError:
        executable = runtime.executable
    return {
        "executable": executable,
        "python_version": runtime.version,
        "tensorflow_version": runtime.tensorflow_version,
        "tensorflow_metal_version": runtime.tensorflow_metal_version,
        "python_no_user_site": environ.get("PYTHONNOUSERSITE", "1"),
    }


def _gpu_probe_cache_ttl_seconds(environ: Mapping[str, str]) -> int:
    raw = str(environ.get("SPOTIFY_RESOURCE_GPU_PROBE_TTL_SECONDS", "")).strip()
    if not raw:
        return GPU_PROBE_CACHE_TTL_SECONDS
    try:
        return max(0, int(raw))
    except ValueError:
        return GPU_PROBE_CACHE_TTL_SECONDS


def _gpu_probe_timeout_seconds(environ: Mapping[str, str]) -> int:
    raw = str(environ.get("SPOTIFY_RESOURCE_GPU_PROBE_TIMEOUT_SECONDS", "")).strip()
    if not raw:
        return GPU_PROBE_TIMEOUT_SECONDS
    try:
        return max(1, int(raw))
    except ValueError:
        return GPU_PROBE_TIMEOUT_SECONDS


def _load_cached_gpu_probe(
    runtime: PythonRuntime,
    *,
    cache_path: Path | None,
    environ: Mapping[str, str],
) -> PythonRuntime | None:
    if cache_path is None or _is_true(environ.get("SPOTIFY_RESOURCE_REFRESH_GPU_PROBE")):
        return None
    ttl_seconds = _gpu_probe_cache_ttl_seconds(environ)
    if ttl_seconds <= 0 or not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        age_seconds = time.time() - float(payload["checked_at"])
        gpu_count = int(payload["gpu_count"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if age_seconds < 0 or age_seconds > ttl_seconds:
        return None
    if payload.get("identity") != _gpu_probe_cache_identity(runtime, environ=environ):
        return None
    return PythonRuntime(
        executable=runtime.executable,
        available=runtime.available,
        version=runtime.version,
        has_tensorflow=runtime.has_tensorflow,
        has_tensorflow_metal=runtime.has_tensorflow_metal,
        tensorflow_version=runtime.tensorflow_version,
        tensorflow_metal_version=runtime.tensorflow_metal_version,
        gpu_count=gpu_count,
    )


def _save_gpu_probe_cache(
    runtime: PythonRuntime,
    *,
    cache_path: Path | None,
    environ: Mapping[str, str],
) -> None:
    if cache_path is None or runtime.gpu_count is None or runtime.probe_error:
        return
    payload = {
        "identity": _gpu_probe_cache_identity(runtime, environ=environ),
        "gpu_count": runtime.gpu_count,
        "checked_at": time.time(),
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.tmp")
        temporary_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary_path.replace(cache_path)
    except OSError:
        return


def _with_gpu_probe(
    runtime: PythonRuntime,
    *,
    cache_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PythonRuntime:
    if not runtime.available or not runtime.has_tensorflow:
        return runtime
    effective_environ = os.environ if environ is None else environ
    cached = _load_cached_gpu_probe(
        runtime,
        cache_path=cache_path,
        environ=effective_environ,
    )
    if cached is not None:
        return cached
    probe = """
import json
import tensorflow as tf
print(json.dumps({"gpu_count": len(tf.config.list_logical_devices("GPU"))}))
"""
    try:
        result = subprocess.run(
            [runtime.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=_gpu_probe_timeout_seconds(effective_environ),
            env={
                **os.environ,
                **effective_environ,
                "PYTHONNOUSERSITE": effective_environ.get("PYTHONNOUSERSITE", "1"),
                "TF_CPP_MIN_LOG_LEVEL": "3",
            },
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        probed_runtime = PythonRuntime(
            executable=runtime.executable,
            available=runtime.available,
            version=runtime.version,
            has_tensorflow=runtime.has_tensorflow,
            has_tensorflow_metal=runtime.has_tensorflow_metal,
            tensorflow_version=runtime.tensorflow_version,
            tensorflow_metal_version=runtime.tensorflow_metal_version,
            gpu_count=int(payload.get("gpu_count", 0)),
        )
        _save_gpu_probe_cache(
            probed_runtime,
            cache_path=cache_path,
            environ=effective_environ,
        )
        return probed_runtime
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError, ValueError) as exc:
        return PythonRuntime(
            executable=runtime.executable,
            available=runtime.available,
            version=runtime.version,
            has_tensorflow=runtime.has_tensorflow,
            has_tensorflow_metal=runtime.has_tensorflow_metal,
            tensorflow_version=runtime.tensorflow_version,
            tensorflow_metal_version=runtime.tensorflow_metal_version,
            gpu_count=0,
            probe_error=str(exc),
        )


def _version_tuple(version: str) -> tuple[int, int]:
    try:
        major, minor, *_ = version.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return (0, 0)


def detect_python_runtimes(
    *,
    root_dir: Path,
    host: HostResources,
    environ: Mapping[str, str],
    requested_profile: str,
) -> tuple[PythonRuntime, PythonRuntime | None]:
    explicit_python = str(environ.get("PYTHON_BIN", "")).strip()
    if explicit_python:
        default_path = explicit_python
    elif (root_dir / ".venv/bin/python").is_file():
        default_path = str(root_dir / ".venv/bin/python")
    else:
        default_path = shutil.which("python3") or "python3"

    default_python = _probe_metadata(default_path)
    metal_path = root_dir / ".venv-metal/bin/python"
    same_python = False
    if metal_path.is_file():
        try:
            same_python = Path(default_path).resolve() == metal_path.resolve()
        except OSError:
            same_python = False
        metal_python = default_python if same_python else _probe_metadata(str(metal_path))
    else:
        metal_python = None

    forced_cpu = _is_true(environ.get("SPOTIFY_FORCE_CPU"))
    device_override = _normalized_device_mode(environ.get("SPOTIFY_TF_DEVICE_MODE"))
    auto_route_python = not _is_false(environ.get("SPOTIFY_AUTO_ROUTE_TF_PYTHON"))
    needs_gpu_probe = not forced_cpu and (
        device_override == "gpu"
        or requested_profile == "gpu"
        or (requested_profile == "auto" and auto_route_python and device_override != "cpu")
    )
    configured_cache_path = str(environ.get("SPOTIFY_RESOURCE_GPU_PROBE_CACHE_PATH", "")).strip()
    if configured_cache_path:
        gpu_probe_cache_path = Path(configured_cache_path).expanduser()
        if not gpu_probe_cache_path.is_absolute():
            gpu_probe_cache_path = root_dir / gpu_probe_cache_path
    else:
        gpu_probe_cache_path = (
            root_dir / "outputs" / "cache" / "resource_planning" / "gpu_probe.json"
        )

    if needs_gpu_probe and host.is_apple_silicon and metal_python is not None and metal_python.has_tensorflow:
        metal_python = _with_gpu_probe(
            PythonRuntime(
                executable=metal_python.executable,
                available=metal_python.available,
                version=metal_python.version,
                has_tensorflow=metal_python.has_tensorflow,
                has_tensorflow_metal=metal_python.has_tensorflow_metal,
                tensorflow_version=metal_python.tensorflow_version,
                tensorflow_metal_version=metal_python.tensorflow_metal_version,
            ),
            cache_path=gpu_probe_cache_path,
            environ=environ,
        )
        if same_python:
            default_python = metal_python

    unsafe_default_probe = (
        host.is_apple_silicon
        and _version_tuple(default_python.version) >= (3, 13)
        and not default_python.has_tensorflow_metal
    )
    if needs_gpu_probe and not same_python and not unsafe_default_probe:
        if explicit_python or not _runtime_supports_gpu(metal_python or PythonRuntime("", False)):
            default_python = _with_gpu_probe(
                default_python,
                cache_path=gpu_probe_cache_path,
                environ=environ,
            )
    return default_python, metal_python


def render_shell(plan: ResourcePlan) -> str:
    exports = dict(plan.environment)
    exports.update(
        {
            "SPOTIFY_RESOURCE_PROFILE_REQUESTED": plan.requested_profile,
            "SPOTIFY_RESOURCE_PROFILE_RESOLVED": plan.resolved_profile,
            "SPOTIFY_RESOURCE_DEVICE": plan.device,
            "SPOTIFY_RESOURCE_LOGICAL_CPUS": str(plan.host.logical_cpus),
            "SPOTIFY_RESOURCE_TOTAL_RAM_GB": str(plan.host.total_ram_gb),
            "SPOTIFY_RESOURCE_PYTHON_VERSION": plan.python.version,
            "SPOTIFY_RESOURCE_GPU_COUNT": (
                "unknown" if plan.python.gpu_count is None else str(plan.python.gpu_count)
            ),
            "SPOTIFY_RESOURCE_PLAN_SUMMARY": plan.summary(),
            "SPOTIFY_RESOURCE_PLAN_REPORT": plan.report(),
        }
    )
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(exports.items()))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve full-run Python, device, and resource settings.")
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="auto")
    parser.add_argument("--format", choices=("shell", "json", "text"), default="text")
    parser.add_argument("--root-dir", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root_dir = args.root_dir.resolve()
    host = detect_host_resources()
    default_python, metal_python = detect_python_runtimes(
        root_dir=root_dir,
        host=host,
        environ=os.environ,
        requested_profile=args.profile,
    )
    plan = build_resource_plan(
        args.profile,
        environ=os.environ,
        host=host,
        default_python=default_python,
        metal_python=metal_python,
    )
    if plan.errors:
        print(plan.report(), file=sys.stderr)
        return 2
    if args.format == "shell":
        print(render_shell(plan))
    elif args.format == "json":
        print(
            json.dumps(
                {
                    "requested_profile": plan.requested_profile,
                    "resolved_profile": plan.resolved_profile,
                    "device": plan.device,
                    "python": plan.python.executable,
                    "python_version": plan.python.version,
                    "gpu_count": plan.python.gpu_count,
                    "logical_cpus": plan.host.logical_cpus,
                    "total_ram_gb": plan.host.total_ram_gb,
                    "environment": plan.environment,
                    "overrides": plan.overrides,
                    "warnings": plan.warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(plan.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
