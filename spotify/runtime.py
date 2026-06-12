from __future__ import annotations

import importlib.metadata
import multiprocessing
import os
import platform
import sys


_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))
_DEVICE_MODE_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "cpu_only": "cpu",
    "gpu": "gpu",
    "gpu_only": "gpu",
}


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: 0, 1, false, true, no, yes, off, on; got {raw!r}.")


def _installed_dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _parse_python_version(raw: str) -> tuple[int, int]:
    parts = str(raw).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return (0, 0)
    return (major, minor)


def detect_acceleration_environment() -> dict[str, object]:
    return {
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "force_cpu": _env_flag("SPOTIFY_FORCE_CPU"),
        "device_mode": resolve_tensorflow_device_mode(),
        "tensorflow_version": _installed_dist_version("tensorflow"),
        "tensorflow_metal_version": _installed_dist_version("tensorflow-metal"),
        "tensorflow_macos_version": _installed_dist_version("tensorflow-macos"),
    }


def resolve_tensorflow_device_mode() -> str:
    if _env_flag("SPOTIFY_FORCE_CPU"):
        return "cpu"
    raw = os.getenv("SPOTIFY_TF_DEVICE_MODE", "auto").strip().lower()
    try:
        return _DEVICE_MODE_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "SPOTIFY_TF_DEVICE_MODE must be one of: auto, cpu, gpu "
            f"(cpu_only/gpu_only are accepted aliases); got {raw!r}."
        ) from exc


def build_acceleration_hint(summary: dict[str, object], *, logical_gpu_count: int) -> str | None:
    if str(summary.get("device_mode", "")).lower() == "cpu" or bool(summary.get("force_cpu")):
        return "TensorFlow CPU mode is enabled (SPOTIFY_FORCE_CPU/device mode), so GPU devices are hidden."

    if summary.get("platform") == "darwin" and summary.get("machine") == "arm64" and logical_gpu_count <= 0:
        python_version = str(summary.get("python_version", "unknown"))
        metal_version = summary.get("tensorflow_metal_version")
        if metal_version:
            return (
                "No TensorFlow GPU devices were detected on Apple Silicon even though "
                f"tensorflow-metal={metal_version} is installed. Run `python scripts/check_acceleration.py` "
                "to inspect plugin conflicts and environment state."
            )
        if _parse_python_version(python_version) >= (3, 13):
            return (
                "No TensorFlow GPU devices were detected on Apple Silicon. This environment is using "
                f"Python {python_version} without tensorflow-metal. Prefer a separate Python 3.11 Metal venv "
                "and run `python scripts/check_acceleration.py` for setup guidance."
            )
        return (
            "No TensorFlow GPU devices were detected on Apple Silicon and tensorflow-metal is not installed. "
            "Run `python scripts/check_acceleration.py` for setup guidance."
        )

    return None


def should_disable_deep_models_for_cpu_only_full_pass(summary: dict[str, object]) -> tuple[bool, str | None]:
    policy = os.getenv("SPOTIFY_FULL_DEEP_MODE_POLICY", "auto").strip().lower()
    if policy in ("0", "false", "no", "off", "disable", "disabled"):
        return True, "policy=off"
    if policy in ("1", "true", "yes", "on", "enable", "enabled"):
        return False, None
    if policy != "auto":
        return False, None

    if bool(summary.get("force_cpu")):
        return True, "force_cpu"
    if str(summary.get("device_mode", "")).lower() == "cpu":
        return True, "device_mode_cpu"

    if summary.get("platform") != "darwin" or summary.get("machine") != "arm64":
        return False, None

    python_version = str(summary.get("python_version", "0.0"))
    if _parse_python_version(python_version) < (3, 13):
        return False, None

    if summary.get("tensorflow_metal_version"):
        return False, None

    return True, "apple_silicon_python_313_no_tensorflow_metal"


def should_prefer_compatibility_python_for_deep_runtime(summary: dict[str, object]) -> tuple[bool, str | None]:
    if bool(summary.get("force_cpu")) or str(summary.get("device_mode", "")).lower() == "cpu":
        return False, None
    if summary.get("platform") != "darwin" or summary.get("machine") != "arm64":
        return False, None
    python_version = str(summary.get("python_version", "0.0"))
    if _parse_python_version(python_version) < (3, 13):
        return False, None
    if summary.get("tensorflow_metal_version"):
        return False, None
    return True, "apple_silicon_python_313_no_tensorflow_metal"


def should_fail_fast_for_deep_tensorflow_runtime(summary: dict[str, object]) -> tuple[bool, str | None]:
    policy = os.getenv("SPOTIFY_FAIL_FAST_PY313_DEEP", "auto").strip().lower()
    if policy in ("0", "false", "no", "off"):
        return False, None
    if bool(summary.get("force_cpu")) or str(summary.get("device_mode", "")).lower() == "cpu":
        return False, None
    prefer_alt, reason = should_prefer_compatibility_python_for_deep_runtime(summary)
    if not prefer_alt:
        return False, None
    if policy in ("1", "true", "yes", "on"):
        return True, reason
    return True, reason


def configure_process_env() -> None:
    device_mode = resolve_tensorflow_device_mode()
    if device_mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_METAL_ENABLE_FUSED_OPERATIONS"] = "0"
    elif device_mode == "gpu":
        # Let TensorFlow place GPU work on dedicated threads when available.
        os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
        os.environ.setdefault("TF_GPU_THREAD_COUNT", "4")

    # Avoid noisy and repeated physical-core detection in joblib/loky.
    if os.getenv("LOKY_MAX_CPU_COUNT") is None:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count() or 1)


def _positive_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}.")
    return value


def configure_tensorflow_threading(tf, logger=None) -> tuple[int | None, int | None]:
    intra_threads = _positive_int_env("TF_NUM_INTRAOP_THREADS")
    inter_threads = _positive_int_env("TF_NUM_INTEROP_THREADS")
    cpu_count = multiprocessing.cpu_count() or 1

    if intra_threads is None and inter_threads is None:
        if logger is not None:
            logger.info("TensorFlow CPU threads: auto (logical_cpus=%d)", cpu_count)
        return None, None

    try:
        if intra_threads is not None:
            tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        if inter_threads is not None:
            tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "TensorFlow thread configuration failed. Set TF_NUM_INTRAOP_THREADS and "
            "TF_NUM_INTEROP_THREADS before importing TensorFlow, using positive integers."
        ) from exc

    if logger is not None:
        logger.info(
            "TensorFlow CPU threads: intra_op=%s inter_op=%s logical_cpus=%d",
            intra_threads if intra_threads is not None else "auto",
            inter_threads if inter_threads is not None else "auto",
            cpu_count,
        )
    return intra_threads, inter_threads


def configure_tensorflow_devices(tf, logger=None) -> tuple[list[object], list[object], str]:
    device_mode = resolve_tensorflow_device_mode()
    try:
        physical_gpus = list(tf.config.list_physical_devices("GPU"))
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not enumerate physical GPU devices. "
            "Run `python scripts/check_acceleration.py --json` for environment diagnostics."
        ) from exc

    if device_mode == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "TensorFlow was initialized before CPU mode could hide GPU devices. "
                "Set SPOTIFY_TF_DEVICE_MODE=cpu (or SPOTIFY_FORCE_CPU=1) before starting Python "
                "and avoid importing TensorFlow early."
            ) from exc
    else:
        for device in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except (RuntimeError, ValueError) as exc:
                if logger is not None:
                    logger.debug("TensorFlow memory-growth configuration skipped for %s: %s", device, exc)

    try:
        logical_gpus = list(tf.config.list_logical_devices("GPU"))
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not initialize logical GPU devices. "
            "Run `python scripts/check_acceleration.py --json` for environment diagnostics."
        ) from exc
    if device_mode == "cpu" and logical_gpus:
        raise RuntimeError(
            "TensorFlow CPU mode was requested, but a logical GPU is still visible. "
            "Ensure CPU mode is set before Python starts and before any TensorFlow import."
        )
    if device_mode == "gpu" and not logical_gpus:
        raise RuntimeError(
            "TensorFlow GPU mode requested, but no logical GPU is available. "
            "On Apple Silicon, use a native arm64 Python environment with a compatible "
            "tensorflow-metal installation. Run `python scripts/check_acceleration.py --json`."
        )
    if device_mode == "auto" and physical_gpus and not logical_gpus and logger is not None:
        logger.warning(
            "TensorFlow discovered %d physical GPU(s), but none initialized; auto mode will use CPU.",
            len(physical_gpus),
        )
    return physical_gpus, logical_gpus, device_mode


def resolve_mixed_precision_policy(
    summary: dict[str, object],
    *,
    logical_gpu_count: int,
) -> tuple[str, str]:
    override = os.getenv("SPOTIFY_MIXED_PRECISION", "auto").strip().lower()
    if override in ("off", "0", "false", "no"):
        return "float32", "disabled"
    if override in ("on", "1", "true", "yes"):
        if logical_gpu_count <= 0:
            raise ValueError(
                "SPOTIFY_MIXED_PRECISION=on requires a visible TensorFlow GPU. "
                "Use SPOTIFY_MIXED_PRECISION=off for CPU execution."
            )
        return "mixed_float16", "enabled"
    if override != "auto":
        raise ValueError(
            "SPOTIFY_MIXED_PRECISION must be one of: auto, off, on "
            f"(boolean aliases are accepted); got {override!r}."
        )
    if logical_gpu_count <= 0:
        return "float32", "auto(no_gpu)"
    if (
        summary.get("platform") == "darwin"
        and summary.get("machine") == "arm64"
    ):
        return "float32", "auto(metal_stability)"
    return "mixed_float16", "auto(gpu)"


def load_tensorflow_runtime(logger):
    configure_process_env()
    summary = detect_acceleration_environment()
    fail_fast, fail_fast_reason = should_fail_fast_for_deep_tensorflow_runtime(summary)
    if fail_fast:
        raise RuntimeError(
            "Refusing to initialize TensorFlow for deep training on Apple Silicon with "
            f"Python {summary.get('python_version')} and no tensorflow-metal "
            f"({fail_fast_reason}). Re-run via `scripts/run_everything.sh`, "
            "set `PYTHON_BIN=.venv-metal/bin/python`, or override with "
            "`SPOTIFY_FAIL_FAST_PY313_DEEP=off`."
        )
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow failed to import. Install a compatible TensorFlow runtime and run "
            "`python scripts/check_acceleration.py --json` to inspect package and plugin errors. "
            f"Original error: {exc!r}"
        ) from exc

    configure_tensorflow_threading(tf, logger=logger)

    physical_gpus, logical_gpus, device_mode = configure_tensorflow_devices(tf, logger=logger)
    logical_gpu_count = len(logical_gpus)
    precision_policy, precision_reason = resolve_mixed_precision_policy(
        summary,
        logical_gpu_count=logical_gpu_count,
    )
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy(precision_policy)
        logger.info(
            "TensorFlow precision policy: %s (%s).",
            precision_policy,
            precision_reason,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to set TensorFlow precision policy {precision_policy!r} "
            f"({precision_reason}). Set SPOTIFY_MIXED_PRECISION=off for float32 execution."
        ) from exc

    logger.info(
        "TensorFlow devices: mode=%s physical_gpus=%d logical_gpus=%d",
        device_mode,
        len(physical_gpus),
        logical_gpu_count,
    )
    hint = build_acceleration_hint(summary, logical_gpu_count=logical_gpu_count)
    if hint:
        if device_mode == "cpu":
            logger.info(hint)
        else:
            logger.warning(hint)

    try:
        if sys.platform != "darwin":
            tf.config.optimizer.set_jit(True)
        else:
            logger.info("Skipping XLA JIT on macOS/Metal due to platform support.")
    except Exception as exc:  # pragma: no cover - backend dependent
        logger.warning("XLA JIT enable failed: %s", exc)

    return tf


def select_distribution_strategy(tf, logger=None):
    mode = os.getenv("SPOTIFY_DISTRIBUTION_STRATEGY", "auto").strip().lower()
    if mode in ("default", "none"):
        return tf.distribute.get_strategy()

    try:
        logical_gpus = tf.config.list_logical_devices("GPU")
    except Exception:
        logical_gpus = []

    if mode in ("mirrored", "auto") and len(logical_gpus) > 1:
        if logger is not None:
            logger.info("Using MirroredStrategy across %d GPUs.", len(logical_gpus))
        return tf.distribute.MirroredStrategy()

    return tf.distribute.get_strategy()
