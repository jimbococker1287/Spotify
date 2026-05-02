from __future__ import annotations

import importlib.metadata
import multiprocessing
import os
import platform
import sys


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
        "force_cpu": os.getenv("SPOTIFY_FORCE_CPU", "0").strip().lower() in ("1", "true", "yes"),
        "tensorflow_version": _installed_dist_version("tensorflow"),
        "tensorflow_metal_version": _installed_dist_version("tensorflow-metal"),
        "tensorflow_macos_version": _installed_dist_version("tensorflow-macos"),
    }


def build_acceleration_hint(summary: dict[str, object], *, logical_gpu_count: int) -> str | None:
    if bool(summary.get("force_cpu")):
        return "SPOTIFY_FORCE_CPU is enabled, so TensorFlow will stay on CPU."

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

    if summary.get("platform") != "darwin" or summary.get("machine") != "arm64":
        return False, None

    python_version = str(summary.get("python_version", "0.0"))
    if _parse_python_version(python_version) < (3, 13):
        return False, None

    if summary.get("tensorflow_metal_version"):
        return False, None

    return True, "apple_silicon_python_313_no_tensorflow_metal"


def should_prefer_compatibility_python_for_deep_runtime(summary: dict[str, object]) -> tuple[bool, str | None]:
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
    prefer_alt, reason = should_prefer_compatibility_python_for_deep_runtime(summary)
    if not prefer_alt:
        return False, None
    if policy in ("1", "true", "yes", "on"):
        return True, reason
    return True, reason


def configure_process_env() -> None:
    force_cpu = os.getenv("SPOTIFY_FORCE_CPU", "0").strip().lower() in ("1", "true", "yes")
    if force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_METAL_ENABLE_FUSED_OPERATIONS", "0")
    else:
        # Let TensorFlow place GPU work on dedicated threads when available.
        os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
        os.environ.setdefault("TF_GPU_THREAD_COUNT", "4")
        os.environ.setdefault("TF_METAL_ENABLE_FUSED_OPERATIONS", "1")

    # Avoid noisy and repeated physical-core detection in joblib/loky.
    if os.getenv("LOKY_MAX_CPU_COUNT") is None:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count() or 1)
    os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
    os.environ.setdefault("TF_GPU_THREAD_COUNT", "4")


def load_tensorflow_runtime(logger):
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
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies first: pip install -r requirements.txt"
        ) from exc

    def _to_pos_int(value: str | None, fallback: int) -> int:
        try:
            parsed = int(str(value))
            if parsed > 0:
                return parsed
        except Exception:
            pass
        return fallback

    cpu_count = multiprocessing.cpu_count()
    intra_raw = os.getenv("TF_NUM_INTRAOP_THREADS")
    inter_raw = os.getenv("TF_NUM_INTEROP_THREADS")
    if intra_raw is not None or inter_raw is not None:
        intra_threads = _to_pos_int(intra_raw, cpu_count)
        inter_threads = _to_pos_int(inter_raw, max(2, min(8, cpu_count // 2)))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
            tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
            logger.info("TensorFlow CPU threads: intra_op=%d inter_op=%d logical_cpus=%d", intra_threads, inter_threads, cpu_count)
        except Exception as exc:
            logger.warning("Unable to set TensorFlow thread config: %s", exc)
    else:
        logger.info("TensorFlow CPU threads: auto (logical_cpus=%d)", cpu_count)

    logical_gpu_count = 0
    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
        for device in physical_gpus:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        physical_gpus = []

    policy_override = os.getenv("SPOTIFY_MIXED_PRECISION", "auto").strip().lower()
    if policy_override in ("off", "0", "false"):
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("float32")
            logger.info("Mixed precision disabled via SPOTIFY_MIXED_PRECISION.")
        except Exception as exc:
            logger.warning("Failed to set float32 policy: %s", exc)
    elif policy_override in ("on", "1", "true"):
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled via SPOTIFY_MIXED_PRECISION.")
        except Exception as exc:
            logger.warning("Failed to set mixed_float16 policy: %s", exc)
    else:
        try:
            from tensorflow.keras import mixed_precision

            logical_gpus = tf.config.list_logical_devices("GPU")
            logical_gpu_count = len(logical_gpus)
            if logical_gpus:
                mixed_precision.set_global_policy("mixed_float16")
                logger.info("Mixed precision auto-enabled (GPUs detected: %d).", len(logical_gpus))
            else:
                logger.info("Using TensorFlow default precision policy.")
        except Exception as exc:
            logger.warning("Mixed precision auto policy setup failed: %s", exc)
            logical_gpus = []

    if not logical_gpu_count:
        try:
            logical_gpu_count = len(tf.config.list_logical_devices("GPU"))
        except Exception:
            logical_gpu_count = 0

    logger.info(
        "TensorFlow GPU devices: physical=%d logical=%d",
        len(physical_gpus),
        logical_gpu_count,
    )
    hint = build_acceleration_hint(summary, logical_gpu_count=logical_gpu_count)
    if hint:
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
