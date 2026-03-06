from __future__ import annotations

import multiprocessing
import os
import sys


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

    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
        for device in physical_gpus:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

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
            if logical_gpus:
                mixed_precision.set_global_policy("mixed_float16")
                logger.info("Mixed precision auto-enabled (GPUs detected: %d).", len(logical_gpus))
            else:
                logger.info("Using TensorFlow default precision policy.")
        except Exception as exc:
            logger.warning("Mixed precision auto policy setup failed: %s", exc)

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
