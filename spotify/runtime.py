from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Tuple


def configure_process_env() -> None:
    cpu_count = str(multiprocessing.cpu_count())
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", cpu_count)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", cpu_count)
    os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
    os.environ.setdefault("TF_GPU_THREAD_COUNT", "4")
    os.environ.setdefault("TF_METAL_ENABLE_FUSED_OPERATIONS", "1")


def load_tensorflow_runtime(logger):
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies first: pip install -r requirements.txt"
        ) from exc

    mixed_precision.set_global_policy("mixed_float16")

    try:
        if sys.platform != "darwin":
            tf.config.optimizer.set_jit(True)
        else:
            logger.info("Skipping XLA JIT on macOS/Metal due to platform support.")
    except Exception as exc:  # pragma: no cover - backend dependent
        logger.warning("XLA JIT enable failed: %s", exc)

    return tf


def select_distribution_strategy(tf):
    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            return tf.distribute.OneDeviceStrategy(device=gpus[0].name)
    except Exception:
        pass
    return tf.distribute.get_strategy()
