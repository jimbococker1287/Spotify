from __future__ import annotations

from pathlib import Path
import json
import site
import sys

from spotify.runtime import build_acceleration_hint, detect_acceleration_environment


def _safe_user_site() -> str:
    try:
        return site.getusersitepackages()
    except Exception:
        return ""


def _parse_python_version(raw: str) -> tuple[int, int]:
    parts = str(raw).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return (0, 0)
    return (major, minor)


def main() -> int:
    summary = detect_acceleration_environment()
    summary["executable"] = sys.executable
    summary["user_site"] = _safe_user_site()

    try:
        import tensorflow as tf
    except Exception as exc:
        summary["tensorflow_import_error"] = repr(exc)
        logical_gpu_count = 0
        physical_gpu_count = 0
    else:
        summary["tensorflow_import_error"] = ""
        summary["tensorflow_runtime_version"] = tf.__version__
        physical_gpus = tf.config.list_physical_devices("GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        physical_gpu_count = len(physical_gpus)
        logical_gpu_count = len(logical_gpus)
        summary["physical_gpus"] = [repr(device) for device in physical_gpus]
        summary["logical_gpus"] = [repr(device) for device in logical_gpus]

    hint = build_acceleration_hint(summary, logical_gpu_count=logical_gpu_count)
    summary["hint"] = hint or ""

    python_version = str(summary.get("python_version", "unknown"))
    user_plugin_dirs = sorted(
        path
        for path in (Path.home() / "Library" / "Python").glob("*/lib/python/site-packages/tensorflow-plugins")
        if path.exists()
    )
    summary["user_plugin_dirs"] = [str(path) for path in user_plugin_dirs]
    summary["user_plugin_dir_exists"] = bool(user_plugin_dirs)

    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary.get("platform") == "darwin" and summary.get("machine") == "arm64":
        print()
        print("Apple Silicon notes:")
        if int(logical_gpu_count) > 0:
            print("- TensorFlow can already see a GPU in this environment.")
        else:
            print("- TensorFlow cannot currently see a GPU in this environment.")
            if _parse_python_version(python_version) >= (3, 13):
                print("- This repo's current Python 3.13 environment could not install `tensorflow-metal` in local verification.")
                print("- Prefer a dedicated Python 3.11 venv when testing Metal support.")
            if not summary.get("tensorflow_metal_version"):
                print("- `tensorflow-metal` is not installed in the active environment.")
            if summary.get("user_plugin_dir_exists"):
                print("- Found user-site TensorFlow plugin directories:")
                for path in user_plugin_dirs:
                    print(f"  - `{path}`")
                print("- A stale global Metal plugin there can break isolated virtualenv tests; use `PYTHONNOUSERSITE=1` when probing.")
            print("- Suggested commands:")
            print("  /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv-metal")
            print("  PYTHONNOUSERSITE=1 .venv-metal/bin/python -m pip install --upgrade pip setuptools wheel")
            print("  PYTHONNOUSERSITE=1 .venv-metal/bin/python -m pip install tensorflow==2.20.0 tensorflow-metal==1.2.0")
            print("  PYTHONNOUSERSITE=1 .venv-metal/bin/python scripts/check_acceleration.py")

    if summary.get("tensorflow_import_error"):
        return 1
    if int(logical_gpu_count) <= 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
