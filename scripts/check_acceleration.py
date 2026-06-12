from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import platform
import site
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spotify.runtime import (  # noqa: E402
    build_acceleration_hint,
    configure_process_env,
    configure_tensorflow_devices,
    detect_acceleration_environment,
)


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


def _base_summary() -> dict[str, object]:
    return {
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requested_device_mode": os.getenv("SPOTIFY_TF_DEVICE_MODE", "auto"),
        "force_cpu_raw": os.getenv("SPOTIFY_FORCE_CPU", "0"),
    }


def collect_diagnostics() -> tuple[dict[str, object], int]:
    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    try:
        configure_process_env()
        summary = detect_acceleration_environment()
    except ValueError as exc:
        summary = _base_summary()
        errors.append(str(exc))
        summary.update(
            {
                "errors": errors,
                "warnings": warnings,
                "recommendations": recommendations,
                "status": "error",
                "exit_code": 3,
            }
        )
        return summary, 3

    summary["executable"] = sys.executable
    summary["base_prefix"] = sys.base_prefix
    summary["prefix"] = sys.prefix
    summary["in_virtualenv"] = sys.prefix != sys.base_prefix
    summary["user_site"] = _safe_user_site()
    summary["user_site_enabled"] = bool(site.ENABLE_USER_SITE)
    summary["python_no_user_site"] = os.getenv("PYTHONNOUSERSITE", "")
    summary["tensorflow_import_error"] = ""
    summary["tensorflow_configuration_error"] = ""
    summary["tensorflow_runtime_version"] = None
    summary["physical_gpus"] = []
    summary["logical_gpus"] = []
    logical_gpu_count = 0

    try:
        import tensorflow as tf
    except Exception as exc:
        summary["tensorflow_import_error"] = repr(exc)
        errors.append(
            "TensorFlow failed to import. Check that the installed TensorFlow package, Python version, "
            "and tensorflow-metal plugin are mutually compatible."
        )
    else:
        summary["tensorflow_runtime_version"] = tf.__version__
        try:
            discovered_physical_gpus = list(tf.config.list_physical_devices("GPU"))
        except Exception:
            discovered_physical_gpus = []
        summary["physical_gpus"] = [repr(device) for device in discovered_physical_gpus]

        try:
            physical_gpus, logical_gpus, device_mode = configure_tensorflow_devices(tf)
        except Exception as exc:
            summary["tensorflow_configuration_error"] = repr(exc)
            errors.append(str(exc))
            try:
                logical_gpus = list(tf.config.list_logical_devices("GPU"))
            except Exception:
                logical_gpus = []
        else:
            summary["device_mode"] = device_mode
            summary["physical_gpus"] = [repr(device) for device in physical_gpus]

        logical_gpu_count = len(logical_gpus)
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

    if summary.get("platform") == "darwin" and summary.get("machine") == "arm64":
        if summary.get("device_mode") == "cpu" and logical_gpu_count == 0 and not errors:
            recommendations.append("CPU mode is active and TensorFlow GPU devices are hidden.")
        elif logical_gpu_count > 0:
            recommendations.append("TensorFlow can see a logical GPU in this environment.")
        else:
            warnings.append("TensorFlow cannot currently see a logical GPU in this environment.")
            if _parse_python_version(python_version) >= (3, 13):
                recommendations.append(
                    "Prefer a dedicated native arm64 Python 3.11 environment for the pinned Metal runtime."
                )
            if not summary.get("tensorflow_metal_version"):
                warnings.append("tensorflow-metal is not installed in the active environment.")
            if summary.get("user_plugin_dir_exists"):
                warnings.append(
                    "User-site TensorFlow plugin directories can leak stale plugins into virtual environments: "
                    + ", ".join(str(path) for path in user_plugin_dirs)
                )
                recommendations.append("Set PYTHONNOUSERSITE=1 when probing an isolated Metal environment.")
            recommendations.extend(
                [
                    "/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv-metal",
                    "PYTHONNOUSERSITE=1 .venv-metal/bin/python -m pip install --upgrade pip setuptools wheel",
                    "PYTHONNOUSERSITE=1 .venv-metal/bin/python -m pip install "
                    "tensorflow-macos==2.16.2 tensorflow-metal==1.2.0",
                    "PYTHONNOUSERSITE=1 .venv-metal/bin/python scripts/check_acceleration.py --json",
                ]
            )

    if errors:
        exit_code = 1 if summary.get("tensorflow_import_error") else 3
        status = "error"
    elif summary.get("device_mode") == "cpu":
        exit_code = 0
        status = "ok"
    elif logical_gpu_count <= 0:
        exit_code = 2
        status = "warning"
    else:
        exit_code = 0
        status = "ok"

    summary["errors"] = errors
    summary["warnings"] = warnings
    summary["recommendations"] = recommendations
    summary["status"] = status
    summary["exit_code"] = exit_code
    return summary, exit_code


def _print_human_notes(summary: dict[str, object]) -> None:
    if summary.get("platform") != "darwin" or summary.get("machine") != "arm64":
        return

    print()
    print("Apple Silicon notes:")
    for message in summary.get("errors", []):
        print(f"- ERROR: {message}")
    for message in summary.get("warnings", []):
        print(f"- WARNING: {message}")
    for message in summary.get("recommendations", []):
        print(f"- {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect TensorFlow device acceleration and Metal readiness.")
    parser.add_argument("--json", action="store_true", help="Print one machine-readable JSON object with no extra text.")
    args = parser.parse_args(argv)

    summary, exit_code = collect_diagnostics()
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    if not args.json:
        _print_human_notes(summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
