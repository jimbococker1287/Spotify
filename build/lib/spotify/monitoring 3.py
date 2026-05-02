from __future__ import annotations

import re
import subprocess
import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self, logger, sample_interval_s: float = 1.0):
        self.logger = logger
        self.sample_interval_s = sample_interval_s
        self.cpu_usage: list[float] = []
        self.gpu_usage: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        try:
            import GPUtil  # noqa: F401

            self._gputil_available = True
        except ImportError:
            self._gputil_available = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self._sample_gpu()
            time.sleep(self.sample_interval_s)

    def _sample_gpu(self) -> None:
        if self._gputil_available:
            try:
                from GPUtil import getGPUs

                gpus = getGPUs()
                if gpus:
                    self.gpu_usage.append(gpus[0].load * 100)
                return
            except Exception:
                return

        try:
            result = subprocess.run(
                [
                    "powermetrics",
                    "--samplers",
                    "gpu_power",
                    "-n",
                    "1",
                    "-i",
                    "1000",
                    "--format",
                    "text",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                match = re.search(r"GPU_Utilization=(\d+)", result.stdout)
                if match:
                    self.gpu_usage.append(float(match.group(1)))
        except Exception:
            return
