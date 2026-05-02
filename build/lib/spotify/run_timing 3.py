from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import time


def _json_ready(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_ready(item_method())
        except Exception:
            return str(value)
    return str(value)


class RunPhaseRecorder:
    def __init__(self, *, run_id: str):
        self.run_id = str(run_id)
        self._started_at = datetime.now(timezone.utc)
        self._started_perf = time.perf_counter()
        self._records: list[dict[str, object]] = []

    @contextmanager
    def phase(self, name: str, **metadata: object):
        phase_name = str(name).strip() or "unnamed_phase"
        phase_metadata = {str(key): _json_ready(value) for key, value in metadata.items()}
        started_perf = time.perf_counter()
        started_offset = started_perf - self._started_perf
        status = "ok"
        try:
            yield phase_metadata
        except Exception as exc:
            status = "failed"
            phase_metadata.setdefault("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            finished_perf = time.perf_counter()
            finished_offset = finished_perf - self._started_perf
            self._records.append(
                {
                    "phase_name": phase_name,
                    "status": status,
                    "started_offset_seconds": round(started_offset, 6),
                    "finished_offset_seconds": round(finished_offset, 6),
                    "duration_seconds": round(finished_perf - started_perf, 6),
                    "metadata": _json_ready(phase_metadata),
                }
            )

    def skip(self, name: str, *, reason: str, **metadata: object) -> None:
        phase_name = str(name).strip() or "unnamed_phase"
        payload = {str(key): _json_ready(value) for key, value in metadata.items()}
        payload["reason"] = str(reason)
        offset = round(time.perf_counter() - self._started_perf, 6)
        self._records.append(
            {
                "phase_name": phase_name,
                "status": "skipped",
                "started_offset_seconds": offset,
                "finished_offset_seconds": offset,
                "duration_seconds": 0.0,
                "metadata": payload,
            }
        )

    def summary(self, *, final_status: str) -> dict[str, object]:
        finished_at = datetime.now(timezone.utc)
        total_seconds = round(time.perf_counter() - self._started_perf, 6)
        phases = [dict(record) for record in self._records]
        non_skipped = [record for record in phases if str(record.get("status")) != "skipped"]
        completed = [record for record in phases if str(record.get("status")) == "ok"]
        measured_seconds = round(
            sum(float(record.get("duration_seconds", 0.0)) for record in non_skipped),
            6,
        )
        ranked = sorted(
            non_skipped,
            key=lambda record: float(record.get("duration_seconds", 0.0)),
            reverse=True,
        )
        slowest_phases = [
            {
                "phase_name": str(record.get("phase_name", "")),
                "status": str(record.get("status", "")),
                "duration_seconds": float(record.get("duration_seconds", 0.0)),
            }
            for record in ranked[:5]
        ]
        return {
            "run_id": self.run_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "final_status": str(final_status),
            "total_seconds": total_seconds,
            "measured_seconds": measured_seconds,
            "unmeasured_overhead_seconds": round(max(total_seconds - measured_seconds, 0.0), 6),
            "phase_count": len(phases),
            "completed_phase_count": len(completed),
            "non_skipped_phase_count": len(non_skipped),
            "slowest_phase": (slowest_phases[0] if slowest_phases else {}),
            "slowest_phases": slowest_phases,
            "phases": phases,
        }

    def write_artifacts(self, *, run_dir: Path, final_status: str) -> tuple[Path, Path, dict[str, object]]:
        payload = self.summary(final_status=final_status)
        json_path = run_dir / "run_phase_timings.json"
        csv_path = run_dir / "run_phase_timings.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "phase_name",
                    "status",
                    "started_offset_seconds",
                    "finished_offset_seconds",
                    "duration_seconds",
                    "metadata_json",
                ],
            )
            writer.writeheader()
            for record in payload["phases"]:
                writer.writerow(
                    {
                        "phase_name": record.get("phase_name", ""),
                        "status": record.get("status", ""),
                        "started_offset_seconds": record.get("started_offset_seconds", 0.0),
                        "finished_offset_seconds": record.get("finished_offset_seconds", 0.0),
                        "duration_seconds": record.get("duration_seconds", 0.0),
                        "metadata_json": json.dumps(record.get("metadata", {}), sort_keys=True),
                    }
                )
        return json_path, csv_path, payload
