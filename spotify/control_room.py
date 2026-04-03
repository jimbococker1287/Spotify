from __future__ import annotations

from .control_room_core import build_control_room_report, main, write_control_room_report

__all__ = [
    "build_control_room_report",
    "main",
    "write_control_room_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
