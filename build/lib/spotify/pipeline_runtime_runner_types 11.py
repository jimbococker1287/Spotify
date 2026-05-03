from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineRunSetup:
    artifact_paths: list[Path]
    history_dir: Path
    logger: Any
    manifest_path: Path
    phase_recorder: Any
    run_dir: Path
    run_id: str


@dataclass
class PipelineRunPolicy:
    run_classical_models: bool
    run_deep_backtest: bool
    run_deep_models: bool


__all__ = [
    "PipelineRunPolicy",
    "PipelineRunSetup",
]
