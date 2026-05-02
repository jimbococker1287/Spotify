from __future__ import annotations

from dataclasses import dataclass

from .pipeline_runtime_dependency_types import PipelineRuntimeDeps
from .tracking import MlflowTracker


@dataclass
class PipelineBootstrapOutputs:
    tracker: MlflowTracker
    deps: PipelineRuntimeDeps
    raw_df: object
    prepared: object
    cache_info_payload: dict[str, object]
    artist_labels: list[str]


@dataclass
class PipelinePreparedDataOutputs:
    raw_df: object
    prepared: object
    cache_info_payload: dict[str, object]
    artist_labels: list[str]


@dataclass
class PipelinePreparedTrainingState:
    raw_df: object
    prepared: object
    cache_info: object


__all__ = [
    "PipelineBootstrapOutputs",
    "PipelinePreparedDataOutputs",
    "PipelinePreparedTrainingState",
]
