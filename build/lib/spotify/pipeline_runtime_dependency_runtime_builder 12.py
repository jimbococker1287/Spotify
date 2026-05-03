from __future__ import annotations

from collections.abc import Mapping

from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def build_runtime_deps(*, imported_deps: Mapping[str, object]) -> PipelineRuntimeDeps:
    return PipelineRuntimeDeps(**dict(imported_deps))


__all__ = ["build_runtime_deps"]
