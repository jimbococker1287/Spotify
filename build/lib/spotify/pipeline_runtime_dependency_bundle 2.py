from __future__ import annotations

from .pipeline_runtime_dependency_builders import build_analysis_deps, build_experiment_deps, build_runtime_deps
from .pipeline_runtime_dependency_imports import load_pipeline_runtime_imports


def load_pipeline_runtime_dependencies(*, phase_recorder):
    imported_deps = load_pipeline_runtime_imports(phase_recorder=phase_recorder)
    return build_runtime_deps(imported_deps=imported_deps)


__all__ = [
    "build_analysis_deps",
    "build_experiment_deps",
    "load_pipeline_runtime_dependencies",
]
