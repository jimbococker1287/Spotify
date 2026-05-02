from __future__ import annotations

from .pipeline_runtime_dependency_analysis_imports import load_pipeline_runtime_analysis_imports
from .pipeline_runtime_dependency_core_imports import load_pipeline_runtime_core_imports
from .pipeline_runtime_dependency_experiment_imports import load_pipeline_runtime_experiment_imports


def load_pipeline_runtime_imports(*, phase_recorder) -> dict[str, object]:
    with phase_recorder.phase("dependency_imports") as phase:
        core_imports = load_pipeline_runtime_core_imports()
        experiment_imports = load_pipeline_runtime_experiment_imports()
        analysis_imports = load_pipeline_runtime_analysis_imports()
        phase["import_group"] = "pipeline_dependencies"
        phase["import_sections"] = ["core", "experiment", "analysis"]

    return core_imports | experiment_imports | analysis_imports


__all__ = ["load_pipeline_runtime_imports"]
