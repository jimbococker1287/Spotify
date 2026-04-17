from __future__ import annotations

from .pipeline_runtime_analysis_artifacts import run_analysis_artifacts
from .pipeline_runtime_analysis_finalize import log_analysis_leaderboard, run_analysis_postrun
from .pipeline_runtime_analysis_types import (
    PipelineAnalysisContext,
    PipelineAnalysisDeps,
    PipelineAnalysisOutputs,
)


def run_analysis_and_postrun(*, context: PipelineAnalysisContext, deps: PipelineAnalysisDeps) -> PipelineAnalysisOutputs:
    run_analysis_artifacts(context=context, deps=deps)
    outputs = run_analysis_postrun(context=context, deps=deps)
    log_analysis_leaderboard(context=context)
    return outputs


__all__ = [
    "PipelineAnalysisContext",
    "PipelineAnalysisDeps",
    "PipelineAnalysisOutputs",
    "run_analysis_and_postrun",
]
