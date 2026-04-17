from __future__ import annotations

from .pipeline_postrun_finalize import finalize_postrun
from .pipeline_postrun_stages import run_postrun_stages
from .pipeline_postrun_types import PipelinePostRunContext, PipelinePostRunDeps, PipelinePostRunResult


def run_pipeline_postrun(*, context: PipelinePostRunContext, deps: PipelinePostRunDeps) -> PipelinePostRunResult:
    stage_outputs = run_postrun_stages(context=context, deps=deps)
    return finalize_postrun(context=context, deps=deps, stage_outputs=stage_outputs)


__all__ = ["run_pipeline_postrun"]
