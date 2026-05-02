from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineExperimentPlan:
    classical_feature_bundle: Any
    deep_backtest_cache_inspection: Any
    deep_cache_plan: Any
    deep_cache_stats: dict[str, object]
    needs_tf_for_deep_backtest: bool
    needs_tf_for_deep_training: bool


__all__ = ["PipelineExperimentPlan"]
