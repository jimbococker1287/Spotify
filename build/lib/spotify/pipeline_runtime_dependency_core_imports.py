from __future__ import annotations


def load_pipeline_runtime_core_imports() -> dict[str, object]:
    from .analytics_db import refresh_analytics_database
    from .backtesting import inspect_temporal_backtest_cache, run_temporal_backtest
    from .control_room import write_control_room_report
    from .data import PreparedDataCacheInfo, SKEW_CONTEXT_FEATURES, load_or_prepare_training_data, load_streaming_history
    from .data_quality import run_data_quality_gate
    from .governance import evaluate_champion_gate
    from .monitoring import ResourceMonitor
    from .training_preflight import compute_baselines, resolve_cached_deep_training_artifacts

    return {
        "PreparedDataCacheInfo": PreparedDataCacheInfo,
        "SKEW_CONTEXT_FEATURES": SKEW_CONTEXT_FEATURES,
        "ResourceMonitor": ResourceMonitor,
        "compute_baselines": compute_baselines,
        "evaluate_champion_gate": evaluate_champion_gate,
        "inspect_temporal_backtest_cache": inspect_temporal_backtest_cache,
        "load_or_prepare_training_data": load_or_prepare_training_data,
        "load_streaming_history": load_streaming_history,
        "refresh_analytics_database": refresh_analytics_database,
        "resolve_cached_deep_training_artifacts": resolve_cached_deep_training_artifacts,
        "run_data_quality_gate": run_data_quality_gate,
        "run_temporal_backtest": run_temporal_backtest,
        "write_control_room_report": write_control_room_report,
    }


__all__ = ["load_pipeline_runtime_core_imports"]
