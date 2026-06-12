from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import spotify.pipeline_helpers as pipeline_helpers
from spotify.pipeline_runtime_tensorflow_stage import release_deep_runtime_resources
from spotify.run_timing import RunPhaseRecorder


def _logger() -> logging.Logger:
    logger = logging.getLogger("spotify.test.pipeline_runtime.cleanup")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


class _TensorFlowBackend:
    def __init__(self) -> None:
        self.free_memory_values: list[bool] = []

    def clear_session(self, *, free_memory: bool = True) -> None:
        self.free_memory_values.append(free_memory)


@pytest.mark.parametrize(
    ("configured_policy", "expected_policy", "expected_gc_calls", "expected_generation"),
    [
        (None, "young", [(0,)], 0),
        ("off", "off", [], None),
        ("full", "full", [()], 2),
    ],
)
def test_release_deep_runtime_resources_uses_one_configurable_gc_pass(
    monkeypatch,
    configured_policy: str | None,
    expected_policy: str,
    expected_gc_calls: list[tuple[int, ...]],
    expected_generation: int | None,
) -> None:
    backend = _TensorFlowBackend()
    tensorflow_runtime = SimpleNamespace(keras=SimpleNamespace(backend=backend))
    gc_calls: list[tuple[int, ...]] = []

    monkeypatch.delenv("SPOTIFY_TF_CLEANUP_GC", raising=False)

    def _collect(*args: int) -> int:
        gc_calls.append(args)
        return 7

    monkeypatch.setattr(pipeline_helpers.gc, "collect", _collect)

    metadata = pipeline_helpers._release_deep_runtime_resources(
        tensorflow_runtime,
        _logger(),
        gc_policy=configured_policy,
    )

    assert backend.free_memory_values == [False]
    assert gc_calls == expected_gc_calls
    assert metadata["clear_session_succeeded"] is True
    assert metadata["clear_session_free_memory"] is False
    assert metadata["gc_policy"] == expected_policy
    assert metadata["gc_generation"] == expected_generation
    assert metadata["gc_collection_count"] == (1 if expected_gc_calls else 0)
    assert metadata["gc_collected_objects"] == (7 if expected_gc_calls else 0)
    assert metadata["gc_error"] == ""


def test_release_deep_runtime_resources_honors_gc_environment(monkeypatch) -> None:
    backend = _TensorFlowBackend()
    tensorflow_runtime = SimpleNamespace(keras=SimpleNamespace(backend=backend))
    gc_calls: list[tuple[int, ...]] = []

    monkeypatch.setenv("SPOTIFY_TF_CLEANUP_GC", "off")
    monkeypatch.setattr(
        pipeline_helpers.gc,
        "collect",
        lambda *args: gc_calls.append(args) or 0,
    )

    metadata = pipeline_helpers._release_deep_runtime_resources(tensorflow_runtime, _logger())

    assert gc_calls == []
    assert metadata["gc_policy"] == "off"
    assert metadata["gc_policy_configured"] == "off"


def test_release_deep_runtime_resources_records_cleanup_phase_metadata(monkeypatch) -> None:
    backend = _TensorFlowBackend()
    tensorflow_runtime = SimpleNamespace(keras=SimpleNamespace(backend=backend))
    recorder = RunPhaseRecorder(run_id="cleanup-metadata")
    context = SimpleNamespace(phase_recorder=recorder, logger=_logger())

    monkeypatch.setenv("SPOTIFY_TF_CLEANUP_GC", "off")

    release_deep_runtime_resources(
        context=context,
        tf=tensorflow_runtime,
        release_point="before_temporal_backtest",
        next_stage="temporal_backtest",
        deep_backtest_required=False,
    )

    phase = recorder.summary(final_status="FINISHED")["phases"][0]
    metadata = phase["metadata"]
    assert phase["phase_name"] == "release_deep_runtime_resources"
    assert metadata["release_point"] == "before_temporal_backtest"
    assert metadata["next_stage"] == "temporal_backtest"
    assert metadata["deep_backtest_required"] is False
    assert metadata["gc_policy"] == "off"
    assert metadata["clear_session_free_memory"] is False
    assert metadata["clear_session_seconds"] >= 0.0
    assert metadata["gc_seconds"] >= 0.0
