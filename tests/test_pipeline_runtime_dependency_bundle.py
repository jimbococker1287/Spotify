from __future__ import annotations

import builtins
from contextlib import contextmanager

from spotify.pipeline_runtime_dependency_bundle import load_pipeline_runtime_dependencies


class _PhaseRecorderStub:
    @contextmanager
    def phase(self, _name: str):
        yield {}


def test_load_pipeline_runtime_dependencies_defers_spotify_training_import(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"training", "spotify.training"} or name.endswith(".training"):
            raise AssertionError("spotify.training should not be imported during dependency bundle setup")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    deps = load_pipeline_runtime_dependencies(phase_recorder=_PhaseRecorderStub())

    assert callable(deps.compute_baselines)
    assert callable(deps.resolve_cached_deep_training_artifacts)
    assert callable(deps.train_and_evaluate_models)
